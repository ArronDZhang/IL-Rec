#!/usr/bin/env python
"""Entry point for incremental ILRec policy training work.

The dry-run path remains a deterministic smoke update for fast validation. The
non-dry-run path is a compact ILRec actor-critic trainer that uses actual
``ILRecEnv.step`` transitions, ILRec discriminator/demo weighting helpers, and
emits rollout JSON compatible with the il-rec exporter.
"""

import argparse
import json
import math
import pickle
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "src", ROOT / "src" / "DeepCTR-Torch", ROOT / "src" / "tianshou"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.policy.demo_advantage import compute_demo_advantages, fit_demo_q_network, fit_demo_value_network
from core.policy.demo_returns import annotate_world_model_demo_returns
from core.policy.discriminator import TransitionDiscriminator
from core.policy.ilrec_a2c import compute_demo_weights, compute_ilrec_policy_loss, compute_mixed_replay_policy_loss
from core.policy.qv_critic import StateQVCritic, qv_critic_training_step, update_target_critic
from core.policy.replay_buffers import DemoReplayBuffer, MixedReplayBuffer, ReplayBuffer
from core.policy.state_action_embedding import DEFAULT_LLAMA_MODEL_PATH, StateActionEmbeddingCache
from core.policy.state_action_text import TEMPLATE_VERSION


DEFAULT_MESSAGE = "ILRec"
DEFAULT_ILREC_ENV_SOURCE = "precomputed_matpre"
DEFAULT_ILREC_STATE_TRACKER = "roler_attention"
DEFAULT_ILREC_DISCRIMINATOR_RETRAIN_INTERVAL = 5000
DEFAULT_ILREC_POLICY_LR = 1e-3
DEFAULT_ILREC_DISCOUNT = 0.5
DEFAULT_ILREC_MIXED_REPLAY_SAMPLING = "global_priority"
ILREC_DEMO_ADVANTAGE_METHOD = "world_model_return_q_demo_v_demo"
DEFAULT_ILREC_HYPERPARAMETER_ENV = "AmazonEnv-v0"
PUBLIC_FIXED_DEFAULTS = {
    "train_continue_bonus": 0.0,
    "train_distance_terminal_penalty": 0.0,
    "train_terminal_repeat_td_penalty": 0.0,
    "train_unsafe_repeat_penalty": 0.0,
    "train_unsafe_distance_penalty": 0.0,
    "train_action_repeat_penalty": 0.0,
    "train_action_distance_penalty": 0.0,
    "train_action_shaping_loss": "env_only",
    "train_bc_kl_coef": 0.0,
    "train_bc_kl_label_smoothing": 0.0,
    "train_logit_l2_penalty": 0.0,
    "train_logit_std_penalty": 0.0,
    "train_logit_std_target": 10.0,
    "train_actor_row_norm_penalty": 0.0,
    "train_actor_row_norm_target": 10.0,
    "train_entropy_floor_ratio": 0.0,
    "train_entropy_floor_coef": 0.0,
}
ILREC_HYPERPARAMETER_DEFAULTS = {
    "AmazonEnv-v0": {
        "beta": 1.0,
        "alpha": 0.25,
        "alpha_ent": 0.1,
        "lambda_imit": 0.5,
    },
    "SteamEnv-v0": {
        "beta": 10.0,
        "alpha": 0.5,
        "alpha_ent": 0.1,
        "lambda_imit": 0.25,
    },
}

FINAL_POLICY_LOGIT_CLAMP = {
    "AmazonEnv-v0": 1.0,
    "SteamEnv-v0": 15.0,
}


def ilrec_hyperparameter_defaults(env):
    default_env = env if env in ILREC_HYPERPARAMETER_DEFAULTS else DEFAULT_ILREC_HYPERPARAMETER_ENV
    return dict(ILREC_HYPERPARAMETER_DEFAULTS[default_env])


def apply_ilrec_hyperparameter_defaults(args):
    defaults = ilrec_hyperparameter_defaults(args.env)
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def apply_ilrec_defaults(args):
    for name, value in PUBLIC_FIXED_DEFAULTS.items():
        setattr(args, name, value)
    if args.lr is None:
        args.lr = DEFAULT_ILREC_POLICY_LR
    if args.discount is None:
        args.discount = DEFAULT_ILREC_DISCOUNT
    if args.mixed_replay_sampling is None:
        args.mixed_replay_sampling = DEFAULT_ILREC_MIXED_REPLAY_SAMPLING
    if args.train_env_source is None:
        args.train_env_source = DEFAULT_ILREC_ENV_SOURCE
    if args.eval_env_source is None:
        args.eval_env_source = DEFAULT_ILREC_ENV_SOURCE
    if args.state_action_feature_mode is None:
        args.state_action_feature_mode = "llama"
    if args.state_tracker_type is None:
        args.state_tracker_type = DEFAULT_ILREC_STATE_TRACKER
    if args.discriminator_retrain_interval is None:
        args.discriminator_retrain_interval = DEFAULT_ILREC_DISCRIMINATOR_RETRAIN_INTERVAL
    if args.policy_logit_clamp is None:
        args.policy_logit_clamp = FINAL_POLICY_LOGIT_CLAMP.get(args.env, 0.0)
    return args


def discount_standard_status(args):
    if math.isclose(float(args.discount), DEFAULT_ILREC_DISCOUNT, rel_tol=0.0, abs_tol=1e-12):
        return "official"
    return "legacy_ablation_nonstandard"


@dataclass(frozen=True)
class WorldModelArtifacts:
    mat_pre: object
    mat_var: object
    params: object
    mat_pre_path: Path
    mat_var_path: Path
    params_path: Path


@dataclass(frozen=True)
class UnsafeActionPenalty:
    total_loss: torch.Tensor
    repeat_mass: torch.Tensor
    distance_mass: torch.Tensor


@dataclass(frozen=True)
class ActionLogitShapingResult:
    adjusted_logits: torch.Tensor
    repeat_mass: torch.Tensor
    distance_mass: torch.Tensor


@dataclass(frozen=True)
class AdvantageTransformResult:
    advantages: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    clipped_count: int


@dataclass(frozen=True)
class LogitScalePenalty:
    total_loss: torch.Tensor
    logit_l2: torch.Tensor
    logit_std_excess: torch.Tensor
    actor_row_norm_excess: torch.Tensor
    logit_std_mean: torch.Tensor
    actor_row_norm_max: torch.Tensor


@dataclass(frozen=True)
class EntropyFloorPenalty:
    total_loss: torch.Tensor
    entropy_mean: torch.Tensor
    entropy_ratio_mean: torch.Tensor
    entropy_floor_gap: torch.Tensor


class UserActorCritic(nn.Module):
    """Small actor-critic with user and recent interaction state."""

    def __init__(
        self,
        num_users,
        num_items,
        hidden_size,
        window_size=4,
        state_tracker_type=DEFAULT_ILREC_STATE_TRACKER,
        num_att_heads=1,
        num_att_layers=2,
        att_dropout=0.1,
    ):
        super().__init__()
        if num_users <= 0:
            raise ValueError("num_users must be positive.")
        if num_items <= 0:
            raise ValueError("num_items must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if state_tracker_type != DEFAULT_ILREC_STATE_TRACKER:
            raise ValueError(f"Unsupported state_tracker_type: {state_tracker_type}")
        self.num_items = int(num_items)
        self.window_size = int(window_size)
        self.state_tracker_type = str(state_tracker_type)
        self.padding_item_id = int(num_items)
        self.user_embedding = nn.Embedding(int(num_users), int(hidden_size))
        self.item_embedding = nn.Embedding(int(num_items) + 1, int(hidden_size), padding_idx=self.padding_item_id)
        self.reward_projection = nn.Linear(1, int(hidden_size))
        self.position_embedding = nn.Embedding(self.window_size, int(hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_size),
            nhead=int(num_att_heads),
            dim_feedforward=int(hidden_size),
            dropout=float(att_dropout),
            batch_first=True,
        )
        self.history_attention_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_att_layers),
        )
        self.state_norm = nn.LayerNorm(int(hidden_size))
        self.actor = nn.Linear(int(hidden_size), int(num_items))
        self.critic = nn.Linear(int(hidden_size), 1)

    def _normalize_history_batch(self, history, batch_size):
        if history is None:
            return [[] for _ in range(batch_size)]
        if hasattr(history, "detach"):
            history = history.detach().cpu().tolist()
        if isinstance(history, np.ndarray):
            history = history.tolist()
        if batch_size == 1:
            if not isinstance(history, (list, tuple)):
                return [[history]]
            if not history:
                return [[]]
            if all(not isinstance(item, (list, tuple, np.ndarray)) for item in history):
                return [list(history)]
        if len(history) != batch_size:
            raise ValueError("history batch size must match user_ids batch size.")
        rows = []
        for row in history:
            if hasattr(row, "detach"):
                row = row.detach().cpu().tolist()
            if isinstance(row, np.ndarray):
                row = row.tolist()
            if row is None:
                rows.append([])
            elif isinstance(row, (list, tuple)):
                rows.append(list(row))
            else:
                rows.append([row])
        return rows

    def _history_tensors(self, history_actions, history_rewards, batch_size, device):
        action_rows = self._normalize_history_batch(history_actions, batch_size)
        reward_rows = self._normalize_history_batch(history_rewards, batch_size)
        actions = torch.full(
            (batch_size, self.window_size),
            self.padding_item_id,
            dtype=torch.long,
            device=device,
        )
        rewards = torch.zeros((batch_size, self.window_size), dtype=torch.float32, device=device)
        mask = torch.zeros((batch_size, self.window_size), dtype=torch.bool, device=device)

        for row_idx, row_actions in enumerate(action_rows):
            row_rewards = reward_rows[row_idx] if row_idx < len(reward_rows) else []
            paired = []
            for index, action in enumerate(row_actions):
                action = int(action)
                if action < 0 or action >= self.num_items:
                    continue
                reward = float(row_rewards[index]) if index < len(row_rewards) else 0.0
                paired.append((action, reward))
            paired = paired[-self.window_size :]
            offset = self.window_size - len(paired)
            for index, (action, reward) in enumerate(paired):
                col = offset + index
                actions[row_idx, col] = action
                rewards[row_idx, col] = reward
                mask[row_idx, col] = True
        return actions, rewards, mask

    def encode_state(self, user_ids, history_actions=None, history_rewards=None):
        user_ids = torch.as_tensor(user_ids, dtype=torch.long, device=self.user_embedding.weight.device)
        if user_ids.ndim == 0:
            user_ids = user_ids.unsqueeze(0)
        batch_size = int(user_ids.shape[0])
        user_hidden = self.user_embedding(user_ids)
        actions, rewards, mask = self._history_tensors(
            history_actions,
            history_rewards,
            batch_size=batch_size,
            device=user_ids.device,
        )
        positions = torch.arange(self.window_size, device=user_ids.device).unsqueeze(0).expand(batch_size, -1)
        item_hidden = self.item_embedding(actions)
        reward_hidden = self.reward_projection((rewards / 5.0).unsqueeze(-1))
        history_hidden = torch.tanh(item_hidden + reward_hidden + self.position_embedding(positions))
        return self._encode_roler_attention_history(user_hidden, history_hidden, mask)

    def _encode_roler_attention_history(self, user_hidden, history_hidden, mask):
        padding_mask = ~mask
        all_padding = padding_mask.all(dim=1)
        padding_mask = padding_mask.clone()
        if all_padding.any():
            padding_mask[all_padding, -1] = False
        tracked = self.history_attention_encoder(history_hidden, src_key_padding_mask=padding_mask)
        last_indices = mask.long().sum(dim=1).clamp_min(1) - 1
        last_indices = last_indices + (self.window_size - mask.long().sum(dim=1).clamp_min(1))
        last_indices = last_indices.clamp(0, self.window_size - 1)
        history_state = tracked[torch.arange(tracked.shape[0], device=tracked.device), last_indices]
        history_state = torch.where(all_padding.unsqueeze(1), torch.zeros_like(history_state), history_state)
        hidden = torch.relu(self.state_norm(user_hidden + history_state))
        return hidden

    def forward(self, user_ids, history_actions=None, history_rewards=None):
        hidden = self.encode_state(user_ids, history_actions=history_actions, history_rewards=history_rewards)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run ILRec policy training for ILRec datasets.")
    parser.add_argument("--env", required=True, help="Environment id, e.g. AmazonEnv-v0.")
    parser.add_argument("--demo-buffer", required=True, type=Path, help="Path to demo_gpt35.pkl.")
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--alpha-ent", dest="alpha_ent", type=float, default=None)
    parser.add_argument("--lambda-imit", dest="lambda_imit", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--read_message", default="ilrec_gpt35")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="World-model root. Defaults to saved_models/{env}/DeepFM.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Checkpoint output directory. Defaults to saved_models/{env}/ILRec.",
    )
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--dry-run", action="store_true", help="Run only the smoke update.")
    parser.add_argument("--smoke-steps", type=int, default=1)
    parser.add_argument("--smoke-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint to load.")
    parser.add_argument("--train-episodes", type=int, default=100000)
    parser.add_argument(
        "--train-action-selection",
        choices=("sample",),
        default="sample",
        help="Final ILRec training action selection mode.",
    )
    parser.add_argument(
        "--train-advantage-clip",
        type=float,
        default=5.0,
        help="Clamp policy advantages into [-value, value].",
    )
    parser.add_argument(
        "--train-normalize-advantages",
        action="store_true",
        help="Normalize sampled policy advantages before clipping. Enabled by default for ILRec.",
    )
    parser.set_defaults(train_normalize_advantages=True)
    parser.add_argument(
        "--policy-logit-clamp",
        type=float,
        default=None,
        help="Clamp actor logits before policy losses and action selection. Defaults to the dataset standard.",
    )
    parser.add_argument(
        "--policy-logit-clamp-mode",
        choices=("tanh",),
        default="tanh",
        help="Smooth tanh clipping for policy-logit-clamp.",
    )
    parser.add_argument(
        "--train-actor-row-norm-project",
        type=float,
        default=10.0,
        help="Project actor output weight rows to this max norm after optimizer.step().",
    )
    parser.add_argument(
        "--train-actor-bias-clamp",
        type=float,
        default=5.0,
        help="Clamp actor output bias to [-value, value] after optimizer.step().",
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--eval-action-selection",
        choices=("sample",),
        default="sample",
        help="Final ILRec evaluation action selection mode.",
    )
    parser.add_argument(
        "--embedding-model-path",
        default=DEFAULT_LLAMA_MODEL_PATH,
        help="Local LLaMA model path used for ILRec state-action embeddings.",
    )
    parser.add_argument(
        "--state-action-cache-path",
        type=Path,
        default=None,
        help="Optional persistent cache for rendered state-action LLaMA embeddings.",
    )
    parser.add_argument(
        "--state-action-feature-mode",
        choices=("llama", "state_tracker"),
        default=None,
        help="Discriminator feature source for ILRec.",
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument(
        "--state-tracker-type",
        choices=(DEFAULT_ILREC_STATE_TRACKER,),
        default=None,
        help="Policy state tracker. The public ILRec setting uses ROLeR-style Transformer attention.",
    )
    parser.add_argument("--state-tracker-att-heads", type=int, default=1)
    parser.add_argument("--state-tracker-att-layers", type=int, default=2)
    parser.add_argument("--state-tracker-att-dropout", type=float, default=0.1)
    parser.add_argument(
        "--discount",
        type=float,
        default=DEFAULT_ILREC_DISCOUNT,
        help="RL discount factor fixed to the ILRec reproduction standard.",
    )
    parser.add_argument("--irl-gamma", type=float, default=1.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--disc-lr", type=float, default=1e-3)
    parser.add_argument("--disc-hidden-size", type=int, default=32)
    parser.add_argument("--discriminator-retrain-interval", type=int, default=None)
    parser.add_argument("--demo-batch-size", type=int, default=8)
    parser.add_argument("--demo-value-steps", type=int, default=100)
    parser.add_argument("--demo-value-lr", type=float, default=1e-2)
    parser.add_argument("--qv-lr", type=float, default=1e-3)
    parser.add_argument("--target-update-interval", type=int, default=10)
    parser.add_argument("--target-update-tau", type=float, default=1.0)
    parser.add_argument("--mixed-replay-batch-size", type=int, default=8)
    parser.add_argument("--mixed-replay-demo-fraction", type=float, default=0.5)
    parser.add_argument(
        "--mixed-replay-env-priority-scale",
        type=float,
        default=0.05,
        help="Scale applied to env replay priority under global_priority sampling.",
    )
    parser.add_argument(
        "--mixed-replay-sampling",
        choices=(
            MixedReplayBuffer.GLOBAL_PRIORITY,
        ),
        default=None,
        help="Mixed replay sampling mode. The public ILRec setting uses global priority over B_env union B_demo.",
    )
    parser.add_argument(
        "--demo-weight-mode",
        choices=("ilrec",),
        default="ilrec",
        help="Use ILRec demo weights for demo replay/imitation.",
    )
    parser.add_argument("--max-turn", type=int, default=None)
    parser.add_argument("--num-leave-compute", type=int, default=4)
    parser.add_argument("--leave-threshold", type=float, default=None)
    parser.add_argument(
        "--train-env-source",
        choices=("precomputed_matpre",),
        default=None,
        help="Use precomputed train/test DeepFM matPre matrices.",
    )
    parser.add_argument(
        "--eval-env-source",
        choices=("precomputed_matpre",),
        default=None,
        help="Use precomputed train/test DeepFM matPre matrices.",
    )
    parser.add_argument(
        "--ilrec-root",
        type=Path,
        default=ROOT,
        help="il-rec root used when loading true ILRec matrices.",
    )
    parser.add_argument(
        "--read-user-num",
        type=int,
        default=None,
        help="Optional user-row limit when loading true ILRec matrices.",
    )
    parser.add_argument(
        "--eval-user-ids",
        default=None,
        help="Comma-separated encoded user ids to evaluate, in order.",
    )
    parser.add_argument(
        "--eval-users-json",
        type=Path,
        default=None,
        help="JSON file containing eval users, such as il-rec data/{dataset}/test.json with userid_encoded.",
    )
    parser.add_argument(
        "--rollout-json",
        type=Path,
        default=None,
        help="Path for learned-policy simple rollout JSON.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path for learned-policy summary JSON.",
    )
    args = apply_ilrec_hyperparameter_defaults(parser.parse_args(argv))
    return apply_ilrec_defaults(args)


def default_model_root(env):
    return ROOT / "saved_models" / env / "DeepFM"


def default_output_dir(env):
    return ROOT / "saved_models" / env / "ILRec"


def default_ilrec_message(env, seed):
    dataset = dataset_for_env(env)
    return f"ilrec_{dataset}_gpt35_seed{int(seed)}_100k_fb"


def default_state_action_cache_path(output_dir, message):
    return Path(output_dir) / "embedding_cache" / f"[{message}]_state_action_embeddings.pt"


def resolve_state_action_cache_path(args, output_dir):
    if args.state_action_cache_path is not None:
        return args.state_action_cache_path
    return default_state_action_cache_path(output_dir, resolve_output_message(args))


def resolve_output_message(args):
    if args.message != DEFAULT_MESSAGE:
        return args.message
    return default_ilrec_message(args.env, args.seed)


def env_source_label(source):
    if source == "artifacts":
        return "world_model_artifacts"
    if source == "precomputed_matpre":
        return "precomputed_deepfm_matpre"
    if source == "true":
        return "direct_matrix_diagnostic"
    return str(source)


def feature_type_for_mode(feature_mode):
    if feature_mode == "llama":
        return "llama_textual_state_action_embedding"
    if feature_mode == "numeric":
        return "numeric_state_action_features"
    if feature_mode == "state_tracker":
        return "state_tracker_state_action_features"
    return str(feature_mode)


def ilrec_metadata(args):
    embedding_cache_path = getattr(args, "resolved_state_action_cache_path", args.state_action_cache_path)
    return {
        "feature_type": feature_type_for_mode(args.state_action_feature_mode),
        "target_feature_type": "llama_textual_state_action_embedding",
        "state_action_feature_mode": args.state_action_feature_mode,
        "embedding_model_path": str(args.embedding_model_path),
        "embedding_cache_path": str(embedding_cache_path) if embedding_cache_path is not None else None,
        "state_action_template_version": TEMPLATE_VERSION,
        "world_model_source": env_source_label(args.train_env_source),
        "train_world_model_source": env_source_label(args.train_env_source),
        "eval_world_model_source": env_source_label(args.eval_env_source),
        "train_matrix_source": env_source_label(args.train_env_source),
        "eval_matrix_source": env_source_label(args.eval_env_source),
        "grounding_similarity": "cosine_similarity",
        "target_grounding_similarity": "cosine_similarity",
        "demo_advantage_method": ILREC_DEMO_ADVANTAGE_METHOD,
        "replay_method": "mixed_env_demo_replay",
        "critic_method": "q_v_td_target",
    }


def artifact_file(model_root, read_message, subdir, suffix):
    pickle_path = model_root / subdir / f"[{read_message}]_{suffix}.pickle"
    if pickle_path.exists():
        return pickle_path
    return pickle_path.with_suffix(".npy")


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def load_matrix_artifact(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    return load_pickle(path)


def load_demo_buffer(path):
    demo = load_pickle(path)
    if not isinstance(demo, dict):
        raise ValueError(f"{path} must contain a dictionary demo buffer.")
    transitions = demo.get("transitions")
    if not isinstance(transitions, list):
        raise ValueError(f"{path} must contain a transitions list.")
    return demo


def load_world_model_artifacts(model_root, read_message):
    model_root = Path(model_root)
    mat_pre_path = artifact_file(model_root, read_message, "matsPre", "matPre")
    mat_var_path = artifact_file(model_root, read_message, "matsVar", "matVar")
    params_path = artifact_file(model_root, read_message, "params", "params")
    missing = [path for path in (mat_pre_path, mat_var_path, params_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing world-model artifacts: " + ", ".join(str(path) for path in missing))
    return WorldModelArtifacts(
        mat_pre=load_matrix_artifact(mat_pre_path),
        mat_var=load_matrix_artifact(mat_var_path),
        params=load_pickle(params_path),
        mat_pre_path=mat_pre_path,
        mat_var_path=mat_var_path,
        params_path=params_path,
    )


def dataset_for_env(env):
    if env == "AmazonEnv-v0":
        return "amazon"
    if env == "SteamEnv-v0":
        return "steam"
    raise ValueError(f"Unsupported ILRec env: {env}")


def env_class_and_default_threshold(env):
    from environments.ILRec.env.ILRecEnv import AmazonEnv, SteamEnv

    if env == "AmazonEnv-v0":
        return AmazonEnv, 15
    if env == "SteamEnv-v0":
        return SteamEnv, 50
    raise ValueError(f"Unsupported ILRec env: {env}")


def as_reward_matrix(value, label):
    mat = np.asarray(value, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[1] == 0:
        raise ValueError(f"{label} must be a non-empty 2D matrix.")
    return mat


def load_true_matrix(args, split="test", matrix_label="true ILRec reward matrix"):
    mat_path, distance_path = ilrec_matrix_paths(args, split)
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing {matrix_label}: {mat_path}")
    if not distance_path.exists():
        raise FileNotFoundError(f"Missing true ILRec distance matrix: {distance_path}")
    mat = np.load(mat_path, mmap_mode="r")
    if args.read_user_num is not None:
        mat = mat[: int(args.read_user_num)]
    with distance_path.open("rb") as f:
        mat_distance = pickle.load(f)
    return as_reward_matrix(mat, str(mat_path)), np.asarray(mat_distance)


def ilrec_matrix_paths(args, split):
    dataset = dataset_for_env(args.env)
    root = Path(args.ilrec_root) / "env" / dataset
    return root / f"{dataset}_{split}.npy", root / f"{split}_distance_mat.pickle"


def make_policy_env(args, artifacts, source, split):
    env_cls, default_threshold = env_class_and_default_threshold(args.env)
    max_turn = int(args.max_turn) if args.max_turn is not None else 100
    leave_threshold = default_threshold if args.leave_threshold is None else float(args.leave_threshold)
    env_kwargs = {
        "num_leave_compute": int(args.num_leave_compute),
        "leave_threshold": leave_threshold,
        "max_turn": max_turn,
    }
    if source == "artifacts":
        if artifacts is None:
            raise ValueError("World-model artifacts are required when env source is 'artifacts'.")
        mat = as_reward_matrix(artifacts.mat_pre, "mat_pre")
        mat_distance = None
    elif source == "precomputed_matpre":
        mat_path, distance_path = ilrec_matrix_paths(args, split)
        return env_cls.from_files(
            mat_path,
            distance_path,
            read_user_num=args.read_user_num,
            cache=True,
            mmap_mode="r",
            **env_kwargs,
        )
    elif source == "true":
        mat_path, distance_path = ilrec_matrix_paths(args, split)
        return env_cls.from_files(
            mat_path,
            distance_path,
            read_user_num=args.read_user_num,
            cache=True,
            mmap_mode="r",
            **env_kwargs,
        )
    else:
        raise ValueError(f"Unknown env source: {source}")

    return env_cls(
        mat=mat,
        mat_distance=mat_distance,
        **env_kwargs,
    )


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_policy_checkpoint(model, discriminator, checkpoint):
    if checkpoint is None:
        return False
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint must contain model_state_dict for policy restore.")
    model.load_state_dict(model_state)
    discriminator_state = checkpoint.get("discriminator_state_dict")
    if isinstance(discriminator_state, dict):
        discriminator.load_state_dict(discriminator_state)
    return True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(cuda_index):
    cuda_index = int(cuda_index)
    if cuda_index < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{cuda_index}")


def _demo_batch(transitions, batch_size):
    usable = [transition for transition in transitions if "action_id" in transition]
    if not usable:
        raise ValueError("Demo buffer has no transitions with action_id.")
    return usable[: max(1, min(int(batch_size), len(usable)))]


def _usable_demo_transitions(transitions, num_users, num_items):
    usable = []
    history_by_trajectory = {}
    for transition in transitions:
        if "user_id" not in transition or "action_id" not in transition:
            continue
        user_id = int(transition["user_id"])
        action_id = int(transition["action_id"])
        if 0 <= user_id < num_users and 0 <= action_id < num_items:
            trajectory_id = transition.get("trajectory_id", transition.get("traj_id", "__global__"))
            history_actions, history_rewards = history_by_trajectory.setdefault(trajectory_id, ([], []))
            transition_with_history = dict(transition)
            transition_with_history.setdefault("history_action_ids", list(history_actions))
            transition_with_history.setdefault("history_rewards", list(history_rewards))
            usable.append(transition_with_history)
            history_actions.append(action_id)
            history_rewards.append(float(transition.get("reward", 0.0)))
            if transition.get("done"):
                history_by_trajectory[trajectory_id] = ([], [])
    if not usable:
        raise ValueError("No usable demonstration transitions fit the selected env user/action space.")
    return usable


def _cycled_demo_batch(transitions, batch_size, offset):
    batch_size = max(1, min(int(batch_size), len(transitions)))
    return [transitions[(offset + index) % len(transitions)] for index in range(batch_size)]


def _transition_history(transition):
    action_keys = ("history_action_ids", "history_actions", "prior_action_ids", "prev_action_ids")
    reward_keys = ("history_rewards", "prior_rewards", "prev_rewards")
    actions = []
    rewards = []
    for key in action_keys:
        if key in transition:
            actions = transition[key]
            break
    for key in reward_keys:
        if key in transition:
            rewards = transition[key]
            break
    return list(actions or []), list(rewards or [])


def _state_action_features(user_ids, action_ids, rewards, num_users, num_items):
    user_ids = torch.as_tensor(user_ids, dtype=torch.float32)
    action_ids = torch.as_tensor(action_ids, dtype=torch.float32)
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    denom_user = max(1.0, float(num_users - 1))
    denom_item = max(1.0, float(num_items - 1))
    return torch.stack(
        [
            user_ids / denom_user,
            action_ids / denom_item,
            rewards / 5.0,
        ],
        dim=1,
    )


def _optional_id2item(ilrec_root, dataset):
    try:
        from environments.ILRec.paths import load_id2item

        id2item, _ = load_id2item(ilrec_root, dataset)
        return id2item
    except (FileNotFoundError, ValueError):
        return {}


def shape_training_reward(raw_reward, done, info, continue_bonus=0.0, distance_terminal_penalty=0.0):
    continue_bonus = float(continue_bonus)
    distance_terminal_penalty = float(distance_terminal_penalty)
    if continue_bonus < 0:
        raise ValueError("train_continue_bonus must be non-negative.")
    if distance_terminal_penalty < 0:
        raise ValueError("train_distance_terminal_penalty must be non-negative.")

    shaped_reward = float(raw_reward)
    reason = (info or {}).get("reason")
    if not done:
        shaped_reward += continue_bonus
    elif reason == "distance":
        shaped_reward -= distance_terminal_penalty
    return float(shaped_reward)


def _record_training_reward(record):
    if "train_reward" in record:
        return float(record["train_reward"])
    if "world_model_reward" in record:
        return float(record["world_model_reward"])
    return float(record.get("reward", 0.0))


def _record_terminal_reason(record):
    return record.get("reason") or record.get("terminal_reason") or record.get("done_reason")


def _is_terminal_exact_repeat_distance_record(record):
    if not bool(record.get("done", False)):
        return False
    if _record_terminal_reason(record) != "distance":
        return False
    if "action_id" not in record:
        return False
    action_id = int(record["action_id"])
    history_actions, _ = _transition_history(record)
    return action_id in {int(action) for action in history_actions}


def _record_td_target_reward(record, terminal_repeat_penalty=0.0):
    terminal_repeat_penalty = float(terminal_repeat_penalty)
    if terminal_repeat_penalty < 0:
        raise ValueError("train_terminal_repeat_td_penalty must be non-negative.")
    reward = _record_training_reward(record)
    if terminal_repeat_penalty > 0.0 and _is_terminal_exact_repeat_distance_record(record):
        reward -= terminal_repeat_penalty
    return float(reward)


def _policy_transition_records(
    users,
    actions,
    rewards,
    history_action_rows,
    history_reward_rows,
    train_rewards=None,
    infos=None,
):
    records = []
    train_rewards = list(train_rewards) if train_rewards is not None else None
    infos = list(infos) if infos is not None else None
    for index, (user_id, action_id, reward) in enumerate(zip(users, actions, rewards)):
        history_actions = history_action_rows[index] if index < len(history_action_rows) else []
        history_rewards = history_reward_rows[index] if index < len(history_reward_rows) else []
        record = {
            "user_id": int(user_id),
            "action_id": int(action_id),
            "reward": float(reward),
            "history_action_ids": list(history_actions or []),
            "history_rewards": list(history_rewards or []),
        }
        if train_rewards is not None:
            record["train_reward"] = float(train_rewards[index])
        if infos is not None and index < len(infos):
            info = dict(infos[index] or {})
            if "reason" in info:
                record["reason"] = info["reason"]
        records.append(record)
    return records


class NumericDiscriminatorFeatureBuilder:
    feature_type = "numeric_state_action_features"

    def __init__(self, num_users, num_items, device):
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.device = device
        self.input_dim = 3

    def policy_features(self, users, actions, rewards, history_action_rows=None, history_reward_rows=None):
        return _state_action_features(users, actions, rewards, self.num_users, self.num_items).to(self.device)

    def demo_features(self, transitions):
        users = [int(transition["user_id"]) for transition in transitions]
        actions = [int(transition["action_id"]) for transition in transitions]
        rewards = [float(transition.get("reward", 0.0)) for transition in transitions]
        return _state_action_features(users, actions, rewards, self.num_users, self.num_items).to(self.device)

    def save(self):
        return False


class LlamaDiscriminatorFeatureBuilder:
    feature_type = "llama_textual_state_action_embedding"

    def __init__(self, args, demo_transitions, device):
        self.dataset = dataset_for_env(args.env)
        self.device = device
        self.item_lookup = _optional_id2item(args.ilrec_root, self.dataset)
        self.cache = StateActionEmbeddingCache(
            cache_path=args.resolved_state_action_cache_path,
            model_path=args.embedding_model_path,
        )
        probe = self.cache.encode_transitions(
            [demo_transitions[0]],
            dataset=self.dataset,
            item_lookup=self.item_lookup,
        )
        if probe.ndim != 2 or probe.shape[1] <= 0:
            raise ValueError("State-action embedding cache must return non-empty 2D features.")
        self.input_dim = int(probe.shape[1])

    def policy_features(self, users, actions, rewards, history_action_rows=None, history_reward_rows=None):
        transitions = _policy_transition_records(
            users,
            actions,
            rewards,
            history_action_rows or [],
            history_reward_rows or [],
        )
        return self.cache.encode_transitions(
            transitions,
            dataset=self.dataset,
            item_lookup=self.item_lookup,
        ).to(self.device)

    def demo_features(self, transitions):
        return self.cache.encode_transitions(
            transitions,
            dataset=self.dataset,
            item_lookup=self.item_lookup,
        ).to(self.device)

    def save(self):
        self.cache.save()
        return self.cache.cache_path is not None


class StateTrackerDiscriminatorFeatureBuilder:
    feature_type = "state_tracker_state_action_features"

    def __init__(self, state_model, device):
        self.state_model = state_model
        self.device = device
        self.input_dim = int(state_model.actor.in_features + state_model.item_embedding.embedding_dim)

    def _features_for_records(self, records):
        was_training = self.state_model.training
        self.state_model.eval()
        try:
            with torch.no_grad():
                states = _state_features_for_records(self.state_model, records, self.device).detach()
                action_ids = [int(record["action_id"]) for record in records]
                invalid = [
                    action_id
                    for action_id in action_ids
                    if action_id < 0 or action_id >= int(self.state_model.num_items)
                ]
                if invalid:
                    raise ValueError(f"state-tracker features received invalid action ids: {invalid[:3]}")
                actions = torch.tensor(action_ids, dtype=torch.long, device=self.device)
                action_features = self.state_model.item_embedding(actions).detach()
                return torch.cat([states, action_features], dim=1).detach()
        finally:
            self.state_model.train(was_training)

    def policy_features(self, users, actions, rewards, history_action_rows=None, history_reward_rows=None):
        transitions = _policy_transition_records(
            users,
            actions,
            rewards,
            history_action_rows or [],
            history_reward_rows or [],
        )
        return self._features_for_records(transitions)

    def demo_features(self, transitions):
        return self._features_for_records(transitions)

    def save(self):
        return False


def make_discriminator_feature_builder(args, demo_transitions, num_users, num_items, device, state_model=None):
    if args.state_action_feature_mode == "numeric":
        return NumericDiscriminatorFeatureBuilder(num_users, num_items, device)
    if args.state_action_feature_mode == "llama":
        return LlamaDiscriminatorFeatureBuilder(args, demo_transitions, device)
    if args.state_action_feature_mode == "state_tracker":
        if state_model is None:
            raise ValueError("state_model is required for state_tracker state-action features.")
        return StateTrackerDiscriminatorFeatureBuilder(state_model, device)
    raise ValueError(f"Unsupported state-action feature mode: {args.state_action_feature_mode}")


def prepare_demo_advantages(args, feature_builder, demo_transitions, device, state_model=None):
    demo_features = feature_builder.demo_features(demo_transitions).detach()
    if state_model is None:
        value_features = demo_features
    else:
        was_training = state_model.training
        state_model.eval()
        with torch.no_grad():
            value_features = _state_features_for_records(state_model, demo_transitions, device).detach()
        state_model.train(was_training)
    demo_returns = torch.tensor(
        [float(transition["demo_return"]) for transition in demo_transitions],
        dtype=torch.float32,
        device=device,
    )
    value_fit_result = fit_demo_value_network(
        value_features,
        demo_returns,
        hidden_size=int(args.disc_hidden_size),
        lr=float(args.demo_value_lr),
        steps=int(args.demo_value_steps),
    )
    q_fit_result = fit_demo_q_network(
        demo_features,
        demo_returns,
        hidden_size=int(args.disc_hidden_size),
        lr=float(args.demo_value_lr),
        steps=int(args.demo_value_steps),
    )
    with torch.no_grad():
        demo_values = value_fit_result.model(value_features).detach()
        demo_q_values = q_fit_result.model(demo_features).detach()
        demo_advantages = compute_demo_advantages(
            demo_returns,
            demo_values,
            demo_q_values=demo_q_values,
        )

    preview = []
    for transition, demo_value, demo_q_value, demo_advantage in zip(
        demo_transitions,
        demo_values,
        demo_q_values,
        demo_advantages,
    ):
        transition["demo_value"] = float(demo_value.detach().cpu().item())
        transition["demo_q_value"] = float(demo_q_value.detach().cpu().item())
        transition["demo_advantage"] = float(demo_advantage.detach().cpu().item())
        preview.append(
            {
                "trajectory_id": transition.get("trajectory_id"),
                "user_id": int(transition["user_id"]),
                "action_id": int(transition["action_id"]),
                "demo_return": float(transition["demo_return"]),
                "demo_value": float(transition["demo_value"]),
                "demo_q_value": float(transition["demo_q_value"]),
                "demo_advantage": float(transition["demo_advantage"]),
            }
        )
    return {
        "method": ILREC_DEMO_ADVANTAGE_METHOD,
        "fit_result": value_fit_result,
        "value_fit_result": value_fit_result,
        "q_fit_result": q_fit_result,
        "preview": preview[:3],
    }


def make_replay_buffers(args, demo_transitions):
    env_replay = ReplayBuffer(source="env")
    demo_replay = build_demo_replay_buffer(demo_transitions)
    return (
        env_replay,
        demo_replay,
        MixedReplayBuffer(
            env_replay,
            demo_replay,
            env_priority_scale=float(args.mixed_replay_env_priority_scale),
        ),
    )


def build_demo_replay_buffer(demo_transitions):
    demo_replay = DemoReplayBuffer()
    for transition in demo_transitions:
        demo_replay.add_demo(transition, weight=transition.get("demo_weight", 1.0))
    return demo_replay


def _history_after_transition(transition):
    history_actions, history_rewards = _transition_history(transition)
    history_actions = list(history_actions)
    history_rewards = list(history_rewards)
    history_actions.append(int(transition["action_id"]))
    history_rewards.append(float(transition.get("world_model_reward", transition.get("reward", 0.0))))
    return history_actions, history_rewards


def _state_features_for_records(model, records, device, use_next_state=False):
    if not records:
        return torch.empty((0, model.actor.in_features), dtype=torch.float32, device=device)
    user_ids = [int(record["user_id"]) for record in records]
    history_actions = []
    history_rewards = []
    for record in records:
        if use_next_state:
            actions, rewards = _history_after_transition(record)
        else:
            actions, rewards = _transition_history(record)
        history_actions.append(actions)
        history_rewards.append(rewards)
    return model.encode_state(
        torch.tensor(user_ids, dtype=torch.long, device=device),
        history_actions=history_actions,
        history_rewards=history_rewards,
    )


def _actor_inputs_for_records(records, device):
    if not records:
        raise ValueError("records must contain at least one transition.")
    user_ids = torch.tensor([int(record["user_id"]) for record in records], dtype=torch.long, device=device)
    actions = torch.tensor([int(record["action_id"]) for record in records], dtype=torch.long, device=device)
    history_rows = [_transition_history(record) for record in records]
    history_actions = [history[0] for history in history_rows]
    history_rewards = [history[1] for history in history_rows]
    return user_ids, actions, history_actions, history_rewards


def _discounted_returns(rewards, discount):
    if not rewards:
        raise ValueError("Cannot compute returns for an empty episode.")
    running = 0.0
    values = []
    for reward in reversed(rewards):
        running = float(reward) + float(discount) * running
        values.append(running)
    values.reverse()
    return torch.tensor(values, dtype=torch.float32)


def _masked_categorical(logits, valid_actions):
    if valid_actions >= logits.shape[-1]:
        return torch.distributions.Categorical(logits=logits)
    masked = logits.clone()
    masked[..., int(valid_actions) :] = torch.finfo(masked.dtype).min
    return torch.distributions.Categorical(logits=masked)


def _normalize_history_rows(history_action_rows, batch_size):
    if history_action_rows is None:
        return [[] for _ in range(batch_size)]
    if hasattr(history_action_rows, "detach"):
        history_action_rows = history_action_rows.detach().cpu().tolist()
    if isinstance(history_action_rows, np.ndarray):
        history_action_rows = history_action_rows.tolist()
    if batch_size == 1:
        if history_action_rows is None:
            return [[]]
        if not isinstance(history_action_rows, (list, tuple)):
            return [[history_action_rows]]
        if not history_action_rows:
            return [[]]
        if all(not isinstance(item, (list, tuple, np.ndarray)) for item in history_action_rows):
            return [list(history_action_rows)]
    if len(history_action_rows) != batch_size:
        raise ValueError("history_action_rows batch size must match action_logits.")
    rows = []
    for row in history_action_rows:
        if hasattr(row, "detach"):
            row = row.detach().cpu().tolist()
        if isinstance(row, np.ndarray):
            row = row.tolist()
        if row is None:
            rows.append([])
        elif isinstance(row, (list, tuple)):
            rows.append(list(row))
        else:
            rows.append([row])
    return rows


def apply_train_action_logit_shaping(
    action_logits,
    history_action_rows=None,
    valid_actions=None,
    repeat_penalty=0.0,
    distance_penalty=0.0,
    mat_distance=None,
    leave_threshold=None,
    num_leave_compute=4,
    row_mask=None,
):
    repeat_penalty = float(repeat_penalty)
    distance_penalty = float(distance_penalty)
    if repeat_penalty < 0 or distance_penalty < 0:
        raise ValueError("train action logit shaping penalties must be non-negative.")

    logits = torch.as_tensor(action_logits)
    squeeze = False
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    elif logits.ndim != 2:
        raise ValueError("action_logits must be 1D or 2D.")

    action_dim = int(logits.shape[1])
    valid_actions = action_dim if valid_actions is None else int(valid_actions)
    if valid_actions <= 0 or valid_actions > action_dim:
        raise ValueError("valid_actions must be in [1, action_logits.shape[-1]].")

    device = logits.device
    adjusted = logits.clone()
    zero = logits.new_tensor(0.0)
    if row_mask is None:
        active_rows = torch.ones(int(logits.shape[0]), dtype=torch.bool, device=device)
    else:
        active_rows = torch.as_tensor(row_mask, dtype=torch.bool, device=device)
        if active_rows.ndim != 1 or active_rows.shape[0] != logits.shape[0]:
            raise ValueError("row_mask must be 1D and match the logits batch size.")

    if repeat_penalty == 0.0 and distance_penalty == 0.0:
        adjusted_logits = adjusted.squeeze(0) if squeeze else adjusted
        return ActionLogitShapingResult(
            adjusted_logits=adjusted_logits,
            repeat_mass=zero,
            distance_mass=zero,
        )

    rows = _normalize_history_rows(history_action_rows, int(logits.shape[0]))
    repeat_masses = []
    distance_masses = []
    distance_array = None if mat_distance is None else np.asarray(mat_distance)
    threshold = None if leave_threshold is None else float(leave_threshold)
    recent_window = max(0, int(num_leave_compute))

    for row_index, history in enumerate(rows):
        if not bool(active_rows[row_index].item()):
            repeat_masses.append(zero)
            distance_masses.append(zero)
            continue

        valid_history = [int(action) for action in history if 0 <= int(action) < valid_actions]
        if repeat_penalty > 0.0 and valid_history:
            repeat_counts = torch.zeros(valid_actions, dtype=logits.dtype, device=device)
            for action in valid_history:
                repeat_counts[action] += 1.0
            adjusted[row_index, :valid_actions] = adjusted[row_index, :valid_actions] - repeat_penalty * repeat_counts
            repeat_masses.append(repeat_counts.mean())
        else:
            repeat_masses.append(zero)

        distance_weights = None
        if (
            distance_penalty > 0.0
            and distance_array is not None
            and threshold is not None
            and threshold > 0.0
            and recent_window > 0
        ):
            recent_history = valid_history[-recent_window:]
            closeness = []
            action_limit = min(valid_actions, int(distance_array.shape[0]))
            for history_item in recent_history:
                if history_item >= distance_array.shape[1] or action_limit <= 0:
                    continue
                weights = np.zeros(valid_actions, dtype=np.float32)
                distances = np.asarray(distance_array[:action_limit, history_item], dtype=np.float32)
                weights[:action_limit] = np.clip((threshold - distances) / threshold, 0.0, None)
                closeness.append(weights)
            if closeness:
                distance_weights = torch.as_tensor(
                    np.max(np.stack(closeness, axis=0), axis=0),
                    dtype=logits.dtype,
                    device=device,
                )
        if distance_weights is None:
            distance_masses.append(zero)
        else:
            adjusted[row_index, :valid_actions] = (
                adjusted[row_index, :valid_actions] - distance_penalty * distance_weights
            )
            distance_masses.append(distance_weights.mean())

    adjusted_logits = adjusted.squeeze(0) if squeeze else adjusted
    return ActionLogitShapingResult(
        adjusted_logits=adjusted_logits,
        repeat_mass=torch.stack(repeat_masses).mean(),
        distance_mass=torch.stack(distance_masses).mean(),
    )


def unsafe_action_probability_penalty(
    action_logits,
    history_action_rows=None,
    valid_actions=None,
    repeat_penalty=0.0,
    distance_penalty=0.0,
    mat_distance=None,
    leave_threshold=None,
    num_leave_compute=4,
):
    repeat_penalty = float(repeat_penalty)
    distance_penalty = float(distance_penalty)
    if repeat_penalty < 0 or distance_penalty < 0:
        raise ValueError("unsafe action penalties must be non-negative.")

    logits = torch.as_tensor(action_logits, dtype=torch.float32)
    if logits.ndim != 2:
        raise ValueError("action_logits must be a 2D tensor of shape [batch, actions].")
    action_dim = int(logits.shape[1])
    valid_actions = action_dim if valid_actions is None else int(valid_actions)
    if valid_actions <= 0 or valid_actions > action_dim:
        raise ValueError("valid_actions must be in [1, action_logits.shape[1]].")

    device = logits.device
    zero = logits.new_tensor(0.0)
    if repeat_penalty == 0.0 and distance_penalty == 0.0:
        return UnsafeActionPenalty(total_loss=zero, repeat_mass=zero, distance_mass=zero)

    rows = _normalize_history_rows(history_action_rows, int(logits.shape[0]))
    probs = torch.softmax(logits[:, :valid_actions], dim=-1)
    repeat_masses = []
    distance_masses = []
    distance_array = None if mat_distance is None else np.asarray(mat_distance)
    threshold = None if leave_threshold is None else float(leave_threshold)
    recent_window = max(0, int(num_leave_compute))

    for row_index, history in enumerate(rows):
        valid_history = [int(action) for action in history if 0 <= int(action) < valid_actions]
        if repeat_penalty > 0.0 and valid_history:
            repeat_counts = torch.zeros(valid_actions, dtype=probs.dtype, device=device)
            for action in valid_history:
                repeat_counts[action] += 1.0
            repeat_masses.append((probs[row_index] * repeat_counts).sum())
        else:
            repeat_masses.append(zero)

        distance_weights = None
        if (
            distance_penalty > 0.0
            and distance_array is not None
            and threshold is not None
            and threshold > 0.0
            and recent_window > 0
        ):
            recent_history = valid_history[-recent_window:]
            closeness = []
            for history_item in recent_history:
                if history_item >= distance_array.shape[1]:
                    continue
                distances = np.asarray(distance_array[:valid_actions, history_item], dtype=np.float32)
                closeness.append(np.clip((threshold - distances) / threshold, 0.0, None))
            if closeness:
                distance_weights = torch.as_tensor(
                    np.max(np.stack(closeness, axis=0), axis=0),
                    dtype=probs.dtype,
                    device=device,
                )
        if distance_weights is None:
            distance_masses.append(zero)
        else:
            distance_masses.append((probs[row_index] * distance_weights).sum())

    repeat_mass = torch.stack(repeat_masses).mean()
    distance_mass = torch.stack(distance_masses).mean()
    total_loss = repeat_penalty * repeat_mass + distance_penalty * distance_mass
    return UnsafeActionPenalty(
        total_loss=total_loss,
        repeat_mass=repeat_mass,
        distance_mass=distance_mass,
    )


def behavior_clone_kl_anchor_loss(action_logits, actions, is_demo=None, label_smoothing=0.0):
    label_smoothing = float(label_smoothing)
    if label_smoothing < 0.0 or label_smoothing >= 1.0:
        raise ValueError("train_bc_kl_label_smoothing must be in [0, 1).")
    logits = torch.as_tensor(action_logits, dtype=torch.float32)
    if logits.ndim != 2:
        raise ValueError("action_logits must be a 2D tensor of shape [batch, actions].")
    batch_size = int(logits.shape[0])
    actions = torch.as_tensor(actions, dtype=torch.long, device=logits.device)
    if actions.ndim != 1 or actions.shape[0] != batch_size:
        raise ValueError("actions must be 1D and match action_logits batch size.")
    if torch.any(actions < 0) or torch.any(actions >= logits.shape[1]):
        raise ValueError("actions contains an action outside the logits range.")

    if is_demo is None:
        demo_mask = torch.ones(batch_size, dtype=torch.bool, device=logits.device)
    else:
        demo_mask = torch.as_tensor(is_demo, dtype=torch.bool, device=logits.device)
        if demo_mask.ndim != 1 or demo_mask.shape[0] != batch_size:
            raise ValueError("is_demo must be 1D and match action_logits batch size.")
    if not bool(demo_mask.any().item()):
        return logits.new_tensor(0.0)
    return F.cross_entropy(
        logits[demo_mask],
        actions[demo_mask],
        label_smoothing=label_smoothing,
    )


def transform_policy_advantages(advantages, normalize=False, clip_value=0.0):
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    clip_value = float(clip_value)
    if clip_value < 0:
        raise ValueError("train_advantage_clip must be non-negative.")

    original = advantages
    if normalize and advantages.numel() > 1:
        mean = advantages.mean()
        std = advantages.std(unbiased=False).clamp_min(1e-8)
        advantages = (advantages - mean) / std
    if clip_value > 0.0:
        clipped_count = int((advantages.abs() > clip_value).sum().detach().cpu().item())
        advantages = advantages.clamp(min=-clip_value, max=clip_value)
    else:
        clipped_count = 0
    return AdvantageTransformResult(
        advantages=advantages,
        mean=original.mean() if original.numel() else original.new_tensor(0.0),
        std=original.std(unbiased=False) if original.numel() > 1 else original.new_tensor(0.0),
        clipped_count=clipped_count,
    )


def apply_policy_logit_clamp(logits, clamp_value=0.0, mode="tanh"):
    logits = torch.as_tensor(logits)
    clamp_value = float(clamp_value)
    if clamp_value < 0.0:
        raise ValueError("policy_logit_clamp must be non-negative.")
    if mode != "tanh":
        raise ValueError(f"Unsupported policy logit clamp mode: {mode}")
    if clamp_value == 0.0:
        return logits
    return clamp_value * torch.tanh(logits / clamp_value)


def project_actor_output_layer(model, row_norm_max=0.0, bias_clamp=0.0):
    row_norm_max = float(row_norm_max)
    bias_clamp = float(bias_clamp)
    if row_norm_max < 0.0:
        raise ValueError("train_actor_row_norm_project must be non-negative.")
    if bias_clamp < 0.0:
        raise ValueError("train_actor_bias_clamp must be non-negative.")
    if row_norm_max == 0.0 and bias_clamp == 0.0:
        return
    if not hasattr(model, "actor") or not hasattr(model.actor, "weight"):
        raise ValueError("actor projection requires model.actor.weight.")
    with torch.no_grad():
        if row_norm_max > 0.0:
            weight = model.actor.weight
            norms = weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
            scale = torch.clamp(row_norm_max / norms, max=1.0)
            weight.mul_(scale)
        if bias_clamp > 0.0:
            if getattr(model.actor, "bias", None) is None:
                raise ValueError("actor bias clamp requires model.actor.bias.")
            model.actor.bias.clamp_(min=-bias_clamp, max=bias_clamp)


def compute_entropy_floor_penalty(logits, valid_actions, floor_ratio=0.0, coef=0.0):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    if logits.ndim != 2:
        raise ValueError("logits must be a 2D tensor.")
    valid_actions = int(valid_actions)
    floor_ratio = float(floor_ratio)
    coef = float(coef)
    if floor_ratio < 0.0 or floor_ratio > 1.0:
        raise ValueError("train_entropy_floor_ratio must be in [0, 1].")
    if coef < 0.0:
        raise ValueError("train_entropy_floor_coef must be non-negative.")
    if floor_ratio == 0.0 or coef == 0.0:
        zero = logits.new_tensor(0.0)
        return EntropyFloorPenalty(
            total_loss=zero,
            entropy_mean=zero,
            entropy_ratio_mean=zero,
            entropy_floor_gap=zero,
        )
    if valid_actions <= 1:
        raise ValueError("valid_actions must be greater than 1 for entropy floor.")
    if valid_actions > logits.shape[-1]:
        raise ValueError("valid_actions cannot exceed logits dimension.")

    active_logits = logits[:, :valid_actions]
    probs = F.softmax(active_logits, dim=-1)
    log_probs = F.log_softmax(active_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    max_entropy = math.log(valid_actions)
    entropy_ratio = entropy / max_entropy
    floor_gap = F.relu(entropy.new_tensor(floor_ratio) - entropy_ratio)
    total_loss = coef * floor_gap.square().mean()
    floor_gap_mean = floor_gap.mean()
    return EntropyFloorPenalty(
        total_loss=total_loss,
        entropy_mean=entropy.mean(),
        entropy_ratio_mean=entropy_ratio.mean(),
        entropy_floor_gap=floor_gap_mean,
    )


def compute_logit_scale_penalty(
    logits,
    model,
    logit_l2_penalty=0.0,
    logit_std_penalty=0.0,
    logit_std_target=10.0,
    actor_row_norm_penalty=0.0,
    actor_row_norm_target=10.0,
):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    if logits.ndim != 2:
        raise ValueError("logits must be a 2D tensor.")

    logit_l2_penalty = float(logit_l2_penalty)
    logit_std_penalty = float(logit_std_penalty)
    logit_std_target = float(logit_std_target)
    actor_row_norm_penalty = float(actor_row_norm_penalty)
    actor_row_norm_target = float(actor_row_norm_target)
    if min(logit_l2_penalty, logit_std_penalty, actor_row_norm_penalty) < 0.0:
        raise ValueError("logit scale penalty coefficients must be non-negative.")
    if logit_std_target < 0.0 or actor_row_norm_target < 0.0:
        raise ValueError("logit scale penalty targets must be non-negative.")

    zero = logits.new_tensor(0.0)
    logit_l2 = logits.square().mean() if logit_l2_penalty > 0.0 else zero
    row_std = logits.std(dim=1, unbiased=False)
    logit_std_mean = row_std.mean()
    if logit_std_penalty > 0.0:
        logit_std_excess = F.relu(row_std - logit_std_target).square().mean()
    else:
        logit_std_excess = zero

    if hasattr(model, "actor") and hasattr(model.actor, "weight"):
        row_norms = model.actor.weight.norm(dim=1)
        actor_row_norm_max = row_norms.max()
        if actor_row_norm_penalty > 0.0:
            violating_rows = row_norms > actor_row_norm_target
            if bool(violating_rows.any().item()):
                actor_row_norm_excess = (row_norms[violating_rows] - actor_row_norm_target).square().mean()
            else:
                actor_row_norm_excess = zero
        else:
            actor_row_norm_excess = zero
    elif actor_row_norm_penalty > 0.0:
        raise ValueError("actor_row_norm_penalty requires model.actor.weight.")
    else:
        actor_row_norm_max = zero
        actor_row_norm_excess = zero

    total_loss = (
        logit_l2_penalty * logit_l2
        + logit_std_penalty * logit_std_excess
        + actor_row_norm_penalty * actor_row_norm_excess
    )
    return LogitScalePenalty(
        total_loss=total_loss,
        logit_l2=logit_l2,
        logit_std_excess=logit_std_excess,
        actor_row_norm_excess=actor_row_norm_excess,
        logit_std_mean=logit_std_mean,
        actor_row_norm_max=actor_row_norm_max,
    )


def summarize_rollout_diagnostics(rollouts):
    lengths = []
    all_actions = []
    first_actions = []
    termination_reasons = []
    repeat_after_first_count = 0
    action_after_first_count = 0
    terminal_exact_repeat_episodes = 0

    for rollout in rollouts:
        steps = list(rollout.get("steps", []))
        actions = [int(step["action_id"]) for step in steps if "action_id" in step]
        lengths.append(len(actions))
        all_actions.extend(actions)
        if actions:
            first_actions.append(actions[0])
        seen = set()
        for index, action in enumerate(actions):
            if index > 0:
                action_after_first_count += 1
                if action in seen:
                    repeat_after_first_count += 1
            seen.add(action)
        if actions and actions[-1] in set(actions[:-1]):
            terminal_exact_repeat_episodes += 1

        reason = rollout.get("termination_reason")
        if reason is None and steps:
            reason = steps[-1].get("reason")
        termination_reasons.append(str(reason or "unknown"))

    rollout_count = len(rollouts)
    total_steps = len(all_actions)
    repeat_rate = (
        float(repeat_after_first_count) / float(action_after_first_count)
        if action_after_first_count
        else 0.0
    )
    return {
        "rollout_count": int(rollout_count),
        "total_steps": int(total_steps),
        "avg_length": float(sum(lengths) / rollout_count) if rollout_count else 0.0,
        "length_counts": {str(key): int(value) for key, value in Counter(lengths).items()},
        "termination_reasons": {str(key): int(value) for key, value in Counter(termination_reasons).items()},
        "unique_actions": int(len(set(all_actions))),
        "unique_first_actions": int(len(set(first_actions))),
        "top_first_actions": {
            str(key): int(value)
            for key, value in Counter(first_actions).most_common(10)
        },
        "repeat_after_first_count": int(repeat_after_first_count),
        "action_after_first_count": int(action_after_first_count),
        "repeat_after_first_rate": float(repeat_rate),
        "terminal_exact_repeat_episodes": int(terminal_exact_repeat_episodes),
    }


def select_evaluation_action(
    logits,
    valid_actions,
    action_selection="sample",
):
    logits = torch.as_tensor(logits)
    if logits.ndim == 2:
        if logits.shape[0] != 1:
            raise ValueError("Evaluation action selection expects a single-row logits tensor.")
        logits = logits.squeeze(0)
    if logits.ndim != 1:
        raise ValueError("Evaluation action selection expects 1D logits.")

    valid_actions = int(valid_actions)
    if valid_actions <= 0:
        raise ValueError("valid_actions must be positive.")
    if valid_actions > logits.shape[-1]:
        raise ValueError("valid_actions cannot exceed logits dimension.")
    if action_selection not in {"sample"}:
        raise ValueError(f"Unsupported evaluation action selection: {action_selection}")

    def _select_from_logits(candidate_logits):
        return int(torch.distributions.Categorical(logits=candidate_logits).sample().item())

    return _select_from_logits(logits[:valid_actions])


def _select_user_id(obs):
    return int(np.asarray(obs).reshape(-1)[0])


def _step_env(env, action):
    obs, reward, done, info = env.step(int(action))
    return obs, float(reward), bool(done), dict(info)


def _parse_user_ids(value):
    if value is None:
        return []
    user_ids = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            user_ids.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid user id {part!r} in comma-separated user ids.") from exc
    return user_ids


def _extract_json_user_ids(payload, path):
    if isinstance(payload, dict):
        if "user_ids" in payload:
            payload = payload["user_ids"]
        else:
            payload = list(payload.values())

    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list of user records or a user_ids list.")

    user_ids = []
    for index, record in enumerate(payload):
        if isinstance(record, dict):
            for key in ("userid_encoded", "user_id", "userid"):
                if key in record:
                    user_ids.append(int(record[key]))
                    break
            else:
                raise ValueError(f"{path} record {index} does not contain a user id field.")
        else:
            user_ids.append(int(record))
    return user_ids


def load_user_ids_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return _extract_json_user_ids(payload, path)


def resolve_eval_user_ids(args):
    user_ids = []
    if args.eval_users_json is not None:
        user_ids.extend(load_user_ids_json(args.eval_users_json))
    user_ids.extend(_parse_user_ids(args.eval_user_ids))
    return user_ids or None


def _validate_user_ids(user_ids, num_users, label):
    if user_ids is None:
        return None
    validated = []
    for user_id in user_ids:
        user_id = int(user_id)
        if user_id < 0 or user_id >= num_users:
            raise ValueError(f"{label} contains user_id {user_id} outside [0, {num_users})")
        validated.append(user_id)
    if not validated:
        raise ValueError(f"{label} cannot be empty when provided.")
    return validated


def run_smoke_update(args, demo_buffer):
    batch = _demo_batch(demo_buffer["transitions"], args.smoke_batch_size)
    action_ids = torch.tensor([int(transition["action_id"]) for transition in batch], dtype=torch.long)
    rewards = torch.tensor([float(transition.get("reward", 0.0)) for transition in batch], dtype=torch.float32)
    action_dim = max(2, int(action_ids.max().item()) + 1)
    batch_size = len(batch)

    policy_logits = torch.nn.Parameter(torch.zeros(batch_size, action_dim))
    values = torch.nn.Parameter(torch.zeros(batch_size))
    demo_logits = torch.nn.Parameter(torch.zeros(batch_size, action_dim))
    optimizer = torch.optim.Adam([policy_logits, values, demo_logits], lr=args.lr)

    last_loss = None
    for _ in range(max(1, int(args.smoke_steps))):
        optimizer.zero_grad()
        discriminator_probs = torch.full((batch_size,), 0.5)
        demo_weights = compute_demo_weights(
            demo_advantages=rewards,
            discriminator_probs=discriminator_probs,
            beta=args.beta,
            alpha=args.alpha,
            irl_gamma=1.0,
        ).weights
        loss_parts = compute_ilrec_policy_loss(
            action_logits=policy_logits,
            actions=action_ids,
            advantages=rewards,
            values=values,
            returns=rewards,
            demo_action_logits=demo_logits,
            demo_actions=action_ids,
            demo_weights=demo_weights,
            lambda_imit=args.lambda_imit,
            vf_coef=0.5,
            alpha_ent=args.alpha_ent,
        )
        loss_parts.total_loss.backward()
        optimizer.step()
        last_loss = loss_parts

    return {
        "loss": float(last_loss.total_loss.detach().cpu().item()),
        "actor_loss": float(last_loss.actor_loss.detach().cpu().item()),
        "value_loss": float(last_loss.value_loss.detach().cpu().item()),
        "entropy_loss": float(last_loss.entropy_loss.detach().cpu().item()),
        "imitation_loss": float(last_loss.imitation_loss.detach().cpu().item()),
        "action_dim": action_dim,
        "batch_size": batch_size,
    }


def _checkpoint_float(checkpoint, key, default=None):
    if not checkpoint:
        return default
    value = checkpoint.get(key)
    if value is None:
        return default
    return float(value)


def train_direct_actor_critic(args, demo_buffer, artifacts, loaded_checkpoint=None):
    train_env = make_policy_env(args, artifacts, args.train_env_source, split="train")
    eval_env = make_policy_env(args, artifacts, args.eval_env_source, split="test")
    num_users = int(train_env.num_user)
    num_items = int(train_env.num_item)
    demo_transitions = _usable_demo_transitions(demo_buffer["transitions"], num_users, num_items)
    demo_return_result = annotate_world_model_demo_returns(
        demo_transitions,
        train_env.lookup_rewards,
        discount=args.discount,
    )
    demo_transitions = demo_return_result.transitions
    eval_user_ids = _validate_user_ids(resolve_eval_user_ids(args), eval_env.num_user, "eval user ids")
    device = resolve_device(args.cuda)
    model = UserActorCritic(
        num_users,
        num_items,
        int(args.hidden_size),
        window_size=int(args.window_size),
        state_tracker_type=args.state_tracker_type,
        num_att_heads=int(args.state_tracker_att_heads),
        num_att_layers=int(args.state_tracker_att_layers),
        att_dropout=float(args.state_tracker_att_dropout),
    ).to(device)
    feature_builder = make_discriminator_feature_builder(
        args,
        demo_transitions,
        num_users,
        num_items,
        device,
        state_model=model,
    )
    demo_advantage_result = prepare_demo_advantages(
        args,
        feature_builder,
        demo_transitions,
        device,
        state_model=model,
    )
    env_replay, demo_replay, mixed_replay = make_replay_buffers(args, demo_transitions)
    discriminator = TransitionDiscriminator(
        input_dim=feature_builder.input_dim,
        hidden_sizes=(int(args.disc_hidden_size),),
    ).to(device)
    checkpoint_loaded_for_policy = _load_policy_checkpoint(model, discriminator, loaded_checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=float(args.disc_lr))
    qv_critic = None
    target_qv_critic = None
    qv_optimizer = None
    qv_update_count = 0
    target_qv_update_count = 0
    qv_demo_weighted_samples = 0
    last_qv_loss = None
    mixed_policy_update_count = 0
    mixed_policy_sample_count = 0
    mixed_policy_demo_sample_count = 0
    mixed_policy_env_sample_count = 0
    terminal_repeat_td_penalized_samples = 0
    qv_critic = StateQVCritic(
        input_dim=int(args.hidden_size),
        num_actions=num_items,
        hidden_size=int(args.disc_hidden_size),
    ).to(device)
    target_qv_critic = StateQVCritic(
        input_dim=int(args.hidden_size),
        num_actions=num_items,
        hidden_size=int(args.disc_hidden_size),
    ).to(device)
    target_qv_critic.load_state_dict(qv_critic.state_dict())
    qv_optimizer = torch.optim.Adam(qv_critic.parameters(), lr=float(args.qv_lr))

    episode_summaries = []
    transition_count = 0
    last_policy_loss = None
    last_regularized_total_loss = None
    last_disc_loss = None
    last_unsafe_action_penalty = None
    last_bc_kl_loss = None
    last_train_action_shaping = None
    last_logit_scale_penalty = None
    last_entropy_floor_penalty = None
    last_advantage_transform = None
    advantage_clipped_sample_count = 0
    unsafe_action_update_count = 0
    bc_kl_update_count = 0
    logit_scale_update_count = 0
    entropy_floor_update_count = 0
    train_action_shaping_sample_count = 0
    train_action_shaping_policy_update_count = 0
    discriminator_update_count = 0
    demo_weight_refresh_count = 0
    demo_offset = 0
    train_episodes = int(args.train_episodes)
    if train_episodes < 0:
        raise ValueError("train_episodes must be non-negative.")
    if float(args.train_continue_bonus) < 0:
        raise ValueError("train_continue_bonus must be non-negative.")
    if float(args.train_distance_terminal_penalty) < 0:
        raise ValueError("train_distance_terminal_penalty must be non-negative.")
    if float(args.train_terminal_repeat_td_penalty) < 0:
        raise ValueError("train_terminal_repeat_td_penalty must be non-negative.")
    if float(args.train_unsafe_repeat_penalty) < 0:
        raise ValueError("train_unsafe_repeat_penalty must be non-negative.")
    if float(args.train_unsafe_distance_penalty) < 0:
        raise ValueError("train_unsafe_distance_penalty must be non-negative.")
    if float(args.train_action_repeat_penalty) < 0:
        raise ValueError("train_action_repeat_penalty must be non-negative.")
    if float(args.train_action_distance_penalty) < 0:
        raise ValueError("train_action_distance_penalty must be non-negative.")
    if float(args.train_bc_kl_coef) < 0:
        raise ValueError("train_bc_kl_coef must be non-negative.")
    if float(args.train_bc_kl_label_smoothing) < 0 or float(args.train_bc_kl_label_smoothing) >= 1.0:
        raise ValueError("train_bc_kl_label_smoothing must be in [0, 1).")
    if float(args.train_logit_l2_penalty) < 0:
        raise ValueError("train_logit_l2_penalty must be non-negative.")
    if float(args.train_logit_std_penalty) < 0:
        raise ValueError("train_logit_std_penalty must be non-negative.")
    if float(args.train_logit_std_target) < 0:
        raise ValueError("train_logit_std_target must be non-negative.")
    if float(args.train_actor_row_norm_penalty) < 0:
        raise ValueError("train_actor_row_norm_penalty must be non-negative.")
    if float(args.train_actor_row_norm_target) < 0:
        raise ValueError("train_actor_row_norm_target must be non-negative.")
    if float(args.train_advantage_clip) < 0:
        raise ValueError("train_advantage_clip must be non-negative.")
    if float(args.policy_logit_clamp) < 0:
        raise ValueError("policy_logit_clamp must be non-negative.")
    if float(args.train_actor_row_norm_project) < 0:
        raise ValueError("train_actor_row_norm_project must be non-negative.")
    if float(args.train_actor_bias_clamp) < 0:
        raise ValueError("train_actor_bias_clamp must be non-negative.")
    if float(args.train_entropy_floor_ratio) < 0 or float(args.train_entropy_floor_ratio) > 1:
        raise ValueError("train_entropy_floor_ratio must be in [0, 1].")
    if float(args.train_entropy_floor_coef) < 0:
        raise ValueError("train_entropy_floor_coef must be non-negative.")
    if float(args.mixed_replay_env_priority_scale) < 0:
        raise ValueError("mixed_replay_env_priority_scale must be non-negative.")
    if train_episodes == 0 and not checkpoint_loaded_for_policy:
        raise ValueError("train_episodes=0 requires --checkpoint with model_state_dict.")

    for episode_idx in range(train_episodes):
        obs = train_env.reset(user_id=episode_idx % num_users)
        logits_list = []
        values_list = []
        actions = []
        rewards = []
        train_rewards = []
        infos = []
        users = []
        done = False
        history_actions = []
        history_rewards = []
        policy_history_actions = []
        policy_history_rewards = []

        while not done and len(actions) < train_env.max_turn:
            user_id = _select_user_id(obs)
            current_history_actions = list(history_actions)
            current_history_rewards = list(history_rewards)
            logits, value = model(
                torch.tensor([user_id], dtype=torch.long, device=device),
                history_actions=current_history_actions,
                history_rewards=current_history_rewards,
            )
            policy_logits = apply_policy_logit_clamp(
                logits,
                clamp_value=args.policy_logit_clamp,
                mode=args.policy_logit_clamp_mode,
            )
            action_shaping = apply_train_action_logit_shaping(
                policy_logits.squeeze(0),
                history_action_rows=current_history_actions,
                valid_actions=train_env.num_item,
                repeat_penalty=args.train_action_repeat_penalty,
                distance_penalty=args.train_action_distance_penalty,
                mat_distance=getattr(train_env, "mat_distance", None),
                leave_threshold=getattr(train_env, "leave_threshold", None),
                num_leave_compute=getattr(train_env, "num_leave_compute", args.num_leave_compute),
            )
            if float(args.train_action_repeat_penalty) > 0.0 or float(args.train_action_distance_penalty) > 0.0:
                train_action_shaping_sample_count += 1
                last_train_action_shaping = action_shaping
            dist = _masked_categorical(action_shaping.adjusted_logits, train_env.num_item)
            action = dist.sample()
            next_obs, reward, done, info = _step_env(train_env, int(action.item()))
            train_reward = shape_training_reward(
                reward,
                done,
                info,
                continue_bonus=args.train_continue_bonus,
                distance_terminal_penalty=args.train_distance_terminal_penalty,
            )

            users.append(user_id)
            actions.append(int(action.item()))
            rewards.append(float(reward))
            train_rewards.append(float(train_reward))
            infos.append(dict(info))
            policy_history_actions.append(current_history_actions)
            policy_history_rewards.append(current_history_rewards)
            logits_list.append(policy_logits.squeeze(0))
            values_list.append(value.squeeze(0))
            history_actions.append(int(action.item()))
            history_rewards.append(float(reward))
            obs = next_obs

        if not actions:
            continue
        policy_records = _policy_transition_records(
            users,
            actions,
            rewards,
            policy_history_actions,
            policy_history_rewards,
            train_rewards=train_rewards,
            infos=infos,
        )
        for step_index, record in enumerate(policy_records):
            record["trajectory_id"] = f"env-{episode_idx}"
            record["step_index"] = step_index
            record["done"] = step_index == len(policy_records) - 1 and bool(done)
        if env_replay is not None:
            env_replay.extend(policy_records)

        demo_batch = _cycled_demo_batch(demo_transitions, args.demo_batch_size, demo_offset)
        demo_offset += len(demo_batch)
        demo_users = [int(transition["user_id"]) for transition in demo_batch]
        demo_actions = [int(transition["action_id"]) for transition in demo_batch]
        demo_rewards = [float(transition.get("reward", 0.0)) for transition in demo_batch]
        demo_histories = [_transition_history(transition) for transition in demo_batch]
        demo_history_actions = [history[0] for history in demo_histories]
        demo_history_rewards = [history[1] for history in demo_histories]

        discriminator_retrain_interval = max(1, int(args.discriminator_retrain_interval))
        should_retrain_discriminator = episode_idx % discriminator_retrain_interval == 0
        demo_features = None
        if should_retrain_discriminator:
            policy_features = feature_builder.policy_features(
                users,
                actions,
                rewards,
                policy_history_actions,
                policy_history_rewards,
            )
            demo_features = feature_builder.demo_features(demo_batch)
            disc_features = torch.cat([demo_features, policy_features], dim=0)
            disc_labels = torch.cat(
                [
                    torch.zeros(len(demo_features), dtype=torch.float32, device=device),
                    torch.ones(len(policy_features), dtype=torch.float32, device=device),
                ],
                dim=0,
            )
            discriminator_optimizer.zero_grad()
            disc_loss = discriminator.bce_loss(disc_features, disc_labels)
            disc_loss.backward()
            discriminator_optimizer.step()
            discriminator_update_count += 1
            last_disc_loss = disc_loss

        returns = _discounted_returns(train_rewards, args.discount).to(device)
        action_logits = torch.stack(logits_list)
        values = torch.stack(values_list)
        action_tensor = torch.tensor(actions, dtype=torch.long, device=device)
        advantages = returns - values.detach()

        demo_user_tensor = torch.tensor(demo_users, dtype=torch.long, device=device)
        demo_action_tensor = torch.tensor(demo_actions, dtype=torch.long, device=device)
        demo_reward_tensor = torch.tensor(demo_rewards, dtype=torch.float32, device=device)
        demo_logits, demo_values = model(
            demo_user_tensor,
            history_actions=demo_history_actions,
            history_rewards=demo_history_rewards,
        )
        demo_logits = apply_policy_logit_clamp(
            demo_logits,
            clamp_value=args.policy_logit_clamp,
            mode=args.policy_logit_clamp_mode,
        )
        with torch.no_grad():
            if should_retrain_discriminator:
                demo_disc_probs = discriminator(demo_features)
                demo_advantages = torch.tensor(
                    [float(transition["demo_advantage"]) for transition in demo_batch],
                    dtype=torch.float32,
                    device=device,
                )
                demo_weights = compute_demo_weights(
                    demo_advantages=demo_advantages,
                    discriminator_probs=demo_disc_probs,
                    beta=args.beta,
                    alpha=args.alpha,
                    irl_gamma=args.irl_gamma,
                ).weights
                if args.demo_weight_mode == "uniform":
                    demo_weights = torch.ones_like(demo_weights)
                for transition, weight in zip(demo_batch, demo_weights.detach().cpu().tolist()):
                    transition["demo_weight"] = float(weight)
                demo_replay = build_demo_replay_buffer(demo_transitions)
                mixed_replay = MixedReplayBuffer(
                    env_replay,
                    demo_replay,
                    env_priority_scale=float(args.mixed_replay_env_priority_scale),
                )
                demo_weight_refresh_count += 1
            else:
                demo_weights = torch.tensor(
                    [float(transition.get("demo_weight", 1.0)) for transition in demo_batch],
                    dtype=torch.float32,
                    device=device,
                )

        qv_records = mixed_replay.sample(
            args.mixed_replay_batch_size,
            demo_fraction=args.mixed_replay_demo_fraction,
            sampling_mode=args.mixed_replay_sampling,
            seed=int(args.seed) + episode_idx + 10_000_000,
        )
        if not qv_records:
            raise RuntimeError("Mixed replay produced no samples for ILRec critic update.")
        qv_state_features = _state_features_for_records(model, qv_records, device).detach()
        qv_next_state_features = _state_features_for_records(
            model,
            qv_records,
            device,
            use_next_state=True,
        ).detach()
        qv_actions = torch.tensor([int(record["action_id"]) for record in qv_records], dtype=torch.long, device=device)
        qv_rewards = torch.tensor(
            [
                _record_td_target_reward(
                    record,
                    terminal_repeat_penalty=args.train_terminal_repeat_td_penalty,
                )
                for record in qv_records
            ],
            dtype=torch.float32,
            device=device,
        )
        if float(args.train_terminal_repeat_td_penalty) > 0.0:
            terminal_repeat_td_penalized_samples += sum(
                1 for record in qv_records if _is_terminal_exact_repeat_distance_record(record)
            )
        qv_dones = torch.tensor([bool(record.get("done", False)) for record in qv_records], dtype=torch.bool, device=device)
        qv_sample_weights = torch.tensor(
            [
                float(record.get("demo_weight", 1.0))
                if record.get("source") == "demo"
                else 1.0
                for record in qv_records
            ],
            dtype=torch.float32,
            device=device,
        )
        qv_demo_weighted_samples += sum(1 for record in qv_records if record.get("source") == "demo")
        last_qv_loss = qv_critic_training_step(
            qv_critic,
            target_qv_critic,
            qv_optimizer,
            state_features=qv_state_features,
            actions=qv_actions,
            rewards=qv_rewards,
            next_state_features=qv_next_state_features,
            dones=qv_dones,
            discount=args.discount,
            sample_weights=qv_sample_weights,
        )
        qv_update_count += 1
        target_update_interval = max(1, int(args.target_update_interval))
        if qv_update_count % target_update_interval == 0:
            update_target_critic(target_qv_critic, qv_critic, tau=float(args.target_update_tau))
            target_qv_update_count += 1

        optimizer.zero_grad()
        mixed_samples = mixed_replay.sample(
            args.mixed_replay_batch_size,
            demo_fraction=args.mixed_replay_demo_fraction,
            sampling_mode=args.mixed_replay_sampling,
            seed=int(args.seed) + episode_idx,
        )
        if not mixed_samples:
            raise RuntimeError("Mixed replay produced no samples for ILRec policy update.")
        (
            mixed_user_tensor,
            mixed_action_tensor,
            mixed_history_actions,
            mixed_history_rewards,
        ) = _actor_inputs_for_records(mixed_samples, device)
        mixed_logits, _ = model(
            mixed_user_tensor,
            history_actions=mixed_history_actions,
            history_rewards=mixed_history_rewards,
        )
        mixed_logits = apply_policy_logit_clamp(
            mixed_logits,
            clamp_value=args.policy_logit_clamp,
            mode=args.policy_logit_clamp_mode,
        )
        with torch.no_grad():
            mixed_state_features = _state_features_for_records(model, mixed_samples, device).detach()
            mixed_q_values, mixed_values = qv_critic(mixed_state_features)
            mixed_selected_q = mixed_q_values.gather(1, mixed_action_tensor.unsqueeze(1)).squeeze(1)
            mixed_advantages = mixed_selected_q - mixed_values
            mixed_advantage_transform = transform_policy_advantages(
                mixed_advantages,
                normalize=bool(args.train_normalize_advantages),
                clip_value=args.train_advantage_clip,
            )
            mixed_advantages = mixed_advantage_transform.advantages.to(device)
            last_advantage_transform = mixed_advantage_transform
            advantage_clipped_sample_count += int(mixed_advantage_transform.clipped_count)
        mixed_is_demo = torch.tensor(
            [record.get("source") == "demo" for record in mixed_samples],
            dtype=torch.bool,
            device=device,
        )
        mixed_demo_weights = torch.tensor(
            [float(record.get("demo_weight", 1.0)) for record in mixed_samples],
            dtype=torch.float32,
            device=device,
        )
        mixed_policy_logits = mixed_logits
        if (
            args.train_action_shaping_loss != "none"
            and (
                float(args.train_action_repeat_penalty) > 0.0
                or float(args.train_action_distance_penalty) > 0.0
            )
        ):
            row_mask = None
            if args.train_action_shaping_loss == "env_only":
                row_mask = ~mixed_is_demo
            mixed_action_shaping = apply_train_action_logit_shaping(
                mixed_logits,
                history_action_rows=mixed_history_actions,
                valid_actions=train_env.num_item,
                repeat_penalty=args.train_action_repeat_penalty,
                distance_penalty=args.train_action_distance_penalty,
                mat_distance=getattr(train_env, "mat_distance", None),
                leave_threshold=getattr(train_env, "leave_threshold", None),
                num_leave_compute=getattr(train_env, "num_leave_compute", args.num_leave_compute),
                row_mask=row_mask,
            )
            mixed_policy_logits = mixed_action_shaping.adjusted_logits
            last_train_action_shaping = mixed_action_shaping
            train_action_shaping_policy_update_count += 1
        regularization_logits = mixed_logits
        regularization_history_actions = mixed_history_actions
        bc_logits = mixed_logits
        bc_actions = mixed_action_tensor
        bc_is_demo = mixed_is_demo
        policy_loss = compute_mixed_replay_policy_loss(
            action_logits=mixed_policy_logits,
            actions=mixed_action_tensor,
            advantages=mixed_advantages,
            is_demo=mixed_is_demo,
            demo_weights=mixed_demo_weights,
            lambda_imit=args.lambda_imit,
            alpha_ent=args.alpha_ent,
        )
        mixed_policy_update_count += 1
        mixed_policy_sample_count += int(policy_loss.sample_count)
        mixed_policy_demo_sample_count += int(policy_loss.demo_sample_count)
        mixed_policy_env_sample_count += int(policy_loss.env_sample_count)
        unsafe_penalty = unsafe_action_probability_penalty(
            regularization_logits,
            history_action_rows=regularization_history_actions,
            valid_actions=train_env.num_item,
            repeat_penalty=args.train_unsafe_repeat_penalty,
            distance_penalty=args.train_unsafe_distance_penalty,
            mat_distance=getattr(train_env, "mat_distance", None),
            leave_threshold=getattr(train_env, "leave_threshold", None),
            num_leave_compute=getattr(train_env, "num_leave_compute", args.num_leave_compute),
        )
        bc_kl_loss = behavior_clone_kl_anchor_loss(
            bc_logits,
            bc_actions,
            is_demo=bc_is_demo,
            label_smoothing=args.train_bc_kl_label_smoothing,
        )
        logit_scale_penalty = compute_logit_scale_penalty(
            regularization_logits,
            model,
            logit_l2_penalty=args.train_logit_l2_penalty,
            logit_std_penalty=args.train_logit_std_penalty,
            logit_std_target=args.train_logit_std_target,
            actor_row_norm_penalty=args.train_actor_row_norm_penalty,
            actor_row_norm_target=args.train_actor_row_norm_target,
        )
        entropy_floor_penalty = compute_entropy_floor_penalty(
            regularization_logits,
            valid_actions=train_env.num_item,
            floor_ratio=args.train_entropy_floor_ratio,
            coef=args.train_entropy_floor_coef,
        )
        regularized_total_loss = (
            policy_loss.total_loss
            + unsafe_penalty.total_loss
            + float(args.train_bc_kl_coef) * bc_kl_loss
            + logit_scale_penalty.total_loss
            + entropy_floor_penalty.total_loss
        )
        regularized_total_loss.backward()
        optimizer.step()
        project_actor_output_layer(
            model,
            row_norm_max=args.train_actor_row_norm_project,
            bias_clamp=args.train_actor_bias_clamp,
        )

        transition_count += len(actions)
        last_policy_loss = policy_loss
        last_regularized_total_loss = regularized_total_loss.detach()
        last_unsafe_action_penalty = unsafe_penalty
        last_bc_kl_loss = bc_kl_loss.detach()
        last_logit_scale_penalty = logit_scale_penalty
        last_entropy_floor_penalty = entropy_floor_penalty
        if float(args.train_unsafe_repeat_penalty) > 0.0 or float(args.train_unsafe_distance_penalty) > 0.0:
            unsafe_action_update_count += 1
        if float(args.train_bc_kl_coef) > 0.0:
            bc_kl_update_count += 1
        if (
            float(args.train_logit_l2_penalty) > 0.0
            or float(args.train_logit_std_penalty) > 0.0
            or float(args.train_actor_row_norm_penalty) > 0.0
        ):
            logit_scale_update_count += 1
        if float(args.train_entropy_floor_coef) > 0.0 and float(args.train_entropy_floor_ratio) > 0.0:
            entropy_floor_update_count += 1
        episode_summaries.append(
            {
                "episode": episode_idx,
                "length": len(actions),
                "return": float(sum(rewards)),
                "train_return": float(sum(train_rewards)),
                "last_reward": float(rewards[-1]),
                "last_train_reward": float(train_rewards[-1]),
                "unsafe_action_loss": float(unsafe_penalty.total_loss.detach().cpu().item()),
                "unsafe_repeat_mass": float(unsafe_penalty.repeat_mass.detach().cpu().item()),
                "unsafe_distance_mass": float(unsafe_penalty.distance_mass.detach().cpu().item()),
                "bc_kl_loss": float(bc_kl_loss.detach().cpu().item()),
                "logit_scale_loss": float(logit_scale_penalty.total_loss.detach().cpu().item()),
                "entropy_floor_loss": float(entropy_floor_penalty.total_loss.detach().cpu().item()),
            }
        )

    if train_episodes > 0 and last_policy_loss is None:
        raise RuntimeError("Training produced no policy transitions.")

    rollouts = rollout_policy(
        model,
        eval_env,
        int(args.eval_episodes),
        user_ids=eval_user_ids,
        action_selection=args.eval_action_selection,
        logit_clamp=args.policy_logit_clamp,
        logit_clamp_mode=args.policy_logit_clamp_mode,
    )
    rollout_user_ids = [int(rollout["userid"]) for rollout in rollouts]
    rollout_diagnostics = summarize_rollout_diagnostics(rollouts)
    rollout_path = write_rollout_json(args, rollouts)
    if last_policy_loss is None:
        loss = _checkpoint_float(loaded_checkpoint, "loss")
        regularized_loss = loss
        actor_loss = _checkpoint_float(loaded_checkpoint, "actor_loss")
        value_loss = _checkpoint_float(loaded_checkpoint, "value_loss")
        entropy_loss = _checkpoint_float(loaded_checkpoint, "entropy_loss")
        imitation_loss = _checkpoint_float(loaded_checkpoint, "imitation_loss")
        discriminator_loss = _checkpoint_float(loaded_checkpoint, "discriminator_loss")
    else:
        loss = float(last_regularized_total_loss.detach().cpu().item())
        regularized_loss = loss
        actor_loss = float(last_policy_loss.actor_loss.detach().cpu().item())
        value_loss = float(last_policy_loss.value_loss.detach().cpu().item())
        entropy_loss = float(last_policy_loss.entropy_loss.detach().cpu().item())
        imitation_loss = float(last_policy_loss.imitation_loss.detach().cpu().item())
        discriminator_loss = float(last_disc_loss.detach().cpu().item())
    state_action_embedding_cache_saved = feature_builder.save()
    payload = {
        "trainer_type": "ilrec_direct_actor_critic",
        "loss": loss,
        "base_policy_loss": (
            float(last_policy_loss.total_loss.detach().cpu().item())
            if last_policy_loss is not None
            else _checkpoint_float(loaded_checkpoint, "loss")
        ),
        "regularized_loss": regularized_loss,
        "actor_loss": actor_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "imitation_loss": imitation_loss,
        "discriminator_loss": discriminator_loss,
        "discriminator_retrain_interval": max(1, int(args.discriminator_retrain_interval)),
        "discriminator_updates": int(discriminator_update_count),
        "demo_weight_refreshes": int(demo_weight_refresh_count),
        "train_episodes": train_episodes,
        "train_continue_bonus": float(args.train_continue_bonus),
        "train_distance_terminal_penalty": float(args.train_distance_terminal_penalty),
        "train_terminal_repeat_td_penalty": float(args.train_terminal_repeat_td_penalty),
        "terminal_repeat_td_penalized_samples": int(terminal_repeat_td_penalized_samples),
        "train_unsafe_repeat_penalty": float(args.train_unsafe_repeat_penalty),
        "train_unsafe_distance_penalty": float(args.train_unsafe_distance_penalty),
        "train_action_repeat_penalty": float(args.train_action_repeat_penalty),
        "train_action_distance_penalty": float(args.train_action_distance_penalty),
        "train_action_shaping_loss": str(args.train_action_shaping_loss),
        "train_bc_kl_coef": float(args.train_bc_kl_coef),
        "train_bc_kl_label_smoothing": float(args.train_bc_kl_label_smoothing),
        "train_logit_l2_penalty": float(args.train_logit_l2_penalty),
        "train_logit_std_penalty": float(args.train_logit_std_penalty),
        "train_logit_std_target": float(args.train_logit_std_target),
        "train_actor_row_norm_penalty": float(args.train_actor_row_norm_penalty),
        "train_actor_row_norm_target": float(args.train_actor_row_norm_target),
        "train_advantage_clip": float(args.train_advantage_clip),
        "train_normalize_advantages": bool(args.train_normalize_advantages),
        "policy_logit_clamp": float(args.policy_logit_clamp),
        "policy_logit_clamp_mode": str(args.policy_logit_clamp_mode),
        "train_actor_row_norm_project": float(args.train_actor_row_norm_project),
        "train_actor_bias_clamp": float(args.train_actor_bias_clamp),
        "train_entropy_floor_ratio": float(args.train_entropy_floor_ratio),
        "train_entropy_floor_coef": float(args.train_entropy_floor_coef),
        "unsafe_action_updates": int(unsafe_action_update_count),
        "bc_kl_updates": int(bc_kl_update_count),
        "logit_scale_updates": int(logit_scale_update_count),
        "entropy_floor_updates": int(entropy_floor_update_count),
        "advantage_clipped_sample_count": int(advantage_clipped_sample_count),
        "train_action_shaping_sample_count": int(train_action_shaping_sample_count),
        "train_action_shaping_policy_update_count": int(train_action_shaping_policy_update_count),
        "train_action_shaping_repeat_mass": (
            float(last_train_action_shaping.repeat_mass.detach().cpu().item())
            if last_train_action_shaping is not None
            else None
        ),
        "train_action_shaping_distance_mass": (
            float(last_train_action_shaping.distance_mass.detach().cpu().item())
            if last_train_action_shaping is not None
            else None
        ),
        "unsafe_action_loss": (
            float(last_unsafe_action_penalty.total_loss.detach().cpu().item())
            if last_unsafe_action_penalty is not None
            else None
        ),
        "unsafe_repeat_mass": (
            float(last_unsafe_action_penalty.repeat_mass.detach().cpu().item())
            if last_unsafe_action_penalty is not None
            else None
        ),
        "unsafe_distance_mass": (
            float(last_unsafe_action_penalty.distance_mass.detach().cpu().item())
            if last_unsafe_action_penalty is not None
            else None
        ),
        "bc_kl_loss": (
            float(last_bc_kl_loss.detach().cpu().item())
            if last_bc_kl_loss is not None
            else None
        ),
        "logit_scale_loss": (
            float(last_logit_scale_penalty.total_loss.detach().cpu().item())
            if last_logit_scale_penalty is not None
            else _checkpoint_float(loaded_checkpoint, "logit_scale_loss", 0.0)
        ),
        "entropy_floor_loss": (
            float(last_entropy_floor_penalty.total_loss.detach().cpu().item())
            if last_entropy_floor_penalty is not None
            else _checkpoint_float(loaded_checkpoint, "entropy_floor_loss", 0.0)
        ),
        "entropy_floor_entropy_mean": (
            float(last_entropy_floor_penalty.entropy_mean.detach().cpu().item())
            if last_entropy_floor_penalty is not None
            else _checkpoint_float(loaded_checkpoint, "entropy_floor_entropy_mean", 0.0)
        ),
        "entropy_floor_ratio_mean": (
            float(last_entropy_floor_penalty.entropy_ratio_mean.detach().cpu().item())
            if last_entropy_floor_penalty is not None
            else _checkpoint_float(loaded_checkpoint, "entropy_floor_ratio_mean", 0.0)
        ),
        "entropy_floor_gap": (
            float(last_entropy_floor_penalty.entropy_floor_gap.detach().cpu().item())
            if last_entropy_floor_penalty is not None
            else _checkpoint_float(loaded_checkpoint, "entropy_floor_gap", 0.0)
        ),
        "logit_l2_loss": (
            float(last_logit_scale_penalty.logit_l2.detach().cpu().item())
            if last_logit_scale_penalty is not None
            else _checkpoint_float(loaded_checkpoint, "logit_l2_loss", 0.0)
        ),
        "logit_std_excess_loss": (
            float(last_logit_scale_penalty.logit_std_excess.detach().cpu().item())
            if last_logit_scale_penalty is not None
            else _checkpoint_float(loaded_checkpoint, "logit_std_excess_loss", 0.0)
        ),
        "actor_row_norm_excess_loss": (
            float(last_logit_scale_penalty.actor_row_norm_excess.detach().cpu().item())
            if last_logit_scale_penalty is not None
            else _checkpoint_float(loaded_checkpoint, "actor_row_norm_excess_loss", 0.0)
        ),
        "logit_std_mean": (
            float(last_logit_scale_penalty.logit_std_mean.detach().cpu().item())
            if last_logit_scale_penalty is not None
            else _checkpoint_float(loaded_checkpoint, "logit_std_mean", 0.0)
        ),
        "actor_row_norm_max": (
            float(last_logit_scale_penalty.actor_row_norm_max.detach().cpu().item())
            if last_logit_scale_penalty is not None
            else _checkpoint_float(loaded_checkpoint, "actor_row_norm_max", 0.0)
        ),
        "advantage_mean": (
            float(last_advantage_transform.mean.detach().cpu().item())
            if last_advantage_transform is not None
            else _checkpoint_float(loaded_checkpoint, "advantage_mean", 0.0)
        ),
        "advantage_std": (
            float(last_advantage_transform.std.detach().cpu().item())
            if last_advantage_transform is not None
            else _checkpoint_float(loaded_checkpoint, "advantage_std", 0.0)
        ),
        "eval_episodes": int(args.eval_episodes),
        "eval_user_ids": rollout_user_ids,
        "eval_user_count": len(rollout_user_ids),
        "available_eval_user_count": len(eval_user_ids) if eval_user_ids is not None else None,
        "discount": float(args.discount),
        "official_ilrec_discount": DEFAULT_ILREC_DISCOUNT,
        "discount_standard_status": discount_standard_status(args),
        "training_transition_count": int(transition_count),
        "demo_usable_transitions": len(demo_transitions),
        "demo_return_method": demo_return_result.diagnostics["return_method"],
        "demo_return_source": demo_return_result.diagnostics["return_source"],
        "demo_return_count": demo_return_result.diagnostics["transition_count"],
        "demo_return_discount": demo_return_result.diagnostics["discount"],
        "demo_return_preview": [
            {
                "trajectory_id": transition.get("trajectory_id"),
                "user_id": int(transition["user_id"]),
                "action_id": int(transition["action_id"]),
                "world_model_reward": float(transition["world_model_reward"]),
                "demo_return": float(transition["demo_return"]),
            }
            for transition in demo_transitions[:3]
        ],
        "demo_advantage_method": demo_advantage_result["method"],
        "demo_value_fit_steps": (
            demo_advantage_result["fit_result"].steps
            if demo_advantage_result["fit_result"] is not None
            else None
        ),
        "demo_value_fit_initial_loss": (
            demo_advantage_result["fit_result"].initial_loss
            if demo_advantage_result["fit_result"] is not None
            else None
        ),
        "demo_value_fit_final_loss": (
            demo_advantage_result["fit_result"].final_loss
            if demo_advantage_result["fit_result"] is not None
            else None
        ),
        "demo_q_fit_steps": (
            demo_advantage_result["q_fit_result"].steps
            if demo_advantage_result.get("q_fit_result") is not None
            else None
        ),
        "demo_q_fit_initial_loss": (
            demo_advantage_result["q_fit_result"].initial_loss
            if demo_advantage_result.get("q_fit_result") is not None
            else None
        ),
        "demo_q_fit_final_loss": (
            demo_advantage_result["q_fit_result"].final_loss
            if demo_advantage_result.get("q_fit_result") is not None
            else None
        ),
        "demo_advantage_preview": demo_advantage_result["preview"],
        "replay_method": "mixed_env_demo_replay",
        "policy_update_method": "mixed_replay_policy_loss",
        "env_replay_size": len(env_replay) if env_replay is not None else 0,
        "demo_replay_size": len(demo_replay) if demo_replay is not None else 0,
        "mixed_replay_size": len(mixed_replay) if mixed_replay is not None else 0,
        "mixed_replay_batch_size": int(args.mixed_replay_batch_size),
        "mixed_replay_sampling": str(args.mixed_replay_sampling),
        "mixed_replay_demo_fraction": float(args.mixed_replay_demo_fraction),
        "mixed_replay_env_priority_scale": float(args.mixed_replay_env_priority_scale),
        "mixed_replay_priority": mixed_replay.priority_totals() if mixed_replay is not None else None,
        "demo_weight_mode": str(args.demo_weight_mode),
        "mixed_policy_updates": int(mixed_policy_update_count),
        "mixed_policy_sample_count": int(mixed_policy_sample_count),
        "mixed_policy_demo_sample_count": int(mixed_policy_demo_sample_count),
        "mixed_policy_env_sample_count": int(mixed_policy_env_sample_count),
        "mixed_replay_demo_weighted": (
            all("demo_weight" in record for record in demo_replay.records())
            if demo_replay is not None
            else False
        ),
        "critic_method": "q_v_td_target",
        "qv_critic_updates": int(qv_update_count),
        "target_critic_updates": int(target_qv_update_count),
        "target_update_interval": int(args.target_update_interval),
        "target_update_tau": float(args.target_update_tau),
        "qv_demo_weighted_samples": int(qv_demo_weighted_samples),
        "qv_critic_loss": (
            float(last_qv_loss.total_loss.detach().cpu().item())
            if last_qv_loss is not None
            else None
        ),
        "qv_q_loss": (
            float(last_qv_loss.q_loss.detach().cpu().item())
            if last_qv_loss is not None
            else None
        ),
        "qv_v_loss": (
            float(last_qv_loss.v_loss.detach().cpu().item())
            if last_qv_loss is not None
            else None
        ),
        "feature_type": feature_builder.feature_type,
        "discriminator_input_dim": int(feature_builder.input_dim),
        "state_action_embedding_cache_saved": bool(state_action_embedding_cache_saved),
        "train_env_source": args.train_env_source,
        "eval_env_source": args.eval_env_source,
        "train_action_selection": str(args.train_action_selection),
        "eval_action_selection": str(args.eval_action_selection),
        "evaluation_mode": "FB",
        "checkpoint_loaded_for_policy": checkpoint_loaded_for_policy,
        "state_tracker": {
            "type": str(args.state_tracker_type),
            "window_size": int(args.window_size),
            "uses_action_history": True,
            "uses_reward_feedback": True,
            "attention_heads": int(args.state_tracker_att_heads),
            "attention_layers": int(args.state_tracker_att_layers),
            "attention_dropout": float(args.state_tracker_att_dropout),
        },
        "device": str(device),
        "num_users": num_users,
        "num_items": num_items,
        "rollout_json": str(rollout_path),
        "rollout_diagnostics": rollout_diagnostics,
        "eval_unique_actions": rollout_diagnostics["unique_actions"],
        "eval_unique_first_actions": rollout_diagnostics["unique_first_actions"],
        "eval_repeat_after_first_rate": rollout_diagnostics["repeat_after_first_rate"],
        "eval_terminal_exact_repeat_episodes": rollout_diagnostics["terminal_exact_repeat_episodes"],
        "eval_termination_reasons": rollout_diagnostics["termination_reasons"],
        "eval_length_counts": rollout_diagnostics["length_counts"],
        "episode_summaries": episode_summaries,
        "model_state_dict": model.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
    }
    return payload


def rollout_policy(
    model,
    env,
    eval_episodes,
    user_ids=None,
    action_selection="sample",
    logit_clamp=0.0,
    logit_clamp_mode="tanh",
):
    model.eval()
    rollouts = []
    eval_episodes = max(0, int(eval_episodes))
    if user_ids is None:
        rollout_user_ids = [episode_idx % env.num_user for episode_idx in range(eval_episodes)]
    else:
        validated_user_ids = _validate_user_ids(user_ids, env.num_user, "eval user ids")
        rollout_user_ids = [validated_user_ids[index % len(validated_user_ids)] for index in range(eval_episodes)]
    device = model.user_embedding.weight.device
    with torch.no_grad():
        for episode_idx, rollout_user_id in enumerate(rollout_user_ids):
            obs = env.reset(user_id=rollout_user_id)
            userid = _select_user_id(obs)
            steps = []
            done = False
            step_index = 0
            recommended_actions = []
            termination_reason = None
            while not done and step_index < env.max_turn:
                user_id = _select_user_id(obs)
                logits, _ = model(
                    torch.tensor(
                        [min(user_id, model.user_embedding.num_embeddings - 1)],
                        dtype=torch.long,
                        device=device,
                    ),
                    history_actions=list(recommended_actions),
                    history_rewards=[step["reward"] for step in steps],
                )
                logits = apply_policy_logit_clamp(
                    logits,
                    clamp_value=logit_clamp,
                    mode=logit_clamp_mode,
                )
                valid_actions = min(env.num_item, logits.shape[-1])
                action = select_evaluation_action(
                    logits,
                    valid_actions=valid_actions,
                    action_selection=action_selection,
                )
                next_obs, reward, done, info = _step_env(env, action)
                recommended_actions.append(action)
                termination_reason = info.get("reason")
                steps.append(
                    {
                        "step_index": step_index,
                        "action_id": action,
                        "reward": reward,
                        "done": done,
                        "reason": info.get("reason"),
                        "terminal_failure": bool(info.get("reason") == "low_reward"),
                    }
                )
                obs = next_obs
                step_index += 1
            rollouts.append(
                {
                    "trajectory_id": f"eval-{episode_idx}",
                    "userid": userid,
                    "evaluation_mode": "FB",
                    "eval_action_selection": str(action_selection),
                    "termination_reason": termination_reason,
                    "steps": steps,
                }
            )
    model.train()
    return rollouts


def default_rollout_path(output_dir, message):
    return Path(output_dir) / "rollouts" / f"[{message}]_rollouts.json"


def default_summary_path(output_dir, message):
    return Path(output_dir) / "logs" / f"[{message}]_summary.json"


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def write_rollout_json(args, rollouts):
    output_dir = args.output_dir or default_output_dir(args.env)
    message = getattr(args, "output_message", args.message)
    rollout_path = args.rollout_json or default_rollout_path(output_dir, message)
    return write_json(rollout_path, rollouts)


def write_summary_json(args, summary):
    output_dir = args.output_dir or default_output_dir(args.env)
    message = getattr(args, "output_message", args.message)
    summary_path = args.summary_json or default_summary_path(output_dir, message)
    summary["summary_json"] = str(summary_path)
    return write_json(summary_path, summary)


def save_checkpoint(output_dir, message, summary, loss_payload, suffix="smoke"):
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"[{message}]_{suffix}.pt"
    torch.save({"summary": summary, **loss_payload}, checkpoint_path)
    return checkpoint_path


def run(args):
    set_seed(args.seed)
    model_root = args.model_root or default_model_root(args.env)
    output_dir = args.output_dir or default_output_dir(args.env)
    args.output_message = resolve_output_message(args)
    args.resolved_state_action_cache_path = resolve_state_action_cache_path(args, output_dir)
    demo_buffer = load_demo_buffer(args.demo_buffer)
    needs_artifacts = bool(
        args.dry_run or args.train_env_source == "artifacts" or args.eval_env_source == "artifacts"
    )
    artifacts = load_world_model_artifacts(model_root, args.read_message) if needs_artifacts else None
    loaded_checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None
    if args.dry_run:
        loss_payload = run_smoke_update(args, demo_buffer)
        checkpoint_suffix = "smoke"
    else:
        loss_payload = train_direct_actor_critic(args, demo_buffer, artifacts, loaded_checkpoint=loaded_checkpoint)
        checkpoint_suffix = "policy"

    summary = {
        "env": args.env,
        "message": args.output_message,
        "requested_message": args.message,
        "read_message": args.read_message,
        "lr": float(args.lr),
        "demo_buffer": str(args.demo_buffer),
        "demo_transitions": len(demo_buffer["transitions"]),
        "world_model_root": str(model_root) if artifacts is not None else None,
        "mat_pre_path": str(artifacts.mat_pre_path) if artifacts is not None else None,
        "mat_var_path": str(artifacts.mat_var_path) if artifacts is not None else None,
        "params_path": str(artifacts.params_path) if artifacts is not None else None,
        "dry_run": bool(args.dry_run),
        "smoke_steps": max(1, int(args.smoke_steps)),
        "loaded_checkpoint": str(args.checkpoint) if loaded_checkpoint is not None else None,
    }
    summary.update(ilrec_metadata(args))
    checkpoint_payload = {
        key: value for key, value in loss_payload.items()
        if key not in {"model_state_dict", "discriminator_state_dict", "episode_summaries"}
    }
    checkpoint_payload.update(
        {
            key: value for key, value in loss_payload.items()
            if key in {"model_state_dict", "discriminator_state_dict"}
        }
    )
    checkpoint_path = save_checkpoint(output_dir, args.output_message, summary, checkpoint_payload, checkpoint_suffix)
    summary.update(
        {
            key: value for key, value in loss_payload.items()
            if key not in {"model_state_dict", "discriminator_state_dict"}
        }
    )
    summary["checkpoint_path"] = str(checkpoint_path)
    write_summary_json(args, summary)
    return summary


def main(argv=None):
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
