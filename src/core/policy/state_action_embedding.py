"""State-action embedding cache for ILRec ILRec."""

from pathlib import Path

import torch

from core.policy.state_action_text import TEMPLATE_VERSION, render_transition_state_action_text


DEFAULT_LLAMA_MODEL_PATH = "/home/hehui/llama2-7bhf"


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class LlamaStateActionEncoder:
    """Encode state-action texts with a local LLaMA backbone."""

    def __init__(self, model_path=DEFAULT_LLAMA_MODEL_PATH, batch_size=16, device=None):
        from transformers import AutoModel, AutoTokenizer

        self.model_path = str(model_path)
        self.batch_size = max(1, int(batch_size))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token or self.tokenizer.eos_token

        if self.device == "cuda":
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
            ).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.model.eval()

    def encode(self, texts):
        texts = [str(text) for text in texts]
        if not texts:
            return torch.empty((0, 0), dtype=torch.float32)

        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(
                    encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    output_hidden_states=True,
                )
                embeddings.append(outputs.hidden_states[-1][:, -1, :].detach().cpu())
        return torch.cat(embeddings, dim=0).float()


class StateActionEmbeddingCache:
    """Persistent cache keyed by deterministic state-action text."""

    def __init__(
        self,
        encoder=None,
        cache_path=None,
        model_path=DEFAULT_LLAMA_MODEL_PATH,
        template_version=TEMPLATE_VERSION,
        encoder_batch_size=16,
        device=None,
    ):
        self.model_path = str(model_path)
        self.template_version = str(template_version)
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.encoder = encoder or LlamaStateActionEncoder(
            model_path=self.model_path,
            batch_size=encoder_batch_size,
            device=device,
        )
        self.entries = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._load()

    def metadata(self):
        return {
            "model_path": self.model_path,
            "template_version": self.template_version,
            "encoder": "llama_state_action",
        }

    def _load(self):
        if self.cache_path is None or not self.cache_path.exists():
            return
        payload = _torch_load(self.cache_path)
        if not isinstance(payload, dict):
            return
        if payload.get("metadata") != self.metadata():
            return
        entries = payload.get("entries", {})
        if not isinstance(entries, dict):
            return
        for text, embedding in entries.items():
            tensor = torch.as_tensor(embedding).detach().cpu().float()
            if tensor.ndim == 1:
                self.entries[str(text)] = tensor

    def save(self):
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata(),
            "entries": self.entries,
        }
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(self.cache_path)

    def encode_texts(self, texts):
        texts = [str(text) for text in texts]
        if not texts:
            return torch.empty((0, 0), dtype=torch.float32)

        missing = []
        queued = set()
        for text in texts:
            if text in self.entries:
                self.cache_hits += 1
                continue
            self.cache_misses += 1
            if text not in queued:
                queued.add(text)
                missing.append(text)

        if missing:
            encoded = torch.as_tensor(self.encoder.encode(missing)).detach().cpu().float()
            if encoded.ndim != 2:
                raise ValueError("State-action encoder must return a 2D tensor.")
            if encoded.shape[0] != len(missing):
                raise ValueError(
                    f"State-action encoder returned {encoded.shape[0]} rows for {len(missing)} texts."
                )
            for text, embedding in zip(missing, encoded):
                self.entries[text] = embedding.detach().cpu().float()

        return torch.stack([self.entries[text] for text in texts], dim=0)

    def encode_transitions(self, transitions, dataset=None, item_lookup=None):
        texts = [
            render_transition_state_action_text(
                transition,
                item_lookup=item_lookup,
                dataset=dataset,
            )
            for transition in transitions
        ]
        return self.encode_texts(texts)
