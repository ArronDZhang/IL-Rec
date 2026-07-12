import pickle
from pathlib import Path

import numpy as np

try:
    import gym
    from gym import spaces
except ImportError:  # pragma: no cover - exercised implicitly when gym is absent.
    class _Env:
        pass

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

    class _Discrete:
        def __init__(self, n):
            self.n = int(n)

    class _Spaces:
        Box = _Box
        Discrete = _Discrete

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()


_RESOURCE_CACHE = {}
_MATRIX_STATS_CACHE = {}


def _resource_cache_key(mat_path, distance_path, read_user_num, mmap_mode):
    return (
        str(Path(mat_path).resolve()),
        str(Path(distance_path).resolve()),
        None if read_user_num is None else int(read_user_num),
        mmap_mode,
    )


def _matrix_stats_cache_key(mat):
    if isinstance(mat, np.memmap):
        return (
            "memmap",
            str(Path(mat.filename).resolve()),
            tuple(mat.shape),
            str(mat.dtype),
            int(mat.offset),
        )
    return ("array", id(mat), tuple(mat.shape), str(mat.dtype))


def _matrix_min_max(mat):
    key = _matrix_stats_cache_key(mat)
    if key not in _MATRIX_STATS_CACHE:
        _MATRIX_STATS_CACHE[key] = (float(np.min(mat)), float(np.max(mat)))
    return _MATRIX_STATS_CACHE[key]


class ILRecEnv(gym.Env):
    default_leave_threshold = None
    default_num_leave_compute = 4
    default_max_turn = 100

    def __init__(
        self,
        mat,
        mat_distance=None,
        num_leave_compute=None,
        leave_threshold=None,
        max_turn=None,
        **_,
    ):
        self.mat = mat if isinstance(mat, np.memmap) else np.asarray(mat)
        if self.mat.ndim != 2 or self.mat.shape[0] == 0 or self.mat.shape[1] == 0:
            raise ValueError("mat must be a non-empty 2D reward matrix")

        self.mat_distance = None if mat_distance is None else np.asarray(mat_distance)
        if self.mat_distance is not None and self.mat_distance.shape != (self.mat.shape[1], self.mat.shape[1]):
            raise ValueError("mat_distance must have shape (num_items, num_items)")

        if self.default_leave_threshold is None and leave_threshold is None:
            raise ValueError("leave_threshold is required for the base ILRecEnv")

        self.num_leave_compute = (
            self.default_num_leave_compute if num_leave_compute is None else int(num_leave_compute)
        )
        self.leave_threshold = self.default_leave_threshold if leave_threshold is None else leave_threshold
        self.max_turn = self.default_max_turn if max_turn is None else int(max_turn)

        if self.num_leave_compute < 0:
            raise ValueError("num_leave_compute must be non-negative")
        if self.max_turn <= 0:
            raise ValueError("max_turn must be positive")

        self.num_user, self.num_item = self.mat.shape
        self.observation_space = spaces.Box(
            low=0,
            high=self.num_user - 1,
            shape=(1,),
            dtype=np.int64,
        )
        self.action_space = spaces.Discrete(self.num_item)
        self.MIN_R, self.MAX_R = _matrix_min_max(self.mat)

        self._next_user_id = 0
        self.cur_user = 0
        self.state = np.array([0], dtype=np.int64)
        self._reset_episode_state()

    @classmethod
    def clear_resource_cache(cls):
        _RESOURCE_CACHE.clear()

    @classmethod
    def load_resources(cls, mat_path, distance_path, read_user_num=None, cache=True, mmap_mode="r"):
        mat_path = Path(mat_path)
        distance_path = Path(distance_path)
        key = _resource_cache_key(mat_path, distance_path, read_user_num, mmap_mode)
        if cache and key in _RESOURCE_CACHE:
            return _RESOURCE_CACHE[key]

        if not mat_path.exists():
            raise FileNotFoundError(f"Required ILRec reward matrix is missing: {mat_path}")
        if not distance_path.exists():
            raise FileNotFoundError(f"Required ILRec distance matrix is missing: {distance_path}")

        mat = np.load(mat_path, mmap_mode=mmap_mode)
        if read_user_num is not None:
            mat = mat[: int(read_user_num)]
        with distance_path.open("rb") as f:
            mat_distance = pickle.load(f)

        resources = (mat, mat_distance)
        if cache:
            _RESOURCE_CACHE[key] = resources
        return resources

    @classmethod
    def from_files(cls, mat_path, distance_path, read_user_num=None, cache=True, mmap_mode="r", **kwargs):
        mat, mat_distance = cls.load_resources(
            mat_path,
            distance_path,
            read_user_num=read_user_num,
            cache=cache,
            mmap_mode=mmap_mode,
        )
        return cls(mat=mat, mat_distance=mat_distance, **kwargs)

    def seed(self, sd=0):
        np.random.seed(sd)

    def reset(self, user_id=None, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        if user_id is None and options:
            user_id = options.get("user_id")
        if user_id is None:
            user_id = self._next_user_id
            self._next_user_id = (self._next_user_id + 1) % self.num_user

        user_id = self._normalize_user_id(user_id)
        self.cur_user = user_id
        self.state = np.array([user_id], dtype=np.int64)
        self._reset_episode_state()
        return self.state.copy()

    def step(self, action):
        action_id = self._normalize_action_id(action)
        user_id = self._normalize_user_id(self.cur_user)

        reward = float(self.mat[user_id, action_id])
        reason = "continue"
        matched_history_item = None

        if reward < 2:
            reason = "low_reward"
        else:
            matched_history_item = self._find_close_history_item(action_id)
            if matched_history_item is not None:
                reason = "distance"

        self.total_turn += 1
        if reason == "continue" and self.total_turn >= self.max_turn:
            reason = "max_turn"

        self.cur_user = user_id
        self.state = np.array([user_id], dtype=np.int64)
        self.action = action_id
        self.reward = reward
        self.cum_reward += reward
        self.history_action.append(action_id)

        info = {
            "user_id": user_id,
            "item_id": action_id,
            "reason": reason,
        }
        if matched_history_item is not None:
            info["matched_history_item"] = matched_history_item

        return self.state.copy(), reward, reason != "continue", info

    def lookup_rewards(self, user_ids, actions):
        user_arr = np.asarray(user_ids).astype(int)
        action_arr = np.asarray(actions).astype(int)
        user_arr, action_arr = np.broadcast_arrays(user_arr, action_arr)
        if np.any(user_arr < 0) or np.any(user_arr >= self.num_user):
            raise ValueError(f"user_id values must be inside [0, {self.num_user})")
        if np.any(action_arr < 0) or np.any(action_arr >= self.num_item):
            raise ValueError(f"action values must be inside [0, {self.num_item})")
        return np.asarray(self.mat[user_arr, action_arr], dtype=float)

    def render(self, mode="human"):
        return None

    def _reset_episode_state(self):
        self.total_turn = 0
        self.cum_reward = 0.0
        self.reward = 0.0
        self.action = None
        self.history_action = []

    def _find_close_history_item(self, action_id):
        if self.mat_distance is None or self.num_leave_compute == 0:
            return None

        recent_history = self.history_action[-self.num_leave_compute :]
        for history_item in reversed(recent_history):
            distance = float(self.mat_distance[action_id, history_item])
            if distance < self.leave_threshold:
                return int(history_item)
        return None

    def _normalize_user_id(self, user_id):
        user_id = self._as_scalar_int(user_id, "user_id")
        if user_id < 0 or user_id >= self.num_user:
            raise ValueError(f"user_id {user_id} is outside [0, {self.num_user})")
        return user_id

    def _normalize_action_id(self, action):
        action_id = self._as_scalar_int(action, "action")
        if action_id < 0 or action_id >= self.num_item:
            raise ValueError(f"action {action_id} is outside [0, {self.num_item})")
        return action_id

    @staticmethod
    def _as_scalar_int(value, label):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value)
        if arr.size != 1:
            raise ValueError(f"{label} must be a scalar id")
        return int(arr.reshape(-1)[0])


class AmazonEnv(ILRecEnv):
    default_leave_threshold = 15


class SteamEnv(ILRecEnv):
    default_leave_threshold = 50
