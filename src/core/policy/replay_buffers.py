"""Explicit replay buffers for the ILRec ILRec branch."""

import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=None, source="env"):
        self.capacity = None if capacity is None else int(capacity)
        if self.capacity is not None and self.capacity <= 0:
            raise ValueError("capacity must be positive when provided.")
        self.source = str(source)
        self._records = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self._records)

    def add(self, record):
        if not isinstance(record, dict):
            raise ValueError("record must be a dictionary.")
        stored = dict(record)
        stored.setdefault("source", self.source)
        self._records.append(stored)

    def extend(self, records):
        for record in records:
            self.add(record)

    def records(self):
        return [dict(record) for record in self._records]

    def get(self, index):
        return dict(self._records[int(index)])

    def _sample_available_index(self, rng, selected_indices):
        record_count = len(self._records)
        if len(selected_indices) >= record_count:
            return None
        if record_count <= 32 or len(selected_indices) > record_count // 2:
            available = [index for index in range(record_count) if index not in selected_indices]
            return rng.choice(available)
        while True:
            index = rng.randrange(record_count)
            if index not in selected_indices:
                return index

    def sample(self, batch_size, seed=None):
        record_count = len(self._records)
        if record_count == 0:
            return []
        batch_size = min(max(0, int(batch_size)), record_count)
        rng = random.Random(seed)
        indices = rng.sample(range(record_count), batch_size)
        return [dict(self._records[index]) for index in indices]


class DemoReplayBuffer(ReplayBuffer):
    def __init__(self, capacity=None):
        super().__init__(capacity=capacity, source="demo")

    def add_demo(self, transition, weight):
        stored = dict(transition)
        stored["demo_weight"] = float(weight)
        self.add(stored)

    def priority(self, index):
        return max(0.0, float(self._records[int(index)].get("demo_weight", 1.0)))

    def priority_sum(self, indices=None):
        if indices is None:
            indices = range(len(self._records))
        return sum(self.priority(index) for index in indices)

    def _sample_available_index_by_priority(self, rng, available):
        if not available:
            return None
        total_weight = self.priority_sum(available)
        if total_weight <= 0.0:
            return rng.choice(available)

        threshold = rng.random() * total_weight
        running = 0.0
        for index in available:
            running += self.priority(index)
            if running >= threshold:
                return index
        return available[-1]

    def sample(self, batch_size, seed=None):
        record_count = len(self._records)
        if record_count == 0:
            return []
        batch_size = min(max(0, int(batch_size)), record_count)
        rng = random.Random(seed)
        available = list(range(record_count))
        samples = []
        for _ in range(batch_size):
            weights = [
                max(0.0, float(self._records[index].get("demo_weight", 1.0)))
                for index in available
            ]
            total_weight = sum(weights)
            if total_weight <= 0.0:
                selected_position = rng.randrange(len(available))
            else:
                threshold = rng.random() * total_weight
                running = 0.0
                selected_position = len(available) - 1
                for position, weight in enumerate(weights):
                    running += weight
                    if running >= threshold:
                        selected_position = position
                        break
            selected_index = available.pop(selected_position)
            samples.append(dict(self._records[selected_index]))
        return samples


class MixedReplayBuffer:
    FIXED_DEMO_FRACTION = "fixed_demo_fraction"
    GLOBAL_PRIORITY = "global_priority"

    def __init__(self, env_buffer, demo_buffer, env_priority_scale=1.0):
        if env_priority_scale < 0:
            raise ValueError("env_priority_scale must be non-negative.")
        self.env_buffer = env_buffer
        self.demo_buffer = demo_buffer
        self.env_priority_scale = float(env_priority_scale)

    def __len__(self):
        return len(self.env_buffer) + len(self.demo_buffer)

    def priority_totals(self):
        env_priority = self.env_priority_scale * float(len(self.env_buffer))
        demo_priority = float(self.demo_buffer.priority_sum())
        return {
            "env_priority": env_priority,
            "demo_priority": demo_priority,
            "total_priority": env_priority + demo_priority,
            "env_priority_scale": self.env_priority_scale,
        }

    def sample(self, batch_size, demo_fraction=0.5, seed=None, sampling_mode=FIXED_DEMO_FRACTION):
        if sampling_mode == self.GLOBAL_PRIORITY:
            return self.sample_global_priority(batch_size, seed=seed)
        if sampling_mode != self.FIXED_DEMO_FRACTION:
            raise ValueError(f"Unsupported mixed replay sampling mode: {sampling_mode}")
        return self.sample_fixed_demo_fraction(batch_size, demo_fraction=demo_fraction, seed=seed)

    def sample_fixed_demo_fraction(self, batch_size, demo_fraction=0.5, seed=None):
        batch_size = max(0, int(batch_size))
        if batch_size == 0:
            return []
        demo_fraction = min(max(float(demo_fraction), 0.0), 1.0)
        demo_count = int(round(batch_size * demo_fraction))
        env_count = batch_size - demo_count

        if len(self.env_buffer) == 0:
            demo_count = batch_size
            env_count = 0
        if len(self.demo_buffer) == 0:
            env_count = batch_size
            demo_count = 0

        rng = random.Random(seed)
        demo_seed = rng.randrange(2**31)
        env_seed = rng.randrange(2**31)
        demo_samples = self.demo_buffer.sample(demo_count, seed=demo_seed)
        env_samples = self.env_buffer.sample(env_count, seed=env_seed)
        mixed = demo_samples + env_samples
        rng.shuffle(mixed)
        return mixed

    def sample_global_priority(self, batch_size, seed=None):
        batch_size = min(max(0, int(batch_size)), len(self))
        if batch_size == 0:
            return []

        rng = random.Random(seed)
        selected_env_indices = set()
        available_demo_indices = list(range(len(self.demo_buffer)))
        samples = []

        for _ in range(batch_size):
            env_priority = self.env_priority_scale * float(len(self.env_buffer) - len(selected_env_indices))
            demo_priority = float(self.demo_buffer.priority_sum(available_demo_indices))
            total_priority = env_priority + demo_priority

            if total_priority <= 0.0:
                if env_priority > 0.0:
                    source = "env"
                elif available_demo_indices:
                    source = "demo"
                else:
                    break
            else:
                source = "env" if rng.random() * total_priority < env_priority else "demo"

            if source == "env":
                selected_index = self.env_buffer._sample_available_index(rng, selected_env_indices)
                if selected_index is None:
                    if not available_demo_indices:
                        break
                    selected_demo_index = self.demo_buffer._sample_available_index_by_priority(
                        rng,
                        available_demo_indices,
                    )
                    available_demo_indices.remove(selected_demo_index)
                    samples.append(self.demo_buffer.get(selected_demo_index))
                else:
                    selected_env_indices.add(selected_index)
                    samples.append(self.env_buffer.get(selected_index))
            else:
                selected_demo_index = self.demo_buffer._sample_available_index_by_priority(
                    rng,
                    available_demo_indices,
                )
                if selected_demo_index is None:
                    selected_env_index = self.env_buffer._sample_available_index(rng, selected_env_indices)
                    if selected_env_index is None:
                        break
                    selected_env_indices.add(selected_env_index)
                    samples.append(self.env_buffer.get(selected_env_index))
                else:
                    available_demo_indices.remove(selected_demo_index)
                    samples.append(self.demo_buffer.get(selected_demo_index))

        rng.shuffle(samples)
        return samples
