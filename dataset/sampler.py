"""Project-balanced sampling for reduced BCSD datasets."""

import math
import random
from collections import defaultdict
from collections.abc import Iterator
from numbers import Real

import torch.distributed as dist
from torch.utils.data import Sampler


class SqrtBalancedSampler(Sampler[int]):
    """Sample each project up to ``sqrt(count) * scale_factor`` per epoch."""

    def __init__(
        self,
        dataset,
        scale_factor: float = 50.0,
        rank: int | None = None,
        world_size: int | None = None,
        seed: int = 42,
    ):
        if (
            not isinstance(scale_factor, Real)
            or isinstance(scale_factor, bool)
            or not math.isfinite(scale_factor)
            or scale_factor <= 0
        ):
            raise ValueError("scale_factor must be a positive finite number")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if not hasattr(dataset, "_group_keys"):
            raise TypeError("dataset must expose _group_keys")

        self._group_keys = list(dataset._group_keys)
        if not self._group_keys:
            raise ValueError("cannot sample an empty dataset")
        self._scale_factor = float(scale_factor)
        self._seed = seed
        self._epoch = 0

        distributed = dist.is_available() and dist.is_initialized()
        self._rank = (
            dist.get_rank() if distributed and rank is None else (0 if rank is None else rank)
        )
        self._world_size = (
            dist.get_world_size()
            if distributed and world_size is None
            else (1 if world_size is None else world_size)
        )
        if (
            not isinstance(self._world_size, int)
            or isinstance(self._world_size, bool)
            or self._world_size <= 0
        ):
            raise ValueError("world_size must be a positive integer")
        if not isinstance(self._rank, int) or isinstance(self._rank, bool):
            raise TypeError("rank must be an integer")
        if not 0 <= self._rank < self._world_size:
            raise ValueError(f"rank must be in [0, {self._world_size}), got {self._rank}")

        self._project_indices = defaultdict(list)
        for index, key in enumerate(self._group_keys):
            if not isinstance(key, tuple) or not key or not isinstance(key[0], str):
                raise ValueError("dataset._group_keys entries must start with a project name")
            self._project_indices[key[0]].append(index)

        self._targets = {}
        for project, indices in self._project_indices.items():
            scaled = int(math.sqrt(len(indices)) * self._scale_factor)
            self._targets[project] = min(len(indices), max(1, scaled))

        self._total_global = sum(self._targets.values())
        self._total_per_rank = math.ceil(self._total_global / self._world_size)
        self._total_padded = self._total_per_rank * self._world_size

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            raise TypeError("epoch must be an integer")
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self._seed + self._epoch)
        sampled = []
        for project, indices in self._project_indices.items():
            sampled.extend(rng.sample(indices, self._targets[project]))
        rng.shuffle(sampled)

        missing = self._total_padded - len(sampled)
        if missing:
            repeats = math.ceil(missing / len(sampled))
            sampled.extend((sampled * repeats)[:missing])

        return iter(sampled[self._rank :: self._world_size])

    def __len__(self) -> int:
        return self._total_per_rank

    def summary(self) -> str:
        lines = []
        for project in sorted(self._project_indices):
            count = len(self._project_indices[project])
            target = self._targets[project]
            percentage = target / self._total_global * 100
            lines.append(f"  {project}: {count} groups -> {target}/epoch ({percentage:.1f}%)")
        lines.append(f"  Total: {self._total_global}/epoch ({self._total_per_rank}/rank)")
        return "\n".join(lines)
