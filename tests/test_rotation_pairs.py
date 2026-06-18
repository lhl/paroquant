"""Equivalence tests for the rotation-pair independent-set selection.

The original `_get_independent_channel_pairs` uses torch scalar indexing inside
nested Python loops, which is pathologically slow on real model dimensions.
These tests pin its exact output so it can be replaced by a faster,
set-based implementation that must be bit-identical.
"""

import random

import torch

from paroquant.optim import train


def _reference(pairs, dim, num_rotations, num_pairs_each):
    """Verbatim copy of the original implementation (the oracle)."""
    pairs = pairs.cpu().tolist()
    rotations_pairs = []
    available = torch.ones(dim, dim)
    available.fill_diagonal_(0)

    for _ in range(num_rotations):
        independent_pairs = []
        available_in_rotation = available.clone()
        for i, j in pairs:
            if len(independent_pairs) == num_pairs_each:
                break
            if available_in_rotation[i, j] == 0:
                continue
            independent_pairs.append((i, j))
            available_in_rotation[i, :] = 0
            available_in_rotation[j, :] = 0
            available_in_rotation[:, i] = 0
            available_in_rotation[:, j] = 0
            available[i, j] = 0
            available[j, i] = 0
        rotations_pairs.append(independent_pairs)

    return rotations_pairs


def _build_pairs(group_size, seed):
    """Mirror how get_random_rotation_pairs builds the per-group pair list."""
    rand = random.Random(seed)
    p = []
    for i in range(group_size):
        for j in range(i + 1, group_size):
            p.append((i, j))
    rand.shuffle(p)
    return torch.tensor(p)


def _cases():
    for group_size in (4, 8, 16, 32):
        for seed in (0, 1, 2, 7, 42):
            pairs = _build_pairs(group_size, seed)
            for num_rotations in (1, 2, 5, 8):
                for num_pairs_each in (
                    1,
                    max(1, group_size // 4),
                    max(1, group_size // 2),
                    group_size,
                ):
                    yield pairs, group_size, num_rotations, num_pairs_each


def test_fast_matches_reference():
    for pairs, group_size, num_rotations, num_pairs_each in _cases():
        expected = _reference(pairs, group_size, num_rotations, num_pairs_each)
        got = train._get_independent_channel_pairs(
            pairs, group_size, num_rotations, num_pairs_each
        )
        assert got == expected, (
            f"mismatch for group_size={group_size}, "
            f"num_rotations={num_rotations}, num_pairs_each={num_pairs_each}"
        )
