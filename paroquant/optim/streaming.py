"""Disk-streaming model loader for layerwise optimization of very large models.

The default optimize path materializes the *entire* model in host RAM (fp16)
before the layerwise loop runs. For block-FP8 checkpoints (e.g. MiniMax-M2,
DeepSeek) the on-disk fp8 weights are upcast to fp16, roughly doubling the
footprint, which makes large MoE models impossible to load on commodity RAM.

This module keeps the decoder layers on the ``meta`` device (zero storage) and
materializes one layer at a time directly from the safetensors shards, applying
block-wise FP8 dequantization (``weight * weight_scale_inv``) on the fly. Only
the non-layer parameters (embeddings, final norm, lm_head) plus the single
active layer are ever resident.

The layerwise optimizer already streams layers on/off the GPU and computes each
layer's input from the previous (already-optimized) layer's activations, so it
never needs more than one decoder layer's weights at once.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from paroquant.optim.util import logger

_SCALE_SUFFIX = "_scale_inv"


def dequantize_fp8_block(weight: torch.Tensor, scale_inv: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Dequantize a block-wise FP8 weight: ``out[i, j] = w[i, j] * scale[i//bs, j//bs]``.

    ``weight`` is 2D FP8 ``[R, C]``; ``scale_inv`` is ``[ceil(R/bs), ceil(C/bs)]``.
    """
    if weight.ndim != 2:
        raise ValueError(f"FP8 block dequant expects a 2D weight, got shape {tuple(weight.shape)}")
    rows, cols = weight.shape
    exp_rows = math.ceil(rows / block_size)
    exp_cols = math.ceil(cols / block_size)
    if tuple(scale_inv.shape) != (exp_rows, exp_cols):
        raise ValueError(
            f"scale_inv shape {tuple(scale_inv.shape)} does not match weight {tuple(weight.shape)} "
            f"for block_size={block_size} (expected {(exp_rows, exp_cols)})"
        )
    w = weight.to(torch.float32)
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block_size, dim=0)[:rows]
    s = s.repeat_interleave(block_size, dim=1)[:, :cols]
    return w * s


def _assign_tensor_by_name(root: nn.Module, qualified_name: str, tensor: torch.Tensor, *, is_param: bool) -> None:
    """Assign ``tensor`` to the parameter/buffer ``qualified_name`` relative to ``root``."""
    *parents, leaf = qualified_name.split(".")
    module = root
    for part in parents:
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    if is_param:
        module._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
    else:
        module._buffers[leaf] = tensor


class StreamingModelLoader:
    """Builds a meta-device model and materializes weights per-layer from disk."""

    def __init__(self, model_path: str, *, dtype: torch.dtype = torch.float16, block_size: int = 128):
        self.path = Path(model_path)
        self.dtype = dtype
        self.block_size = block_size

        index_file = self.path / "model.safetensors.index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"Sharded safetensors index not found: {index_file}")
        self.weight_map: dict[str, str] = json.loads(index_file.read_text())["weight_map"]
        self._handles: dict[str, "safe_open"] = {}

    # --- raw tensor access -------------------------------------------------
    def _handle(self, shard: str) -> "safe_open":
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.path / shard), framework="pt", device="cpu")
            self._handles[shard] = handle
        return handle

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def _raw(self, name: str) -> torch.Tensor:
        return self._handle(self.weight_map[name]).get_tensor(name)

    def _load_weight(self, name: str) -> torch.Tensor:
        """Load a checkpoint tensor, applying FP8 block dequant when a scale exists."""
        scale_name = name + _SCALE_SUFFIX
        if self.has(scale_name):
            tensor = dequantize_fp8_block(self._raw(name), self._raw(scale_name), self.block_size)
        else:
            tensor = self._raw(name)
        return tensor.to(self.dtype).contiguous()

    # --- model construction ------------------------------------------------
    def build_meta_model(self) -> nn.Module:
        """Instantiate the model with all parameters on ``meta`` (buffers are real)."""
        config = AutoConfig.from_pretrained(self.path, trust_remote_code=True)
        with init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        return model

    def materialize(self, root: nn.Module, prefix: str) -> None:
        """Fill every parameter under ``root`` from the checkpoint under ``prefix``."""
        for rel_name, _ in list(root.named_parameters(recurse=True)):
            full = f"{prefix}.{rel_name}" if prefix else rel_name
            if not self.has(full):
                continue  # tied / structurally-present-but-unsaved weight
            _assign_tensor_by_name(root, rel_name, self._load_weight(full), is_param=True)

    def release(self, root: nn.Module) -> None:
        """Return every parameter under ``root`` to the meta device, freeing RAM."""
        for rel_name, param in list(root.named_parameters(recurse=True)):
            meta = torch.empty(param.shape, dtype=param.dtype, device="meta")
            _assign_tensor_by_name(root, rel_name, meta, is_param=True)

    def close(self) -> None:
        self._handles.clear()


def materialize_nonlayer(loader: StreamingModelLoader, model: nn.Module, block_prefix: str = "model.layers") -> None:
    """Materialize all params except the decoder layers (embeddings, norm, lm_head)."""
    materialized = 0
    for name, _ in list(model.named_parameters(recurse=True)):
        if name.startswith(block_prefix + "."):
            continue
        if not loader.has(name):
            continue
        _assign_tensor_by_name(model, name, loader._load_weight(name), is_param=True)
        materialized += 1
    logger.info("Streaming loader: materialized %d non-layer tensors.", materialized)


def move_real_tensors(model: nn.Module, device: torch.device | str) -> None:
    """Move all *non-meta* params and buffers to ``device`` (leaves meta layers alone)."""
    for module in model.modules():
        for n, p in list(module._parameters.items()):
            if p is not None and p.device.type != "meta":
                module._parameters[n] = nn.Parameter(p.data.to(device), requires_grad=p.requires_grad)
        for n, b in list(module._buffers.items()):
            if b is not None and b.device.type != "meta":
                module._buffers[n] = b.to(device)
