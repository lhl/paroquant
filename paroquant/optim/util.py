from __future__ import annotations

import gc
import glob
import json
import logging
import math
import random
import shutil
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable, TypeVar

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


RETAINED_KWARG_KEYS = ("shared_kv_states",)


def to_device(value, device: torch.device | str):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


def get_blocks(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        model = model.model.language_model
    elif hasattr(model, "model"):
        model = model.model

    if hasattr(model, "layers"):
        return model.layers

    raise NotImplementedError(type(model))


_Linear_T = TypeVar("Linear", bound=nn.Module)


def get_named_linears(module: nn.Module, subclass: type[_Linear_T] = nn.Linear) -> dict[str, _Linear_T]:
    return {name: m for name, m in module.named_modules() if isinstance(m, subclass)}


def get_module_by_name(module, module_name):
    for name, m in module.named_modules():
        if name == module_name:
            return m
    return None


def set_module_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def load_model(model_path: str, device_map: str | None = None, dtype=torch.float32, **kwargs) -> nn.Module:
    kwargs.setdefault("trust_remote_code", True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, dtype=dtype, **kwargs)
    return model


def load_tokenizer(model_path: str, **kwargs) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def move_embed(model, device):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        # AutoModelForCausalLM returns Gemma4ForConditionalGeneration for Gemma 4. Need to unwrap.
        model = model.model.language_model
    elif hasattr(model, "model"):
        model = model.model
    else:
        raise NotImplementedError(type(model))

    def _move(module_name):
        module = getattr(model, module_name, None)
        if module is not None:
            module.to(device)

    _move("embed_tokens")
    _move("rotary_emb")

    # Gemma 4
    _move("embed_tokens_per_layer")
    _move("per_layer_model_projection")
    _move("per_layer_projection_norm")


def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()


def _normalize_chat_role(role: Any) -> str:
    role_str = str(role or "user").strip().lower()
    if role_str in {"human", "user", "instruction", "input"}:
        return "user"
    if role_str in {"gpt", "assistant", "model", "bot", "teacher"}:
        return "assistant"
    if role_str == "system":
        return "system"
    return "user"


def _messages_to_text(messages: Any, tokenizer) -> str:
    if isinstance(messages, str):
        with suppress(json.JSONDecodeError):
            messages = json.loads(messages)
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        return ""

    normalized = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", msg.get("from", msg.get("speaker", "user")))
            content = msg.get("content", msg.get("value", msg.get("text", "")))
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            role, content = msg
        else:
            continue
        if content is None:
            continue
        normalized.append({"role": _normalize_chat_role(role), "content": str(content)})

    if not normalized:
        return ""

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is not None:
        try:
            return apply_chat_template(normalized, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass

    eos = getattr(tokenizer, "eos_token", "") or ""
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in normalized).strip() + eos


def _jsonl_row_to_text(row: dict[str, Any], tokenizer) -> str:
    for key in ("text", "content"):
        value = row.get(key)
        if isinstance(value, str):
            return value

    for key in ("conversations", "messages", "conversation", "chat", "dialogue"):
        if key in row:
            text = _messages_to_text(row[key], tokenizer)
            if text:
                return text

    # Preference data: calibrate on the preferred trajectory only.
    if "chosen" in row:
        text = _messages_to_text(row["chosen"], tokenizer)
        if text:
            return text

    # GAD-style rows: prompt is a chat/message list and teacher is the answer.
    if "prompt" in row and "teacher" in row:
        messages = row["prompt"]
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            messages = []
        teacher = row["teacher"]
        if isinstance(teacher, dict):
            messages = [*messages, teacher]
        elif isinstance(teacher, str):
            messages = [*messages, {"role": "assistant", "content": teacher}]
        text = _messages_to_text(messages, tokenizer)
        if text:
            return text

    # Generic prompt/completion fallback.
    if isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str):
        return f"{row['prompt']}\n{row['completion']}"

    return ""


def _resolve_jsonl_paths(data: str) -> list[Path]:
    source = data.removeprefix("jsonl:")
    source = str(Path(source).expanduser())
    if any(ch in source for ch in "*?[]"):
        paths = [Path(p) for p in glob.glob(source)]
    else:
        path = Path(source)
        if path.is_dir():
            paths = list(path.glob("*.jsonl"))
        elif path.is_file() and path.suffix == ".jsonl":
            paths = [path]
        else:
            paths = []
    paths = sorted(p for p in paths if p.is_file() and p.suffix == ".jsonl")
    if not paths:
        raise FileNotFoundError(f"No JSONL calibration files found for {data!r}")
    return paths


def _is_jsonl_calib_source(data: str) -> bool:
    if data.startswith("jsonl:"):
        return True
    try:
        _resolve_jsonl_paths(data)
        return True
    except FileNotFoundError:
        return False


def _iter_jsonl_texts(data: str, tokenizer, seed: int) -> Iterable[str]:
    paths = _resolve_jsonl_paths(data)
    rand = random.Random(seed)
    rand.shuffle(paths)
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, str):
                    text = row
                elif isinstance(row, dict):
                    text = _jsonl_row_to_text(row, tokenizer)
                else:
                    text = ""
                text = text.strip()
                if text:
                    yield text


def get_mixed_calib_dataset(
    datasets: list[str],
    *,
    tokenizer,
    n_samples: int,
    block_size: int,
    seed: int,
    split: str,
) -> list[torch.Tensor]:
    per_dataset_len = n_samples // len(datasets)
    results = []
    for i, dataset in enumerate(datasets):
        dataset_samples = per_dataset_len if i < len(datasets) - 1 else n_samples - len(results)
        results.extend(
            get_calib_dataset(
                data=dataset,
                tokenizer=tokenizer,
                n_samples=dataset_samples,
                block_size=block_size,
                seed=seed,
                split=split,
            )
        )
    if len(results) < n_samples:
        if len(results) == 0:
            raise ValueError(f"Mixed calibration produced no samples, requested {n_samples}")
        missing = n_samples - len(results)
        logger.warning(
            "Mixed calibration produced %d samples, requested %d (%.1f%% coverage). "
            "Padding with %d deterministic duplicate sample(s) to preserve fixed batch shapes.",
            len(results), n_samples, 100 * len(results) / n_samples, missing,
        )
        for i in range(missing):
            sample = results[i % len(results)]
            results.append(sample.clone() if isinstance(sample, torch.Tensor) else sample)
    results = results[:n_samples]

    rand = random.Random(seed)
    rand.shuffle(results)

    return results


# Adapted from awq-llm
def get_calib_dataset(
    data="pileval",
    *,
    tokenizer,
    n_samples: int,
    block_size: int,
    seed: int,
    split: str,
) -> list[torch.Tensor]:
    allow_long_rows = False
    if _is_jsonl_calib_source(data):
        line_iter = _iter_jsonl_texts(data, tokenizer, seed)
        allow_long_rows = True
    else:
        if data == "pileval":
            if split != "validation":
                warnings.warn("The split argument is ignored when data is 'pileval'.")
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            dataset = dataset.shuffle(seed=seed)
        elif data == "wikitext2":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            dataset = dataset.shuffle(seed=seed)
        elif data == "c4":
            if split == "train":
                dataset = load_dataset(
                    "allenai/c4",
                    data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                    split=split,
                )
            elif split == "validation":
                dataset = load_dataset(
                    "allenai/c4",
                    data_files={"validation": "en/c4-validation.00001-of-00008.json.gz"},
                    split=split,
                )
            dataset = dataset.shuffle(seed=seed)
        elif data == "redpajama":
            test_split, val_split = 0.2, 0.1
            dataset = load_dataset(
                "liang2kl/RedPajama-Data-1T-Sample-Backup",
                split="train",
                trust_remote_code=True,
            )
            dataset = dataset.shuffle(seed=seed)
            test_size = int(len(dataset) * test_split)
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - test_size - val_size
            if split == "test":
                dataset = dataset.select(range(len(dataset) - test_size, len(dataset)))
            elif split == "validation":
                dataset = dataset.select(range(len(dataset) - test_size - val_size, len(dataset) - test_size))
            elif split == "train":
                dataset = dataset.select(range(0, train_size))
            else:
                raise ValueError(f"Invalid split: {split}")
        else:
            raise NotImplementedError
        line_iter = (str(row["text"]).strip() for row in dataset if row.get("text") is not None)

    token_ids: list[int] = []
    target_len = n_samples * block_size
    for line in line_iter:
        if not line:
            continue
        line_encoded = tokenizer.encode(line)
        if not line_encoded:
            continue
        if len(line_encoded) > block_size and not allow_long_rows:
            continue
        token_ids.extend(line_encoded)
        if len(token_ids) >= target_len:
            break

    if not token_ids:
        raise ValueError(f"Calibration source {data!r} produced no usable text")
    samples = torch.tensor(token_ids[:target_len])
    n_split = min(samples.shape[0] // block_size, n_samples)

    return [samples[i * block_size : (i + 1) * block_size] for i in range(n_split)]


@torch.no_grad()
def capture_layer_inputs_and_args(
    model: nn.Module,
    layers: nn.ModuleList,
    samples: torch.Tensor,
    batch_size: int | None,
    *,
    first_layer_input_batches=None,
) -> tuple[Any, list[dict], list[list[tuple[torch.Tensor, ...]]], list[dict]]:
    device = samples.device
    kwargs_list: list[dict] = [{} for _ in range(len(layers))]
    if first_layer_input_batches is None:
        first_layer_input_batches = []
    other_args_batches_list: list[list[tuple[torch.Tensor, ...]]] = [[] for _ in range(len(layers))]
    retained_kwargs_batches: list[dict] = []

    class Catcher(nn.Module):

        def __init__(self, module, layer_idx):
            super().__init__()
            # Bypass __setattr__ of nn.Module
            object.__setattr__(self, "module", module)
            object.__setattr__(self, "layer_idx", layer_idx)

        def forward(self, *args, **kwargs):
            if self.layer_idx == 0:
                first_layer_input_batches.append(args[0].cpu())

            # Capture kwargs for all layers.
            layer_kwargs = kwargs_list[self.layer_idx]
            if len(layer_kwargs) == 0:
                layer_kwargs.update({k: v for k, v in kwargs.items() if k not in RETAINED_KWARG_KEYS})
                layer_kwargs.pop("use_cache", None)
                layer_kwargs.pop("past_key_value", None)
                layer_kwargs.pop("past_key_values", None)

            other_args_batches_list[self.layer_idx].append(
                tuple(arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args[1:])
            )
            retained_kwargs = {k: to_device(kwargs[k], "cpu") for k in RETAINED_KWARG_KEYS if k in kwargs}
            if self.layer_idx == 0:
                retained_kwargs_batches.append(retained_kwargs)

            return torch.empty_like(args[0], device=device)

        def __getattr__(self, name):
            return getattr(self.module, name)

    for layer_idx, layer in enumerate(layers):
        layers[layer_idx] = Catcher(layer, layer_idx)

    batch_size = samples.shape[0] if batch_size is None or batch_size <= 0 else batch_size
    if samples.shape[0] % batch_size != 0:
        raise ValueError(
            f"Number of calibration samples ({samples.shape[0]}) must be divisible by batch_size ({batch_size}). "
            "Ragged batches cannot safely reuse captured layer kwargs."
        )
    for samples in samples.split(batch_size):
        model(samples)

    for layer_idx, layer in enumerate(layers):
        layers[layer_idx] = layer.module

    return (
        first_layer_input_batches,
        kwargs_list,
        other_args_batches_list,
        retained_kwargs_batches,
    )


def _move_tensor_batch(batch, device: torch.device | str) -> torch.Tensor | tuple[torch.Tensor, ...]:
    return to_device(batch, device)


def _tensor_batch_device(batch) -> torch.device:
    if isinstance(batch, tuple):
        return batch[0].device
    return batch.device


class DiskTensorBatchStore:
    """Disk-backed sequence for activation batches.

    Batches are saved one file at a time and loaded lazily by __getitem__.
    This keeps inter-layer activation streams out of host RAM while preserving
    the list-like API used by the layerwise optimizer.
    """

    def __init__(
        self,
        path: str | Path,
        batches=None,
        *,
        overwrite: bool = False,
        delete_on_cleanup: bool = True,
    ):
        self.path = Path(path)
        if overwrite and self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.delete_on_cleanup = delete_on_cleanup
        self._len = 0
        self._closed = False

        if batches is not None:
            self.extend(batches)

    def _batch_path(self, index: int) -> Path:
        return self.path / f"{index:06d}.pt"

    def append(self, batch) -> None:
        if self._closed:
            raise RuntimeError(f"Cannot append to closed activation store: {self.path}")
        torch.save(to_device(batch, "cpu"), self._batch_path(self._len))
        self._len += 1

    def extend(self, batches) -> None:
        for batch in batches:
            self.append(batch)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._len)
            return [self[i] for i in range(start, stop, step)]
        if index < 0:
            index += self._len
        if index < 0 or index >= self._len:
            raise IndexError(index)
        return torch.load(self._batch_path(index), map_location="cpu")

    def __iter__(self):
        for index in range(self._len):
            yield self[index]

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.delete_on_cleanup and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


class LayerArgsBatches:
    """Lazy tuple(input_batch, *other_args) view over layer activation batches."""

    def __init__(self, input_batches, other_args_batches: list[tuple[torch.Tensor, ...]]):
        if len(input_batches) != len(other_args_batches):
            raise ValueError(
                f"Mismatched layer args: {len(input_batches)} input batches vs "
                f"{len(other_args_batches)} other-args batches"
            )
        self.input_batches = input_batches
        self.other_args_batches = other_args_batches

    def __len__(self) -> int:
        return len(self.input_batches)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        input_batch = self.input_batches[index]
        other_args_batch = self.other_args_batches[index]
        return (input_batch, *other_args_batch)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class CachedTensorShards:

    def __init__(
        self,
        batches: list[torch.Tensor] | list[tuple[torch.Tensor, ...]],
        num_shards: int,
        *,
        target_device: torch.device | str,
        offload_device: torch.device | str = torch.device("cpu"),
    ):
        if num_shards > len(batches):
            num_shards = max(1, len(batches))
        if _tensor_batch_device(batches[0]) != offload_device:
            self.batches = [_move_tensor_batch(b, offload_device) for b in batches]
        else:
            self.batches = batches
        self.num_shards = num_shards
        self.current_shard: int = None
        self.cached_shard: list[torch.Tensor] | list[tuple[torch.Tensor, ...]] = None
        self.target_device = target_device

    def _switch_shard(self, shard_index: int) -> None:
        if self.current_shard == shard_index:
            return
        self.clear_cache()
        self.current_shard = shard_index
        start, end = self._get_shard_range(shard_index)
        self.cached_shard = self.batches[start:end]
        self.cached_shard = [_move_tensor_batch(b, self.target_device) for b in self.cached_shard]

    def clear_cache(self) -> None:
        self.cached_shard = None
        self.current_shard = None

    def _get_shard_range(self, index: int) -> tuple[int, int]:
        if self.num_shards == 1:
            return 0, len(self.batches)
        shard_size = len(self.batches) // self.num_shards
        start = shard_size * index
        if index == self.num_shards - 1:
            end = len(self.batches)
        else:
            end = shard_size * (index + 1)
        return start, end

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, ...]:
        shard_len = len(self.batches) // self.num_shards
        shard_index = min(index // shard_len, self.num_shards - 1)
        if self.current_shard != shard_index:
            self._switch_shard(shard_index)
        shard_start, _ = self._get_shard_range(shard_index)
        return self.cached_shard[index - shard_start]

    def __iter__(self) -> "Iterator":
        return self.Iterator(self)

    def __len__(self) -> int:
        return len(self.batches)

    class Iterator:
        def __init__(self, batches: "CachedTensorShards"):
            self.batches = batches
            self.current_index = 0

        def __iter__(self):
            return self

        def __next__(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
            if self.current_index >= len(self.batches):
                raise StopIteration
            result = self.batches[self.current_index]
            self.current_index += 1
            return result

        def __len__(self) -> int:
            return len(self.batches)


class CosineAnnealingParam:
    def __init__(self, start_value: float, end_value: float, T_max: int):
        """
        Args:
            start_value (float): The initial value (equivalent to eta_max).
            end_value (float): The final value (equivalent to eta_min).
            T_max (int): Maximum number of steps.
        """
        self.start_value = start_value
        self.end_value = end_value
        self.T_max = T_max
        self._step = -1

    def step(self) -> float:
        self._step += 1

        if self._step >= self.T_max:
            return self.end_value

        cos_val = math.cos(math.pi * self._step / self.T_max)
        return self.end_value + (self.start_value - self.end_value) * (1 + cos_val) / 2


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = TqdmLoggingHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


logger = get_logger("ParoQuant")
