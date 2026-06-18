from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import simple_parsing
import torch
import torch.nn as nn
try:
    import wandb
except ImportError:  # Optional unless --use-wandb is set.
    wandb = None
from tqdm import tqdm

from paroquant.optim.train import optimize_module, get_random_rotation_pairs
from paroquant.optim.qlinear import PseudoQuantizedLinear
from paroquant.optim.qexperts import PseudoQuantizedMoEExperts, get_named_moe_experts, is_fused_moe_experts
from paroquant.optim.util import (
    set_module_by_name,
    load_model,
    move_embed,
    load_tokenizer,
    get_blocks,
    get_calib_dataset,
    get_mixed_calib_dataset,
    capture_layer_inputs_and_args,
    get_named_linears,
    empty_cache,
    logger,
    CachedTensorShards,
    DiskTensorBatchStore,
    LayerArgsBatches,
    RETAINED_KWARG_KEYS,
    to_device,
)
from paroquant.optim.rotation import transform_to_kernel_data
from paroquant.optim.streaming import StreamingModelLoader, materialize_nonlayer, move_real_tensors


@dataclass(kw_only=True)
class Config:
    # Huggingface model path.
    model: str
    # The parameters to optimize at each stage and the corresponding learning rates,
    # e.g., --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6"
    params: list[str]
    # The number of epochs for each stage of optimization,
    # e.g., --epochs 10 10
    epochs: list[int]

    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-10
    # Loss function to use.
    loss: Literal["mse", "smooth_l1"] = "smooth_l1"

    # Quantization & rotation group size.
    group_size: int
    # Bit width.
    n_bit: int
    # Number of rotations.
    num_rotations: int

    skipped_modules: list[str] = field(default_factory=list)

    # Calibration datasets. If more than one dataset is provided,
    # they will be sampled evenly and shuffled.
    datasets: list[str]
    val_dataset: str
    train_size: int
    validation_size: int
    batch_size: int
    val_batch_size: int | None = None
    # Accumulate gradients over multiple micro-batches before each optimizer step.
    # Increasing this reduces peak memory usage but increases training time.
    gradient_accumulation_steps: int = 1
    seqlen: int

    # Number of shards to cache the input/output tensors. At any time, only one shard
    # will be moved to GPU for optimization. The rest will be kept in CPU memory.
    # Increasing this reduces GPU memory usage but increases training time.
    cache_shards: int = 1

    # Optional directory for disk-backed activation spill. When set, large
    # inter-layer activation streams are written to this directory and loaded
    # lazily by shard, reducing host RAM for large calibration sets such as 8192.
    activation_spill_dir: str | None = None
    # Keep spilled activation files after the run for debugging. By default the
    # temporary per-stream spill directories are removed as soon as they are no
    # longer needed.
    keep_activation_spill: bool = False

    # Directory to save state dicts of optimized linear layers.
    output_dir: str

    # Optional existing optimizer result directory to initialize from. This is
    # for non-destructive continuation: load states from this directory, run
    # more optimization, and save the new states into `output_dir`.
    init_from_dir: str | None = None

    # Whether to resume from previously saved results in `output_dir`.
    resume: bool = False
    # Whether to enable gradient checkpointing.
    checkpointing: bool = False
    # Optional smoke/debug limit: optimize only the first N transformer layers.
    # Full artifacts should leave this unset.
    max_layers: int | None = None

    # Stream the model from disk one layer at a time instead of holding the whole
    # model in host RAM. Decoder layers stay on the meta device and are
    # materialized (with block-FP8 dequantization) just before they are
    # optimized, then released. Required to optimize models that do not fit in
    # RAM (e.g. block-FP8 MoE checkpoints such as MiniMax-M2 / DeepSeek).
    stream_from_disk: bool = False

    seed: int

    use_wandb: bool = False


def setup_wandb(args: Config) -> wandb.Run | None:
    if not args.use_wandb:
        return None
    if wandb is None:
        raise RuntimeError("wandb is required when --use-wandb is set")
    wandb_run = wandb.init(config=vars(args))
    logger.info(
        f"wandb logging enabled: entity={wandb_run.entity}, project={wandb_run.project}, run_name={wandb_run.name}"
    )

    return wandb_run


def main():
    args = simple_parsing.parse(Config, add_option_string_dash_variants=simple_parsing.DashVariant.DASH)
    print(args)

    # Store the results in a subdirectory.
    model_name = args.model.split("/")[-1]
    output_dir = Path(args.output_dir)
    output_dir = output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    init_from_dir: Path | None = None
    if args.init_from_dir is not None:
        init_from_base = Path(args.init_from_dir)
        init_from_model_dir = init_from_base / model_name
        init_from_dir = init_from_model_dir if init_from_model_dir.exists() else init_from_base
        if not init_from_dir.exists():
            raise FileNotFoundError(f"--init-from-dir does not exist: {init_from_dir}")
        if init_from_dir.resolve() == output_dir.resolve():
            raise ValueError("--init-from-dir must be different from --output-dir for non-destructive continuation")
        logger.info(f"Initializing optimization states from: {init_from_dir}")

    # Currently only support single GPU training.
    device = "cuda"

    activation_spill_dir: Path | None = None
    if args.activation_spill_dir is not None:
        activation_spill_dir = Path(args.activation_spill_dir)
        if activation_spill_dir.exists() and not args.keep_activation_spill:
            shutil.rmtree(activation_spill_dir)
        activation_spill_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using disk-backed activation spill directory: %s", activation_spill_dir)

    def make_activation_store(name: str):
        if activation_spill_dir is None:
            return []
        return DiskTensorBatchStore(
            activation_spill_dir / name,
            overwrite=True,
            delete_on_cleanup=not args.keep_activation_spill,
        )

    def cleanup_activation_batches(batches) -> None:
        cleanup = getattr(batches, "cleanup", None)
        if cleanup is not None:
            cleanup()

    wandb_run = setup_wandb(args)

    # Determine which params to optimize.
    params_to_optimize: list[dict[str, float]] = []
    for params in args.params:
        params = params.strip().split(",")
        param_dict = {}
        for param in params:
            param, lr = param.strip().split(":")
            param_dict[param.strip()] = float(lr.strip())
        params_to_optimize.append(param_dict)
    print(f"Parameters to optimize: {params_to_optimize}")

    # Save args to output directory
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model.
    weight_loader = None
    if args.stream_from_disk:
        logger.info("Streaming model from disk (per-layer FP8 dequant); decoder layers stay on meta.")
        weight_loader = StreamingModelLoader(args.model, dtype=torch.float16)
        model = weight_loader.build_meta_model()
        materialize_nonlayer(weight_loader, model)
    else:
        model = load_model(args.model, device_map="cpu", dtype=torch.float16).half()
        move_embed(model, device)
    tokenizer = load_tokenizer(args.model)
    blocks = get_blocks(model)

    # Get calibration dataset.
    samples = get_mixed_calib_dataset(
        args.datasets,
        tokenizer=tokenizer,
        n_samples=args.train_size,
        block_size=args.seqlen,
        seed=args.seed,
        split="train",
    )
    samples = torch.stack(samples, dim=0).to(device)

    val_samples = get_calib_dataset(
        args.val_dataset,
        tokenizer=tokenizer,
        n_samples=args.validation_size,
        block_size=args.seqlen,
        seed=args.seed,
        split="validation",
    )
    val_samples = torch.stack(val_samples, dim=0).to(device)

    # Capture per-batch positional args and layer kwargs.
    logger.info("Capturing layer positional args and kwargs...")
    if weight_loader is not None:
        # Decoder layers are on meta; only move the real (non-layer) tensors.
        move_real_tensors(model, device)
    else:
        model.to(device)
    (
        og_layer_input_batches,
        kwargs_list,
        other_args_batches_list,
        og_retained_kwargs_batches,
    ) = capture_layer_inputs_and_args(
        model,
        blocks,
        samples,
        batch_size=args.batch_size,
        first_layer_input_batches=make_activation_store("train/layer000-original-input"),
    )

    val_batch_size = args.val_batch_size or args.batch_size
    (
        og_layer_val_input_batches,
        val_kwargs_list,
        val_other_args_batches_list,
        og_val_retained_kwargs_batches,
    ) = capture_layer_inputs_and_args(
        model,
        blocks,
        val_samples,
        batch_size=val_batch_size,
        first_layer_input_batches=make_activation_store("val/layer000-original-input"),
    )
    new_retained_kwargs_batches = deepcopy(og_retained_kwargs_batches)
    new_val_retained_kwargs_batches = deepcopy(og_val_retained_kwargs_batches)

    if weight_loader is not None:
        move_real_tensors(model, "cpu")
    else:
        model.cpu()

    del samples, val_samples
    empty_cache()

    @torch.no_grad()
    def forward_layer_batch(
        layer: nn.Module,
        args_batched,
        *,
        kwargs: dict,
        retained_kwargs_batches: list[dict],
        store_device: torch.device | str,
        store_name: str,
    ):
        output_batched = make_activation_store(store_name)

        layer.to(device)
        for batch_idx, args_batch in enumerate(args_batched):
            batch_kwargs = kwargs.copy()
            for key in RETAINED_KWARG_KEYS:
                if key in retained_kwargs_batches[batch_idx]:
                    batch_kwargs[key] = to_device(retained_kwargs_batches[batch_idx][key], device)
            output = layer(*to_device(args_batch, device), **batch_kwargs)
            if isinstance(output, tuple):
                output = output[0]
            output = output.detach()
            if output.device != store_device:
                output = output.to(store_device)
            output_batched.append(output)
            del output
            for key in RETAINED_KWARG_KEYS:
                if key in batch_kwargs:
                    retained_kwargs_batches[batch_idx][key] = to_device(batch_kwargs[key], "cpu")
        layer.cpu()

        empty_cache()
        return output_batched

    def make_layer_args_batches(
        input_batches,
        other_args_batches: list[tuple[torch.Tensor, ...]],
    ) -> LayerArgsBatches:
        return LayerArgsBatches(input_batches, other_args_batches)

    def init_rotation_data(
        weight: torch.Tensor,
        *,
        seed: int,
        group_size: int,
        num_rotations: int,
    ) -> list[torch.Tensor]:
        weight_grouped = weight.view(weight.shape[0], -1, group_size).permute(1, 0, 2)
        all_pairs = get_random_rotation_pairs(
            weight_grouped,
            group_size=group_size,
            num_rotations=num_rotations,
            num_pairs_factor=0.5,
            seed=seed,
        )
        all_pairs = [torch.tensor(pairs, device="cpu", dtype=torch.int32) for pairs in all_pairs]
        initial_angles = [torch.zeros(pairs.shape[0], device="cpu") for pairs in all_pairs]
        npairs, angles, mask = transform_to_kernel_data(
            all_pairs,
            initial_angles,
            group_size=group_size,
        )
        return [npairs.to(device), angles.to(device), mask.to(device)]

    def set_checkpointing_enabled(pseudo_modules: dict[str, nn.Module], enable: bool) -> None:
        for pseudo_module in pseudo_modules.values():
            if hasattr(pseudo_module, "enable_checkpoint"):
                pseudo_module.enable_checkpoint = enable

    # Layerwise, multi-stage optimization.
    blocks_to_optimize = blocks
    if args.max_layers is not None:
        if args.max_layers <= 0:
            raise ValueError("--max-layers must be positive when set")
        blocks_to_optimize = blocks[: args.max_layers]
        logger.warning("Debug/smoke mode: optimizing only the first %d layer(s).", len(blocks_to_optimize))

    for layer_idx, layer in enumerate(tqdm(blocks_to_optimize)):
        empty_cache()
        if weight_loader is not None:
            # Materialize this layer's weights from disk (block-FP8 dequant).
            weight_loader.materialize(layer, f"model.layers.{layer_idx}")
        layer_eval_dtype = next(layer.parameters()).dtype
        logger.info(f"Capturing original layer output...")
        # Original output of this layer.
        og_layer_args_batches = make_layer_args_batches(
            og_layer_input_batches,
            other_args_batches_list[layer_idx],
        )
        og_layer_val_args_batches = make_layer_args_batches(
            og_layer_val_input_batches,
            val_other_args_batches_list[layer_idx],
        )
        og_layer_output_batches = forward_layer_batch(
            layer,
            og_layer_args_batches,
            kwargs=kwargs_list[layer_idx],
            retained_kwargs_batches=og_retained_kwargs_batches,
            store_device="cpu",
            store_name=f"train/layer{layer_idx:03d}-original-output",
        )
        og_layer_val_output_batches = forward_layer_batch(
            layer,
            og_layer_val_args_batches,
            kwargs=val_kwargs_list[layer_idx],
            retained_kwargs_batches=og_val_retained_kwargs_batches,
            store_device="cpu",
            store_name=f"val/layer{layer_idx:03d}-original-output",
        )

        # The original input stream for this layer is no longer needed once
        # original outputs have been captured. For layer 0 it is also the
        # quantized-path input, so defer that cleanup until after new output
        # capture below.
        if layer_idx > 0:
            cleanup_activation_batches(og_layer_input_batches)
            cleanup_activation_batches(og_layer_val_input_batches)
            del og_layer_args_batches, og_layer_val_args_batches
            og_layer_input_batches = None
            og_layer_val_input_batches = None
            empty_cache()

        if layer_idx > 0:
            layer_args_batches = make_layer_args_batches(
                new_layer_output_batches,
                other_args_batches_list[layer_idx],
            )
            layer_val_args_batches = make_layer_args_batches(
                new_layer_val_output_batches,
                val_other_args_batches_list[layer_idx],
            )
        else:
            layer_args_batches = og_layer_args_batches
            layer_val_args_batches = og_layer_val_args_batches

        train_args_batches = layer_args_batches
        train_output_batches = og_layer_output_batches

        train_args_batches = CachedTensorShards(train_args_batches, args.cache_shards, target_device=device)
        train_output_batches = CachedTensorShards(train_output_batches, args.cache_shards, target_device=device)

        val_args_batches = [to_device(args_batch, device) for args_batch in layer_val_args_batches]
        val_output_batches = [b.to(device) for b in og_layer_val_output_batches]

        # Freeze all parameters
        for param in layer.parameters():
            param.requires_grad = False

        linear_modules = get_named_linears(layer)
        expert_modules = get_named_moe_experts(layer)
        optim_modules: dict[str, nn.Module] = {}
        optim_modules.update(linear_modules)
        optim_modules.update(expert_modules)
        if args.resume:
            all_files_exist = True
            for name in optim_modules.keys():
                file_name = f"{layer_idx}.{name}.pt"
                file_path = output_dir / file_name
                if not file_path.exists() and name not in args.skipped_modules:
                    all_files_exist = False
                    break
        else:
            all_files_exist = False

        if not all_files_exist:
            logger.info(f"Initializing rotation parameters...")

        linear_names_to_optimize = [name for name in linear_modules.keys() if name not in args.skipped_modules]
        expert_names_to_optimize = [name for name in expert_modules.keys() if name not in args.skipped_modules]
        skipped_linear_names = [name for name in linear_modules.keys() if name in args.skipped_modules]
        skipped_expert_names = [name for name in expert_modules.keys() if name in args.skipped_modules]
        logger.info(f"Linear modules to optimize: {linear_names_to_optimize}")
        logger.info(f"MoE expert modules to optimize: {expert_names_to_optimize}")
        logger.info(f"Skipped linear modules: {skipped_linear_names}")
        logger.info(f"Skipped MoE expert modules: {skipped_expert_names}")

        named_pseudo_modules: dict[str, nn.Module] = {}
        for name, old_module in optim_modules.items():
            if name in args.skipped_modules:
                continue

            if all_files_exist:
                existing_result_file = output_dir / f"{layer_idx}.{name}.pt"
                sd = torch.load(existing_result_file, map_location=device)
                if isinstance(old_module, nn.Linear):
                    new_module = PseudoQuantizedLinear.from_state_dict(sd)
                    set_module_by_name(layer, name, new_module)
                elif is_fused_moe_experts(old_module):
                    new_module = PseudoQuantizedMoEExperts.from_state_dict(sd, old_module, device)
                    set_module_by_name(layer, name, new_module)
                else:
                    raise NotImplementedError(f"Unsupported module type: {type(old_module)}")
                named_pseudo_modules[name] = new_module
                continue

            if init_from_dir is not None:
                init_result_file = init_from_dir / f"{layer_idx}.{name}.pt"
                if init_result_file.exists():
                    sd = torch.load(init_result_file, map_location=device)
                    if isinstance(old_module, nn.Linear):
                        new_module = PseudoQuantizedLinear.from_state_dict(sd)
                        set_module_by_name(layer, name, new_module)
                    elif is_fused_moe_experts(old_module):
                        new_module = PseudoQuantizedMoEExperts.from_state_dict(sd, old_module, device)
                        set_module_by_name(layer, name, new_module)
                    else:
                        raise NotImplementedError(f"Unsupported module type: {type(old_module)}")
                    named_pseudo_modules[name] = new_module
                    logger.info(f"Initialized {layer_idx}.{name} from {init_result_file}")
                    continue

            old_module.to(device)
            if isinstance(old_module, nn.Linear):
                weight = old_module.weight.float()
                rotation_pairs = init_rotation_data(
                    weight,
                    seed=args.seed + layer_idx,
                    group_size=args.group_size,
                    num_rotations=args.num_rotations,
                )
                channel_scales = torch.ones(1, weight.shape[1], dtype=old_module.weight.dtype, device=device)

                new_module = PseudoQuantizedLinear(
                    old_module,
                    rotation_pairs,
                    channel_scales,
                    group_size=args.group_size,
                    n_bits=args.n_bit,
                    num_rotations=args.num_rotations,
                )
                set_module_by_name(layer, name, new_module)
            elif is_fused_moe_experts(old_module):
                gate_up = old_module.gate_up_proj.float().view(-1, old_module.gate_up_proj.shape[-1])
                down = old_module.down_proj.float().view(-1, old_module.down_proj.shape[-1])
                gate_up_rotation_pairs = init_rotation_data(
                    gate_up,
                    seed=args.seed + layer_idx,
                    group_size=args.group_size,
                    num_rotations=args.num_rotations,
                )
                down_rotation_pairs = init_rotation_data(
                    down,
                    seed=args.seed + layer_idx + 97,
                    group_size=args.group_size,
                    num_rotations=args.num_rotations,
                )
                gate_up_channel_scales = torch.ones(
                    1,
                    gate_up.shape[1],
                    dtype=old_module.gate_up_proj.dtype,
                    device=device,
                )
                down_channel_scales = torch.ones(
                    1,
                    down.shape[1],
                    dtype=old_module.down_proj.dtype,
                    device=device,
                )

                new_module = PseudoQuantizedMoEExperts(
                    old_module,
                    gate_up_rotation_pairs,
                    down_rotation_pairs,
                    gate_up_channel_scales,
                    down_channel_scales,
                    group_size=args.group_size,
                    n_bits=args.n_bit,
                    num_rotations=args.num_rotations,
                )
                set_module_by_name(layer, name, new_module)
            else:
                raise NotImplementedError(f"Unsupported module type: {type(old_module)}")

            named_pseudo_modules[name] = new_module
            old_module.cpu()

        if not all_files_exist:
            layer.to(device).float()

            set_checkpointing_enabled(named_pseudo_modules, args.checkpointing)
            layer_step = 0
            wandb_metric_logger = None
            if wandb_run is not None:
                layer_prefix = f"layer_{layer_idx}"
                layer_step_metric = f"{layer_prefix}/step"
                wandb.define_metric(layer_step_metric)
                wandb.define_metric(f"{layer_prefix}/loss", step_metric=layer_step_metric)
                wandb.define_metric(f"{layer_prefix}/val_loss", step_metric=layer_step_metric)
                wandb.define_metric(f"{layer_prefix}/best_val_loss", step_metric=layer_step_metric)

                def _wandb_layer_logger(
                    metrics: dict[str, float], metric_step: int, *, _prefix=layer_prefix, _step_metric=layer_step_metric
                ) -> None:
                    payload = {_step_metric: metric_step}
                    payload.update({f"{_prefix}/{metric_name}": value for metric_name, value in metrics.items()})
                    wandb_run.log(payload)

                wandb_metric_logger = _wandb_layer_logger

            def _reset_named_angles(_: nn.Module) -> None:
                for pseudo_module in named_pseudo_modules.values():
                    if hasattr(pseudo_module, "reset_angles_by_mask"):
                        pseudo_module.reset_angles_by_mask()

            train_kwargs_batches = [
                kwargs_list[layer_idx] | {k: retained_kwargs[k] for k in RETAINED_KWARG_KEYS if k in retained_kwargs}
                for retained_kwargs in new_retained_kwargs_batches
            ]
            val_kwargs_batches = [
                val_kwargs_list[layer_idx]
                | {k: retained_kwargs[k] for k in RETAINED_KWARG_KEYS if k in retained_kwargs}
                for retained_kwargs in new_val_retained_kwargs_batches
            ]

            for step, step_params_dict in enumerate(params_to_optimize):
                empty_cache()
                optim_params = []
                for new_module in named_pseudo_modules.values():
                    new_module.set_optim_enabled(
                        **{param_name: True for param_name in step_params_dict.keys()},
                    )
                    for param_name, lr in step_params_dict.items():
                        optim_params.append(
                            dict(
                                params=new_module.get_optim_params(param_name),
                                lr=lr,
                                weight_decay=args.weight_decay,
                                betas=args.betas,
                                eps=args.eps,
                            )
                        )

                logger.info(
                    f"Optimizing layer {layer_idx}, step {step + 1}/{len(params_to_optimize)}: "
                    f"{', '.join([k for k in step_params_dict])}"
                )

                layer_step = optimize_module(
                    layer,
                    (train_args_batches, train_output_batches),
                    (val_args_batches, val_output_batches),
                    train_kwargs_batches,
                    val_kwargs_batches,
                    optim_params,
                    loss_fn=args.loss,
                    n_iter=args.epochs[step],
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    early_stop=None,
                    post_optim_callback=_reset_named_angles,
                    metric_logger=wandb_metric_logger,
                    start_step=layer_step,
                )

            set_checkpointing_enabled(named_pseudo_modules, False)

            train_args_batches.clear_cache()
            train_output_batches.clear_cache()
            del (
                train_args_batches,
                train_output_batches,
                val_args_batches,
                val_output_batches,
                train_kwargs_batches,
                val_kwargs_batches,
            )
            empty_cache()

        else:
            logger.info(f"Skipping optimization for layer {layer_idx}: already been optimized.")
            train_args_batches.clear_cache()
            train_output_batches.clear_cache()
            del (
                train_args_batches,
                train_output_batches,
                val_args_batches,
                val_output_batches,
            )
            empty_cache()

        layer.to(device=device, dtype=layer_eval_dtype)

        logger.info("Capturing new layer output...")
        next_new_layer_output_batches = forward_layer_batch(
            layer,
            layer_args_batches,
            kwargs=kwargs_list[layer_idx],
            retained_kwargs_batches=new_retained_kwargs_batches,
            store_device="cpu",
            store_name=f"train/layer{layer_idx:03d}-quant-output",
        )
        next_new_layer_val_output_batches = forward_layer_batch(
            layer,
            layer_val_args_batches,
            kwargs=val_kwargs_list[layer_idx],
            retained_kwargs_batches=new_val_retained_kwargs_batches,
            store_device="cpu",
            store_name=f"val/layer{layer_idx:03d}-quant-output",
        )

        if layer_idx > 0:
            cleanup_activation_batches(new_layer_output_batches)
            cleanup_activation_batches(new_layer_val_output_batches)
        else:
            cleanup_activation_batches(og_layer_input_batches)
            cleanup_activation_batches(og_layer_val_input_batches)
        del layer_args_batches, layer_val_args_batches
        if layer_idx == 0:
            del og_layer_args_batches, og_layer_val_args_batches

        new_layer_output_batches = next_new_layer_output_batches
        new_layer_val_output_batches = next_new_layer_val_output_batches
        og_layer_input_batches = og_layer_output_batches
        og_layer_val_input_batches = og_layer_val_output_batches

        if all_files_exist:
            layer.cpu()
            if weight_loader is not None:
                weight_loader.release(layer)
            continue

        # Save the optimized result
        for name, module in named_pseudo_modules.items():
            result_file = output_dir / f"{layer_idx}.{name}.pt"
            torch.save(
                module.state_dict(),
                result_file,
            )

        layer.cpu()
        if weight_loader is not None:
            weight_loader.release(layer)

    cleanup_activation_batches(og_layer_input_batches)
    cleanup_activation_batches(og_layer_val_input_batches)
    if "new_layer_output_batches" in locals():
        cleanup_activation_batches(new_layer_output_batches)
    if "new_layer_val_output_batches" in locals():
        cleanup_activation_batches(new_layer_val_output_batches)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
