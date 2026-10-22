import logging
import math
import os, copy
import sys
import functools
from contextlib import nullcontext

import torch
from tqdm import tqdm

from vlmrot import data_utils, flatquant_utils, model_utils, utils, visual_pruning


logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def _resolve_config_path(config_path):
    if not config_path or os.path.exists(config_path):
        return config_path
    project_root = os.path.dirname(os.path.abspath(__file__))
    candidates = (
        os.path.join(project_root, config_path),
        os.path.join(project_root, "configs", config_path),
    )
    return next((path for path in candidates if os.path.exists(path)), config_path)


def parse_args():
    original_argv = sys.argv
    resolved_argv = list(original_argv)
    for index, value in enumerate(resolved_argv):
        if value == "--config" and index + 1 < len(resolved_argv):
            resolved_argv[index + 1] = _resolve_config_path(resolved_argv[index + 1])
        elif value.startswith("--config="):
            config_path = value.split("=", 1)[1]
            resolved_argv[index] = f"--config={_resolve_config_path(config_path)}"
    sys.argv = resolved_argv
    try:
        args = utils.parser_gen()
    finally:
        sys.argv = original_argv
    return args


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _detach_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    return value


def _layer_forward(layer, hidden_states, layer_kwargs):
    output = layer(hidden_states, **layer_kwargs)
    if isinstance(output, tuple):
        return output[0]
    return output


def _set_layer_flatquant_mode(layer, ori_mode):
    layer.self_attn._ori_mode = ori_mode
    layer.mlp._ori_mode = ori_mode


def _validate_args(args):
    model_identity = str(args.model or "").lower()
    supported_model = any(
        name in model_identity
        for name in ("qwen2.5-vl", "qwen2-vl", "llava-v1.6", "llava-1.5", "llava-v1.5")
    )
    if args.model and not supported_model:
        raise ValueError(
            "FlatQuant training currently supports Qwen2-VL, LLaVA-1.5, "
            "and LLaVA-Next"
        )
    if args.a_groupsize > 0:
        raise NotImplementedError("FlatQuant activation groupsize is not supported yet")
    if not any([args.cali_trans, args.add_diag, args.lwc, args.lac]):
        logging.warning(
            "No FlatQuant trainable option enabled; saving initialized matrices only."
        )
    if getattr(args, "flatquant_prune_calibration", False):
        if not getattr(args, "visual_prune", False):
            raise ValueError(
                "flatquant_prune_calibration requires visual_prune=True"
            )
        calibration_prune_strategy = getattr(args, "visual_prune_strategy", None)
        if calibration_prune_strategy not in ("vispruner", "crisp", "p4q"):
            raise ValueError(
                "flatquant_prune_calibration requires "
                "visual_prune_strategy=vispruner, crisp, or p4q"
            )
        if calibration_prune_strategy == "vispruner":
            if not getattr(args, "visual_prune_vispruner_quant_vit_features", False):
                raise ValueError(
                    "VisPruner FlatQuant calibration requires quantized ViT "
                    "features for diversity selection"
                )
        if calibration_prune_strategy == "vispruner":
            strategy_label = "VisPruner"
            visual_bits = getattr(args, "visual_prune_quant_bits", None)
            visual_bits = args.a_bits if visual_bits is None else int(visual_bits)
            if visual_bits != int(args.a_bits):
                raise ValueError(
                    "visual_prune_quant_bits must equal a_bits during "
                    f"{strategy_label} FlatQuant calibration"
                )
            if int(getattr(args, "visual_prune_quant_groupsize", -1)) != int(
                args.a_groupsize
            ):
                raise ValueError(
                    "visual_prune_quant_groupsize must equal a_groupsize during "
                    f"{strategy_label} FlatQuant calibration"
                )
            if bool(getattr(args, "visual_prune_quant_asym", False)) != bool(args.a_asym):
                raise ValueError(
                    "visual_prune_quant_asym must match a_asym during "
                    f"{strategy_label} FlatQuant calibration"
                )
        if getattr(args, "visual_prune_recycle", False):
            raise ValueError("pruned FlatQuant calibration does not support recycling")

    real_export_dir = getattr(args, "real_quant_export_dir", None)
    if getattr(args, "real_quant_export_only", False) and not real_export_dir:
        raise ValueError("real_quant_export_only requires real_quant_export_dir")
    if real_export_dir:
        if getattr(args, "flatquant_mode", "train") != "eval":
            raise ValueError(
                "real_quant_export_dir currently requires flatquant_mode=eval "
                "with an existing flat_matrices.pth"
            )
        if args.w_bits != 4 or args.a_bits != 4:
            raise ValueError("real quant export requires W4A4")
        if bool(getattr(args, "w_asym", False)):
            raise ValueError("real quant export requires symmetric W4 weights")


@torch.no_grad()
def capture_first_layer_inputs(model, processor, dataloader, args, device):
    layers = model_utils.get_layers(model)
    inps = []
    layer_kwargs = []
    pruning_stats = []

    class StopForward(Exception):
        pass

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, *unused_args, **kwargs):
            kwargs["use_cache"] = False
            inps.append(hidden_states.detach().cpu())
            layer_kwargs.append(_detach_to_cpu(kwargs))
            raise StopForward

    layers[0] = Catcher(layers[0])
    model = utils.dispatch_model_to_cuda(model)
    input_device = utils.model_first_device(model)
    try:
        for batch in dataloader:
            if len(inps) >= args.nsamples:
                break
            inputs = model_utils.build_loss_inputs(batch, processor, model)
            inputs = _move_to_device(inputs, input_device)
            model_inputs = inputs
            if getattr(args, "flatquant_prune_calibration", False):
                model_inputs, sample_pruning_stats = (
                    visual_pruning.prepare_pruned_generation_inputs(
                        model,
                        inputs,
                        args,
                        query_text="\n".join(
                            str(item.get("value", ""))
                            for item in batch
                            if item.get("type") == "text"
                        ),
                        tokenizer=getattr(processor, "tokenizer", None),
                    )
                )
                pruning_stats.append(sample_pruning_stats)
            try:
                model(**model_inputs, use_cache=False)
            except StopForward:
                pass
    finally:
        layers[0] = layers[0].module
        model = utils.move_model_to_cpu(model)
        utils.cleanup_memory(verbos=False)

    if not inps:
        raise RuntimeError("Failed to capture any FlatQuant calibration inputs")
    if pruning_stats:
        original_visual = sum(item.original_visual_tokens for item in pruning_stats)
        kept_visual = sum(item.kept_visual_tokens for item in pruning_stats)
        original_prompt = sum(item.original_prompt_tokens for item in pruning_stats)
        kept_prompt = sum(item.kept_prompt_tokens for item in pruning_stats)
        logging.info(
            "[FlatQuant calibration %s] samples=%d visual_tokens=%d->%d "
            "(keep=%.6f) prompt_tokens=%d->%d (keep=%.6f)",
            getattr(args, "visual_prune_strategy", "unknown"),
            len(pruning_stats),
            original_visual,
            kept_visual,
            kept_visual / max(1, original_visual),
            original_prompt,
            kept_prompt,
            kept_prompt / max(1, original_prompt),
        )
    return inps[: args.nsamples], layer_kwargs[: args.nsamples]


def _prepare_trainable_params(layer, args):
    for param in layer.parameters():
        param.requires_grad = False
    optimizer_groups = []
    if args.cali_trans:
        params = [
            param
            for name, param in layer.named_parameters()
            if "trans" in name and "diag_scale" not in name
        ]
        if params:
            optimizer_groups.append({"params": params, "lr": args.flat_lr})
    if args.add_diag:
        params = [
            param
            for name, param in layer.named_parameters()
            if "diag_scale" in name
        ]
        if params:
            optimizer_groups.append({"params": params, "lr": args.flat_lr})
    if args.lwc:
        params = [
            param
            for name, param in layer.named_parameters()
            if "clip_factor_w" in name
        ]
        if params:
            optimizer_groups.append({"params": params, "lr": args.flat_lr * 10})
    if args.lac:
        params = [
            param
            for name, param in layer.named_parameters()
            if "clip_factor_a" in name
        ]
        if params:
            optimizer_groups.append({"params": params, "lr": args.flat_lr * 10})

    for group in optimizer_groups:
        for param in group["params"]:
            param.requires_grad = True
    return optimizer_groups or None


def _get_traincast(device):
    if device.type == "cuda":
        return functools.partial(torch.amp.autocast, device_type="cuda", dtype=torch.float16)
    if device.type == "cpu":
        return functools.partial(torch.amp.autocast, device_type="cpu", dtype=torch.bfloat16)
    return nullcontext

def _set_flat_quantizers_enabled(module, enabled):
    states = []
    for child in module.modules():
        if isinstance(
            child,
            (flatquant_utils.WeightQuantizer, flatquant_utils.ActivationQuantizer),
        ):
            states.append((child, child.enable))
            child.enable = enabled
    return states


def _restore_flat_quantizer_states(states):
    for quantizer, enabled in states:
        quantizer.enable = enabled


def _reparameterize_layer(layer):
    layer.self_attn.reparameterize()
    layer.mlp.reparameterize()
    if layer.self_attn.ln_trans is not None and layer.self_attn.ln_trans.add_diag:
        flatquant_utils.reparameterize_ln(
            layer.input_layernorm, layer.self_attn.ln_trans
        )
    if layer.mlp.up_gate_trans is not None and layer.mlp.up_gate_trans.add_diag:
        flatquant_utils.reparameterize_ln(
            layer.post_attention_layernorm, layer.mlp.up_gate_trans
        )
    return layer

@torch.no_grad()
def _check_reparameterization_consistency(
    layer, fp_inps, layer_kwargs, args, device, layer_idx
):
    """Compare online transforms with folded transforms, without quantization/RTN."""
    if not all(
        hasattr(part, "reparameterize") for part in (layer.self_attn, layer.mlp)
    ):
        return
    sample_count = min(
        len(fp_inps), int(getattr(args, "flatquant_fold_check_samples", 1))
    )
    if sample_count <= 0:
        return

    # folded_layer = copy.deepcopy(layer).to(device)
    online_states = _set_flat_quantizers_enabled(layer, False)
    # folded_states = _set_flat_quantizers_enabled(folded_layer, False)

    max_abs = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    element_count = 0
    try:
        online_outputs = []
        for sample_idx in range(sample_count):
            hidden = fp_inps[sample_idx].to(device)
            kwargs = _move_to_device(layer_kwargs[sample_idx], device)
            online_outputs.append(_layer_forward(layer, hidden, kwargs).float())

        _reparameterize_layer(layer)
        for sample_idx, online_output in enumerate(online_outputs):
            hidden = fp_inps[sample_idx].to(device)
            kwargs = _move_to_device(layer_kwargs[sample_idx], device)
            folded_output = _layer_forward(layer, hidden, kwargs).float()
            diff = online_output - folded_output
            max_abs = max(max_abs, float(diff.abs().amax().cpu()))
            squared_error += float(diff.square().sum().cpu())
            squared_reference += float(online_output.square().sum().cpu())
            element_count += diff.numel()
    finally:
        _restore_flat_quantizer_states(online_states)

    rmse = math.sqrt(squared_error / max(1, element_count))
    relative_l2 = math.sqrt(squared_error / max(squared_reference, 1e-24))
    tolerance = float(getattr(args, "flatquant_fold_check_tol", 1e-3))
    logging.info(
        "[FlatQuant fold check/no RTN] layer %d samples=%d max_abs=%.6g "
        "rmse=%.6g relative_l2=%.6g tol=%.6g",
        layer_idx,
        sample_count,
        max_abs,
        rmse,
        relative_l2,
        tolerance,
    )
    if relative_l2 > tolerance:
        logging.warning(
            "FlatQuant matrix folding mismatch at layer %d: relative_l2=%.6g",
            layer_idx,
            relative_l2,
        )

    utils.cleanup_memory(verbos=False)


def calibrate_flatquant(model, inps, layer_kwargs, args, device):
    if type(model) == getattr(model_utils, "QWEN2_5_VL", None):
        from vlmrot.qwen25_flatquant_training import calibrate
        return calibrate(model, inps, layer_kwargs, args, device)
    layers = model_utils.get_layers(model)
    loss_fn = torch.nn.MSELoss()
    traincast = _get_traincast(device)

    fp_inps = inps
    for layer_idx, layer in enumerate(tqdm(layers, desc="FlatQuant layers")):
        logging.info("Calibrating FlatQuant layer %d", layer_idx)
        layer = layer.to(device)
        layer.float()
        fp_outs = []

        _set_layer_flatquant_mode(layer, True)
        with torch.no_grad():
            for sample_idx, sample in enumerate(fp_inps):
                hidden = sample.to(device)
                kwargs = _move_to_device(layer_kwargs[sample_idx], device)
                fp_outs.append(_layer_forward(layer, hidden, kwargs).detach().cpu())
        _set_layer_flatquant_mode(layer, False)

        if args.diag_init == "sq_style":
            layer.self_attn.init_diag_scale(alpha=args.diag_alpha)
            layer.mlp.init_diag_scale(alpha=args.diag_alpha)
        elif args.diag_init != "one_style":
            raise ValueError(f"Unknown diag_init: {args.diag_init}")

        optimizer_groups = _prepare_trainable_params(layer, args)
        if optimizer_groups is not None:
            optimizer = torch.optim.AdamW(optimizer_groups)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs * math.ceil(len(fp_inps) / args.cali_bsz)),
                eta_min=args.flat_lr * 1e-3,
            )
            for epoch in range(args.epochs):
                total_loss = 0.0
                optimizer.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    for sample_idx, sample in enumerate(fp_inps):
                        hidden = sample.to(device)
                        kwargs = _move_to_device(layer_kwargs[sample_idx], device)
                        target = fp_outs[sample_idx].to(device)
                        with traincast():
                            quant_out = _layer_forward(layer, hidden, kwargs)
                            loss = loss_fn(quant_out, target)
                            normalized_loss = loss / loss.detach().clamp_min(1e-12)
                        (normalized_loss / args.cali_bsz).backward()
                        total_loss += float(loss.detach().cpu())
                        if (sample_idx + 1) % args.cali_bsz == 0 or sample_idx + 1 == len(fp_inps):
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                logging.info(
                    "layer %d epoch %d/%d mse %.8f",
                    layer_idx,
                    epoch + 1,
                    args.epochs,
                    total_loss / max(1, len(fp_inps)),
                )

        _check_reparameterization_consistency(
            layer, fp_inps, layer_kwargs, args, device, layer_idx
        )

        layers[layer_idx] = layer.cpu()
        fp_inps = fp_outs
        utils.cleanup_memory(verbos=False)

    return model


def _restore_flatquant_decoder_dtype(model):
    """Restore folded decoder weights to the base model deployment dtype.

    FlatQuant matrices and clipping parameters intentionally remain FP32, but
    every folded Linear executed during inference must match the decoder input
    dtype before Accelerate rebuilds its device hooks.
    """
    deployment_dtype = model_utils.get_lm_head(model).weight.dtype
    for layer in model_utils.get_layers(model):
        layer.input_layernorm.to(dtype=deployment_dtype)
        layer.post_attention_layernorm.to(dtype=deployment_dtype)
        for module in layer.modules():
            if isinstance(module, flatquant_utils.FlatQuantizedLinear):
                module.linear.to(dtype=deployment_dtype)
    logging.info(
        "[FlatQuant] restored folded decoder weights to %s",
        deployment_dtype,
    )
    return deployment_dtype


def evaluate_loaded_flatquant(model, processor, args, device):
    from vlmrot import eval_utils, gptq_utils

    logging.info("[FlatQuant direct eval] loading matrices from %s", args.flatquant_matrix_path)
    flatquant_utils.load_flat_matrices(args, model)
    flatquant_utils.reparameterize_model(model)
    flatquant_utils.set_flatquant_attention_backend(model, "sdpa")

    if getattr(args, "quant_method", "rtn") != "rtn":
        raise ValueError("FlatQuant direct evaluation currently requires quant_method=rtn")
    quantizers = None
    if args.w_bits < 16:
        original_w_clip = getattr(args, "w_clip", False)
        args.w_clip = False
        try:
            logging.info(
                "[FlatQuant direct eval] applying RTN W%d without extra MSE clipping",
                args.w_bits,
            )
            quantizers = gptq_utils.rtn_fwrd(model, device, args)
        finally:
            args.w_clip = original_w_clip

    if getattr(args, "real_quant_export_dir", None):
        if args.w_bits != 4 or args.a_bits != 4:
            raise ValueError(
                "real_quant_export_dir requires the W4A4 FlatQuant setting"
            )
        from vlmrot.real_quant_export import export_real_quantized_language_model

        export_path = export_real_quantized_language_model(
            model,
            quantizers,
            args.real_quant_export_dir,
            model_name=args.model_path or args.model,
        )
        if getattr(args, "real_quant_export_only", False):
            return {"real_quant_checkpoint": str(export_path)}

    # RTN derives its output dtype from the layer parameter ordering.  Learned
    # FlatQuant parameters are FP32, so direct eval must explicitly restore
    # folded decoder linears before Accelerate captures weights for dispatch.
    _restore_flatquant_decoder_dtype(model)

    model.eval()
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    matrix_path = args.flatquant_matrix_path
    if not str(matrix_path).lower().endswith(".pth"):
        matrix_path = os.path.join(matrix_path, "flat_matrices.pth")
    checkpoint_tag = f"flatquant_loaded_{os.stat(matrix_path).st_mtime_ns}"
    args.eval_result_dir = os.path.join(args.eval_result_dir, checkpoint_tag)
    os.makedirs(args.eval_result_dir, exist_ok=True)

    results = {}
    for eval_dataset in args.eval_dataset:
        logging.info("[FlatQuant direct eval] evaluating %s", eval_dataset)
        test_loader = data_utils.get_loaders(eval_dataset)
        results[eval_dataset] = eval_utils.evaluator(
            model, test_loader, device, args, processor, eval_dataset
        )
    return results


def train():
    args = parse_args()
    _validate_args(args)
    utils.set_seed(args.seed)
    device = utils.get_dev()

    model, processor = model_utils.get_model(
        args.model,
        args.model_path,
        device_map=args.device_map,
        max_memory_per_gpu=args.max_memory_per_gpu,
    )
    model.eval()
    model = utils.move_model_to_cpu(model)
    utils.cleanup_memory(verbos=False)
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    flatquant_utils.apply_flatquant_to_model(args, model)
    if args.flatquant_mode == "eval":
        evaluate_loaded_flatquant(model, processor, args, device)
        return

    # Preserve the official FlatQuant eager attention path for calibration,
    # including the original target, fake-quant training and fold checks.
    flatquant_utils.set_flatquant_attention_backend(model, "eager")
    dataloader = data_utils.get_loaders(
        args.cal_dataset,
        args.nsamples,
        args.seed,
        model.seqlen,
        args.question_file,
        args.image_base,
    )
    inps, layer_kwargs = capture_first_layer_inputs(
        model, processor, dataloader, args, device
    )
    calibrate_flatquant(model, inps, layer_kwargs, args, device)
    output_dir = flatquant_utils.get_flatquant_output_dir(args)
    if getattr(args, "flatquant_prune_calibration", False):
        output_dir += visual_pruning.result_suffix(args)
    os.makedirs(output_dir, exist_ok=True)
    flatquant_utils.save_flat_matrices(args, model, output_dir)
    exit()

    # Validate the exact in-memory model produced by calibration.  The fold
    # check reparameterizes checked layers in place; fold only layers for which
    # that check was disabled, otherwise transforms would be applied twice.
    from vlmrot import eval_utils, gptq_utils

    for layer in model_utils.get_layers(model):
        if not layer.self_attn._eval_mode:
            _reparameterize_layer(layer)

    # Validation uses SDPA to avoid materializing the full attention matrix.
    # Switch the language-model causal-mask producer at the same time.
    flatquant_utils.set_flatquant_attention_backend(model, "sdpa")

    # calibrate_flatquant calls layer.float(), while the untouched projector,
    # final norm and lm_head remain FP16.  Restore only the original decoder
    # weights to the lm_head deployment dtype; keep learned FlatQuant matrices
    # and clipping parameters in FP32 as they are in the normal main.py path.
    deployment_dtype = _restore_flatquant_decoder_dtype(model)

    if getattr(args, "quant_method", "rtn") != "rtn":
        raise ValueError(
            "Post-training FlatQuant validation currently requires quant_method=rtn"
        )
    if args.w_bits < 16:
        original_w_clip = getattr(args, "w_clip", False)
        args.w_clip = False
        try:
            logging.info(
                "[FlatQuant current-model validation] applying RTN W%d without extra MSE clipping",
                args.w_bits,
            )
            gptq_utils.rtn_fwrd(model, device, args)
        finally:
            args.w_clip = original_w_clip

    model.eval()
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    matrix_path = os.path.join(output_dir, "flat_matrices.pth")
    validation_tag = f"flatquant_current_model_{os.stat(matrix_path).st_mtime_ns}"
    args.eval_result_dir = os.path.join(args.eval_result_dir, validation_tag)
    os.makedirs(args.eval_result_dir, exist_ok=True)

    # FlatQuantizedLinear already applies the trained activation quantizer/LAC;
    # do not add a second VLMRot ActQuantWrapper.
    for eval_dataset in args.eval_dataset:
        logging.info(
            "[FlatQuant current-model validation] evaluating %s", eval_dataset
        )
        test_loader = data_utils.get_loaders(eval_dataset)
        result = eval_utils.evaluator(
            model, test_loader, device, args, processor, eval_dataset
        )
        logging.info(
            "[FlatQuant current-model validation] %s result: %s",
            eval_dataset,
            result,
        )


if __name__ == "__main__":
    train()
