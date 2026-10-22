import torch
import torch.nn.functional as F
from vlmrot import model_utils, utils
import csv
import json
import logging
import math
import os
import re
from collections import defaultdict
from tqdm import tqdm


ONLINE_SPLIT_TARGET_SUFFIXES = ("self_attn.o_proj",)

def _clone_to_device(value, device):
    if torch.is_tensor(value):
        return value.detach().clone().to(device)
    if isinstance(value, dict):
        return {k: _clone_to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [_clone_to_device(v, device) for v in value]
    return value

def _prepare_error_check_inputs(args, image_processor, model, cached_inputs=None):
    from vlmrot import data_utils

    n_samples = max(1, args.error_check_samples)
    cpu = torch.device('cpu')
    if cached_inputs:
        return [_clone_to_device(inp, cpu) for inp in cached_inputs[:n_samples]]

    dataloader = data_utils.get_loaders(
        args.cal_dataset, n_samples, args.seed, model.seqlen,
        args.question_file, args.image_base
    )
    inputs = []
    for batch in dataloader:
        # input = model_utils.message_to_prompt(batch, image_processor, type(model))
        input = model_utils.build_loss_inputs(batch, image_processor, model)
        inputs.append(_clone_to_device(input, cpu))
        if len(inputs) >= n_samples:
            break
    return inputs

def _extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        if len(output) > 1 and torch.is_tensor(output[0]) and output[0].ndim == 0:
            return output[1]
        return output[0]
    raise TypeError(f"Cannot extract logits from output type {type(output)}")


def _extract_error_check_loss(output):
    loss = getattr(output, "loss", None)
    if torch.is_tensor(loss):
        return loss
    return _extract_logits(output).sum()


def _forward_logits_on_cpu(model, input):
    model.eval()
    forward_input = _clone_to_device(input, utils.get_dev())
    # forward_input.pop("labels", None)
    with torch.no_grad():
        output = model(**forward_input, use_cache=False)
        logits = _extract_logits(output).detach().float().cpu()
    del output, forward_input
    return logits


def _compute_global_output_error(output_pairs, eps=1e-5):
    diff_sq_sum = 0.0
    original_sq_sum = 0.0
    current_sq_sum = 0.0
    dot_sum = 0.0
    max_abs_error = 0.0
    top1_matches = 0
    kl_sum = 0.0
    valid_samples = 0
    valid_tokens = 0

    for sample_idx, (original_logits, current_logits, attention_mask) in enumerate(
        output_pairs
    ):
        if original_logits.shape != current_logits.shape:
            logging.warning(
                f'[error_check] sample {sample_idx}: output shape mismatch, '
                f'original={tuple(original_logits.shape)}, '
                f'current={tuple(current_logits.shape)}'
            )
            continue
        mask = attention_mask.detach().cpu().bool()
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if original_logits.ndim != 3 or mask.shape != original_logits.shape[:2]:
            logging.warning(
                f'[error_check] sample {sample_idx}: attention mask/logit token '
                f'shape mismatch, mask={tuple(mask.shape)}, '
                f'logits={tuple(original_logits.shape)}'
            )
            continue

        original_valid = original_logits[mask].float()
        current_valid = current_logits[mask].float()
        token_count = int(original_valid.shape[0])
        if token_count == 0:
            continue

        diff = current_valid - original_valid
        diff_sq_sum += float(diff.double().square().sum())
        original_sq_sum += float(original_valid.double().square().sum())
        current_sq_sum += float(current_valid.double().square().sum())
        dot_sum += float(
            (original_valid.double() * current_valid.double()).sum()
        )
        max_abs_error = max(max_abs_error, float(diff.abs().max()))
        top1_matches += int(
            (
                original_valid.argmax(dim=-1)
                == current_valid.argmax(dim=-1)
            ).sum()
        )
        original_log_probs = F.log_softmax(original_valid, dim=-1)
        current_log_probs = F.log_softmax(current_valid, dim=-1)
        kl_sum += float(
            F.kl_div(
                current_log_probs,
                original_log_probs,
                reduction="sum",
                log_target=True,
            )
        )
        valid_samples += 1
        valid_tokens += token_count

    if valid_tokens == 0:
        return {
            "relative_l2": float("nan"),
            "cosine_error": float("nan"),
            "max_abs_error": float("nan"),
            "top1_agreement": float("nan"),
            "kl_divergence": float("nan"),
            "valid_samples": 0,
            "valid_tokens": 0,
        }

    relative_l2 = (diff_sq_sum ** 0.5) / max(
        original_sq_sum ** 0.5, eps
    )
    cosine = dot_sum / max(
        (original_sq_sum * current_sq_sum) ** 0.5, eps
    )
    cosine = min(1.0, max(-1.0, cosine))
    return {
        "relative_l2": relative_l2,
        "cosine_error": 1.0 - cosine,
        "max_abs_error": max_abs_error,
        "top1_agreement": top1_matches / valid_tokens,
        "kl_divergence": kl_sum / valid_tokens,
        "valid_samples": valid_samples,
        "valid_tokens": valid_tokens,
    }


def _log_model_output_error(original_model, current_model, inputs):
    original_model = utils.dispatch_model_to_cuda(original_model)
    current_model = utils.dispatch_model_to_cuda(current_model)
    try:
        def output_pairs():
            for input in inputs:
                original_logits = _forward_logits_on_cpu(original_model, input)
                current_logits = _forward_logits_on_cpu(current_model, input)
                yield (
                    original_logits,
                    current_logits,
                    input["attention_mask"],
                )
                del original_logits, current_logits

        metrics = _compute_global_output_error(output_pairs())
        logging.info(
            '[error_check] summary: '
            f'relative_l2={metrics["relative_l2"]:.6e}, '
            f'cosine_error={metrics["cosine_error"]:.6e}, '
            f'max_abs_error={metrics["max_abs_error"]:.6e}, '
            f'top1_agreement={metrics["top1_agreement"]:.8f}, '
            f'kl_divergence={metrics["kl_divergence"]:.6e}, '
            f'valid_samples={metrics["valid_samples"]}, '
            f'valid_tokens={metrics["valid_tokens"]}'
        )
        return metrics
    finally:
        original_model = utils.move_model_to_cpu(original_model)
        current_model = utils.move_model_to_cpu(current_model)
        utils.cleanup_memory(verbos=False)


def _first_tensor_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Cannot extract tensor output from {type(output)}")


def _rotate_original_activation_to_current_basis(original_activation, rotation_matrix):
    if rotation_matrix is None:
        return original_activation
    if original_activation.shape[-1] != rotation_matrix.shape[0]:
        raise ValueError(
            "Rotation shape mismatch: "
            f"activation_hidden={original_activation.shape[-1]}, "
            f"rotation={tuple(rotation_matrix.shape)}"
        )
    rotation = rotation_matrix.to(
        device=original_activation.device,
        dtype=original_activation.dtype,
    )
    return torch.matmul(original_activation, rotation)


def _compute_layer_input_relative_errors(
    original_input,
    current_input,
    rotation_matrix=None,
    eps=1e-5,
):
    return _compute_layer_input_channel_metrics(
        original_input=original_input,
        current_input=current_input,
        original_grad=None,
        rotation_matrix=rotation_matrix,
        eps=eps,
    )["relative_errors"]


def _inverse_rotate_current_activation_to_baseline_basis(current_activation, rotation_matrix):
    if rotation_matrix is None:
        return current_activation
    if current_activation.shape[-1] != rotation_matrix.shape[0]:
        raise ValueError(
            "Rotation shape mismatch: "
            f"activation_hidden={current_activation.shape[-1]}, "
            f"rotation={tuple(rotation_matrix.shape)}"
        )
    rotation = rotation_matrix.to(
        device=current_activation.device,
        dtype=current_activation.dtype,
    )
    return torch.matmul(current_activation, rotation.t())


def _compute_inverse_rotated_channel_metrics(
    baseline_input,
    current_input,
    rotation_matrix=None,
    eps=1e-5,
):
    baseline_input = baseline_input.detach().float().cpu()
    current_input = current_input.detach().float().cpu()
    if baseline_input.shape != current_input.shape:
        raise ValueError(
            "MLP input shape mismatch: "
            f"baseline={tuple(baseline_input.shape)}, "
            f"current={tuple(current_input.shape)}"
        )
    if baseline_input.ndim < 2:
        raise ValueError(f"Expected MLP input with channel dim, got {tuple(baseline_input.shape)}")

    current_aligned = _inverse_rotate_current_activation_to_baseline_basis(
        current_input,
        rotation_matrix.detach().float().cpu() if torch.is_tensor(rotation_matrix) else rotation_matrix,
    )
    relative_error = (current_aligned - baseline_input).abs() / (
        baseline_input.abs() + eps
    ) * 100.0
    return {
        "relative_errors": _channel_mean_abs(relative_error),
    }


def _channel_mean_abs(tensor):
    return tensor.abs().reshape(-1, tensor.shape[-1]).mean(dim=0)


def _parse_split_sweep_percents(args):
    value = getattr(args, "online_channel_split_topk_percent_sweep", None)
    if value is None or value == "":
        return [float(getattr(args, "online_channel_split_topk_percent", 10.0))]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    percents = [float(part) for part in parts]
    if not percents:
        raise ValueError("online_channel_split_topk_percent_sweep must not be empty")
    invalid = [percent for percent in percents if percent <= 0.0 or percent > 100.0]
    if invalid:
        raise ValueError(
            "online_channel_split_topk_percent_sweep values must be in (0, 100], "
            f"got {invalid}"
        )
    return percents


def _layer_index_from_split_gradient_name(name):
    match = re.search(r"layers\.(\d+)\.self_attn\.o_proj", name)
    if match is None:
        return None
    return int(match.group(1))


def _build_split_sweep_important_indices(
    gradient_magnitudes,
    topk_percent,
    group_size,
):
    from vlmrot import rotation_split_utils

    important_by_layer = {}
    for name, gradient in gradient_magnitudes.items():
        layer_idx = _layer_index_from_split_gradient_name(name)
        if layer_idx is None:
            continue
        gradient = gradient.detach().float().cpu().reshape(-1)
        if group_size <= 0 or int(gradient.numel()) % int(group_size) != 0:
            raise ValueError(
                f"Invalid split sweep group_size={group_size} for {gradient.numel()} channels"
            )

        layer_indices = []
        for offset in range(0, int(gradient.numel()), int(group_size)):
            important, _ = rotation_split_utils.split_channels_by_gradient(
                gradient[offset: offset + int(group_size)],
                topk=None,
                topk_percent=topk_percent,
                topp_percent=None,
            )
            if important.numel() > 0:
                layer_indices.append(important + offset)
        if layer_indices:
            important_by_layer[layer_idx] = torch.cat(layer_indices).sort().values
        else:
            important_by_layer[layer_idx] = torch.empty(0, dtype=torch.long)
    return important_by_layer


def _online_channel_split_cache_path(args):
    cache_root = (
        getattr(args, "online_channel_split_grad_cache_dir", None)
        or getattr(args, "save_backward_result_dir", None)
        or os.path.join(".", "cache", "online_channel_split_gradients")
    )
    model_name = getattr(args, "model", "model").split("/")[-1]
    return os.path.join(
        cache_root,
        model_name,
        f"online_channel_split_gradients_ns{int(args.nsamples)}.pt",
    )


def collect_online_channel_split_gradients(
    model,
    image_processor,
    args,
    target_suffixes=ONLINE_SPLIT_TARGET_SUFFIXES,
):
    from vlmrot import data_utils

    dataloader = data_utils.get_loaders(
        args.cal_dataset,
        args.nsamples,
        args.seed,
        model.seqlen,
        args.question_file,
        args.image_base,
    )
    stats_sum = defaultdict(lambda: None)
    stats_count = defaultdict(int)
    retained_tensors = {}

    def make_hook(name):
        def hook(module, inp):
            tensor = _first_tensor_output(inp)
            if tensor.requires_grad:
                tensor.retain_grad()
                retained_tensors[name].append(tensor)
        return hook

    layers = model_utils.get_layers(model)
    hooks = []
    for layer_idx, layer in enumerate(layers):
        for module_name, module in layer.named_modules():
            if module_name.endswith(target_suffixes):
                full_name = f"layers.{layer_idx}.{module_name}"
                retained_tensors[full_name] = []
                hooks.append(module.register_forward_pre_hook(make_hook(full_name)))

    if not hooks:
        raise ValueError("No online channel split target modules were found")

    model.eval()
    model = utils.dispatch_model_to_cuda(model)
    try:
        logging.info(
            "[online_channel_split] collecting gradient statistics with %d samples",
            args.nsamples,
        )
        with torch.enable_grad():
            for sample_idx, batch in enumerate(tqdm(dataloader, total=args.nsamples)):
                if sample_idx >= args.nsamples:
                    break
                model.zero_grad(set_to_none=True)
                for values in retained_tensors.values():
                    values.clear()
                inputs = model_utils.build_loss_inputs(batch, image_processor, model)
                output = model(**inputs, use_cache=False)
                loss = _extract_error_check_loss(output) / max(int(args.nsamples), 1)
                loss.backward()
                for name, tensors in retained_tensors.items():
                    for tensor in tensors:
                        if tensor.grad is None:
                            continue
                        channel_grad = _channel_mean_abs(tensor.grad.detach().float().cpu())
                        if stats_sum[name] is None:
                            stats_sum[name] = channel_grad
                        else:
                            stats_sum[name] += channel_grad
                        stats_count[name] += 1
                del inputs, output, loss
                utils.cleanup_memory(verbos=False)
    finally:
        _remove_hooks(hooks)
        model = utils.move_model_to_cpu(model)
        utils.cleanup_memory(verbos=False)

    gradients = {}
    for name, summed in stats_sum.items():
        if summed is not None and stats_count[name] > 0:
            gradients[name] = summed / stats_count[name]
    if not gradients:
        raise ValueError("No online channel split gradients were collected")
    return gradients


def load_or_collect_online_channel_split_gradients(model, image_processor, args):
    path = _online_channel_split_cache_path(args)
    if os.path.exists(path):
        saved = torch.load(path, weights_only=False)
        nsamples = saved.get("nsamples")
        gradients = saved.get("gradient_magnitudes", saved)
        if nsamples is not None and int(nsamples) != int(args.nsamples):
            logging.warning(
                "[online_channel_split] cache nsamples=%s differs from args.nsamples=%s; recomputing",
                nsamples,
                args.nsamples,
            )
        elif gradients:
            logging.info("[online_channel_split] loaded gradient statistics from %s", path)
            return gradients

    gradients = collect_online_channel_split_gradients(model, image_processor, args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "gradient_magnitudes": gradients,
            "nsamples": int(args.nsamples),
            "model": getattr(args, "model", None),
        },
        path,
    )
    logging.info("[online_channel_split] saved gradient statistics to %s", path)
    return gradients


def _r1_split_cache_path(args):
    cache_root = (
        getattr(args, "r1_split_grad_cache_dir", None)
        or getattr(args, "save_backward_result_dir", None)
        or os.path.join(".", "cache", "r1_split_gradients")
    )
    model_name = getattr(args, "model", "model").split("/")[-1]
    return os.path.join(
        cache_root,
        model_name,
        f"r1_split_gradients_ns{int(args.nsamples)}.pt",
    )


def collect_r1_split_gradients(model, image_processor, args):
    from vlmrot import data_utils

    dataloader = data_utils.get_loaders(
        args.cal_dataset,
        args.nsamples,
        args.seed,
        model.seqlen,
        args.question_file,
        args.image_base,
    )
    stats_sum = defaultdict(lambda: None)
    stats_count = defaultdict(int)
    retained_tensors = {}

    def make_hook(name):
        def hook(module, inp):
            tensor = _first_tensor_output(inp)
            if tensor.requires_grad:
                tensor.retain_grad()
                retained_tensors[name].append(tensor)
        return hook

    layers = model_utils.get_layers(model)
    hooks = []
    for layer_idx, layer in enumerate(layers):
        full_name = f"layers.{layer_idx}.input"
        retained_tensors[full_name] = []
        hooks.append(layer.register_forward_pre_hook(make_hook(full_name)))

    if not hooks:
        raise ValueError("No transformer layers were found for split-R1 importance collection")

    model.eval()
    model = utils.dispatch_model_to_cuda(model)
    try:
        logging.info(
            "[split_r1] collecting layer-input activation-gradient statistics with %d samples",
            args.nsamples,
        )
        with torch.enable_grad():
            for sample_idx, batch in enumerate(tqdm(dataloader, total=args.nsamples)):
                if sample_idx >= args.nsamples:
                    break
                model.zero_grad(set_to_none=True)
                for values in retained_tensors.values():
                    values.clear()
                inputs = model_utils.build_loss_inputs(batch, image_processor, model)
                output = model(**inputs, use_cache=False)
                loss = _extract_error_check_loss(output) / max(int(args.nsamples), 1)
                loss.backward()
                for name, tensors in retained_tensors.items():
                    for tensor in tensors:
                        if tensor.grad is None:
                            continue
                        importance = _channel_mean_abs(
                            tensor.detach().float().cpu()
                            * tensor.grad.detach().float().cpu()
                        )
                        if stats_sum[name] is None:
                            stats_sum[name] = importance
                        else:
                            stats_sum[name] += importance
                        stats_count[name] += 1
                del inputs, output, loss
                utils.cleanup_memory(verbos=False)
    finally:
        _remove_hooks(hooks)
        model = utils.move_model_to_cpu(model)
        utils.cleanup_memory(verbos=False)

    gradients = {}
    for name, summed in stats_sum.items():
        if summed is not None and stats_count[name] > 0:
            gradients[name] = summed / stats_count[name]
    if not gradients:
        raise ValueError("No split-R1 activation-gradient statistics were collected")
    return gradients


def _aggregate_r1_split_importance(per_layer_importance, reduce_method):
    if not per_layer_importance:
        raise ValueError("split-R1 importance is empty")

    def layer_input_sort_key(name):
        match = re.search(r"layers\.(\d+)\.input", str(name))
        if match is not None:
            return int(match.group(1))
        return _layer_sort_key(str(name))

    layer_names = sorted(per_layer_importance, key=layer_input_sort_key)
    ordered = [
        per_layer_importance[name].detach().float().cpu().reshape(-1)
        for name in layer_names
    ]
    hidden_size = int(ordered[0].numel())
    for name, values in zip(layer_names, ordered):
        if int(values.numel()) != hidden_size:
            raise ValueError(
                "All split-R1 layer importance tensors must share hidden size: "
                f"expected {hidden_size}, got {int(values.numel())} for {name}"
            )
    stacked = torch.stack(ordered, dim=0)
    if reduce_method == "mean":
        return stacked.mean(dim=0)
    if reduce_method == "p95":
        return torch.quantile(stacked, 0.95, dim=0)
    raise ValueError(f"Unknown r1_split_importance_reduce: {reduce_method}")


def build_split_r1_rotation(global_importance, topk_percent=10.0, seed=0):
    from vlmrot import rotation_split_utils

    importance = global_importance.detach().float().cpu().reshape(-1)
    channel_count = int(importance.numel())
    if channel_count == 0:
        raise ValueError("Cannot build split-R1 rotation for zero hidden channels")
    if float(topk_percent) <= 0.0 or float(topk_percent) > 100.0:
        raise ValueError(f"r1_split_topk_percent must be in (0, 100], got {topk_percent}")

    topk = math.ceil(channel_count * float(topk_percent) / 100.0)
    if channel_count > 1:
        topk = max(1, min(int(topk), channel_count - 1))
    else:
        topk = 1

    important = torch.topk(importance, k=topk, largest=True).indices.sort().values
    mask = torch.ones(channel_count, dtype=torch.bool)
    mask[important] = False
    unimportant = torch.arange(channel_count, dtype=torch.long)[mask]

    rotation = torch.zeros((channel_count, channel_count), dtype=torch.float64)
    for group_id, indices in enumerate((important, unimportant)):
        if indices.numel() == 0:
            continue
        block = rotation_split_utils._random_orthogonal_matrix(
            int(indices.numel()),
            seed=int(seed) + group_id,
            dtype=torch.float64,
        )
        rotation[indices[:, None], indices[None, :]] = block

    metadata = {
        "important_indices": important,
        "unimportant_indices": unimportant,
        "topk_percent": float(topk_percent),
        "important_count": int(important.numel()),
        "unimportant_count": int(unimportant.numel()),
    }
    return rotation, metadata


def _split_r1_cache_is_compatible(saved, args):
    expected = {
        "nsamples": int(args.nsamples),
        "topk_percent": float(getattr(args, "r1_split_topk_percent", 10.0)),
        "importance_reduce": str(getattr(args, "r1_split_importance_reduce", "p95")),
        "seed": int(getattr(args, "seed", 0)),
    }
    for key, value in expected.items():
        if key not in saved:
            return False
        if isinstance(value, float):
            if abs(float(saved[key]) - value) > 1e-12:
                return False
        elif saved[key] != value:
            return False
    model_name = getattr(args, "model", None)
    return model_name is None or saved.get("model") in (None, model_name)


def load_or_collect_r1_split_rotation(model, image_processor, args):
    path = _r1_split_cache_path(args)
    if os.path.exists(path):
        saved = torch.load(path, weights_only=False)
        rotation = saved.get("rotation_matrix")
        if _split_r1_cache_is_compatible(saved, args) and torch.is_tensor(rotation):
            logging.info("[split_r1] loaded shared split-R1 rotation from %s", path)
            return rotation.to(device=utils.get_dev(), dtype=torch.float64)
        logging.warning("[split_r1] cache at %s is incompatible; recomputing", path)

    per_layer_importance = collect_r1_split_gradients(model, image_processor, args)
    global_importance = _aggregate_r1_split_importance(
        per_layer_importance,
        getattr(args, "r1_split_importance_reduce", "p95"),
    )
    rotation, metadata = build_split_r1_rotation(
        global_importance,
        topk_percent=getattr(args, "r1_split_topk_percent", 10.0),
        seed=getattr(args, "seed", 0),
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "per_layer_importance": per_layer_importance,
            "global_importance": global_importance,
            "important_indices": metadata["important_indices"],
            "unimportant_indices": metadata["unimportant_indices"],
            "rotation_matrix": rotation,
            "topk_percent": float(getattr(args, "r1_split_topk_percent", 10.0)),
            "importance_reduce": getattr(args, "r1_split_importance_reduce", "p95"),
            "nsamples": int(args.nsamples),
            "model": getattr(args, "model", None),
            "seed": int(getattr(args, "seed", 0)),
        },
        path,
    )
    logging.info(
        "[split_r1] saved shared split-R1 cache to %s; important=%d unimportant=%d reduce=%s",
        path,
        metadata["important_count"],
        metadata["unimportant_count"],
        getattr(args, "r1_split_importance_reduce", "p95"),
    )
    return rotation.to(device=utils.get_dev(), dtype=torch.float64)


def _masked_channel_mean_abs(tensor, mask):
    mask = mask.detach().cpu().bool()
    if mask.shape != tensor.shape[:2]:
        raise ValueError(
            "Modality mask shape mismatch: "
            f"mask={tuple(mask.shape)}, tensor_tokens={tuple(tensor.shape[:2])}"
        )
    selected = tensor[mask]
    if selected.numel() == 0:
        return torch.zeros(tensor.shape[-1], dtype=tensor.dtype)
    return selected.abs().reshape(-1, tensor.shape[-1]).mean(dim=0)


def _compute_layer_input_channel_metrics(
    original_input,
    current_input,
    original_grad=None,
    rotation_matrix=None,
    vision_mask=None,
    text_mask=None,
    eps=1e-5,
):
    original_input = original_input.detach().float().cpu()
    current_input = current_input.detach().float().cpu()
    if original_input.shape != current_input.shape:
        raise ValueError(
            "Layer input shape mismatch: "
            f"original={tuple(original_input.shape)}, "
            f"current={tuple(current_input.shape)}"
        )
    if original_input.ndim < 2:
        raise ValueError(f"Expected layer input with channel dim, got {tuple(original_input.shape)}")

    original_rotated = _rotate_original_activation_to_current_basis(
        original_input,
        rotation_matrix.detach().float().cpu() if torch.is_tensor(rotation_matrix) else rotation_matrix,
    )
    relative_error = (current_input - original_rotated).abs() / (
        original_rotated.abs() + eps
    ) * 100.0
    if original_grad is None:
        grad_rotated = torch.zeros_like(original_rotated)
    else:
        original_grad = original_grad.detach().float().cpu()
        if original_grad.shape != original_input.shape:
            raise ValueError(
                "Layer input gradient shape mismatch: "
                f"input={tuple(original_input.shape)}, "
                f"grad={tuple(original_grad.shape)}"
            )
        grad_rotated = _rotate_original_activation_to_current_basis(
            original_grad,
            rotation_matrix.detach().float().cpu() if torch.is_tensor(rotation_matrix) else rotation_matrix,
        )
    return {
        "relative_errors": _channel_mean_abs(relative_error),
        "activation_magnitudes": _channel_mean_abs(original_rotated),
        "gradient_magnitudes": _channel_mean_abs(grad_rotated),
        "vision_activation_magnitudes": (
            _masked_channel_mean_abs(original_rotated, vision_mask)
            if vision_mask is not None else torch.zeros(original_rotated.shape[-1])
        ),
        "text_activation_magnitudes": (
            _masked_channel_mean_abs(original_rotated, text_mask)
            if text_mask is not None else torch.zeros(original_rotated.shape[-1])
        ),
    }


def _empty_layer_channel_metrics():
    return {
        "relative_errors": [],
        "activation_magnitudes": [],
        "gradient_magnitudes": [],
        "vision_activation_magnitudes": [],
        "text_activation_magnitudes": [],
    }


def _append_layer_input_relative_errors(results, layer_name, metrics):
    if layer_name not in results:
        results[layer_name] = _empty_layer_channel_metrics()
    for key in results[layer_name]:
        results[layer_name][key].append(metrics[key].detach().float().cpu())


def _finalize_layer_input_relative_errors(results):
    for layer_name in sorted(results.keys(), key=_layer_sort_key):
        for key, per_sample_values in results[layer_name].items():
            if not per_sample_values:
                continue
            results[layer_name][key] = torch.stack(
                per_sample_values,
                dim=0,
            ).mean(dim=0)


def _layer_sort_key(layer_name):
    try:
        return int(layer_name.split("_")[-1])
    except ValueError:
        return layer_name


def _get_error_check_output_dir(args):
    return getattr(args, "output_dir", None) or getattr(args, "profile_dir", None) or "."


def _split_error_best_k_path(args):
    explicit_path = getattr(args, "online_channel_split_topk_percent_cache", None)
    if explicit_path:
        return os.fspath(explicit_path)
    return os.path.join(
        os.fspath(_get_error_check_output_dir(args)),
        "split_error_sweep_best_k.json",
    )


def _select_split_error_best_k(rows):
    best_by_layer = {}
    for row in rows:
        if row.get("phase") != "after_quantization":
            continue
        layer = int(row["layer"])
        topk_percent = float(row["topk_percent"])
        mean_error = float(row["mean_error"])
        p95_error = float(row["p95_error"])
        candidate = {
            "layer": layer,
            "topk_percent": topk_percent,
            "mean_error": mean_error,
            "p95_error": p95_error,
            "important_count": int(row["important_count"]),
        }
        current = best_by_layer.get(layer)
        if current is None:
            best_by_layer[layer] = candidate
            continue
        if (p95_error, topk_percent) < (
            float(current["p95_error"]),
            float(current["topk_percent"]),
        ):
            best_by_layer[layer] = candidate
    return dict(sorted(best_by_layer.items()))


def _write_split_error_best_k(best_by_layer, output_dir, path=None):
    json_path = os.fspath(path) if path is not None else os.path.join(
        os.fspath(output_dir),
        "split_error_sweep_best_k.json",
    )
    csv_path = os.path.splitext(json_path)[0] + ".csv"
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

    layers = [
        dict(best_by_layer[layer])
        for layer in sorted(best_by_layer)
    ]
    payload = {
        "metric": "p95_error",
        "phase": "after_quantization",
        "topk_percent_by_layer": {
            str(row["layer"]): float(row["topk_percent"])
            for row in layers
        },
        "layers": layers,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "topk_percent", "mean_error", "p95_error", "important_count"],
        )
        writer.writeheader()
        writer.writerows(layers)

    logging.info("[split_error_sweep] saved best k cache to %s and %s", json_path, csv_path)
    return csv_path, json_path


def load_split_error_best_k(args):
    path = _split_error_best_k_path(args)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        payload = json.load(f)
    values = payload.get("topk_percent_by_layer")
    if not isinstance(values, dict):
        raise ValueError(
            "split error best-k cache must contain a topk_percent_by_layer object: "
            f"{path}"
        )

    topk_by_layer = {}
    for layer, topk_percent in values.items():
        topk_by_layer[int(layer)] = float(topk_percent)
    logging.info(
        "[online_channel_split] loaded per-layer topk_percent for %d layers from %s",
        len(topk_by_layer),
        path,
    )
    return topk_by_layer or None


def _get_error_check_plot_mode(args):
    return getattr(args, "error_check_plot_mode", "relative_error_histogram")


def _cache_fp16_layer_inputs(model, inputs, args):
    cpu = torch.device("cpu")
    device = utils.get_dev()
    layers = model_utils.get_layers(model)
    cached = {layer_idx: [] for layer_idx in range(len(layers))}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp, kwargs):
            tensor = _first_tensor_output(inp).detach().to(cpu, dtype=torch.float16).clone()
            cached[layer_idx].append(
                {
                    "input": tensor,
                    "kwargs": _clone_to_device(kwargs, cpu),
                }
            )
        return hook

    for layer_idx, layer in enumerate(layers):
        hooks.append(layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))

    model.eval()
    model = utils.dispatch_model_to_cuda(model)
    try:
        for input_data in inputs:
            forward_input = _clone_to_device(input_data, device)
            with torch.no_grad():
                model(**forward_input, use_cache=False)
            del forward_input
            utils.cleanup_memory(verbos=False)
    finally:
        _remove_hooks(hooks)
        model = utils.move_model_to_cpu(model)
        utils.cleanup_memory(verbos=False)

    return cached


def _run_single_layer_and_capture_mlp(
    layer,
    layer_input,
    layer_kwargs,
    device,
    input_rotation_matrix=None,
):
    captured = {}

    def hook(module, inp):
        captured["mlp_input"] = _first_tensor_output(inp).detach().float().cpu()

    handle = layer.mlp.register_forward_pre_hook(hook)
    layer.eval()
    layer = layer.to(device)
    try:
        forward_input = layer_input.to(device=device, dtype=next(iter(layer.parameters())).dtype)
        if input_rotation_matrix is not None:
            forward_input = _rotate_original_activation_to_current_basis(
                forward_input,
                input_rotation_matrix,
            )
        kwargs = _clone_to_device(layer_kwargs, device)
        with torch.no_grad():
            output = layer(forward_input, **kwargs)
        output_tensor = _first_tensor_output(output).detach().float().cpu()
        if "mlp_input" not in captured:
            raise ValueError("Could not capture target layer MLP input")
        return output_tensor, captured["mlp_input"]
    finally:
        handle.remove()
        layer = layer.cpu()
        utils.cleanup_memory(verbos=False)


def _compute_single_layer_mlp_error(
    reference_mlp_input,
    current_mlp_input,
    rotation_matrix=None,
    eps=1e-5,
):
    reference_mlp_input = reference_mlp_input.detach().float().cpu()
    current_mlp_input = current_mlp_input.detach().float().cpu()
    current_mlp_input = _inverse_rotate_current_activation_to_baseline_basis(
        current_mlp_input,
        rotation_matrix.detach().float().cpu() if torch.is_tensor(rotation_matrix) else rotation_matrix,
    )
    if reference_mlp_input.shape != current_mlp_input.shape:
        raise ValueError(
            "Single-layer MLP input shape mismatch: "
            f"reference={tuple(reference_mlp_input.shape)}, "
            f"current={tuple(current_mlp_input.shape)}"
        )
    relative_error = (current_mlp_input - reference_mlp_input).abs() / (
        reference_mlp_input.abs() + eps
    ) * 100.0
    return _channel_mean_abs(relative_error)


def _collect_single_layer_reference_mlp_inputs(layer, cached_samples, device):
    reference_mlp_inputs = []
    for sample in cached_samples:
        _, mlp_input = _run_single_layer_and_capture_mlp(
            layer,
            sample["input"],
            sample["kwargs"],
            device,
        )
        reference_mlp_inputs.append(mlp_input)
    return reference_mlp_inputs


def _compute_single_layer_mlp_relative_errors(
    layer,
    cached_samples,
    reference_mlp_inputs,
    device,
    layer_idx,
    phase,
    rotation_matrix=None,
):
    per_sample_errors = []
    for sample, reference_mlp_input in zip(cached_samples, reference_mlp_inputs):
        _, current_mlp_input = _run_single_layer_and_capture_mlp(
            layer,
            sample["input"],
            sample["kwargs"],
            device,
            input_rotation_matrix=rotation_matrix,
        )
        per_sample_errors.append(
            _compute_single_layer_mlp_error(
                reference_mlp_input,
                current_mlp_input,
                rotation_matrix=rotation_matrix,
            )
        )

    if not per_sample_errors:
        return {}

    errors = torch.stack(per_sample_errors, dim=0).mean(dim=0)
    layer_name = f"layer_{layer_idx}"
    return {
        layer_name: {
            "relative_errors": errors,
        }
    }


def _plot_layer_input_error_check_results(vis_utils, results, phase, output_dir, plot_mode):
    if plot_mode in ("relative_error_histogram", "both"):
        vis_utils.plot_layer_input_relative_error_histograms(results, phase, output_dir)
    if plot_mode in ("channel_diagnostics", "both"):
        vis_utils.plot_layer_channel_diagnostics(results, phase, output_dir)
    if plot_mode not in ("relative_error_histogram", "channel_diagnostics", "both"):
        raise ValueError(f"Unknown error_check_plot_mode: {plot_mode}")


def _remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def _compute_and_log_channel_relative_error(
    original_model,
    current_model,
    inputs,
    args,
    phase='after_rotation',
    rotation_matrix=None,
):
    from vlmrot import vis_utils

    original_model = utils.dispatch_model_to_cuda(original_model)
    current_model = utils.dispatch_model_to_cuda(current_model)

    try:
        device = utils.get_dev()
        results = {}

        for input_idx, input_data in enumerate(inputs):
            logging.info(f'[error_check] processing sample {input_idx + 1}/{len(inputs)}')
            forward_input = _clone_to_device(input_data, device)
            vision_mask, text_mask = model_utils.get_input_masks(current_model, forward_input)

            layer_inputs_original = {}
            layer_input_tensors_original = {}
            layer_grads_original = {}
            layer_inputs_current = {}

            def create_original_pre_hook(layer_name):
                def hook(module, inp):
                    tensor = _first_tensor_output(inp)
                    layer_inputs_original[layer_name] = tensor.detach().float().cpu()
                    layer_input_tensors_original[layer_name] = tensor
                    if tensor.requires_grad:
                        tensor.retain_grad()
                return hook

            def create_current_pre_hook(layer_name):
                def hook(module, inp):
                    layer_inputs_current[layer_name] = _first_tensor_output(inp).detach().float().cpu()
                return hook

            original_layers = model_utils.get_layers(original_model)
            current_layers = model_utils.get_layers(current_model)
            hooks = []

            for layer_idx, layer in enumerate(original_layers):
                layer_name = f'layer_{layer_idx}'
                hooks.append(layer.register_forward_pre_hook(create_original_pre_hook(layer_name)))

            original_model.eval()
            current_model.eval()
            original_model.zero_grad(set_to_none=True)
            try:
                with torch.enable_grad():
                    output_orig = original_model(**forward_input, use_cache=False)
                    loss_orig = _extract_error_check_loss(output_orig)
                    loss_orig.backward()
            finally:
                _remove_hooks(hooks)

            for layer_name, tensor in layer_input_tensors_original.items():
                if tensor.grad is not None:
                    layer_grads_original[layer_name] = tensor.grad.detach().float().cpu()

            hooks = []
            for layer_idx, layer in enumerate(current_layers):
                layer_name = f'layer_{layer_idx}'
                hooks.append(layer.register_forward_pre_hook(create_current_pre_hook(layer_name)))

            try:
                with torch.no_grad():
                    output_curr = current_model(**forward_input, use_cache=False)
            finally:
                _remove_hooks(hooks)

            for layer_idx in range(min(len(original_layers), len(current_layers))):
                layer_name = f'layer_{layer_idx}'
                if layer_name not in layer_inputs_original or layer_name not in layer_inputs_current:
                    continue

                try:
                    metrics = _compute_layer_input_channel_metrics(
                        original_input=layer_inputs_original[layer_name],
                        current_input=layer_inputs_current[layer_name],
                        original_grad=layer_grads_original.get(layer_name),
                        rotation_matrix=rotation_matrix,
                        vision_mask=vision_mask,
                        text_mask=text_mask,
                    )
                except ValueError as exc:
                    logging.warning(f'[error_check] {layer_name}: {exc}')
                    continue
                _append_layer_input_relative_errors(results, layer_name, metrics)

            del (
                forward_input,
                layer_inputs_original,
                layer_input_tensors_original,
                layer_grads_original,
                layer_inputs_current,
                output_orig,
                output_curr,
            )
            original_model.zero_grad(set_to_none=True)
            utils.cleanup_memory(verbos=False)

        _finalize_layer_input_relative_errors(results)
        for layer_name in sorted(results.keys(), key=_layer_sort_key):
            errors = results[layer_name]["relative_errors"]
            logging.info(
                f'[error_check] {phase} {layer_name} input activation relative error: '
                f'mean={errors.mean().item():.4f}%, '
                f'median={errors.median().item():.4f}%, '
                f'p95={torch.quantile(errors, 0.95).item():.4f}%, '
                f'max={errors.max().item():.4f}%, '
                f'channels={errors.numel()}, '
                f'act_mean={results[layer_name]["activation_magnitudes"].mean().item():.6e}, '
                f'grad_mean={results[layer_name]["gradient_magnitudes"].mean().item():.6e}'
            )

        _plot_layer_input_error_check_results(
            vis_utils,
            results,
            phase,
            _get_error_check_output_dir(args),
            _get_error_check_plot_mode(args),
        )
        return results

    finally:
        original_model = utils.move_model_to_cpu(original_model)
        current_model = utils.move_model_to_cpu(current_model)
        utils.cleanup_memory(verbos=False)


def _compute_and_log_mlp_relative_error_to_fp16_baseline(
    baseline_model,
    current_model,
    inputs,
    args,
    phase,
    rotation_matrix=None,
):
    baseline_model = utils.dispatch_model_to_cuda(baseline_model)
    current_model = utils.dispatch_model_to_cuda(current_model)

    try:
        device = utils.get_dev()
        results = {}

        for input_idx, input_data in enumerate(inputs):
            logging.info(
                f'[split_error_sweep] {phase} processing sample {input_idx + 1}/{len(inputs)}'
            )
            forward_input = _clone_to_device(input_data, device)
            baseline_mlp_inputs = {}
            current_mlp_inputs = {}

            def create_mlp_pre_hook(storage, layer_name):
                def hook(module, inp):
                    storage[layer_name] = _first_tensor_output(inp).detach().float().cpu()
                return hook

            baseline_layers = model_utils.get_layers(baseline_model)
            current_layers = model_utils.get_layers(current_model)
            hooks = []

            for layer_idx, layer in enumerate(baseline_layers):
                layer_name = f'layer_{layer_idx}'
                hooks.append(layer.mlp.register_forward_pre_hook(
                    create_mlp_pre_hook(baseline_mlp_inputs, layer_name)
                ))

            baseline_model.eval()
            current_model.eval()
            try:
                with torch.no_grad():
                    output_baseline = baseline_model(**forward_input, use_cache=False)
            finally:
                _remove_hooks(hooks)

            hooks = []
            for layer_idx, layer in enumerate(current_layers):
                layer_name = f'layer_{layer_idx}'
                hooks.append(layer.mlp.register_forward_pre_hook(
                    create_mlp_pre_hook(current_mlp_inputs, layer_name)
                ))

            try:
                with torch.no_grad():
                    output_current = current_model(**forward_input, use_cache=False)
            finally:
                _remove_hooks(hooks)

            for layer_idx in range(min(len(baseline_layers), len(current_layers))):
                layer_name = f'layer_{layer_idx}'
                if layer_name not in baseline_mlp_inputs or layer_name not in current_mlp_inputs:
                    continue
                try:
                    metrics = _compute_inverse_rotated_channel_metrics(
                        baseline_input=baseline_mlp_inputs[layer_name],
                        current_input=current_mlp_inputs[layer_name],
                        rotation_matrix=rotation_matrix,
                    )
                except ValueError as exc:
                    logging.warning(f'[split_error_sweep] {phase} {layer_name}: {exc}')
                    continue
                if layer_name not in results:
                    results[layer_name] = {"relative_errors": []}
                results[layer_name]["relative_errors"].append(
                    metrics["relative_errors"].detach().float().cpu()
                )

            del (
                forward_input,
                baseline_mlp_inputs,
                current_mlp_inputs,
                output_baseline,
                output_current,
            )
            utils.cleanup_memory(verbos=False)

        for layer_name in sorted(results.keys(), key=_layer_sort_key):
            per_sample_values = results[layer_name]["relative_errors"]
            results[layer_name]["relative_errors"] = torch.stack(
                per_sample_values,
                dim=0,
            ).mean(dim=0)
            errors = results[layer_name]["relative_errors"]
            logging.info(
                f'[split_error_sweep] {phase} {layer_name} MLP input relative error: '
                f'mean={errors.mean().item():.4f}%, '
                f'p95={torch.quantile(errors, 0.95).item():.4f}%, '
                f'max={errors.max().item():.4f}%, '
                f'channels={errors.numel()}'
            )
        return results

    finally:
        baseline_model = utils.move_model_to_cpu(baseline_model)
        current_model = utils.move_model_to_cpu(current_model)
        utils.cleanup_memory(verbos=False)


def _summarize_split_error_sweep_phase(
    results,
    important_indices_by_layer,
    topk_percent,
    phase,
):
    rows = []
    for layer_name in sorted(results.keys(), key=_layer_sort_key):
        layer_idx = _layer_sort_key(layer_name)
        important = important_indices_by_layer.get(layer_idx)
        if important is None or important.numel() == 0:
            continue
        errors = results[layer_name]["relative_errors"].detach().float().cpu()
        important = important.to(dtype=torch.long)
        important = important[important < errors.numel()]
        if important.numel() == 0:
            continue
        selected = errors.index_select(0, important)
        rows.append(
            {
                "phase": phase,
                "layer": int(layer_idx),
                "topk_percent": float(topk_percent),
                "important_count": int(important.numel()),
                "mean_error": float(selected.mean().item()),
                "p95_error": float(torch.quantile(selected, 0.95).item()),
            }
        )
    return rows


def _clone_model_state_dict_to_cpu(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _clone_quantizer_attrs(quantizer):
    return {
        "bits": getattr(quantizer, "bits", 16),
        "groupsize": getattr(quantizer, "groupsize", -1),
        "sym": getattr(quantizer, "sym", False),
        "clip_ratio": getattr(quantizer, "clip_ratio", 1.0),
    }


def _restore_quantizer_attrs(quantizer, attrs):
    for key, value in attrs.items():
        setattr(quantizer, key, value)


def _clone_wrapper_attrs(model):
    from vlmrot import quant_utils

    attrs = {}
    for name, wrapper in quant_utils.find_qlayers(
        model,
        layers=[quant_utils.ActQuantWrapper],
    ).items():
        had_K = getattr(wrapper, "had_K", None)
        if torch.is_tensor(had_K):
            had_K = had_K.detach().cpu().clone()
        attrs[name] = {
            "had_K": had_K,
            "K": getattr(wrapper, "K", 1),
            "online_full_had": getattr(wrapper, "online_full_had", False),
            "online_full_hadnopow2": getattr(wrapper, "online_full_hadnopow2", False),
            "online_partial_had": getattr(wrapper, "online_partial_had", False),
            "online_random_had": getattr(wrapper, "online_random_had", False),
            "online_channel_split_had": getattr(wrapper, "online_channel_split_had", False),
            "channel_split_groups": getattr(wrapper, "channel_split_groups", None),
            "had_dim": getattr(wrapper, "had_dim", 0),
            "fp32_had": getattr(wrapper, "fp32_had", False),
            "fuse_bias": getattr(wrapper, "fuse_bias", False),
            "fuse_linear": getattr(wrapper, "fuse_linear", False),
            "bias_param": getattr(wrapper, "bias_param", None),
            "quantizer": _clone_quantizer_attrs(wrapper.quantizer),
            "out_quantizer": _clone_quantizer_attrs(wrapper.out_quantizer),
        }
    return attrs


def _clone_split_sweep_model_state(model):
    return {
        "state_dict": _clone_model_state_dict_to_cpu(model),
        "wrapper_attrs": _clone_wrapper_attrs(model),
    }


def _restore_wrapper_attrs(model, wrapper_attrs):
    if not wrapper_attrs:
        return
    from vlmrot import quant_utils

    wrappers = quant_utils.find_qlayers(
        model,
        layers=[quant_utils.ActQuantWrapper],
    )
    for name, attrs in wrapper_attrs.items():
        if name not in wrappers:
            continue
        wrapper = wrappers[name]
        for key, value in attrs.items():
            if key == "quantizer":
                _restore_quantizer_attrs(wrapper.quantizer, value)
            elif key == "out_quantizer":
                _restore_quantizer_attrs(wrapper.out_quantizer, value)
            else:
                setattr(wrapper, key, value)


def _restore_model_state_dict(model, state_dict):
    if "state_dict" in state_dict:
        wrapper_attrs = state_dict.get("wrapper_attrs", {})
        state_dict = state_dict["state_dict"]
    else:
        wrapper_attrs = {}
    # Restore optional buffers such as ActQuantWrapper.had_K before asking the
    # module for state_dict keys. Otherwise a previous sweep candidate can leave
    # extra keys that were absent from the saved baseline state.
    _restore_wrapper_attrs(model, wrapper_attrs)
    current_state = model.state_dict()
    model.load_state_dict(
        {
            name: state_dict[name].to(device=value.device, dtype=value.dtype)
            for name, value in current_state.items()
        },
        strict=True,
    )
    _restore_wrapper_attrs(model, wrapper_attrs)


def _write_split_error_sweep_summary(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "split_error_sweep_summary.csv")
    json_path = os.path.join(output_dir, "split_error_sweep_summary.json")
    fieldnames = [
        "phase",
        "layer",
        "topk_percent",
        "important_count",
        "mean_error",
        "p95_error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    logging.info("[split_error_sweep] saved summary to %s and %s", csv_path, json_path)
    return csv_path, json_path


def _split_sweep_layer_indices(gradient_magnitudes):
    layer_indices = set()
    for name in gradient_magnitudes:
        layer_idx = _layer_index_from_split_gradient_name(name)
        if layer_idx is not None:
            layer_indices.add(layer_idx)
    return sorted(layer_indices)


def run_split_error_sweep(
    baseline_model,
    current_model,
    inputs,
    args,
    rotation_matrix,
    gradient_magnitudes,
    apply_split_rotation,
    apply_weight_quantization,
    configure_activation_quantization,
):
    from vlmrot import vis_utils

    if getattr(args, "online_channel_split_topk", None) is not None:
        raise ValueError("split_error_sweep requires online_channel_split_topk=None")
    if getattr(args, "online_channel_split_topp_percent", None) is not None:
        raise ValueError("split_error_sweep does not support online_channel_split_topp_percent")

    config = model_utils.get_lm_config(current_model)
    head_dim = config.hidden_size // config.num_attention_heads
    percents = _parse_split_sweep_percents(args)
    output_dir = _get_error_check_output_dir(args)
    baseline_state = _clone_split_sweep_model_state(current_model)
    original_percent = getattr(args, "online_channel_split_topk_percent", None)
    layer_indices = _split_sweep_layer_indices(gradient_magnitudes)
    rows = []

    try:
        _restore_model_state_dict(current_model, baseline_state)
        cached_layer_inputs = _cache_fp16_layer_inputs(baseline_model, inputs, args)
        baseline_layers = model_utils.get_layers(baseline_model)
        current_layers = model_utils.get_layers(current_model)
        device = utils.get_dev()

        for layer_idx in tqdm(layer_indices, desc="Split error sweep layers"):
            if layer_idx >= len(current_layers) or layer_idx >= len(baseline_layers):
                logging.warning(
                    "[split_error_sweep] skipping layer_%d because baseline/current layer counts are %d/%d",
                    layer_idx,
                    len(baseline_layers),
                    len(current_layers),
                )
                continue
            cached_samples = cached_layer_inputs.get(layer_idx, [])
            if not cached_samples:
                logging.warning(
                    "[split_error_sweep] skipping layer_%d because no cached inputs were captured",
                    layer_idx,
                )
                continue

            _restore_model_state_dict(current_model, baseline_state)
            baseline_layers = model_utils.get_layers(baseline_model)
            current_layers = model_utils.get_layers(current_model)
            reference_mlp_inputs = _collect_single_layer_reference_mlp_inputs(
                baseline_layers[layer_idx],
                cached_samples,
                device,
            )
            layer_rows = []

            for topk_percent in percents:
                args.online_channel_split_topk_percent = float(topk_percent)
                important_indices = _build_split_sweep_important_indices(
                    gradient_magnitudes=gradient_magnitudes,
                    topk_percent=float(topk_percent),
                    group_size=head_dim,
                )
                important_indices = {
                    layer_idx: important_indices.get(
                        layer_idx,
                        torch.empty(0, dtype=torch.long),
                    )
                }

                _restore_model_state_dict(current_model, baseline_state)
                apply_split_rotation(target_layer_idx=layer_idx)
                current_layers = model_utils.get_layers(current_model)
                after_rotation = _compute_single_layer_mlp_relative_errors(
                    current_layers[layer_idx],
                    cached_samples,
                    reference_mlp_inputs,
                    device,
                    layer_idx,
                    "after_rotation",
                    rotation_matrix=rotation_matrix,
                )
                phase_rows = _summarize_split_error_sweep_phase(
                    after_rotation,
                    important_indices,
                    topk_percent,
                    "after_rotation",
                )
                rows.extend(phase_rows)
                layer_rows.extend(phase_rows)

                apply_weight_quantization(target_layer_idx=layer_idx)
                configure_activation_quantization(target_layer_idx=layer_idx)
                current_layers = model_utils.get_layers(current_model)
                after_quantization = _compute_single_layer_mlp_relative_errors(
                    current_layers[layer_idx],
                    cached_samples,
                    reference_mlp_inputs,
                    device,
                    layer_idx,
                    "after_quantization",
                    rotation_matrix=rotation_matrix,
                )
                phase_rows = _summarize_split_error_sweep_phase(
                    after_quantization,
                    important_indices,
                    topk_percent,
                    "after_quantization",
                )
                rows.extend(phase_rows)
                layer_rows.extend(phase_rows)

            layer_best_by_layer = _select_split_error_best_k(layer_rows)
            vis_utils.plot_split_error_sweep_layer_curves(
                layer_rows,
                output_dir,
                best_k_by_layer=layer_best_by_layer,
            )
    finally:
        _restore_model_state_dict(current_model, baseline_state)
        if original_percent is not None:
            args.online_channel_split_topk_percent = original_percent

    _write_split_error_sweep_summary(rows, output_dir)
    best_by_layer = _select_split_error_best_k(rows)
    _write_split_error_best_k(
        best_by_layer,
        output_dir,
        path=_split_error_best_k_path(args),
    )
    for layer, best in best_by_layer.items():
        logging.info(
            "[split_error_sweep] selected layer_%d topk_percent=%.4f by %s p95_error=%.6f",
            layer,
            float(best["topk_percent"]),
            "after_quantization",
            float(best["p95_error"]),
        )
    return rows
