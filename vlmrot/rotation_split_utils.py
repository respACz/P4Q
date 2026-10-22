from collections import defaultdict
import logging
import math
import re

import torch

from vlmrot import hadamard_utils, model_utils


def _as_long_index(indices, device):
    if indices is None:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def _as_rotation(rotation, device, dtype):
    if rotation is None:
        return None
    return rotation.to(device=device, dtype=dtype)


def _random_orthogonal_matrix(size, seed, dtype=torch.float32):
    if size <= 0:
        return torch.empty((0, 0), dtype=dtype)
    if size == 1:
        return torch.ones((1, 1), dtype=dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(size, size, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q.to(dtype=dtype)


def _linear_from_module(module):
    if isinstance(module, torch.nn.Linear):
        return module
    wrapped = getattr(module, "module", None)
    if isinstance(wrapped, torch.nn.Linear):
        return wrapped
    raise TypeError(f"Expected Linear or ActQuantWrapper-like module, got {type(module)}")


def _layer_index_from_qlayer_name(name):
    match = re.search(r"layers\.(\d+)", name)
    if match is None:
        raise ValueError(f"Cannot infer layer index from quantized layer name: {name}")
    return int(match.group(1))


def _get_split_gradient(gradient_magnitudes, layer_idx, module_suffix):
    key = f"layers.{layer_idx}.{module_suffix}"
    if key in gradient_magnitudes:
        return gradient_magnitudes[key]
    matches = [
        value for name, value in gradient_magnitudes.items()
        if name.endswith(key) or name.endswith(f"{layer_idx}.{module_suffix}")
    ]
    if not matches:
        raise ValueError(f"Missing gradient statistics for {key}")
    return matches[0]


def _get_matching_v_proj(qlayers, o_proj_name, layer_idx):
    candidates = [
        o_proj_name.replace("self_attn.o_proj", "self_attn.v_proj"),
        f"layers.{layer_idx}.self_attn.v_proj",
    ]
    for candidate in candidates:
        if candidate in qlayers:
            return qlayers[candidate]

    suffix = f"layers.{layer_idx}.self_attn.v_proj"
    matches = [
        qlayer for name, qlayer in qlayers.items()
        if name.endswith(suffix) or name.endswith(f"{layer_idx}.self_attn.v_proj")
    ]
    if not matches:
        raise ValueError(
            "Missing matching v_proj for fused OV split rotation: "
            f"o_proj={o_proj_name}, expected layer {layer_idx} self_attn.v_proj"
        )
    return matches[0]


def _topk_from_topp(gradients, topp_percent):
    if topp_percent <= 0.0 or topp_percent > 100.0:
        raise ValueError(f"topp_percent must be in (0, 100], got {topp_percent}")

    channel_count = int(gradients.numel())
    if channel_count <= 1:
        return channel_count

    sorted_gradients = torch.sort(gradients, descending=True).values
    total = sorted_gradients.sum()
    if total <= 0:
        return 1

    cumulative = torch.cumsum(sorted_gradients, dim=0) / total
    threshold = float(topp_percent) / 100.0
    topk = int(torch.searchsorted(cumulative, torch.tensor(threshold), right=False).item()) + 1
    return max(1, min(topk, channel_count - 1))


def split_channels_by_gradient(gradient_magnitudes, topk=None, topk_percent=10.0, topp_percent=None):
    gradients = gradient_magnitudes.detach().float().cpu().reshape(-1)
    channel_count = int(gradients.numel())
    if channel_count == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    if topp_percent is not None:
        topk = _topk_from_topp(gradients, float(topp_percent))
    elif topk is None:
        topk = math.ceil(channel_count * float(topk_percent) / 100.0)
    if channel_count > 1:
        topk = max(1, min(int(topk), channel_count - 1))
    else:
        topk = 1
    important = torch.topk(gradients, k=topk, largest=True).indices.sort().values
    mask = torch.ones(channel_count, dtype=torch.bool)
    mask[important] = False
    unimportant = torch.arange(channel_count, dtype=torch.long)[mask]
    return important, unimportant


def make_channel_split_rotation_groups(
    gradient_magnitudes,
    topk=None,
    topk_percent=10.0,
    topp_percent=None,
    seed=0,
    group_size=None,
):
    gradients = gradient_magnitudes.detach().float().cpu().reshape(-1)
    channel_count = int(gradients.numel())
    if group_size is None:
        group_size = channel_count
    if group_size <= 0 or channel_count % group_size != 0:
        raise ValueError(
            f"Invalid channel split group_size={group_size} for {channel_count} channels"
        )

    groups = []
    group_id = 0
    for offset in range(0, channel_count, group_size):
        local_grad = gradients[offset: offset + group_size]
        important, unimportant = split_channels_by_gradient(
            local_grad,
            topk=topk,
            topk_percent=topk_percent,
            topp_percent=topp_percent,
        )
        for indices in (important, unimportant):
            if indices.numel() == 0:
                continue
            global_indices = indices + offset
            rotation = _random_orthogonal_matrix(
                int(indices.numel()),
                seed=int(seed) + group_id,
                dtype=torch.float32,
            )
            groups.append((global_indices, rotation))
            group_id += 1
    return groups


def count_channel_split_important_channels(
    gradient_magnitudes,
    topk=None,
    topk_percent=10.0,
    topp_percent=None,
    group_size=None,
):
    gradients = gradient_magnitudes.detach().float().cpu().reshape(-1)
    channel_count = int(gradients.numel())
    if channel_count == 0:
        return 0
    if group_size is None:
        group_size = channel_count
    if group_size <= 0 or channel_count % group_size != 0:
        raise ValueError(
            f"Invalid channel split group_size={group_size} for {channel_count} channels"
        )

    important_count = 0
    for offset in range(0, channel_count, group_size):
        important, _ = split_channels_by_gradient(
            gradients[offset: offset + group_size],
            topk=topk,
            topk_percent=topk_percent,
            topp_percent=topp_percent,
        )
        important_count += int(important.numel())
    return important_count


def _get_layer_topk_percent(args, layer_idx):
    by_layer = getattr(args, "online_channel_split_topk_percent_by_layer", None)
    if by_layer:
        if layer_idx in by_layer:
            return float(by_layer[layer_idx])
        str_layer_idx = str(layer_idx)
        if str_layer_idx in by_layer:
            return float(by_layer[str_layer_idx])
    return float(args.online_channel_split_topk_percent)


def apply_channel_group_rotations(x, groups, fp32=False):
    if not groups:
        return x
    out = x.clone()
    compute = x.float() if fp32 else x
    for indices, rotation in groups:
        idx = _as_long_index(indices, x.device)
        if idx.numel() == 0:
            continue
        rot = _as_rotation(rotation, x.device, compute.dtype)
        out[..., idx] = torch.matmul(compute.index_select(-1, idx), rot).to(x.dtype)
    return out


def apply_channel_split_rotation(
    x,
    important_indices,
    unimportant_indices,
    important_rotation,
    unimportant_rotation,
    fp32=False,
):
    groups = [
        (important_indices, important_rotation),
        (unimportant_indices, unimportant_rotation),
    ]
    return apply_channel_group_rotations(x, groups, fp32=fp32)


def apply_channel_group_rotation_to_linear(module, groups, output=False):
    linear = _linear_from_module(module)
    if not groups:
        return

    weight = linear.weight.data
    dtype = weight.dtype
    device = weight.device
    work = weight.float()
    for indices, rotation in groups:
        idx = _as_long_index(indices, device)
        if idx.numel() == 0:
            continue
        rot = _as_rotation(rotation, device, work.dtype)
        if output:
            work[idx, :] = torch.matmul(rot.t(), work.index_select(0, idx))
        else:
            work[:, idx] = torch.matmul(work.index_select(1, idx), rot)
    linear.weight.data = work.to(device=device, dtype=dtype)

    if output and linear.bias is not None:
        bias = linear.bias.data
        bias_work = bias.float()
        for indices, rotation in groups:
            idx = _as_long_index(indices, bias.device)
            if idx.numel() == 0:
                continue
            rot = _as_rotation(rotation, bias.device, bias_work.dtype)
            bias_work[idx] = torch.matmul(bias_work.index_select(0, idx), rot)
        linear.bias.data = bias_work.to(device=bias.device, dtype=bias.dtype)


def apply_channel_split_rotation_to_linear(
    module,
    important_indices,
    unimportant_indices,
    important_rotation,
    unimportant_rotation,
    output=False,
):
    groups = [
        (important_indices, important_rotation),
        (unimportant_indices, unimportant_rotation),
    ]
    apply_channel_group_rotation_to_linear(module, groups, output=output)


def apply_spinquant_down_proj_rotation(module):
    linear = _linear_from_module(module)
    hadamard_utils.apply_exact_had_to_linear(linear, had_dim=-1, output=False)


def apply_spinquant_ov_rotation(v_proj, o_proj, head_dim, r2=None):
    v_linear = _linear_from_module(v_proj)
    o_linear = _linear_from_module(o_proj)
    if r2 is None:
        hadamard_utils.apply_exact_had_to_linear(v_linear, had_dim=head_dim, output=True)
        hadamard_utils.apply_exact_had_to_linear(o_linear, had_dim=head_dim, output=False)
        return

    _apply_head_rotation_to_linear(v_linear, head_dim=head_dim, rotation=r2, output=True)
    _apply_head_rotation_to_linear(o_linear, head_dim=head_dim, rotation=r2, output=False)


def _apply_head_rotation_to_linear(module, head_dim, rotation, output=False):
    weight = module.weight.data
    dtype = weight.dtype
    device = weight.device
    work = weight.to(dtype=torch.float64)
    rot = rotation.to(device=device, dtype=work.dtype)

    if output:
        work = work.t()
        shape = work.shape
        work = torch.matmul(
            work.reshape(-1, shape[-1] // head_dim, head_dim),
            rot,
        ).reshape(shape).t()
    else:
        shape = work.shape
        work = torch.matmul(
            work.reshape(-1, shape[-1] // head_dim, head_dim),
            rot,
        ).reshape(shape)

    module.weight.data = work.to(device=device, dtype=dtype)


def configure_down_proj_online_rotation(
    model,
    qlayers,
    args,
    apply_weight_rotation=False,
    target_layer_idx=None,
):
    config = model_utils.get_lm_config(model)
    had_K, K = hadamard_utils.get_hadK(config.intermediate_size)
    configured = 0

    for name, qlayer in qlayers.items():
        if "down_proj" not in name:
            continue
        layer_idx = _layer_index_from_qlayer_name(name)
        if target_layer_idx is not None and layer_idx != int(target_layer_idx):
            continue
        if apply_weight_rotation:
            apply_spinquant_down_proj_rotation(qlayer)
        qlayer.online_full_had = True
        qlayer.had_K = had_K
        qlayer.K = K
        qlayer.fp32_had = args.fp32_had
        qlayer.online_channel_split_had = False
        qlayer.channel_split_groups = None
        configured += 1

    if target_layer_idx is None:
        logging.info(
            "[online_channel_split] configured standard down_proj online rotation for %d wrappers",
            configured,
        )
    return configured


def configure_split_online_rotation(
    model,
    qlayers,
    gradient_magnitudes,
    args,
    target_layer_idx=None,
):
    config = model_utils.get_lm_config(model)
    head_dim = config.hidden_size // config.num_attention_heads
    configured = 0
    important_counts = defaultdict(dict)

    for name, qlayer in qlayers.items():
        if "o_proj" not in name:
            continue

        layer_idx = _layer_index_from_qlayer_name(name)
        if target_layer_idx is not None and layer_idx != int(target_layer_idx):
            continue
        gradient = _get_split_gradient(
            gradient_magnitudes,
            layer_idx,
            "self_attn.o_proj",
        )
        topk_percent = _get_layer_topk_percent(args, layer_idx)
        groups = make_channel_split_rotation_groups(
            gradient,
            topk=args.online_channel_split_topk,
            topk_percent=topk_percent,
            topp_percent=getattr(args, "online_channel_split_topp_percent", None),
            seed=args.seed + layer_idx * 1000 + 100000,
            group_size=head_dim,
        )
        v_proj = _get_matching_v_proj(qlayers, name, layer_idx)

        apply_channel_group_rotation_to_linear(
            v_proj,
            groups,
            output=True,
        )
        apply_channel_group_rotation_to_linear(
            qlayer,
            groups,
            output=False,
        )
        qlayer.online_channel_split_had = False
        qlayer.channel_split_groups = None
        v_proj.online_channel_split_had = False
        v_proj.channel_split_groups = None
        important_counts[layer_idx]["o_proj"] = (
            count_channel_split_important_channels(
                gradient,
                topk=args.online_channel_split_topk,
                topk_percent=topk_percent,
                topp_percent=getattr(args, "online_channel_split_topp_percent", None),
                group_size=head_dim,
            )
        )
        configured += 1

    if target_layer_idx is None:
        logging.info(
            "[online_channel_split] configured split online rotation for %d wrappers",
            configured,
        )
    return dict(important_counts)
