"""Training-free FP16 token-condensation baselines.

The implementations in this module follow the local reference repositories:

* ``VisionZip-main`` (Apache-2.0): dominant CLS-attention tokens plus evenly
  sampled contextual sinks populated by key-space nearest-neighbour merging.
* ``SpecFlow-main``: low-entropy CLS-attention seeds, kNN heat diffusion,
  energy-adaptive quadtree allocation, and coreset sinks.

They are kept independent from the existing VisPruner/CRISP/P4Q selectors so
adding a baseline cannot mutate any of those paths.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def visionzip_condense(
    hidden_states: torch.Tensor,
    cls_attention: torch.Tensor,
    key_metric: torch.Tensor,
    dominant_tokens: int = 54,
    contextual_tokens: int = 10,
):
    """Condense CLIP tokens using the official VisionZip inference rule.

    ``hidden_states`` and ``key_metric`` include the CLS token.  The configured
    dominant budget also includes CLS, matching ``visionzip(model,
    dominant=54, contextual=10)`` in the reference implementation.
    """
    if hidden_states.dim() != 3 or key_metric.dim() != 3:
        raise ValueError("VisionZip hidden states and key metric must be 3D")
    if hidden_states.shape[:2] != key_metric.shape[:2]:
        raise ValueError("VisionZip hidden-state and key-metric token counts differ")

    batch_size, sequence_length, _ = hidden_states.shape
    patch_count = sequence_length - 1
    dominant_tokens = int(dominant_tokens)
    contextual_tokens = int(contextual_tokens)
    if dominant_tokens < 1 or dominant_tokens > sequence_length:
        raise ValueError(
            f"VisionZip dominant_tokens must be in [1, {sequence_length}]"
        )
    residual_count = sequence_length - dominant_tokens
    if contextual_tokens < 0 or contextual_tokens > residual_count:
        raise ValueError(
            "VisionZip contextual_tokens must be non-negative and no larger "
            f"than the {residual_count} residual tokens"
        )

    if cls_attention.dim() == 4:
        cls_attention = cls_attention[:, :, 0, :]
    if cls_attention.dim() == 3:
        cls_attention = cls_attention.sum(dim=1)
    if cls_attention.shape != (batch_size, patch_count):
        raise ValueError(
            "VisionZip CLS attention must address patch tokens only; got "
            f"{tuple(cls_attention.shape)} for {patch_count} patches"
        )

    dominant_patch_count = dominant_tokens - 1
    if dominant_patch_count:
        dominant_patch_indices = torch.topk(
            cls_attention,
            k=dominant_patch_count,
            dim=1,
        ).indices + 1
    else:
        dominant_patch_indices = torch.empty(
            batch_size,
            0,
            dtype=torch.long,
            device=hidden_states.device,
        )
    cls_indices = torch.zeros(
        batch_size,
        1,
        dtype=torch.long,
        device=hidden_states.device,
    )
    dominant_indices = torch.cat((cls_indices, dominant_patch_indices), dim=1)
    # The reference selects with a boolean mask, hence dominant tokens retain
    # their original spatial order rather than Top-K score order.
    dominant_indices = dominant_indices.sort(dim=1).values
    dominant_mask = torch.zeros(
        batch_size,
        sequence_length,
        dtype=torch.bool,
        device=hidden_states.device,
    )
    dominant_mask.scatter_(1, dominant_indices, True)
    dominant = hidden_states[dominant_mask].view(
        batch_size, dominant_tokens, hidden_states.shape[-1]
    )

    if contextual_tokens == 0:
        return dominant, dominant_indices

    residual_hidden = hidden_states[~dominant_mask].view(
        batch_size, residual_count, hidden_states.shape[-1]
    )
    residual_metric = key_metric[~dominant_mask].view(
        batch_size, residual_count, key_metric.shape[-1]
    )
    residual_metric = F.normalize(residual_metric.float(), p=2, dim=-1)

    step = max(1, residual_count // contextual_tokens)
    target_offsets = torch.arange(
        0, residual_count, step, device=hidden_states.device
    )[:contextual_tokens]
    target_metric = residual_metric[:, target_offsets]
    residual_offsets = torch.arange(residual_count, device=hidden_states.device)
    merge_mask = ~torch.isin(residual_offsets, target_offsets)
    merge_metric = residual_metric[:, merge_mask]
    merge_hidden = residual_hidden[:, merge_mask]

    if merge_metric.shape[1] == 0:
        contextual = residual_hidden[:, target_offsets]
    else:
        similarity = merge_metric @ target_metric.transpose(1, 2)
        assignment = F.one_hot(
            similarity.argmax(dim=-1), num_classes=contextual_tokens
        ).to(dtype=hidden_states.dtype)
        counts = assignment.sum(dim=1).clamp(min=1).unsqueeze(-1)
        aggregated = assignment.transpose(1, 2) @ merge_hidden
        contextual = residual_hidden[:, target_offsets] + aggregated / counts

    return torch.cat((dominant, contextual), dim=1), dominant_indices


def specflow_heat_flow(
    attention: torch.Tensor,
    features: torch.Tensor,
    use_best_heads: bool = True,
    n_best: int = 16,
    k_neighbors: int = 25,
    diffusion_steps: int = 2,
    alpha: float = 0.7,
) -> torch.Tensor:
    """Diffuse SpecFlow CLS-attention seeds over a symmetric kNN graph."""
    if attention.dim() == 3:
        attention = attention.unsqueeze(2)
    if attention.dim() != 4:
        raise ValueError("SpecFlow attention must be [B, H, 1, N] or [B, H, N]")
    batch_size, num_heads, _, num_tokens = attention.shape
    if features.shape[:2] != (batch_size, num_tokens):
        raise ValueError("SpecFlow attention and feature token counts differ")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("SpecFlow alpha must be in [0, 1]")

    energy_maps = []
    for batch_idx in range(batch_size):
        cls_attn = attention[batch_idx, :, 0, :].float()
        if use_best_heads:
            entropy = -(cls_attn * torch.log(cls_attn + 1e-8)).sum(dim=-1)
            best_heads = torch.topk(
                -entropy, k=min(int(n_best), num_heads)
            ).indices
            seed = cls_attn[best_heads].mean(dim=0)
        else:
            seed = cls_attn.mean(dim=0)
        seed = seed / (seed.sum() + 1e-8)

        normalized = F.normalize(features[batch_idx].float(), p=2, dim=-1)
        similarity = normalized @ normalized.T
        k = min(max(1, int(k_neighbors)), num_tokens)
        neighbours = torch.topk(similarity, k=k, dim=-1).indices
        adjacency = torch.zeros(
            num_tokens,
            num_tokens,
            dtype=torch.bool,
            device=features.device,
        )
        adjacency.scatter_(1, neighbours, True)
        adjacency = adjacency | adjacency.T
        transition = F.softmax(
            similarity.masked_fill(~adjacency, float("-inf")), dim=-1
        )
        transition = torch.nan_to_num(transition)

        energy = seed
        for _ in range(int(diffusion_steps)):
            energy = (1.0 - float(alpha)) * seed + float(alpha) * (
                transition.T @ energy
            )
            energy = energy / (energy.sum() + 1e-8)
        energy_maps.append(energy)
    return torch.stack(energy_maps)


def _normalize_grid_segments(grid_spec, num_tokens: int):
    if grid_spec is None:
        return []
    if isinstance(grid_spec, tuple) and len(grid_spec) == 2:
        height, width = map(int, grid_spec)
        return [(0, num_tokens, height, width)] if height * width == num_tokens else []

    segments = []
    cursor = 0
    for item in grid_spec:
        if len(item) == 2:
            height, width = map(int, item)
            start, end = cursor, cursor + height * width
            cursor = end
        elif len(item) == 4:
            start, end, height, width = map(int, item)
        else:
            raise ValueError(f"Invalid SpecFlow grid segment: {item}")
        if start < 0 or end > num_tokens or start >= end:
            raise ValueError(f"Invalid SpecFlow grid segment bounds: {item}")
        if height <= 0 or width <= 0 or end - start != height * width:
            raise ValueError(f"SpecFlow grid segment length mismatch: {item}")
        segments.append((start, end, height, width))
    return segments


def _quadtree_split(energy_grid, min_size, threshold):
    height, width = energy_grid.shape
    crops = []

    def split(y0, y1, x0, x1):
        crop_height, crop_width = y1 - y0, x1 - x0
        crop = energy_grid[y0:y1, x0:x1]
        if (
            crop_height < 2 * min_size
            or crop_width < 2 * min_size
            or crop.std(unbiased=False) <= threshold
        ):
            crops.append((y0, y1, x0, x1))
            return
        middle_y, middle_x = y0 + crop_height // 2, x0 + crop_width // 2
        split(y0, middle_y, x0, middle_x)
        split(y0, middle_y, middle_x, x1)
        split(middle_y, y1, x0, middle_x)
        split(middle_y, y1, middle_x, x1)

    split(0, height, 0, width)
    return crops


def _allocate_capacity_quota(
    weights: List[float], capacities: List[int], total_quota: int
) -> List[int]:
    if not capacities:
        return []
    total_quota = min(max(0, int(total_quota)), sum(capacities))
    weights = [max(float(value), 0.0) for value in weights]
    if sum(weights) <= 0:
        weights = list(capacities)
    total_weight = sum(weights)
    if total_weight <= 0 or total_quota == 0:
        return [0] * len(capacities)

    quotas = [
        min(int(total_quota * weight / total_weight), capacity)
        for weight, capacity in zip(weights, capacities)
    ]
    remaining = total_quota - sum(quotas)
    ranking = sorted(
        range(len(weights)),
        key=lambda idx: (weights[idx], capacities[idx]),
        reverse=True,
    )
    while remaining:
        progressed = False
        for idx in ranking:
            if remaining == 0:
                break
            if quotas[idx] < capacities[idx]:
                quotas[idx] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return quotas


def _top_tokens_in_crop(energy_grid, crop, quota):
    y0, y1, x0, x1 = crop
    crop_energy = energy_grid[y0:y1, x0:x1]
    if quota >= crop_energy.numel():
        local_indices = torch.arange(crop_energy.numel(), device=energy_grid.device)
    else:
        local_indices = torch.topk(crop_energy.flatten(), quota).indices
    width = energy_grid.shape[1]
    crop_width = x1 - x0
    return [
        int((y0 + index // crop_width) * width + x0 + index % crop_width)
        for index in local_indices
    ]


def _select_from_grid_segments(
    energy, segments, total_quota, min_crop_size, adaptive_threshold
):
    region_info = []
    weights = []
    capacities = []
    for start, end, height, width in segments:
        energy_grid = energy[start:end].view(height, width)
        for crop in _quadtree_split(
            energy_grid, int(min_crop_size), float(adaptive_threshold)
        ):
            y0, y1, x0, x1 = crop
            crop_energy = torch.nan_to_num(
                energy_grid[y0:y1, x0:x1].float(),
                nan=0.0,
                posinf=1e6,
                neginf=0.0,
            )
            region_info.append((start, energy_grid, crop))
            weights.append(max(float(crop_energy.sum()), 0.0))
            capacities.append(crop_energy.numel())
    quotas = _allocate_capacity_quota(weights, capacities, total_quota)
    selected = []
    for (offset, energy_grid, crop), quota in zip(region_info, quotas):
        if quota:
            selected.extend(
                offset + idx for idx in _top_tokens_in_crop(energy_grid, crop, quota)
            )
    return torch.tensor(selected, device=energy.device, dtype=torch.long)


def specflow_condense(
    image_tokens: torch.Tensor,
    energy: torch.Tensor,
    token_budget: int = 192,
    min_crop_size: int = 4,
    adaptive_threshold: float = 1e-4,
    use_coreset_sinks: bool = True,
    n_sinks: int = 2,
    grid_shape=None,
) -> torch.Tensor:
    """Apply SpecFlow quadtree selection and conservative sink condensation."""
    batch_size, num_tokens, hidden_size = image_tokens.shape
    token_budget = int(token_budget)
    n_sinks = max(0, int(n_sinks)) if use_coreset_sinks else 0
    if token_budget < 1 or token_budget > num_tokens:
        raise ValueError(
            f"SpecFlow token budget must be in [1, {num_tokens}], got {token_budget}"
        )
    selection_budget = max(0, token_budget - n_sinks)
    selection_budget = min(selection_budget, num_tokens)
    energy = torch.nan_to_num(energy.float(), nan=0.0, posinf=1e6, neginf=-1e6)

    if grid_shape is None:
        side = int(num_tokens**0.5)
        grid_shapes = [(side, side)] * batch_size
    elif isinstance(grid_shape, tuple):
        grid_shapes = [grid_shape] * batch_size
    else:
        if len(grid_shape) != batch_size:
            raise ValueError("SpecFlow grid metadata batch size mismatch")
        grid_shapes = grid_shape

    outputs = []
    for batch_idx in range(batch_size):
        segments = _normalize_grid_segments(grid_shapes[batch_idx], num_tokens)
        if segments:
            selected = _select_from_grid_segments(
                energy[batch_idx],
                segments,
                selection_budget,
                min_crop_size,
                adaptive_threshold,
            )
        elif selection_budget:
            selected = torch.topk(energy[batch_idx], selection_budget).indices.sort().values
        else:
            selected = torch.empty(0, dtype=torch.long, device=image_tokens.device)

        if selected.numel() < selection_budget:
            remaining_mask = torch.ones(
                num_tokens, dtype=torch.bool, device=image_tokens.device
            )
            remaining_mask[selected] = False
            remaining = remaining_mask.nonzero(as_tuple=False).flatten()
            needed = selection_budget - selected.numel()
            additions = remaining[
                torch.topk(energy[batch_idx, remaining], min(needed, remaining.numel())).indices
            ]
            selected = torch.cat((selected, additions))
        elif selected.numel() > selection_budget:
            selected = selected[
                torch.topk(energy[batch_idx, selected], selection_budget).indices
            ]

        selected_tokens = image_tokens[batch_idx, selected]
        if n_sinks:
            dropped_mask = torch.ones(
                num_tokens, dtype=torch.bool, device=image_tokens.device
            )
            dropped_mask[selected] = False
            dropped = image_tokens[batch_idx, dropped_mask]
            if dropped.numel():
                mean_sink = dropped.mean(dim=0, keepdim=True)
                sinks = [mean_sink]
                if n_sinks >= 2 and dropped.shape[0] > 1:
                    farthest = torch.norm(dropped - mean_sink, dim=-1).argmax()
                    sinks.append(dropped[farthest : farthest + 1])
                if n_sinks >= 3 and dropped.shape[0] > 2:
                    dropped_energy = energy[batch_idx, dropped_mask]
                    weights = F.softmax(dropped_energy, dim=0)
                    sinks.append((dropped * weights.unsqueeze(-1)).sum(dim=0, keepdim=True))
                selected_tokens = torch.cat((selected_tokens, *sinks[:n_sinks]), dim=0)

        if selected_tokens.shape[0] < token_budget:
            padding = selected_tokens.new_zeros(
                token_budget - selected_tokens.shape[0], hidden_size
            )
            selected_tokens = torch.cat((selected_tokens, padding), dim=0)
        outputs.append(selected_tokens[:token_budget])
    return torch.stack(outputs).to(image_tokens.dtype)

