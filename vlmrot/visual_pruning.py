import logging
import math
import re
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vlmrot import baseline_pruning, model_utils, quant_utils


VISUAL_PRUNE_STRATEGY_ACTIVATION_SIMILARITY = "activation_similarity"
VISUAL_PRUNE_STRATEGY_VISPRUNER = "vispruner"
VISUAL_PRUNE_STRATEGY_CRISP = "crisp"
VISUAL_PRUNE_STRATEGY_P4Q = "p4q"
VISUAL_PRUNE_STRATEGY_VISIONZIP = "visionzip"
VISUAL_PRUNE_STRATEGY_SPECFLOW = "specflow"
VISUAL_PRUNE_STRATEGIES = (
    VISUAL_PRUNE_STRATEGY_ACTIVATION_SIMILARITY,
    VISUAL_PRUNE_STRATEGY_VISPRUNER,
    VISUAL_PRUNE_STRATEGY_CRISP,
    VISUAL_PRUNE_STRATEGY_P4Q,
    VISUAL_PRUNE_STRATEGY_VISIONZIP,
    VISUAL_PRUNE_STRATEGY_SPECFLOW,
)


_CRISP_SPACY_PIPELINES = {}
_CRISP_HAN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CRISP_RESPONSE_INSTRUCTION_PATTERNS = (
    re.compile(r"(?is)\n?When the provided information is insufficient.*?$"),
    re.compile(r"(?is)\n?Answer the question using.*?$"),
    re.compile(r"(?is)\n?Please answer (?:the question|directly).*?$"),
    re.compile(r"(?is)\n?Answer with (?:the option's letter|a single word|yes or no).*?$"),
    re.compile(r"(?is)\n?Respond with (?:only )?.*?$"),
    re.compile(r"(?is)\n?请(?:直接)?回答.*?$"),
    re.compile(r"(?is)\n?请(?:只)?(?:输出|回答)(?:选项|答案|字母).*?$"),
)


@dataclass
class VisualPruningStats:
    original_visual_tokens: int
    kept_visual_tokens: int
    original_prompt_tokens: int
    kept_prompt_tokens: int
    recycled_visual_tokens: int = 0


def is_enabled(args):
    return bool(getattr(args, "visual_prune", False))


def is_supported_model(model):
    return type(model) in (
        model_utils.LLAVA_HF,
        model_utils.LLAVA_NEXT_HF,
        model_utils.QWEN2_VL,
        getattr(model_utils, "QWEN2_5_VL", None),
    )


def strategy_from_args(args):
    strategy = str(
        getattr(
            args,
            "visual_prune_strategy",
            VISUAL_PRUNE_STRATEGY_ACTIVATION_SIMILARITY,
        )
    )
    if strategy not in VISUAL_PRUNE_STRATEGIES:
        raise ValueError(
            "visual_prune_strategy must be one of: "
            + ", ".join(VISUAL_PRUNE_STRATEGIES)
        )
    return strategy


def _format_config_value(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _safe_config_label(value):
    return re.sub(r"[^A-Za-z0-9_.]+", "_", str(value)).strip("_") or "none"


def result_suffix(args):
    if not is_enabled(args):
        return ""
    strategy = strategy_from_args(args)
    if strategy == VISUAL_PRUNE_STRATEGY_VISIONZIP:
        return "_" + "-".join(
            [
                "vprune",
                "stvisionzip",
                f"d{int(getattr(args, 'visual_prune_visionzip_dominant_tokens', 54))}",
                f"c{int(getattr(args, 'visual_prune_visionzip_contextual_tokens', 10))}",
            ]
        )

    if strategy == VISUAL_PRUNE_STRATEGY_SPECFLOW:
        budget_mode = str(
            getattr(args, "visual_prune_specflow_budget_mode", "fixed")
        ).lower()
        budget_part = (
            "br" + _format_config_value(
                float(getattr(args, "visual_prune_keep_ratio", 1.0))
            )
            if budget_mode == "ratio"
            else f"b{int(getattr(args, 'visual_prune_specflow_token_budget', 192))}"
        )
        return "_" + "-".join(
            [
                "vprune",
                "stspecflow",
                budget_part,
                f"h{int(getattr(args, 'visual_prune_specflow_num_best_heads', 16))}",
                f"k{int(getattr(args, 'visual_prune_specflow_k_neighbors', 25))}",
                f"ds{int(getattr(args, 'visual_prune_specflow_diffusion_steps', 2))}",
                f"a{_format_config_value(float(getattr(args, 'visual_prune_specflow_alpha', 0.7)))}",
                f"mc{int(getattr(args, 'visual_prune_specflow_min_crop_size', 4))}",
                f"t{_format_config_value(float(getattr(args, 'visual_prune_specflow_adaptive_threshold', 1e-4)))}",
                f"s{int(getattr(args, 'visual_prune_specflow_num_sinks', 2))}",
            ]
        )
    if strategy == VISUAL_PRUNE_STRATEGY_VISPRUNER:
        parts = [
            "vprune",
            "stvispruner",
            f"kr{_format_config_value(float(getattr(args, 'visual_prune_keep_ratio', 1.0)))}",
            f"ir{_format_config_value(float(getattr(args, 'visual_prune_important_ratio', 0.5)))}",
            f"mk{int(getattr(args, 'visual_prune_min_keep_tokens', 1))}",
        ]
        model_name = str(getattr(args, "model", "")).lower()
        if "qwen2-vl" in model_name or "qwen2vl" in model_name:
            parts.append(
                "qvl" + str(int(getattr(args, "visual_prune_qwen_vision_layer", -1)))
            )
        if bool(getattr(args, "visual_prune_vispruner_quant_vit_features", True)):
            quant_bits = getattr(args, "visual_prune_quant_bits", None)
            if quant_bits is None:
                quant_bits = getattr(args, "a_bits", 16)
            quant_groupsize = int(getattr(args, "visual_prune_quant_groupsize", -1))
            quant_group_label = "all" if quant_groupsize == -1 else str(quant_groupsize)
            quant_mode = "asym" if bool(getattr(args, "visual_prune_quant_asym", False)) else "sym"
            parts.extend(
                [
                    f"vitq{quant_bits}",
                    f"vitg{quant_group_label}",
                    quant_mode,
                    f"vitc{_format_config_value(float(getattr(args, 'visual_prune_quant_clip_ratio', 1.0)))}",
                ]
            )
        return "_" + "-".join(parts)

    if strategy == VISUAL_PRUNE_STRATEGY_CRISP:
        crisp_en_model = str(
            getattr(args, "visual_prune_crisp_en_model", "en_core_web_sm")
        )
        crisp_zh_model = str(
            getattr(args, "visual_prune_crisp_zh_model", "zh_core_web_sm")
        )
        parts = [
            "vprune",
            "stcrisp",
            f"kr{_format_config_value(float(getattr(args, 'visual_prune_keep_ratio', 1.0)))}",
            f"a{_format_config_value(float(getattr(args, 'visual_prune_crisp_alpha', 0.3)))}",
            f"la{_format_config_value(float(getattr(args, 'visual_prune_crisp_low_budget_alpha', 0.5)))}",
            f"lt{int(getattr(args, 'visual_prune_crisp_low_budget_threshold', 100))}",
            f"lang{getattr(args, 'visual_prune_crisp_language', 'auto')}",
            f"fb{int(bool(getattr(args, 'visual_prune_crisp_allow_heuristic_fallback', False)))}",
            f"mk{int(getattr(args, 'visual_prune_min_keep_tokens', 1))}",
        ]
        if crisp_en_model != "en_core_web_sm":
            parts.append("en" + _safe_config_label(crisp_en_model))
        if crisp_zh_model != "zh_core_web_sm":
            parts.append("zh" + _safe_config_label(crisp_zh_model))
        if bool(getattr(args, "visual_prune_crisp_quantize_weights", False)):
            parts.extend(
                [
                    f"vwq{int(getattr(args, 'w_bits', 16))}",
                    "wasym" if bool(getattr(args, "w_asym", False)) else "wsym",
                    "wmse" if bool(getattr(args, "w_clip", True)) else "wmax",
                ]
            )
        model_name = str(getattr(args, "model", "")).lower()
        if "qwen2-vl" in model_name or "qwen2vl" in model_name:
            parts.extend(
                [
                    "qvl"
                    + str(
                        int(getattr(args, "visual_prune_qwen_vision_layer", -1))
                    ),
                    "qpw"
                    + _format_config_value(
                        float(
                            getattr(
                                args,
                                "visual_prune_qwen_crisp_prior_weight",
                                0.25,
                            )
                        )
                    ),
                ]
            )
        return "_" + "-".join(parts)

    if strategy == VISUAL_PRUNE_STRATEGY_P4Q:
        crisp_en_model = str(
            getattr(args, "visual_prune_crisp_en_model", "en_core_web_sm")
        )
        crisp_zh_model = str(
            getattr(args, "visual_prune_crisp_zh_model", "zh_core_web_sm")
        )
        parts = [
            "vprune",
            "stp4q",
            f"kr{_format_config_value(float(getattr(args, 'visual_prune_keep_ratio', 1.0)))}",
            f"b{_format_config_value(float(getattr(args, 'visual_prune_p4q_beta', 0.5)))}",
            f"l{_format_config_value(float(getattr(args, 'visual_prune_p4q_lambda', 0.6)))}",
            f"lang{getattr(args, 'visual_prune_crisp_language', 'auto')}",
            f"fb{int(bool(getattr(args, 'visual_prune_crisp_allow_heuristic_fallback', False)))}",
            f"mk{int(getattr(args, 'visual_prune_min_keep_tokens', 1))}",
            _p4q_feature_suffix(args),
        ]
        if crisp_en_model != "en_core_web_sm":
            parts.append("en" + _safe_config_label(crisp_en_model))
        if crisp_zh_model != "zh_core_web_sm":
            parts.append("zh" + _safe_config_label(crisp_zh_model))
        return "_" + "-".join(parts)

    quant_bits = getattr(args, "visual_prune_quant_bits", None)
    if quant_bits is None:
        quant_bits = getattr(args, "a_bits", 16)
    quant_groupsize = int(getattr(args, "visual_prune_quant_groupsize", -1))
    quant_group_label = "all" if quant_groupsize == -1 else str(quant_groupsize)
    quant_mode = "asym" if bool(getattr(args, "visual_prune_quant_asym", False)) else "sym"
    parts = [
        "vprune",
        f"kr{_format_config_value(float(getattr(args, 'visual_prune_keep_ratio', 1.0)))}",
        f"q{quant_bits}",
        f"g{quant_group_label}",
        quant_mode,
        f"c{_format_config_value(float(getattr(args, 'visual_prune_quant_clip_ratio', 1.0)))}",
        f"aw{_format_config_value(float(getattr(args, 'visual_prune_score_activation_weight', 0.5)))}",
        f"sw{_format_config_value(float(getattr(args, 'visual_prune_score_similarity_weight', 0.5)))}",
        f"mk{int(getattr(args, 'visual_prune_min_keep_tokens', 1))}",
    ]
    if bool(getattr(args, "visual_prune_recycle", False)):
        recycle_mode = getattr(args, "visual_prune_recycle_mode", "knn")
        parts.extend(
            [
                f"rec{recycle_mode}",
                f"ra{_format_config_value(float(getattr(args, 'visual_prune_recycle_alpha', 0.25)))}",
            ]
        )
        if recycle_mode == "window":
            parts.append(
                f"rw{int(getattr(args, 'visual_prune_recycle_window_size', 64))}"
            )
    return "_" + "-".join(parts)


def _minmax_normalize(values):
    if values.numel() == 0:
        return values
    min_value = values.min()
    max_value = values.max()
    denom = max_value - min_value
    if denom.abs() <= torch.finfo(values.dtype).eps:
        return torch.zeros_like(values)
    return (values - min_value) / denom


def _activation_magnitude(visual_embeds):
    return visual_embeds.float().abs().mean(dim=-1)


def _visual_text_similarity(visual_embeds, text_embeds):
    if text_embeds.numel() == 0:
        return torch.zeros(visual_embeds.shape[0], device=visual_embeds.device)
    text_center = text_embeds.float().mean(dim=0, keepdim=True)
    return F.cosine_similarity(visual_embeds.float(), text_center, dim=-1)


def visual_pruning_scores(
    visual_embeds,
    text_embeds,
    activation_weight=0.5,
    similarity_weight=0.5,
):
    activation_score = _minmax_normalize(_activation_magnitude(visual_embeds))
    similarity_score = _minmax_normalize(
        _visual_text_similarity(visual_embeds, text_embeds)
    )
    total_weight = float(activation_weight) + float(similarity_weight)
    if total_weight <= 0:
        raise ValueError("visual pruning score weights must sum to a positive value")
    return (
        float(activation_weight) * activation_score
        + float(similarity_weight) * similarity_score
    ) / total_weight


def select_topk_visual_tokens(
    visual_embeds,
    text_embeds,
    keep_ratio,
    activation_weight=0.5,
    similarity_weight=0.5,
    min_keep_tokens=1,
):
    if visual_embeds.dim() != 2:
        raise ValueError(
            f"visual_embeds must be [num_visual_tokens, hidden], got {visual_embeds.shape}"
        )
    num_visual_tokens = visual_embeds.shape[0]
    if num_visual_tokens == 0:
        return torch.zeros(0, dtype=torch.bool, device=visual_embeds.device), torch.zeros(
            0, device=visual_embeds.device
        )

    keep_ratio = float(keep_ratio)
    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise ValueError("visual_prune_keep_ratio must be in (0, 1]")

    keep_count = int(math.ceil(num_visual_tokens * keep_ratio))
    keep_count = max(int(min_keep_tokens), keep_count)
    keep_count = min(num_visual_tokens, keep_count)
    scores = visual_pruning_scores(
        visual_embeds,
        text_embeds,
        activation_weight=activation_weight,
        similarity_weight=similarity_weight,
    )
    keep_indices = torch.topk(scores, k=keep_count, largest=True, sorted=False).indices
    keep_mask = torch.zeros(num_visual_tokens, dtype=torch.bool, device=visual_embeds.device)
    keep_mask[keep_indices] = True
    return keep_mask, scores


def _coerce_vispruner_attention(image_attentions, num_tokens):
    if image_attentions.dim() == 1:
        attention_scores = image_attentions.float()
    elif image_attentions.dim() == 2:
        attention_scores = image_attentions.float().mean(dim=0)
    else:
        raise ValueError(
            "image_attentions must be [num_tokens] or [num_heads, num_tokens], "
            f"got {image_attentions.shape}"
        )
    if attention_scores.shape[0] != num_tokens:
        raise ValueError(
            "attention token count does not match feature token count: "
            f"{attention_scores.shape[0]} vs {num_tokens}"
        )
    return attention_scores


def _vispruner_diverse_residual_indices(normalized_features, residual_indices, diverse_count):
    residual_indices = residual_indices.clone()
    diverse_count = int(diverse_count)
    while residual_indices.numel() > diverse_count:
        residual_count = residual_indices.numel()
        remove_count = min(8, residual_count - diverse_count)
        if remove_count <= 0:
            break

        residual_tokens = normalized_features[residual_indices]
        even_tokens = residual_tokens[::2]
        odd_tokens = residual_tokens[1::2]
        even_indices = residual_indices[::2]
        odd_indices = residual_indices[1::2]
        if even_tokens.numel() == 0 or odd_tokens.numel() == 0:
            residual_indices = residual_indices[:diverse_count]
            break

        scores = even_tokens @ odd_tokens.T
        scores = scores.max(dim=-1).values
        distinct_offsets = scores.argsort(dim=-1, descending=True)[remove_count:]
        residual_indices = torch.cat(
            [even_indices[distinct_offsets], odd_indices],
            dim=0,
        )

    return residual_indices[:diverse_count]


def _vispruner_diverse_residual_indices_batched(
    normalized_features,
    residual_indices,
    diverse_count,
):
    """Run the official VisPruner matching loop for a batch of equal-size crops.

    LLaVA-Next encodes all AnyRes crops in one tensor, and every crop contains
    the same number of CLIP patch tokens.  Keeping the crop dimension here
    preserves the original per-crop selection rule while replacing one Python
    loop per crop with batched matrix multiplications.
    """
    residual_indices = residual_indices.clone()
    diverse_count = int(diverse_count)
    batch_size = residual_indices.shape[0]
    batch_indices = torch.arange(
        batch_size,
        device=residual_indices.device,
    ).unsqueeze(-1)

    while residual_indices.shape[1] > diverse_count:
        residual_count = residual_indices.shape[1]
        remove_count = min(8, residual_count - diverse_count)
        if remove_count <= 0:
            break

        residual_tokens = normalized_features[batch_indices, residual_indices]
        even_tokens = residual_tokens[:, ::2]
        odd_tokens = residual_tokens[:, 1::2]
        even_indices = residual_indices[:, ::2]
        odd_indices = residual_indices[:, 1::2]
        if even_tokens.shape[1] == 0 or odd_tokens.shape[1] == 0:
            residual_indices = residual_indices[:, :diverse_count]
            break

        scores = even_tokens @ odd_tokens.transpose(-1, -2)
        scores = scores.max(dim=-1).values
        distinct_offsets = scores.argsort(dim=-1, descending=True)[:, remove_count:]
        remaining_even_indices = torch.gather(
            even_indices,
            dim=1,
            index=distinct_offsets,
        )
        residual_indices = torch.cat(
            [remaining_even_indices, odd_indices],
            dim=1,
        )

    return residual_indices[:, :diverse_count]


def select_vispruner_visual_tokens_batched(
    image_features,
    image_attentions,
    keep_count,
    important_ratio=0.5,
    diversity_features=None,
):
    """Select VisPruner tokens for equal-size crops in one batched operation.

    This is the batch counterpart of ``select_vispruner_visual_tokens`` for the
    LLaVA-Next crop path.  It intentionally keeps the strict VisPruner rule of
    removing at most eight residual tokens per matching round.  The generic
    single-sample function remains unchanged because other model paths may use
    an ``always_keep_mask`` or differently shaped visual groups.
    """
    if image_features.dim() != 3:
        raise ValueError(
            "image_features must be [batch, num_visual_tokens, hidden], "
            f"got {image_features.shape}"
        )
    batch_size, num_tokens, _ = image_features.shape
    if num_tokens == 0:
        return torch.zeros(
            batch_size,
            0,
            dtype=torch.bool,
            device=image_features.device,
        )

    keep_count = int(keep_count)
    if keep_count < 1:
        raise ValueError("keep_count must be >= 1")
    keep_count = min(keep_count, num_tokens)
    important_ratio = float(important_ratio)
    if important_ratio < 0.0 or important_ratio > 1.0:
        raise ValueError("visual_prune_important_ratio must be in [0, 1]")

    if image_attentions.dim() == 2:
        attention_scores = image_attentions.float()
    elif image_attentions.dim() == 3:
        attention_scores = image_attentions.float().mean(dim=1)
    else:
        raise ValueError(
            "image_attentions must be [batch, num_visual_tokens] or "
            "[batch, num_heads, num_visual_tokens], "
            f"got {image_attentions.shape}"
        )
    if attention_scores.shape != (batch_size, num_tokens):
        raise ValueError(
            "attention shape does not match batched image features: "
            f"{attention_scores.shape} vs {(batch_size, num_tokens)}"
        )

    sorted_indices = attention_scores.argsort(dim=-1, descending=True)
    important_count = int(keep_count * important_ratio)
    important_count = min(max(0, important_count), keep_count)
    diverse_count = keep_count - important_count
    important_indices = sorted_indices[:, :important_count]

    if diverse_count > 0:
        if diversity_features is None:
            diversity_features = image_features
        if (
            diversity_features.dim() != 3
            or diversity_features.shape[:2] != image_features.shape[:2]
        ):
            raise ValueError(
                "diversity_features must match the batch and token dimensions "
                f"of image_features: {diversity_features.shape} vs "
                f"{image_features.shape}"
            )
        normalized_features = F.normalize(
            diversity_features.detach().float(),
            dim=-1,
        )
        residual_indices = sorted_indices[:, important_count:]
        diverse_indices = _vispruner_diverse_residual_indices_batched(
            normalized_features,
            residual_indices,
            diverse_count,
        )
        selected_indices = torch.cat(
            [important_indices, diverse_indices],
            dim=1,
        )
    else:
        selected_indices = important_indices

    keep_mask = torch.zeros(
        batch_size,
        num_tokens,
        dtype=torch.bool,
        device=image_features.device,
    )
    keep_mask.scatter_(1, selected_indices, True)
    return keep_mask


def select_vispruner_visual_tokens(
    image_features,
    image_attentions,
    keep_count,
    important_ratio=0.5,
    always_keep_mask=None,
    diversity_features=None,
):
    if image_features.dim() != 2:
        raise ValueError(
            f"image_features must be [num_visual_tokens, hidden], got {image_features.shape}"
        )
    num_tokens = image_features.shape[0]
    if num_tokens == 0:
        return torch.zeros(0, dtype=torch.bool, device=image_features.device)

    keep_count = int(keep_count)
    if keep_count < 1:
        raise ValueError("keep_count must be >= 1")
    keep_count = min(keep_count, num_tokens)
    important_ratio = float(important_ratio)
    if important_ratio < 0.0 or important_ratio > 1.0:
        raise ValueError("visual_prune_important_ratio must be in [0, 1]")

    if always_keep_mask is None:
        always_keep_mask = torch.zeros(num_tokens, dtype=torch.bool, device=image_features.device)
    else:
        always_keep_mask = always_keep_mask.to(device=image_features.device, dtype=torch.bool)
        if always_keep_mask.shape[0] != num_tokens:
            raise ValueError(
                "always_keep_mask length does not match image feature count: "
                f"{always_keep_mask.shape[0]} vs {num_tokens}"
            )

    keep_mask = always_keep_mask.clone()
    budget = keep_count - int(always_keep_mask.sum().item())
    if budget <= 0:
        return keep_mask

    attention_scores = _coerce_vispruner_attention(image_attentions, num_tokens)
    candidate_indices = torch.nonzero(~always_keep_mask, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return keep_mask

    budget = min(budget, candidate_indices.numel())
    sorted_offsets = attention_scores[candidate_indices].argsort(dim=-1, descending=True)
    sorted_indices = candidate_indices[sorted_offsets]
    important_count = int(keep_count * important_ratio)
    important_count = min(max(0, important_count), budget)
    diverse_count = budget - important_count

    important_indices = sorted_indices[:important_count]
    residual_indices = sorted_indices[important_count:]
    if diverse_count > 0:
        if diversity_features is None:
            diversity_features = image_features
        if (
            diversity_features.dim() != 2
            or diversity_features.shape[0] != image_features.shape[0]
        ):
            raise ValueError(
                "diversity_features must have the same token count as image_features: "
                f"{diversity_features.shape} vs {image_features.shape}"
            )
        normalized_features = F.normalize(diversity_features.detach().float(), dim=-1)
        diverse_indices = _vispruner_diverse_residual_indices(
            normalized_features,
            residual_indices,
            diverse_count,
        )
        selected_indices = torch.cat([important_indices, diverse_indices], dim=0)
    else:
        selected_indices = important_indices

    keep_mask[selected_indices] = True
    return keep_mask


def _strip_crisp_response_instructions(query_text):
    """Remove answer-format suffixes before CRISP noun extraction.

    CRISP removes response instructions because words such as ``answer`` and
    ``option`` describe the output format rather than the image content.  The
    benchmark prompt builders append several English and Chinese variants, so
    the patterns intentionally cover both without modifying the actual model
    prompt.
    """
    cleaned = str(query_text or "").strip()
    for pattern in _CRISP_RESPONSE_INSTRUCTION_PATTERNS:
        cleaned = pattern.sub("", cleaned).strip()
    return cleaned


def _detect_crisp_language(query_text, configured_language="auto"):
    language = str(configured_language).lower()
    if language not in ("auto", "en", "zh"):
        raise ValueError("visual_prune_crisp_language must be one of: auto, en, zh")
    if language != "auto":
        return language
    return "zh" if _CRISP_HAN_PATTERN.search(str(query_text or "")) else "en"


def _load_crisp_spacy_pipeline(language, args):
    model_name = (
        str(getattr(args, "visual_prune_crisp_zh_model", "zh_core_web_sm"))
        if language == "zh"
        else str(getattr(args, "visual_prune_crisp_en_model", "en_core_web_sm"))
    )
    cache_key = (language, model_name)
    if cache_key in _CRISP_SPACY_PIPELINES:
        return _CRISP_SPACY_PIPELINES[cache_key]
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "CRISP requires spaCy. Install the optional dependency with "
            "`pip install -r requirements-crisp.txt`, then install the requested "
            f"language model with `python -m spacy download {model_name}`."
        ) from exc
    try:
        pipeline = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"CRISP could not load spaCy model {model_name!r}. Install it with "
            f"`python -m spacy download {model_name}`. For Chinese evaluation, "
            "use zh_core_web_sm rather than en_core_web_sm."
        ) from exc
    _CRISP_SPACY_PIPELINES[cache_key] = pipeline
    return pipeline


def _crisp_heuristic_keywords(query_text, language):
    """Explicit opt-in fallback when a spaCy model is unavailable.

    This is a portability extension, not the paper-exact POS path.  It is kept
    disabled by default and deliberately conservative.
    """
    if language == "zh":
        return re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{1,8}", query_text)
    stopwords = {
        "a", "an", "and", "are", "be", "do", "does", "in", "is", "of",
        "on", "or", "the", "this", "to", "what", "which", "who", "with",
    }
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", query_text)
    return [word for word in words if word.lower() not in stopwords]


def extract_crisp_keywords(query_text, args):
    cleaned_query = _strip_crisp_response_instructions(query_text)
    language = _detect_crisp_language(
        cleaned_query,
        getattr(args, "visual_prune_crisp_language", "auto"),
    )
    try:
        pipeline = _load_crisp_spacy_pipeline(language, args)
    except (ImportError, RuntimeError):
        if not bool(
            getattr(args, "visual_prune_crisp_allow_heuristic_fallback", False)
        ):
            raise
        logging.warning(
            "CRISP is using the opt-in heuristic %s keyword fallback; results "
            "are not paper-exact because spaCy POS tagging is unavailable.",
            language,
        )
        return _crisp_heuristic_keywords(cleaned_query, language), language

    document = pipeline(cleaned_query)
    keywords = [
        token.text.strip()
        for token in document
        if token.pos_ in ("NOUN", "PROPN") and token.text.strip()
    ]
    return keywords, language


def embed_crisp_keywords(model, tokenizer, keywords, device, batched=False, verify_exact=False):
    if batched:
        from vlmrot.p4q_execution import embed_keywords
        result = embed_keywords(model, tokenizer, keywords, device)
        if verify_exact:
            expected = embed_crisp_keywords(model, tokenizer, keywords, device)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        return result
    if not keywords:
        language_model = _get_language_model(model)
        hidden_size = language_model.embed_tokens.weight.shape[-1]
        return torch.empty(0, hidden_size, dtype=torch.float32, device=device)
    if tokenizer is None:
        raise ValueError("CRISP requires the model processor tokenizer")

    language_model = _get_language_model(model)
    embed_tokens = language_model.embed_tokens
    keyword_embeddings = []
    with torch.no_grad():
        for keyword in keywords:
            tokenized = tokenizer(
                keyword,
                add_special_tokens=False,
                return_tensors="pt",
            )
            token_ids = tokenized["input_ids"]
            if token_ids.numel() == 0:
                continue
            embedding_device = embed_tokens.weight.device
            embedded = embed_tokens(token_ids.to(embedding_device))
            keyword_embeddings.append(
                embedded.float().mean(dim=1).squeeze(0).to(device)
            )
    if not keyword_embeddings:
        hidden_size = embed_tokens.weight.shape[-1]
        return torch.empty(0, hidden_size, dtype=torch.float32, device=device)
    return torch.stack(keyword_embeddings, dim=0)


def _crisp_critical_count(keep_count, alpha, low_budget_alpha, low_budget_threshold):
    effective_alpha = (
        float(low_budget_alpha)
        if int(keep_count) < int(low_budget_threshold)
        else float(alpha)
    )
    # The paper's floor rule is preserved.  The lower bound is a necessary
    # guard for tiny synthetic budgets where floor(alpha * K) would be zero.
    return max(1, min(int(keep_count), int(math.floor(effective_alpha * keep_count))))


def select_crisp_visual_tokens(
    projected_features,
    image_attentions,
    keyword_embeddings,
    keep_count,
    alpha=0.3,
    low_budget_alpha=0.5,
    low_budget_threshold=100,
):
    """Paper-faithful two-stage CRISP selection in the LLM embedding space.

    Stage 1 distributes a critical-token budget across noun embeddings using
    CLS attention mass.  Stage 2 greedily adds the token least similar to the
    current selected set, which preserves complementary visual information.
    """
    if projected_features.dim() != 2:
        raise ValueError(
            "projected_features must be [num_visual_tokens, hidden], got "
            f"{projected_features.shape}"
        )
    num_tokens, hidden_size = projected_features.shape
    if num_tokens == 0:
        return torch.zeros(0, dtype=torch.bool, device=projected_features.device)
    keep_count = min(int(keep_count), num_tokens)
    if keep_count < 1:
        raise ValueError("keep_count must be >= 1")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("visual_prune_crisp_alpha must be in (0, 1]")
    if not 0.0 < float(low_budget_alpha) <= 1.0:
        raise ValueError("visual_prune_crisp_low_budget_alpha must be in (0, 1]")
    if int(low_budget_threshold) < 1:
        raise ValueError("visual_prune_crisp_low_budget_threshold must be >= 1")

    attention_scores = _coerce_vispruner_attention(image_attentions, num_tokens)
    attention_scores = attention_scores.to(projected_features.device)
    critical_count = _crisp_critical_count(
        keep_count,
        alpha,
        low_budget_alpha,
        low_budget_threshold,
    )
    normalized_features = F.normalize(
        projected_features.detach().float(), dim=-1
    )
    selected_mask = torch.zeros(
        num_tokens, dtype=torch.bool, device=projected_features.device
    )

    if keyword_embeddings is not None and keyword_embeddings.numel() > 0:
        if keyword_embeddings.dim() != 2 or keyword_embeddings.shape[-1] != hidden_size:
            raise ValueError(
                "keyword_embeddings must be [num_keywords, hidden] with the "
                f"same hidden size as visual features: {keyword_embeddings.shape}"
            )
        normalized_keywords = F.normalize(
            keyword_embeddings.detach().float().to(projected_features.device),
            dim=-1,
        )
        relevance = normalized_features @ normalized_keywords.T
        per_word_candidates = critical_count // normalized_keywords.shape[0]
        if per_word_candidates > 0:
            top_offsets = torch.topk(
                relevance,
                k=per_word_candidates,
                dim=0,
                largest=True,
                sorted=False,
            ).indices
            grounding_strength = attention_scores[top_offsets].sum(dim=0)
            strength_sum = grounding_strength.sum()
            if strength_sum.abs() > torch.finfo(torch.float32).eps:
                quotas = torch.floor(
                    grounding_strength / strength_sum * critical_count
                ).to(torch.long)
            else:
                quotas = torch.full_like(
                    grounding_strength,
                    critical_count // normalized_keywords.shape[0],
                    dtype=torch.long,
                )
            for word_index, quota in enumerate(quotas.tolist()):
                if quota <= 0:
                    continue
                selected_indices = torch.topk(
                    relevance[:, word_index],
                    k=min(quota, num_tokens),
                    largest=True,
                    sorted=False,
                ).indices
                selected_mask[selected_indices] = True

    # CRISP's noun-free fallback is top vision-attention.  The same fallback
    # also makes the exact floor/union allocation robust when it selects none.
    if not selected_mask.any():
        fallback_indices = torch.topk(
            attention_scores,
            k=critical_count,
            largest=True,
            sorted=False,
        ).indices
        selected_mask[fallback_indices] = True

    # The union in stage 1 may contain fewer (duplicates/floored quotas) or
    # more than Kcritical tokens.  The paper keeps the union and stage 2 fills
    # from its actual size; only a pathological >K budget needs deterministic
    # clipping.
    if int(selected_mask.sum()) > keep_count:
        selected_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
        kept_offsets = torch.topk(
            attention_scores[selected_indices],
            k=keep_count,
            largest=True,
            sorted=False,
        ).indices
        selected_mask.zero_()
        selected_mask[selected_indices[kept_offsets]] = True

    selected_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
    max_similarity = (
        normalized_features @ normalized_features[selected_indices].T
    ).max(dim=-1).values
    max_similarity[selected_mask] = torch.inf
    while int(selected_mask.sum()) < keep_count:
        next_index = torch.argmin(max_similarity)
        selected_mask[next_index] = True
        new_similarity = normalized_features @ normalized_features[next_index]
        max_similarity = torch.maximum(max_similarity, new_similarity)
        max_similarity[selected_mask] = torch.inf
    return selected_mask


def _rank_normalize_visual_scores(values):
    """Map each score column to [0, 1] percentile ranks over visual tokens."""
    squeeze_last = values.dim() == 1
    if squeeze_last:
        values = values.unsqueeze(-1)
    if values.dim() != 2:
        raise ValueError(f"visual scores must have 1 or 2 dimensions, got {values.shape}")
    num_tokens = values.shape[0]
    if num_tokens <= 1:
        ranks = torch.zeros_like(values, dtype=torch.float32)
    else:
        order = torch.argsort(values.float(), dim=0)
        rank_values = torch.arange(
            num_tokens,
            dtype=torch.float32,
            device=values.device,
        ).unsqueeze(-1)
        rank_values = rank_values.expand_as(order) / float(num_tokens - 1)
        ranks = torch.empty_like(values, dtype=torch.float32)
        ranks.scatter_(0, order, rank_values)
    return ranks.squeeze(-1) if squeeze_last else ranks


def _qwen_crisp_visual_prior(attention_scores):
    """Return rank prior and confidence from normalized-attention entropy."""
    attention_scores = attention_scores.detach().float().clamp_min(0.0)
    prior_ranks = _rank_normalize_visual_scores(attention_scores)
    num_tokens = attention_scores.numel()
    total = attention_scores.sum()
    eps = torch.finfo(torch.float32).eps
    if num_tokens <= 1 or float(total) <= eps:
        return prior_ranks, attention_scores.new_tensor(0.0)
    probabilities = attention_scores / total
    entropy = -(
        probabilities * probabilities.clamp_min(eps).log()
    ).sum() / math.log(num_tokens)
    confidence = (1.0 - entropy).clamp(0.0, 1.0)
    return prior_ranks, confidence


def select_qwen_crisp_visual_tokens(
    projected_features,
    image_attentions,
    keyword_embeddings,
    keep_count,
    alpha=0.3,
    low_budget_alpha=0.5,
    low_budget_threshold=100,
    visual_prior_weight=0.25,
):
    """Qwen2-VL CRISP adapter with text relevance as the primary signal.

    Qwen2-VL has no CLS token, so mean incoming visual attention is not a
    semantic grounding score. Stage 1 therefore ranks tokens primarily by
    noun--visual cosine relevance. Incoming-attention centrality contributes
    only a bounded prior, and normalized entropy suppresses that prior when
    attention is diffuse. Stage 2 retains CRISP's max-min semantic diversity.
    """
    if projected_features.dim() != 2:
        raise ValueError(
            "projected_features must be [num_visual_tokens, hidden], got "
            f"{projected_features.shape}"
        )
    num_tokens, hidden_size = projected_features.shape
    if num_tokens == 0:
        return torch.zeros(0, dtype=torch.bool, device=projected_features.device)
    keep_count = min(int(keep_count), num_tokens)
    if keep_count < 1:
        raise ValueError("keep_count must be >= 1")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("visual_prune_crisp_alpha must be in (0, 1]")
    if not 0.0 < float(low_budget_alpha) <= 1.0:
        raise ValueError("visual_prune_crisp_low_budget_alpha must be in (0, 1]")
    if int(low_budget_threshold) < 1:
        raise ValueError("visual_prune_crisp_low_budget_threshold must be >= 1")
    if not 0.0 <= float(visual_prior_weight) <= 1.0:
        raise ValueError("visual_prune_qwen_crisp_prior_weight must be in [0, 1]")

    attention_scores = _coerce_vispruner_attention(
        image_attentions, num_tokens
    ).to(projected_features.device)
    prior_ranks, prior_confidence = _qwen_crisp_visual_prior(attention_scores)
    effective_prior_weight = float(visual_prior_weight) * prior_confidence
    critical_count = _crisp_critical_count(
        keep_count,
        alpha,
        low_budget_alpha,
        low_budget_threshold,
    )
    normalized_features = F.normalize(
        projected_features.detach().float(), dim=-1
    )
    selected_mask = torch.zeros(
        num_tokens, dtype=torch.bool, device=projected_features.device
    )

    if keyword_embeddings is not None and keyword_embeddings.numel() > 0:
        if keyword_embeddings.dim() != 2 or keyword_embeddings.shape[-1] != hidden_size:
            raise ValueError(
                "keyword_embeddings must be [num_keywords, hidden] with the "
                f"same hidden size as visual features: {keyword_embeddings.shape}"
            )
        normalized_keywords = F.normalize(
            keyword_embeddings.detach().float().to(projected_features.device),
            dim=-1,
        )
        relevance = normalized_features @ normalized_keywords.T
        relevance_ranks = _rank_normalize_visual_scores(relevance)
        joint_scores = relevance_ranks + (
            effective_prior_weight * prior_ranks.unsqueeze(-1)
        )

        # Give every active noun a balanced quota. This avoids reinterpreting
        # non-CLS attention mass as keyword grounding strength, while resolving
        # duplicate token choices by taking the next best unselected token.
        keyword_priority = torch.argsort(
            relevance.max(dim=0).values,
            descending=True,
        )
        active_keyword_count = min(critical_count, normalized_keywords.shape[0])
        active_keywords = keyword_priority[:active_keyword_count]
        base_quota, extra_quota = divmod(critical_count, active_keyword_count)
        for priority_offset, keyword_index in enumerate(active_keywords.tolist()):
            quota = base_quota + int(priority_offset < extra_quota)
            candidates = torch.argsort(
                joint_scores[:, keyword_index],
                descending=True,
            )
            chosen = 0
            for token_index in candidates.tolist():
                if selected_mask[token_index]:
                    continue
                selected_mask[token_index] = True
                chosen += 1
                if chosen == quota:
                    break

    # When no nouns are available, visual centrality is the only model-native
    # cue. This is deliberately a fallback, not the dominant grounded score.
    if not selected_mask.any():
        fallback_indices = torch.topk(
            attention_scores,
            k=critical_count,
            largest=True,
            sorted=False,
        ).indices
        selected_mask[fallback_indices] = True

    selected_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
    max_similarity = (
        normalized_features @ normalized_features[selected_indices].T
    ).max(dim=-1).values
    max_similarity[selected_mask] = torch.inf
    while int(selected_mask.sum()) < keep_count:
        next_index = torch.argmin(max_similarity)
        selected_mask[next_index] = True
        new_similarity = normalized_features @ normalized_features[next_index]
        max_similarity = torch.maximum(max_similarity, new_similarity)
        max_similarity[selected_mask] = torch.inf
    return selected_mask


def select_crisp_visual_tokens_batched(
    projected_features,
    image_attentions,
    keyword_embeddings,
    keep_count,
    alpha=0.3,
    low_budget_alpha=0.5,
    low_budget_threshold=100,
):
    if projected_features.dim() != 3:
        raise ValueError(
            "projected_features must be [batch, num_visual_tokens, hidden], got "
            f"{projected_features.shape}"
        )
    if image_attentions.dim() not in (2, 3):
        raise ValueError("batched CRISP attentions must have 2 or 3 dimensions")
    masks = []
    for batch_index in range(projected_features.shape[0]):
        masks.append(
            select_crisp_visual_tokens(
                projected_features[batch_index],
                image_attentions[batch_index],
                keyword_embeddings,
                keep_count,
                alpha=alpha,
                low_budget_alpha=low_budget_alpha,
                low_budget_threshold=low_budget_threshold,
            )
        )
    return torch.stack(masks, dim=0)


def _p4q_budget_counts(keep_count, beta=0.5, text_lambda=0.6):
    """Return the exact sequential P4Q text/visual/diversity budgets."""
    keep_count = int(keep_count)
    if keep_count < 1:
        raise ValueError("keep_count must be >= 1")
    if not 0.0 < float(beta) <= 1.0:
        raise ValueError("visual_prune_p4q_beta must be in (0, 1]")
    if not 0.0 <= float(text_lambda) <= 1.0:
        raise ValueError("visual_prune_p4q_lambda must be in [0, 1]")

    anchor_count = max(1, min(keep_count, int(math.floor(float(beta) * keep_count))))
    text_count = int(math.floor(float(text_lambda) * anchor_count))
    visual_count = anchor_count - text_count
    diversity_count = keep_count - anchor_count
    return text_count, visual_count, diversity_count


def _select_p4q_crisp_text_anchors(
    normalized_features,
    attention_scores,
    normalized_keywords,
    text_count,
):
    """Run CRISP Stage 1 inside P4Q's fixed text-anchor budget."""
    num_tokens = normalized_features.shape[0]
    selected_mask = torch.zeros(
        num_tokens,
        dtype=torch.bool,
        device=normalized_features.device,
    )
    text_count = min(int(text_count), num_tokens)
    if text_count < 1 or normalized_keywords.numel() == 0:
        return selected_mask

    relevance = normalized_features @ normalized_keywords.T
    per_word_candidates = text_count // normalized_keywords.shape[0]
    if per_word_candidates < 1:
        return selected_mask

    top_offsets = torch.topk(
        relevance,
        k=per_word_candidates,
        dim=0,
        largest=True,
        sorted=False,
    ).indices
    grounding_strength = attention_scores[top_offsets].sum(dim=0)
    strength_sum = grounding_strength.sum()
    if strength_sum.abs() > torch.finfo(torch.float32).eps:
        quotas = torch.floor(
            grounding_strength / strength_sum * text_count
        ).to(torch.long)
    else:
        quotas = torch.full_like(
            grounding_strength,
            text_count // normalized_keywords.shape[0],
            dtype=torch.long,
        )

    for word_index, quota in enumerate(quotas.tolist()):
        if quota <= 0:
            continue
        selected_indices = torch.topk(
            relevance[:, word_index],
            k=min(quota, num_tokens),
            largest=True,
            sorted=False,
        ).indices
        selected_mask[selected_indices] = True
    return selected_mask


def select_p4q_visual_tokens(
    selection_features,
    image_attentions,
    keyword_embeddings,
    keep_count,
    beta=0.5,
    text_lambda=0.6,
):
    """Select P4Q tokens with disjoint text, visual, and diversity budgets.

    ``selection_features`` follow the configured P4Q feature mode. The caller
    always gathers from projector outputs, not the feature-quantized copy.
    """
    if selection_features.dim() != 2:
        raise ValueError(
            "selection_features must be [num_visual_tokens, hidden], got "
            f"{selection_features.shape}"
        )
    num_tokens, hidden_size = selection_features.shape
    if num_tokens == 0:
        return torch.zeros(0, dtype=torch.bool, device=selection_features.device)
    keep_count = min(int(keep_count), num_tokens)
    text_count, visual_count, diversity_count = _p4q_budget_counts(
        keep_count,
        beta=beta,
        text_lambda=text_lambda,
    )

    normalized_features = F.normalize(
        selection_features.detach().float(), dim=-1
    )
    attention_scores = _coerce_vispruner_attention(
        image_attentions, num_tokens
    ).to(selection_features.device)
    selected_mask = torch.zeros(
        num_tokens, dtype=torch.bool, device=selection_features.device
    )

    selected_text_count = 0
    if (
        text_count > 0
        and keyword_embeddings is not None
        and keyword_embeddings.numel() > 0
    ):
        if keyword_embeddings.dim() != 2 or keyword_embeddings.shape[-1] != hidden_size:
            raise ValueError(
                "keyword_embeddings must be [num_keywords, hidden] with the "
                f"same hidden size as P4Q features: {keyword_embeddings.shape}"
            )
        normalized_keywords = F.normalize(
            keyword_embeddings.detach().float().to(selection_features.device),
            dim=-1,
        )
        selected_mask |= _select_p4q_crisp_text_anchors(
            normalized_features,
            attention_scores,
            normalized_keywords,
            text_count,
        )
        selected_text_count = int(selected_mask.sum())

    # CRISP's floored per-noun quotas and union can leave part of the planned
    # text budget unused. Transfer any shortfall to visual saliency so P4Q's
    # anchor budget and final total remain exact.
    visual_budget = visual_count + (text_count - selected_text_count)
    if visual_budget > 0:
        remaining_indices = torch.nonzero(~selected_mask, as_tuple=False).flatten()
        visual_budget = min(visual_budget, remaining_indices.numel())
        visual_offsets = torch.topk(
            attention_scores[remaining_indices],
            k=visual_budget,
            largest=True,
            sorted=False,
        ).indices
        selected_mask[remaining_indices[visual_offsets]] = True

    # Sequential max-min coverage uses only candidates outside both anchor sets.
    # At least one anchor is guaranteed because beta > 0 and keep_count >= 1.
    selected_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
    max_similarity = (
        normalized_features @ normalized_features[selected_indices].T
    ).max(dim=-1).values
    max_similarity[selected_mask] = torch.inf
    # The anchor stage above always fills exactly text_count+visual_count
    # distinct positions.  Use the already-known diversity budget instead of
    # synchronizing selected_mask.sum() from CUDA on every iteration.
    for _ in range(diversity_count):
        next_index = torch.argmin(max_similarity)
        selected_mask[next_index] = True
        new_similarity = normalized_features @ normalized_features[next_index]
        max_similarity = torch.maximum(max_similarity, new_similarity)
        max_similarity[selected_mask] = torch.inf
    return selected_mask


def select_p4q_visual_tokens_batched(
    selection_features,
    image_attentions,
    keyword_embeddings,
    keep_count,
    beta=0.5,
    text_lambda=0.6,
    batch_crops=False,
    gpu_budget=False,
    verify_exact=False,
):
    if selection_features.dim() != 3:
        raise ValueError(
            "selection_features must be [batch, num_visual_tokens, hidden], got "
            f"{selection_features.shape}"
        )
    if image_attentions.dim() not in (2, 3):
        raise ValueError("batched P4Q attentions must have 2 or 3 dimensions")
    if batch_crops or gpu_budget:
        from vlmrot.p4q_execution import select_crops
        result = select_crops(selection_features, image_attentions, keyword_embeddings,
                              keep_count, beta, text_lambda, batch_crops, gpu_budget)
        if verify_exact:
            expected = select_p4q_visual_tokens_batched(
                selection_features, image_attentions, keyword_embeddings,
                keep_count, beta, text_lambda)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        return result
    masks = [
        select_p4q_visual_tokens(
            selection_features[batch_index],
            image_attentions[batch_index],
            keyword_embeddings,
            keep_count,
            beta=beta,
            text_lambda=text_lambda,
        )
        for batch_index in range(selection_features.shape[0])
    ]
    return torch.stack(masks, dim=0)


def recycle_dropped_visual_tokens_knn(image_features, keep_mask, alpha=0.25):
    if alpha <= 0.0 or keep_mask.all() or (~keep_mask).sum() == 0:
        return image_features
    if keep_mask.sum() == 0:
        raise ValueError("Cannot recycle dropped visual tokens with no kept tokens")

    kept = image_features[keep_mask]
    dropped = image_features[~keep_mask]
    kept_norm = F.normalize(kept.float(), dim=-1)
    dropped_norm = F.normalize(dropped.float(), dim=-1)
    owner = (dropped_norm @ kept_norm.T).argmax(dim=-1)

    recycled = kept.float().clone()
    counts = torch.ones(
        kept.shape[0],
        dtype=torch.float32,
        device=kept.device,
    )
    weighted_dropped = dropped.float() * float(alpha)
    recycled.index_add_(0, owner, weighted_dropped)
    counts.index_add_(
        0,
        owner,
        torch.full(
            (dropped.shape[0],),
            float(alpha),
            dtype=torch.float32,
            device=kept.device,
        ),
    )
    recycled = recycled / counts.unsqueeze(-1)

    output = image_features.clone()
    output[keep_mask] = recycled.to(image_features.dtype)
    return output


def _index_add_recycled_tokens(image_features, keep_mask, owner, dropped_indices, alpha):
    recycled = image_features.float().clone()
    counts = torch.ones(
        image_features.shape[0],
        dtype=torch.float32,
        device=image_features.device,
    )
    recycled.index_add_(0, owner, image_features[dropped_indices].float() * float(alpha))
    counts.index_add_(
        0,
        owner,
        torch.full(
            (dropped_indices.shape[0],),
            float(alpha),
            dtype=torch.float32,
            device=image_features.device,
        ),
    )
    output = image_features.clone()
    output[keep_mask] = (
        recycled[keep_mask] / counts[keep_mask].unsqueeze(-1)
    ).to(image_features.dtype)
    return output


def recycle_dropped_visual_tokens_window(
    image_features,
    keep_mask,
    alpha=0.25,
    window_size=64,
):
    if alpha <= 0.0 or keep_mask.all() or (~keep_mask).sum() == 0:
        return image_features
    if keep_mask.sum() == 0:
        raise ValueError("Cannot recycle dropped visual tokens with no kept tokens")
    window_size = int(window_size)
    if window_size < 1:
        raise ValueError("visual_prune_recycle_window_size must be >= 1")

    num_tokens = image_features.shape[0]
    owner_chunks = []
    dropped_chunks = []
    fallback_dropped = []
    all_indices = torch.arange(num_tokens, device=image_features.device)

    for start in range(0, num_tokens, window_size):
        end = min(start + window_size, num_tokens)
        window_indices = all_indices[start:end]
        window_keep = window_indices[keep_mask[start:end]]
        window_dropped = window_indices[(~keep_mask[start:end])]
        if window_dropped.numel() == 0:
            continue
        if window_keep.numel() == 0:
            fallback_dropped.append(window_dropped)
            continue

        kept_norm = F.normalize(image_features[window_keep].float(), dim=-1)
        dropped_norm = F.normalize(image_features[window_dropped].float(), dim=-1)
        owner = window_keep[(dropped_norm @ kept_norm.T).argmax(dim=-1)]
        owner_chunks.append(owner)
        dropped_chunks.append(window_dropped)

    if fallback_dropped:
        fallback_dropped = torch.cat(fallback_dropped, dim=0)
        global_keep = all_indices[keep_mask]
        kept_norm = F.normalize(image_features[global_keep].float(), dim=-1)
        dropped_norm = F.normalize(image_features[fallback_dropped].float(), dim=-1)
        owner_chunks.append(global_keep[(dropped_norm @ kept_norm.T).argmax(dim=-1)])
        dropped_chunks.append(fallback_dropped)

    if not dropped_chunks:
        return image_features
    owner = torch.cat(owner_chunks, dim=0)
    dropped_indices = torch.cat(dropped_chunks, dim=0)
    return _index_add_recycled_tokens(
        image_features,
        keep_mask,
        owner,
        dropped_indices,
        alpha,
    )


def recycle_dropped_visual_tokens(image_features, keep_mask, args):
    if not bool(getattr(args, "visual_prune_recycle", False)):
        return image_features
    mode = getattr(args, "visual_prune_recycle_mode", "knn")
    alpha = float(getattr(args, "visual_prune_recycle_alpha", 0.25))
    if mode == "knn":
        return recycle_dropped_visual_tokens_knn(image_features, keep_mask, alpha)
    if mode == "window":
        return recycle_dropped_visual_tokens_window(
            image_features,
            keep_mask,
            alpha=alpha,
            window_size=int(getattr(args, "visual_prune_recycle_window_size", 64)),
        )
    raise ValueError(f"Unsupported visual_prune_recycle_mode: {mode}")


def prune_prompt_embeddings(
    inputs_embeds,
    attention_mask,
    image_mask,
    visual_keep_mask,
    position_ids=None,
):
    if inputs_embeds.shape[0] != 1:
        raise ValueError("visual token pruning currently expects eval batch size 1")
    if attention_mask is None:
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )
    sequence_keep_mask = attention_mask[0].bool().clone()
    visual_positions = image_mask[0].nonzero(as_tuple=False).flatten()
    if visual_positions.numel() != visual_keep_mask.numel():
        raise ValueError(
            "visual keep mask length does not match image token positions: "
            f"{visual_keep_mask.numel()} vs {visual_positions.numel()}"
        )
    sequence_keep_mask[visual_positions] = visual_keep_mask.to(sequence_keep_mask.device)

    pruned_embeds = inputs_embeds[:, sequence_keep_mask, :]
    pruned_attention_mask = attention_mask[:, sequence_keep_mask]
    pruned_position_ids = None
    if position_ids is not None:
        if position_ids.dim() == 3:
            pruned_position_ids = position_ids[:, :, sequence_keep_mask]
        else:
            pruned_position_ids = position_ids[:, sequence_keep_mask]
    return pruned_embeds, pruned_attention_mask, pruned_position_ids


def _get_language_model(model):
    return model_utils.get_language_model(model)


def _get_visual_module(model):
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    return None


def _normalize_image_features(image_features):
    if hasattr(image_features, "pooler_output"):
        image_features = image_features.pooler_output
    if isinstance(image_features, (list, tuple)):
        image_features = [item for item in image_features if item is not None]
        if len(image_features) == 1 and torch.is_tensor(image_features[0]):
            image_features = image_features[0]
        else:
            image_features = torch.cat(image_features, dim=0)
    if image_features.dim() == 3 and image_features.shape[0] == 1:
        image_features = image_features[0]
    if image_features.dim() != 2:
        image_features = image_features.reshape(-1, image_features.shape[-1])
    return image_features


def _llava_component(model, name):
    if hasattr(model, name):
        return getattr(model, name)
    if hasattr(model, "model") and hasattr(model.model, name):
        return getattr(model.model, name)
    raise AttributeError(f"Cannot locate LLaVA component: {name}")


def _config_value(config, names, default=None):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _llava_next_image_num_patches(image_sizes, model):
    try:
        from transformers.image_processing_utils import select_best_resolution
    except ImportError as exc:
        raise ImportError(
            "LLaVA-Next visual pruning fallback requires transformers.image_processing_utils"
        ) from exc

    grid_pinpoints = model.config.image_grid_pinpoints
    image_size = model.config.vision_config.image_size
    counts = []
    for imsize in image_sizes:
        if not isinstance(imsize, (list, tuple)):
            imsize = imsize.tolist()
        best_height, best_width = select_best_resolution(imsize, grid_pinpoints)
        num_patches = 0
        for _ in range(0, best_height, image_size):
            for _ in range(0, best_width, image_size):
                num_patches += 1
        counts.append(num_patches + 1)
    return counts


def _llava_next_image_newline(model):
    if hasattr(model, "image_newline"):
        return model.image_newline
    if hasattr(model, "model") and hasattr(model.model, "image_newline"):
        return model.model.image_newline
    return None


def _llava_next_unpad_image(tensor, original_size):
    try:
        from transformers.models.llava_next.modeling_llava_next import unpad_image

        return unpad_image(tensor, original_size)
    except ImportError:
        if not isinstance(original_size, (list, tuple)):
            original_size = original_size.tolist()
        original_height, original_width = original_size
        current_height, current_width = tensor.shape[1:]
        original_aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            scale_factor = current_width / original_width
            new_height = int(round(original_height * scale_factor, 7))
            padding = (current_height - new_height) // 2
            return tensor[:, padding : current_height - padding, :]
        scale_factor = current_height / original_height
        new_width = int(round(original_width * scale_factor, 7))
        padding = (current_width - new_width) // 2
        return tensor[:, :, padding : current_width - padding]


def _llava_next_component(model, name):
    return _llava_component(model, name)


def _extract_llava_v15_vision_outputs(model, inputs, output_attentions=False):
    pixel_values = inputs.get("pixel_values", None)
    if pixel_values is None:
        raise ValueError("LLaVA-1.5 visual pruning requires pixel_values")
    if pixel_values.dim() != 4:
        raise ValueError(
            f"LLaVA-1.5 pixel_values must be 4D, got {pixel_values.shape}"
        )

    vision_feature_layer = _config_value(
        model.config,
        ("vision_feature_layer",),
        default=-2,
    )
    if not isinstance(vision_feature_layer, int):
        raise ValueError("LLaVA-1.5 pruning requires a single integer vision_feature_layer")
    vision_feature_select_strategy = _config_value(
        model.config,
        ("vision_feature_select_strategy",),
        default="default",
    )
    vision_tower = _llava_component(model, "vision_tower")
    image_outputs = vision_tower(
        pixel_values.to(dtype=getattr(vision_tower, "dtype", pixel_values.dtype)),
        output_hidden_states=True,
        output_attentions=output_attentions,
    )
    selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
    if vision_feature_select_strategy == "default":
        selected_image_feature = selected_image_feature[:, 1:]
    elif vision_feature_select_strategy != "full":
        raise ValueError(
            "Unsupported LLaVA-1.5 vision_feature_select_strategy: "
            f"{vision_feature_select_strategy}"
        )
    return image_outputs, selected_image_feature, vision_feature_layer


def _visionzip_key_metric(vision_tower, image_outputs, vision_feature_layer):
    """Reconstruct VisionZip's mean-head raw key metric without monkeypatching.

    The reference repository replaces CLIPAttention.forward to retain this
    tensor.  Computing it from the returned layer input is equivalent and keeps
    the shared HF model class untouched for every other pruning branch.
    """
    vision_model = getattr(vision_tower, "vision_model", None)
    if vision_model is None and hasattr(vision_tower, "model"):
        vision_model = getattr(vision_tower.model, "vision_model", None)
    if vision_model is None or not hasattr(vision_model, "encoder"):
        raise AttributeError("VisionZip requires a CLIP-style vision encoder")
    layers = vision_model.encoder.layers
    layer_count = len(layers)
    layer_index = int(vision_feature_layer)
    if layer_index < 0:
        layer_index += layer_count
    if layer_index < 0 or layer_index >= layer_count:
        raise ValueError("VisionZip vision feature layer is outside the CLIP encoder")

    layer = layers[layer_index]
    hidden_input = image_outputs.hidden_states[layer_index]
    normalized = layer.layer_norm1(hidden_input)
    attention = layer.self_attn
    key_states = attention.k_proj(normalized)
    num_heads = int(getattr(attention, "num_heads", 0))
    if num_heads <= 0 or key_states.shape[-1] % num_heads:
        raise ValueError("VisionZip could not infer the CLIP attention head layout")
    head_dim = key_states.shape[-1] // num_heads
    return key_states.view(
        key_states.shape[0], key_states.shape[1], num_heads, head_dim
    ).transpose(1, 2).mean(dim=1)


def _scatter_condensed_visual_features(condensed, original_visual_tokens):
    """Represent newly condensed tokens in the existing hard-mask interface."""
    condensed = _normalize_image_features(condensed)
    original_visual_tokens = int(original_visual_tokens)
    if condensed.shape[0] > original_visual_tokens:
        raise ValueError(
            "condensed visual-token count exceeds the model prompt capacity: "
            f"{condensed.shape[0]} vs {original_visual_tokens}"
        )
    dense = condensed.new_zeros(original_visual_tokens, condensed.shape[-1])
    keep_mask = torch.zeros(
        original_visual_tokens,
        dtype=torch.bool,
        device=condensed.device,
    )
    dense[: condensed.shape[0]] = condensed
    keep_mask[: condensed.shape[0]] = True
    return dense, keep_mask


def _extract_llava_v15_visionzip_features(
    model, inputs, args, original_visual_tokens
):
    image_outputs, _, vision_feature_layer = _extract_llava_v15_vision_outputs(
        model, inputs, output_attentions=True
    )
    if image_outputs.attentions is None:
        raise RuntimeError("LLaVA-1.5 did not return attentions required by VisionZip")
    vision_tower = _llava_component(model, "vision_tower")
    full_hidden = image_outputs.hidden_states[vision_feature_layer]
    cls_attention = image_outputs.attentions[vision_feature_layer][:, :, 0, 1:]
    key_metric = _visionzip_key_metric(
        vision_tower, image_outputs, vision_feature_layer
    )
    condensed, _ = baseline_pruning.visionzip_condense(
        full_hidden,
        cls_attention,
        key_metric,
        dominant_tokens=int(
            getattr(args, "visual_prune_visionzip_dominant_tokens", 54)
        ),
        contextual_tokens=int(
            getattr(args, "visual_prune_visionzip_contextual_tokens", 10)
        ),
    )
    projected = _llava_component(model, "multi_modal_projector")(condensed)
    return _scatter_condensed_visual_features(projected, original_visual_tokens)


def _extract_llava_v15_specflow_features(
    model, inputs, args, original_visual_tokens
):
    image_outputs, selected, vision_feature_layer = _extract_llava_v15_vision_outputs(
        model, inputs, output_attentions=True
    )
    if image_outputs.attentions is None:
        raise RuntimeError("LLaVA-1.5 did not return attentions required by SpecFlow")
    projected = _llava_component(model, "multi_modal_projector")(selected)
    attention = image_outputs.attentions[vision_feature_layer][:, :, 0, 1:]
    energy = baseline_pruning.specflow_heat_flow(
        attention,
        projected,
        use_best_heads=bool(
            getattr(args, "visual_prune_specflow_use_best_heads", True)
        ),
        n_best=int(getattr(args, "visual_prune_specflow_num_best_heads", 16)),
        k_neighbors=int(getattr(args, "visual_prune_specflow_k_neighbors", 25)),
        diffusion_steps=int(
            getattr(args, "visual_prune_specflow_diffusion_steps", 2)
        ),
        alpha=float(getattr(args, "visual_prune_specflow_alpha", 0.7)),
    )
    side = int(round(math.sqrt(projected.shape[1])))
    grid_shape = (side, side) if side * side == projected.shape[1] else None
    condensed = baseline_pruning.specflow_condense(
        projected,
        energy,
        token_budget=int(
            getattr(args, "visual_prune_specflow_token_budget", 192)
        ),
        min_crop_size=int(
            getattr(args, "visual_prune_specflow_min_crop_size", 4)
        ),
        adaptive_threshold=float(
            getattr(args, "visual_prune_specflow_adaptive_threshold", 1e-4)
        ),
        use_coreset_sinks=bool(
            getattr(args, "visual_prune_specflow_use_coreset_sinks", True)
        ),
        n_sinks=int(getattr(args, "visual_prune_specflow_num_sinks", 2)),
        grid_shape=grid_shape,
    )
    return _scatter_condensed_visual_features(condensed, original_visual_tokens)


def _extract_llava_v15_image_features(model, inputs):
    _, selected_image_feature, _ = _extract_llava_v15_vision_outputs(
        model,
        inputs,
        output_attentions=False,
    )
    multi_modal_projector = _llava_component(model, "multi_modal_projector")
    return _normalize_image_features(multi_modal_projector(selected_image_feature))


def _extract_llava_v15_vispruner_features(model, inputs, args):
    image_outputs, selected_image_feature, vision_feature_layer = (
        _extract_llava_v15_vision_outputs(
            model,
            inputs,
            output_attentions=True,
        )
    )
    if image_outputs.attentions is None:
        raise RuntimeError(
            "LLaVA-1.5 vision tower did not return attentions required by VisPruner"
        )
    image_attentions = image_outputs.attentions[vision_feature_layer][:, :, 0, 1:]
    if selected_image_feature.shape[1] != image_attentions.shape[-1]:
        raise ValueError(
            "LLaVA-1.5 visual feature and CLS-attention token counts differ: "
            f"{selected_image_feature.shape[1]} vs {image_attentions.shape[-1]}"
        )

    diversity_image_feature = selected_image_feature
    if bool(getattr(args, "visual_prune_vispruner_quant_vit_features", True)):
        diversity_image_feature = fake_quantize_visual_features(
            selected_image_feature,
            args,
        )
    keep_ratio = float(getattr(args, "visual_prune_keep_ratio", 1.0))
    min_keep_tokens = int(getattr(args, "visual_prune_min_keep_tokens", 1))
    important_ratio = float(getattr(args, "visual_prune_important_ratio", 0.5))
    tokens_per_image = selected_image_feature.shape[1]
    keep_count = int(math.ceil(tokens_per_image * keep_ratio))
    keep_count = max(min_keep_tokens, keep_count)
    keep_count = min(tokens_per_image, keep_count)
    keep_masks = select_vispruner_visual_tokens_batched(
        selected_image_feature,
        image_attentions,
        keep_count=keep_count,
        important_ratio=important_ratio,
        diversity_features=diversity_image_feature,
    )

    multi_modal_projector = _llava_component(model, "multi_modal_projector")
    if keep_count == tokens_per_image:
        projected_features = multi_modal_projector(selected_image_feature)
    else:
        projected_features = _project_selected_llava_tokens(
            multi_modal_projector,
            selected_image_feature,
            keep_masks,
            context="LLaVA-1.5",
        )
    return (
        _normalize_image_features(projected_features),
        keep_masks.reshape(-1),
    )


def _crisp_selector_kwargs(args):
    return {
        "alpha": float(getattr(args, "visual_prune_crisp_alpha", 0.3)),
        "low_budget_alpha": float(
            getattr(args, "visual_prune_crisp_low_budget_alpha", 0.5)
        ),
        "low_budget_threshold": int(
            getattr(args, "visual_prune_crisp_low_budget_threshold", 100)
        ),
    }


@torch.no_grad()
def _apply_crisp_weight_fake_quant(module, args, component_name):
    """Idempotently replace Linear weights with dequantized RTN weights.

    This mirrors the repository's normal RTN path: per-output-channel weight
    quantization, optional asymmetric ranges, and optional MSE clipping. Only
    Linear weights are quantized; biases, normalization, embeddings, and patch
    convolutions stay in their original precision.
    """
    if not bool(getattr(args, "visual_prune_crisp_quantize_weights", False)):
        return 0

    bits = int(getattr(args, "w_bits", 16))
    groupsize = int(getattr(args, "w_groupsize", -1))
    if bits < 2:
        raise ValueError("CRISP weight fake quantization requires w_bits >= 2")
    if groupsize != -1:
        raise ValueError(
            "CRISP RTN weight fake quantization currently requires w_groupsize=-1"
        )
    signature = (
        bits,
        bool(getattr(args, "w_asym", False)),
        bool(getattr(args, "w_clip", True)),
    )
    previous_signature = getattr(module, "_crisp_weight_fake_quant_signature", None)
    if previous_signature is not None:
        if previous_signature != signature:
            raise RuntimeError(
                f"{component_name} was already CRISP-fake-quantized with "
                f"{previous_signature}, cannot re-quantize it with {signature}"
            )
        return 0
    if bits >= 16:
        module._crisp_weight_fake_quant_signature = signature
        return 0

    quantized_count = 0
    for layer in module.modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        weight = layer.weight.data
        quantizer = quant_utils.WeightQuantizer()
        quantizer.configure(
            bits,
            perchannel=True,
            sym=not signature[1],
            mse=signature[2],
        )
        quantizer.find_params(weight)
        layer.weight.data = quantizer.quantize(weight).to(
            device=weight.device,
            dtype=weight.dtype,
        )
        quantized_count += 1

    module._crisp_weight_fake_quant_signature = signature
    logging.info(
        "CRISP fake-quantized %d %s Linear weights with W%d %s %s RTN",
        quantized_count,
        component_name,
        bits,
        "asymmetric" if signature[1] else "symmetric",
        "MSE-clipped" if signature[2] else "min-max",
    )
    return quantized_count


def _prepare_llava_crisp_weight_fake_quant(vision_tower, projector, args):
    _apply_crisp_weight_fake_quant(vision_tower, args, "ViT")
    _apply_crisp_weight_fake_quant(projector, args, "projector")


def _p4q_feature_mode(args):
    # This release always selects tokens using native projector features.
    # Keep this independent of YAML/CLI settings and LLM W4A4 quantization.
    return "native"


def _p4q_feature_suffix(args):
    mode = _p4q_feature_mode(args)
    if mode == "native":
        return "featnative"
    prefix = "feature" if mode == "feature_quant" else "weight"
    bits = int(getattr(args, f"visual_prune_p4q_{prefix}_bits", 4))
    group = int(getattr(args, f"visual_prune_p4q_{prefix}_groupsize", -1))
    asym = bool(getattr(args, f"visual_prune_p4q_{prefix}_asym", False))
    clip = (float(getattr(args, "visual_prune_p4q_feature_clip_ratio", 0.9))
            if prefix == "feature" else int(bool(getattr(args, "visual_prune_p4q_weight_clip", True))))
    return f"feat{mode}-q{bits}-g{group}-{'asym' if asym else 'sym'}-c{_format_config_value(clip)}"


def _p4q_selection_features(features, args):
    if _p4q_feature_mode(args) != "feature_quant":
        return features
    return fake_quantize_visual_features(features, SimpleNamespace(
        visual_prune_quant_bits=getattr(args, "visual_prune_p4q_feature_bits", 4),
        visual_prune_quant_groupsize=getattr(args, "visual_prune_p4q_feature_groupsize", -1),
        visual_prune_quant_asym=getattr(args, "visual_prune_p4q_feature_asym", False),
        visual_prune_quant_clip_ratio=getattr(args, "visual_prune_p4q_feature_clip_ratio", 0.9),
    ))


@torch.no_grad()
def _prepare_p4q_weights(args, *modules):
    mode = _p4q_feature_mode(args)
    signature = (
        int(getattr(args, "visual_prune_p4q_weight_bits", 4)),
        int(getattr(args, "visual_prune_p4q_weight_groupsize", -1)),
        bool(getattr(args, "visual_prune_p4q_weight_asym", False)),
        bool(getattr(args, "visual_prune_p4q_weight_clip", True)),
    )
    if mode == "weight_quant" and (not 2 <= signature[0] <= 16 or signature[1] != -1):
        raise ValueError("P4Q weight quantization requires bits in [2,16] and groupsize=-1")
    # Preflight all modules before mutating any weights. Changing modes requires
    # a freshly loaded model, since fake quantization overwrites Linear weights.
    for module in modules:
        previous = getattr(module, "_p4q_weight_quant_signature", None)
        if previous is not None and (mode != "weight_quant" or previous != signature):
            raise RuntimeError("Reload the model before changing P4Q weight quantization mode/settings")
    if mode != "weight_quant" or signature[0] >= 16:
        return
    for module in modules:
        if getattr(module, "_p4q_weight_quant_signature", None) == signature:
            continue
        for layer in module.modules():
            if not isinstance(layer, torch.nn.Linear):
                continue
            weight = layer.weight.data
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(signature[0], perchannel=True, sym=not signature[2], mse=signature[3])
            quantizer.find_params(weight)
            layer.weight.data = quantizer.quantize(weight).to(weight)
        module._p4q_weight_quant_signature = signature


def _p4q_selector_kwargs(args):
    result = {
        "beta": float(getattr(args, "visual_prune_p4q_beta", 0.5)),
        "text_lambda": float(getattr(args, "visual_prune_p4q_lambda", 0.6)),
    }
    for name, option in (('p4q_batch_crops', 'batch_crops'),
                         ('p4q_gpu_budget', 'gpu_budget'),
                         ('p4q_verify_exact', 'verify_exact')):
        if getattr(args, name, False):
            result[option] = True
    return result


def _extract_llava_v15_crisp_features(model, inputs, args, keyword_embeddings):
    vision_tower = _llava_component(model, "vision_tower")
    multi_modal_projector = _llava_component(model, "multi_modal_projector")
    _prepare_llava_crisp_weight_fake_quant(
        vision_tower,
        multi_modal_projector,
        args,
    )
    image_outputs, selected_image_feature, vision_feature_layer = (
        _extract_llava_v15_vision_outputs(
            model,
            inputs,
            output_attentions=True,
        )
    )
    if image_outputs.attentions is None:
        raise RuntimeError(
            "LLaVA-1.5 vision tower did not return attentions required by CRISP"
        )
    image_attentions = image_outputs.attentions[vision_feature_layer][:, :, 0, 1:]
    projected_features = multi_modal_projector(selected_image_feature)
    if projected_features.shape[1] != image_attentions.shape[-1]:
        raise ValueError(
            "LLaVA-1.5 CRISP feature and CLS-attention token counts differ: "
            f"{projected_features.shape[1]} vs {image_attentions.shape[-1]}"
        )

    keep_ratio = float(getattr(args, "visual_prune_keep_ratio", 1.0))
    min_keep_tokens = int(getattr(args, "visual_prune_min_keep_tokens", 1))
    keep_count = int(math.ceil(projected_features.shape[1] * keep_ratio))
    keep_count = min(
        projected_features.shape[1], max(min_keep_tokens, keep_count)
    )
    keep_masks = select_crisp_visual_tokens_batched(
        projected_features,
        image_attentions,
        keyword_embeddings,
        keep_count,
        **_crisp_selector_kwargs(args),
    )
    return _normalize_image_features(projected_features), keep_masks.reshape(-1)


def _extract_llava_v15_p4q_features(model, inputs, args, keyword_embeddings):
    _prepare_p4q_weights(args, _llava_component(model, "vision_tower"),
                         _llava_component(model, "multi_modal_projector"))
    image_outputs, selected_image_feature, vision_feature_layer = (
        _extract_llava_v15_vision_outputs(
            model,
            inputs,
            output_attentions=True,
        )
    )
    if image_outputs.attentions is None:
        raise RuntimeError(
            "LLaVA-1.5 vision tower did not return attentions required by P4Q"
        )
    image_attentions = image_outputs.attentions[vision_feature_layer][:, :, 0, 1:]
    multi_modal_projector = _llava_component(model, "multi_modal_projector")
    projected_features = multi_modal_projector(selected_image_feature)
    if projected_features.shape[1] != image_attentions.shape[-1]:
        raise ValueError(
            "LLaVA-1.5 P4Q feature and CLS-attention token counts differ: "
            f"{projected_features.shape[1]} vs {image_attentions.shape[-1]}"
        )

    keep_ratio = float(getattr(args, "visual_prune_keep_ratio", 1.0))
    min_keep_tokens = int(getattr(args, "visual_prune_min_keep_tokens", 1))
    keep_count = int(math.ceil(projected_features.shape[1] * keep_ratio))
    keep_count = min(
        projected_features.shape[1], max(min_keep_tokens, keep_count)
    )
    selection_features = _p4q_selection_features(projected_features, args)
    keep_masks = select_p4q_visual_tokens_batched(
        selection_features,
        image_attentions,
        keyword_embeddings,
        keep_count,
        **_p4q_selector_kwargs(args),
    )
    return _normalize_image_features(projected_features), keep_masks.reshape(-1)


def _llava_next_anyres_grid_shape(image_size, model):
    try:
        from transformers.image_processing_utils import select_best_resolution
    except ImportError as exc:
        raise ImportError(
            "LLaVA-Next visual pruning requires transformers.image_processing_utils"
        ) from exc

    if not isinstance(image_size, (list, tuple)):
        image_size = image_size.tolist()
    best_height, best_width = select_best_resolution(
        image_size,
        model.config.image_grid_pinpoints,
    )
    patch_size = model.config.vision_config.image_size
    return best_height // patch_size, best_width // patch_size


def _pack_llava_next_keep_masks(model, crop_keep_masks, image_num_patches, image_sizes):
    split_masks = torch.split(crop_keep_masks, image_num_patches, dim=0)
    packed_masks = []
    has_newline = _llava_next_image_newline(model) is not None
    for image_idx, image_mask in enumerate(split_masks):
        if image_mask.shape[0] > 1:
            base_mask = image_mask[0]
            patch_mask = image_mask[1:]
            height = width = (
                model.config.vision_config.image_size
                // model.config.vision_config.patch_size
            )
            num_patch_height, num_patch_width = _llava_next_anyres_grid_shape(
                image_sizes[image_idx],
                model,
            )
            patch_mask = patch_mask.view(num_patch_height, num_patch_width, height, width)
            patch_mask = patch_mask.permute(0, 2, 1, 3).contiguous().unsqueeze(0)
            patch_mask = patch_mask.flatten(1, 2).flatten(2, 3)
            patch_mask = _llava_next_unpad_image(patch_mask, image_sizes[image_idx])
            if has_newline:
                newline_mask = torch.ones(
                    *patch_mask.shape[:-1],
                    1,
                    dtype=torch.bool,
                    device=patch_mask.device,
                )
                patch_mask = torch.cat((patch_mask, newline_mask), dim=-1)
            patch_mask = patch_mask.flatten(1, 2).squeeze(0)
            packed_masks.append(torch.cat((base_mask, patch_mask), dim=0))
        else:
            image_mask = image_mask[0]
            if has_newline:
                newline_mask = torch.ones(1, dtype=torch.bool, device=image_mask.device)
                image_mask = torch.cat((image_mask, newline_mask), dim=0)
            packed_masks.append(image_mask)
    return torch.cat(packed_masks, dim=0)


def _pack_llava_next_image_features(
    model,
    projected_features,
    image_num_patches,
    image_sizes,
    vision_feature_select_strategy,
):
    try:
        pack_image_features = _llava_next_component(model, "pack_image_features")
    except AttributeError:
        return projected_features.reshape(-1, projected_features.shape[-1])

    split_features = torch.split(projected_features, image_num_patches, dim=0)
    try:
        packed_features, _ = pack_image_features(
            split_features,
            image_sizes,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_newline=_llava_next_image_newline(model),
        )
    except TypeError:
        packed_features, _ = pack_image_features(
            split_features,
            image_sizes,
            image_newline=_llava_next_image_newline(model),
        )
    return packed_features


def _extract_llava_next_image_features_with_forward_path(model, inputs):
    pixel_values = inputs.get("pixel_values", None)
    image_sizes = inputs.get("image_sizes", None)
    if pixel_values is None or image_sizes is None:
        raise ValueError("LLaVA-Next visual pruning requires pixel_values and image_sizes")

    vision_feature_layer = _config_value(
        model.config,
        ("vision_feature_layer",),
        default=-2,
    )
    vision_feature_select_strategy = _config_value(
        model.config,
        ("vision_feature_select_strategy",),
        default="default",
    )
    image_num_patches = _llava_next_image_num_patches(image_sizes, model)

    if pixel_values.dim() == 5:
        pixel_values = torch.cat(
            [
                patch_values[:num_patch]
                for patch_values, num_patch in zip(pixel_values, image_num_patches)
            ],
            dim=0,
        )
    elif pixel_values.dim() != 4:
        raise ValueError(
            f"LLaVA-Next pixel_values must be 4D or 5D, got {pixel_values.shape}"
        )

    vision_tower = _llava_next_component(model, "vision_tower")
    multi_modal_projector = _llava_next_component(model, "multi_modal_projector")
    image_outputs = vision_tower(pixel_values, output_hidden_states=True)
    selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
    if vision_feature_select_strategy == "default":
        selected_image_feature = selected_image_feature[:, 1:]
    elif vision_feature_select_strategy != "full":
        raise ValueError(
            "Unsupported LLaVA-Next vision_feature_select_strategy: "
            f"{vision_feature_select_strategy}"
        )

    projected_features = multi_modal_projector(selected_image_feature)
    return _pack_llava_next_image_features(
        model,
        projected_features,
        image_num_patches,
        image_sizes,
        vision_feature_select_strategy,
    )


def _extract_llava_next_image_features(model, inputs):
    kwargs = {
        "pixel_values": inputs["pixel_values"],
        "image_sizes": inputs.get("image_sizes", None),
    }
    for name in ("vision_feature_layer", "vision_feature_select_strategy"):
        value = getattr(model.config, name, None)
        if value is not None:
            kwargs[name] = value
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if hasattr(model, "get_image_features"):
        return _normalize_image_features(model.get_image_features(**kwargs))
    return _normalize_image_features(
        _extract_llava_next_image_features_with_forward_path(model, inputs)
    )


def _project_selected_llava_tokens(
    multi_modal_projector,
    selected_image_feature,
    keep_masks,
    context="LLaVA",
):
    """Project only kept tokens while preserving the dense visual layout.

    LLaVA's projector is token-wise, so gathering the kept tokens before
    projection is equivalent to projecting every token and selecting the same
    outputs afterwards. Scattering back keeps the existing model-specific
    packing behavior intact; zero placeholders are removed by the hard mask.
    """
    if keep_masks.dtype != torch.bool:
        raise TypeError(f"{context} keep masks must be boolean")
    if selected_image_feature.shape[:2] != keep_masks.shape:
        raise ValueError(
            f"{context} feature and keep mask shapes differ: "
            f"{selected_image_feature.shape[:2]} vs {keep_masks.shape}"
        )

    kept_features = selected_image_feature[keep_masks]
    if kept_features.shape[0] == 0:
        raise ValueError(f"{context} VisPruner must keep at least one visual token")

    projected_kept_features = multi_modal_projector(kept_features)
    if projected_kept_features.dim() != 2:
        raise ValueError(
            f"{context} projector must preserve the gathered token dimension, "
            f"got shape {projected_kept_features.shape}"
        )

    projected_features = projected_kept_features.new_zeros(
        *keep_masks.shape,
        projected_kept_features.shape[-1],
    )
    projected_features[keep_masks] = projected_kept_features
    return projected_features


def _project_selected_llava_next_tokens(
    multi_modal_projector,
    selected_image_feature,
    crop_keep_masks,
):
    return _project_selected_llava_tokens(
        multi_modal_projector,
        selected_image_feature,
        crop_keep_masks,
        context="LLaVA-Next crop",
    )


def _extract_llava_next_vispruner_features(model, inputs, args):
    pixel_values = inputs.get("pixel_values", None)
    image_sizes = inputs.get("image_sizes", None)
    if pixel_values is None or image_sizes is None:
        raise ValueError("LLaVA-Next VisPruner requires pixel_values and image_sizes")

    vision_feature_layer = _config_value(
        model.config,
        ("vision_feature_layer",),
        default=-2,
    )
    if not isinstance(vision_feature_layer, int):
        raise ValueError("VisPruner requires a single integer vision_feature_layer")
    vision_feature_select_strategy = _config_value(
        model.config,
        ("vision_feature_select_strategy",),
        default="default",
    )
    if vision_feature_select_strategy != "default":
        raise ValueError(
            "LLaVA-Next VisPruner currently requires vision_feature_select_strategy='default'"
        )

    image_num_patches = _llava_next_image_num_patches(image_sizes, model)
    if pixel_values.dim() == 5:
        pixel_values = torch.cat(
            [
                patch_values[:num_patch]
                for patch_values, num_patch in zip(pixel_values, image_num_patches)
            ],
            dim=0,
        )
    elif pixel_values.dim() != 4:
        raise ValueError(
            f"LLaVA-Next pixel_values must be 4D or 5D, got {pixel_values.shape}"
        )

    vision_tower = _llava_next_component(model, "vision_tower")
    multi_modal_projector = _llava_next_component(model, "multi_modal_projector")
    image_outputs = vision_tower(
        pixel_values,
        output_hidden_states=True,
        output_attentions=True,
    )
    selected_image_feature = image_outputs.hidden_states[vision_feature_layer][:, 1:]
    image_attentions = image_outputs.attentions[vision_feature_layer][:, :, 0, 1:]
    diversity_image_feature = selected_image_feature
    if bool(getattr(args, "visual_prune_vispruner_quant_vit_features", True)):
        diversity_image_feature = fake_quantize_visual_features(selected_image_feature, args)

    keep_ratio = float(getattr(args, "visual_prune_keep_ratio", 1.0))
    min_keep_tokens = int(getattr(args, "visual_prune_min_keep_tokens", 1))
    important_ratio = float(getattr(args, "visual_prune_important_ratio", 0.5))
    tokens_per_crop = selected_image_feature.shape[1]
    keep_count = int(math.ceil(tokens_per_crop * keep_ratio))
    keep_count = max(min_keep_tokens, keep_count)
    keep_count = min(tokens_per_crop, keep_count)
    crop_keep_masks = select_vispruner_visual_tokens_batched(
        selected_image_feature,
        image_attentions,
        keep_count=keep_count,
        important_ratio=important_ratio,
        diversity_features=diversity_image_feature,
    )
    if keep_count == tokens_per_crop:
        # Preserve the original full-projection path (and its exact numerics)
        # when pruning is effectively disabled by the configured keep count.
        projected_features = multi_modal_projector(selected_image_feature)
    else:
        projected_features = _project_selected_llava_next_tokens(
            multi_modal_projector,
            selected_image_feature,
            crop_keep_masks,
        )
    packed_features = _normalize_image_features(
        _pack_llava_next_image_features(
            model,
            projected_features,
            image_num_patches,
            image_sizes,
            vision_feature_select_strategy,
        )
    )
    packed_keep_mask = _pack_llava_next_keep_masks(
        model,
        crop_keep_masks,
        image_num_patches,
        image_sizes,
    )
    if packed_features.shape[0] != packed_keep_mask.shape[0]:
        raise ValueError(
            "packed LLaVA-Next feature and VisPruner mask lengths differ: "
            f"{packed_features.shape[0]} vs {packed_keep_mask.shape[0]}"
        )
    return packed_features, packed_keep_mask


def _extract_llava_next_crisp_features(model, inputs, args, keyword_embeddings):
    pixel_values = inputs.get("pixel_values", None)
    image_sizes = inputs.get("image_sizes", None)
    if pixel_values is None or image_sizes is None:
        raise ValueError("LLaVA-Next CRISP requires pixel_values and image_sizes")

    vision_feature_layer = _config_value(
        model.config,
        ("vision_feature_layer",),
        default=-2,
    )
    if not isinstance(vision_feature_layer, int):
        raise ValueError("CRISP requires a single integer vision_feature_layer")
    vision_feature_select_strategy = _config_value(
        model.config,
        ("vision_feature_select_strategy",),
        default="default",
    )
    if vision_feature_select_strategy != "default":
        raise ValueError(
            "LLaVA-Next CRISP currently requires "
            "vision_feature_select_strategy='default'"
        )

    image_num_patches = _llava_next_image_num_patches(image_sizes, model)
    if pixel_values.dim() == 5:
        pixel_values = torch.cat(
            [
                patch_values[:num_patch]
                for patch_values, num_patch in zip(pixel_values, image_num_patches)
            ],
            dim=0,
        )
    elif pixel_values.dim() != 4:
        raise ValueError(
            f"LLaVA-Next pixel_values must be 4D or 5D, got {pixel_values.shape}"
        )

    vision_tower = _llava_next_component(model, "vision_tower")
    multi_modal_projector = _llava_next_component(model, "multi_modal_projector")
    _prepare_llava_crisp_weight_fake_quant(
        vision_tower,
        multi_modal_projector,
        args,
    )
    image_outputs = vision_tower(
        pixel_values,
        output_hidden_states=True,
        output_attentions=True,
    )
    selected_image_feature = image_outputs.hidden_states[vision_feature_layer][:, 1:]
    image_attentions = image_outputs.attentions[vision_feature_layer][:, :, 0, 1:]
    # CRISP defines relevance in the LLM embedding space, so unlike VisPruner
    # all crop tokens must pass through the projector before selection.
    projected_features = multi_modal_projector(selected_image_feature)
    expected = (image_attentions.shape[0], image_attentions.shape[-1])
    if tuple(projected_features.shape[:2]) != expected:
        raise ValueError(
            "LLaVA-Next CRISP feature and CLS-attention token counts differ: "
            f"{projected_features.shape[:2]} vs {expected}"
        )

    keep_ratio = float(getattr(args, "visual_prune_keep_ratio", 1.0))
    min_keep_tokens = int(getattr(args, "visual_prune_min_keep_tokens", 1))
    keep_count = int(math.ceil(projected_features.shape[1] * keep_ratio))
    keep_count = min(
        projected_features.shape[1], max(min_keep_tokens, keep_count)
    )
    crop_keep_masks = select_crisp_visual_tokens_batched(
        projected_features,
        image_attentions,
        keyword_embeddings,
        keep_count,
        **_crisp_selector_kwargs(args),
    )
    packed_features = _normalize_image_features(
        _pack_llava_next_image_features(
            model,
            projected_features,
            image_num_patches,
            image_sizes,
            vision_feature_select_strategy,
        )
    )
    packed_keep_mask = _pack_llava_next_keep_masks(
        model,
        crop_keep_masks,
        image_num_patches,
        image_sizes,
    )
    if packed_features.shape[0] != packed_keep_mask.shape[0]:
        raise ValueError(
            "packed LLaVA-Next feature and CRISP mask lengths differ: "
            f"{packed_features.shape[0]} vs {packed_keep_mask.shape[0]}"
        )
    return packed_features, packed_keep_mask


def _extract_llava_next_p4q_features(model, inputs, args, keyword_embeddings):
    _prepare_p4q_weights(args, _llava_next_component(model, "vision_tower"),
                         _llava_next_component(model, "multi_modal_projector"))
    pixel_values = inputs.get("pixel_values", None)
    image_sizes = inputs.get("image_sizes", None)
    if pixel_values is None or image_sizes is None:
        raise ValueError("LLaVA-Next P4Q requires pixel_values and image_sizes")

    vision_feature_layer = _config_value(
        model.config,
        ("vision_feature_layer",),
        default=-2,
    )
    if not isinstance(vision_feature_layer, int):
        raise ValueError("P4Q requires a single integer vision_feature_layer")
    vision_feature_select_strategy = _config_value(
        model.config,
        ("vision_feature_select_strategy",),
        default="default",
    )
    if vision_feature_select_strategy != "default":
        raise ValueError(
            "LLaVA-Next P4Q currently requires "
            "vision_feature_select_strategy='default'"
        )

    image_num_patches = _llava_next_image_num_patches(image_sizes, model)
    if pixel_values.dim() == 5:
        pixel_values = torch.cat(
            [
                patch_values[:num_patch]
                for patch_values, num_patch in zip(pixel_values, image_num_patches)
            ],
            dim=0,
        )
    elif pixel_values.dim() != 4:
        raise ValueError(
            f"LLaVA-Next pixel_values must be 4D or 5D, got {pixel_values.shape}"
        )

    vision_tower = _llava_next_component(model, "vision_tower")
    multi_modal_projector = _llava_next_component(model, "multi_modal_projector")
    image_outputs = vision_tower(
        pixel_values,
        output_hidden_states=True,
        output_attentions=True,
    )
    selected_image_feature = image_outputs.hidden_states[vision_feature_layer][:, 1:]
    image_attentions = image_outputs.attentions[vision_feature_layer][:, :, 0, 1:]
    projected_features = multi_modal_projector(selected_image_feature)
    expected = (image_attentions.shape[0], image_attentions.shape[-1])
    if tuple(projected_features.shape[:2]) != expected:
        raise ValueError(
            "LLaVA-Next P4Q feature and CLS-attention token counts differ: "
            f"{projected_features.shape[:2]} vs {expected}"
        )

    keep_ratio = float(getattr(args, "visual_prune_keep_ratio", 1.0))
    min_keep_tokens = int(getattr(args, "visual_prune_min_keep_tokens", 1))
    keep_count = int(math.ceil(projected_features.shape[1] * keep_ratio))
    keep_count = min(
        projected_features.shape[1], max(min_keep_tokens, keep_count)
    )
    selection_features = _p4q_selection_features(projected_features, args)
    crop_keep_masks = select_p4q_visual_tokens_batched(
        selection_features,
        image_attentions,
        keyword_embeddings,
        keep_count,
        **_p4q_selector_kwargs(args),
    )
    packed_features = _normalize_image_features(
        _pack_llava_next_image_features(
            model,
            projected_features,
            image_num_patches,
            image_sizes,
            vision_feature_select_strategy,
        )
    )
    packed_keep_mask = _pack_llava_next_keep_masks(
        model,
        crop_keep_masks,
        image_num_patches,
        image_sizes,
    )
    if packed_features.shape[0] != packed_keep_mask.shape[0]:
        raise ValueError(
            "packed LLaVA-Next feature and P4Q mask lengths differ: "
            f"{packed_features.shape[0]} vs {packed_keep_mask.shape[0]}"
        )
    return packed_features, packed_keep_mask


def _llava_next_flatten_pixel_values(pixel_values, image_num_patches):
    if pixel_values.dim() == 5:
        return torch.cat(
            [
                patch_values[:num_patch]
                for patch_values, num_patch in zip(pixel_values, image_num_patches)
            ],
            dim=0,
        )
    if pixel_values.dim() != 4:
        raise ValueError(
            f"LLaVA-Next pixel_values must be 4D or 5D, got {pixel_values.shape}"
        )
    return pixel_values


def _pack_llava_next_visionzip_condensed(
    model,
    condensed_crops,
    dominant_indices,
    image_num_patches,
    image_sizes,
    dominant_tokens,
):
    """Reproduce VisionZip's AnyRes restore-and-append token ordering."""
    feature_splits = torch.split(condensed_crops, image_num_patches, dim=0)
    index_splits = torch.split(dominant_indices, image_num_patches, dim=0)
    packed = []
    patch_side = (
        model.config.vision_config.image_size
        // model.config.vision_config.patch_size
    )
    patch_count = patch_side * patch_side
    image_newline = _llava_next_image_newline(model)

    for image_idx, (features, indices) in enumerate(
        zip(feature_splits, index_splits)
    ):
        if features.shape[0] == 1:
            current = features[0]
            # This matches the reference single-crop LLaVA-NeXT branch.
            if image_newline is not None:
                current = torch.cat(
                    (current, image_newline[None].to(current)), dim=0
                )
            packed.append(current)
            continue

        base_features = features[0]
        crop_features = features[1:]
        crop_indices = indices[1:]
        crop_count = crop_features.shape[0]
        dense = crop_features.new_zeros(crop_count, patch_count, features.shape[-1])
        dense_mask = torch.zeros(
            crop_count,
            patch_count,
            dtype=torch.bool,
            device=features.device,
        )
        patch_indices = crop_indices[:, 1:dominant_tokens] - 1
        batch_indices = torch.arange(crop_count, device=features.device).unsqueeze(1)
        dense[batch_indices, patch_indices] = crop_features[:, 1:dominant_tokens]
        dense_mask[batch_indices, patch_indices] = True

        grid_height, grid_width = _llava_next_anyres_grid_shape(
            image_sizes[image_idx], model
        )
        if grid_height * grid_width != crop_count:
            raise ValueError(
                "VisionZip AnyRes crop grid does not match encoded crops: "
                f"{grid_height}x{grid_width} vs {crop_count}"
            )
        spatial = dense.view(
            grid_height,
            grid_width,
            patch_side,
            patch_side,
            features.shape[-1],
        ).permute(4, 0, 2, 1, 3).contiguous()
        spatial = spatial.flatten(1, 2).flatten(2, 3)
        spatial = _llava_next_unpad_image(spatial, image_sizes[image_idx])
        spatial = spatial.permute(1, 2, 0).reshape(-1, features.shape[-1])

        spatial_mask = dense_mask.view(
            grid_height, grid_width, patch_side, patch_side
        ).permute(0, 2, 1, 3).contiguous().unsqueeze(0)
        spatial_mask = spatial_mask.flatten(1, 2).flatten(2, 3)
        spatial_mask = _llava_next_unpad_image(
            spatial_mask, image_sizes[image_idx]
        ).flatten()
        selected_spatial = spatial[spatial_mask]
        crop_cls = crop_features[:, 0].reshape(-1, features.shape[-1])
        contextual = crop_features[:, dominant_tokens:].reshape(
            -1, features.shape[-1]
        )
        packed.append(
            torch.cat(
                (base_features, selected_spatial, crop_cls, contextual), dim=0
            )
        )
    return torch.cat(packed, dim=0)


def _extract_llava_next_visionzip_features(
    model, inputs, args, original_visual_tokens
):
    pixel_values = inputs.get("pixel_values")
    image_sizes = inputs.get("image_sizes")
    if pixel_values is None or image_sizes is None:
        raise ValueError("LLaVA-Next VisionZip requires pixel_values and image_sizes")
    vision_feature_layer = _config_value(
        model.config, ("vision_feature_layer",), default=-2
    )
    vision_feature_select_strategy = _config_value(
        model.config, ("vision_feature_select_strategy",), default="default"
    )
    if not isinstance(vision_feature_layer, int):
        raise ValueError("VisionZip requires a single integer vision_feature_layer")
    if vision_feature_select_strategy != "default":
        raise ValueError("VisionZip requires vision_feature_select_strategy='default'")

    image_num_patches = _llava_next_image_num_patches(image_sizes, model)
    pixel_values = _llava_next_flatten_pixel_values(
        pixel_values, image_num_patches
    )
    vision_tower = _llava_next_component(model, "vision_tower")
    outputs = vision_tower(
        pixel_values,
        output_hidden_states=True,
        output_attentions=True,
    )
    if outputs.attentions is None:
        raise RuntimeError("LLaVA-Next did not return attentions required by VisionZip")
    full_hidden = outputs.hidden_states[vision_feature_layer]
    cls_attention = outputs.attentions[vision_feature_layer][:, :, 0, 1:]
    key_metric = _visionzip_key_metric(
        vision_tower, outputs, vision_feature_layer
    )
    dominant_tokens = int(
        getattr(args, "visual_prune_visionzip_dominant_tokens", 54)
    )
    condensed, dominant_indices = baseline_pruning.visionzip_condense(
        full_hidden,
        cls_attention,
        key_metric,
        dominant_tokens=dominant_tokens,
        contextual_tokens=int(
            getattr(args, "visual_prune_visionzip_contextual_tokens", 10)
        ),
    )
    projected = _llava_next_component(model, "multi_modal_projector")(condensed)
    packed = _pack_llava_next_visionzip_condensed(
        model,
        projected,
        dominant_indices,
        image_num_patches,
        image_sizes,
        dominant_tokens,
    )
    return _scatter_condensed_visual_features(packed, original_visual_tokens)


def _pack_llava_next_specflow_inputs(
    model,
    projected_crops,
    crop_attention,
    image_num_patches,
    image_sizes,
):
    feature_splits = torch.split(projected_crops, image_num_patches, dim=0)
    attention_splits = torch.split(crop_attention, image_num_patches, dim=0)
    patch_side = (
        model.config.vision_config.image_size
        // model.config.vision_config.patch_size
    )
    packed = []
    for image_idx, (features, attention) in enumerate(
        zip(feature_splits, attention_splits)
    ):
        if features.shape[0] == 1:
            packed.append(
                (
                    features[0],
                    attention[0],
                    [(0, features.shape[1], patch_side, patch_side)],
                )
            )
            continue

        base_features = features[0]
        base_attention = attention[0]
        crop_features = features[1:]
        crop_attention_values = attention[1:]
        grid_height, grid_width = _llava_next_anyres_grid_shape(
            image_sizes[image_idx], model
        )
        if grid_height * grid_width != crop_features.shape[0]:
            raise ValueError("SpecFlow AnyRes crop grid does not match encoded crops")

        spatial = crop_features.view(
            grid_height,
            grid_width,
            patch_side,
            patch_side,
            features.shape[-1],
        ).permute(4, 0, 2, 1, 3).contiguous()
        spatial = spatial.flatten(1, 2).flatten(2, 3)
        spatial = _llava_next_unpad_image(spatial, image_sizes[image_idx])
        unpadded_height, unpadded_width = spatial.shape[1:]
        spatial = spatial.flatten(1, 2).transpose(0, 1)

        heads = crop_attention_values.shape[1]
        spatial_attention = crop_attention_values.view(
            grid_height,
            grid_width,
            heads,
            patch_side,
            patch_side,
        ).permute(2, 0, 3, 1, 4).contiguous()
        spatial_attention = spatial_attention.flatten(1, 2).flatten(2, 3)
        spatial_attention = _llava_next_unpad_image(
            spatial_attention, image_sizes[image_idx]
        ).flatten(1, 2)

        packed_features = torch.cat((base_features, spatial), dim=0)
        packed_attention = torch.cat(
            (base_attention, spatial_attention), dim=1
        )
        base_count = base_features.shape[0]
        segments = [
            (0, base_count, patch_side, patch_side),
            (
                base_count,
                base_count + unpadded_height * unpadded_width,
                unpadded_height,
                unpadded_width,
            ),
        ]
        packed.append((packed_features, packed_attention, segments))
    return packed


def _extract_llava_next_specflow_features(
    model, inputs, args, original_visual_tokens
):
    pixel_values = inputs.get("pixel_values")
    image_sizes = inputs.get("image_sizes")
    if pixel_values is None or image_sizes is None:
        raise ValueError("LLaVA-Next SpecFlow requires pixel_values and image_sizes")
    vision_feature_layer = _config_value(
        model.config, ("vision_feature_layer",), default=-2
    )
    vision_feature_select_strategy = _config_value(
        model.config, ("vision_feature_select_strategy",), default="default"
    )
    if not isinstance(vision_feature_layer, int):
        raise ValueError("SpecFlow requires a single integer vision_feature_layer")
    if vision_feature_select_strategy != "default":
        raise ValueError("SpecFlow requires vision_feature_select_strategy='default'")

    image_num_patches = _llava_next_image_num_patches(image_sizes, model)
    pixel_values = _llava_next_flatten_pixel_values(
        pixel_values, image_num_patches
    )
    vision_tower = _llava_next_component(model, "vision_tower")
    outputs = vision_tower(
        pixel_values,
        output_hidden_states=True,
        output_attentions=True,
    )
    if outputs.attentions is None:
        raise RuntimeError("LLaVA-Next did not return attentions required by SpecFlow")
    selected = outputs.hidden_states[vision_feature_layer][:, 1:]
    crop_attention = outputs.attentions[vision_feature_layer][:, :, 0, 1:]
    projected = _llava_next_component(model, "multi_modal_projector")(selected)
    packed_inputs = _pack_llava_next_specflow_inputs(
        model,
        projected,
        crop_attention,
        image_num_patches,
        image_sizes,
    )

    condensed_images = []
    for packed_features, packed_attention, segments in packed_inputs:
        batched_features = packed_features.unsqueeze(0)
        energy = baseline_pruning.specflow_heat_flow(
            packed_attention.unsqueeze(0),
            batched_features,
            use_best_heads=bool(
                getattr(args, "visual_prune_specflow_use_best_heads", True)
            ),
            n_best=int(
                getattr(args, "visual_prune_specflow_num_best_heads", 16)
            ),
            k_neighbors=int(
                getattr(args, "visual_prune_specflow_k_neighbors", 25)
            ),
            diffusion_steps=int(
                getattr(args, "visual_prune_specflow_diffusion_steps", 2)
            ),
            alpha=float(getattr(args, "visual_prune_specflow_alpha", 0.7)),
        )
        condensed_images.append(
            baseline_pruning.specflow_condense(
                batched_features,
                energy,
                token_budget=int(
                    getattr(args, "visual_prune_specflow_token_budget", 192)
                ),
                min_crop_size=int(
                    getattr(args, "visual_prune_specflow_min_crop_size", 4)
                ),
                adaptive_threshold=float(
                    getattr(
                        args,
                        "visual_prune_specflow_adaptive_threshold",
                        1e-4,
                    )
                ),
                use_coreset_sinks=bool(
                    getattr(
                        args,
                        "visual_prune_specflow_use_coreset_sinks",
                        True,
                    )
                ),
                n_sinks=int(
                    getattr(args, "visual_prune_specflow_num_sinks", 2)
                ),
                grid_shape=[segments],
            )[0]
        )
    return _scatter_condensed_visual_features(
        torch.cat(condensed_images, dim=0), original_visual_tokens
    )


def _extract_llava_ov_vispruner_features(model, inputs, args):
    raise NotImplementedError(
        "VisPruner for LLaVA-OneVision is reserved for a SigLIP-compatible "
        "visual-attention implementation and is not enabled yet"
    )


def _qwen2vl_rotate_half(hidden_states):
    first = hidden_states[..., : hidden_states.shape[-1] // 2]
    second = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _qwen2vl_apply_vision_rotary(query, key, position_embeddings):
    cos, sin = position_embeddings
    query_dtype = query.dtype
    key_dtype = key.dtype
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    query_float = query.float()
    key_float = key.float()
    query = (query_float * cos) + (_qwen2vl_rotate_half(query_float) * sin)
    key = (key_float * cos) + (_qwen2vl_rotate_half(key_float) * sin)
    return query.to(query_dtype), key.to(key_dtype)


def _qwen2vl_received_vision_attention(
    attention,
    hidden_states,
    cu_seqlens,
    position_embeddings,
    query_chunk_size=128,
    qkv_states=None,
):
    """Return per-head attention received by every Qwen vision patch.

    Qwen2-VL has no CLS token.  Its strict adapter therefore uses the mean
    incoming attention over all queries within each independent visual grid.
    Query chunking avoids allocating a full global visual attention matrix.
    """
    if not hasattr(attention, "qkv"):
        raise AttributeError("Qwen2-VL vision attention must expose qkv")
    num_heads = int(getattr(attention, "num_heads"))
    if qkv_states is None:
        if hidden_states is None:
            raise ValueError("hidden_states is required when qkv_states is not supplied")
        qkv_states = attention.qkv(hidden_states)
    token_count = qkv_states.shape[0]
    head_dim = qkv_states.shape[-1] // (3 * num_heads)
    query, key, _ = (
        qkv_states
        .reshape(token_count, 3, num_heads, head_dim)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    position_embeddings = tuple(
        value.to(device=qkv_states.device) for value in position_embeddings
    )
    query, key = _qwen2vl_apply_vision_rotary(
        query,
        key,
        position_embeddings,
    )

    received = torch.zeros(
        num_heads,
        token_count,
        dtype=torch.float32,
        device=qkv_states.device,
    )
    boundaries = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
    scale = head_dim**-0.5
    query_chunk_size = max(1, int(query_chunk_size))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        segment_key = key[start:end].float()
        segment_received = torch.zeros(
            num_heads,
            end - start,
            dtype=torch.float32,
            device=qkv_states.device,
        )
        for query_start in range(start, end, query_chunk_size):
            query_end = min(end, query_start + query_chunk_size)
            logits = torch.einsum(
                "qhd,khd->hqk",
                query[query_start:query_end].float(),
                segment_key,
            ) * scale
            segment_received += F.softmax(logits, dim=-1).sum(dim=1)
        received[:, start:end] = segment_received / float(end - start)
    return received


def _extract_qwen2vl_vispruner_features(model, inputs, args):
    pixel_values = inputs.get("pixel_values", None)
    image_grid_thw = inputs.get("image_grid_thw", None)
    if pixel_values is None or image_grid_thw is None:
        raise ValueError(
            "Qwen2-VL VisPruner requires pixel_values and image_grid_thw"
        )
    if inputs.get("pixel_values_videos", None) is not None:
        raise ValueError("Qwen2-VL VisPruner currently supports images, not videos")

    visual = _get_visual_module(model)
    if visual is None:
        raise AttributeError("Cannot locate Qwen2-VL visual module")
    required = ("patch_embed", "rot_pos_emb", "blocks", "merger")
    missing = [name for name in required if not hasattr(visual, name)]
    if missing:
        raise AttributeError(
            "Qwen2-VL visual module is missing required components: "
            + ", ".join(missing)
        )

    blocks = visual.blocks
    layer_idx = int(getattr(args, "visual_prune_qwen_vision_layer", -1))
    if layer_idx < 0:
        layer_idx += len(blocks)
    if layer_idx < 0 or layer_idx >= len(blocks):
        raise ValueError(
            "visual_prune_qwen_vision_layer is outside the Qwen2-VL vision "
            f"tower: {layer_idx} for {len(blocks)} blocks"
        )

    target_block = blocks[layer_idx]
    captured = {}

    def capture_attention_inputs(module, positional_args, keyword_args):
        cu_seqlens = keyword_args.get("cu_seqlens")
        position_embeddings = keyword_args.get("position_embeddings")
        if position_embeddings is None:
            rotary_pos_emb = keyword_args.get("rotary_pos_emb")
            if rotary_pos_emb is not None:
                rotary_embedding = torch.cat(
                    (rotary_pos_emb, rotary_pos_emb),
                    dim=-1,
                )
                position_embeddings = (
                    rotary_embedding.cos(),
                    rotary_embedding.sin(),
                )
        captured["cu_seqlens"] = (
            cu_seqlens.detach() if torch.is_tensor(cu_seqlens) else cu_seqlens
        )
        captured["position_embeddings"] = (
            tuple(value.detach() for value in position_embeddings)
            if position_embeddings is not None
            else None
        )

    def capture_qkv(module, positional_args, output):
        captured["qkv_states"] = output.detach()

    def capture_block_output(module, positional_args, output):
        captured["diversity_hidden_states"] = output.detach()

    handles = (
        target_block.attn.register_forward_pre_hook(
            capture_attention_inputs,
            with_kwargs=True,
        ),
        target_block.attn.qkv.register_forward_hook(capture_qkv),
        target_block.register_forward_hook(capture_block_output),
    )
    try:
        visual_dtype = (
            visual.get_dtype()
            if hasattr(visual, "get_dtype")
            else getattr(visual, "dtype", pixel_values.dtype)
        )
        projected_features = _normalize_image_features(
            visual(pixel_values.to(dtype=visual_dtype), grid_thw=image_grid_thw)
        )
    finally:
        for handle in handles:
            handle.remove()

    required_captures = (
        "cu_seqlens",
        "position_embeddings",
        "qkv_states",
        "diversity_hidden_states",
    )
    missing_captures = [
        name for name in required_captures if captured.get(name) is None
    ]
    if missing_captures:
        raise RuntimeError(
            "Failed to capture Qwen2-VL VisPruner visual cues: "
            + ", ".join(missing_captures)
        )
    diversity_hidden_states = captured["diversity_hidden_states"]
    patch_attentions = _qwen2vl_received_vision_attention(
        target_block.attn,
        None,
        captured["cu_seqlens"],
        captured["position_embeddings"],
        qkv_states=captured["qkv_states"],
    )

    merge_size = int(getattr(visual, "spatial_merge_size", 1))
    merge_unit = merge_size * merge_size
    patch_count = captured["qkv_states"].shape[0]
    if patch_count % merge_unit != 0:
        raise ValueError(
            "Qwen2-VL patch count must be divisible by spatial_merge_size^2: "
            f"{patch_count} vs {merge_unit}"
        )

    # Qwen orders each spatial_merge_size x spatial_merge_size patch group
    # contiguously before PatchMerger.  Aggregate the attention cue and keep
    # the complete pre-merger group as the diversity descriptor.
    merged_attentions = patch_attentions.reshape(
        patch_attentions.shape[0], -1, merge_unit
    ).mean(dim=-1)
    grouped_diversity_features = diversity_hidden_states.reshape(
        -1, merge_unit * diversity_hidden_states.shape[-1]
    )
    merged_attentions = merged_attentions.to(projected_features.device)
    grouped_diversity_features = grouped_diversity_features.to(
        projected_features.device
    )
    if bool(getattr(args, "visual_prune_vispruner_quant_vit_features", True)):
        grouped_diversity_features = fake_quantize_visual_features(
            grouped_diversity_features,
            args,
        )
    merged_counts = []
    for grid in image_grid_thw:
        temporal, height, width = [int(value) for value in grid.tolist()]
        raw_count = temporal * height * width
        if raw_count % merge_unit != 0:
            raise ValueError(
                "Each Qwen2-VL image grid must be divisible by "
                f"spatial_merge_size^2: {(temporal, height, width)}"
            )
        merged_counts.append(raw_count // merge_unit)
    if sum(merged_counts) != projected_features.shape[0]:
        raise ValueError(
            "Qwen2-VL merged grid count does not match projected features: "
            f"{sum(merged_counts)} vs {projected_features.shape[0]}"
        )

    keep_ratio = float(getattr(args, "visual_prune_keep_ratio", 1.0))
    min_keep_tokens = int(getattr(args, "visual_prune_min_keep_tokens", 1))
    important_ratio = float(getattr(args, "visual_prune_important_ratio", 0.5))
    feature_chunks = torch.split(projected_features, merged_counts, dim=0)
    attention_chunks = torch.split(merged_attentions, merged_counts, dim=1)
    diversity_chunks = torch.split(
        grouped_diversity_features,
        merged_counts,
        dim=0,
    )
    keep_masks = []
    for features, attentions, diversity_features in zip(
        feature_chunks,
        attention_chunks,
        diversity_chunks,
    ):
        keep_count = int(math.ceil(features.shape[0] * keep_ratio))
        keep_count = max(min_keep_tokens, keep_count)
        keep_count = min(features.shape[0], keep_count)
        keep_masks.append(
            select_vispruner_visual_tokens(
                features,
                attentions,
                keep_count=keep_count,
                important_ratio=important_ratio,
                diversity_features=diversity_features,
            )
        )
    return projected_features, torch.cat(keep_masks, dim=0)


def _extract_qwen2vl_crisp_features(model, inputs, args, keyword_embeddings):
    pixel_values = inputs.get("pixel_values", None)
    image_grid_thw = inputs.get("image_grid_thw", None)
    if pixel_values is None or image_grid_thw is None:
        raise ValueError("Qwen2-VL CRISP requires pixel_values and image_grid_thw")
    if inputs.get("pixel_values_videos", None) is not None:
        raise ValueError("Qwen2-VL CRISP currently supports images, not videos")

    visual = _get_visual_module(model)
    if visual is None:
        raise AttributeError("Cannot locate Qwen2-VL visual module")
    # Qwen2-VL's visual module contains both the ViT blocks and PatchMerger,
    # which is its multimodal projector.
    _apply_crisp_weight_fake_quant(visual, args, "Qwen2-VL ViT+projector")
    required = ("patch_embed", "rot_pos_emb", "blocks", "merger")
    missing = [name for name in required if not hasattr(visual, name)]
    if missing:
        raise AttributeError(
            "Qwen2-VL visual module is missing required components: "
            + ", ".join(missing)
        )

    blocks = visual.blocks
    layer_idx = int(getattr(args, "visual_prune_qwen_vision_layer", -1))
    if layer_idx < 0:
        layer_idx += len(blocks)
    if layer_idx < 0 or layer_idx >= len(blocks):
        raise ValueError(
            "visual_prune_qwen_vision_layer is outside the Qwen2-VL vision "
            f"tower: {layer_idx} for {len(blocks)} blocks"
        )

    target_block = blocks[layer_idx]
    captured = {}

    def capture_attention_inputs(module, positional_args, keyword_args):
        cu_seqlens = keyword_args.get("cu_seqlens")
        position_embeddings = keyword_args.get("position_embeddings")
        if position_embeddings is None:
            rotary_pos_emb = keyword_args.get("rotary_pos_emb")
            if rotary_pos_emb is not None:
                rotary_embedding = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                position_embeddings = (
                    rotary_embedding.cos(),
                    rotary_embedding.sin(),
                )
        captured["cu_seqlens"] = (
            cu_seqlens.detach() if torch.is_tensor(cu_seqlens) else cu_seqlens
        )
        captured["position_embeddings"] = (
            tuple(value.detach() for value in position_embeddings)
            if position_embeddings is not None
            else None
        )

    def capture_qkv(module, positional_args, output):
        captured["qkv_states"] = output.detach()

    handles = (
        target_block.attn.register_forward_pre_hook(
            capture_attention_inputs,
            with_kwargs=True,
        ),
        target_block.attn.qkv.register_forward_hook(capture_qkv),
    )
    try:
        visual_dtype = (
            visual.get_dtype()
            if hasattr(visual, "get_dtype")
            else getattr(visual, "dtype", pixel_values.dtype)
        )
        projected_features = _normalize_image_features(
            visual(pixel_values.to(dtype=visual_dtype), grid_thw=image_grid_thw)
        )
    finally:
        for handle in handles:
            handle.remove()

    missing_captures = [
        name
        for name in ("cu_seqlens", "position_embeddings", "qkv_states")
        if captured.get(name) is None
    ]
    if missing_captures:
        raise RuntimeError(
            "Failed to capture Qwen2-VL CRISP visual cues: "
            + ", ".join(missing_captures)
        )
    patch_attentions = _qwen2vl_received_vision_attention(
        target_block.attn,
        None,
        captured["cu_seqlens"],
        captured["position_embeddings"],
        qkv_states=captured["qkv_states"],
    )

    # Qwen2-VL has no CLS token. Mean incoming attention is mapped from each
    # pre-merger patch group to its projected token and used only as an
    # entropy-gated visual-centrality prior by the Qwen-specific selector.
    merge_size = int(getattr(visual, "spatial_merge_size", 1))
    merge_unit = merge_size * merge_size
    patch_count = captured["qkv_states"].shape[0]
    if patch_count % merge_unit != 0:
        raise ValueError(
            "Qwen2-VL patch count must be divisible by spatial_merge_size^2: "
            f"{patch_count} vs {merge_unit}"
        )
    merged_attentions = patch_attentions.reshape(
        patch_attentions.shape[0], -1, merge_unit
    ).mean(dim=-1).to(projected_features.device)

    merged_counts = []
    for grid in image_grid_thw:
        temporal, height, width = [int(value) for value in grid.tolist()]
        raw_count = temporal * height * width
        if raw_count % merge_unit != 0:
            raise ValueError(
                "Each Qwen2-VL image grid must be divisible by "
                f"spatial_merge_size^2: {(temporal, height, width)}"
            )
        merged_counts.append(raw_count // merge_unit)
    if sum(merged_counts) != projected_features.shape[0]:
        raise ValueError(
            "Qwen2-VL merged grid count does not match projected features: "
            f"{sum(merged_counts)} vs {projected_features.shape[0]}"
        )

    keep_ratio = float(getattr(args, "visual_prune_keep_ratio", 1.0))
    min_keep_tokens = int(getattr(args, "visual_prune_min_keep_tokens", 1))
    feature_chunks = torch.split(projected_features, merged_counts, dim=0)
    attention_chunks = torch.split(merged_attentions, merged_counts, dim=1)
    keep_masks = []
    for features, attentions in zip(feature_chunks, attention_chunks):
        keep_count = int(math.ceil(features.shape[0] * keep_ratio))
        keep_count = min(features.shape[0], max(min_keep_tokens, keep_count))
        keep_masks.append(
            select_qwen_crisp_visual_tokens(
                features,
                attentions,
                keyword_embeddings,
                keep_count,
                **_crisp_selector_kwargs(args),
                visual_prior_weight=float(
                    getattr(
                        args,
                        "visual_prune_qwen_crisp_prior_weight",
                        0.25,
                    )
                ),
            )
        )
    return projected_features, torch.cat(keep_masks, dim=0)


def extract_vispruner_image_features_and_mask(model, inputs, args):
    if type(model) == model_utils.LLAVA_HF:
        return _extract_llava_v15_vispruner_features(model, inputs, args)
    if type(model) == model_utils.LLAVA_NEXT_HF:
        return _extract_llava_next_vispruner_features(model, inputs, args)
    if type(model) == model_utils.LLAVA_OV_HF:
        return _extract_llava_ov_vispruner_features(model, inputs, args)
    if type(model) == model_utils.QWEN2_VL:
        return _extract_qwen2vl_vispruner_features(model, inputs, args)
    raise TypeError(f"VisPruner does not support model type {type(model)}")


def extract_crisp_image_features_and_mask(
    model,
    inputs,
    args,
    keyword_embeddings,
):
    if type(model) == model_utils.LLAVA_HF:
        return _extract_llava_v15_crisp_features(
            model, inputs, args, keyword_embeddings
        )
    if type(model) == model_utils.LLAVA_NEXT_HF:
        return _extract_llava_next_crisp_features(
            model, inputs, args, keyword_embeddings
        )
    if type(model) == model_utils.QWEN2_VL:
        return _extract_qwen2vl_crisp_features(
            model, inputs, args, keyword_embeddings
        )
    raise TypeError(f"CRISP does not support model type {type(model)}")


def extract_p4q_image_features_and_mask(
    model,
    inputs,
    args,
    keyword_embeddings,
):
    if type(model) == model_utils.LLAVA_HF:
        return _extract_llava_v15_p4q_features(
            model, inputs, args, keyword_embeddings
        )
    if type(model) == model_utils.LLAVA_NEXT_HF:
        return _extract_llava_next_p4q_features(
            model, inputs, args, keyword_embeddings
        )
    raise TypeError(
        "P4Q currently supports only LLaVA-1.5 and LLaVA-NeXT, got "
        f"{type(model)}"
    )


def extract_visionzip_image_features_and_mask(
    model, inputs, args, original_visual_tokens
):
    if type(model) == model_utils.LLAVA_HF:
        return _extract_llava_v15_visionzip_features(
            model, inputs, args, original_visual_tokens
        )
    if type(model) == model_utils.LLAVA_NEXT_HF:
        return _extract_llava_next_visionzip_features(
            model, inputs, args, original_visual_tokens
        )
    raise TypeError(
        "VisionZip baseline supports only LLaVA-1.5 and LLaVA-NeXT, got "
        f"{type(model)}"
    )


def extract_specflow_image_features_and_mask(
    model, inputs, args, original_visual_tokens
):
    if type(model) == model_utils.LLAVA_HF:
        return _extract_llava_v15_specflow_features(
            model, inputs, args, original_visual_tokens
        )
    if type(model) == model_utils.LLAVA_NEXT_HF:
        return _extract_llava_next_specflow_features(
            model, inputs, args, original_visual_tokens
        )
    raise TypeError(
        "SpecFlow baseline supports only LLaVA-1.5 and LLaVA-NeXT, got "
        f"{type(model)}"
    )


def _require_fp16_condensation_baseline(model, args, strategy):
    supported_types = (model_utils.LLAVA_HF, model_utils.LLAVA_NEXT_HF)
    if strategy == "SpecFlow":
        supported_types += (getattr(model_utils, "QWEN2_5_VL", None),)
    if type(model) not in supported_types:
        supported_names = "LLaVA-1.5 and LLaVA-NeXT"
        if strategy == "SpecFlow":
            supported_names += " and Qwen2.5-VL"
        raise TypeError(
            f"{strategy} supports only {supported_names}"
        )
    bit_fields = ("w_bits", "a_bits", "q_bits", "k_bits", "v_bits")
    invalid_bits = [
        name for name in bit_fields if int(getattr(args, name, 16)) != 16
    ]
    incompatible_flags = (
        "quant_vit",
        "flatquant",
        "scale_smooth",
        "smoothquant",
        "quarot",
        "offline_rotate",
        "online_rotate",
        "spinquant",
        "split_rotate",
        "split_r1_rotate",
        "parrot",
    )
    enabled_flags = [
        name for name in incompatible_flags if bool(getattr(args, name, False))
    ]
    if invalid_bits or enabled_flags:
        details = []
        if invalid_bits:
            details.append("16-bit fields required: " + ", ".join(invalid_bits))
        if enabled_flags:
            details.append("disabled transforms required: " + ", ".join(enabled_flags))
        raise ValueError(
            f"{strategy} is an FP16-only comparison baseline (" + "; ".join(details) + ")"
        )


def _extract_qwen2vl_image_features(model, inputs):
    pixel_values = inputs.get("pixel_values", None)
    image_grid_thw = inputs.get("image_grid_thw", None)
    if pixel_values is None or image_grid_thw is None:
        raise ValueError("Qwen2VL visual pruning requires pixel_values and image_grid_thw")
    if hasattr(model, "get_image_features"):
        return _normalize_image_features(model.get_image_features(pixel_values, image_grid_thw))
    visual = _get_visual_module(model)
    if visual is None:
        raise AttributeError("Cannot locate Qwen2VL visual module")
    dtype = getattr(visual, "dtype", pixel_values.dtype)
    return _normalize_image_features(visual(pixel_values.to(dtype), grid_thw=image_grid_thw))


def extract_projected_image_features(model, inputs):
    if type(model) == model_utils.LLAVA_HF:
        return _extract_llava_v15_image_features(model, inputs)
    if type(model) == model_utils.LLAVA_NEXT_HF:
        return _extract_llava_next_image_features(model, inputs)
    if type(model) == model_utils.QWEN2_VL:
        if inputs.get("pixel_values_videos", None) is not None:
            raise ValueError("visual token pruning currently supports image inputs, not video")
        return _extract_qwen2vl_image_features(model, inputs)
    raise TypeError(f"visual token pruning does not support model type {type(model)}")


def fake_quantize_visual_features(image_features, args):
    bits = getattr(args, "visual_prune_quant_bits", None)
    if bits is None:
        bits = getattr(args, "a_bits", 16)
    bits = int(bits)
    if bits >= 16:
        return image_features
    quantizer = quant_utils.ActQuantizer()
    quantizer.configure(
        bits=bits,
        groupsize=int(getattr(args, "visual_prune_quant_groupsize", -1)),
        sym=not bool(getattr(args, "visual_prune_quant_asym", False)),
        clip_ratio=float(getattr(args, "visual_prune_quant_clip_ratio", 1.0)),
    )
    dtype = image_features.dtype
    quantizer.find_params(image_features.detach())
    quantized = quantizer(image_features).to(dtype)
    quantizer.free()
    return quantized


def _compute_qwen_position_ids(model, inputs, attention_mask):
    explicit_position_ids = inputs.get("position_ids", None)
    if explicit_position_ids is not None:
        return explicit_position_ids

    rope_owner = model
    if not hasattr(rope_owner, "get_rope_index") and hasattr(model, "model"):
        rope_owner = model.model
    if hasattr(rope_owner, "get_rope_index"):
        try:
            position_ids, _ = rope_owner.get_rope_index(
                inputs.get("input_ids", None),
                inputs.get("image_grid_thw", None),
                inputs.get("video_grid_thw", None),
                attention_mask,
            )
            return position_ids
        except TypeError:
            logging.debug("Qwen2VL get_rope_index signature did not match pruning call")
    return None


def prepare_pruned_generation_inputs(
    model,
    inputs,
    args,
    query_text=None,
    tokenizer=None,
):
    if type(model) == getattr(model_utils, "QWEN2_5_VL", None):
        from vlmrot.qwen25_p4q import prepare_inputs
        return prepare_inputs(model, inputs, args, query_text, tokenizer)
    if not is_supported_model(model):
        raise TypeError(f"Unsupported visual pruning model type: {type(model)}")
    input_ids = inputs["input_ids"]
    if input_ids.shape[0] != 1:
        raise ValueError("visual token pruning currently expects eval batch size 1")
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    image_token_id = _config_value(
        model.config,
        ("image_token_id", "image_token_index"),
        default=None,
    )
    if image_token_id is None:
        raise ValueError("model.config.image_token_id is required for visual pruning")
    image_mask = input_ids == image_token_id
    num_image_tokens = int(image_mask.sum().item())

    strategy = strategy_from_args(args)
    if strategy == VISUAL_PRUNE_STRATEGY_VISPRUNER:
        if bool(getattr(args, "visual_prune_recycle", False)):
            raise ValueError("visual_prune_recycle is not part of strict VisPruner")
        image_features, visual_keep_mask = extract_vispruner_image_features_and_mask(
            model,
            inputs,
            args,
        )
    elif strategy == VISUAL_PRUNE_STRATEGY_CRISP:
        if bool(getattr(args, "visual_prune_recycle", False)):
            raise ValueError("visual_prune_recycle is not part of strict CRISP")
        if query_text is None:
            raise ValueError(
                "CRISP requires the raw user question via query_text for noun extraction"
            )
        keywords, crisp_language = extract_crisp_keywords(query_text, args)
        language_model = _get_language_model(model)
        keyword_embeddings = embed_crisp_keywords(
            model,
            tokenizer,
            keywords,
            language_model.embed_tokens.weight.device,
        )
        logging.debug(
            "CRISP extracted %d %s keywords: %s",
            len(keywords),
            crisp_language,
            keywords,
        )
        image_features, visual_keep_mask = extract_crisp_image_features_and_mask(
            model,
            inputs,
            args,
            keyword_embeddings,
        )
    elif strategy == VISUAL_PRUNE_STRATEGY_P4Q:
        if type(model) not in (model_utils.LLAVA_HF, model_utils.LLAVA_NEXT_HF):
            raise TypeError(
                "P4Q currently supports only LLaVA-1.5 and LLaVA-NeXT, got "
                f"{type(model)}"
            )
        if bool(getattr(args, "visual_prune_recycle", False)):
            raise ValueError("visual_prune_recycle is not part of P4Q")
        if query_text is None:
            raise ValueError(
                "P4Q requires the raw user question via query_text for noun extraction"
            )
        keywords, p4q_language = extract_crisp_keywords(query_text, args)
        language_model = _get_language_model(model)
        keyword_embeddings = embed_crisp_keywords(
            model,
            tokenizer,
            keywords,
            language_model.embed_tokens.weight.device,
            **({'batched': True, 'verify_exact': bool(getattr(args, 'p4q_verify_exact', False))}
               if getattr(args, 'p4q_batch_keywords', False) else {}),
        )
        logging.debug(
            "P4Q extracted %d %s keywords: %s",
            len(keywords),
            p4q_language,
            keywords,
        )
        image_features, visual_keep_mask = extract_p4q_image_features_and_mask(
            model,
            inputs,
            args,
            keyword_embeddings,
        )
    elif strategy == VISUAL_PRUNE_STRATEGY_VISIONZIP:
        if bool(getattr(args, "visual_prune_recycle", False)):
            raise ValueError("visual_prune_recycle is not part of VisionZip")
        _require_fp16_condensation_baseline(model, args, "VisionZip")
        image_features, visual_keep_mask = (
            extract_visionzip_image_features_and_mask(
                model, inputs, args, num_image_tokens
            )
        )
    elif strategy == VISUAL_PRUNE_STRATEGY_SPECFLOW:
        if bool(getattr(args, "visual_prune_recycle", False)):
            raise ValueError("visual_prune_recycle is not part of SpecFlow")
        _require_fp16_condensation_baseline(model, args, "SpecFlow")
        image_features, visual_keep_mask = extract_specflow_image_features_and_mask(
            model, inputs, args, num_image_tokens
        )
    else:
        image_features = extract_projected_image_features(model, inputs)
        image_features = fake_quantize_visual_features(image_features, args)
        visual_keep_mask = None
    if image_features.shape[0] != num_image_tokens:
        raise ValueError(
            "projected image feature count does not match image token count: "
            f"{image_features.shape[0]} vs {num_image_tokens}"
        )

    language_model = _get_language_model(model)
    inputs_embeds = language_model.embed_tokens(input_ids).to(image_features.device)
    image_mask = image_mask.to(inputs_embeds.device)
    attention_mask = attention_mask.to(inputs_embeds.device)
    inputs_embeds[image_mask] = image_features.to(inputs_embeds.dtype)

    if visual_keep_mask is None:
        text_mask = attention_mask.bool() & (~image_mask)
        text_embeds = inputs_embeds[text_mask]
        visual_keep_mask, _ = select_topk_visual_tokens(
            image_features,
            text_embeds,
            keep_ratio=float(getattr(args, "visual_prune_keep_ratio", 1.0)),
            activation_weight=float(getattr(args, "visual_prune_score_activation_weight", 0.5)),
            similarity_weight=float(getattr(args, "visual_prune_score_similarity_weight", 0.5)),
            min_keep_tokens=int(getattr(args, "visual_prune_min_keep_tokens", 1)),
        )
    recycled_visual_tokens = 0
    if bool(getattr(args, "visual_prune_recycle", False)):
        recycled_visual_tokens = int((~visual_keep_mask).sum().item())
        image_features = recycle_dropped_visual_tokens(
            image_features,
            visual_keep_mask,
            args,
        )
        visual_positions = image_mask[0].nonzero(as_tuple=False).flatten()
        inputs_embeds[0, visual_positions, :] = image_features.to(inputs_embeds.dtype)

    position_ids = None
    if type(model) == model_utils.QWEN2_VL:
        position_ids = _compute_qwen_position_ids(model, inputs, attention_mask)
        if position_ids is not None:
            position_ids = position_ids.to(inputs_embeds.device)

    pruned_embeds, pruned_attention_mask, pruned_position_ids = prune_prompt_embeddings(
        inputs_embeds,
        attention_mask,
        image_mask,
        visual_keep_mask,
        position_ids=position_ids,
    )

    prepared = {
        "inputs_embeds": pruned_embeds,
        "attention_mask": pruned_attention_mask,
    }
    if pruned_position_ids is not None:
        prepared["position_ids"] = pruned_position_ids
    stats = VisualPruningStats(
        original_visual_tokens=num_image_tokens,
        kept_visual_tokens=int(visual_keep_mask.sum().item()),
        original_prompt_tokens=int(attention_mask.sum().item()),
        kept_prompt_tokens=int(pruned_attention_mask.sum().item()),
        recycled_visual_tokens=recycled_visual_tokens,
    )
    return prepared, stats


def prepare_dense_generation_inputs(model, inputs):
    """Build the unpruned multimodal prompt used by the comparable runner.

    This mirrors the embedding construction in ``prepare_pruned_generation_inputs``
    without running a selector or dropping tokens.  Keeping preparation separate
    from decoding lets dense and pruned evaluation share exactly the same greedy
    generation loop while the legacy ``model.generate`` path remains available.
    """
    if type(model) == getattr(model_utils, "QWEN2_5_VL", None):
        from vlmrot.qwen25_p4q import prepare_inputs
        return prepare_inputs(model, inputs)
    if not is_supported_model(model):
        raise TypeError(f"Unsupported dense generation model type: {type(model)}")
    input_ids = inputs["input_ids"]
    if input_ids.shape[0] != 1:
        raise ValueError("comparable generation currently expects eval batch size 1")
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    image_token_id = _config_value(
        model.config,
        ("image_token_id", "image_token_index"),
        default=None,
    )
    if image_token_id is None:
        raise ValueError("model.config.image_token_id is required for dense generation")
    image_mask = input_ids == image_token_id
    image_features = extract_projected_image_features(model, inputs)
    num_image_tokens = int(image_mask.sum().item())
    if image_features.shape[0] != num_image_tokens:
        raise ValueError(
            "projected image feature count does not match image token count: "
            f"{image_features.shape[0]} vs {num_image_tokens}"
        )

    language_model = _get_language_model(model)
    inputs_embeds = language_model.embed_tokens(input_ids).to(image_features.device)
    image_mask = image_mask.to(inputs_embeds.device)
    attention_mask = attention_mask.to(inputs_embeds.device)
    inputs_embeds[image_mask] = image_features.to(inputs_embeds.dtype)

    prepared = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }
    if type(model) == model_utils.QWEN2_VL:
        position_ids = _compute_qwen_position_ids(model, inputs, attention_mask)
        if position_ids is not None:
            prepared["position_ids"] = position_ids.to(inputs_embeds.device)

    prompt_tokens = int(attention_mask.sum().item())
    stats = VisualPruningStats(
        original_visual_tokens=num_image_tokens,
        kept_visual_tokens=num_image_tokens,
        original_prompt_tokens=prompt_tokens,
        kept_prompt_tokens=prompt_tokens,
    )
    return prepared, stats


def _get_past_key_values(outputs):
    if hasattr(outputs, "past_key_values"):
        return outputs.past_key_values
    return outputs[1] if len(outputs) > 1 else None


def _next_generated_text_position_ids(position_ids):
    """Return the first cached-decode position after a prepared prompt.

    Qwen2-VL represents prompt positions on temporal, height, and width axes,
    but generated text must advance from the largest prompt coordinate with
    the same scalar position on all three axes.  Taking the global maximum is
    also correct for an ordinary 2-D text position tensor and remains robust
    when a multimodal prompt ends immediately after a visual span.
    """
    if position_ids is None:
        return None
    if position_ids.dim() == 3:
        next_positions = position_ids.amax(dim=(0, 2), keepdim=False) + 1
        return next_positions.view(1, -1, 1).expand(
            position_ids.shape[0],
            -1,
            -1,
        )
    if position_ids.dim() == 2:
        return position_ids.amax(dim=-1, keepdim=True) + 1
    raise ValueError(
        "prepared position_ids must be [batch, sequence] or "
        f"[axes, batch, sequence], got {position_ids.shape}"
    )


def greedy_generate_from_prepared_inputs(
    model,
    prepared,
    max_new_tokens=16,
    eos_token_id=None,
    pad_token_id=None,
):
    """Greedy decode from prepared embeddings for dense/pruned comparisons."""
    generated = []
    past_key_values = None
    attention_mask = prepared["attention_mask"]
    position_ids = prepared.get("position_ids", None)
    next_position_ids = _next_generated_text_position_ids(position_ids)

    eos_ids = eos_token_id
    if eos_ids is None:
        eos_ids = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_ids = set(eos_ids or [])

    next_input_ids = None
    for step in range(int(max_new_tokens)):
        if step == 0:
            model_inputs = dict(prepared)
        else:
            model_inputs = {
                "input_ids": next_input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
            if next_position_ids is not None:
                # Qwen2-VL's three M-RoPE axes are identical for generated
                # text.  Supplying them explicitly avoids relying on the
                # unpruned prompt's rope_deltas after an inputs_embeds prefill.
                model_inputs["position_ids"] = next_position_ids

        outputs = model(
            **model_inputs,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_token)
        past_key_values = _get_past_key_values(outputs)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    attention_mask.shape[0],
                    1,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )
        next_input_ids = next_token
        # The token sampled by the prompt prefill has not entered the cache
        # yet.  Keep ``prompt_max + 1`` for the first cached token, then advance
        # only after a cached-decode step has consumed its input token.
        if next_position_ids is not None and step > 0:
            next_position_ids = next_position_ids + 1
        if eos_ids and all(int(token) in eos_ids for token in next_token.flatten().tolist()):
            break
        if pad_token_id is not None and all(
            int(token) == int(pad_token_id) for token in next_token.flatten().tolist()
        ):
            break

    if not generated:
        return torch.empty(1, 0, dtype=torch.long, device=attention_mask.device)
    return torch.cat(generated, dim=1)


def generate_without_visual_pruning(
    model,
    inputs,
    max_new_tokens=16,
    eos_token_id=None,
    pad_token_id=None,
):
    """Run the dense prompt through the same decoder loop as pruning."""
    prepared, stats = prepare_dense_generation_inputs(model, inputs)
    generated = greedy_generate_from_prepared_inputs(
        model,
        prepared,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    return generated, stats


def generate_with_visual_pruning(
    model,
    inputs,
    args,
    max_new_tokens=16,
    eos_token_id=None,
    pad_token_id=None,
    query_text=None,
    tokenizer=None,
):
    prepared, stats = prepare_pruned_generation_inputs(
        model,
        inputs,
        args,
        query_text=query_text,
        tokenizer=tokenizer,
    )
    generated = greedy_generate_from_prepared_inputs(
        model,
        prepared,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    return generated, stats
