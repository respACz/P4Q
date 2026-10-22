"""Validation for the released LLaVA FlatQuant W4A4 + native P4Q workflows."""
import math
from pathlib import Path


SUPPORTED_MODELS = (
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-v1.6-vicuna-7b-hf",
)


def validate_p4q_args(args):
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    require(args.model in SUPPORTED_MODELS, "P4Q supports LLaVA-1.5-7B and LLaVA-NeXT-7B only")
    require(args.flatquant_mode in ("train", "eval"), "flatquant_mode must be train or eval; run calibration and evaluation separately")
    require(args.visual_prune and args.visual_prune_strategy == "p4q", "this workflow requires visual_prune=true and visual_prune_strategy=p4q")
    require(not args.visual_prune_recycle, "P4Q does not support visual_prune_recycle")
    require(args.w_bits == 4 and args.a_bits == 4, "this workflow requires w_bits=4 and a_bits=4")
    require(all(getattr(args, name) == 16 for name in ("q_bits", "k_bits", "v_bits")), "this workflow requires q_bits=k_bits=v_bits=16")
    require(args.w_groupsize == -1 and args.a_groupsize == -1, "this workflow requires w_groupsize=a_groupsize=-1")
    require(args.quant_method == "rtn", "FlatQuant evaluation requires quant_method=rtn")
    require(not args.int8_down_proj, "W4A4 requires int8_down_proj=false")
    require(args.device_map in ("auto", "balanced", "balanced_low_0", "sequential", "none"), "invalid device_map")
    require(args.diag_init in ("sq_style", "one_style"), "diag_init must be sq_style or one_style")
    require(args.visual_prune_crisp_language in ("auto", "en", "zh"), "visual_prune_crisp_language must be auto, en or zh")

    for name in ("visual_prune_keep_ratio", "visual_prune_p4q_beta"):
        value = float(getattr(args, name))
        require(math.isfinite(value) and 0 < value <= 1, f"{name} must be in (0, 1]")
    value = float(args.visual_prune_p4q_lambda)
    require(math.isfinite(value) and 0 <= value <= 1, "visual_prune_p4q_lambda must be in [0, 1]")
    require(args.visual_prune_min_keep_tokens >= 1, "visual_prune_min_keep_tokens must be >= 1")
    require(args.visual_prune_log_every >= 1, "visual_prune_log_every must be >= 1")

    # These flags belong to other experiments. Their omitted defaults must not
    # activate TLQ, rotation or analysis requirements for the released configs.
    disabled = ("scale_smooth", "smoothquant", "quant_vit", "spinquant",
                "offline_rotate", "online_rotate", "split_rotate",
                "split_r1_rotate", "parrot", "quarot", "profile",
                "error_check", "split_error_sweep")
    for name in disabled:
        require(not getattr(args, name, False), f"{name} is not supported in the FlatQuant + P4Q workflow")
    require(not getattr(args, "real_quant_export_dir", None) and not getattr(args, "real_quant_export_only", False), "real quant export is not part of this calibration/evaluation workflow")

    model_path = args.model_path
    require(bool(model_path) and (model_path == args.model or Path(model_path).is_dir()), "model_path must be an existing model directory or the same Hugging Face ID as model")

    if isinstance(args.eval_dataset, str):
        args.eval_dataset = [args.eval_dataset]
    require(isinstance(args.eval_dataset, list), "eval_dataset must be a list or dataset name")

    if args.flatquant_mode == "train":
        require(args.flatquant_prune_calibration, "P4Q calibration requires flatquant_prune_calibration=true")
        require(args.cal_dataset == "ScienceQA_Train", "P4Q calibration requires cal_dataset=ScienceQA_Train")
        require(bool(args.question_file) and Path(args.question_file).is_file(), "question_file must point to the ScienceQA calibration JSON")
        require(bool(args.image_base) and Path(args.image_base).is_dir(), "image_base must point to the ScienceQA image directory")
        for name in ("epochs", "cali_bsz", "nsamples"):
            require(getattr(args, name) >= 1, f"{name} must be >= 1")
        require(math.isfinite(args.flat_lr) and args.flat_lr > 0, "flat_lr must be positive")
        require(math.isfinite(args.diag_alpha) and 0 <= args.diag_alpha <= 1, "diag_alpha must be in [0, 1]")
        require(any((args.cali_trans, args.add_diag, args.lwc, args.lac)), "enable at least one FlatQuant trainable option")
        require(args.flatquant_fold_check_samples >= 0, "flatquant_fold_check_samples must be >= 0")
        require(math.isfinite(args.flatquant_fold_check_tol) and args.flatquant_fold_check_tol >= 0, "flatquant_fold_check_tol must be nonnegative")
    elif args.flatquant_mode == "eval":
        path = Path(args.flatquant_matrix_path) if args.flatquant_matrix_path else None
        if path is not None and path.suffix != ".pth":
            path = path / "flat_matrices.pth"
        require(path is not None and path.is_file(), "flatquant_matrix_path must point to an existing flat_matrices.pth; run calibration first")
        require(bool(args.eval_dataset), "evaluation requires a nonempty eval_dataset")
        # No calibration data or TLQ caption files are needed to load matrices.

    require(args.eval_generation_mode in ("legacy", "comparable"), "invalid eval_generation_mode")
    require(args.eval_timing_warmup_samples >= 0, "eval_timing_warmup_samples must be >= 0")
    if args.eval_max_samples is not None:
        require(args.eval_max_samples >= 1 and args.eval_inference_only, "eval_max_samples requires a positive count and eval_inference_only=true")

    if errors:
        raise ValueError("Argument validation failed:\n- " + "\n- ".join(errors))
