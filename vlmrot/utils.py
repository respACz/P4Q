import argparse, os, yaml, logging, pprint, random, gc
import torch
import torch.nn as nn
import torch.distributed as dist
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.hooks import remove_hook_from_submodules
import numpy as np
from vlmrot import model_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


def resolve_project_path(path):
    """Resolve a user path against the VLMRot_PQ project root.

    Configuration files live under ``configs/``, but cache paths in those
    files are intentionally written from the repository root.  Resolving here
    also makes their meaning independent of the shell's current directory.
    """
    if path is None:
        return None
    expanded_path = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(expanded_path):
        return os.path.normpath(expanded_path)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(project_root, expanded_path))

supported_models = [
            # 'liuhaotian/llava-v1.5-7b',
            # 'liuhaotian/llava-v1.5-13b',
            # 'liuhaotian/llava-v1.6-vicuna-7b',
            # 'liuhaotian/llava-v1.6-vicuna-13b',
            'llava-hf/llava-1.5-7b-hf',
            'llava-hf/llava-v1.6-vicuna-7b-hf',
            'llava-hf/llava-v1.6-vicuna-13b-hf',
            'llava-hf/llava-onevision-qwen2-7b-ov-hf',
            'llava-hf/llava-onevision-qwen2-0.5b-ov-hf',
            'Qwen/Qwen2.5-VL-7B-Instruct',
            'Qwen/Qwen2-VL-7B-Instruct',
            'Qwen/Qwen2-VL-2B-Instruct'
            ]

supported_cal_datasets = [
    'ScienceQA_Train',
    'COCO_ShareGPT4V',
]

supported_eval_datasets = [
    'ScienceQA_TEST',
    'MMMU_DEV_VAL',
    'OCRBench',
    'VizWiz',
    'TextVQA_VAL',
    'ChartQA_TEST',
    'SEEDBench2_Plus',
    'MME',
    'MMMB_en',
    'MMMB_cn',
    'GQA_TestDev_Balanced',
    'MMBench_dev_cn',
    'POPE',
    'SEEDBench_IMG',
]
    # # For lmms-eval
    # 'mmmu_val', 
    # 'seedbench', 
    # 'ocrbench', 
    # 'vizwiz_vqa_val', 
    # 'scienceqa', 
    # 'textvqa_val', 
    # 'docvqa_val', 
    # 'seedbench_2_plus', 
    # 'chartqa', 
    # 'mme', 
    # 'mme_realworld', 
    # 'mme_realworld_cn'

# Unified parameter choices configuration (single source of truth)
PARAM_CHOICES = {
    'run_mode': ['single_card', 'data_parallel', 'model_distribute'],
    'device_map': ['auto', 'balanced', 'balanced_low_0', 'sequential', 'none'],
    'model': supported_models,
    'quant_method': ['rtn', 'gptq'],
    'cal_dataset': supported_cal_datasets,
    'eval_dataset': supported_eval_datasets
}

# Parameters that accept multiple values and need list conversion
MULTI_VALUE_PARAMS = {'eval_dataset'}

def parser_gen():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--run_mode', type=str, default='single_card', help='GPU situations decide the run mode',
                         choices=PARAM_CHOICES['run_mode'])
    parser.add_argument('--config', default=None, type=str, help='Path to the config file.')

    # model
    parser.add_argument('--model', type=str, default='liuhaotian/llava-v1.5-7b',
                        help='Model to load', choices=PARAM_CHOICES['model'])
    parser.add_argument('--model_path', type=str, default=None, help='Path to the selected model weights.')
    parser.add_argument('--device_map', type=str, default='auto',
                        choices=PARAM_CHOICES['device_map'],
                        help='HuggingFace device_map used when loading the model.')
    parser.add_argument('--max_memory_per_gpu', type=str, default=None,
                        help='Optional HF max_memory value applied to each visible GPU, e.g. 22GiB.')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')

    # calibration
    parser.add_argument('--quant_vit', action=argparse.BooleanOptionalAction, default=False, help='Quant the vit encoder or not')
    parser.add_argument('--scale_smooth', action=argparse.BooleanOptionalAction, default=False, help='Smoothquant-like scale smooth')
    parser.add_argument('--smooth_min_free_gpu_gb', type=float, default=1.0,
                        help='Minimum free memory required on each smooth-scale model GPU before backward.')
    parser.add_argument('--smooth_gradient_checkpointing', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable gradient checkpointing during smooth-scale backward statistics.')
    parser.add_argument('--smooth_log_gpu_memory', action=argparse.BooleanOptionalAction, default=True,
                        help='Log GPU memory around smooth-scale backward samples.')
    parser.add_argument('--smoothquant', type=bool, default=False, help='Use smoothquant to scale or not')
    parser.add_argument('--spinquant', type=bool, default=False, help='load the trained R1 or not')
    # FlatQuant is an isolated decoder-quantization workflow.  These defaults
    # mirror the upstream FlatQuant contract while remaining disabled unless a
    # FlatQuant config explicitly enables it.
    parser.add_argument('--flatquant', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable the dedicated FlatQuant workflow.')
    parser.add_argument('--flatquant_prune_calibration', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply the configured runtime visual pruning once before capturing FlatQuant calibration inputs.')
    parser.add_argument('--flatquant_mode', type=str, default='train',
                        choices=('train', 'eval', 'train_eval'),
                        help='Train matrices, evaluate an existing checkpoint, or train then evaluate.')
    parser.add_argument('--flatquant_matrix_path', type=str, default=None,
                        help='Path to flat_matrices.pth, or its containing directory; relative paths resolve from the project root.')
    parser.add_argument('--save_flatquant_dir', type=str, default='cache/flatquant',
                        help='Root directory used to save FlatQuant matrices.')
    parser.add_argument('--real_quant_export_dir', type=str, default=None,
                        help='Optional directory for a packed real-W4 language checkpoint consumed by flatquant_speed.')
    parser.add_argument('--real_quant_export_only', action=argparse.BooleanOptionalAction, default=False,
                        help='Exit after exporting the packed real-W4 checkpoint without running accuracy datasets.')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of per-layer FlatQuant calibration epochs.')
    parser.add_argument('--cali_bsz', type=int, default=1,
                        help='FlatQuant gradient accumulation batch size.')
    parser.add_argument('--flat_lr', type=float, default=0.005,
                        help='FlatQuant transform learning rate.')
    parser.add_argument('--cali_trans', action=argparse.BooleanOptionalAction, default=False,
                        help='Optimize FlatQuant transform matrices.')
    parser.add_argument('--add_diag', action=argparse.BooleanOptionalAction, default=False,
                        help='Add and optimize per-channel diagonal scales.')
    parser.add_argument('--lwc', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable learnable weight clipping for FlatQuant.')
    parser.add_argument('--lac', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable learnable activation clipping for FlatQuant.')
    parser.add_argument('--diag_init', type=str, default='sq_style',
                        choices=('sq_style', 'one_style'),
                        help='FlatQuant diagonal-scale initialization strategy.')
    parser.add_argument('--diag_alpha', type=float, default=0.3,
                        help='SmoothQuant-style diagonal initialization exponent.')
    parser.add_argument('--direct_inv', action=argparse.BooleanOptionalAction, default=False,
                        help='Use directly inverted transforms instead of SVD transforms.')
    parser.add_argument('--separate_vtrans', action=argparse.BooleanOptionalAction, default=False,
                        help='Keep the V-cache transform separate from v_proj.')
    parser.add_argument('--flatquant_fold_check_samples', type=int, default=1,
                        help='Samples used to compare online and folded FlatQuant transforms.')
    parser.add_argument('--flatquant_fold_check_tol', type=float, default=0.001,
                        help='Relative-L2 warning tolerance for FlatQuant folding.')
    parser.add_argument('--offline_rotate', type=bool, default=False, help='Rotate the model without online hadmard.')
    parser.add_argument('--online_rotate', type=bool, default=False,
                        help='Add online hadmard for o, v, and down_proj linear')
    parser.add_argument('--split_rotate', action=argparse.BooleanOptionalAction, default=False,
                        help='Split OV rotation channels by gradient importance and fuse the split rotation into weights')
    parser.add_argument('--split_r1_rotate', action=argparse.BooleanOptionalAction, default=False,
                        help='Use a shared whole-hidden split-R1 rotation when offline_rotate is enabled')
    parser.add_argument('--r1_split_topk_percent', type=float, default=10.0,
                        help='Percentage of hidden channels assigned to the important split-R1 group')
    parser.add_argument('--r1_split_importance_reduce', type=str, default='p95',
                        choices=('mean', 'p95'),
                        help='How to aggregate per-layer split-R1 importance into one shared hidden importance')
    parser.add_argument('--r1_split_grad_cache_dir', type=str, default=None,
                        help='Dir to cache shared split-R1 activation-gradient statistics')
    parser.add_argument('--online_channel_split_grad_cache_dir', type=str, default=None,
                        help='Dir to cache gradient statistics for split online rotation')
    parser.add_argument('--online_channel_split_topk', type=int, default=None,
                        help='Absolute number of important channels per split group for split online rotation')
    parser.add_argument('--online_channel_split_topk_percent', type=float, default=10.0,
                        help='Percentage of important channels per split group for split online rotation')
    parser.add_argument('--online_channel_split_topk_percent_sweep', type=str, default=None,
                        help='Comma-separated topk_percent values for split_error_sweep, e.g. 2,5,10,20')
    parser.add_argument('--online_channel_split_topk_percent_cache', type=str, default=None,
                        help='Path to per-layer topk_percent JSON cache written by split_error_sweep')
    parser.add_argument('--online_channel_split_topp_percent', type=float, default=None,
                        help='(0, 100] Cumulative gradient percentage for adaptive important channel selection in split rotation')
    parser.add_argument('--parrot', type=bool, default=False)
    parser.add_argument('--parrot_mode', type=str, default='activation', choices=('activation', 'gradient'))
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply Hadamard rotation in FP32 (default: False)')
    parser.add_argument('--int8_down_proj', action=argparse.BooleanOptionalAction, default=False,
                        help='Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8')
    parser.add_argument('--w_bits', type=int, default=16,
                        help='Number of bits for weights of the Linear layers')
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)''')
    parser.add_argument('--q_bits', type=int, default=16,
                        help='Number of bits for query-cache quantization.')
    parser.add_argument('--k_bits', type=int, default=16, help='''Number of bits for K-cache quantization. 
                        Note that quantizing the K-cache needs another rotation for the keys/queries''')
    parser.add_argument('--v_bits', type=int, default=16, help='''Number of bits for V-cache quantization. 
                        Note that quantizing the V-cache does not need any other rotation''')
    parser.add_argument('--visual_prune', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable optional visual-token pruning at evaluation runtime.')
    parser.add_argument('--visual_prune_strategy', type=str, default='activation_similarity',
                        choices=('activation_similarity', 'vispruner', 'crisp', 'p4q', 'visionzip', 'specflow'),
                        help='Visual token pruning strategy. VisionZip and SpecFlow are FP16-only comparison baselines for LLaVA-1.5/NeXT.')
    parser.add_argument('--visual_prune_keep_ratio', type=float, default=1.0,
                        help='Ratio of visual tokens kept by runtime top-k pruning. Must be in (0, 1].')
    parser.add_argument('--visual_prune_important_ratio', type=float, default=0.5,
                        help='For vispruner, fraction of retained patch tokens selected by ViT attention before diversity selection.')
    parser.add_argument('--visual_prune_vispruner_quant_vit_features', action=argparse.BooleanOptionalAction, default=True,
                        help='For vispruner, fake-quantize selected ViT hidden features before cosine diversity selection. Attention scores are not quantized.')
    parser.add_argument('--visual_prune_qwen_vision_layer', type=int, default=-1,
                        help='Qwen2-VL vision-block index used for VisPruner/CRISP incoming-attention cues; negative indices count from the end.')
    parser.add_argument('--visual_prune_qwen_crisp_prior_weight', type=float, default=0.25,
                        help='Qwen2-VL CRISP weight for the entropy-gated incoming-attention centrality prior; noun--visual relevance remains primary.')
    parser.add_argument('--visual_prune_crisp_alpha', type=float, default=0.3,
                        help='CRISP critical-token fraction for keep budgets at or above the low-budget threshold.')
    parser.add_argument('--visual_prune_crisp_low_budget_alpha', type=float, default=0.5,
                        help='CRISP critical-token fraction when the retained-token budget is below the configured threshold.')
    parser.add_argument('--visual_prune_crisp_low_budget_threshold', type=int, default=100,
                        help='CRISP retained-token threshold below which low_budget_alpha is used.')
    parser.add_argument('--visual_prune_crisp_quantize_weights', action=argparse.BooleanOptionalAction, default=False,
                        help='Run CRISP with per-output-channel RTN fake-quantized ViT and projector Linear weights. Uses w_bits/w_asym/w_clip and requires w_groupsize=-1.')
    parser.add_argument('--visual_prune_crisp_language', type=str, default='auto',
                        choices=('auto', 'en', 'zh'),
                        help='spaCy language used for CRISP noun extraction; auto selects zh when the question contains Han characters.')
    parser.add_argument('--visual_prune_crisp_en_model', type=str, default='en_core_web_sm',
                        help='spaCy English pipeline used by CRISP.')
    parser.add_argument('--visual_prune_crisp_zh_model', type=str, default='zh_core_web_sm',
                        help='spaCy Chinese pipeline used by CRISP.')
    parser.add_argument('--visual_prune_crisp_allow_heuristic_fallback', action=argparse.BooleanOptionalAction, default=False,
                        help='Allow a non-paper heuristic keyword fallback if spaCy or its language model is unavailable.')
    parser.add_argument('--visual_prune_p4q_beta', type=float, default=0.5,
                        help='P4Q anchor-budget fraction. The remaining budget is assigned to max-min diversity.')
    parser.add_argument('--visual_prune_p4q_lambda', type=float, default=0.6,
                        help='P4Q text share within the anchor budget; the remainder uses ViT CLS-attention saliency.')
    parser.add_argument('--visual_prune_p4q_feature_bits', type=int, default=4)
    parser.add_argument('--visual_prune_p4q_feature_groupsize', type=int, default=-1)
    parser.add_argument('--visual_prune_p4q_feature_asym', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--visual_prune_p4q_feature_clip_ratio', type=float, default=0.9)
    parser.add_argument('--visual_prune_p4q_weight_bits', type=int, default=4)
    parser.add_argument('--visual_prune_p4q_weight_groupsize', type=int, default=-1,
                        help='P4Q visual Linear RTN currently supports per-output-channel quantization only (-1).')
    parser.add_argument('--visual_prune_p4q_weight_asym', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--visual_prune_p4q_weight_clip', action=argparse.BooleanOptionalAction, default=True,
                        help='Use MSE clipping search for P4Q visual Linear weights.')
    parser.add_argument('--visual_prune_visionzip_dominant_tokens', type=int, default=54,
                        help='VisionZip dominant-token budget including CLS (paper/reference default: 54).')
    parser.add_argument('--visual_prune_visionzip_contextual_tokens', type=int, default=10,
                        help='VisionZip contextual merge-token budget per CLIP crop (paper/reference default: 10).')
    parser.add_argument('--visual_prune_specflow_token_budget', type=int, default=192,
                        help='SpecFlow final token budget per image (reference default: 192).')
    parser.add_argument('--visual_prune_specflow_budget_mode', choices=('fixed', 'ratio'), default='fixed',
                        help='SpecFlow budget mode; ratio is supported only for Qwen2.5-VL dynamic-resolution evaluation.')
    parser.add_argument('--visual_prune_specflow_use_best_heads', action=argparse.BooleanOptionalAction, default=True,
                        help='Use lowest-entropy CLS-attention heads as SpecFlow heat seeds.')
    parser.add_argument('--visual_prune_specflow_num_best_heads', type=int, default=16)
    parser.add_argument('--visual_prune_specflow_k_neighbors', type=int, default=25)
    parser.add_argument('--visual_prune_specflow_diffusion_steps', type=int, default=2)
    parser.add_argument('--visual_prune_specflow_alpha', type=float, default=0.7)
    parser.add_argument('--visual_prune_specflow_min_crop_size', type=int, default=4)
    parser.add_argument('--visual_prune_specflow_adaptive_threshold', type=float, default=1e-4)
    parser.add_argument('--visual_prune_specflow_use_coreset_sinks', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--visual_prune_specflow_num_sinks', type=int, default=2)
    parser.add_argument('--visual_prune_quant_bits', type=int, default=None,
                        help='Bits for pseudo-quantizing ViT+projector features before pruning; defaults to a_bits.')
    parser.add_argument('--visual_prune_quant_groupsize', type=int, default=-1,
                        help='Group size for pseudo-quantizing ViT+projector features before pruning.')
    parser.add_argument('--visual_prune_quant_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='Use asymmetric pseudo-quantization for ViT+projector features before pruning.')
    parser.add_argument('--visual_prune_quant_clip_ratio', type=float, default=1.0,
                        help='Clip ratio for pseudo-quantizing ViT+projector features before pruning.')
    parser.add_argument('--visual_prune_score_activation_weight', type=float, default=0.5,
                        help='Weight for activation magnitude in visual token pruning score.')
    parser.add_argument('--visual_prune_score_similarity_weight', type=float, default=0.5,
                        help='Weight for visual-text cosine similarity in visual token pruning score.')
    parser.add_argument('--visual_prune_min_keep_tokens', type=int, default=1,
                        help='Minimum number of visual tokens kept per eval sample.')
    parser.add_argument('--visual_prune_recycle', action=argparse.BooleanOptionalAction, default=False,
                        help='Merge dropped visual tokens into kept visual tokens before pruning them away.')
    parser.add_argument('--visual_prune_recycle_mode', type=str, default='knn',
                        choices=('knn', 'window'),
                        help='How dropped visual tokens are assigned to kept tokens for recycling.')
    parser.add_argument('--visual_prune_recycle_alpha', type=float, default=0.25,
                        help='Weight of each dropped visual token when merging into its assigned kept token.')
    parser.add_argument('--visual_prune_recycle_window_size', type=int, default=64,
                        help='Local token-window size used by window recycle mode.')
    parser.add_argument('--visual_prune_log_every', type=int, default=50,
                        help='Log visual pruning stats every N eval samples.')
    parser.add_argument('--a_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric Activation quantization (default: False)')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--q_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='Use asymmetric query-cache quantization.')
    parser.add_argument('--k_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='Use asymmetric K-cache quantization.')
    parser.add_argument('--v_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='Use asymmetric V-cache quantization.')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=True,
                        help='''Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--a_groupsize', type=int, default=-1, 
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--a_clip_ratio', type=float, default=1.0,
                        help='Clip ratio for activation quantization. new_max = max * clip_ratio')
    parser.add_argument('--lma_clip_ratio', type=float, default=1.0,
                        help='override Clip ratio for lm activation quantization. new_max = max * clip_ratio')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default=['MMMU_val'], 
                        help='Calibration dataset', choices=PARAM_CHOICES['cal_dataset'])
    parser.add_argument('--question_file', type=str, default=None, 
                        help="Path to calibration dataset question file(ScienceQA)")
    parser.add_argument('--caption_file', type=str, default=None,
                        help="Path to COCO+ShareGPT4V calibration JSON or JSONL")
    parser.add_argument('--image_base', type=str, default=None, 
                        help="Path to calibration dataset image root")
    parser.add_argument('--smooth_token_importance_mode', type=str,
                        choices=['tlq_topk'], default='tlq_topk',
                        help='Strict TLQ Linear grad_input Top-K used by scale_smooth.')
    parser.add_argument('--smooth_tlq_topk_ratio', type=float, default=0.5,
                        help='Fraction of shared token positions selected by TLQ Top-K')
    parser.add_argument('--smooth_tlq_padding_side', type=str,
                        choices=('auto', 'left', 'right'), default='auto',
                        help='Shared TLQ token grid: Qwen2-VL uses left padding; LLaVA uses right.')

    # quantization
    parser.add_argument('--quant_method', type=str, default='rtn',
                        help='Method when quantize the model weights', choices=PARAM_CHOICES['quant_method'])
    parser.add_argument('--load_qmodel', type=bool, default=False, help='Load the quantized model')
    parser.add_argument('--save_qmodel_dir', type=str, default=None, 
                        help='Path to save the quantized model, None means do not save')
    parser.add_argument('--load_backward_result', type=bool, default=False, help='Load the gard statistic in backward')
    parser.add_argument('--save_backward_result_dir', type=str, default=None, help='Dir to the grad statistic')
    parser.add_argument('--backward_result_filename', type=str, default='backward_result.pt',
                        help='Filename for the cached backward result (.pt only; no path components)')
    parser.add_argument('--load_smooth_result', type=bool, default=False, help="load the result of scale smooth")
    parser.add_argument('--save_smooth_result_dir', type=str, default=None, help='Dir to result of scale smooth')
    parser.add_argument('--smooth_search_parallel', action=argparse.BooleanOptionalAction, default=False,
                        help='Evaluate the fixed smooth-scale ratio grid on multiple GPUs.')
    parser.add_argument('--smooth_search_max_workers', type=int, default=4,
                        help='Maximum number of GPUs used by exact ratio-parallel smoothing.')
    parser.add_argument('--smooth_search_min_free_gpu_gb', type=float, default=3.0,
                        help='Free-memory reserve required before a GPU receives a search worker.')
    parser.add_argument('--smooth_search_checkpoint', action=argparse.BooleanOptionalAction, default=False,
                        help='Atomically save per-layer smoothing progress.')
    parser.add_argument('--smooth_search_resume', action=argparse.BooleanOptionalAction, default=False,
                        help='Explicitly resume a compatible per-layer smoothing checkpoint.')
    # evaluation
    parser.add_argument('--eval_dataset', type=str, default=['MMMU_val'],
                        help='evaluation tasks', nargs="+", choices=PARAM_CHOICES['eval_dataset'])
    parser.add_argument('--eval_log_path', type=str, default='./log/', help='Path to save the output eval results log.')
    parser.add_argument('--eval_result_dir', type=str, default='./cache/eval_results', help="Path to save eval results")
    parser.add_argument('--eval_generation_mode', type=str, default='legacy',
                        choices=('legacy', 'comparable'),
                        help='legacy preserves the existing generation paths; comparable uses one greedy inputs_embeds runner for dense and pruned evaluation.')
    parser.add_argument('--eval_timing', action=argparse.BooleanOptionalAction, default=False,
                        help='Record synchronized per-sample preprocessing and model-generation latency as JSONL.')
    parser.add_argument('--eval_timing_warmup_samples', type=int, default=10,
                        help='Number of leading timed samples retained in JSONL but excluded from the timing summary.')
    parser.add_argument('--eval_ignore_cache', action=argparse.BooleanOptionalAction, default=False,
                        help='Ignore existing prediction caches and recompute every selected evaluation sample.')
    parser.add_argument('--eval_inference_only', action=argparse.BooleanOptionalAction, default=False,
                        help='Stop after prediction generation; skip spreadsheet export and dataset judging.')
    parser.add_argument('--eval_max_samples', type=int, default=None,
                        help='Optional maximum number of evaluation samples, useful for matched speed tests.')

    # speed test

    # profile
    parser.add_argument('--profile', type=bool, default=False, help='profile the quantize process')
    parser.add_argument('--profile_dir', type=str, default=None, help='Save dir of the profile output')
    parser.add_argument('--error_check', action=argparse.BooleanOptionalAction, default=False,
                        help='Compare the original model and the current model on identical inputs.')
    parser.add_argument('--error_check_samples', type=int, default=10,
                        help='Number of samples used for original/current model output error checking.')
    parser.add_argument('--error_check_topk', type=int, default=None,
                        help='Absolute number of top important channels to track by gradient magnitude.')
    parser.add_argument('--error_check_topk_percent', type=float, default=90.0,
                        help='Percentage of channels to consider important (default: 10%)')
    parser.add_argument('--error_check_plot_mode', type=str, default='channel_diagnostics',
                        choices=('relative_error_histogram', 'channel_diagnostics', 'both'),
                        help='Plot mode for layer input error_check results.')
    parser.add_argument('--split_error_sweep', action=argparse.BooleanOptionalAction, default=False,
                        help='Run error_check sweep over online_channel_split_topk_percent values.')

    args = parser.parse_args()

    # load config from file
    if args.config:
        assert os.path.exists(args.config)
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for k, v in config.items():
                setattr(args, k, v)
    if getattr(args, 'flatquant_matrix_path', None):
        args.flatquant_matrix_path = resolve_project_path(args.flatquant_matrix_path)
    if getattr(args, "split_error_sweep", False):
        args.error_check = True
    validate_args(args)

    # print config
    logging.info('Arguments: ')
    logging.info(pprint.pformat(vars(args)))
    logging.info('--' * 30)

    set_seed(args.seed)

    return args


def validate_args(args):
    """
    Validate all argument values are within allowed choices.

    This function:
    - Validates all choice-restricted parameters against PARAM_CHOICES
    - Normalizes multi-value parameters to lists
    - Raises ValueError with clear error messages if validation fails

    Args:
        args: argparse.Namespace object

    Raises:
        ValueError: If any parameter value is not in the allowed choices
    """
    if getattr(args, "flatquant", False):
        from vlmrot.p4q_config import validate_p4q_args
        validate_p4q_args(args)
        invalid_datasets = [name for name in args.eval_dataset if name not in supported_eval_datasets]
        if invalid_datasets:
            raise ValueError(f"Unsupported eval_dataset: {invalid_datasets}")
        return

    errors = []
    if getattr(args, "eval_generation_mode", "legacy") not in (
        "legacy",
        "comparable",
    ):
        errors.append("eval_generation_mode must be one of: legacy, comparable")
    eval_max_samples = getattr(args, "eval_max_samples", None)
    if eval_max_samples is not None and int(eval_max_samples) < 1:
        errors.append("eval_max_samples must be >= 1 or None")
    if eval_max_samples is not None and not bool(
        getattr(args, "eval_inference_only", False)
    ):
        errors.append("eval_max_samples requires eval_inference_only=True")
    if int(getattr(args, "eval_timing_warmup_samples", 10)) < 0:
        errors.append("eval_timing_warmup_samples must be >= 0")
    flatquant_eval_only = bool(getattr(args, "flatquant", False)) and getattr(
        args, "flatquant_mode", "train"
    ) == "eval"
    path_vars = ['model_path']
    if not flatquant_eval_only:
        path_vars.extend(['question_file', 'caption_file', 'image_base'])

    for param_name, valid_choices in PARAM_CHOICES.items():
        value = getattr(args, param_name, None)
        if value is None:
            continue

        # Handle multi-value parameters
        if param_name in MULTI_VALUE_PARAMS:
            # Convert to list if string
            if isinstance(value, str):
                values = [value]
            else:
                values = list(value) if value else []

            # Validate each value
            invalid_values = [v for v in values if v not in valid_choices]
            if invalid_values:
                errors.append(
                    f"{param_name}: {invalid_values} not in {valid_choices}"
                )
            else:
                # Update args with normalized list
                setattr(args, param_name, values)
        else:
            # Single value validation
            if value not in valid_choices:
                errors.append(
                    f"{param_name}: '{value}' not in valid choices {valid_choices}"
                )

    for key in path_vars:
        value = getattr(args, key, None)
        if value is not None:
            is_model_hub_id = key == "model_path" and value == getattr(
                args, "model", None
            )
            if not os.path.exists(value) and not is_model_hub_id:
                errors.append(f"{key}: '{value}' not exist!")
    flatquant_prune_calibration = bool(
        getattr(args, "flatquant_prune_calibration", False)
    )
    if flatquant_prune_calibration and not getattr(args, "flatquant", False):
        errors.append("flatquant_prune_calibration requires flatquant=True")
    if getattr(args, "flatquant", False):
        flatquant_mode = getattr(args, "flatquant_mode", "train")
        if flatquant_mode not in ("train", "eval", "train_eval"):
            errors.append(
                "flatquant_mode must be one of: train, eval, train_eval"
            )
        conflicts = (
            "scale_smooth",
            "smoothquant",
            "spinquant",
            "offline_rotate",
            "online_rotate",
            "split_rotate",
            "split_r1_rotate",
        )
        enabled_conflicts = [name for name in conflicts if getattr(args, name, False)]
        if enabled_conflicts:
            errors.append(
                "flatquant is mutually exclusive with: "
                + ", ".join(enabled_conflicts)
            )
        if int(getattr(args, "a_groupsize", -1)) > 0:
            errors.append("flatquant does not support a_groupsize > 0")
        if flatquant_prune_calibration:
            if not getattr(args, "visual_prune", False):
                errors.append(
                    "flatquant_prune_calibration requires visual_prune=True"
                )
            calibration_prune_strategy = getattr(
                args, "visual_prune_strategy", None
            )
            if calibration_prune_strategy not in ("vispruner", "crisp", "p4q"):
                errors.append(
                    "flatquant_prune_calibration requires "
                    "visual_prune_strategy=vispruner, crisp, or p4q"
                )
            if calibration_prune_strategy == "vispruner":
                if not getattr(
                    args, "visual_prune_vispruner_quant_vit_features", False
                ):
                    errors.append(
                        "VisPruner FlatQuant calibration requires "
                        "visual_prune_vispruner_quant_vit_features=True"
                    )
            if calibration_prune_strategy == "vispruner":
                strategy_label = "VisPruner"
                visual_quant_bits = getattr(args, "visual_prune_quant_bits", None)
                visual_quant_bits = (
                    int(getattr(args, "a_bits", 16))
                    if visual_quant_bits is None
                    else int(visual_quant_bits)
                )
                if visual_quant_bits != int(getattr(args, "a_bits", 16)):
                    errors.append(
                        "visual_prune_quant_bits must equal a_bits during "
                        f"{strategy_label} FlatQuant calibration"
                    )
                if int(getattr(args, "visual_prune_quant_groupsize", -1)) != int(
                    getattr(args, "a_groupsize", -1)
                ):
                    errors.append(
                        "visual_prune_quant_groupsize must equal a_groupsize during "
                        f"{strategy_label} FlatQuant calibration"
                    )
                if bool(getattr(args, "visual_prune_quant_asym", False)) != bool(
                    getattr(args, "a_asym", False)
                ):
                    errors.append(
                        "visual_prune_quant_asym must match a_asym during "
                        f"{strategy_label} FlatQuant calibration"
                    )
            if getattr(args, "visual_prune_recycle", False):
                errors.append(
                    "flatquant_prune_calibration does not support visual token recycling"
                )
        if int(getattr(args, "cali_bsz", 1)) < 1:
            errors.append("cali_bsz must be >= 1")
        if int(getattr(args, "epochs", 15)) < 1:
            errors.append("epochs must be >= 1")
        if float(getattr(args, "flat_lr", 0.005)) <= 0:
            errors.append("flat_lr must be > 0")
        if int(getattr(args, "flatquant_fold_check_samples", 1)) < 0:
            errors.append("flatquant_fold_check_samples must be >= 0")
        if float(getattr(args, "flatquant_fold_check_tol", 0.001)) < 0:
            errors.append("flatquant_fold_check_tol must be >= 0")
        if flatquant_mode in ("eval", "train_eval"):
            if getattr(args, "quant_method", "rtn") != "rtn":
                errors.append("FlatQuant evaluation currently requires quant_method=rtn")
            if int(getattr(args, "w_groupsize", -1)) != -1:
                errors.append("FlatQuant RTN evaluation requires w_groupsize=-1")
        if flatquant_mode == "eval":
            matrix_path = getattr(args, "flatquant_matrix_path", None)
            if not matrix_path:
                errors.append("flatquant_mode=eval requires flatquant_matrix_path")
            else:
                resolved_matrix_path = str(matrix_path)
                if not resolved_matrix_path.lower().endswith(".pth"):
                    resolved_matrix_path = os.path.join(
                        resolved_matrix_path, "flat_matrices.pth"
                    )
                if not os.path.exists(resolved_matrix_path):
                    errors.append(
                        f"flatquant_matrix_path: '{resolved_matrix_path}' not exist!"
                    )
    split_k_cache = getattr(args, "online_channel_split_topk_percent_cache", None)
    if split_k_cache is not None and split_k_cache != "":
        if not str(split_k_cache).lower().endswith(".json"):
            errors.append("online_channel_split_topk_percent_cache must be a .json path")
    r1_split_topk_percent = float(getattr(args, "r1_split_topk_percent", 10.0))
    if r1_split_topk_percent <= 0.0 or r1_split_topk_percent > 100.0:
        errors.append("r1_split_topk_percent must be in (0, 100]")
    if getattr(args, "r1_split_importance_reduce", "p95") not in ("mean", "p95"):
        errors.append("r1_split_importance_reduce must be one of: mean, p95")
    if getattr(args, "split_r1_rotate", False):
        if not getattr(args, "offline_rotate", False):
            errors.append("split_r1_rotate requires offline_rotate=True")
        if getattr(args, "spinquant", False):
            errors.append("split_r1_rotate is mutually exclusive with spinquant")
    if getattr(args, "split_error_sweep", False):
        if not getattr(args, "offline_rotate", False):
            errors.append("split_error_sweep requires offline_rotate=True")
        if not getattr(args, "split_rotate", False):
            errors.append("split_error_sweep requires split_rotate=True")
        if getattr(args, "online_channel_split_topk", None) is not None:
            errors.append("split_error_sweep requires online_channel_split_topk=None")
        if getattr(args, "online_channel_split_topp_percent", None) is not None:
            errors.append("split_error_sweep does not support online_channel_split_topp_percent")
        sweep = getattr(args, "online_channel_split_topk_percent_sweep", None)
        if sweep is not None:
            try:
                if isinstance(sweep, (list, tuple)):
                    values = [float(value) for value in sweep]
                else:
                    values = [float(part.strip()) for part in str(sweep).split(",") if part.strip()]
            except ValueError:
                values = []
                errors.append(
                    "online_channel_split_topk_percent_sweep must be a comma-separated float list"
                )
            invalid = [value for value in values if value <= 0.0 or value > 100.0]
            if not values:
                errors.append("online_channel_split_topk_percent_sweep must not be empty")
            if invalid:
                errors.append(
                    "online_channel_split_topk_percent_sweep values must be in (0, 100]"
                )
    if getattr(args, "visual_prune", False):
        visual_strategy = getattr(args, "visual_prune_strategy", "activation_similarity")
        if visual_strategy not in ("activation_similarity", "vispruner", "crisp", "p4q", "visionzip", "specflow"):
            errors.append("visual_prune_strategy must be one of: activation_similarity, vispruner, crisp, p4q, visionzip, specflow")
        visual_keep_ratio = float(getattr(args, "visual_prune_keep_ratio", 1.0))
        if visual_keep_ratio <= 0.0 or visual_keep_ratio > 1.0:
            errors.append("visual_prune_keep_ratio must be in (0, 1]")
        visual_important_ratio = float(getattr(args, "visual_prune_important_ratio", 0.5))
        if visual_important_ratio < 0.0 or visual_important_ratio > 1.0:
            errors.append("visual_prune_important_ratio must be in [0, 1]")
        if visual_strategy == "p4q":
            p4q_beta = float(getattr(args, "visual_prune_p4q_beta", 0.5))
            if p4q_beta <= 0.0 or p4q_beta > 1.0:
                errors.append("visual_prune_p4q_beta must be in (0, 1]")
            p4q_lambda = float(getattr(args, "visual_prune_p4q_lambda", 0.6))
            if p4q_lambda < 0.0 or p4q_lambda > 1.0:
                errors.append("visual_prune_p4q_lambda must be in [0, 1]")
        if visual_strategy == "visionzip":
            dominant = int(
                getattr(args, "visual_prune_visionzip_dominant_tokens", 54)
            )
            contextual = int(
                getattr(args, "visual_prune_visionzip_contextual_tokens", 10)
            )
            if dominant < 1:
                errors.append("visual_prune_visionzip_dominant_tokens must be >= 1")
            if contextual < 0:
                errors.append("visual_prune_visionzip_contextual_tokens must be >= 0")
            if dominant + contextual > 576:
                errors.append(
                    "VisionZip dominant + contextual tokens must not exceed "
                    "the 576-token CLIP prompt capacity"
                )
        if visual_strategy == "specflow":
            specflow_budget_mode = str(
                getattr(args, "visual_prune_specflow_budget_mode", "fixed")
            ).lower()
            if specflow_budget_mode not in ("fixed", "ratio"):
                errors.append(
                    "visual_prune_specflow_budget_mode must be fixed or ratio"
                )
            if (
                specflow_budget_mode == "ratio"
                and getattr(args, "model", None)
                != "Qwen/Qwen2.5-VL-7B-Instruct"
            ):
                errors.append(
                    "visual_prune_specflow_budget_mode=ratio is supported only "
                    "for Qwen2.5-VL-7B"
                )
            if int(getattr(args, "visual_prune_specflow_token_budget", 192)) < 1:
                errors.append("visual_prune_specflow_token_budget must be >= 1")
            if int(getattr(args, "visual_prune_specflow_num_best_heads", 16)) < 1:
                errors.append("visual_prune_specflow_num_best_heads must be >= 1")
            if int(getattr(args, "visual_prune_specflow_k_neighbors", 25)) < 1:
                errors.append("visual_prune_specflow_k_neighbors must be >= 1")
            if int(getattr(args, "visual_prune_specflow_diffusion_steps", 2)) < 0:
                errors.append("visual_prune_specflow_diffusion_steps must be >= 0")
            specflow_alpha = float(
                getattr(args, "visual_prune_specflow_alpha", 0.7)
            )
            if specflow_alpha < 0.0 or specflow_alpha > 1.0:
                errors.append("visual_prune_specflow_alpha must be in [0, 1]")
            if int(getattr(args, "visual_prune_specflow_min_crop_size", 4)) < 1:
                errors.append("visual_prune_specflow_min_crop_size must be >= 1")
            if float(
                getattr(args, "visual_prune_specflow_adaptive_threshold", 1e-4)
            ) < 0.0:
                errors.append(
                    "visual_prune_specflow_adaptive_threshold must be >= 0"
                )
            if int(getattr(args, "visual_prune_specflow_num_sinks", 2)) < 0:
                errors.append("visual_prune_specflow_num_sinks must be >= 0")
        if visual_strategy in ("visionzip", "specflow"):
            fp16_models = [
                "llava-hf/llava-1.5-7b-hf",
                "llava-hf/llava-v1.6-vicuna-7b-hf",
            ]
            if visual_strategy == "specflow":
                fp16_models.append("Qwen/Qwen2.5-VL-7B-Instruct")
            if getattr(args, "model", None) not in fp16_models:
                supported_names = "LLaVA-1.5-7B and LLaVA-NeXT-7B"
                if visual_strategy == "specflow":
                    supported_names += " and Qwen2.5-VL-7B"
                errors.append(
                    f"{visual_strategy} supports only {supported_names}"
                )
            non_fp16_bits = [
                name
                for name in ("w_bits", "a_bits", "q_bits", "k_bits", "v_bits")
                if int(getattr(args, name, 16)) != 16
            ]
            if non_fp16_bits:
                errors.append(
                    f"{visual_strategy} is FP16-only; set to 16: "
                    + ", ".join(non_fp16_bits)
                )
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
                name
                for name in incompatible_flags
                if bool(getattr(args, name, False))
            ]
            if enabled_flags:
                errors.append(
                    f"{visual_strategy} FP16 baseline requires disabled: "
                    + ", ".join(enabled_flags)
                )
        visual_min_keep = int(getattr(args, "visual_prune_min_keep_tokens", 1))
        if visual_min_keep < 1:
            errors.append("visual_prune_min_keep_tokens must be >= 1")
        visual_quant_bits = getattr(args, "visual_prune_quant_bits", None)
        if visual_quant_bits is not None and int(visual_quant_bits) < 2:
            errors.append("visual_prune_quant_bits must be >= 2 or None")
        visual_clip_ratio = float(getattr(args, "visual_prune_quant_clip_ratio", 1.0))
        if visual_clip_ratio <= 0.0 or visual_clip_ratio > 1.0:
            errors.append("visual_prune_quant_clip_ratio must be in (0, 1]")
        activation_weight = float(
            getattr(args, "visual_prune_score_activation_weight", 0.5)
        )
        similarity_weight = float(
            getattr(args, "visual_prune_score_similarity_weight", 0.5)
        )
        if activation_weight < 0.0 or similarity_weight < 0.0:
            errors.append("visual pruning score weights must be non-negative")
        if activation_weight + similarity_weight <= 0.0:
            errors.append("visual pruning score weights must sum to a positive value")
        if visual_strategy == "crisp":
            crisp_alpha = float(getattr(args, "visual_prune_crisp_alpha", 0.3))
            if crisp_alpha <= 0.0 or crisp_alpha > 1.0:
                errors.append("visual_prune_crisp_alpha must be in (0, 1]")
            crisp_low_alpha = float(
                getattr(args, "visual_prune_crisp_low_budget_alpha", 0.5)
            )
            if crisp_low_alpha <= 0.0 or crisp_low_alpha > 1.0:
                errors.append("visual_prune_crisp_low_budget_alpha must be in (0, 1]")
            if int(getattr(args, "visual_prune_crisp_low_budget_threshold", 100)) < 1:
                errors.append("visual_prune_crisp_low_budget_threshold must be >= 1")
            if bool(getattr(args, "visual_prune_crisp_quantize_weights", False)):
                if int(getattr(args, "w_bits", 16)) < 2:
                    errors.append("CRISP weight fake quantization requires w_bits >= 2")
                if int(getattr(args, "w_groupsize", -1)) != -1:
                    errors.append(
                        "CRISP RTN weight fake quantization currently requires "
                        "w_groupsize=-1"
                    )
            qwen_crisp_prior_weight = float(
                getattr(args, "visual_prune_qwen_crisp_prior_weight", 0.25)
            )
            if qwen_crisp_prior_weight < 0.0 or qwen_crisp_prior_weight > 1.0:
                errors.append(
                    "visual_prune_qwen_crisp_prior_weight must be in [0, 1]"
                )
        if visual_strategy in ("crisp", "p4q") and getattr(
            args, "visual_prune_crisp_language", "auto"
        ) not in ("auto", "en", "zh"):
            errors.append("visual_prune_crisp_language must be one of: auto, en, zh")
        if visual_strategy in ("vispruner", "crisp", "p4q", "visionzip", "specflow") and getattr(args, "visual_prune_recycle", False):
            errors.append(
                "visual_prune_recycle is not compatible with strict "
                f"{visual_strategy} strategy"
            )
        if getattr(args, "visual_prune_recycle", False):
            recycle_mode = getattr(args, "visual_prune_recycle_mode", "knn")
            if recycle_mode not in ("knn", "window"):
                errors.append("visual_prune_recycle_mode must be one of: knn, window")
            recycle_alpha = float(getattr(args, "visual_prune_recycle_alpha", 0.25))
            if recycle_alpha < 0.0 or recycle_alpha > 1.0:
                errors.append("visual_prune_recycle_alpha must be in [0, 1]")
            recycle_window_size = int(
                getattr(args, "visual_prune_recycle_window_size", 64)
            )
            if recycle_window_size < 1:
                errors.append("visual_prune_recycle_window_size must be >= 1")
        if int(getattr(args, "visual_prune_log_every", 50)) < 1:
            errors.append("visual_prune_log_every must be >= 1")
    if int(getattr(args, "smooth_search_max_workers", 4)) < 1:
        errors.append("smooth_search_max_workers must be >= 1")
    if float(getattr(args, "smooth_search_min_free_gpu_gb", 3.0)) < 0.0:
        errors.append("smooth_search_min_free_gpu_gb must be >= 0")
    if getattr(args, "scale_smooth", False):
        if getattr(args, "smooth_token_importance_mode", "tlq_topk") != "tlq_topk":
            errors.append("scale_smooth supports only strict TLQ tlq_topk semantics")
        strict_run_values = {
            "cal_dataset": (
                "COCO_ShareGPT4V",
                getattr(args, "cal_dataset", None),
            ),
            "nsamples": (128, int(getattr(args, "nsamples", 0))),
            "seed": (42, int(getattr(args, "seed", -1))),
            "smooth_tlq_topk_ratio": (
                0.5,
                float(getattr(args, "smooth_tlq_topk_ratio", 0.0)),
            ),
            "quant_method": ("rtn", getattr(args, "quant_method", None)),
        }
        for name, (expected, actual) in strict_run_values.items():
            if actual != expected:
                errors.append(
                    f"strict TLQ scale_smooth requires {name}={expected}; got {actual!r}"
                )
        if not getattr(args, "caption_file", None):
            errors.append("strict TLQ scale_smooth requires a ShareGPT4V caption_file")
        if not getattr(args, "image_base", None):
            errors.append("strict TLQ scale_smooth requires a ShareGPT4V image_base")
        if bool(getattr(args, "smoothquant", False)):
            errors.append(
                "scale_smooth cannot be combined with plain SmoothQuant because "
                "both branches rescale the same LM weights"
            )
        if (
            getattr(args, "quant_method", None) == "rtn"
            and int(getattr(args, "w_groupsize", -1)) != -1
        ):
            errors.append(
                "RTN supports only w_groupsize=-1; group-wise RTN weight "
                "quantization is not implemented"
            )
        for name in ("a_clip_ratio", "lma_clip_ratio"):
            value = float(getattr(args, name, 1.0))
            if value <= 0.0 or value > 1.0:
                errors.append(f"{name} must be in (0, 1]")
    if getattr(args, "smooth_search_resume", False) and not getattr(
        args, "smooth_search_checkpoint", False
    ):
        errors.append("smooth_search_resume requires smooth_search_checkpoint=True")
    if getattr(args, "smooth_search_checkpoint", False) and not getattr(
        args, "save_smooth_result_dir", None
    ):
        errors.append("smooth_search_checkpoint requires save_smooth_result_dir")
    if getattr(args, "smooth_search_checkpoint", False) and not getattr(
        args, "save_backward_result_dir", None
    ):
        errors.append(
            "smooth_search_checkpoint requires save_backward_result_dir so resume "
            "can verify the exact backward artifact"
        )
    if errors:
        error_msg = "Argument validation failed:\n- " + "\n- ".join(errors)
        logging.error(error_msg)
        raise ValueError(error_msg)
    
def set_seed(seed):
    np.random.seed(seed)  # Set NumPy seed
    random.seed(seed)  # Set Python's built-in random seed
    torch.manual_seed(seed)  # Set PyTorch seed for CPU & CUDA
    torch.cuda.manual_seed(seed)  # Set seed for CUDA (if using GPU)
    torch.cuda.manual_seed_all(seed)  # If multi-GPU, set for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-optimizations for determinism

def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

def get_dev():
    return (
        torch.device(f'cuda:{dist.get_rank()}') if dist.is_initialized() else 
        torch.device('cuda') if torch.cuda.is_available() else 
        torch.device('cpu'))

def llama_down_proj_groupsize(model, groupsize):
    
    assert groupsize > 1, 'groupsize should be greater than 1!'
    
    if model.config.intermediate_size % groupsize == 0:
        logging.info(f'(Act.) Groupsiz = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size/groupsize)
    assert groupsize*group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size//group_num
    assert down_proj_groupsize*group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logging.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")

def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")

def set_op_by_name(layer, name, new_module):
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

def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x

def normalize_device(device):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f'cuda:{device}')
    if isinstance(device, str):
        if device.isdigit():
            return torch.device(f'cuda:{device}')
        if device.startswith('cuda') or device == 'cpu':
            return torch.device(device)
        if device in ('disk', 'meta'):
            return torch.device('cpu')
    return torch.device(device)

def move_nested_tensor_to_device(value, device, detach=False):
    device = normalize_device(device)
    if torch.is_tensor(value):
        tensor = value.detach() if detach else value
        return tensor.to(device)
    if isinstance(value, dict):
        return {k: move_nested_tensor_to_device(v, device, detach=detach) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_nested_tensor_to_device(v, device, detach=detach) for v in value)
    if isinstance(value, list):
        return [move_nested_tensor_to_device(v, device, detach=detach) for v in value]
    return value

def detach_nested_tensor_to_cpu(value):
    return move_nested_tensor_to_device(value, torch.device('cpu'), detach=True)

def model_first_device(model):
    device_map = getattr(model, 'hf_device_map', None) or {}
    preferred_names = (
        'model.language_model.embed_tokens',
        'language_model.embed_tokens',
        'model.embed_tokens',
        'embed_tokens',
        'model.language_model',
        'language_model',
        'model',
    )
    for preferred_name in preferred_names:
        if preferred_name in device_map:
            return normalize_device(device_map[preferred_name])
    for device in device_map.values():
        normalized = normalize_device(device)
        if normalized.type == 'cuda':
            return normalized
    return get_dev()

def layer_device(model, layer_idx):
    device_map = getattr(model, 'hf_device_map', None) or {}
    candidate_names = (
        f'model.language_model.layers.{layer_idx}',
        f'language_model.layers.{layer_idx}',
        f'model.layers.{layer_idx}',
        f'layers.{layer_idx}',
    )
    for name in candidate_names:
        if name in device_map:
            return normalize_device(device_map[name])
    layer_marker = f'layers.{layer_idx}'
    for name, device in device_map.items():
        if layer_marker in name:
            return normalize_device(device)
    return model_first_device(model) if device_map else torch.device('cpu')

def dispatch_model_to_cuda(model):
    device_map = getattr(model, 'hf_device_map', None)
    if not device_map:
        return model.cuda()
    model = dispatch_model(model, device_map=device_map)
    if type(model) == model_utils.LLAVA_NEXT_HF:
        model.model.image_newline.data = model.model.image_newline.data.to(model_first_device(model))
    # for name, param in model.named_parameters():
    #     print(name, param.device, param.dtype, tuple(param.shape))
    return model

def move_model_to_cpu(model):
    remove_hook_from_submodules(model)
    model = model.cpu()
    return model

@torch.no_grad()
def get_free_gpu():
    free_mem = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_mem.append(free)
    best_gpu = int(torch.argmax(torch.tensor(free_mem)))
    return best_gpu, free_mem

def list_tensors_on_gpu(device_id=0):
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.device.index == device_id:
                tensors.append(obj)
        except:
            pass
    return tensors

def move_list_tensor(x: list, device):
    device = normalize_device(device)
    for i in range(len(x)):
        old = x[i]
        x[i] = old.detach().to(device)
        del old
    torch.cuda.empty_cache()

def del_list_tensor(x: list):
    for tensor in x:
        del tensor
    torch.cuda.empty_cache()
