import torch
import torch.nn as nn
from torch.autograd.graph import save_on_cpu
import argparse, logging, functools, os, ntpath, copy, time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from vlmrot import utils, model_utils, data_utils, rotation_utils, smooth_utils, parrot_utils, \
                   quant_utils, hadamard_utils, gptq_utils, eval_utils, vis_utils, check_utils, rotation_split_utils
from vlmrot.smooth_parallel import atomic_torch_save

logging.basicConfig(level=logging.INFO, format='%(message)s')


def _matches_target_layer(name, target_layer_idx):
    return target_layer_idx is None or f"layers.{target_layer_idx}." in name


def _apply_weight_quantization(model, args, save_qmodel=True, target_layer_idx=None):
    if args.w_bits >= 16:
        return {}

    save_dict = {}
    if args.load_qmodel and args.save_qmodel_dir:
        rotate = '_rotate' if args.offline_rotate else ''
        save_path = os.path.join(args.save_qmodel_dir, f"{args.model.split('/')[-1]}_W{args.w_bits}A{args.a_bits}{rotate}.pt")
        assert os.path.exists(save_path)
        logging.info(f'Loading quantized model from {save_path}')
        save_dict = torch.load(save_path, weights_only=False)
        model.load_state_dict(save_dict['model'])
    elif args.quant_method == 'rtn':
        quantizers = gptq_utils.rtn_fwrd(
            model,
            utils.get_dev(),
            args,
            target_layer_idx=target_layer_idx,
        )
        save_dict["w_quantizers"] = quantizers

    if save_qmodel and args.save_qmodel_dir:
        if not os.path.exists(args.save_qmodel_dir):
            os.makedirs(args.save_qmodel_dir)
        save_dict["model"] = model.state_dict()
        rotate = '_rotate' if args.offline_rotate else ''
        save_path = os.path.join(args.save_qmodel_dir, f"{args.model.split('/')[-1]}_W{args.w_bits}A{args.a_bits}{rotate}.pt")
        torch.save(save_dict, save_path)
        logging.info(f'Quantized model saved at: {save_path}')

    return save_dict


def _configure_activation_quantization(model, args, target_layer_idx=None):
    if args.a_bits >= 16 and args.v_bits >= 16:
        return

    if target_layer_idx is None:
        logging.info(f'setting a clip ratio in lm {min(args.a_clip_ratio, args.lma_clip_ratio)}')
    qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
    down_proj_groupsize = -1
    if args.a_groupsize > 0 and "llama" in args.model:
        down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
    for name in qlayers:
        if not _matches_target_layer(name, target_layer_idx):
            continue
        layer_input_bits = args.a_bits
        layer_groupsize = args.a_groupsize
        layer_a_sym = not(args.a_asym)
        layer_a_clip = min(args.a_clip_ratio, args.lma_clip_ratio)

        if 'v_proj' in name and args.v_bits < 16:
            qlayers[name].out_quantizer.configure(
                bits=args.v_bits,
                groupsize=getattr(args, "v_groupsize", -1),
                sym=not(getattr(args, "v_asym", False)),
                clip_ratio=getattr(args, "v_clip_ratio", 1.0),
            )

        if 'lm_head' in name:
            layer_input_bits = 16

        if 'down_proj' in name:
            if args.int8_down_proj:
                layer_input_bits = 8
            layer_groupsize = down_proj_groupsize

        qlayers[name].quantizer.configure(
            bits=layer_input_bits,
            groupsize=layer_groupsize,
            sym=layer_a_sym,
            clip_ratio=layer_a_clip,
        )


def _build_tlq_quant_config(model, args):
    """Resolve the exact fake-quantization contract used during TLQ search."""
    down_proj_groupsize = -1
    a_groupsize = int(getattr(args, 'a_groupsize', -1))
    if a_groupsize > 0 and "llama" in str(args.model).lower():
        down_proj_groupsize = utils.llama_down_proj_groupsize(model, a_groupsize)
    return {
        'w_sym': not bool(getattr(args, 'w_asym', False)),
        'w_clip': bool(getattr(args, 'w_clip', True)),
        'a_sym': not bool(getattr(args, 'a_asym', False)),
        'a_groupsize': a_groupsize,
        'down_proj_a_groupsize': down_proj_groupsize,
        'a_clip_ratio': min(
            float(getattr(args, 'a_clip_ratio', 1.0)),
            float(getattr(args, 'lma_clip_ratio', 1.0)),
        ),
        'int8_down_proj': bool(getattr(args, 'int8_down_proj', False)),
    }


def _configure_split_rotation(
    model,
    split_online_gradients,
    args,
    plot_counts=True,
    target_layer_idx=None,
):
    qlayers = quant_utils.find_qlayers(model)
    split_important_counts = rotation_split_utils.configure_split_online_rotation(
        model,
        qlayers,
        split_online_gradients,
        args,
        target_layer_idx=target_layer_idx,
    )
    rotation_split_utils.configure_down_proj_online_rotation(
        model,
        qlayers,
        args,
        apply_weight_rotation=True,
        target_layer_idx=target_layer_idx,
    )
    if plot_counts:
        vis_utils.plot_split_online_rotation_important_counts(
            split_important_counts,
            args,
        )
    return split_important_counts

def _smooth_result_path(args):
    return os.path.join(
        args.save_smooth_result_dir,
        args.model.split('/')[-1],
        'smooth_results.pt',
    )


def _backward_result_path(args):
    filename = getattr(args, "backward_result_filename", "backward_result.pt")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("backward_result_filename must be a non-empty string")
    filename = filename.strip()
    if ntpath.splitdrive(filename)[0] or "/" in filename or "\\" in filename:
        raise ValueError(
            "backward_result_filename must not contain path components or a drive prefix"
        )
    if not filename.lower().endswith(".pt"):
        raise ValueError("backward_result_filename must end with .pt")
    return os.path.join(
        args.save_backward_result_dir,
        args.model.split('/')[-1],
        filename,
    )


def _smooth_search_checkpoint_path(args):
    return os.path.join(
        args.save_smooth_result_dir,
        args.model.split('/')[-1],
        'smooth_search_checkpoint.pt',
    )


def _artifact_identity(path):
    if not path:
        return None
    normalized = os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))
    if not os.path.exists(normalized):
        return {'path': normalized, 'exists': False}
    if os.path.isfile(normalized):
        stat = os.stat(normalized)
        return {
            'path': normalized,
            'size': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns),
        }
    tracked = []
    for name in sorted(os.listdir(normalized)):
        lower_name = name.lower()
        if not (
            lower_name == 'config.json'
            or lower_name.endswith('.safetensors')
            or lower_name.endswith('.bin')
            or lower_name.endswith('.index.json')
        ):
            continue
        item_path = os.path.join(normalized, name)
        if os.path.isfile(item_path):
            stat = os.stat(item_path)
            tracked.append((name, int(stat.st_size), int(stat.st_mtime_ns)))
    return {'path': normalized, 'files': tracked}


def _smooth_search_signature(args, layer_count):
    backward_path = None
    if getattr(args, 'save_backward_result_dir', None):
        backward_path = _backward_result_path(args)
    return {
        'version': 5,
        'tlq_definition': 'grad_input_model_padding_v2',
        'profile_layer_index': 0,
        'padding_side': str(getattr(args, '_resolved_tlq_padding_side', None)),
        'candidate_quant_scope': 'target_linear_only',
        'post_selection_propagation': 'whole_layer_wa',
        'loss_reduction': 'global_token',
        'model': args.model,
        'model_artifact': _artifact_identity(getattr(args, 'model_path', None)),
        'model_revision': str(getattr(args, 'revision', None)),
        'backward_artifact': _artifact_identity(backward_path),
        'layer_count': int(layer_count),
        'nsamples': int(args.nsamples),
        'seed': int(args.seed),
        'cal_dataset': str(args.cal_dataset),
        'question_file': str(getattr(args, 'question_file', None)),
        'caption_file': str(getattr(args, 'caption_file', None)),
        'calibration_artifact': _artifact_identity(
            getattr(args, 'caption_file', None)
            or getattr(args, 'question_file', None)
        ),
        'image_base': str(getattr(args, 'image_base', None)),
        'processor_identity': getattr(args, '_tlq_processor_identity', None),
        'smooth_token_importance_mode': str(
            getattr(args, 'smooth_token_importance_mode', 'tlq_topk')
        ),
        'smooth_tlq_topk_ratio': float(
            getattr(args, 'smooth_tlq_topk_ratio', 0.5)
        ),
        'w_bits': int(args.w_bits),
        'a_bits': int(args.a_bits),
        'w_asym': bool(getattr(args, 'w_asym', False)),
        'w_clip': bool(getattr(args, 'w_clip', True)),
        'w_groupsize': int(getattr(args, 'w_groupsize', -1)),
        'int8_down_proj': bool(getattr(args, 'int8_down_proj', False)),
        'a_asym': bool(getattr(args, 'a_asym', False)),
        'a_groupsize': int(getattr(args, 'a_groupsize', -1)),
        'a_clip_ratio': float(args.a_clip_ratio),
        'lma_clip_ratio': float(args.lma_clip_ratio),
        'quant_contract': dict(getattr(args, '_tlq_quant_config', {})),
        'backward_result_filename': getattr(
            args, 'backward_result_filename', 'backward_result.pt'
        ),
        'ratio_grid_size': 20,
        'loss_mode': 'tlq_topk_mae',
    }


def _tlq_calibration_signature(args):
    """Identity of the data and Top-K definition behind a TLQ backward cache."""
    return {
        'definition': 'grad_input_model_padding_v2',
        'profile_layer_index': 0,
        'padding_side': str(getattr(args, '_resolved_tlq_padding_side', None)),
        'nsamples': int(args.nsamples),
        'seed': int(args.seed),
        'cal_dataset': str(args.cal_dataset),
        'question_file': str(getattr(args, 'question_file', None)),
        'caption_file': str(getattr(args, 'caption_file', None)),
        'calibration_artifact': _artifact_identity(
            getattr(args, 'caption_file', None)
            or getattr(args, 'question_file', None)
        ),
        'image_base': str(getattr(args, 'image_base', None)),
        'processor_identity': getattr(args, '_tlq_processor_identity', None),
        'topk_ratio': float(getattr(args, 'smooth_tlq_topk_ratio', 0.5)),
    }


def _validate_smooth_result_payload(payload, args, layer_count):
    if not isinstance(payload, dict) or 'scale' not in payload:
        raise ValueError('Strict TLQ smooth cache must be a dictionary with scales')
    expected = _smooth_search_signature(args, layer_count)
    if payload.get('metadata') != expected:
        raise ValueError(
            'Strict TLQ smooth cache metadata does not match this run. '
            f'Expected {expected}, found {payload.get("metadata")}. Rebuild the cache.'
        )
    return payload


def _load_smooth_search_checkpoint(args, layer_count):
    path = _smooth_search_checkpoint_path(args)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'smooth_search_resume=True but checkpoint does not exist: {path}'
        )
    payload = torch.load(path, weights_only=False, map_location='cpu')
    expected = _smooth_search_signature(args, layer_count)
    if payload.get('signature') != expected:
        raise ValueError(
            'Smooth-search checkpoint does not match the current calibration setting. '
            f'Expected {expected}, found {payload.get("signature")}. '
            'Set smooth_search_resume: false to start a new search.'
        )
    next_layer = int(payload['next_layer'])
    if next_layer < 0 or next_layer > layer_count:
        raise ValueError(f'Invalid checkpoint next_layer={next_layer}')
    return path, next_layer, payload['smooth_results'], payload['inps']


def _save_smooth_search_checkpoint(args, layer_count, next_layer, smooth_results, inps):
    path = _smooth_search_checkpoint_path(args)
    atomic_torch_save(
        {
            'signature': _smooth_search_signature(args, layer_count),
            'next_layer': int(next_layer),
            'smooth_results': utils.detach_nested_tensor_to_cpu(smooth_results),
            'inps': utils.detach_nested_tensor_to_cpu(inps),
        },
        path,
    )
    logging.info('[smooth_scale] checkpoint saved: next_layer=%d path=%s', next_layer, path)
    return path


def _model_cuda_device_indices(model):
    device_map = getattr(model, 'hf_device_map', None) or {}
    device_indices = set()
    for device in device_map.values():
        normalized = utils.normalize_device(device)
        if normalized.type == 'cuda':
            device_indices.add(0 if normalized.index is None else normalized.index)
    if not device_indices and torch.cuda.is_available():
        device_indices.add(torch.cuda.current_device())
    return sorted(device_indices)


def _check_smooth_gpu_memory(model, args, phase):
    if not torch.cuda.is_available():
        return
    min_free_gb = float(getattr(args, 'smooth_min_free_gpu_gb', 1.0))
    if min_free_gb <= 0:
        return
    low_devices = []
    summaries = []
    for device_idx in _model_cuda_device_indices(model):
        free, total = torch.cuda.mem_get_info(device_idx)
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        summaries.append(f'cuda:{device_idx} free={free_gb:.2f}GB total={total_gb:.2f}GB')
        if free_gb < min_free_gb:
            low_devices.append((device_idx, free_gb, total_gb))
    logging.info('[smooth_scale] GPU memory before %s: %s', phase, '; '.join(summaries))
    if low_devices:
        details = ', '.join(
            f'cuda:{idx} free={free_gb:.2f}GB total={total_gb:.2f}GB'
            for idx, free_gb, total_gb in low_devices
        )
        raise RuntimeError(
            f'Insufficient free GPU memory before smooth scale {phase}: {details}. '
            f'Required at least {min_free_gb:.2f}GB free per model GPU. '
            'Stop other GPU jobs or lower max_memory_per_gpu in the YAML.'
        )


def _log_smooth_gpu_memory(model, label):
    if not torch.cuda.is_available():
        return
    summaries = []
    for device_idx in _model_cuda_device_indices(model):
        free, total = torch.cuda.mem_get_info(device_idx)
        allocated = torch.cuda.memory_allocated(device_idx)
        reserved = torch.cuda.memory_reserved(device_idx)
        summaries.append(
            f'cuda:{device_idx} free={free / (1024 ** 3):.2f}GB '
            f'alloc={allocated / (1024 ** 3):.2f}GB '
            f'reserved={reserved / (1024 ** 3):.2f}GB '
            f'total={total / (1024 ** 3):.2f}GB'
        )
    logging.info('[smooth_scale] GPU memory %s: %s', label, '; '.join(summaries))


def _set_smooth_gradient_checkpointing(model, enabled):
    if not enabled:
        return False
    target = model
    if hasattr(target, "gradient_checkpointing_enable"):
        target.gradient_checkpointing_enable()
        if hasattr(target, "config"):
            target.config.use_cache = False
        return True
    language_model = getattr(model, "language_model", None)
    if language_model is None and hasattr(model, "model"):
        language_model = getattr(model.model, "language_model", None)
    if language_model is not None and hasattr(language_model, "gradient_checkpointing_enable"):
        language_model.gradient_checkpointing_enable()
        if hasattr(language_model, "config"):
            language_model.config.use_cache = False
        return True
    return False


if __name__ == '__main__':
    args = utils.parser_gen()
    model, image_processor = model_utils.get_model(
        args.model,
        args.model_path,
        device_map=getattr(args, 'device_map', 'auto'),
        max_memory_per_gpu=getattr(args, 'max_memory_per_gpu', None),
    )
    if args.scale_smooth:
        args._tlq_quant_config = _build_tlq_quant_config(model, args)
        logging.info(
            '[smooth_scale] search/replay quantization contract: %s',
            args._tlq_quant_config,
        )
        args._resolved_tlq_padding_side = model_utils.resolve_tlq_padding_side(
            args.model, getattr(args, 'smooth_tlq_padding_side', 'auto')
        )
        args._tlq_processor_identity = model_utils.get_processor_identity(image_processor)
        logging.info(
            '[smooth_scale] strict TLQ padding side: %s',
            args._resolved_tlq_padding_side,
        )
    model.eval()
    model = utils.move_model_to_cpu(model)
    utils.cleanup_memory(verbos=False)

    if args.error_check:
        logging.info('[error_check] keeping a copy of the original model before calibration')
        original_model = copy.deepcopy(model).eval()
        error_check_inputs = check_utils._prepare_error_check_inputs(
            args, image_processor, model, cached_inputs=None
        )

    if args.quant_vit:
        dataloader = data_utils.get_loaders(
            args.cal_dataset, args.nsamples, args.seed, model.seqlen, 
            args.question_file, args.image_base, getattr(args, 'caption_file', None)
            )
        utils.set_seed(args.seed)
        inpsfp16 = _profile(dataloader, model, image_processor)
        utils.cleanup_memory()

        if args.offline_rotate:
            rotation_utils.fuse_layernorm(model, 'vit')
            if args.online_rotate:
                pass
        if args.quant_method == 'rtn':
            pass

    # Plain LLM RTN path: no SmoothQuant and no rotation.
    if (not args.scale_smooth) and (not args.offline_rotate) \
        and (not args.smoothquant) and args.quant_method == 'rtn':
        if args.a_bits < 16 or args.v_bits < 16:
            logging.info('plain RTN: adding activation quantization wrappers to LM layers')
            qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
            if not qlayers:
                quant_utils.add_actquant(model_utils.get_layers(model))
                
    # Plain LLM SmoothQuant path: SmoothQuant without rotation.
    if args.smoothquant:
        assert not args.scale_smooth
        assert not args.offline_rotate
        if args.load_smooth_result and args.save_smooth_result_dir:
            load_path = os.path.join(args.save_smooth_result_dir,
                                     args.model.split('/')[-1], 'smoothquant_scales.pt')
            smooth_results = torch.load(load_path, weights_only=False)
            logging.info(f'loaded SmoothQuant act scales from {load_path}')
        else:
            dataloader = data_utils.get_loaders(
                args.cal_dataset, args.nsamples, args.seed,
                model.seqlen, args.question_file, args.image_base,
                getattr(args, 'caption_file', None)
            )
            utils.set_seed(args.seed)
            smooth_results = smooth_utils.get_smoothquant_act_scales(
                dataloader, model, image_processor, args
            )
        if args.save_smooth_result_dir and not args.load_smooth_result:
            save_dir = os.path.join(args.save_smooth_result_dir, args.model.split('/')[-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, 'smoothquant_scales.pt')
            torch.save(smooth_results, save_path)
            logging.info(f'SmoothQuant act scales saved at {save_path}')
        logging.info('plain SmoothQuant: applying scales to LM layers')
        smooth_utils.smooth_lm(model.language_model, smooth_results)
        if args.a_bits < 16 or args.v_bits < 16:
            qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
            if not qlayers:
                quant_utils.add_actquant(model_utils.get_layers(model))

    if args.scale_smooth:
        layers = model_utils.get_layers(model)
        if args.load_smooth_result and args.save_smooth_result_dir:
            load_path = _smooth_result_path(args)
            if not os.path.exists(load_path):
                raise FileNotFoundError(
                    'load_smooth_result=True but the strict TLQ smooth cache is missing: '
                    f'{load_path}. Re-run with load_smooth_result: false.'
                )
            smooth_results = _validate_smooth_result_payload(
                torch.load(load_path, weights_only=False, map_location='cpu'),
                args,
                len(layers),
            )
            logging.info(f'loaded smooth_results from {load_path}')
        else:
            smooth_results = {
                "scale": [],
                "metadata": _smooth_search_signature(args, len(layers)),
            }

            if args.load_backward_result and args.save_backward_result_dir:
                path = _backward_result_path(args)
                save_dict = torch.load(path, weights_only=False, map_location='cpu')
                importance_mode = 'tlq_topk'
                cached_mode = save_dict.get('smooth_token_importance_mode')
                if cached_mode != importance_mode:
                    raise ValueError(
                        f"Backward cache importance mode mismatch: expected {importance_mode}, "
                        f"found {cached_mode}. Rebuild the backward cache."
                    )
                if save_dict.get('tlq_topk_definition') != 'grad_input_model_padding_v2':
                    raise ValueError(
                        'Backward cache does not use the TLQ grad_input Top-K '
                        'definition. Rebuild the backward cache.'
                    )
                if save_dict.get('tlq_calibration_signature') != _tlq_calibration_signature(args):
                    raise ValueError(
                        'Backward cache calibration data or TLQ Top-K ratio '
                        'does not match the current run. Rebuild the backward cache.'
                    )
                grad_avg_dict = save_dict['grad_avg_dict']
                important_tokens = save_dict['important_tokens']
                tlq_topk_positions = save_dict.get('tlq_topk_positions', {})
                if not tlq_topk_positions:
                    raise ValueError(
                        'TLQ Top-K smooth calibration requires tlq_topk_positions in the backward cache. '
                        'Rebuild the backward cache.'
                    )
                inpsfp16 = save_dict['inpsfp16']
                layer_kwargs = save_dict['layer_kwargs']
                inputs = save_dict['inputs']
                vision_masks = save_dict['vision_masks']
                caption_masks = save_dict['caption_masks']
                if save_dict.get('profile_layer_index') != 0:
                    raise ValueError('Backward cache was not captured at decoder layer 0')
                if not (
                    len(inpsfp16) == len(layer_kwargs) == len(inputs)
                    == len(vision_masks) == len(caption_masks) == args.nsamples
                ):
                    raise ValueError('Backward cache sample arrays are not aligned')
                logging.info(f'loaded backward result from {path}')
                del save_dict
                utils.cleanup_memory(verbos=False)
            else:
                dataloader = data_utils.get_loaders(
                    args.cal_dataset, args.nsamples, args.seed, 
                    model.seqlen, args.question_file, args.image_base,
                    getattr(args, 'caption_file', None)
                    )
                utils.set_seed(args.seed)
                
                inpsfp16, layer_kwargs, inputs, vision_masks, caption_masks = smooth_utils._profile(
                    dataloader, model, image_processor, args, sample_module=0, out_mask=True)
                importance_mode = 'tlq_topk'
                grad_cache = smooth_utils.GradCacheHook(
                    'tlq_topk',
                    'per_modal',
                    threshold=0.5,
                    vis_masks=vision_masks,
                    cap_masks=caption_masks,
                    n_samples=args.nsamples,
                    tlq_topk_ratio=float(getattr(args, 'smooth_tlq_topk_ratio', 0.5)),
                    tlq_padding_side=args._resolved_tlq_padding_side,
                )
                grad_cache.register_hooks(layers)

                utils.cleanup_memory()
                # model = model.cuda()
                model = utils.dispatch_model_to_cuda(model)
                checkpointing_enabled = _set_smooth_gradient_checkpointing(
                    model,
                    bool(getattr(args, 'smooth_gradient_checkpointing', True)),
                )
                logging.info('[smooth_scale] gradient checkpointing enabled: %s', checkpointing_enabled)
                _check_smooth_gpu_memory(model, args, 'backward')
                backward_input_device = utils.model_first_device(model)
                with torch.enable_grad():
                    for i in tqdm(range(args.nsamples)):
                        if bool(getattr(args, 'smooth_log_gpu_memory', True)):
                            _log_smooth_gpu_memory(model, f'before backward sample {i + 1}/{args.nsamples}')
                        input = utils.move_nested_tensor_to_device(
                            inputs[i],
                            backward_input_device,
                        )
                        input["use_cache"] = False
                        output = model(**input)
                        loss = output[0] / args.nsamples
                        with save_on_cpu():
                            loss.backward()
                        # TLQ Top-K consumes Linear grad_input and uses the
                        # same backward pass to derive vision/answer reweight.
                        model.zero_grad(set_to_none=True)
                        del input, output, loss
                        utils.cleanup_memory(verbos=False)
                        if bool(getattr(args, 'smooth_log_gpu_memory', True)):
                            _log_smooth_gpu_memory(model, f'after backward sample {i + 1}/{args.nsamples}')

                utils.cleanup_memory()
                grad_avg_dict = grad_cache.get_avg_grad_dict()
                important_tokens = grad_cache.important_tokens
                tlq_topk_positions = grad_cache.get_tlq_topk_positions()
                grad_cache.remove_hooks()
                del grad_cache
                model = utils.move_model_to_cpu(model)
            if args.save_backward_result_dir and not args.load_backward_result:
                save_dir = os.path.join(args.save_backward_result_dir, args.model.split('/')[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                path = _backward_result_path(args)
                save_dict = {'grad_avg_dict': grad_avg_dict, 'important_tokens': important_tokens, 
                            'tlq_topk_positions': tlq_topk_positions,
                            'smooth_token_importance_mode': importance_mode,
                            'tlq_topk_definition': 'grad_input_model_padding_v2',
                            'profile_layer_index': 0,
                            'tlq_padding_side': args._resolved_tlq_padding_side,
                            'tlq_calibration_signature': _tlq_calibration_signature(args),
                            'inpsfp16': utils.detach_nested_tensor_to_cpu(inpsfp16),
                            'layer_kwargs': utils.detach_nested_tensor_to_cpu(layer_kwargs),
                            'inputs': utils.detach_nested_tensor_to_cpu(inputs), 
                            'vision_masks': vision_masks, 'caption_masks': caption_masks}
                torch.save(save_dict, path)
                logging.info(f'backward result saved at {path}')

            importance_mode = 'tlq_topk'
            attn_list = []
            mlp_list = []

            for key_name in grad_avg_dict:
                vis_avg = float(grad_avg_dict[key_name]["vis_avg_grad"])
                cap_avg = float(grad_avg_dict[key_name]["cap_avg_grad"])
                if not np.isfinite(vis_avg) or not np.isfinite(cap_avg) or cap_avg == 0:
                    raise ValueError(
                        f'Non-finite TLQ modality gradient statistic for {key_name}'
                    )
                if "down_" in key_name or "w2" in key_name:
                    mlp_list.append(vis_avg / cap_avg)
                if "o_proj" in key_name or "wo" in key_name:
                    attn_list.append(vis_avg / cap_avg)

            if not attn_list or not mlp_list:
                raise ValueError('Strict TLQ requires attention and MLP gradient ratios')
            attn_median = float(np.median(attn_list))
            mlp_median = float(np.median(mlp_list))
            if not np.isfinite(attn_median) or not np.isfinite(mlp_median):
                raise ValueError('Strict TLQ global modality medians must be finite')

            inps = inpsfp16
            start_layer = 0
            profile_layer_index = 0
            checkpoint_path = None
            if bool(getattr(args, 'smooth_search_checkpoint', False)) and not bool(
                getattr(args, 'smooth_search_resume', False)
            ):
                stale_checkpoint = _smooth_search_checkpoint_path(args)
                if os.path.exists(stale_checkpoint):
                    os.remove(stale_checkpoint)
                    logging.info(
                        '[smooth_scale] removed stale checkpoint before a fresh search: %s',
                        stale_checkpoint,
                    )
            if bool(getattr(args, 'smooth_search_resume', False)):
                checkpoint_path, start_layer, smooth_results, inps = (
                    _load_smooth_search_checkpoint(args, len(layers))
                )
                logging.info(
                    '[smooth_scale] resumed checkpoint at layer %d/%d from %s',
                    start_layer, len(layers), checkpoint_path,
                )
            elif start_layer != profile_layer_index:
                raise RuntimeError('Fresh TLQ replay must begin at decoder layer 0')
            layer_progress = tqdm(
                range(start_layer, len(layers)),
                desc='Per layer smoothing...',
                initial=start_layer,
                total=len(layers),
            )
            for i in layer_progress:
                layer_started = time.perf_counter()
                # get input of every linear first
                current_device = utils.layer_device(model, i)
                layer = layers[i]
                layer = layer.to(current_device)
                # Keep the complete calibration set on CPU. batch1_infer moves only
                # the current sample to the layer device, avoiding multi-GiB peaks.
                utils.cleanup_memory(verbos=False)
                current_layer_kwargs = layer_kwargs

                named_linears = utils.get_named_linears(layer)

                def cache_input_hook(m, x, y, name, feat_dict):
                    x = x[0]
                    x = x.detach().cpu()
                    feat_dict[name].append(x)

                input_feat = defaultdict(list)
                handles = []
                for name in named_linears:
                    handles.append(
                        named_linears[name].register_forward_hook(
                            functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                        )
                    )
                
                pass_act_quant = True
                if pass_act_quant:
                    smooth_utils.batch1_infer(layer, inps, current_layer_kwargs)
                else:
                    inps = smooth_utils.batch1_infer(layer, inps, current_layer_kwargs)

                
                for h in handles:
                    h.remove()
                utils.cleanup_memory(verbos=False)

                # min value protection for modal-level reweight
                scale_reweight_ratio_dict = {'attn': None, 'mlp': None}
                for key, value in grad_avg_dict.items():
                    item_list = key.split(".")
                    if str(i) in item_list:
                        if "wo" in item_list or "o_proj" in item_list: 
                            scale_reweight_ratio_dict["attn"] = max((value["vis_avg_grad"] / value["cap_avg_grad"]), attn_median)
                        elif "w2" in item_list or "down_proj" in item_list:
                            scale_reweight_ratio_dict["mlp"] = max((value["vis_avg_grad"] / value["cap_avg_grad"]), mlp_median)

                if any(
                    ratio is None or not np.isfinite(float(ratio))
                    for ratio in scale_reweight_ratio_dict.values()
                ):
                    raise ValueError(
                        f'Layer {i} is missing finite TLQ attention/MLP modality ratios'
                    )

                # Candidate scope remains source-faithful and target-only;
                # its numeric quantization contract matches final RTN.
                layer_important_tokens = {}
                token_positions = tlq_topk_positions
                for k in token_positions:
                    if str(i) in k.split('.'):
                        new_k = k.replace(f'layers.{i}.', '')
                        layer_important_tokens[new_k] = token_positions[k]
                scales_list = smooth_utils.auto_scale_block_wa(
                    layer,
                    current_layer_kwargs,
                    w_bit=args.w_bits,
                    a_bit=args.a_bits,
                    input_feat=input_feat,
                    ans_mask=caption_masks,
                    vis_mask=vision_masks,
                    reweight_ratio_dict=scale_reweight_ratio_dict,
                    loss_mode='tlq_topk_mae',
                    device=current_device,
                    important_tokens=layer_important_tokens,
                    layer_id=i,
                    tlq_padding_side=args._resolved_tlq_padding_side,
                    search_parallel=bool(getattr(args, 'smooth_search_parallel', False)),
                    search_max_workers=int(getattr(args, 'smooth_search_max_workers', 4)),
                    search_min_free_gpu_gb=float(
                        getattr(args, 'smooth_search_min_free_gpu_gb', 3.0)
                    ),
                    quant_config=args._tlq_quant_config,
                )
                
                qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
                assert not qlayers
                if pass_act_quant:
                    new_layer = copy.deepcopy(layer)
                    # Search replay must not mutate the live model. Apply the
                    # candidate scales only to the temporary replay layer; the
                    # complete live model receives every saved scale once below.
                    smooth_utils.apply_scale(new_layer, scales_list)
                    new_layer = smooth_utils._quantize_tlq_layer_wa(
                        new_layer,
                        args.w_bits,
                        args.a_bits,
                        copy_block=False,
                        quant_config=args._tlq_quant_config,
                    )
                    inps = smooth_utils.batch1_infer(new_layer, inps, current_layer_kwargs)
                    del new_layer
                layer = layer.cpu()
                del input_feat
                utils.cleanup_memory(verbos=False)
                
                smooth_results["scale"] += utils.append_str_prefix(
                    scales_list, utils.get_op_name(model.model, layer) + "."
                )
                if bool(getattr(args, 'smooth_search_checkpoint', False)):
                    checkpoint_path = _save_smooth_search_checkpoint(
                        args, len(layers), i + 1, smooth_results, inps
                    )
                logging.info(
                    '[smooth_scale] completed layer %d/%d elapsed=%.1fs',
                    i + 1, len(layers), time.perf_counter() - layer_started,
                )

        if args.save_smooth_result_dir and not args.load_smooth_result:
            smooth_results["metadata"] = _smooth_search_signature(args, len(layers))
            save_dir = os.path.join(args.save_smooth_result_dir, args.model.split('/')[-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = _smooth_result_path(args)
            atomic_torch_save(smooth_results, save_path)
            logging.info(f'smooth_results saved at {save_path}')
            if bool(getattr(args, 'smooth_search_checkpoint', False)):
                checkpoint_path = _smooth_search_checkpoint_path(args)
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    logging.info('[smooth_scale] removed completed checkpoint %s', checkpoint_path)
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        assert not qlayers
        smooth_utils.apply_scale(model.model, smooth_results["scale"], type_model=type(model))
        # smooth_utils.smooth_lm(model.model, smooth_results)
        if not args.offline_rotate:
            quant_utils.add_actquant(model_utils.get_layers(model))
    
        if args.profile:
            results = vis_utils.collect_layer_quant_errors(
                model=model,
                inputs=inputs,
                vision_mask=vision_masks,
                text_mask=caption_masks,
                n_bits=8,
            )
            save_dir = os.path.join(args.profile_dir, args.model.split('/')[-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, "layerwise_relative_error.pdf")
            vis_utils.plot_quant_errors(results, save_path)
            

    # llm activation rotation and quantization
    Q = None
    if args.offline_rotate:
        rotation_utils.fuse_layernorm(model, 'llm')
        R1 = None
        if args.spinquant:
            load_path = "cache/rotation/R_w4a8_scale.bin"
            R_dict = torch.load(load_path, map_location='cpu')
            R1 = R_dict['R1']
            R1 = R1.to(dtype=torch.float64, device=utils.get_dev())
            logging.info(f'loaded SpinQuant result R1 from {load_path}')
        if getattr(args, "split_r1_rotate", False):
            R1 = check_utils.load_or_collect_r1_split_rotation(
                model,
                image_processor,
                args,
            )
            logging.info('using shared whole-hidden split-R1 rotation')
        assert not (args.online_rotate and args.split_rotate)
        Q = rotation_utils.rotate_model(model, 'llm', args.online_rotate, R1)
        utils.cleanup_memory()
        split_online_gradients = None
        if args.split_rotate:
            split_online_gradients = check_utils.load_or_collect_online_channel_split_gradients(
                model,
                image_processor,
                args,
            )
        if args.profile:
            assert args.profile_dir
            if not os.path.exists(args.profile_dir):
                os.mkdir(args.profile_dir)
            target_dir = os.path.join(args.profile_dir, args.model.split('/')[-1])
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            rotate_ckpt = model.state_dict()
            rotate_dict = {'Q':Q, 'model': rotate_ckpt}
            torch.save(rotate_dict, os.path.join(target_dir, 'no_fuse_rotate_model.pth'))
            logging.info(f"Rotated model saved at: {os.path.join(target_dir, 'rotate_model.pth')}")
        quant_utils.add_actquant(model_utils.get_layers(model)) #Add Activation Wrapper to only the LM model
        if args.online_rotate:
            qlayers = quant_utils.find_qlayers(model)
            for name in qlayers:# for clip, todo here
                if 'down_proj' in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.text_config.intermediate_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                if 'o_proj' in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.text_config.num_attention_heads)
                    qlayers[name].online_partial_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].had_dim = model.config.text_config.hidden_size//model.config.text_config.num_attention_heads
                    qlayers[name].fp32_had = args.fp32_had
        if args.split_error_sweep:
            check_utils.run_split_error_sweep(
                baseline_model=original_model,
                current_model=model,
                inputs=error_check_inputs,
                args=args,
                rotation_matrix=Q,
                gradient_magnitudes=split_online_gradients,
                apply_split_rotation=lambda target_layer_idx=None: _configure_split_rotation(
                    model,
                    split_online_gradients,
                    args,
                    plot_counts=False,
                    target_layer_idx=target_layer_idx,
                ),
                apply_weight_quantization=lambda target_layer_idx=None: _apply_weight_quantization(
                    model,
                    args,
                    save_qmodel=False,
                    target_layer_idx=target_layer_idx,
                ),
                configure_activation_quantization=lambda target_layer_idx=None: _configure_activation_quantization(
                    model,
                    args,
                    target_layer_idx=target_layer_idx,
                ),
            )
            del original_model, error_check_inputs
            utils.cleanup_memory(verbos=False)
            exit()
        if args.split_rotate:
            split_best_k = check_utils.load_split_error_best_k(args)
            if split_best_k is not None:
                args.online_channel_split_topk_percent_by_layer = split_best_k
            _configure_split_rotation(
                model,
                split_online_gradients,
                args,
                plot_counts=True,
            )

    if args.parrot:
        parrot_samples = 20
        parrot_topk = 10

        parrot_samples = min(args.nsamples, parrot_samples)
        dataloader = data_utils.get_loaders(
            args.cal_dataset,
            parrot_samples,
            args.seed,
            model.seqlen,
            args.question_file,
            args.image_base,
        )
        output_dir = parrot_utils.get_parrot_output_dir(
            args.profile_dir, args.model
        )
        result = parrot_utils.run_parrot_analysis(
            model=model,
            image_processor=image_processor,
            dataloader=dataloader,
            output_dir=output_dir,
            sample_limit=parrot_samples,
            topk=parrot_topk,
            mode=args.parrot_mode
        )
        logging.info(
            "Parrot analysis saved %d samples and %d channel observations to %s",
            result["sample_count"],
            result["observation_count"],
            result["output_dir"],
        )
        exit()

    if args.error_check:
        logging.info('[error_check] comparing original model with current calibration model')
        # check_utils._log_model_output_error(original_model, model, error_check_inputs)
        check_utils._compute_and_log_channel_relative_error(
            original_model, model, error_check_inputs, args,
            phase='after_rotation', rotation_matrix=Q
        )
        utils.cleanup_memory(verbos=False)

    # llm weight quantization
    if args.w_bits < 16:
        _apply_weight_quantization(model, args, save_qmodel=True)

    # Activation quantization configs
    if args.a_bits < 16 or args.v_bits < 16:
        _configure_activation_quantization(model, args)
            
    if args.error_check:
        logging.info('[error_check] comparing original model with current quantization model')
        check_utils._log_model_output_error(original_model, model, error_check_inputs)
        check_utils._compute_and_log_channel_relative_error(
            original_model, model, error_check_inputs, args,
            phase='after_quantization', rotation_matrix=Q
        )
        del original_model, error_check_inputs
        utils.cleanup_memory(verbos=False)
        exit()

    for eval_dataset in args.eval_dataset:
        test_loader = data_utils.get_loaders(eval_dataset)
        dataset_ppl = eval_utils.evaluator(model, test_loader, utils.get_dev(), args, image_processor, eval_dataset)
