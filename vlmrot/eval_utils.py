import torch
from vlmeval.smp import *
from vlmeval.smp import dump, get_rank_and_world_size, load
import json
import os, logging, string, re, statistics, time
from tqdm import tqdm
from vlmrot import model_utils, utils, visual_pruning
import pandas as pd
from tabulate import tabulate

def is_vizwiz_dataset(eval_dataset):
    return str(eval_dataset).lower() == 'vizwiz'

def build_prompt_llava(line, dataset, eval_dataset):
    tgt_path = dataset.dump_image(line)
    question = line['question']
    hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
    if hint is not None:
        question = hint + "\n" + question 
    options = {
        cand: line[cand]
        for cand in string.ascii_uppercase
        if cand in line and not pd.isna(line[cand])
    }
    if is_vizwiz_dataset(eval_dataset):
        question += "\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase."
    else:
        for key, item in options.items():
            question += f"\n{key}. {item}"
        question += "\nAnswer with the option's letter from the given choices directly."
    message = [dict(type="image", value=s) for s in tgt_path]
    message.append(dict(type="text", value=question))
    return message

def output_process(answer):
    if "<s>" in answer:
        answer = answer.replace("<s>", "").strip()
    if "[/INST]" in answer:
        answer = answer.split("[/INST]")[1].strip()
    elif "ASSISTANT:" in answer:
        answer = answer.split("ASSISTANT:")[1].strip()
    elif "assistant\n" in answer:
        answer = answer.split("assistant\n")[1].strip()
    elif "<|end_header_id|>\n\n" in answer:
        answer = answer.split("<|end_header_id|>\n\n")[2].strip()

    if "</s>" in answer:
        answer = answer.split("</s>")[0].strip()
    elif "<|im_end|>" in answer:
        answer = answer.split("<|im_end|>")[0].strip()
    elif "<|eot_id|>" in answer:
        answer = answer.split("<|eot_id|>")[0].strip()
    return answer

def _visual_prune_result_suffix(args):
    return visual_pruning.result_suffix(args)


def calibration_abbreviation(cal_dataset):
    """Return a stable, path-safe short name for a calibration dataset."""
    raw_name = str(cal_dataset).strip()
    known = {
        "scienceqa_train": "SQA",
        "mmmu_dev_val": "MMMU",
        "textvqa_val": "TextVQA",
    }
    normalized = re.sub(r"[^a-z0-9]+", "_", raw_name.lower()).strip("_")
    if normalized in known:
        return known[normalized]

    split_tokens = {"train", "val", "validation", "dev", "test"}
    readable_tokens = [
        token
        for token in re.split(r"[^A-Za-z0-9]+", raw_name)
        if token and token.lower() not in split_tokens
    ]
    abbreviation = "".join(readable_tokens)
    return abbreviation or "Cal"


def eval_run_name(args, visual_prune_suffix=""):
    """Build the cache-owning run directory while preserving legacy non-smooth names."""
    model_name = args.model.split('/')[-1]
    bit_tag = f"W{args.w_bits}A{args.a_bits}"
    if args.scale_smooth:
        calibration_tag = calibration_abbreviation(args.cal_dataset)
        run_name = f"{model_name}_smooth_{bit_tag}_{calibration_tag}{args.nsamples}"
        w_group = int(getattr(args, "w_groupsize", -1))
        a_group = int(getattr(args, "a_groupsize", -1))
        w_group_tag = "all" if w_group == -1 else str(w_group)
        a_group_tag = "all" if a_group == -1 else str(a_group)
        effective_a_clip = min(
            float(getattr(args, "a_clip_ratio", 1.0)),
            float(getattr(args, "lma_clip_ratio", 1.0)),
        )
        clip_tag = f"{effective_a_clip:.6g}".replace("-", "m").replace(".", "p")
        weight_tag = (
            ("Wasym" if bool(getattr(args, "w_asym", False)) else "Wsym")
            + ("MSE" if bool(getattr(args, "w_clip", True)) else "Max")
            + f"g{w_group_tag}"
        )
        activation_tag = (
            ("Aasym" if bool(getattr(args, "a_asym", False)) else "Asym")
            + f"C{clip_tag}g{a_group_tag}"
        )
        down_proj_tag = (
            "DP8" if bool(getattr(args, "int8_down_proj", False)) else "DPbase"
        )
        run_name += f"_{weight_tag}_{activation_tag}_{down_proj_tag}"
    else:
        run_name = model_name + f"_{bit_tag}"
        run_name += "_Smoothquant" if args.smoothquant else ""

    run_name += (
        ("_fullRotate" if (args.offline_rotate and args.online_rotate) else "")
        + ("_offlineRotate" if (
            args.offline_rotate and not args.online_rotate and not args.split_rotate
        ) else "")
        + ("_splitOV" if (args.offline_rotate and args.split_rotate) else "")
        + ("_splitR1" if (args.offline_rotate and args.split_r1_rotate) else "")
        + ("SpinQuant" if args.spinquant else "")
        + ("_rtn" if (
            args.quant_method == "rtn" and (args.a_bits < 16 or args.v_bits < 16)
        ) else "")
        + visual_prune_suffix
    )
    if getattr(args, "eval_generation_mode", "legacy") == "comparable":
        run_name += "_gen-comparable"
    return run_name


def eval_output_dir(args, eval_dataset, visual_prune_suffix=""):
    return os.path.join(
        args.eval_result_dir,
        eval_run_name(args, visual_prune_suffix),
        eval_dataset,
    )


def _synchronize_cuda():
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


def _timing_summary(records):
    measured_records = [record for record in records if not record.get("warmup", False)]
    if not measured_records:
        return {
            "total_sample_count": len(records),
            "warmup_sample_count": len(records),
            "sample_count": 0,
        }

    def summarize(field):
        values = sorted(float(record[field]) for record in measured_records)
        p90_index = min(len(values) - 1, max(0, int(0.9 * len(values)) - 1))
        return {
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "p90_ms": values[p90_index],
            "total_s": sum(values) / 1000.0,
        }

    return {
        "total_sample_count": len(records),
        "warmup_sample_count": len(records) - len(measured_records),
        "sample_count": len(measured_records),
        "preprocess": summarize("preprocess_ms"),
        "generation": summarize("generation_ms"),
        "sample_total": summarize("sample_total_ms"),
        "output_tokens": sum(record["output_tokens"] for record in measured_records),
    }

def infer_data_llava(model, image_processor, dataset, args, eval_dataset):
    if visual_pruning.is_enabled(args) and not visual_pruning.is_supported_model(model):
        raise ValueError(
            "visual_prune is only implemented for llava-1.5, llava-next, and qwen2vl models"
        )
    if (
        getattr(args, "eval_generation_mode", "legacy") == "comparable"
        and not visual_pruning.is_supported_model(model)
    ):
        raise ValueError(
            "comparable generation is implemented for llava-1.5, llava-next, "
            "and qwen2vl models"
        )
    max_samples = getattr(args, "eval_max_samples", None)
    selected_length = len(dataset)
    if max_samples is not None:
        selected_length = min(selected_length, max(0, int(max_samples)))
    sheet_indices = range(selected_length)
    data = dataset.data.iloc[sheet_indices]
    length1 = len(data)
    out_dir = eval_output_dir(
        args,
        eval_dataset,
        _visual_prune_result_suffix(args),
    )
    logging.info(f'eval out dir: {out_dir}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, f'vlm_eval.pkl')
    data_indices = [i for i in data['index']]
    res = {}
    ignore_cache = bool(getattr(args, "eval_ignore_cache", False))
    if os.path.exists(out_file) and not ignore_cache:
        res.update(load(out_file))
    data = data[~data['index'].isin(res)]   # resume, avoiding recompute the results
    length2 = len(data)
    if length1 > length2:
        logging.info(f'Resumed! Original length of {eval_dataset}: {length1}, current: {length2}')
    else:
        logging.info(f'{length1} samples in {eval_dataset}')
    timing_enabled = bool(getattr(args, "eval_timing", False))
    timing_records = []
    timing_file = os.path.join(out_dir, 'timing.jsonl')
    if timing_enabled and (ignore_cache or length1 == length2):
        with open(timing_file, 'w', encoding='utf-8'):
            pass
    for i in tqdm(range(len(data))):
        sample_started = time.perf_counter()
        idx = data.iloc[i]['index']
        message = build_prompt_llava(data.iloc[i], dataset, eval_dataset)
        inputs = model_utils.message_to_prompt(
        message, image_processor, type(model)
        )
        if timing_enabled:
            _synchronize_cuda()
        generation_started = time.perf_counter()
        generation_stats = None
        if type(model) in (
            model_utils.LLAVA_HF,
            model_utils.LLAVA_NEXT_HF,
            model_utils.QWEN2_VL,
            getattr(model_utils, "QWEN2_5_VL", None),
        ):
            if type(model) in (model_utils.QWEN2_VL, getattr(model_utils, "QWEN2_5_VL", None)):
                model.generation_config.temperature = None
                model.generation_config.top_k = None
                model.generation_config.top_p = None
            if visual_pruning.is_enabled(args):
                output, prune_stats = visual_pruning.generate_with_visual_pruning(
                    model,
                    inputs,
                    args,
                    max_new_tokens=16,
                    query_text="\n".join(
                        str(item.get("value", ""))
                        for item in message
                        if item.get("type") == "text"
                    ),
                    tokenizer=getattr(image_processor, "tokenizer", None),
                )
                if (i + 1) % max(1, int(getattr(args, "visual_prune_log_every", 50))) == 0:
                    logging.info(
                        "visual pruning sample %d/%d: visual %d -> %d, prompt %d -> %d",
                        i + 1,
                        len(data),
                        prune_stats.original_visual_tokens,
                        prune_stats.kept_visual_tokens,
                        prune_stats.original_prompt_tokens,
                        prune_stats.kept_prompt_tokens,
                    )
                    if prune_stats.recycled_visual_tokens > 0:
                        logging.info(
                            "visual pruning recycle: mode=%s alpha=%.3f window=%d dropped=%d merged_into=%d",
                            getattr(args, "visual_prune_recycle_mode", "knn"),
                            float(getattr(args, "visual_prune_recycle_alpha", 0.25)),
                            int(getattr(args, "visual_prune_recycle_window_size", 64)),
                            prune_stats.recycled_visual_tokens,
                            prune_stats.kept_visual_tokens,
                        )
            else:
                if getattr(args, "eval_generation_mode", "legacy") == "comparable":
                    output, generation_stats = (
                        visual_pruning.generate_without_visual_pruning(
                            model,
                            inputs,
                            max_new_tokens=16,
                        )
                    )
                else:
                    output = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=16,
                        top_p=None,
                        num_beams=1
                    )
        elif type(model) == model_utils.LLAVA_OV_HF:
            output = model.generate(
                **inputs,
                do_sample=False, 
                max_new_tokens=16,
                top_p=None,
                num_beams=1,
                eos_token_id=image_processor.tokenizer.pad_token_id,
                pad_token_id=image_processor.tokenizer.pad_token_id
            )
        if visual_pruning.is_enabled(args):
            generation_stats = prune_stats
        if timing_enabled:
            _synchronize_cuda()
        generation_finished = time.perf_counter()
        if timing_enabled:
            generated_tokens = int(output.shape[-1])
            if (
                getattr(args, "eval_generation_mode", "legacy") == "legacy"
                and not visual_pruning.is_enabled(args)
                and "input_ids" in inputs
            ):
                generated_tokens = max(
                    0,
                    generated_tokens - int(inputs["input_ids"].shape[-1]),
                )
            record = {
                "index": int(idx) if hasattr(idx, "__int__") else str(idx),
                "dataset": str(eval_dataset),
                "generation_mode": getattr(args, "eval_generation_mode", "legacy"),
                "visual_prune": visual_pruning.is_enabled(args),
                "preprocess_ms": (generation_started - sample_started) * 1000.0,
                "generation_ms": (generation_finished - generation_started) * 1000.0,
                "sample_total_ms": (generation_finished - sample_started) * 1000.0,
                "output_tokens": generated_tokens,
                "warmup": i < int(getattr(args, "eval_timing_warmup_samples", 10)),
            }
            if generation_stats is not None:
                record.update({
                    "original_visual_tokens": generation_stats.original_visual_tokens,
                    "kept_visual_tokens": generation_stats.kept_visual_tokens,
                    "original_prompt_tokens": generation_stats.original_prompt_tokens,
                    "kept_prompt_tokens": generation_stats.kept_prompt_tokens,
                })
            timing_records.append(record)
            with open(timing_file, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        response = image_processor.decode(output[0], skip_special_tokens=True)
        response = output_process(response) 
        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)
    dump({k: res[k] for k in data_indices}, out_file)
    if timing_enabled:
        summary = _timing_summary(timing_records)
        summary_file = os.path.join(out_dir, 'timing_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        logging.info('timing summary: %s', json.dumps(summary, ensure_ascii=False))
    return out_file

@torch.no_grad()
def evaluator(model, dataset, device, args, image_processor, eval_dataset):
    model = utils.dispatch_model_to_cuda(model)
    # model = model.cuda()
    if not os.path.exists(args.eval_result_dir):
        os.mkdir(args.eval_result_dir)
    out_file = infer_data_llava(model, image_processor, dataset, args, eval_dataset)
    if bool(getattr(args, "eval_inference_only", False)):
        model = utils.move_model_to_cpu(model)
        utils.cleanup_memory()
        return {"prediction_file": out_file}
    result_file = os.path.join(args.eval_result_dir, 'vlm_result.xlsx')
    if os.path.exists(result_file):
        for file in os.listdir(args.eval_result_dir):
            if file.split('_')[0] == 'vlm':
                remove_file = os.path.join(args.eval_result_dir, file)
                os.remove(remove_file)
                logging.info(f'removed previous file: {remove_file}')
    data_all = load(out_file)
    data = dataset.data

    for x in data['index']:
        assert x in data_all
    data['prediction'] = [str(data_all[x]) for x in data['index']]
    if 'image' in data:
        data.pop('image')
    dump(data, result_file)

    judge_kwargs = {'nproc': 4, 'verbose': True, 'retry': 3}
    eval_results = dataset.evaluate(result_file, **judge_kwargs)

    import json
    logging.info(eval_results)
    if isinstance(eval_results, dict):
        logging.info('\n' + json.dumps(eval_results, indent=4))
    elif isinstance(eval_results, pd.DataFrame):
        if len(eval_results) < len(eval_results.columns):
            eval_results = eval_results.T
        logging.info('\n' + tabulate(eval_results))
    model = utils.move_model_to_cpu(model)
    utils.cleanup_memory()
    return eval_results
