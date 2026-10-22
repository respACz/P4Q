from vlmeval.dataset import build_dataset
import json, os, math, random
import numpy as np

def get_loaders(name, nsamples=128, seed=42, seqlen=2048, question_file=None,
                image_base=None, caption_file=None):
    if 'scienceqa' in name.lower():
        return get_vlm_data(name, nsamples, seed, question_file, image_base)
    if name.lower() == 'coco_sharegpt4v':
        return get_coco_sharegpt4v_data(
            nsamples,
            caption_file or question_file,
            image_base,
        )
    else:
        return build_dataset(name)


def _load_json_or_jsonl(data_path):
    if not data_path:
        raise ValueError(
            "COCO_ShareGPT4V requires caption_file (or the legacy question_file alias)"
        )
    data_path = os.path.expanduser(data_path)
    if data_path.endswith('.jsonl'):
        with open(data_path, 'r', encoding='utf-8') as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    raise ValueError(f"Unsupported COCO_ShareGPT4V file type: {data_path}")


def _sharegpt_record_to_message(record, image_base):
    image_paths = record.get('image', [])
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    if not isinstance(image_paths, list):
        raise ValueError("COCO_ShareGPT4V record image must be a string or list")

    conversations = record.get('conversations', [])
    if len(conversations) < 2:
        raise ValueError("COCO_ShareGPT4V record must contain a question and answer")
    question = conversations[0].get('value', conversations[0].get('content', ''))
    answer = conversations[1].get('value', conversations[1].get('content', ''))
    if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError("COCO_ShareGPT4V conversation values must be strings")

    message = [
        dict(type='image', value=os.path.join(image_base, path))
        for path in image_paths
    ]
    message.append(dict(type='text', value=question.replace('<image>', '').strip()))
    message.append(dict(type='textanw', value=answer.strip()))
    return message


def get_coco_sharegpt4v_data(nsamples, caption_file, image_base):
    dataset = _load_json_or_jsonl(caption_file)
    if not image_base:
        raise ValueError("COCO_ShareGPT4V requires image_base")
    if not dataset:
        raise ValueError("COCO_ShareGPT4V calibration data is empty")
    rng = np.random.default_rng(seed=42)
    rng.shuffle(dataset)
    return [
        _sharegpt_record_to_message(dataset[index % len(dataset)], image_base)
        for index in range(nsamples)
    ]
    
def get_vlm_data(name, nsamples, seed, question_file, image_base):
    if 'train' in name.lower():
        def build_prompt_llava_train(line, pathbase):
            question = line['conversations'][0]
            question = question['value'].replace('<image>', '').strip()
            tgt_path = [os.path.join(pathbase, line["image"])]
            anwser = line['conversations'][1]['value'].strip()
            message = [dict(type="image", value=s) for s in tgt_path]
            message.append(dict(type="text", value=question))
            message.append(dict(type="textanw", value=anwser))
            return message
        
        dataloader = []
        dataset = json.load(open(os.path.expanduser(question_file), "r"))
        random.seed(seed)
        indices = random.sample(range(len(dataset)), nsamples)
        for i in indices:
            line = dataset[i]
            dataloader.append(build_prompt_llava_train(line, image_base))
        return dataloader

    else:
        dataset = build_dataset(name)
        return dataset
