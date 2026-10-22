import transformers
import torch
from PIL import Image

LLAVA_HF = None
LLAVA_NEXT_HF = None
LLAVA_OV_HF = None
QWEN2_VL = None
try:
    LLAVA_HF = transformers.models.llava.modeling_llava.LlavaForConditionalGeneration
    LLAVA_NEXT_HF = transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration
    LLAVA_OV_HF = transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration
    QWEN2_VL = transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration
except ImportError:
    # Keep model-independent utilities importable when generation/runtime
    # dependencies are intentionally stubbed (for example in CPU-only tests).
    pass
QWEN2_5_VL = None
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QWEN2_5_VL
except ImportError:
    pass

LLAVA_HF_MODELS = tuple(
    model_type
    for model_type in (LLAVA_HF, LLAVA_NEXT_HF, LLAVA_OV_HF)
    if model_type is not None
)

QUESTION_REPLACE_MAPPING = {
    "\nOptions:": "\nChoices:",
    "Please select the correct answer from the options above.": "Answer with the letter.",
}


def resolve_tlq_padding_side(model_or_name, requested="auto"):
    """Resolve the padded token coordinates used by the TLQ reference path."""
    requested = str(requested).lower()
    if requested in ("left", "right"):
        return requested
    if requested != "auto":
        raise ValueError("smooth_tlq_padding_side must be auto, left, or right")

    if isinstance(model_or_name, str):
        identity = model_or_name.lower()
    elif isinstance(model_or_name, type):
        identity = model_or_name.__name__.lower()
    else:
        identity = type(model_or_name).__name__.lower()
    if "qwen2-vl" in identity or "qwen2vl" in identity:
        return "left"
    if "llava" in identity:
        return "right"
    raise ValueError(
        f"Cannot infer TLQ padding side for {model_or_name!r}; set it explicitly"
    )


def get_processor_identity(processor):
    """Return deterministic processor/tokenizer fields used by cache signatures."""
    tokenizer = getattr(processor, "tokenizer", None)
    return {
        "processor_class": f"{type(processor).__module__}.{type(processor).__qualname__}",
        "processor_name_or_path": str(getattr(processor, "name_or_path", None)),
        "tokenizer_class": (
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
            if tokenizer is not None else None
        ),
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", None)),
        "tokenizer_padding_side": str(getattr(tokenizer, "padding_side", None)),
    }

def _normalize_device_map(device_map):
    if device_map == "none":
        return None
    return device_map


def _resolve_max_memory(max_memory_per_gpu=None):
    if max_memory_per_gpu is None or str(max_memory_per_gpu).strip() == "":
        return None
    if not torch.cuda.is_available():
        return None
    return {idx: str(max_memory_per_gpu) for idx in range(torch.cuda.device_count())}


def _load_kwargs(device_map="auto", max_memory_per_gpu=None):
    kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    resolved_device_map = _normalize_device_map(device_map)
    if resolved_device_map is not None:
        kwargs["device_map"] = resolved_device_map
        max_memory = _resolve_max_memory(max_memory_per_gpu)
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
    return kwargs


def get_model(model_name, model_path, device_map="auto", max_memory_per_gpu=None):
    """Load a supported LLaVA or Qwen2-VL checkpoint."""
    model_identity = str(model_name).lower()
    if 'qwen2.5-vl' in model_identity:
        from vlmrot.qwen25_p4q import load_model
        return load_model(model_path, device_map, max_memory_per_gpu)
    if 'llava-1.5' in model_identity or 'llava-v1.5' in model_identity:
        return get_llava_v15(model_path, device_map, max_memory_per_gpu)
    if 'llava-v1.6' in model_identity:
        return get_llava_v16(model_path, device_map, max_memory_per_gpu)
    elif 'llava-onevision' in model_identity:
        return get_llava_ov(model_path, device_map, max_memory_per_gpu)
    elif 'qwen2-vl' in model_identity:
        return get_qwen2_vl(model_path, device_map, max_memory_per_gpu)
    else:
        raise ValueError(f'Unknown model {model_name}')


def get_model_shell(model):
    llava_hf = globals().get("LLAVA_HF")
    llava_types = (LLAVA_NEXT_HF, LLAVA_OV_HF) + (
        (llava_hf,) if llava_hf is not None else ()
    )
    if type(model) in llava_types and hasattr(model, "model"):
        return model.model
    return model


def get_language_model(model):
    if type(model) == globals().get("QWEN2_5_VL"):
        return model.model.language_model
    shell = get_model_shell(model)
    language_model = None
    if hasattr(model, "language_model"):
        language_model = model.language_model
    elif hasattr(shell, "language_model"):
        language_model = shell.language_model
    elif hasattr(shell, "layers"):
        language_model = shell

    # Depending on the Transformers release, LLaVA exposes either the bare
    # LlamaModel or a LlamaForCausalLM wrapper as ``language_model``.  FlatQuant
    # needs the bare decoder because its layers, embeddings and final norm live
    # there.  Resolve both layouts without changing the outer model contract.
    for _ in range(3):
        if language_model is None:
            break
        if hasattr(language_model, "layers"):
            return language_model
        language_model = getattr(language_model, "model", None)
    raise ValueError(f"Cannot find language_model on {type(model)}")


def get_vit_config(model):
    if type(model) == QWEN2_5_VL:
        return model.model.visual.config
    model_type = type(model)
    if model_type in LLAVA_HF_MODELS:
        return model.vision_tower.config
    elif model_type == QWEN2_VL:
        return model.visual.config
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
def get_lm_config(model):
    if type(model) in (*LLAVA_HF_MODELS, QWEN2_VL, QWEN2_5_VL):
        return model.config.text_config
    else:
        return model.config

def get_lm_norm_type(model):
    if type(model) == QWEN2_5_VL:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm
        return Qwen2RMSNorm
    if type(model) in (LLAVA_HF, LLAVA_NEXT_HF):
        return transformers.models.llama.modeling_llama.LlamaRMSNorm
    elif type(model) == LLAVA_OV_HF:
        return transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
    elif type(model) == QWEN2_VL:
        return transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2RMSNorm
    else:
        raise ValueError(f'Unknown model type {type(model)}')
    
def get_input_masks(model, inputs):
    if type(model) in (*LLAVA_HF_MODELS, QWEN2_VL, QWEN2_5_VL):
        assert inputs['input_ids'].shape[0] == 1
        image_token_id = getattr(
            model.config,
            "image_token_id",
            getattr(model.config, "image_token_index", None),
        )
        if image_token_id is None:
            raise ValueError("LLaVA/Qwen config must define an image token id")
        vision_mask = (inputs['input_ids'] == image_token_id)
        caption_mask = (inputs['attention_mask'] == 1) & (~vision_mask)
        return vision_mask, caption_mask
    else:
        raise ValueError(f'Unknown model type {type(model)}')
    
def _clean_question(text):
    text = text.strip()
    for src, dst in QUESTION_REPLACE_MAPPING.items():
        text = text.replace(src, dst)
    return text


def build_answer_only_labels(inputs, question_inputs, tokenizer=None, config=None):
    labels = inputs["input_ids"].clone()
    prompt_len = min(question_inputs["input_ids"].shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100

    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        labels[attention_mask == 0] = -100

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100

    image_token_id = getattr(config, "image_token_id", None)
    if image_token_id is not None:
        labels[labels == image_token_id] = -100

    video_token_id = getattr(config, "video_token_id", None)
    if video_token_id is not None:
        labels[labels == video_token_id] = -100

    return labels


def _message_has_answer(message):
    return any(item.get("type") == "textanw" for item in message)


def build_qwen2vl_train_inputs(message, image_processor, model, device=None):
    from qwen_vl_utils import process_vision_info

    content = []
    answer = ""
    for msg in message:
        if msg["type"] == "image":
            content.append({"type": "image", "image": msg["value"]})
        elif msg["type"] == "text":
            content.append({"type": "text", "text": _clean_question(msg["value"])})
        elif msg["type"] == "textanw":
            answer += msg["value"].strip()

    if not answer:
        raise ValueError("Training sample does not contain a textanw answer field")

    question_conversation = [{"role": "user", "content": content}]
    full_conversation = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]

    question_prompt = image_processor.apply_chat_template(
        question_conversation,
        add_generation_prompt=True,
    )
    full_prompt = image_processor.apply_chat_template(
        full_conversation,
        add_generation_prompt=False,
    )

    image_inputs, video_inputs = process_vision_info(full_conversation)
    inputs = image_processor(
        text=[full_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    question_inputs = image_processor(
        text=[question_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if device is not None:
        inputs = inputs.to(device, torch.float16)
        question_inputs = question_inputs.to(device, torch.float16)

    inputs["labels"] = build_answer_only_labels(
        inputs,
        question_inputs,
        getattr(image_processor, "tokenizer", None),
        model.config,
    )
    return inputs


def build_llava_train_inputs(message, image_processor, model, device=None):
    model_type = type(model)
    if model_type not in LLAVA_HF_MODELS:
        raise ValueError(f'Unsupported LLaVA model type: {model_type}')

    content = []
    images = []
    answer = ''
    for msg in message:
        if msg['type'] == 'image':
            if model_type == LLAVA_NEXT_HF:
                content.append({'type': 'image'})
            else:
                content.append({'type': 'image', 'image': None})
            images.append(Image.open(msg['value']).convert('RGB'))
        elif msg['type'] == 'text':
            content.append({'type': 'text', 'text': _clean_question(msg['value'])})
        elif msg['type'] == 'textanw':
            answer += msg['value'].strip()

    if not answer:
        raise ValueError('Training sample does not contain a textanw answer field')

    question_conversation = [{'role': 'user', 'content': content}]
    full_conversation = [
        {'role': 'user', 'content': content},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
    ]

    def process(conversation, add_generation_prompt):
        if model_type == LLAVA_HF:
            user_text = next(
                (
                    item['text']
                    for turn in conversation
                    if turn['role'] == 'user'
                    for item in turn['content']
                    if item['type'] == 'text'
                ),
                '',
            )
            image_token = getattr(image_processor, 'image_token', '<image>')
            prompt = '\n'.join([image_token] * len(images) + [user_text])
            prompt = f"USER: {prompt}\nASSISTANT:"
            if not add_generation_prompt:
                assistant_text = next(
                    (
                        item['text']
                        for turn in conversation
                        if turn['role'] == 'assistant'
                        for item in turn['content']
                        if item['type'] == 'text'
                    ),
                    '',
                )
                prompt += f" {assistant_text}"
            return image_processor(
                images=images,
                text=prompt,
                return_tensors='pt',
            )
        prompt = image_processor.apply_chat_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
        )
        if model_type == LLAVA_NEXT_HF:
            return image_processor(prompt, images, return_tensors='pt')
        return image_processor(text=prompt, images=images, return_tensors='pt')

    inputs = process(full_conversation, add_generation_prompt=False)
    question_inputs = process(question_conversation, add_generation_prompt=True)
    if device is not None:
        inputs = inputs.to(device, torch.float16)
        question_inputs = question_inputs.to(device, torch.float16)

    inputs['labels'] = build_answer_only_labels(
        inputs,
        question_inputs,
        getattr(image_processor, 'tokenizer', None),
        model.config,
    )
    return inputs


def build_loss_inputs(message, image_processor, model, device=None):
    if type(model) in LLAVA_HF_MODELS and _message_has_answer(message):
        return build_llava_train_inputs(
            message,
            image_processor,
            model,
            device if device is not None else 'cuda',
        )
    if type(model) in (QWEN2_VL, QWEN2_5_VL) and _message_has_answer(message):
        return build_qwen2vl_train_inputs(
            message,
            image_processor,
            model,
            device if device is not None else "cuda",
        )

    inputs = message_to_prompt(
        message,
        image_processor,
        type(model),
        device=device if device is not None else "cuda",
    )
    vision_mask, _ = get_input_masks(model, inputs)
    labels = inputs["input_ids"].clone()
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        labels[attention_mask == 0] = -100
    labels[vision_mask] = -100
    tokenizer = getattr(image_processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    inputs["labels"] = labels
    return inputs
    
def get_embeddings(model):
    if type(model) in (*LLAVA_HF_MODELS, QWEN2_VL, QWEN2_5_VL):
        return [get_language_model(model).embed_tokens]
    else:
        raise ValueError(f'Unknown model type {type(model)}')

def get_projector(model) -> list[torch.nn.Module]:
    if type(model) == QWEN2_5_VL:
        return [model.model.visual.merger]
    model_type = type(model)
    if model_type in LLAVA_HF_MODELS:
        return [model.multi_modal_projector]
    elif model_type == QWEN2_VL:
        return [model.visual.merger]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
def get_vit_layers(model):
    if type(model) == QWEN2_5_VL:
        return list(model.model.visual.blocks)
    model_type = type(model)
    if model_type in LLAVA_HF_MODELS:
        return [layer for layer in model.vision_tower.vision_model.encoder.layers]
    elif model_type == QWEN2_VL:
        return [layer for layer in model.visual.blocks]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
def get_transformer_layers(model):
    if type(model) in (*LLAVA_HF_MODELS, QWEN2_VL, QWEN2_5_VL):
        return [layer for layer in get_language_model(model).layers]
    else:
        raise ValueError(f'Unknown model type {type(model)}')
    
def get_layers(model):
    llava_hf = globals().get("LLAVA_HF")
    supported_types = (LLAVA_NEXT_HF, LLAVA_OV_HF, QWEN2_VL, globals().get("QWEN2_5_VL")) + (
        (llava_hf,) if llava_hf is not None else ()
    )
    if type(model) in supported_types:
        return get_language_model(model).layers
    raise NotImplementedError
    
def get_pre_head_layernorm(model):
    model_type = type(model)
    pre_head_layernorm = get_language_model(model).norm
    norm_type = get_lm_norm_type(model)
    assert isinstance(pre_head_layernorm, norm_type)
    return pre_head_layernorm

def get_lm_head(model):
    if type(model) in (*LLAVA_HF_MODELS, QWEN2_VL, QWEN2_5_VL):
        shell = get_model_shell(model)
        owners = (
            model,
            getattr(model, "language_model", None),
            getattr(shell, "language_model", None),
            get_language_model(model),
        )
        for owner in owners:
            if owner is not None and hasattr(owner, "lm_head"):
                return owner.lm_head
        raise ValueError(f'Cannot find lm_head on {type(model)}')
    else:
        raise ValueError(f'Unknown model type {type(model)}')


def _validate_llava_v15_loading_info(model_path, loading_info):
    missing_keys = list((loading_info or {}).get("missing_keys", ()))
    missing_decoder_keys = [
        name
        for name in missing_keys
        if "language_model" in name
        and any(
            token in name
            for token in ("embed_tokens", ".layers.", ".norm.", "lm_head")
        )
    ]
    if not missing_decoder_keys:
        return

    preview = ", ".join(missing_decoder_keys[:3])
    raise RuntimeError(
        "The LLaVA-1.5 checkpoint is not compatible with Transformers "
        "LlavaForConditionalGeneration: "
        f"{len(missing_decoder_keys)} language-model weights were not loaded "
        f"from '{model_path}' (for example: {preview}). This usually means "
        "the directory contains the original liuhaotian/LLaVA checkpoint, "
        "not the converted llava-hf/llava-1.5-7b-hf checkpoint. Do not train "
        "or quantize this randomly initialized model. Download or convert the "
        "complete Hugging Face checkpoint, including config.json, the model "
        "shards, tokenizer files, processor_config.json, and "
        "preprocessor_config.json."
    )


def get_llava_v15(model_path, device_map="auto", max_memory_per_gpu=None):
    from transformers import LlavaProcessor, LlavaForConditionalGeneration
    model, loading_info = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        output_loading_info=True,
        **_load_kwargs(device_map, max_memory_per_gpu),
    )
    _validate_llava_v15_loading_info(model_path, loading_info)
    model.seqlen = 2048
    try:
        processor = LlavaProcessor.from_pretrained(model_path)
    except OSError as exc:
        if "preprocessor_config.json" not in str(exc):
            raise
        raise OSError(
            f"LLaVA-1.5 processor files are incomplete under '{model_path}'. "
            "Use the complete converted llava-hf/llava-1.5-7b-hf directory; "
            "copying only model weights/config is insufficient because "
            "LlavaProcessor also requires preprocessor_config.json and the "
            "tokenizer/processor files."
        ) from exc
    return model, processor


def get_llava_v16(model_path, device_map="auto", max_memory_per_gpu=None):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, 
        **_load_kwargs(device_map, max_memory_per_gpu),
    )
    model.seqlen = 2048
    processor = LlavaNextProcessor.from_pretrained(model_path)
    return model, processor

def get_llava_ov(model_path, device_map="auto", max_memory_per_gpu=None):
    from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        **_load_kwargs(device_map, max_memory_per_gpu),
    )
    model.seqlen = 2048
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    return model, processor

def get_qwen2_vl(model_path, device_map="auto", max_memory_per_gpu=None):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        **_load_kwargs(device_map, max_memory_per_gpu),
    )
    model.seqlen = 2048
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def message_to_prompt(message, image_processor, model_type, device="cuda"):
    if model_type == LLAVA_HF:
        images = []
        question_parts = []
        for msg in message:
            if msg['type'] == 'image':
                images.append(Image.open(msg['value']).convert('RGB'))
            elif msg['type'] == 'text':
                question_parts.append(msg['value'])
        image_token = getattr(image_processor, 'image_token', '<image>')
        prompt = '\n'.join(
            [image_token] * len(images) + question_parts
        )
        prompt = f"USER: {prompt}\nASSISTANT:"
        inputs = image_processor(
            images=images,
            text=prompt,
            return_tensors='pt',
        )
        if device is not None:
            inputs = inputs.to(device, torch.float16)
        return inputs

    elif model_type == LLAVA_NEXT_HF:
        content, images = [], []
        for msg in message:
            if msg['type'] == 'image':
                content.append({"type": "image"})
                images.append(Image.open(msg['value']).convert('RGB'))
            elif msg['type'] == 'text':
                content.append({"type": "text", "text": msg['value']})
        conversation = [{"role": "user", "content": content}]
        prompt = image_processor.apply_chat_template(conversation,
                                                    add_generation_prompt=True)
        inputs = image_processor(prompt, images, return_tensors='pt')
        if device is not None:
            inputs = inputs.to(device, torch.float16)
        return inputs

    elif model_type == LLAVA_OV_HF:
        content, images = [], []
        for msg in message:
            if msg['type'] == 'image':
                content.append({"type": "image", "image": None})
                images.append(Image.open(msg['value']).convert('RGB'))
            elif msg['type'] == 'text':
                content.append({"type": "text", "text": msg['value']})
        conversation = [{"role": "user", "content": content}]
        prompt = image_processor.apply_chat_template(conversation,
                                                     add_generation_prompt=True)
        inputs = image_processor(text=prompt, images=images, return_tensors='pt')
        if device is not None:
            inputs = inputs.to(device, torch.float16)
        return inputs

    elif model_type in (QWEN2_VL, QWEN2_5_VL):
        from qwen_vl_utils import process_vision_info
        content, images = [], []
        for msg in message:
            if msg['type'] == 'image':
                content.append({"type": "image", "image": msg['value']})
            elif msg['type'] == 'text':
                content.append({"type": "text", "text": msg['value']})
        conversation = [{"role": "user", "content": content}]
        prompt = image_processor.apply_chat_template(conversation,
                                                    add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        assert video_inputs is None
        inputs = image_processor(text=[prompt], images=image_inputs, videos=video_inputs,
                                 padding=True, return_tensors='pt')
        if device is not None:
            inputs = inputs.to(device, torch.float16)
        return inputs

    else:
        raise NotImplementedError

class LN_(torch.nn.Module):
    def __init__(self, ori_layer_norm) -> None:
        super().__init__()
        self.ln = ori_layer_norm

    def forward(self, x):
        out = self.ln(x)
        outdtype = out.dtype
        out_ = out.double()
        out_ = out_ - out_.mean(dim=-2, keepdim=True)
        return out_.to(outdtype)
    
class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)

# class RMSLN(torch.nn.Module):
#     """
#     This class implements the Root Mean Square Normalization (RMSN) layer.
#     We use the implementation from LLAMARMSNorm here:
#     https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
#     """

#     def __init__(self, mean_dim: int, eps=1e-5):
#         super().__init__()
#         self.eps = eps
#         self.mean_dim = mean_dim
#         self.weight = torch.nn.Parameter(torch.ones(1))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         input_dtype = x.dtype
#         if x.dtype == torch.float16:
#             x = x.to(torch.float32)
#         variance = x.pow(2).mean(-1, keepdim=True)
#         x = x * torch.rsqrt(variance + self.eps)
#         return x.to(input_dtype)
    
def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)
