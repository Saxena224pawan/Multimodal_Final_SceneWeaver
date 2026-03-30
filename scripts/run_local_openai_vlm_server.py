#!/usr/bin/env python3
import argparse
import base64
import io
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Type

from PIL import Image
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
import torch
import uvicorn
from transformers import AutoConfig, AutoProcessor

try:
    from transformers import AutoModelForImageTextToText
except Exception:  # pragma: no cover - depends on transformers version
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:  # pragma: no cover - depends on transformers version
    AutoModelForVision2Seq = None

try:
    from transformers import AutoModelForCausalLM
except Exception:  # pragma: no cover - depends on transformers version
    AutoModelForCausalLM = None

try:
    from transformers import LlavaForConditionalGeneration
except Exception:  # pragma: no cover - depends on transformers version
    LlavaForConditionalGeneration = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover - depends on transformers version
    Qwen2_5_VLForConditionalGeneration = None

MODEL = None
PROCESSOR = None
MODEL_NAME = None
MODEL_TYPE = None
MODEL_TEXT_MAX_POSITIONS = None
MODEL_IMAGE_TOKEN_COUNT = None
DEFAULT_MAX_NEW_TOKENS = 256


def decode_data_url(url: str) -> Image.Image:
    if not url.startswith('data:image'):
        raise ValueError('Only data:image URLs are supported')
    _, payload = url.split(',', 1)
    image = Image.open(io.BytesIO(base64.b64decode(payload))).convert('RGB')
    return image


def convert_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    converted: List[Dict[str, Any]] = []
    images: List[Image.Image] = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if isinstance(content, str):
            converted.append({'role': role, 'content': [{'type': 'text', 'text': content}]})
            continue
        parts: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                parts.append({'type': 'text', 'text': item})
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get('type')
            if item_type == 'text':
                parts.append({'type': 'text', 'text': item.get('text', '')})
            elif item_type == 'image_url':
                image_url = (item.get('image_url') or {}).get('url', '')
                image = decode_data_url(image_url)
                images.append(image)
                parts.append({'type': 'image', 'image': image})
        converted.append({'role': role, 'content': parts})
    return converted, images


def estimate_llava_sequence_length(prompt_text: str, image_count: int) -> Optional[int]:
    tokenizer = getattr(PROCESSOR, 'tokenizer', None)
    if tokenizer is None:
        return None
    if MODEL_TEXT_MAX_POSITIONS is None or MODEL_IMAGE_TOKEN_COUNT is None:
        return None
    try:
        tokenized = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=True)
    except Exception:
        return None
    input_ids = tokenized.get('input_ids')
    if input_ids is None:
        return None
    text_len = int(input_ids.shape[1])
    # The serialized prompt typically contains one placeholder token per image.
    return text_len + image_count * max(MODEL_IMAGE_TOKEN_COUNT - 1, 0)


def maybe_inject_scoring_reminder(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if MODEL_TYPE not in {'llava', 'smolvlm'}:
        return messages

    reminder = (
        'Answer briefly. Put the score on the first line exactly as: Score: <integer 1-5>. '
        'Then give at most two short sentences of justification.'
    )
    last_user_index = None
    for index, message in enumerate(messages):
        if message.get('role') == 'user':
            last_user_index = index
    if last_user_index is None:
        return messages

    patched: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        content = list(message.get('content', []))
        if index == last_user_index:
            already_present = any(
                item.get('type') == 'text' and reminder in item.get('text', '')
                for item in content
                if isinstance(item, dict)
            )
            if not already_present:
                content.append({'type': 'text', 'text': reminder})
        patched.append({'role': message.get('role', 'user'), 'content': content})
    return patched


def generate_completion(payload: Dict[str, Any]) -> Dict[str, Any]:
    global MODEL, PROCESSOR, MODEL_NAME, MODEL_TYPE
    model_name = payload.get('model') or MODEL_NAME
    raw_messages = payload.get('messages') or []
    max_tokens = int(payload.get('max_tokens') or payload.get('max_completion_tokens') or DEFAULT_MAX_NEW_TOKENS)

    messages, images = convert_messages(raw_messages)
    messages = maybe_inject_scoring_reminder(messages)
    prompt_text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if MODEL_TYPE == 'llava' and images:
        estimated_seq_len = estimate_llava_sequence_length(prompt_text, len(images))
        if estimated_seq_len is not None and MODEL_TEXT_MAX_POSITIONS is not None:
            print(
                f"LLaVA request: images={len(images)} estimated_sequence_length={estimated_seq_len} "
                f"max_position_embeddings={MODEL_TEXT_MAX_POSITIONS}",
                flush=True,
            )
            if estimated_seq_len > MODEL_TEXT_MAX_POSITIONS:
                raise ValueError(
                    "Refusing oversized LLaVA request: "
                    f"{len(images)} images expand to an estimated sequence length of {estimated_seq_len}, "
                    f"which exceeds max_position_embeddings={MODEL_TEXT_MAX_POSITIONS}. "
                    "Reduce the number of frames or use grid images."
                )

    processor_kwargs: Dict[str, Any] = {
        'text': [prompt_text],
        'padding': True,
        'return_tensors': 'pt',
    }
    if images:
        processor_kwargs['images'] = images

    inputs = PROCESSOR(**processor_kwargs)
    for key, value in list(inputs.items()):
        if hasattr(value, 'to'):
            inputs[key] = value.to(MODEL.device)

    with torch.inference_mode():
        generated_ids = MODEL.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    prompt_len = inputs['input_ids'].shape[1]
    completion_ids = generated_ids[:, prompt_len:]
    output_text = PROCESSOR.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    completion_tokens = int(completion_ids.shape[1]) if completion_ids.ndim == 2 else 0
    prompt_tokens = int(prompt_len)
    now = int(time.time())
    return {
        'id': f'chatcmpl-{uuid.uuid4().hex}',
        'object': 'chat.completion',
        'created': now,
        'model': model_name,
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': output_text},
                'finish_reason': 'stop',
            }
        ],
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        },
    }


async def models_endpoint(_: Request) -> JSONResponse:
    return JSONResponse({'object': 'list', 'data': [{'id': MODEL_NAME, 'object': 'model', 'owned_by': 'local'}]})


async def chat_completions_endpoint(request: Request) -> JSONResponse:
    payload = await request.json()
    try:
        response = generate_completion(payload)
        return JSONResponse(response)
    except Exception as exc:
        return JSONResponse({'error': {'message': str(exc), 'type': 'server_error'}}, status_code=500)


def _candidate_model_loaders(model_type: str) -> List[Type]:
    candidates: List[Optional[Type]] = []
    if model_type == 'qwen2_5_vl' and Qwen2_5_VLForConditionalGeneration is not None:
        candidates.append(Qwen2_5_VLForConditionalGeneration)
    if model_type == 'llava' and LlavaForConditionalGeneration is not None:
        candidates.append(LlavaForConditionalGeneration)
    if AutoModelForImageTextToText is not None:
        candidates.append(AutoModelForImageTextToText)
    if AutoModelForVision2Seq is not None:
        candidates.append(AutoModelForVision2Seq)
    if AutoModelForCausalLM is not None:
        candidates.append(AutoModelForCausalLM)

    unique: List[Type] = []
    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate in seen:
            continue
        unique.append(candidate)
        seen.add(candidate)
    return unique


def _load_with_class(model_cls: Type, model_dir: str):
    return model_cls.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )


def load_model(model_dir: str, served_model_name: str) -> None:
    global MODEL, PROCESSOR, MODEL_NAME, MODEL_TYPE, MODEL_TEXT_MAX_POSITIONS, MODEL_IMAGE_TOKEN_COUNT
    MODEL_NAME = served_model_name
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_type = getattr(config, 'model_type', '')
    MODEL_TYPE = model_type
    text_config = getattr(config, 'text_config', None)
    vision_config = getattr(config, 'vision_config', None)
    MODEL_TEXT_MAX_POSITIONS = getattr(text_config, 'max_position_embeddings', None)
    image_size = getattr(vision_config, 'image_size', None)
    patch_size = getattr(vision_config, 'patch_size', None)
    if isinstance(image_size, int) and isinstance(patch_size, int) and patch_size > 0:
        MODEL_IMAGE_TOKEN_COUNT = (image_size // patch_size) ** 2
    else:
        MODEL_IMAGE_TOKEN_COUNT = None
    PROCESSOR = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    errors: List[str] = []
    for model_cls in _candidate_model_loaders(model_type):
        try:
            MODEL = _load_with_class(model_cls, model_dir)
            break
        except Exception as exc:  # pragma: no cover - hardware/model dependent
            errors.append(f"{model_cls.__name__}: {exc}")
    else:
        details = "; ".join(errors) if errors else "no compatible model loader found"
        raise RuntimeError(
            f"Unable to load multimodal model from {model_dir} "
            f"(model_type={model_type!r}): {details}"
        )

    MODEL.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--served-model-name', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--api-key', default='local-videobench')
    args = parser.parse_args()

    load_model(args.model_dir, args.served_model_name)
    app = Starlette(
        debug=False,
        routes=[
            Route('/v1/models', models_endpoint, methods=['GET']),
            Route('/v1/chat/completions', chat_completions_endpoint, methods=['POST']),
        ],
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    main()
