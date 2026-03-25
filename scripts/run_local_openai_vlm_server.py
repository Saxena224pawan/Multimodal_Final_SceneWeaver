#!/usr/bin/env python3
import argparse
import base64
import io
import json
import time
import uuid
from typing import Any, Dict, List, Tuple

from PIL import Image
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
import torch
import uvicorn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = None
PROCESSOR = None
MODEL_NAME = None
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


def generate_completion(payload: Dict[str, Any]) -> Dict[str, Any]:
    global MODEL, PROCESSOR, MODEL_NAME
    model_name = payload.get('model') or MODEL_NAME
    raw_messages = payload.get('messages') or []
    max_tokens = int(payload.get('max_tokens') or payload.get('max_completion_tokens') or DEFAULT_MAX_NEW_TOKENS)

    messages, images = convert_messages(raw_messages)
    prompt_text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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


def load_model(model_dir: str, served_model_name: str) -> None:
    global MODEL, PROCESSOR, MODEL_NAME
    MODEL_NAME = served_model_name
    PROCESSOR = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
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
