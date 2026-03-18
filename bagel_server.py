#!/usr/bin/env python3
"""
BAGEL Model Server with OpenAI-Compatible API

This server wraps the BAGEL-7B-MoT model to provide an OpenAI-compatible
/v1/chat/completions endpoint, allowing it to integrate seamlessly with
the Theory-of-Space evaluation framework.

Usage:
    python bagel_server.py --model-path /path/to/BAGEL-7B-MoT --port 9999
"""

import argparse
import base64
import io
import os
import sys
from typing import List, Dict, Any

from flask import Flask, request, jsonify
from PIL import Image
import torch

# Add Bagel to Python path
BAGEL_DIR = os.path.join(os.path.dirname(__file__), 'Bagel')
sys.path.insert(0, BAGEL_DIR)

# Import BAGEL utilities
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation

app = Flask(__name__)

# Global model variables
model = None
tokenizer = None
new_token_ids = None
image_transform = None


def parse_args():
    parser = argparse.ArgumentParser(description="BAGEL Model Server")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to BAGEL model directory")
    parser.add_argument("--port", type=int, default=9999,
                        help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    return parser.parse_args()


def decode_base64_image(image_url: str) -> Image.Image:
    """Decode base64 image data to PIL Image."""
    if image_url.startswith('data:image'):
        # Extract base64 data after comma
        base64_data = image_url.split(',', 1)[1]
    else:
        base64_data = image_url

    image_bytes = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_bytes))
    return image


def extract_answer(response: str) -> str:
    """Extract final answer from response, removing <think> tags if present."""
    if '<think>' in response and '</think>' in response:
        # Split by </think> and take everything after
        parts = response.split('</think>')
        if len(parts) > 1:
            return parts[-1].strip()
    return response.strip()


def process_openai_messages(messages: List[Dict[str, Any]]) -> tuple:
    """
    Convert OpenAI format messages to BAGEL format.

    Returns:
        images: List of PIL Images
        prompt: Combined text prompt
    """
    images = []
    prompt_parts = []

    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', [])

        # Handle string content (simple text)
        if isinstance(content, str):
            prompt_parts.append(content)
            continue

        # Handle list content (multimodal)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get('type', 'text')

                    if item_type == 'text':
                        prompt_parts.append(item.get('text', ''))

                    elif item_type == 'image_url':
                        image_data = item.get('image_url', {})
                        if isinstance(image_data, dict):
                            url = image_data.get('url', '')
                        else:
                            url = image_data

                        if url:
                            try:
                                # Check if it's a local file path
                                if os.path.exists(url):
                                    img = Image.open(url)
                                else:
                                    # Assume it's base64
                                    img = decode_base64_image(url)
                                images.append(img)
                            except Exception as e:
                                print(f"Error loading image: {e}")

    prompt = ' '.join(prompt_parts)
    return images, prompt


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        max_tokens = data.get('max_new_tokens', data.get('max_tokens', 512))
        temperature = data.get('temperature', 0.0)

        # Process messages to extract images and prompt
        images, prompt = process_openai_messages(messages)

        if not prompt:
            return jsonify({
                "error": {
                    "message": "No text prompt provided",
                    "type": "invalid_request_error"
                }
            }), 400

        # Prepare conversation for BAGEL
        conversation = [{"role": "user", "content": prompt}]
        images_processed, conversation_processed = process_conversation(images, conversation)

        # Build prompt string
        prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.cuda()

        # Process images
        if images_processed:
            pixel_values_list = []
            for img in images_processed:
                pixel_values = image_transform(img)
                pixel_values_list.append(pixel_values)
            pixel_values = torch.stack(pixel_values_list).cuda()
        else:
            pixel_values = None

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                new_token_ids=new_token_ids,
                max_length=max_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 0.01) if temperature > 0 else 1.0,
            )

        # Decode response
        response_text = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Extract answer (remove <think> tags)
        answer = extract_answer(response_text)

        # Return OpenAI format response
        return jsonify({
            "id": "bagel-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "BAGEL-7B-MoT",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_ids.shape[1],
                "completion_tokens": output_ids.shape[1] - input_ids.shape[1],
                "total_tokens": output_ids.shape[1]
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }), 500


def main():
    global model, tokenizer, new_token_ids, image_transform

    args = parse_args()

    print("="*80)
    print("BAGEL Model Server")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Loading model...")

    # Change to Bagel directory for proper path resolution
    original_cwd = os.getcwd()
    os.chdir(BAGEL_DIR)

    try:
        # Create a simple args object for load_model_and_tokenizer
        class ModelArgs:
            def __init__(self, model_path):
                self.model_path = model_path

        model_args = ModelArgs(args.model_path)

        # Load model and tokenizer
        model, tokenizer, new_token_ids = load_model_and_tokenizer(model_args)
        print("✓ Model loaded successfully")

        # Build image transform
        image_transform = build_transform()
        print("✓ Image transform initialized")

    finally:
        # Restore original working directory
        os.chdir(original_cwd)

    print(f"\nStarting server on {args.host}:{args.port}")
    print("="*80)

    # Run Flask app
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()

