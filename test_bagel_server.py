#!/usr/bin/env python3
"""
Quick test script for BAGEL server

Usage:
    python test_bagel_server.py --port 9999
"""

import argparse
import requests
import json
import time


def test_health_check(port: int):
    """Test the /health endpoint."""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Health check passed")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_text_only(port: int):
    """Test text-only inference."""
    print("\nTesting text-only inference...")

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is 2+2?"}
                ]
            }
        ],
        "max_tokens": 100,
        "temperature": 0.0
    }

    try:
        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            print(f"✓ Text-only inference passed")
            print(f"  Question: What is 2+2?")
            print(f"  Answer: {answer}")
            return True
        else:
            print(f"✗ Text-only inference failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Text-only inference failed: {e}")
        return False


def test_multimodal(port: int, image_path: str = None):
    """Test multimodal inference with image."""
    if not image_path:
        print("\nSkipping multimodal test (no image provided)")
        return True

    print(f"\nTesting multimodal inference with {image_path}...")

    import base64
    from PIL import Image
    import io

    # Load and encode image
    try:
        img = Image.open(image_path)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        img_url = f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return False

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": "Describe this image."}
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0.0
    }

    try:
        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            print(f"✓ Multimodal inference passed")
            print(f"  Question: Describe this image.")
            print(f"  Answer: {answer[:200]}...")
            return True
        else:
            print(f"✗ Multimodal inference failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Multimodal inference failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test BAGEL server")
    parser.add_argument("--port", type=int, default=9999,
                        help="Port where BAGEL server is running")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to test image (optional)")
    parser.add_argument("--wait", type=int, default=0,
                        help="Wait N seconds before testing (for server startup)")
    args = parser.parse_args()

    if args.wait > 0:
        print(f"Waiting {args.wait} seconds for server to start...")
        time.sleep(args.wait)

    print("="*80)
    print("BAGEL Server Test Suite")
    print("="*80)

    results = []

    # Test 1: Health check
    results.append(("Health Check", test_health_check(args.port)))

    # Test 2: Text-only inference
    if results[-1][1]:  # Only if health check passed
        results.append(("Text-only Inference", test_text_only(args.port)))

    # Test 3: Multimodal inference
    if results[-1][1] and args.image:  # Only if previous tests passed
        results.append(("Multimodal Inference", test_multimodal(args.port, args.image)))

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! BAGEL server is working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the server logs.")
        return 1


if __name__ == "__main__":
    exit(main())
