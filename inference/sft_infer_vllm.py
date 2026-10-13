"""Run inference and save parsed answers."""

import os
import json
import re
import argparse

from swift.llm import PtEngine, RequestConfig
try:
    from swift.llm import VllmEngine
    VLLM_AVAILABLE = True
    print("vLLM available; enable with --use_vllm.")
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM unavailable; install vllm to enable it.")


os.environ['SWIFT_PATCH_CONV3D'] = '1'

PREFIX = "/path/to/your/project"

def parse_json_from_xml(text):
    pattern = r"<answer>\s*(\{.*?\})\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None

def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Swift LLM Inference Script Accelerated")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation jsonl file")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save results")
    parser.add_argument("--use_vllm", action='store_true', help="Use VLLM engine for acceleration")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for PtEngine")
    
    args = parser.parse_args()

    # Input data
    val_path = os.path.join(PREFIX, args.val_file)
    print(f"Loading data: {val_path}")
    
    val_items = []
    with open(val_path, 'r', encoding='utf-8') as val_file:
        for line in val_file:
            if line.strip():
                val_items.append(json.loads(line))
    
    print(f"Validation samples: {len(val_items)}. Preparing inputs.")

    all_infer_inputs = []
    meta_data = []

    for i, item in enumerate(val_items):
        images = item.get('images', [])
        messages = item['messages']

        system_prompt = None
        user_query = None
        gt_content = None

        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            elif msg['role'] == 'user':
                user_query = msg['content']
            elif msg['role'] == 'assistant':
                gt_content = msg['content']

        if user_query is None:
            if len(messages) > 0: system_prompt = messages[0].get('content')
            if len(messages) > 1: user_query = messages[1].get('content')
            if len(messages) > 2: gt_content = messages[2].get('content')

        input_msgs = []
        if system_prompt:
            input_msgs.append({'role': 'system', 'content': system_prompt})
        if user_query:
            input_msgs.append({'role': 'user', 'content': user_query})

        all_infer_inputs.append({
            'messages': input_msgs,
            'images': images
        })

        meta_data.append({
            "id": i,
            "user_query": user_query,
            "gt_content": gt_content
        })

    # Model and engine
    model_path = os.path.join(PREFIX, args.ckpt_dir)
    print(f"Loading model: {model_path}")

    request_config = RequestConfig(
        max_tokens=2048,
        temperature=0.0,
        top_p=0.9,
        repetition_penalty=1.0,
    )

    engine = None
    
    if args.use_vllm and VLLM_AVAILABLE:
        print("Using VllmEngine.")
        try:
            engine = VllmEngine(
                model_path, 
                tensor_parallel_size=1,
                max_model_len=14672,
                gpu_memory_utilization=0.95,
                limit_mm_per_prompt={'image': 24}
            )
        except Exception as e:
            print(f"vLLM initialization failed: {e}. Using PtEngine.")
    
    if engine is None:
        print(f"Using PtEngine, batch_size={args.batch_size}.")
        engine = PtEngine(model_path, max_batch_size=args.batch_size, device_map='auto')

    # Inference
    print(f"Running inference on {len(all_infer_inputs)} samples.")
    resp_list = engine.infer(all_infer_inputs, request_config)

    # Parse and save
    print("Parsing results.")
    results = []

    for idx, (resp, meta) in enumerate(zip(resp_list, meta_data)):
        response_text = resp.choices[0].message.content
        
        gt_content = meta['gt_content']
        gt_json = parse_json_from_xml(gt_content) if gt_content else None
        pred_json = parse_json_from_xml(response_text)

        if idx < 1:
            print(f"\nSample {idx} Input: {meta['user_query']}")
            print(f"Sample {idx} GT Raw: {gt_content}")
            print(f"Sample {idx} Pred Raw: {response_text}\n")

        task_type = "unknown"
        if gt_json:
            keys = list(gt_json.keys())
            if "syntax_score" in keys:
                task_type = "syntax"
            else:
                task_type = keys[0] if len(keys) > 0 else "unknown"

        results.append({
            "id": meta['id'],
            "task_type": task_type,
            "pred_raw": response_text,
            "pred_parsed": pred_json,
            "gt_raw": gt_content,
            "gt_parsed": gt_json
        })

    save_dir_path = os.path.join(PREFIX, args.save_dir)
    os.makedirs(save_dir_path, exist_ok=True)
    output_file = os.path.join(save_dir_path, 'inference_results.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {output_file}")

if __name__ == "__main__":
    main()
