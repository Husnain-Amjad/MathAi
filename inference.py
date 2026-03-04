import argparse
import os
import re
import json
import tempfile
import time
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Inference and PPO Data Preparation for Math Models.")
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace Hub ID or path to the saved model/LoRA adapter.")
    parser.add_argument("--is_lora", action="store_true", help="Flag to indicate if the model_path is a LoRA adapter that needs merging.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the validation/test CSV data.")
    parser.add_argument("--output_path", type=str, default="ppo_training_data.jsonl", help="Path to save the generated PPO dataset.")
    parser.add_argument("--prompt_column", type=str, default="problem", help="Column name for the input question.")
    parser.add_argument("--solution_column", type=str, default="solution", help="Column name for the ground truth solution.")
    parser.add_argument("--num_responses", type=int, default=4, help="Number of responses to generate per prompt (n parameter in vLLM).")
    
    # Updated: Higher default max_tokens and added max_model_len
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum tokens to generate per response.")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Total context window size (prompt + generation) for vLLM.")
    
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling (must be > 0 for multiple unique responses).")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter.")
    return parser.parse_args()

def extract_boxed_answer(text):
    match = re.search(r'\\boxed{', text)
    if not match:
        return None
    start_index = match.end() - 1 
    stack = []
    for i in range(start_index, len(text)):
        if text[i] == '{':
            stack.append('{')
        elif text[i] == '}':
            stack.pop()
            if len(stack) == 0:
                return text[start_index + 1 : i].strip()
    return None

def check_correctness(prediction, ground_truth):
    pred_ans = extract_boxed_answer(prediction)
    gt_ans = extract_boxed_answer(ground_truth)
    if not pred_ans:
        return 0.0
    if pred_ans.replace(" ", "") == gt_ans.replace(" ", ""):
        return 1.0
    return 0.0

def merge_lora_to_temp(lora_path):
    print(f"\n[Timeline] Loading and merging LoRA adapter from {lora_path}...")
    model = AutoPeftModelForCausalLM.from_pretrained(lora_path, device_map="auto", torch_dtype="auto")
    merged_model = model.merge_and_unload()
    
    temp_dir = tempfile.mkdtemp(prefix="merged_math_model_")
    merged_model.save_pretrained(temp_dir)
    AutoTokenizer.from_pretrained(lora_path).save_pretrained(temp_dir)
    return temp_dir

def main():
    start_time = time.time()
    args = parse_args()

    # 1. Handle LoRA merging
    model_dir_for_vllm = args.model_path
    if args.is_lora:
        lora_start = time.time()
        model_dir_for_vllm = merge_lora_to_temp(args.model_path)
        print(f"[Timeline] LoRA merge completed in {time.time() - lora_start:.2f} seconds.")

    # 2. Initialize vLLM
    print(f"\n[Timeline] Initializing vLLM from {model_dir_for_vllm}...")
    vllm_start = time.time()
    
    # Updated: Passed max_model_len to the vLLM engine
    llm = LLM(
        model=model_dir_for_vllm, 
        trust_remote_code=True, 
        max_model_len=args.max_model_len
    )
    tokenizer = llm.get_tokenizer()
    print(f"[Timeline] vLLM initialized in {time.time() - vllm_start:.2f} seconds.")

    # 3. Load & Format Data
    print(f"\nLoading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    formatted_prompts = []
    print("Formatting prompts...")
    for problem in tqdm(df[args.prompt_column], desc="Applying Chat Templates"):
        messages = [{"role": "user", "content": problem}]
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"Question: {problem}\nAnswer:"
        formatted_prompts.append(prompt)

    # 4 & 5. vLLM Inference
    sampling_params = SamplingParams(
        n=args.num_responses,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print(f"\n[Timeline] Starting inference for {len(formatted_prompts)} prompts ({args.num_responses} responses each)...")
    inference_start = time.time()
    # vLLM automatically displays a tqdm progress bar for this step!
    outputs = llm.generate(formatted_prompts, sampling_params)
    print(f"[Timeline] Inference completed in {time.time() - inference_start:.2f} seconds.")

    # 6. Evaluate
    ppo_dataset = []
    print("\nEvaluating responses and preparing PPO data...")
    eval_start = time.time()
    for idx, output in enumerate(tqdm(outputs, desc="Evaluating Correctness")):
        original_problem = df.iloc[idx][args.prompt_column]
        ground_truth = df.iloc[idx][args.solution_column]
        
        # Output outputs is a list of completions because n=num_responses
        responses = [output.outputs[i].text for i in range(args.num_responses)]
        rewards = [check_correctness(r, ground_truth) for r in responses]
            
        ppo_dataset.append({
            "prompt": original_problem,
            "ground_truth": ground_truth,
            "responses": responses,     
            "rewards": rewards          
        })
    print(f"[Timeline] Evaluation completed in {time.time() - eval_start:.2f} seconds.")

    # 7. Save
    with open(args.output_path, "w") as f:
        for record in ppo_dataset:
            f.write(json.dumps(record) + "\n")
            
    # Cleanup
    if args.is_lora:
        import shutil
        shutil.rmtree(model_dir_for_vllm)

    total_time = time.time() - start_time
    print(f"\n=== SUCCESS ===")
    print(f"PPO data saved to: {args.output_path}")
    print(f"Total script execution time: {total_time / 60:.2f} minutes.")

if __name__ == "__main__":
    main()