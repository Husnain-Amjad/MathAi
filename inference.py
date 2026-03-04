import argparse
import os
import re
import json
import tempfile
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Inference and PPO Data Preparation for Math Models.")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True, 
                        help="HuggingFace Hub ID or path to the saved model/LoRA adapter.")
    parser.add_argument("--is_lora", action="store_true", 
                        help="Flag to indicate if the model_path is a LoRA adapter that needs merging.")
    parser.add_argument("--base_model_name", type=str, default=None,
                        help="Base model name (required only if merging LoRA and adapter config is missing it).")
    
    # Data configuration
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the validation/test CSV data.")
    parser.add_argument("--output_path", type=str, default="ppo_training_data.jsonl", 
                        help="Path to save the generated PPO dataset.")
    parser.add_argument("--prompt_column", type=str, default="problem", 
                        help="Column name for the input question.")
    parser.add_argument("--solution_column", type=str, default="solution", 
                        help="Column name for the ground truth solution.")
    
    # vLLM Generation parameters
    parser.add_argument("--num_responses", type=int, default=4, 
                        help="Number of responses to generate per prompt (n parameter in vLLM).")
    parser.add_argument("--max_tokens", type=int, default=1024, 
                        help="Maximum tokens to generate per response.")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for sampling (must be > 0 for multiple unique responses).")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p sampling parameter.")
    
    return parser.parse_args()

def extract_boxed_answer(text):
    """
    Extracts the content inside \boxed{} from a LaTeX string.
    Uses a stack to handle nested curly braces correctly.
    """
    match = re.search(r'\\boxed{', text)
    if not match:
        return None
    
    start_index = match.end() - 1 # Point to the opening '{'
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
    """
    Compares the generated answer to the ground truth.
    Returns 1.0 if correct, 0.0 if incorrect.
    """
    pred_ans = extract_boxed_answer(prediction)
    gt_ans = extract_boxed_answer(ground_truth)
    
    # If the generation didn't output a boxed answer, it's wrong (or malformed)
    if not pred_ans:
        return 0.0
        
    # Basic string matching. For production, consider using libraries like `math_verify`
    # or SymPy for algebraic equivalent checking.
    if pred_ans.replace(" ", "") == gt_ans.replace(" ", ""):
        return 1.0
    return 0.0

def merge_lora_to_temp(lora_path):
    """
    Merges a LoRA adapter into its base model and saves it to a temporary directory 
    so vLLM can load it efficiently.
    """
    print(f"Loading and merging LoRA adapter from {lora_path}...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_path, 
        device_map="auto", 
        torch_dtype="auto"
    )
    merged_model = model.merge_and_unload()
    
    temp_dir = tempfile.mkdtemp(prefix="merged_math_model_")
    print(f"Saving merged model to temporary directory: {temp_dir}")
    merged_model.save_pretrained(temp_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    tokenizer.save_pretrained(temp_dir)
    
    return temp_dir

def main():
    args = parse_args()

    # 1. Handle LoRA merging if necessary
    model_dir_for_vllm = args.model_path
    if args.is_lora:
        model_dir_for_vllm = merge_lora_to_temp(args.model_path)

    # 2. Initialize vLLM
    print(f"Initializing vLLM with model from {model_dir_for_vllm}...")
    llm = LLM(model=model_dir_for_vllm, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()

    # 3. Load Evaluation/Reward Data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    # Format prompts using the chat template (crucial for models like Qwen/DeepSeek)
    formatted_prompts = []
    for problem in df[args.prompt_column]:
        messages = [{"role": "user", "content": problem}]
        # Apply chat template if available, otherwise fallback to raw prompt
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"Question: {problem}\nAnswer:"
        formatted_prompts.append(prompt)

    # 4. Configure Sampling Params for Multiple Generations
    # Setting 'n' > 1 generates multiple completions per prompt simultaneously
    sampling_params = SamplingParams(
        n=args.num_responses,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # 5. Run Batched Inference
    print(f"Running inference (Generating {args.num_responses} responses per prompt)...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # 6. Evaluate and Format for PPO/Reward Model
    ppo_dataset = []
    
    print("Evaluating responses and preparing PPO data...")
    for idx, output in enumerate(outputs):
        original_problem = df.iloc[idx][args.prompt_column]
        ground_truth = df.iloc[idx][args.solution_column]
        
        responses = []
        rewards = []
        
        for i in range(args.num_responses):
            generated_text = output.outputs[i].text
            responses.append(generated_text)
            
            # Calculate reward (1 for correct, 0 for incorrect)
            reward = check_correctness(generated_text, ground_truth)
            rewards.append(reward)
            
        # Structure ideal for RLHF, PPO, or Reward Model training
        ppo_record = {
            "prompt": original_problem,
            "ground_truth": ground_truth,
            "responses": responses,     # List of N generated responses
            "rewards": rewards          # List of N rewards (e.g., [1.0, 0.0, 1.0, 0.0])
        }
        ppo_dataset.append(ppo_record)

    # 7. Save to JSONL
    with open(args.output_path, "w") as f:
        for record in ppo_dataset:
            f.write(json.dumps(record) + "\n")
            
    print(f"Success! PPO data saved to {args.output_path}")

    # Cleanup temp directory if we merged a LoRA
    if args.is_lora:
        import shutil
        shutil.rmtree(model_dir_for_vllm)
        print("Cleaned up temporary merged model directory.")

if __name__ == "__main__":
    main()