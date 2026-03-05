"""
vLLM Inference & PPO Data Preparation for Math LLMs
=====================================================
Loads a base model or LoRA adapter, runs batched inference via vLLM,
and writes a JSONL dataset ready for PPO / RLHF training.

Usage examples
--------------
# Base model
python vllm_inference.py \
    --model_path "Qwen/Qwen2.5-Math-7B" \
    --data_path  "math_val.csv" \
    --output_path "ppo_data.jsonl"

# LoRA adapter (auto-merged before loading into vLLM)
python vllm_inference.py \
    --model_path "checkpoints/lora-math-v1" \
    --is_lora \
    --data_path  "math_val.csv" \
    --output_path "ppo_data.jsonl" \
    --num_responses 8 \
    --temperature 0.8
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM Inference and PPO Data Preparation for Math Models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace Hub ID or local path to the saved model / LoRA adapter.",
    )
    model_grp.add_argument(
        "--is_lora",
        action="store_true",
        help="Treat --model_path as a LoRA adapter directory. "
             "The script will merge it with its base model before inference.",
    )
    model_grp.add_argument(
        "--base_model_for_lora",
        type=str,
        default=None,
        help="Base model required when --is_lora is set and the adapter config "
             "does not specify base_model_name_or_path.",
    )

    # ── Data ───────────────────────────────────────────────────────────────
    data_grp = parser.add_argument_group("Data")
    data_grp.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the validation / test CSV file.",
    )
    data_grp.add_argument(
        "--output_path",
        type=str,
        default="ppo_training_data.jsonl",
        help="Destination file for the generated PPO JSONL dataset.",
    )
    data_grp.add_argument(
        "--prompt_column",
        type=str,
        default="problem",
        help="CSV column that contains the input question / problem.",
    )
    data_grp.add_argument(
        "--solution_column",
        type=str,
        default="solution",
        help="CSV column that contains the ground-truth solution.",
    )

    # ── Generation ─────────────────────────────────────────────────────────
    gen_grp = parser.add_argument_group("Generation")
    gen_grp.add_argument(
        "--num_responses",
        type=int,
        default=4,
        help="Number of independent responses to sample per prompt (vLLM `n`).",
    )
    gen_grp.add_argument(
        "--max_tokens",
        type=int,
        default=3000,
        help="Maximum new tokens to generate per response.",
    )
    gen_grp.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Total context window length (prompt + generation) for vLLM.",
    )
    gen_grp.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (must be > 0 when num_responses > 1).",
    )
    gen_grp.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter.",
    )

    # ── Prompt template ────────────────────────────────────────────────────
    prompt_grp = parser.add_argument_group("Prompt template")
    prompt_grp.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are an expert mathematician. "
            "Solve the following problem step by step, "
            "then state your final answer clearly."
        ),
        help="System prompt prepended to every problem.",
    )

    # ── vLLM engine ────────────────────────────────────────────────────────
    vllm_grp = parser.add_argument_group("vLLM engine")
    vllm_grp.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism.",
    )
    vllm_grp.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory vLLM may use (0.0 – 1.0).",
    )
    vllm_grp.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Weight dtype for the vLLM engine.",
    )
    vllm_grp.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of prompts to submit to vLLM in one call.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# LoRA merge helper
# ---------------------------------------------------------------------------
def merge_lora_and_save(adapter_path: str, base_model: str | None) -> str:
    """
    Merge a PEFT LoRA adapter into its base model weights and save the
    merged checkpoint to a temporary directory.

    Returns the path to the merged model directory.
    """
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "peft and transformers are required for LoRA merging. "
            "Install them with: pip install peft transformers"
        ) from exc

    import torch

    # Resolve base model name
    if base_model is None:
        adapter_cfg_path = Path(adapter_path) / "adapter_config.json"
        if not adapter_cfg_path.exists():
            raise FileNotFoundError(
                f"adapter_config.json not found in {adapter_path}. "
                "Provide --base_model_for_lora explicitly."
            )
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_model = adapter_cfg.get("base_model_name_or_path")
        if not base_model:
            raise ValueError(
                "base_model_name_or_path not found in adapter_config.json. "
                "Provide --base_model_for_lora explicitly."
            )

    log.info("Loading base model '%s' for LoRA merge …", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    log.info("Applying LoRA adapter from '%s' …", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    tmp_dir = tempfile.mkdtemp(prefix="merged_model_")
    log.info("Saving merged model to '%s' …", tmp_dir)
    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    return tmp_dir


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_prompt(problem: str, system_prompt: str) -> str:
    """
    Wraps a math problem in a simple chat-style prompt.
    Extend this function to match your model's exact chat template.
    """
    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{problem}\n"
        f"<|assistant|>\n"
    )


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_final_answer(text: str) -> str:
    """
    Attempt to extract the final boxed answer from a LaTeX-style solution.
    Falls back to the last non-empty line of the response.
    """
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# Reward: exact-match against ground truth
# ---------------------------------------------------------------------------
def compute_reward(predicted: str, ground_truth: str) -> float:
    """
    Binary reward: 1.0 if the predicted answer matches the extracted
    ground-truth answer, else 0.0.
    """
    gt_answer = extract_final_answer(ground_truth)
    pred_answer = extract_final_answer(predicted)
    return 1.0 if pred_answer.strip() == gt_answer.strip() else 0.0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(data_path: str, prompt_col: str, solution_col: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    missing = [c for c in [prompt_col, solution_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in '{data_path}': {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[[c for c in df.columns]].dropna(subset=[prompt_col, solution_col])
    log.info("Loaded %d rows from '%s'.", len(df), data_path)
    return df


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
def run_inference(args: argparse.Namespace) -> None:
    # ── Optional LoRA merge ────────────────────────────────────────────────
    merged_tmp_dir: str | None = None
    model_path = args.model_path

    if args.is_lora:
        log.info("LoRA mode — merging adapter before loading into vLLM …")
        merged_tmp_dir = merge_lora_and_save(
            adapter_path=args.model_path,
            base_model=args.base_model_for_lora,
        )
        model_path = merged_tmp_dir

    # ── Load vLLM engine ───────────────────────────────────────────────────
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM is not installed. Install it with: pip install vllm"
        ) from exc

    log.info("Initialising vLLM engine with model '%s' …", model_path)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=args.num_responses,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|user|>", "<|system|>", "</s>"],
    )

    # ── Load data ──────────────────────────────────────────────────────────
    df = load_dataset(args.data_path, args.prompt_column, args.solution_column)

    prompts: list[str] = [
        build_prompt(row[args.prompt_column], args.system_prompt)
        for _, row in df.iterrows()
    ]
    ground_truths: list[str] = df[args.solution_column].tolist()

    # Gather extra metadata columns (level, type, …) if present
    meta_cols = [c for c in df.columns if c not in (args.prompt_column, args.solution_column)]

    # ── Batched inference ─────────────────────────────────────────────────
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    total_written = 0
    total_correct = 0
    total_responses = 0

    log.info(
        "Starting inference: %d prompts × %d responses = %d total generations.",
        len(prompts),
        args.num_responses,
        len(prompts) * args.num_responses,
    )

    with open(output_path, "w", encoding="utf-8") as fout:
        for batch_idx in tqdm(range(total_batches), desc="Batches", unit="batch"):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(prompts))

            batch_prompts = prompts[start:end]
            batch_gt = ground_truths[start:end]
            batch_rows = df.iloc[start:end]

            # Submit entire batch in one vLLM call
            vllm_outputs = llm.generate(batch_prompts, sampling_params)

            for i, (vllm_out, gt, (_, row)) in enumerate(
                zip(vllm_outputs, batch_gt, batch_rows.iterrows())
            ):
                prompt_text = vllm_out.prompt
                responses: list[dict[str, Any]] = []

                for output in vllm_out.outputs:
                    generated_text = output.text
                    reward = compute_reward(generated_text, gt)
                    total_correct += reward
                    total_responses += 1

                    responses.append(
                        {
                            "response": generated_text,
                            "extracted_answer": extract_final_answer(generated_text),
                            "reward": reward,
                            "finish_reason": output.finish_reason,
                        }
                    )

                record: dict[str, Any] = {
                    "prompt": row[args.prompt_column],
                    "prompt_with_template": prompt_text,
                    "ground_truth_solution": gt,
                    "ground_truth_answer": extract_final_answer(gt),
                    "responses": responses,
                    "num_correct": sum(r["reward"] for r in responses),
                    "num_responses": len(responses),
                }

                # Attach any extra metadata (level, type, …)
                for col in meta_cols:
                    record[col] = row[col]

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

    # ── Summary ────────────────────────────────────────────────────────────
    accuracy = total_correct / total_responses if total_responses else 0.0
    log.info("─" * 60)
    log.info("Inference complete.")
    log.info("  Problems processed : %d", total_written)
    log.info("  Total responses    : %d", total_responses)
    log.info("  Correct responses  : %d  (%.1f%%)", int(total_correct), accuracy * 100)
    log.info("  Output written to  : %s", output_path.resolve())
    log.info("─" * 60)

    # ── Cleanup merged model ───────────────────────────────────────────────
    if merged_tmp_dir and Path(merged_tmp_dir).exists():
        log.info("Removing temporary merged model directory …")
        shutil.rmtree(merged_tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Validation
    if args.num_responses > 1 and args.temperature == 0.0:
        log.warning(
            "temperature=0.0 with num_responses=%d will produce identical "
            "outputs. Consider using temperature > 0.",
            args.num_responses,
        )

    if not Path(args.data_path).exists():
        log.error("Data file not found: %s", args.data_path)
        sys.exit(1)

    run_inference(args)


if __name__ == "__main__":
    main()