
import argparse
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from data_processing import Preprocessing
from base_data_loader import BaseDataLoader
from tokenization import Tokenization, TokenizedDataset
from no_trainer_based_training import ManualTraining
from lora_config import lora_default_args
from bnb_config import four_bit_args
from training_args_config import non_trainer_args_defaults
from types import SimpleNamespace
from inference import InferenceModule

# Available models
MODEL_NAMES = [
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "Qwen/Qwen2.5-Math-1.5B",
    "deepseek-ai/deepseek-math-7b-rl",
    "deepseek-ai/deepseek-math-7b-base",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "AI-MO/NuminaMath-7B-CoT",
    "AIDC-AI/Marco-o1",
    "microsoft/phi-4",
    "meta-math/MetaMath-Mistral-7B",
    "vanillaOVO/WizardMath-7B-V1.0",
    "tiiuae/falcon-40b",
    "EleutherAI/gpt-neo-2.7B",
    "google/gemma-3-27b-it"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Train a math model with flexible configurations.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B", 
                        choices=MODEL_NAMES, help="Model to use for training.")
    parser.add_argument("--data_dir", type=str, default="AUG_MATH", help="Directory containing train.csv and validation.csv.")
    parser.add_argument("--sample_ratio", type=float, default=1.0, 
                        help="Ratio of data to use (0.0 to 1.0).")
    parser.add_argument("--stratify_column", type=str, default=None, 
                        help="Column to use for stratified sampling (e.g., 'problem_type').")
    parser.add_argument("--use_quantization", action="store_true", 
                        help="Enable 4-bit quantization.")
    parser.add_argument("--use_lora", action="store_true", 
                        help="Enable LoRA fine-tuning.")
    parser.add_argument("--lora_rank", type=int, default=16, 
                        help="LoRA rank parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, 
                        help="LoRA dropout rate.")
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="Number of training epochs.")
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="Directory for saving model outputs.")
    parser.add_argument("--boxed", action="store_true", 
                        help="Extract only boxed solutions instead of full solutions.")
    return parser.parse_args()

def stratified_sample(df, sample_ratio, random_state=42):
    if sample_ratio >= 1.0:
        return df
    df['type_level'] = df['type'] + '_' + df['level']

    # Use train_test_split to sample a fraction while stratifying
    sampled_df, _ = train_test_split(
        df,
        train_size=sample_ratio,
        stratify=df['type_level'],
        random_state=random_state
    )
    sampled_df = sampled_df.drop(columns=['type_level'])
    return sampled_df

def main():
    args = parse_args()

    # Set up paths
    train_path = os.path.join(args.data_dir, "train.csv")
    val_path = os.path.join(args.data_dir, "validation.csv")

    # Data Loading
    train_loader = BaseDataLoader(train_path)
    train_data = train_loader.load()
    if args.sample_ratio < 1.0:
        train_data = stratified_sample(train_data, args.sample_ratio, args.stratify_column)
    print(f"Train data len: {train_data.shape[0]}")
    print("Train Data is Loaded:\n", train_data.head())

    val_loader = BaseDataLoader(val_path)
    val_data = val_loader.load()
    if args.sample_ratio < 1.0:
        val_data = stratified_sample(val_data, args.sample_ratio, args.stratify_column)
    print(f"Validation data len: {val_data.shape[0]}")
    print("Validation Data is Loaded:\n", val_data.head())

    # Data Preprocessing
    train_cleaned = Preprocessing.process_data(train_data, boxed=args.boxed)
    print("\nTrain Data Info:", train_cleaned.info())
    print("Rows with missing solutions:\n", train_cleaned[train_cleaned["solution"].isna()])

    val_cleaned = Preprocessing.process_data(val_data, boxed=args.boxed)
    print("\nValidation Data Info:", val_cleaned.info())
    print("Rows with missing solutions:\n", val_cleaned[val_cleaned["solution"].isna()])

    # Tokenizer Setup
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer_obj = Tokenization(tokenizer, max_length=1024)

    # Tokenize Data
    train_tokens = tokenizer_obj.training_unified_tokenization(train_cleaned)
    print(f"\nTokenization complete: {len(train_tokens['input_ids'])} samples")
    val_tokens = tokenizer_obj.training_unified_tokenization(val_cleaned)
    print(f"Validation tokenization complete: {len(val_tokens['input_ids'])} samples")

    train_dataset = TokenizedDataset(train_tokens)
    val_dataset = TokenizedDataset(val_tokens)

    # CUDA Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is NOT available.")

    # Model Setup
    bnb_config = BitsAndBytesConfig(**four_bit_args) if args.use_quantization else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    if args.use_lora:
        lora_config = lora_default_args.copy()
        lora_config["r"] = args.lora_rank
        lora_config["lora_dropout"] = args.lora_dropout
        peft_config = LoraConfig(**lora_config)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
    
    model = model.to(device)
    if args.use_lora:
        model.print_trainable_parameters()

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training Setup
    training_args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_epochs,
        **non_trainer_args_defaults
    }
    training_args = SimpleNamespace(**training_args_dict)

    # Training
    trainer_wrapper = ManualTraining(model=model, tokenizer=tokenizer)
    trained_model, model_tokenizer = trainer_wrapper.training(
        training_args=training_args,
        tokenized_train_dataset=train_dataset,
        tokenized_validation_dataset=val_dataset,
        data_collator=data_collator
    )

    # Inference
    # inference_wrapper = InferenceModule(trained_model, model_tokenizer)
    # question = "What is 2+2"
    # generated_output = inference_wrapper.generator(question)
    # print(f"Inference result for '{question}': {generated_output}")

if __name__ == "__main__":
    main()