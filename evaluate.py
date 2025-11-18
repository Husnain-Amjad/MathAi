
import argparse
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from sklearn.model_selection import train_test_split
from data_processing import Preprocessing
from base_data_loader import BaseDataLoader
# from tokenization import Tokenization, TokenizedDataset
# from no_trainer_based_training import ManualTraining
# from lora_config import lora_default_args
from bnb_config import four_bit_args, eight_bit_args
# from training_args_config import non_trainer_args_defaults
from types import SimpleNamespace
from inference import InferenceModule


def parse_args():
    parser = argparse.ArgumentParser(description="Inference a math model with flexible configurations.")
    # Inference Model 
    parser.add_argument("--model_name", type=str, default=None, 
                        help="Model to use for inference.")
    # Enable Quantization 
    parser.add_argument("--use_quantization", action="store_true", 
                        help="Enable 4-bit quantization.")
    # Quantization type selection
    parser.add_argument("--bnb_config", type=str, default= "four_bit_args", 
                            help="When use_quantization then BitsAndBytesConfig (four_bit_args/eight_bit_args)")
    # Data to Use for Training
    parser.add_argument("--data_dir", type=str, default="AUG_MATH", help="Directory containing train.csv and validation.csv.")
    # Fraction of data to be used for training
    parser.add_argument("--sample_ratio", type=float, default=1.0, 
                        help="Ratio of data to use (0.0 to 1.0).")
    # Processing
    parser.add_argument("--boxed", action="store_true", 
                        help="Extract only boxed solutions instead of full solutions.")
    

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
    test_path = os.path.join(args.data_dir, "test.csv")

    # Data Loading
    test_loader = BaseDataLoader(test_path)
    test_data = test_loader.load()
    if args.sample_ratio < 1.0:
        test_data = stratified_sample(test_data, args.sample_ratio, args.stratify_column)
    print(f"Train data len: {test_data.shape[0]}")
    print("Train Data is Loaded:\n", test_data.head())

    # Data Preprocessing
    test_cleaned = Preprocessing.process_data(test_data, boxed=args.boxed)
    print("\nTrain Data Info:", test_cleaned.info())
    print("Rows with missing solutions:\n", test_cleaned[test_cleaned["solution"].isna()])

    # Model Setup
    bnb_config = None
    if args.use_quantization:
        if args.bnb_config == "four_bit_args":
            bnb_config = BitsAndBytesConfig(**four_bit_args)
        elif args.bnb_config == "eight_bit_args":
            bnb_config = BitsAndBytesConfig(**eight_bit_args)

    # Inference
    model_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    trained_model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb_config, device_map="Auto")

    inference_wrapper = InferenceModule(trained_model, model_tokenizer)
    results = []
    for question in test_cleaned:
        generated_output = inference_wrapper.generator(question)
        print(f"Inference result for '{question}': {generated_output}")
        results.append({"problem":{question}, "solution":{generated_output}})
    # Convert to DataFrame and save
    inference_results = pd.DataFrame(results)
    inference_results.to_csv('inference_results.csv', index=False, encoding='utf-8')
    print("All results saved to inference_results.csv")

    inference_results = Preprocessing.process_data(inference_results, boxed=args.boxed)
    inference_results["ref_solution"] = test_cleaned["solution"]
    print("##### Results Head #####", inference_results.head())


    
