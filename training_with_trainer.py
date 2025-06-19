model_names = [
    "Qwen/Qwen2.5-Math-7B-Instruct", #0
    "Qwen/Qwen2.5-Math-1.5B", #1
    "deepseek-ai/deepseek-math-7b-rl", #2
    "deepseek-ai/deepseek-math-7b-base", #3
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", #4
    "AI-MO/NuminaMath-7B-CoT", #5
    "AIDC-AI/Marco-o1", #6
    "microsoft/phi-4", #7
    "meta-math/MetaMath-Mistral-7B", #8
    "vanillaOVO/WizardMath-7B-V1.0", #9
    "tiiuae/falcon-40b", #10
    "EleutherAI/gpt-neo-2.7B", #11
    "google/gemma-3-27b-it" #12
]

from data_processing import Preprocessing
from base_data_loader import BaseDataLoader
import pandas as pd
import torch
from bnb_config import four_bit_args
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoTokenizer , DataCollatorForLanguageModeling
from bnb_config import four_bit_args
from tokenization import Tokenization, TokenizedDataset
from trainer_based_training import TrainerBasedTraining
from lora_config import lora_default_args
from training_args_config import training_args_defaults
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from inference import InferenceModule

########################################################################################################################
################################################# Data Loading #########################################################
########################################################################################################################

train_loader = BaseDataLoader('AUG_MATH\\train.csv')
train_data = train_loader.load()

# train_data = train_data.head(int(len(train_data)*0.1))
print("train data len: ", train_data.shape[0])
print("Train Data is Loaded: \n", train_data.head())

val_loader = BaseDataLoader('AUG_MATH\\validation.csv')
val_data = val_loader.load()
# val_data = val_data.head(int(len(val_data)*0.1))
print("val data len: ", val_data.shape[0])
print("Train Data is Loaded: \n", val_data.head())

########################################################################################################################
######################################## Extracting Problem-Solution(\boxed{}) #########################################
########################################################################################################################

# # train_cleaned = Preprocessing.process_data(train_data, boxed=True)

# # print("\n\nInfo of Data: ", train_cleaned.info())
# # print("\n\nInfo of Data: \n", train_cleaned[train_cleaned["solution"].isna()]) # None are at index [133, 240, 5462, 5643]
# # # Handling None
# # for i in [133, 240, 5462, 5643]:
# #     print(Preprocessing.refine_box_extraction(train_data['solution'][i]))
# #     train_cleaned['solution'][i] = Preprocessing.refine_box_extraction(train_data['solution'][i])
# #     print(f"train_cleaned solution number{i}", train_cleaned['solution'][i])

# # print("\n\nInfo of Data: ", train_cleaned.info())
# # print("\n\nInfo of Data: 33\n", train_cleaned[train_cleaned["solution"].isna()])

########################################################################################################################
###################################### Extracting Problem-Solution(FULL SOLUTION) ######################################
########################################################################################################################
train_cleaned = Preprocessing.process_data(train_data, boxed=False)

print("\n\nInfo of Data: ", train_cleaned.info())
print("\n\nInfo of Data: \n", train_cleaned[train_cleaned["solution"].isna()]) # None are at index [133, 240, 5462, 5643]

print(train_cleaned.head())

val_cleaned = Preprocessing.process_data(val_data, boxed=False)

print("\n\nInfo of Data: ", val_cleaned.info())
print("\n\nInfo of Data: \n", val_cleaned[val_cleaned["solution"].isna()]) # None are at index [133, 240, 5462, 5643]

print(val_cleaned.head())

########################################################################################################################
################################################## Tokenizer Calling ###################################################
########################################################################################################################

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
tokenizer_obj = Tokenization(tokenizer, max_length=1024)

                          ######################## Train Tokenization #########################

train_tokens = tokenizer_obj.training_unified_tokenization(train_cleaned)
print("\n\ntokenization is compelte\n", len(train_tokens["labels"]), len(train_tokens["attention_mask"]), len(train_tokens["input_ids"]))
print(train_tokens["labels"][0])
print(train_tokens["attention_mask"][0], "\n")
print(train_tokens["input_ids"][0], "\n")

                          ####################### Validation Tokenization ######################

val_tokens = tokenizer_obj.training_unified_tokenization(val_cleaned)
print("\n\ntokenization is compelte\n", len(val_tokens["labels"]), len(val_tokens["attention_mask"]), len(val_tokens["input_ids"]))
print(val_tokens["labels"][0], "\n")
print(val_tokens["attention_mask"][0], "\n")
print(val_tokens["input_ids"][0], "\n")


train_dataset = TokenizedDataset(train_tokens)  # Wrap tokenized train data
val_dataset = TokenizedDataset(val_tokens)     # Wrap tokenized validation data

######################################## CUDA CHECK ########################################

if torch.cuda.is_available():
    print("CUDA is available.")
    print(f"Device count: {torch.cuda.device_count()}")  # Number of GPUs available
    print(f"Current device: {torch.cuda.current_device()}")  # Current GPU device
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")  # Name of the GPU
else:
    print("CUDA is NOT available.")

########################################################################################################################
#################################################### Model Setting #####################################################
########################################################################################################################

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


                            ######################### BNB SETUP #########################

# Optional: print args for visibility
print("Using BitsAndBytesConfig with:")
for k, v in four_bit_args.items():
    print(f"{k}: {v}")

bnb_args= BitsAndBytesConfig(**four_bit_args)

                            ######################### LORA SETUP #########################

my_lora_config = lora_default_args.copy()

# Modify parameters if needed
my_lora_config["r"] = 16
my_lora_config["lora_dropout"] = 0.1

peft_config = LoraConfig(**my_lora_config)

                          ######################### MODEL LOADING #########################

model_path = model_names[1]  # "Qwen/Qwen2.5-Math-1.5B"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_args, # Here provide the bnb quantization args
    device_map='auto')

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config).to(device)

# Verify trainable parameters
model.print_trainable_parameters()

########################################################################################################################
#################################################### Model Training ####################################################
########################################################################################################################

custom_args = {
    "output_dir": "./results",
    "logging_strategy": "epoch",
    # "eval_strategy": "epoch", 
    "save_strategy": "epoch",
    "report_to": "none",
    # "metric_for_best_model": "eval_loss",
    # "load_best_model_at_end": True,
    "do_eval": True,
    "num_train_epochs": 3,
    # "use_cache":False,
    "per_device_train_batch_size":  16,
    "per_device_eval_batch_size": 16,
}

args_dict = {**training_args_defaults, **custom_args}
training_args = TrainingArguments(**args_dict)

trainer_wrapper = TrainerBasedTraining(model=model, tokenizer=tokenizer)
trained_model, model_tokenizer = trainer_wrapper.training(training_args=training_args,
                     tokenized_train_dataset=train_dataset,
                     tokenized_validation_dataset=val_dataset,
                     data_collator=data_collator)

########################################################################################################################
#################################################### Model Inference ###################################################
########################################################################################################################

question = "What is 2+2"
# test_loader = BaseDataLoader('AUG_MATH\\test.csv')
# test_data = test_loader.load()

inference_wrapr = InferenceModule(trained_model, model_tokenizer)
# generated_output = inference_wrapr.generator(test_data['problem'])

generated_output = inference_wrapr.generator(question)


