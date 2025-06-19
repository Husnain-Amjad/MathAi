"""
bnb_config.py

This file provides pre-defined configuration dictionaries for using
bitsandbytes 8-bit and 4-bit quantization settings with Hugging Face Transformers.

Usage Example:
--------------
from bnb_config import four_bit_args, eight_bit_args

bnb_args= BitsAndBytesConfig(**four_bit_args)
bnb_args= BitsAndBytesConfig(**eight_bit_args)

You can also override default values like:

bnb_args = {**four_bit_args, "bnb_4bit_use_double_quant": False}
bnb_args = {**eight_bit_args, "llm_int8_enable_fp32_cpu_offload": True}

You can quantize the model as following:

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_args, # Here provide the bnb quantization args
    device_map="auto"  # auto-assign GPU/CPU
)
--------------
"""
import torch
# -------------------------------
# 8-BIT QUANTIZATION CONFIGURATION
# -------------------------------

eight_bit_args = {
    "load_in_8bit": False,  # (bool) Enable 8-bit quantization. Set to True to reduce memory usage.
    
    "llm_int8_threshold": 6.0,  # (float) Threshold for outlier detection. Higher means fewer outliers kept in FP16.

    "llm_int8_skip_modules": None,  # (list of str or None) Skip these modules from quantization, e.g., ["lm_head"]

    "llm_int8_enable_fp32_cpu_offload": False,  # (bool) Offload FP32 modules to CPU to avoid GPU OOM.

    "llm_int8_has_fp16_weight": False,  # (bool) Set True if the model weights are already in fp16.
}

# -------------------------------
# 4-BIT QUANTIZATION CONFIGURATION
# -------------------------------

four_bit_args = {
    "load_in_4bit": True,  # (bool) Enable 4-bit quantization for maximum memory saving.
    
    "bnb_4bit_quant_type": "nf4",  # (str) Quantization type.
                                   # Options: "nf4" (recommended), "fp4"
    
    "bnb_4bit_compute_dtype": torch.float16,  # (torch.dtype) Dtype used during computation.
                                              # Options: torch.float16, torch.bfloat16, torch.float32

    "bnb_4bit_use_double_quant": True,  # (bool) Apply second quantization step. Saves additional memory.

    "bnb_4bit_quant_storage": None,  # (torch.dtype or None) Data type used to store quantized weights.
                                     # If None, default is usually torch.uint8
}
