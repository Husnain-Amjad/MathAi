"""
lora_config.py

LoRA Configuration Defaults and Metadata

LoRA (Low-Rank Adaptation) config contains settings for applying parameter-efficient fine-tuning
to transformer models.

Default target_modules are set to cover common LLaMA/Mistral models' projection layers:
  ["q_proj", "k_proj", "v_proj", "o_proj"]

Alternative target_modules for BERT-style models:
  ["query", "key", "value", "dense"]

Alternative target_modules for GPT-style models:
  ["c_attn", "c_proj"]

Users can modify target_modules based on their model architecture.
You can see by using following lines of Code:

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)

---

LoRA Arguments:

- r (int): LoRA rank. Controls bottleneck dimension. Typical values: 4-64.
- alpha (int): LoRA scaling factor. Usually set equal or proportional to r.
- dropout (float): Dropout probability for LoRA layers. Default 0.0 (no dropout).
- target_modules (list[str]): Names of model modules to apply LoRA to.
- merge_weights (bool): Whether to merge LoRA weights into base model weights after training.
- fan_in_fan_out (bool): Whether to transpose weight matrices in LoRA layers.
- bias (str): Bias type in LoRA layers: "none", "all", or "lora_only".

"""

default_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # LLaMA/Mistral style projections
bert_style_target_modules = ["query", "key", "value", "dense"]     # BERT-style attention + FFN
gpt_style_target_modules = ["c_attn", "c_proj"]     # GPT-2 and GPT-J
lora_default_args = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": default_target_modules,
    "inference_mode": False,
    "fan_in_fan_out": False,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

lora_bert_args = {
    **lora_default_args,
    "target_modules": bert_style_target_modules,
}

lora_gpt_args= {
    **lora_default_args,
    "target_modules": gpt_style_target_modules,
}

def print_lora_args_meta():
    print("""
LoRA Configuration Arguments and Description:

- r (int): Low-rank dimension for LoRA adapters (default: 8).
- alpha (int): Scaling factor to apply after LoRA (default: 16).
- dropout (float): Dropout rate in LoRA adapters (default: 0.05).
- target_modules (list of str): List of module names to apply LoRA.
    - Default: ["q_proj", "k_proj", "v_proj", "o_proj"] (LLaMA/Mistral style/General)
    - Alternative: ["query", "key", "value", "dense"] (BERT style)
    - Alternative: ["c_attn", "c_proj"] (GPT style)
- merge_weights (bool): Whether to merge LoRA weights into base model after training (default: False).
- fan_in_fan_out (bool): If True, transpose LoRA weight matrices (default: False).
- bias (str): Bias option for LoRA layers: "none", "all", "lora_only" (default: "none").

Usage Example:

from lora_config import lora_default_args, lora_bert_args

# Use default LLaMA-style LoRA config
my_lora_config = lora_default_args.copy()

# Modify parameters if needed
my_lora_config["r"] = 16
my_lora_config["dropout"] = 0.1

# Or use BERT-style LoRA config
my_lora_config = lora_bert_args.copy()
my_lora_config["alpha"] = 32

# Pass my_lora_config as **kwargs when initializing LoRA adapters in your training pipeline.

Note on QLoRA:
QLoRA combines 4-bit quantization (e.g. bitsandbytes) with LoRA fine-tuning.
Typically, you prepare a quantization config (e.g. from bnb_config.py) and a LoRA config,
then apply LoRA on the quantized model for memory-efficient fine-tuning.

""")

