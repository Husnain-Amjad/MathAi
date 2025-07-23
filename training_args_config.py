"""
training_args_config.py

Configuration for Hugging Face Trainer's TrainingArguments
with metadata for each argument: default value, description, and usage examples.

Usage:
    from transformers import TrainingArguments
    from training_args_config import training_args_defaults

    # Update args as needed
    custom_args = {
        "output_dir": "./results",
        "num_train_epochs": 3,
    }
    args_dict = {**training_args_defaults, **custom_args}

    training_args = TrainingArguments(**args_dict)

"""

from dataclasses import dataclass, field

training_args_metadata = {
    "output_dir": {
        "default": "./results",
        "description": "The output directory where the model predictions and checkpoints will be written.",
    },
    "overwrite_output_dir": {
        "default": False,
        "description": "Overwrite the content of the output directory. Use with caution.",
    },
    "do_train": {
        "default": True,
        "description": "Whether to run training.",
    },
    "do_eval": {
        "default": True,
        "description": "Whether to run evaluation on the validation set.",
    },
    "per_device_train_batch_size": {
        "default": 8,
        "description": "Batch size per GPU/CPU for training.",
    },
    "per_device_eval_batch_size": {
        "default": 8,
        "description": "Batch size per GPU/CPU for evaluation.",
    },
    "learning_rate": {
        "default": 5e-5,
        "description": "The initial learning rate for Adam optimizer.",
    },
    "weight_decay": {
        "default": 0.0,
        "description": "Weight decay to apply (if any).",
    },
    "gradient_accumulation_steps": {
        "default": 1,
        "description": "Number of updates steps to accumulate before performing a backward/update pass.",
    },
    "warmup_steps": {
        "default": 0,
        "description": "Number of steps for learning rate warmup.",
    },
    "warmup_ratio": {
        "default": 0.0,
        "description": "Ratio of total steps for LR warmup (overridden by warmup_steps if set).",
    },
    "report_to": {
        "default": "tensorboard",
        "description": "The list of integrations to report the results and logs to.",
    },
    "adam_beta1": {
        "default": 0.9,
        "description": "Beta1 parameter for Adam optimizer.",
    },
    "adam_beta2": {
        "default": 0.999,
        "description": "Beta2 parameter for Adam optimizer.",
    },
    "adam_epsilon": {
        "default": 1e-8,
        "description": "Epsilon parameter for Adam optimizer.",
    },
    "max_grad_norm": {
        "default": 1.0,
        "description": "Maximum gradient norm for gradient clipping.",
    },
    "lr_scheduler_type": {
    "default": "linear",
    "description": (
        "Type of learning rate scheduler to use during training. "
        "Supported options include:\n"
        "  - 'linear': Linear warmup and then linear decay\n"
        "  - 'cosine': Linear warmup followed by cosine decay\n"
        "  - 'cosine_with_restarts': Cosine decay with restarts\n"
        "  - 'polynomial': Polynomial warmup and decay\n"
        "  - 'constant': Constant learning rate (no decay)\n"
        "  - 'constant_with_warmup': Constant LR after a warmup phase\n"
        "Choose the scheduler that best fits your training behavior and model sensitivity."),
    },
    "num_train_epochs": {
        "default": 1,
        "description": "Total number of training epochs to perform.",
    },
    "logging_dir": {
        "default": "./logs",
        "description": "Tensorboard log directory.",
    },
    "logging_steps": {
        "default": 50,
        "description": "Log every X update steps.",
    },
    "eval_strategy": {
        "default": "steps",
        "description": "Evaluation strategy to adopt during training. Options: 'no', 'steps', 'epoch'.",
    },
    "eval_steps": {
        "default": 500,
        "description": "Run evaluation every X steps if evaluation_strategy is 'steps'.",
    },
    "save_strategy": {
        "default": "epoch",
        "description": "The checkpoint save strategy. Options: 'no', 'epoch', 'steps'.",
    },
    "save_steps": {
        "default": 500,
        "description": "Save checkpoint every X steps if save_strategy is 'steps'.",
    },
    "save_total_limit": {
        "default": 2,
        "description": "Limit the total amount of checkpoints. Deletes the older checkpoints.",
    },
    "seed": {
        "default": 42,
        "description": "Random seed for reproducibility.",
    },
    "fp16": {
        "default": True,
        "description": "Whether to use 16-bit (mixed) precision training.",
    },
    "load_best_model_at_end": {
        "default": True,
        "description": "Whether to load the best model found during training at the end of training.",
    },
    "metric_for_best_model": {
        "default": None,
        "description": "Metric to use to evaluate the best model if load_best_model_at_end is True.",
    },
    "greater_is_better": {
        "default": None,
        "description": "Whether the metric for the best model should be maximized or minimized.",
    },
    "load_checkpoint": {
        "default": False,
        "description": "Whether to load checkpoint or not"
    }
}

# Default training arguments dictionary extracted from metadata
training_args_defaults = {k: v["default"] for k, v in training_args_metadata.items()}

# For ManualTraining class (non-HF Trainer)
non_trainer_args_defaults = {
    "output_dir": "./manual_output",
    "num_train_epochs": 10,
    "per_device_eval_batch_size": 2,
    "learning_rate": 2e-5,
    "warmup_steps": 100,
    "save_steps": 500,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "save_total_limit": 2,
    "fp16": True,
    "logging_strategy": "steps",
    "log_level": "info",
    "load_best_model_at_end": True,        
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "logging_steps": 50,
    "load_checkpoint": False
}


non_trainer_args_metadata = {
    "output_dir": "Directory where checkpoints and the final model will be saved.",
    "num_train_epochs": "Total number of training epochs.",
    "per_device_train_batch_size": "Batch size allocated to each device during training.",
    "per_device_eval_batch_size": "Batch size allocated to each device during evaluation.",
    "learning_rate": "Initial learning rate for the optimizer.",
    "weight_decay": "Weight decay to apply (L2 regularization).",
    "warmup_steps": "Number of warmup steps for the learning rate scheduler.",
    "save_steps": "Number of steps between model checkpoint saves.",
    "eval_strategy": "Evaluation strategy (e.g., 'steps' or 'epoch').",
    "eval_steps": "Number of steps between evaluation runs.",
    "save_total_limit": "Maximum number of checkpoints to keep. Older ones will be deleted.",
    "fp16": "Whether to use 16-bit (mixed) precision training (if supported by hardware).",
    "logging_strategy": "Logging strategy to use (e.g., 'steps').",
    "log_level": "Verbosity level of logging (e.g., 'info', 'debug').",
    "load_best_model_at_end": "Whether to load the best model (based on evaluation metric) at the end of training.",
    "logging_steps": "Number of steps between logging updates.",
    "load_checkpoint": "Whether to load checkpoint or not"
}


def print_training_args_metadata():
    print("Training Arguments Metadata:\n")
    for arg, meta in training_args_metadata.items():
        print(f"{arg} (default={meta['default']}):")
        print(f"    {meta['description']}\n")


if __name__ == "__main__":
    # Example: print all metadata to console
    print_training_args_metadata()
