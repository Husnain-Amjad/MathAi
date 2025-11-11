# MathAi

## Overview

MathAi is a project dedicated to enhancing the reasoning capabilities of Large Language Models (LLMs) on complex mathematical word problems. This repository contains a pipeline designed to fine-tune and evaluate LLMs using the MATH dataset by Hendrycks, which includes a diverse set of challenging math problems ranging from algebra to calculus.

## Table of Contents

- Installation
- Dataset
- Pipeline
- Usage
- Contributing
- License
- Acknowledgements

## Installation

To set up the MathAi pipeline, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Husnain-Amjad/MathAi.git
   cd MathAi
   ```

2. **Install dependencies**: Ensure you have Python 3.8+ installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **MATH dataset**: The MATH dataset by Hendrycks is placed in the `AUG_MATH/` directory.

## Dataset

The MATH dataset, created by Hendrycks et al., consists of 12500 math word problems across various difficulty levels and topics, including:

- Algebra
- Geometry
- Number Theory
- Calculus
- Probability

Each problem includes a question, a step-by-step solution, and the final answer, making it ideal for training and evaluating LLMs on mathematical reasoning.

## Pipeline

The MathAi pipeline is designed to:

1. **Preprocess the MATH dataset**: Clean and format the dataset for model training.
2. **Fine-tune LLMs**: Adapt pre-trained LLMs to improve their performance on math word problems.
3. **Evaluate performance**: Use metrics such as accuracy and step-by-step reasoning correctness to assess model improvements.
4. **Generate solutions**: Allow the fine-tuned model to generate solutions for new math problems.

The pipeline supports popular LLM frameworks like PyTorch and Hugging Face Transformers.

### Directory Structure

```
MathAi/
├── AUG_MATH/              # MATH dataset and related files
├── __pycache__/           # Compiled Python files
├── README.md              # Project documentation
├── base_data_loader.py    # Data loading utilities
├── bnb_config.py          # Configuration for training
├── data_processing.py     # Data preprocessing script (updated for padding tokens)
├── inference.py           # Inference script
├── lora_config.py         # LoRA configuration
├── main.py                
├── no_trainer_based_training.py  # Trainer-less training with comments cleaned
├── requirements.txt       # Dependency list (updated configuration)
├── tokenization.py        # Tokenization script (cleaned of useless code)
├── train.py               # Main Training script (updated for padding tokens)
├── trainer_based_training.py  # Trainer-based training script
├── training_args_config.py    # Updated argument passing for no-trainer
├── training_with_trainer.py   # Training with trainer script
└── training_without_trainer.py  # Updated stratified sampling
```

## Usage

To use the MathAi complete pipeline:

**Train the model**:

   ```bash
   python train.py --model_name <model_path>  --sample_ratio 0.30 --save_steps 300 --logging_steps 1500 --epochs 5 --use_quantization --use_lora --lora_rank 16 --lora_dropout 0.1 --output_dir <directory_path>
   ```

Replace `<model_path>`  with the desired model and `<directory_path>` with output directory path.

## Contributing
Currently repository is under maintainance. 
Contributions will be welcomed soon.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- The MATH dataset by Hendrycks et al. for providing a comprehensive set of math problems.

The open-source community for tools like PyTorch and Hugging Face Transformers.
