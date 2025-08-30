# MathAi

## Overview

MathAi is a project dedicated to enhancing the reasoning capabilities of Large Language Models (LLMs) on complex mathematical word problems. This repository contains a pipeline designed to fine-tune and evaluate LLMs using the MATH dataset by Hendrycks, which includes a diverse set of challenging math problems ranging from algebra to calculus.
MathAi
Overview
MathAi is a project dedicated to enhancing the reasoning capabilities of Large Language Models (LLMs) on complex mathematical word problems. This repository contains a pipeline designed to fine-tune and evaluate LLMs using the MATH dataset by Hendrycks, which includes a diverse set of challenging math problems ranging from algebra to calculus.
Table of Contents

Installation
Dataset
Pipeline
Usage
Contributing
License
Acknowledgements

Installation
To set up the MathAi pipeline, follow these steps:

Clone the repository:
git clone https://github.com/Husnain-Amjad/MathAi.git
cd MathAi


Install dependencies:Ensure you have Python 3.8+ installed. Then, install the required packages:
pip install -r requirements.txt


Download the MATH dataset:The MATH dataset by Hendrycks is required. Download it from the official repository and place it in the data/ directory.


Dataset
The MATH dataset, created by Hendrycks et al., consists of thousands of math word problems across various difficulty levels and topics, including:

Algebra
Geometry
Number Theory
Calculus
Probability

Each problem includes a question, a step-by-step solution, and the final answer, making it ideal for training and evaluating LLMs on mathematical reasoning.
Pipeline
The MathAi pipeline is designed to:

Preprocess the MATH dataset: Clean and format the dataset for model training.
Fine-tune LLMs: Adapt pre-trained LLMs to improve their performance on math word problems.
Evaluate performance: Use metrics such as accuracy and step-by-step reasoning correctness to assess model improvements.
Generate solutions: Allow the fine-tuned model to generate solutions for new math problems.

The pipeline supports popular LLM frameworks like PyTorch and Hugging Face Transformers.
Directory Structure
MathAi/
├── data/                # MATH dataset and preprocessed files
├── models/              # Fine-tuned model checkpoints
├── scripts/             # Training and evaluation scripts
├── requirements.txt     # Python dependencies
└── README.md            # This file

Usage
To use the MathAi pipeline:

Preprocess the dataset:
python scripts/preprocess.py --data_path data/MATH/


Train the model:
python scripts/train.py --model_name <model_name> --data_path data/processed/


Evaluate the model:
python scripts/evaluate.py --model_path models/<model_checkpoint> --test_data data/processed/test.json


Generate solutions:
python scripts/generate.py --model_path models/<model_checkpoint> --input "Solve the equation 2x + 3 = 7"



Replace <model_name> and <model_checkpoint> with the desired model (e.g., bert-base-uncased or a custom checkpoint).
Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit (git commit -m "Add feature").
Push to the branch (git push origin feature-branch).
Open a pull request.

Please ensure your code follows the project's coding standards and includes relevant tests.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgements

The MATH dataset by Hendrycks et al. for providing a comprehensive set of math problems.
The open-source community for tools like PyTorch and Hugging Face Transformers.
