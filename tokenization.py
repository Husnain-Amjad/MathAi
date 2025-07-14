# tokenization.py

"""
Tokenization module for training and inference preprocessing.

Description:
    This module provides a Tokenizer class that encapsulates tokenization logic
    for training and inference phases of a causal language model task.
    It supports concatenation of question and answer with EOS tokens, padding,
    truncation, and label preparation.


Example:

from tokenization import Tokenization
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-model-name")

tokenizer_obj = Tokenization(tokenizer)

train_tokens = tokenizer_obj.training_tokenization({
    "question": ["What is AI?", "Define machine learning."],
    "answer": ["Artificial Intelligence.", "A subset of AI."]
})

inference_tokens = tokenizer_obj.inference_tokenization({
    "question": ["What is AI?", "Define machine learning."]
})

"""

from typing import List, Dict
from torch.utils.data import Dataset

class TokenizedDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["labels"][idx],
        }

class Tokenization:
    def __init__(self, tokenizer, max_length: int = 512):
        """
        Initialize the Tokenization class.

        Args:
            tokenizer: A Hugging Face tokenizer instance.
            max_length (int): Maximum token length for padding/truncation.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos = self.tokenizer.eos_token if self.tokenizer.eos_token else ""
        print("\n EOS TOKEN", self.eos)

    def training_unified_tokenization(self, examples: Dict[str, List[str]]) -> Dict:
        """
        Tokenize examples for training.
        Concatenates question and answer with EOS token and prepares labels.

        Args:
            examples (Dict): Dictionary with keys "problem" and "solution", both lists of strings.

        Returns:
            Dict: Dictionary with tokenized inputs and labels.
        """
        print("\n NEW EOS TOKEN", self.eos)
        self.eos = self.tokenizer.eos_token if self.tokenizer.eos_token else ""

        inputs = []
        
        for q, a in zip(examples["problem"], examples["solution"]):
            print(f"question: {q}\n", f"\nAnswer: {a}\n", f"\nUnified Answer: {(q + self.eos + a + self.eos)}")
            inputs.append(q + self.eos + a + self.eos)

        tokenized_inputs = self.tokenizer(
            inputs,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        new_labels = []

        for input_ids in tokenized_inputs["input_ids"]:
            labels = input_ids.copy()

            eos_positions = [i for i, token in enumerate(input_ids) if token == self.eos]

            if len(eos_positions) >= 2:
                first_eos_index = eos_positions[0]

                for i in range(first_eos_index + 1):
                    labels[i] = -100
            else:

                labels[:] = [-100] * len(labels)

            new_labels.append(labels)

        tokenized_inputs["labels"] = new_labels
        


        return tokenized_inputs
    
    def training_paired_tokenization(self, examples: Dict[str, List[str]]) -> Dict:
        """
        Tokenize examples for training a Seq2Seq model.
        Tokenizes inputs (problems) and target outputs (solutions) separately.

        Args:
            examples (Dict): Dictionary with keys "problem" and "solution", both lists of strings.

        Returns:
            Dict: Tokenized inputs with labels for training.
        """
        inputs = []
        targets = []
        for q, a in zip(examples["problem"], examples["solution"]):
            inputs.append(q + self.eos)
            targets.append(a + self.eos)

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


    def inference_tokenization(self, examples: Dict[str, List[str]]) -> Dict:
        """
        Tokenize examples for inference.
        Concatenates question only with EOS token, no labels needed.

        Args:
            examples (Dict): Dictionary with key "question" as list of strings.

        Returns:
            Dict: Tokenized inputs without labels.
        """
        inputs = []
        for q in zip(examples["problem"]):
            inputs.append(q + self.eos)

        tokenized_inputs = self.tokenizer(
            inputs,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )

        return tokenized_inputs
