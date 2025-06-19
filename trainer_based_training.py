from transformers import Trainer, TrainerCallback
import time
from peft import PeftModel

# Assume `training_args_defaults` comes from your training_args_config.py
# training_args = TrainingArguments(**training_args_defaults)

class TimingCallback(TrainerCallback):
    """Custom callback to track training times."""
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.total_start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_start_time = time.time()
        print("Training started...")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        print(f"Starting epoch {int(state.epoch) + 1}/{int(args.num_train_epochs)}")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        print(f"Epoch {int(state.epoch) + 1} completed in {epoch_time:.2f} seconds")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.total_start_time
        print(f"Total training time: {total_time:.2f} seconds")
        if self.epoch_times:
            avg_time = sum(self.epoch_times) / len(self.epoch_times)
            print(f"Average time per epoch: {avg_time:.2f} seconds")

class TrainerBasedTraining:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def training(self, training_args, tokenized_train_dataset, tokenized_validation_dataset, data_collator):
        trainer = Trainer(
            model=self.model,
            args=training_args,  
            train_dataset=tokenized_train_dataset,  
            eval_dataset=tokenized_validation_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,  
            callbacks=[TimingCallback()]
        )

        trainer.train()

        # Merge LoRA weights into base model after training
        if isinstance(trainer.model, PeftModel):
            print("Mering Weights of LoRa...")
            merged_model = trainer.model.merge_and_unload()
            print("Weights Merged")
        else:
            merged_model = trainer.model

        # Save merged model and tokenizer
        merged_model.save_pretrained(training_args.output_dir)
        self.tokenizer.save_pretrained(training_args.output_dir)

        return trainer, self.tokenizer

