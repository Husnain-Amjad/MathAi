import torch
import time
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from peft import PeftModel
from accelerate import Accelerator


class TimingTracker:
    """Custom class to track training times (replaces TimingCallback)."""
    def __init__(self):
        self.epoch_times = []
        self.total_start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, num_epochs):
        self.total_start_time = time.time()
        print("Training started...")

    def on_epoch_begin(self, epoch, num_epochs):
        self.epoch_start_time = time.time()
        print(f"Starting epoch {epoch + 1}/{num_epochs}")

    def on_epoch_end(self, epoch):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")

    def on_train_end(self):
        total_time = time.time() - self.total_start_time
        print(f"Total training time: {total_time:.2f} seconds")
        if self.epoch_times:
            avg_time = sum(self.epoch_times) / len(self.epoch_times)
            print(f"Average time per epoch: {avg_time:.2f} seconds")


class ManualTraining:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")
        self.model.to(self.device)

    def training(self, training_args, tokenized_train_dataset, tokenized_validation_dataset, data_collator):
        # Prepare timing
        timing_tracker = TimingTracker()

        # Dataloaders
        train_dataloader = DataLoader(
            tokenized_train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator
        )

        eval_dataloader = None
        if tokenized_validation_dataset is not None:
            eval_dataloader = DataLoader(
                tokenized_validation_dataset,
                batch_size=getattr(training_args, 'per_device_eval_batch_size', training_args.per_device_train_batch_size),
                shuffle=False,
                collate_fn=data_collator
            )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=getattr(training_args, 'weight_decay', 0.01)
        )

        # Scheduler
        num_epochs = training_args.num_train_epochs
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = getattr(training_args, 'warmup_steps', 0)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1 if warmup_steps > 0 else 1.0,
            total_iters=warmup_steps if warmup_steps > 0 else 1
        )

        eval_steps = getattr(training_args, 'eval_steps', len(train_dataloader))
        save_steps = getattr(training_args, 'save_steps', len(train_dataloader))
        logging_steps = getattr(training_args, 'logging_steps', 50)
        save_total_limit = getattr(training_args, 'save_total_limit', None)
        load_best_model_at_end = getattr(training_args, 'load_best_model_at_end', False)
        eval_strategy = getattr(training_args, 'eval_strategy', 'steps')

        self.model, optimizer, train_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader
        )
        if eval_dataloader:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)

        best_eval_loss = float("inf")
        best_model_state = None
        global_step = 0
        total_loss = 0.0
        saved_checkpoints = []

        timing_tracker.on_train_begin(num_epochs)
        self.model.train()

        for epoch in range(num_epochs):
            timing_tracker.on_epoch_begin(epoch, num_epochs)
            epoch_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                outputs = self.model(**batch)
                loss = outputs.loss / self.accelerator.gradient_accumulation_steps
                self.accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step_loss = loss.item()
                total_loss += step_loss
                epoch_loss += step_loss
                global_step += 1

                progress_bar.set_postfix({
                    "loss": f"{step_loss:.4f}",
                    "avg_loss": f"{total_loss / global_step:.4f}"
                })

                if global_step % logging_steps == 0:
                    print(f"Step {global_step}: Loss = {step_loss:.4f}")

                if eval_dataloader and eval_strategy == "steps" and global_step % eval_steps == 0:
                    eval_loss = self._evaluate(eval_dataloader)
                    print(f"Step {global_step}: Eval Loss = {eval_loss:.4f}")
                    self.model.train()

                    if load_best_model_at_end and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_model_state = self.model.state_dict()

                if global_step % save_steps == 0:
                    ckpt_path = self._save_checkpoint(training_args.output_dir, global_step)
                    saved_checkpoints.append(ckpt_path)
                    if save_total_limit and len(saved_checkpoints) > save_total_limit:
                        to_remove = saved_checkpoints.pop(0)
                        shutil.rmtree(to_remove, ignore_errors=True)

            print(f"Epoch {epoch + 1} average loss: {epoch_loss / len(train_dataloader):.4f}")
            timing_tracker.on_epoch_end(epoch)

        timing_tracker.on_train_end()

        if load_best_model_at_end and best_model_state is not None:
            print("Loading best model weights based on validation loss...")
            self.model.load_state_dict(best_model_state)

        self._save_model(training_args.output_dir)
        print(f"Training completed! Model saved to {training_args.output_dir}")
        return self.model, self.tokenizer

    def _evaluate(self, eval_dataloader):
        self.model.eval()
        total_eval_loss = 0.0
        steps = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                outputs = self.model(**batch)
                total_eval_loss += outputs.loss.item()
                steps += 1
        return total_eval_loss / steps

    def _save_checkpoint(self, output_dir, step):
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        print(f"Checkpoint saved at step {step}")
        return ckpt_dir

    def _save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if isinstance(self.model, PeftModel):
            print("Merging weights of LoRA model...")
            model_to_save = self.model.merge_and_unload()
        else:
            model_to_save = self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


# Example usage:
"""
# Initialize the manual trainer
manual_trainer = ManualTraining(model, tokenizer)

# Train the model
manual_trainer.training(
    training_args=training_args,
    tokenized_train_dataset=tokenized_train_dataset,
    tokenized_validation_dataset=tokenized_validation_dataset,
    data_collator=data_collator
)
"""