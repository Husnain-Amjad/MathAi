import torch
import time
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from peft import PeftModel

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
        self.model.to(self.device)

    def training(self, training_args, tokenized_train_dataset, tokenized_validation_dataset, data_collator):
        """
        Manual training function that replicates Trainer functionality.
        
        Args:
            training_args: Should contain attributes like:
                - num_train_epochs
                - per_device_train_batch_size
                - per_device_eval_batch_size
                - learning_rate
                - output_dir
                - eval_steps (optional)
                - save_steps (optional)
                - logging_steps (optional)
                - warmup_steps (optional)
                - weight_decay (optional)
        """
        
        # Initialize timing tracker
        timing_tracker = TimingTracker()
        
        # Create data loaders
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
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=getattr(training_args, 'weight_decay', 0.01)
        )
        
        # Calculate total training steps
        num_epochs = training_args.num_train_epochs
        total_steps = len(train_dataloader) * num_epochs
        
        # Setup learning rate scheduler
        warmup_steps = getattr(training_args, 'warmup_steps', 0)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1 if warmup_steps > 0 else 1.0,
            total_iters=warmup_steps if warmup_steps > 0 else 1
        )
        
        # Training configuration
        eval_steps = getattr(training_args, 'eval_steps', len(train_dataloader))
        save_steps = getattr(training_args, 'save_steps', len(train_dataloader))
        logging_steps = getattr(training_args, 'logging_steps', 100)
        
        # Start training
        timing_tracker.on_train_begin(num_epochs)
        self.model.train()
        
        global_step = 0
        total_loss = 0.0
        
        for epoch in range(num_epochs):
            timing_tracker.on_epoch_begin(epoch, num_epochs)
            epoch_loss = 0.0
            
            # Training loop for one epoch
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update metrics
                global_step += 1
                step_loss = loss.item()
                total_loss += step_loss
                epoch_loss += step_loss
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{step_loss:.4f}',
                    'avg_loss': f'{total_loss / global_step:.4f}'
                })
                
                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = total_loss / global_step
                    print(f"Step {global_step}: Loss = {step_loss:.4f}, Avg Loss = {avg_loss:.4f}")
                
                # Evaluation
                if eval_dataloader is not None and global_step % eval_steps == 0:
                    eval_loss = self._evaluate(eval_dataloader)
                    print(f"Step {global_step}: Eval Loss = {eval_loss:.4f}")
                    self.model.train()  # Switch back to training mode
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    self._save_checkpoint(training_args.output_dir, global_step)
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
            timing_tracker.on_epoch_end(epoch)
            
            # Evaluation at end of epoch
            if eval_dataloader is not None:
                eval_loss = self._evaluate(eval_dataloader)
                print(f"End of epoch {epoch + 1}: Eval Loss = {eval_loss:.4f}")
                self.model.train()
        
            # End training
            timing_tracker.on_train_end()

            # Save final model
            if isinstance(self.model, PeftModel):
                print("LoRA detected. Merging and saving final model...")
                self.model = self.model.merge_and_unload()  # merge LoRA weights into base model
            else:
                print("Saving standard model...")

            self._save_model(training_args.output_dir)
            print(f"Training completed! Model saved to {training_args.output_dir}")
        # Return the final model and tokenizer
        return self.model, self.tokenizer
    
    def _evaluate(self, eval_dataloader):
        """Evaluate the model on validation dataset."""
        self.model.eval()
        total_eval_loss = 0.0
        num_eval_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_eval_loss += outputs.loss.item()
                num_eval_steps += 1
        
        return total_eval_loss / num_eval_steps if num_eval_steps > 0 else 0.0
    
    def _save_checkpoint(self, output_dir, step):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model state
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        print(f"Checkpoint saved at step {step}")

    def _save_model(self, output_dir):
        """Save the final trained model."""
        os.makedirs(output_dir, exist_ok=True)
        if isinstance(self.model, PeftModel):
            print("Mering Weights of LoRa...")
            merged_model = self.model.merge_and_unload()
            print("Weights Merged")
        else:
            merged_model = self.model
        merged_model.save_pretrained(output_dir)
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