import torch
import time
import os
import glob
import re
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import shutil
import math
from peft import PeftModel
from accelerate import Accelerator
import matplotlib.pyplot as plt
from collections import deque
import threading


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
    """Enhanced ManualTraining class that matches your existing interface."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")
        self.model.to(self.device)
        
        # Real-time plotting
        self.step_losses = deque(maxlen=1000)
        self.step_eval_losses = deque(maxlen=1000)
        self.steps = deque(maxlen=1000)
        self.epoch_losses = []
        self.epoch_eval_losses = []
        self.epochs = []
        self.fig = None
        self.plot_thread = None

    def find_latest_checkpoint(self, output_dir):
        """Find the latest checkpoint in the output directory."""
        if not os.path.exists(output_dir) or not os.listdir(output_dir):
            print(f"📁 Output directory {output_dir} is empty. Starting training from scratch.")
            return None
        
        # Find all checkpoint directories
        checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
        checkpoint_dirs = glob.glob(checkpoint_pattern)
        
        if not checkpoint_dirs:
            print(f"📁 No checkpoints found in {output_dir}. Starting training from scratch.")
            return None
        
        # Extract step/epoch numbers and find the latest
        latest_checkpoint = None
        latest_number = -1
        
        for ckpt_dir in checkpoint_dirs:
            match = re.search(r'checkpoint-(\d+)', os.path.basename(ckpt_dir))
            if match:
                number = int(match.group(1))
                if number > latest_number:
                    latest_number = number
                    latest_checkpoint = ckpt_dir
        
        if latest_checkpoint:
            print(f"🔄 Found latest checkpoint: {latest_checkpoint}")
        
        return latest_checkpoint

    def load_checkpoint(self, checkpoint_path, optimizer, scheduler):
        """Load checkpoint and return training state."""
        print(f"⏳ Loading checkpoint from: {checkpoint_path}")
        
        # Load training state
        train_state_path = os.path.join(checkpoint_path, "training_state.pt")
        
        # Initialize default values
        global_step = 0
        start_epoch = 0
        best_eval_loss = float("inf")
        completed_steps = 0
        
        # Load custom training state
        if os.path.exists(train_state_path):
            state = torch.load(train_state_path, map_location="cpu")
            global_step = state.get("global_step", 0)
            start_epoch = state.get("epoch", 0)
            best_eval_loss = state.get("best_eval_loss", float("inf"))
            completed_steps = state.get("completed_steps", 0)
            print(f"✅ Training state loaded: step {global_step}, epoch {start_epoch}")
        else:
            print("⚠️ No training_state.pt found in checkpoint.")
        
        # Load accelerator state (model, optimizer, scheduler)
        try:
            self.accelerator.load_state(checkpoint_path)
            print("✅ Accelerator state (model, optimizer, scheduler) loaded successfully.")
        except Exception as e:
            print(f"⚠️ Could not load accelerator state: {e}")
            print("Continuing with current model state...")
        
        # Return the training state values
        return global_step, start_epoch, best_eval_loss, completed_steps
        
    def _init_plots(self):
        """Initialize real-time plots."""
        plt.ion()
        self.fig, ((self.ax1, self.ax2)) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle('Training Progress')
        
        self.ax1.set_title('Step Loss')
        self.ax1.set_xlabel('Step')
        self.ax1.set_ylabel('Loss')
        
        self.ax2.set_title('Epoch Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Loss')
        
        plt.tight_layout()
        plt.show(block=False)

    def _update_plots(self):
        """Update plots with current data."""
        if not hasattr(self, 'ax1'):
            return
            
        # Step plot
        self.ax1.clear()
        self.ax1.set_title('Step Loss')
        self.ax1.set_xlabel('Step')
        self.ax1.set_ylabel('Loss')
        if self.steps and self.step_losses:
            self.ax1.plot(list(self.steps), list(self.step_losses), 'b-', label='Train', alpha=0.7)
        if self.steps and self.step_eval_losses:
            eval_steps = [s for i, s in enumerate(self.steps) if i < len(self.step_eval_losses)]
            self.ax1.plot(eval_steps, list(self.step_eval_losses), 'r-', label='Eval', alpha=0.7)
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Epoch plot
        self.ax2.clear()
        self.ax2.set_title('Epoch Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Loss')
        if self.epochs and self.epoch_losses:
            self.ax2.plot(self.epochs, self.epoch_losses, 'b-o', label='Train', alpha=0.7)
        if self.epochs and self.epoch_eval_losses:
            self.ax2.plot(self.epochs, self.epoch_eval_losses, 'r-o', label='Eval', alpha=0.7)
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def training(self, training_args, tokenized_train_dataset, tokenized_validation_dataset, data_collator):
        """Main training method that matches your existing interface."""
        
        # Prepare timing
        timing_tracker = TimingTracker()
        
        # Check for existing checkpoints
        latest_checkpoint = self.find_latest_checkpoint(training_args.output_dir)

        # Combine train and validation datasets for training (if validation exists)
        if tokenized_validation_dataset is not None:
            print(f"🔗 Combining train ({len(tokenized_train_dataset)}) and validation ({len(tokenized_validation_dataset)}) datasets for training")
            combined_dataset = ConcatDataset([tokenized_train_dataset, tokenized_validation_dataset])
        else:
            combined_dataset = tokenized_train_dataset
            print("📊 Using only training dataset (no validation dataset provided)")

        # Dataloaders
        train_dataloader = DataLoader(
            combined_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator
        )

        # Keep separate validation dataloader for evaluation
        eval_dataloader = None
        if tokenized_validation_dataset is not None:
            eval_dataloader = DataLoader(
                tokenized_validation_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=data_collator
            )

        # Initialize training variables
        best_eval_loss = float("inf")
        best_model_state = None
        global_step = 0
        start_epoch = 0
        total_loss = 0.0
        saved_checkpoints = []
        completed_steps = 0
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay
        )

        # Scheduler
        num_epochs = training_args.num_train_epochs
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = training_args.warmup_steps
        
        if warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_steps
            )
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)

        # Training arguments
        eval_steps = training_args.eval_steps
        save_steps = training_args.save_steps
        logging_steps = training_args.logging_steps
        save_total_limit = training_args.save_total_limit
        load_best_model_at_end = training_args.load_best_model_at_end
        eval_strategy = training_args.eval_strategy

        # Prepare with accelerator
        self.model, optimizer, train_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader
        )
        if eval_dataloader:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)

        # Load checkpoint if found
        if latest_checkpoint:
            global_step, start_epoch, best_eval_loss, completed_steps = self.load_checkpoint(
                latest_checkpoint, optimizer, scheduler
            )
            # Adjust total_loss based on completed steps
            total_loss = 0.0  # Reset for current session tracking

        # Create output directory
        os.makedirs(training_args.output_dir, exist_ok=True)

        # Initialize plots if enabled
        show_plots = getattr(training_args, 'show_plots', True)
        if show_plots:
            self._init_plots()

        timing_tracker.on_train_begin(num_epochs - start_epoch)
        self.model.train()

        print(f"🚀 Starting training from epoch {start_epoch + 1}/{num_epochs}")
        print(f"📊 Combined training samples: {len(combined_dataset)}")
        if eval_dataloader:
            print(f"📈 Validation samples for evaluation: {len(tokenized_validation_dataset)}")

        for epoch in range(start_epoch, num_epochs):
            timing_tracker.on_epoch_begin(epoch, num_epochs)
            epoch_loss = 0.0
            epoch_steps = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            # Skip batches if we're resuming from a checkpoint
            batch_skip_count = 0
            if global_step > 0 and epoch == start_epoch:
                # Calculate how many batches to skip based on global_step
                batches_per_epoch = len(train_dataloader)
                batch_skip_count = global_step % batches_per_epoch
                print(f"🔄 Resuming from step {global_step}, skipping {batch_skip_count} batches in epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                # Skip steps if resuming from checkpoint
                if batch_skip_count > 0:
                    batch_skip_count -= 1
                    continue

                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step_loss = loss.item()
                total_loss += step_loss
                epoch_loss += step_loss
                global_step += 1
                epoch_steps += 1

                # Update step plots
                if show_plots:
                    self.steps.append(global_step)
                    self.step_losses.append(step_loss)
                    if global_step % 10 == 0:  # Update every 10 steps
                        self._update_plots()

                progress_bar.set_postfix({
                    "loss": f"{step_loss:.4f}",
                    "avg_loss": f"{total_loss / max(global_step, 1):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })

                # Logging
                if global_step % logging_steps == 0:
                    print(f"📊 Step {global_step}: Loss = {step_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}")

                # Evaluation
                if eval_dataloader and eval_strategy == "steps" and global_step % eval_steps == 0:
                    train_loss = step_loss
                    eval_loss = self._evaluate(eval_dataloader)
                    print(f"📈 Step {global_step}: Train Loss = {train_loss:.4f}, Eval Loss = {eval_loss:.4f}")
                    self.model.train()

                    # Update eval plots
                    if show_plots:
                        self.step_eval_losses.append(eval_loss)
                        self._update_plots()

                    # Save best model
                    if load_best_model_at_end and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        print(f"💾 New best model found! Eval Loss: {eval_loss:.4f}")

                # Save checkpoint
                if global_step % save_steps == 0:
                    ckpt_path = self._save_checkpoint(training_args.output_dir, global_step, epoch, 
                                                   best_eval_loss, saved_checkpoints, save_total_limit)
                    if ckpt_path:
                        saved_checkpoints.append(ckpt_path)

            # End of epoch evaluation
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            print(f"📊 Epoch {epoch + 1} completed - Average Train Loss: {avg_epoch_loss:.4f}")
            
            # Update epoch plots
            if show_plots:
                self.epochs.append(epoch + 1)
                self.epoch_losses.append(avg_epoch_loss)
            
            if eval_dataloader and eval_strategy == "epoch":
                eval_loss = self._evaluate(eval_dataloader)
                print(f"📈 Epoch {epoch + 1}: Train Loss = {avg_epoch_loss:.4f}, Eval Loss = {eval_loss:.4f}")
                self.model.train()

                # Update epoch eval plots
                if show_plots:
                    self.epoch_eval_losses.append(eval_loss)

                if load_best_model_at_end and eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    print(f"💾 New best model found! Eval Loss: {eval_loss:.4f}")

            # Update plots at end of epoch
            if show_plots:
                self._update_plots()

            timing_tracker.on_epoch_end(epoch)

            # Save checkpoint at end of epoch if using epoch-based saving
            if hasattr(training_args, 'save_strategy') and getattr(training_args, 'save_strategy') == "epoch":
                ckpt_path = self._save_checkpoint(training_args.output_dir, f"epoch_{epoch+1}", epoch+1, 
                                               best_eval_loss, saved_checkpoints, save_total_limit)
                if ckpt_path:
                    saved_checkpoints.append(ckpt_path)

        timing_tracker.on_train_end()

        # Load best model if requested
        if load_best_model_at_end and best_model_state is not None:
            print("🏆 Loading best model weights based on validation loss...")
            self.model.load_state_dict(best_model_state)

        # Save final model
        self._save_model(training_args.output_dir)
        
        # Keep plots open
        if show_plots:
            plt.ioff()
            plt.show()
            
        print(f"🎉 Training completed! Model saved to {training_args.output_dir}")
        return self.model, self.tokenizer

    def _evaluate(self, eval_dataloader):
        """Evaluate the model and return average loss."""
        self.model.eval()
        total_eval_loss = 0.0
        steps = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
                outputs = self.model(**batch)
                total_eval_loss += outputs.loss.item()
                steps += 1
        return total_eval_loss / max(steps, 1)

    def _save_checkpoint(self, output_dir, step_or_epoch, epoch, best_eval_loss, saved_checkpoints, save_total_limit):
        """Save checkpoint and manage checkpoint limits."""
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step_or_epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        try:
            # For quantized LoRA models, save components separately
            if hasattr(self.model, 'peft_config'):
                # Save LoRA adapter
                self.model.save_pretrained(ckpt_dir)
                self.tokenizer.save_pretrained(ckpt_dir)
                
                # Save optimizer and scheduler manually
                torch.save(self.accelerator.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.bin"))
                torch.save(self.accelerator.lr_scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.bin"))
                
                print(f"✅ LoRA checkpoint saved to {ckpt_dir}")
            else:
                # For non-LoRA models, use accelerator
                self.accelerator.save_state(ckpt_dir)
                print(f"✅ Standard checkpoint saved to {ckpt_dir}")
            
            # Save custom training state
            training_state = {
                "epoch": epoch,
                "global_step": step_or_epoch if isinstance(step_or_epoch, int) else 0,
                "best_eval_loss": best_eval_loss,
                "completed_steps": 0
            }
            torch.save(training_state, os.path.join(ckpt_dir, "training_state.pt"))
            
            # Manage checkpoint limit
            if save_total_limit and len(saved_checkpoints) >= save_total_limit:
                oldest_checkpoint = saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    shutil.rmtree(oldest_checkpoint, ignore_errors=True)
                    print(f"🗑️ Removed old checkpoint: {oldest_checkpoint}")
            
            return ckpt_dir
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            return None

    def _save_model(self, output_dir):
        """Save the final trained model."""
        os.makedirs(output_dir, exist_ok=True)
        try:
            if isinstance(self.model, PeftModel):
                print("🔗 Merging LoRA weights...")
                model_to_save = self.model.merge_and_unload()
            else:
                model_to_save = self.model
            
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"✅ Final model saved to {output_dir}")
        except Exception as e:
            print(f"❌ Failed to save model: {e}")