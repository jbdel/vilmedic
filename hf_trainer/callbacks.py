"""Custom callbacks for HF Trainer"""
import time
import logging
import os
import glob
import shutil
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class SimplifiedProgressCallback(TrainerCallback):
    """Simplified single-line progress logging with experiment time remaining."""
    
    def __init__(self, logger=None):
        self.experiment_start_time = time.time()
        self.total_steps = None
        self.logger = logger or logging.getLogger("training")
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Initialize at training start."""
        self.experiment_start_time = time.time()
        self.total_steps = state.max_steps
        self.logger.info(f"[Training] Starting training with {self.total_steps} total steps")
        return control
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Override default logging with simplified single-line format."""
        # Only log on the global main process
        if getattr(args, "process_index", 0) != 0:
            return control
        # Backward-compatibility fallback
        if hasattr(args, "local_rank") and args.local_rank not in (-1, 0):
            return control
            
        if logs is None or self.total_steps is None or state.global_step == 0:
            return control
        
        # Calculate time remaining
        elapsed_time = time.time() - self.experiment_start_time
        steps_done = state.global_step
        steps_remaining = self.total_steps - steps_done
        
        if steps_done > 0:
            time_per_step = elapsed_time / steps_done
            time_remaining = steps_remaining * time_per_step
            
            # Format time remaining
            hours = int(time_remaining // 3600)
            minutes = int((time_remaining % 3600) // 60)
            seconds = int(time_remaining % 60)
            time_str = f"{hours:02d}h{minutes:02d}m{seconds:02d}s"
        else:
            time_str = "calculating..."
        
        # Extract metrics, handling both dict and number types
        loss = logs.get('loss', 0.0)
        grad_norm = logs.get('grad_norm', 0.0)
        lr = logs.get('learning_rate', 0.0)
        epoch = logs.get('epoch', state.epoch)
        
        # Format and print single line
        log_line = (f"[Training] Step {steps_done}/{self.total_steps} | "
                   f"loss: {loss:.4f} | grad_norm: {grad_norm:.4f} | "
                   f"lr: {lr:.2e} | epoch: {epoch:.2f} | "
                   f"time_remaining: {time_str}")
        
        # Use print to override tqdm output
        print(log_line, flush=True)
        
        # Also log to file
        if self.logger:
            self.logger.info(log_line)
        
        return control
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Log training completion."""
        total_time = time.time() - self.experiment_start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        self.logger.info(f"[Training] Training completed in {hours:02d}h{minutes:02d}m{seconds:02d}s")
        return control


class EpochCheckpointCallback(TrainerCallback):
    """Save checkpoints at the end of each epoch with epoch number and seed in the name."""
    
    def __init__(self, save_dir, seed=None, save_epochs=True, logger=None):
        self.save_dir = save_dir
        self.seed = seed
        self.save_epochs = save_epochs
        self.logger = logger or logging.getLogger("training")
        self.last_saved_epoch = -1
        
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Rename checkpoint directory to include epoch number and seed after save."""
        if not self.save_epochs:
            return control
        
        # Only act on the global main process
        if getattr(args, "process_index", 0) != 0:
            return control
        if hasattr(args, "local_rank") and args.local_rank not in (-1, 0):
            return control
            
        # Calculate current epoch
        current_epoch = int(state.epoch)
        
        # Only rename if we haven't already renamed for this epoch
        if current_epoch > self.last_saved_epoch:
            try:
                # Re-scan for freshly saved step checkpoint
                checkpoint_pattern = os.path.join(self.save_dir, "checkpoint-[0-9]*")
                checkpoints = glob.glob(checkpoint_pattern)
                step_checkpoints = [cp for cp in checkpoints if "epoch" not in os.path.basename(cp)]

                if not step_checkpoints:
                    # Nothing to rename (could already be renamed)
                    epoch_dir = (
                        f"checkpoint-epoch-{current_epoch}-seed-{self.seed}"
                        if self.seed else f"checkpoint-epoch-{current_epoch}"
                    )
                    if os.path.exists(os.path.join(self.save_dir, epoch_dir)):
                        self.last_saved_epoch = current_epoch
                    return control

                # Pick the highest step checkpoint
                def _step_num(path):
                    try:
                        return int(os.path.basename(path).split("-")[-1])
                    except Exception:
                        return -1
                latest_checkpoint = max(step_checkpoints, key=_step_num)

                if not os.path.exists(latest_checkpoint):
                    return control

                # New checkpoint directory name
                new_name = os.path.join(
                    self.save_dir,
                    f"checkpoint-epoch-{current_epoch}" + (f"-seed-{self.seed}" if self.seed else "")
                )

                # If target exists, skip renaming
                if os.path.exists(new_name):
                    self.last_saved_epoch = current_epoch
                    return control

                # Perform rename
                os.rename(latest_checkpoint, new_name)

                if self.logger:
                    self.logger.info(
                        f"[Checkpoint] Saved checkpoint for epoch {current_epoch} to: {os.path.basename(new_name)}"
                    )
                self.last_saved_epoch = current_epoch
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[Checkpoint] Rename skipped due to: {e}")
        
        return control
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Log epoch completion."""
        epoch_num = int(state.epoch)
        self.logger.info(f"[Training] Completed epoch {epoch_num}/{int(args.num_train_epochs)}")
        return control