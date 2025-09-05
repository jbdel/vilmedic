import torch
from transformers import Trainer
from typing import Dict, Union, Any, Optional, List, Tuple
import numpy as np
from torch import nn


class VisionLanguageTrainer(Trainer):
    """Custom Trainer for Vision-Language models"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Compute the training loss for vision-language model
        """
        # Get labels from inputs (do not mutate inputs)
        labels = inputs["input_ids"]
        
        # Forward pass through our custom model
        outputs = model(
            input_ids=labels,
            attention_mask=inputs["attention_mask"],
            images=inputs["images"],
            images_mask=inputs.get("images_mask", None)
        )
        
        # Get loss from outputs
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step for vision-language model
        """
        model.eval()
        # Get labels
        with torch.no_grad():
            labels = inputs["input_ids"]
            
            # Always compute loss for validation
            outputs = model(
                input_ids=labels,
                attention_mask=inputs["attention_mask"],
                images=inputs["images"],
                images_mask=inputs.get("images_mask", None)
            )
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            
            # For evaluation with generation
            if getattr(self.args, "predict_with_generate", False) and not prediction_loss_only:
                # Generation mode for metrics computation
                # Pass all inputs to model's generate method which handles multi-image
                tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
                # Fallbacks from model config if tokenizer not set
                pad_id = tokenizer.pad_token_id if tokenizer is not None else getattr(getattr(model, "model", model).config, "pad_token_id", 0)
                bos_id = tokenizer.cls_token_id if tokenizer is not None else getattr(getattr(model, "model", model).config, "decoder_start_token_id", None)
                eos_id = tokenizer.sep_token_id if tokenizer is not None else getattr(getattr(model, "model", model).config, "eos_token_id", None)
                generated_tokens = model.generate(
                    images=inputs["images"],
                    images_mask=inputs.get("images_mask", None),
                    max_new_tokens=getattr(self, 'gen_max_length', 128),
                    num_beams=getattr(self, 'beam_width', 2),
                    early_stopping=True,
                    pad_token_id=pad_id,
                    bos_token_id=bos_id,
                    eos_token_id=eos_id,
                    decoder_start_token_id=bos_id,
                )
                return (loss, generated_tokens, labels)
            else:
                # Standard evaluation - just return loss
                return (loss, None, labels)
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Override evaluation loop to set predict_with_generate=True for metrics
        """
        # Ensure predict_with_generate attribute exists and set it True during eval
        original_predict_with_generate = getattr(self.args, "predict_with_generate", False)
        setattr(self.args, "predict_with_generate", True)
        
        # Set tokenizer for generation if not already set
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self.tokenizer = dataloader.dataset.tokenizer
        
        # Run the parent evaluation loop
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )
        
        # Restore original setting
        setattr(self.args, "predict_with_generate", original_predict_with_generate)
        
        return output