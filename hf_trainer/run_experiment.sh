#!/bin/bash

# Datasets
dataset1="IAMJB/report-generation-rexgradient-noimage"
dataset2="IAMJB/report-generation-chexpert-png-noimage"
dataset3="IAMJB/report-generation-mimic-cxr-noimage"

# Configuration
mode="findings"
vision_backbone="microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft"
export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=4
NUM_LAYERS=6

# Training hyperparameters
BATCH_SIZE=8         # Per GPU batch size
GRAD_ACCU=4          # Gradient accumulation steps

# Calculate training steps

MAX_EPOCHS=20
LEARNING_RATE=2e-05   
MIN_LR=1e-07         

# Optimization settings
WEIGHT_DECAY=0.1     
ADAM_BETA1=0.9       
ADAM_BETA2=0.999     
ADAM_EPSILON=1e-06   
GRAD_CLIP=1.0        

# Run training with Accelerate (uses HF Trainer under the hood)
accelerate launch \
    --config_file accelerate_config_multigpu.yaml \
    --num_processes $NUM_GPUS \
    hf_trainer/train.py config/RRG/baseline-mimic-HF.yml \
    dataset.seq.processing=ifcc_clean_report \
    dataset.seq.hf_dataset=[$dataset1,$dataset2,$dataset3] \
    dataset.seq.hf_field=$mode \
    dataset.seq.hf_filter='lambda e:e["'"${mode}"'"]' \
    dataset.seq.tokenizer_max_len=175 \
    dataset.image.hf_dataset=[$dataset1,$dataset2,$dataset3] \
    dataset.image.hf_field=images_path \
    dataset.image.hf_filter='lambda e:e["'"${mode}"'"]' \
    dataset.image.multi_image=2 \
    dataset.image.resize=420 \
    dataset.image.crop=384 \
    dataset.image.image_path=/fss/jb/vilmedic_datasets/data/images/rex_mimic_chex/ \
    model.proto=RRG_HF \
    model.vision=$vision_backbone \
    model.decoder.proto_config_args.num_hidden_layers=$NUM_LAYERS \
    model.decoder.proto_config_args.hidden_size=1024 \
    model.decoder.proto_config_args.hidden_dropout_prob=0.1 \
    model.decoder.proto_config_args.attention_probs_dropout_prob=0.1 \
    trainor.batch_size=$BATCH_SIZE \
    trainor.grad_accu=$GRAD_ACCU \
    trainor.use_amp=true \
    trainor.optim_params.lr=$LEARNING_RATE \
    trainor.optimizer=AdamW \
    trainor.clip_grad_norm=$GRAD_CLIP \
    trainor.optim_params.weight_decay=$WEIGHT_DECAY \
    trainor.optim_params.eps=$ADAM_EPSILON \
    trainor.optim_params.betas=[$ADAM_BETA1,$ADAM_BETA2] \
    trainor.lr_decay=LinearWarmupCosineSchedule \
    trainor.early_stop_metric=radevalbertscore \
    trainor.early_stop=15 \
    trainor.eval_start=2 \
    trainor.early_stop_start=5 \
    trainor.epochs=$MAX_EPOCHS \
    validator.batch_size=64 \
    validator.beam_width=2 \
    validator.metrics=[radevalbertscore] \
    validator.splits=[val] \
    ckpt_dir=ckpt \
    name=mix3_${mode}_${NUM_LAYERS}_HF_TRAINER 

    # trainor.eval_only=true
