squeue -u $USER
# scancel -u $USER
dataset1=IAMJB/report-generation-rexgradient-noimage
dataset2=IAMJB/report-generation-chexpert-png-noimage
dataset3=IAMJB/report-generation-mimic-cxr-noimage


for dataset in IAMJB/report-generation-rexgradient-noimage IAMJB/report-generation-chexpert-png-noimage IAMJB/report-generation-mimic-cxr-noimage
do
  for mode in impression findings
  do
    sanitize_folder_name() {
      echo "$1" | sed 's/[^a-zA-Z0-9_-]/_/g'
    }
    FOLDER_NAME=$(sanitize_folder_name "$dataset")
    NUM_LAYERS=6
    BATCH_SIZE=64
    GRAD_ACCU=1
    MAX_EPOCHS=20
    LEARNING_RATE=2e-05
    MIN_LR=1e-07
    WEIGHT_DECAY=0.05
    ADAM_BETA1=0.9
    ADAM_BETA2=0.999
    ADAM_EPSILON=1e-06
    GRAD_CLIP=1.0
    NUM_GPUS=4
    MAX_GEN_LENGTH=175
    NAME=dino_modern_${FOLDER_NAME}_${mode}_${NUM_LAYERS}_HF_TRAINER
        # Generate a unique port for this job to avoid conflicts
    # Use job name hash to generate consistent but unique port  
    PORT_OFFSET=$(echo "$NAME" | cksum | cut -d' ' -f1)
    PORT=$((29500 + (PORT_OFFSET % 1000)))
    
    echo "Submitting job: $NAME with port=$PORT"

    sbatch --time=24:00:00 --gpus=$NUM_GPUS --cpus-per-task=16 --mem=128G --job-name="${NAME}" \
    --wrap "source /fss/jb/miniconda3/etc/profile.d/conda.sh
    conda init
    conda activate vilmedic
    cd /fss/jb/vilmedic

    # Launch
    accelerate launch \
      --config_file accelerate_config_multigpu.yaml \
      --num_processes $NUM_GPUS \
      --main_process_port $PORT \
      hf_trainer/train.py config/RRG/baseline-mimic-HF.yml \
      dataset.seq.processing=ifcc_clean_report \
      dataset.seq.hf_dataset=[$dataset] \
      dataset.seq.hf_field=$mode \
      dataset.seq.hf_filter='lambda e:e[\"${mode}\"]' \
      dataset.seq.tokenizer_max_len=$MAX_GEN_LENGTH \
      dataset.image.hf_dataset=[$dataset] \
      dataset.image.hf_field=images_path \
      dataset.image.hf_filter='lambda e:e[\"${mode}\"]' \
      dataset.image.multi_image=2 \
      dataset.image.resize=420 \
      dataset.image.crop=384 \
      dataset.image.image_path=/fss/jb/vilmedic_datasets/data/images/rex_mimic_chex/ \
      model.proto=VisionLanguageModernBert \
      model.vision_model_name=IAMJB/maira-2-dinov2 \
      model.tokenizer_name=answerdotai/ModernBERT-base \
      model.decoder_config_name=jhu-clsp/ettin-decoder-17m \
      model.decoder_config_args.num_hidden_layers=$NUM_LAYERS \
      model.decoder_config_args.hidden_size=1024 \
      model.decoder_config_args.hidden_dropout_prob=0.1 \
      model.decoder_config_args.attention_probs_dropout_prob=0.1 \
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
      validator.gen_max_length=$MAX_GEN_LENGTH \
      validator.metrics=[radevalbertscore] \
      validator.splits=[val] \
      ckpt_dir=ckpt \
      name=${NAME}
    "
  done
done






# single 
for dataset in IAMJB/report-generation-mimic-cxr-noimage
do
  for mode in findings
  do
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    sanitize_folder_name() {
      echo "$1" | sed 's/[^a-zA-Z0-9_-]/_/g'
    }
    FOLDER_NAME=$(sanitize_folder_name "$dataset")
    NUM_LAYERS=6
    BATCH_SIZE=64
    GRAD_ACCU=1
    MAX_EPOCHS=20
    LEARNING_RATE=2e-05
    MIN_LR=1e-07
    WEIGHT_DECAY=0.05
    ADAM_BETA1=0.9
    ADAM_BETA2=0.999
    ADAM_EPSILON=1e-06
    GRAD_CLIP=1.0
    NUM_GPUS=4
    MAX_GEN_LENGTH=175
    NAME=dino_modern_${FOLDER_NAME}_${mode}_${NUM_LAYERS}_HF_TRAINER
        # Generate a unique port for this job to avoid conflicts
    # Use job name hash to generate consistent but unique port  
    PORT_OFFSET=$(echo "$NAME" | cksum | cut -d' ' -f1)
    PORT=$((29500 + (PORT_OFFSET % 1000)))
accelerate launch \
      --config_file accelerate_config_multigpu.yaml \
      --num_processes $NUM_GPUS \
      --main_process_port $PORT \
      hf_trainer/train.py config/RRG/baseline-mimic-HF.yml \
      dataset.seq.processing=ifcc_clean_report \
      dataset.seq.hf_dataset=[$dataset] \
      dataset.seq.hf_field=$mode \
      dataset.seq.hf_filter='lambda e:e["'"${mode}"'"]' \
      dataset.seq.tokenizer_max_len=$MAX_GEN_LENGTH \
      dataset.image.hf_dataset=[$dataset] \
      dataset.image.hf_field=images_path \
      dataset.image.hf_filter='lambda e:e["'"${mode}"'"]' \
      dataset.image.multi_image=2 \
      dataset.image.resize=420 \
      dataset.image.crop=384 \
      dataset.image.image_path=/fss/jb/vilmedic_datasets/data/images/rex_mimic_chex/ \
      model.proto=VisionLanguageModernBert \
      model.vision_model_name=IAMJB/maira-2-dinov2 \
      model.tokenizer_name=answerdotai/ModernBERT-base \
      model.decoder_config_name=jhu-clsp/ettin-decoder-17m \
      model.decoder_config_args.num_hidden_layers=$NUM_LAYERS \
      model.decoder_config_args.hidden_size=1024 \
      model.decoder_config_args.hidden_dropout_prob=0.1 \
      model.decoder_config_args.attention_probs_dropout_prob=0.1 \
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
      validator.batch_size=16 \
      validator.beam_width=2 \
      validator.gen_max_length=$MAX_GEN_LENGTH \
      validator.metrics=[radevalbertscore] \
      validator.splits=[val,test] \
      ckpt_dir=ckpt \
      name=${NAME}  
    done
done

