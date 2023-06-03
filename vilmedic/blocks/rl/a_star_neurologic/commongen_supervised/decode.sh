#!/usr/bin/env bash

export PYTHONPATH=/home/ximinglu/a_star_neurologic

DATA_DIR='../dataset/'
SPLIT='dev'
MODEL_RECOVER_PATH='finetune_model/gpt2-large/checkpoint-1800/'

DEVICES=$1
OUTPUT_FILE=$2

# neurologic with greedy look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${DATA_DIR}/${SPLIT}.txt --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
  --key_constraint_file ${DATA_DIR}/constraint/${SPLIT}_key.constraint.json \
  --batch_size 16 --beam_size 20 --max_tgt_length 48 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 50 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.25 --look_ahead_width 1 #--fusion_t 1.0

# neurologic with sampling look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${DATA_DIR}/${SPLIT}.txt --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
  --key_constraint_file ${DATA_DIR}/constraint/${SPLIT}_key.constraint.json \
  --batch_size 16 --beam_size 20 --max_tgt_length 48 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 50 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.3  --look_ahead_sample --look_ahead_width 5

# neurologic with beam look-ahead
#CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${DATA_DIR}/${SPLIT}.txt --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
  --key_constraint_file ${DATA_DIR}/constraint/${SPLIT}_key.constraint.json \
  --batch_size 16 --beam_size 20 --max_tgt_length 48 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 50 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.45 --look_ahead_width 4