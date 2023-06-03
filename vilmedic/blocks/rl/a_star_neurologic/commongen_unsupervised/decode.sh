#!/usr/bin/env bash

export PYTHONPATH=/home/ximinglu/a_star_neurologic

DATA_DIR='../dataset'
SPLIT='dev'

DEVICES=$1
OUTPUT_FILE=$2

# neurologic with greedy look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python decode_gpt2.py --model_name 'gpt2-large' \
  --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
  --key_constraint_file ${DATA_DIR}/constraint/${SPLIT}_key.constraint.json \
  --batch_size 16 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.175 --look_ahead_width 1 #--fusion_t 1.0
