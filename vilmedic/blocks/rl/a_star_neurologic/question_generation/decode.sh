#!/usr/bin/env bash

export PYTHONPATH=/home/ximinglu/a_star_neurologic

DEVICES=$1
OUTPUT_FILE=$2
DATA_DIR='../dataset/question_generation'

# neurologic
CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name 'gpt2-large' \
  --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/constraints.jsonl \
  --batch_size 16 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0 --look_ahead_width 1

# neurologic with greedy look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name 'gpt2-large' \
  --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/constraints.jsonl \
  --batch_size 16 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.175 --look_ahead_width 1 #--fusion_t 1.0

# neurologic with sample look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name 'gpt2-large' \
  --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/constraints.jsonl \
  --batch_size 16 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.175 --look_ahead_sample --look_ahead_width 2

# neurologic with beam look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name 'gpt2-large' \
  --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/constraints.jsonl \
  --batch_size 16 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.175 --look_ahead_width 2
