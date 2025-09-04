#!/usr/bin/env bash
set -euo pipefail

############################################
# User-configurable
############################################
DATASETS=(
  "IAMJB/report-generation-rexgradient-noimage"
)
MODES=("impression")

VISION_BACKBONE="microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft"
NUM_GPUS=4
NUM_LAYERS=6

BATCH_SIZE=8
GRAD_ACCU=4
STEPS_PER_EPOCH=5292
MAX_EPOCHS=40
TOTAL_STEPS=$((STEPS_PER_EPOCH * MAX_EPOCHS))
WARMUP_STEPS=500
LEARNING_RATE=2e-05
MIN_LR=1e-07
WEIGHT_DECAY=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.999
ADAM_EPSILON=1e-06
GRAD_CLIP=1.0

export NCCL_TIMEOUT=7200
IMAGE_ROOT="/fss/jb/vilmedic_datasets/data/images/rex_mimic_chex/"
ACCELERATE_CFG="accelerate_config_multigpu.yaml"
ENV_NAME="vilmedic"
# Where your repo lives on the remote nodes (adjust if needed)
WORKDIR="${WORKDIR:-$HOME/vilmedic}"

# --- add near the top ---
SCREEN_NAME_MAX=80

short_hash() {
  # 6-char stable hash from args
  printf "%s" "$*" | md5sum | awk '{print substr($1,1,6)}'
}

make_session_name() {
  local dataset="$1"  # full path-like name
  local mode="$2"
  local layers="$3"
  local ts
  ts="$(date +%m%d_%H%M)"                # short timestamp

  local base
  base="$(basename "$dataset")"          # e.g., report-generation-rexgradient-noimage
  # keep it compact: first 24 chars
  base="${base:0:24}"

  local mshort
  case "$mode" in
    impression) mshort="imp" ;;
    findings)   mshort="fin" ;;
    *)          mshort="${mode:0:3}" ;;
  esac

  local h
  h="$(short_hash "$dataset|$mode|$layers|$ts")"

  local name="exp_${base}_${mshort}_${layers}_${ts}_${h}"

  # enforce max length (screen limit is 80)
  if [ "${#name}" -gt "$SCREEN_NAME_MAX" ]; then
    name="${name:0:$SCREEN_NAME_MAX}"
  fi
  printf "%s" "$name"
}

############################################
# Discover a node with ≥4 free GPUs and return: "<HOST> <CSV_4_GPU_INDICES>"
############################################
pick_node_and_gpus() {
  pdsh -R ssh -w "$(sinfo -N -h -o '%N' | paste -sd,)" \
    'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
     | awk -F", *" '"'"'{if ($2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/) {u=$2+0; m=$3+0; if (u==0 && m<5) free=(free?free","$1:$1)}} END{print (free?free:"-")}'"'"'' \
  | awk '
      # Expect lines like: HOST: 1,2,3,4,5,6,7 or HOST: -
      match($0, /^([^:]+):[[:space:]]*([0-9, -]+)$/, a) {
        host = a[1]; list = a[2];
        gsub(/[[:space:]]/, "", list);
        if (list == "-" || list == "") next;
        n = split(list, arr, ",");
        if (n >= 4) {
          printf "%s %s,%s,%s,%s\n", host, arr[1], arr[2], arr[3], arr[4];
          exit;  # take the first eligible host
        }
      }
    '
}

############################################
# Launch one experiment remotely in screen (robust: run a script)
############################################
launch_remote() {
  local HOST="$1"
  local GPUS="$2"
  local DATASET="$3"
  local MODE="$4"

  local NAME
  NAME="$(make_session_name "$DATASET" "$MODE" "$NUM_LAYERS")"
  local SESSION="$NAME"

  echo "[INFO] Launching on ${HOST} with GPUs ${GPUS} -> ${SESSION}"

  # Build the remote script
  local REMOTE_SCRIPT
  REMOTE_SCRIPT="$(cat <<'RS'
#!/usr/bin/env bash
set -euo pipefail
set -x

LOGDIR=~/.screenlogs
mkdir -p "$LOGDIR"

echo "[START] $(date) on $(hostname)"
echo "[SESSION] __SESSION__"
echo "[CUDA_VISIBLE_DEVICES] __GPUS__"
export PYTHONUNBUFFERED=1

# Make conda available in non-interactive shells
infer_conda_sh() {
  local conda_exe; conda_exe="$(command -v conda || true)"
  if [ -n "$conda_exe" ]; then
    local conda_root; conda_root="$(dirname "$(dirname "$conda_exe")")"
    if [ -f "$conda_root/etc/profile.d/conda.sh" ]; then
      echo "$conda_root/etc/profile.d/conda.sh"; return 0
    fi
  fi
  return 1
}whi

if CONDA_SH="$(infer_conda_sh 2>/dev/null)"; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
else
  if CONDA_EXE="$(command -v conda)"; then
    eval "$("$CONDA_EXE" shell.bash hook)"
  fi
fi

conda activate __ENV_NAME__

export CUDA_VISIBLE_DEVICES="__GPUS__"
export NCCL_TIMEOUT="__NCCL_TIMEOUT__"
export TERM=xterm

# Go to repo (fail loudly if missing)
cd "__WORKDIR__" || { echo "[ERROR] WORKDIR not found: __WORKDIR__" | tee -a "$LOGDIR/__NAME__.log"; exit 2; }
if [ ! -f bin/train_accelerate.py ]; then
  echo "[ERROR] Missing bin/train_accelerate.py in $(pwd)" | tee -a "$LOGDIR/__NAME__.log"
  exit 2
fi
if [ ! -f "__ACCELERATE_CFG__" ]; then
  echo "[ERROR] Missing accelerate config: __ACCELERATE_CFG__" | tee -a "$LOGDIR/__NAME__.log"
  exit 2
fi

nvidia-smi || true

accelerate launch \
  --config_file "__ACCELERATE_CFG__" \
  --num_processes __NUM_GPUS__ \
  bin/train_accelerate.py config/RRG/baseline-mimic-HF.yml \
  dataset.seq.processing=ifcc_clean_report \
  dataset.seq.hf_dataset='["__DATASET__"]' \
  dataset.seq.hf_field=__MODE__ \
  dataset.seq.hf_filter='lambda e:e["__MODE__"]' \
  dataset.seq.tokenizer_max_len=128 \
  dataset.seq.file=null \
  dataset.image.hf_dataset='["__DATASET__"]' \
  dataset.image.hf_field=images_path \
  dataset.image.hf_filter='lambda e:e["__MODE__"]' \
  dataset.image.multi_image=2 \
  dataset.image.resize=420 \
  dataset.image.crop=384 \
  dataset.image.image_path="__IMAGE_ROOT__" \
  dataset.image.file=null \
  model.proto=RRG_HF \
  model.vision="__VISION_BACKBONE__" \
  model.decoder.proto_config_args.num_hidden_layers=__NUM_LAYERS__ \
  model.decoder.proto_config_args.hidden_size=1024 \
  model.decoder.proto_config_args.hidden_dropout_prob=0.1 \
  model.decoder.proto_config_args.attention_probs_dropout_prob=0.1 \
  trainor.batch_size=__BATCH_SIZE__ \
  trainor.grad_accu=__GRAD_ACCU__ \
  trainor.use_amp=true \
  trainor.optimizer=AdamW \
  trainor.clip_grad_norm=__GRAD_CLIP__ \
  trainor.optim_params.lr=__LEARNING_RATE__ \
  trainor.optim_params.weight_decay=__WEIGHT_DECAY__ \
  trainor.optim_params.eps=__ADAM_EPSILON__ \
  trainor.optim_params.betas='[__ADAM_BETA1__,__ADAM_BETA2__]' \
  trainor.lr_decay=LinearWarmupCosineSchedule \
  trainor.lr_decay_params.num_warmup_steps=__WARMUP_STEPS__ \
  trainor.lr_decay_params.num_training_steps=__TOTAL_STEPS__ \
  trainor.early_stop_metric=radevalbertscore \
  trainor.early_stop=15 \
  trainor.eval_start=2 \
  trainor.early_stop_start=5 \
  trainor.epochs=__MAX_EPOCHS__ \
  validator.batch_size=64 \
  validator.beam_width=2 \
  validator.metrics='[radevalbertscore]' \
  validator.splits='[val]' \
  ckpt_dir=ckpt \
  name="__NAME__" \
  |& tee -a "$LOGDIR/__NAME__.log"

echo "[DONE] $(date)"
RS
)"

  # Fill in placeholders without fighting shell quoting
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__SESSION__/${SESSION}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__GPUS__/${GPUS}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__ENV_NAME__/${ENV_NAME}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__NCCL_TIMEOUT__/${NCCL_TIMEOUT}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__WORKDIR__/${WORKDIR}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__ACCELERATE_CFG__/${ACCELERATE_CFG}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__NUM_GPUS__/${NUM_GPUS}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__DATASET__/${DATASET}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__MODE__/${MODE}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__IMAGE_ROOT__/${IMAGE_ROOT}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__VISION_BACKBONE__/${VISION_BACKBONE}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__NUM_LAYERS__/${NUM_LAYERS}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__BATCH_SIZE__/${BATCH_SIZE}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__GRAD_ACCU__/${GRAD_ACCU}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__GRAD_CLIP__/${GRAD_CLIP}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__LEARNING_RATE__/${LEARNING_RATE}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__WEIGHT_DECAY__/${WEIGHT_DECAY}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__ADAM_EPSILON__/${ADAM_EPSILON}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__ADAM_BETA1__/${ADAM_BETA1}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__ADAM_BETA2__/${ADAM_BETA2}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__WARMUP_STEPS__/${WARMUP_STEPS}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__TOTAL_STEPS__/${TOTAL_STEPS}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__MAX_EPOCHS__/${MAX_EPOCHS}}"
  REMOTE_SCRIPT="${REMOTE_SCRIPT//__NAME__/${NAME}}"

  # Stage script on remote and make executable
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${HOST}" "cat > /tmp/${SESSION}.sh <<'EOS'
${REMOTE_SCRIPT}
EOS
chmod +x /tmp/${SESSION}.sh
ls -l /tmp/${SESSION}.sh
"

  # Ensure log dir exists
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${HOST}" 'mkdir -p ~/.screenlogs'

  # Start screen with login shell; enable screen's own logging too
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${HOST}" \
    "TERM=xterm screen -dmL -Logfile \$HOME/.screenlogs/${SESSION}.screen.log -S '${SESSION}' bash -lc '/bin/bash /tmp/${SESSION}.sh'"

  sleep 10
  # Verify the session is there; fail fast if not
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${HOST}" "screen -ls | grep -F '.${SESSION}' >/dev/null"; then
    echo "[ERROR] Screen session '${SESSION}' was not created on ${HOST}."
    echo "Debug:"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${HOST}" "screen -ls || true; command -v screen || true; echo 'TERM='\"\$TERM\"; ls -l /tmp/${SESSION}.sh || true"
    exit 1
  fi

  echo "[OK] ${HOST} :: ${SESSION}"
  echo "Attach:  ssh ${HOST} 'screen -r ${SESSION}'"
  echo "Logs:    ssh ${HOST} 'tail -f ~/.screenlogs/${SESSION}.log'"
  echo "Screen:  ssh ${HOST} 'tail -f ~/.screenlogs/${SESSION}.screen.log'"
  echo "List:    ssh ${HOST} 'screen -ls'"
}

############################################
# Iterate datasets × modes:
# for each, find a node with ≥4 free GPUs and launch there
############################################
main() {
  for ds in "${DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do
      echo "[INFO] Seeking node with ≥${NUM_GPUS} free GPUs..."
      local sel
      while true; do
        sel="$(pick_node_and_gpus || true)"
        if [[ -n "${sel}" ]]; then
          break
        fi
        echo "[INFO] None available yet; retry in 60s..."
        sleep 60
      done
      # sel = "HOST CSV"
      local host gpus
      host="$(awk '{print $1}' <<<"${sel}")"
      gpus="$(awk '{print $2}' <<<"${sel}")"
      launch_remote "${host}" "${gpus}" "${ds}" "${mode}"
    done
  done
}

main
