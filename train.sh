#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage(){
  cat >&2 <<EOF
Description: Train a lanauage pair

Usage: $(basename "${0}") [options] [--]

Options:
  -f
    Use FP16 training
  -s
    Source language code
  -g
    Set GPU indices to use ('0' or '0,1,2', etc)
  -d
    Set script debug mode
  -h
    Show usage and exit

EOF
}

CUDA_VISIBLE_DEVICES=0
TARGET_LANG=en
EXTRA_ARGS=()

while getopts ":dfg:s:h" opt; do
  case "$opt" in
    f)
      EXTRA_ARGS+=(--fp16)
      ;;
    g)
      CUDA_VISIBLE_DEVICES="${OPTARG}"
      ;;
    s)
      SOURCE_LANG="${OPTARG}"
      ;;
    d)
      DEBUG=1
      set -x
      ;;
    h)
      usage
      exit 0
      ;;
    *)
      echo -e "\nOption does not exist: $OPTARG\n" >&2
      usage
      exit 1
      ;;
  esac
done

shift "$((OPTIND-1))"

export CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

if [[ ${#CUDA_VISIBLE_DEVICES} -le 1 ]]; then
  EXTRA_ARGS+=(--update-freq 4)
fi

if [[ -z ${SOURCE_LANG-} ]]; then
  echo -e >&2 "SOURCE_LANG is required!\n"
  usage
  exit 1
fi

DATA_DIR="data-bin/wiki_${SOURCE_LANG}_en_bpe5000"

if [[ ! -d "$DATA_DIR" ]]; then
  echo >&2 "DATA_DIR $DATA_DIR is missing!"
  usage
  exit 1
fi

MODEL_DIR="models/$SOURCE_LANG"
mkdir -p "$MODEL_DIR"

echo >&2 "Started training at $(date)"

time fairseq-train \
  "$DATA_DIR" \
  --source-lang "$SOURCE_LANG" \
  --target-lang en \
  --arch transformer \
  --share-all-embeddings \
  --encoder-layers 5 \
  --decoder-layers 5 \
  --encoder-embed-dim 512 \
  --decoder-embed-dim 512 \
  --encoder-ffn-embed-dim 2048 \
  --decoder-ffn-embed-dim 2048 \
  --encoder-attention-heads 2 \
  --decoder-attention-heads 2 \
  --encoder-normalize-before \
  --decoder-normalize-before \
  --dropout 0.4 \
  --attention-dropout 0.2 \
  --relu-dropout 0.2 \
  --weight-decay 0.0001 \
  --label-smoothing 0.2 \
  --criterion label_smoothed_cross_entropy \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --clip-norm 0 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 \
  --warmup-init-lr 1e-7 \
  --lr 1e-3 \
  --min-lr 1e-9 \
  --max-tokens 4000 \
  --max-epoch 100 \
  --save-dir "$MODEL_DIR" \
  --save-interval 10 \
  "${EXTRA_ARGS[@]}"

echo >&2 "Done training at $(date)"
