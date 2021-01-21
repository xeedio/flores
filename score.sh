#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage(){
  cat >&2 <<EOF
Description: Score a lanauage pair

Usage: $(basename "${0}") [options] [--]

Options:
  -f
    Set FP16 for running the model
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

fairseq-generate \
  "$DATA_DIR" \
  --source-lang "$SOURCE_LANG" \
  --target-lang en \
  --path "$MODEL_DIR/checkpoint_best.pt" \
  --beam 5 \
  --lenpen 1.2 \
  --gen-subset valid \
  --remove-bpe=sentencepiece \
  --scoring sacrebleu \
  "${EXTRA_ARGS[@]}"
