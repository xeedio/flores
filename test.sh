#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

export CUDA_VISIBLE_DEVICES=0
export PYTHONFAULTHANDLER=true

cat mono.test |\
  python -m cProfile -o output.pstats ~/anaconda3/bin/fairseq-interactive /home/williamss/gocode/src/github.com/facebookresearch/flores/data-bin/wiki_ne_en_bpe5000/ \
  --source-lang ne \
  --target-lang en \
  --path /home/williamss/gocode/src/github.com/facebookresearch/flores/scripts/../experiments/sup/ne-en/model/checkpoint_best.pt \
  --lenpen 1.5 \
  --max-len-a 1.8 \
  --max-len-b 10 \
  --buffer-size 10000 \
  --max-tokens 4000 \
  --skip-invalid-size-inputs-valid-test \
  --num-workers 8 > mono.test.log
