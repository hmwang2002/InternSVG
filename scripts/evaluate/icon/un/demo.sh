#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

python evaluate_un.py \
    --gt_path "PATH_TO_GT" \
    --pred_path "PATH_TO_PRED" \
    --output_path "PATH_TO_OUTPUT"