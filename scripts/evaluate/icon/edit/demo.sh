#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

python evaluate_edit.py \
    --model_name "MODEL_NAME" \
    --gt_dir "PATH_TO_GT_DIR" \
    --test_dir "PATH_TO_TEST_DIR" \
    --tokenizer_path "PATH_TO_TOKENIZER"