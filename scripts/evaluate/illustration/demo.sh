#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

python evaluate_gen.py \
    --model_name "MODEL_NAME" \
    --text2svg_test_dir "PATH_TO_TEXT2SVG_TEST_DIR" \
    --img2svg_test_dir "PATH_TO_IMG2SVG_TEST_DIR" \
    --tokenizer_path "PATH_TO_TOKENIZER" \
    --test_file_path "PATH_TO_TEST_FILE" \
    --gt_img_dir "PATH_TO_GT_IMG_DIR" \
    --gt_svg_dir "PATH_TO_GT_SVG_DIR" \
    --caption_path "PATH_TO_CAPTION_FILE" \
    --bench_name "Illustration" \