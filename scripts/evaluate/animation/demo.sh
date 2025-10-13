#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

python evaluate_animation.py \
    --model_name "MODEL_NAME" \
    --test_file_path "PATH_TO_TEST_FILE" \
    --gt_video_dir "PATH_TO_GT_VIDEO_DIR" \
    --gt_svg_dir "PATH_TO_GT_SVG_DIR" \
    --overall_video_dir "PATH_TO_OVERALL_VIDEO_DIR" \
    --text2sani_test_dir "PATH_TO_TEXT2SANI_TEST_DIR" \
    --video2sani_test_dir "PATH_TO_VIDEO2SANI_TEST_DIR" \
    --tokenizer_path "PATH_TO_TOKENIZER"

