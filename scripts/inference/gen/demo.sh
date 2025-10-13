#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

BASE_URL="BASE_URL"
API_KEY="API_KEY"
MODEL_NAME="MODEL_NAME"
TEXT2SVG_TEST_PATH="PATH_TO_TEXT2SVG_TEST_PATH"
IMG2SVG_TEST_PATH="PATH_TO_IMG2SVG_TEST_PATH"
OUTPUT_DIR="PATH_TO_OUTPUT_DIR"
RETRY=1
TEMPERATURE=0.0
MAX_TOKENS=4000
MAX_WORKERS=32

python metrics/inference/inference.py \
--base_url ${BASE_URL} \
--api_key ${API_KEY} \
--model_name ${MODEL_NAME} \
--text2svg_test_path ${TEXT2SVG_TEST_PATH} \
--img2svg_test_path ${IMG2SVG_TEST_PATH} \
--output_dir ${OUTPUT_DIR} \
--temperature ${TEMPERATURE} \
--max_tokens ${MAX_TOKENS} \
--max_workers ${MAX_WORKERS}
