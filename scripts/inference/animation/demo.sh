#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

BASE_URL="BASE_URL"
API_KEY="API_KEY"
MODEL_NAME="MODEL_NAME"
TIMEOUT=500
TEXT2SANI_TEST_PATH="PATH_TO_TEXT2SANI_TEST_PATH"
VIDEO2SANI_TEST_PATH="PATH_TO_VIDEO2SANI_TEST_PATH"
OUTPUT_DIR="PATH_TO_OUTPUT_DIR"
RETRY=1
MAX_TOKENS=4000
TEMPERATURE=0.0
MAX_WORKERS=32

python metrics/inference/inference_animation.py --base_url ${BASE_URL} --api_key ${API_KEY} --model_name ${MODEL_NAME} --text2sani_test_path ${TEXT2SANI_TEST_PATH} --video2sani_test_path ${VIDEO2SANI_TEST_PATH} --output_dir ${OUTPUT_DIR} --retry ${RETRY} --max_tokens ${MAX_TOKENS} --timeout ${TIMEOUT} --temperature ${TEMPERATURE} --max_workers ${MAX_WORKERS}
