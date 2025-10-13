#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

BASE_URL="BASE_URL"
API_KEY="API_KEY"
MODEL_NAME="MODEL_NAME"
TIMEOUT=500
EDIT_TEST_DIR="PATH_TO_EDIT_TEST_DIR"
OUTPUT_DIR="PATH_TO_OUTPUT_DIR"
RETRY=1
TEMPERATURE=0.0
MAX_TOKENS=4000
MAX_WORKERS=32

python metrics/inference/inference_edit.py --base_url ${BASE_URL} --api_key ${API_KEY} --model_name ${MODEL_NAME} --edit_test_dir ${EDIT_TEST_DIR} --output_dir ${OUTPUT_DIR} --temperature ${TEMPERATURE} --max_tokens ${MAX_TOKENS} --max_workers ${MAX_WORKERS}
