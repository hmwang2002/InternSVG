import re
import os
import json
import argparse
from typing import Union

from metrics.inference.video_api import build_client, chat
from utils.parallel_mapper import parallel_map
from dataclasses import dataclass


@dataclass
class TestData:
    id: int
    question: str
    video_path: Union[str, None]
    base_url: str
    api_key: str
    model_name: str
    timeout: int
    output_path: str
    max_tokens: int
    retry: int
    temperature: float
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_url', type=str, default='BASE_URL')
    parser.add_argument('--api_key', type=str, default='API_KEY')
    parser.add_argument('--model_name', type=str, default='MODEL_NAME')
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--text2sani_test_path', type=str, default='PATH_TO_TEXT2SANI_TEST_PATH')
    parser.add_argument('--video2sani_test_path', type=str, default='PATH_TO_VIDEO2SANI_TEST_PATH')
    parser.add_argument('--output_dir', type=str, default='PATH_TO_OUTPUT_DIR')
    parser.add_argument('--retry', type=int, default=2)
    parser.add_argument('--max_tokens', type=int, default=4000)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_workers', type=int, default=32)
    return parser.parse_args()


def generate_svg(data: TestData):
    client = build_client(base_url=data.base_url, api_key=data.api_key, timeout=data.timeout)
    
    question = data.question
    response = chat(client=client, model=data.model_name, text=question, video_path=data.video_path, temperature=data.temperature, max_tokens=data.max_tokens, retry=data.retry)
    if response is None:
        return
    output_json_path = os.path.join(data.output_path, 'output.jsonl')
    
    with open(output_json_path, 'a') as f:
        f.write(json.dumps({'id': data.id, 'answer': response}) + '\n')
    
    output_svg_path = os.path.join(data.output_path, 'svg', f'{data.id}.svg')
    os.makedirs(os.path.dirname(output_svg_path), exist_ok=True)
    """
    get the content start with <svg> and end with </svg> in the response
    """
    svg_code = re.search(r'<svg[^>]*>.*?</svg>', response, re.DOTALL)
    if svg_code:
        svg_code = svg_code.group(0)
        with open(output_svg_path, 'w') as f:
            f.write(svg_code)
    

if __name__ == "__main__":
    args = parse_args()
    base_url = args.base_url
    api_key = args.api_key
    model_name = args.model_name
    text2sani_test_path = args.text2sani_test_path
    video2sani_test_path = args.video2sani_test_path
    output_dir = args.output_dir
    max_tokens = args.max_tokens
    retry = args.retry
    timeout = args.timeout
    temperature = args.temperature
    max_workers = args.max_workers
    text2sani_output_path = os.path.join(output_dir, 'text2sani')
    video2sani_output_path = os.path.join(output_dir, 'video2sani')
    os.makedirs(text2sani_output_path, exist_ok=True)
    os.makedirs(video2sani_output_path, exist_ok=True)
    
    text2sani_data = []
    video2sani_data = []
    with open(text2sani_test_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            id = int(data['id'])
            svg_path = os.path.join(text2sani_output_path, 'svg', f'{id}.svg')
            if not os.path.exists(svg_path):
                question = data['conversations'][0]['value']
                text2sani_data.append(TestData(id=id, question=question, video_path=None, base_url=base_url, api_key=api_key, model_name=model_name, output_path=text2sani_output_path, max_tokens=max_tokens, retry=retry, timeout=timeout, temperature=temperature))
    
    print("Start processing text2svg data...")
    parallel_map(generate_svg, text2sani_data, max_workers=max_workers)
    
    
    with open(video2sani_test_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            id = int(data['id'])
            question = data['conversations'][0]['value']
            svg_path = os.path.join(video2sani_output_path, 'svg', f'{id}.svg')
            if not os.path.exists(svg_path):
                video_path = os.path.join(os.path.dirname(video2sani_test_path), data['video'])
                video2sani_data.append(TestData(id=id, question=question, video_path=video_path, base_url=base_url, api_key=api_key, model_name=model_name, output_path=video2sani_output_path, max_tokens=max_tokens, retry=retry, timeout=timeout, temperature=temperature))
    
    print("Start processing video2svg data...")
    parallel_map(generate_svg, video2sani_data, max_workers=max_workers)

    