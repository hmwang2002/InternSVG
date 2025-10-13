import re
import os
import json
import argparse
from typing import Union

from utils.api import build_client, chat
from utils.raster_svg import InputData, raster_svg
from utils.parallel_mapper import parallel_map
from dataclasses import dataclass


@dataclass
class TestData:
    id: int
    question: str
    img_path: Union[str, None]
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
    parser.add_argument('--text2svg_test_path', type=str, default='PATH_TO_TEXT2SVG_TEST_PATH')
    parser.add_argument('--img2svg_test_path', type=str, default='PATH_TO_IMG2SVG_TEST_PATH')
    parser.add_argument('--output_dir', type=str, default='PATH_TO_OUTPUT_DIR')
    parser.add_argument('--retry', type=int, default=2)
    parser.add_argument('--max_tokens', type=int, default=4000)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_workers', type=int, default=32)
    return parser.parse_args()


def generate_svg(data: TestData):
    client = build_client(base_url=data.base_url, api_key=data.api_key, timeout=data.timeout)
    
    question = data.question
    response = chat(client=client, model=data.model_name, text=question, image_path=data.img_path, temperature=data.temperature, max_tokens=data.max_tokens, retry=data.retry)
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
    text2svg_test_path = args.text2svg_test_path
    img2svg_test_path = args.img2svg_test_path
    output_dir = args.output_dir
    max_tokens = args.max_tokens
    retry = args.retry
    timeout = args.timeout
    temperature = args.temperature
    max_workers = args.max_workers
    text2svg_output_path = os.path.join(output_dir, 'text2svg')
    img2svg_output_path = os.path.join(output_dir, 'img2svg')
    os.makedirs(text2svg_output_path, exist_ok=True)
    os.makedirs(img2svg_output_path, exist_ok=True)
    
    text2svg_data = []
    img2svg_data = []
    with open(text2svg_test_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            id = int(data['id'])
            svg_path = os.path.join(text2svg_output_path, 'svg', f'{id}.svg')
            if not os.path.exists(svg_path):
                question = data['conversations'][0]['value']
                text2svg_data.append(TestData(id=id, question=question, img_path=None, base_url=base_url, api_key=api_key, model_name=model_name, output_path=text2svg_output_path, max_tokens=max_tokens, retry=retry, timeout=timeout, temperature=temperature))
    
    print("Start processing text2svg data...")
    parallel_map(generate_svg, text2svg_data, max_workers=max_workers)
    
    
    with open(img2svg_test_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            id = int(data['id'])
            question = data['conversations'][0]['value']
            svg_path = os.path.join(img2svg_output_path, 'svg', f'{id}.svg')
            if not os.path.exists(svg_path):
                img2svg_data.append(TestData(id=id, question=question, img_path=data['image'], base_url=base_url, api_key=api_key, model_name=model_name, output_path=img2svg_output_path, max_tokens=max_tokens, retry=retry, timeout=timeout, temperature=temperature))
    
    print("Start processing img2svg data...")
    parallel_map(generate_svg, img2svg_data, max_workers=max_workers)
    
    text2svg_svg_dir = os.path.join(output_dir, 'text2svg', 'svg')
    img2svg_svg_dir = os.path.join(output_dir, 'img2svg', 'svg')
    
    print("Start rastering text2svg svg...")
    output_images_dir = os.path.join(output_dir, 'text2svg', 'images')
    os.makedirs(output_images_dir, exist_ok=True)
    input_data_list = [InputData(svg_path=os.path.join(text2svg_svg_dir, file_name), output_dir=output_images_dir, width=448, height=448) for file_name in os.listdir(text2svg_svg_dir)]
    parallel_map(raster_svg, input_data_list, max_workers=max_workers)
    
    print("Start rastering img2svg svg...")
    output_images_dir = os.path.join(output_dir, 'img2svg', 'images')
    os.makedirs(output_images_dir, exist_ok=True)
    input_data_list = [InputData(svg_path=os.path.join(img2svg_svg_dir, file_name), output_dir=output_images_dir, width=448, height=448) for file_name in os.listdir(img2svg_svg_dir)]
    parallel_map(raster_svg, input_data_list, max_workers=max_workers)
    
    