import os
import json
import argparse

from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from utils.api import build_client, chat
from utils.parallel_mapper import parallel_map


@dataclass
class TestData:
    id: int
    question: str
    base_url: str
    api_key: str
    model_name: str
    timeout: int
    output_path: Path
    max_tokens: int
    retry: int
    temperature: float

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_url', type=str, default='BASE_URL')
    parser.add_argument('--api_key', type=str, default='API_KEY')
    parser.add_argument('--model_name', type=str, default='MODEL_NAME')
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--un_test_path', type=str, default='PATH_TO_UN_TEST_PATH')
    parser.add_argument('--output_dir', type=str, default='PATH_TO_OUTPUT_DIR')
    parser.add_argument('--retry', type=int, default=2)
    parser.add_argument('--max_tokens', type=int, default=4000)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_workers', type=int, default=32)
    return parser.parse_args()

def _load_existing(out_file: Path) -> Dict[int, str]:
    """If the result file exists, load the {id: result} mapping, tolerate bad lines."""
    existing: Dict[int, str] = {}
    if not out_file.exists():
        return existing
    with out_file.open("r", encoding="utf-8") as fp:
        for ln in fp:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                existing[int(obj["id"])] = obj["result"]
            except Exception:
                # Half-written bad lines or format problems – skip directly
                continue
    return existing

def load_dataset(path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open( encoding="utf‑8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples

def postprocess_response(raw_rsp: str) -> str:
    """Normalize model output to final answer field (strip whitespace)."""
    return raw_rsp.strip()

def generate_answer(data: TestData):
    client = build_client(base_url=data.base_url, api_key=data.api_key, timeout=data.timeout)
    response = chat(client=client, model=data.model_name, text=data.question, temperature=data.temperature, max_tokens=data.max_tokens, retry=data.retry)
    
    if response is None:
        return
    with data.output_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps({"id": data.id, "result": postprocess_response(response)}) + "\n")

def evaluate_model(
    base_url: str,
    api_key: str,
    model_name: str,
    timeout: int,
    dataset: List[Dict[str, Any]],
    output_path: Path,
    retry: int,
    max_tokens: int,
    temperature: float,
    max_workers: int
):
    existing_results = _load_existing(output_path)
    done_ids = set(existing_results.keys())
    remaining_dataset = [
        sample
        for sample in dataset
        if sample["id"] not in done_ids
    ]
    
    if not remaining_dataset:
        print(f"[SKIP] {model_name}: all done, sort the file directly.")
    else:
        print(f"[INFO] {model_name}: {len(done_ids)} items done, {len(remaining_dataset)} items remaining.")
        
        data_list = [
            TestData(
                id=sample["id"],
                question=sample["conversations"][0]["value"],
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                timeout=timeout,
                output_path=output_path,
                retry=retry,
                max_tokens=max_tokens,
                temperature=temperature
            )
            for sample in remaining_dataset
        ]
        parallel_map(generate_answer, data_list, max_workers=max_workers)

    existing_results = {}
    with output_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            existing_results[obj["id"]] = obj["result"]
    sorted_items = sorted(existing_results.items(), key=lambda kv: kv[0])
    with output_path.open("w", encoding="utf-8") as fp:
        for _id, _res in sorted_items:
            fp.write(json.dumps({"id": _id, "result": _res}, ensure_ascii=False) + "\n")

    print(f"[OK] {model_name}: total {len(sorted_items)} results sorted and written to {output_path}")

def main():
    args = parse_args()
    base_url = args.base_url
    api_key = args.api_key
    model_name = args.model_name
    timeout = args.timeout
    un_test_path = args.un_test_path
    output_dir = args.output_dir
    retry = args.retry
    max_tokens = args.max_tokens
    temperature = args.temperature
    max_workers = args.max_workers
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{model_name}_un_result.jsonl')

    dataset = load_dataset(Path(un_test_path))

    evaluate_model(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        timeout=timeout,
        dataset=dataset,
        output_path=Path(output_path),
        retry=retry,
        max_tokens=max_tokens,
        temperature=temperature,
        max_workers=max_workers
    )

if __name__ == '__main__':
    main()