import json
import argparse

from collections import defaultdict
from typing import Dict, Tuple, List, Optional


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()

def extract_choice(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return text[0]

def load_gt(path: str) -> Dict[int, Tuple[str, str]]:
    """
    return dict: cid -> (gt_choice, subject)
    """
    gt = {}
    with open(path, "r") as fp:
        for ln in fp:
            obj = json.loads(ln)
            cid = int(obj["id"])
            ans_raw = obj["conversations"][1]["value"]
            choice = extract_choice(ans_raw)
            if choice is None:
                raise ValueError(f"GT id={cid} extract failed, raw: {ans_raw}")
            subject = obj.get("Subject", "unknown")
            gt[cid] = (choice, subject)
    return gt

def load_pred(path: str) -> Dict[int, str]:
    pred = {}
    with open(path, "r") as fp:
        for ln in fp:
            obj = json.loads(ln)
            cid = int(obj["id"])
            choice = extract_choice(obj.get("result"))
            if choice:
                pred[cid] = choice
    return pred

def evaluate(
    gt: Dict[int, Tuple[str, str]],
    pred: Dict[int, str]
) -> Tuple[float, Dict[str, float], List[int]]:
    """
    Return:
      - overall_acc
      - per_subject_acc  dict[str, float]
      - wrong_ids        list[int]
    """
    total = len(gt)
    correct = 0
    wrong_ids: List[int] = []

    per_tot   = defaultdict(int)  # subject -> total
    per_corr  = defaultdict(int)  # subject -> correct

    for cid, (gt_choice, subj) in gt.items():
        per_tot[subj] += 1
        if pred.get(cid) == gt_choice:
            correct      += 1
            per_corr[subj] += 1
        else:
            wrong_ids.append(cid)

    overall_acc = correct / total * 100
    per_subject_acc = {
        s: per_corr[s] / per_tot[s] * 100
        for s in per_tot
    }
    return overall_acc, per_subject_acc, wrong_ids

def main():
    args = parse_args()
    gt_path = args.gt_path
    pred_path = args.pred_path
    output_path = args.output_path

    print(f"[INFO] loading GT from {gt_path}")
    gt = load_gt(gt_path)
    
    with open(output_path, "w") as result_file:
        try:
            pred = load_pred(pred_path)
            overall, per_subject, wrong_ids = evaluate(gt, pred)
            print(f"{pred_path} : {overall:.1f}%  ({len(gt)-len(wrong_ids)}/{len(gt)})")
            for s, acc in sorted(per_subject.items()):
                print(f"  - {s:<12}: {acc:.1f}%  ({acc/100}")

            result_file.write(f"{pred_path} : {overall:.1f}\n")
            for s, acc in sorted(per_subject.items()):
                result_file.write(f"  {s} : {acc:.1f}\n")
        except Exception as e:
            error_msg = f"Error processing {pred_path}: {str(e)}"
            print(error_msg)
            result_file.write(f"{pred_path} : ERROR - {str(e)}\n")


if __name__ == "__main__":
    main()