from pathlib import Path
import json
from typing import List, Tuple, Dict
import shutil

import torch
from transformers import AutoTokenizer, AddedToken


SRC = "PATH_TO_SRC_MODEL"
DST = "OUTPUT_DIR"
REPORT_JSON = f"{DST}/added_tokens_report.json"
REPORT_TXT = f"{DST}/added_tokens_report.txt"

TRY_RESIZE_MODEL = True # If you want to resize the model, set to True

TAG_TOKENS = [
    "<svg", "</svg>",
    "</g>",
    "<defs", "</defs>",
    "<use", "</use>",
     "</path>",
    "<rect", "</rect>",
    "<circle", "</circle>",
    "<ellipse", "</ellipse>",
    "<line", "</line>",
    "<polyline", "</polyline>",
    "<polygon", "</polygon>",
    "<text", "</text>",
    "<tspan", "</tspan>",
    "<textPath", "</textPath>",
    "<linearGradient", "</linearGradient>",
    "<radialGradient", "</radialGradient>",
    "<stop", "</stop>",
    "<clipPath", "</clipPath>",
    "<mask", "</mask>",
    "<filter", "</filter>",
    "<feGaussianBlur", "</feGaussianBlur>",
    "<feColorMatrix", "</feColorMatrix>",
    "<feComposite", "</feComposite>",
    "<feBlend", "</feBlend>",
    "/>","<animate","</animate>",
"<animateMotion", "</animateMotion>",
"<animateTransform", "</animateTransform>"
]  

ATTR_TOKENS = [
    ' width="', ' height="', ' viewBox="',
    ' id="', ' class="',
    ' x="', ' y="', ' x1="', ' y1="', ' x2="', ' y2="',
    ' cx="', ' cy="', ' r="', ' rx="', ' ry="',
    ' d="', ' points="',
    ' fill="', ' stroke="', ' stroke-width="',
    ' stroke-linecap="', ' stroke-linejoin="', ' stroke-miterlimit="',
    ' fill-rule="', ' clip-path="',
    ' opacity="', ' transform="',
    ' font-size="', ' font-family="', ' text-anchor="',
    ' gradientUnits="', ' gradientTransform="',
    ' offset="', ' stop-color="',' dur="',
    ' from="',' to="',' repeatCount="',' begin="',' rotate="',' path="'
]

def gen_numeric_tokens() -> Tuple[List[str], List[str], List[str]]:
    ints = [str(i) for i in range(-128, 129)]
    frac2 = [f".{i:02d}" for i in range(100)]
    frac1 = [f".{i}" for i in range(10)]
    return ints, frac2, frac1

def add_tokens_unique(tok, strings: List[str]) -> Tuple[int, List[str]]:
    vocab = tok.get_vocab()
    new_tokens: List[AddedToken] = []
    actually_added: List[str] = []
    for s in strings:
        if s in vocab:
            continue
        new_tokens.append(AddedToken(s, lstrip=False, rstrip=False, normalized=False))
        actually_added.append(s)
    n_new = tok.add_tokens(new_tokens, special_tokens=False) if new_tokens else 0
    if n_new < len(actually_added):
        actually_added = actually_added[:n_new]
    return n_new, actually_added

def _get_out_layer_and_weight(model):
    out_layer, out_w = None, None
    try:
        if hasattr(model, "get_output_embeddings") and callable(model.get_output_embeddings):
            ol = model.get_output_embeddings()
            if ol is not None and hasattr(ol, "weight"):
                out_layer, out_w = ol, ol.weight
    except Exception:
        pass

    if out_layer is None and hasattr(model, "language_model"):
        lm = model.language_model
        try:
            if hasattr(lm, "get_output_embeddings") and callable(lm.get_output_embeddings):
                ol = lm.get_output_embeddings()
                if ol is not None and hasattr(ol, "weight"):
                    out_layer, out_w = ol, ol.weight
        except Exception:
            pass
    return out_layer, out_w

def _is_tied(in_w: torch.Tensor, out_w: torch.Tensor) -> bool:
    try:
        return (in_w is out_w) or (in_w.data_ptr() == out_w.data_ptr())
    except Exception:
        return False

def semantic_initialize_new_tokens(
    model,
    tok_new,
    tok_ref,
    actually_added_lists: Dict[str, List[str]],
) -> Dict:
    model.eval()

    in_embed = model.get_input_embeddings()
    old_in_w = in_embed.weight.detach().clone()        # [old_vocab, hidden] (CPU)
    old_vocab = old_in_w.size(0)

    out_layer, out_w_cur = _get_out_layer_and_weight(model)
    old_out_w = None
    if out_w_cur is not None:
        try:
            old_out_w = out_w_cur.detach().clone()
        except Exception:
            old_out_w = None

    model.resize_token_embeddings(len(tok_new))

    try:
        if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
            model.config.vocab_size = len(tok_new)
        if hasattr(model, "config") and hasattr(model.config, "llm_config") and hasattr(model.config.llm_config, "vocab_size"):
            model.config.llm_config.vocab_size = len(tok_new)
        if hasattr(model, "language_model") and hasattr(model.language_model, "config") and hasattr(model.language_model.config, "vocab_size"):
            model.language_model.config.vocab_size = len(tok_new)
    except Exception:
        pass

    in_w = model.get_input_embeddings().weight
    out_layer, out_w = _get_out_layer_and_weight(model)

    tied = False
    if out_w is not None:
        tied = _is_tied(in_w, out_w)

    global_in_mean = old_in_w.mean(dim=0)                                # CPU
    global_out_mean = old_out_w.mean(dim=0) if old_out_w is not None else None  # CPU

    new_tokens_all: List[str] = []
    for k in ["integers", "frac2", "frac1", "attrs", "tags"]:
        new_tokens_all.extend(actually_added_lists.get(k, []))

    unk_id = getattr(tok_ref, "unk_token_id", None)

    stats = {
        "total_new_tokens": len(new_tokens_all),
        "semantic_inited": 0,
        "fallback_global": 0,
        "missing_in_output_head": 0,
        "examples": [],  # 存 3~5 个示例
        "tied_embeddings": bool(tied),
    }

    def _maybe_log_example(token, piece_ids, used_fallback, vec_sum):
        if len(stats["examples"]) < 5:
            stats["examples"].append({
                "token": token,
                "pieces": piece_ids,
                "used_fallback": bool(used_fallback),
                "in_vec_sum": float(vec_sum),
            })

    for token in new_tokens_all:
        new_id = tok_new.convert_tokens_to_ids(token)
        if not isinstance(new_id, int) or new_id < 0:
            with torch.no_grad():
                in_w[new_id].copy_(global_in_mean.to(in_w.device, dtype=in_w.dtype))
                if (out_w is not None) and (not tied):
                    try:
                        base = global_out_mean if global_out_mean is not None else global_in_mean
                        out_w[new_id].copy_(base.to(out_w.device, dtype=out_w.dtype))
                    except Exception:
                        stats["missing_in_output_head"] += 1
            stats["fallback_global"] += 1
            _maybe_log_example(token, [], True, float(in_w[new_id].sum().item()))
            continue

        piece_ids = tok_ref.encode(token, add_special_tokens=False)

        valid_ids = []
        for i in piece_ids:
            if not isinstance(i, int):
                continue
            if i < 0 or i >= old_vocab:
                continue
            if (unk_id is not None) and (i == unk_id):
                continue
            valid_ids.append(i)

        if len(valid_ids) == 0:
            with torch.no_grad():
                in_w[new_id].copy_(global_in_mean.to(in_w.device, dtype=in_w.dtype))
                if (out_w is not None) and (not tied):
                    try:
                        base = global_out_mean if global_out_mean is not None else global_in_mean
                        out_w[new_id].copy_(base.to(out_w.device, dtype=out_w.dtype))
                    except Exception:
                        stats["missing_in_output_head"] += 1
            stats["fallback_global"] += 1
            _maybe_log_example(token, piece_ids, True, float(in_w[new_id].sum().item()))
        else:
            vec_in = old_in_w[valid_ids].mean(dim=0)  # CPU
            with torch.no_grad():
                in_w[new_id].copy_(vec_in.to(in_w.device, dtype=in_w.dtype))

                if (out_w is not None) and (not tied):
                    try:
                        if old_out_w is not None:
                            vec_out = old_out_w[valid_ids].mean(dim=0)  # CPU
                            out_w[new_id].copy_(vec_out.to(out_w.device, dtype=out_w.dtype))
                        else:
                            out_w[new_id].copy_(vec_in.to(out_w.device, dtype=out_w.dtype))
                    except Exception:
                        stats["missing_in_output_head"] += 1

            stats["semantic_inited"] += 1
            _maybe_log_example(token, valid_ids, False, float(vec_in.sum().item()))

    try:
        if getattr(model.config, "tie_word_embeddings", None):
            model.tie_weights()
    except Exception:
        pass

    return stats


def write_report(dst_dir: str, summary: Dict):
    json_path = Path(dst_dir) / REPORT_JSON
    txt_path = Path(dst_dir) / REPORT_TXT

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("========== SVG Token Extension Report ==========\n")
    lines.append(f"Source: {summary['paths']['src']}")
    lines.append(f"Saved : {summary['paths']['dst']}")
    lines.append(f"Original vocab size : {summary['sizes']['orig_vocab']}")
    lines.append(f"New vocab size      : {summary['sizes']['new_vocab']}")
    lines.append("")
    lines.append(f"Added integers (-128..128): {summary['added_counts']['integers']}")
    lines.append(f"Added frac .00.. .99     : {summary['added_counts']['frac2']}")
    lines.append(f"Added frac .0 .. .9      : {summary['added_counts']['frac1']}")
    lines.append(f"Added ATTR tokens        : {summary['added_counts']['attrs']}")
    lines.append(f"Added TAG tokens         : {summary['added_counts']['tags']}")
    lines.append("")
    def block(title, items: List[str]):
        lines.append(f"[{title}] ({len(items)})")
        for s in items:
            lines.append(s)
        lines.append("")
    block("ATTR_TOKENS (actually added)", summary["actually_added"]["attrs"])
    block("TAG_TOKENS  (actually added)", summary["actually_added"]["tags"])
    block("Integers (actually added)", summary["actually_added"]["integers"])
    block("Frac2 .00.. .99 (actually added)", summary["actually_added"]["frac2"])
    block("Frac1 .0.. .9 (actually added)", summary["actually_added"]["frac1"])

    lines.append("---------- Model Resize & Semantic Init ----------")
    for k, v in summary["model_resize"].items():
        if k == "examples":
            continue
        lines.append(f"{k}: {v}")
    ex = summary["model_resize"].get("examples", [])
    if ex:
        lines.append("\nExamples (up to 5):")
        for e in ex:
            lines.append(f"- token={e['token']} | pieces={e['pieces']} | used_fallback={e['used_fallback']} | in_vec_sum={e['in_vec_sum']:.6f}")
    lines.append("===============================================")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))

def try_resize_and_save_model(tok_new, actually_added_lists: Dict[str, List[str]]) -> Dict[str, str]:
    info = {}
    if not TRY_RESIZE_MODEL:
        info["status"] = "skip"
        info["message"] = "TRY_RESIZE_MODEL=False, only save tokenizer."
        return info

    try:
        tok_ref = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)

        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            SRC, trust_remote_code=True, low_cpu_mem_usage=False
        )
        info["loader"] = "AutoModelForImageTextToText"

        sem_stats = semantic_initialize_new_tokens(
            model=model,
            tok_new=tok_new,
            tok_ref=tok_ref,
            actually_added_lists=actually_added_lists,
        )

        Path(DST).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(DST)

        info.update({
            "status": "ok",
            "message": "Model token embeddings matrix has been extended and semantic initialized and saved.",
            "total_new_tokens": sem_stats["total_new_tokens"],
            "semantic_inited": sem_stats["semantic_inited"],
            "fallback_global": sem_stats["fallback_global"],
            "missing_in_output_head": sem_stats["missing_in_output_head"],
            "tied_embeddings": sem_stats["tied_embeddings"],
            "examples": sem_stats["examples"],
        })
        return info

    except Exception as e:
        info["status"] = "fail"
        info["message"] = (
            f"Extension/initialization failed: {type(e).__name__}: {e}\n"
            "Only saved tokenizer; can call "
            "model.resize_token_embeddings(len(tokenizer)) and refer to the logic of this script to complete semantic initialization."
        )
        return info

def main():
    Path(DST).mkdir(parents=True, exist_ok=True)

    tok_ref = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
    tok_new = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
    orig_size = len(tok_new)

    ints, frac2, frac1 = gen_numeric_tokens()
    n_int, added_int = add_tokens_unique(tok_new, ints)
    n_f2, added_f2 = add_tokens_unique(tok_new, frac2)
    n_f1, added_f1 = add_tokens_unique(tok_new, frac1)

    n_attr, added_attr = add_tokens_unique(tok_new, ATTR_TOKENS)

    n_tag, added_tag = add_tokens_unique(tok_new, TAG_TOKENS)

    new_size = len(tok_new)
    tok_new.save_pretrained(DST)

    actually_added_lists = {
        "integers": added_int,
        "frac2": added_f2,
        "frac1": added_f1,
        "attrs": added_attr,
        "tags": added_tag,
    }

    model_info = try_resize_and_save_model(tok_new, actually_added_lists)

    summary = {
        "paths": {"src": SRC, "dst": DST},
        "sizes": {"orig_vocab": orig_size, "new_vocab": new_size},
        "added_counts": {
            "integers": n_int,
            "frac2": n_f2,
            "frac1": n_f1,
            "attrs": n_attr,
            "tags": n_tag,
        },
        "actually_added": actually_added_lists,
        "model_resize": model_info,
    }
    write_report(DST, summary)

    to_copy_files = [
        "preprocessor_config.json",
        "processor_config.json",
        "image_processor.json",
        "vision_config.json",
    ]

    for fname in to_copy_files:
        src_path = Path(SRC) / fname
        dst_path = Path(DST) / fname
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"[COPY] {fname}")

if __name__ == "__main__":
    main()
