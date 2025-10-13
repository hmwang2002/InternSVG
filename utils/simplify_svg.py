import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


SVGO_BIN = "svgo"
CONFIG   = "./svgo/config.mjs"
ANIMATION_CONFIG = "./svgo/config_smil.mjs"


def compress_svg(src_path: Path, dst_path: Path):
    cmd = [
        SVGO_BIN,
        str(src_path),
        "-o",
        str(dst_path),
        "--config",
        CONFIG,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return dst_path, result.returncode, result.stderr.strip()

def compress_animation_svg(src_path: Path, dst_path: Path):
    cmd = [
        SVGO_BIN,
        str(src_path),
        "-o",
        str(dst_path),
        "--config",
        ANIMATION_CONFIG,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return dst_path, result.returncode, result.stderr.strip()

def gather_svg_paths(directory: Path) -> list[Path]:
    return [p for p in directory.rglob("*.svg") if p.is_file()]

def simplify_svg(svg_dir: Path, output_dir: Path, max_workers: int = 64):
    svg_paths = gather_svg_paths(svg_dir)
    if not svg_paths:
        print("⚠️  No SVGs to compress")
        return

    print(f"🔧 Compressing {len(svg_paths)} SVGs (parallel {max_workers} processes)")

    os.makedirs(output_dir, exist_ok=True)
    
    # Slice, process 50000 SVGs at a time
    svg_paths = [svg_paths[i: min(i+50000, len(svg_paths))] for i in range(0, len(svg_paths), 50000)]
    failed = 0
    for i, paths in enumerate(svg_paths):
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(compress_svg, p, output_dir) for p in paths]

            for f in tqdm(as_completed(futures), total=len(futures)):
                dst, rc, err = f.result()
                if rc != 0:
                    failed += 1
                    tqdm.write(f"❌ {dst.name} failed (code={rc})\n{err}")

    print(f"🎉 All done, failed {failed} SVGs")
    
def simplify_animation_svg(svg_dir: Path, output_dir: Path, max_workers: int = 64):
    svg_paths = gather_svg_paths(svg_dir)
    if not svg_paths:
        print("⚠️  No SVGs to compress")
        return

    print(f"🔧 Compressing {len(svg_paths)} SVGs (parallel {max_workers} processes)")

    os.makedirs(output_dir, exist_ok=True)
    
    # Slice, process 50000 SVGs at a time
    svg_paths = [svg_paths[i: min(i+50000, len(svg_paths))] for i in range(0, len(svg_paths), 50000)]
    failed = 0
    for i, paths in enumerate(svg_paths):
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(compress_animation_svg, p, output_dir) for p in paths]

            for f in tqdm(as_completed(futures), total=len(futures)):
                dst, rc, err = f.result()
                if rc != 0:
                    failed += 1
                    tqdm.write(f"❌ {dst.name} failed (code={rc})\n{err}")

    print(f"🎉 All done, failed {failed} SVGs")
        

if __name__ == "__main__":
    svg_dir = Path("PATH_TO_SVG_DIR")
    output_dir = Path("PATH_TO_OUTPUT_DIR")
    simplify_svg(svg_dir, output_dir, max_workers=128)