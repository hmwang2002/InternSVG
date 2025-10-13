import asyncio
import os
import tempfile
import glob
import nest_asyncio
from pyppeteer import launch
from moviepy import ImageSequenceClip
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import shutil

# --- Configuration parameters ---
DURATION_SECONDS = 5   # Video duration (seconds)
FPS = 30               # Frame rate

# Allow controlling concurrency through environment variables
MAX_WORKERS = int(os.getenv("SVG2MP4_MAX_WORKERS", "0"))  # 0 means automatic
if MAX_WORKERS <= 0:
    MAX_WORKERS = min(max(1, mp.cpu_count() // 2), 16)

# Chromium startup parameters (more stable in multiprocessing)
CHROME_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-software-rasterizer",
]

async def svg_to_mp4(svg_file_path: str, output_dir: str, width: int = 448, height: int = 448):
    VIEWPORT = {'width': width, 'height': height}
    """
    Single file rendering and export logic: called in the [current process].
    Each call will start and close a Chromium (unrelated in multiprocessing).
    """
    os.makedirs(output_dir, exist_ok=True)
    output_mp4_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(svg_file_path))[0] + ".mp4"
    )

    num_frames = int(DURATION_SECONDS * FPS)

    # Use independent user-data-dir for the current rendering instance, avoid multiprocessing conflicts
    tmp_profile_dir = tempfile.mkdtemp(prefix="pptr_profile_")

    try:
        with tempfile.TemporaryDirectory() as frame_dir:
            # Launch browser
            browser = await launch(
                headless=True,
                args=CHROME_ARGS + [f"--user-data-dir={tmp_profile_dir}"]
            )
            page = await browser.newPage()
            await page.setViewport(VIEWPORT)

            # Load SVG (wrap with HTML + CSS, make the container fill the viewport)
            with open(svg_file_path, "r", encoding="utf-8") as f:
                svg_content = f.read()

            html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {{
      margin: 0; padding: 0;
      width: 100%; height: 100%;
      background: #ffffff;
      overflow: hidden;
    }}
    #stage {{
      width: 100vw;
      height: 100vh;
      display: block;
    }}
    #stage > svg {{
      width: 100%;
      height: 100%;
      display: block;
    }}
  </style>
</head>
<body>
  <div id="stage">{svg_content}</div>
</body>
</html>
"""
            await page.setContent(html)
            await page.waitForSelector("svg", {"timeout": 10000})

            # Insert white background & standardize SVG dimensions (select the actual <svg> node)
            await page.evaluate(
    """(Width, Height) => {
      const svg = document.querySelector('svg');
      if (!svg) return;
      const hasViewBox = svg.hasAttribute('viewBox');
      if (!hasViewBox) {
        const wAttr = parseFloat((svg.getAttribute('width')  || '').replace('px','')) || Width;
        const hAttr = parseFloat((svg.getAttribute('height') || '').replace('px','')) || Height;
        svg.setAttribute('viewBox', `0 0 ${wAttr} ${hAttr}`);
      }

      // Width and height fill the container, keeping aspect ratio centered
      svg.removeAttribute('x');
      svg.removeAttribute('y');
      svg.setAttribute('width', '100%');
      svg.setAttribute('height', '100%');
      svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

      // Calculate a white background rect based on viewBox (if no viewBox, fall back to Width x Height)
      const vb = svg.viewBox && svg.viewBox.baseVal;
      let x = 0, y = 0, w, h;
      if (vb && vb.width && vb.height) {
        x = vb.x; y = vb.y; w = vb.width; h = vb.height;
      } else {
        w = Width; h = Height;
      }

      if (!svg.querySelector("rect[data-bg='1']")) {
        const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        bg.setAttribute("x", x);
        bg.setAttribute("y", y);
        bg.setAttribute("width", w);
        bg.setAttribute("height", h);
        bg.setAttribute("fill", "#ffffff");
        bg.setAttribute("data-bg", "1");
        svg.insertBefore(bg, svg.firstChild);
      }
    }""",
    width, height
)

            frame_paths = []
            for i in range(num_frames):
                t = i / FPS
                frame_path = os.path.join(frame_dir, f"frame_{i:05d}.png")

                # Use <svg> SMIL control API
                await page.evaluate("""
(() => {
  const root = document.querySelector('svg');
  if (root && typeof root.pauseAnimations === 'function') root.pauseAnimations();
})
""")
                await page.evaluate(f"""
(() => {{
  const root = document.querySelector('svg');
  if (root && typeof root.setCurrentTime === 'function') {{
    root.setCurrentTime({t});
  }}
}})
""")

                await page.screenshot({
                    "path": frame_path,
                    "omitBackground": False
                })
                frame_paths.append(frame_path)

            await browser.close()

            if not frame_paths:
                raise RuntimeError("No frames captured.")

            frame_paths.sort()
            clip = ImageSequenceClip(frame_paths, fps=FPS)
            clip.write_videofile(
                output_mp4_path,
                codec="libx264",
                audio=False,
                preset="medium",
                ffmpeg_params=["-pix_fmt", "yuv420p"]
            )
            clip.close()

    finally:
        try:
            shutil.rmtree(tmp_profile_dir, ignore_errors=True)
        except Exception:
            pass


def _ensure_event_loop():
    """Ensure there is an available event loop in the subprocess."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
        return loop
    except Exception:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def wrapper(svg_file: str, output_dir: str, width: int = 448, height: int = 448):
    """
    Subprocess entry: create event loop and call asynchronous rendering function.
    Throw exception when failed, captured by main process.
    """
    loop = _ensure_event_loop()
    loop.run_until_complete(svg_to_mp4(svg_file, output_dir, width, height))
    return svg_file  # Return the successfully processed file path


def run_all_mp():
    width = 448
    height = 448
    # Your output and input directories (keep as is)
    OUTPUT_DIRS = [
        'PATH_TO_OUTPUT_DIR',
    ]

    FILE_DIRS = [
        'PATH_TO_SVG_DIR',
    ]

    # Collect tasks to process
    tasks = []
    for file_dir, out_dir in zip(FILE_DIRS, OUTPUT_DIRS):
        os.makedirs(out_dir, exist_ok=True)
        if '_128' in out_dir:
            width = 128
            height = 128
        else:
            width = 448
            height = 448
        svg_files = glob.glob(os.path.join(file_dir, '*.svg'))
        for svg_file in svg_files:
            mp4_file_path = os.path.join(out_dir, os.path.basename(svg_file).replace('.svg', '.mp4'))
            if not os.path.exists(mp4_file_path):
                tasks.append((svg_file, out_dir, width, height))

    if not tasks:
        print("No SVG to process.")
        return

    failed_files = []
    # Multiprocessing parallel
    print(f"Concurrent process number: {MAX_WORKERS}, number of files to process: {len(tasks)}")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(wrapper, svg, out_dir, width, height) for svg, out_dir, width, height in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="SVG→MP4"):
            try:
                fut.result()
            except Exception as e:
                # Cannot know which file failed here, so print it before throwing exception in wrapper
                failed_files.append(str(e))

    if failed_files:
        print(f"Failed files (count={len(failed_files)}):")
        for f in failed_files:
            print(f"  - {f}")
    else:
        print("All files processed, no failed files.")


def run_all():
    width = 448
    height = 448
    """
    If you still want to keep the original serial running, you can call this function;
    Recommended to use run_all_mp() for multiprocessing acceleration.
    """
    nest_asyncio.apply()
    OUTPUT_DIRS = [
        'PATH_TO_OUTPUT_DIR',
    ]

    file_dirs = [
        'PATH_TO_SVG_DIR',
    ]
       
    failed_files = []
    for file_dir, output_dir in zip(file_dirs, OUTPUT_DIRS):
        if '_128' in output_dir:
            width = 128
            height = 128
        else:
            width = 448
            height = 448
        svg_files = glob.glob(os.path.join(file_dir, '*.svg'))
        for svg_file in tqdm(svg_files):
            mp4_file_path = os.path.join(output_dir, os.path.basename(svg_file).replace('.svg', '.mp4'))
            if not os.path.exists(mp4_file_path):
                try:
                    asyncio.get_event_loop().run_until_complete(svg_to_mp4(svg_file, output_dir, width, height))
                except Exception as e:
                    print(f"Error processing {svg_file}: {e}")
                    failed_files.append(svg_file)
    print(f"Failed files: {failed_files}")

def run_single(svg_path, out_dir, width, height):
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(svg_to_mp4(svg_path, out_dir, width, height))

if __name__ == "__main__":
    # Recommended to use multiprocessing version
    run_all_mp()
    # If you want to run serially, change to: run_all()
    # run_single('PATH_TO_SVG_PATH', 'PATH_TO_OUTPUT_DIR', 128, 128)