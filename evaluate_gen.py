import os
import json
import argparse

from PIL import Image
from metrics.metrics import MetricsConfig, InternSVGMetrics


def load_images(paths):
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(im.copy())
    return images

def load_config(bench_name: str, task: str):
    if task == 'text2svg':
        if bench_name == 'Icon' or bench_name == 'Illustration':
            return MetricsConfig(
                use_FID = True,
                use_FID_C = True,
                use_CLIP_Score_T2I= True,
                use_CLIP_Score_I2I = True,
                use_DINO_Score = False,
                use_LPIPS = False,
                use_SSIM = False,
                use_PSNR = False,
                use_token_length = True
            )
        elif bench_name == 'Chem':
            return MetricsConfig(
                use_FID = True,
                use_FID_C = True,
                use_CLIP_Score_T2I= False,
                use_CLIP_Score_I2I = True,
                use_DINO_Score = False,
                use_LPIPS = False,
                use_SSIM = False,
                use_PSNR = False,
                use_token_length = True
            )
        else:
            raise ValueError(f'Invalid bench name: {bench_name}')

    elif task == 'img2svg':
        return MetricsConfig(
            use_FID = False,
            use_FID_C = False,
            use_CLIP_Score_T2I = False,
            use_CLIP_Score_I2I = False,
            use_DINO_Score = True,
            use_LPIPS = True,
            use_SSIM = True,
            use_PSNR = True,
            use_token_length = True
        )
    else:
        raise ValueError(f'Invalid task: {task}')

def evaluate(test_dir: str, tokenizer_path: str, task: str, test_file_path: str, gt_img_dir: str, gt_svg_dir: str, bench_name: str, caption_path: str):
    if bench_name != 'Chem' and len(caption_path) > 0:
        caption_dict = {}
        with open(caption_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                caption = data['caption'][0]['generated_text'].strip()
                caption_dict[data['image']] = caption
    
    gt_img_paths = []
    pred_img_paths = []
    gt_svg_paths = []
    pred_svg_paths = []
    caption = []
    total_samples = 0
    with open(test_file_path, 'r') as f:
        lines = f.readlines()
        total_samples = len(lines)
        for line in lines:
            data = json.loads(line)
            id = data['id']
            # whether pred_image exists
            pred_img_path = os.path.join(test_dir, 'images', f'{id}.png')
            if os.path.exists(pred_img_path):
                pred_img_paths.append(pred_img_path)
                pred_svg_path = os.path.join(test_dir, 'svg', f'{id}.svg')
                pred_svg_paths.append(pred_svg_path)
            else:
                pred_img_paths.append('example/pure_black.png')
                pred_svg_paths.append(None)
            
            gt_img_path = os.path.join(gt_img_dir, f'{id}.png')
            gt_img_paths.append(gt_img_path)
            gt_svg_path = os.path.join(gt_svg_dir, f'{id}.svg')
            gt_svg_paths.append(gt_svg_path)

            if bench_name != 'Chem' and len(caption_path) > 0:
                gt_img_name = os.path.basename(gt_img_path)
                caption.append(caption_dict[gt_img_name])

    gt_imgs = load_images(gt_img_paths)
    pred_imgs = load_images(pred_img_paths)

    gt_svgs = []
    pred_svgs = []
    valid_preds = 0
    for gt_svg_path, pred_svg_path in zip(gt_svg_paths, pred_svg_paths):
        if pred_svg_path is None:
            pred_svgs.append('')
            gt_svgs.append('')
            continue
        
        with open(pred_svg_path, 'r') as f:
            pred_svg = f.read()
        pred_svgs.append(pred_svg)
        
        with open(gt_svg_path, 'r') as f:
            gt_svg = f.read()
        gt_svgs.append(gt_svg)
        
        valid_preds += 1

    success_rate = valid_preds / total_samples
    print(f'success rate: {success_rate:.4%}')

    if bench_name == 'Chem' or len(caption_path) == 0:
        batch = {
            'gt_im': gt_imgs,
            'pred_im': pred_imgs,
            'gt_svg': gt_svgs,
            'pred_svg': pred_svgs
        }
    else:
        batch = {
            'gt_im': gt_imgs,
            'pred_im': pred_imgs,
            'gt_svg': gt_svgs,
            'pred_svg': pred_svgs,
            'caption': caption
        }

    config = load_config(bench_name, task)
    calculator = InternSVGMetrics(config, tokenizer_path)
    calculator.calculate_metrics(batch)
    print(calculator.summarize_metrics())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MODEL_NAME')
    parser.add_argument('--text2svg_test_dir', type=str, default='')
    parser.add_argument('--img2svg_test_dir', type=str, default='')
    parser.add_argument('--tokenizer_path', type=str, default='PATH_TO_TOKENIZER')
    parser.add_argument('--test_file_path', type=str, default='PATH_TO_TEST_FILE')
    parser.add_argument('--gt_img_dir', type=str, default='PATH_TO_GT_IMG_DIR')
    parser.add_argument('--gt_svg_dir', type=str, default='PATH_TO_GT_SVG_DIR')
    parser.add_argument('--caption_path', type=str, default='PATH_TO_CAPTION_FILE')
    parser.add_argument('--bench_name', type=str, default='Icon', choices=['Icon', 'Illustration', 'Chem'])
    return parser.parse_args()

def main():
    args = parse_args()
    MODEL_NAME = args.model_name
    print(f'Evaluating {MODEL_NAME}...')
    text2svg_test_dir = args.text2svg_test_dir
    img2svg_test_dir = args.img2svg_test_dir
    tokenizer_path = args.tokenizer_path
    test_file_path = args.test_file_path
    gt_img_dir = args.gt_img_dir
    gt_svg_dir = args.gt_svg_dir
    caption_path = args.caption_path
    bench_name = args.bench_name

    if len(text2svg_test_dir) > 0:
        evaluate(text2svg_test_dir, tokenizer_path, 'text2svg', test_file_path, gt_img_dir, gt_svg_dir, bench_name, caption_path)
    if len(img2svg_test_dir) > 0:
        evaluate(img2svg_test_dir, tokenizer_path, 'img2svg', test_file_path, gt_img_dir, gt_svg_dir, bench_name, caption_path)

if __name__ == '__main__':
    main()

