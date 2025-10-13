import os
import json
import argparse

from PIL import Image
from metrics.metrics import MetricsConfig, InternSVGMetrics


TASK_DICT = {
    'color_complex': 'Semantic-level Color Editing',
    'color_simple': 'Low-level Color Editing',
    'crop': 'Cropping',
    'flip': 'Flipping',
    'rotate': 'Rotation',
    'scale': 'Scaling',
    'translate': 'Translation',
    'opacity': 'Transparency',
    'styletransform_openmoji': 'Style Transfer',
    'outline': 'Adding stroke',
}

def load_images(paths):
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(im.copy())
    return images

def evaluate(gt_dir, test_dir: str, tokenizer_path: str):
    config = MetricsConfig(
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
    calculator = InternSVGMetrics(config, tokenizer_path)
    
    global_samples = 0
    global_valid_preds = 0
    for sub_dir in os.listdir(test_dir):
        task_name = sub_dir
        print(f'Evaluating {task_name}...')
        
        test_file_path = os.path.join(gt_dir, f'{task_name}.jsonl')
        test_task_dir = os.path.join(test_dir, sub_dir)
        gt_img_dir = os.path.join(gt_dir, 'data', task_name, 'images')
        gt_svg_dir = os.path.join(gt_dir, 'data', task_name, 'svg')

        gt_img_paths = []
        pred_img_paths = []
        gt_svg_paths = []
        pred_svg_paths = []
        total_samples = 0
        with open(test_file_path, 'r') as f:
            lines = f.readlines()
            total_samples = len(lines)
            global_samples += total_samples
            for line in lines:
                data = json.loads(line)
                id = data['id']
                # whether pred_image exists
                pred_img_path = os.path.join(test_task_dir, 'images', f'{id}.png')
                if os.path.exists(pred_img_path):
                    pred_img_paths.append(pred_img_path)
                    pred_svg_path = os.path.join(test_task_dir, 'svg', f'{id}.svg')
                    pred_svg_paths.append(pred_svg_path)
                else:
                    pred_img_paths.append('example/pure_black.png')
                    pred_svg_paths.append(None)
                
                gt_img_path = os.path.join(gt_img_dir, f'{id}.png')
                gt_img_paths.append(gt_img_path)
                gt_svg_path = os.path.join(gt_svg_dir, f'{id}.svg')
                gt_svg_paths.append(gt_svg_path)

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
        global_valid_preds += valid_preds
        batch = {
            'gt_im': gt_imgs,
            'pred_im': pred_imgs,
            'gt_svg': gt_svgs,
            'pred_svg': pred_svgs,
        }
        
        results_dict = calculator.calculate_metrics(batch)
        print(TASK_DICT[task_name] + ':')
        print(results_dict)
    print('--------------------------------')
    print(f'global success rate: {global_valid_preds / global_samples:.4%}')
    print(calculator.summarize_metrics())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MODEL_NAME')
    parser.add_argument('--gt_dir', type=str, default='PATH_TO_GT_DIR')
    parser.add_argument('--test_dir', type=str, default='PATH_TO_TEST_DIR')
    parser.add_argument('--tokenizer_path', type=str, default='PATH_TO_TOKENIZER')
    return parser.parse_args()

def main():
    args = parse_args()
    MODEL_NAME = args.model_name
    print(f'Evaluating {MODEL_NAME}...')
    gt_dir = args.gt_dir
    test_dir = args.test_dir
    tokenizer_path = args.tokenizer_path
    if len(test_dir) > 0:
        evaluate(gt_dir, test_dir, tokenizer_path)


if __name__ == '__main__':
    main()

