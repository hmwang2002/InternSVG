import os
import json
import argparse
import shutil

from metrics.video.video_metrics import InternSVGVideoMetrics, VideoMetricsConfig


def evaluate(test_file_path: str, gt_video_dir: str, gt_svg_dir: str, overall_video_dir: str, test_dir: str, tokenizer_path: str, task: str):
    pred_videos = []
    gt_videos = []
    caption = []
    gt_svg_paths = []
    pred_svg_paths = []
    overall_video = []
    pred_video_fvd = []
    total_samples = 0
    with open(test_file_path, 'r') as f:
        lines = f.readlines()
        total_samples = len(lines)
        for line in lines:
            data = json.loads(line)
            id = data['id']
            pred_video_path = os.path.join(test_dir, 'video', f'{id}.mp4')
            if os.path.exists(pred_video_path):
                pred_videos.append(pred_video_path)
                pred_svg_path = os.path.join(test_dir, 'svg', f'{id}.svg')
                pred_svg_paths.append(pred_svg_path)
            else:
                pred_videos.append('example/pure_black.mp4')
                pred_svg_paths.append(None)
                
            pred_fvd_video_path = os.path.join(test_dir, 'video_128', f'{id}.mp4')
            if not os.path.exists(pred_fvd_video_path):
                shutil.copy('example/pure_black_128.mp4', pred_fvd_video_path)
            pred_video_fvd.append(os.path.dirname(pred_fvd_video_path))
            overall_video.append(overall_video_dir)
            
            gt_video_path = os.path.join(gt_video_dir, f'{id}.mp4')
            gt_videos.append(gt_video_path)
            gt_svg_path = os.path.join(gt_svg_dir, f'{id}.svg')
            gt_svg_paths.append(gt_svg_path)
            # get instruction as caption
            # e.g. xxxx Instruction: xxxx
            question = data['conversations'][0]['value']
            caption.append(question.split('Instruction: ')[1])
    
    gt_svgs = []
    pred_svgs = []
    valid_preds = 0
    for gt_svg_path, pred_svg_path in zip(gt_svg_paths, pred_svg_paths):
        if pred_svg_path is None:
            gt_svgs.append('')
            pred_svgs.append('')
            continue
        
        with open(pred_svg_path, 'r') as f:
            pred_svg = f.read()
        pred_svgs.append(pred_svg)
        
        with open(gt_svg_path, 'r') as f:
            gt_svg = f.read()
        gt_svgs.append(gt_svg)
        
        valid_preds += 1
        
    success_rate = valid_preds / total_samples
    print(f"Success rate: {success_rate}")
    
    batch = {
        'gt_video': gt_videos,
        'pred_video': pred_videos,
        'caption': caption,
        'gt_svg': gt_svgs,
        'pred_svg': pred_svgs,
        'overall_video': overall_video,
        'pred_video_fvd': pred_video_fvd,
    }
    
    if task == 'text2sani':
        config = VideoMetricsConfig(
            use_FVD=True,
            use_ViCLIP_T2V=True,
            use_ViCLIP_V2V=True,
            use_DINO_Video=False,
            use_SSIM_Video=False,
            use_LPIPS_Video=False,
            use_PSNR_Video=False,
            use_token_length=True,
        )
    elif task == 'video2sani':
        config = VideoMetricsConfig(
            use_FVD=False,
            use_ViCLIP_T2V=False,
            use_ViCLIP_V2V=False,
            use_DINO_Video=True,
            use_SSIM_Video=True,
            use_LPIPS_Video=True,
            use_PSNR_Video=True,
            use_token_length=True,
        )
        
    calculator = InternSVGVideoMetrics(config, tokenizer_path)
    calculator.calculate_metrics(batch)
    print(calculator.summarize_metrics())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MODEL_NAME')
    parser.add_argument('--test_file_path', type=str, default='PATH_TO_TEST_FILE')
    parser.add_argument('--gt_video_dir', type=str, default='PATH_TO_GT_VIDEO_DIR')
    parser.add_argument('--gt_svg_dir', type=str, default='PATH_TO_GT_SVG_DIR')
    parser.add_argument('--overall_video_dir', type=str, default='PATH_TO_OVERALL_VIDEO_DIR')
    parser.add_argument('--text2sani_test_dir', type=str, default='PATH_TO_TEXT2SANI_TEST_DIR')
    parser.add_argument('--video2sani_test_dir', type=str, default='PATH_TO_VIDEO2SANI_TEST_DIR')
    parser.add_argument('--tokenizer_path', type=str, default='PATH_TO_TOKENIZER')
    return parser.parse_args()

def main():
    args = parse_args()
    MODEL_NAME = args.model_name
    print(f'Evaluating {MODEL_NAME}...')
    TEST_FILE_PATH = args.test_file_path
    GT_VIDEO_DIR = args.gt_video_dir
    GT_SVG_DIR = args.gt_svg_dir
    OVERALL_VIDEO_DIR = args.overall_video_dir
    TEXT2SANI_TEST_DIR = args.text2sani_test_dir
    VIDEO2SANI_TEST_DIR = args.video2sani_test_dir
    TOKENIZER_PATH = args.tokenizer_path
    
    if len(TEXT2SANI_TEST_DIR) > 0:
        evaluate(TEST_FILE_PATH, GT_VIDEO_DIR, GT_SVG_DIR, OVERALL_VIDEO_DIR, TEXT2SANI_TEST_DIR, TOKENIZER_PATH, 'text2sani')
    if len(VIDEO2SANI_TEST_DIR) > 0:
        evaluate(TEST_FILE_PATH, GT_VIDEO_DIR, GT_SVG_DIR, OVERALL_VIDEO_DIR, VIDEO2SANI_TEST_DIR, TOKENIZER_PATH, 'video2sani')

if __name__ == '__main__':
    main()
    
    