import os
import json


def InternVL2Aplaca(file_path: str, output_path: str, task: str='understanding'):
    if task == 'understanding':
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                with open(output_path, 'a') as fout:
                    fout.write(json.dumps({
                        'instruction': data['conversations'][0]['value'],
                        'input': '',
                        'output': data['conversations'][1]['value'],
                    }) + '\n')
    elif task == 'text2svg':
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                with open(output_path, 'a') as fout:
                    fout.write(json.dumps({
                        'instruction': data['conversations'][0]['value'],
                        'input': '',
                        'output': data['conversations'][1]['value'],
                    }) + '\n')
    elif task == 'img2svg':
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                
                with open(output_path, 'a') as fout:
                    fout.write(json.dumps({
                        'instruction': data['conversations'][0]['value'],
                        'input': '',
                        'output': data['conversations'][1]['value'],
                        'images': [data['image']],
                    }) + '\n')
    else:
        raise ValueError(f"Invalid task: {task}")
    print("Process finished: save to ", output_path)
                

def Alpaca2InternVL(file_path: str, output_path: str, task: str='understanding'):
    if task == 'understanding':
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                data = json.loads(line)
                with open(output_path, 'a') as fout:
                    fout.write(json.dumps({
                        'id': idx,
                        'conversations': [
                            {
                                'from': 'human',
                                'value': data['instruction'] + '\n' + data['input'],
                            },
                            {
                                'from': 'gpt',
                                'value': data['output'],
                            }
                        ],
                    }) + '\n')
    elif task == 'text2svg':
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                data = json.loads(line)
                with open(output_path, 'a') as fout:
                    fout.write(json.dumps({
                        'id': idx,
                        'conversations': [
                            {
                                'from': 'human',
                                'value': data['instruction'] + '\n' + data['input'],
                            },
                            {
                                'from': 'gpt',
                                'value': data['output'],
                            }
                        ],
                    }) + '\n')
    elif task == 'img2svg':
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                data = json.loads(line)
                image = data['images'][0]
                with open(output_path, 'a') as fout:
                    fout.write(json.dumps({
                        'id': idx,
                        'image': image,
                        'image_wh': [[448, 448]],
                        'conversations': [
                            {
                                'from': 'human',
                                'value': data['instruction'] + '\n' + data['input'],
                            },
                            {
                                'from': 'gpt',
                                'value': data['output'],
                            }
                        ],
                    }) + '\n')
    else:
        raise ValueError(f"Invalid task: {task}")
    print("Process finished: save to ", output_path)
        