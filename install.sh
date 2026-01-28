#!/bin/bash

pip install -r requirements.txt

# install clip
pip install git+https://github.com/openai/CLIP.git

# Download ViCLIP
mkdir sarena_ckpt
cd sarena_ckpt
# You need to login first and have the access to the repo https://huggingface.co/OpenGVLab/ViCLIP. Use the command "huggingface-cli login" to login.
huggingface-cli download --resume-download OpenGVLab/ViCLIP ViClip-InternVid-10M-FLT.pth --local-dir .
cd ..

# install svgo
conda install nodejs
npm install -g svgo

# For training
pip install deepspeed==0.16.9
pip install av==14.4.0
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..