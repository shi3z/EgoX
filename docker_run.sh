#!/bin/bash
set -e

# Install dependencies
pip install -q diffusers==0.34.0 transformers accelerate sentencepiece peft imageio imageio-ffmpeg tyro ftfy opencv-python-headless wandb

# Run inference
python infer.py \
  --prompt ./webui_output/IMG_3128/caption.txt \
  --exo_video_path ./webui_output/IMG_3128/exo_path.txt \
  --ego_prior_video_path ./webui_output/IMG_3128/ego_prior_path.txt \
  --meta_data_file ./webui_output/IMG_3128/camera_params.json \
  --depth_root ./webui_output/IMG_3128/depth/ \
  --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
  --lora_path ./checkpoints/EgoX/pytorch_lora_weights.safetensors \
  --lora_rank 256 \
  --out ./results_docker \
  --seed 42 \
  --cos_sim_scaling_factor 3.0 \
  --idx 0 \
  --use_GGA
