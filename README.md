# EgoX: Egocentric Video Generation from a Single Exocentric Video with WebUI

<video src="https://github.com/user-attachments/assets/06bea3a8-afa8-437c-b570-e0e845d744cc" controls width="100%"></video>


> **This is an extended fork of [DAVIAN-Robotics/EgoX](https://github.com/DAVIAN-Robotics/EgoX)**
>
> This fork includes **independently implemented Ego Prior generation** and **WebUI** features.
> The original repository has not yet released the data preprocessing code for Ego Prior generation.
> We reverse-engineered the Ego Prior pipeline based on the paper and implemented it from scratch.

[![Hugging Face Paper](https://img.shields.io/badge/HuggingFace-Paper%20of%20the%20Day%20%231-orange)](https://huggingface.co/papers/2512.08269)
[![arXiv](https://img.shields.io/badge/arXiv-2510.26236-b31b1b.svg)](https://arxiv.org/abs/2512.08269)
[![Project Page](https://img.shields.io/badge/Project_Page-Visit-blue.svg)](https://keh0t0.github.io/EgoX/)
[![Original Repo](https://img.shields.io/badge/Original-DAVIAN--Robotics%2FEgoX-green)](https://github.com/DAVIAN-Robotics/EgoX)

> [Taewoong Kang\*](https://keh0t0.github.io/), [Kinam Kim\*](https://kinam0252.github.io/), [Dohyeon Kim\*](https://linkedin.com/in/dohyeon-kim-a79231347), [Minho Park](https://pmh9960.github.io/), [Junha Hyung](https://junhahyung.github.io/), and [Jaegul Choo](https://sites.google.com/site/jaegulchoo/)
>
> **DAVIAN Robotics, KAIST AI, SNU**
> arXiv 2025. (\* indicates equal contribution)

## 🆕 New Features (Fork by shi3z)

### Flask WebUI

A Flask-based web interface with background worker processing and real-time progress tracking. This WebUI integrates **EgoX-EgoPriorRenderer** for high-quality Ego Prior generation using ViPE (Video Pose Engine).

#### Docker Setup (Required for NVIDIA Blackwell GPUs / GB10)

For NVIDIA Blackwell architecture GPUs (sm_121, e.g., GB10), you must use the NGC 25.11 container:

```bash
# 1. Clone the repository with submodules
git clone --recursive https://github.com/shi3z/EgoX.git
cd EgoX

# 2. Start the Docker container
docker compose up -d

# 3. Install dependencies inside the container
docker exec -it egox-egox-webui-1 bash -c "
  cd /workspace/EgoX && \
  pip install flask diffusers==0.34.0 transformers accelerate sentencepiece peft imageio imageio-ffmpeg tyro ftfy opencv-python-headless wandb && \
  apt-get update && apt-get install -y ffmpeg
"

# 4. Install EgoX-EgoPriorRenderer dependencies
docker exec -it egox-egox-webui-1 bash -c "
  cd /workspace/EgoX/EgoX-EgoPriorRenderer && \
  pip install -e .
"

# 5. Download model weights (see Model Weights Download section below)
# Make sure models are in ./checkpoints/ directory

# 6. Start the Flask WebUI (on host machine)
python flask_webui.py

# 7. Open in browser: http://localhost:7861
```

#### docker-compose.yml Configuration

```yaml
services:
  egox-webui:
    image: nvcr.io/nvidia/pytorch:25.11-py3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "7860:7860"
    volumes:
      - .:/workspace/EgoX
      - ~/.cache:/root/.cache
    working_dir: /workspace/EgoX
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    command: sleep infinity  # Keep container running
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
```

#### Processing Pipeline

The Flask WebUI executes the following 5-step pipeline:

1. **Video Preparation** - Extract frames, resize to 784x448, limit to 49 frames
2. **ViPE Inference** - Depth estimation and camera pose extraction using ViPE
3. **Ego Prior Rendering** - 3D point cloud rendering from egocentric viewpoint
4. **Depth Conversion** - Convert ViPE depth maps to EgoX format
5. **EgoX Inference** - Generate final egocentric video using Wan2.1-I2V-14B + LoRA

**Features:**
- Upload any exocentric (3rd person) video
- High-quality Ego Prior generation using EgoX-EgoPriorRenderer + ViPE
- Background worker processing with real-time progress tracking
- Job queue management with status monitoring
- Customizable prompt text (auto-generated or manual)
- Accessible over network

**Processing Time:**
- ViPE + Ego Prior rendering: ~5-10 minutes
- EgoX inference: ~30-40 minutes (50 diffusion steps)

#### Output Structure

```
webui_output/
└── <job-uuid>/
    ├── exo.mp4              # Resized input video (784x448, 49 frames)
    ├── prompt.txt           # Scene description prompt
    ├── vipe_results/        # ViPE depth maps and camera poses
    ├── videos/
    │   └── ego_Prior.mp4    # Generated Ego Prior video
    ├── depth_maps/          # Converted depth maps for EgoX
    └── results/
        └── EgoX.mp4         # Final egocentric video output
```

### Legacy Gradio WebUI

A simpler Gradio-based interface (uses Depth Anything V2 instead of ViPE):

```bash
python webui.py --host 0.0.0.0 --port 7860
```

---

## 🛠️ Environment Setup

### System Requirements

- **GPU**: < 80GB (for inference)
- **CUDA**: 12.1 or higher
- **Python**: 3.10
- **PyTorch**: Compatible with CUDA 12.1

### Installation

Create a conda environment and install dependencies:

```bash
# Create conda environment
conda create -n egox python=3.10 -y
conda activate egox

# Install PyTorch with CUDA 12.1
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

## 📥 Model Weights Download

### 💾 Wan2.1-I2V-14B Pretrained Model

Download the [Wan2.1-I2V-14B](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) model and save it to the `checkpoints/pretrained_model/` folder.

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers')"
```

### 💾 EgoX Model Weights Download

Download the trained EgoX LoRA weights using one of the following methods:

**Option 1: Hugging Face**
```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='DAVIAN-Robotics/EgoX', local_dir='./checkpoints/EgoX', allow_patterns='*.safetensors')"
```

**Option 2: Google Drive**
- Download from [Google Drive](https://drive.google.com/file/d/1Q7j7LVI4YiSkwzNMBBiyLS1rT3HMcNVB/view?usp=drive_link) and save to the `checkpoints/EgoX/` folder.


## 🙏 Acknowledgements

This project is built upon the following works:

- [4DNeX](https://github.com/3DTopia/4DNeX)
- [Ego-Exo4D](https://github.com/facebookresearch/Ego-Exo)

## 📝 Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@misc{kang2025egoxegocentricvideogeneration,
      title={EgoX: Egocentric Video Generation from a Single Exocentric Video}, 
      author={Taewoong Kang and Kinam Kim and Dohyeon Kim and Minho Park and Junha Hyung and Jaegul Choo},
      year={2025},
      eprint={2512.08269},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.08269}, 
}
```
