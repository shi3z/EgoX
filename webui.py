#!/usr/bin/env python3
"""
EgoX WebUI - Gradio-based web interface for EgoX video generation
Upload any exocentric video and automatically convert to egocentric view
"""
import os
import subprocess
import json
import shutil
import tempfile
from pathlib import Path
import gradio as gr

def check_model_status():
    """Check if models are downloaded"""
    model_path = "./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers"
    lora_path = "./checkpoints/EgoX/pytorch_lora_weights.safetensors"

    model_exists = os.path.exists(os.path.join(model_path, "transformer"))
    lora_exists = os.path.exists(lora_path)

    status = []
    if model_exists:
        status.append("Pretrained model: Ready")
    else:
        status.append("Pretrained model: Not found")

    if lora_exists:
        status.append("EgoX LoRA: Ready")
    else:
        status.append("EgoX LoRA: Not found")

    return "\n".join(status)


def generate_ego_prior(video_path: str, output_dir: str, camera_position: str = "center", look_direction: str = "front"):
    """Generate Ego Prior from exocentric video using generate_ego_prior.py"""
    cmd = [
        "python", "generate_ego_prior.py",
        "--exo_video", video_path,
        "--output_dir", output_dir,
        "--trajectory", "center_look",
        "--ego_depth", "0.5",
        "--device", "cpu",  # Use CPU for depth estimation (GB10/Blackwell not yet supported by PyTorch)
        "--camera_position", camera_position,
        "--look_direction", look_direction,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    if result.returncode != 0:
        raise Exception(f"Ego Prior generation failed:\n{result.stderr}")

    return True


def prepare_inference_files(output_dir: str, take_name: str, caption: str):
    """Prepare files for inference"""
    with open(os.path.join(output_dir, "exo_path.txt"), 'w') as f:
        f.write(f"./{output_dir}/exo.mp4\n")

    with open(os.path.join(output_dir, "ego_prior_path.txt"), 'w') as f:
        f.write(f"./{output_dir}/ego_Prior.mp4\n")

    with open(os.path.join(output_dir, "caption.txt"), 'w') as f:
        f.write(caption.strip() + "\n")

    depth_root = os.path.join(output_dir, "depth")
    depth_target = os.path.join(depth_root, take_name)
    os.makedirs(depth_target, exist_ok=True)

    depth_src = os.path.join(output_dir, "depth_maps")
    if os.path.exists(depth_src):
        for f in os.listdir(depth_src):
            if f.endswith('.npy'):
                shutil.move(os.path.join(depth_src, f), os.path.join(depth_target, f))

    return True


def run_inference_with_progress(output_dir: str, seed: int, use_gga: bool, cos_sim_scale: float):
    """Run EgoX inference with progress updates (generator) using NVIDIA Docker container"""
    import re

    model_path = "./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers"
    lora_path = "./checkpoints/EgoX/pytorch_lora_weights.safetensors"

    take_name = os.path.basename(output_dir.rstrip('/'))
    results_dir = f"./results_{take_name}"

    # Build inference command for inside container
    infer_cmd = " ".join([
        "python", "-u", "infer.py",
        "--prompt", os.path.join(output_dir, "caption.txt"),
        "--exo_video_path", os.path.join(output_dir, "exo_path.txt"),
        "--ego_prior_video_path", os.path.join(output_dir, "ego_prior_path.txt"),
        "--meta_data_file", os.path.join(output_dir, "camera_params.json"),
        "--depth_root", os.path.join(output_dir, "depth/"),
        "--model_path", model_path,
        "--lora_path", lora_path,
        "--lora_rank", "256",
        "--out", results_dir,
        "--seed", str(seed),
        "--cos_sim_scaling_factor", str(cos_sim_scale),
        "--idx", "0",
        "--use_GGA" if use_gga else ""
    ])

    # Get absolute path for mounting
    workspace_path = os.path.dirname(os.path.abspath(__file__))

    # Docker command for NVIDIA container (GB10/Blackwell support)
    cmd = [
        "docker", "run", "--gpus", "all", "--rm",
        "--ipc=host", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-v", f"{workspace_path}:/workspace/EgoX",
        "-w", "/workspace/EgoX",
        "nvcr.io/nvidia/pytorch:25.09-py3",
        "bash", "-c",
        f"pip install -q diffusers==0.34.0 transformers accelerate sentencepiece peft imageio imageio-ffmpeg tyro ftfy opencv-python-headless wandb && {infer_cmd}"
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    last_progress = ""
    for line in process.stdout:
        # Parse progress from tqdm output like "50%|█████ | 25/50"
        match = re.search(r'(\d+)%\|[^|]+\|\s*(\d+)/(\d+)', line)
        if match:
            percent = match.group(1)
            current = match.group(2)
            total = match.group(3)
            last_progress = f"Diffusion: {percent}% ({current}/{total} steps)"
            yield last_progress, None
        elif "Loading" in line:
            yield "Loading model...", None
        elif "Video saved" in line:
            yield "Video saved!", None

    process.wait()

    if process.returncode != 0:
        raise Exception(f"Inference failed with code {process.returncode}")

    output_video = os.path.join(results_dir, f"{take_name}.mp4")
    if os.path.exists(output_video):
        yield "Complete!", output_video
    else:
        raise Exception(f"Output video not found at {output_video}")


def process_video(
    video_file,
    seed: int,
    use_gga: bool,
    cos_sim_scale: float,
    caption: str,
    camera_position: str,
    look_direction: str
):
    """Full pipeline: Upload -> Ego Prior -> Inference (generator for progressive updates)"""

    if video_file is None:
        yield None, None, "Please upload a video file."
        return

    model_path = "./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers"
    lora_path = "./checkpoints/EgoX/pytorch_lora_weights.safetensors"

    if not os.path.exists(os.path.join(model_path, "transformer")):
        yield None, None, "Error: Pretrained model not found."
        return

    if not os.path.exists(lora_path):
        yield None, None, "Error: EgoX LoRA weights not found."
        return

    try:
        video_name = Path(video_file).stem
        output_dir = f"./webui_output/{video_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Generate Ego Prior
        yield None, None, f"Step 1/3: Generating Ego Prior...\nPosition: {camera_position}, Direction: {look_direction}\nThis takes about 1-2 minutes."

        generate_ego_prior(video_file, output_dir, camera_position, look_direction)
        ego_prior_path = os.path.join(output_dir, "ego_Prior.mp4")

        # Show Ego Prior immediately after generation
        yield ego_prior_path, None, "Step 1/3: Ego Prior generated!\n\nStep 2/3: Preparing inference files..."

        # Step 2: Prepare inference files
        take_name = video_name
        prepare_inference_files(output_dir, take_name, caption)

        yield ego_prior_path, None, "Step 2/3: Inference files ready!\n\nStep 3/3: Running EgoX inference..."

        # Step 3: Run inference with progress
        output_video = None
        for progress_msg, video_path in run_inference_with_progress(output_dir, seed, use_gga, cos_sim_scale):
            if video_path:
                output_video = video_path
            yield ego_prior_path, output_video, f"Step 3/3: {progress_msg}"

        yield ego_prior_path, output_video, f"Complete!\n\nEgo Prior: {ego_prior_path}\nOutput: {output_video}"

    except Exception as e:
        yield None, None, f"Error: {str(e)}"


def create_ui():
    """Create the Gradio interface"""

    with gr.Blocks(title="EgoX - Egocentric Video Generation") as demo:
        gr.Markdown("""
        # EgoX: Egocentric Video Generation from a Single Exocentric Video

        Generate first-person (egocentric) videos from third-person (exocentric) video input.

        **Paper:** [arXiv:2512.08269](https://arxiv.org/abs/2512.08269) | **Project:** [EgoX Project Page](https://keh0t0.github.io/EgoX/)

        ---

        ## How to Use
        1. Upload an exocentric (third-person) video
        2. Adjust parameters if needed
        3. Click "Generate Egocentric Video"
        4. Wait for processing (~30-40 minutes)
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Status")
                model_status = gr.Textbox(
                    value=check_model_status(),
                    label="Status",
                    interactive=False,
                    lines=2
                )
                refresh_btn = gr.Button("Refresh Status", size="sm")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Video")

                video_input = gr.Video(
                    label="Exocentric (3rd person) Video",
                    sources=["upload"]
                )

                gr.Markdown("### Settings")

                seed_input = gr.Number(
                    value=42,
                    label="Seed",
                    precision=0
                )

                use_gga = gr.Checkbox(
                    value=True,
                    label="Use GGA (Geometry-Guided Attention)"
                )

                cos_sim_scale = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    value=3.0,
                    step=0.1,
                    label="Cosine Similarity Scaling Factor"
                )

                caption_input = gr.Textbox(
                    value="[Exo view] A person performing an action captured from a third-person perspective. The subject moves through the scene. [Ego view] From the first-person perspective, the hands and immediate surroundings are visible as the person interacts with the environment.",
                    label="Caption",
                    lines=4,
                    placeholder="Describe the scene from both Exo and Ego perspectives..."
                )

                gr.Markdown("### Viewpoint Settings")

                camera_position = gr.Radio(
                    choices=["center", "left", "right"],
                    value="center",
                    label="Camera Position (whose viewpoint?)"
                )

                look_direction = gr.Radio(
                    choices=["front", "left", "right", "up", "down"],
                    value="front",
                    label="Look Direction"
                )

                generate_btn = gr.Button("Generate Egocentric Video", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Output")

                output_status = gr.Textbox(label="Progress", interactive=False, lines=5)

                with gr.Row():
                    ego_prior_output = gr.Video(label="Generated Ego Prior (Step 1)")
                    ego_video_output = gr.Video(label="Final Egocentric Video (Step 3)")

        # Event handlers
        refresh_btn.click(
            fn=check_model_status,
            outputs=model_status
        )

        generate_btn.click(
            fn=process_video,
            inputs=[video_input, seed_input, use_gga, cos_sim_scale, caption_input, camera_position, look_direction],
            outputs=[ego_prior_output, ego_video_output, output_status]
        )

        gr.Markdown("""
        ---
        ### Notes
        - **GPU Requirement:** ~80GB VRAM
        - **Processing Time:** ~30-40 minutes per video
        - Video will be resized to 784x448 with 49 frames
        - Ego Prior is generated using Depth Anything V2
        """)

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    args = parser.parse_args()

    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft()
    )
