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


def generate_ego_prior(video_path: str, output_dir: str, progress_callback=None):
    """Generate Ego Prior from exocentric video using generate_ego_prior.py"""
    cmd = [
        "python", "generate_ego_prior.py",
        "--exo_video", video_path,
        "--output_dir", output_dir,
        "--trajectory", "center_look",
        "--ego_depth", "0.5",
        "--device", "cuda"
    ]

    if progress_callback:
        progress_callback(0.1, desc="Generating Ego Prior (depth estimation)...")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    if result.returncode != 0:
        raise Exception(f"Ego Prior generation failed:\n{result.stderr}")

    return True


def prepare_inference_files(output_dir: str, take_name: str):
    """Prepare files for inference"""
    # Create path files
    exo_path = os.path.join(output_dir, "exo.mp4")
    ego_prior_path = os.path.join(output_dir, "ego_Prior.mp4")

    with open(os.path.join(output_dir, "exo_path.txt"), 'w') as f:
        f.write(f"./{output_dir}/exo.mp4\n")

    with open(os.path.join(output_dir, "ego_prior_path.txt"), 'w') as f:
        f.write(f"./{output_dir}/ego_Prior.mp4\n")

    # Create caption (generic)
    with open(os.path.join(output_dir, "caption.txt"), 'w') as f:
        f.write("[Exo view] A person performing an action from third-person view. [Ego view] First-person perspective of the same scene.\n")

    # Setup depth directory structure
    depth_root = os.path.join(output_dir, "depth")
    depth_target = os.path.join(depth_root, take_name)
    os.makedirs(depth_target, exist_ok=True)

    # Move depth maps
    depth_src = os.path.join(output_dir, "depth_maps")
    if os.path.exists(depth_src):
        for f in os.listdir(depth_src):
            if f.endswith('.npy'):
                shutil.move(os.path.join(depth_src, f), os.path.join(depth_target, f))

    return True


def run_inference(output_dir: str, seed: int, use_gga: bool, cos_sim_scale: float, progress_callback=None):
    """Run EgoX inference"""
    model_path = "./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers"
    lora_path = "./checkpoints/EgoX/pytorch_lora_weights.safetensors"

    take_name = os.path.basename(output_dir.rstrip('/'))
    results_dir = f"./results_{take_name}"

    cmd = [
        "python", "infer.py",
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
        "--idx", "0"
    ]

    if use_gga:
        cmd.append("--use_GGA")

    if progress_callback:
        progress_callback(0.5, desc="Running EgoX inference (this takes ~30 minutes)...")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    if result.returncode != 0:
        raise Exception(f"Inference failed:\n{result.stderr[-2000:]}")

    # Find output video
    output_video = os.path.join(results_dir, f"{take_name}.mp4")
    if os.path.exists(output_video):
        return output_video
    else:
        raise Exception(f"Output video not found at {output_video}")


def process_video(
    video_file,
    seed: int,
    use_gga: bool,
    cos_sim_scale: float,
    progress=gr.Progress()
):
    """Full pipeline: Upload -> Ego Prior -> Inference"""

    if video_file is None:
        return None, None, "Please upload a video file."

    # Check models
    model_path = "./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers"
    lora_path = "./checkpoints/EgoX/pytorch_lora_weights.safetensors"

    if not os.path.exists(os.path.join(model_path, "transformer")):
        return None, None, "Error: Pretrained model not found."

    if not os.path.exists(lora_path):
        return None, None, "Error: EgoX LoRA weights not found."

    try:
        # Create output directory
        video_name = Path(video_file).stem
        output_dir = f"./webui_output/{video_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Generate Ego Prior
        progress(0.1, desc="Step 1/3: Generating Ego Prior...")
        generate_ego_prior(video_file, output_dir)

        # Get ego prior video path
        ego_prior_path = os.path.join(output_dir, "ego_Prior.mp4")

        # Step 2: Prepare inference files
        progress(0.3, desc="Step 2/3: Preparing inference files...")
        take_name = video_name
        prepare_inference_files(output_dir, take_name)

        # Step 3: Run inference
        progress(0.4, desc="Step 3/3: Running EgoX inference (~30 min)...")
        output_video = run_inference(output_dir, seed, use_gga, cos_sim_scale)

        progress(1.0, desc="Complete!")

        return ego_prior_path, output_video, f"Success! Output saved to: {output_video}"

    except Exception as e:
        return None, None, f"Error: {str(e)}"


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

                generate_btn = gr.Button("Generate Egocentric Video", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Output")

                with gr.Row():
                    ego_prior_output = gr.Video(label="Generated Ego Prior")
                    ego_video_output = gr.Video(label="Final Egocentric Video")

                output_status = gr.Textbox(label="Status", interactive=False, lines=5)

        # Event handlers
        refresh_btn.click(
            fn=check_model_status,
            outputs=model_status
        )

        generate_btn.click(
            fn=process_video,
            inputs=[video_input, seed_input, use_gga, cos_sim_scale],
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
