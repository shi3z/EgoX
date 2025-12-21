#!/usr/bin/env python3
"""
EgoX WebUI - Gradio-based web interface for EgoX video generation
Uses subprocess to run inference to avoid library compatibility issues
"""
import os
import subprocess
import json
from pathlib import Path
import gradio as gr

def get_available_examples():
    """Get list of available examples"""
    example_root = Path("./example/in_the_wild")
    if not example_root.exists():
        return []

    exo_path_file = example_root / "exo_path.txt"
    if not exo_path_file.exists():
        return []

    with open(exo_path_file, 'r') as f:
        paths = [line.strip() for line in f.readlines()]

    return [p.split('/')[-2] for p in paths if p.strip()]


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


def generate_video(
    example_name: str,
    seed: int,
    use_gga: bool,
    cos_sim_scaling_factor: float,
    progress=gr.Progress()
):
    """Generate egocentric video by calling infer.py"""

    # Check if models exist
    model_path = "./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers"
    lora_path = "./checkpoints/EgoX/pytorch_lora_weights.safetensors"

    if not os.path.exists(os.path.join(model_path, "transformer")):
        return None, "Error: Pretrained model not found. Please wait for download to complete."

    if not os.path.exists(lora_path):
        return None, "Error: EgoX LoRA weights not found. Please wait for download to complete."

    progress(0.1, desc="Starting inference...")

    # Get the index of the selected example
    examples = get_available_examples()
    if example_name not in examples:
        return None, f"Example '{example_name}' not found"

    idx = examples.index(example_name)

    # Build command
    cmd = [
        "python3", "infer.py",
        "--prompt", "./example/in_the_wild/caption.txt",
        "--exo_video_path", "./example/in_the_wild/exo_path.txt",
        "--ego_prior_video_path", "./example/in_the_wild/ego_prior_path.txt",
        "--meta_data_file", "./example/in_the_wild/camera_params.json",
        "--depth_root", "./example/in_the_wild/depth_maps/",
        "--model_path", model_path,
        "--lora_path", lora_path,
        "--lora_rank", "256",
        "--out", "./results",
        "--seed", str(seed),
        "--idx", str(idx),
        "--cos_sim_scaling_factor", str(cos_sim_scaling_factor),
        "--in_the_wild"
    ]

    if use_gga:
        cmd.append("--use_GGA")

    progress(0.2, desc=f"Running inference for {example_name}...")

    try:
        # Run inference
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/home/shi3z/git/EgoX"
        )

        if result.returncode != 0:
            return None, f"Error during inference:\n{result.stderr[-2000:]}"

        progress(0.9, desc="Inference complete!")

        # Find output video
        output_path = f"./results/{example_name}.mp4"
        if os.path.exists(output_path):
            return output_path, f"Video generated successfully!\nSaved to: {output_path}"
        else:
            return None, f"Inference completed but output video not found.\n\nStdout:\n{result.stdout[-1000:]}"

    except Exception as e:
        return None, f"Exception during inference: {str(e)}"


def preview_example(example_name: str):
    """Preview the exo and ego prior videos for an example"""
    if not example_name or example_name == "No examples found":
        return None, None, "No example selected"

    example_root = Path("./example/in_the_wild/videos") / example_name

    exo_video = example_root / "exo.mp4"
    ego_prior = example_root / "ego_Prior.mp4"

    exo_path = str(exo_video) if exo_video.exists() else None
    ego_path = str(ego_prior) if ego_prior.exists() else None

    info = f"Example: {example_name}\n"
    if exo_path:
        info += f"Exo video: Found\n"
    if ego_path:
        info += f"Ego prior: Found\n"

    return exo_path, ego_path, info


def create_ui():
    """Create the Gradio interface"""

    with gr.Blocks(title="EgoX - Egocentric Video Generation") as demo:
        gr.Markdown("""
        # EgoX: Egocentric Video Generation from a Single Exocentric Video

        Generate first-person (egocentric) videos from third-person (exocentric) video input.

        **Paper:** [arXiv:2512.08269](https://arxiv.org/abs/2512.08269) | **Project:** [EgoX Project Page](https://keh0t0.github.io/EgoX/)
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
                gr.Markdown("### Generation Settings")

                examples = get_available_examples()
                example_dropdown = gr.Dropdown(
                    choices=examples if examples else ["No examples found"],
                    value=examples[0] if examples else None,
                    label="Select Example"
                )

                preview_btn = gr.Button("Preview Input Videos", size="sm")

                seed_input = gr.Number(
                    value=846514,
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

                generate_btn = gr.Button("Generate Video", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Input Preview")
                with gr.Row():
                    exo_preview = gr.Video(label="Exocentric (3rd person) Video")
                    ego_prior_preview = gr.Video(label="Ego Prior Video")
                preview_info = gr.Textbox(label="Info", interactive=False, lines=3)

                gr.Markdown("### Output")
                output_video = gr.Video(label="Generated Egocentric Video")
                output_status = gr.Textbox(label="Status", interactive=False, lines=5)

        # Event handlers
        refresh_btn.click(
            fn=check_model_status,
            outputs=model_status
        )

        preview_btn.click(
            fn=preview_example,
            inputs=[example_dropdown],
            outputs=[exo_preview, ego_prior_preview, preview_info]
        )

        example_dropdown.change(
            fn=preview_example,
            inputs=[example_dropdown],
            outputs=[exo_preview, ego_prior_preview, preview_info]
        )

        generate_btn.click(
            fn=generate_video,
            inputs=[example_dropdown, seed_input, use_gga, cos_sim_scale],
            outputs=[output_video, output_status]
        )

        gr.Markdown("""
        ---
        ### Instructions
        1. Check that models are downloaded (status shows "Ready")
        2. Select an example from the dropdown (or preview input videos)
        3. Adjust parameters if needed (seed, GGA, scaling factor)
        4. Click "Generate Video" to start generation

        **Note:** Video generation requires ~80GB VRAM and takes several minutes.

        ### Available Examples
        - **joker**: Joker movie scene
        - **ironman**: Iron Man scene
        - **hulk_blackwidow**: Avengers scene
        - **tabletennis**: Table tennis at Paris 2024
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
