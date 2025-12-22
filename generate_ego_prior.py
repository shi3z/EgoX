#!/usr/bin/env python3
"""
Ego Prior Video Generator for EgoX

This script generates Ego Prior videos from Exocentric videos.
Ego Prior = 3D point cloud rendered from egocentric (first-person) viewpoint

The ego camera should be INSIDE the scene, at the position where a person
would be looking from (e.g., the subject's eye level).

Pipeline:
1. Extract frames from Exo video
2. Estimate depth maps (Depth Anything V2)
3. Build 3D point cloud from depth + camera params
4. Render from Ego camera trajectory (inside the scene)
5. Save as Ego Prior video

Usage:
    python generate_ego_prior.py --exo_video path/to/exo.mp4 --output_dir output/
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from typing import Optional, Tuple, List
from tqdm import tqdm


def extract_frames(video_path: str, num_frames: int = 49) -> List[np.ndarray]:
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames > num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = list(range(total_frames))

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames[:num_frames]


def estimate_depth_dav2(frames: List[np.ndarray], device: str = "cuda") -> List[np.ndarray]:
    """Estimate depth using Depth Anything V2"""
    try:
        from transformers import pipeline

        print("Loading Depth Anything V2...")
        depth_estimator = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Large-hf",
            device=0 if device == "cuda" else -1
        )

        depth_maps = []
        for frame in tqdm(frames, desc="Estimating depth"):
            from PIL import Image
            pil_image = Image.fromarray(frame)
            result = depth_estimator(pil_image)
            depth = np.array(result["depth"])
            depth = depth.astype(np.float32)

            # Convert to metric depth (approximate scale)
            # Depth Anything outputs relative depth, scale to reasonable range
            depth_min, depth_max = depth.min(), depth.max()
            depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)
            depth = depth * 8.0 + 2.0  # Scale to 2-10m range

            depth_maps.append(depth)

        return depth_maps

    except ImportError:
        print("Depth Anything V2 not available. Using MiDaS fallback...")
        return estimate_depth_midas(frames, device)


def estimate_depth_midas(frames: List[np.ndarray], device: str = "cuda") -> List[np.ndarray]:
    """Fallback: Estimate depth using MiDaS"""
    print("Loading MiDaS...")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    midas = midas.to(device).eval()

    depth_maps = []
    for frame in tqdm(frames, desc="Estimating depth (MiDaS)"):
        input_batch = transform(frame).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        depth = 1.0 / (depth + 1e-6)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 8.0 + 2.0

        depth_maps.append(depth.astype(np.float32))

    return depth_maps


def create_exo_camera_intrinsics(width: int, height: int) -> np.ndarray:
    """Create camera intrinsics for exo camera (standard lens)"""
    # Standard ~60 degree FOV
    fov = 60.0
    fx = width / (2.0 * np.tan(np.radians(fov / 2)))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)


def create_ego_camera_intrinsics(width: int, height: int) -> np.ndarray:
    """Create camera intrinsics for ego camera (wide-angle/fisheye)"""
    # Wide FOV like fisheye lens (similar to original: fx=fy=150 for 512x512)
    # This gives roughly 120-140 degree FOV
    fx = 150.0 * (width / 512.0)
    fy = fx
    cx = width / 2.0 + 0.5
    cy = height / 2.0 + 0.5

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Create rotation matrix that rotates vec1 to vec2"""
    a = vec1 / (np.linalg.norm(vec1) + 1e-8)
    b = vec2 / (np.linalg.norm(vec2) + 1e-8)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.linalg.norm(v) < 1e-8:
        if c > 0:
            return np.eye(3)
        else:
            # 180 degree rotation
            return -np.eye(3)

    s = np.linalg.norm(v)
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s * s + 1e-8))
    return R


def create_ego_camera_trajectory(
    num_frames: int,
    depth_maps: List[np.ndarray],
    exo_intrinsics: np.ndarray,
    trajectory_type: str = "center_look",
    ego_depth_ratio: float = 0.6,  # How deep into the scene (0=near, 1=far)
    camera_position: str = "center",  # center, left, right
    look_direction: str = "front",  # front, left, right, up, down
) -> List[np.ndarray]:
    """
    Create ego camera trajectory INSIDE the scene.

    The ego camera should be positioned where a person would be looking from,
    typically in the middle-depth of the scene.

    Args:
        num_frames: Number of frames
        depth_maps: Depth maps to estimate scene geometry
        exo_intrinsics: Exo camera intrinsics
        trajectory_type: Type of camera motion
        ego_depth_ratio: Where in depth to place ego camera (0=near, 1=far)
        camera_position: Where to place camera horizontally (center/left/right)
        look_direction: Which direction to look (front/left/right/up/down)
    """
    h, w = depth_maps[0].shape
    fx, fy = exo_intrinsics[0, 0], exo_intrinsics[1, 1]
    cx, cy = exo_intrinsics[0, 2], exo_intrinsics[1, 2]

    extrinsics = []

    # Determine horizontal offset based on camera_position
    if camera_position == "left":
        h_offset_ratio = 0.25  # Left quarter
    elif camera_position == "right":
        h_offset_ratio = 0.75  # Right quarter
    else:  # center
        h_offset_ratio = 0.5

    for i in range(num_frames):
        t = i / (num_frames - 1)  # 0 to 1
        depth = depth_maps[i]

        # Estimate ego position from depth
        # Select region based on camera_position
        if camera_position == "left":
            region = depth[h//4:3*h//4, :w//2]
        elif camera_position == "right":
            region = depth[h//4:3*h//4, w//2:]
        else:
            region = depth[h//4:3*h//4, w//4:3*w//4]

        # Sort depths and pick at ego_depth_ratio percentile
        sorted_depths = np.sort(region.flatten())
        target_depth = sorted_depths[int(len(sorted_depths) * ego_depth_ratio)]

        # Ego camera position in world coordinates
        ego_u = int(w * h_offset_ratio)
        ego_v = h // 2
        ego_z = target_depth
        ego_x = (ego_u - cx) * ego_z / fx
        ego_y = (ego_v - cy) * ego_z / fy

        # Add some vertical offset (eye level, slightly above center)
        ego_y -= 0.3  # 30cm above center

        # Ego camera position
        position = np.array([ego_x, ego_y, ego_z])

        # Camera orientation based on look_direction
        if look_direction == "left":
            base_angle = -np.pi / 3  # -60 degrees
        elif look_direction == "right":
            base_angle = np.pi / 3  # +60 degrees
        elif look_direction == "up":
            base_angle = 0
            # Will adjust y component below
        elif look_direction == "down":
            base_angle = 0
            # Will adjust y component below
        else:  # front
            base_angle = 0

        # Camera orientation - look outward from center, with some rotation
        if trajectory_type == "center_look":
            # Look toward the edges of the scene, rotating over time
            angle = base_angle + t * np.pi * 0.3 - np.pi * 0.15  # +/- 27 degrees sweep
            look_dir = np.array([np.sin(angle), 0, np.cos(angle)])
        elif trajectory_type == "forward_look":
            # Always look in the specified direction
            look_dir = np.array([np.sin(base_angle), 0, np.cos(base_angle)])
        elif trajectory_type == "scan":
            # Scan left to right from base direction
            angle = base_angle + (t - 0.5) * np.pi * 0.8  # -72 to +72 degrees from base
            look_dir = np.array([np.sin(angle), 0, np.cos(angle)])
        else:
            look_dir = np.array([np.sin(base_angle), 0, np.cos(base_angle)])

        # Adjust for up/down looking
        if look_direction == "up":
            look_dir[1] = -0.5  # Look upward
            look_dir = look_dir / np.linalg.norm(look_dir)
        elif look_direction == "down":
            look_dir[1] = 0.5  # Look downward
            look_dir = look_dir / np.linalg.norm(look_dir)

        # Build rotation matrix (camera looks along -Z in camera space)
        forward = -look_dir / (np.linalg.norm(look_dir) + 1e-8)
        up = np.array([0, 1, 0])  # Y-up convention (standard)
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)

        # Rotation matrix (world to camera)
        R = np.stack([right, up, forward], axis=0)  # 3x3, rows are camera axes

        # Translation (world to camera)
        t_vec = -R @ position

        # Extrinsic matrix (world to camera, 3x4)
        ext = np.zeros((3, 4), dtype=np.float32)
        ext[:3, :3] = R
        ext[:3, 3] = t_vec

        extrinsics.append(ext)

    return extrinsics


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray = None
) -> np.ndarray:
    """Convert depth map to 3D point cloud"""
    h, w = depth.shape

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_cam = np.stack([x, y, z], axis=-1)

    if extrinsics is not None:
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        # Camera to world: p_world = R^T @ (p_cam - t) = R^T @ p_cam - R^T @ t
        # But extrinsics is typically world-to-camera, so inverse:
        R_inv = R.T
        t_inv = -R.T @ t
        points_world = points_cam @ R_inv.T + t_inv
        return points_world

    return points_cam


def render_pointcloud_zbuffer(
    points_world: np.ndarray,
    colors: np.ndarray,
    ego_intrinsics: np.ndarray,
    ego_extrinsics: np.ndarray,
    output_size: Tuple[int, int] = (448, 448)
) -> np.ndarray:
    """Render point cloud from ego camera viewpoint using z-buffer"""
    h_out, w_out = output_size

    points = points_world.reshape(-1, 3)
    colors = colors.reshape(-1, 3)

    # World to camera transform
    R = ego_extrinsics[:3, :3]
    t = ego_extrinsics[:3, 3]
    points_cam = points @ R.T + t

    # Filter points behind camera
    valid = points_cam[:, 2] > 0.1
    points_cam = points_cam[valid]
    colors_valid = colors[valid]

    if len(points_cam) == 0:
        return np.zeros((h_out, w_out, 3), dtype=np.uint8)

    # Project to image plane
    fx, fy = ego_intrinsics[0, 0], ego_intrinsics[1, 1]
    cx, cy = ego_intrinsics[0, 2], ego_intrinsics[1, 2]

    u = (points_cam[:, 0] * fx / points_cam[:, 2] + cx).astype(np.int32)
    v = (points_cam[:, 1] * fy / points_cam[:, 2] + cy).astype(np.int32)
    z = points_cam[:, 2]

    # Filter points outside image
    valid = (u >= 0) & (u < w_out) & (v >= 0) & (v < h_out)
    u, v, z = u[valid], v[valid], z[valid]
    colors_valid = colors_valid[valid]

    if len(u) == 0:
        return np.zeros((h_out, w_out, 3), dtype=np.uint8)

    # Z-buffer rendering (sort far to near)
    image = np.zeros((h_out, w_out, 3), dtype=np.uint8)
    zbuffer = np.full((h_out, w_out), np.inf, dtype=np.float32)

    order = np.argsort(-z)
    for idx in order:
        if z[idx] < zbuffer[v[idx], u[idx]]:
            zbuffer[v[idx], u[idx]] = z[idx]
            image[v[idx], u[idx]] = colors_valid[idx]

    # Hole filling with dilation
    kernel = np.ones((3, 3), np.uint8)
    mask = (image.sum(axis=-1) == 0).astype(np.uint8)
    for _ in range(7):  # More iterations for better filling
        dilated = cv2.dilate(image, kernel, iterations=1)
        image = np.where(mask[:, :, np.newaxis], dilated, image)
        new_mask = (image.sum(axis=-1) == 0).astype(np.uint8)
        if np.sum(new_mask) == np.sum(mask):
            break
        mask = new_mask

    return image


def generate_ego_prior(
    exo_video_path: str,
    output_dir: str,
    trajectory_type: str = "center_look",
    num_frames: int = 49,
    output_size: Tuple[int, int] = (448, 448),
    device: str = "cuda",
    save_depth: bool = True,
    save_camera_params: bool = True,
    ego_depth_ratio: float = 0.5,
    camera_position: str = "center",  # center, left, right
    look_direction: str = "front",  # front, left, right, up, down
) -> str:
    """
    Generate Ego Prior video from Exo video
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {exo_video_path}")

    # Step 1: Extract frames
    print("Step 1: Extracting frames...")
    frames = extract_frames(exo_video_path, num_frames)
    h, w = frames[0].shape[:2]
    print(f"  Extracted {len(frames)} frames, size: {w}x{h}")

    # Step 2: Estimate depth
    print("Step 2: Estimating depth...")
    depth_maps = estimate_depth_dav2(frames, device)

    if save_depth:
        depth_dir = output_dir / "depth_maps"
        depth_dir.mkdir(exist_ok=True)
        for i, dm in enumerate(depth_maps):
            dm_resized = cv2.resize(dm, (w, h))
            np.save(depth_dir / f"{i:05d}.npy", dm_resized)
        print(f"  Saved depth maps to {depth_dir}")

    # Step 3: Setup camera parameters
    print("Step 3: Setting up camera parameters...")

    exo_intrinsics = create_exo_camera_intrinsics(w, h)
    exo_extrinsics = np.eye(4, dtype=np.float32)[:3, :]  # Identity

    # Create ego trajectory INSIDE the scene
    ego_trajectory = create_ego_camera_trajectory(
        num_frames,
        depth_maps,
        exo_intrinsics,
        trajectory_type=trajectory_type,
        ego_depth_ratio=ego_depth_ratio,
        camera_position=camera_position,
        look_direction=look_direction,
    )

    ego_intrinsics = create_ego_camera_intrinsics(output_size[1], output_size[0])

    # Step 4: Generate Ego Prior frames
    print("Step 4: Rendering Ego Prior frames...")
    ego_frames = []

    for i, (frame, depth, ego_ext) in enumerate(tqdm(
        zip(frames, depth_maps, ego_trajectory),
        total=num_frames,
        desc="Rendering"
    )):
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h))

        # Point cloud in world coordinates (exo camera at origin)
        points_world = depth_to_pointcloud(depth, exo_intrinsics, None)

        # Render from ego viewpoint
        ego_frame = render_pointcloud_zbuffer(
            points_world,
            frame,
            ego_intrinsics,
            ego_ext,
            output_size
        )

        ego_frames.append(ego_frame)

    # Step 5: Save video
    print("Step 5: Saving Ego Prior video...")
    output_video_path = output_dir / "ego_Prior.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, 30, output_size)

    for frame in ego_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

    print(f"  Saved: {output_video_path}")

    # Step 6: Save camera parameters
    if save_camera_params:
        camera_params = {
            "test_datasets": [{
                "path": str(exo_video_path),
                "best_camera": "none",
                "source_frame_start": 0,
                "source_frame_end": num_frames,
                "camera_intrinsics": exo_intrinsics.tolist(),
                "camera_extrinsics": exo_extrinsics.tolist(),
                "ego_intrinsics": ego_intrinsics.tolist(),
                "ego_extrinsics": [ext.tolist() for ext in ego_trajectory]
            }]
        }

        params_path = output_dir / "camera_params.json"
        with open(params_path, 'w') as f:
            json.dump(camera_params, f, indent=2)
        print(f"  Saved: {params_path}")

    # Also copy exo video
    import shutil
    exo_copy_path = output_dir / "exo.mp4"
    shutil.copy(exo_video_path, exo_copy_path)
    print(f"  Copied exo video to: {exo_copy_path}")

    return str(output_video_path)


def main():
    parser = argparse.ArgumentParser(description="Generate Ego Prior video from Exo video")
    parser.add_argument("--exo_video", type=str, required=True, help="Path to exocentric video")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--trajectory", type=str, default="center_look",
                       choices=["center_look", "forward_look", "scan"],
                       help="Camera trajectory type")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames")
    parser.add_argument("--output_size", type=int, nargs=2, default=[448, 448],
                       help="Output size (height width)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--no_depth", action="store_true", help="Don't save depth maps")
    parser.add_argument("--no_camera_params", action="store_true", help="Don't save camera params")
    parser.add_argument("--ego_depth", type=float, default=0.5,
                       help="Where to place ego camera in depth (0=near, 1=far)")
    parser.add_argument("--camera_position", type=str, default="center",
                       choices=["center", "left", "right"],
                       help="Camera horizontal position")
    parser.add_argument("--look_direction", type=str, default="front",
                       choices=["front", "left", "right", "up", "down"],
                       help="Camera look direction")

    args = parser.parse_args()

    generate_ego_prior(
        exo_video_path=args.exo_video,
        output_dir=args.output_dir,
        trajectory_type=args.trajectory,
        num_frames=args.num_frames,
        output_size=tuple(args.output_size),
        device=args.device,
        save_depth=not args.no_depth,
        save_camera_params=not args.no_camera_params,
        ego_depth_ratio=args.ego_depth,
        camera_position=args.camera_position,
        look_direction=args.look_direction,
    )


if __name__ == "__main__":
    main()
