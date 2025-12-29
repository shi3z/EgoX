#!/usr/bin/env python3
"""
EgoX Flask WebUI - Full pipeline with EgoPriorRenderer integration
Background worker processes with real-time progress tracking
"""

import os
import sys
import json
import uuid
import shutil
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/shi3z/git/EgoX/webui_uploads'
app.config['OUTPUT_FOLDER'] = '/home/shi3z/git/EgoX/webui_output'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Global job storage
jobs = {}
jobs_lock = threading.Lock()

def get_job(job_id):
    with jobs_lock:
        return jobs.get(job_id, {}).copy()

def update_job(job_id, **kwargs):
    with jobs_lock:
        if job_id not in jobs:
            jobs[job_id] = {}
        jobs[job_id].update(kwargs)
        jobs[job_id]['updated_at'] = datetime.now().isoformat()

def run_command(cmd, job_id, step_name, cwd=None, env=None):
    """Run a command and capture output with progress updates"""
    update_job(job_id, current_step=step_name, step_status='running')

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        env=full_env,
        text=True,
        bufsize=1
    )

    output_lines = []
    for line in iter(process.stdout.readline, ''):
        output_lines.append(line.rstrip())
        # Parse progress from output
        if '%|' in line or 'it/s]' in line:
            # Extract percentage from tqdm output
            try:
                if '%|' in line:
                    pct = int(line.split('%|')[0].strip().split()[-1])
                    update_job(job_id, step_progress=pct, last_output=line.rstrip())
            except:
                pass
        update_job(job_id, last_output=line.rstrip())

    process.wait()

    if process.returncode != 0:
        update_job(job_id, step_status='error', error='\n'.join(output_lines[-20:]))
        return False, '\n'.join(output_lines)

    update_job(job_id, step_status='completed', step_progress=100)
    return True, '\n'.join(output_lines)


def worker_process(job_id, video_path, prompt, num_frames=49, use_gga=False):
    """Main worker process - runs the full EgoX pipeline"""
    try:
        job_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        video_name = "EgoX"
        videos_dir = job_dir / 'videos' / video_name
        videos_dir.mkdir(parents=True, exist_ok=True)

        update_job(job_id,
                   status='processing',
                   total_steps=5,
                   current_step_num=0,
                   steps=['Video準備', 'ViPE推論', 'Ego Prior生成', 'Meta.json作成', 'EgoX推論'])

        # Step 1: Prepare video (resize to 448 height)
        update_job(job_id, current_step_num=1, current_step='Video準備', step_progress=0)

        exo_path = videos_dir / 'exo.mp4'
        # Convert paths for container
        video_path_container = f'/workspace/EgoX/{Path(video_path).relative_to("/home/shi3z/git/EgoX")}'
        exo_path_container = f'/workspace/EgoX/{exo_path.relative_to("/home/shi3z/git/EgoX")}'

        cmd = [
            'docker', 'exec', 'egox-egox-webui-1',
            'ffmpeg', '-y', '-i', video_path_container,
            '-vf', 'scale=-2:448',
            '-frames:v', str(num_frames),
            '-c:v', 'libx264', '-preset', 'fast',
            '-an', exo_path_container
        ]
        success, output = run_command(cmd, job_id, 'Video準備')
        if not success:
            update_job(job_id, status='error', error=f'Video準備に失敗: {output}')
            return

        update_job(job_id, step_progress=100)

        # Step 2: ViPE inference (depth maps and camera poses)
        update_job(job_id, current_step_num=2, current_step='ViPE推論', step_progress=0)

        vipe_output = Path('/home/shi3z/git/EgoX/EgoX-EgoPriorRenderer/vipe_results') / video_name
        vipe_output.mkdir(parents=True, exist_ok=True)

        # Run ViPE inside Docker container
        exo_video_container = f'/workspace/EgoX/{videos_dir.relative_to("/home/shi3z/git/EgoX")}/exo.mp4'
        vipe_output_container = f'/workspace/EgoX/EgoX-EgoPriorRenderer/vipe_results/{video_name}'
        docker_cmd = [
            'docker', 'exec', 'egox-egox-webui-1',
            'bash', '-c',
            f'''cd /workspace/EgoX/EgoX-EgoPriorRenderer && \
            vipe infer "{exo_video_container}" \
                -o "{vipe_output_container}" \
                --start_frame 0 \
                --end_frame {num_frames - 1} \
                --assume_fixed_camera_pose \
                --pipeline lyra'''
        ]
        success, output = run_command(docker_cmd, job_id, 'ViPE推論')
        if not success:
            update_job(job_id, status='error', error=f'ViPE推論に失敗: {output}')
            return

        # Step 3: Render ego prior from point cloud
        update_job(job_id, current_step_num=3, current_step='Ego Prior生成', step_progress=0)

        # Create meta.json for rendering
        meta_for_render = create_meta_json(job_dir, video_name, prompt, num_frames, vipe_output)

        render_cmd = [
            'docker', 'exec', 'egox-egox-webui-1',
            'python', '/workspace/EgoX/EgoX-EgoPriorRenderer/vipe/render_vipe_pointcloud.py',
            '--meta_file', f'/workspace/EgoX/{job_dir.relative_to("/home/shi3z/git/EgoX")}/meta.json',
            '--vipe_results_root', '/workspace/EgoX/EgoX-EgoPriorRenderer/vipe_results',
            '--output_height', '448',
            '--output_width', '448'
        ]
        success, output = run_command(render_cmd, job_id, 'Ego Prior生成')
        if not success:
            update_job(job_id, status='error', error=f'Ego Prior生成に失敗: {output}')
            return

        # Step 4: Prepare depth maps and meta.json
        update_job(job_id, current_step_num=4, current_step='Meta.json作成', step_progress=0)

        # Convert depth maps from .exr to .npy
        depth_output = Path('/home/shi3z/git/EgoX/depth_maps') / video_name
        depth_output.mkdir(parents=True, exist_ok=True)

        convert_cmd = [
            'docker', 'exec', 'egox-egox-webui-1',
            'python', '-c', f'''
import numpy as np
import cv2
from pathlib import Path

depth_src = Path("/workspace/EgoX/EgoX-EgoPriorRenderer/vipe_results/{video_name}/depth")
depth_dst = Path("/workspace/EgoX/depth_maps/{video_name}")
depth_dst.mkdir(parents=True, exist_ok=True)

for i, exr_file in enumerate(sorted(depth_src.glob("*.exr"))):
    depth = cv2.imread(str(exr_file), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is not None:
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        npy_file = depth_dst / f"{{i:06d}}.npy"
        np.save(str(npy_file), depth.astype(np.float32))
        print(f"Converted {{exr_file.name}} -> {{npy_file.name}}")
'''
        ]
        success, output = run_command(convert_cmd, job_id, 'depth変換')
        if not success:
            update_job(job_id, status='error', error=f'depth変換に失敗: {output}')
            return

        update_job(job_id, step_progress=100)

        # Step 5: Run EgoX inference
        update_job(job_id, current_step_num=5, current_step='EgoX推論', step_progress=0)

        results_dir = job_dir / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)

        infer_cmd = [
            'docker', 'exec', 'egox-egox-webui-1',
            'python', '-u', '/workspace/EgoX/infer.py',
            '--model_path', '/workspace/EgoX/checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers',
            '--lora_path', '/workspace/EgoX/checkpoints/EgoX/pytorch_lora_weights.safetensors',
            '--meta_data_file', f'/workspace/EgoX/{job_dir.relative_to("/home/shi3z/git/EgoX")}/meta.json',
            '--out', f'/workspace/EgoX/{results_dir.relative_to("/home/shi3z/git/EgoX")}',
            '--seed', '42',
            '--in_the_wild'
        ]

        if use_gga:
            infer_cmd.append('--use_GGA')

        success, output = run_command(infer_cmd, job_id, 'EgoX推論')
        if not success:
            update_job(job_id, status='error', error=f'EgoX推論に失敗: {output}')
            return

        # Find output video
        output_video = results_dir / f'{video_name}.mp4'
        if output_video.exists():
            update_job(job_id,
                       status='completed',
                       output_video=str(output_video),
                       completed_at=datetime.now().isoformat())
        else:
            update_job(job_id, status='error', error='出力ビデオが見つかりません')

    except Exception as e:
        import traceback
        update_job(job_id, status='error', error=f'{str(e)}\n{traceback.format_exc()}')


def create_meta_json(job_dir, video_name, prompt, num_frames, vipe_output):
    """Create meta.json with camera parameters from ViPE results"""
    import numpy as np

    videos_dir = job_dir / 'videos' / video_name

    # Load ViPE results
    intrinsics_path = vipe_output / 'intrinsics' / f'{video_name}.npz'
    pose_path = vipe_output / 'pose' / f'{video_name}.npz'

    # Default intrinsics if ViPE results not found
    fx, fy, cx, cy = 300.0, 300.0, 224.0, 224.0

    if intrinsics_path.exists():
        try:
            intrinsics_data = np.load(str(intrinsics_path))
            fx, fy, cx, cy = intrinsics_data['data'][0]
        except:
            pass

    # Scale intrinsics to 448 height
    scale = 448 / 1080  # Assume original is 1080p
    camera_intrinsics = [
        [float(fx * scale), 0.0, float(cx * scale)],
        [0.0, float(fy * scale), float(cy * scale)],
        [0.0, 0.0, 1.0]
    ]

    # Identity extrinsics for exo camera
    camera_extrinsics = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ]

    # Ego intrinsics (448x448)
    ego_intrinsics = [
        [150.0, 0.0, 224.0],
        [0.0, 150.0, 224.0],
        [0.0, 0.0, 1.0]
    ]

    # Generate ego_extrinsics
    ego_extrinsics = []
    for i in range(num_frames):
        angle = 0.3 + 0.02 * np.sin(i * 0.2)
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)]
        ])
        tx = 0.0 + 0.02 * np.sin(i * 0.15)
        ty = 0.3 + 0.01 * np.sin(i * 0.1)
        tz = 1.5

        extrinsic = [
            [float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(tx)],
            [float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(ty)],
            [float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(tz)]
        ]
        ego_extrinsics.append(extrinsic)

    meta_data = {
        "test_datasets": [
            {
                "exo_path": f"/workspace/EgoX/{videos_dir.relative_to('/home/shi3z/git/EgoX')}/exo.mp4",
                "ego_prior_path": f"/workspace/EgoX/{videos_dir.relative_to('/home/shi3z/git/EgoX')}/ego_Prior.mp4",
                "prompt": prompt,
                "camera_intrinsics": camera_intrinsics,
                "camera_extrinsics": camera_extrinsics,
                "ego_intrinsics": ego_intrinsics,
                "ego_extrinsics": ego_extrinsics
            }
        ]
    }

    meta_path = job_dir / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)

    return meta_data


# HTML Templates
INDEX_HTML = '''
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EgoX WebUI</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d4ff; text-align: center; }
        .card {
            background: #16213e;
            border-radius: 12px;
            padding: 24px;
            margin: 20px 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        label { display: block; margin: 12px 0 6px; color: #aaa; }
        input[type="file"], textarea, input[type="number"] {
            width: 100%;
            padding: 12px;
            border: 1px solid #333;
            border-radius: 8px;
            background: #0f0f23;
            color: #fff;
            font-size: 14px;
        }
        textarea { min-height: 150px; resize: vertical; }
        button {
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            color: #000;
            border: none;
            padding: 14px 32px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
            transition: transform 0.2s;
        }
        button:hover { transform: scale(1.02); }
        button:disabled { background: #666; cursor: not-allowed; transform: none; }
        .checkbox-group { display: flex; align-items: center; gap: 10px; margin: 12px 0; }
        .checkbox-group input { width: auto; }
        .info { color: #888; font-size: 12px; margin-top: 4px; }
        .jobs-list { margin-top: 30px; }
        .job-item {
            background: #0f0f23;
            border-radius: 8px;
            padding: 16px;
            margin: 10px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .job-status { padding: 4px 12px; border-radius: 20px; font-size: 12px; }
        .status-processing { background: #ff9800; color: #000; }
        .status-completed { background: #4caf50; color: #fff; }
        .status-error { background: #f44336; color: #fff; }
        .status-pending { background: #666; color: #fff; }
        a { color: #00d4ff; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>EgoX WebUI</h1>
    <p style="text-align:center;color:#888;">Exocentric → Egocentric Video Generation</p>

    <div class="card">
        <form action="/upload" method="post" enctype="multipart/form-data" id="uploadForm">
            <label>動画ファイル (MP4, MOV)</label>
            <input type="file" name="video" accept="video/*" required>
            <p class="info">最大500MB、49フレーム使用</p>

            <label>プロンプト (シーン説明)</label>
            <textarea name="prompt" placeholder="[Exo view]&#10;Scene Overview:&#10;...&#10;&#10;[Ego view]&#10;Scene Overview:&#10;..."></textarea>
            <p class="info">Exo視点とEgo視点の両方の説明を含めてください</p>

            <label>フレーム数</label>
            <input type="number" name="num_frames" value="49" min="13" max="97">

            <div class="checkbox-group">
                <input type="checkbox" name="use_gga" id="use_gga">
                <label for="use_gga" style="margin:0;">GGA使用 (高品質、メモリ多め)</label>
            </div>

            <button type="submit">処理開始</button>
        </form>
    </div>

    <div class="jobs-list">
        <h2>処理一覧</h2>
        <div id="jobsList"></div>
    </div>

    <script>
        function loadJobs() {
            fetch('/api/jobs')
                .then(r => r.json())
                .then(data => {
                    const list = document.getElementById('jobsList');
                    if (data.jobs.length === 0) {
                        list.innerHTML = '<p style="color:#666;">処理中のジョブはありません</p>';
                        return;
                    }
                    list.innerHTML = data.jobs.map(job => `
                        <div class="job-item">
                            <div>
                                <strong>${job.id.substring(0,8)}...</strong>
                                <br><small style="color:#888;">${job.created_at || ''}</small>
                            </div>
                            <div>
                                <span class="job-status status-${job.status}">${job.status}</span>
                            </div>
                            <div>
                                <a href="/job/${job.id}">詳細</a>
                            </div>
                        </div>
                    `).join('');
                });
        }
        loadJobs();
        setInterval(loadJobs, 5000);
    </script>
</body>
</html>
'''

JOB_HTML = '''
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Job {{ job_id[:8] }} - EgoX WebUI</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d4ff; }
        .card {
            background: #16213e;
            border-radius: 12px;
            padding: 24px;
            margin: 20px 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .progress-container {
            background: #0f0f23;
            border-radius: 8px;
            padding: 20px;
            margin: 16px 0;
        }
        .step {
            display: flex;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #333;
        }
        .step:last-child { border-bottom: none; }
        .step-num {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: #333;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 16px;
            font-weight: bold;
        }
        .step-num.active { background: #ff9800; color: #000; }
        .step-num.completed { background: #4caf50; }
        .step-name { flex: 1; }
        .step-progress {
            width: 100px;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
        }
        .step-progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #0099cc);
            transition: width 0.3s;
        }
        .output-log {
            background: #0a0a14;
            border-radius: 8px;
            padding: 16px;
            font-family: monospace;
            font-size: 12px;
            max-height: 200px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .status-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: bold;
        }
        .status-processing { background: #ff9800; color: #000; }
        .status-completed { background: #4caf50; color: #fff; }
        .status-error { background: #f44336; color: #fff; }
        .status-pending { background: #666; color: #fff; }
        video {
            width: 100%;
            border-radius: 8px;
            margin-top: 16px;
        }
        a { color: #00d4ff; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .error-box {
            background: #3d1515;
            border: 1px solid #f44336;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
        }
        .back-link { margin-bottom: 20px; display: block; }
    </style>
</head>
<body>
    <a href="/" class="back-link">← 戻る</a>
    <h1>Job: {{ job_id[:8] }}...</h1>

    <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span class="status-badge status-{{ status }}" id="statusBadge">{{ status }}</span>
            <span style="color:#888;" id="updatedAt"></span>
        </div>

        <div class="progress-container" id="progressContainer">
            <!-- Steps will be rendered by JS -->
        </div>

        <h3>出力ログ</h3>
        <div class="output-log" id="outputLog">待機中...</div>

        <div id="errorBox" class="error-box" style="display:none;">
            <strong>エラー:</strong>
            <pre id="errorText"></pre>
        </div>

        <div id="resultSection" style="display:none;">
            <h3>生成結果</h3>
            <video id="resultVideo" controls></video>
            <p><a id="downloadLink" href="#">ダウンロード</a></p>
        </div>
    </div>

    <script>
        const jobId = '{{ job_id }}';
        let lastStatus = '';

        function updateUI(data) {
            // Status badge
            const badge = document.getElementById('statusBadge');
            badge.textContent = data.status || 'pending';
            badge.className = 'status-badge status-' + (data.status || 'pending');

            // Updated time
            if (data.updated_at) {
                document.getElementById('updatedAt').textContent = '更新: ' + data.updated_at;
            }

            // Progress steps
            const steps = data.steps || [];
            const currentStepNum = data.current_step_num || 0;
            const stepProgress = data.step_progress || 0;

            let stepsHtml = '';
            steps.forEach((step, i) => {
                const num = i + 1;
                let stepClass = '';
                let progress = 0;

                if (num < currentStepNum) {
                    stepClass = 'completed';
                    progress = 100;
                } else if (num === currentStepNum) {
                    stepClass = 'active';
                    progress = stepProgress;
                }

                stepsHtml += `
                    <div class="step">
                        <div class="step-num ${stepClass}">${num}</div>
                        <div class="step-name">${step}</div>
                        <div class="step-progress">
                            <div class="step-progress-bar" style="width:${progress}%"></div>
                        </div>
                    </div>
                `;
            });
            document.getElementById('progressContainer').innerHTML = stepsHtml;

            // Output log
            if (data.last_output) {
                document.getElementById('outputLog').textContent = data.last_output;
            }

            // Error
            if (data.error) {
                document.getElementById('errorBox').style.display = 'block';
                document.getElementById('errorText').textContent = data.error;
            }

            // Result
            if (data.status === 'completed' && data.output_video) {
                document.getElementById('resultSection').style.display = 'block';
                document.getElementById('resultVideo').src = '/video/' + jobId;
                document.getElementById('downloadLink').href = '/download/' + jobId;
            }

            lastStatus = data.status;
        }

        function poll() {
            fetch('/api/job/' + jobId)
                .then(r => r.json())
                .then(data => {
                    updateUI(data);
                    if (data.status === 'processing' || data.status === 'pending') {
                        setTimeout(poll, 1000);
                    }
                })
                .catch(err => {
                    console.error(err);
                    setTimeout(poll, 3000);
                });
        }

        poll();
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return INDEX_HTML


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Create job
    job_id = str(uuid.uuid4())
    upload_dir = Path(app.config['UPLOAD_FOLDER'])
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    filename = secure_filename(file.filename)
    video_path = upload_dir / f"{job_id}_{filename}"
    file.save(str(video_path))

    # Get parameters
    prompt = request.form.get('prompt', '')
    num_frames = int(request.form.get('num_frames', 49))
    use_gga = 'use_gga' in request.form

    # Initialize job
    update_job(job_id,
               status='pending',
               created_at=datetime.now().isoformat(),
               video_path=str(video_path),
               prompt=prompt,
               num_frames=num_frames,
               use_gga=use_gga)

    # Start worker thread
    thread = threading.Thread(
        target=worker_process,
        args=(job_id, video_path, prompt, num_frames, use_gga)
    )
    thread.daemon = True
    thread.start()

    return redirect(url_for('job_page', job_id=job_id))


@app.route('/job/<job_id>')
def job_page(job_id):
    job = get_job(job_id)
    if not job:
        return "Job not found", 404

    from flask import render_template_string
    return render_template_string(JOB_HTML, job_id=job_id, status=job.get('status', 'pending'))


@app.route('/api/job/<job_id>')
def api_job(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


@app.route('/api/jobs')
def api_jobs():
    with jobs_lock:
        job_list = [
            {'id': k, 'status': v.get('status', 'pending'), 'created_at': v.get('created_at', '')}
            for k, v in jobs.items()
        ]
    job_list.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify({'jobs': job_list[:20]})


@app.route('/video/<job_id>')
def serve_video(job_id):
    job = get_job(job_id)
    if not job or 'output_video' not in job:
        return "Video not found", 404
    return send_file(job['output_video'], mimetype='video/mp4')


@app.route('/download/<job_id>')
def download_video(job_id):
    job = get_job(job_id)
    if not job or 'output_video' not in job:
        return "Video not found", 404
    return send_file(job['output_video'], as_attachment=True, download_name=f'egox_{job_id[:8]}.mp4')


if __name__ == '__main__':
    # Ensure directories exist
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    Path(app.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7861, help='Port to run on')
    args = parser.parse_args()

    print("Starting EgoX Flask WebUI...")
    print(f"Open http://localhost:{args.port} in your browser")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
