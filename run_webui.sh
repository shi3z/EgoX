#!/bin/bash
# EgoX WebUI Launch Script
# Access from VPN: http://<your-ip>:7860

echo "Starting EgoX WebUI..."
echo "Access URL: http://0.0.0.0:7860"
echo "From other devices on VPN, use your machine's IP address"

cd "$(dirname "$0")"

python webui.py --host 0.0.0.0 --port 7860
