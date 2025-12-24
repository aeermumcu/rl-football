#!/bin/bash
# ============================================================
# RL Football - Google Cloud VM Setup Script
# ============================================================
# Run this ONCE to set up the VM after SSH'ing in
# 
# Usage: bash setup_gcp.sh
# ============================================================

set -e

echo "=============================================="
echo "🚀 Setting up RL Football Training Environment"
echo "=============================================="

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv git tmux htop

# Clone repo
echo "📥 Cloning repository..."
if [ ! -d "rl-football" ]; then
    git clone https://github.com/aeermumcu/rl-football.git
fi
cd rl-football

# Create virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "📦 Installing TensorFlow and dependencies..."
pip install --quiet --upgrade pip
pip install --quiet tensorflow numpy

# Verify GPU
echo "🔍 Checking GPU..."
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs found: {len(gpus)}'); [print(f'  - {g}') for g in gpus]"

echo ""
echo "=============================================="
echo "✅ Setup complete!"
echo "=============================================="
echo ""
echo "To start training, run:"
echo "  cd rl-football && bash train_gcp.sh"
echo ""
