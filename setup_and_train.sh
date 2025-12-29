#!/bin/bash
set -e

echo "=== Installing TensorFlow ==="
pip install tensorflow

echo "=== Verifying GPU ==="
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

echo "=== Installing required packages ==="
pip install pygame gfootball

echo "=== Creating training directory ==="
mkdir -p ~/rl-football

echo "=== Setup complete, ready to run training ==="
