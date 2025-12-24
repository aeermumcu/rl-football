#!/bin/bash
# ============================================================
# RL Football - Start Training in Background (tmux)
# ============================================================
# This runs training in tmux so it survives SSH disconnection
#
# Usage: bash train_gcp.sh
# ============================================================

EPISODES=${1:-100000}
PARALLEL=${2:-64}

echo "=============================================="
echo "🚀 Starting Training in Background"
echo "=============================================="
echo "Episodes: $EPISODES"
echo "Parallel envs: $PARALLEL"
echo ""

# Activate venv
source .venv/bin/activate

# Create weights directory
mkdir -p trainer/weights

# Kill existing session if any
tmux kill-session -t training 2>/dev/null || true

# Start training in tmux
tmux new-session -d -s training "
    cd $(pwd) && \
    source .venv/bin/activate && \
    python trainer/fast_trainer.py --episodes=$EPISODES --parallel-envs=$PARALLEL 2>&1 | tee training.log
"

echo "✅ Training started in background!"
echo ""
echo "Commands:"
echo "  tmux attach -t training    # View live progress"
echo "  Ctrl+B then D              # Detach (training continues)"
echo "  tail -f training.log       # View log file"
echo ""
echo "When done, weights will be in: trainer/weights/trained.json"
echo ""
echo "To download weights to your Mac:"
echo "  gcloud compute scp VM_NAME:~/rl-football/trainer/weights/trained.json ."
echo ""
