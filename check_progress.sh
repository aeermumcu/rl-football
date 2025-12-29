#!/bin/bash
# Check RL Football training progress on GCP VM

echo "CONNECTING TO VM TO CHECK TRAINING STATUS..."
echo "------------------------------------------------"

gcloud compute ssh rl-training-2-vm \
    --zone=us-east4-c \
    --tunnel-through-iap \
    --command="tmux capture-pane -pt training -S -20 && echo '------------------------------------------------'"
