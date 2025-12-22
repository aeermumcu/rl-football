#!/bin/bash
# Auto-restart training script with memory management
# Usage: ./train-loop.sh 10000

TARGET_EPISODES=${1:-10000}
MATCH_TIME=${2:-30}

echo "🔄 Auto-restart trainer - Target: $TARGET_EPISODES episodes"
echo "💾 Memory management enabled (GC exposed)"

while true; do
    # Find latest checkpoint
    LATEST=$(ls -t weights/weights-*.json 2>/dev/null | head -1)
    
    if [ -n "$LATEST" ]; then
        echo "📂 Resuming from $LATEST"
        caffeinate node --expose-gc train.js --resume="$LATEST" --episodes=$TARGET_EPISODES --match-time=$MATCH_TIME
    else
        echo "🆕 Starting fresh training"
        caffeinate node --expose-gc train.js --episodes=$TARGET_EPISODES --match-time=$MATCH_TIME
    fi
    
    EXIT_CODE=$?
    
    # Check if training completed successfully
    if [ -f "weights/weights-$TARGET_EPISODES.json" ]; then
        echo "✅ Training complete! Final weights saved."
        break
    fi
    
    echo ""
    echo "⚠️ Process exited (code: $EXIT_CODE). Restarting in 5 seconds..."
    echo ""
    sleep 5
done
