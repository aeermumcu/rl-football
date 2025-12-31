# RL Football Champions ⚽

A browser-based football game featuring an AI opponent trained using **Dueling Double DQN** reinforcement learning.

![Game Screenshot](https://img.shields.io/badge/Status-Experimental-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

## 🎮 Play Now

Simply open `index.html` in your browser to play against the AI!

**Game Modes:**
- **Play:** Control Blip (left player) with arrow keys + spacebar to kick
- **Watch:** Watch two AI agents compete against each other
- **Train:** Train the agents in-browser (slow, for demonstration only)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (index.html)                      │
├─────────────────────────────────────────────────────────────┤
│  game.js          │  dqn-agent.js      │  main.js           │
│  Game physics     │  Neural network    │  UI & game loop    │
│  Ball/player sim  │  TensorFlow.js     │  Training control  │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ Load weights
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Training (Python)                         │
├─────────────────────────────────────────────────────────────┤
│  trainer/fast_trainer.py                                     │
│  - NumPy-accelerated game simulation                         │
│  - TensorFlow/Keras Dueling Double DQN                       │
│  - Parallel environments (512x speedup)                      │
│  - Self-play training                                        │
└─────────────────────────────────────────────────────────────┘
```

## 🧠 AI Details

### Network Architecture
- **Type:** Dueling Double DQN
- **Input:** 12 features (positions of ball, players, goal)
- **Hidden Layers:** 256 → 256 → 128 neurons (shared), then separate value/advantage streams
- **Output:** 10 discrete actions (movement + kick combinations)

### Training Phases Completed
| Phase | Episodes | Description |
|-------|----------|-------------|
| 1 | 100,000 | Dense rewards vs SimpleAI opponent |
| 2 | 500,000 | Dense rewards + self-play |
| 3 | 100,000 | Sparse rewards (win/loss only) + self-play |

**Total: 700,000 episodes** trained on GCP with NVIDIA L4 GPU.

## 📁 Project Structure

```
rl-football/
├── index.html              # Main game page
├── styles.css              # Game styling
├── js/
│   ├── game.js             # Game physics and state
│   ├── dqn-agent.js        # Neural network agent (TF.js)
│   ├── main.js             # Game loop and UI
│   ├── player.js           # Player entity
│   └── visualizer.js       # Debug visualization
├── trainer/
│   ├── fast_trainer.py     # Main Python training script
│   ├── setup_gcp.sh        # GCP VM setup script
│   └── rl_football_colab.ipynb  # Colab notebook
├── weights/
│   ├── trained.json        # Current best weights (Phase 3)
│   └── trained_sparse.json # Phase 3 sparse reward weights
└── README.md
```

## 🚀 Training Your Own Agent

### Quick Start (Browser)
1. Open `index.html`
2. Select "Train" mode
3. Click "Start" — training is slow but demonstrates the concept

### Serious Training (Python + GPU)
```bash
cd trainer
pip install tensorflow numpy

# Train from scratch
python fast_trainer.py --episodes 100000 --parallel 512

# Continue from checkpoint
python fast_trainer.py --episodes 100000 --load weights/trained.json
```

### GCP Training (Recommended)
```bash
# Setup VM with GPU
./trainer/setup_gcp.sh

# Start training in tmux
tmux new -s training
python fast_trainer.py --episodes 100000 --parallel 512 --sparse
```

## 🔧 Technical Challenges & Lessons Learned

### Weight Loading Issue
Keras Functional API uses non-deterministic topological sorting. This caused weight shape mismatches when loading across different environments. 

**Solution:** Manual layer-name-based weight injection via `debug_load_vm.py`.

### Training Speed
Initial browser-based training: ~0.01 episodes/sec
Optimized Python + GPU: ~7-18 episodes/sec (700x speedup)

### Self-Play Limitations
Both agents learning together often converge to simple "chase ball" strategies rather than sophisticated play.

## 🔮 Future Improvements (Not Implemented)

These changes could significantly improve AI performance but require substantial rework:

| Improvement | Expected Impact | Effort |
|-------------|-----------------|--------|
| Switch to PPO/SAC | High - better for continuous control | High |
| Imitation learning (pretrain on human demos) | High | Medium |
| Richer state (velocities, angles, predictions) | Medium | Medium |
| Hierarchical RL (tactics + control) | High | Very High |
| Curriculum learning | Medium | Medium |

## 📊 Current AI Performance

**Honest assessment:** The AI actively chases the ball and attempts to score, but remains easy for humans to beat. This is a common challenge with RL in continuous control tasks — achieving human-level play in even simple sports games is research-level difficulty.

## 🛠️ Development

```bash
# Clone
git clone https://github.com/aeermumcu/rl-football.git
cd rl-football

# Play (no build needed)
open index.html

# Or serve locally
python -m http.server 8000
# Visit http://localhost:8000
```

## 📜 License

MIT License — feel free to use, modify, and learn from this project.

---

*Built as a learning project to explore reinforcement learning in browser games.*
