# RL Football Champions - Headless Trainer

Train the AI without a browser! Runs much faster and can run on cloud servers.

## Quick Start

```bash
cd trainer
npm install
npm run train
```

## Commands

```bash
# Train 5000 episodes (default)
npm run train

# Train 10000 episodes
npm run train:10k

# Custom training
node train.js --episodes=3000 --match-time=30 --save-every=500
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--episodes=N` | 5000 | Number of episodes to train |
| `--match-time=N` | 30 | Match length in seconds |
| `--save-every=N` | 500 | Save checkpoint every N episodes |

## Output

Weights are saved to `trainer/weights/`:
- `weights-500.json`, `weights-1000.json`, etc.
- `latest.json` - most recent save

## Loading Weights in Browser

1. After training, copy `trainer/weights/latest.json`
2. In the browser game, click **Load**
3. Select the JSON file
4. Your trained AI is ready!

## Estimated Training Times

| Episodes | Time (approx) | Difficulty |
|----------|---------------|------------|
| 2,000 | 10-15 min | Medium |
| 5,000 | 30-45 min | Hard |
| 10,000 | 1-2 hours | Very Hard |
| 20,000 | 3-4 hours | Expert |

*Times vary based on CPU speed*

## Running on Cloud

### Replit
1. Upload the `trainer` folder to a new Node.js Repl
2. Run `npm install && npm run train`

### Railway / Render
1. Deploy the `trainer` folder as a Node.js app
2. Set start command: `npm run train:10k`

### Google Cloud / AWS
```bash
# SSH into your VM
git clone <your-repo>
cd "Reinforcement Learning Demo/trainer"
npm install
nohup npm run train:10k &
```

The training will continue even if you disconnect!
