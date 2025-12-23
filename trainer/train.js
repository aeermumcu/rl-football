/**
 * Headless Training Script for RL Football Champions
 * Run with: npm run train or node train.js --episodes=5000
 * Resume with: node train.js --resume=weights/weights-4000.json --episodes=10000
 * 
 * NEW: Trains DQN against aggressive SimpleAI opponent to learn active play
 */
const fs = require('fs');
const path = require('path');
const DQNAgent = require('./dqn-agent');
const SimpleAI = require('./simple-ai');
const { Game } = require('./game');

// Parse command line arguments
const args = process.argv.slice(2);
let targetEpisodes = 5000;
let saveEvery = 500;
let matchTime = 30;
let resumeFile = null;
let startEpisode = 1;
let useSimpleOpponent = true;  // NEW: Use SimpleAI as opponent

args.forEach(arg => {
    if (arg.startsWith('--episodes=')) {
        targetEpisodes = parseInt(arg.split('=')[1]);
    }
    if (arg.startsWith('--save-every=')) {
        saveEvery = parseInt(arg.split('=')[1]);
    }
    if (arg.startsWith('--match-time=')) {
        matchTime = parseInt(arg.split('=')[1]);
    }
    if (arg.startsWith('--resume=')) {
        resumeFile = arg.split('=')[1];
    }
    if (arg === '--self-play') {
        useSimpleOpponent = false;  // Old behavior: train both DQN agents
    }
});

console.log('='.repeat(50));
console.log('🤖 RL Football Champions - Headless Trainer');
console.log('='.repeat(50));
console.log(`Episodes: ${targetEpisodes}`);
console.log(`Match time: ${matchTime}s`);
console.log(`Save every: ${saveEvery} episodes`);
console.log(`Opponent: ${useSimpleOpponent ? '🎯 SimpleAI (ball-chaser)' : 'DQN (self-play)'}`);
if (resumeFile) console.log(`Resuming from: ${resumeFile}`);
console.log('='.repeat(50));

// Initialize
const game = new Game();
game.matchTime = matchTime;

// DQN agent learns, opponent is either SimpleAI or another DQN
const blipAgent = new DQNAgent('Blip', 'blip');
const bloopAgent = useSimpleOpponent
    ? new SimpleAI('Bloop', 'bloop')
    : new DQNAgent('Bloop', 'bloop');

// Stats
let stats = {
    blipWins: 0,
    bloopWins: 0,
    draws: 0,
    totalGoals: 0
};

// Load checkpoint if resuming
async function loadCheckpoint() {
    if (!resumeFile) return;

    const filePath = path.join(__dirname, resumeFile);
    if (!fs.existsSync(filePath)) {
        console.error(`❌ Resume file not found: ${filePath}`);
        process.exit(1);
    }

    console.log(`📂 Loading checkpoint from ${resumeFile}...`);
    const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));

    // Load weights (only for DQN agents)
    await blipAgent.importWeights(data.blip);
    if (!useSimpleOpponent && bloopAgent.importWeights) {
        await bloopAgent.importWeights(data.bloop);
    }

    // Restore stats and episode count
    stats = data.stats || stats;
    startEpisode = (data.episode || 0) + 1;

    console.log(`✅ Resumed from episode ${data.episode}`);
    console.log(`   Stats: Blip ${stats.blipWins} | Bloop ${stats.bloopWins} | Draws ${stats.draws}`);
}

// Training loop
async function train() {
    await loadCheckpoint();

    const startTime = Date.now();

    for (let episode = startEpisode; episode <= targetEpisodes; episode++) {
        game.reset();
        game.resetScores();

        let steps = 0;
        let done = false;

        while (!done) {
            steps++;

            // Get states
            const blipState = blipAgent.getState(
                game.blip, game.ball, game.bloop,
                game.fieldWidth, game.fieldHeight
            );
            const bloopState = bloopAgent.getState(
                game.bloop, game.ball, game.blip,
                game.fieldWidth, game.fieldHeight
            );

            // Choose actions
            const blipAction = blipAgent.chooseAction(blipState, true);
            const bloopAction = bloopAgent.chooseAction(bloopState, true);

            // Apply actions
            game.applyAction(game.blip, blipAction);
            game.applyAction(game.bloop, bloopAction);

            // Update game
            const result = game.update(1);
            done = result.done;

            // Handle events
            let blipEvent = null;
            let bloopEvent = null;

            if (result.event === 'blip_scored') {
                blipEvent = 'scored';
                bloopEvent = 'conceded';
                stats.totalGoals++;
            } else if (result.event === 'bloop_scored') {
                blipEvent = 'conceded';
                bloopEvent = 'scored';
                stats.totalGoals++;
            }

            // Calculate rewards
            const blipReward = blipAgent.calculateReward(
                game.blip, game.ball, game.bloop, blipEvent, game.fieldWidth
            );
            const bloopReward = bloopAgent.calculateReward(
                game.bloop, game.ball, game.blip, bloopEvent, game.fieldWidth
            );

            // Get new states
            const newBlipState = blipAgent.getState(
                game.blip, game.ball, game.bloop,
                game.fieldWidth, game.fieldHeight
            );
            const newBloopState = bloopAgent.getState(
                game.bloop, game.ball, game.blip,
                game.fieldWidth, game.fieldHeight
            );

            // Store experiences (only Blip learns, Bloop is SimpleAI)
            blipAgent.remember(blipState, blipAgent.lastAction, blipReward, newBlipState, done);
            if (!useSimpleOpponent) {
                bloopAgent.remember(bloopState, bloopAgent.lastAction, bloopReward, newBloopState, done);
            }

            // Train every 4 steps (only the learning agent)
            if (steps % 4 === 0) {
                await blipAgent.train();
                if (!useSimpleOpponent) {
                    await bloopAgent.train();
                }
            }
        }

        // Episode end
        const winner = game.getWinner();
        if (winner === 'blip') stats.blipWins++;
        else if (winner === 'bloop') stats.bloopWins++;
        else stats.draws++;

        blipAgent.reset();
        bloopAgent.reset();

        // Progress logging
        if (episode % 100 === 0 || episode === 1) {
            const elapsed = (Date.now() - startTime) / 1000;
            const epsPerSec = (episode - startEpisode + 1) / elapsed;
            const remaining = (targetEpisodes - episode) / epsPerSec;

            const epsilon = blipAgent.epsilon?.toFixed(3) || 'N/A';

            console.log(`Episode ${episode}/${targetEpisodes} | ε: ${epsilon} | ` +
                `Blip: ${stats.blipWins} | Bloop: ${stats.bloopWins} | Draws: ${stats.draws} | ` +
                `Goals: ${stats.totalGoals} | ETA: ${formatTime(remaining)}`);
        }

        // Auto-save
        if (episode % saveEvery === 0) {
            await saveWeights(episode);

            // Proactive memory management - garbage collect after saving
            if (global.gc) {
                global.gc();
                const used = process.memoryUsage();
                const heapMB = Math.round(used.heapUsed / 1024 / 1024);
                console.log(`🧹 Memory cleaned: ${heapMB}MB heap used`);
            }
        }
    }

    // Final save
    await saveWeights(targetEpisodes);

    console.log('='.repeat(50));
    console.log('✅ Training complete!');
    console.log(`Final stats: Blip ${stats.blipWins} | Bloop ${stats.bloopWins} | Draws ${stats.draws}`);
    console.log(`Total goals: ${stats.totalGoals}`);
    console.log('='.repeat(50));
}

async function saveWeights(episode) {
    const outputDir = path.join(__dirname, 'weights');
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir);
    }

    const blipWeights = await blipAgent.exportWeights();
    // Bloop only has weights if it's a DQN agent (self-play mode)
    const bloopWeights = !useSimpleOpponent ? await bloopAgent.exportWeights() : blipWeights;

    const saveData = {
        version: 2,
        aiType: 'dqn',
        episode,
        episodeCount: episode,
        timestamp: new Date().toISOString(),
        trainedAgainst: useSimpleOpponent ? 'SimpleAI' : 'DQN',
        stats,
        blipAgent: blipWeights,
        bloopAgent: bloopWeights,  // Use Blip's weights for both when vs SimpleAI
        // Also save in old format for trainer resume compatibility
        blip: blipWeights,
        bloop: bloopWeights
    };

    // Save with episode number
    const filename = `weights-${episode}.json`;
    fs.writeFileSync(path.join(outputDir, filename), JSON.stringify(saveData));

    // Also save as "latest"
    fs.writeFileSync(path.join(outputDir, 'latest.json'), JSON.stringify(saveData));

    console.log(`💾 Saved weights to ${filename}`);
}

function formatTime(seconds) {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
}

// Run training
train().catch(err => {
    console.error('Training error:', err);
    process.exit(1);
});
