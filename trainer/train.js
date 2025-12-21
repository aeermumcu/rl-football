/**
 * Headless Training Script for RL Football Champions
 * Run with: npm run train or node train.js --episodes=5000
 */
const fs = require('fs');
const path = require('path');
const DQNAgent = require('./dqn-agent');
const { Game } = require('./game');

// Parse command line arguments
const args = process.argv.slice(2);
let targetEpisodes = 5000;
let saveEvery = 500;
let matchTime = 30;

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
});

console.log('='.repeat(50));
console.log('🤖 RL Football Champions - Headless Trainer');
console.log('='.repeat(50));
console.log(`Episodes: ${targetEpisodes}`);
console.log(`Match time: ${matchTime}s`);
console.log(`Save every: ${saveEvery} episodes`);
console.log('='.repeat(50));

// Initialize
const game = new Game();
game.matchTime = matchTime;

const blipAgent = new DQNAgent('Blip', 'blip');
const bloopAgent = new DQNAgent('Bloop', 'bloop');

// Stats
const stats = {
    blipWins: 0,
    bloopWins: 0,
    draws: 0,
    totalGoals: 0
};

// Training loop
async function train() {
    const startTime = Date.now();

    for (let episode = 1; episode <= targetEpisodes; episode++) {
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

            // Store experiences
            blipAgent.remember(blipState, blipAgent.lastAction, blipReward, newBlipState, done);
            bloopAgent.remember(bloopState, bloopAgent.lastAction, bloopReward, newBloopState, done);

            // Train every 4 steps
            if (steps % 4 === 0) {
                await blipAgent.train();
                await bloopAgent.train();
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
            const epsPerSec = episode / elapsed;
            const remaining = (targetEpisodes - episode) / epsPerSec;

            console.log(`Episode ${episode}/${targetEpisodes} | ε: ${blipAgent.epsilon.toFixed(3)} | ` +
                `Blip: ${stats.blipWins} | Bloop: ${stats.bloopWins} | Draws: ${stats.draws} | ` +
                `Goals: ${stats.totalGoals} | ETA: ${formatTime(remaining)}`);
        }

        // Auto-save
        if (episode % saveEvery === 0) {
            await saveWeights(episode);
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
    const bloopWeights = await bloopAgent.exportWeights();

    const saveData = {
        version: 'ddqn-256-128-64',
        episode,
        timestamp: new Date().toISOString(),
        stats,
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
