/**
 * Main Application Controller
 */
class App {
    constructor() {
        // Get DOM elements
        this.gameCanvas = document.getElementById('gameCanvas');
        this.chartCanvas = document.getElementById('chartCanvas');

        // Initialize game and AI
        this.game = new Game(this.gameCanvas);
        this.blipAgent = new RLAgent('Blip', 'blip');
        this.bloopAgent = new RLAgent('Bloop', 'bloop');
        this.visualizer = new Visualizer(this.chartCanvas);

        // State
        this.mode = 'train'; // train, watch, play
        this.running = false;
        this.speed = 1;
        this.episodeCount = 0;

        // Stats
        this.stats = {
            blipWins: 0,
            bloopWins: 0,
            draws: 0,
            totalGoals: 0
        };

        // Human input
        this.keys = {};

        // Bind controls
        this.bindControls();

        // Initial render
        this.game.draw();
        this.visualizer.draw();
        this.updateUI();
    }

    bindControls() {
        // Mode buttons
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const mode = btn.dataset.mode;
                this.setMode(mode);
            });
        });

        // Speed buttons
        document.querySelectorAll('.speed-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.setSpeed(parseInt(btn.dataset.speed));
            });
        });

        // Match length buttons
        document.querySelectorAll('.match-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.setMatchTime(parseInt(btn.dataset.time));
            });
        });

        // Start/Stop button
        document.getElementById('btnStart').addEventListener('click', () => {
            this.toggleRunning();
        });

        // Reset button
        document.getElementById('btnReset').addEventListener('click', () => {
            this.reset();
        });

        // Save button
        document.getElementById('btnSave').addEventListener('click', () => {
            this.saveTraining();
        });

        // Load button
        document.getElementById('btnLoad').addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });

        // File input for loading
        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.loadTraining(e.target.files[0]);
        });

        // Keyboard for play mode
        document.addEventListener('keydown', (e) => {
            // Prevent arrow keys and space from scrolling the page
            if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) {
                e.preventDefault();
            }
            this.keys[e.key] = true;
        });
        document.addEventListener('keyup', (e) => {
            this.keys[e.key] = false;
        });
    }

    setMode(mode) {
        this.mode = mode;

        // Update UI
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        // Reset game when changing modes
        if (this.running) {
            this.game.reset();
        }
    }

    setSpeed(speed) {
        this.speed = speed;

        document.querySelectorAll('.speed-btn').forEach(btn => {
            btn.classList.toggle('active', parseInt(btn.dataset.speed) === speed);
        });
    }

    setMatchTime(seconds) {
        this.game.matchTime = seconds;
        this.game.timeRemaining = seconds;

        document.querySelectorAll('.match-btn').forEach(btn => {
            btn.classList.toggle('active', parseInt(btn.dataset.time) === seconds);
        });
    }

    toggleRunning() {
        this.running = !this.running;

        const btn = document.getElementById('btnStart');
        if (this.running) {
            btn.innerHTML = '<span>⏸️ Pause</span>';
            btn.classList.add('running');
            this.run();
        } else {
            btn.innerHTML = '<span>▶️ Start Training</span>';
            btn.classList.remove('running');
        }
    }

    reset() {
        this.running = false;
        this.episodeCount = 0;
        this.stats = { blipWins: 0, bloopWins: 0, draws: 0, totalGoals: 0 };

        this.blipAgent = new RLAgent('Blip', 'blip');
        this.bloopAgent = new RLAgent('Bloop', 'bloop');
        this.visualizer.reset();

        this.game.reset();
        this.game.resetScores();

        document.getElementById('btnStart').innerHTML = '<span>▶️ Start Training</span>';
        document.getElementById('btnStart').classList.remove('running');

        this.game.draw();
        this.updateUI();
    }

    run() {
        if (!this.running) return;

        // Run multiple steps per frame at higher speeds
        const stepsPerFrame = this.speed;
        const renderEvery = Math.max(1, Math.floor(this.speed / 5));

        for (let i = 0; i < stepsPerFrame; i++) {
            this.step();
        }

        // Render (less frequently at high speeds)
        if (this.speed <= 5 || Math.random() < 0.1) {
            this.game.draw();
        }
        this.visualizer.draw();
        this.updateUI();

        requestAnimationFrame(() => this.run());
    }

    step() {
        // Get states for both agents
        const blipState = this.blipAgent.getState(
            this.game.blip,
            this.game.ball,
            this.game.bloop,
            this.game.fieldWidth,
            this.game.fieldHeight,
            this.game.goalY
        );

        const bloopState = this.bloopAgent.getState(
            this.game.bloop,
            this.game.ball,
            this.game.blip,
            this.game.fieldWidth,
            this.game.fieldHeight,
            this.game.goalY
        );

        // Choose actions
        let blipAction, bloopAction;

        if (this.mode === 'play') {
            // Human controls Blip
            blipAction = this.getHumanAction();
            bloopAction = this.bloopAgent.chooseAction(bloopState, false);
        } else {
            // Both AI
            const training = this.mode === 'train';
            blipAction = this.blipAgent.chooseAction(blipState, training);
            bloopAction = this.bloopAgent.chooseAction(bloopState, training);
        }

        // Apply actions
        this.game.applyAction(this.game.blip, blipAction);
        this.game.applyAction(this.game.bloop, bloopAction);

        // Update game
        const { event, done } = this.game.update(1);

        // Handle goal events and training
        if (this.mode === 'train') {
            let blipEvent = null;
            let bloopEvent = null;

            if (event === 'blip_scored') {
                blipEvent = 'scored';
                bloopEvent = 'conceded';
                this.stats.totalGoals++;
            } else if (event === 'bloop_scored') {
                blipEvent = 'conceded';
                bloopEvent = 'scored';
                this.stats.totalGoals++;
            }

            // Calculate rewards
            const blipReward = this.blipAgent.calculateReward(
                this.game.blip, this.game.ball, this.game.bloop,
                blipEvent, this.game.fieldWidth
            );
            const bloopReward = this.bloopAgent.calculateReward(
                this.game.bloop, this.game.ball, this.game.blip,
                bloopEvent, this.game.fieldWidth
            );

            // Get new states
            const newBlipState = this.blipAgent.getState(
                this.game.blip, this.game.ball, this.game.bloop,
                this.game.fieldWidth, this.game.fieldHeight, this.game.goalY
            );
            const newBloopState = this.bloopAgent.getState(
                this.game.bloop, this.game.ball, this.game.blip,
                this.game.fieldWidth, this.game.fieldHeight, this.game.goalY
            );

            // Learn
            this.blipAgent.learn(blipReward, newBlipState, done);
            this.bloopAgent.learn(bloopReward, newBloopState, done);
        }

        // Episode end
        if (done) {
            this.endEpisode();
        }
    }

    getHumanAction() {
        let dx = 0, dy = 0, kick = false;

        if (this.keys['ArrowUp'] || this.keys['w']) dy = -1;
        if (this.keys['ArrowDown'] || this.keys['s']) dy = 1;
        if (this.keys['ArrowLeft'] || this.keys['a']) dx = -1;
        if (this.keys['ArrowRight'] || this.keys['d']) dx = 1;
        if (this.keys[' ']) kick = true;

        return { dx, dy, kick, name: 'human' };
    }

    endEpisode() {
        this.episodeCount++;

        const winner = this.game.getWinner();

        // Update stats
        if (winner === 'blip') {
            this.stats.blipWins++;
        } else if (winner === 'bloop') {
            this.stats.bloopWins++;
        } else {
            this.stats.draws++;
        }

        // Update visualizer
        this.visualizer.addResult(winner);

        // Reset agents for new episode
        this.blipAgent.reset();
        this.bloopAgent.reset();

        // Reset game
        this.game.reset();
        this.game.resetScores();
    }

    updateUI() {
        // Scores
        document.getElementById('blipScore').textContent = this.game.blipScore;
        document.getElementById('bloopScore').textContent = this.game.bloopScore;

        // Episode count
        document.getElementById('episodeCount').textContent = this.episodeCount;

        // Timer
        document.getElementById('matchTimer').textContent = this.game.getTimeString();

        // Stats
        document.getElementById('blipWins').textContent = this.stats.blipWins;
        document.getElementById('bloopWins').textContent = this.stats.bloopWins;
        document.getElementById('draws').textContent = this.stats.draws;
        document.getElementById('totalGoals').textContent = this.stats.totalGoals;

        // AI info
        document.getElementById('qTableSize').textContent =
            (this.blipAgent.getQTableSize() + this.bloopAgent.getQTableSize()).toLocaleString();
        document.getElementById('epsilon').textContent =
            this.blipAgent.getEpsilon().toFixed(3);
    }

    saveTraining() {
        const data = {
            version: 1,
            timestamp: new Date().toISOString(),
            episodeCount: this.episodeCount,
            stats: this.stats,
            blipAgent: this.blipAgent.exportQTable(),
            bloopAgent: this.bloopAgent.exportQTable(),
            winHistory: {
                blip: this.visualizer.blipWinHistory,
                bloop: this.visualizer.bloopWinHistory
            }
        };

        const json = JSON.stringify(data);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `rl-football-${this.episodeCount}-episodes.json`;
        a.click();

        URL.revokeObjectURL(url);
    }

    loadTraining(file) {
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = JSON.parse(e.target.result);

                // Restore state
                this.episodeCount = data.episodeCount || 0;
                this.stats = data.stats || { blipWins: 0, bloopWins: 0, draws: 0, totalGoals: 0 };

                // Restore agents
                this.blipAgent.importQTable(data.blipAgent);
                this.bloopAgent.importQTable(data.bloopAgent);

                // Restore visualizer history
                if (data.winHistory) {
                    this.visualizer.blipWinHistory = data.winHistory.blip || [];
                    this.visualizer.bloopWinHistory = data.winHistory.bloop || [];
                }

                // Update UI
                this.game.reset();
                this.game.resetScores();
                this.game.draw();
                this.visualizer.draw();
                this.updateUI();

                alert(`✅ Loaded ${this.episodeCount} episodes of training!`);
            } catch (err) {
                alert('❌ Error loading file: ' + err.message);
            }
        };
        reader.readAsText(file);

        // Reset file input so same file can be loaded again
        document.getElementById('fileInput').value = '';
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
