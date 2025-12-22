/**
 * Main Application Controller
 */
class App {
    constructor() {
        // Get DOM elements
        this.gameCanvas = document.getElementById('gameCanvas');
        this.chartCanvas = document.getElementById('chartCanvas');

        // Initialize game
        this.game = new Game(this.gameCanvas);
        this.visualizer = new Visualizer(this.chartCanvas);

        // AI type: 'qlearning' or 'dqn'
        this.aiType = 'dqn';

        // Initialize agents based on AI type
        this.initAgents();

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

    initAgents() {
        if (this.aiType === 'dqn') {
            this.blipAgent = new DQNAgent('Blip', 'blip');
            this.bloopAgent = new DQNAgent('Bloop', 'bloop');
        } else {
            this.blipAgent = new RLAgent('Blip', 'blip');
            this.bloopAgent = new RLAgent('Bloop', 'bloop');
        }
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

        // AI type buttons
        document.querySelectorAll('.ai-type-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.setAIType(btn.dataset.type);
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

    setAIType(type) {
        if (this.running) {
            alert('Please pause training before switching AI type!');
            return;
        }

        this.aiType = type;

        // Update UI
        document.querySelectorAll('.ai-type-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.type === type);
        });
        document.getElementById('aiTypeLabel').textContent = type === 'dqn' ? 'DQN' : 'Q-Learn';

        // Reset and reinitialize agents
        this.reset();
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
            // Clear training interval when paused
            if (this.trainInterval) {
                clearInterval(this.trainInterval);
                this.trainInterval = null;
            }
        }
    }

    reset() {
        this.running = false;
        this.episodeCount = 0;
        this.stats = { blipWins: 0, bloopWins: 0, draws: 0, totalGoals: 0 };

        // Reinitialize agents based on current AI type
        this.initAgents();
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

        // Clear any existing interval
        if (this.trainInterval) {
            clearInterval(this.trainInterval);
        }

        // Use setInterval for training (works in background tabs!)
        // 16ms = ~60fps equivalent
        this.trainInterval = setInterval(() => {
            if (!this.running) {
                clearInterval(this.trainInterval);
                return;
            }

            // Cap steps per frame to prevent freezing
            const maxStepsPerFrame = this.aiType === 'dqn' ? 10 : 50;
            const stepsPerFrame = Math.min(this.speed, maxStepsPerFrame);

            for (let i = 0; i < stepsPerFrame; i++) {
                this.step();
            }
        }, 16);

        // Use requestAnimationFrame only for rendering (pauses in background, that's fine)
        this.renderLoop();
    }

    renderLoop() {
        if (!this.running) return;

        // Track frames for consistent rendering
        this.frameCount = (this.frameCount || 0) + 1;

        // Always render every N frames to prevent black screen
        const renderEveryNFrames = this.speed >= 50 ? 5 : (this.speed >= 10 ? 2 : 1);
        if (this.frameCount % renderEveryNFrames === 0) {
            this.game.draw();
            this.visualizer.draw();
            this.updateUI();
        }

        requestAnimationFrame(() => this.renderLoop());
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

            // Learn based on AI type
            if (this.aiType === 'dqn') {
                // DQN: store experience and train
                this.blipAgent.remember(blipState, this.blipAgent.lastAction, blipReward, newBlipState, done);
                this.bloopAgent.remember(bloopState, this.bloopAgent.lastAction, bloopReward, newBloopState, done);

                // Throttle training to reduce CPU load at high speeds
                // Only train every 4th step
                this.trainCounter = (this.trainCounter || 0) + 1;
                if (this.trainCounter % 4 === 0) {
                    // Asymmetric training: only train one agent at a time to prevent collapse
                    const alternateEvery = 500;
                    const phase = Math.floor(this.episodeCount / alternateEvery);
                    const trainBlip = phase % 2 === 0;

                    if (trainBlip) {
                        this.blipAgent.train();
                    } else {
                        this.bloopAgent.train();
                    }
                }
            } else {
                // Q-learning: direct update
                this.blipAgent.learn(blipReward, newBlipState, done);
                this.bloopAgent.learn(bloopReward, newBloopState, done);
            }
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

        // Autosave every 500 episodes
        const autosaveEnabled = document.getElementById('autosaveCheckbox')?.checked;
        if (autosaveEnabled && this.episodeCount > 0 && this.episodeCount % 500 === 0) {
            console.log(`🔄 Autosaving at episode ${this.episodeCount}...`);
            this.saveTraining();
        }

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

        // AI info - different display for each type
        if (this.aiType === 'dqn') {
            // DQN: show replay buffer size
            const bufferSize = (this.blipAgent.getBufferSize?.() || 0) +
                (this.bloopAgent.getBufferSize?.() || 0);
            document.getElementById('qTableSize').textContent = bufferSize.toLocaleString();
        } else {
            // Q-learning: show Q-table size
            document.getElementById('qTableSize').textContent =
                (this.blipAgent.getQTableSize() + this.bloopAgent.getQTableSize()).toLocaleString();
        }

        document.getElementById('epsilon').textContent =
            this.blipAgent.getEpsilon().toFixed(3);
    }

    async saveTraining() {
        let blipAgentData, bloopAgentData;

        if (this.aiType === 'dqn') {
            // DQN: export neural network weights
            blipAgentData = await this.blipAgent.exportWeights();
            bloopAgentData = await this.bloopAgent.exportWeights();
        } else {
            // Q-learning: export Q-table
            blipAgentData = this.blipAgent.exportQTable();
            bloopAgentData = this.bloopAgent.exportQTable();
        }

        const data = {
            version: 2,
            aiType: this.aiType,
            timestamp: new Date().toISOString(),
            episodeCount: this.episodeCount,
            stats: this.stats,
            blipAgent: blipAgentData,
            bloopAgent: bloopAgentData,
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
        a.download = `rl-football-${this.aiType}-${this.episodeCount}-episodes.json`;
        a.click();

        URL.revokeObjectURL(url);
    }

    async loadTraining(file) {
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const data = JSON.parse(e.target.result);
                console.log('Loading file, keys:', Object.keys(data));

                // Detect AI type - check multiple possible locations
                let savedType = data.aiType;
                if (!savedType) {
                    // Try to detect from data structure
                    if (data.blipAgent?.weights || data.blip?.weights) {
                        savedType = 'dqn';
                    } else if (data.blipAgent?.qTable || data.blip?.qTable) {
                        savedType = 'qlearning';
                    } else {
                        savedType = 'dqn'; // Default to DQN for trainer files
                    }
                }
                console.log('Detected AI type:', savedType);

                if (savedType !== this.aiType) {
                    const switchType = confirm(
                        `This save file is for ${savedType.toUpperCase()}. ` +
                        `You're currently using ${this.aiType.toUpperCase()}. ` +
                        `Switch to ${savedType.toUpperCase()} to load?`
                    );
                    if (switchType) {
                        this.aiType = savedType;
                        document.querySelectorAll('.ai-type-btn').forEach(btn => {
                            btn.classList.toggle('active', btn.dataset.type === savedType);
                        });
                        document.getElementById('aiTypeLabel').textContent = savedType === 'dqn' ? 'DQN' : 'Q-Learn';
                        this.initAgents();
                    } else {
                        return;
                    }
                }

                // Restore state
                this.episodeCount = data.episodeCount || data.episode || 0;
                this.stats = data.stats || { blipWins: 0, bloopWins: 0, draws: 0, totalGoals: 0 };

                // Restore agents based on type - handle both browser and trainer formats
                if (this.aiType === 'dqn') {
                    const blipData = data.blipAgent || data.blip;
                    const bloopData = data.bloopAgent || data.bloop;
                    console.log('Loading DQN weights, blipData exists:', !!blipData, 'bloopData exists:', !!bloopData);
                    await this.blipAgent.importWeights(blipData);
                    await this.bloopAgent.importWeights(bloopData);
                } else {
                    const blipData = data.blipAgent || data.blip;
                    const bloopData = data.bloopAgent || data.bloop;
                    this.blipAgent.importQTable(blipData);
                    this.bloopAgent.importQTable(bloopData);
                }

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

                alert(`✅ Loaded ${this.episodeCount} episodes of ${this.aiType.toUpperCase()} training!`);
            } catch (err) {
                alert('❌ Error loading file: ' + err.message);
                console.error(err);
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
