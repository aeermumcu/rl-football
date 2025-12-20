/**
 * Deep Q-Network Agent using TensorFlow.js
 */
class DQNAgent {
    constructor(name, team) {
        this.name = name;
        this.team = team;

        // State and action dimensions
        this.stateSize = 12;  // Continuous state features
        this.actionSize = 10; // Same actions as Q-learning

        // Hyperparameters
        this.learningRate = 0.001;
        this.gamma = 0.99;           // Discount factor
        this.epsilon = 1.0;          // Exploration rate
        this.epsilonMin = 0.05;
        this.epsilonDecay = 0.995;    // Decay per EPISODE (not per step)

        // Experience replay
        this.replayBuffer = [];
        this.bufferSize = 10000;
        this.batchSize = 32;
        this.minBufferSize = 100;    // Start training after this many experiences

        // Target network update frequency
        this.targetUpdateFreq = 100;
        this.trainStepCount = 0;

        // Networks
        this.model = null;
        this.targetModel = null;
        this.isInitialized = false;

        // Actions (same as Q-learning)
        this.actions = [
            { dx: 0, dy: -1, kick: false, name: 'up' },
            { dx: 0, dy: 1, kick: false, name: 'down' },
            { dx: -1, dy: 0, kick: false, name: 'left' },
            { dx: 1, dy: 0, kick: false, name: 'right' },
            { dx: -1, dy: -1, kick: false, name: 'up-left' },
            { dx: 1, dy: -1, kick: false, name: 'up-right' },
            { dx: -1, dy: 1, kick: false, name: 'down-left' },
            { dx: 1, dy: 1, kick: false, name: 'down-right' },
            { dx: 0, dy: 0, kick: true, name: 'kick' },
            { dx: 0, dy: 0, kick: false, name: 'stay' }
        ];

        // Field dimensions for normalization
        this.fieldWidth = 720;
        this.fieldHeight = 420;

        // State tracking
        this.lastState = null;
        this.lastAction = null;

        // Initialize networks
        this.initNetworks();
    }

    async initNetworks() {
        // Main Q-network
        this.model = this.createNetwork();

        // Target network (copy of main)
        this.targetModel = this.createNetwork();

        // Copy weights to target
        await this.updateTargetNetwork();

        this.isInitialized = true;
        console.log(`${this.name} DQN initialized!`);
    }

    createNetwork() {
        const model = tf.sequential();

        // Input layer + first hidden layer
        model.add(tf.layers.dense({
            inputShape: [this.stateSize],
            units: 128,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        // Second hidden layer
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        // Output layer (Q-values for each action)
        model.add(tf.layers.dense({
            units: this.actionSize,
            activation: 'linear',
            kernelInitializer: 'heNormal'
        }));

        // Compile with MSE loss and Adam optimizer
        model.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: 'meanSquaredError'
        });

        return model;
    }

    async updateTargetNetwork() {
        const weights = this.model.getWeights();
        const targetWeights = weights.map(w => w.clone());
        this.targetModel.setWeights(targetWeights);
    }

    /**
     * Extract continuous state features from game state
     * Note: uses same method name as RLAgent for compatibility
     */
    getState(player, ball, opponent, fieldWidth, fieldHeight) {
        // Normalize all values to 0-1 range
        const playerX = player.x / fieldWidth;
        const playerY = player.y / fieldHeight;
        const ballX = ball.x / fieldWidth;
        const ballY = ball.y / fieldHeight;

        // Ball velocity (normalized, capped)
        const maxVel = 15;
        const ballVx = Math.max(-1, Math.min(1, ball.vx / maxVel));
        const ballVy = Math.max(-1, Math.min(1, ball.vy / maxVel));

        const opponentX = opponent.x / fieldWidth;
        const opponentY = opponent.y / fieldHeight;

        // Distance and angle to ball
        const distBall = player.distanceTo(ball) / Math.sqrt(fieldWidth ** 2 + fieldHeight ** 2);
        const angleBall = (player.angleTo(ball) + Math.PI) / (2 * Math.PI);

        // Distance and angle to goal
        const goalX = this.team === 'blip' ? fieldWidth : 0;
        const goalY = fieldHeight / 2;
        const distGoal = Math.sqrt((player.x - goalX) ** 2 + (player.y - goalY) ** 2) /
            Math.sqrt(fieldWidth ** 2 + fieldHeight ** 2);
        const angleGoal = (Math.atan2(goalY - player.y, goalX - player.x) + Math.PI) / (2 * Math.PI);

        return [
            playerX, playerY,
            ballX, ballY, ballVx, ballVy,
            opponentX, opponentY,
            distBall, angleBall,
            distGoal, angleGoal
        ];
    }

    /**
     * Choose action using epsilon-greedy policy
     */
    chooseAction(state, training = true) {
        if (!this.isInitialized) {
            // Random action while initializing
            const actionIndex = Math.floor(Math.random() * this.actionSize);
            return this.actions[actionIndex];
        }

        let actionIndex;

        if (training && Math.random() < this.epsilon) {
            // Explore: random action
            actionIndex = Math.floor(Math.random() * this.actionSize);
        } else {
            // Exploit: use neural network with proper tensor cleanup
            actionIndex = tf.tidy(() => {
                const stateTensor = tf.tensor2d([state], [1, this.stateSize]);
                const qValues = this.model.predict(stateTensor);
                return qValues.argMax(1).dataSync()[0];
            });
        }

        this.lastState = state;
        this.lastAction = actionIndex;

        return this.actions[actionIndex];
    }

    /**
     * Store experience in replay buffer
     */
    remember(state, action, reward, nextState, done) {
        this.replayBuffer.push({
            state, action, reward, nextState, done
        });

        // Remove oldest if buffer is full
        if (this.replayBuffer.length > this.bufferSize) {
            this.replayBuffer.shift();
        }
    }

    /**
     * Train on a batch from replay buffer
     */
    async train() {
        if (this.replayBuffer.length < this.minBufferSize) {
            return; // Not enough experiences yet
        }

        // Sample random batch
        const batch = [];
        for (let i = 0; i < this.batchSize; i++) {
            const idx = Math.floor(Math.random() * this.replayBuffer.length);
            batch.push(this.replayBuffer[idx]);
        }

        // Prepare training data
        const states = batch.map(e => e.state);
        const nextStates = batch.map(e => e.nextState);

        // Use tf.tidy for intermediate tensors that can be auto-cleaned
        const { currentQsArray, nextQsArray } = tf.tidy(() => {
            const statesTensor = tf.tensor2d(states, [this.batchSize, this.stateSize]);
            const nextStatesTensor = tf.tensor2d(nextStates, [this.batchSize, this.stateSize]);
            const currentQs = this.model.predict(statesTensor);
            const nextQs = this.targetModel.predict(nextStatesTensor);
            return {
                currentQsArray: currentQs.arraySync(),
                nextQsArray: nextQs.arraySync()
            };
        });

        // Calculate target Q values (pure JS, no tensors)
        for (let i = 0; i < this.batchSize; i++) {
            const { action, reward, done } = batch[i];
            if (done) {
                currentQsArray[i][action] = reward;
            } else {
                const maxNextQ = Math.max(...nextQsArray[i]);
                currentQsArray[i][action] = reward + this.gamma * maxNextQ;
            }
        }

        // Train the model with proper cleanup
        const statesTensor = tf.tensor2d(states, [this.batchSize, this.stateSize]);
        const targetTensor = tf.tensor2d(currentQsArray, [this.batchSize, this.actionSize]);

        await this.model.fit(statesTensor, targetTensor, {
            epochs: 1,
            verbose: 0
        });

        // Clean up training tensors
        statesTensor.dispose();
        targetTensor.dispose();

        // Update target network periodically
        this.trainStepCount++;
        if (this.trainStepCount % this.targetUpdateFreq === 0) {
            await this.updateTargetNetwork();
        }

        // Note: epsilon decay moved to reset() - happens per episode, not per step
    }

    /**
     * Calculate reward (same as Q-learning version)
     */
    calculateReward(player, ball, opponent, event, fieldWidth) {
        let reward = 0;

        // Goal events (big rewards)
        if (event === 'scored') {
            reward += 100;
        } else if (event === 'conceded') {
            reward -= 100;
        }

        // Distance to ball
        const distToBall = player.distanceTo(ball);
        if (distToBall < 50) {
            reward += 2;
        } else if (distToBall < 100) {
            reward += 1;
        }

        // Moving toward ball
        if (this.lastDistToBall !== null && distToBall < this.lastDistToBall) {
            reward += 0.5;
        }
        this.lastDistToBall = distToBall;

        // Ball moving toward goal
        const ballMovingTowardGoal = (this.team === 'blip' && ball.vx > 0) ||
            (this.team === 'bloop' && ball.vx < 0);
        if (ballMovingTowardGoal && distToBall < 80) {
            reward += 1;
        }

        // Time penalty
        reward -= 0.1;

        // Corner penalty
        const fieldHeight = 420;
        const cornerMargin = 60;
        const inCorner = (player.x < cornerMargin || player.x > fieldWidth - cornerMargin) &&
            (player.y < cornerMargin || player.y > fieldHeight - cornerMargin);
        if (inCorner) {
            reward -= 2;
        }

        // Far from ball penalty
        if (distToBall > 200) {
            reward -= 0.5;
        }

        return reward;
    }

    /**
     * Reset for new episode
     */
    reset() {
        this.lastState = null;
        this.lastAction = null;
        this.lastDistToBall = null;

        // Decay epsilon once per episode
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }

    /**
     * Get stats for UI
     */
    getEpsilon() {
        return this.epsilon;
    }

    getBufferSize() {
        return this.replayBuffer.length;
    }

    /**
     * Export model weights for saving
     */
    async exportWeights() {
        if (!this.isInitialized) return null;

        const weights = this.model.getWeights();
        const weightData = await Promise.all(weights.map(async w => ({
            shape: w.shape,
            data: Array.from(await w.data())
        })));

        return {
            weights: weightData,
            epsilon: this.epsilon,
            trainStepCount: this.trainStepCount
        };
    }

    /**
     * Import model weights from saved data
     */
    async importWeights(data) {
        if (!this.isInitialized || !data || !data.weights) return;

        const weights = data.weights.map(w =>
            tf.tensor(w.data, w.shape)
        );

        this.model.setWeights(weights);
        await this.updateTargetNetwork();

        this.epsilon = data.epsilon || this.epsilon;
        this.trainStepCount = data.trainStepCount || 0;

        // Clean up
        weights.forEach(w => w.dispose());
    }
}

// Export
window.DQNAgent = DQNAgent;
