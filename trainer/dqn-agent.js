/**
 * Advanced DQN Agent for Node.js Training
 * Features: Dueling Architecture, Optimized Hyperparameters
 */
const tf = require('@tensorflow/tfjs-node');

/**
 * Custom layer to combine Value and Advantage streams for Dueling DQN
 * Input: [value (1), advantages (actionSize)] concatenated
 * Output: Q(s,a) = V(s) + (A(s,a) - mean(A))
 */
class DuelingCombineLayer extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.actionSize = config.actionSize;
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], this.actionSize];
    }

    call(inputs) {
        return tf.tidy(() => {
            const input = Array.isArray(inputs) ? inputs[0] : inputs;
            // Split: first column is value, rest is advantages
            const value = tf.slice(input, [0, 0], [-1, 1]);
            const advantages = tf.slice(input, [0, 1], [-1, this.actionSize]);
            // Q = V + (A - mean(A))
            const meanAdvantage = tf.mean(advantages, -1, true);
            return tf.add(value, tf.sub(advantages, meanAdvantage));
        });
    }

    getConfig() {
        const config = super.getConfig();
        config.actionSize = this.actionSize;
        return config;
    }

    static get className() {
        return 'DuelingCombineLayer';
    }
}
tf.serialization.registerClass(DuelingCombineLayer);


class DQNAgent {
    constructor(name, team) {
        this.name = name;
        this.team = team;

        // State and action dimensions
        this.stateSize = 12;
        this.actionSize = 10;

        // OPTIMIZED Hyperparameters for better learning
        this.learningRate = 0.0005;     // Lower for stability (was 0.001)
        this.gamma = 0.995;             // Value future more (was 0.99)
        this.epsilon = 1.0;
        this.epsilonMin = 0.02;         // Lower minimum for more exploitation
        this.epsilonDecay = 0.9999;     // Much slower decay (was 0.9995)

        // Larger experience replay for more diverse learning
        this.replayBuffer = [];
        this.bufferSize = 50000;        // 5x larger (was 10000)
        this.batchSize = 64;            // Larger batches (was 32)
        this.minBufferSize = 500;       // Wait longer before training

        // Target network update frequency
        this.targetUpdateFreq = 500;    // Less frequent updates for stability (was 100)
        this.trainStepCount = 0;

        // Networks
        this.model = null;
        this.targetModel = null;

        // Actions
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

        // State tracking
        this.lastState = null;
        this.lastAction = null;
        this.lastDistToBall = null;

        // Initialize networks
        this.initNetworks();
    }

    initNetworks() {
        this.model = this.createNetwork();
        this.targetModel = this.createNetwork();
        this.updateTargetNetwork();
        console.log(`${this.name} DQN initialized!`);
    }

    /**
     * Create Dueling DQN network
     * Separates value stream V(s) and advantage stream A(s,a)
     * Q(s,a) = V(s) + (A(s,a) - mean(A))
     */
    createNetwork() {
        // Input layer
        const input = tf.input({ shape: [this.stateSize] });

        // Shared feature layers
        let shared = tf.layers.dense({
            units: 256,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }).apply(input);

        shared = tf.layers.dense({
            units: 256,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }).apply(shared);

        shared = tf.layers.dense({
            units: 128,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }).apply(shared);

        // Value stream - estimates V(s)
        let valueStream = tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }).apply(shared);

        const value = tf.layers.dense({
            units: 1,
            activation: 'linear',
            kernelInitializer: 'heNormal',
            name: 'value'
        }).apply(valueStream);

        // Advantage stream - estimates A(s,a)
        let advantageStream = tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }).apply(shared);

        const advantage = tf.layers.dense({
            units: this.actionSize,
            activation: 'linear',
            kernelInitializer: 'heNormal',
            name: 'advantage'
        }).apply(advantageStream);

        // Combine: Q(s,a) = V(s) + (A(s,a) - mean(A))
        // Using concatenation + custom layer for compatibility
        const combined = tf.layers.concatenate().apply([value, advantage]);
        const qValues = new DuelingCombineLayer({ actionSize: this.actionSize }).apply(combined);

        const model = tf.model({ inputs: input, outputs: qValues });

        model.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: 'meanSquaredError'
        });

        return model;
    }

    updateTargetNetwork() {
        const weights = this.model.getWeights();
        const targetWeights = weights.map(w => w.clone());
        this.targetModel.setWeights(targetWeights);
    }

    getState(player, ball, opponent, fieldWidth, fieldHeight) {
        const playerX = player.x / fieldWidth;
        const playerY = player.y / fieldHeight;
        const ballX = ball.x / fieldWidth;
        const ballY = ball.y / fieldHeight;

        const maxVel = 15;
        const ballVx = Math.max(-1, Math.min(1, ball.vx / maxVel));
        const ballVy = Math.max(-1, Math.min(1, ball.vy / maxVel));

        const opponentX = opponent.x / fieldWidth;
        const opponentY = opponent.y / fieldHeight;

        const distBall = this.distance(player, ball) / Math.sqrt(fieldWidth ** 2 + fieldHeight ** 2);
        const angleBall = (this.angle(player, ball) + Math.PI) / (2 * Math.PI);

        const goalX = this.team === 'blip' ? fieldWidth : 0;
        const goalY = fieldHeight / 2;
        const distGoal = Math.sqrt((player.x - goalX) ** 2 + (player.y - goalY) ** 2) /
            Math.sqrt(fieldWidth ** 2 + fieldHeight ** 2);
        const angleGoal = (Math.atan2(goalY - player.y, goalX - player.x) + Math.PI) / (2 * Math.PI);

        return [playerX, playerY, ballX, ballY, ballVx, ballVy, opponentX, opponentY, distBall, angleBall, distGoal, angleGoal];
    }

    distance(a, b) {
        return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
    }

    angle(a, b) {
        return Math.atan2(b.y - a.y, b.x - a.x);
    }

    chooseAction(state, training = true) {
        let actionIndex;

        if (training && Math.random() < this.epsilon) {
            actionIndex = Math.floor(Math.random() * this.actionSize);
        } else {
            const result = tf.tidy(() => {
                const stateTensor = tf.tensor2d([state], [1, this.stateSize]);
                const qValues = this.model.predict(stateTensor);
                return qValues.argMax(1).dataSync()[0];
            });
            actionIndex = result;
        }

        this.lastState = state;
        this.lastAction = actionIndex;

        return this.actions[actionIndex];
    }

    remember(state, action, reward, nextState, done) {
        this.replayBuffer.push({ state, action, reward, nextState, done });
        if (this.replayBuffer.length > this.bufferSize) {
            this.replayBuffer.shift();
        }
    }

    async train() {
        if (this.replayBuffer.length < this.minBufferSize) return;

        const batch = [];
        for (let i = 0; i < this.batchSize; i++) {
            const idx = Math.floor(Math.random() * this.replayBuffer.length);
            batch.push(this.replayBuffer[idx]);
        }

        const states = batch.map(e => e.state);
        const nextStates = batch.map(e => e.nextState);

        const { currentQsArray, nextQsMainArray, nextQsTargetArray } = tf.tidy(() => {
            const statesTensor = tf.tensor2d(states, [this.batchSize, this.stateSize]);
            const nextStatesTensor = tf.tensor2d(nextStates, [this.batchSize, this.stateSize]);
            const currentQs = this.model.predict(statesTensor);
            const nextQsMain = this.model.predict(nextStatesTensor);
            const nextQsTarget = this.targetModel.predict(nextStatesTensor);
            return {
                currentQsArray: currentQs.arraySync(),
                nextQsMainArray: nextQsMain.arraySync(),
                nextQsTargetArray: nextQsTarget.arraySync()
            };
        });

        for (let i = 0; i < this.batchSize; i++) {
            const { action, reward, done } = batch[i];
            if (done) {
                currentQsArray[i][action] = reward;
            } else {
                const bestAction = nextQsMainArray[i].indexOf(Math.max(...nextQsMainArray[i]));
                const nextQ = nextQsTargetArray[i][bestAction];
                currentQsArray[i][action] = reward + this.gamma * nextQ;
            }
        }

        const statesTensor = tf.tensor2d(states, [this.batchSize, this.stateSize]);
        const targetTensor = tf.tensor2d(currentQsArray, [this.batchSize, this.actionSize]);

        await this.model.fit(statesTensor, targetTensor, { epochs: 1, verbose: 0 });

        statesTensor.dispose();
        targetTensor.dispose();

        this.trainStepCount++;
        if (this.trainStepCount % this.targetUpdateFreq === 0) {
            this.updateTargetNetwork();
        }
    }

    calculateReward(player, ball, opponent, event, fieldWidth) {
        let reward = 0;
        const fieldHeight = 420;
        const distToBall = this.distance(player, ball);

        // === GOAL EVENTS (massive rewards) ===
        if (event === 'scored') {
            reward += 500;  // HUGE reward for scoring
        } else if (event === 'conceded') {
            reward -= 300;  // Big penalty for conceding
        }

        // === ALWAYS CHASE THE BALL (most important) ===
        // Continuous reward for being close to ball
        const maxDist = Math.sqrt(fieldWidth ** 2 + fieldHeight ** 2);
        const normalizedDist = distToBall / maxDist;
        reward += (1 - normalizedDist) * 5;  // Up to +5 for being at ball, 0 for far away

        // Big bonus for actually touching the ball
        if (distToBall < 40) {
            reward += 10;  // Ball possession is king!
        }

        // Reward for moving TOWARD the ball (not just being close)
        if (this.lastDistToBall !== null) {
            const distDelta = this.lastDistToBall - distToBall;  // Positive = getting closer
            reward += distDelta * 0.5;  // Continuous reward for approach

            // Extra bonus for reducing distance quickly
            if (distDelta > 2) {
                reward += 3;  // Fast approach bonus
            }
        }
        this.lastDistToBall = distToBall;

        // === MOVEMENT IS MANDATORY ===
        const playerSpeed = Math.sqrt(player.vx * player.vx + player.vy * player.vy);

        // Heavy penalty for standing still when not at the ball
        if (playerSpeed < 0.5 && distToBall > 50) {
            reward -= 8;  // MUCH stronger - don't just stand there!
        }

        // Bonus for moving (any movement is good)
        if (playerSpeed > 1) {
            reward += 1;  // Always reward movement
        }

        // === ATTACKING BONUSES ===
        const attackingGoalX = this.team === 'blip' ? fieldWidth : 0;

        // Ball moving toward opponent's goal
        const ballMovingTowardGoal = (this.team === 'blip' && ball.vx > 2) ||
            (this.team === 'bloop' && ball.vx < -2);
        if (ballMovingTowardGoal && distToBall < 80) {
            reward += 8;  // You probably caused this!
        }

        // Ball close to opponent's goal
        const distBallToGoal = Math.abs(ball.x - attackingGoalX);
        if (distBallToGoal < 100) {
            reward += 5;  // Dangerous position
        }

        // === PENALTIES ===
        // Corner penalty
        const cornerMargin = 80;
        const inCorner = (player.x < cornerMargin || player.x > fieldWidth - cornerMargin) &&
            (player.y < cornerMargin || player.y > fieldHeight - cornerMargin);
        if (inCorner) {
            reward -= 5;
            if (distToBall > 100) reward -= 5;  // Extra if corner is pointless
        }

        // Far from ball penalty (escalating)
        if (distToBall > 300) {
            reward -= 5;  // Way too far
        } else if (distToBall > 200) {
            reward -= 3;
        } else if (distToBall > 150) {
            reward -= 1;
        }

        // Opponent closer to ball than you = bad
        const opponentDistToBall = this.distance(opponent, ball);
        if (opponentDistToBall < distToBall && distToBall > 60) {
            reward -= 2;  // They're beating you to the ball!
        }

        // Small time penalty to encourage action
        reward -= 0.1;

        return reward;
    }

    reset() {
        this.lastState = null;
        this.lastAction = null;
        this.lastDistToBall = null;
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }

    async exportWeights() {
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

    async importWeights(data) {
        if (!data || !data.weights) return;
        const weights = data.weights.map(w => tf.tensor(w.data, w.shape));
        this.model.setWeights(weights);
        this.updateTargetNetwork();
        this.epsilon = data.epsilon || this.epsilon;
        this.trainStepCount = data.trainStepCount || 0;
        weights.forEach(w => w.dispose());
    }
}

module.exports = DQNAgent;
