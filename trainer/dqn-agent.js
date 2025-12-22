/**
 * Headless DQN Agent for Node.js Training
 * Compatible with browser version - exports weights in same format
 */
const tf = require('@tensorflow/tfjs-node');

class DQNAgent {
    constructor(name, team) {
        this.name = name;
        this.team = team;

        // State and action dimensions
        this.stateSize = 12;
        this.actionSize = 10;

        // Hyperparameters
        this.learningRate = 0.001;
        this.gamma = 0.99;
        this.epsilon = 1.0;
        this.epsilonMin = 0.05;
        this.epsilonDecay = 0.9995;  // Slower decay for more exploration

        // Experience replay
        this.replayBuffer = [];
        this.bufferSize = 10000;
        this.batchSize = 32;
        this.minBufferSize = 100;

        // Target network update frequency
        this.targetUpdateFreq = 100;
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

    createNetwork() {
        const model = tf.sequential();

        model.add(tf.layers.dense({
            inputShape: [this.stateSize],
            units: 256,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        model.add(tf.layers.dense({
            units: this.actionSize,
            activation: 'linear',
            kernelInitializer: 'heNormal'
        }));

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

        if (event === 'scored') reward += 200;
        else if (event === 'conceded') reward -= 150;

        const distToBall = this.distance(player, ball);
        if (distToBall < 40) reward += 3;
        else if (distToBall < 80) reward += 1.5;
        else if (distToBall < 150) reward += 0.5;

        if (this.lastDistToBall !== null && distToBall < this.lastDistToBall) {
            reward += 0.8;
        }
        this.lastDistToBall = distToBall;

        const attackingGoalX = this.team === 'blip' ? fieldWidth : 0;
        const ballInAttackHalf = this.team === 'blip' ? ball.x > fieldWidth / 2 : ball.x < fieldWidth / 2;
        if (ballInAttackHalf && distToBall < 100) reward += 2;

        const ballMovingTowardGoal = (this.team === 'blip' && ball.vx > 2) || (this.team === 'bloop' && ball.vx < -2);
        if (ballMovingTowardGoal) {
            reward += 3;
            if (distToBall < 60) reward += 2;
        }

        const distBallToGoal = Math.abs(ball.x - attackingGoalX);
        if (distBallToGoal < 150) reward += 3;
        else if (distBallToGoal < 250) reward += 1;

        reward -= 0.05;

        const cornerMargin = 80;
        const inCorner = (player.x < cornerMargin || player.x > fieldWidth - cornerMargin) &&
            (player.y < cornerMargin || player.y > fieldHeight - cornerMargin);
        if (inCorner) {
            reward -= 10;  // Much stronger corner penalty
            if (distToBall > 100) reward -= 5;
        }

        // Midfield positioning bonus - encourage being in the middle of the field
        const centerX = fieldWidth / 2;
        const centerY = fieldHeight / 2;
        const distFromCenter = Math.sqrt((player.x - centerX) ** 2 + (player.y - centerY) ** 2);
        const maxDist = Math.sqrt(centerX ** 2 + centerY ** 2);
        if (distFromCenter < maxDist * 0.4) {
            reward += 0.5;  // Small bonus for being in central area
        }

        if (distToBall > 250) reward -= 1.5;
        else if (distToBall > 180) reward -= 0.8;

        const playerSpeed = Math.sqrt(player.vx * player.vx + player.vy * player.vy);
        if (playerSpeed < 0.5 && distToBall > 60) reward -= 2;
        if (playerSpeed > 1 && this.lastDistToBall !== null && distToBall < this.lastDistToBall) {
            reward += 1;
        }

        const ballMovingTowardOwnGoal = (this.team === 'blip' && ball.vx < -2) || (this.team === 'bloop' && ball.vx > 2);
        if (ballMovingTowardOwnGoal && distToBall < 100) reward -= 2;

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
