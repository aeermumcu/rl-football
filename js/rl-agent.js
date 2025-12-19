/**
 * Q-Learning Agent for Football
 */
class RLAgent {
    constructor(name, team) {
        this.name = name;
        this.team = team;

        // Q-Learning parameters
        this.learningRate = 0.1;      // Alpha
        this.discountFactor = 0.95;   // Gamma
        this.epsilon = 1.0;           // Exploration rate
        this.epsilonDecay = 0.9995;
        this.epsilonMin = 0.05;

        // Q-Table: state -> action -> value
        this.qTable = new Map();

        // Actions: [dx, dy, kick]
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

        // Stats
        this.totalReward = 0;
    }

    /**
     * Discretize the game state into a string key
     */
    getState(player, ball, opponent, fieldWidth, fieldHeight, goalY) {
        // Discretize positions into grid cells
        const gridSize = 5; // 5x5 grid sections

        const playerGridX = Math.floor(player.x / fieldWidth * gridSize);
        const playerGridY = Math.floor(player.y / fieldHeight * gridSize);

        const ballGridX = Math.floor(ball.x / fieldWidth * gridSize);
        const ballGridY = Math.floor(ball.y / fieldHeight * gridSize);

        const opponentGridX = Math.floor(opponent.x / fieldWidth * gridSize);
        const opponentGridY = Math.floor(opponent.y / fieldHeight * gridSize);

        // Relative ball position (simplified)
        const ballDx = Math.sign(ball.x - player.x);
        const ballDy = Math.sign(ball.y - player.y);

        // Distance to ball (discretized)
        const distToBall = player.distanceTo(ball);
        const distCategory = distToBall < 50 ? 'close' : distToBall < 150 ? 'mid' : 'far';

        // Ball moving? And direction?
        const ballMoving = Math.abs(ball.vx) > 0.5 || Math.abs(ball.vy) > 0.5;
        const ballDir = ballMoving ? Math.sign(ball.vx) : 0;

        // Create state key
        return `${playerGridX},${playerGridY}|${ballGridX},${ballGridY}|${ballDx},${ballDy}|${distCategory}|${ballDir}|${opponentGridX}`;
    }

    /**
     * Get Q-value for a state-action pair
     */
    getQValue(state, actionIndex) {
        if (!this.qTable.has(state)) {
            this.qTable.set(state, new Array(this.actions.length).fill(0));
        }
        return this.qTable.get(state)[actionIndex];
    }

    /**
     * Set Q-value for a state-action pair
     */
    setQValue(state, actionIndex, value) {
        if (!this.qTable.has(state)) {
            this.qTable.set(state, new Array(this.actions.length).fill(0));
        }
        this.qTable.get(state)[actionIndex] = value;
    }

    /**
     * Get best action for a state
     */
    getBestAction(state) {
        if (!this.qTable.has(state)) {
            return Math.floor(Math.random() * this.actions.length);
        }

        const qValues = this.qTable.get(state);
        let bestValue = -Infinity;
        let bestActions = [];

        for (let i = 0; i < qValues.length; i++) {
            if (qValues[i] > bestValue) {
                bestValue = qValues[i];
                bestActions = [i];
            } else if (qValues[i] === bestValue) {
                bestActions.push(i);
            }
        }

        // Random tie-break
        return bestActions[Math.floor(Math.random() * bestActions.length)];
    }

    /**
     * Choose action using epsilon-greedy policy
     */
    chooseAction(state, training = true) {
        let actionIndex;

        if (training && Math.random() < this.epsilon) {
            // Explore: random action
            actionIndex = Math.floor(Math.random() * this.actions.length);
        } else {
            // Exploit: best known action
            actionIndex = this.getBestAction(state);
        }

        // Store for learning
        this.lastState = state;
        this.lastAction = actionIndex;

        return this.actions[actionIndex];
    }

    /**
     * Update Q-value based on reward
     */
    learn(reward, newState, done) {
        if (this.lastState === null || this.lastAction === null) return;

        const oldQ = this.getQValue(this.lastState, this.lastAction);

        let newQ;
        if (done) {
            newQ = oldQ + this.learningRate * (reward - oldQ);
        } else {
            // Get max Q-value for next state
            const maxNextQ = Math.max(...(this.qTable.get(newState) || new Array(this.actions.length).fill(0)));
            newQ = oldQ + this.learningRate * (reward + this.discountFactor * maxNextQ - oldQ);
        }

        this.setQValue(this.lastState, this.lastAction, newQ);
        this.totalReward += reward;

        // Decay epsilon
        if (done && this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }

    /**
     * Calculate reward based on game state
     */
    calculateReward(player, ball, opponent, event, fieldWidth) {
        let reward = 0;

        // Goal events
        if (event === 'scored') {
            reward += 100;
        } else if (event === 'conceded') {
            reward -= 100;
        }

        // Distance to ball reward
        const distToBall = player.distanceTo(ball);
        if (distToBall < 50) {
            reward += 2; // Close to ball is good
        } else if (distToBall < 100) {
            reward += 1;
        }

        // Moving toward ball
        const oldDist = this.lastDistToBall || distToBall;
        if (distToBall < oldDist) {
            reward += 0.5;
        }
        this.lastDistToBall = distToBall;

        // Ball moving toward opponent's goal (good for attacker)
        const targetGoalX = this.team === 'blip' ? fieldWidth : 0;
        const ballMovingTowardGoal = (this.team === 'blip' && ball.vx > 0) ||
            (this.team === 'bloop' && ball.vx < 0);
        if (ballMovingTowardGoal && distToBall < 80) {
            reward += 1;
        }

        // Small time penalty to encourage action
        reward -= 0.1;

        // Corner penalty - discourage getting stuck in corners
        const fieldHeight = 420; // approximate field height
        const cornerMargin = 60;
        const inCorner = (player.x < cornerMargin || player.x > fieldWidth - cornerMargin) &&
            (player.y < cornerMargin || player.y > fieldHeight - cornerMargin);
        if (inCorner) {
            reward -= 2; // Strong penalty for being in corners
        }

        // Penalty for being far from ball when ball is far from you
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
        this.totalReward = 0;
    }

    /**
     * Get Q-table size
     */
    getQTableSize() {
        return this.qTable.size;
    }

    /**
     * Get current epsilon
     */
    getEpsilon() {
        return this.epsilon;
    }

    /**
     * Export Q-table for saving
     */
    exportQTable() {
        const obj = {};
        this.qTable.forEach((values, key) => {
            obj[key] = values;
        });
        return {
            qTable: obj,
            epsilon: this.epsilon
        };
    }

    /**
     * Import Q-table from saved data
     */
    importQTable(data) {
        this.qTable.clear();
        for (const key in data.qTable) {
            this.qTable.set(key, data.qTable[key]);
        }
        this.epsilon = data.epsilon || this.epsilon;
    }
}

// Export
window.RLAgent = RLAgent;
