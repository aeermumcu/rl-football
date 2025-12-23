/**
 * Simple Rule-Based AI - Always chases ball aggressively
 * Used as training opponent to teach DQN agents active play
 */

class SimpleAI {
    constructor(name, team) {
        this.name = name;
        this.team = team;

        // Same action format as DQN
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

        this.lastAction = 0;
    }

    /**
     * Smart ball-chasing behavior:
     * - Chase ball when far
     * - When close, push ball toward opponent's goal
     */
    chooseAction(state, training = true) {
        // State format: [playerX, playerY, ballX, ballY, ballVx, ballVy, opponentX, opponentY, distBall, angleBall, distGoal, angleGoal]
        const playerX = state[0];
        const playerY = state[1];
        const ballX = state[2];
        const ballY = state[3];

        // Distance to ball (normalized, 0 = at ball, 1 = max distance)
        const distToBall = state[8];

        // Calculate direction to ball
        const dxToBall = ballX - playerX;
        const dyToBall = ballY - playerY;

        // If very close to ball, push it toward opponent's goal
        if (distToBall < 0.08) {
            // Bloop attacks left goal (x=0), so push left
            // This is a simple heuristic - move in direction of goal while kicking
            const goalDir = this.team === 'bloop' ? -1 : 1;

            // Kick if at ball
            if (distToBall < 0.04) {
                this.lastAction = 8;  // kick
                return this.actions[8];
            }

            // Push toward goal - primarily horizontal
            let moveDx = goalDir;
            let moveDy = dyToBall > 0.01 ? 1 : (dyToBall < -0.01 ? -1 : 0);

            // Find matching action (prioritize diagonal toward goal)
            for (let i = 0; i < 8; i++) {
                if (this.actions[i].dx === moveDx && this.actions[i].dy === moveDy) {
                    this.lastAction = i;
                    return this.actions[i];
                }
            }
        }

        // Otherwise, chase the ball
        let moveDx = 0, moveDy = 0;
        const threshold = 0.02;

        if (Math.abs(dxToBall) > threshold) {
            moveDx = dxToBall > 0 ? 1 : -1;
        }
        if (Math.abs(dyToBall) > threshold) {
            moveDy = dyToBall > 0 ? 1 : -1;
        }

        // Find matching action
        for (let i = 0; i < 8; i++) {
            if (this.actions[i].dx === moveDx && this.actions[i].dy === moveDy) {
                this.lastAction = i;
                return this.actions[i];
            }
        }

        // Fallback: move toward ball center
        this.lastAction = 3;  // right
        return this.actions[3];
    }

    // Compatibility methods (no-op for simple AI)
    getState(player, ball, opponent, fieldWidth, fieldHeight) {
        const playerX = player.x / fieldWidth;
        const playerY = player.y / fieldHeight;
        const ballX = ball.x / fieldWidth;
        const ballY = ball.y / fieldHeight;
        const ballVx = ball.vx / 15;
        const ballVy = ball.vy / 15;
        const oppX = opponent.x / fieldWidth;
        const oppY = opponent.y / fieldHeight;
        const distBall = this.distance(player, ball) / Math.sqrt(fieldWidth ** 2 + fieldHeight ** 2);
        const angleBall = (this.angle(player, ball) + Math.PI) / (2 * Math.PI);
        const goalX = this.team === 'blip' ? fieldWidth : 0;
        const goalY = fieldHeight / 2;
        const distGoal = Math.sqrt((player.x - goalX) ** 2 + (player.y - goalY) ** 2) /
            Math.sqrt(fieldWidth ** 2 + fieldHeight ** 2);
        const angleGoal = (Math.atan2(goalY - player.y, goalX - player.x) + Math.PI) / (2 * Math.PI);

        return [playerX, playerY, ballX, ballY, ballVx, ballVy, oppX, oppY, distBall, angleBall, distGoal, angleGoal];
    }

    distance(a, b) {
        return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
    }

    angle(a, b) {
        return Math.atan2(b.y - a.y, b.x - a.x);
    }

    remember() { }  // No-op
    async train() { }  // No-op
    reset() { }  // No-op
    calculateReward() { return 0; }  // No-op
    async exportWeights() { return null; }  // No-op
    async importWeights() { }  // No-op
    getEpsilon() { return 0; }  // Always exploit
    getBufferSize() { return 0; }
}

module.exports = SimpleAI;
