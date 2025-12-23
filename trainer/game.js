/**
 * Simplified Game Simulation for Headless Training
 * No rendering - just physics and game logic
 */

class Player {
    constructor(x, y, team) {
        this.x = x;
        this.y = y;
        this.startX = x;
        this.startY = y;
        this.team = team;
        this.vx = 0;
        this.vy = 0;
        this.radius = 25;
        this.speed = 4;
        this.friction = 0.85;
    }

    reset() {
        this.x = this.startX;
        this.y = this.startY;
        this.vx = 0;
        this.vy = 0;
    }

    update(dt = 1) {
        this.x += this.vx * dt;
        this.y += this.vy * dt;
        this.vx *= this.friction;
        this.vy *= this.friction;
    }

    move(dx, dy) {
        this.vx += dx * this.speed * 0.5;
        this.vy += dy * this.speed * 0.5;
        const maxSpeed = this.speed;
        const currentSpeed = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
        if (currentSpeed > maxSpeed) {
            this.vx = (this.vx / currentSpeed) * maxSpeed;
            this.vy = (this.vy / currentSpeed) * maxSpeed;
        }
    }

    kick() {
        this.kickAnimation = 1;
    }
}

class Game {
    constructor() {
        this.width = 800;
        this.height = 500;
        this.padding = 40;

        this.fieldLeft = this.padding;
        this.fieldRight = this.width - this.padding;
        this.fieldTop = this.padding;
        this.fieldBottom = this.height - this.padding;
        this.fieldWidth = this.fieldRight - this.fieldLeft;
        this.fieldHeight = this.fieldBottom - this.fieldTop;

        this.goalWidth = 15;
        this.goalHeight = 120;
        this.goalY = this.height / 2;

        this.blip = new Player(this.fieldLeft + 80, this.height / 2, 'blip');
        this.bloop = new Player(this.fieldRight - 80, this.height / 2, 'bloop');

        this.ball = {
            x: this.width / 2,
            y: this.height / 2,
            vx: 0,
            vy: 0,
            radius: 12,
            startX: this.width / 2,
            startY: this.height / 2
        };

        this.blipScore = 0;
        this.bloopScore = 0;
        this.matchTime = 30;
        this.timeRemaining = this.matchTime;
    }

    reset() {
        this.blip.reset();
        this.bloop.reset();
        this.ball.x = this.ball.startX;
        this.ball.y = this.ball.startY;
        this.ball.vx = 0;
        this.ball.vy = 0;
        this.timeRemaining = this.matchTime;
    }

    resetScores() {
        this.blipScore = 0;
        this.bloopScore = 0;
    }

    update(dt = 1) {
        this.timeRemaining -= dt / 60;

        this.blip.update(dt);
        this.bloop.update(dt);
        this.updateBall(dt);
        this.checkPlayerBallCollision(this.blip);
        this.checkPlayerBallCollision(this.bloop);
        this.constrainPlayer(this.blip);
        this.constrainPlayer(this.bloop);

        const goalEvent = this.checkGoal();
        const done = this.timeRemaining <= 0;

        return { event: goalEvent, done };
    }

    updateBall(dt) {
        this.ball.x += this.ball.vx * dt;
        this.ball.y += this.ball.vy * dt;
        this.ball.vx *= 0.98;
        this.ball.vy *= 0.98;

        if (this.ball.y - this.ball.radius < this.fieldTop) {
            this.ball.y = this.fieldTop + this.ball.radius;
            this.ball.vy *= -0.8;
        }
        if (this.ball.y + this.ball.radius > this.fieldBottom) {
            this.ball.y = this.fieldBottom - this.ball.radius;
            this.ball.vy *= -0.8;
        }

        const inGoalRange = this.ball.y > this.goalY - this.goalHeight / 2 &&
            this.ball.y < this.goalY + this.goalHeight / 2;

        if (!inGoalRange) {
            if (this.ball.x - this.ball.radius < this.fieldLeft) {
                this.ball.x = this.fieldLeft + this.ball.radius;
                this.ball.vx *= -0.8;
            }
            if (this.ball.x + this.ball.radius > this.fieldRight) {
                this.ball.x = this.fieldRight - this.ball.radius;
                this.ball.vx *= -0.8;
            }
        }
    }

    checkPlayerBallCollision(player) {
        const dx = this.ball.x - player.x;
        const dy = this.ball.y - player.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const minDist = player.radius + this.ball.radius;

        if (dist < minDist && dist > 0) {
            const nx = dx / dist;
            const ny = dy / dist;
            this.ball.x = player.x + nx * minDist;
            this.ball.y = player.y + ny * minDist;
            const kickPower = player.kickAnimation > 0 ? 12 : 6;
            this.ball.vx = nx * kickPower + player.vx * 0.5;
            this.ball.vy = ny * kickPower + player.vy * 0.5;
            player.kickAnimation = 0;  // Clear kick after use
            return true;
        }
        return false;
    }

    constrainPlayer(player) {
        const margin = player.radius;
        if (player.x - margin < this.fieldLeft) { player.x = this.fieldLeft + margin; player.vx = 0; }
        if (player.x + margin > this.fieldRight) { player.x = this.fieldRight - margin; player.vx = 0; }
        if (player.y - margin < this.fieldTop) { player.y = this.fieldTop + margin; player.vy = 0; }
        if (player.y + margin > this.fieldBottom) { player.y = this.fieldBottom - margin; player.vy = 0; }
    }

    checkGoal() {
        const inGoalY = this.ball.y > this.goalY - this.goalHeight / 2 &&
            this.ball.y < this.goalY + this.goalHeight / 2;

        if (inGoalY && this.ball.x < this.fieldLeft) {
            this.bloopScore++;
            this.resetBallAfterGoal();
            return 'bloop_scored';
        }

        if (inGoalY && this.ball.x > this.fieldRight) {
            this.blipScore++;
            this.resetBallAfterGoal();
            return 'blip_scored';
        }

        return null;
    }

    resetBallAfterGoal() {
        this.ball.x = this.ball.startX;
        this.ball.y = this.ball.startY;
        this.ball.vx = 0;
        this.ball.vy = 0;
        this.blip.x = this.blip.startX;
        this.blip.y = this.blip.startY;
        this.bloop.x = this.bloop.startX;
        this.bloop.y = this.bloop.startY;
    }

    applyAction(player, action) {
        if (action.dx !== 0 || action.dy !== 0) {
            player.move(action.dx, action.dy);
        }
        if (action.kick) {
            player.kick();
        }
    }

    getWinner() {
        if (this.blipScore > this.bloopScore) return 'blip';
        if (this.bloopScore > this.blipScore) return 'bloop';
        return 'draw';
    }
}

module.exports = { Game, Player };
