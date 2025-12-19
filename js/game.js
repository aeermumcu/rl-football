/**
 * Football Game Engine
 */
class Game {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        // Field dimensions
        this.width = canvas.width;
        this.height = canvas.height;
        this.padding = 40;

        // Field area
        this.fieldLeft = this.padding;
        this.fieldRight = this.width - this.padding;
        this.fieldTop = this.padding;
        this.fieldBottom = this.height - this.padding;
        this.fieldWidth = this.fieldRight - this.fieldLeft;
        this.fieldHeight = this.fieldBottom - this.fieldTop;

        // Goal setup
        this.goalWidth = 15;
        this.goalHeight = 120;
        this.goalY = this.height / 2;

        // Create players
        this.blip = new Player(this.fieldLeft + 80, this.height / 2, 'blip', 'Blip');
        this.bloop = new Player(this.fieldRight - 80, this.height / 2, 'bloop', 'Bloop');

        // Create ball
        this.ball = {
            x: this.width / 2,
            y: this.height / 2,
            vx: 0,
            vy: 0,
            radius: 12,
            startX: this.width / 2,
            startY: this.height / 2
        };

        // Scores
        this.blipScore = 0;
        this.bloopScore = 0;

        // Match settings
        this.matchTime = 60; // seconds per match (adjustable via UI)
        this.timeRemaining = this.matchTime;
        this.lastTime = Date.now();

        // Particles for effects
        this.particles = [];

        // Goal event tracking
        this.goalScored = null;
        this.goalCelebration = 0;

        // Field pattern
        this.grassPattern = this.createGrassPattern();
    }

    createGrassPattern() {
        const patternCanvas = document.createElement('canvas');
        patternCanvas.width = 40;
        patternCanvas.height = 40;
        const pCtx = patternCanvas.getContext('2d');

        // Base grass color
        pCtx.fillStyle = '#2d8a4e';
        pCtx.fillRect(0, 0, 40, 40);

        // Lighter stripe
        pCtx.fillStyle = '#35a05a';
        pCtx.fillRect(0, 0, 20, 40);

        return this.ctx.createPattern(patternCanvas, 'repeat');
    }

    reset() {
        // Reset positions
        this.blip.reset();
        this.bloop.reset();

        // Reset ball with random direction
        this.ball.x = this.ball.startX;
        this.ball.y = this.ball.startY;
        this.ball.vx = 0;
        this.ball.vy = 0;

        // Reset time
        this.timeRemaining = this.matchTime;
        this.lastTime = Date.now();

        // Clear goal event
        this.goalScored = null;
        this.goalCelebration = 0;
    }

    resetScores() {
        this.blipScore = 0;
        this.bloopScore = 0;
    }

    update(dt = 1) {
        // Update timer
        const now = Date.now();
        const deltaSeconds = (now - this.lastTime) / 1000;
        this.lastTime = now;
        this.timeRemaining = Math.max(0, this.timeRemaining - deltaSeconds * dt);

        // Goal celebration pause
        if (this.goalCelebration > 0) {
            this.goalCelebration -= dt;
            this.updateParticles(dt);
            return { event: null, done: false };
        }

        // Update players
        this.blip.update(dt);
        this.bloop.update(dt);

        // Update ball
        this.updateBall(dt);

        // Check collisions
        this.checkPlayerBallCollision(this.blip);
        this.checkPlayerBallCollision(this.bloop);

        // Check wall collisions for players
        this.constrainPlayer(this.blip);
        this.constrainPlayer(this.bloop);

        // Check goal
        const goalEvent = this.checkGoal();

        // Update particles
        this.updateParticles(dt);

        // Check if match ended
        const done = this.timeRemaining <= 0;

        return { event: goalEvent, done };
    }

    updateBall(dt) {
        // Apply velocity
        this.ball.x += this.ball.vx * dt;
        this.ball.y += this.ball.vy * dt;

        // Friction
        this.ball.vx *= 0.98;
        this.ball.vy *= 0.98;

        // Bounce off walls (top/bottom)
        if (this.ball.y - this.ball.radius < this.fieldTop) {
            this.ball.y = this.fieldTop + this.ball.radius;
            this.ball.vy *= -0.8;
            this.addSparks(this.ball.x, this.ball.y, 3);
        }
        if (this.ball.y + this.ball.radius > this.fieldBottom) {
            this.ball.y = this.fieldBottom - this.ball.radius;
            this.ball.vy *= -0.8;
            this.addSparks(this.ball.x, this.ball.y, 3);
        }

        // Bounce off side walls (if not in goal area)
        const inGoalRange = this.ball.y > this.goalY - this.goalHeight / 2 &&
            this.ball.y < this.goalY + this.goalHeight / 2;

        if (!inGoalRange) {
            if (this.ball.x - this.ball.radius < this.fieldLeft) {
                this.ball.x = this.fieldLeft + this.ball.radius;
                this.ball.vx *= -0.8;
                this.addSparks(this.ball.x, this.ball.y, 3);
            }
            if (this.ball.x + this.ball.radius > this.fieldRight) {
                this.ball.x = this.fieldRight - this.ball.radius;
                this.ball.vx *= -0.8;
                this.addSparks(this.ball.x, this.ball.y, 3);
            }
        }
    }

    checkPlayerBallCollision(player) {
        const dx = this.ball.x - player.x;
        const dy = this.ball.y - player.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const minDist = player.radius + this.ball.radius;

        if (dist < minDist && dist > 0) {
            // Collision! Push ball away
            const nx = dx / dist;
            const ny = dy / dist;

            // Move ball out of player
            this.ball.x = player.x + nx * minDist;
            this.ball.y = player.y + ny * minDist;

            // Apply kick force
            const kickPower = player.kickAnimation > 0 ? 12 : 6;
            this.ball.vx = nx * kickPower + player.vx * 0.5;
            this.ball.vy = ny * kickPower + player.vy * 0.5;

            // Effects
            this.addSparks(this.ball.x, this.ball.y, 5);
            player.kick();

            return true;
        }
        return false;
    }

    constrainPlayer(player) {
        const margin = player.radius;

        if (player.x - margin < this.fieldLeft) {
            player.x = this.fieldLeft + margin;
            player.vx = 0;
        }
        if (player.x + margin > this.fieldRight) {
            player.x = this.fieldRight - margin;
            player.vx = 0;
        }
        if (player.y - margin < this.fieldTop) {
            player.y = this.fieldTop + margin;
            player.vy = 0;
        }
        if (player.y + margin > this.fieldBottom) {
            player.y = this.fieldBottom - margin;
            player.vy = 0;
        }
    }

    checkGoal() {
        const inGoalY = this.ball.y > this.goalY - this.goalHeight / 2 &&
            this.ball.y < this.goalY + this.goalHeight / 2;

        // Ball in left goal (Bloop scores)
        if (inGoalY && this.ball.x < this.fieldLeft) {
            this.bloopScore++;
            this.goalScored = 'bloop';
            this.bloop.setExpression('happy', 90);
            this.blip.setExpression('sad', 90);
            this.addGoalCelebration(this.fieldLeft, this.goalY, '#ff4d6d');
            this.goalCelebration = 60;
            this.resetBallAfterGoal();
            return 'bloop_scored';
        }

        // Ball in right goal (Blip scores)  
        if (inGoalY && this.ball.x > this.fieldRight) {
            this.blipScore++;
            this.goalScored = 'blip';
            this.blip.setExpression('happy', 90);
            this.bloop.setExpression('sad', 90);
            this.addGoalCelebration(this.fieldRight, this.goalY, '#00d4ff');
            this.goalCelebration = 60;
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

        // Reset player positions
        this.blip.x = this.blip.startX;
        this.blip.y = this.blip.startY;
        this.bloop.x = this.bloop.startX;
        this.bloop.y = this.bloop.startY;
    }

    addSparks(x, y, count) {
        for (let i = 0; i < count; i++) {
            this.particles.push({
                x, y,
                vx: (Math.random() - 0.5) * 8,
                vy: (Math.random() - 0.5) * 8,
                life: 1,
                size: 3 + Math.random() * 3,
                color: '#fff'
            });
        }
    }

    addGoalCelebration(x, y, color) {
        for (let i = 0; i < 30; i++) {
            const angle = (Math.PI * 2 / 30) * i;
            this.particles.push({
                x, y,
                vx: Math.cos(angle) * (5 + Math.random() * 5),
                vy: Math.sin(angle) * (5 + Math.random() * 5),
                life: 1,
                size: 4 + Math.random() * 4,
                color
            });
        }
    }

    updateParticles(dt) {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.vx *= 0.95;
            p.vy *= 0.95;
            p.life -= 0.03 * dt;

            if (p.life <= 0) {
                this.particles.splice(i, 1);
            }
        }
    }

    draw() {
        const ctx = this.ctx;

        // Clear canvas
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, this.width, this.height);

        // Draw field
        this.drawField();

        // Draw goals
        this.drawGoals();

        // Draw particles (behind players)
        this.drawParticles();

        // Draw ball
        this.drawBall();

        // Draw players
        this.blip.draw(ctx);
        this.bloop.draw(ctx);

        // Draw goal celebration overlay
        if (this.goalCelebration > 0) {
            this.drawGoalOverlay();
        }
    }

    drawField() {
        const ctx = this.ctx;

        // Field background with grass pattern
        ctx.save();
        ctx.fillStyle = this.grassPattern;
        ctx.fillRect(this.fieldLeft, this.fieldTop, this.fieldWidth, this.fieldHeight);

        // Field border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = 3;
        ctx.strokeRect(this.fieldLeft, this.fieldTop, this.fieldWidth, this.fieldHeight);

        // Center line
        ctx.beginPath();
        ctx.moveTo(this.width / 2, this.fieldTop);
        ctx.lineTo(this.width / 2, this.fieldBottom);
        ctx.stroke();

        // Center circle
        ctx.beginPath();
        ctx.arc(this.width / 2, this.height / 2, 60, 0, Math.PI * 2);
        ctx.stroke();

        // Center dot
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.beginPath();
        ctx.arc(this.width / 2, this.height / 2, 5, 0, Math.PI * 2);
        ctx.fill();

        // Penalty areas
        const penaltyWidth = 80;
        const penaltyHeight = 160;

        // Left penalty area
        ctx.strokeRect(
            this.fieldLeft,
            this.goalY - penaltyHeight / 2,
            penaltyWidth,
            penaltyHeight
        );

        // Right penalty area
        ctx.strokeRect(
            this.fieldRight - penaltyWidth,
            this.goalY - penaltyHeight / 2,
            penaltyWidth,
            penaltyHeight
        );

        ctx.restore();
    }

    drawGoals() {
        const ctx = this.ctx;

        // Left goal (Blip's goal - Bloop tries to score here)
        ctx.fillStyle = 'rgba(255, 77, 109, 0.3)';
        ctx.fillRect(
            this.fieldLeft - this.goalWidth,
            this.goalY - this.goalHeight / 2,
            this.goalWidth,
            this.goalHeight
        );
        ctx.strokeStyle = '#ff4d6d';
        ctx.lineWidth = 4;
        ctx.strokeRect(
            this.fieldLeft - this.goalWidth,
            this.goalY - this.goalHeight / 2,
            this.goalWidth,
            this.goalHeight
        );

        // Right goal (Bloop's goal - Blip tries to score here)
        ctx.fillStyle = 'rgba(0, 212, 255, 0.3)';
        ctx.fillRect(
            this.fieldRight,
            this.goalY - this.goalHeight / 2,
            this.goalWidth,
            this.goalHeight
        );
        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth = 4;
        ctx.strokeRect(
            this.fieldRight,
            this.goalY - this.goalHeight / 2,
            this.goalWidth,
            this.goalHeight
        );
    }

    drawBall() {
        const ctx = this.ctx;
        const ball = this.ball;

        // Ball shadow
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.beginPath();
        ctx.ellipse(ball.x + 3, ball.y + 5, ball.radius * 0.8, ball.radius * 0.4, 0, 0, Math.PI * 2);
        ctx.fill();

        // Ball gradient
        const gradient = ctx.createRadialGradient(
            ball.x - 4, ball.y - 4, 0,
            ball.x, ball.y, ball.radius
        );
        gradient.addColorStop(0, '#ffffff');
        gradient.addColorStop(0.8, '#e0e0e0');
        gradient.addColorStop(1, '#b0b0b0');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
        ctx.fill();

        // Ball pattern (pentagon shapes)
        ctx.fillStyle = '#333';
        const hexRadius = ball.radius * 0.4;
        for (let i = 0; i < 5; i++) {
            const angle = (Math.PI * 2 / 5) * i + Math.atan2(ball.vy, ball.vx);
            const hx = ball.x + Math.cos(angle) * ball.radius * 0.5;
            const hy = ball.y + Math.sin(angle) * ball.radius * 0.5;
            ctx.beginPath();
            ctx.arc(hx, hy, hexRadius * 0.5, 0, Math.PI * 2);
            ctx.fill();
        }

        // Motion blur effect when moving fast
        const speed = Math.sqrt(ball.vx * ball.vx + ball.vy * ball.vy);
        if (speed > 5) {
            ctx.strokeStyle = `rgba(255, 255, 255, ${Math.min(0.3, speed / 30)})`;
            ctx.lineWidth = ball.radius * 2;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(ball.x, ball.y);
            ctx.lineTo(ball.x - ball.vx * 2, ball.y - ball.vy * 2);
            ctx.stroke();
        }
    }

    drawParticles() {
        const ctx = this.ctx;
        for (const p of this.particles) {
            ctx.globalAlpha = p.life;
            ctx.fillStyle = p.color;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.globalAlpha = 1;
    }

    drawGoalOverlay() {
        const ctx = this.ctx;
        const alpha = this.goalCelebration / 60;

        ctx.fillStyle = this.goalScored === 'blip'
            ? `rgba(0, 212, 255, ${alpha * 0.2})`
            : `rgba(255, 77, 109, ${alpha * 0.2})`;
        ctx.fillRect(0, 0, this.width, this.height);

        // GOAL! text
        ctx.save();
        ctx.font = 'bold 60px Outfit';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = this.goalScored === 'blip' ? '#00d4ff' : '#ff4d6d';
        ctx.shadowColor = ctx.fillStyle;
        ctx.shadowBlur = 20;
        ctx.globalAlpha = alpha;
        ctx.fillText('GOAL!', this.width / 2, this.height / 2);
        ctx.restore();
    }

    // Get state for AI
    getState(forPlayer) {
        const player = forPlayer === 'blip' ? this.blip : this.bloop;
        const opponent = forPlayer === 'blip' ? this.bloop : this.blip;
        return { player, opponent, ball: this.ball };
    }

    // Apply action to player
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

    getTimeString() {
        const seconds = Math.ceil(this.timeRemaining);
        return `0:${seconds.toString().padStart(2, '0')}`;
    }
}

// Export
window.Game = Game;
