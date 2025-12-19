/**
 * Player - Cute robot football player
 */
class Player {
    constructor(x, y, team, name) {
        this.x = x;
        this.y = y;
        this.startX = x;
        this.startY = y;
        this.team = team; // 'blip' or 'bloop'
        this.name = name;
        
        // Physics
        this.vx = 0;
        this.vy = 0;
        this.radius = 25;
        this.speed = 4;
        this.friction = 0.85;
        
        // Animation
        this.angle = 0;
        this.bobOffset = 0;
        this.expression = 'normal'; // normal, happy, sad, focused
        this.expressionTimer = 0;
        this.kickAnimation = 0;
        
        // Colors
        if (team === 'blip') {
            this.primaryColor = '#00d4ff';
            this.secondaryColor = '#0099cc';
            this.glowColor = 'rgba(0, 212, 255, 0.4)';
        } else {
            this.primaryColor = '#ff4d6d';
            this.secondaryColor = '#cc3355';
            this.glowColor = 'rgba(255, 77, 109, 0.4)';
        }
    }
    
    reset() {
        this.x = this.startX;
        this.y = this.startY;
        this.vx = 0;
        this.vy = 0;
        this.expression = 'normal';
    }
    
    update(dt = 1) {
        // Apply velocity
        this.x += this.vx * dt;
        this.y += this.vy * dt;
        
        // Apply friction
        this.vx *= this.friction;
        this.vy *= this.friction;
        
        // Animation updates
        this.bobOffset = Math.sin(Date.now() / 200) * 3;
        
        // Update facing angle based on movement
        if (Math.abs(this.vx) > 0.1 || Math.abs(this.vy) > 0.1) {
            this.angle = Math.atan2(this.vy, this.vx);
        }
        
        // Update expression timer
        if (this.expressionTimer > 0) {
            this.expressionTimer--;
            if (this.expressionTimer === 0) {
                this.expression = 'normal';
            }
        }
        
        // Kick animation decay
        if (this.kickAnimation > 0) {
            this.kickAnimation -= 0.2;
        }
    }
    
    move(dx, dy) {
        this.vx += dx * this.speed * 0.5;
        this.vy += dy * this.speed * 0.5;
        
        // Limit max speed
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
    
    setExpression(expr, duration = 60) {
        this.expression = expr;
        this.expressionTimer = duration;
    }
    
    draw(ctx) {
        ctx.save();
        ctx.translate(this.x, this.y + this.bobOffset);
        
        // Glow effect
        const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, this.radius * 1.5);
        gradient.addColorStop(0, this.glowColor);
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(0, 0, this.radius * 1.5, 0, Math.PI * 2);
        ctx.fill();
        
        // Shadow
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.beginPath();
        ctx.ellipse(0, this.radius - 5, this.radius * 0.8, 8, 0, 0, Math.PI * 2);
        ctx.fill();
        
        // Body
        const bodyGradient = ctx.createLinearGradient(-this.radius, -this.radius, this.radius, this.radius);
        bodyGradient.addColorStop(0, this.primaryColor);
        bodyGradient.addColorStop(1, this.secondaryColor);
        
        ctx.fillStyle = bodyGradient;
        ctx.beginPath();
        ctx.arc(0, 0, this.radius, 0, Math.PI * 2);
        ctx.fill();
        
        // Body shine
        ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.beginPath();
        ctx.arc(-8, -8, this.radius * 0.4, 0, Math.PI * 2);
        ctx.fill();
        
        // Face
        this.drawFace(ctx);
        
        // Antenna
        ctx.strokeStyle = this.secondaryColor;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(0, -this.radius);
        ctx.lineTo(0, -this.radius - 12);
        ctx.stroke();
        
        // Antenna ball
        ctx.fillStyle = this.primaryColor;
        ctx.beginPath();
        ctx.arc(0, -this.radius - 15, 5, 0, Math.PI * 2);
        ctx.fill();
        
        // Antenna glow
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.beginPath();
        ctx.arc(0, -this.radius - 15, 3, 0, Math.PI * 2);
        ctx.fill();
        
        // Kick effect
        if (this.kickAnimation > 0) {
            ctx.strokeStyle = `rgba(255, 255, 255, ${this.kickAnimation * 0.8})`;
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(0, 0, this.radius + 10 + (1 - this.kickAnimation) * 20, 0, Math.PI * 2);
            ctx.stroke();
        }
        
        ctx.restore();
    }
    
    drawFace(ctx) {
        // Eyes
        const eyeSpacing = 12;
        const eyeY = -5;
        
        // Eye whites
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(-eyeSpacing, eyeY, 8, 0, Math.PI * 2);
        ctx.arc(eyeSpacing, eyeY, 8, 0, Math.PI * 2);
        ctx.fill();
        
        // Pupils - look toward movement direction
        const lookX = Math.cos(this.angle) * 3;
        const lookY = Math.sin(this.angle) * 3;
        
        ctx.fillStyle = '#333';
        ctx.beginPath();
        ctx.arc(-eyeSpacing + lookX, eyeY + lookY, 4, 0, Math.PI * 2);
        ctx.arc(eyeSpacing + lookX, eyeY + lookY, 4, 0, Math.PI * 2);
        ctx.fill();
        
        // Eye shine
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(-eyeSpacing + lookX - 1, eyeY + lookY - 1, 1.5, 0, Math.PI * 2);
        ctx.arc(eyeSpacing + lookX - 1, eyeY + lookY - 1, 1.5, 0, Math.PI * 2);
        ctx.fill();
        
        // Mouth based on expression
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';
        
        switch (this.expression) {
            case 'happy':
                ctx.beginPath();
                ctx.arc(0, 8, 8, 0.2, Math.PI - 0.2);
                ctx.stroke();
                break;
            case 'sad':
                ctx.beginPath();
                ctx.arc(0, 18, 8, Math.PI + 0.3, -0.3);
                ctx.stroke();
                break;
            case 'focused':
                ctx.beginPath();
                ctx.moveTo(-6, 10);
                ctx.lineTo(6, 10);
                ctx.stroke();
                break;
            default:
                ctx.beginPath();
                ctx.arc(0, 10, 5, 0.1, Math.PI - 0.1);
                ctx.stroke();
        }
    }
    
    // Get bounds for collision
    getBounds() {
        return {
            left: this.x - this.radius,
            right: this.x + this.radius,
            top: this.y - this.radius,
            bottom: this.y + this.radius
        };
    }
    
    distanceTo(other) {
        const dx = other.x - this.x;
        const dy = other.y - this.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    angleTo(other) {
        return Math.atan2(other.y - this.y, other.x - this.x);
    }
}

// Export for use in other modules
window.Player = Player;
