/**
 * Training Visualizer - Charts and Stats
 */
class Visualizer {
    constructor(chartCanvas) {
        this.canvas = chartCanvas;
        this.ctx = chartCanvas.getContext('2d');

        // History for charts
        this.blipWinHistory = [];
        this.bloopWinHistory = [];
        this.historySize = 50;

        // Colors
        this.blipColor = '#00d4ff';
        this.bloopColor = '#ff4d6d';
        this.gridColor = 'rgba(255, 255, 255, 0.1)';
    }

    addResult(winner) {
        // Add to history (1 for win, 0 for loss/draw)
        this.blipWinHistory.push(winner === 'blip' ? 1 : 0);
        this.bloopWinHistory.push(winner === 'bloop' ? 1 : 0);

        // Limit history size
        if (this.blipWinHistory.length > this.historySize) {
            this.blipWinHistory.shift();
            this.bloopWinHistory.shift();
        }
    }

    getWinRate(history) {
        if (history.length === 0) return 0;
        return history.reduce((a, b) => a + b, 0) / history.length;
    }

    getRunningWinRates(history) {
        const rates = [];
        let sum = 0;
        for (let i = 0; i < history.length; i++) {
            sum += history[i];
            rates.push(sum / (i + 1));
        }
        return rates;
    }

    draw() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // Clear
        ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
        ctx.fillRect(0, 0, w, h);

        // Grid lines
        ctx.strokeStyle = this.gridColor;
        ctx.lineWidth = 1;

        // Horizontal grid (0%, 50%, 100%)
        for (let i = 0; i <= 2; i++) {
            const y = (h / 2) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        if (this.blipWinHistory.length < 2) {
            // Show placeholder text
            ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.font = '12px Outfit';
            ctx.textAlign = 'center';
            ctx.fillText('Training data will appear here...', w / 2, h / 2);
            return;
        }

        // Calculate running win rates
        const blipRates = this.getRunningWinRates(this.blipWinHistory);
        const bloopRates = this.getRunningWinRates(this.bloopWinHistory);

        // Draw lines
        this.drawLine(blipRates, this.blipColor, 3);
        this.drawLine(bloopRates, this.bloopColor, 3);
    }

    drawLine(data, color, lineWidth) {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        const padding = 5;

        if (data.length < 2) return;

        const stepX = (w - padding * 2) / (this.historySize - 1);

        // Line
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        ctx.beginPath();
        for (let i = 0; i < data.length; i++) {
            const x = padding + i * stepX;
            const y = h - padding - (data[i] * (h - padding * 2));

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();

        // Glow effect
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth + 4;
        ctx.globalAlpha = 0.2;
        ctx.stroke();
        ctx.globalAlpha = 1;

        // End point dot
        if (data.length > 0) {
            const lastX = padding + (data.length - 1) * stepX;
            const lastY = h - padding - (data[data.length - 1] * (h - padding * 2));

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = 'white';
            ctx.beginPath();
            ctx.arc(lastX, lastY, 2, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    reset() {
        this.blipWinHistory = [];
        this.bloopWinHistory = [];
        this.draw();
    }
}

// Export
window.Visualizer = Visualizer;
