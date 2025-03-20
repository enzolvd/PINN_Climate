// Create a WandbStyle interactive chart handler
class WandbChart {
    constructor(containerId, data, options = {}) {
        this.container = document.getElementById(containerId);
        this.data = data;
        this.options = Object.assign({
            width: 100,
            height: 400,
            margin: { top: 20, right: 30, bottom: 50, left: 60 },
            xAxis: 'step',
            colors: ['#5b8a72', '#3a506b', '#c45b64', '#8854d0', '#2e86c1'],
            title: 'Model Performance Comparison',
            yAxisLabel: 'MSE Loss',
            xAxisLabel: 'Training Step',
            showLegend: true,
            smoothing: 0.1,
            showTooltip: true,
            showGrid: true,
            animation: true
        }, options);
        
        this.init();
    }
    
    init() {
        // Clear any existing content
        this.container.innerHTML = '';
        
        // Calculate actual dimensions
        this.width = this.container.offsetWidth - this.options.margin.left - this.options.margin.right;
        this.height = this.options.height - this.options.margin.top - this.options.margin.bottom;
        
        // Create SVG element
        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.setAttribute('width', this.width + this.options.margin.left + this.options.margin.right);
        this.svg.setAttribute('height', this.height + this.options.margin.top + this.options.margin.bottom);
        this.svg.setAttribute('class', 'wandb-chart');
        
        // Create chart group with margin
        this.chart = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.chart.setAttribute('transform', `translate(${this.options.margin.left},${this.options.margin.top})`);
        this.svg.appendChild(this.chart);
        
        // Prepare data
        this.processData();
        
        // Create scales
        this.createScales();
        
        // Add axes
        this.addAxes();
        
        // Add grid
        if (this.options.showGrid) {
            this.addGrid();
        }
        
        // Add title
        this.addTitle();
        
        // Add axis labels
        this.addAxisLabels();
        
        // Add lines for each model
        this.addLines();
        
        // Add legend if enabled
        if (this.options.showLegend) {
            this.addLegend();
        }
        
        // Add tooltip if enabled
        if (this.options.showTooltip) {
            this.addTooltip();
        }
        
        // Add to container
        this.container.appendChild(this.svg);
    }
    
    processData() {
        // Flatten the data and get unique models and steps
        this.flatData = [];
        this.models = [];
        
        let allSteps = new Set();
        
        for (const model in this.data) {
            if (!this.models.includes(model)) {
                this.models.push(model);
            }
            
            for (const point of this.data[model]) {
                const step = point.step;
                const value = point.value;
                
                allSteps.add(step);
                
                this.flatData.push({
                    model: model,
                    step: step,
                    value: value
                });
            }
        }
        
        // Sort steps
        this.steps = Array.from(allSteps).sort((a, b) => a - b);
        
        // Apply smoothing if needed
        if (this.options.smoothing > 0) {
            this.applySmoothing();
        }
    }
    
    applySmoothing() {
        const alpha = this.options.smoothing;
        const smoothedData = {};
        
        for (const model of this.models) {
            const modelData = this.data[model].slice().sort((a, b) => a.step - b.step);
            
            if (!smoothedData[model]) {
                smoothedData[model] = [];
            }
            
            let lastValue = null;
            
            for (const point of modelData) {
                const currentValue = point.value;
                const smoothedValue = lastValue === null ? 
                    currentValue : 
                    alpha * currentValue + (1 - alpha) * lastValue;
                
                smoothedData[model].push({
                    step: point.step,
                    value: smoothedValue
                });
                
                lastValue = smoothedValue;
            }
        }
        
        // Update the flat data with smoothed values
        this.flatData = [];
        for (const model in smoothedData) {
            for (const point of smoothedData[model]) {
                this.flatData.push({
                    model: model,
                    step: point.step,
                    value: point.value
                });
            }
        }
    }
    
    createScales() {
        // X scale (steps)
        const xDomain = [Math.min(...this.steps), Math.max(...this.steps)];
        this.xScale = (step) => {
            return this.width * (step - xDomain[0]) / (xDomain[1] - xDomain[0]);
        };
        
        // Y scale (values)
        const values = this.flatData.map(d => d.value);
        const yMin = Math.min(...values);
        const yMax = Math.max(...values);
        const yPadding = (yMax - yMin) * 0.1; // Add 10% padding
        this.yDomain = [yMin - yPadding, yMax + yPadding];
        
        this.yScale = (value) => {
            return this.height - this.height * (value - this.yDomain[0]) / (this.yDomain[1] - this.yDomain[0]);
        };
    }
    
    addAxes() {
        // X Axis
        const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        xAxis.setAttribute('class', 'x-axis');
        xAxis.setAttribute('transform', `translate(0,${this.height})`);
        
        // X axis line
        const xAxisLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        xAxisLine.setAttribute('x1', 0);
        xAxisLine.setAttribute('y1', 0);
        xAxisLine.setAttribute('x2', this.width);
        xAxisLine.setAttribute('y2', 0);
        xAxisLine.setAttribute('stroke', '#aaaaaa');
        xAxisLine.setAttribute('stroke-width', 1);
        xAxis.appendChild(xAxisLine);
        
        // X axis ticks
        const tickCount = 5;
        const tickInterval = Math.floor(this.steps.length / tickCount);
        
        for (let i = 0; i < this.steps.length; i += tickInterval) {
            const step = this.steps[i];
            const x = this.xScale(step);
            
            // Tick line
            const tickLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            tickLine.setAttribute('x1', x);
            tickLine.setAttribute('y1', 0);
            tickLine.setAttribute('x2', x);
            tickLine.setAttribute('y2', 5);
            tickLine.setAttribute('stroke', '#aaaaaa');
            tickLine.setAttribute('stroke-width', 1);
            xAxis.appendChild(tickLine);
            
            // Tick label
            const tickLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            tickLabel.setAttribute('x', x);
            tickLabel.setAttribute('y', 20);
            tickLabel.setAttribute('text-anchor', 'middle');
            tickLabel.setAttribute('fill', '#aaaaaa');
            tickLabel.setAttribute('font-size', '12px');
            tickLabel.textContent = step;
            xAxis.appendChild(tickLabel);
        }
        
        this.chart.appendChild(xAxis);
        
        // Y Axis
        const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        yAxis.setAttribute('class', 'y-axis');
        
        // Y axis line
        const yAxisLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        yAxisLine.setAttribute('x1', 0);
        yAxisLine.setAttribute('y1', 0);
        yAxisLine.setAttribute('x2', 0);
        yAxisLine.setAttribute('y2', this.height);
        yAxisLine.setAttribute('stroke', '#aaaaaa');
        yAxisLine.setAttribute('stroke-width', 1);
        yAxis.appendChild(yAxisLine);
        
        // Y axis ticks
        const yTickCount = 5;
        const yTickStep = (this.yDomain[1] - this.yDomain[0]) / yTickCount;
        
        for (let i = 0; i <= yTickCount; i++) {
            const value = this.yDomain[0] + i * yTickStep;
            const y = this.yScale(value);
            
            // Tick line
            const tickLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            tickLine.setAttribute('x1', -5);
            tickLine.setAttribute('y1', y);
            tickLine.setAttribute('x2', 0);
            tickLine.setAttribute('y2', y);
            tickLine.setAttribute('stroke', '#aaaaaa');
            tickLine.setAttribute('stroke-width', 1);
            yAxis.appendChild(tickLine);
            
            // Tick label
            const tickLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            tickLabel.setAttribute('x', -10);
            tickLabel.setAttribute('y', y + 4);
            tickLabel.setAttribute('text-anchor', 'end');
            tickLabel.setAttribute('fill', '#aaaaaa');
            tickLabel.setAttribute('font-size', '12px');
            tickLabel.textContent = value.toExponential(2);
            yAxis.appendChild(tickLabel);
        }
        
        this.chart.appendChild(yAxis);
    }
    
    addGrid() {
        const grid = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        grid.setAttribute('class', 'grid');
        
        // Horizontal grid lines (based on y-axis ticks)
        const yTickCount = 5;
        const yTickStep = (this.yDomain[1] - this.yDomain[0]) / yTickCount;
        
        for (let i = 0; i <= yTickCount; i++) {
            const value = this.yDomain[0] + i * yTickStep;
            const y = this.yScale(value);
            
            // Grid line
            const gridLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            gridLine.setAttribute('x1', 0);
            gridLine.setAttribute('y1', y);
            gridLine.setAttribute('x2', this.width);
            gridLine.setAttribute('y2', y);
            gridLine.setAttribute('stroke', '#333333');
            gridLine.setAttribute('stroke-width', 0.5);
            gridLine.setAttribute('stroke-dasharray', '3,3');
            grid.appendChild(gridLine);
        }
        
        this.chart.appendChild(grid);
    }
    
    addTitle() {
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        title.setAttribute('x', this.width / 2);
        title.setAttribute('y', -5);
        title.setAttribute('text-anchor', 'middle');
        title.setAttribute('fill', '#e0e0e0');
        title.setAttribute('font-size', '16px');
        title.setAttribute('font-weight', 'bold');
        title.textContent = this.options.title;
        this.chart.appendChild(title);
    }
    
    addAxisLabels() {
        // X axis label
        const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        xLabel.setAttribute('x', this.width / 2);
        xLabel.setAttribute('y', this.height + 40);
        xLabel.setAttribute('text-anchor', 'middle');
        xLabel.setAttribute('fill', '#aaaaaa');
        xLabel.setAttribute('font-size', '14px');
        xLabel.textContent = this.options.xAxisLabel;
        this.chart.appendChild(xLabel);
        
        // Y axis label
        const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        yLabel.setAttribute('transform', `translate(-40,${this.height / 2}) rotate(-90)`);
        yLabel.setAttribute('text-anchor', 'middle');
        yLabel.setAttribute('fill', '#aaaaaa');
        yLabel.setAttribute('font-size', '14px');
        yLabel.textContent = this.options.yAxisLabel;
        this.chart.appendChild(yLabel);
    }
    
    addLines() {
        this.lineGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.lineGroup.setAttribute('class', 'line-group');
        
        // Create a line for each model
        this.models.forEach((model, index) => {
            // Get data for this model
            const modelData = this.flatData.filter(d => d.model === model)
                                .sort((a, b) => a.step - b.step);
            
            // Skip if no data
            if (modelData.length === 0) return;
            
            // Generate path data
            let pathData = `M${this.xScale(modelData[0].step)},${this.yScale(modelData[0].value)}`;
            
            for (let i = 1; i < modelData.length; i++) {
                pathData += ` L${this.xScale(modelData[i].step)},${this.yScale(modelData[i].value)}`;
            }
            
            // Create path element
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', pathData);
            path.setAttribute('fill', 'none');
            path.setAttribute('stroke', this.options.colors[index % this.options.colors.length]);
            path.setAttribute('stroke-width', 2);
            path.setAttribute('data-model', model);
            
            // Add animation if enabled
            if (this.options.animation) {
                path.setAttribute('stroke-dasharray', path.getTotalLength());
                path.setAttribute('stroke-dashoffset', path.getTotalLength());
                path.innerHTML = `
                    <animate 
                        attributeName="stroke-dashoffset" 
                        from="${path.getTotalLength()}" 
                        to="0" 
                        dur="1s" 
                        fill="freeze" 
                    />
                `;
            }
            
            this.lineGroup.appendChild(path);
            
            // Add data points
            modelData.forEach(point => {
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', this.xScale(point.step));
                circle.setAttribute('cy', this.yScale(point.value));
                circle.setAttribute('r', 3);
                circle.setAttribute('fill', this.options.colors[index % this.options.colors.length]);
                circle.setAttribute('data-model', model);
                circle.setAttribute('data-step', point.step);
                circle.setAttribute('data-value', point.value);
                
                // Add animation
                if (this.options.animation) {
                    circle.setAttribute('opacity', 0);
                    circle.innerHTML = `
                        <animate 
                            attributeName="opacity" 
                            from="0" 
                            to="1" 
                            dur="1s" 
                            fill="freeze" 
                            begin="0.8s"
                        />
                    `;
                }
                
                this.lineGroup.appendChild(circle);
            });
        });
        
        this.chart.appendChild(this.lineGroup);
    }
    
    addLegend() {
        const legendGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        legendGroup.setAttribute('class', 'legend');
        
        const legendHeight = 20;
        const legendSpacing = 20;
        let legendY = 20;
        
        this.models.forEach((model, index) => {
            const legendItem = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            legendItem.setAttribute('transform', `translate(${this.width - 120}, ${legendY})`);
            
            // Color indicator
            const colorIndicator = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            colorIndicator.setAttribute('width', 15);
            colorIndicator.setAttribute('height', 3);
            colorIndicator.setAttribute('y', -6);
            colorIndicator.setAttribute('fill', this.options.colors[index % this.options.colors.length]);
            legendItem.appendChild(colorIndicator);
            
            // Model name
            const modelName = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            modelName.setAttribute('x', 20);
            modelName.setAttribute('y', 0);
            modelName.setAttribute('fill', '#e0e0e0');
            modelName.setAttribute('font-size', '12px');
            modelName.textContent = model;
            legendItem.appendChild(modelName);
            
            legendGroup.appendChild(legendItem);
            legendY += legendHeight;
        });
        
        this.chart.appendChild(legendGroup);
    }
    
    addTooltip() {
        // Create tooltip element (HTML, not SVG)
        const tooltip = document.createElement('div');
        tooltip.className = 'wandb-tooltip';
        tooltip.style.position = 'absolute';
        tooltip.style.padding = '10px';
        tooltip.style.backgroundColor = 'rgba(30, 30, 30, 0.9)';
        tooltip.style.borderRadius = '5px';
        tooltip.style.pointerEvents = 'none';
        tooltip.style.opacity = 0;
        tooltip.style.transition = 'opacity 0.2s';
        tooltip.style.color = '#e0e0e0';
        tooltip.style.fontSize = '12px';
        tooltip.style.zIndex = 1000;
        tooltip.style.boxShadow = '0 2px 5px rgba(0, 0, 0, 0.3)';
        this.container.style.position = 'relative';
        this.container.appendChild(tooltip);
        
        // Add event listeners to data points
        const circles = this.lineGroup.querySelectorAll('circle');
        
        circles.forEach(circle => {
            circle.addEventListener('mouseover', (event) => {
                const model = circle.getAttribute('data-model');
                const step = circle.getAttribute('data-step');
                const value = circle.getAttribute('data-value');
                
                tooltip.innerHTML = `
                    <div><strong>${model}</strong></div>
                    <div>Step: ${step}</div>
                    <div>MSE: ${Number(value).toExponential(4)}</div>
                `;
                
                tooltip.style.opacity = 1;
                tooltip.style.left = `${event.offsetX + 15}px`;
                tooltip.style.top = `${event.offsetY - 15}px`;
                
                // Highlight active point
                circle.setAttribute('r', 5);
            });
            
            circle.addEventListener('mousemove', (event) => {
                tooltip.style.left = `${event.offsetX + 15}px`;
                tooltip.style.top = `${event.offsetY - 15}px`;
            });
            
            circle.addEventListener('mouseout', () => {
                tooltip.style.opacity = 0;
                
                // Restore point size
                circle.setAttribute('r', 3);
            });
        });
    }
}

// Function to initialize the chart when data is loaded
function initWandbChart(data) {
    const chartContainer = document.getElementById('mse-chart-container');
    if (!chartContainer) return;
    
    const chart = new WandbChart('mse-chart-container', data, {
        height: 400,
        title: 'Model MSE Comparison',
        yAxisLabel: 'MSE Loss',
        xAxisLabel: 'Training Step',
        smoothing: 0.2
    });
}

// Add some styles
document.addEventListener('DOMContentLoaded', function() {
    const style = document.createElement('style');
    style.textContent = `
        .wandb-chart {
            background-color: var(--header-bg);
            border-radius: 8px;
        }
        
        .wandb-tooltip {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
    `;
    
    document.head.appendChild(style);
    
    // Load the data
    fetch('wandb_data.json')
        .then(response => response.json())
        .then(data => {
            initWandbChart(data);
        })
        .catch(error => {
            console.error('Error loading chart data:', error);
        });
});