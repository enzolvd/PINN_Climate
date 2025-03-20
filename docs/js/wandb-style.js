// Create a WandbStyle interactive chart handler
class WandbChart {
    constructor(containerId, data, options = {}) {
        this.container = document.getElementById(containerId);
        this.data = data;
        this.options = Object.assign({
            height: 400,
            margin: { top: 20, right: 80, bottom: 50, left: 60 },
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
        if (!this.container) {
            console.error('Container element not found');
            return;
        }
        
        // Clear any existing content
        this.container.innerHTML = '';
        
        // Calculate actual dimensions
        this.width = this.container.offsetWidth - this.options.margin.left - this.options.margin.right;
        this.height = this.options.height - this.options.margin.top - this.options.margin.bottom;
        
        // Create SVG element
        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.setAttribute('width', '100%');
        this.svg.setAttribute('height', this.options.height);
        this.svg.setAttribute('viewBox', `0 0 ${this.container.offsetWidth} ${this.options.height}`);
        this.svg.setAttribute('class', 'wandb-chart');
        this.svg.style.backgroundColor = 'var(--header-bg, #1a1a1a)';
        this.svg.style.borderRadius = '8px';
        
        // Create chart group with margin
        this.chart = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.chart.setAttribute('transform', `translate(${this.options.margin.left},${this.options.margin.top})`);
        this.svg.appendChild(this.chart);
        
        // Prepare data
        this.processData();
        
        // Create scales
        this.createScales();
        
        // Add grid
        if (this.options.showGrid) {
            this.addGrid();
        }
        
        // Add axes
        this.addAxes();
        
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
        let minValue = Infinity;
        let maxValue = -Infinity;
        
        for (const model in this.data) {
            if (!this.models.includes(model)) {
                this.models.push(model);
            }
            
            for (const point of this.data[model]) {
                const step = Number(point.step);
                const value = Number(point.value);
                
                allSteps.add(step);
                
                // Update min/max values
                if (value < minValue) minValue = value;
                if (value > maxValue) maxValue = value;
                
                this.flatData.push({
                    model: model,
                    step: step,
                    value: value
                });
            }
        }
        
        // Sort steps
        this.steps = Array.from(allSteps).sort((a, b) => a - b);
        
        // Store min/max values for scaling
        this.minValue = minValue;
        this.maxValue = maxValue;
        
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
        const xMin = Math.min(...this.steps);
        const xMax = Math.max(...this.steps);
        this.xDomain = [xMin, xMax];
        
        this.xScale = (step) => {
            return this.width * (step - this.xDomain[0]) / (this.xDomain[1] - this.xDomain[0]);
        };
        
        // Y scale (values)
        const yPadding = (this.maxValue - this.minValue) * 0.1; // Add 10% padding
        this.yDomain = [this.minValue - yPadding, this.maxValue + yPadding];
        
        this.yScale = (value) => {
            return this.height - this.height * (value - this.yDomain[0]) / (this.yDomain[1] - this.yDomain[0]);
        };
    }
    
    addGrid() {
        const grid = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        grid.setAttribute('class', 'grid');
        
        // Horizontal grid lines
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
        
        // Vertical grid lines
        const xTickCount = 5;
        const xTickStep = (this.xDomain[1] - this.xDomain[0]) / xTickCount;
        
        for (let i = 0; i <= xTickCount; i++) {
            const step = this.xDomain[0] + i * xTickStep;
            const x = this.xScale(step);
            
            // Grid line
            const gridLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            gridLine.setAttribute('x1', x);
            gridLine.setAttribute('y1', 0);
            gridLine.setAttribute('x2', x);
            gridLine.setAttribute('y2', this.height);
            gridLine.setAttribute('stroke', '#333333');
            gridLine.setAttribute('stroke-width', 0.5);
            gridLine.setAttribute('stroke-dasharray', '3,3');
            grid.appendChild(gridLine);
        }
        
        this.chart.appendChild(grid);
    }
    
    addAxes() {
        // X Axis
        const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        xAxis.setAttribute('class', 'x-axis');
        
        // X axis line
        const xAxisLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        xAxisLine.setAttribute('x1', 0);
        xAxisLine.setAttribute('y1', this.height);
        xAxisLine.setAttribute('x2', this.width);
        xAxisLine.setAttribute('y2', this.height);
        xAxisLine.setAttribute('stroke', '#aaaaaa');
        xAxisLine.setAttribute('stroke-width', 1);
        xAxis.appendChild(xAxisLine);
        
        // X axis ticks
        const xTickCount = 5;
        const xTickStep = (this.xDomain[1] - this.xDomain[0]) / xTickCount;
        
        for (let i = 0; i <= xTickCount; i++) {
            const step = this.xDomain[0] + i * xTickStep;
            const x = this.xScale(step);
            
            // Tick line
            const tickLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            tickLine.setAttribute('x1', x);
            tickLine.setAttribute('y1', this.height);
            tickLine.setAttribute('x2', x);
            tickLine.setAttribute('y2', this.height + 5);
            tickLine.setAttribute('stroke', '#aaaaaa');
            tickLine.setAttribute('stroke-width', 1);
            xAxis.appendChild(tickLine);
            
            // Tick label
            const tickLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            tickLabel.setAttribute('x', x);
            tickLabel.setAttribute('y', this.height + 20);
            tickLabel.setAttribute('text-anchor', 'middle');
            tickLabel.setAttribute('fill', '#aaaaaa');
            tickLabel.setAttribute('font-size', '12px');
            tickLabel.textContent = Math.round(step);
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
            
            // Format the value in scientific notation
            let formattedValue;
            if (value < 0.001 || value > 1000) {
                formattedValue = value.toExponential(2);
            } else {
                formattedValue = value.toPrecision(3);
            }
            
            tickLabel.textContent = formattedValue;
            yAxis.appendChild(tickLabel);
        }
        
        this.chart.appendChild(yAxis);
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
            const color = this.options.colors[index % this.options.colors.length];
            
            // Get data for this model
            const modelData = this.flatData.filter(d => d.model === model)
                                .sort((a, b) => a.step - b.step);
            
            // Skip if no data
            if (modelData.length === 0) return;
            
            // Generate path data
            let pathData = '';
            
            for (let i = 0; i < modelData.length; i++) {
                const x = this.xScale(modelData[i].step);
                const y = this.yScale(modelData[i].value);
                
                if (i === 0) {
                    pathData += `M${x},${y}`;
                } else {
                    pathData += ` L${x},${y}`;
                }
            }
            
            // Create path element
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', pathData);
            path.setAttribute('fill', 'none');
            path.setAttribute('stroke', color);
            path.setAttribute('stroke-width', 2);
            path.setAttribute('data-model', model);
            
            // Add animation if enabled
            if (this.options.animation) {
                const length = path.getTotalLength ? path.getTotalLength() : 1000;
                path.setAttribute('stroke-dasharray', length);
                path.setAttribute('stroke-dashoffset', length);
                
                const animation = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
                animation.setAttribute('attributeName', 'stroke-dashoffset');
                animation.setAttribute('from', length);
                animation.setAttribute('to', '0');
                animation.setAttribute('dur', '1s');
                animation.setAttribute('fill', 'freeze');
                
                path.appendChild(animation);
            }
            
            this.lineGroup.appendChild(path);
            
            // Add data points
            modelData.forEach(point => {
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', this.xScale(point.step));
                circle.setAttribute('cy', this.yScale(point.value));
                circle.setAttribute('r', 3);
                circle.setAttribute('fill', color);
                circle.setAttribute('data-model', model);
                circle.setAttribute('data-step', point.step);
                circle.setAttribute('data-value', point.value);
                
                // Add animation
                if (this.options.animation) {
                    const animOpacity = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
                    animOpacity.setAttribute('attributeName', 'opacity');
                    animOpacity.setAttribute('from', '0');
                    animOpacity.setAttribute('to', '1');
                    animOpacity.setAttribute('dur', '1s');
                    animOpacity.setAttribute('fill', 'freeze');
                    animOpacity.setAttribute('begin', '0.8s');
                    
                    circle.setAttribute('opacity', '0');
                    circle.appendChild(animOpacity);
                }
                
                this.lineGroup.appendChild(circle);
            });
        });
        
        this.chart.appendChild(this.lineGroup);
    }
    
    addLegend() {
        const legendGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        legendGroup.setAttribute('class', 'legend');
        legendGroup.setAttribute('transform', `translate(${this.width + 5}, 0)`);
        
        const legendHeight = 20;
        let legendY = 0;
        
        this.models.forEach((model, index) => {
            const color = this.options.colors[index % this.options.colors.length];
            const formattedModel = model.replace(/_/g, '_');
            
            const legendItem = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            legendItem.setAttribute('transform', `translate(0, ${legendY})`);
            
            // Color indicator
            const colorIndicator = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            colorIndicator.setAttribute('x1', 0);
            colorIndicator.setAttribute('y1', 0);
            colorIndicator.setAttribute('x2', 15);
            colorIndicator.setAttribute('y2', 0);
            colorIndicator.setAttribute('stroke', color);
            colorIndicator.setAttribute('stroke-width', 2);
            legendItem.appendChild(colorIndicator);
            
            // Model name
            const modelName = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            modelName.setAttribute('x', 20);
            modelName.setAttribute('y', 4);
            modelName.setAttribute('fill', '#e0e0e0');
            modelName.setAttribute('font-size', '12px');
            modelName.textContent = formattedModel;
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
        tooltip.style.opacity = '0';
        tooltip.style.transition = 'opacity 0.2s';
        tooltip.style.color = '#e0e0e0';
        tooltip.style.fontSize = '12px';
        tooltip.style.zIndex = '1000';
        tooltip.style.boxShadow = '0 2px 5px rgba(0, 0, 0, 0.3)';
        this.container.style.position = 'relative';
        this.container.appendChild(tooltip);
        
        // Add event listeners
        this.svg.addEventListener('mousemove', (event) => {
            const rect = this.svg.getBoundingClientRect();
            const mouseX = event.clientX - rect.left - this.options.margin.left;
            const mouseY = event.clientY - rect.top - this.options.margin.top;
            
            // Find the closest point
            let closestPoint = null;
            let closestDistance = Infinity;
            
            this.flatData.forEach(point => {
                const x = this.xScale(point.step);
                const y = this.yScale(point.value);
                const distance = Math.sqrt((x - mouseX) ** 2 + (y - mouseY) ** 2);
                
                if (distance < closestDistance && distance < 20) {
                    closestDistance = distance;
                    closestPoint = point;
                }
            });
            
            if (closestPoint) {
                const model = closestPoint.model;
                const step = closestPoint.step;
                const value = closestPoint.value;
                const modelIndex = this.models.indexOf(model);
                const color = this.options.colors[modelIndex % this.options.colors.length];
                
                tooltip.innerHTML = `
                    <div style="color:${color}"><strong>${model}</strong></div>
                    <div>Step: ${step}</div>
                    <div>MSE: ${Number(value).toExponential(4)}</div>
                `;
                
                tooltip.style.opacity = '1';
                tooltip.style.left = `${event.clientX - rect.left + 15}px`;
                tooltip.style.top = `${event.clientY - rect.top - 15}px`;
                
                // Highlight active point by increasing circle size
                const circles = this.lineGroup.querySelectorAll('circle');
                circles.forEach(circle => {
                    if (circle.getAttribute('data-model') === model && 
                        Number(circle.getAttribute('data-step')) === step) {
                        circle.setAttribute('r', '5');
                    } else {
                        circle.setAttribute('r', '3');
                    }
                });
            } else {
                tooltip.style.opacity = '0';
                
                // Reset all circles
                const circles = this.lineGroup.querySelectorAll('circle');
                circles.forEach(circle => {
                    circle.setAttribute('r', '3');
                });
            }
        });
        
        this.svg.addEventListener('mouseleave', () => {
            tooltip.style.opacity = '0';
            
            // Reset all circles
            const circles = this.lineGroup.querySelectorAll('circle');
            circles.forEach(circle => {
                circle.setAttribute('r', '3');
            });
        });
    }
}

// Fallback data in case the JSON file can't be loaded
const fallbackData = {
  "model_0": [
    {"step": 0, "value": 0.0076},
    {"step": 10, "value": 0.0054},
    {"step": 20, "value": 0.0043},
    {"step": 30, "value": 0.0035},
    {"step": 40, "value": 0.0029},
    {"step": 50, "value": 0.0025},
    {"step": 60, "value": 0.0022},
    {"step": 70, "value": 0.0020},
    {"step": 80, "value": 0.0019},
    {"step": 90, "value": 0.0018},
    {"step": 100, "value": 0.0017}
  ],
  "model_1": [
    {"step": 0, "value": 0.0089},
    {"step": 10, "value": 0.0084},
    {"step": 20, "value": 0.0079},
    {"step": 30, "value": 0.0076},
    {"step": 40, "value": 0.0072},
    {"step": 50, "value": 0.0070},
    {"step": 60, "value": 0.0068},
    {"step": 70, "value": 0.0066},
    {"step": 80, "value": 0.0065},
    {"step": 90, "value": 0.0064},
    {"step": 100, "value": 0.0063}
  ],
  "model_0_Re": [
    {"step": 0, "value": 0.0058},
    {"step": 10, "value": 0.0039},
    {"step": 20, "value": 0.0028},
    {"step": 30, "value": 0.0021},
    {"step": 40, "value": 0.0016},
    {"step": 50, "value": 0.0013},
    {"step": 60, "value": 0.0011},
    {"step": 70, "value": 0.0010},
    {"step": 80, "value": 0.0010},
    {"step": 90, "value": 0.0009},
    {"step": 100, "value": 0.0009}
  ],
  "model_2": [
    {"step": 0, "value": 0.0043},
    {"step": 10, "value": 0.0026},
    {"step": 20, "value": 0.0016},
    {"step": 30, "value": 0.0012},
    {"step": 40, "value": 0.0009},
    {"step": 50, "value": 0.0008},
    {"step": 60, "value": 0.0007},
    {"step": 70, "value": 0.0006},
    {"step": 80, "value": 0.0006},
    {"step": 90, "value": 0.0006},
    {"step": 100, "value": 0.0006}
  ],
  "model_3": [
    {"step": 0, "value": 0.0046},
    {"step": 10, "value": 0.0029},
    {"step": 20, "value": 0.0018},
    {"step": 30, "value": 0.0014},
    {"step": 40, "value": 0.0010},
    {"step": 50, "value": 0.0009},
    {"step": 60, "value": 0.0008},
    {"step": 70, "value": 0.0007},
    {"step": 80, "value": 0.0006},
    {"step": 90, "value": 0.0006},
    {"step": 100, "value": 0.0006}
  ]
};

// Function to initialize the chart
function initWandbChart(data) {
    const chartContainer = document.getElementById('mse-chart-container');
    if (!chartContainer) {
        console.error('Chart container not found');
        return;
    }
    
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
            background-color: var(--header-bg, #1a1a1a);
            border-radius: 8px;
        }
        
        .wandb-tooltip {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
    `;
    
    document.head.appendChild(style);
    
    // Try to load data from file, fall back to hardcoded data if there's an error
    fetch('/PINN_Climate/wandb_data.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Successfully loaded data from JSON file");
            initWandbChart(data);
        })
        .catch(error => {
            console.warn("Could not load JSON file, using fallback data", error);
            initWandbChart(fallbackData);
        });
});