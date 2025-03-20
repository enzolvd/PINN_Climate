document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelector('.nav-links');
    
    if (hamburger) {
        hamburger.addEventListener('click', function() {
            navLinks.classList.toggle('active');
        });
    }
    
    // Model variant tabs
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    if (tabButtons.length > 0) {
        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                // Remove active class from all buttons
                tabButtons.forEach(btn => btn.classList.remove('active'));
                
                // Add active class to clicked button
                this.classList.add('active');
                
                // Hide all tab panes
                tabPanes.forEach(pane => pane.classList.remove('active'));
                
                // Show selected tab pane
                const tabId = this.getAttribute('data-tab');
                document.getElementById(tabId).classList.add('active');
            });
        });
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            // Only handle internal links
            if (this.getAttribute('href').startsWith('#')) {
                e.preventDefault();
                
                const targetId = this.getAttribute('href');
                const targetElement = document.querySelector(targetId);
                
                if (targetElement) {
                    window.scrollTo({
                        top: targetElement.offsetTop - 100, // Offset for navbar
                        behavior: 'smooth'
                    });
                    
                    // Update URL hash without jumping
                    history.pushState(null, null, targetId);
                }
            }
        });
    });
    
    // Handle URL hash on page load
    if (window.location.hash) {
        const targetElement = document.querySelector(window.location.hash);
        if (targetElement) {
            setTimeout(() => {
                window.scrollTo({
                    top: targetElement.offsetTop - 100,
                    behavior: 'smooth'
                });
            }, 300);
        }
    }
    
    // Add active class to current page in navigation
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav-links a').forEach(link => {
        const linkPage = link.getAttribute('href');
        if (linkPage === currentPage) {
            link.classList.add('active');
        }
    });

    // MSE Chart Integration
    initMSEChart();
});

// Initialize the MSE Chart
function initMSEChart() {
    // Replace MSE chart image with interactive chart if it exists
    const chartImg = document.querySelector('.chart img[alt="MSE Performance Chart"]');
    if (!chartImg) return;

    // Create container for chart
    const chartContainer = document.createElement('div');
    chartContainer.id = 'mse-chart-container';
    chartContainer.style.width = '100%';
    chartContainer.style.height = '400px';
    chartContainer.style.position = 'relative';
    
    // Replace image with container
    chartImg.parentNode.replaceChild(chartContainer, chartImg);
    
    // Add controls
    const controlsDiv = document.createElement('div');
    controlsDiv.className = 'chart-controls';
    controlsDiv.innerHTML = `
        <div class="chart-legend"></div>
        <div class="chart-options">
            <div class="smoothing-control">
                <label for="smoothing-slider">Smoothing: <span id="smoothing-value">0.5</span></label>
                <input type="range" id="smoothing-slider" min="0" max="0.99" step="0.01" value="0.5">
            </div>
            <div class="scale-toggle">
                <label>
                    <input type="checkbox" id="log-scale-toggle" checked>
                    Log Scale
                </label>
            </div>
        </div>
    `;
    
    chartImg.parentNode.appendChild(controlsDiv);
    
    // Load D3.js if not already loaded
    if (typeof d3 === 'undefined') {
        loadScript('https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js', renderMSEChart);
    } else {
        renderMSEChart();
    }
}

// Helper function to load scripts
function loadScript(url, callback) {
    const script = document.createElement('script');
    script.src = url;
    script.onload = callback;
    document.head.appendChild(script);
}

// Render the MSE Chart
function renderMSEChart() {
    // Try to fetch data from wandb_data.json
    fetch('wandb_data.json')
        .then(response => response.json())
        .then(modelData => {
            createMSEChart(modelData);
            setupChartControls(modelData);
        })
        .catch(error => {
            console.warn('Could not load wandb_data.json, using sample data instead:', error);
            // Fallback to sample data
            const sampleData = generateSampleModelData();
            createMSEChart(sampleData);
            setupChartControls(sampleData);
        });
}

// Generate sample data for demonstration
function generateSampleModelData() {
    return {
        "model_0": {
            "data_loss": generateSampleData(25000, 0.0017, 0.0002)
        },
        "model_1": {
            "data_loss": generateSampleData(25000, 0.0063, 0.0008)
        },
        "model_0_Re": {
            "data_loss": generateSampleData(25000, 0.00094, 0.0001)
        },
        "model_2": {
            "data_loss": generateSampleData(25000, 0.00059, 0.00007)
        },
        "model_3": {
            "data_loss": generateSampleData(25000, 0.00061, 0.00008)
        }
    };
}

function generateSampleData(steps, meanValue, stdDev) {
    const data = [];
    for (let i = 0; i < steps; i += 100) {
        const noise = (Math.random() - 0.5) * stdDev;
        const decayFactor = Math.exp(-i / (steps / 3));
        const startValue = meanValue * 10;
        const value = meanValue + (startValue - meanValue) * decayFactor + noise;
        data.push([i, value]);
    }
    return data;
}

// Create the MSE Chart with D3
function createMSEChart(modelData) {
    // Color scheme matching the site's dark theme
    const colorScheme = {
        model_0: "#3a506b",
        model_1: "#5b8a72", 
        model_0_Re: "#8e6fbd",
        model_2: "#d17a4e",
        model_3: "#c5384b"
    };
    
    // Set up dimensions
    const container = document.getElementById('mse-chart-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    const margin = {top: 20, right: 50, bottom: 50, left: 60};
    
    // Create the SVG container
    const svg = d3.select('#mse-chart-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);
    
    // Set up scales
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Find the data bounds
    let allPoints = [];
    Object.entries(modelData).forEach(([model, data]) => {
        allPoints = allPoints.concat(data.data_loss);
    });
    
    const xExtent = d3.extent(allPoints, d => d[0]);
    const yExtent = d3.extent(allPoints, d => d[1]);
    const yMinLog = Math.max(1e-5, yExtent[0]); // Prevent zero in log scale
    
    const xScale = d3.scaleLinear()
        .domain(xExtent)
        .range([0, innerWidth]);
    
    const yScaleLinear = d3.scaleLinear()
        .domain([0, yExtent[1] * 1.1])
        .range([innerHeight, 0]);
    
    const yScaleLog = d3.scaleLog()
        .domain([yMinLog, yExtent[1] * 1.1])
        .range([innerHeight, 0]);
    
    // Initial scale is log
    let yScale = yScaleLog;
    
    // Create axes
    const xAxis = d3.axisBottom(xScale)
        .ticks(5)
        .tickFormat(d => `${d/1000}k`);
    
    const yAxisLinear = d3.axisLeft(yScaleLinear)
        .ticks(5)
        .tickFormat(d => d.toExponential(1));
    
    const yAxisLog = d3.axisLeft(yScaleLog)
        .ticks(5)
        .tickFormat(d => d.toExponential(1));
    
    // Add axes to SVG
    const xAxisG = svg.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0, ${innerHeight})`)
        .call(xAxis);
    
    const yAxisG = svg.append('g')
        .attr('class', 'y-axis')
        .call(yAxisLog);
    
    // Apply dark theme styling to axes
    svg.selectAll('.domain, .tick line')
        .attr('stroke', '#555');
    
    svg.selectAll('.tick text')
        .attr('fill', '#e0e0e0')
        .style('font-size', '12px');
    
    // Add axis labels
    svg.append('text')
        .attr('class', 'x-label')
        .attr('text-anchor', 'middle')
        .attr('x', innerWidth / 2)
        .attr('y', innerHeight + 40)
        .attr('fill', '#aaaaaa')
        .text('Training Steps');
    
    svg.append('text')
        .attr('class', 'y-label')
        .attr('text-anchor', 'middle')
        .attr('transform', `translate(-40, ${innerHeight / 2}) rotate(-90)`)
        .attr('fill', '#aaaaaa')
        .text('MSE Loss');
    
    // Create line generators
    const rawLine = d3.line()
        .x(d => xScale(d[0]))
        .y(d => yScale(d[1]));
    
    const smoothedLine = d3.line()
        .x(d => xScale(d[0]))
        .y(d => yScale(d[1]))
        .curve(d3.curveMonotoneX);
    
    // Create a clip path
    svg.append('clipPath')
        .attr('id', 'chart-area')
        .append('rect')
        .attr('width', innerWidth)
        .attr('height', innerHeight);
    
    // Create a group for the lines
    const linesGroup = svg.append('g')
        .attr('clip-path', 'url(#chart-area)');
    
    // Create the chart legend
    const legend = d3.select('.chart-legend');
    
    // Function to apply smoothing to data
    function smoothData(data, factor) {
        if (factor === 0) return data;
        
        const smoothed = [];
        let lastValue = data[0][1];
        
        for (let i = 0; i < data.length; i++) {
            const point = data[i];
            lastValue = lastValue * factor + point[1] * (1 - factor);
            smoothed.push([point[0], lastValue]);
        }
        
        return smoothed;
    }
    
    // Object to track active models
    const activeModels = {
        model_0: true,
        model_1: true,
        model_0_Re: true,
        model_2: true,
        model_3: true
    };
    
    // Function to update the chart
    function updateChart(smoothingFactor, useLogScale) {
        // Update y scale based on log scale toggle
        yScale = useLogScale ? yScaleLog : yScaleLinear;
        yAxisG.call(useLogScale ? yAxisLog : yAxisLinear);
        
        // Clear existing lines
        linesGroup.selectAll('.line-group').remove();
        
        // Add lines for each model
        Object.entries(modelData).forEach(([model, data]) => {
            if (!activeModels[model]) return;
            
            const color = colorScheme[model] || '#999';
            const rawData = data.data_loss;
            const smoothedData = smoothData(rawData, smoothingFactor);
            
            const lineGroup = linesGroup.append('g')
                .attr('class', 'line-group')
                .attr('data-model', model);
            
            // Draw raw data line with low opacity
            lineGroup.append('path')
                .datum(rawData)
                .attr('class', 'raw-line')
                .attr('fill', 'none')
                .attr('stroke', color)
                .attr('stroke-width', 1)
                .attr('opacity', 0.3)
                .attr('d', rawLine);
            
            // Draw smoothed line
            lineGroup.append('path')
                .datum(smoothedData)
                .attr('class', 'smoothed-line')
                .attr('fill', 'none')
                .attr('stroke', color)
                .attr('stroke-width', 2.5)
                .attr('d', smoothedLine);
            
            // Create invisible hover area
            lineGroup.append('path')
                .datum(smoothedData)
                .attr('class', 'hover-line')
                .attr('fill', 'none')
                .attr('stroke', 'transparent')
                .attr('stroke-width', 10)
                .attr('d', smoothedLine)
                .on('mouseover', function(event) {
                    // Highlight the corresponding line
                    lineGroup.select('.smoothed-line')
                        .attr('stroke-width', 4);
                    
                    // Show tooltip with model name
                    showTooltip(model, event);
                })
                .on('mousemove', function(event) {
                    // Get mouse position
                    const [mouseX] = d3.pointer(event, this);
                    const x0 = xScale.invert(mouseX);
                    
                    // Find closest data point
                    const bisect = d3.bisector(d => d[0]).left;
                    const index = bisect(smoothedData, x0);
                    const d0 = smoothedData[Math.max(0, index - 1)];
                    const d1 = smoothedData[Math.min(smoothedData.length - 1, index)];
                    const d = x0 - d0[0] > d1[0] - x0 ? d1 : d0;
                    
                    // Update tooltip
                    updateTooltip(model, d, event);
                })
                .on('mouseout', function() {
                    // Reset line width
                    lineGroup.select('.smoothed-line')
                        .attr('stroke-width', 2.5);
                    
                    // Hide tooltip
                    hideTooltip();
                });
        });
    }
    
    // Create tooltip
    const tooltip = d3.select('body').append('div')
        .attr('class', 'chart-tooltip')
        .style('position', 'absolute')
        .style('visibility', 'hidden')
        .style('background-color', 'rgba(0, 0, 0, 0.8)')
        .style('color', 'white')
        .style('padding', '8px')
        .style('border-radius', '4px')
        .style('font-size', '12px')
        .style('pointer-events', 'none')
        .style('z-index', '999');
    
    function showTooltip(model, event) {
        tooltip.style('visibility', 'visible');
    }
    
    function updateTooltip(model, dataPoint, event) {
        const [step, value] = dataPoint;
        const modelName = {
            model_0: "Model 0",
            model_1: "Model 1",
            model_0_Re: "Model 0 Re",
            model_2: "Model 2",
            model_3: "Model 3"
        }[model] || model;
        
        tooltip.html(`
            <div><strong>${modelName}</strong></div>
            <div>Step: ${step.toLocaleString()}</div>
            <div>MSE: ${value.toExponential(4)}</div>
        `)
        .style('left', (event.pageX + 15) + 'px')
        .style('top', (event.pageY - 30) + 'px');
    }
    
    function hideTooltip() {
        tooltip.style('visibility', 'hidden');
    }
    
    // Build the legend
    Object.entries(modelData).forEach(([model, data]) => {
        const modelName = {
            model_0: "Model 0",
            model_1: "Model 1",
            model_0_Re: "Model 0 Re",
            model_2: "Model 2",
            model_3: "Model 3"
        }[model] || model;
        
        const color = colorScheme[model] || '#999';
        
        const legendItem = legend.append('div')
            .style('display', 'inline-block')
            .style('margin', '0 10px')
            .style('cursor', 'pointer');
        
        legendItem.append('span')
            .style('display', 'inline-block')
            .style('width', '15px')
            .style('height', '15px')
            .style('background-color', color)
            .style('border-radius', '3px')
            .style('margin-right', '5px')
            .style('vertical-align', 'middle');
        
        legendItem.append('span')
            .text(modelName)
            .style('color', '#e0e0e0')
            .style('font-size', '14px')
            .style('vertical-align', 'middle');
        
        legendItem.on('click', function() {
            activeModels[model] = !activeModels[model];
            
            // Update styling to show active/inactive
            if (activeModels[model]) {
                legendItem.style('opacity', 1);
            } else {
                legendItem.style('opacity', 0.4);
            }
            
            // Update chart with current settings
            const smoothingFactor = parseFloat(document.getElementById('smoothing-slider').value);
            const useLogScale = document.getElementById('log-scale-toggle').checked;
            updateChart(smoothingFactor, useLogScale);
        });
    });
    
    // Initial render with default smoothing
    updateChart(0.5, true);
    
    // Export update function for controls
    window.updateMSEChart = updateChart;
}

// Set up the chart controls
function setupChartControls(modelData) {
    // Set up smoothing slider
    const smoothingSlider = document.getElementById('smoothing-slider');
    const smoothingValue = document.getElementById('smoothing-value');
    
    if (smoothingSlider) {
        smoothingSlider.addEventListener('input', function() {
            const value = parseFloat(this.value);
            smoothingValue.textContent = value.toFixed(2);
            
            // Update chart
            const useLogScale = document.getElementById('log-scale-toggle').checked;
            window.updateMSEChart(value, useLogScale);
        });
    }
    
    // Set up log scale toggle
    const logScaleToggle = document.getElementById('log-scale-toggle');
    
    if (logScaleToggle) {
        logScaleToggle.addEventListener('change', function() {
            const useLogScale = this.checked;
            const smoothingFactor = parseFloat(smoothingSlider.value);
            
            // Update chart
            window.updateMSEChart(smoothingFactor, useLogScale);
        });
    }
    
    // Add styles for the controls
    const style = document.createElement('style');
    style.textContent = `
        .chart-controls {
            margin-top: 10px;
            padding: 15px;
            background-color: var(--header-bg, #1a1a1a);
            border-radius: 8px;
            color: var(--text-color, #e0e0e0);
        }
        
        .chart-legend {
            margin-bottom: 15px;
            text-align: center;
        }
        
        .chart-options {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .smoothing-control {
            display: flex;
            align-items: center;
            margin-right: 20px;
        }
        
        .smoothing-control label {
            margin-right: 10px;
        }
        
        input[type="range"] {
            background-color: var(--border-color, #333333);
            border-radius: 5px;
            height: 6px;
            width: 150px;
        }
        
        .scale-toggle {
            display: flex;
            align-items: center;
        }
        
        .chart-tooltip {
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
            line-height: 1.4;
        }
        
        .x-axis path, .y-axis path, .x-axis line, .y-axis line {
            stroke: var(--border-color, #333333);
        }
        
        @media (max-width: 768px) {
            .chart-options {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .smoothing-control {
                margin-bottom: 10px;
            }
        }
    `;
    document.head.appendChild(style);
}