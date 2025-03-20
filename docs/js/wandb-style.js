// Simple but reliable chart implementation
document.addEventListener('DOMContentLoaded', function() {
    // Sample data - we'll use this if loading the JSON fails
    const modelData = {
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
  
    // Try to load data, or use the sample data
    drawChart(modelData);
  
    function drawChart(data) {
      // Chart configuration
      const config = {
        container: 'mse-chart-container',
        width: 600,
        height: 400,
        margin: {top: 40, right: 120, bottom: 50, left: 60},
        backgroundColor: '#1a1a1a',
        textColor: '#e0e0e0',
        gridColor: '#333333',
        lineColors: ['#5b8a72', '#c45b64', '#e3a914', '#8854d0', '#2e86c1'],
        title: 'Model MSE Comparison',
        xAxisLabel: 'Training Step',
        yAxisLabel: 'MSE Loss'
      };
  
      // Create canvas
      const container = document.getElementById(config.container);
      if (!container) return;
      
      // Clear previous chart if any
      container.innerHTML = '';
      
      // Get actual chart dimensions
      const chartWidth = config.width - config.margin.left - config.margin.right;
      const chartHeight = config.height - config.margin.top - config.margin.bottom;
      
      // Create SVG
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('width', '100%'); 
      svg.setAttribute('height', config.height);
      svg.setAttribute('viewBox', `0 0 ${config.width} ${config.height}`);
      svg.style.backgroundColor = config.backgroundColor;
      svg.style.borderRadius = '8px';
      container.appendChild(svg);
      
      // Create chart group
      const chartGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      chartGroup.setAttribute('transform', `translate(${config.margin.left},${config.margin.top})`);
      svg.appendChild(chartGroup);
      
      // Find min/max values for scales
      let minX = Infinity, maxX = -Infinity;
      let minY = Infinity, maxY = -Infinity;
      
      for (const model in data) {
        for (const point of data[model]) {
          const x = Number(point.step);
          const y = Number(point.value);
          
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
      
      // Add padding to Y scale
      const yPadding = (maxY - minY) * 0.1;
      minY = minY - yPadding;
      maxY = maxY + yPadding;
      
      // Scale functions
      function xScale(value) {
        return (value - minX) / (maxX - minX) * chartWidth;
      }
      
      function yScale(value) {
        return chartHeight - (value - minY) / (maxY - minY) * chartHeight;
      }
      
      // Draw grid
      const grid = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      grid.setAttribute('class', 'grid');
      
      // Horizontal grid lines
      const yTicks = 5;
      for (let i = 0; i <= yTicks; i++) {
        const y = i / yTicks * chartHeight;
        const gridLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        gridLine.setAttribute('x1', 0);
        gridLine.setAttribute('y1', y);
        gridLine.setAttribute('x2', chartWidth);
        gridLine.setAttribute('y2', y);
        gridLine.setAttribute('stroke', config.gridColor);
        gridLine.setAttribute('stroke-dasharray', '3,3');
        grid.appendChild(gridLine);
      }
      
      // Vertical grid lines
      const xTicks = 5;
      for (let i = 0; i <= xTicks; i++) {
        const x = i / xTicks * chartWidth;
        const gridLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        gridLine.setAttribute('x1', x);
        gridLine.setAttribute('y1', 0);
        gridLine.setAttribute('x2', x);
        gridLine.setAttribute('y2', chartHeight);
        gridLine.setAttribute('stroke', config.gridColor);
        gridLine.setAttribute('stroke-dasharray', '3,3');
        grid.appendChild(gridLine);
      }
      
      chartGroup.appendChild(grid);
      
      // Draw axes
      const axes = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      axes.setAttribute('class', 'axes');
      
      // X-axis
      const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      xAxis.setAttribute('x1', 0);
      xAxis.setAttribute('y1', chartHeight);
      xAxis.setAttribute('x2', chartWidth);
      xAxis.setAttribute('y2', chartHeight);
      xAxis.setAttribute('stroke', config.textColor);
      axes.appendChild(xAxis);
      
      // Y-axis
      const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      yAxis.setAttribute('x1', 0);
      yAxis.setAttribute('y1', 0);
      yAxis.setAttribute('x2', 0);
      yAxis.setAttribute('y2', chartHeight);
      yAxis.setAttribute('stroke', config.textColor);
      axes.appendChild(yAxis);
      
      // X-axis ticks and labels
      for (let i = 0; i <= xTicks; i++) {
        const value = minX + (i / xTicks) * (maxX - minX);
        const x = xScale(value);
        
        // Tick
        const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        tick.setAttribute('x1', x);
        tick.setAttribute('y1', chartHeight);
        tick.setAttribute('x2', x);
        tick.setAttribute('y2', chartHeight + 5);
        tick.setAttribute('stroke', config.textColor);
        axes.appendChild(tick);
        
        // Label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', x);
        label.setAttribute('y', chartHeight + 20);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('fill', config.textColor);
        label.textContent = Math.round(value);
        axes.appendChild(label);
      }
      
      // Y-axis ticks and labels
      for (let i = 0; i <= yTicks; i++) {
        const value = minY + (i / yTicks) * (maxY - minY);
        const y = yScale(value);
        
        // Tick
        const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        tick.setAttribute('x1', -5);
        tick.setAttribute('y1', y);
        tick.setAttribute('x2', 0);
        tick.setAttribute('y2', y);
        tick.setAttribute('stroke', config.textColor);
        axes.appendChild(tick);
        
        // Label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', -10);
        label.setAttribute('y', y + 4);
        label.setAttribute('text-anchor', 'end');
        label.setAttribute('fill', config.textColor);
        label.textContent = value.toExponential(2);
        axes.appendChild(label);
      }
      
      chartGroup.appendChild(axes);
      
      // Draw title
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      title.setAttribute('x', chartWidth / 2);
      title.setAttribute('y', -20);
      title.setAttribute('text-anchor', 'middle');
      title.setAttribute('fill', config.textColor);
      title.setAttribute('font-size', '16px');
      title.setAttribute('font-weight', 'bold');
      title.textContent = config.title;
      chartGroup.appendChild(title);
      
      // Draw axis labels
      const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      xLabel.setAttribute('x', chartWidth / 2);
      xLabel.setAttribute('y', chartHeight + 40);
      xLabel.setAttribute('text-anchor', 'middle');
      xLabel.setAttribute('fill', config.textColor);
      xLabel.textContent = config.xAxisLabel;
      chartGroup.appendChild(xLabel);
      
      const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      yLabel.setAttribute('transform', `translate(-40,${chartHeight/2}) rotate(-90)`);
      yLabel.setAttribute('text-anchor', 'middle');
      yLabel.setAttribute('fill', config.textColor);
      yLabel.textContent = config.yAxisLabel;
      chartGroup.appendChild(yLabel);
      
      // Draw lines for each model
      const linesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      linesGroup.setAttribute('class', 'lines');
      
      let modelIndex = 0;
      for (const model in data) {
        const points = data[model].sort((a, b) => a.step - b.step);
        const color = config.lineColors[modelIndex % config.lineColors.length];
        
        // Create line path
        let pathData = '';
        points.forEach((point, i) => {
          const x = xScale(point.step);
          const y = yScale(point.value);
          
          if (i === 0) {
            pathData += `M ${x} ${y}`;
          } else {
            pathData += ` L ${x} ${y}`;
          }
        });
        
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', pathData);
        path.setAttribute('stroke', color);
        path.setAttribute('stroke-width', 2);
        path.setAttribute('fill', 'none');
        linesGroup.appendChild(path);
        
        // Add data points
        points.forEach(point => {
          const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          circle.setAttribute('cx', xScale(point.step));
          circle.setAttribute('cy', yScale(point.value));
          circle.setAttribute('r', 3);
          circle.setAttribute('fill', color);
          circle.setAttribute('data-model', model);
          circle.setAttribute('data-step', point.step);
          circle.setAttribute('data-value', point.value);
          linesGroup.appendChild(circle);
        });
        
        modelIndex++;
      }
      
      chartGroup.appendChild(linesGroup);
      
      // Add legend
      const legendGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      legendGroup.setAttribute('class', 'legend');
      legendGroup.setAttribute('transform', `translate(${chartWidth + 10}, 0)`);
      
      modelIndex = 0;
      for (const model in data) {
        const color = config.lineColors[modelIndex % config.lineColors.length];
        const y = modelIndex * 20;
        
        // Line
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', 0);
        line.setAttribute('y1', y + 8);
        line.setAttribute('x2', 20);
        line.setAttribute('y2', y + 8);
        line.setAttribute('stroke', color);
        line.setAttribute('stroke-width', 2);
        legendGroup.appendChild(line);
        
        // Label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', 25);
        label.setAttribute('y', y + 12);
        label.setAttribute('fill', config.textColor);
        label.textContent = model;
        legendGroup.appendChild(label);
        
        modelIndex++;
      }
      
      chartGroup.appendChild(legendGroup);
      
      // Add tooltip
      const tooltip = document.createElement('div');
      tooltip.style.position = 'absolute';
      tooltip.style.padding = '8px 12px';
      tooltip.style.backgroundColor = 'rgba(40, 40, 40, 0.9)';
      tooltip.style.borderRadius = '4px';
      tooltip.style.color = '#ffffff';
      tooltip.style.fontSize = '12px';
      tooltip.style.pointerEvents = 'none';
      tooltip.style.opacity = '0';
      tooltip.style.transition = 'opacity 0.2s';
      tooltip.style.zIndex = '100';
      container.style.position = 'relative';
      container.appendChild(tooltip);
      
      // Add interactivity
      const circles = linesGroup.querySelectorAll('circle');
      
      circles.forEach(circle => {
        circle.addEventListener('mouseover', (e) => {
          const rect = svg.getBoundingClientRect();
          const model = circle.getAttribute('data-model');
          const step = circle.getAttribute('data-step');
          const value = Number(circle.getAttribute('data-value')).toExponential(4);
          const modelIndex = Object.keys(data).indexOf(model);
          const color = config.lineColors[modelIndex % config.lineColors.length];
          
          tooltip.innerHTML = `
            <div style="font-weight: bold; color: ${color}">${model}</div>
            <div>Step: ${step}</div>
            <div>MSE: ${value}</div>
          `;
          
          tooltip.style.opacity = '1';
          tooltip.style.left = `${e.clientX - rect.left + 10}px`;
          tooltip.style.top = `${e.clientY - rect.top - 40}px`;
          
          circle.setAttribute('r', '5');
        });
        
        circle.addEventListener('mouseout', () => {
          tooltip.style.opacity = '0';
          circle.setAttribute('r', '3');
        });
        
        circle.addEventListener('mousemove', (e) => {
          const rect = svg.getBoundingClientRect();
          tooltip.style.left = `${e.clientX - rect.left + 10}px`;
          tooltip.style.top = `${e.clientY - rect.top - 40}px`;
        });
      });
    }
  });