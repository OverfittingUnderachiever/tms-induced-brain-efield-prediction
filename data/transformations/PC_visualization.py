#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np

def visualize_point_clouds(point_clouds_data, output_path=None, point_size=1.2):
    """
    Generates an HTML file visualizing one or more point clouds.
    
    Parameters:
    -----------
    point_clouds_data : list of dict
        List of dictionaries, each representing a point cloud with:
        - 'points': list of 3D coordinates [[x, y, z], ...]
        - 'name': name/identifier for the point cloud (optional, default: "Point Cloud {index}")
        - 'intensities': list of intensity values for coloring (optional)
        - 'color': [r, g, b] values between.0-1 (optional, random color if not provided)
        
    output_path : str
        Path to save the HTML visualization file.
        If None, will save to /home/freyhe/MA_Henry/subject_visualizations with a timestamp.
        
    point_size : float
        Size of the points in the visualization (default: 1.2)
    
    Returns:
    --------
    str : Path to the generated HTML file
    """
    # Generate a unique filename with timestamp if output_path is not provided
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/home/freyhe/MA_Henry/subject_visualizations/pointcloud_visualization_{timestamp}.html"
    """
    Generates an HTML file visualizing one or more point clouds.
    
    Parameters:
    -----------
    point_clouds_data : list of dict
        List of dictionaries, each representing a point cloud with:
        - 'points': list of 3D coordinates [[x, y, z], ...]
        - 'name': name/identifier for the point cloud (optional, default: "Point Cloud {index}")
        - 'intensities': list of intensity values for coloring (optional)
        - 'color': [r, g, b] values between 0-1 (optional, random color if not provided)
        
    output_path : str
        Path to save the HTML visualization file
        
    point_size : float
        Size of the points in the visualization (default: 1.2)
    
    Returns:
    --------
    str : Path to the generated HTML file
    """
    # Define a color palette if not provided
    def get_color(index, total):
        """Generate a distinct color from HSV color wheel based on index"""
        hue = index / total
        # Convert HSV to RGB (simplified version with S=1, V=1)
        h_i = int(hue * 6)
        f = hue * 6 - h_i
        
        if h_i == 0:
            return [1.0, f, 0.0]
        elif h_i == 1:
            return [1.0-f, 1.0, 0.0]
        elif h_i == 2:
            return [0.0, 1.0, f]
        elif h_i == 3:
            return [0.0, 1.0-f, 1.0]
        elif h_i == 4:
            return [f, 0.0, 1.0]
        else:
            return [1.0, 0.0, 1.0-f]
    
    # Process point cloud data
    processed_point_clouds = []
    for i, pc_data in enumerate(point_clouds_data):
        
        # Extract basic data
        points = pc_data.get('points', [])
        # Check for empty points array properly (works with numpy arrays too)
        if isinstance(points, np.ndarray):
            if points.size == 0:
                print(f"Warning: Point cloud {i} has no points. Skipping.")
                continue
        elif len(points) == 0:
            print(f"Warning: Point cloud {i} has no points. Skipping.")
            continue
            
        # Get or generate name
        name = pc_data.get('name', f"Point Cloud {i+1}")
        
        # Get or generate color
        color = pc_data.get('color', get_color(i, len(point_clouds_data)))
        
        # Get or generate intensities
        if 'intensities' in pc_data:
            intensities = pc_data['intensities']
        else:
            # Create default intensities array with the right length
            if isinstance(points, np.ndarray):
                points_length = points.shape[0]
            else:
                points_length = len(points)
            intensities = np.ones(points_length).tolist()
            
        # Ensure all data is converted to list, not numpy array
        if isinstance(points, np.ndarray):
            points = points.tolist()
        if isinstance(intensities, np.ndarray):
            intensities = intensities.tolist()
        if isinstance(color, np.ndarray):
            color = color.tolist()
            
        processed_point_clouds.append({
            'index': i + 1,
            'name': name,
            'points': points,
            'intensities': intensities,
            'color': color,
            'point_count': len(points)
        })
    
    if not processed_point_clouds:
        print("Error: No valid point clouds to display")
        return None
    
    # Create data object for visualization
    data = {
        'pointClouds': processed_point_clouds,
        'pointSize': point_size,
        'names': [pc['name'] for pc in processed_point_clouds]
    }
    
    # Generate the HTML content
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Point Cloud Visualization</title>
    <style>
        body { margin: 0; overflow: hidden; font-family: Arial, sans-serif; }
        #controls { 
            position: absolute; 
            top: 10px; 
            right: 10px; 
            background: rgba(0,0,0,0.7); 
            color: white; 
            padding: 10px; 
            border-radius: 5px;
            max-width: 350px;
            z-index: 10;
            max-height: 90vh;
            overflow-y: auto;
        }
        #info { 
            position: absolute; 
            top: 10px; 
            left: 10px; 
            background: rgba(0,0,0,0.7); 
            color: white; 
            padding: 10px;
            border-radius: 5px;
            max-width: 300px;
            z-index: 10;
            max-height: 90vh;
            overflow-y: auto;
        }
        button { 
            margin: 5px;
            padding: 5px 10px;
            cursor: pointer;
        }
        .slider-container {
            margin: 10px 0;
            display: flex;
            align-items: center;
        }
        label {
            display: inline-block;
            width: 120px;
        }
        input[type="range"] {
            flex-grow: 1;
        }
        .value-display {
            width: 40px;
            text-align: right;
            margin-left: 10px;
        }
        .legend-box {
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 5px;
            vertical-align: middle;
        }
        .point-cloud-toggle {
            display: flex;
            align-items: center;
            margin: 5px 0;
        }
        .point-cloud-toggle button {
            flex-grow: 1;
            margin-left: 5px;
        }
    </style>
</head>
<body>
    <div id="info">
        <h3>Point Cloud Visualization</h3>
        <div id="legend"></div>
        <p><strong>Controls:</strong></p>
        <p>- Left-click and drag: Rotate camera</p>
        <p>- Right-click and drag: Pan camera</p>
        <p>- Mouse wheel: Zoom in/out</p>
    </div>
    <div id="controls">
        <h3>Point Cloud Controls</h3>
        <div id="pointcloud-toggles"></div>
        <div>
            <button id="show-all">Show All</button>
            <button id="hide-all">Hide All</button>
        </div>
        <div class="slider-container">
            <label for="point-size">Point Size:</label>
            <input type="range" id="point-size-slider" min="0.1" max="10" step="0.1" value="${data.pointSize}">
            <span id="point-size-value" class="value-display">${data.pointSize}</span>
        </div>
        <h3>Display Controls</h3>
        <div>
            <button id="toggle-axes">Hide Axes</button>
            <button id="toggle-grid">Hide Grid</button>
        </div>
        <div>
            <button id="reset-view">Reset View</button>
            <button id="save-image">Save Image</button>
        </div>
        <div class="slider-container">
            <label for="background-color">Background:</label>
            <select id="background-color">
                <option value="dark">Dark</option>
                <option value="light">Light</option>
                <option value="gradient">Gradient</option>
            </select>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Embedded data
        const data = DATA_PLACEHOLDER;
        
        // Main variables
        let scene, camera, renderer;
        let pointClouds = {};
        let axesHelper, gridHelper;
        let pointSize = data.pointSize;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            // Create legend
            createLegend();
            
            // Create point cloud toggle controls
            createPointCloudToggles();
                
            // Setup scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x111111);
            
            // Setup camera
            camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
            
            // Setup renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);
            
            // Create visualization elements
            createPointClouds();
            
            // Add axes
            axesHelper = new THREE.AxesHelper(30);
            scene.add(axesHelper);
            
            // Add grid
            gridHelper = new THREE.GridHelper(100, 10);
            scene.add(gridHelper);
            
            // Add light
            scene.add(new THREE.AmbientLight(0xffffff, 0.8));
            
            // Center camera
            centerCamera();
            
            // Add event listeners
            setupControls();
            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });
            
            // Show/hide all point clouds
            document.getElementById('show-all').addEventListener('click', () => {
                data.pointClouds.forEach(pc => {
                    const pcId = `pc-${pc.index}`;
                    pointClouds[pcId].visible = true;
                    document.getElementById(`toggle-${pcId}`).textContent = `Hide ${pc.name}`;
                });
            });
            
            document.getElementById('hide-all').addEventListener('click', () => {
                data.pointClouds.forEach(pc => {
                    const pcId = `pc-${pc.index}`;
                    pointClouds[pcId].visible = false;
                    document.getElementById(`toggle-${pcId}`).textContent = `Show ${pc.name}`;
                });
            });
            
            // Point size slider
            document.getElementById('point-size-slider').addEventListener('input', function() {
                pointSize = parseFloat(this.value);
                document.getElementById('point-size-value').textContent = pointSize.toFixed(1);
                updatePointSize();
            });
            
            // Button controls for axes and grid
            document.getElementById('toggle-axes').addEventListener('click', () => {
                axesHelper.visible = !axesHelper.visible;
                document.getElementById('toggle-axes').textContent = 
                    axesHelper.visible ? 'Hide Axes' : 'Show Axes';
            });
            
            document.getElementById('toggle-grid').addEventListener('click', () => {
                gridHelper.visible = !gridHelper.visible;
                document.getElementById('toggle-grid').textContent = 
                    gridHelper.visible ? 'Hide Grid' : 'Show Grid';
            });
            
            document.getElementById('reset-view').addEventListener('click', centerCamera);
            
            document.getElementById('save-image').addEventListener('click', () => {
                renderer.render(scene, camera);
                const link = document.createElement('a');
                link.href = renderer.domElement.toDataURL('image/png');
                link.download = 'pointcloud_visualization.png';
                link.click();
            });
            
            // Background color selector
            document.getElementById('background-color').addEventListener('change', function() {
                const value = this.value;
                
                if (value === 'dark') {
                    scene.background = new THREE.Color(0x111111);
                } else if (value === 'light') {
                    scene.background = new THREE.Color(0xf0f0f0);
                } else if (value === 'gradient') {
                    const canvas = document.createElement('canvas');
                    canvas.width = 2;
                    canvas.height = 2;
                    
                    const context = canvas.getContext('2d');
                    const gradient = context.createLinearGradient(0, 0, 0, 2);
                    gradient.addColorStop(0, '#1a2a3a');
                    gradient.addColorStop(1, '#2c3e50');
                    
                    context.fillStyle = gradient;
                    context.fillRect(0, 0, 2, 2);
                    
                    const texture = new THREE.CanvasTexture(canvas);
                    texture.needsUpdate = true;
                    scene.background = texture;
                }
            });
            
            // Start rendering
            animate();
        });
        
        function createLegend() {
            const legend = document.getElementById('legend');
            
            // Create legend entries for each point cloud
            data.pointClouds.forEach(pc => {
                const colorStyle = `rgb(${Math.floor(pc.color[0]*255)}, ${Math.floor(pc.color[1]*255)}, ${Math.floor(pc.color[2]*255)})`;
                
                const legendEntry = document.createElement('p');
                legendEntry.innerHTML = `<span class="legend-box" style="background-color: ${colorStyle};"></span> ${pc.name}: <span id="point-count-${pc.index}">${pc.point_count}</span> points`;
                
                legend.appendChild(legendEntry);
            });
        }
        
        function createPointCloudToggles() {
            const togglesContainer = document.getElementById('pointcloud-toggles');
            
            // Create toggle buttons for each point cloud
            data.pointClouds.forEach(pc => {
                const pcId = `pc-${pc.index}`;
                
                const toggleDiv = document.createElement('div');
                toggleDiv.classList.add('point-cloud-toggle');
                
                const colorBox = document.createElement('span');
                colorBox.classList.add('legend-box');
                colorBox.style.backgroundColor = `rgb(${Math.floor(pc.color[0]*255)}, ${Math.floor(pc.color[1]*255)}, ${Math.floor(pc.color[2]*255)})`;
                
                const toggleButton = document.createElement('button');
                toggleButton.id = `toggle-${pcId}`;
                toggleButton.textContent = `Hide ${pc.name}`;
                toggleButton.addEventListener('click', () => {
                    pointClouds[pcId].visible = !pointClouds[pcId].visible;
                    toggleButton.textContent = pointClouds[pcId].visible ? 
                        `Hide ${pc.name}` : `Show ${pc.name}`;
                });
                
                toggleDiv.appendChild(colorBox);
                toggleDiv.appendChild(toggleButton);
                togglesContainer.appendChild(toggleDiv);
            });
        }
        
        function createPointClouds() {
            // Calculate the overall bounding box
            let minX = Infinity, minY = Infinity, minZ = Infinity;
            let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
            
            // Create point cloud for each dataset
            data.pointClouds.forEach(pc => {
                const pcId = `pc-${pc.index}`;
                
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(pc.points.flat());
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                
                // Update bounding box
                for (let i = 0; i < pc.points.length; i++) {
                    const point = pc.points[i];
                    minX = Math.min(minX, point[0]);
                    minY = Math.min(minY, point[1]);
                    minZ = Math.min(minZ, point[2]);
                    maxX = Math.max(maxX, point[0]);
                    maxY = Math.max(maxY, point[1]);
                    maxZ = Math.max(maxZ, point[2]);
                }
                
                // Set colors based on intensities and assigned color
                const colors = new Float32Array(pc.points.length * 3);
                for (let i = 0, j = 0; i < pc.points.length; i++, j += 3) {
                    // Get intensity safely (use 1.0 as default if missing)
                    const intensity = (i < pc.intensities.length) ? pc.intensities[i] : 1.0;
                    // Base color with intensity scaling
                    colors[j] = pc.color[0] * (0.3 + 0.7 * intensity);
                    colors[j+1] = pc.color[1] * (0.3 + 0.7 * intensity);
                    colors[j+2] = pc.color[2] * (0.3 + 0.7 * intensity);
                }
                geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                
                // Create point cloud
                pointClouds[pcId] = new THREE.Points(
                    geometry,
                    new THREE.PointsMaterial({ 
                        size: pointSize, 
                        vertexColors: true, 
                        sizeAttenuation: true 
                    })
                );
                scene.add(pointClouds[pcId]);
            });
            
            // Store bounding box for camera positioning
            window.boundingBox = {
                min: { x: minX, y: minY, z: minZ },
                max: { x: maxX, y: maxY, z: maxZ },
                center: { 
                    x: (minX + maxX) / 2,
                    y: (minY + maxY) / 2,
                    z: (minZ + maxZ) / 2
                },
                size: {
                    x: maxX - minX,
                    y: maxY - minY,
                    z: maxZ - minZ
                }
            };
        }
        
        function updatePointSize() {
            // Update the point size for all point clouds
            data.pointClouds.forEach(pc => {
                const pcId = `pc-${pc.index}`;
                pointClouds[pcId].material.size = pointSize;
            });
        }
        
        function setupControls() {
            let isMouseDown = false;
            let previousMousePosition = { x: 0, y: 0 };
            let targetPosition;
            
            if (window.boundingBox) {
                targetPosition = new THREE.Vector3(
                    window.boundingBox.center.x,
                    window.boundingBox.center.y,
                    window.boundingBox.center.z
                );
            } else {
                targetPosition = new THREE.Vector3(0, 0, 0);
            }
            
            // Mouse controls
            document.addEventListener('mousedown', event => {
                isMouseDown = true;
                previousMousePosition = { x: event.clientX, y: event.clientY };
            });
            
            document.addEventListener('mousemove', event => {
                if (!isMouseDown) return;
                
                const deltaMove = {
                    x: event.clientX - previousMousePosition.x,
                    y: event.clientY - previousMousePosition.y
                };
                
                if (event.buttons === 1) { // Left click = rotate
                    const rotateSpeed = 0.005;
                    
                    // Get camera position relative to target
                    const offset = new THREE.Vector3().subVectors(camera.position, targetPosition);
                    
                    // Rotate around Y axis
                    const angleY = -deltaMove.x * rotateSpeed;
                    offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), angleY);
                    
                    // Rotate around local X axis
                    const angleX = -deltaMove.y * rotateSpeed;
                    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
                    offset.applyAxisAngle(right, angleX);
                    
                    // Update camera position
                    camera.position.copy(targetPosition).add(offset);
                    camera.lookAt(targetPosition);
                } else if (event.buttons === 2) { // Right click = pan
                    const panSpeed = 0.1;
                    
                    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion).multiplyScalar(deltaMove.x * panSpeed);
                    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion).multiplyScalar(-deltaMove.y * panSpeed);
                    
                    camera.position.add(right).add(up);
                    targetPosition.add(right).add(up);
                }
                
                previousMousePosition = { x: event.clientX, y: event.clientY };
            });
            
            document.addEventListener('mouseup', () => { isMouseDown = false; });
            document.addEventListener('contextmenu', event => { event.preventDefault(); });
            
            // Mouse wheel zoom
            document.addEventListener('wheel', event => {
                event.preventDefault();
                
                const zoomSpeed = 0.1;
                const direction = event.deltaY > 0 ? 1 : -1;
                
                const offset = new THREE.Vector3().subVectors(camera.position, targetPosition);
                const distance = offset.length();
                offset.normalize().multiplyScalar(distance * (1 + direction * zoomSpeed));
                
                camera.position.copy(targetPosition).add(offset);
            });
        }
        
        function centerCamera() {
            if (window.boundingBox) {
                const center = new THREE.Vector3(
                    window.boundingBox.center.x,
                    window.boundingBox.center.y,
                    window.boundingBox.center.z
                );
                
                const maxDimension = Math.max(
                    window.boundingBox.size.x,
                    window.boundingBox.size.y,
                    window.boundingBox.size.z
                );
                
                const distance = maxDimension * 2;
                
                // Position camera
                camera.position.set(
                    center.x + distance,
                    center.y + distance,
                    center.z + distance
                );
                camera.lookAt(center);
            } else {
                // Fallback if no bounding box
                camera.position.set(50, 50, 50);
                camera.lookAt(0, 0, 0);
            }
        }
        
        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }
    </script>
</body>
</html>"""

    # Insert data
    json_data = json.dumps(data, separators=(',', ':'))
    html_content = html_content.replace('DATA_PLACEHOLDER', json_data)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Saved point cloud visualization to: {output_path}")
    print(f"This visualization includes {len(data['pointClouds'])} point clouds.")
    for pc in processed_point_clouds:
        print(f"- {pc['name']}: {pc['point_count']} points")
    
    return output_path


def example_usage():
    """Example function that creates and visualizes some sample point clouds."""
    # Create some example point clouds
    import numpy as np
    
    # First point cloud: a simple spiral
    t = np.linspace(0, 10 * np.pi, 1000)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = t
    spiral_points = np.column_stack((x, y, z))
    
    # Second point cloud: a sphere
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 20)
    phi, theta = np.meshgrid(phi, theta)
    
    r = 30
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    sphere_points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    
    # Create point cloud data structures
    point_clouds = [
        {
            'name': 'Spiral',
            'points': spiral_points,
            'color': [1, 0, 0]  # Red
        },
        {
            'name': 'Sphere',
            'points': sphere_points,
            'color': [0, 0, 1]  # Blue
        }
    ]
    
    # Visualize point clouds (uses default path with timestamp)
    output_path = visualize_point_clouds(point_clouds)
    print(f"Example point clouds visualization created at: {output_path}")


if __name__ == "__main__":
    # Run the example if called directly
    example_usage()