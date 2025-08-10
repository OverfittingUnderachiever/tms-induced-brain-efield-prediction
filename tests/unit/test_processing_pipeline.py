import numpy as np
import os
import sys
import json

# Add project path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import only what we need
from tms_efield_prediction.data.transformations.mesh_to_grid import MeshToGridTransformer

def main():
    """Process TMS E-field data and create a self-contained interactive visualization."""
    print("=== TMS E-field Self-Contained Visualization Generator ===")
    
    # 1. Load the E-field data
    efield_path = '/home/freyhe/MA_Henry/data/sub-005/experiment/multi_sim_100/sub-005_all_efields.npy'
    
    print(f"Loading E-field data from: {efield_path}")
    efield_data = np.load(efield_path, allow_pickle=True)
    print(f"E-field data shape: {efield_data.shape}")
    
    # 2. Process the first position
    position_idx = 0
    print(f"Processing position {position_idx}")
    efield_sample = efield_data[position_idx]
    print(f"E-field sample shape: {efield_sample.shape}")
    
    # 3. Create minimal context for the transformer
    class MinimalContext:
        def __init__(self):
            self.config = {'mask_dilation': False}
    
    # 4. Create coordinates for voxelization
    # Use a grid to distribute points in 3D space (this is an approximation)
    efield_coords = np.zeros((efield_sample.shape[0], 3))
    
    # Create a cubic lattice of points
    side_length = int(np.ceil(np.cbrt(efield_sample.shape[0])))
    coords_1d = np.linspace(-1, 1, side_length)
    
    # Fill in coordinates
    idx = 0
    for x in coords_1d:
        for y in coords_1d:
            for z in coords_1d:
                if idx < efield_sample.shape[0]:
                    efield_coords[idx] = [x, y, z]
                    idx += 1
    
    # 5. Create transformer and voxelize the data
    n_bins = 48  # Using smaller grid for better browser performance
    print(f"Voxelizing E-field data to {n_bins}x{n_bins}x{n_bins} grid...")
    
    transformer = MeshToGridTransformer(context=MinimalContext(), debug_hook=None, resource_monitor=None)
    
    # 6. Check if we have vector or scalar E-field data
    if len(efield_sample.shape) > 1 and efield_sample.shape[1] > 1:
        # Vector data
        voxelized_efield = np.zeros((n_bins, n_bins, n_bins, efield_sample.shape[1]), dtype=np.float32)
        
        for i in range(efield_sample.shape[1]):
            component_data = efield_sample[:, i]
            voxelized_component, _, _ = transformer.transform(component_data, efield_coords, n_bins)
            voxelized_efield[..., i] = voxelized_component
            
        print(f"Voxelized E-field shape: {voxelized_efield.shape} (vector data)")
    else:
        # Scalar data
        voxelized_efield, _, _ = transformer.transform(efield_sample, efield_coords, n_bins)
        print(f"Voxelized E-field shape: {voxelized_efield.shape} (scalar data)")
    
    # 7. Calculate E-field magnitude if it's vector data
    is_vector = len(voxelized_efield.shape) > 3
    
    if is_vector:
        # Calculate magnitude (Euclidean norm along the last axis)
        magnitude = np.sqrt(np.sum(voxelized_efield**2, axis=-1))
        print(f"E-field magnitude shape: {magnitude.shape}")
    else:
        # Already scalar data
        magnitude = voxelized_efield
        print("Using direct scalar values for visualization")
    
    # 8. Normalize magnitude for visualization
    # Remove any NaN values
    magnitude = np.nan_to_num(magnitude, nan=0.0)
    
    # Get the range
    min_val = np.min(magnitude)
    max_val = np.max(magnitude)
    print(f"E-field magnitude range: {min_val} to {max_val}")
    
    # Normalize to 0-255 range for visualization
    if max_val > min_val:
        normalized = ((magnitude - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(magnitude, dtype=np.uint8)
    
    # 9. Create output directory
    output_dir = "./visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # 10. Prepare the data for embedding in HTML
    # We'll save as a flat array of values with dimensions for the JS code
    flat_data = normalized.flatten().tolist()
    
    # Create a JSON structure with the data and dimensions
    data_json = {
        "dimensions": [n_bins, n_bins, n_bins],
        "data": flat_data,
        "min_value": float(min_val),
        "max_value": float(max_val)
    }
    
    # Convert to a JSON string for embedding in the HTML
    data_json_str = json.dumps(data_json)
    
    # 11. Create the HTML file with embedded data
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TMS E-field Visualization</title>
    <style>
        body {{ margin: 0; overflow: hidden; }}
        canvas {{ display: block; }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            background-color: rgba(0,0,0,0.7);
            padding: 10px;
            font-family: Arial, sans-serif;
            border-radius: 5px;
            pointer-events: none;
        }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            background-color: rgba(0,0,0,0.7);
            padding: 20px;
            font-family: Arial, sans-serif;
            border-radius: 5px;
            z-index: 100;
        }}
        #controls {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            color: white;
            background-color: rgba(0,0,0,0.7);
            padding: 10px;
            font-family: Arial, sans-serif;
            border-radius: 5px;
        }}
        .slider-container {{
            margin: 10px 0;
        }}
        label {{
            display: inline-block;
            width: 150px;
        }}
    </style>
</head>
<body>
    <div id="loading">Initializing visualization...</div>
    <div id="info">
        <h2>TMS E-field Visualization</h2>
        <p>Subject: 005, Position: 0</p>
        <p>Click and drag to rotate. Scroll to zoom.</p>
        <p id="value-info">Value at cursor: N/A</p>
    </div>
    <div id="controls">
        <div class="slider-container">
            <label for="threshold">Threshold:</label>
            <input type="range" id="threshold" min="0" max="255" value="50">
            <span id="threshold-value">50</span>
        </div>
        <div class="slider-container">
            <label for="opacity">Opacity:</label>
            <input type="range" id="opacity" min="0" max="100" value="80">
            <span id="opacity-value">0.8</span>
        </div>
        <div class="slider-container">
            <label for="colormap">Color Map:</label>
            <select id="colormap">
                <option value="viridis">Viridis</option>
                <option value="plasma">Plasma</option>
                <option value="inferno">Inferno</option>
                <option value="magma">Magma</option>
                <option value="hot">Hot</option>
                <option value="cool">Cool</option>
                <option value="rainbow">Rainbow</option>
            </select>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Embedded data (loaded directly instead of fetched)
        const efieldData = {data_json_str};
        
        // Create the scene, camera, and renderer
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        // Set background color
        scene.background = new THREE.Color(0x000000);
        
        // Add some lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);
        
        // Set up camera position
        camera.position.z = 150;
        
        // Add orbit controls for interaction
        class OrbitControls {{
            constructor(camera, domElement) {{
                this.camera = camera;
                this.domElement = domElement;
                this.target = new THREE.Vector3();
                
                // State
                this.rotateStart = new THREE.Vector2();
                this.rotateEnd = new THREE.Vector2();
                this.rotateDelta = new THREE.Vector2();
                
                this.zoomSpeed = 1.0;
                this.rotateSpeed = 1.0;
                
                this.isRotating = false;
                
                // Distance constraints
                this.minDistance = 10;
                this.maxDistance = 500;
                
                // Events
                this.domElement.addEventListener('mousedown', this.onMouseDown.bind(this), false);
                this.domElement.addEventListener('mousemove', this.onMouseMove.bind(this), false);
                this.domElement.addEventListener('mouseup', this.onMouseUp.bind(this), false);
                this.domElement.addEventListener('wheel', this.onMouseWheel.bind(this), false);
                
                this.update();
            }}
            
            onMouseDown(event) {{
                this.isRotating = true;
                this.rotateStart.set(event.clientX, event.clientY);
            }}
            
            onMouseMove(event) {{
                if (!this.isRotating) return;
                
                this.rotateEnd.set(event.clientX, event.clientY);
                this.rotateDelta.subVectors(this.rotateEnd, this.rotateStart);
                
                // Rotate camera
                const element = this.domElement;
                this.rotateLeft(2 * Math.PI * this.rotateDelta.x / element.clientWidth * this.rotateSpeed);
                this.rotateUp(2 * Math.PI * this.rotateDelta.y / element.clientHeight * this.rotateSpeed);
                
                this.rotateStart.copy(this.rotateEnd);
                this.update();
            }}
            
            onMouseUp() {{
                this.isRotating = false;
            }}
            
            onMouseWheel(event) {{
                event.preventDefault();
                
                // Zoom
                if (event.deltaY < 0) {{
                    this.dollyOut();
                }} else {{
                    this.dollyIn();
                }}
                
                this.update();
            }}
            
            rotateLeft(angle) {{
                const quat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), angle);
                const offset = new THREE.Vector3().subVectors(this.camera.position, this.target);
                offset.applyQuaternion(quat);
                this.camera.position.copy(this.target).add(offset);
                this.camera.lookAt(this.target);
            }}
            
            rotateUp(angle) {{
                const quat = new THREE.Quaternion().setFromAxisAngle(
                    new THREE.Vector3(1, 0, 0).cross(this.camera.position).normalize(),
                    angle
                );
                const offset = new THREE.Vector3().subVectors(this.camera.position, this.target);
                offset.applyQuaternion(quat);
                this.camera.position.copy(this.target).add(offset);
                this.camera.lookAt(this.target);
            }}
            
            dollyIn() {{
                const zoomScale = 0.95;
                const offset = new THREE.Vector3().subVectors(this.camera.position, this.target);
                const distance = offset.length();
                offset.multiplyScalar(zoomScale);
                if (offset.length() < this.minDistance) {{
                    offset.normalize().multiplyScalar(this.minDistance);
                }}
                this.camera.position.copy(this.target).add(offset);
            }}
            
            dollyOut() {{
                const zoomScale = 1.05;
                const offset = new THREE.Vector3().subVectors(this.camera.position, this.target);
                const distance = offset.length();
                offset.multiplyScalar(zoomScale);
                if (offset.length() > this.maxDistance) {{
                    offset.normalize().multiplyScalar(this.maxDistance);
                }}
                this.camera.position.copy(this.target).add(offset);
            }}
            
            update() {{
                this.camera.lookAt(this.target);
            }}
        }}
        
        const controls = new OrbitControls(camera, renderer.domElement);
        
        // Color maps
        const colorMaps = {{
            viridis: [
                [0.267004, 0.004874, 0.329415],
                [0.282656, 0.140926, 0.457517],
                [0.253935, 0.265254, 0.529983],
                [0.206756, 0.371758, 0.553117],
                [0.163625, 0.471133, 0.558148],
                [0.127568, 0.566949, 0.550556],
                [0.134692, 0.658636, 0.517649],
                [0.266941, 0.748751, 0.440573],
                [0.477504, 0.821444, 0.318195],
                [0.741388, 0.873449, 0.149561],
                [0.993248, 0.906157, 0.143936]
            ],
            plasma: [
                [0.050383, 0.029803, 0.527975],
                [0.258099, 0.038279, 0.604253],
                [0.438813, 0.012249, 0.633101],
                [0.611205, 0.013284, 0.610176],
                [0.756447, 0.089923, 0.541490],
                [0.871788, 0.195811, 0.440369],
                [0.951130, 0.336057, 0.321569],
                [0.981591, 0.489951, 0.225683],
                [0.963961, 0.650391, 0.147607],
                [0.897922, 0.813513, 0.076105],
                [0.720893, 0.954031, 0.134296]
            ],
            inferno: [
                [0.001462, 0.000466, 0.013866],
                [0.087411, 0.020179, 0.145290],
                [0.210793, 0.017019, 0.246995],
                [0.340697, 0.039772, 0.294176],
                [0.469012, 0.092513, 0.285889],
                [0.589330, 0.159766, 0.246066],
                [0.701108, 0.235281, 0.196066],
                [0.804149, 0.328332, 0.142881],
                [0.892778, 0.444455, 0.116128],
                [0.962886, 0.586037, 0.137530],
                [0.988350, 0.751213, 0.265551]
            ],
            magma: [
                [0.001462, 0.000466, 0.013866],
                [0.093350, 0.019737, 0.175238],
                [0.209885, 0.018004, 0.298619],
                [0.329398, 0.047214, 0.390945],
                [0.444801, 0.092968, 0.453418],
                [0.555581, 0.148516, 0.488173],
                [0.661355, 0.214981, 0.495662],
                [0.763408, 0.293143, 0.474586],
                [0.858765, 0.387057, 0.420837],
                [0.940015, 0.504053, 0.345110],
                [0.988350, 0.751213, 0.265551]
            ],
            hot: [
                [0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [1.0, 0.3, 0.0],
                [1.0, 0.6, 0.0],
                [1.0, 0.9, 0.0],
                [1.0, 1.0, 0.3],
                [1.0, 1.0, 0.6],
                [1.0, 1.0, 0.9],
                [1.0, 1.0, 1.0]
            ],
            cool: [
                [0.0, 1.0, 1.0],
                [0.1, 0.9, 1.0],
                [0.2, 0.8, 1.0],
                [0.3, 0.7, 1.0],
                [0.4, 0.6, 1.0],
                [0.5, 0.5, 1.0],
                [0.6, 0.4, 1.0],
                [0.7, 0.3, 1.0],
                [0.8, 0.2, 1.0],
                [0.9, 0.1, 1.0],
                [1.0, 0.0, 1.0]
            ],
            rainbow: [
                [0.5, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.5, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.5],
                [0.0, 1.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.5]
            ]
        }};
        
        function getColorFromMap(value, colorMap) {{
            if (value <= 0) return new THREE.Color(0, 0, 0);
            
            const normValue = Math.min(Math.max(value / 255, 0), 1);
            const index = normValue * (colorMap.length - 1);
            const lowerIndex = Math.floor(index);
            const upperIndex = Math.ceil(index);
            const blend = index - lowerIndex;
            
            const lowerColor = colorMap[lowerIndex];
            const upperColor = colorMap[upperIndex];
            
            const r = lowerColor[0] * (1 - blend) + upperColor[0] * blend;
            const g = lowerColor[1] * (1 - blend) + upperColor[1] * blend;
            const b = lowerColor[2] * (1 - blend) + upperColor[2] * blend;
            
            return new THREE.Color(r, g, b);
        }}
        
        // Variables for the visualization
        let voxelSize = 1;
        let thresholdValue = 50;
        let opacity = 0.8;
        let currentColorMap = 'viridis';
        let points;
        
        // Initialize with the embedded data
        const voxelData = efieldData.data;
        const dimensions = efieldData.dimensions;
        
        // Remove loading indicator and add data info
        document.getElementById('loading').style.display = 'none';
        document.getElementById('info').innerHTML += `<p>Data range: ${{efieldData.min_value.toFixed(6)}} to ${{efieldData.max_value.toFixed(6)}}</p>`;
        
        // Create the visualization
        createVisualization();
        
        // Set up threshold slider
        const thresholdSlider = document.getElementById('threshold');
        const thresholdValueDisplay = document.getElementById('threshold-value');
        
        thresholdSlider.addEventListener('input', function() {{
            thresholdValue = parseInt(this.value);
            thresholdValueDisplay.textContent = thresholdValue;
            updateVisualization();
        }});
        
        // Set up opacity slider
        const opacitySlider = document.getElementById('opacity');
        const opacityValueDisplay = document.getElementById('opacity-value');
        
        opacitySlider.addEventListener('input', function() {{
            opacity = parseInt(this.value) / 100;
            opacityValueDisplay.textContent = opacity.toFixed(2);
            updateVisualization();
        }});
        
        // Set up colormap selector
        const colormapSelector = document.getElementById('colormap');
        
        colormapSelector.addEventListener('change', function() {{
            currentColorMap = this.value;
            updateVisualization();
        }});
        
        function createVisualization() {{
            const [nx, ny, nz] = dimensions;
            const geometry = new THREE.BufferGeometry();
            const material = new THREE.PointsMaterial({{
                size: voxelSize,
                vertexColors: true,
                sizeAttenuation: true,
                transparent: true,
                opacity: opacity
            }});
            
            // Center the visualization
            const offsetX = -nx / 2;
            const offsetY = -ny / 2;
            const offsetZ = -nz / 2;
            
            // Create positions and colors arrays
            let positions = [];
            let colors = [];
            
            // For each voxel above the threshold
            for (let i = 0; i < nx; i++) {{
                for (let j = 0; j < ny; j++) {{
                    for (let k = 0; k < nz; k++) {{
                        const idx = i + j * nx + k * nx * ny;
                        const value = voxelData[idx];
                        
                        if (value > thresholdValue) {{
                            // Add position
                            positions.push(i + offsetX, j + offsetY, k + offsetZ);
                            
                            // Add color from colormap
                            const color = getColorFromMap(value, colorMaps[currentColorMap]);
                            colors.push(color.r, color.g, color.b);
                        }}
                    }}
                }}
            }}
            
            // Set attributes
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            
            // Create points object
            points = new THREE.Points(geometry, material);
            scene.add(points);
            
            // Center camera target on the object
            controls.target.set(0, 0, 0);
            controls.update();
        }}
        
        function updateVisualization() {{
            // Remove existing points
            if (points) {{
                scene.remove(points);
                points.geometry.dispose();
                points.material.dispose();
            }}
            
            // Create new visualization with updated parameters
            createVisualization();
        }}
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }}
        
        animate();
        
        // Handle window resize
        window.addEventListener('resize', function() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
    """
    
    # Save the HTML file
    html_path = os.path.join(output_dir, "efield_visualization_standalone.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Saved self-contained visualization to: {html_path}")
    print(f"Open this file in any web browser to view the interactive visualization")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()