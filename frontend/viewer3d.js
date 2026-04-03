/**
 * ADDS 3D Medical Viewer
 * Three.js-based interactive viewer for CT organs and tumors
 */

// Global state
let scene, camera, renderer, controls;
let meshObjects = {
    body: null,
    soft_tissue: null,
    colon: null,
    tumors: []
};
let config = {
    patientId: 'default',
    apiBaseUrl: 'http://localhost:8000/api/v1/ct'
};

// Initialize viewer
async function init() {
    console.log('[Viewer3D] Initializing...');

    // Get URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    config.patientId = urlParams.get('patient_id') || 'default';

    // Setup scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    // Add subtle ambient light and directional light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight1.position.set(1, 1, 1);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
    dirLight2.position.set(-1, -1, -1);
    scene.add(dirLight2);

    // Setup camera
    const container = document.getElementById('canvas-3d');
    const aspect = container.clientWidth / container.clientHeight;
    camera = new THREE.PerspectiveCamera(50, aspect, 1, 5000);
    camera.position.set(300, 300, 300);

    // Setup renderer
    renderer = new THREE.WebGLRenderer({
        canvas: container,
        antialias: true,
        alpha: true
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Setup controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 50;
    controls.maxDistance = 2000;

    // Add coordinate axes helper
    const axesHelper = new THREE.AxesHelper(100);
    scene.add(axesHelper);

    // Load mesh data
    await loadMeshData();

    // Setup UI controls
    setupControls();

    // Start animation loop
    animate();

    // Hide loading screen
    document.getElementById('loading').style.display = 'none';
    document.getElementById('controls-panel').style.display = 'block';

    console.log('[Viewer3D] Initialization complete');
}

// Load mesh data from API
async function loadMeshData() {
    console.log(`[Viewer3D] Loading mesh data for patient ${config.patientId}...`);

    try {
        const response = await fetch(
            `${config.apiBaseUrl}/${config.patientId}/mesh3d?mesh_type=all`
        );

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        console.log('[Viewer3D] Mesh data loaded:', data);

        // Create organ meshes
        if (data.organs) {
            if (data.organs.body) {
                meshObjects.body = createMesh(
                    data.organs.body,
                    0xcccccc,
                    0.1,
                    'Body'
                );
            }

            if (data.organs.soft_tissue) {
                meshObjects.soft_tissue = createMesh(
                    data.organs.soft_tissue,
                    0xffccaa,
                    0.2,
                    'Soft Tissue'
                );
            }

            if (data.organs.colon) {
                meshObjects.colon = createMesh(
                    data.organs.colon,
                    0xff8800,
                    0.3,
                    'Colon'
                );
            }
        }

        // Create tumor meshes
        if (data.tumors && data.tumors.length > 0) {
            console.log(`[Viewer3D] Creating ${data.tumors.length} tumor markers`);

            data.tumors.forEach((tumorData, index) => {
                const tumorMesh = createMesh(
                    tumorData,
                    0xff0000,
                    0.9,
                    `Tumor ${index + 1}`
                );
                meshObjects.tumors.push(tumorMesh);
            });
        }

        // Center camera on scene
        centerCamera();

    } catch (error) {
        console.error('[Viewer3D] Error loading mesh data:', error);
        document.getElementById('loading').innerHTML = `
            <div style="color: #ff0000; font-weight: bold;">Error loading 3D data</div>
            <div style="margin-top: 10px; font-size: 12px;">${error.message}</div>
            <div style="margin-top: 10px; font-size: 11px;">
                Make sure CT analysis has been run and 3D segmentation files exist.
            </div>
        `;
    }
}

// Create Three.js mesh from mesh data
function createMesh(meshData, color, opacity, name) {
    console.log(`[Viewer3D] Creating mesh: ${name}`);

    const vertices = new Float32Array(meshData.vertices.flat());
    const indices = new Uint32Array(meshData.faces.flat());

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
        color: color,
        transparent: true,
        opacity: opacity,
        side: THREE.DoubleSide,
        flatShading: false,
        shininess: 30
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = name;
    scene.add(mesh);

    console.log(`[Viewer3D] Mesh created: ${name} (${meshData.num_vertices} vertices, ${meshData.num_faces} faces)`);

    return mesh;
}

// Center camera on all visible objects
function centerCamera() {
    const box = new THREE.Box3();

    scene.traverse((object) => {
        if (object.isMesh && object.visible) {
            box.expandByObject(object);
        }
    });

    if (!box.isEmpty()) {
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());

        // Position camera to view entire scene
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = camera.fov * (Math.PI / 180);
        let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
        cameraZ *= 1.5; // Add some margin

        camera.position.set(center.x + cameraZ / 2, center.y + cameraZ / 2, center.z + cameraZ);
        camera.lookAt(center);

        controls.target.copy(center);
        controls.update();

        console.log('[Viewer3D] Camera centered on scene');
    }
}

// Setup UI control event listeners
function setupControls() {
    // Reset view button
    document.getElementById('btn-reset').addEventListener('click', () => {
        centerCamera();
    });

    // Visibility toggles
    document.getElementById('show-body').addEventListener('change', (e) => {
        if (meshObjects.body) {
            meshObjects.body.visible = e.target.checked;
        }
    });

    document.getElementById('show-soft-tissue').addEventListener('change', (e) => {
        if (meshObjects.soft_tissue) {
            meshObjects.soft_tissue.visible = e.target.checked;
        }
    });

    document.getElementById('show-colon').addEventListener('change', (e) => {
        if (meshObjects.colon) {
            meshObjects.colon.visible = e.target.checked;
        }
    });

    document.getElementById('show-tumors').addEventListener('change', (e) => {
        meshObjects.tumors.forEach(tumor => {
            tumor.visible = e.target.checked;
        });
    });

    // Opacity sliders
    document.getElementById('opacity-colon').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        document.getElementById('val-colon').textContent = value.toFixed(2);
        if (meshObjects.colon) {
            meshObjects.colon.material.opacity = value;
        }
    });

    document.getElementById('opacity-tissue').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        document.getElementById('val-tissue').textContent = value.toFixed(2);
        if (meshObjects.soft_tissue) {
            meshObjects.soft_tissue.material.opacity = value;
        }
    });

    document.getElementById('opacity-body').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        document.getElementById('val-body').textContent = value.toFixed(2);
        if (meshObjects.body) {
            meshObjects.body.material.opacity = value;
        }
    });

    // Tumor scale slider
    document.getElementById('scale-tumor').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        document.getElementById('val-tumor').textContent = value.toFixed(1);
        meshObjects.tumors.forEach(tumor => {
            tumor.scale.set(value, value, value);
        });
    });
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Handle window resize
window.addEventListener('resize', () => {
    const container = document.getElementById('canvas-3d');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});

// Start initialization when page loads
window.addEventListener('DOMContentLoaded', init);
