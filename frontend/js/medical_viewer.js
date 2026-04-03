/**
 * ADDS 3D Medical Viewer
 * Three.js 기반 인터랙티브 의료 영상 뷰어
 */

// Three.js 전역 변수
let scene, camera, renderer, controls;
let organMeshes = {};
let tumorMeshes = [];
let autoRotate = false;
let currentPatientId = 'PT001';
let currentTimepoint = 'T0';

// 초기화
function init() {
    console.log('Initializing 3D Medical Viewer...');

    // Scene 생성
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1e3c72);
    scene.fog = new THREE.Fog(0x1e3c72, 200, 1000);

    // Camera
    const canvas = document.getElementById('viewer-canvas');
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
    camera.position.set(0, 200, 400);
    camera.lookAt(0, 0, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true,
        alpha: true
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;

    // Controls (OrbitControls)
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 100;
    controls.maxDistance = 800;
    controls.maxPolarAngle = Math.PI;

    // 조명
    setupLights();

    // 그리드 헬퍼
    const gridHelper = new THREE.GridHelper(400, 20, 0x888888, 0x444444);
    gridHelper.position.y = -100;
    scene.add(gridHelper);

    // 축 헬퍼
    const axesHelper = new THREE.AxesHelper(150);
    axesHelper.position.y = -100;
    scene.add(axesHelper);

    // 윈도우 리사이즈
    window.addEventListener('resize', onWindowResize, false);

    // 데이터 로드
    loadPatientData(currentPatientId, currentTimepoint);

    // 애니메이션 시작
    animate();

    // 로딩 숨기기
    setTimeout(() => {
        document.getElementById('loading').style.display = 'none';
    }, 2000);
}

// 조명 설정
function setupLights() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // Directional lights
    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.7);
    dirLight1.position.set(200, 200, 200);
    dirLight1.castShadow = true;
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    dirLight2.position.set(-200, 100, -200);
    scene.add(dirLight2);

    // Hemisphere light
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.3);
    hemiLight.position.set(0, 200, 0);
    scene.add(hemiLight);
}

// 환자 데이터 로드
async function loadPatientData(patientId, timepoint) {
    console.log(`Loading data for ${patientId} at ${timepoint}`);

    try {
        // API에서 3D 메시 데이터 가져오기
        const response = await fetch(`http://localhost:8001/api/patients/${patientId}/3d-meshes?timepoint=${timepoint}`);
        const data = await response.json();

        // 기존 메시 제거
        clearScene();

        // 장기 메시 로드
        if (data.organs) {
            loadOrganMeshes(data.organs);
        }

        // 종양 메시 로드
        if (data.tumors) {
            loadTumorMeshes(data.tumors);
        }

        // UI 업데이트
        updateUI(data);

    } catch (error) {
        console.error('Failed to load patient data:', error);
        // 데모 데이터 로드
        loadDemoData();
    }
}

// 장기 메시 로드
function loadOrganMeshes(organs) {
    const organColors = {
        'colon': 0xff9664,
        'liver': 0x964632,
        'kidneys': 0xc86496,
        'spleen': 0x7f3f2f,
        'stomach': 0xffc8a0
    };

    organs.forEach(organ => {
        const geometry = createGeometryFromVertices(organ.vertices, organ.faces);
        const material = new THREE.MeshPhongMaterial({
            color: organColors[organ.name] || 0x888888,
            transparent: true,
            opacity: 0.5,
            side: THREE.DoubleSide,
            flatShading: false
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.name = organ.name;

        scene.add(mesh);
        organMeshes[organ.name] = mesh;
    });
}

// 종양 메시 로드
function loadTumorMeshes(tumors) {
    tumors.forEach((tumor, index) => {
        const geometry = createGeometryFromVertices(tumor.vertices, tumor.faces);
        const material = new THREE.MeshPhongMaterial({
            color: 0xff3232,
            transparent: true,
            opacity: 0.85,
            emissive: 0xff0000,
            emissiveIntensity: 0.2
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.name = `tumor_${index}`;

        // 위치 설정 (centroid)
        if (tumor.centroid) {
            mesh.position.set(
                tumor.centroid[0],
                tumor.centroid[1],
                tumor.centroid[2]
            );
        }

        scene.add(mesh);
        tumorMeshes.push(mesh);
    });
}

// Vertices + Faces에서 Geometry 생성
function createGeometryFromVertices(vertices, faces) {
    const geometry = new THREE.BufferGeometry();

    // Vertices
    const vertexArray = new Float32Array(vertices.flat());
    geometry.setAttribute('position', new THREE.BufferAttribute(vertexArray, 3));

    // Faces (indices)
    if (faces && faces.length > 0) {
        const indexArray = new Uint32Array(faces.flat());
        geometry.setIndex(new THREE.BufferAttribute(indexArray, 1));
    }

    geometry.computeVertexNormals();
    return geometry;
}

// 데모 데이터 (API 없을 때)
function loadDemoData() {
    console.log('Loading demo data...');

    // 대장 (토러스)
    const colonGeometry = new THREE.TorusGeometry(80, 25, 16, 100);
    const colonMaterial = new THREE.MeshPhongMaterial({
        color: 0xff9664,
        transparent: true,
        opacity: 0.5
    });
    const colon = new THREE.Mesh(colonGeometry, colonMaterial);
    colon.position.set(0, 0, 0);
    colon.rotation.x = Math.PI / 2;
    scene.add(colon);
    organMeshes['colon'] = colon;

    // 간 (박스)
    const liverGeometry = new THREE.BoxGeometry(120, 60, 80);
    const liverMaterial = new THREE.MeshPhongMaterial({
        color: 0x964632,
        transparent: true,
        opacity: 0.5
    });
    const liver = new THREE.Mesh(liverGeometry, liverMaterial);
    liver.position.set(50, 40, 0);
    scene.add(liver);
    organMeshes['liver'] = liver;

    // 종양 1
    const tumor1Geometry = new THREE.SphereGeometry(15, 32, 32);
    const tumorMaterial = new THREE.MeshPhongMaterial({
        color: 0xff3232,
        transparent: true,
        opacity: 0.9,
        emissive: 0xff0000,
        emissiveIntensity: 0.3
    });
    const tumor1 = new THREE.Mesh(tumor1Geometry, tumorMaterial);
    tumor1.position.set(20, -10, 30);
    scene.add(tumor1);
    tumorMeshes.push(tumor1);

    // 종양 2
    const tumor2 = new THREE.Mesh(tumor1Geometry, tumorMaterial.clone());
    tumor2.scale.set(0.7, 0.7, 0.7);
    tumor2.position.set(-30, 5, -20);
    scene.add(tumor2);
    tumorMeshes.push(tumor2);
}

// Scene 클리어
function clearScene() {
    Object.values(organMeshes).forEach(mesh => scene.remove(mesh));
    tumorMeshes.forEach(mesh => scene.remove(mesh));
    organMeshes = {};
    tumorMeshes = [];
}

// 애니메이션 루프
function animate() {
    requestAnimationFrame(animate);

    // 자동 회전
    if (autoRotate) {
        scene.rotation.y += 0.005;
    }

    // 종양 펄스 효과
    const time = Date.now() * 0.001;
    tumorMeshes.forEach((tumor, index) => {
        const scale = 1 + Math.sin(time * 2 + index) * 0.05;
        tumor.scale.set(scale, scale, scale);
    });

    controls.update();
    renderer.render(scene, camera);
}

// 윈도우 리사이즈
function onWindowResize() {
    const canvas = document.getElementById('viewer-canvas');
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

// 레이어 토글
function toggleLayer(layerName) {
    const checkbox = document.getElementById(`layer-${layerName}`);
    checkbox.checked = !checkbox.checked;

    if (layerName === 'tumors') {
        tumorMeshes.forEach(mesh => {
            mesh.visible = checkbox.checked;
        });
    } else {
        const mesh = organMeshes[layerName];
        if (mesh) {
            mesh.visible = checkbox.checked;
        }
    }
}

// 카메라 리셋
function resetCamera() {
    camera.position.set(0, 200, 400);
    camera.lookAt(0, 0, 0);
    controls.reset();
}

// 자동 회전 토글
function toggleRotation() {
    autoRotate = !autoRotate;
}

// 스크린샷
function takeScreenshot() {
    const dataURL = renderer.domElement.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = `3d_viewer_${currentPatientId}_${currentTimepoint}.png`;
    link.href = dataURL;
    link.click();
}

// 전체화면
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
}

// Timeline 슬라이더
document.getElementById('timeline-slider').addEventListener('input', (e) => {
    const timepoints = ['T0', 'T1', 'T2'];
    currentTimepoint = timepoints[e.target.value];
    document.getElementById('info-timepoint').textContent =
        currentTimepoint === 'T0' ? 'T0 (Baseline)' :
            currentTimepoint === 'T1' ? 'T1 (Week 2)' :
                'T2 (Week 4)';

    loadPatientData(currentPatientId, currentTimepoint);
});

// 전체 투명도
document.getElementById('global-opacity').addEventListener('input', (e) => {
    const opacity = e.target.value / 100;
    Object.values(organMeshes).forEach(mesh => {
        if (mesh.material) {
            mesh.material.opacity = opacity * 0.5;
        }
    });
});

// UI 업데이트
function updateUI(data) {
    if (data.patient_id) {
        document.getElementById('patient-id').textContent = data.patient_id;
        document.getElementById('info-patient').textContent = data.patient_id;
    }

    if (data.tumor_stats) {
        document.getElementById('tumor-count').textContent =
            `${data.tumor_stats.count}개`;
        document.getElementById('tumor-volume').textContent =
            `${data.tumor_stats.total_volume.toLocaleString()} mm³`;
        document.getElementById('tumor-diameter').textContent =
            `${data.tumor_stats.max_diameter.toFixed(1)} mm`;

        if (data.tumor_stats.change_percent) {
            const change = data.tumor_stats.change_percent;
            const element = document.getElementById('tumor-change');
            element.textContent = `${change > 0 ? '+' : ''}${change.toFixed(1)}%`;
            element.style.color = change < 0 ? '#28a745' : '#dc3545';
        }
    }

    if (data.scan_date) {
        document.getElementById('info-date').textContent = data.scan_date;
    }
}

// 페이지 로드 시 초기화
window.addEventListener('DOMContentLoaded', init);
