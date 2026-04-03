import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

/**
 * Multi-Layer Organ 3D Viewer
 * 장기별 독립 레이어 제어 및 시각화
 */
const MultiLayerOrgan3DViewer = () => {
    const containerRef = useRef(null);
    const sceneRef = useRef(null);
    const rendererRef = useRef(null);
    const cameraRef = useRef(null);
    const controlsRef = useRef(null);
    const meshesRef = useRef({});

    const [layers, setLayers] = useState({
        fat: { visible: true, opacity: 0.3, color: '#FFD700', name: '지방' },
        lung_tissue: { visible: true, opacity: 0.2, color: '#87CEEB', name: '폐 조직' },
        muscle: { visible: true, opacity: 0.4, color: '#CD5C5C', name: '근육' },
        liver: { visible: true, opacity: 0.5, color: '#8B4513', name: '간' },
        soft_tissue: { visible: true, opacity: 0.5, color: '#FFB6C1', name: '연조직' },
        bone: { visible: true, opacity: 0.8, color: '#FFFFFF', name: '뼈' },
    });

    const [loading, setLoading] = useState(true);
    const [selectedOrgan, setSelectedOrgan] = useState(null);

    // Three.js 초기화
    useEffect(() => {
        if (!containerRef.current) return;

        // Scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);
        sceneRef.current = scene;

        // Camera
        const camera = new THREE.PerspectiveCamera(
            75,
            containerRef.current.clientWidth / containerRef.current.clientHeight,
            0.1,
            10000
        );
        camera.position.set(300, 300, 300);
        cameraRef.current = camera;

        // Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
        containerRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controlsRef.current = controls;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight1.position.set(1, 1, 1);
        scene.add(directionalLight1);

        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight2.position.set(-1, -1, -1);
        scene.add(directionalLight2);

        // Animation loop
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        // Cleanup
        return () => {
            if (containerRef.current && renderer.domElement) {
                containerRef.current.removeChild(renderer.domElement);
            }
            renderer.dispose();
        };
    }, []);

    // 메시 로딩
    useEffect(() => {
        if (!sceneRef.current) return;

        const loadMeshes = async () => {
            try {
                setLoading(true);

                // 각 장기 메시 로딩
                for (const [organName, layerConfig] of Object.entries(layers)) {
                    try {
                        const response = await fetch(`/api/meshes/${organName}_mesh.json`);
                        const meshData = await response.json();

                        // Geometry 생성
                        const geometry = new THREE.BufferGeometry();

                        const vertices = new Float32Array(meshData.vertices.flat());
                        const indices = new Uint32Array(meshData.faces.flat());

                        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
                        geometry.computeVertexNormals();

                        // Material
                        const material = new THREE.MeshPhongMaterial({
                            color: new THREE.Color(layerConfig.color),
                            opacity: layerConfig.opacity,
                            transparent: true,
                            side: THREE.DoubleSide,
                            depthWrite: true,
                        });

                        // Mesh
                        const mesh = new THREE.Mesh(geometry, material);
                        mesh.name = organName;
                        mesh.visible = layerConfig.visible;

                        sceneRef.current.add(mesh);
                        meshesRef.current[organName] = mesh;

                    } catch (error) {
                        console.warn(`Failed to load ${organName}:`, error);
                    }
                }

                setLoading(false);

            } catch (error) {
                console.error('Failed to load meshes:', error);
                setLoading(false);
            }
        };

        loadMeshes();
    }, []);

    // 레이어 가시성 토글
    const toggleLayer = (organName) => {
        setLayers((prev) => ({
            ...prev,
            [organName]: { ...prev[organName], visible: !prev[organName].visible },
        }));

        if (meshesRef.current[organName]) {
            meshesRef.current[organName].visible = !meshesRef.current[organName].visible;
        }
    };

    // 투명도 변경
    const updateOpacity = (organName, opacity) => {
        setLayers((prev) => ({
            ...prev,
            [organName]: { ...prev[organName], opacity },
        }));

        if (meshesRef.current[organName]) {
            meshesRef.current[organName].material.opacity = opacity;
        }
    };

    // 모든 레이어 토글
    const toggleAllLayers = () => {
        const allVisible = Object.values(layers).every((layer) => layer.visible);
        const newVisibility = !allVisible;

        const updatedLayers = {};
        for (const [name, config] of Object.entries(layers)) {
            updatedLayers[name] = { ...config, visible: newVisibility };
            if (meshesRef.current[name]) {
                meshesRef.current[name].visible = newVisibility;
            }
        }
        setLayers(updatedLayers);
    };

    // 장기 선택 (하이라이트)
    const selectOrgan = (organName) => {
        // 이전 선택 해제
        if (selectedOrgan && meshesRef.current[selectedOrgan]) {
            meshesRef.current[selectedOrgan].material.emissive = new THREE.Color(0x000000);
        }

        // 새 선택
        if (meshesRef.current[organName]) {
            meshesRef.current[organName].material.emissive = new THREE.Color(0x444444);
            setSelectedOrgan(organName);
        }
    };

    return (
        <div style={{ display: 'flex', width: '100%', height: '100vh' }}>
            {/* 3D Viewer */}
            <div ref={containerRef} style={{ flex: 1, position: 'relative' }}>
                {loading && (
                    <div
                        style={{
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            color: 'white',
                            fontSize: '20px',
                            background: 'rgba(0,0,0,0.7)',
                            padding: '20px',
                            borderRadius: '8px',
                        }}
                    >
                        Loading 3D Models...
                    </div>
                )}
            </div>

            {/* Control Panel */}
            <div
                style={{
                    width: '350px',
                    background: '#1a1a1a',
                    color: 'white',
                    padding: '20px',
                    overflowY: 'auto',
                }}
            >
                <h2 style={{ marginTop: 0 }}>장기 레이어 제어</h2>

                <button
                    onClick={toggleAllLayers}
                    style={{
                        width: '100%',
                        padding: '10px',
                        marginBottom: '20px',
                        background: '#444',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                    }}
                >
                    {Object.values(layers).every((l) => l.visible) ? '모두 숨기기' : '모두 표시'}
                </button>

                {Object.entries(layers).map(([organName, config]) => (
                    <div
                        key={organName}
                        style={{
                            marginBottom: '20px',
                            padding: '15px',
                            background: selectedOrgan === organName ? '#333' : '#2a2a2a',
                            borderRadius: '8px',
                            border: selectedOrgan === organName ? '2px solid #fff' : '2px solid transparent',
                            cursor: 'pointer',
                        }}
                        onClick={() => selectOrgan(organName)}
                    >
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                            {/* Color Indicator */}
                            <div
                                style={{
                                    width: '20px',
                                    height: '20px',
                                    background: config.color,
                                    marginRight: '10px',
                                    borderRadius: '4px',
                                }}
                            />

                            {/* Name */}
                            <span style={{ flex: 1, fontWeight: 'bold' }}>{config.name}</span>

                            {/* Toggle */}
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    toggleLayer(organName);
                                }}
                                style={{
                                    padding: '5px 10px',
                                    background: config.visible ? '#4CAF50' : '#f44336',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '4px',
                                    cursor: 'pointer',
                                }}
                            >
                                {config.visible ? '표시' : '숨김'}
                            </button>
                        </div>

                        {/* Opacity Slider */}
                        <div>
                            <label style={{ fontSize: '12px', color: '#aaa' }}>
                                투명도: {(config.opacity * 100).toFixed(0)}%
                            </label>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.05"
                                value={config.opacity}
                                onChange={(e) => updateOpacity(organName, parseFloat(e.target.value))}
                                onClick={(e) => e.stopPropagation()}
                                style={{ width: '100%', marginTop: '5px' }}
                            />
                        </div>
                    </div>
                ))}

                <div style={{ marginTop: '30px', padding: '15px', background: '#2a2a2a', borderRadius: '8px' }}>
                    <h3 style={{ marginTop: 0, fontSize: '14px' }}>컨트롤</h3>
                    <ul style={{ fontSize: '12px', color: '#aaa', paddingLeft: '20px' }}>
                        <li>마우스 왼쪽: 회전</li>
                        <li>마우스 휠: 확대/축소</li>
                        <li>마우스 오른쪽: 이동</li>
                    </ul>
                </div>
            </div>
        </div>
    );
};

export default MultiLayerOrgan3DViewer;
