import React, { useState, useEffect, useRef } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, Environment, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import { Box, Typography, Paper, Checkbox, FormControlLabel, Slider, Button, CircularProgress } from '@mui/material';

/**
 * 3D CT Tumor Viewer
 * Three.js 기반 대화형 3D 종양 시각화 컴포넌트
 */

// ============================================================================
// 메시 컴포넌트
// ============================================================================

const MeshComponent = ({ meshData, visible, opacity, color, onClick, selected }) => {
    const meshRef = useRef();

    // Geometry 생성
    const geometry = React.useMemo(() => {
        const geo = new THREE.BufferGeometry();

        // Vertices
        const vertices = new Float32Array(meshData.vertices.flat());
        geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

        // Faces (indices)
        const indices = new Uint32Array(meshData.faces.flat());
        geo.setIndex(new THREE.BufferAttribute(indices, 1));

        // Normals
        if (meshData.normals && meshData.normals.length > 0) {
            const normals = new Float32Array(meshData.normals.flat());
            geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
        } else {
            geo.computeVertexNormals();
        }

        return geo;
    }, [meshData]);

    // 선택 시 하이라이트 효과
    useFrame(() => {
        if (selected && meshRef.current) {
            meshRef.current.rotation.y += 0.01;
        }
    });

    if (!visible) return null;

    return (
        <mesh
            ref={meshRef}
            geometry={geometry}
            onClick={onClick}
            onPointerOver={() => document.body.style.cursor = 'pointer'}
            onPointerOut={() => document.body.style.cursor = 'default'}
        >
            <meshStandardMaterial
                color={selected ? '#ffff00' : color}
                transparent
                opacity={opacity}
                side={THREE.DoubleSide}
                metalness={0.3}
                roughness={0.7}
            />
        </mesh>
    );
};


// ============================================================================
// 메인 3D 뷰어
// ============================================================================

const Tumor3DViewer = ({ patientId, scanId }) => {
    // State
    const [meshData, setMeshData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const [selectedTumor, setSelectedTumor] = useState(null);

    const [layerVisibility, setLayerVisibility] = useState({
        skin: true,
        organs: true,
        tumors: true
    });

    const [layerOpacity, setLayerOpacity] = useState({
        skin: 0.3,
        organs: 0.6,
        tumors: 1.0
    });

    // 데이터 로딩
    useEffect(() => {
        const loadMeshData = async () => {
            try {
                setLoading(true);

                // API에서 메시 데이터 가져오기
                const response = await fetch(`/api/ct/3d/mesh/${scanId}`);

                if (!response.ok) {
                    throw new Error('Failed to load 3D mesh data');
                }

                const data = await response.json();

                // 각 메시 JSON 파일 로드
                const loadedData = {
                    skin: null,
                    organs: [],
                    tumors: []
                };

                // Skin mesh
                if (data.skin_mesh) {
                    const skinResponse = await fetch(data.skin_mesh.mesh_url);
                    if (skinResponse.ok) {
                        loadedData.skin = {
                            ...await skinResponse.json(),
                            color: data.skin_mesh.color
                        };
                    }
                }

                // Organ meshes
                for (const organInfo of data.organ_meshes) {
                    const organResponse = await fetch(organInfo.mesh_url);
                    if (organResponse.ok) {
                        loadedData.organs.push({
                            name: organInfo.name,
                            data: await organResponse.json(),
                            color: organInfo.color
                        });
                    }
                }

                // Tumor meshes
                for (const tumorInfo of data.tumor_meshes) {
                    const tumorResponse = await fetch(tumorInfo.mesh_url);
                    if (tumorResponse.ok) {
                        loadedData.tumors.push({
                            id: tumorInfo.tumor_id,
                            data: await tumorResponse.json(),
                            volume: tumorInfo.volume_ml,
                            diameter: tumorInfo.diameter_mm
                        });
                    }
                }

                setMeshData(loadedData);
                setLoading(false);

            } catch (err) {
                console.error('Error loading mesh data:', err);
                setError(err.message);
                setLoading(false);
            }
        };

        if (scanId) {
            loadMeshData();
        }
    }, [scanId]);

    // 레이어 토글
    const toggleLayer = (layer) => {
        setLayerVisibility(prev => ({
            ...prev,
            [layer]: !prev[layer]
        }));
    };

    // 투명도 변경
    const handleOpacityChange = (layer, value) => {
        setLayerOpacity(prev => ({
            ...prev,
            [layer]: value
        }));
    };

    // 종양 선택
    const handleTumorClick = (tumor) => {
        setSelectedTumor(selectedTumor?.id === tumor.id ? null : tumor);
    };

    // 로딩 중
    if (loading) {
        return (
            <Box display="flex" justifyContent="center" alignItems="center" height={600}>
                <CircularProgress />
                <Typography ml={2}>Loading 3D visualization...</Typography>
            </Box>
        );
    }

    // 에러
    if (error) {
        return (
            <Box p={3}>
                <Typography color="error">Error: {error}</Typography>
            </Box>
        );
    }

    // 데이터 없음
    if (!meshData) {
        return (
            <Box p={3}>
                <Typography>No 3D data available</Typography>
            </Box>
        );
    }

    return (
        <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
            {/* 3D Canvas */}
            <Box sx={{ width: '100%', height: 600, bgcolor: '#000' }}>
                <Canvas>
                    <PerspectiveCamera makeDefault position={[0, 0, 500]} fov={50} />

                    <ambientLight intensity={0.4} />
                    <directionalLight position={[10, 10, 5]} intensity={0.8} />
                    <directionalLight position={[-10, -10, -5]} intensity={0.3} />

                    {/* Skin mesh */}
                    {meshData.skin && (
                        <MeshComponent
                            meshData={meshData.skin}
                            visible={layerVisibility.skin}
                            opacity={layerOpacity.skin}
                            color={`rgb(${meshData.skin.color.join(',')})`}
                            onClick={() => { }}
                            selected={false}
                        />
                    )}

                    {/* Organ meshes */}
                    {layerVisibility.organs && meshData.organs.map((organ, idx) => (
                        <MeshComponent
                            key={`organ-${idx}`}
                            meshData={organ.data}
                            visible={true}
                            opacity={layerOpacity.organs}
                            color={`rgb(${organ.color.join(',')})`}
                            onClick={() => { }}
                            selected={false}
                        />
                    ))}

                    {/* Tumor meshes */}
                    {layerVisibility.tumors && meshData.tumors.map((tumor, idx) => (
                        <MeshComponent
                            key={`tumor-${idx}`}
                            meshData={tumor.data}
                            visible={true}
                            opacity={layerOpacity.tumors}
                            color="#ff0000"
                            onClick={() => handleTumorClick(tumor)}
                            selected={selectedTumor?.id === tumor.id}
                        />
                    ))}

                    <OrbitControls
                        enablePan={true}
                        enableZoom={true}
                        enableRotate={true}
                        minDistance={100}
                        maxDistance={1000}
                    />

                    <Environment preset="studio" />
                </Canvas>
            </Box>

            {/* 컨트롤 패널 */}
            <Paper
                sx={{
                    position: 'absolute',
                    top: 20,
                    left: 20,
                    p: 2,
                    width: 280,
                    bgcolor: 'rgba(255, 255, 255, 0.95)'
                }}
                elevation={3}
            >
                <Typography variant="h6" gutterBottom>
                    레이어 컨트롤
                </Typography>

                {/* Skin */}
                <FormControlLabel
                    control={
                        <Checkbox
                            checked={layerVisibility.skin}
                            onChange={() => toggleLayer('skin')}
                        />
                    }
                    label="피부"
                />
                {layerVisibility.skin && (
                    <Box px={2}>
                        <Typography variant="caption">투명도</Typography>
                        <Slider
                            value={layerOpacity.skin}
                            onChange={(e, v) => handleOpacityChange('skin', v)}
                            min={0}
                            max={1}
                            step={0.1}
                            size="small"
                        />
                    </Box>
                )}

                {/* Organs */}
                <FormControlLabel
                    control={
                        <Checkbox
                            checked={layerVisibility.organs}
                            onChange={() => toggleLayer('organs')}
                        />
                    }
                    label="내장 기관"
                />
                {layerVisibility.organs && (
                    <Box px={2}>
                        <Typography variant="caption">투명도</Typography>
                        <Slider
                            value={layerOpacity.organs}
                            onChange={(e, v) => handleOpacityChange('organs', v)}
                            min={0}
                            max={1}
                            step={0.1}
                            size="small"
                        />
                    </Box>
                )}

                {/* Tumors */}
                <FormControlLabel
                    control={
                        <Checkbox
                            checked={layerVisibility.tumors}
                            onChange={() => toggleLayer('tumors')}
                        />
                    }
                    label="종양"
                />
                {layerVisibility.tumors && (
                    <Box px={2}>
                        <Typography variant="caption">투명도</Typography>
                        <Slider
                            value={layerOpacity.tumors}
                            onChange={(e, v) => handleOpacityChange('tumors', v)}
                            min={0}
                            max={1}
                            step={0.1}
                            size="small"
                        />
                    </Box>
                )}

                <Box mt={2}>
                    <Button
                        variant="outlined"
                        size="small"
                        fullWidth
                        onClick={() => setSelectedTumor(null)}
                    >
                        선택 초기화
                    </Button>
                </Box>
            </Paper>

            {/* 종양 정보 패널 */}
            {selectedTumor && (
                <Paper
                    sx={{
                        position: 'absolute',
                        bottom: 20,
                        right: 20,
                        p: 2,
                        width: 300,
                        bgcolor: 'rgba(255, 255, 255, 0.95)'
                    }}
                    elevation={3}
                >
                    <Typography variant="h6" gutterBottom>
                        종양 #{selectedTumor.id}
                    </Typography>

                    <Box mt={1}>
                        <Typography variant="body2">
                            <strong>부피:</strong> {selectedTumor.volume?.toFixed(2)} ml
                        </Typography>
                        <Typography variant="body2">
                            <strong>직경:</strong> {selectedTumor.diameter?.toFixed(1)} mm
                        </Typography>

                        {selectedTumor.data.centroid && (
                            <Typography variant="body2">
                                <strong>좌표:</strong><br />
                                X: {selectedTumor.data.centroid[0]?.toFixed(1)} mm<br />
                                Y: {selectedTumor.data.centroid[1]?.toFixed(1)} mm<br />
                                Z: {selectedTumor.data.centroid[2]?.toFixed(1)} mm
                            </Typography>
                        )}
                    </Box>

                    <Box mt={2}>
                        <Button
                            variant="contained"
                            size="small"
                            fullWidth
                            onClick={() => {
                                // TODO: 상세 분석 페이지로 이동
                                alert(`종양 #${selectedTumor.id} 상세 분석`);
                            }}
                        >
                            상세 분석
                        </Button>
                    </Box>
                </Paper>
            )}

            {/* 종양 목록 */}
            <Paper
                sx={{
                    position: 'absolute',
                    top: 20,
                    right: 20,
                    p: 2,
                    width: 200,
                    maxHeight: 400,
                    overflow: 'auto',
                    bgcolor: 'rgba(255, 255, 255, 0.95)'
                }}
                elevation={3}
            >
                <Typography variant="h6" gutterBottom>
                    종양 목록
                </Typography>

                {meshData.tumors.map((tumor, idx) => (
                    <Box
                        key={idx}
                        p={1}
                        mb={1}
                        sx={{
                            border: selectedTumor?.id === tumor.id ? 2 : 1,
                            borderColor: selectedTumor?.id === tumor.id ? 'primary.main' : 'grey.300',
                            borderRadius: 1,
                            cursor: 'pointer',
                            bgcolor: selectedTumor?.id === tumor.id ? 'primary.light' : 'white'
                        }}
                        onClick={() => handleTumorClick(tumor)}
                    >
                        <Typography variant="body2" fontWeight="bold">
                            종양 #{tumor.id}
                        </Typography>
                        <Typography variant="caption" display="block">
                            {tumor.volume?.toFixed(2)} ml
                        </Typography>
                        <Typography variant="caption" display="block">
                            {tumor.diameter?.toFixed(1)} mm
                        </Typography>
                    </Box>
                ))}
            </Paper>
        </Box>
    );
};

export default Tumor3DViewer;
