import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Line, Html } from '@react-three/drei';
import * as THREE from 'three';
import {
    Box,
    Typography,
    Paper,
    Grid,
    Button,
    Chip,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Card,
    CardContent,
    Slider
} from '@mui/material';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';

/**
 * Before/After Tumor Comparison Viewer
 * 치료 전후 종양 변화 비교 시각화
 */

// RECIST 반응에 따른 컬러
const RECIST_COLORS = {
    CR: '#00ff00',  // Complete Response
    PR: '#90ee90',  // Partial Response
    SD: '#ffff00',  // Stable Disease
    PD: '#ff0000'   // Progressive Disease
};

const RECIST_LABELS = {
    CR: '완전 관해',
    PR: '부분 관해',
    SD: '안정',
    PD: '진행'
};

// ============================================================================
// 화살표 컴포넌트 (종양 이동 표시)
// ============================================================================

const TumorMovementArrow = ({ start, end, color, thickness = 2 }) => {
    const points = [
        new THREE.Vector3(...start),
        new THREE.Vector3(...end)
    ];

    return (
        <>
            <Line
                points={points}
                color={color}
                lineWidth={thickness}
            />

            {/* 끝점에 구 표시 */}
            <mesh position={end}>
                <sphereGeometry args={[3, 16, 16]} />
                <meshStandardMaterial color={color} />
            </mesh>
        </>
    );
};

// ============================================================================
// 메인 비교 뷰어
// ============================================================================

const BeforeAfterComparison = ({ patientId, baselineScanId, followupScanId }) => {
    // State
    const [comparisonData, setComparisonData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [viewMode, setViewMode] = useState('overlay');  // 'overlay', 'sidebyside', 'diff'
    const [showArrows, setShowArrows] = useState(true);
    const [animationProgress, setAnimationProgress] = useState(0);

    // 데이터 로딩
    useEffect(() => {
        const loadComparisonData = async () => {
            try {
                setLoading(true);

                // API에서 비교 데이터 가져오기
                const response = await fetch(
                    `/api/ct/comparison?baseline=${baselineScanId}&followup=${followupScanId}`
                );

                if (!response.ok) {
                    throw new Error('Failed to load comparison data');
                }

                const data = await response.json();
                setComparisonData(data);
                setLoading(false);

            } catch (err) {
                console.error('Error loading comparison:', err);
                setLoading(false);
            }
        };

        if (baselineScanId && followupScanId) {
            loadComparisonData();
        }
    }, [baselineScanId, followupScanId]);

    if (loading || !comparisonData) {
        return <Typography>Loading comparison...</Typography>;
    }

    const { evaluation, matches, arrows, baseline_meshes, followup_meshes } = comparisonData;

    return (
        <Box>
            {/* 상단 요약 */}
            <Paper sx={{ p: 3, mb: 3 }}>
                <Grid container spacing={3}>
                    {/* 전체 평가 */}
                    <Grid item xs={12} md={4}>
                        <Typography variant="h6" gutterBottom>
                            전체 평가
                        </Typography>
                        <Chip
                            label={RECIST_LABELS[evaluation.overall_response]}
                            sx={{
                                bgcolor: RECIST_COLORS[evaluation.overall_response],
                                color: '#000',
                                fontWeight: 'bold',
                                fontSize: 18,
                                height: 40,
                                px: 2
                            }}
                        />
                    </Grid>

                    {/* 부피 변화 */}
                    <Grid item xs={12} md={4}>
                        <Typography variant="h6" gutterBottom>
                            총 종양 부피
                        </Typography>
                        <Box display="flex" alignItems="center" gap={2}>
                            <Typography variant="body1">
                                {evaluation.total_tumor_volume_baseline.toFixed(1)} cm³
                            </Typography>
                            <ArrowForwardIcon />
                            <Typography variant="body1">
                                {evaluation.total_tumor_volume_followup.toFixed(1)} cm³
                            </Typography>
                            <Chip
                                label={`${evaluation.volume_change_percent > 0 ? '+' : ''}${evaluation.volume_change_percent.toFixed(1)}%`}
                                color={evaluation.volume_change_percent < 0 ? 'success' : 'error'}
                                size="small"
                            />
                        </Box>
                    </Grid>

                    {/* 반응 분포 */}
                    <Grid item xs={12} md={4}>
                        <Typography variant="h6" gutterBottom>
                            반응 분포
                        </Typography>
                        <Box display="flex" gap={1}>
                            {evaluation.num_cr > 0 && (
                                <Chip label={`CR: ${evaluation.num_cr}`} size="small" sx={{ bgcolor: RECIST_COLORS.CR, color: '#000' }} />
                            )}
                            {evaluation.num_pr > 0 && (
                                <Chip label={`PR: ${evaluation.num_pr}`} size="small" sx={{ bgcolor: RECIST_COLORS.PR, color: '#000' }} />
                            )}
                            {evaluation.num_sd > 0 && (
                                <Chip label={`SD: ${evaluation.num_sd}`} size="small" sx={{ bgcolor: RECIST_COLORS.SD, color: '#000' }} />
                            )}
                            {evaluation.num_pd > 0 && (
                                <Chip label={`PD: ${evaluation.num_pd}`} size="small" sx={{ bgcolor: RECIST_COLORS.PD, color: '#000' }} />
                            )}
                        </Box>
                    </Grid>
                </Grid>
            </Paper>

            {/* 3D 시각화 */}
            <Grid container spacing={3}>
                <Grid item xs={12} lg={8}>
                    <Paper sx={{ p: 2 }}>
                        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                            <Typography variant="h6">
                                3D 비교 뷰
                            </Typography>

                            <Box display="flex" gap={1}>
                                <Button
                                    variant={viewMode === 'overlay' ? 'contained' : 'outlined'}
                                    size="small"
                                    onClick={() => setViewMode('overlay')}
                                >
                                    오버레이
                                </Button>
                                <Button
                                    variant={viewMode === 'sidebyside' ? 'contained' : 'outlined'}
                                    size="small"
                                    onClick={() => setViewMode('sidebyside')}
                                >
                                    나란히
                                </Button>
                                <Button
                                    variant={showArrows ? 'contained' : 'outlined'}
                                    size="small"
                                    onClick={() => setShowArrows(!showArrows)}
                                >
                                    이동 화살표
                                </Button>
                            </Box>
                        </Box>

                        {/* 3D Canvas */}
                        <Box sx={{ height: 600, bgcolor: '#000' }}>
                            <Canvas>
                                <ambientLight intensity={0.4} />
                                <directionalLight position={[10, 10, 5]} intensity={0.8} />

                                {viewMode === 'overlay' && (
                                    <>
                                        {/* Baseline tumors (반투명) */}
                                        {baseline_meshes && baseline_meshes.map((mesh, idx) => (
                                            <mesh key={`baseline-${idx}`} geometry={mesh.geometry}>
                                                <meshStandardMaterial color="#ff0000" transparent opacity={0.3} />
                                            </mesh>
                                        ))}

                                        {/* Follow-up tumors (불투명) */}
                                        {followup_meshes && followup_meshes.map((mesh, idx) => (
                                            <mesh key={`followup-${idx}`} geometry={mesh.geometry}>
                                                <meshStandardMaterial color="#00ff00" transparent opacity={0.7} />
                                            </mesh>
                                        ))}

                                        {/* 이동 화살표 */}
                                        {showArrows && arrows && arrows.map((arrow, idx) => (
                                            <TumorMovementArrow
                                                key={`arrow-${idx}`}
                                                start={arrow.start}
                                                end={arrow.end}
                                                color={arrow.color}
                                                thickness={arrow.thickness}
                                            />
                                        ))}
                                    </>
                                )}

                                {viewMode === 'sidebyside' && (
                                    <>
                                        {/* Baseline (왼쪽) */}
                                        <group position={[-150, 0, 0]}>
                                            {baseline_meshes && baseline_meshes.map((mesh, idx) => (
                                                <mesh key={`baseline-${idx}`} geometry={mesh.geometry}>
                                                    <meshStandardMaterial color="#ff6666" />
                                                </mesh>
                                            ))}
                                            <Html position={[0, 100, 0]}>
                                                <Typography
                                                    variant="h6"
                                                    sx={{ color: '#fff', bgcolor: 'rgba(0,0,0,0.7)', p: 1, borderRadius: 1 }}
                                                >
                                                    Before
                                                </Typography>
                                            </Html>
                                        </group>

                                        {/* Follow-up (오른쪽) */}
                                        <group position={[150, 0, 0]}>
                                            {followup_meshes && followup_meshes.map((mesh, idx) => (
                                                <mesh key={`followup-${idx}`} geometry={mesh.geometry}>
                                                    <meshStandardMaterial color="#66ff66" />
                                                </mesh>
                                            ))}
                                            <Html position={[0, 100, 0]}>
                                                <Typography
                                                    variant="h6"
                                                    sx={{ color: '#fff', bgcolor: 'rgba(0,0,0,0.7)', p: 1, borderRadius: 1 }}
                                                >
                                                    After
                                                </Typography>
                                            </Html>
                                        </group>
                                    </>
                                )}

                                <OrbitControls />
                            </Canvas>
                        </Box>

                        {/* 애니메이션 슬라이더 */}
                        {viewMode === 'overlay' && (
                            <Box mt={2}>
                                <Typography variant="caption" gutterBottom>
                                    시간 경과 애니메이션
                                </Typography>
                                <Slider
                                    value={animationProgress}
                                    onChange={(e, v) => setAnimationProgress(v)}
                                    min={0}
                                    max={100}
                                    marks={[
                                        { value: 0, label: 'Before' },
                                        { value: 100, label: 'After' }
                                    ]}
                                />
                            </Box>
                        )}
                    </Paper>
                </Grid>

                {/* 종양별 상세 변화 */}
                <Grid item xs={12} lg={4}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            종양별 변화
                        </Typography>

                        <TableContainer sx={{ maxHeight: 560, overflow: 'auto' }}>
                            <Table size="small" stickyHeader>
                                <TableHead>
                                    <TableRow>
                                        <TableCell>종양</TableCell>
                                        <TableCell align="right">부피 변화</TableCell>
                                        <TableCell align="center">평가</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {matches && matches.map((match, idx) => (
                                        <TableRow key={idx} hover>
                                            <TableCell>#{match.baseline_tumor_id}</TableCell>
                                            <TableCell align="right">
                                                <Typography
                                                    variant="body2"
                                                    color={match.volume_change_percent < 0 ? 'success.main' : 'error.main'}
                                                    fontWeight="bold"
                                                >
                                                    {match.volume_change_percent > 0 ? '+' : ''}
                                                    {match.volume_change_percent.toFixed(1)}%
                                                </Typography>
                                                <Typography variant="caption" display="block">
                                                    {match.baseline_volume.toFixed(1)} → {match.followup_volume.toFixed(1)} cm³
                                                </Typography>
                                            </TableCell>
                                            <TableCell align="center">
                                                <Chip
                                                    label={RECIST_LABELS[match.recist_response]}
                                                    size="small"
                                                    sx={{
                                                        bgcolor: RECIST_COLORS[match.recist_response],
                                                        color: '#000',
                                                        fontSize: 11
                                                    }}
                                                />
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </Paper>
                </Grid>
            </Grid>

            {/* 리포트 생성 버튼 */}
            <Box mt={3} display="flex" justifyContent="center">
                <Button
                    variant="contained"
                    size="large"
                    onClick={() => {
                        // TODO: 리포트 생성 API 호출
                        alert('RECIST 리포트 생성 중...');
                    }}
                >
                    RECIST 리포트 생성
                </Button>
            </Box>
        </Box>
    );
};

export default BeforeAfterComparison;
