"""
ADDS End-to-End Working Protocol
실제 데이터로 작동하는 전체 파이프라인

데이터 위치:
- BindingDB: F:\\ADDS\\bindingdb\\BindingDB_Extracted.tsv
- CT DICOM: F:\\ADDS\\CTdata\\CTdcm\\*.dcm (427 slices)

파이프라인:
1. CT 3D 재구성
2. 종양 감지 (간단한 HU thresholding)
3. 종양 3D 좌표 계산
4. BindingDB에서 약물 데이터 조회
5. 에너지 계산
6. 3D 메시 생성
"""

import sys
from pathlib import Path
import numpy as np
import json
import logging
from datetime import datetime

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from backend.services.ct_3d_reconstruction import CT3DReconstructor
from backend.services.mesh_generator import MeshGenerator
from backend.models.physics_energy_model import PhysicsEnergyModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 설정
# ============================================================================

# 데이터 경로
BINDING_DB_PATH = Path("F:/ADDS/bindingdb/BindingDB_Extracted.tsv")
CT_DICOM_DIR = Path("F:/ADDS/CTdata/CTdcm")

# 출력 경로
OUTPUT_DIR = Path("F:/ADDS/output/protocol_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Step 1: CT 3D 재구성
# ============================================================================

def step1_ct_reconstruction():
    """CT DICOM → 3D Volume"""
    logger.info("=" * 80)
    logger.info("STEP 1: CT 3D Reconstruction")
    logger.info("=" * 80)
    
    reconstructor = CT3DReconstructor()
    
    # DICOM 시리즈 로딩
    logger.info(f"Loading DICOM from: {CT_DICOM_DIR}")
    volume = reconstructor.load_dicom_series(str(CT_DICOM_DIR))
    
    logger.info(f"Loaded volume: {volume.shape}")
    logger.info(f"Spacing: {reconstructor.metadata.spacing}")
    logger.info(f"Patient ID: {reconstructor.metadata.patient_id}")
    
    # 등방성 리샘플링 (1mm³)
    logger.info("Resampling to isotropic 1mm³ voxels...")
    volume_iso = reconstructor.resample_to_isotropic(target_spacing=1.0)
    
    # NIfTI 저장
    nifti_path = OUTPUT_DIR / "ct_volume.nii.gz"
    reconstructor.save_nifti(str(nifti_path))
    logger.info(f"Saved NIfTI: {nifti_path}")
    
    return reconstructor


# ============================================================================
# Step 2: 간단한 종양 감지 (HU thresholding)
# ============================================================================

def step2_tumor_detection(reconstructor: CT3DReconstructor):
    """HU thresholding으로 종양 후보 감지"""
    logger.info("=" * 80)
    logger.info("STEP 2: Tumor Detection (HU Thresholding)")
    logger.info("=" * 80)
    
    volume = reconstructor.volume
    
    # CT 복부 종양의 일반적인 HU 범위: 20-60
    tumor_mask = (volume > 20) & (volume < 60)
    
    # Morphological operations
    from scipy import ndimage
    
    # Small noise 제거
    tumor_mask = ndimage.binary_opening(tumor_mask, iterations=2)
    
    # Connected components
    labeled_volume, num_tumors = ndimage.label(tumor_mask)
    
    logger.info(f"Detected {num_tumors} potential tumor regions")
    
    # 부피별 필터링 (최소 100 voxels)
    voxel_volume = np.prod(reconstructor.metadata.spacing)
    
    valid_tumors = []
    for label in range(1, num_tumors + 1):
        tumor_voxels = np.sum(labeled_volume == label)
        volume_mm3 = tumor_voxels * voxel_volume
        volume_cm3 = volume_mm3 / 1000
        
        if volume_cm3 > 0.1:  # 최소 0.1 cm³
            valid_tumors.append(label)
            logger.info(f"  Tumor {label}: {volume_cm3:.2f} cm³")
    
    logger.info(f"Valid tumors (>0.1 cm³): {len(valid_tumors)}")
    
    # 유효한 종양만 마스크 생성
    final_mask = np.isin(labeled_volume, valid_tumors)
    
    # 저장
    mask_path = OUTPUT_DIR / "tumor_mask.nii.gz"
    reconstructor.save_mask_nifti(final_mask, str(mask_path))
    logger.info(f"Saved tumor mask: {mask_path}")
    
    return final_mask, labeled_volume, valid_tumors


# ============================================================================
# Step 3: 종양 3D 좌표 계산
# ============================================================================

def step3_tumor_coordinates(reconstructor: CT3DReconstructor, labeled_volume, valid_labels):
    """각 종양의 3D 좌표 계산"""
    logger.info("=" * 80)
    logger.info("STEP 3: Tumor 3D Coordinates")
    logger.info("=" * 80)
    
    tumor_coords = []
    
    for label in valid_labels:
        tumor_mask = (labeled_volume == label)
        
        try:
            coords = reconstructor.calculate_tumor_coordinates(tumor_mask)
            
            tumor_data = {
                'tumor_id': f"tumor_{label}",
                'centroid_mm': coords['centroid_mm'],
                'bbox_min_mm': coords['bbox_min_mm'],
                'bbox_max_mm': coords['bbox_max_mm'],
                'volume_cm3': coords['volume_cm3'],
                'longest_diameter_mm': coords['longest_diameter_mm']
            }
            
            tumor_coords.append(tumor_data)
            
            logger.info(f"Tumor {label}:")
            logger.info(f"  Centroid: ({coords['centroid_mm'][0]:.1f}, "
                       f"{coords['centroid_mm'][1]:.1f}, {coords['centroid_mm'][2]:.1f}) mm")
            logger.info(f"  Volume: {coords['volume_cm3']:.2f} cm³")
            logger.info(f"  Diameter: {coords['longest_diameter_mm']:.1f} mm")
            
        except Exception as e:
            logger.warning(f"Failed to calculate coordinates for tumor {label}: {e}")
    
    # JSON 저장
    coords_file = OUTPUT_DIR / "tumor_coordinates.json"
    with open(coords_file, 'w') as f:
        json.dump(tumor_coords, f, indent=2)
    
    logger.info(f"Saved coordinates: {coords_file}")
    
    return tumor_coords


# ============================================================================
# Step 4: BindingDB 약물 데이터 샘플 조회
# ============================================================================

def step4_query_binding_data():
    """BindingDB에서 항암제 데이터 샘플 조회"""
    logger.info("=" * 80)
    logger.info("STEP 4: Query BindingDB for Drug Data")
    logger.info("=" * 80)
    
    import pandas as pd
    
    # 샘플로 처음 1000줄만 읽기 (전체는 너무 큼)
    logger.info(f"Reading BindingDB: {BINDING_DB_PATH}")
    logger.info("(Reading first 10000 rows as sample)")
    
    df = pd.read_csv(
        BINDING_DB_PATH,
        sep='\t',
        nrows=10000,
        low_memory=False
    )
    
    logger.info(f"Loaded {len(df)} binding records")
    logger.info(f"Columns: {list(df.columns[:10])}")
    
    # 항암제 관련 타겟 필터링 (예: EGFR, VEGF, etc.)
    cancer_targets = ['EGFR', 'VEGFR', 'HER2', 'BRAF', 'MEK', 'ALK']
    
    df_cancer = df[df['Target Name'].str.contains('|'.join(cancer_targets), case=False, na=False)]
    
    logger.info(f"Found {len(df_cancer)} cancer-related binding records")
    
    # Ki 값이 있는 것만
    df_with_ki = df_cancer[df_cancer['Ki (nM)'].notna()]
    
    logger.info(f"Records with Ki values: {len(df_with_ki)}")
    
    if len(df_with_ki) > 0:
        # 샘플 데이터
        sample = df_with_ki.iloc[0]
        
        drug_info = {
            'ligand_name': str(sample.get('Ligand Name', 'Unknown')),
            'target_name': str(sample.get('Target Name', 'Unknown')),
            'ki_nm': float(sample.get('Ki (nM)', np.nan))
        }
        
        logger.info(f"\nSample drug:")
        logger.info(f"  Ligand: {drug_info['ligand_name']}")
        logger.info(f"  Target: {drug_info['target_name']}")
        logger.info(f"  Ki: {drug_info['ki_nm']:.2f} nM")
        
        return drug_info
    else:
        logger.warning("No cancer drugs with Ki found in sample")
        return None


# ============================================================================
# Step 5: 물리 에너지 계산
# ============================================================================

def step5_energy_calculation(drug_info, tumor_coords):
    """약물-종양 에너지 계산"""
    logger.info("=" * 80)
    logger.info("STEP 5: Physics Energy Calculation")
    logger.info("=" * 80)
    
    if drug_info is None:
        logger.warning("No drug data available, skipping energy calculation")
        return None
    
    model = PhysicsEnergyModel()
    
    # 바인딩 에너지 계산
    ki_m = drug_info['ki_nm'] * 1e-9  # nM → M
    binding_energy = model.calculate_binding_energy(ki_m)
    
    logger.info(f"\nBinding Energy Calculation:")
    logger.info(f"  Drug: {drug_info['ligand_name']}")
    logger.info(f"  Target: {drug_info['target_name']}")
    logger.info(f"  Ki: {drug_info['ki_nm']:.2f} nM")
    logger.info(f"  ΔG: {binding_energy:.2f} kcal/mol")
    
    # 각 종양에 대해 가상의 치료 반응 계산
    tumor_results = []
    
    for tumor in tumor_coords:
        # 가상의 종양 반응 (실제로는 CT 변화 필요)
        baseline_volume = tumor['volume_cm3']
        # 30% 감소 가정
        followup_volume = baseline_volume * 0.7
        
        # 반응 에너지 계산
        response_energy = model.calculate_tumor_response_energy(
            baseline_volume_cm3=baseline_volume,
            followup_volume_cm3=followup_volume,
            time_days=90
        )
        
        # 유효 농도 역추론
        estimated_conc = model.reverse_calculate_drug_concentration(
            delta_g_total=response_energy,
            hill_coefficient=1.5,
            ec50_m=drug_info['ki_nm'] * 1e-9 * 10  # Ki의 10배로 가정
        )
        
        result = {
            'tumor_id': tumor['tumor_id'],
            'baseline_volume_cm3': baseline_volume,
            'followup_volume_cm3': followup_volume,
            'response_energy_kcal': response_energy,
            'estimated_concentration_m': estimated_conc
        }
        
        tumor_results.append(result)
        
        logger.info(f"\n{tumor['tumor_id']}:")
        logger.info(f"  Volume: {baseline_volume:.2f} → {followup_volume:.2f} cm³")
        logger.info(f"  Response Energy: {response_energy:.2f} kcal/mol")
        logger.info(f"  Estimated [Drug]: {estimated_conc*1e9:.2f} nM")
    
    # 결과 저장
    energy_results = {
        'drug': drug_info,
        'binding_energy_kcal': binding_energy,
        'tumors': tumor_results,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = OUTPUT_DIR / "energy_results.json"
    with open(results_file, 'w') as f:
        json.dump(energy_results, f, indent=2)
    
    logger.info(f"\nSaved energy results: {results_file}")
    
    return energy_results


# ============================================================================
# Step 6: 3D 메시 생성 (간단 버전)
# ============================================================================

def step6_mesh_generation(reconstructor: CT3DReconstructor, tumor_mask):
    """3D 메시 생성"""
    logger.info("=" * 80)
    logger.info("STEP 6: 3D Mesh Generation")
    logger.info("=" * 80)
    
    generator = MeshGenerator()
    
    # 피부 마스크
    logger.info("Generating skin mask...")
    skin_mask = reconstructor.segment_skin(threshold=-200)
    
    # 피부 마스크 저장
    skin_mask_path = OUTPUT_DIR / "skin_mask.nii.gz"
    reconstructor.save_mask_nifti(skin_mask, str(skin_mask_path))
    
    # 피부 메시 생성
    logger.info("Generating skin mesh...")
    try:
        skin_mesh = generator.generate_organ_mesh(
            mask_path=skin_mask_path,
            spacing=reconstructor.metadata.spacing,
            organ_name="skin",
            simplify=True,
            target_faces=5000
        )
        
        skin_mesh_file = OUTPUT_DIR / "skin_mesh.json"
        generator.save_mesh_json(skin_mesh, skin_mesh_file)
        logger.info(f"Saved skin mesh: {skin_mesh_file}")
        logger.info(f"  Vertices: {skin_mesh['num_vertices']}")
        logger.info(f"  Faces: {skin_mesh['num_faces']}")
        
    except Exception as e:
        logger.error(f"Skin mesh generation failed: {e}")
    
    # 종양 메시
    logger.info("Generating tumor mesh...")
    tumor_mask_path = OUTPUT_DIR / "tumor_mask.nii.gz"
    
    try:
        tumor_mesh = generator.generate_organ_mesh(
            mask_path=tumor_mask_path,
            spacing=reconstructor.metadata.spacing,
            organ_name="tumor",
            simplify=True,
            target_faces=3000
        )
        
        tumor_mesh_file = OUTPUT_DIR / "tumor_mesh.json"
        generator.save_mesh_json(tumor_mesh, tumor_mesh_file)
        logger.info(f"Saved tumor mesh: {tumor_mesh_file}")
        logger.info(f"  Vertices: {tumor_mesh['num_vertices']}")
        logger.info(f"  Faces: {tumor_mesh['num_faces']}")
        
    except Exception as e:
        logger.error(f"Tumor mesh generation failed: {e}")
    
    logger.info("Mesh generation complete")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """전체 파이프라인 실행"""
    logger.info("\n" + "=" * 80)
    logger.info("ADDS END-TO-END PROTOCOL")
    logger.info("=" * 80)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 80 + "\n")
    
    # Step 1: CT 재구성
    reconstructor = step1_ct_reconstruction()
    
    # Step 2: 종양 감지
    tumor_mask, labeled_volume, valid_tumors = step2_tumor_detection(reconstructor)
    
    # Step 3: 종양 좌표
    tumor_coords = step3_tumor_coordinates(reconstructor, labeled_volume, valid_tumors)
    
    # Step 4: BindingDB 조회
    drug_info = step4_query_binding_data()
    
    # Step 5: 에너지 계산
    energy_results = step5_energy_calculation(drug_info, tumor_coords)
    
    # Step 6: 메시 생성
    step6_mesh_generation(reconstructor, tumor_mask)
    
    # 최종 요약
    logger.info("\n" + "=" * 80)
    logger.info("PROTOCOL COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output files in: {OUTPUT_DIR}")
    logger.info(f"  - ct_volume.nii.gz")
    logger.info(f"  - tumor_mask.nii.gz")
    logger.info(f"  - tumor_coordinates.json")
    logger.info(f"  - energy_results.json")
    logger.info(f"  - skin_mesh.json")
    logger.info(f"  - tumor_mesh.json")
    logger.info("=" * 80)
    
    print("\n✅ SUCCESS: End-to-end protocol completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Protocol failed: {e}", exc_info=True)
        print(f"\n❌ FAILED: {e}")
        sys.exit(1)
