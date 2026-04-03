"""
nnU-Net 학습을 위한 Masked Dataset 생성
Body mask와 Colon mask를 사용하여 원본 이미지를 마스킹하고
nnU-Net 전처리를 위한 데이터셋 생성
"""
import nibabel as nib
import numpy as np
from pathlib import Path
import json
import shutil


def apply_anatomical_mask(image_data, body_mask_data, colon_mask_data):
    """
    해부학적 마스크를 적용하여 body 외부 영역 제거
    
    Parameters:
    -----------
    image_data : numpy array
        원본 CT 이미지 데이터
    body_mask_data : numpy array
        Body mask (1=body, 0=외부)
    colon_mask_data : numpy array
        Colon mask (1=colon, 0=기타)
        
    Returns:
    --------
    masked_image : numpy array
        Body 영역만 유지한 마스킹된 이미지
    """
    # Body mask 적용: body 외부는 -1024 (air HU value)로 설정
    masked_image = image_data.copy()
    masked_image[body_mask_data == 0] = -1024
    
    return masked_image


def create_masked_dataset(
    raw_dir="f:/ADDS/nnUNet_raw/Dataset010_Colon",
    output_dir="f:/ADDS/nnUNet_raw/Dataset011_ColonMasked"
):
    """
    Body/Colon mask를 적용한 새로운 nnU-Net 데이터셋 생성
    
    Parameters:
    -----------
    raw_dir : str
        원본 Dataset010_Colon 디렉토리 경로
    output_dir : str
        출력 Dataset011_ColonMasked 디렉토리 경로
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    # 출력 디렉토리 구조 생성
    (output_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (output_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (output_path / "imagesTs").mkdir(parents=True, exist_ok=True)
    
    # 원본 이미지 및 레이블 경로
    images_dir = raw_path / "imagesTr"
    labels_dir = raw_path / "labelsTr"
    body_masks_dir = raw_path / "body_masks"
    colon_masks_dir = raw_path / "colon_masks"
    
    # 모든 케이스 처리
    image_files = sorted(images_dir.glob("*.nii.gz"))
    
    print(f"\n{'='*70}")
    print(f"nnU-Net Masked Dataset 생성")
    print(f"{'='*70}")
    print(f"원본 디렉토리: {raw_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"총 케이스 수: {len(image_files)}")
    print(f"{'='*70}\n")
    
    success_count = 0
    failed_cases = []
    
    for idx, image_file in enumerate(image_files, 1):
        # colon_001_0000.nii.gz -> colon_001
        case_id = image_file.stem.replace(".nii", "").replace("_0000", "")
        
        try:
            # 파일 경로
            label_file = labels_dir / f"{case_id}.nii.gz"
            body_mask_file = body_masks_dir / f"{case_id}_body.nii.gz"
            colon_mask_file = colon_masks_dir / f"{case_id}_colon.nii.gz"
            
            # 파일 존재 확인
            if not all([f.exists() for f in [image_file, label_file, body_mask_file, colon_mask_file]]):
                missing = []
                if not image_file.exists(): missing.append("image")
                if not label_file.exists(): missing.append("label")
                if not body_mask_file.exists(): missing.append("body_mask")
                if not colon_mask_file.exists(): missing.append("colon_mask")
                raise FileNotFoundError(f"Missing files: {', '.join(missing)}")
            
            # 데이터 로딩
            image_nii = nib.load(image_file)
            image_data = image_nii.get_fdata()
            
            body_mask_nii = nib.load(body_mask_file)
            body_mask_data = body_mask_nii.get_fdata()
            
            colon_mask_nii = nib.load(colon_mask_file)
            colon_mask_data = colon_mask_nii.get_fdata()
            
            # 마스킹 적용
            masked_image = apply_anatomical_mask(image_data, body_mask_data, colon_mask_data)
            
            # 마스킹된 이미지 저장 (원본과 동일한 affine, header 사용)
            output_image_file = output_path / "imagesTr" / f"{case_id}_0000.nii.gz"
            masked_nii = nib.Nifti1Image(masked_image, image_nii.affine, image_nii.header)
            nib.save(masked_nii, output_image_file)
            
            # 레이블은 그대로 복사 (tumor annotations는 변경 없음)
            output_label_file = output_path / "labelsTr" / f"{case_id}.nii.gz"
            shutil.copy2(label_file, output_label_file)
            
            # 진행 상황 출력
            if idx % 10 == 0 or idx == len(image_files):
                print(f"[{idx}/{len(image_files)}] {case_id} OK")
            
            success_count += 1
            
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] {case_id} ERROR: {str(e)}")
            failed_cases.append((case_id, str(e)))
    
    # dataset.json 생성 (nnU-Net v2 형식)
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "tumor": 1
        },
        "numTraining": success_count,
        "file_ending": ".nii.gz"
    }
    
    # dataset.json 저장
    with open(output_path / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    # 최종 결과 출력
    print(f"\n{'='*70}")
    print(f"처리 완료!")
    print(f"{'='*70}")
    print(f"성공: {success_count}/{len(image_files)} 케이스")
    
    if failed_cases:
        print(f"\n실패한 케이스 ({len(failed_cases)}):")
        for case_id, error in failed_cases:
            print(f"  - {case_id}: {error}")
    
    print(f"\n출력 위치:")
    print(f"  이미지: {output_path / 'imagesTr'}")
    print(f"  레이블: {output_path / 'labelsTr'}")
    print(f"  dataset.json: {output_path / 'dataset.json'}")
    print(f"{'='*70}\n")
    
    return success_count, failed_cases


if __name__ == "__main__":
    success, failures = create_masked_dataset()
    
    if success > 0 and len(failures) == 0:
        print("[OK] MASKED DATASET 생성 완료! nnU-Net 전처리를 진행할 수 있습니다.")
    elif success > 0:
        print(f"[WARNING] 일부 케이스 실패. {success}개 케이스는 성공적으로 생성되었습니다.")
    else:
        print("[ERROR] DATASET 생성 실패!")
