"""
진짜 Volume Rendering - 원본 CT HU 값 사용
"""

import pyvista as pv
import nibabel as nib
import numpy as np

print("="*60)
print("원본 CT Volume Rendering (HU 값)")
print("="*60)

# 원본 CT 로드
ct_path = "F:/ADDS/CTdata/nifti/inha_ct_arterial.nii.gz"
print(f"\nCT 로딩: {ct_path}")

img = nib.load(ct_path)
ct_data = img.get_fdata()
spacing = img.header.get_zooms()

print(f"Shape: {ct_data.shape}")
print(f"HU 범위: {ct_data.min():.0f} ~ {ct_data.max():.0f}")
print(f"Spacing: {spacing}")

# 다운샘플링 (메모리 효율)
print("\n다운샘플링...")
ct_small = ct_data[::2, ::2, ::2].astype(np.float32)
small_spacing = tuple(s*2 for s in spacing)

print(f"다운샘플 Shape: {ct_small.shape}")

# PyVista Grid
grid = pv.ImageData()
grid.dimensions = ct_small.shape
grid.spacing = small_spacing
grid.point_data['HU'] = ct_small.flatten(order='F')

# Plotter
pl = pv.Plotter()
pl.set_background('black')

# 간단한 colormap 사용
volume = pl.add_volume(
    grid,
    scalars='HU',
    cmap='bone',  # 또는 'viridis', 'hot', 'cool'
    opacity='sigmoid',  # 자동 opacity
    shade=True
)

pl.camera_position = 'iso'
pl.add_axes()

print("\n" + "="*60)
print("진짜 Volume Rendering 완료!")
print("="*60)
print("\n조작:")
print("  좌클릭 드래그: 회전")
print("  우클릭 드래그: 줌")
print("  Q: 종료")

print("\n3D 창 열림...")
pl.show()
