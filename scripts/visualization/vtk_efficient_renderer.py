"""
메모리 효율적 VTK Volume Rendering
다운샘플링 + 실제 인체 색상
"""

import vtk
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

print("="*60)
print("메모리 효율적 VTK Volume Rendering")
print("="*60)

# CT 데이터 로드 (메모리 맵)
print("\n1. CT 데이터 로딩 중...")
organ_path = "F:/ADDS/output/organs_simple/organs_multilabel_hu.nii.gz"
organ_img = nib.load(organ_path)

print(f"   원본 Shape: {organ_img.shape}")

# 다운샘플링 (메모리 절약)
print("\n2. 다운샘플링 중 (50% 크기)...")
downsample_factor = 0.5

# 일부만 로드해서 다운샘플
organ_data_subset = organ_img.dataobj[::2, ::2, ::2]  # 2배 다운샘플
organ_data_small = np.array(organ_data_subset, dtype=np.uint8)

print(f"   다운샘플 Shape: {organ_data_small.shape}")
print(f"   메모리: {organ_data_small.nbytes / 1024 / 1024:.1f} MB")

# VTK ImageData
print("\n3. VTK Volume 생성 중...")
vtk_data = vtk.vtkImageData()
vtk_data.SetDimensions(organ_data_small.shape)
vtk_data.SetSpacing(2.0, 2.0, 2.0)  # 2배 spacing

# NumPy → VTK
flat_data = organ_data_small.flatten(order='F')
vtk_array = vtk.vtkUnsignedCharArray()
vtk_array.SetNumberOfTuples(flat_data.size)
for i in range(flat_data.size):
    vtk_array.SetValue(i, int(flat_data[i]))
    
vtk_data.GetPointData().SetScalars(vtk_array)

# Volume Mapper
print("\n4. Volume Rendering 설정...")
volumeMapper = vtk.vtkSmartVolumeMapper()
volumeMapper.SetInputData(vtk_data)

# 실제 인체 색상
colorFunc = vtk.vtkColorTransferFunction()
# label 값별 색상 (segmentation labels)
colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)        # 배경
colorFunc.AddRGBPoint(2, 1.0, 0.84, 0.0)       # 지방: 금색
colorFunc.AddRGBPoint(3, 0.68, 0.85, 0.90)     # 폐: 밝은 하늘색
colorFunc.AddRGBPoint(4, 0.80, 0.40, 0.40)     # 근육: 붉은 갈색
colorFunc.AddRGBPoint(5, 0.55, 0.27, 0.07)     # 간: 짙은 갈색
colorFunc.AddRGBPoint(6, 1.0, 0.75, 0.80)      # 연조직: 살색
colorFunc.AddRGBPoint(7, 0.95, 0.95, 0.95)     # 뼈: 밝은 회색

# 투명도
opacityFunc = vtk.vtkPiecewiseFunction()
opacityFunc.AddPoint(0, 0.0)     # 배경: 투명
opacityFunc.AddPoint(2, 0.4)     # 지방
opacityFunc.AddPoint(3, 0.3)     # 폐
opacityFunc.AddPoint(4, 0.6)     # 근육
opacityFunc.AddPoint(5, 0.7)     # 간
opacityFunc.AddPoint(6, 0.5)     # 연조직  
opacityFunc.AddPoint(7, 0.9)     # 뼈

# Volume Property
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorFunc)
volumeProperty.SetScalarOpacity(opacityFunc)
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.ShadeOn()
volumeProperty.SetAmbient(0.3)
volumeProperty.SetDiffuse(0.7)
volumeProperty.SetSpecular(0.3)

# Volume
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

# Renderer
print("\n5. 3D 창 생성 중...")
renderer = vtk.vtkRenderer()
renderer.AddVolume(volume)
renderer.SetBackground(0.15, 0.15, 0.15)
renderer.ResetCamera()

# Window
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindow.SetSize(1200, 800)
renderWindow.SetWindowName("ADDS - Realistic Human Body Rendering")

# Interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renderWindow)

# 컨트롤
style = vtk.vtkInteractorStyleTrackballCamera()
interactor.SetInteractorStyle(style)

print("\n" + "="*60)
print("3D Volume Rendering 완료!")
print("="*60)
print("\n조직별 색상:")
print("  - 지방: 금색")
print("  - 폐: 하늘색")
print("  - 근육: 붉은 갈색")
print("  - 간: 짙은 갈색")
print("  - 연조직: 살색")
print("  - 뼈: 흰색")
print("\n조작:")
print("  - 좌클릭 드래그: 회전")
print("  - 우클릭 드래그: 줌")
print("  - Q 키: 종료")
print("\n창이 열립니다...")

renderWindow.Render()
interactor.Start()
