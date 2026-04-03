"""
VTK Volume Rendering - 실제 인체처럼 표현
종양 위치/크기 정확히 시각화
"""

import vtk
import nibabel as nib
import numpy as np

print("="*60)
print("VTK Volume Rendering - 실제 인체 시각화")
print("="*60)

# CT 데이터 로드
print("\n1. CT 데이터 로딩 중...")
organ_path = "F:/ADDS/output/organs_simple/organs_multilabel_hu.nii.gz"
tumor_path = "F:/ADDS/output/tumor_organ_mapping/tumors_unique_3d.nii.gz"

organ_img = nib.load(organ_path)
organ_data = organ_img.get_fdata()

tumor_img = nib.load(tumor_path)
tumor_data = tumor_img.get_fdata()

print(f"   Shape: {organ_data.shape}")
print(f"   Spacing: {organ_img.header.get_zooms()}")

# VTK ImageData 생성
print("\n2. VTK Volume 생성 중...")

# 장기 데이터 → VTK
vtk_organ_data = vtk.vtkImageData()
vtk_organ_data.SetDimensions(organ_data.shape)
vtk_organ_data.SetSpacing(organ_img.header.get_zooms())
vtk_organ_data.AllocateScalars(vtk.VTK_FLOAT, 1)

# NumPy → VTK (flatten)
vtk_array = vtk.vtkFloatArray()
vtk_array.SetNumberOfTuples(organ_data.size)
vtk_array.SetVoidArray(organ_data.flatten('F'), organ_data.size, 1)
vtk_organ_data.GetPointData().SetScalars(vtk_array)

# 종양 데이터 추가 (강조용)
vtk_tumor_data = vtk.vtkImageData()
vtk_tumor_data.SetDimensions(tumor_data.shape)
vtk_tumor_data.SetSpacing(tumor_img.header.get_zooms())
vtk_tumor_data.AllocateScalars(vtk.VTK_FLOAT, 1)

vtk_tumor_array = vtk.vtkFloatArray()
vtk_tumor_array.SetNumberOfTuples(tumor_data.size)
vtk_tumor_array.SetVoidArray(tumor_data.flatten('F'), tumor_data.size, 1)
vtk_tumor_data.GetPointData().SetScalars(vtk_tumor_array)

# Volume Mapper
print("\n3. Volume Rendering 설정 중...")
volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
volumeMapper.SetInputData(vtk_organ_data)

# Color Transfer Function (조직별 색상)
colorFunc = vtk.vtkColorTransferFunction()

# HU값별 실제 인체 색상
colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)        # 공기: 검정
colorFunc.AddRGBPoint(2, 1.0, 0.84, 0.0)      # 지방: 금색
colorFunc.AddRGBPoint(3, 0.53, 0.81, 0.98)    # 폐: 하늘색
colorFunc.AddRGBPoint(4, 0.8, 0.36, 0.36)     # 근육: 빨간 갈색
colorFunc.AddRGBPoint(5, 0.55, 0.27, 0.07)    # 간: 짙은 갈색
colorFunc.AddRGBPoint(6, 1.0, 0.71, 0.76)     # 연조직: 분홍
colorFunc.AddRGBPoint(7, 1.0, 1.0, 1.0)       # 뼈: 흰색

# Opacity Transfer Function (투명도)
opacityFunc = vtk.vtkPiecewiseFunction()
opacityFunc.AddPoint(0, 0.0)    # 공기: 투명
opacityFunc.AddPoint(2, 0.3)    # 지방
opacityFunc.AddPoint(3, 0.2)    # 폐
opacityFunc.AddPoint(4, 0.5)    # 근육
opacityFunc.AddPoint(5, 0.6)    # 간
opacityFunc.AddPoint(6, 0.4)    # 연조직
opacityFunc.AddPoint(7, 0.8)    # 뼈

# Gradient Opacity (경계 강조)
gradientFunc = vtk.vtkPiecewiseFunction()
gradientFunc.AddPoint(0, 0.0)
gradientFunc.AddPoint(90, 0.5)
gradientFunc.AddPoint(100, 1.0)

# Volume Property
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorFunc)
volumeProperty.SetScalarOpacity(opacityFunc)
volumeProperty.SetGradientOpacity(gradientFunc)
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.ShadeOn()
volumeProperty.SetAmbient(0.4)
volumeProperty.SetDiffuse(0.6)
volumeProperty.SetSpecular(0.2)

# Volume Actor
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

# 종양 Volume (빨강 강조)
tumorMapper = vtk.vtkGPUVolumeRayCastMapper()
tumorMapper.SetInputData(vtk_tumor_data)

tumorColorFunc = vtk.vtkColorTransferFunction()
tumorColorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)      # 배경: 투명
tumorColorFunc.AddRGBPoint(0.5, 1.0, 0.0, 0.0)    # 종양: 빨강

tumorOpacityFunc = vtk.vtkPiecewiseFunction()
tumorOpacityFunc.AddPoint(0, 0.0)
tumorOpacityFunc.AddPoint(0.5, 0.9)

tumorProperty = vtk.vtkVolumeProperty()
tumorProperty.SetColor(tumorColorFunc)
tumorProperty.SetScalarOpacity(tumorOpacityFunc)
tumorProperty.ShadeOn()

tumorVolume = vtk.vtkVolume()
tumorVolume.SetMapper(tumorMapper)
tumorVolume.SetProperty(tumorProperty)

# Renderer 설정
print("\n4. 3D 뷰어 창 생성 중...")
renderer = vtk.vtkRenderer()
renderer.AddVolume(volume)
renderer.AddVolume(tumorVolume)
renderer.SetBackground(0.1, 0.1, 0.1)  # 어두운 배경

# 카메라 자동 설정
renderer.ResetCamera()

# Render Window
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindow.SetSize(1200, 900)
renderWindow.SetWindowName("ADDS - Realistic CT Volume Rendering")

# Interactor (마우스 컨트롤)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# 상호작용 스타일
style = vtk.vtkInteractorStyleTrackballCamera()
renderWindowInteractor.SetInteractorStyle(style)

print("\n" + "="*60)
print("3D Volume Rendering 완료!")
print("="*60)
print("\n사용법:")
print("- 좌클릭 + 드래그: 회전")
print("- 우클릭 + 드래그: 줌")
print("- 중간클릭 + 드래그: 이동")
print("- 'q' 키: 종료")
print("\n창이 열립니다...")

# 렌더링 시작
renderWindow.Render()
renderWindowInteractor.Start()
