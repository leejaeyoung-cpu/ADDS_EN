"""
ADDS CT 3D Reconstruction System
=================================
가이드 기반 구현:
DICOM 로드 -> HU 변환 -> 조직 분할 -> Marching Cubes -> Smoothing -> Render

파이프라인:
1. DICOM 시리즈 로드 (pydicom)
2. HU 변환 (RescaleSlope/Intercept)
3. 조직별 분할 (HU threshold)
4. Marching Cubes (skimage.measure)
5. Laplacian Smoothing
6. PyVista 렌더링
7. STL/OBJ 내보내기
"""

import os
import sys
import numpy as np
import pydicom
from pathlib import Path
from skimage import measure
from scipy import ndimage

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def log(msg):
    print(msg, flush=True)


class CT3DReconstructor:
    """CT DICOM -> 3D Mesh"""

    def __init__(self):
        self.volume = None
        self.spacing = None
        self.meshes = {}

    # =========================================================
    # 1. DICOM 로드
    # =========================================================
    def load_dicom_series(self, dicom_dir):
        """DICOM 시리즈를 3D 볼륨으로 로드"""
        log("[1/7] DICOM 로드...")

        dcm_dir = Path(dicom_dir)
        dcm_files = sorted(dcm_dir.glob("*.dcm"))
        log(f"  파일 수: {len(dcm_files)}")

        # Series 분류
        series_dict = {}
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                sid = ds.SeriesInstanceUID
                if sid not in series_dict:
                    series_dict[sid] = []
                series_dict[sid].append(str(f))
            except Exception:
                pass

        log(f"  Series 수: {len(series_dict)}")

        # 가장 큰 시리즈 = 메인 CT
        main_files = max(series_dict.values(), key=len)
        log(f"  메인 시리즈: {len(main_files)} slices")

        # 슬라이스 로드 + Z 정렬
        slices = []
        for f in main_files:
            ds = pydicom.dcmread(f)
            slices.append(ds)

        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # HU 변환 + 3D 스택
        self.volume = np.stack([
            s.pixel_array.astype(np.float32) * float(s.RescaleSlope) + float(s.RescaleIntercept)
            for s in slices
        ], axis=0)  # axis=0: z축 (슬라이스 방향)

        pixel_sp = [float(x) for x in slices[0].PixelSpacing]
        slice_th = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
        if slice_th == 0:
            slice_th = float(slices[0].SliceThickness)

        self.spacing = (slice_th, pixel_sp[0], pixel_sp[1])

        log(f"  Volume: {self.volume.shape}")
        log(f"  Spacing (z,y,x): {self.spacing} mm")
        log(f"  HU: {self.volume.min():.0f} ~ {self.volume.max():.0f}")
        phys = [self.volume.shape[i] * self.spacing[i] for i in range(3)]
        log(f"  Physical: {phys[0]:.0f} x {phys[1]:.0f} x {phys[2]:.0f} mm")

        return self.volume

    # =========================================================
    # 2. 전처리
    # =========================================================
    def preprocess(self, target_spacing=1.5, sigma=0.8):
        """Isotropic resampling + 노이즈 제거"""
        from scipy.ndimage import zoom as scipy_zoom

        log(f"\n[2/7] 전처리 (isotropic={target_spacing}mm, sigma={sigma})...")

        # 현재 spacing -> 목표 spacing 비율 계산
        zoom_factors = [self.spacing[i] / target_spacing for i in range(3)]
        log(f"  Zoom factors (z,y,x): {[f'{z:.2f}' for z in zoom_factors]}")

        # Isotropic resampling
        self.volume = scipy_zoom(self.volume, zoom_factors, order=1).astype(np.float32)
        self.spacing = (target_spacing, target_spacing, target_spacing)

        # Gaussian smoothing
        self.volume = ndimage.gaussian_filter(self.volume, sigma=sigma)

        log(f"  Volume: {self.volume.shape}")
        log(f"  Spacing: {self.spacing} mm (isotropic)")

    # =========================================================
    # 3. 조직 분할
    # =========================================================
    def segment_tissues(self):
        """HU 기반 조직 분할"""
        log("\n[3/7] 조직 분할...")

        tissues = {
            'bone':        {'range': (300, 3000),  'color': 'white',     'opacity': 0.9},
            'muscle':      {'range': (40, 100),    'color': '#CD5C5C',   'opacity': 0.4},
            'soft_tissue': {'range': (-50, 40),    'color': '#FFB6C1',   'opacity': 0.25},
            'fat':         {'range': (-100, -50),  'color': '#FFD700',   'opacity': 0.15},
            'lung':        {'range': (-900, -400), 'color': '#87CEEB',   'opacity': 0.1},
            'body':        {'range': (-400, 3000), 'color': 'peachpuff', 'opacity': 0.15},
        }

        for name, info in tissues.items():
            lo, hi = info['range']
            mask = (self.volume >= lo) & (self.volume <= hi)
            voxels = mask.sum()
            vol_mm3 = voxels * np.prod(self.spacing)
            log(f"  {name:15s}: {voxels:>10,} voxels  ({vol_mm3/1000:.1f} cm3)")
            info['mask'] = mask

        self.tissues = tissues
        return tissues

    # =========================================================
    # 4. Marching Cubes
    # =========================================================
    def extract_surfaces(self, targets=None):
        """Marching Cubes로 각 조직 surface 추출"""
        log("\n[4/7] Marching Cubes 표면 추출...")

        if targets is None:
            targets = ['bone', 'body']

        for name in targets:
            if name not in self.tissues:
                continue

            info = self.tissues[name]
            mask = info['mask'].astype(np.float32)

            if mask.sum() < 100:
                log(f"  {name}: 복셀 부족, 건너뜀")
                continue

            # 형태학 정리
            mask = ndimage.binary_closing(mask, iterations=2).astype(np.float32)

            try:
                verts, faces, normals, values = measure.marching_cubes(
                    mask,
                    level=0.5,
                    spacing=self.spacing,
                    allow_degenerate=False
                )

                self.meshes[name] = {
                    'verts': verts,
                    'faces': faces,
                    'normals': normals,
                    'color': info['color'],
                    'opacity': info['opacity'],
                }
                log(f"  {name:15s}: {len(verts):>8,} verts, {len(faces):>8,} faces")

            except Exception as e:
                log(f"  {name}: 실패 - {e}")

        return self.meshes

    # =========================================================
    # 5. Smoothing
    # =========================================================
    def smooth_meshes(self, iterations=50):
        """PyVista built-in smoothing (C++ 구현, 빠름)"""
        log(f"\n[5/7] Smoothing (iter={iterations})...")

        import pyvista as pv

        for name, mesh in self.meshes.items():
            verts = mesh['verts']
            faces = mesh['faces']

            faces_pv = np.column_stack([np.full(len(faces), 3), faces]).astype(np.int32)
            poly = pv.PolyData(verts, faces=faces_pv)

            # PyVista smooth (내부 C++ VTK)
            poly = poly.smooth(n_iter=iterations, relaxation_factor=0.1)

            # Decimate (50% 감소)
            poly = poly.decimate(0.5)

            mesh['poly'] = poly  # 바로 사용
            log(f"  {name}: {poly.n_points:,} pts, {poly.n_cells:,} faces")

    # =========================================================
    # 6. Render
    # =========================================================
    def render(self):
        """PyVista 인터랙티브 3D 렌더링"""
        import pyvista as pv

        log("\n[6/7] PyVista 3D Rendering...")

        pl = pv.Plotter()
        pl.set_background('black')

        for name, mesh in self.meshes.items():
            poly = mesh.get('poly')
            if poly is None:
                continue

            pl.add_mesh(
                poly,
                color=mesh['color'],
                opacity=mesh['opacity'],
                smooth_shading=True,
                name=name
            )
            log(f"  {name}: rendered ({poly.n_points:,} pts)")

        pl.camera_position = 'iso'
        pl.add_axes()
        pl.add_text(
            "ADDS CT 3D Model",
            position='upper_left',
            font_size=12,
            color='white'
        )

        log("\n  3D window opening...")
        log("  Controls: Left-drag=Rotate, Right-drag=Zoom, Q=Quit")
        pl.show()

    # =========================================================
    # 7. Export
    # =========================================================
    def export(self, output_dir="F:/ADDS/output/3d_models"):
        """STL/OBJ 내보내기"""
        log(f"\n[7/7] 파일 내보내기 -> {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        for name, mesh in self.meshes.items():
            poly = mesh.get('poly')
            if poly is None:
                log(f"  {name}: no smoothed mesh, skip export")
                continue

            # PyVista에서 STL/OBJ 직접 저장
            stl_path = os.path.join(output_dir, f"{name}.stl")
            poly.save(stl_path)

            obj_path = os.path.join(output_dir, f"{name}.obj")
            poly.save(obj_path)

            stl_size = os.path.getsize(stl_path) / 1024
            obj_size = os.path.getsize(obj_path) / 1024
            log(f"  {name}: STL={stl_size:.0f}KB, OBJ={obj_size:.0f}KB")


# =============================================================
# MAIN
# =============================================================
if __name__ == '__main__':
    log("=" * 60)
    log("ADDS CT 3D Reconstruction System")
    log("=" * 60)

    recon = CT3DReconstructor()

    # 1. DICOM 로드
    recon.load_dicom_series("F:/ADDS/CTdata/CTdcm")

    # 2. 전처리 (isotropic resampling)
    recon.preprocess(target_spacing=1.5, sigma=0.8)

    # 3. 조직 분할
    recon.segment_tissues()

    # 4. Surface 추출 (모든 조직)
    recon.extract_surfaces(targets=['bone', 'muscle', 'soft_tissue', 'fat', 'lung', 'body'])

    # 5. Smoothing (강화)
    recon.smooth_meshes(iterations=50)

    # 6. Export
    recon.export()

    # 7. Render
    recon.render()

    log("\nDone.")
