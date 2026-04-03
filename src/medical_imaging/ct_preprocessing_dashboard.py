"""
CT Preprocessing Dashboard Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
각 전처리 단계별 before/after를 시각화하는 자립형(standalone) HTML 대시보드를
생성합니다. 모든 이미지는 base64로 인라인 임베드됩니다.

출력:
  {output_root}/dashboard_{series_id}.html

Author : ADDS Research Team
Date   : 2026-03-17
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Image Utilities
# ═══════════════════════════════════════════════════════════════════

def _arr_to_png_b64(arr: np.ndarray, cmap: str = 'gray', vmin=None, vmax=None) -> str:
    """Convert a 2-D numpy array to a base64-encoded PNG string."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _get_center_axial(vol: np.ndarray) -> np.ndarray:
    """Return the center axial slice of a 3D volume."""
    z = vol.shape[0] // 2
    return vol[z]


def _get_3views(vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Axial, coronal, sagittal center slices."""
    D, H, W = vol.shape
    axial   = vol[D // 2, :, :]
    coronal = vol[:, H // 2, :]
    sagittal= vol[:, :, W // 2]
    return axial, coronal, sagittal


# ═══════════════════════════════════════════════════════════════════
# Stage Comparison Generator
# ═══════════════════════════════════════════════════════════════════

class DashboardGenerator:
    """
    Loads preprocessed outputs from a single series directory
    and produces a rich HTML comparison dashboard.
    """

    STAGE_LABELS = [
        ('raw',             'Raw DICOM (HU)'),
        ('body_mask',       'Body Isolation'),
        ('preprocessed',    'N4 + NLM Denoised'),
        ('clahe',           'CLAHE Enhanced'),
        ('isotropic',       'Isotropic 1mm³'),
    ]

    WINDOW_NAMES = ['Soft Tissue', 'Liver', 'Colon', 'Lung', 'Bone']

    def __init__(self, series_dir: str):
        self.series_dir = Path(series_dir)
        self.series_id = self.series_dir.name
        self._volumes: Dict[str, np.ndarray] = {}
        self._qc: dict = {}
        self._annotation: dict = {}
        self._multi_window: Optional[np.ndarray] = None

    # ---------------------------------------------------------------
    def load(self) -> 'DashboardGenerator':
        """Load all available volumes from the series directory."""
        import SimpleITK as sitk

        def _load_nii(name: str) -> Optional[np.ndarray]:
            p = self.series_dir / f'{name}.nii.gz'
            if p.exists():
                arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
                logger.debug(f"  Loaded {name}: {arr.shape}")
                return arr
            return None

        for key, _ in self.STAGE_LABELS:
            vol = _load_nii(key)
            if vol is not None:
                self._volumes[key] = vol

        mw_path = self.series_dir / 'multi_window_5ch.npy'
        if mw_path.exists():
            self._multi_window = np.load(str(mw_path))

        qc_path = self.series_dir / 'qc_report.json'
        if qc_path.exists():
            with open(qc_path) as f:
                self._qc = json.load(f)

        ann_path = self.series_dir / 'annotation.json'
        if ann_path.exists():
            with open(ann_path) as f:
                self._annotation = json.load(f)

        return self

    # ---------------------------------------------------------------
    def _make_stage_cards(self) -> str:
        """Generate HTML cards for each stage (axial / coronal / sagittal)."""
        cards_html = ''
        for key, label in self.STAGE_LABELS:
            if key not in self._volumes:
                continue
            vol = self._volumes[key]

            # Use body mask as overlay
            is_mask = (key == 'body_mask')
            vmin = 0 if is_mask else None
            vmax = 1 if is_mask else None

            axial, coronal, sagittal = _get_3views(vol)
            imgs = [
                ('Axial',    axial,    'gray', vmin, vmax),
                ('Coronal',  coronal,  'gray', vmin, vmax),
                ('Sagittal', sagittal, 'gray', vmin, vmax),
            ]

            views_html = ''
            for view_name, arr, cmap, vn, vx in imgs:
                b64 = _arr_to_png_b64(arr, cmap=cmap, vmin=vn, vmax=vx)
                views_html += f'''
                <div class="view-item">
                  <div class="view-label">{view_name}</div>
                  <img src="data:image/png;base64,{b64}" />
                </div>'''

            cards_html += f'''
            <div class="stage-card">
              <h3>{label}</h3>
              <div class="view-row">{views_html}</div>
            </div>'''

        return cards_html

    # ---------------------------------------------------------------
    def _make_multiwindow_cards(self) -> str:
        """Generate 5-channel window comparison cards."""
        if self._multi_window is None:
            return '<p style="color:#aaa">Multi-window data not available.</p>'

        cards = ''
        D = self._multi_window.shape[1]
        z = D // 2

        for i, win_name in enumerate(self.WINDOW_NAMES):
            if i >= self._multi_window.shape[0]:
                break
            slc = self._multi_window[i, z, :, :]
            b64 = _arr_to_png_b64(slc, cmap='gray', vmin=0, vmax=1)
            cards += f'''
            <div class="win-card">
              <div class="win-label">Ch{i+1}: {win_name}</div>
              <img src="data:image/png;base64,{b64}" />
            </div>'''

        return f'<div class="win-row">{cards}</div>'

    # ---------------------------------------------------------------
    def _make_qc_panel(self) -> str:
        """Generate QC metrics bars."""
        if not self._qc:
            return '<p style="color:#aaa">QC data not available.</p>'

        metrics = [
            ('SNR',          self._qc.get('snr', 0),          10.0,  '#4fc3f7'),
            ('CNR',          self._qc.get('cnr', 0),          5.0,   '#aed581'),
            ('Motion Score', 1.0 - self._qc.get('motion_score', 0), 1.0, '#ff8a65'),
            ('Body Coverage',self._qc.get('body_coverage_ratio', 0), 1.0,'#ce93d8'),
        ]

        bars = ''
        for name, val, max_val, color in metrics:
            pct = min(100, max(0, (val / max_val) * 100))
            bars += f'''
            <div class="qc-row">
              <div class="qc-label">{name}</div>
              <div class="qc-bar-bg">
                <div class="qc-bar" style="width:{pct:.1f}%;background:{color}"></div>
              </div>
              <div class="qc-val">{val:.3f}</div>
            </div>'''

        verdict = self._qc.get('qc_verdict', 'UNKNOWN')
        verdict_color = {'PASS': '#66bb6a', 'WARN': '#ffa726', 'FAIL': '#ef5350'}.get(verdict, '#90a4ae')

        timing = self._qc.get('stage_timings', {}).get('total', 0)

        return f'''
        <div class="qc-verdict" style="border-left:4px solid {verdict_color}">
          QC Verdict: <strong style="color:{verdict_color}">{verdict}</strong>
          &nbsp;&nbsp;|&nbsp;&nbsp; Pipeline: <strong>{timing:.1f}s</strong>
        </div>
        {bars}'''

    # ---------------------------------------------------------------
    def _make_annotation_panel(self) -> str:
        """Generate annotation summary panel."""
        if not self._annotation:
            return '<p style="color:#aaa">No annotation data.</p>'

        rows = ''
        key_labels = {
            'diaphragm_z_mm': 'Diaphragm Z (mm)',
            'pelvis_z_mm': 'Pelvic Floor Z (mm)',
            'abdomen_slice_coverage': 'Abd. Coverage',
        }
        for k, label in key_labels.items():
            val = self._annotation.get(k, 'N/A')
            if isinstance(val, float) and k == 'abdomen_slice_coverage':
                val = f'{val:.1%}'
            rows += f'<tr><td class="ann-key">{label}</td><td class="ann-val">{val}</td></tr>'

        # HU percentiles
        hu = self._annotation.get('hu_percentiles', {})
        if hu:
            rows += f'''<tr><td class="ann-key">HU P5/P50/P95</td>
              <td class="ann-val">{hu.get("p5","?")} / {hu.get("p50","?")} / {hu.get("p95","?")} HU</td></tr>'''

        # Organ presence
        organs = self._annotation.get('organs', {})
        if organs:
            present = [o for o, d in organs.items() if d.get('present')]
            rows += f'<tr><td class="ann-key">Organs detected</td><td class="ann-val">{", ".join(present) or "none"}</td></tr>'

        return f'<table class="ann-table">{rows}</table>'

    # ---------------------------------------------------------------
    def generate_html(self) -> str:
        """Assemble the complete HTML dashboard."""
        stage_cards = self._make_stage_cards()
        win_cards   = self._make_multiwindow_cards()
        qc_panel    = self._make_qc_panel()
        ann_panel   = self._make_annotation_panel()

        snr = self._qc.get('snr', 0)
        verdict = self._qc.get('qc_verdict', '—')
        shape = self._qc.get('isotropic_shape', self._qc.get('volume_shape', '—'))

        return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADDS CT 전처리 대시보드 — {self.series_id}</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --accent: #58a6ff; --text: #e6edf3; --sub: #8b949e;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; padding: 24px; }}
  h1 {{ color: var(--accent); font-size: 1.5rem; margin-bottom: 4px; }}
  h2 {{ color: var(--sub); font-size: 1.1rem; font-weight: 500; margin: 28px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  h3 {{ color: #cdd9e5; font-size: .95rem; margin-bottom: 8px; }}
  .subtitle {{ color: var(--sub); font-size: .85rem; margin-bottom: 24px; }}
  .stats-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 8px; }}
  .stat-chip {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 18px; }}
  .stat-chip .label {{ font-size: .7rem; color: var(--sub); text-transform: uppercase; letter-spacing: .05em; }}
  .stat-chip .value {{ font-size: 1.1rem; font-weight: 600; color: var(--accent); }}

  .stage-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 14px; }}
  .view-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .view-item {{ display: flex; flex-direction: column; align-items: center; gap: 4px; }}
  .view-label {{ font-size: .7rem; color: var(--sub); text-transform: uppercase; }}
  .view-item img {{ width: 200px; height: 200px; object-fit: contain; border-radius: 4px; border: 1px solid var(--border); background: #000; }}

  .win-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .win-card {{ display: flex; flex-direction: column; align-items: center; gap: 4px; }}
  .win-label {{ font-size: .72rem; color: var(--sub); }}
  .win-card img {{ width: 160px; height: 160px; object-fit: contain; border-radius: 4px; border: 1px solid var(--border); background: #000; }}

  .qc-verdict {{ background: var(--surface); border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; font-size: .9rem; }}
  .qc-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
  .qc-label {{ width: 120px; font-size: .8rem; color: var(--sub); }}
  .qc-bar-bg {{ flex: 1; height: 10px; background: #21262d; border-radius: 5px; overflow: hidden; }}
  .qc-bar {{ height: 100%; border-radius: 5px; transition: width .4s ease; }}
  .qc-val {{ width: 60px; font-size: .8rem; text-align: right; color: var(--text); }}

  .ann-table {{ border-collapse: collapse; width: 100%; }}
  .ann-table tr {{ border-bottom: 1px solid var(--border); }}
  .ann-key {{ padding: 6px 12px 6px 0; font-size: .82rem; color: var(--sub); width: 180px; }}
  .ann-val {{ padding: 6px 0; font-size: .82rem; color: var(--text); font-weight: 500; }}

  footer {{ margin-top: 36px; font-size: .72rem; color: var(--sub); text-align: center; }}
</style>
</head>
<body>

<h1>🧠 ADDS CT 전처리 대시보드</h1>
<p class="subtitle">Series: <strong>{self.series_id}</strong></p>

<div class="stats-row">
  <div class="stat-chip"><div class="label">QC Verdict</div><div class="value">{verdict}</div></div>
  <div class="stat-chip"><div class="label">SNR</div><div class="value">{snr:.2f}</div></div>
  <div class="stat-chip"><div class="label">Isotropic Shape</div><div class="value">{shape}</div></div>
  <div class="stat-chip"><div class="label">Channels</div><div class="value">5-ch</div></div>
</div>

<h2>📊 10단계 파이프라인 — Stage 비교</h2>
{stage_cards}

<h2>🪟 5-채널 Multi-Window Fusion (중심 슬라이스)</h2>
{win_cards}

<h2>✅ QC 메트릭</h2>
{qc_panel}

<h2>📍 복부 ROI 어노테이션 (Stage 10)</h2>
{ann_panel}

<footer>Generated by ADDS CT Preprocessing Dashboard v2.0 &nbsp;|&nbsp; 2026-03-17</footer>
</body>
</html>"""

    # ---------------------------------------------------------------
    def save(self, output_path: Optional[str] = None) -> str:
        """Generate and save the dashboard HTML."""
        html = self.generate_html()
        if output_path is None:
            output_path = str(self.series_dir.parent.parent / f'dashboard_{self.series_id}.html')

        Path(output_path).write_text(html, encoding='utf-8')
        logger.info(f"  Dashboard saved → {output_path}")
        return output_path


# ═══════════════════════════════════════════════════════════════════
# Batch dashboard generation
# ═══════════════════════════════════════════════════════════════════

def generate_all_dashboards(
    preprocessed_root: str,
    overwrite: bool = False,
) -> List[str]:
    """
    Scan preprocessed_root for qc_report.json, generate one dashboard per series.
    """
    root = Path(preprocessed_root)
    qc_files = sorted(root.rglob('qc_report.json'))
    paths = []

    for qc_f in qc_files:
        series_dir = qc_f.parent
        out_html = root / f'dashboard_{series_dir.name}.html'

        if out_html.exists() and not overwrite:
            paths.append(str(out_html))
            continue

        try:
            gen = DashboardGenerator(str(series_dir))
            gen.load()
            saved = gen.save(str(out_html))
            paths.append(saved)
            print(f"  ✓ {series_dir.name} → {out_html.name}")
        except Exception as e:
            logger.error(f"  ✗ {series_dir.name}: {e}")

    return paths


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    generate_all_dashboards(r'f:\ADDS\preprocessed', overwrite=True)
