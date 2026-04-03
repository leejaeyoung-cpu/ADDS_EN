"""
run_ct_advanced_preprocessing.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADDS CT 고급 전처리 통합 실행 진입점 (CLI)

사용법:
  # 전체 파이프라인 (CTdata1 + CTdata2)
  python run_ct_advanced_preprocessing.py

  # 드라이런 (파일 탐색만, 실제 처리 없음)
  python run_ct_advanced_preprocessing.py --dry-run

  # 특정 폴더 + N4 스킵 (속도 우선)
  python run_ct_advanced_preprocessing.py --sources CTdata2 --skip-stages 3

  # 1개 시리즈만 테스트
  python run_ct_advanced_preprocessing.py --limit 1 --dashboard

  # 대시보드만 재생성
  python run_ct_advanced_preprocessing.py --dashboard-only

Author : ADDS Research Team
Date   : 2026-03-17
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── 경로 설정 ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description='ADDS CT Advanced Preprocessing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--sources', nargs='+',
        default=[str(ROOT / 'CTdata1'), str(ROOT / 'CTdata2')],
        help='Source directories to scan (default: CTdata1 CTdata2)',
    )
    parser.add_argument(
        '--output',
        default=str(ROOT / 'preprocessed'),
        help='Output root directory (default: ./preprocessed)',
    )
    parser.add_argument(
        '--skip-stages', nargs='*', type=int, default=[],
        help='Stage numbers to skip (e.g. 3 4 for N4+NLM skip)',
    )
    parser.add_argument(
        '--spacing', nargs=3, type=float, default=[1.0, 1.0, 1.0],
        metavar=('Z', 'Y', 'X'),
        help='Target isotropic spacing in mm (default: 1.0 1.0 1.0)',
    )
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Parallel workers (default: 1, recommend ≤2 for RAM)',
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Process at most N series (for testing)',
    )
    parser.add_argument(
        '--patch-size', type=int, default=96,
        help='3D patch size for dataset builder (default: 96)',
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.2,
        help='Validation split ratio (default: 0.2)',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Re-process even if outputs already exist',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Scan sources and report — no processing',
    )
    parser.add_argument(
        '--dashboard', action='store_true',
        help='Generate HTML dashboards after preprocessing',
    )
    parser.add_argument(
        '--dashboard-only', action='store_true',
        help='Only regenerate dashboards (skip preprocessing)',
    )
    parser.add_argument(
        '--no-dataset', action='store_true',
        help='Skip dataset builder (patches + manifest)',
    )
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Force UTF-8 output on Windows
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    Path(args.output).mkdir(parents=True, exist_ok=True)

    log_handlers = [logging.StreamHandler(sys.stdout)]
    try:
        log_handlers.append(
            logging.FileHandler(str(Path(args.output) / 'pipeline.log'),
                                encoding='utf-8', mode='a')
        )
    except Exception:
        pass

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=log_handlers,
    )
    logger = logging.getLogger(__name__)

    SEP = '=' * 64
    print('\n' + SEP)
    print('   ADDS CT Advanced Preprocessing Pipeline  v2.0')
    print('   10-Stage: HU > Body > N4 > NLM > CLAHE >')
    print('             5ch > ROI > QC > Isotropic > Annotation')
    print(SEP)

    t_start = time.time()

    # ── Dashboard only mode ──────────────────────────────────────
    if args.dashboard_only:
        print('\n[DASHBOARD-ONLY] Regenerating dashboards...')
        from src.medical_imaging.ct_preprocessing_dashboard import generate_all_dashboards
        paths = generate_all_dashboards(args.output, overwrite=True)
        print(f'\n  Generated {len(paths)} dashboards in {time.time()-t_start:.1f}s')
        return

    # ── Batch preprocessing ──────────────────────────────────────
    if not args.dry_run:
        print('\n[STEP 1] Scanning CT sources...')

    from src.medical_imaging.ct_batch_preprocessor import scan_ct_sources, BatchPreprocessor

    series_list = scan_ct_sources(args.sources)

    if args.dry_run:
        print(f'\n  DRY RUN — {len(series_list)} series found:\n')
        for i, s in enumerate(series_list, 1):
            print(f'  {i:3d}. [{s.input_type.upper():5s}] {s.series_id}  '
                  f'({len(s.file_paths)} files)')
        print(f'\n  Estimated output: ~{len(series_list)} qc_report.json + dashboard.html pairs')
        print(f'  Skip stages: {args.skip_stages or "none"}')
        print(f'  Target spacing: {args.spacing} mm')
        return

    print(f'  Found {len(series_list)} series.')
    print(f'\n[STEP 2] Running 10-stage preprocessing...\n')

    bp = BatchPreprocessor(
        output_root=args.output,
        skip_stages=args.skip_stages,
        target_spacing=tuple(args.spacing),
        overwrite=args.overwrite,
    )
    batch_results = bp.run_all(
        series_list,
        max_workers=args.workers,
        limit=args.limit,
    )

    n_ok = sum(1 for r in batch_results.values() if r.get('status') == 'OK')
    n_err = len(batch_results) - n_ok
    print(f'\n  Preprocessing done: {n_ok} OK, {n_err} ERROR ({time.time()-t_start:.1f}s elapsed)')

    # ── Dataset builder ──────────────────────────────────────────
    if not args.no_dataset:
        print('\n[STEP 3] Building training dataset (patches + manifest)...\n')
        from src.medical_imaging.ct_dataset_builder import CTDatasetBuilder
        builder = CTDatasetBuilder(
            output_root=args.output,
            patch_size=args.patch_size,
        )
        manifest = builder.build_all(
            qc_pass_only=True,
            val_ratio=args.val_ratio,
            overwrite=args.overwrite,
        )
        print(f'\n  Dataset: {manifest["total_patches"]} patches '
              f'({manifest["train_patches"]} train / {manifest["val_patches"]} val)')

    # ── Dashboard generation ─────────────────────────────────────
    if args.dashboard:
        print('\n[STEP 4] Generating HTML dashboards...\n')
        from src.medical_imaging.ct_preprocessing_dashboard import generate_all_dashboards
        paths = generate_all_dashboards(args.output, overwrite=args.overwrite)
        print(f'\n  Generated {len(paths)} dashboards')

    # ── Final summary ────────────────────────────────────────────
    elapsed = time.time() - t_start
    print('\n' + '═' * 64)
    print(f'  Pipeline complete in {elapsed:.1f}s')
    print(f'  Output root: {args.output}')
    print(f'  Batch summary: {args.output}/batch_summary.json')
    if not args.no_dataset:
        print(f'  Manifest: {args.output}/manifest.json')
    if args.dashboard:
        print(f'  Dashboards: {args.output}/dashboard_*.html')
    print('═' * 64 + '\n')


if __name__ == '__main__':
    main()
