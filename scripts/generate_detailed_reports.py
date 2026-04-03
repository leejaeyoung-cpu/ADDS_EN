"""
Generate detailed analysis report for each case
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import label
import json


def analyze_single_case(case_id, dataset_dir, output_dir):
    """Create detailed analysis report for a single case"""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    case_output_dir = output_dir / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    ct_file = dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
    body_mask_file = dataset_dir / "body_masks" / f"{case_id}_body.nii.gz"
    colon_mask_file = dataset_dir / "colon_masks" / f"{case_id}_colon.nii.gz"
    
    if not ct_file.exists():
        print(f"[ERROR] {case_id}: CT file not found")
        return None
    
    if not body_mask_file.exists():
        print(f"[ERROR] {case_id}: Body mask not found")
        return None
    
    if not colon_mask_file.exists():
        print(f"[WARNING] {case_id}: Colon mask not found")
        colon_mask_file = None
    
    ct_data = nib.load(ct_file).get_fdata()
    body_mask = nib.load(body_mask_file).get_fdata()
    colon_mask = nib.load(colon_mask_file).get_fdata() if colon_mask_file else None
    
    # Overall statistics
    total_slices = ct_data.shape[2]
    
    # Compute metrics for each slice
    slice_stats = []
    for slice_idx in range(total_slices):
        ct_slice = ct_data[:, :, slice_idx]
        body_slice = body_mask[:, :, slice_idx]
        colon_slice = colon_mask[:, :, slice_idx] if colon_mask is not None else np.zeros_like(body_slice)
        
        actual_body = ct_slice > -500
        
        total_pixels = ct_slice.size
        actual_body_pixels = actual_body.sum()
        body_mask_pixels = body_slice.astype(bool).sum()
        colon_pixels = colon_slice.astype(bool).sum()
        
        # Check bottom region
        height = ct_slice.shape[1]
        y_10pct = int(height * 0.10)
        body_in_bottom = actual_body[:, :y_10pct].sum()
        
        slice_stats.append({
            'slice': slice_idx,
            'actual_body_coverage': actual_body_pixels / total_pixels * 100,
            'mask_coverage': body_mask_pixels / total_pixels * 100,
            'colon_pixels': colon_pixels,
            'body_in_bottom_10': body_in_bottom / actual_body_pixels * 100 if actual_body_pixels > 0 else 0
        })
    
    # Find key slices
    coverages = [s['mask_coverage'] for s in slice_stats]
    max_coverage_idx = np.argmax(coverages)
    mid_slice_idx = total_slices // 2
    
    # Find slices with colon
    colon_slices = [s['slice'] for s in slice_stats if s['colon_pixels'] > 0]
    if colon_slices:
        mid_colon_idx = colon_slices[len(colon_slices) // 2]
    else:
        mid_colon_idx = mid_slice_idx
    
    key_slices = [
        ('Bottom', int(total_slices * 0.2)),
        ('Lower', int(total_slices * 0.35)),
        ('Middle', mid_slice_idx),
        ('Max Coverage', max_coverage_idx),
        ('Colon Region', mid_colon_idx),
        ('Upper', int(total_slices * 0.7)),
    ]
    
    # Create visualization
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(6, 5, hspace=0.3, wspace=0.3)
    
    for row, (label_text, slice_idx) in enumerate(key_slices):
        ct_slice = ct_data[:, :, slice_idx]
        body_slice = body_mask[:, :, slice_idx]
        colon_slice = colon_mask[:, :, slice_idx] if colon_mask is not None else np.zeros_like(body_slice)
        actual_body = ct_slice > -500
        
        stats = slice_stats[slice_idx]
        
        # Column 1: Original CT
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(ct_slice.T, cmap='gray', origin='lower')
        ax.set_title(f'{label_text}\nSlice {slice_idx}', fontsize=9)
        ax.axis('off')
        
        # Column 2: Actual body (HU > -500)
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(ct_slice.T, cmap='gray', origin='lower')
        ax.imshow(actual_body.T, cmap='Greens', alpha=0.3, origin='lower')
        ax.set_title(f'Actual Body (HU>-500)\n{stats["actual_body_coverage"]:.1f}%', fontsize=9)
        ax.axis('off')
        
        # Column 3: Body mask
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(ct_slice.T, cmap='gray', origin='lower')
        ax.imshow(body_slice.T, cmap='Reds', alpha=0.5, origin='lower')
        ax.set_title(f'Body Mask\n{stats["mask_coverage"]:.1f}%', fontsize=9)
        ax.axis('off')
        
        # Column 4: Colon mask
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(ct_slice.T, cmap='gray', origin='lower')
        if stats['colon_pixels'] > 0:
            ax.imshow(colon_slice.T, cmap='Greens', alpha=0.5, origin='lower')
            ax.set_title(f'Colon Mask\n{stats["colon_pixels"]} px', fontsize=9)
        else:
            ax.set_title(f'Colon Mask\nNone', fontsize=9, color='gray')
        ax.axis('off')
        
        # Column 5: Combined
        ax = fig.add_subplot(gs[row, 4])
        ax.imshow(ct_slice.T, cmap='gray', origin='lower')
        ax.imshow(body_slice.T, cmap='Reds', alpha=0.3, origin='lower')
        if stats['colon_pixels'] > 0:
            ax.imshow(colon_slice.T, cmap='Greens', alpha=0.5, origin='lower')
        ax.set_title(f'Combined\nBody(R)+Colon(G)', fontsize=9)
        ax.axis('off')
    
    plt.suptitle(f'{case_id} - Detailed Analysis Report', fontsize=14, fontweight='bold')
    
    # Save visualization
    viz_file = case_output_dir / f"{case_id}_detailed_analysis.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    slices = [s['slice'] for s in slice_stats]
    actual_cov = [s['actual_body_coverage'] for s in slice_stats]
    mask_cov = [s['mask_coverage'] for s in slice_stats]
    colon_px = [s['colon_pixels'] for s in slice_stats]
    bottom_pct = [s['body_in_bottom_10'] for s in slice_stats]
    
    # Plot 1: Coverage comparison
    axes[0, 0].plot(slices, actual_cov, 'g-', label='Actual Body (HU>-500)', linewidth=2)
    axes[0, 0].plot(slices, mask_cov, 'r-', label='Body Mask', linewidth=2)
    axes[0, 0].fill_between(slices, 0, mask_cov, alpha=0.2, color='red')
    axes[0, 0].set_xlabel('Slice Index')
    axes[0, 0].set_ylabel('Coverage (%)')
    axes[0, 0].set_title('Body Coverage Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Colon distribution
    axes[0, 1].bar(slices, colon_px, color='green', alpha=0.6)
    axes[0, 1].set_xlabel('Slice Index')
    axes[0, 1].set_ylabel('Colon Pixels')
    axes[0, 1].set_title('Colon Mask Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Coverage difference
    diff = [a - m for a, m in zip(actual_cov, mask_cov)]
    axes[1, 0].plot(slices, diff, 'b-', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].fill_between(slices, 0, diff, where=[d >= 0 for d in diff], alpha=0.3, color='orange', label='Under-inclusion')
    axes[1, 0].fill_between(slices, 0, diff, where=[d < 0 for d in diff], alpha=0.3, color='blue', label='Over-inclusion')
    axes[1, 0].set_xlabel('Slice Index')
    axes[1, 0].set_ylabel('Difference (%)')
    axes[1, 0].set_title('Mask Accuracy (Actual - Mask)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Bottom region check
    axes[1, 1].plot(slices, bottom_pct, 'purple', linewidth=2)
    axes[1, 1].axhline(y=5, color='r', linestyle='--', linewidth=2, label='5% threshold')
    axes[1, 1].fill_between(slices, 0, bottom_pct, where=[b > 5 for b in bottom_pct], alpha=0.3, color='red')
    axes[1, 1].set_xlabel('Slice Index')
    axes[1, 1].set_ylabel('% of Body in Bottom 10%')
    axes[1, 1].set_title('Bottom Region Body Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    metrics_file = case_output_dir / f"{case_id}_metrics.png"
    plt.savefig(metrics_file, dpi=120, bbox_inches='tight')
    plt.close()
    
    # Compute summary statistics
    total_actual_body = sum(s['actual_body_coverage'] for s in slice_stats) / len(slice_stats)
    total_mask_coverage = sum(s['mask_coverage'] for s in slice_stats) / len(slice_stats)
    total_colon_pixels = sum(s['colon_pixels'] for s in slice_stats)
    problematic_slices = [s['slice'] for s in slice_stats if s['body_in_bottom_10'] > 5]
    
    avg_diff = sum(abs(s['actual_body_coverage'] - s['mask_coverage']) for s in slice_stats) / len(slice_stats)
    
    summary = {
        'case_id': case_id,
        'total_slices': int(total_slices),
        'avg_actual_body_coverage': round(float(total_actual_body), 2),
        'avg_mask_coverage': round(float(total_mask_coverage), 2),
        'total_colon_pixels': int(total_colon_pixels),
        'colon_slices': int(len(colon_slices)),
        'problematic_slices': int(len(problematic_slices)),
        'avg_abs_difference': round(float(avg_diff), 2),
        'max_coverage_slice': int(max_coverage_idx),
        'max_coverage_value': round(float(coverages[max_coverage_idx]), 2),
        'quality_score': 'PASS' if len(problematic_slices) < total_slices * 0.2 and avg_diff < 5 else 'WARNING'
    }
    
    # Save summary JSON
    summary_file = case_output_dir / f"{case_id}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create text report
    report_lines = [
        f"{'='*70}",
        f"DETAILED ANALYSIS REPORT - {case_id}",
        f"{'='*70}",
        f"",
        f"OVERVIEW:",
        f"  Total slices: {total_slices}",
        f"  Image dimensions: {ct_data.shape}",
        f"",
        f"BODY MASK STATISTICS:",
        f"  Average actual body (HU>-500): {total_actual_body:.2f}%",
        f"  Average mask coverage: {total_mask_coverage:.2f}%",
        f"  Average absolute difference: {avg_diff:.2f}%",
        f"  Max coverage slice: {max_coverage_idx} ({coverages[max_coverage_idx]:.2f}%)",
        f"",
        f"COLON MASK STATISTICS:",
        f"  Total colon pixels: {int(total_colon_pixels):,}",
        f"  Slices with colon: {len(colon_slices)}/{total_slices} ({len(colon_slices)/total_slices*100:.1f}%)",
        f"  Colon slice range: {min(colon_slices) if colon_slices else 'N/A'} - {max(colon_slices) if colon_slices else 'N/A'}",
        f"",
        f"QUALITY ASSESSMENT:",
        f"  Problematic slices (>5% body in bottom 10%): {len(problematic_slices)}/{total_slices}",
        f"  Quality score: {summary['quality_score']}",
        f"",
        f"FILES GENERATED:",
        f"  - {viz_file.name}",
        f"  - {metrics_file.name}",
        f"  - {summary_file.name}",
        f"  - {case_id}_report.txt",
        f"",
        f"{'='*70}",
    ]
    
    report_file = case_output_dir / f"{case_id}_report.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"[{case_id}] Analysis complete - Quality: {summary['quality_score']}")
    
    return summary


def main():
    """Generate reports for all cases"""
    dataset_dir = Path("f:/ADDS/nnUNet_raw/Dataset010_Colon")
    output_dir = Path("f:/ADDS/outputs/detailed_case_reports")
    
    # Find all cases
    images_dir = dataset_dir / "imagesTr"
    case_files = sorted(images_dir.glob("*_0000.nii.gz"))
    case_ids = [f.name.replace("_0000.nii.gz", "") for f in case_files]
    
    print("="*70)
    print(f"GENERATING DETAILED ANALYSIS REPORTS")
    print("="*70)
    print(f"Total cases: {len(case_ids)}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    print()
    
    all_summaries = []
    
    for idx, case_id in enumerate(case_ids, 1):
        print(f"[{idx}/{len(case_ids)}] Processing {case_id}...")
        summary = analyze_single_case(case_id, dataset_dir, output_dir)
        if summary:
            all_summaries.append(summary)
    
    # Create master summary
    print("\n" + "="*70)
    print("CREATING MASTER SUMMARY")
    print("="*70)
    
    master_summary_file = output_dir / "master_summary.json"
    with open(master_summary_file, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    # Create master report
    pass_count = sum(1 for s in all_summaries if s['quality_score'] == 'PASS')
    warning_count = len(all_summaries) - pass_count
    
    master_lines = [
        f"{'='*70}",
        f"MASTER ANALYSIS REPORT",
        f"{'='*70}",
        f"",
        f"TOTAL CASES: {len(all_summaries)}",
        f"  PASS: {pass_count} ({pass_count/len(all_summaries)*100:.1f}%)",
        f"  WARNING: {warning_count} ({warning_count/len(all_summaries)*100:.1f}%)",
        f"",
        f"AVERAGE STATISTICS:",
        f"  Avg mask coverage: {sum(s['avg_mask_coverage'] for s in all_summaries)/len(all_summaries):.2f}%",
        f"  Avg colon pixels: {sum(s['total_colon_pixels'] for s in all_summaries)/len(all_summaries):.0f}",
        f"  Avg abs difference: {sum(s['avg_abs_difference'] for s in all_summaries)/len(all_summaries):.2f}%",
        f"",
        f"CASES WITH WARNINGS:",
    ]
    
    for s in all_summaries:
        if s['quality_score'] == 'WARNING':
            master_lines.append(f"  - {s['case_id']}: {s['problematic_slices']} problematic slices, diff={s['avg_abs_difference']:.2f}%")
    
    master_lines.extend([
        f"",
        f"Output directory: {output_dir}",
        f"Master summary: {master_summary_file}",
        f"",
        f"{'='*70}",
    ])
    
    master_report_file = output_dir / "master_report.txt"
    with open(master_report_file, 'w') as f:
        f.write('\n'.join(master_lines))
    
    print("\n" + '\n'.join(master_lines))
    
    print(f"\nAll reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
