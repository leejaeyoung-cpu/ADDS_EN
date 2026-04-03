"""
Comprehensive Validation Analysis for Perfect Reconstruction Results

Analyzes tumor measurement results to validate:
- Distribution quality
- Shape metrics accuracy
- Inside/Outside classification
- Outlier detection
- Quality assurance
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def analyze_tumor_results(enhanced_json_path):
    """
    Comprehensive analysis of tumor measurement results
    
    Generates:
    - Distribution plots
    - Statistical summaries
    - Outlier reports
    - Quality metrics
    """
    
    print("="*80)
    print("VALIDATION ANALYSIS - PERFECT RECONSTRUCTION RESULTS")
    print("="*80)
    
    # Load results
    with open(enhanced_json_path) as f:
        data = json.load(f)
    
    tumors = data['tumors']
    print(f"\nLoaded {len(tumors)} tumors")
    
    # Extract metrics
    volumes = np.array([t['volume_mm3'] for t in tumors])
    sphericities = np.array([t['sphericity'] for t in tumors])
    elongations = np.array([t['elongation'] for t in tumors])
    flatnesses = np.array([t['flatness'] for t in tumors])
    surface_areas = np.array([t['surface_area_mm2'] for t in tumors])
    diameters = np.array([t['max_diameter_mm'] for t in tumors])
    
    inside_colon = [t['distance_to_colon']['inside_colon'] for t in tumors]
    distances = [t['distance_to_colon']['min_distance_mm'] for t in tumors 
                 if not t['distance_to_colon']['inside_colon']]
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    print("\n[1/5] Generating distribution plots...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Row 1: Volume metrics
    plt.subplot(3, 4, 1)
    plt.hist(volumes, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Volume (mm³)', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.title('Volume Distribution', fontsize=11, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(3, 4, 2)
    plt.hist(diameters, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
    plt.xlabel('Diameter (mm)', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.title('Diameter Distribution', fontsize=11, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(3, 4, 3)
    plt.hist(surface_areas, bins=30, color='coral', edgecolor='black', alpha=0.7)
    plt.xlabel('Surface Area (mm²)', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.title('Surface Area Distribution', fontsize=11, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(3, 4, 4)
    plt.scatter(volumes, surface_areas, alpha=0.6, c=sphericities, cmap='RdYlGn')
    plt.xlabel('Volume (mm³)', fontsize=10)
    plt.ylabel('Surface Area (mm²)', fontsize=10)
    plt.title('SA vs Volume', fontsize=11, fontweight='bold')
    plt.colorbar(label='Sphericity')
    plt.grid(alpha=0.3)
    
    # Row 2: Shape metrics
    plt.subplot(3, 4, 5)
    plt.hist(sphericities, bins=30, color='gold', edgecolor='black', alpha=0.7)
    plt.xlabel('Sphericity', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.title('Sphericity Distribution', fontsize=11, fontweight='bold')
    plt.axvline(0.3, color='r', linestyle='--', label='Irregular threshold')
    plt.axvline(0.7, color='g', linestyle='--', label='Spherical threshold')
    plt.legend(fontsize=8)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(3, 4, 6)
    plt.hist(elongations, bins=30, color='orchid', edgecolor='black', alpha=0.7)
    plt.xlabel('Elongation', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.title('Elongation Distribution', fontsize=11, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(3, 4, 7)
    plt.hist(flatnesses, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.xlabel('Flatness', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.title('Flatness Distribution', fontsize=11, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(3, 4, 8)
    plt.scatter(sphericities, elongations, alpha=0.6, c=volumes, cmap='viridis')
    plt.xlabel('Sphericity', fontsize=10)
    plt.ylabel('Elongation', fontsize=10)
    plt.title('Sphericity vs Elongation', fontsize=11, fontweight='bold')
    plt.colorbar(label='Volume (mm³)')
    plt.grid(alpha=0.3)
    
    # Row 3: Location metrics
    plt.subplot(3, 4, 9)
    inside_count = sum(inside_colon)
    outside_count = len(tumors) - inside_count
    bars = plt.bar(['Inside\nColon', 'Outside\nColon'], [inside_count, outside_count],
                   color=['#FF4444', '#FFAA00'], edgecolor='black', alpha=0.8)
    plt.ylabel('Count', fontsize=10)
    plt.title('Location Classification', fontsize=11, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(tumors)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(3, 4, 10)
    if distances:
        plt.hist(distances, bins=20, color='orange', edgecolor='black', alpha=0.7)
        plt.xlabel('Distance to Colon (mm)', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.title('Distance for Outside Tumors', fontsize=11, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
    
    # Shape categories
    plt.subplot(3, 4, 11)
    very_irregular = sum(sphericities < 0.3)
    moderate = sum((sphericities >= 0.3) & (sphericities < 0.7))
    spherical = sum(sphericities >= 0.7)
    
    categories = ['Very\nIrregular\n(<0.3)', 'Moderate\n(0.3-0.7)', 'Nearly\nSpherical\n(≥0.7)']
    counts = [very_irregular, moderate, spherical]
    colors_cat = ['#FF6B6B', '#FFD93D', '#6BCF7F']
    
    bars = plt.bar(categories, counts, color=colors_cat, edgecolor='black', alpha=0.8)
    plt.ylabel('Count', fontsize=10)
    plt.title('Shape Categories', fontsize=11, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(tumors)*100:.1f}%)',
                ha='center', va='bottom', fontsize=8)
    plt.grid(axis='y', alpha=0.3)
    
    # Size categories
    plt.subplot(3, 4, 12)
    small = sum(volumes < 500)
    medium = sum((volumes >= 500) & (volumes < 2000))
    large = sum(volumes >= 2000)
    
    size_cats = ['Small\n(<500)', 'Medium\n(500-2000)', 'Large\n(≥2000)']
    size_counts = [small, medium, large]
    size_colors = ['#E3F2FD', '#90CAF9', '#1976D2']
    
    bars = plt.bar(size_cats, size_counts, color=size_colors, edgecolor='black', alpha=0.8)
    plt.ylabel('Count', fontsize=10)
    plt.title('Size Categories (mm³)', fontsize=11, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(tumors)*100:.1f}%)',
                ha='center', va='bottom', fontsize=8)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path('outputs/inha_3d_analysis/validation_distributions.png')
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    
    # ========================================================================
    # STATISTICAL ANALYSIS
    # ========================================================================
    
    print("\n[2/5] Computing statistical summaries...")
    
    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nTotal Tumors: {len(tumors)}")
    
    print(f"\n--- Volume Statistics (mm³) ---")
    print(f"  Min:      {volumes.min():.1f}")
    print(f"  Q1:       {np.percentile(volumes, 25):.1f}")
    print(f"  Median:   {np.median(volumes):.1f}")
    print(f"  Q3:       {np.percentile(volumes, 75):.1f}")
    print(f"  Max:      {volumes.max():.1f}")
    print(f"  Mean:     {volumes.mean():.1f}")
    print(f"  Std:      {volumes.std():.1f}")
    
    print(f"\n--- Sphericity Statistics ---")
    print(f"  Min:      {sphericities.min():.3f}")
    print(f"  Q1:       {np.percentile(sphericities, 25):.3f}")
    print(f"  Median:   {np.median(sphericities):.3f}")
    print(f"  Q3:       {np.percentile(sphericities, 75):.3f}")
    print(f"  Max:      {sphericities.max():.3f}")
    print(f"  Mean:     {sphericities.mean():.3f}")
    print(f"  Std:      {sphericities.std():.3f}")
    
    print(f"\n--- Shape Categories ---")
    print(f"  Very Irregular (<0.3):   {very_irregular:2d} ({very_irregular/len(tumors)*100:5.1f}%)")
    print(f"  Moderate (0.3-0.7):      {moderate:2d} ({moderate/len(tumors)*100:5.1f}%)")
    print(f"  Nearly Spherical (≥0.7): {spherical:2d} ({spherical/len(tumors)*100:5.1f}%)")
    
    print(f"\n--- Size Categories ---")
    print(f"  Small (<500 mm³):        {small:2d} ({small/len(tumors)*100:5.1f}%)")
    print(f"  Medium (500-2000 mm³):   {medium:2d} ({medium/len(tumors)*100:5.1f}%)")
    print(f"  Large (≥2000 mm³):       {large:2d} ({large/len(tumors)*100:5.1f}%)")
    
    print(f"\n--- Location Classification ---")
    print(f"  Inside Colon:  {inside_count:2d} ({inside_count/len(tumors)*100:5.1f}%)")
    print(f"  Outside Colon: {outside_count:2d} ({outside_count/len(tumors)*100:5.1f}%)")
    
    if distances:
        print(f"\n--- Distance to Colon (Outside Tumors) ---")
        print(f"  Min:      {min(distances):.1f} mm")
        print(f"  Median:   {np.median(distances):.1f} mm")
        print(f"  Max:      {max(distances):.1f} mm")
        print(f"  Mean:     {np.mean(distances):.1f} mm")
    
    # ========================================================================
    # OUTLIER DETECTION
    # ========================================================================
    
    print(f"\n[3/5] Detecting outliers...")
    
    print(f"\n{'='*60}")
    print("OUTLIER ANALYSIS")
    print(f"{'='*60}")
    
    # Large tumors
    large_tumors = [t for t in tumors if t['volume_mm3'] > 5000]
    print(f"\nLarge Tumors (>5000 mm³): {len(large_tumors)}")
    for t in large_tumors[:5]:
        print(f"  Tumor #{t['tumor_id']:2d}: {t['volume_mm3']:8.1f} mm³, sphericity={t['sphericity']:.3f}")
    
    # Very irregular
    irregular_tumors = [t for t in tumors if t['sphericity'] < 0.3]
    print(f"\nVery Irregular (sphericity <0.3): {len(irregular_tumors)}")
    for t in irregular_tumors[:5]:
        print(f"  Tumor #{t['tumor_id']:2d}: sphericity={t['sphericity']:.3f}, volume={t['volume_mm3']:.1f} mm³")
    
    # High surface area
    high_sa_tumors = [t for t in tumors if t['surface_area_mm2'] > 5000]
    print(f"\nHigh Surface Area (>5000 mm²): {len(high_sa_tumors)}")
    for t in high_sa_tumors[:5]:
        print(f"  Tumor #{t['tumor_id']:2d}: SA={t['surface_area_mm2']:.1f} mm², volume={t['volume_mm3']:.1f} mm³")
    
    # Perfect spheres (potential fallbacks)
    perfect_spheres = [t for t in tumors if t['sphericity'] >= 0.99]
    print(f"\nPerfect Spheres (sphericity ≥0.99): {len(perfect_spheres)}")
    for t in perfect_spheres[:5]:
        print(f"  Tumor #{t['tumor_id']:2d}: sphericity={t['sphericity']:.3f}")
    
    # ========================================================================
    # QUALITY CHECKS
    # ========================================================================
    
    print(f"\n[4/5] Running quality checks...")
    
    print(f"\n{'='*60}")
    print("QUALITY ASSURANCE")
    print(f"{'='*60}")
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: No fallback spheres
    total_checks += 1
    fallback_count = sum(1 for t in tumors if t['sphericity'] == 1.0 and t['elongation'] == 0.0)
    status = "PASS" if fallback_count == 0 else "FAIL"
    if status == "PASS": checks_passed += 1
    print(f"\n[{status}] Fallback Spheres: {fallback_count} (expect 0)")
    
    # Check 2: Sphericity range
    total_checks += 1
    status = "PASS" if 0.15 <= sphericities.min() <= 1.0 and sphericities.max() <= 1.0 else "WARN"
    if status == "PASS": checks_passed += 1
    print(f"[{status}] Sphericity Range: {sphericities.min():.3f} - {sphericities.max():.3f} (expect 0.15-1.0)")
    
    # Check 3: Inside/Outside split
    total_checks += 1
    status = "PASS" if 30 <= inside_count/len(tumors)*100 <= 70 else "WARN"
    if status == "PASS": checks_passed += 1
    print(f"[{status}] Inside/Outside Split: {inside_count/len(tumors)*100:.1f}% inside (expect 30-70%)")
    
    # Check 4: Distance metrics for outside tumors
    total_checks += 1
    outside_tumors_with_distance = [t for t in tumors 
                                    if not t['distance_to_colon']['inside_colon']
                                    and t['distance_to_colon']['min_distance_mm'] > 0]
    status = "PASS" if len(outside_tumors_with_distance) == outside_count else "WARN"
    if status == "PASS": checks_passed += 1
    print(f"[{status}] Distance Metrics: {len(outside_tumors_with_distance)}/{outside_count} outside tumors have distance >0")
    
    # Check 5: SA/Volume relationship
    total_checks += 1
    expected_sa = 4.836 * (volumes ** (2/3))  # For perfect spheres
    sa_ratio = surface_areas / expected_sa
    extreme_sa = sum((sa_ratio > 10) | (sa_ratio < 0.1))
    status = "PASS" if extreme_sa < len(tumors) * 0.05 else "WARN"
    if status == "PASS": checks_passed += 1
    print(f"[{status}] SA/Volume Relationship: {extreme_sa} extreme outliers (expect <{int(len(tumors)*0.05)})")
    
    print(f"\n{'='*60}")
    print(f"QUALITY SCORE: {checks_passed}/{total_checks} checks passed ({checks_passed/total_checks*100:.0f}%)")
    print(f"{'='*60}")
    
    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    
    print(f"\n[5/5] Generating summary report...")
    
    summary = {
        'total_tumors': len(tumors),
        'volume': {
            'min': float(volumes.min()),
            'median': float(np.median(volumes)),
            'max': float(volumes.max()),
            'mean': float(volumes.mean())
        },
        'sphericity': {
            'min': float(sphericities.min()),
            'median': float(np.median(sphericities)),
            'max': float(sphericities.max()),
            'mean': float(sphericities.mean())
        },
        'shape_categories': {
            'very_irregular': int(very_irregular),
            'moderate': int(moderate),
            'nearly_spherical': int(spherical)
        },
        'size_categories': {
            'small': int(small),
            'medium': int(medium),
            'large': int(large)
        },
        'location': {
            'inside': int(inside_count),
            'outside': int(outside_count)
        },
        'quality': {
            'checks_passed': checks_passed,
            'total_checks': total_checks,
            'score': checks_passed / total_checks
        },
        'outliers': {
            'large_tumors': len(large_tumors),
            'very_irregular': len(irregular_tumors),
            'high_surface_area': len(high_sa_tumors),
            'perfect_spheres': len(perfect_spheres)
        }
    }
    
    summary_path = Path('outputs/inha_3d_analysis/validation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Saved: {summary_path}")
    
    print(f"\n{'='*80}")
    print("VALIDATION ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    return summary


if __name__ == "__main__":
    enhanced_json = Path("outputs/inha_3d_analysis/tumors_3d_enhanced_perfect.json")
    
    if not enhanced_json.exists():
        print(f"ERROR: Enhanced JSON not found: {enhanced_json}")
        print("Please run the perfect reconstruction pipeline first.")
        exit(1)
    
    summary = analyze_tumor_results(enhanced_json)
    
    print(f"\n✅ Validation complete!")
    print(f"Quality Score: {summary['quality']['score']*100:.0f}%")
    print(f"Distribution plots: outputs/inha_3d_analysis/validation_distributions.png")
    print(f"Summary report: outputs/inha_3d_analysis/validation_summary.json")
