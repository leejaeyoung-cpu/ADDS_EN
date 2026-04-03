"""
DICOM Series Analyzer
====================
Analyzes all DICOM files in a folder to identify:
- Scout vs. Diagnostic series
- Series descriptions
- Slice thickness
- Image dimensions
"""
import pydicom
from pathlib import Path
from collections import defaultdict
import json


def analyze_dicom_folder(folder_path):
    """
    Analyze all DICOM files and group by series
    
    Returns:
        dict: Series information grouped by SeriesInstanceUID
    """
    folder = Path(folder_path)
    series_dict = defaultdict(lambda: {
        'files': [],
        'series_description': 'Unknown',
        'slice_thickness': None,
        'image_size': None,
        'modality': None,
        'series_number': None,
        'is_scout': False
    })
    
    dicom_files = list(folder.glob("*.dcm"))
    print(f"\n[*] Found {len(dicom_files)} DICOM files in {folder}")
    print(f"[*] Analyzing metadata...\n")
    
    for dcm_file in sorted(dicom_files):
        try:
            ds = pydicom.dcmread(dcm_file, stop_before_pixels=True)
            
            series_uid = ds.get('SeriesInstanceUID', 'UNKNOWN')
            
            # Extract metadata
            series_desc = str(ds.get('SeriesDescription', 'Unknown'))
            slice_thickness = float(ds.get('SliceThickness', 0))
            modality = str(ds.get('Modality', 'Unknown'))
            series_number = int(ds.get('SeriesNumber', 0))
            
            # Check if Scout/Localizer
            is_scout = any(keyword in series_desc.upper() 
                          for keyword in ['SCOUT', 'LOCALIZER', 'TOPO', 'SURVIEW'])
            
            # Update series info
            series_info = series_dict[series_uid]
            series_info['files'].append(str(dcm_file))
            series_info['series_description'] = series_desc
            series_info['slice_thickness'] = slice_thickness
            series_info['image_size'] = f"{ds.Rows}x{ds.Columns}"
            series_info['modality'] = modality
            series_info['series_number'] = series_number
            series_info['is_scout'] = is_scout or slice_thickness > 10  # Thick slices are scouts
            
        except Exception as e:
            print(f"[WARNING] Failed to read {dcm_file.name}: {e}")
    
    return series_dict


def print_series_summary(series_dict):
    """Print formatted summary of series"""
    print("="*80)
    print("DICOM SERIES SUMMARY")
    print("="*80)
    
    # Sort by series number
    sorted_series = sorted(series_dict.items(), 
                          key=lambda x: x[1].get('series_number', 999))
    
    diagnostic_count = 0
    scout_count = 0
    
    for series_uid, info in sorted_series:
        num_files = len(info['files'])
        series_type = "[SCOUT]" if info['is_scout'] else "[DIAGNOSTIC]"
        
        if info['is_scout']:
            scout_count += num_files
        else:
            diagnostic_count += num_files
        
        print(f"\n{series_type}")
        print(f"  Series #{info['series_number']}: {info['series_description']}")
        print(f"  Files: {num_files}")
        print(f"  Image Size: {info['image_size']}")
        print(f"  Slice Thickness: {info['slice_thickness']:.2f} mm")
        print(f"  Modality: {info['modality']}")
        print(f"  Series UID: {series_uid[:40]}...")
    
    print("\n" + "="*80)
    print(f"TOTALS:")
    print(f"  Diagnostic slices: {diagnostic_count}")
    print(f"  Scout slices: {scout_count}")
    print(f"  Total series: {len(series_dict)}")
    print("="*80 + "\n")


def get_diagnostic_series(series_dict):
    """
    Filter and return only diagnostic series
    
    Returns:
        dict: Only diagnostic series (Scout excluded)
    """
    diagnostic = {}
    for series_uid, info in series_dict.items():
        if not info['is_scout']:
            diagnostic[series_uid] = info
    
    return diagnostic


def main():
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = r"f:\ADDS\CTdata\CTdcm"
    
    # Analyze
    series_dict = analyze_dicom_folder(folder_path)
    
    # Print summary
    print_series_summary(series_dict)
    
    # Get diagnostic series
    diagnostic = get_diagnostic_series(series_dict)
    
    if len(diagnostic) > 0:
        print(f"\n[+] Found {len(diagnostic)} DIAGNOSTIC series")
        print(f"[+] Recommended for tumor detection:\n")
        
        for series_uid, info in diagnostic.items():
            print(f"  - {info['series_description']} ({len(info['files'])} slices)")
    else:
        print("\n[!] No diagnostic series found!")
    
    # Save results
    output_file = Path(folder_path) / "series_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'all_series': {k: {**v, 'files': len(v['files'])} for k, v in series_dict.items()},
            'diagnostic_series': {k: {**v, 'files': len(v['files'])} for k, v in diagnostic.items()}
        }, f, indent=2)
    
    print(f"\n[+] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
