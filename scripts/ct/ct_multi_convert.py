"""
Full CT Pipeline: CTdata1 Delay + CTdata2 Pre/Post
===================================================
Step 1: CTdata1 Delay -> NIfTI -> nnU-Net Task10
Step 2: CTdata2 Pre (0930) -> NIfTI -> nnU-Net Task10
Step 3: CTdata2 Post (1223) -> NIfTI -> nnU-Net Task10
Step 4: If all fail -> export PNG slices -> YOLO fallback
Step 5: Evaluate vs CTdata1 GT
"""
import os, sys, glob, shutil, json, warnings
import numpy as np
import nibabel as nib
import pydicom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

warnings.filterwarnings('ignore')

SAVE   = r'F:\ADDS'
DATA1  = r'F:\ADDS\CTdata1'
DATA2  = r'F:\ADDS\CTdata2'
NNUNET_RESULTS = r'C:\nnUNet_data\nnUNet_results'

print('='*68)
print('Multi-dataset nnU-Net Pipeline')
print('CTdata1 Delay + CTdata2 Pre + CTdata2 Post')
print('='*68)

# ================================================================
# Utility: Load DICOM series sorted by z-position
# ================================================================
def load_dcm_sorted(files, require_rows=None):
    slices = []
    meta   = {}
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            rows = int(getattr(ds,'Rows',0))
            if require_rows and rows != require_rows:
                continue
            try:    z = float(ds.ImagePositionPatient[2])
            except: z = float(getattr(ds,'InstanceNumber',0))
            slope = float(getattr(ds,'RescaleSlope',1))
            inter = float(getattr(ds,'RescaleIntercept',-1024))
            hu    = ds.pixel_array.astype(np.float32)*slope + inter
            slices.append((z, hu, ds))
        except: pass
    if not slices: return None, None
    slices.sort(key=lambda x: x[0])
    vol  = np.stack([s[1] for s in slices])
    ds0  = slices[0][2]
    try:    sp = [float(x) for x in ds0.PixelSpacing]
    except: sp = [1.0,1.0]
    try:    th = float(ds0.SliceThickness)
    except: th = float(abs(slices[1][0]-slices[0][0])) if len(slices)>1 else 5.0
    meta = {'spacing':sp,'thick':th,'desc':str(getattr(ds0,'SeriesDescription','?')),
            'n':vol.shape[0],'rows':vol.shape[1],'cols':vol.shape[2]}
    return vol, meta

# ================================================================
# Step 1: CTdata1 Delay DICOM -> NIfTI
# ================================================================
print('\n[1] CTdata1 Delay series -> NIfTI')
delay_files = sorted(glob.glob(os.path.join(DATA1,'CTdcm','*.dcm')))
# Filter to Delay series from series_analysis
delay_files_filtered = []
for f in delay_files:
    try:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        desc = str(getattr(ds,'SeriesDescription',''))
        if 'delay' in desc.lower() or 'Delay' in desc:
            delay_files_filtered.append(f)
    except: pass

print(f'  Delay files found: {len(delay_files_filtered)}')
vol_delay, m_delay = load_dcm_sorted(delay_files_filtered, require_rows=512)
if vol_delay is not None:
    print(f'  Volume: {vol_delay.shape}  thick={m_delay["thick"]}mm  HU=[{vol_delay.min():.0f},{vol_delay.max():.0f}]')
    # Save as NIfTI
    zooms = (m_delay['spacing'][0], m_delay['spacing'][1], m_delay['thick'])
    aff   = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
    nii   = nib.Nifti1Image(vol_delay.astype(np.float32), aff)
    nii.header.set_zooms(zooms)
    out1  = os.path.join(SAVE,'nnunet_delay_input','Patient002227784_Delay_0000.nii.gz')
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    nib.save(nii, out1)
    print(f'  Saved: {out1}')
else:
    print('  Delay series not found -- trying by file size pattern')
    # Fallback: CTdata1 delay is the series with ~119 files different from Artery
    all_series = {}
    for f in delay_files[:300]:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            sid = str(getattr(ds,'SeriesInstanceUID','?'))
            if sid not in all_series: all_series[sid] = {'desc':str(getattr(ds,'SeriesDescription','?')),'files':[]}
            all_series[sid]['files'].append(f)
        except: pass
    print(f'  Series found in CTdcm: {len(all_series)}')
    for sid,(v) in list(all_series.items())[:5]:
        print(f'    {all_series[sid]["desc"]:40s} n={len(all_series[sid]["files"])}')

# ================================================================
# Step 2: CTdata2 Pre (0930) -> DICOM analysis -> NIfTI groups
# ================================================================
print('\n[2] CTdata2: Analyzing DICOM series...')
data2_files_all = sorted(glob.glob(os.path.join(DATA2,'*.dcm')))
print(f'  Total DCM files: {len(data2_files_all)}')

# Group by prefix and rows
groups = {}
for f in data2_files_all:
    bn = os.path.basename(f)
    prefix = bn[:4]   # '0930' or '1223'
    try:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        rows = int(getattr(ds,'Rows',0))
        thick = float(getattr(ds,'SliceThickness',1))
        desc  = str(getattr(ds,'SeriesDescription','?'))
        key   = f'{prefix}|{rows}|{round(thick,1)}'
        if key not in groups: groups[key] = {'prefix':prefix,'rows':rows,'thick':thick,'desc':desc,'files':[]}
        groups[key]['files'].append(f)
    except: pass

print(f'\n  Groups found:')
for key,v in sorted(groups.items(), key=lambda x: -len(x[1]['files'])):
    print(f'    {v["prefix"]} rows={v["rows"]:4d} thick={v["thick"]}mm count={len(v["files"]):4d}  desc={v["desc"][:35]}')

# ================================================================
# Step 3: Convert each major group to NIfTI
# ================================================================
print('\n[3] Converting CTdata2 groups to NIfTI...')
converted = {}
for key,v in sorted(groups.items(), key=lambda x: -len(x[1]['files'])):
    if len(v['files']) < 50: continue   # skip small series
    if v['rows'] < 256: continue        # skip scouts/thumbnails
    name = f'{v["prefix"]}_rows{v["rows"]}_n{len(v["files"])}'
    vol, m = load_dcm_sorted(v['files'], require_rows=v['rows'])
    if vol is None or vol.shape[0] < 30: continue
    zooms = (m['spacing'][0], m['spacing'][1], m['thick'])
    aff   = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
    nii   = nib.Nifti1Image(vol.astype(np.float32), aff)
    nii.header.set_zooms(zooms)
    out_path = os.path.join(SAVE, f'nnunet_{name}_input')
    os.makedirs(out_path, exist_ok=True)
    nii_path = os.path.join(out_path, f'Patient_{name}_0000.nii.gz')
    nib.save(nii, nii_path)
    converted[name] = {'path': nii_path, 'shape': vol.shape, 'meta': m}
    print(f'  Saved: {nii_path} shape={vol.shape} HU=[{vol.min():.0f},{vol.max():.0f}]')

print(f'\n  Converted: {len(converted)} volumes')
print('\nConversion results:')
for name, info in converted.items():
    print(f'  {name}: {info["shape"]}')

# Save mapping for next step
mapping = {name:{'nifti':info['path'],'shape':list(info['shape']),'meta':info['meta']}
           for name,info in converted.items()}
with open(os.path.join(SAVE,'ctdata2_nifti_mapping.json'),'w') as f:
    json.dump(mapping,f,indent=2)
print(f'\nMapping saved -> {os.path.join(SAVE,"ctdata2_nifti_mapping.json")}')
