#!/usr/bin/env python3
"""
Interactive 3D Tumor Mask Viewer
Browse CT slices with tumor mask overlay
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path

def load_nifti(file_path: Path) -> np.ndarray:
    """Load NIfTI file"""
    nii = nib.load(str(file_path))
    return nii.get_fdata()

class TumorMaskViewer:
    """Interactive viewer for tumor masks"""
    
    def __init__(self, ct_volume, binary_mask, multiclass_mask):
        self.ct_volume = ct_volume
        self.binary_mask = binary_mask
        self.multiclass_mask = multiclass_mask
        self.n_slices = ct_volume.shape[0]
        self.current_slice = self.n_slices // 2
        
        # Find slices with tumors
        self.tumor_slices = np.where(binary_mask.sum(axis=(1, 2)) > 0)[0]
        print(f"\n[TUMOR SLICES] Found tumors in {len(self.tumor_slices)} slices")
        print(f"   Slice range: {self.tumor_slices.min()} - {self.tumor_slices.max()}")
        
        # Set initial slice to first tumor slice
        if len(self.tumor_slices) > 0:
            self.current_slice = self.tumor_slices[0]
        
        self.setup_figure()
    
    def setup_figure(self):
        """Setup matplotlib figure with slider"""
        self.fig = plt.figure(figsize=(20, 7))
        
        # Create grid
        gs = self.fig.add_gridspec(2, 3, height_ratios=[20, 1], hspace=0.3, wspace=0.2)
        
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        self.ax_slider = self.fig.add_subplot(gs[1, :])
        
        # Create slider
        self.slider = Slider(
            self.ax_slider,
            'Slice',
            0,
            self.n_slices - 1,
            valinit=self.current_slice,
            valstep=1
        )
        self.slider.on_changed(self.update_slice)
        
        # Initial display
        self.update_display()
        
        # Add keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'right' or event.key == 'up':
            new_slice = min(self.current_slice + 1, self.n_slices - 1)
            self.slider.set_val(new_slice)
        elif event.key == 'left' or event.key == 'down':
            new_slice = max(self.current_slice - 1, 0)
            self.slider.set_val(new_slice)
        elif event.key == 'n':  # Next tumor
            idx = np.searchsorted(self.tumor_slices, self.current_slice + 1)
            if idx < len(self.tumor_slices):
                self.slider.set_val(self.tumor_slices[idx])
        elif event.key == 'p':  # Previous tumor
            idx = np.searchsorted(self.tumor_slices, self.current_slice) - 1
            if idx >= 0:
                self.slider.set_val(self.tumor_slices[idx])
    
    def update_slice(self, val):
        """Update displayed slice"""
        self.current_slice = int(val)
        self.update_display()
    
    def update_display(self):
        """Update all three panels"""
        slice_idx = self.current_slice
        ct_slice = self.ct_volume[slice_idx, :, :]
        binary_2d = self.binary_mask[slice_idx, :, :]
        multiclass_2d = self.multiclass_mask[slice_idx, :, :]
        
        # Count lesions in this slice
        n_tumors = np.sum(multiclass_2d == 1)
        n_suspicious = np.sum(multiclass_2d == 2)
        
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # 1. Original CT
        self.ax1.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
        self.ax1.set_title(f'Original CT - Slice {slice_idx}/{self.n_slices-1}',
                          fontsize=14, fontweight='bold')
        self.ax1.axis('off')
        
        # 2. Binary mask overlay
        self.ax2.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
        tumor_overlay = np.zeros((*ct_slice.shape, 4))
        tumor_overlay[binary_2d > 0] = [1, 0, 0, 0.6]
        self.ax2.imshow(tumor_overlay)
        self.ax2.set_title(f'Binary Tumor Mask\n({binary_2d.sum()} pixels)',
                          fontsize=14, fontweight='bold')
        self.ax2.axis('off')
        
        # 3. Multi-class mask overlay
        self.ax3.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
        multiclass_overlay = np.zeros((*ct_slice.shape, 4))
        multiclass_overlay[multiclass_2d == 1] = [1, 0, 0, 0.7]  # Red: tumor
        multiclass_overlay[multiclass_2d == 2] = [1, 1, 0, 0.5]  # Yellow: suspicious
        self.ax3.imshow(multiclass_overlay)
        self.ax3.set_title(
            f'Multi-class Mask\nTumor: {n_tumors}px, Suspicious: {n_suspicious}px',
            fontsize=14, fontweight='bold', color='red' if n_tumors > 0 else 'black'
        )
        self.ax3.axis('off')
        
        # Add navigation hint
        if slice_idx in self.tumor_slices:
            self.fig.suptitle(
                '[TUMOR DETECTED] Controls: Left/Right arrows=slice, N=next tumor, P=prev tumor',
                fontsize=12, fontweight='bold', color='red'
            )
        else:
            self.fig.suptitle(
                'No tumor | Controls: ←/→ arrows=slice, N=next tumor, P=prev tumor',
                fontsize=12
            )
        
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Display the viewer"""
        plt.show()

def main():
    """Main viewer"""
    print("\n" + "="*70)
    print("3D TUMOR MASK VIEWER")
    print("="*70)
    
    # Load files
    ct_path = Path("CTdata/nifti/inha_ct_arterial.nii.gz")
    binary_mask_path = Path("CTdata/tumor_masks/tumor_mask_binary.nii.gz")
    multiclass_mask_path = Path("CTdata/tumor_masks/tumor_mask_multiclass.nii.gz")
    
    print("\nLoading files...")
    ct_volume = load_nifti(ct_path)
    binary_mask = load_nifti(binary_mask_path)
    multiclass_mask = load_nifti(multiclass_mask_path)
    
    print(f"  CT: {ct_volume.shape}")
    print(f"  Binary mask: {binary_mask.shape}")
    print(f"  Multiclass mask: {multiclass_mask.shape}")
    
    # Statistics
    tumor_volume = np.sum(binary_mask > 0)
    tumor_voxels_class1 = np.sum(multiclass_mask == 1)
    tumor_voxels_class2 = np.sum(multiclass_mask == 2)
    
    print(f"\n[TUMOR STATISTICS]:")
    print(f"  Total tumor volume: {tumor_volume:,} voxels")
    print(f"  Potential tumor (class 1): {tumor_voxels_class1:,} voxels")
    print(f"  Suspicious (class 2): {tumor_voxels_class2:,} voxels")
    
    # Create and show viewer
    print("\n[OPENING VIEWER] Starting interactive viewer...")
    print("\n[CONTROLS]:")
    print("  - Left/Right arrows: Navigate slices")
    print("  - N: Jump to next tumor slice")
    print("  - P: Jump to previous tumor slice")
    print("  - Slider: Go to specific slice")
    
    viewer = TumorMaskViewer(ct_volume, binary_mask, multiclass_mask)
    viewer.show()

if __name__ == '__main__':
    main()
