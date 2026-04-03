#!/usr/bin/env python3
"""
MedSAM Annotation Tool
Interactive CT tumor annotation using Streamlit
"""

import streamlit as st
import numpy as np
import pydicom
from pathlib import Path
import json
import cv2
from PIL import Image
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Check if MedSAM is available
try:
    from medical_imaging.medsam_detector import MedSAMDetector
    MEDSAM_AVAILABLE = True
except Exception as e:
    MEDSAM_AVAILABLE = False
    MEDSAM_ERROR = str(e)

st.set_page_config(
    page_title="MedSAM CT Annotation",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 MedSAM CT Tumor Annotation Tool")

# Initialize session state
if 'annotations' not in st.session_state:
    st.session_state.annotations = {}

if 'current_slice' not in st.session_state:
    st.session_state.current_slice = None

if 'detector' not in st.session_state:
    st.session_state.detector = None

# Sidebar
st.sidebar.header("Settings")

# Model initialization
if MEDSAM_AVAILABLE:
    if st.sidebar.button("Initialize MedSAM"):
        with st.spinner("Loading MedSAM model..."):
            try:
                detector = MedSAMDetector(
                    model_type="vit_h",
                    checkpoint_path="models/sam_vit_h_4b8939.pth"
                )
                st.session_state.detector = detector
                st.sidebar.success("✅ MedSAM loaded!")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")
else:
    st.sidebar.error(f"❌ MedSAM not available: {MEDSAM_ERROR}")

# File selection
data_dir = Path("CTdata/CTdcm")
if data_dir.exists():
    dcm_files = sorted(list(data_dir.glob("*.dcm")))
    
    if dcm_files:
        st.sidebar.subheader("Select CT Slice")
        
        selected_file = st.sidebar.selectbox(
            "DICOM file:",
            dcm_files,
            format_func=lambda x: x.name
        )
        
        st.session_state.current_slice = selected_file
    else:
        st.sidebar.warning("No DICOM files found")
else:
    st.sidebar.error(f"Data directory not found: {data_dir}")

# Main area
if st.session_state.current_slice:
    st.subheader(f"Current Slice: {st.session_state.current_slice.name}")
    
    # Load and display CT
    try:
        dcm = pydicom.dcmread(st.session_state.current_slice)
        image = dcm.pixel_array.astype(float)
        
        # Apply HU conversion
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            image = image * dcm.RescaleSlope + dcm.RescaleIntercept
        
        # Window to soft tissue
        window_level = 40
        window_width = 400
        img_min = window_level - window_width / 2
        img_max = window_level + window_width / 2
        
        image = np.clip(image, img_min, img_max)
        image_display = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original CT")
            st.image(image_display, use_container_width=True, clamp=True)
            
            # Point input
            st.write("**Enter tumor center point:**")
            col_x, col_y = st.columns(2)
            with col_x:
                point_x = st.number_input("X coordinate", 0, image.shape[1], image.shape[1]//2)
            with col_y:
                point_y = st.number_input("Y coordinate", 0, image.shape[0], image.shape[0]//2)
            
            if st.button("🎯 Segment with MedSAM"):
                if st.session_state.detector is None:
                    st.error("Please initialize MedSAM first!")
                else:
                    with st.spinner("Running segmentation..."):
                        # Prepare image
                        image_rgb = cv2.cvtColor(image_display, cv2.COLOR_GRAY2RGB)
                        
                        # Predict
                        points = np.array([[point_x, point_y]])
                        masks, scores, logits = st.session_state.detector.predict_with_points(
                            image_rgb,
                            points,
                            labels=np.array([1])
                        )
                        
                        # Store best mask
                        best_idx = np.argmax(scores)
                        best_mask = masks[best_idx]
                        best_score = scores[best_idx]
                        
                        # Save annotation
                        annotation = {
                            'file': st.session_state.current_slice.name,
                            'point': [int(point_x), int(point_y)],
                            'mask': best_mask.tolist(),
                            'score': float(best_score)
                        }
                        
                        st.session_state.annotations[st.session_state.current_slice.name] = annotation
                        
                        st.success(f"✅ Segmented! Score: {best_score:.3f}")
        
        with col2:
            st.subheader("Segmentation Result")
            
            if st.session_state.current_slice.name in st.session_state.annotations:
                ann = st.session_state.annotations[st.session_state.current_slice.name]
                mask = np.array(ann['mask'])
                
                # Create overlay
                overlay = image_display.copy()
                overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
                overlay[mask > 0] = [255, 0, 0]  # Red overlay
                
                # Blend
                result = cv2.addWeighted(
                    cv2.cvtColor(image_display, cv2.COLOR_GRAY2RGB),
                    0.7,
                    overlay,
                    0.3,
                    0
                )
                
                st.image(result, use_container_width=True, clamp=True)
                st.write(f"**Score:** {ann['score']:.3f}")
                st.write(f"**Point:** ({ann['point'][0]}, {ann['point'][1]})")
                
                # Validation buttons
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    if st.button("✅ Correct"):
                        st.session_state.annotations[st.session_state.current_slice.name]['validated'] = True
                        st.success("Marked as correct!")
                with col_v2:
                    if st.button("❌ Incorrect"):
                        if st.session_state.current_slice.name in st.session_state.annotations:
                            del st.session_state.annotations[st.session_state.current_slice.name]
                        st.warning("Annotation removed")
            else:
                st.info("No segmentation yet. Enter point and click 'Segment'")
        
    except Exception as e:
        st.error(f"Error loading CT: {e}")

# Annotation management
st.sidebar.subheader("Annotations")
st.sidebar.write(f"Total: {len(st.session_state.annotations)}")

if st.session_state.annotations:
    if st.sidebar.button("💾 Save Annotations"):
        output_file = "annotations/medsam_annotations.json"
        Path(output_file).parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(st.session_state.annotations, f, indent=2)
        
        st.sidebar.success(f"✅ Saved to {output_file}")
    
    if st.sidebar.button("🗑️ Clear All"):
        st.session_state.annotations = {}
        st.sidebar.warning("All annotations cleared")

# Instructions
with st.expander("📖 Instructions"):
    st.markdown("""
    ### How to use:
    
    1. **Initialize MedSAM** (sidebar)
    2. **Select CT slice** from dropdown
    3. **Click on tumor center** or enter coordinates
    4. **Click "Segment with MedSAM"**
    5. **Review result** and mark as correct/incorrect
    6. **Repeat** for 5-10 slices with clear tumors
    7. **Save annotations** when done
    
    ### Tips:
    - Choose slices where tumor is clearly visible
    - Click near the tumor center
    - If result is wrong, try different point
    - Only 5-10 good annotations needed for fine-tuning
    """)
