"""
Segmentation Service
Business logic for Cellpose segmentation
"""

import sys
from pathlib import Path
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from preprocessing.image_processor import CellposeProcessor

class SegmentationService:
    """Service for cell segmentation operations"""
    
    def __init__(self):
        self.processor = CellposeProcessor()
    
    async def segment(
        self,
        image_path: str,
        diameter: float = None,
        flow_threshold: float = 0.6,
        cellprob_threshold: float = -1.0,
        batch_size: int = 8
    ):
        """
        Perform cell segmentation
        
        Returns:
            SegmentationResponse dict
        """
        
        # Load image
        image = self.processor._load_image(image_path)
        
        # Segment
        masks, flows, metadata = self.processor.segment_image(
            image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )
        
        # Generate ID
        image_id = f"img_{uuid.uuid4().hex[:8]}"
        
        return {
            "image_id": image_id,
            "cell_count": int(masks.max()),
            "masks_shape": list(masks.shape),
            "metadata": {
                "diameter_used": float(metadata.get('diameter', diameter or 0)),
                "flow_threshold": flow_threshold,
                "cellprob_threshold": cellprob_threshold
            }
        }
