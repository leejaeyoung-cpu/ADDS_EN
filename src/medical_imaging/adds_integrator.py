"""
Stage 6: ADDS Pharmacokinetic System Integration
Converts CT analysis results to ADDS input format for drug recommendation
"""

import json
from typing import Dict, Optional, List
from pathlib import Path
import logging
import requests

logger = logging.getLogger(__name__)


class ADDSIntegrator:
    """
    Integrates CT-based colorectal cancer detection with ADDS
    pharmacokinetic optimization system
    """
    
    def __init__(self, adds_api_url: Optional[str] = None, 
                 api_token: Optional[str] = None):
        """
        Initialize ADDS integrator
        
        Args:
            adds_api_url: ADDS API endpoint
            api_token: Authentication token
        """
        self.adds_api_url = adds_api_url or "http://localhost:8000/api/v1"
        self.api_token = api_token
        
        # Drug sensitivity predictors
        self.drug_models = {}
    
    def prepare_adds_input(self, patient_id: str,
                          volume: 'np.ndarray',
                          tumor_analysis: Dict,
                          radiomics: Dict,
                          classification: Dict) -> Dict:
        """
        Prepare ADDS input JSON from CT analysis
        
        Args:
            patient_id: Patient identifier
            volume: 3D CT volume
            tumor_analysis: Tumor detection results
            radiomics: Radiomics features
            classification: Classification and staging results
            
        Returns:
            adds_input: ADDS-compatible input dictionary
        """
        # Build ADDS input structure
        adds_input = {
            'patient_id': patient_id,
            'analysis_source': 'CT_Automated_Detection',
            'analysis_timestamp': self._get_timestamp(),
            
            # Tumor information
            'tumor_info': {
                'primary_site': 'Colorectal',
                'histology': 'Adenocarcinoma',  # Most common CRC type
                'location': self._determine_colon_location(tumor_analysis),
                'stage': classification.get('overall_stage', 'Unknown'),
                'tnm': classification.get('tnm_stage', {})
            },
            
            # Tumor characteristics
            'tumor_characteristics': {
                'volume_cm3': tumor_analysis.get('volume_mm3', 0) / 1000,
                'diameter_cm': self._estimate_diameter(tumor_analysis),
                'sphericity': radiomics.get('original_shape_Sphericity', 0.5),
                'heterogeneity_index': radiomics.get('original_firstorder_Entropy', 0),
                'texture_contrast': radiomics.get('original_glcm_Contrast', 0),
                'surface_volume_ratio': radiomics.get('original_shape_SurfaceVolumeRatio', 0),
                
                # Vascularization estimate
                'vascularity_estimate': self._estimate_vascularity(radiomics)
            },
            
            # Drug sensitivity predictions
            'predicted_drug_sensitivity': {
                'FOLFOX': self._predict_folfox_sensitivity(radiomics, classification),
                'FOLFIRI': self._predict_folfiri_sensitivity(radiomics, classification),
                'bevacizumab': self._predict_bevacizumab_sensitivity(radiomics, classification),
                'cetuximab': self._predict_cetuximab_sensitivity(radiomics, classification),
                'pembrolizumab': self._predict_pembrolizumab_sensitivity(radiomics, classification)
            },
            
            # Imaging biomarkers
            'imaging_biomarkers': {
                'msi_status_predicted': classification.get('msi_status', {}),
                'kras_likelihood': classification.get('kras_mutation', {}),
                'radiomics_based_prognosis': self._estimate_prognosis(radiomics, classification)
            },
            
            # Quality metrics
            'analysis_quality': {
                'segmentation_confidence': tumor_analysis.get('confidence', 0.0),
                'classification_confidence': classification.get('confidence', 0.0)
            }
        }
        
        return adds_input
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _determine_colon_location(self, tumor_analysis: Dict) -> str:
        """
        Determine tumor location in colon
        
        Args:
            tumor_analysis: Tumor metadata with centroid
            
        Returns:
            location: Colon location (Right/Left/Rectum)
        """
        # Simplified: would need anatomical mapping
        # For now, use Z-coordinate as proxy
        centroid = tumor_analysis.get('centroid', [0, 0, 0])
        z_pos = centroid[0] if len(centroid) > 0 else 0
        
        if z_pos < 100:
            return 'Rectum'
        elif z_pos < 200:
            return 'Left Colon'
        else:
            return 'Right Colon'
    
    def _estimate_diameter(self, tumor_analysis: Dict) -> float:
        """
        Estimate tumor diameter from volume
        
        Args:
            tumor_analysis: Contains volume
            
        Returns:
            diameter_cm: Estimated diameter
        """
        volume_mm3 = tumor_analysis.get('volume_mm3', 0)
        # Assume sphere: V = 4/3 * pi * r^3
        radius_mm = (3 * volume_mm3 / (4 * 3.14159)) ** (1/3)
        diameter_cm = (2 * radius_mm) / 10
        return diameter_cm
    
    def _estimate_vascularity(self, radiomics: Dict) -> float:
        """
        Estimate tumor vascularity from radiomics
        
        High vascularity indicators:
        - High entropy (heterogeneous enhancement)
        - High contrast
        
        Args:
            radiomics: Radiomics features
            
        Returns:
            vascularity_score: 0-1
        """
        entropy = radiomics.get('original_firstorder_Entropy', 0)
        contrast = radiomics.get('original_glcm_Contrast', 0)
        
        # Normalize
        entropy_norm = min(entropy / 5.0, 1.0)
        contrast_norm = min(contrast / 200.0, 1.0)
        
        vascularity = (entropy_norm + contrast_norm) / 2
        return vascularity
    
    def _predict_folfox_sensitivity(self, radiomics: Dict, classification: Dict) -> Dict:
        """
        Predict FOLFOX (5-FU + Oxaliplatin) sensitivity
        
        Args:
            radiomics: Radiomics features
            classification: Tumor classification
            
        Returns:
            sensitivity: Predicted sensitivity
        """
        # FOLFOX is standard first-line for CRC
        # Sensitivity factors:
        # - Stage (lower stage = better response)
        # - Homogeneity (homogeneous tumors respond better)
        
        stage = classification.get('overall_stage')
        entropy = radiomics.get('original_firstorder_Entropy', 4.0)
        
        base_sensitivity = 0.6  # Default for CRC
        
        # Stage adjustment (check if stage is not None)
        if stage:
            if 'I' in stage or 'II' in stage:
                base_sensitivity += 0.2
            elif 'IV' in stage:
                base_sensitivity -= 0.1
        
        # Homogeneity bonus
        if entropy < 3.5:
            base_sensitivity += 0.1
        
        sensitivity = min(max(base_sensitivity, 0.0), 1.0)
        
        return {
            'predicted_response_rate': sensitivity,
            'confidence': 0.6,
            'rationale': 'Standard first-line therapy for CRC'
        }
    
    def _predict_folfiri_sensitivity(self, radiomics: Dict, classification: Dict) -> Dict:
        """Predict FOLFIRI (5-FU + Irinotecan) sensitivity"""
        # Similar to FOLFOX, often used as second-line
        sensitivity = 0.55
        
        return {
            'predicted_response_rate': sensitivity,
            'confidence': 0.5,
            'rationale': 'Second-line therapy option'
        }
    
    def _predict_bevacizumab_sensitivity(self, radiomics: Dict, classification: Dict) -> Dict:
        """
        Predict Bevacizumab (anti-VEGF) sensitivity
        
        Bevacizumab targets angiogenesis
        High vascularity = better response
        """
        vascularity = self._estimate_vascularity(radiomics)
        
        # Vascular tumors respond better
        sensitivity = 0.4 + (vascularity * 0.4)
        
        rationale = "High vascularity detected" if vascularity > 0.6 else "Moderate vascularity"
        
        return {
            'predicted_response_rate': sensitivity,
            'confidence': 0.7,
            'rationale': rationale
        }
    
    def _predict_cetuximab_sensitivity(self, radiomics: Dict, classification: Dict) -> Dict:
        """
        Predict Cetuximab (anti-EGFR) sensitivity
        
        Requires KRAS wild-type (not mutated)
        """
        kras = classification.get('kras_mutation', {})
        kras_status = kras.get('status', 'Unknown')
        
        if kras_status == 'Wild-type':
            sensitivity = 0.6
            rationale = "KRAS wild-type predicted"
        else:
            sensitivity = 0.2
            rationale = "KRAS status uncertain or mutated"
        
        return {
            'predicted_response_rate': sensitivity,
            'confidence': 0.4,
            'rationale': rationale
        }
    
    def _predict_pembrolizumab_sensitivity(self, radiomics: Dict, classification: Dict) -> Dict:
        """
        Predict Pembrolizumab (anti-PD-1) sensitivity
        
        MSI-H tumors respond very well to immunotherapy
        """
        msi = classification.get('msi_status', {})
        msi_status = msi.get('status', 'MSS')
        
        if msi_status == 'MSI-H':
            sensitivity = 0.85  # Very high response rate
            rationale = "MSI-H status predicted - excellent immunotherapy candidate"
        else:
            sensitivity = 0.15  # Low response in MSS tumors
            rationale = "MSS/MSI-L predicted - limited immunotherapy benefit"
        
        return {
            'predicted_response_rate': sensitivity,
            'confidence': msi.get('confidence', 0.5),
            'rationale': rationale
        }
    
    def _estimate_prognosis(self, radiomics: Dict, classification: Dict) -> Dict:
        """
        Estimate prognosis based on radiomics and stage
        
        Args:
            radiomics: Radiomics features
            classification: Tumor classification
            
        Returns:
            prognosis: Prognosis estimate
        """
        stage = classification.get('overall_stage', 'Unknown')
        
        # 5-year survival rates (approximate)
        survival_rates = {
            'Stage I': 0.90,
            'Stage II': 0.75,
            'Stage III': 0.55,
            'Stage IV': 0.12
        }
        
        base_survival = survival_rates.get(stage, 0.50)
        
        # Adjust based on tumor characteristics
        sphericity = radiomics.get('original_shape_Sphericity', 0.5)
        entropy = radiomics.get('original_firstorder_Entropy', 4.0)
        
        # More regular tumors = better prognosis
        if sphericity > 0.7:
            base_survival += 0.05
        elif sphericity < 0.4:
            base_survival -= 0.05
        
        # Less heterogeneous = better prognosis
        if entropy < 3.5:
            base_survival += 0.05
        elif entropy > 5.0:
            base_survival -= 0.05
        
        survival = min(max(base_survival, 0.0), 1.0)
        
        return {
            'estimated_5year_survival': survival,
            'risk_category': self._categorize_risk(survival),
            'confidence': 0.6
        }
    
    def _categorize_risk(self, survival: float) -> str:
        """Categorize risk based on survival"""
        if survival > 0.75:
            return 'Low Risk'
        elif survival > 0.50:
            return 'Moderate Risk'
        else:
            return 'High Risk'
    
    def send_to_adds(self, adds_input: Dict) -> Dict:
        """
        Send analysis to ADDS API
        
        Args:
            adds_input: ADDS input dictionary
            
        Returns:
            treatment_plan: ADDS response with treatment recommendations
        """
        endpoint = f"{self.adds_api_url}/pharmacokinetics/analyze"
        
        headers = {}
        if self.api_token:
            headers['Authorization'] = f'Bearer {self.api_token}'
        
        try:
            response = requests.post(
                endpoint,
                json=adds_input,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                treatment_plan = response.json()
                logger.info("Successfully retrieved treatment plan from ADDS")
                return treatment_plan
            else:
                logger.error(f"ADDS API error: {response.status_code} - {response.text}")
                return self._generate_fallback_plan(adds_input)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to ADDS API: {e}")
            return self._generate_fallback_plan(adds_input)
    
    def _generate_fallback_plan(self, adds_input: Dict) -> Dict:
        """
        Generate fallback treatment plan when ADDS API is unavailable
        
        Args:
            adds_input: Original ADDS input
            
        Returns:
            fallback_plan: Basic treatment recommendations
        """
        stage = adds_input['tumor_info']['stage']
        sensitivities = adds_input['predicted_drug_sensitivity']
        
        # Select drugs with highest predicted sensitivity
        ranked_drugs = sorted(
            sensitivities.items(),
            key=lambda x: x[1]['predicted_response_rate'],
            reverse=True
        )
        
        primary_drug = ranked_drugs[0] if ranked_drugs else ('FOLFOX', {})
        
        fallback_plan = {
            'patient_id': adds_input['patient_id'],
            'note': 'Fallback plan - ADDS API unavailable',
            'recommended_regimen': {
                'primary_drugs': [{
                    'name': primary_drug[0],
                    'predicted_response_rate': primary_drug[1].get('predicted_response_rate', 0.6),
                    'rationale': primary_drug[1].get('rationale', 'Standard therapy')
                }],
                'targeted_therapy': [],
                'immunotherapy': []
            },
            'stage': stage
        }
        
        logger.info("Generated fallback treatment plan")
        return fallback_plan
    
    def save_integration_result(self, adds_input: Dict, treatment_plan: Dict, 
                               output_path: Path):
        """
        Save integration result to file
        
        Args:
            adds_input: ADDS input
            treatment_plan: ADDS response
            output_path: Output JSON file
        """
        result = {
            'adds_input': adds_input,
            'treatment_plan': treatment_plan,
            'timestamp': self._get_timestamp()
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Integration result saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Sample data
    tumor_analysis = {
        'volume_mm3': 5000,
        'centroid': [150, 200, 220],
        'confidence': 0.92
    }
    
    radiomics = {
        'original_shape_Sphericity': 0.55,
        'original_shape_SurfaceVolumeRatio': 0.32,
        'original_firstorder_Entropy': 4.5,
        'original_glcm_Contrast': 165
    }
    
    classification = {
        'classification': 'Malignant',
        'malignancy_probability': 0.88,
        'overall_stage': 'Stage III',
        'tnm_stage': {'T': 'T3', 'N': 'N1', 'M': 'MX'},
        'msi_status': {'status': 'MSI-H', 'confidence': 0.75},
        'confidence': 0.85
    }
    
    # Integrate with ADDS
    integrator = ADDSIntegrator()
    adds_input = integrator.prepare_adds_input(
        patient_id='PT-12345',
        volume=None,
        tumor_analysis=tumor_analysis,
        radiomics=radiomics,
        classification=classification
    )
    
    print("\n✓ ADDS Input prepared:")
    print(f"  Stage: {adds_input['tumor_info']['stage']}")
    print(f"  Top drug: FOLFOX (sensitivity: {adds_input['predicted_drug_sensitivity']['FOLFOX']['predicted_response_rate']:.2f})")
    print(f"  Immunotherapy: Pembrolizumab (sensitivity: {adds_input['predicted_drug_sensitivity']['pembrolizumab']['predicted_response_rate']:.2f})")
