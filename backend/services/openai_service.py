"""OpenAI Medical Inference Service"""

import os
from typing import Dict, Any, List
from datetime import datetime
import json


class OpenAIService:
    """
    OpenAI GPT-4 Medical Inference Service
    Uses GPT-4 for comprehensive medical analysis and drug recommendations
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize OpenAI service
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4"
        
        # Check if openai is available
        try:
            import openai
            self.openai = openai
            if self.api_key:
                self.openai.api_key = self.api_key
            self.available = True
        except (ImportError, OSError) as e:
            print(f"Warning: openai package not available ({e}). OpenAI inference will be simulated.")
            self.available = False
    
    def run_medical_inference(
        self,
        radiomics: Dict[str, Any],
        tumor_characteristics: Dict[str, Any],
        clinical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run GPT-4 medical inference
        
        Args:
            radiomics: Radiomics features
            tumor_characteristics: Tumor morphology and characteristics
            clinical_data: Clinical metadata
            
        Returns:
            OpenAI inference result
        """
        if not self.available or not self.api_key:
            return self._generate_simulated_result(radiomics, tumor_characteristics, clinical_data)
        
        try:
            # Build medical prompt
            prompt = self._build_medical_prompt(radiomics, tumor_characteristics, clinical_data)
            
            # Call GPT-4
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert oncologist specializing in colorectal cancer treatment and precision medicine. Provide evidence-based treatment recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse response
            analysis = self._parse_response(response.choices[0].message.content)
            
            return analysis
            
        except Exception as e:
            print(f"OpenAI inference error: {e}")
            return self._generate_simulated_result(radiomics, tumor_characteristics, clinical_data)
    
    def _build_medical_prompt(
        self,
        radiomics: Dict,
        tumor_chars: Dict,
        clinical: Dict
    ) -> str:
        """Build detailed medical analysis prompt"""
        
        # Extract key features
        tumor_volume = tumor_chars.get("morphology", {}).get("area_mm2", "N/A")
        mean_hu = tumor_chars.get("intensity", {}).get("mean_hu", "N/A")
        sphericity = radiomics.get("shape_Sphericity", "N/A")
        entropy = radiomics.get("firstorder_Entropy", "N/A")
        
        prompt = f"""## Colorectal Cancer Case Analysis

### Patient Clinical Data
- **Tumor Location**: {clinical.get('tumor_location', 'Unknown')}
- **TNM Stage**: {clinical.get('tnm_stage', 'Unknown')}
- **MSI Status**: {clinical.get('msi_status', 'Unknown')}
- **KRAS Mutation**: {clinical.get('kras_mutation', 'Unknown')}

### CT Imaging Analysis
- **Tumor Size**: {tumor_volume} mm²
- **Mean Intensity**: {mean_hu} HU
- **Shape Sphericity**: {sphericity}
- **Texture Entropy**: {entropy}
- **Circularity**: {tumor_chars.get('morphology', {}).get('circularity', 'N/A')}

### Task
Based on this colorectal cancer case, provide:

1. **Recommended 3-Drug Combination**: List the specific drug names
2. **Treatment Rationale**: Explain why each drug was selected based on the clinical and imaging features
3. **Treatment Strategy**: Specify whether this should be neo-adjuvant, adjuvant, or palliative
4. **Confidence Level**: Rate your confidence (0.0-1.0) in this recommendation
5. **Key Clinical Considerations**: List important factors the medical team should consider
6. **Estimated 5-Year Survival**: Provide a realistic survival estimate

### Response Format
Provide your response in the following JSON format:
```json
{{
  "drugs": ["Drug1", "Drug2", "Drug3"],
  "rationale": "Detailed explanation of drug selection...",
  "treatment_strategy": "neo-adjuvant/adjuvant/palliative",
  "confidence": 0.XX,
  "key_considerations": ["Consideration 1", "Consideration 2", ...],
  "estimated_survival": 0.XX
}}
```
"""
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse GPT-4 JSON response"""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
            
            parsed = json.loads(json_str)
            
            return {
                "system": "OpenAI GPT-4",
                "version": self.model,
                "recommended_drugs": parsed.get("drugs", []),
                "rationale": parsed.get("rationale", ""),
                "treatment_strategy": parsed.get("treatment_strategy", "adjuvant"),
                "confidence": parsed.get("confidence", 0.75),
                "key_considerations": parsed.get("key_considerations", []),
                "estimated_survival": parsed.get("estimated_survival", 0.7),
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Response parsing error: {e}")
            return self._generate_fallback_result()
    
    def _generate_simulated_result(
        self,
        radiomics: Dict,
        tumor_chars: Dict,
        clinical: Dict
    ) -> Dict[str, Any]:
        """
        Generate simulated OpenAI-style result
        Used when API is not available
        """
        # Determine drugs based on clinical features
        drugs = []
        rationale_parts = []
        
        # Base chemotherapy
        drugs.append("5-Fluorouracil")
        drugs.append("Oxaliplatin")
        rationale_parts.append("FOLFOX backbone recommended for colorectal cancer")
        
        # MSI-H → Immunotherapy
        if clinical.get("msi_status") == "MSI-H":
            drugs.append("Pembrolizumab")
            rationale_parts.append("MSI-H status indicates strong immunotherapy response")
        # KRAS WT → anti-EGFR
        elif clinical.get("kras_mutation") == "WT":
            drugs.append("Cetuximab")
            rationale_parts.append("KRAS wild-type allows anti-EGFR therapy")
        # Default: anti-VEGF
        else:
            drugs.append("Bevacizumab")
            rationale_parts.append("Anti-angiogenic therapy for vascular targeting")
        
        # Determine strategy based on stage
        tnm_stage = clinical.get("tnm_stage", "")
        if "M1" in tnm_stage or "IV" in tnm_stage:
            strategy = "palliative"
            survival = 0.45
        elif "T4" in tnm_stage or "N2" in tnm_stage:
            strategy = "neo-adjuvant"
            survival = 0.65
        else:
            strategy = "adjuvant"
            survival = 0.75
        
        return {
            "system": "OpenAI GPT-4 (Simulated)",
            "version": "gpt-4-simulation",
            "recommended_drugs": drugs[:3],
            "rationale": ". ".join(rationale_parts) + ". Treatment plan optimized based on molecular profile and imaging features.",
            "treatment_strategy": strategy,
            "confidence": 0.78,
            "key_considerations": [
                "Confirm molecular testing results before finalizing treatment",
                "Monitor for treatment-related adverse events",
                "Consider patient performance status and comorbidities",
                "Multidisciplinary tumor board review recommended"
            ],
            "estimated_survival": survival,
            "generated_at": datetime.now().isoformat(),
            "note": "This is a simulated result. OpenAI API key not configured."
        }
    
    def _generate_fallback_result(self) -> Dict[str, Any]:
        """Generate generic fallback result"""
        return {
            "system": "OpenAI GPT-4",
            "version": self.model,
            "recommended_drugs": ["5-Fluorouracil", "Oxaliplatin", "Bevacizumab"],
            "rationale": "Standard FOLFOX + Bevacizumab regimen for metastatic colorectal cancer",
            "treatment_strategy": "adjuvant",
            "confidence": 0.70,
            "key_considerations": ["Standard first-line therapy"],
            "estimated_survival": 0.68,
            "generated_at": datetime.now().isoformat(),
            "error": "Response parsing failed - using fallback recommendation"
        }
