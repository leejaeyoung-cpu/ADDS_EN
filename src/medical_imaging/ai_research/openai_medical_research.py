"""
OpenAI-powered Medical Research Module
Provides AI-assisted analysis of CT findings and medical insights
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. Install with: pip install openai")


@dataclass
class ResearchResponse:
    """Structure for research responses"""
    content: str
    model: str
    tokens_used: int
    timestamp: float
    cached: bool = False


class MedicalResearcher:
    """
    AI-powered medical research assistant using OpenAI
    Provides intelligent analysis of CT findings and medical insights
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 2000,
        temperature: float = 0.7
    ):
        """
        Initialize Medical Researcher
        
        Args:
            api_key: OpenAI API key (defaults to env variable)
            model: OpenAI model to use (gpt-4o recommended for medical accuracy)
            max_tokens: Maximum tokens per response
            temperature: Response creativity (0.0-1.0, lower is more factual)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package required. Install with: pip install openai")
        
        # Load API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Response cache
        self._cache: Dict[str, ResearchResponse] = {}
        
        logger.info(f"Medical Researcher initialized with model: {model}")
    
    def _create_prompt(self, system_role: str, user_query: str) -> List[Dict[str, str]]:
        """Create messages for OpenAI chat completion"""
        return [
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_query}
        ]
    
    def _query_openai(
        self,
        messages: List[Dict[str, str]],
        cache_key: Optional[str] = None
    ) -> ResearchResponse:
        """
        Query OpenAI API with caching support
        
        Args:
            messages: Chat messages
            cache_key: Optional cache key for response caching
        
        Returns:
            ResearchResponse object
        """
        # Check cache
        if cache_key and cache_key in self._cache:
            cached_response = self._cache[cache_key]
            cached_response.cached = True
            logger.info(f"Using cached response for: {cache_key[:50]}...")
            return cached_response
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract response
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            result = ResearchResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                timestamp=time.time(),
                cached=False
            )
            
            # Cache result
            if cache_key:
                self._cache[cache_key] = result
            
            logger.info(f"OpenAI response: {tokens_used} tokens used")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def analyze_ct_findings(self, findings: Dict[str, Any]) -> ResearchResponse:
        """
        Analyze CT scan findings and provide medical insights
        
        Args:
            findings: Dictionary containing CT analysis results
                - tumor_count: Number of tumors detected
                - tumor_volume_mm3: Total tumor volume
                - max_diameter_mm: Maximum tumor diameter
                - location: Tumor location
                - confidence_score: Detection confidence
        
        Returns:
            ResearchResponse with analysis
        """
        system_role = """You are an expert medical AI assistant specializing in radiology and oncology.
Analyze CT scan findings and provide:
1. Clinical significance of the findings
2. Potential tumor staging (T-staging based on size)
3. Recommended follow-up actions
4. Important considerations for the radiologist

Be precise, evidence-based, and cite relevant medical guidelines when possible.
IMPORTANT: Always note that this is AI-assisted analysis and requires verification by qualified medical professionals."""

        user_query = f"""Analyze these CT scan findings:

Detected Findings:
- Number of lesions: {findings.get('tumor_count', 'N/A')}
- Total volume: {findings.get('tumor_volume_mm3', 'N/A')} mm³
- Maximum diameter: {findings.get('max_diameter_mm', 'N/A')} mm
- Location: {findings.get('location', 'N/A')}
- Detection confidence: {findings.get('confidence_score', 'N/A')}

Provide a comprehensive analysis of these findings including clinical significance, staging implications, and recommended actions."""

        cache_key = f"ct_analysis_{json.dumps(findings, sort_keys=True)}"
        return self._query_openai(
            self._create_prompt(system_role, user_query),
            cache_key=cache_key
        )
    
    def research_tumor_characteristics(self, tumor_data: Dict[str, Any]) -> ResearchResponse:
        """
        Research tumor characteristics and provide literature-based insights
        
        Args:
            tumor_data: Dictionary with tumor characteristics
                - type: Tumor type (if known)
                - size_mm: Tumor size
                - shape: Tumor morphology
                - density_hu: Hounsfield units
        
        Returns:
            ResearchResponse with research insights
        """
        system_role = """You are a medical research AI specializing in oncology and tumor biology.
Provide evidence-based information about tumor characteristics including:
1. Typical characteristics of this tumor type
2. Differential diagnoses based on imaging features
3. Prognostic factors
4. Current treatment approaches
5. Relevant recent research findings

Cite medical literature and guidelines where applicable."""

        user_query = f"""Research these tumor characteristics:

{json.dumps(tumor_data, indent=2)}

Provide comprehensive information about tumors with these characteristics, including differential diagnoses, prognostic factors, and current treatment approaches."""

        cache_key = f"tumor_research_{json.dumps(tumor_data, sort_keys=True)}"
        return self._query_openai(
            self._create_prompt(system_role, user_query),
            cache_key=cache_key
        )
    
    def explain_medical_terms(self, terms: List[str]) -> ResearchResponse:
        """
        Explain medical terminology in clear, accessible language
        
        Args:
            terms: List of medical terms to explain
        
        Returns:
            ResearchResponse with explanations
        """
        system_role = """You are a medical educator AI.
Explain medical terms clearly and accurately in both technical and layperson's language.
Include:
1. Technical definition
2. Simple explanation
3. Clinical significance
4. Related terms"""

        user_query = f"""Explain these medical terms:

{chr(10).join(f'- {term}' for term in terms)}

Provide clear, accurate explanations suitable for both medical professionals and patients."""

        cache_key = f"terms_{','.join(sorted(terms))}"
        return self._query_openai(
            self._create_prompt(system_role, user_query),
            cache_key=cache_key
        )
    
    def suggest_treatment_insights(self, patient_data: Dict[str, Any]) -> ResearchResponse:
        """
        Provide treatment recommendation context based on patient data
        
        Args:
            patient_data: Anonymized patient data
                - tnm_stage: TNM staging
                - tumor_location: Tumor location
                - tumor_size_mm: Tumor size
                - patient_age: Patient age range
                - comorbidities: List of relevant comorbidities
        
        Returns:
            ResearchResponse with treatment insights
        """
        system_role = """You are an oncology AI assistant.
Based on tumor characteristics and patient factors, provide:
1. Standard treatment options according to NCCN/ASCO guidelines
2. Factors influencing treatment selection
3. Clinical trial considerations
4. Multidisciplinary team discussion points

CRITICAL: This is for educational/research purposes only. 
Always emphasize that actual treatment decisions must be made by qualified oncologists 
based on complete patient evaluation."""

        user_query = f"""Based on these patient characteristics, provide treatment context:

{json.dumps(patient_data, indent=2)}

What are the standard treatment approaches and key considerations for this case?"""

        cache_key = f"treatment_{json.dumps(patient_data, sort_keys=True)}"
        return self._query_openai(
            self._create_prompt(system_role, user_query),
            cache_key=cache_key
        )
    
    def analyze_pathology_features(self, pathology_results: Dict[str, Any]) -> ResearchResponse:
        """
        Analyze pathology image features from Cellpose segmentation
        
        Args:
            pathology_results: Dictionary containing Cellpose analysis results
                - num_cells: Cell count
                - mean_area: Average cell area
                - mean_circularity: Average cell circularity
                - cell_line: Cell line name (e.g., HUVEC)
                - treatment: Treatment applied (e.g., TNF-α)
                - condition: Experimental condition
        
        Returns:
            ResearchResponse with biological interpretation
        """
        system_role = """You are a cell biology and pathology AI expert.
Analyze microscopy image analysis results and provide:
1. Biological significance of cell morphology changes
2. Interpretation in context of experimental treatment
3. Comparison with expected normal cell behavior
4. Suggested follow-up experiments or measurements
5. Relevant research literature on similar observations

Focus on translating quantitative metrics into biological insights."""

        user_query = f"""Analyze these cell analysis results:

{json.dumps(pathology_results, indent=2)}

Provide biological interpretation of these findings, especially in the context of the experimental treatment."""

        cache_key = f"pathology_{json.dumps(pathology_results, sort_keys=True)}"
        return self._query_openai(
            self._create_prompt(system_role, user_query),
            cache_key=cache_key
        )
    
    def compare_multiple_findings(self, findings_list: List[Dict[str, Any]], comparison_type: str = "temporal") -> ResearchResponse:
        """
        Compare multiple CT/pathology findings over time or across conditions
        
        Args:
            findings_list: List of analysis results to compare
            comparison_type: "temporal" (time course) or "treatment" (different conditions)
        
        Returns:
            ResearchResponse with comparative analysis
        """
        system_role = """You are a medical imaging and research AI specialist.
Compare multiple imaging or pathology results and provide:
1. Trends and patterns across the dataset
2. Statistical significance of observed changes
3. Clinical or biological interpretation of progression/regression
4. Prognostic implications
5. Recommendations for monitoring or intervention

Be quantitative where possible and highlight clinically meaningful changes."""

        comparison_context = "over time" if comparison_type == "temporal" else "across treatment conditions"
        
        user_query = f"""Compare these {len(findings_list)} analysis results {comparison_context}:

{json.dumps(findings_list, indent=2)}

Provide a comprehensive comparative analysis highlighting key trends and their significance."""

        cache_key = f"compare_{comparison_type}_{json.dumps(findings_list, sort_keys=True)}"
        return self._query_openai(
            self._create_prompt(system_role, user_query),
            cache_key=cache_key
        )
    
    def generate_patient_report(self, medical_findings: Dict[str, Any]) -> ResearchResponse:
        """
        Generate patient-friendly report explaining medical findings
        
        Args:
            medical_findings: CT/MRI/pathology analysis results
        
        Returns:
            ResearchResponse with patient-friendly explanation
        """
        system_role = """You are a medical communication AI assistant.
Translate complex medical findings into clear, compassionate language for patients.

Guidelines:
1. Use simple, non-technical language
2. Explain medical terms when necessary
3. Be honest but reassuring in tone
4. Emphasize next steps and what patients should expect
5. Always include disclaimer that this is educational information

Avoid alarming language while maintaining medical accuracy."""

        user_query = f"""Create a patient-friendly explanation of these medical findings:

{json.dumps(medical_findings, indent=2)}

Write in a clear, compassionate way that helps patients understand their results without causing unnecessary anxiety."""

        cache_key = f"patient_report_{json.dumps(medical_findings, sort_keys=True)}"
        return self._query_openai(
            self._create_prompt(system_role, user_query),
            cache_key=cache_key
        )
    
    def analyze_medical_image(self, image_type: str, analysis_results: Dict[str, Any]) -> ResearchResponse:
        """
        Unified medical image analysis for pathology/CT/MRI
        
        Args:
            image_type: "pathology", "ct", or "mri"
            analysis_results: Analysis results from respective analyzer
        
        Returns:
            ResearchResponse with appropriate analysis
        """
        if image_type.lower() == "pathology":
            return self.analyze_pathology_features(analysis_results)
        elif image_type.lower() == "ct":
            return self.analyze_ct_findings(analysis_results)
        elif image_type.lower() == "mri":
            # Future: MRI-specific analysis
            return self.analyze_ct_findings(analysis_results)  # Fallback to CT for now
        else:
            raise ValueError(f"Unsupported image type: {image_type}. Use 'pathology', 'ct', or 'mri'.")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cached_responses': len(self._cache),
            'total_tokens_saved': sum(r.tokens_used for r in self._cache.values())
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self._cache.clear()
        logger.info("Response cache cleared")


# Convenience functions for direct use

def analyze_ct_findings(findings: Dict[str, Any], api_key: Optional[str] = None) -> str:
    """
    Convenience function to analyze CT findings
    
    Args:
        findings: CT scan findings dictionary
        api_key: Optional OpenAI API key
    
    Returns:
        Analysis text
    """
    try:
        researcher = MedicalResearcher(api_key=api_key)
        response = researcher.analyze_ct_findings(findings)
        return response.content
    except Exception as e:
        logger.error(f"Error analyzing CT findings: {e}")
        return f"⚠️ AI 분석을 사용할 수 없습니다: {str(e)}"


def research_tumor_characteristics(tumor_data: Dict[str, Any], api_key: Optional[str] = None) -> str:
    """
    Convenience function to research tumor characteristics
    
    Args:
        tumor_data: Tumor characteristics
        api_key: Optional OpenAI API key
    
    Returns:
        Research text
    """
    try:
        researcher = MedicalResearcher(api_key=api_key)
        response = researcher.research_tumor_characteristics(tumor_data)
        return response.content
    except Exception as e:
        logger.error(f"Error researching tumor: {e}")
        return f"⚠️ AI 리서치를 사용할 수 없습니다: {str(e)}"


def explain_medical_terms(terms: List[str], api_key: Optional[str] = None) -> str:
    """
    Convenience function to explain medical terms
    
    Args:
        terms: List of medical terms
        api_key: Optional OpenAI API key
    
    Returns:
        Explanations text
    """
    try:
        researcher = MedicalResearcher(api_key=api_key)
        response = researcher.explain_medical_terms(terms)
        return response.content
    except Exception as e:
        logger.error(f"Error explaining terms: {e}")
        return f"⚠️ AI 설명을 사용할 수 없습니다: {str(e)}"


def suggest_treatment_insights(patient_data: Dict[str, Any], api_key: Optional[str] = None) -> str:
    """
    Convenience function to get treatment insights
    
    Args:
        patient_data: Patient characteristics
        api_key: Optional OpenAI API key
    
    Returns:
        Treatment insights text
    """
    try:
        researcher = MedicalResearcher(api_key=api_key)
        response = researcher.suggest_treatment_insights(patient_data)
        return response.content
    except Exception as e:
        logger.error(f"Error getting treatment insights: {e}")
        return f"⚠️ AI 치료 인사이트를 사용할 수 없습니다: {str(e)}"


# Test
if __name__ == "__main__":
    print("Testing Medical Researcher...")
    
    # Test CT findings analysis
    test_findings = {
        'tumor_count': 2,
        'tumor_volume_mm3': 1250.5,
        'max_diameter_mm': 15.2,
        'location': 'Right colon',
        'confidence_score': 0.92
    }
    
    try:
        result = analyze_ct_findings(test_findings)
        print("\n=== CT Findings Analysis ===")
        print(result)
        print("\n✓ Medical Researcher test passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("Make sure OPENAI_API_KEY is set in environment")
