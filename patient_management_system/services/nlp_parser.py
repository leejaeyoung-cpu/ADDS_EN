"""
Physician Notes NLP Parser

Extracts structured information from clinical notes using NLP techniques
to enable automated re-analysis triggers and metadata enrichment.
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PhysicianNotesParser:
    """Parse physician notes to extract structured information"""
    
    # Keywords for severity assessment
    SEVERITY_KEYWORDS = {
        'critical': ['critical', 'emergency', 'urgent', 'severe', 'acute'],
        'moderate': ['moderate', 'concerning', 'worsening', 'progressive'],
        'mild': ['mild', 'stable', 'improved', 'improving', 'controlled']
    }
    
    # Symptoms and conditions
    SYMPTOM_KEYWORDS = [
        'pain', 'nausea', 'vomiting', 'fatigue', 'fever', 'bleeding',
        'shortness of breath', 'dyspnea', 'cough', 'weight loss',
        'diarrhea', 'constipation', 'anemia', 'infection'
    ]
    
    # Tumor-related keywords
    TUMOR_KEYWORDS = {
        'growth': ['growing', 'enlarged', 'increased', 'expansion', 'progression'],
        'reduction': ['reduced', 'decreased', 'shrinking', 'regression', 'responding'],
        'stable': ['stable', 'unchanged', 'no change', 'static']
    }
    
    def __init__(self):
        """Initialize parser"""
        self.logger = logger
    
    def parse(self, clinical_text: str) -> Dict[str, Any]:
        """
        Parse clinical notes and extract structured information
        
        Args:
            clinical_text: Raw physician notes
            
        Returns:
            Dictionary with extracted information
        """
        if not clinical_text:
            return {'error': 'Empty text'}
        
        results = {
            'parsed_at': datetime.now().isoformat(),
            'severity': self._extract_severity(clinical_text),
            'symptoms': self._extract_symptoms(clinical_text),
            'tumor_status': self._extract_tumor_status(clinical_text),
            'requires_reanalysis': self._should_trigger_reanalysis(clinical_text),
            'key_findings': self._extract_key_findings(clinical_text),
            'medications': self._extract_medications(clinical_text)
        }
        
        return results
    
    def _extract_severity(self, text: str) -> Dict[str, Any]:
        """Extract severity level from text"""
        text_lower = text.lower()
        
        severity_scores = {
            'critical': 0,
            'moderate': 0,
            'mild': 0
        }
        
        for severity, keywords in self.SEVERITY_KEYWORDS.items():
            for keyword in keywords:
                severity_scores[severity] += text_lower.count(keyword)
        
        # Determine primary severity
        max_severity = max(severity_scores.items(), key=lambda x: x[1])
        
        return {
            'level': max_severity[0] if max_severity[1] > 0 else 'unknown',
            'score': self._severity_to_numeric(max_severity[0]),
            'keywords_found': severity_scores
        }
    
    def _severity_to_numeric(self, severity: str) -> int:
        """Convert severity level to numeric score (1-10)"""
        mapping = {
            'mild': 3,
            'moderate': 6,
            'critical': 9,
            'unknown': 5
        }
        return mapping.get(severity, 5)
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract mentioned symptoms"""
        text_lower = text.lower()
        found_symptoms = []
        
        for symptom in self.SYMPTOM_KEYWORDS:
            if symptom in text_lower:
                found_symptoms.append(symptom)
        
        return found_symptoms
    
    def _extract_tumor_status(self, text: str) -> Dict[str, Any]:
        """Extract tumor status indicators"""
        text_lower = text.lower()
        
        status_scores = {
            'growth': 0,
            'reduction': 0,
            'stable': 0
        }
        
        for status, keywords in self.TUMOR_KEYWORDS.items():
            for keyword in keywords:
                status_scores[status] += text_lower.count(keyword)
        
        # Determine primary status
        max_status = max(status_scores.items(), key=lambda x: x[1])
        
        return {
            'status': max_status[0] if max_status[1] > 0 else 'unknown',
            'confidence': min(max_status[1] / 3, 1.0),  # Normalize to 0-1
            'indicators': status_scores
        }
    
    def _should_trigger_reanalysis(self, text: str) -> bool:
        """
        Determine if notes indicate need for re-analysis
        
        Triggers re-analysis if:
        - Tumor growth mentioned
        - New symptoms
        - Treatment change mentioned
        - Severity is critical
        """
        text_lower = text.lower()
        
        # Check for growth keywords
        for keyword in self.TUMOR_KEYWORDS['growth']:
            if keyword in text_lower:
                return True
        
        # Check for critical severity
        for keyword in self.SEVERITY_KEYWORDS['critical']:
            if keyword in text_lower:
                return True
        
        # Check for treatment change
        treatment_change_keywords = [
            'change treatment', 'new regimen', 'adjust dose',
            'discontinue', 'switch to', 'add medication'
        ]
        
        for keyword in treatment_change_keywords:
            if keyword in text_lower:
                return True
        
        return False
    
    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key clinical findings from text"""
        findings = []
        
        # Look for sentences with key indicators
        sentences = re.split(r'[.!?]', text)
        
        key_indicators = [
            'finding', 'noted', 'observed', 'assessment', 'impression',
            'diagnosis', 'conclusion', 'result'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(indicator in sentence_lower for indicator in key_indicators):
                if len(sentence.strip()) > 10:  # Filter out very short sentences
                    findings.append(sentence.strip())
        
        return findings[:5]  # Limit to top 5 findings
    
    def _extract_medications(self, text: str) -> List[str]:
        """Extract medication mentions (basic pattern matching)"""
        medications = []
        
        # Common chemotherapy drugs
        chemo_drugs = [
            '5-FU', '5-Fluorouracil', 'Oxaliplatin', 'Leucovorin', 'Irinotecan',
            'Capecitabine', 'Bevacizumab', 'Cetuximab', 'Panitumumab',
            'FOLFOX', 'FOLFIRI', 'CAPOX', 'Cisplatin', 'Carboplatin'
        ]
        
        text_lower = text.lower()
        
        for drug in chemo_drugs:
            if drug.lower() in text_lower:
                medications.append(drug)
        
        return list(set(medications))  # Remove duplicates
    
    def extract_tumor_measurements(self, text: str) -> Optional[Dict[str, float]]:
        """
        Extract tumor size measurements from text
        
        Looks for patterns like:
        - "tumor measures 3.5 cm"
        - "lesion size: 45mm"
        - "mass 2.1 x 3.4 cm"
        """
        measurements = {}
        
        # Pattern for single measurements
        pattern_single = r'(\d+\.?\d*)\s*(mm|cm)'
        matches = re.findall(pattern_single, text.lower())
        
        if matches:
            value, unit = matches[0]
            value = float(value)
            
            # Convert to mm
            if unit == 'cm':
                value *= 10
            
            measurements['size_mm'] = value
        
        # Pattern for dimensions (e.g., "2.1 x 3.4 cm")
        pattern_dimensions = r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*(mm|cm)'
        dim_matches = re.findall(pattern_dimensions, text.lower())
        
        if dim_matches:
            dim1, dim2, unit = dim_matches[0]
            dim1, dim2 = float(dim1), float(dim2)
            
            if unit == 'cm':
                dim1 *= 10
                dim2 *= 10
            
            measurements['dimension1_mm'] = dim1
            measurements['dimension2_mm'] = dim2
            measurements['max_diameter_mm'] = max(dim1, dim2)
        
        return measurements if measurements else None


if __name__ == "__main__":
    # Test the parser
    parser = PhysicianNotesParser()
    
    test_note = """
    Patient presents with moderate abdominal pain and fatigue. 
    CT scan shows tumor has reduced from 4.5 cm to 3.2 cm. 
    Patient responding well to FOLFOX regimen.
    Assessment: Partial response to treatment.
    """
    
    results = parser.parse(test_note)
    print("Parser Test Results:")
    print(f"Severity: {results['severity']}")
    print(f"Symptoms: {results['symptoms']}")
    print(f"Tumor Status: {results['tumor_status']}")
    print(f"Requires Re-analysis: {results['requires_reanalysis']}")
