"""
AI Platform Comparator for ADDS
Compares Cellpose analysis results with OpenAI GPT-4 Vision for validation and hyperparameter optimization
"""

import os
import base64
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from openai import OpenAI
from dotenv import load_dotenv

from utils import get_logger

load_dotenv()
logger = get_logger(__name__)


class AIPlatformComparator:
    """
    AI 플랫폼 비교 분석기
    Cellpose 결과를 GPT-4 Vision과 비교하여 정확도 검증 및 하이퍼파라미터 최적화
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize AI Platform Comparator
        
        Args:
            openai_api_key: OpenAI API key (optional, defaults to env variable)
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized for platform comparison")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("No OpenAI API key available - comparison features disabled")
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available"""
        return self.client is not None
    
    def analyze_with_gpt4v(
        self, 
        image_path: Path,
        cellpose_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze image using GPT-4 Vision
        
        Args:
            image_path: Path to the image file
            cellpose_results: Cellpose analysis results for reference
            
        Returns:
            Dictionary with GPT-4V analysis results
        """
        if not self.client:
            return {'error': 'OpenAI API not available'}
        
        try:
            # Encode image to base64
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Prepare Cellpose summary for context
            cellpose_summary = f"""
Cellpose Analysis Results (for reference):
- Cell count: {cellpose_results.get('num_cells', 0)}
- Mean area: {cellpose_results.get('mean_area', 0):.2f} px²
- Mean circularity: {cellpose_results.get('mean_circularity', 0):.3f}
"""
            
            # Create prompt
            prompt = f"""You are a biomedical image analysis expert. Analyze this cell microscopy image.

{cellpose_summary}

Please provide your own independent analysis with:
1. Your estimate of the number of cells visible (be as accurate as possible)
2. Description of cell characteristics (shape, size uniformity, health indicators)
3. Quality assessment of the image
4. Any anomalies or concerns you notice
5. Your confidence level (0-100%)

Format your response as:
CELL_COUNT: [number]
CHARACTERISTICS: [brief description]
QUALITY: [assessment]
ANOMALIES: [any issues or "None"]
CONFIDENCE: [0-100]%
NOTES: [additional observations]
"""
            
            logger.info(f"Sending image to GPT-4 Vision: {image_path.name}")
            
            # Call GPT-4 Vision API
            response = self.client.chat.completions.create(
                model="gpt-4o",  # GPT-4o has vision capabilities
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3  # Lower temperature for more consistent analysis
            )
            
            # Extract response
            ai_text = response.choices[0].message.content
            logger.info("GPT-4 Vision analysis completed")
            
            # Parse structured response
            parsed = self._parse_gpt4v_response(ai_text)
            parsed['raw_response'] = ai_text
            parsed['model_used'] = 'gpt-4o'
            
            return parsed
        
        except Exception as e:
            logger.error(f"GPT-4 Vision analysis failed: {e}")
            return {'error': str(e)}
    
    def _parse_gpt4v_response(self, response_text: str) -> Dict[str, Any]:
        """Parse GPT-4 Vision response into structured data"""
        result = {
            'cell_count_estimate': 0,
            'cell_characteristics': '',
            'quality_assessment': '',
            'anomalies': [],
            'confidence': 0.0,
            'notes': ''
        }
        
        # Extract cell count
        cell_count_match = re.search(r'CELL_COUNT:\s*(\d+)', response_text, re.IGNORECASE)
        if cell_count_match:
            result['cell_count_estimate'] = int(cell_count_match.group(1))
        
        # Extract characteristics
        char_match = re.search(r'CHARACTERISTICS:\s*(.+?)(?=\n[A-Z]+:|$)', response_text, re.IGNORECASE | re.DOTALL)
        if char_match:
            result['cell_characteristics'] = char_match.group(1).strip()
        
        # Extract quality
        quality_match = re.search(r'QUALITY:\s*(.+?)(?=\n[A-Z]+:|$)', response_text, re.IGNORECASE | re.DOTALL)
        if quality_match:
            result['quality_assessment'] = quality_match.group(1).strip()
        
        # Extract anomalies
        anomaly_match = re.search(r'ANOMALIES:\s*(.+?)(?=\n[A-Z]+:|$)', response_text, re.IGNORECASE | re.DOTALL)
        if anomaly_match:
            anomaly_text = anomaly_match.group(1).strip()
            if anomaly_text.lower() != 'none':
                result['anomalies'] = [anomaly_text]
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response_text, re.IGNORECASE)
        if conf_match:
            result['confidence'] = float(conf_match.group(1)) / 100.0
        
        # Extract notes
        notes_match = re.search(r'NOTES:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)
        if notes_match:
            result['notes'] = notes_match.group(1).strip()
        
        return result
    
    def compare_results(
        self,
        cellpose_results: Dict[str, Any],
        gpt4v_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare Cellpose and GPT-4V results
        
        Args:
            cellpose_results: Cellpose analysis results
            gpt4v_results: GPT-4 Vision analysis results
            
        Returns:
            Comparison analysis with agreement scores and discrepancies
        """
        if 'error' in gpt4v_results:
            return {'error': gpt4v_results['error'], 'comparison_available': False}
        
        cellpose_count = cellpose_results.get('num_cells', 0)
        gpt4v_count = gpt4v_results.get('cell_count_estimate', 0)
        
        # Calculate difference and accuracy
        difference = abs(cellpose_count - gpt4v_count)
        max_count = max(cellpose_count, gpt4v_count, 1)
        accuracy = 1.0 - (difference / max_count)
        
        # Analyze discrepancies
        discrepancies = []
        
        if difference > max_count * 0.2:  # > 20% difference
            discrepancies.append(f"Large cell count difference: {difference} cells ({difference/max_count*100:.1f}%)")
        
        if difference > 50:
            discrepancies.append(f"Absolute difference exceeds 50 cells")
        
        # Determine agreement level
        if accuracy >= 0.9:
            agreement_level = "Excellent"
        elif accuracy >= 0.8:
            agreement_level = "Good"
        elif accuracy >= 0.7:
            agreement_level = "Fair"
        else:
            agreement_level = "Poor"
        
        comparison = {
            'comparison_available': True,
            'cell_count_comparison': {
                'cellpose': cellpose_count,
                'gpt4v': gpt4v_count,
                'difference': difference,
                'difference_percent': (difference / max_count * 100) if max_count > 0 else 0,
                'accuracy_estimate': accuracy
            },
            'agreement_level': agreement_level,
            'agreement_score': accuracy,
            'discrepancies': discrepancies,
            'gpt4v_confidence': gpt4v_results.get('confidence', 0.0),
            'gpt4v_characteristics': gpt4v_results.get('cell_characteristics', ''),
            'gpt4v_quality': gpt4v_results.get('quality_assessment', ''),
            'gpt4v_anomalies': gpt4v_results.get('anomalies', []),
            'gpt4v_notes': gpt4v_results.get('notes', ''),
            'gpt4v_raw': gpt4v_results.get('raw_response', '')
        }
        
        logger.info(f"Comparison complete: {agreement_level} agreement ({accuracy*100:.1f}%)")
        
        return comparison
    
    def generate_hyperparameter_recommendations(
        self,
        comparison_results: Dict[str, Any],
        current_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate hyperparameter optimization recommendations
        
        Args:
            comparison_results: Results from compare_results()
            current_params: Current Cellpose parameters
            
        Returns:
            Dictionary with recommendations and overall assessment
        """
        if 'error' in comparison_results or not comparison_results.get('comparison_available'):
            return {
                'recommendations': [],
                'overall_assessment': 'Comparison not available - cannot generate recommendations'
            }
        
        recommendations = []
        
        accuracy = comparison_results.get('agreement_score', 1.0)
        cell_comp = comparison_results.get('cell_count_comparison', {})
        difference = cell_comp.get('difference', 0)
        cellpose_count = cell_comp.get('cellpose', 0)
        gpt4v_count = cell_comp.get('gpt4v', 0)
        
        # Diameter recommendations (if significant difference)
        if difference > 10 and cellpose_count > 0 and gpt4v_count > 0:
            current_diameter = current_params.get('diameter', 30)
            
            if cellpose_count > gpt4v_count * 1.2:
                # Over-segmentation: increase diameter
                suggested_diameter = min(200, current_diameter + 5)
                recommendations.append({
                    'parameter': 'diameter',
                    'current_value': current_diameter,
                    'suggested_value': suggested_diameter,
                    'reason': f'Cellpose detected {cellpose_count - gpt4v_count} more cells than GPT-4V. May be over-segmenting. Increase diameter to merge small fragments.',
                    'priority': 'high' if difference > 50 else 'medium'
                })
            
            elif gpt4v_count > cellpose_count * 1.2:
                # Under-segmentation: decrease diameter
                suggested_diameter = max(10, current_diameter - 5)
                recommendations.append({
                    'parameter': 'diameter',
                    'current_value': current_diameter,
                    'suggested_value': suggested_diameter,
                    'reason': f'GPT-4V detected {gpt4v_count - cellpose_count} more cells than Cellpose. May be under-segmenting. Decrease diameter for finer detection.',
                    'priority': 'high' if difference > 50 else 'medium'
                })
        
        # Flow threshold recommendations (if low agreement)
        if accuracy < 0.8 and difference > 20:
            current_flow = current_params.get('flow_threshold', 0.4)
            
            if cellpose_count > gpt4v_count:
                # Too many cells detected - increase threshold
                recommendations.append({
                    'parameter': 'flow_threshold',
                    'current_value': current_flow,
                    'suggested_value': min(3.0, current_flow + 0.1),
                    'reason': 'Low agreement with AI (possible over-segmentation). Increase flow_threshold for stricter cell boundaries.',
                    'priority': 'medium'
                })
            else:
                # Too few cells detected - decrease threshold
                recommendations.append({
                    'parameter': 'flow_threshold',
                    'current_value': current_flow,
                    'suggested_value': max(0.0, current_flow - 0.1),
                    'reason': 'Low agreement with AI (possible under-segmentation). Decrease flow_threshold for more lenient boundaries.',
                    'priority': 'medium'
                })
        
        # Cellprob threshold recommendations (if very low agreement)
        if accuracy < 0.7:
            current_cellprob = current_params.get('cellprob_threshold', 0.0)
            recommendations.append({
                'parameter': 'cellprob_threshold',
                'current_value': current_cellprob,
                'suggested_value': min(6.0, current_cellprob + 0.5),
                'reason': 'Very low agreement score. Increase cellprob_threshold to filter out low-confidence detections.',
                'priority': 'low'
            })
        
        # Overall assessment
        if accuracy >= 0.9:
            assessment = f"Excellent performance (Agreement: {accuracy*100:.1f}%). Current parameters are well-tuned."
        elif accuracy >= 0.8:
            assessment = f"Good performance (Agreement: {accuracy*100:.1f}%). Minor adjustments may improve accuracy."
        elif accuracy >= 0.7:
            assessment = f"Fair performance (Agreement: {accuracy*100:.1f}%). Parameter tuning recommended."
        else:
            assessment = f"Poor agreement (Agreement: {accuracy*100:.1f}%). Significant parameter adjustment needed."
        
        # Add GPT-4V insights
        if comparison_results.get('gpt4v_anomalies'):
            assessment += f"\n\nGPT-4V detected anomalies: {', '.join(comparison_results['gpt4v_anomalies'])}"
        
        result = {
            'recommendations': recommendations,
            'overall_assessment': assessment,
            'agreement_score': accuracy,
            'total_recommendations': len(recommendations)
        }
        
        logger.info(f"Generated {len(recommendations)} hyperparameter recommendations")
        
        return result


# Convenience function
def compare_with_ai(
    image_path: Path,
    cellpose_results: Dict[str, Any],
    current_params: Dict[str, Any],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete AI comparison workflow
    
    Args:
        image_path: Path to analyzed image
        cellpose_results: Cellpose analysis results
        current_params: Current Cellpose parameters
        api_key: Optional OpenAI API key
        
    Returns:
        Complete comparison with recommendations
    """
    comparator = AIPlatformComparator(api_key)
    
    if not comparator.is_available():
        return {
            'error': 'OpenAI API not available',
            'comparison_available': False
        }
    
    # Step 1: Analyze with GPT-4V
    gpt4v_results = comparator.analyze_with_gpt4v(image_path, cellpose_results)
    
    # Step 2: Compare results
    comparison = comparator.compare_results(cellpose_results, gpt4v_results)
    
    # Step 3: Generate recommendations
    recommendations = comparator.generate_hyperparameter_recommendations(
        comparison,
        current_params
    )
    
    return {
        'gpt4v_analysis': gpt4v_results,
        'comparison': comparison,
        'recommendations': recommendations
    }
