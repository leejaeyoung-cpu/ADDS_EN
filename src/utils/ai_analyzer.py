"""
AI-powered analysis insights generator
Uses OpenAI API to generate comprehensive analysis insights from cell analysis results
"""

import os
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AIAnalyzer:
    """AI-powered analyzer for cell analysis insights"""
    
    def __init__(self):
        """Initialize AI Analyzer with OpenAI client"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
    
    def get_statistical_interpretation(self, results: Dict[str, Any]) -> str:
        """
        Generate statistical interpretation of analysis results
        Works without OpenAI API
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Statistical interpretation as markdown string
        """
        seg_metadata = results.get('segmentation_metadata', {})
        metrics = results.get('metrics', {})
        quality = results.get('quality_assessment', {})
        
        num_cells = seg_metadata.get('num_cells', 0)
        mean_area = metrics.get('mean_area', 0)
        std_area = metrics.get('std_area', 0)
        mean_circularity = metrics.get('mean_circularity', 0)
        quality_score = quality.get('overall_score', 0)
        
        # ── Thresholds (documented, not arbitrary) ─────────────────────────────
        # Based on typical in-vitro cell density ranges for confocal microscopy images.
        # These thresholds are for RESEARCH USE ONLY and do not constitute clinical guidance.
        CELL_COUNT_LOW_THRESHOLD  = 100   # cells: below = low density
        CELL_COUNT_HIGH_THRESHOLD = 1000  # cells: above = high density

        interpretation = f"""
### 📊 통계적 해석

#### 세포 수 분석
- **검출된 세포 수**: {num_cells:,}개
- {'✅ 정상 범위 (100–2000개)' if 100 <= num_cells <= 2000 else '⚠️ 비정상 범위 — 실험 조건 재확인 권장'}
- **밀도**: {'높음' if num_cells > CELL_COUNT_HIGH_THRESHOLD else '중간' if num_cells > CELL_COUNT_LOW_THRESHOLD else '낮음'}
  _(기준: 낮음 < {CELL_COUNT_LOW_THRESHOLD} < 중간 < {CELL_COUNT_HIGH_THRESHOLD} < 높음 세포/이미지)_

#### 세포 크기 분석
- **평균 면적**: {mean_area:.1f} px²
- **표준편차**: {std_area:.1f} px²
- **변이계수 (CV)**: {(std_area/mean_area*100):.1f}%
- {'✅ 균일한 크기 분포' if (std_area/mean_area) < 0.3 else '⚠️ 크기 변동성이 큼'}

#### 형태 분석
- **평균 원형도**: {mean_circularity:.3f}
- {'✅ 건강한 형태 (원형에 가까움)' if mean_circularity > 0.7 else '⚠️ 불규칙한 형태'}
- **형태 평가**: {'매우 양호' if mean_circularity > 0.8 else '양호' if mean_circularity > 0.7 else '주의 필요'}

#### 이미지 품질
- **종합 품질 점수**: {quality_score:.2f}/1.0
- **등급**: {quality.get('overall_quality', 'N/A')}

> ⚠️ **연구용 지표**: 이 수치는 현미경 이미지 분석 결과이며, 임상 진단 또는 치료 결정의 근거로 단독 사용할 수 없습니다.
"""
        return interpretation
    
    def get_biological_meaning(self, results: Dict[str, Any]) -> str:
        """
        Generate biological interpretation
        Works without OpenAI API
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Biological interpretation as markdown string
        """
        seg_metadata = results.get('segmentation_metadata', {})
        metrics = results.get('metrics', {})
        
        num_cells = seg_metadata.get('num_cells', 0)
        mean_circularity = metrics.get('mean_circularity', 0)
        mean_area = metrics.get('mean_area', 0)
        
        interpretation = f"""
### 🔬 생물학적 의미

#### 세포 건강도 평가
"""
        
        # Circularity-based health assessment
        if mean_circularity > 0.75:
            interpretation += """
- ✅ **건강한 세포 형태**: 높은 원형도는 세포막 무결성과 정상적인 세포 기능을 시사합니다.
- 세포가 스트레스를 받지 않고 정상적인 형태를 유지하고 있습니다.
"""
        elif mean_circularity > 0.6:
            interpretation += """
- ⚠️ **중간 수준의 형태 변화**: 약간의 형태 변화가 관찰됩니다.
- 실험 조건이나 세포 상태를 재확인할 필요가 있습니다.
"""
        else:
            interpretation += """
- 🔴 **비정상적인 형태**: 낮은 원형도는 세포 스트레스, 손상 또는 변형을 나타낼 수 있습니다.
- 실험 조건, 약물 처리 효과 또는 세포 사멸 과정을 의심해볼 수 있습니다.
"""
        
        # Cell density assessment
        interpretation += f"""

#### 세포 밀도 및 증식
- **관찰된 세포 밀도**: {'높음 (confluence 상태 가능)' if num_cells > 1000 else '중간' if num_cells > 500 else '낮음'}
"""
        
        if num_cells > 1000:
            interpretation += """
- 높은 세포 밀도는 증식이 활발하거나 confluence에 도달했음을 의미할 수 있습니다.
- Contact inhibition으로 인한 성장 억제 고려 필요
"""
        elif num_cells < 200:
            interpretation += """
- 낮은 세포 밀도는 세포 생존율 감소, 약물 독성 또는 부적절한 실험 조건을 시사할 수 있습니다.
"""
        
        interpretation += """

#### 실험적 시사점
- 형태학적 특징은 약물 반응, 세포 건강도, 실험 조건의 적절성을 평가하는 중요한 지표입니다.
- 대조군과의 정량적 비교를 통해 실험 효과를 검증할 수 있습니다.
"""
        
        return interpretation
    
    def get_ai_insights(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Generate AI-powered comprehensive insights using OpenAI
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            AI-generated insights or None if API unavailable
        """
        if not self.client:
            return None
        
        try:
            seg_metadata = results.get('segmentation_metadata', {})
            metrics = results.get('metrics', {})
            quality = results.get('quality_assessment', {})
            
            # Prepare data summary for AI
            data_summary = f"""
세포 분석 결과:
- 세포 수: {seg_metadata.get('num_cells', 0)}개
- 평균 면적: {metrics.get('mean_area', 0):.1f} px²
- 표준편차: {metrics.get('std_area', 0):.1f} px²
- 평균 원형도: {metrics.get('mean_circularity', 0):.3f}
- 품질 점수: {quality.get('overall_score', 0):.2f}
- 품질 등급: {quality.get('overall_quality', 'N/A')}

상세 품질 평가:
{quality.get('detailed_assessment', {})}
"""
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 세포 이미지 분석 전문가입니다. 
주어진 분석 결과를 바탕으로 종합적인 인사이트를 제공하세요.
다음 관점에서 분석하세요:
1. 비정상적인 패턴이나 주목할 만한 특징
2. 실험 결과의 신뢰성 평가
3. 추가 분석이 필요한 부분
4. 실험 개선을 위한 제안

한국어로 답변하며, 전문적이면서도 이해하기 쉽게 설명하세요."""
                    },
                    {
                        "role": "user",
                        "content": f"다음 세포 분석 결과에 대한 종합적인 인사이트를 제공해주세요:\n\n{data_summary}"
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None
    
    def get_key_findings(self, results: Dict[str, Any]) -> str:
        """
        Generate key findings from analysis results
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Key findings as markdown string
        """
        seg_metadata = results.get('segmentation_metadata', {})
        metrics = results.get('metrics', {})
        quality = results.get('quality_assessment', {})
        
        num_cells = seg_metadata.get('num_cells', 0)
        mean_circularity = metrics.get('mean_circularity', 0)
        cv = metrics.get('std_area', 0) / metrics.get('mean_area', 1) if metrics.get('mean_area', 0) > 0 else 0
        quality_score = quality.get('overall_score', 0)
        
        findings = ["### 🎯 주요 발견사항\n"]
        
        # Cell count findings
        if num_cells < 100:
            findings.append("- 🔴 **매우 낮은 세포 수**: 세포 생존율 또는 실험 조건 재검토 필요")
        elif num_cells > 1500:
            findings.append("- ⚠️ **높은 세포 밀도**: Confluence 상태일 수 있으며, contact inhibition 고려 필요")
        else:
            findings.append(f"- ✅ **적정 세포 수**: {num_cells:,}개의 세포가 검출되어 분석에 충분함")
        
        # Morphology findings
        if mean_circularity < 0.6:
            findings.append("- ⚠️ **비정상 형태**: 낮은 원형도는 세포 스트레스나 약물 효과를 시사")
        elif mean_circularity > 0.8:
            findings.append("- ✅ **매우 건강한 형태**: 높은 원형도로 세포 건강 상태 양호")
        
        # Size variation findings
        if cv > 0.4:
            findings.append("- ⚠️ **높은 크기 변동성**: 불균일한 세포 집단 또는 다양한 세포 주기 단계")
        elif cv < 0.2:
            findings.append("- ✅ **균일한 세포 크기**: 동질적인 세포 집단")
        
        # Quality findings
        if quality_score < 0.5:
            findings.append("- 🔴 **낮은 이미지 품질**: 분석 결과의 신뢰성이 떨어질 수 있음")
        elif quality_score > 0.8:
            findings.append("- ✅ **우수한 이미지 품질**: 신뢰할 수 있는 분석 결과")
        
        return "\n".join(findings)
    
    def get_recommendations(self, results: Dict[str, Any]) -> str:
        """
        Generate recommendations based on analysis results
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Recommendations as markdown string
        """
        seg_metadata = results.get('segmentation_metadata', {})
        metrics = results.get('metrics', {})
        quality = results.get('quality_assessment', {})
        
        num_cells = seg_metadata.get('num_cells', 0)
        mean_circularity = metrics.get('mean_circularity', 0)
        quality_score = quality.get('overall_score', 0)
        
        recommendations = ["### 💡 권장사항\n"]
        
        # Cell density recommendations
        if num_cells < 100:
            recommendations.append("""
#### 세포 수 개선
- 초기 seeding 밀도 증가 고려
- 배양 시간 연장 검토
- 세포 생존율 확인 (trypan blue staining 등)
""")
        elif num_cells > 1500:
            recommendations.append("""
#### 세포 밀도 조절
- 초기 seeding 밀도 감소
- 더 이른 시점에서 분석 수행
- Sub-confluent 상태에서 실험 진행 권장
""")
        
        # Morphology recommendations
        if mean_circularity < 0.6:
            recommendations.append("""
#### 세포 형태 개선
- 실험 조건 (온도, CO₂, 배지) 재확인
- 약물 농도 및 처리 시간 최적화
- 세포 계대수 확인 (너무 많은 passage는 형태 변화 유발)
""")
        
        # Quality recommendations
        if quality_score < 0.7:
            recommendations.append("""
#### 이미지 품질 향상
- 현미경 초점 재조정
- 조명 조건 최적화
- 배경 노이즈 감소
- 고해상도 이미징 고려
""")
        
        # General recommendations
        recommendations.append("""
#### 추가 분석 제안
- 여러 시간대에서 time-course 분석 수행
- 생물학적 반복 실험 (n≥3) 수행
- 대조군과의 정량적 비교 분석
- Western blot, flow cytometry 등 보완 실험 고려
""")
        
        return "\n".join(recommendations)


# Convenience function for quick analysis
def generate_comprehensive_insights(results: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate all insights at once
    
    Args:
        results: Analysis results dictionary
        
    Returns:
        Dictionary containing all insight types
    """
    analyzer = AIAnalyzer()
    
    insights = {
        'statistical': analyzer.get_statistical_interpretation(results),
        'biological': analyzer.get_biological_meaning(results),
        'ai_insights': analyzer.get_ai_insights(results),
        'key_findings': analyzer.get_key_findings(results),
        'recommendations': analyzer.get_recommendations(results)
    }
    
    return insights
