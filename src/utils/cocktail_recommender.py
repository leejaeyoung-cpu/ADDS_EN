import os
import re
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv(encoding='utf-8')

logger = logging.getLogger(__name__)

# Validated drug names (FDA-approved anticancer agents)
VALIDATED_DRUG_NAMES = {
    "5-FU", "5-Fluorouracil", "Fluorouracil", "Capecitabine",
    "Oxaliplatin", "Irinotecan", "Cisplatin", "Carboplatin",
    "Doxorubicin", "Paclitaxel", "Docetaxel", "Gemcitabine",
    "Bevacizumab", "Cetuximab", "Panitumumab", "Pembrolizumab",
    "Nivolumab", "Ipilimumab", "Atezolizumab", "Aflibercept",
    "Regorafenib", "Trifluridine", "Encorafenib", "Binimetinib",
    "Ramucirumab", "Methotrexate", "Vincristine", "Etoposide",
    "Bortezomib", "nab-Paclitaxel"
}


class CocktailRecommender:
    """AI-powered drug cocktail recommendation system"""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
            self.has_api = True
        else:
            self.client = None
            self.has_api = False
    
    def recommend_cocktail(
        self,
        cancer_type: str,
        patient_info: Dict[str, Any],
        available_drugs: List[str],
        num_recommendations: int = 3
    ) -> Dict[str, Any]:
        """
        Generate AI-powered drug cocktail recommendations
        
        Args:
            cancer_type: Type of cancer (e.g., "Colorectal", "Breast")
            patient_info: Dictionary with patient data (age, sex, ECOG, etc.)
            available_drugs: List of available drug names
            num_recommendations: Number of combinations to recommend
            
        Returns:
            Dictionary with recommendations and rationale
        """
        if not self.has_api:
            return self._fallback_recommendation(cancer_type, available_drugs)
        
        # Construct prompt
        prompt = self._build_recommendation_prompt(
            cancer_type, patient_info, available_drugs, num_recommendations
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "You are an expert oncologist specializing in combination chemotherapy regimens. "
                        "Provide evidence-based drug cocktail recommendations. "
                        "IMPORTANT: Only recommend drugs from the provided available drugs list. "
                        "Do NOT recommend drugs outside this list. "
                        "Do NOT invent drug names. "
                        "Always include evidence levels and cite relevant clinical trials."
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            recommendation_text = response.choices[0].message.content
            
            # Validate GPT output
            validation = self._validate_gpt_output(recommendation_text, available_drugs)
            
            return {
                "success": True,
                "recommendations": recommendation_text,
                "model": "gpt-4o-mini",
                "source": "AI-generated",
                "validation": validation,
                "disclaimer": (
                    "⚠️ AI 생성 권장사항: 이 추천은 AI 모델에 의해 생성되었으며, "
                    "임상 결정의 참고 자료로만 사용해야 합니다. "
                    "실제 처방은 반드시 자격을 갖춘 종양 전문의의 판단에 따라야 합니다."
                )
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "recommendations": self._fallback_recommendation(cancer_type, available_drugs)
            }
    
    def _validate_gpt_output(
        self,
        recommendation_text: str,
        available_drugs: List[str]
    ) -> Dict[str, Any]:
        """
        Validate GPT output for drug name accuracy and safety
        
        Checks:
        1. Drug names mentioned are in available_drugs or VALIDATED_DRUG_NAMES
        2. Flags any unknown/potentially hallucinated drug names
        """
        warnings = []
        
        # Extract potential drug names (capitalized words that could be drug names)
        # Simple heuristic: words that appear near dose-like patterns
        text_upper = recommendation_text.upper()
        
        # Check available drugs are mentioned
        mentioned_drugs = []
        for drug in available_drugs:
            if drug.upper() in text_upper:
                mentioned_drugs.append(drug)
        
        # Check for drugs mentioned but not in available list
        all_known = {d.upper() for d in VALIDATED_DRUG_NAMES} | {d.upper() for d in available_drugs}
        
        # Look for capitalized drug-like names with dosage patterns
        dose_pattern = re.findall(
            r'(\b[A-Z][a-z]+(?:-[A-Z][a-z]+)*\b)\s*(?:\d+\s*(?:mg|mg/m²|mg/kg))',
            recommendation_text
        )
        for potential_drug in dose_pattern:
            if potential_drug.upper() not in all_known:
                warnings.append(
                    f"⚠️ Unknown drug with dosage: '{potential_drug}' — not in validated drug list"
                )
                logger.warning(f"GPT recommended unknown drug: {potential_drug}")
        
        return {
            "mentioned_drugs": mentioned_drugs,
            "warnings": warnings,
            "is_valid": len(warnings) == 0
        }
    
    def _build_recommendation_prompt(
        self,
        cancer_type: str,
        patient_info: Dict[str, Any],
        available_drugs: List[str],
        num_recommendations: int
    ) -> str:
        """Build detailed prompt for AI recommendation"""
        
        prompt = f"""As an expert oncologist, recommend the TOP {num_recommendations} drug cocktail combinations for a patient with {cancer_type} cancer.

**Patient Information:**
- Age: {patient_info.get('age', 'Unknown')}
- Sex: {patient_info.get('sex', 'Unknown')}
- ECOG Performance Status: {patient_info.get('ecog', 'Unknown')}
- Stage: {patient_info.get('stage', 'Unknown')}
- Previous Treatment: {patient_info.get('previous_treatment', 'None')}

**Available Drugs:**
{', '.join(available_drugs)}

**Please provide:**

1. **Top {num_recommendations} Recommended Combinations** (ranked by expected efficacy)
   - For each combination, specify:
     - Drug names and doses
     - Expected synergy mechanism
     - Evidence level (High/Medium/Low)

2. **Rationale for Each Combination**
   - Why this combination is optimal
   - Expected response rate
   - Potential toxicity concerns

3. **Treatment Monitoring Recommendations**
   - Key biomarkers to monitor
   - Expected timeline for response

Format your response clearly with numbered sections."""
        
        return prompt
    
    def _fallback_recommendation(
        self,
        cancer_type: str,
        available_drugs: List[str]
    ) -> str:
        """Provide rule-based recommendations when API unavailable"""
        
        # Simple rule-based recommendations
        recommendations= {
            "Colorectal": ["5-FU + Oxaliplatin (FOLFOX)", "5-FU + Irinotecan + Bevacizumab"],
            "Breast": ["Doxorubicin + Paclitaxel", "Doxorubicin + Cisplatin"],
            "Lung": ["Cisplatin + Paclitaxel", "Gemcitabine + Cisplatin"],
            "Pancreatic": ["Gemcitabine + nab-Paclitaxel", "FOLFIRINOX (5-FU + Oxaliplatin + Irinotecan)"]
        }
        
        suggestions = recommendations.get(cancer_type, ["No specific recommendations available"])
        
        return f"""
**Rule-Based Recommendations for {cancer_type} Cancer:**

{chr(10).join(f"{i+1}. {rec}" for i, rec in enumerate(suggestions))}

**Note:** These are general guidelines. For personalized AI recommendations, please configure your OpenAI API key.

**Evidence:**
- Based on NCCN guidelines and standard treatment protocols
- Consult with oncology team for patient-specific considerations
"""
    
    def explain_synergy(
        self,
        drug1: str,
        drug2: str,
        mechanisms: Dict[str, str]
    ) -> str:
        """
        Explain synergy mechanism between two drugs
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            mechanisms: Dictionary of drug mechanisms
            
        Returns:
            Explanation of synergistic mechanism
        """
        if not self.has_api:
            return self._fallback_synergy_explanation(drug1, drug2)
        
        prompt = f"""Explain the synergistic mechanism between {drug1} and {drug2} in cancer treatment.

**{drug1} mechanism:** {mechanisms.get(drug1, 'Unknown')}
**{drug2} mechanism:** {mechanisms.get(drug2, 'Unknown')}

Provide a concise, scientific explanation of how these drugs work together synergistically."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert pharmacologist explaining drug synergy mechanisms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content
        
        except:
            return self._fallback_synergy_explanation(drug1, drug2)
    
    def _fallback_synergy_explanation(self, drug1: str, drug2: str) -> str:
        """Fallback synergy explanation"""
        return f"{drug1} and {drug2} may work synergistically by targeting different cellular pathways, potentially overcoming resistance and enhancing therapeutic efficacy."
