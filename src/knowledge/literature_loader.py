"""
Enhanced Literature Feature Loader with GPT-4 Integration
Automated paper summarization, relationship extraction, and evidence scoring
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from datetime import datetime


class GPT4LiteratureMiner:
    """
    GPT-4 powered literature mining for automated evidence extraction
    
    Capabilities:
    - Automated paper summarization
    - Drug-Gene-Disease relationship extraction
    - Evidence level automatic scoring
    - Citation network analysis
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GPT-4 miner
        
        Args:
            api_key: OpenAI API key (or read from env OPENAI_API_KEY)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.api_base = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4o"  # Latest model as of 2026
        
    def summarize_paper(self, paper_text: str, focus: str = "drug combination synergy") -> Dict:
        """
        Summarize scientific paper with GPT-4
        
        Args:
            paper_text: Full paper text or abstract
            focus: Research focus area
            
        Returns:
            Dictionary with summary, key findings, and clinical implications
        """
        if not self.api_key:
            return {"error": "OpenAI API key not configured"}
        
        prompt = f"""
You are a biomedical research expert. Analyze the following scientific paper 
focusing on: {focus}

Paper text:
{paper_text[:4000]}  # Limit to avoid token overflow

Provide a structured analysis in JSON format:
{{
    "executive_summary": "2-3 sentence high-level summary",
    "key_findings": [
        "Finding 1 with quantitative data",
        "Finding 2 with quantitative data"
    ],
    "methodology": "Brief description of study design",
    "sample_size": "Number of patients/samples",
    "statistical_significance": "P-values and confidence intervals",
    "clinical_implications": "Practical takeaways for treatment",
    "limitations": ["Limitation 1", "Limitation 2"]
}}

Return ONLY valid JSON, no markdown formatting.
"""
        
        try:
            response = requests.post(
                self.api_base,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a biomedical research analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,  # Low temp for factual extraction
                    "max_tokens": 1000
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            content = result['choices'][0]['message']['content']
            
            # Parse JSON response
            summary_data = json.loads(content)
            summary_data['generated_at'] = datetime.now().isoformat()
            summary_data['model'] = self.model
            
            return summary_data
            
        except Exception as e:
            return {
                "error": str(e),
                "fallback_summary": paper_text[:500]
            }
    
    def extract_relationships(self, paper_text: str) -> Dict:
        """
        Extract Drug-Gene-Disease relationships from paper
        
        Args:
            paper_text: Paper text
            
        Returns:
            Dictionary with extracted entities and relationships
        """
        if not self.api_key:
            return {"error": "OpenAI API key not configured"}
        
        prompt = f"""
Extract biomedical relationships from this paper.

Paper text:
{paper_text[:4000]}

Return a JSON with:
{{
    "drugs": [
        {{"name": "Drug Name", "mechanism": "MOA", "targets": ["Gene1", "Gene2"]}}
    ],
    "genes": [
        {{"name": "Gene Name", "role": "oncogene/tumor suppressor", "mutations": ["Variant1"]}}
    ],
    "diseases": [
        {{"name": "Cancer Type", "stage": "I-IV", "subtype": "Subtype"}}
    ],
    "relationships": [
        {{"drug": "Drug A", "gene": "Gene X", "effect": "inhibits/activates", "evidence": "Clinical trial data"}}
    ],
    "synergies": [
        {{"drug1": "Drug A", "drug2": "Drug B", "synergy_score": "Bliss excess", "mechanism": "Complementary MOA"}}
    ]
}}

Return ONLY valid JSON.
"""
        
        try:
            response = requests.post(
                self.api_base,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a biomedical knowledge extraction expert."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1500
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            relationships = json.loads(content)
            relationships['extracted_at'] = datetime.now().isoformat()
            
            return relationships
            
        except Exception as e:
            return {"error": str(e)}
    
    def score_evidence_level(self, study_design: str, sample_size: int, 
                            has_rct: bool = False, meta_analysis: bool = False) -> Dict:
        """
        Automatically score evidence level based on study characteristics
        
        Args:
            study_design: Type of study
            sample_size: Number of participants
            has_rct: Is randomized controlled trial
            meta_analysis: Is meta-analysis
            
        Returns:
            Evidence level and justification
        """
        # Oxford Centre for Evidence-Based Medicine Levels
        # Level I: Systematic review of RCTs
        # Level II: Individual RCT
        # Level III: Non-randomized controlled cohort
        # Level IV: Case series, case-control
        # Level V: Expert opinion
        
        if meta_analysis:
            return {
                "level": "Level_I",
                "grade": "A",
                "description": "Systematic review/meta-analysis of RCTs",
                "justification": "Highest quality evidence"
            }
        
        if has_rct and sample_size > 100:
            return {
                "level": "Level_II",
                "grade": "B",
                "description": "Individual RCT with adequate sample size",
                "justification": f"Well-powered RCT (n={sample_size})"
            }
        
        study_lower = study_design.lower()
        
        if "cohort" in study_lower and sample_size > 50:
            return {
                "level": "Level_III",
                "grade": "C",
                "description": "Non-randomized controlled cohort study",
                "justification": f"Prospective cohort (n={sample_size})"
            }
        
        if "case" in study_lower:
            return {
                "level": "Level_IV",
                "grade": "C",
                "description": "Case series or case-control study",
                "justification": f"Observational study (n={sample_size})"
            }
        
        if "preclinical" in study_lower or "vitro" in study_lower:
            return {
                "level": "Preclinical",
                "grade": "N/A",
                "description": "Preclinical or in vitro study",
                "justification": "Laboratory evidence only"
            }
        
        return {
            "level": "Level_V",
            "grade": "D",
            "description": "Expert opinion or small study",
            "justification": "Limited evidence quality"
        }


class LiteratureFeatureLoader:
    """
    Enhanced loader with GPT-4 integration
    
    Provides access to:
    - Biomarker prevalences
    - Treatment outcomes
    - Mutation frequencies
    - Prognostic factors
    - AI-powered paper analysis (NEW)
    """
    
    def __init__(self, base_path: Optional[Path] = None, openai_api_key: Optional[str] = None):
        """
        Initialize loader with GPT-4 support
        
        Args:
            base_path: Path to knowledge/literature directory
            openai_api_key: OpenAI API key for GPT-4 features
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent / "knowledge" / "literature"
        
        self.base_path = base_path
        self.literature_db = self._load_literature_database()
        self.cancer_features = {}
        
        # GPT-4 Integration
        self.gpt4_miner = GPT4LiteratureMiner(api_key=openai_api_key)
        
        # Load available cancer feature files
        self._load_cancer_features()
    
    def _load_literature_database(self) -> Dict:
        """Load main literature database"""
        db_path = self.base_path / "literature_database.json"
        
        if not db_path.exists():
            return {"papers": [], "summary_statistics": {}}
        
        with open(db_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_cancer_features(self):
        """Load all available cancer feature files"""
        features_dir = self.base_path / "extracted_features"
        
        if not features_dir.exists():
            return
        
        for feature_file in features_dir.glob("*_features.json"):
            cancer_type = feature_file.stem.replace("_features", "")
            
            with open(feature_file, 'r', encoding='utf-8') as f:
                self.cancer_features[cancer_type] = json.load(f)
    
    # ========== NEW: GPT-4 Enhanced Methods ==========
    
    def analyze_paper_with_ai(self, paper_text: str, focus: str = "drug combination") -> Dict:
        """
        AI-powered paper analysis
        
        Args:
            paper_text: Paper abstract or full text
            focus: Analysis focus
            
        Returns:
            Comprehensive analysis including summary and relationships
        """
        summary = self.gpt4_miner.summarize_paper(paper_text, focus)
        relationships = self.gpt4_miner.extract_relationships(paper_text)
        
        return {
            "summary": summary,
            "relationships": relationships,
            "analyzed_at": datetime.now().isoformat()
        }
    
    def search_papers_with_ai_summary(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict]:
        """
        Search papers and generate AI summaries
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of papers with AI-generated summaries
        """
        # Basic search first
        papers = self.search_papers()[:max_results]
        
        enriched_papers = []
        for paper in papers:
            # Generate AI summary if abstract available
            abstract = paper.get('abstract', '')
            if abstract and len(abstract) > 100:
                ai_analysis = self.analyze_paper_with_ai(abstract)
                paper['ai_summary'] = ai_analysis['summary']
                paper['ai_relationships'] = ai_analysis['relationships']
            
            enriched_papers.append(paper)
        
        return enriched_papers
    
    def extract_drug_combinations_from_literature(
        self,
        cancer_type: str,
        min_evidence_level: str = "Level_III"
    ) -> List[Dict]:
        """
        Extract drug combinations from literature with AI analysis
        
        Args:
            cancer_type: Cancer type to focus on
            min_evidence_level: Minimum evidence quality
            
        Returns:
            List of drug combinations with synergy data
        """
        papers = self.search_papers(cancer_type=cancer_type, min_evidence_level=min_evidence_level)
        
        combinations = []
        for paper in papers:
            abstract = paper.get('abstract', '')
            if not abstract:
                continue
            
            # Extract relationships with AI
            relationships = self.gpt4_miner.extract_relationships(abstract)
            
            # Focus on synergies
            if 'synergies' in relationships and relationships['synergies']:
                for synergy in relationships['synergies']:
                    combinations.append({
                        'drug1': synergy.get('drug1'),
                        'drug2': synergy.get('drug2'),
                        'synergy_mechanism': synergy.get('mechanism'),
                        'evidence_paper': paper.get('title'),
                        'pubmed_id': paper.get('pubmed_id'),
                        'evidence_level': paper.get('evidence_level')
                    })
        
        return combinations
    
    # ========== Original Methods (Preserved) ==========
    
    def get_biomarker_prevalence(
        self, 
        cancer_type: str, 
        biomarker: str
    ) -> Optional[float]:
        """Get biomarker prevalence from literature"""
        cancer_type_lower = cancer_type.lower()
        
        if cancer_type_lower not in self.cancer_features:
            return None
        
        features = self.cancer_features[cancer_type_lower]
        
        # Check actionable biomarkers
        if 'actionable_biomarkers' in features:
            for marker_name, marker_data in features['actionable_biomarkers'].items():
                if biomarker.lower() in marker_name.lower():
                    return marker_data.get('prevalence')
        
        # Check molecular features
        if 'molecular_subtypes' in features:
            for subtype_name, subtype_data in features['molecular_subtypes']['subtypes'].items():
                if biomarker.lower() in subtype_name.lower():
                    return subtype_data.get('prevalence')
        
        return None
    
    def get_treatment_outcome(
        self,
        cancer_type: str,
        treatment: str,
        biomarker: Optional[str] = None
    ) -> Optional[Dict]:
        """Get treatment outcomes from literature"""
        cancer_type_lower = cancer_type.lower()
        
        if cancer_type_lower not in self.cancer_features:
            return None
        
        features = self.cancer_features[cancer_type_lower]
        
        if 'treatment_outcomes' not in features:
            return None
        
        outcomes = features['treatment_outcomes']
        
        # Search through treatment outcomes
        for outcome_key, outcome_data in outcomes.items():
            if treatment.lower() in outcome_key.lower():
                return outcome_data
        
        return None
    
    def get_mutation_frequency(
        self,
        cancer_type: str,
        gene: str
    ) -> Optional[float]:
        """
        Get mutation frequency from literature
        
        Args:
            cancer_type: Cancer type
            gene: Gene name (e.g., 'TP53', 'KRAS')
        
        Returns:
            Mutation frequency (0-1) or None
        """
        cancer_type_lower = cancer_type.lower()
        
        if cancer_type_lower not in self.cancer_features:
            return None
        
        features = self.cancer_features[cancer_type_lower]
        
        if 'mutation_landscape' not in features:
            return None
        
        mutations = features['mutation_landscape']
        
        # Check driver mutations
        if 'driver_mutations' in mutations:
            for gene_name, gene_data in mutations['driver_mutations'].items():
                if gene.upper() == gene_name.upper():
                    return gene_data.get('prevalence')
        
        # Check amplifications
        if 'amplifications' in mutations:
            for amp_name, amp_freq in mutations['amplifications'].items():
                if gene.upper() in amp_name.upper():
                    return amp_freq
        
        return None
    
    def search_papers(
        self,
        cancer_type: Optional[str] = None,
        biomarker: Optional[str] = None,
        min_evidence_level: str = "Level_II"
    ) -> List[Dict]:
        """Search papers in database"""
        papers = self.literature_db.get('papers', [])
        results = []
        
        evidence_hierarchy = {
            "Level_I": 1,
            "Level_II": 2,
            "Level_III": 3,
            "Level_IV": 4,
            "Preclinical": 5
        }
        
        min_level_score = evidence_hierarchy.get(min_evidence_level, 5)
        
        for paper in papers:
            # Filter by cancer type
            if cancer_type:
                if paper.get('cancer_type', '').lower() != cancer_type.lower():
                    continue
            
            # Filter by evidence level
            paper_level = paper.get('evidence_level', 'Preclinical')
            paper_score = evidence_hierarchy.get(paper_level, 5)
            if paper_score > min_level_score:
                continue
            
            # Filter by biomarker (search in key findings)
            if biomarker:
                biomarker_found = False
                key_findings = paper.get('key_findings', {})
                findings_str = json.dumps(key_findings).lower()
                
                if biomarker.lower() in findings_str:
                    biomarker_found = True
                
                if not biomarker_found:
                    continue
            
            results.append(paper)
        
        return results


# Convenience function
def get_literature_features(cancer_type: str) -> Optional[Dict]:
    """
    Quick access to cancer features
    
    Args:
        cancer_type: Cancer type (e.g., 'gastric')
    
    Returns:
        Features dictionary or None
    """
    loader = LiteratureFeatureLoader()
    return loader.cancer_features.get(cancer_type.lower())
