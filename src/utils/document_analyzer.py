"""
Document Analyzer with OpenAI Integration
AI-powered document analysis for medical/scientific documents
"""

import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import openai
import json

# Load environment variables
load_dotenv(encoding='utf-8')


class DocumentAnalyzer:
    """AI-powered document analyzer using OpenAI"""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
            self.has_api = True
        else:
            self.client = None
            self.has_api = False
    
    def analyze_document(
        self,
        parsed_doc: Dict[str, Any],
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Analyze parsed document with AI
        
        Args:
            parsed_doc: Parsed document data
            analysis_type: 'comprehensive', 'summary', 'entities', 'relationships'
            
        Returns:
            Analysis results
        """
        if not self.has_api:
            return self._fallback_analysis(parsed_doc)
        
        full_text =parsed_doc.get("full_text", "")
        
        if not full_text:
            return {"error": "No text content to analyze"}
        
        if analysis_type == "comprehensive":
            return self._comprehensive_analysis(full_text, parsed_doc)
        elif analysis_type == "summary":
            return self._generate_summary(full_text)
        elif analysis_type == "entities":
            return self._extract_entities_ai(full_text)
        elif analysis_type == "relationships":
            return self._extract_relationships(full_text)
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}
    
    def _comprehensive_analysis(
        self,
        text: str,
        parsed_doc: Dict
    ) -> Dict[str, Any]:
        """Comprehensive AI analysis"""
        
        prompt = f"""Analyze this medical/scientific document comprehensively.

Document Title: {parsed_doc.get('metadata', {}).get('title', 'Unknown')}
Document Type: {parsed_doc.get('file_type', 'Unknown')}

Content (first 8000 chars):
{text[:8000]}

Provide a comprehensive analysis with:
1. executive_summary: Brief 2-3 sentence overview
2. key_findings: List of 5-7 main findings
3. drugs_mentioned: List of drugs with their purposes
4. clinical_data: Any dosages, efficacy rates, or trial results
5. mechanisms: Biological mechanisms discussed
6. implications: Clinical or research implications

Respond in clear markdown format with headers."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical document analyst. Provide detailed, structured analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Try to parse as JSON first
            try:
                analysis = json.loads(content)
                analysis['format'] = 'json'
            except json.JSONDecodeError:
                # If not JSON, keep as markdown text
                analysis = {
                    'executive_summary': 'See full analysis below',
                    'raw_analysis': content,
                    'format': 'markdown'
                }
            
            return {
                "success": True,
                "analysis": analysis,
                "model": "gpt-4o-mini",
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback": self._fallback_analysis(parsed_doc)
            }
    
    def _generate_summary(self, text: str) -> Dict[str, Any]:
        """Generate document summary"""
        
        prompt = f"""Summarize this medical/scientific document in 3-5 sentences.
Focus on: main purpose, key findings, and clinical relevance.

Document:
{text[:6000]}

Provide a concise, informative summary."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical document summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            summary = response.choices[0].message.content
            
            return {
                "success": True,
                "summary": summary,
                "model": "gpt-4o-mini"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "summary": text[:500] + "..."
            }
    
    def _extract_entities_ai(self, text: str) -> Dict[str, Any]:
        """Extract medical entities using AI"""
        
        prompt = f"""Extract medical entities from this text.

Text:
{text[:5000]}

Return JSON with:
- drugs: list of drug names
- genes: list of gene names
- proteins: list of protein names
- diseases: list of diseases/conditions
- pathways: list of biological pathways

JSON only, no additional text."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a biomedical entity extraction system. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            content = response.choices[0].message.content
            entities = json.loads(content)
            
            return {
                "success": True,
                "entities": entities,
                "model": "gpt-4o-mini"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_relationships(self, text: str) -> Dict[str, Any]:
        """Extract relationships between entities"""
        
        prompt = f"""Extract relationships from this medical text.

Text:
{text[:5000]}

Return JSON with:
- drug_disease: [{{"drug": "...", "disease": "...", "relationship": "treats/causes/..."}}]
- drug_gene: [{{"drug": "...", "gene": "...", "relationship": "inhibits/activates/..."}}]
- gene_pathway: [{{"gene": "...", "pathway": "...", "relationship": "participates/regulates/..."}}]

JSON only."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a biomedical relationship extraction system.  Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            relationships = json.loads(content)
            
            return {
                "success": True,
                "relationships": relationships,
                "model": "gpt-4o-mini"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _fallback_analysis(self, parsed_doc: Dict) -> Dict[str, Any]:
        """Fallback analysis without API"""
        
        text = parsed_doc.get("full_text", "")
        entities = parsed_doc.get("entities", {})
        
        return {
            "success": False,
            "note": "API key not configured. Using basic analysis.",
            "word_count": len(text.split()),
            "char_count": len(text),
            "sections_count": len(parsed_doc.get("sections", [])),
            "tables_count": len(parsed_doc.get("tables", [])),
            "entities_found": entities
        }
    
    def generate_fine_tuning_data(
        self,
        parsed_doc: Dict,
        analysis: Dict
    ) -> Dict[str, Any]:
        """
        Generate fine-tuning data in OpenAI format
        
        Returns:
            Fine-tuning message format
        """
        text_snippet = parsed_doc.get("full_text", "")[:2000]
        
        if analysis.get("success"):
            assistant_response = json.dumps(analysis.get("analysis", {}), indent=2)
        else:
            assistant_response = "Analysis not available."
        
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical document analyzer specialized in extracting structured information from scientific papers and clinical documents."
                },
                {
                    "role": "user",
                    "content": f"Analyze this medical document:\n\n{text_snippet}"
                },
                {
                    "role": "assistant",
                    "content": assistant_response
                }
            ]
        }
