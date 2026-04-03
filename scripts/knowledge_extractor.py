"""
Knowledge Extraction Pipeline using GPT-4
==========================================
Automatically extracts cancer mechanisms, drug information, and clinical insights
from scientific papers using GPT-4.

Usage:
    python scripts/knowledge_extractor.py --input data/literature/pdfs --output data/extracted
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import PyPDF2
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

class KnowledgeExtractor:
    """Extract structured knowledge from scientific literature"""
    
    def __init__(self):
        self.extraction_prompt = self._create_extraction_prompt()
    
    def _create_extraction_prompt(self) -> str:
        """Create comprehensive extraction prompt for GPT-4"""
        return """
        You are an expert oncologist and molecular biologist extracting structured information 
        from cancer research papers. Extract the following information in JSON format:

        {
          "mechanisms": [
            {
              "pathway_name": "RAS/RAF/MEK/ERK",
              "category": "Growth Signaling",
              "description": "Detailed mechanism description",
              "key_proteins": ["RAS", "RAF", "MEK", "ERK"],
              "regulation_type": "Activation"
            }
          ],
          "drugs": [
            {
              "drug_name": "Cetuximab",
              "generic_name": "cetuximab",
              "drug_class": "Targeted Therapy",
              "mechanism_of_action": "Blocks EGFR receptor",
              "molecular_target": "EGFR",
              "pathways_affected": ["EGFR", "RAS/RAF/MEK/ERK"]
            }
          ],
          "drug_interactions": [
            {
              "drug1": "5-Fluorouracil",
              "drug2": "Oxaliplatin",
              "interaction_type": "Synergy",
              "synergy_score": 0.65,
              "mechanism_basis": "Complementary DNA damage mechanisms"
            }
          ],
          "resistance_mechanisms": [
            {
              "drug": "Cetuximab",
              "resistance_type": "Acquired",
              "mechanism": "KRAS mutation",
              "genetic_alterations": ["KRAS G12D", "KRAS G13D"],
              "bypass_pathways": ["PI3K/AKT"],
              "overcoming_strategies": "Combine with MEK inhibitor"
            }
          ],
          "biomarkers": [
            {
              "name": "KRAS",
              "type": "Genetic",
              "measurement": "NGS",
              "predictive_value": "Predicts resistance to anti-EGFR therapy",
              "drug_associations": {
                "Cetuximab": "Only effective in KRAS wild-type"
              }
            }
          ],
          "clinical_findings": {
            "patient_cohort": "Metastatic colorectal cancer",
            "sample_size": 500,
            "key_result": "FOLFOX + Bevacizumab improved PFS vs FOLFOX alone",
            "statistical_significance": "p < 0.001",
            "clinical_relevance": "First-line treatment standard"
          }
        }

        IMPORTANT: 
        - Extract ONLY information explicitly stated in the paper
        - Include specific molecular details (proteins, genes, binding sites)
        - Include quantitative data (IC50, CI, p-values) when available
        - Note evidence level (in vitro, preclinical, clinical trial phase)
        """
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from PDF"""
        print(f"📄 Reading {pdf_path.name}...")
        
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        
        return text
    
    def chunk_text(self, text: str, max_chars: int = 12000) -> List[str]:
        """Split text into chunks for GPT-4 processing"""
        # Simple chunking by character count
        # TODO: Improve with semantic chunking
        chunks = []
        current_chunk = ""
        
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < max_chars:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def extract_knowledge(self, text: str) -> Dict:
        """Use GPT-4 to extract structured knowledge"""
        print("🤖 Extracting knowledge with GPT-4...")
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert oncologist extracting cancer mechanisms from scientific literature. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": f"{self.extraction_prompt}\n\nPAPER TEXT:\n{text}"
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=4096
            )
            
            extracted_data = json.loads(response.choices[0].message.content)
            return extracted_data
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return {}
    
    def validate_extraction(self, data: Dict) -> Dict:
        """Validate and clean extracted data"""
        # Add validation logic
        # Check required fields, data types, etc.
        return data
    
    def merge_extractions(self, extractions: List[Dict]) -> Dict:
        """Merge multiple extraction chunks"""
        merged = {
            "mechanisms": [],
            "drugs": [],
            "drug_interactions": [],
            "resistance_mechanisms": [],
            "biomarkers": [],
            "clinical_findings": {}
        }
        
        for extraction in extractions:
            for key in merged.keys():
                if key in extraction:
                    if isinstance(merged[key], list):
                        merged[key].extend(extraction[key])
                    elif isinstance(extraction[key], dict):
                        merged[key].update(extraction[key])
        
        # Remove duplicates (simple implementation)
        for key in merged.keys():
            if isinstance(merged[key], list):
                merged[key] = [dict(t) for t in {tuple(sorted(d.items())) for d in merged[key] if isinstance(d, dict)}]
        
        return merged
    
    def process_paper(self, pdf_path: Path, output_dir: Path) -> Dict:
        """Complete pipeline to process one paper"""
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Chunk text
        chunks = self.chunk_text(text)
        print(f"📚 Split into {len(chunks)} chunks")
        
        # Extract from each chunk
        extractions = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            extraction = self.extract_knowledge(chunk)
            if extraction:
                extractions.append(extraction)
        
        # Merge extractions
        merged = self.merge_extractions(extractions)
        
        # Validate
        validated = self.validate_extraction(merged)
        
        # Save
        output_file = output_dir / f"{pdf_path.stem}_extracted.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validated, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved to {output_file}")
        return validated


def main():
    parser = argparse.ArgumentParser(description="Extract knowledge from papers")
    parser.add_argument("--input", type=str, required=True, help="Input PDF directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = KnowledgeExtractor()
    
    # Process all PDFs
    pdf_files = list(input_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_file.name}")
        print('='*60)
        
        try:
            extractor.process_paper(pdf_file, output_dir)
        except Exception as e:
            print(f"❌ Failed to process {pdf_file.name}: {e}")
    
    print("\n🎉 All papers processed!")


if __name__ == "__main__":
    main()
