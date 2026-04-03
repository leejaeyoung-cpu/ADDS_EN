"""
Debug script to see exactly what GPT-4 returns
"""
import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load one paper from the metadata
with open("data/literature/collection_progress.json", encoding='utf-8') as f:
    data = json.load(f)
    papers = data.get('papers', [])
    paper = papers[0]  # First paper

print("=" * 80)
print("TESTING PAPER:")
print("=" * 80)
print(f"PMID: {paper['pmid']}")
print(f"Title: {paper['title'][:80]}...")
print(f"Abstract length: {len(paper['abstract'])} chars")
print()

# The exact prompt from the script
PROMPT_TEMPLATE = """Extract structured cancer knowledge from this abstract. RESPOND ONLY WITH A VALID JSON OBJECT.

Return JSON with this structure:
{
  "mechanisms": [
    {
      "pathway_name": "e.g., PI3K/AKT",
      "category": "Growth Signaling|Apoptosis|DNA Repair|Metabolism|Cell Cycle|Immune|Metastasis",
      "description": "brief description",
      "key_proteins": ["protein names"],
      "evidence_level": "in vitro|preclinical|clinical"
    }
  ],
  "drugs": [
    {
      "drug_name": "name",
      "drug_class": "Chemotherapy|Targeted|Immunotherapy|Hormone",
      "mechanism_of_action": "description",
      "molecular_target": "target"
    }
  ],
  "drug_combinations": [
    {
      "combination": ["Drug A", "Drug B"],
      "synergy_type": "additive|synergistic|antagonistic",
      "evidence": "quote from abstract"
    }
  ],
  "biomarkers": [
    {
      "name": "biomarker name",
      "type": "Genetic|Protein|Metabolic|Immune",
      "predictive_value": "what it predicts"
    }
  ],
  "clinical_findings": {
    "study_type": "Phase 1/2/3|Preclinical|Retrospective",
    "patient_count": null,
    "key_result": "main finding"
  },
  "paper_summary": {
    "cancer_type": "Colorectal|Gastric|Lung|Breast|Pancreatic|Pan-cancer",
    "novelty": "what's new (1 sentence)"
  }
}

RULES:
1. Extract ONLY explicitly stated information
2. Leave arrays empty [] if not mentioned
3. Use null for missing numbers
4. NO additional text, ONLY valid JSON

PAPER:
Title: {title}
Journal: {journal}
Year: {year}

ABSTRACT:
{abstract}
"""

prompt = PROMPT_TEMPLATE.format(
    title=paper.get('title', 'Unknown'),
    journal=paper.get('journal', 'Unknown'),
    year=paper.get('publication_year', 'Unknown'),
    abstract=paper.get('abstract', '')
)

print("=" * 80)
print("SENDING REQUEST TO GPT-4...")
print("=" * 80)
print(f"Prompt length: {len(prompt)} chars")
print()

try:
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are an expert oncology knowledge extractor. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=2000
    )
    
    content = response.choices[0].message.content
    
    print("=" * 80)
    print("RAW RESPONSE FROM GPT-4:")
    print("=" * 80)
    print(content)
    print()
    print("=" * 80)
    print(f"Response length: {len(content)} chars")
    print(f"Starts with: {repr(content[:50])}")
    print(f"Ends with: {repr(content[-50:])}")
    print()
    
    # Try to parse
    print("=" * 80)
    print("ATTEMPTING JSON PARSE...")
    print("=" * 80)
    
    try:
        parsed = json.loads(content)
        print("SUCCESS! Parsed JSON:")
        print(json.dumps(parsed, indent=2)[:500])
        print(f"\nKeys in response: {list(parsed.keys())}")
    except json.JSONDecodeError as e:
        print(f"FAILED: {e}")
        print(f"Error at position {e.pos}")
        print(f"Context around error: {repr(content[max(0, e.pos-50):e.pos+50])}")
        
except Exception as e:
    print(f"API CALL FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
