"""
Knowledge Base Consolidation Script
====================================
Consolidate 88 extracted knowledge files into unified knowledge base.

Features:
- Merge all extracted JSON files
- Remove duplicates (drugs, mechanisms, biomarkers)
- Normalize data (standardize naming, categories)
- Create searchable indexes
- Generate statistics and quality reports

Author: ADDS Team
Date: 2026-01-31
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime

# Paths
EXTRACTED_DIR = Path("data/extracted/abstracts")
OUTPUT_DIR = Path("data/knowledge_base")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
CONSOLIDATED_KB = OUTPUT_DIR / "cancer_knowledge_base.json"
DRUG_INDEX = OUTPUT_DIR / "drug_index.json"
MECHANISM_INDEX = OUTPUT_DIR / "mechanism_index.json"
BIOMARKER_INDEX = OUTPUT_DIR / "biomarker_index.json"
STATISTICS_REPORT = OUTPUT_DIR / "consolidation_report.md"


class KnowledgeConsolidator:
    """Consolidate extracted knowledge into unified database"""
    
    def __init__(self):
        self.papers = []
        self.all_drugs = defaultdict(list)  # drug_name -> [paper_pmid]
        self.all_mechanisms = defaultdict(list)  # pathway -> [paper_pmid]
        self.all_biomarkers = defaultdict(list)  # biomarker -> [paper_pmid]
        self.all_combinations = []
        self.cancer_types = Counter()
        
    def load_all_extractions(self, extracted_dir: Path) -> int:
        """Load all extracted JSON files"""
        print("\n[LOAD] Loading extracted knowledge files...")
        
        count = 0
        for json_file in sorted(extracted_dir.glob("*_extracted.json")):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.papers.append(data)
                    count += 1
            except Exception as e:
                print(f"   [WARN] Failed to load {json_file.name}: {e}")
        
        print(f"   [OK] Loaded {count} papers")
        return count
    
    def normalize_drug_name(self, name: str) -> str:
        """Normalize drug names for matching"""
        # Basic normalization
        name = name.strip().lower()
        
        # Remove common suffixes
        suffixes = [' inhibitor', ' inhibitors', ' (adc)', ' (adcs)']
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        
        return name.strip()
    
    def normalize_pathway_name(self, name: str) -> str:
        """Normalize pathway names"""
        # Standardize common pathway names
        pathway_map = {
            'rtk/mapk': 'RTK/MAPK',
            'pi3k/akt/mtor': 'PI3K/AKT/mTOR',
            'pi3k/akt': 'PI3K/AKT',
            'wnt/β-catenin': 'Wnt/β-catenin',
            'met-stat3': 'MET-STAT3',
            'jak-stat': 'JAK-STAT'
        }
        
        normalized = name.strip()
        lower = normalized.lower()
        
        return pathway_map.get(lower, normalized)
    
    def index_drugs(self):
        """Create drug index"""
        print("\n[INDEX] Indexing drugs...")
        
        drug_details = {}
        
        for paper in self.papers:
            pmid = paper['source']['pmid']
            cancer_type = paper.get('paper_summary', {}).get('cancer_type', 'Unknown')
            
            for drug in paper.get('drugs', []):
                drug_name = drug.get('drug_name', '')
                if not drug_name:
                    continue
                
                normalized = self.normalize_drug_name(drug_name)
                
                # Track which papers mention this drug
                self.all_drugs[normalized].append(pmid)
                
                # Store drug details
                if normalized not in drug_details:
                    drug_details[normalized] = {
                        'name': drug_name,  # Original name
                        'class': drug.get('drug_class', 'Unknown'),
                        'mechanism': drug.get('mechanism_of_action', ''),
                        'target': drug.get('molecular_target', ''),
                        'papers': [],
                        'cancer_types': set()
                    }
                
                drug_details[normalized]['papers'].append({
                    'pmid': pmid,
                    'cancer_type': cancer_type
                })
                drug_details[normalized]['cancer_types'].add(cancer_type)
        
        # Convert sets to lists for JSON serialization
        for drug in drug_details.values():
            drug['cancer_types'] = sorted(list(drug['cancer_types']))
            drug['paper_count'] = len(drug['papers'])
        
        print(f"   [OK] Indexed {len(drug_details)} unique drugs")
        
        return drug_details
    
    def index_mechanisms(self):
        """Create mechanism index"""
        print("\n[INDEX] Indexing mechanisms...")
        
        mechanism_details = {}
        
        for paper in self.papers:
            pmid = paper['source']['pmid']
            cancer_type = paper.get('paper_summary', {}).get('cancer_type', 'Unknown')
            
            for mech in paper.get('mechanisms', []):
                pathway = mech.get('pathway_name', '')
                if not pathway:
                    continue
                
                normalized = self.normalize_pathway_name(pathway)
                
                # Track papers
                self.all_mechanisms[normalized].append(pmid)
                
                # Store mechanism details
                if normalized not in mechanism_details:
                    mechanism_details[normalized] = {
                        'pathway': normalized,
                        'categories': set(),
                        'proteins': set(),
                        'descriptions': [],
                        'papers': [],
                        'cancer_types': set()
                    }
                
                # Add details
                category = mech.get('category', '')
                if category:
                    for cat in category.split('|'):
                        mechanism_details[normalized]['categories'].add(cat.strip())
                
                proteins = mech.get('key_proteins', [])
                mechanism_details[normalized]['proteins'].update(proteins)
                
                desc = mech.get('description', '')
                if desc and desc not in mechanism_details[normalized]['descriptions']:
                    mechanism_details[normalized]['descriptions'].append(desc)
                
                mechanism_details[normalized]['papers'].append({
                    'pmid': pmid,
                    'cancer_type': cancer_type,
                    'evidence_level': mech.get('evidence_level', 'unknown')
                })
                mechanism_details[normalized]['cancer_types'].add(cancer_type)
        
        # Convert sets to lists
        for mech in mechanism_details.values():
            mech['categories'] = sorted(list(mech['categories']))
            mech['proteins'] = sorted(list(mech['proteins']))
            mech['cancer_types'] = sorted(list(mech['cancer_types']))
            mech['paper_count'] = len(mech['papers'])
        
        print(f"   [OK] Indexed {len(mechanism_details)} unique mechanisms")
        
        return mechanism_details
    
    def index_biomarkers(self):
        """Create biomarker index"""
        print("\n[INDEX] Indexing biomarkers...")
        
        biomarker_details = {}
        
        for paper in self.papers:
            pmid = paper['source']['pmid']
            cancer_type = paper.get('paper_summary', {}).get('cancer_type', 'Unknown')
            
            for biomarker in paper.get('biomarkers', []):
                name = biomarker.get('name', '')
                if not name:
                    continue
                
                normalized = name.strip()
                
                # Track papers
                self.all_biomarkers[normalized].append(pmid)
                
                # Store details
                if normalized not in biomarker_details:
                    biomarker_details[normalized] = {
                        'name': normalized,
                        'types': set(),
                        'predictive_values': [],
                        'papers': [],
                        'cancer_types': set()
                    }
                
                bio_type = biomarker.get('type', '')
                if bio_type:
                    biomarker_details[normalized]['types'].add(bio_type)
                
                pred_val = biomarker.get('predictive_value', '')
                if pred_val and pred_val not in biomarker_details[normalized]['predictive_values']:
                    biomarker_details[normalized]['predictive_values'].append(pred_val)
                
                biomarker_details[normalized]['papers'].append({
                    'pmid': pmid,
                    'cancer_type': cancer_type
                })
                biomarker_details[normalized]['cancer_types'].add(cancer_type)
        
        # Convert sets to lists
        for bio in biomarker_details.values():
            bio['types'] = sorted(list(bio['types']))
            bio['cancer_types'] = sorted(list(bio['cancer_types']))
            bio['paper_count'] = len(bio['papers'])
        
        print(f"   [OK] Indexed {len(biomarker_details)} unique biomarkers")
        
        return biomarker_details
    
    def extract_combinations(self):
        """Extract all drug combinations"""
        print("\n[EXTRACT] Extracting drug combinations...")
        
        all_combos = []
        
        for paper in self.papers:
            pmid = paper['source']['pmid']
            cancer_type = paper.get('paper_summary', {}).get('cancer_type', 'Unknown')
            
            for combo in paper.get('drug_combinations', []):
                drugs = combo.get('combination', [])
                if len(drugs) < 2:
                    continue
                
                all_combos.append({
                    'drugs': drugs,
                    'synergy_type': combo.get('synergy_type', 'unknown'),
                    'evidence': combo.get('evidence', ''),
                    'pmid': pmid,
                    'cancer_type': cancer_type
                })
        
        print(f"   [OK] Found {len(all_combos)} drug combinations")
        
        return all_combos
    
    def generate_statistics(self, drug_index, mechanism_index, biomarker_index, combinations):
        """Generate consolidation statistics"""
        print("\n[STATS] Generating statistics...")
        
        # Cancer type distribution
        for paper in self.papers:
            cancer_type = paper.get('paper_summary', {}).get('cancer_type', 'Unknown')
            self.cancer_types[cancer_type] += 1
        
        # Quality distribution
        quality_scores = [p['quality']['score'] for p in self.papers if 'quality' in p]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Study type distribution
        study_types = Counter()
        for paper in self.papers:
            study_type = paper.get('clinical_findings', {}).get('study_type', 'Unknown')
            study_types[study_type] += 1
        
        stats = {
            'total_papers': len(self.papers),
            'unique_drugs': len(drug_index),
            'unique_mechanisms': len(mechanism_index),
            'unique_biomarkers': len(biomarker_index),
            'total_combinations': len(combinations),
            'avg_quality_score': avg_quality,
            'cancer_types': dict(self.cancer_types),
            'study_types': dict(study_types),
            'most_mentioned_drugs': self._get_top_items(self.all_drugs, 10),
            'most_mentioned_mechanisms': self._get_top_items(self.all_mechanisms, 10),
            'most_mentioned_biomarkers': self._get_top_items(self.all_biomarkers, 10)
        }
        
        return stats
    
    def _get_top_items(self, items_dict: defaultdict, top_n: int) -> List[Tuple[str, int]]:
        """Get top N most mentioned items"""
        return sorted(
            [(name, len(pmids)) for name, pmids in items_dict.items()],
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
    
    def generate_report(self, stats: Dict) -> str:
        """Generate markdown report"""
        
        report = f"""# Knowledge Base Consolidation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary

- **Total Papers Processed**: {stats['total_papers']}
- **Unique Drugs**: {stats['unique_drugs']}
- **Unique Mechanisms**: {stats['unique_mechanisms']}
- **Unique Biomarkers**: {stats['unique_biomarkers']}
- **Drug Combinations**: {stats['total_combinations']}
- **Average Quality Score**: {stats['avg_quality_score']:.3f}

---

## Cancer Type Distribution

| Cancer Type | Papers |
|-------------|--------|
"""
        
        for cancer_type, count in sorted(stats['cancer_types'].items(), key=lambda x: x[1], reverse=True):
            report += f"| {cancer_type} | {count} |\n"
        
        report += f"""
---

## Study Type Distribution

| Study Type | Count |
|------------|-------|
"""
        
        for study_type, count in sorted(stats['study_types'].items(), key=lambda x: x[1], reverse=True):
            report += f"| {study_type} | {count} |\n"
        
        report += f"""
---

## Top Mentioned Drugs

| Rank | Drug | Papers |
|------|------|--------|
"""
        
        for rank, (drug, count) in enumerate(stats['most_mentioned_drugs'], 1):
            report += f"| {rank} | {drug} | {count} |\n"
        
        report += f"""
---

## Top Mechanisms

| Rank | Pathway | Papers |
|------|---------|--------|
"""
        
        for rank, (pathway, count) in enumerate(stats['most_mentioned_mechanisms'], 1):
            report += f"| {rank} | {pathway} | {count} |\n"
        
        report += f"""
---

## Top Biomarkers

| Rank | Biomarker | Papers |
|------|-----------|--------|
"""
        
        for rank, (biomarker, count) in enumerate(stats['most_mentioned_biomarkers'], 1):
            report += f"| {rank} | {biomarker} | {count} |\n"
        
        report += f"""
---

## Output Files

- **Consolidated Knowledge Base**: `{CONSOLIDATED_KB}`
- **Drug Index**: `{DRUG_INDEX}`
- **Mechanism Index**: `{MECHANISM_INDEX}`
- **Biomarker Index**: `{BIOMARKER_INDEX}`
- **Statistics Report**: `{STATISTICS_REPORT}`

---

## Next Steps

1. Integrate knowledge base with ADDS CDSS system
2. Create search/query interface
3. Validate against clinical cases
4. Update regularly with new papers
"""
        
        return report
    
    def consolidate(self):
        """Main consolidation process"""
        print("="*70)
        print(" KNOWLEDGE BASE CONSOLIDATION")
        print("="*70)
        
        # Step 1: Load all extractions
        count = self.load_all_extractions(EXTRACTED_DIR)
        
        if count == 0:
            print("[ERROR] No extraction files found!")
            return
        
        # Step 2: Index all data
        drug_index = self.index_drugs()
        mechanism_index = self.index_mechanisms()
        biomarker_index = self.index_biomarkers()
        combinations = self.extract_combinations()
        
        # Step 3: Generate statistics
        stats = self.generate_statistics(drug_index, mechanism_index, biomarker_index, combinations)
        
        # Step 4: Save consolidated knowledge base
        print("\n[SAVE] Saving consolidated knowledge base...")
        
        consolidated = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'total_papers': len(self.papers),
                'version': '1.0'
            },
            'statistics': stats,
            'papers': self.papers
        }
        
        with open(CONSOLIDATED_KB, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)
        print(f"   [OK] Saved to {CONSOLIDATED_KB}")
        
        # Step 5: Save indexes
        print("\n[SAVE] Saving indexes...")
        
        with open(DRUG_INDEX, 'w', encoding='utf-8') as f:
            json.dump(drug_index, f, indent=2, ensure_ascii=False)
        
        with open(MECHANISM_INDEX, 'w', encoding='utf-8') as f:
            json.dump(mechanism_index, f, indent=2, ensure_ascii=False)
        
        with open(BIOMARKER_INDEX, 'w', encoding='utf-8') as f:
            json.dump(biomarker_index, f, indent=2, ensure_ascii=False)
        
        print(f"   [OK] Drug index: {DRUG_INDEX}")
        print(f"   [OK] Mechanism index: {MECHANISM_INDEX}")
        print(f"   [OK] Biomarker index: {BIOMARKER_INDEX}")
        
        # Step 6: Generate report
        print("\n[REPORT] Generating consolidation report...")
        report = self.generate_report(stats)
        
        with open(STATISTICS_REPORT, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"   [OK] Report saved to {STATISTICS_REPORT}")
        
        # Summary
        print("\n" + "="*70)
        print(" CONSOLIDATION COMPLETE!")
        print("="*70)
        print(f"\nFinal Summary:")
        print(f"   Papers: {stats['total_papers']}")
        print(f"   Drugs: {stats['unique_drugs']}")
        print(f"   Mechanisms: {stats['unique_mechanisms']}")
        print(f"   Biomarkers: {stats['unique_biomarkers']}")
        print(f"   Combinations: {stats['total_combinations']}")
        print(f"   Quality: {stats['avg_quality_score']:.3f}")
        print()


def main():
    """Main execution"""
    consolidator = KnowledgeConsolidator()
    consolidator.consolidate()


if __name__ == "__main__":
    main()
