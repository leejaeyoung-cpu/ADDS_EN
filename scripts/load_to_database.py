"""
Database Loader for Extracted Knowledge
========================================
Loads GPT-4 extracted knowledge into MySQL knowledge base
"""

import json
import mysql.connector
from pathlib import Path
from typing import Dict, List
import sys

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Set your MySQL password
    'database': 'cancer_knowledge_db'
}


class KnowledgeDBLoader:
    """Load extracted knowledge into database"""
    
    def __init__(self, db_config: dict):
        self.conn = mysql.connector.connect(**db_config)
        self.cursor = self.conn.cursor()
    
    def load_mechanisms(self, mechanisms: List[Dict], source_ref_id: int):
        """Load mechanisms into database"""
        for mech in mechanisms:
            query = """
            INSERT INTO mechanisms 
            (category, subcategory, pathway_name, description, key_proteins, regulation_type)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(query, (
                mech.get('category'),
                mech.get('subcategory'),
                mech['pathway_name'],
                mech['description'],
                json.dumps(mech.get('key_proteins', [])),
                mech.get('regulation_type')
            ))
            
            print(f"  + Mechanism: {mech['pathway_name']}")
    
    def load_drugs(self, drugs: List[Dict], source_ref_id: int):
        """Load drugs into database"""
        for drug in drugs:
            # Check if drug already exists
            self.cursor.execute(
                "SELECT id FROM drugs WHERE generic_name = %s",
                (drug['generic_name'],)
            )
            result = self.cursor.fetchone()
            
            if not result:
                query = """
                INSERT INTO drugs 
                (drug_name, generic_name, drug_class, mechanism_of_action, 
                 molecular_target, pathway_affected)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                
                self.cursor.execute(query, (
                    drug.get('drug_name'),
                    drug['generic_name'],
                    drug.get('drug_class'),
                    drug.get('mechanism_of_action'),
                    drug.get('molecular_target'),
                    json.dumps(drug.get('pathways_affected', []))
                ))
                
                print(f"  + Drug: {drug['generic_name']}")
            else:
                print(f"  ○ Drug exists: {drug['generic_name']}")
    
    def load_drug_interactions(self, interactions: List[Dict], source_ref_id: int):
        """Load drug interactions"""
        for interaction in interactions:
            # Get drug IDs
            drug1_id = self._get_drug_id(interaction['drug1'])
            drug2_id = self._get_drug_id(interaction['drug2'])
            
            if drug1_id and drug2_id:
                # Ensure ordered pair
                if drug1_id > drug2_id:
                    drug1_id, drug2_id = drug2_id, drug1_id
                
                query = """
                INSERT INTO drug_interactions 
                (drug1_id, drug2_id, interaction_type, synergy_score, mechanism_basis)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                synergy_score = VALUES(synergy_score),
                mechanism_basis = VALUES(mechanism_basis)
                """
                
                self.cursor.execute(query, (
                    drug1_id,
                    drug2_id,
                    interaction.get('interaction_type'),
                    interaction.get('synergy_score'),
                    interaction.get('mechanism_basis')
                ))
                
                print(f"  + Interaction: {interaction['drug1']} + {interaction['drug2']}")
    
    def load_biomarkers(self, biomarkers: List[Dict], source_ref_id: int):
        """Load biomarkers"""
        for marker in biomarkers:
            # Check if exists
            self.cursor.execute(
                "SELECT id FROM biomarkers WHERE biomarker_name = %s",
                (marker['name'],)
            )
            result = self.cursor.fetchone()
            
            if not result:
                query = """
                INSERT INTO biomarkers 
                (biomarker_name, biomarker_type, measurement_method, 
                 predictive_value, drug_response_impact)
                VALUES (%s, %s, %s, %s, %s)
                """
                
                self.cursor.execute(query, (
                    marker['name'],
                    marker.get('type'),
                    marker.get('measurement'),
                    marker.get('predictive_value'),
                    json.dumps(marker.get('drug_associations', {}))
                ))
                
                print(f"  + Biomarker: {marker['name']}")
            else:
                print(f"  ○ Biomarker exists: {marker['name']}")
    
    def load_literature_reference(self, source: Dict) -> int:
        """Load literature reference and return ID"""
        query = """
        INSERT INTO literature_references 
        (reference_type, title, journal, publication_year, doi, pmid)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        self.cursor.execute(query, (
            'Nature' if 'Nature' in source['journal'] else 'SCI',
            source['title'],
            source['journal'],
            int(source['year']),
            source.get('doi'),
            source.get('pmid')
        ))
        
        return self.cursor.lastrowid
    
    def _get_drug_id(self, drug_name: str) -> int:
        """Get drug ID by name"""
        self.cursor.execute(
            "SELECT id FROM drugs WHERE generic_name LIKE %s OR drug_name LIKE %s",
            (f"%{drug_name}%", f"%{drug_name}%")
        )
        result = self.cursor.fetchone()
        return result[0] if result else None
    
    def load_extraction(self, extraction: Dict):
        """Load complete extraction"""
        print(f"\n{'='*70}")
        print(f"Loading: {extraction['source']['title'][:60]}...")
        print(f"{'='*70}")
        
        # Load literature reference
        ref_id = self.load_literature_reference(extraction['source'])
        print(f"  Reference ID: {ref_id}")
        
        # Load each type
        if 'mechanisms' in extraction:
            self.load_mechanisms(extraction['mechanisms'], ref_id)
        
        if 'drugs' in extraction:
            self.load_drugs(extraction['drugs'], ref_id)
        
        if 'drug_interactions' in extraction:
            self.load_drug_interactions(extraction['drug_interactions'], ref_id)
        
        if 'biomarkers' in extraction:
            self.load_biomarkers(extraction['biomarkers'], ref_id)
        
        self.conn.commit()
        print("  Committed to database")
    
    def close(self):
        """Close database connection"""
        self.cursor.close()
        self.conn.close()


def main():
    print("="*70)
    print("  Loading Knowledge into Database")
    print("="*70)
    print()
    
    # Load extracted knowledge
    extraction_file = Path("data/extracted/sample_extracted_knowledge.json")
    
    if not extraction_file.exists():
        print("Error: No extracted knowledge found!")
        print("Run: python scripts/demo_extraction.py first")
        return
    
    with open(extraction_file, 'r', encoding='utf-8') as f:
        extractions = json.load(f)
    
    print(f"Found {len(extractions)} extracted papers")
    print()
    
    # Check database connection
    try:
        loader = KnowledgeDBLoader(DB_CONFIG)
        print("Database connected!")
        print()
    except mysql.connector.Error as e:
        print(f"Database connection failed: {e}")
        print("\nPlease ensure:")
        print("1. MySQL server is running")
        print("2. Database 'cancer_knowledge_db' exists")
        print("3. Schema is loaded (SOURCE database/cancer_knowledge_schema.sql)")
        return
    
    # Load each extraction
    for extraction in extractions:
        try:
            loader.load_extraction(extraction)
        except Exception as e:
            print(f"Error loading: {e}")
            continue
    
    loader.close()
    
    print(f"\n{'='*70}")
    print("Loading complete!")
    print(f"{'='*70}")
    print()
    print("Verify with SQL queries:")
    print("  SELECT * FROM drugs;")
    print("  SELECT * FROM biomarkers;")
    print("  SELECT * FROM high_synergy_combinations;")
    print()


if __name__ == "__main__":
    main()
