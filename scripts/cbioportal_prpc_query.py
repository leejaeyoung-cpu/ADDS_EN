#!/usr/bin/env python3
"""
cBioPortal Query Tool for PRNP Data
Phase 1 of v3.0 AI-First Computational Discovery

Queries cBioPortal API for PRNP expression, mutations, and clinical data
across all cancer studies
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import requests
from tqdm import tqdm
import time

# Configuration
CBIO_API = "https://www.cbioportal.org/api"
OUTPUT_DIR = Path("data/analysis/prpc_validation/open_data/cbioportal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class cBioPortalQuery:
    """Query cBioPortal for PRNP data across all cancer studies"""
    
    def __init__(self):
        self.api_base = CBIO_API
        self.session = requests.Session()
        self.metadata = {
            "query_date": datetime.now().isoformat(),
            "studies_queried": 0,
            "samples_found": 0,
            "errors": []
        }
    
    def get_all_studies(self):
        """Get list of all cancer studies in cBioPortal"""
        print("🔍 Fetching all cancer studies from cBioPortal...")
        
        url = f"{self.api_base}/studies"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            studies = response.json()
            
            print(f"   Found {len(studies)} studies")
            
            # Filter for studies with expression data
            expression_studies = [
                s for s in studies 
                if 'mrna' in s.get('description', '').lower() or
                   'expression' in s.get('description', '').lower() or
                   'rna' in s.get('name', '').lower()
            ]
            
            print(f"   {len(expression_studies)} have expression data")
            
            return expression_studies
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.metadata['errors'].append({
                "step": "get_studies",
                "error": str(e)
            })
            return []
    
    def get_molecular_profiles(self, study_id):
        """Get molecular profiles for a study"""
        url = f"{self.api_base}/studies/{study_id}/molecular-profiles"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            profiles = response.json()
            
            # Find mRNA expression profile
            mrna_profiles = [
                p for p in profiles 
                if p['molecularAlterationType'] == 'MRNA_EXPRESSION'
            ]
            
            return mrna_profiles
            
        except Exception as e:
            return []
    
    def get_prnp_expression(self, study_id, molecular_profile_id):
        """Get PRNP expression data for a specific study"""
        print(f"\n📥 Querying {study_id}...")
        
        url = f"{self.api_base}/molecular-profiles/{molecular_profile_id}/molecular-data"
        
        params = {
            "entrezGeneId": 5621,  # PRNP gene ID
            "projection": "SUMMARY"
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data:
                print(f"   ✓ Found {len(data)} samples with PRNP data")
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                df['study_id'] = study_id
                df['molecular_profile_id'] = molecular_profile_id
                
                return df
            else:
                print(f"   ⚠️  No PRNP data found")
                return None
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.metadata['errors'].append({
                "step": "get_expression",
                "study_id": study_id,
                "error": str(e)
            })
            return None
    
    def get_clinical_data(self, study_id, sample_ids):
        """Get clinical data for samples"""
        url = f"{self.api_base}/studies/{study_id}/clinical-data"
        
        # cBioPortal API can be picky, try simple GET first
        try:
            response = self.session.get(url)
            response.raise_for_status()
            clinical = response.json()
            
            if clinical:
                df = pd.DataFrame(clinical)
                # Filter to our samples
                df = df[df['sampleId'].isin(sample_ids)]
                return df
            else:
                return None
                
        except Exception as e:
            return None
    
    def query_all_studies(self, max_studies=None):
        """Main query function - get PRNP data from all studies"""
        print("=" * 80)
        print("🚀 cBioPortal PRNP Query - Starting")
        print("=" * 80)
        
        # Get all studies
        studies = self.get_all_studies()
        
        if max_studies:
            studies = studies[:max_studies]
        
        self.metadata['studies_queried'] = len(studies)
        
        all_expression = []
        all_clinical = []
        
        # Query each study
        for study in tqdm(studies, desc="Querying studies"):
            study_id = study['studyId']
            
            # Get molecular profiles
            profiles = self.get_molecular_profiles(study_id)
            
            if not profiles:
                continue
            
            # Query each mRNA profile
            for profile in profiles:
                profile_id = profile['molecularProfileId']
                
                # Get PRNP expression
                expr_df = self.get_prnp_expression(study_id, profile_id)
                
                if expr_df is not None:
                    all_expression.append(expr_df)
                    
                    # Get clinical data for these samples
                    sample_ids = expr_df['sampleId'].tolist()
                    clinical_df = self.get_clinical_data(study_id, sample_ids)
                    
                    if clinical_df is not None:
                        all_clinical.append(clinical_df)
                    
                    self.metadata['samples_found'] += len(expr_df)
            
            time.sleep(0.5)  # Be nice to API
        
        # Combine all data
        if all_expression:
            combined_expr = pd.concat(all_expression, ignore_index=True)
            expr_file = OUTPUT_DIR / "cbioportal_prnp_expression.csv"
            combined_expr.to_csv(expr_file, index=False)
            print(f"\n💾 Saved expression data: {expr_file}")
            
            if all_clinical:
                combined_clinical = pd.concat(all_clinical, ignore_index=True)
                clinical_file = OUTPUT_DIR / "cbioportal_clinical_data.csv"
                combined_clinical.to_csv(clinical_file, index=False)
                print(f"💾 Saved clinical data: {clinical_file}")
                
                # Merge expression with clinical
                merged = combined_expr.merge(
                    combined_clinical,
                    on=['sampleId', 'studyId'],
                    how='left'
                )
                merged_file = OUTPUT_DIR / "cbioportal_prnp_with_clinical.csv"
                merged.to_csv(merged_file, index=False)
                print(f"💾 Saved merged data: {merged_file}")
            
            # Summary
            print("\n" + "=" * 80)
            print("✅ Query Complete!")
            print("=" * 80)
            print(f"📊 Summary:")
            print(f"   Studies queried: {self.metadata['studies_queried']}")
            print(f"   Samples found: {self.metadata['samples_found']}")
            print(f"   Errors: {len(self.metadata['errors'])}")
            
            # Save metadata
            self.metadata['expression_file'] = str(expr_file)
            if all_clinical:
                self.metadata['clinical_file'] = str(clinical_file)
                self.metadata['merged_file'] = str(merged_file)
            
            metadata_file = OUTPUT_DIR / "query_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            return combined_expr
        else:
            print("\n❌ No data found")
            return None


def main():
    """Main execution"""
    query = cBioPortalQuery()
    
    # Query all studies (or limit for testing)
    result = query.query_all_studies(max_studies=None)  # None = all studies
    
    if result is not None:
        print(f"\n✅ Success! Found {len(result)} samples")
        print(f"\nPreview:")
        print(result.head())
        print(f"\nColumns: {list(result.columns)}")
        print(f"\nUnique studies: {result['study_id'].nunique()}")
    else:
        print("\n⚠️  Query completed with errors. Check logs.")
    
    return result


if __name__ == "__main__":
    result = main()
