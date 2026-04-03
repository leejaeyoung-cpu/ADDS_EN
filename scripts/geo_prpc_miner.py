#!/usr/bin/env python3
"""
GEO Database Miner for PRNP/PrPc Expression Data
Phase 1 of v3.0 AI-First Computational Discovery

Searches and downloads PRNP expression data from GEO database
Target: 3000-5000 samples from 50-100 datasets
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import requests
from Bio import Entrez
import GEOparse
from tqdm import tqdm
import time

# Configuration
Entrez.email = "brookin@inha.edu"
OUTPUT_DIR = Path("data/analysis/prpc_validation/open_data/geo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class GEOMiner:
    """Automated GEO database miner for PRNP expression"""
    
    def __init__(self, keywords=None):
        self.keywords = keywords or [
            "PRNP cancer",
            "prion protein cancer",
            "PrPc tumor",
            "PRNP colorectal",
            "PRNP pancreatic",
            "PRNP breast cancer"
        ]
        self.results = []
        self.metadata = {
            "download_date": datetime.now().isoformat(),
            "datasets_found": 0,
            "datasets_downloaded": 0,
            "total_samples": 0,
            "errors": []
        }
    
    def search_geo(self, keyword, max_results=100):
        """Search GEO for datasets matching keyword"""
        print(f"\n🔍 Searching GEO for: '{keyword}'")
        
        try:
            # Search GEO DataSets
            handle = Entrez.esearch(
                db="gds",
                term=keyword,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            
            ids = record['IdList']
            print(f"   Found {len(ids)} datasets")
            
            return ids
            
        except Exception as e:
            print(f"   ❌ Error searching: {e}")
            self.metadata['errors'].append({
                "step": "search",
                "keyword": keyword,
                "error": str(e)
            })
            return []
    
    def get_dataset_info(self, gds_id):
        """Get detailed information about a GEO dataset"""
        try:
            handle = Entrez.esummary(db="gds", id=gds_id)
            record = Entrez.read(handle)
            handle.close()
            
            if record:
                summary = record[0]
                return {
                    "gds_id": gds_id,
                    "title": summary.get('title', 'N/A'),
                    "summary": summary.get('summary', 'N/A'),
                    "n_samples": summary.get('n_samples', 0),
                    "pubmed_id": summary.get('PubMedIds', ['N/A'])[0] if 'PubMedIds' in summary else 'N/A'
                }
        except Exception as e:
            print(f"   ⚠️  Error getting info for {gds_id}: {e}")
            return None
    
    def download_gse_series(self, gse_id, max_samples=1000):
        """Download GSE series data and extract PRNP expression"""
        print(f"\n📥 Downloading {gse_id}...")
        
        try:
            # Download GSE
            gse = GEOparse.get_GEO(geo=gse_id, destdir=str(OUTPUT_DIR / "raw"))
            
            # Extract expression data
            if hasattr(gse, 'phenotype_data'):
                pheno = gse.phenotype_data
                n_samples = len(pheno)
                print(f"   Samples: {n_samples}")
                
                # Look for PRNP in expression data
                if hasattr(gse, 'table'):
                    expr_data = gse.table
                    
                    # Try to find PRNP gene
                    prnp_rows = expr_data[
                        expr_data['ID_REF'].str.contains('PRNP', case=False, na=False) |
                        expr_data['IDENTIFIER'].str.contains('PRNP', case=False, na=False) 
                        if 'IDENTIFIER' in expr_data.columns else False
                    ]
                    
                    if len(prnp_rows) > 0:
                        print(f"   ✓ Found PRNP expression data!")
                        
                        # Extract sample columns (usually GSM...)
                        sample_cols = [col for col in expr_data.columns if col.startswith('GSM')]
                        
                        if sample_cols:
                            prnp_expr = prnp_rows[sample_cols].iloc[0]
                            
                            # Combine with phenotype data
                            df = pd.DataFrame({
                                'sample_id': prnp_expr.index,
                                'PRNP_expression': prnp_expr.values,
                                'GSE_id': gse_id
                            })
                            
                            # Add phenotype info if available
                            if n_samples == len(df):
                                for col in pheno.columns:
                                    df[col] = pheno[col].values
                            
                            # Save
                            output_file = OUTPUT_DIR / f"{gse_id}_prnp.csv"
                            df.to_csv(output_file, index=False)
                            print(f"   💾 Saved to {output_file}")
                            
                            self.metadata['datasets_downloaded'] += 1
                            self.metadata['total_samples'] += len(df)
                            
                            return df
                    else:
                        print(f"   ⚠️  PRNP not found in expression data")
                        return None
                else:
                    print(f"   ⚠️  No expression table found")
                    return None
            else:
                print(f"   ⚠️  No phenotype data found")
                return None
                
        except Exception as e:
            print(f"   ❌ Error downloading {gse_id}: {e}")
            self.metadata['errors'].append({
                "step": "download",
                "gse_id": gse_id,
                "error": str(e)
            })
            return None
    
    def mine_all_datasets(self, max_datasets=50):
        """Main mining function - search and download all relevant datasets"""
        print("=" * 80)
        print("🚀 GEO PRNP Miner - Starting")
        print("=" * 80)
        
        all_ids = set()
        
        # Search with all keywords
        for keyword in self.keywords:
            ids = self.search_geo(keyword, max_results=30)
            all_ids.update(ids)
            time.sleep(1)  # Be nice to NCBI
        
        all_ids = list(all_ids)
        self.metadata['datasets_found'] = len(all_ids)
        
        print(f"\n📊 Total unique datasets found: {len(all_ids)}")
        print(f"   Will attempt to download up to {max_datasets} datasets")
        
        # Get detailed info for each
        dataset_info = []
        for gds_id in tqdm(all_ids[:max_datasets], desc="Getting dataset info"):
            info = self.get_dataset_info(gds_id)
            if info:
                dataset_info.append(info)
            time.sleep(0.5)
        
        # Save metadata
        metadata_file = OUTPUT_DIR / "geo_datasets_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        print(f"\n💾 Saved metadata for {len(dataset_info)} datasets")
        
        # Try to download GSE series
        # Note: GDS IDs need to be converted to GSE IDs
        print("\n" + "=" * 80)
        print("📥 Downloading GSE Series Data")
        print("=" * 80)
        
        # For now, we'll use a curated list of known PRNP cancer studies
        known_gse_ids = [
            "GSE39582",  # Colorectal cancer
            "GSE41258",  # Colorectal
            "GSE17536",  # Colorectal
            "GSE28735",  # Pancreatic
            "GSE15471",  # Pancreatic
            "GSE2109",   # Multi-cancer
            "GSE10846",  # Breast cancer
        ]
        
        all_data = []
        for gse_id in tqdm(known_gse_ids, desc="Downloading GSE series"):
            df = self.download_gse_series(gse_id)
            if df is not None:
                all_data.append(df)
            time.sleep(2)  # Be extra nice
        
        # Combine all data
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined_file = OUTPUT_DIR / "geo_all_prnp_combined.csv"
            combined.to_csv(combined_file, index=False)
            
            print("\n" + "=" * 80)
            print("✅ Mining Complete!")
            print("=" * 80)
            print(f"📊 Summary:")
            print(f"   Datasets found: {self.metadata['datasets_found']}")
            print(f"   Datasets downloaded: {self.metadata['datasets_downloaded']}")
            print(f"   Total samples: {self.metadata['total_samples']}")
            print(f"   Combined file: {combined_file}")
            print(f"   Errors: {len(self.metadata['errors'])}")
            
            # Save mining metadata
            self.metadata['combined_file'] = str(combined_file)
            metadata_summary = OUTPUT_DIR / "mining_summary.json"
            with open(metadata_summary, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            return combined
        else:
            print("\n❌ No data downloaded successfully")
            return None


def main():
    """Main execution"""
    miner = GEOMiner()
    
    # Run mining
    result = miner.mine_all_datasets(max_datasets=50)
    
    if result is not None:
        print(f"\n✅ Success! Downloaded {len(result)} total samples")
        print(f"\nPreview:")
        print(result.head())
        print(f"\nColumns: {list(result.columns)}")
    else:
        print("\n⚠️  Mining completed with errors. Check logs.")
    
    return result


if __name__ == "__main__":
    result = main()
