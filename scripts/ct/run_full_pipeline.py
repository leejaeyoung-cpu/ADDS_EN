#!/usr/bin/env python3
"""
Automated Pipeline Orchestrator for v3.0 AI-First Project
Runs all phases sequentially: Mining → Harmonization → ML Training → Validation

Usage:
    python run_full_pipeline.py --phase all
    python run_full_pipeline.py --phase mining
    python run_full_pipeline.py --phase training
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse

class PipelineOrchestrator:
    """Orchestrates the entire v3.0 AI-First pipeline"""
    
    def __init__(self):
        self.base_dir = Path("C:/Users/brook/Desktop/ADDS")
        self.scripts_dir = self.base_dir / "scripts"
        self.data_dir = self.base_dir / "data/analysis/prpc_validation"
        
        self.log = {
            "start_time": datetime.now().isoformat(),
            "phases": {}
        }
    
    def run_script(self, script_name, description):
        """Run a Python script and log results"""
        print(f"\n{'='*80}")
        print(f"🚀 {description}")
        print(f"   Script: {script_name}")
        print(f"{'='*80}\n")
        
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                encoding='utf-8',
                env={'PYTHONIOENCODING': 'utf-8'}
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(result.stdout)
            
            if result.returncode == 0:
                print(f"\n✅ {description} - SUCCESS ({duration:.1f}s)")
                self.log["phases"][script_name] = {
                    "status": "success",
                    "duration_seconds": duration,
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                }
                return True
            else:
                print(f"\n❌ {description} - FAILED")
                print(f"Error: {result.stderr}")
                self.log["phases"][script_name] = {
                    "status": "failed",
                    "error": result.stderr,
                    "duration_seconds": duration
                }
                return False
                
        except Exception as e:
            print(f"\n❌ Error running {script_name}: {e}")
            self.log["phases"][script_name] = {
                "status": "error",
                "error": str(e)
            }
            return False
    
    def phase_1_mining(self):
        """Phase 1: Data Mining"""
        print("\n" + "="*80)
        print("📊 PHASE 1: DATA MINING")
        print("="*80)
        
        # Check if TCGA data exists
        tcga_file = self.data_dir / "open_data/real/tcga_all_cancers_prnp_real.csv"
        if tcga_file.exists():
            print(f"✓ TCGA data found: {tcga_file}")
        else:
            print(f"⚠️  TCGA data not found. Run tcga_prnp_real_download.py first.")
        
        # Run GEO miner (if not already done)
        geo_output = self.data_dir / "open_data/geo/geo_all_prnp_combined.csv"
        if not geo_output.exists():
            success = self.run_script("geo_prpc_miner.py", "GEO Database Mining")
            if not success:
                return False
        else:
            print(f"✓ GEO data already exists: {geo_output}")
        
        # Run cBioPortal query
        cbio_output = self.data_dir / "open_data/cbioportal/cbioportal_prnp_with_clinical.csv"
        if not cbio_output.exists():
            success = self.run_script("cbioportal_prpc_query.py", "cBioPortal API Query")
            if not success:
                return False
        else:
            print(f"✓ cBioPortal data already exists: {cbio_output}")
        
        return True
    
    def phase_2_harmonization(self):
        """Phase 2: Data Harmonization"""
        print("\n" + "="*80)
        print("🔧 PHASE 2: DATA HARMONIZATION")
        print("="*80)
        
        return self.run_script("harmonize_datasets.py", "Data Harmonization & QC")
    
    def phase_3_training(self):
        """Phase 3: Model Training"""
        print("\n" + "="*80)
        print("🤖 PHASE 3: ML MODEL TRAINING")
        print("="*80)
        
        return self.run_script("prpc_transfer_learning.py", "ADDS Transfer Learning")
    
    def phase_4_validation(self):
        """Phase 4: Validation (placeholder for Week 9-10)"""
        print("\n" + "="*80)
        print("🔬 PHASE 4: VALIDATION")
        print("="*80)
        print("⚠️  Validation scripts will be created in Week 9")
        return True
    
    def run_all(self):
        """Run complete pipeline"""
        print("\n" + "="*80)
        print("🚀 v3.0 AI-FIRST PIPELINE - FULL EXECUTION")
        print("="*80)
        print(f"Start time: {self.log['start_time']}")
        
        # Phase 1: Mining
        if not self.phase_1_mining():
            print("\n❌ Pipeline stopped: Mining failed")
            self.save_log()
            return False
        
        # Phase 2: Harmonization
        if not self.phase_2_harmonization():
            print("\n❌ Pipeline stopped: Harmonization failed")
            self.save_log()
            return False
        
        # Phase 3: Training (will be enabled when data is ready)
        # if not self.phase_3_training():
        #     print("\n❌ Pipeline stopped: Training failed")
        #     self.save_log()
        #     return False
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE!")
        print("="*80)
        
        self.log["end_time"] = datetime.now().isoformat()
        self.log["status"] = "success"
        
        self.save_log()
        self.print_summary()
        
        return True
    
    def save_log(self):
        """Save pipeline log"""
        log_file = self.data_dir / "pipeline_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.log, f, indent=2)
        print(f"\n📝 Pipeline log saved: {log_file}")
    
    def print_summary(self):
        """Print execution summary"""
        print("\n" + "="*80)
        print("📊 PIPELINE SUMMARY")
        print("="*80)
        
        for script, info in self.log["phases"].items():
            status_icon = "✅" if info["status"] == "success" else "❌"
            duration = info.get("duration_seconds", 0)
            print(f"{status_icon} {script}: {info['status']} ({duration:.1f}s)")
        
        total_time = 0
        for info in self.log["phases"].values():
            total_time += info.get("duration_seconds", 0)
        
        print(f"\nTotal execution time: {total_time/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description="v3.0 AI-First Pipeline Orchestrator")
    parser.add_argument(
        "--phase",
        choices=["all", "mining", "harmonization", "training", "validation"],
        default="all",
        help="Which phase to run"
    )
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator()
    
    if args.phase == "all":
        orchestrator.run_all()
    elif args.phase == "mining":
        orchestrator.phase_1_mining()
    elif args.phase == "harmonization":
        orchestrator.phase_2_harmonization()
    elif args.phase == "training":
        orchestrator.phase_3_training()
    elif args.phase == "validation":
        orchestrator.phase_4_validation()


if __name__ == "__main__":
    main()
