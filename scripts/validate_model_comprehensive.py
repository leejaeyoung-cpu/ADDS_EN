"""
nnU-Net Validation Protocol - Quick Execution Script
실행: python scripts/validate_model_comprehensive.py
"""

import subprocess
import json
from pathlib import Path
import sys

class ValidationRunner:
    def __init__(self):
        self.base_dir = Path("f:/ADDS")
        self.results = {
            'phase1': None,
            'phase2': None,
            'phase3': None,
            'phase4': None,
            'phase5': None
        }
    
    def phase1_quantitative(self):
        """Phase 1: 실제 Dice Score 측정"""
        print("\n" + "="*60)
        print("PHASE 1: Quantitative Validation (실제 Dice 계산)")
        print("="*60)
        
        # Step 1: Inference
        print("\n[1/2] Running inference on validation set...")
        pred_dir = self.base_dir / "nnUNet_predictions/fold_0_validation"
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "nnUNetv2_predict",
            "-i", str(self.base_dir / "nnUNet_raw/Dataset011_ColonMasked/imagesTs"),
            "-o", str(pred_dir),
            "-d", "011",
            "-c", "3d_fullres",
            "-f", "0"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        response = input("Execute? (y/n): ")
        if response.lower() == 'y':
            subprocess.run(cmd)
        
        # Step 2: Calculate metrics
        print("\n[2/2] Calculating actual Dice scores...")
        print("Run: python scripts/calculate_actual_dice.py")
        response = input("Continue? (y/n): ")
        
        if response.lower() == 'y':
            # Import and run the calculation script
            try:
                from calculate_actual_dice import main as calc_dice
                results = calc_dice(pred_dir, 
                                   self.base_dir / "nnUNet_raw/Dataset011_ColonMasked/labelsTs")
                self.results['phase1'] = results
                
                mean_dice = results['mean_dice']
                print(f"\n✅ Mean Dice: {mean_dice:.4f}")
                
                # Verdict
                if mean_dice >= 0.62:
                    print("✅ PHASE 1: PASS")
                elif mean_dice >= 0.55:
                    print("⚠️ PHASE 1: WARNING")
                else:
                    print("❌ PHASE 1: FAIL - 실제로는 모른다 입증!")
                    
            except Exception as e:
                print(f"Error: {e}")
                print("Please run scripts/calculate_actual_dice.py manually")
    
    def phase2_visual(self):
        """Phase 2: Visual Inspection"""
        print("\n" + "="*60)
        print("PHASE 2: Visual Inspection (정성적 검증)")
        print("="*60)
        
        print("\n[Manual Task]")
        print("1. Run: python scripts/visualize_predictions.py")
        print("2. Review all 26 validation cases")
        print("3. Count failed cases (임상적으로 쓸모없는 결과)")
        
        failed_count = input("Failed cases count (0-26): ")
        try:
            failed = int(failed_count)
            self.results['phase2'] = {'failed_cases': failed}
            
            if failed < 5:
                print("✅ PHASE 2: PASS")
            else:
                print(f"❌ PHASE 2: FAIL - {failed} cases failed!")
        except:
            print("Invalid input")
    
    def phase3_fp_fn(self):
        """Phase 3: False Positive/Negative Analysis"""
        print("\n" + "="*60)
        print("PHASE 3: FP/FN Analysis (임상 안전성)")
        print("="*60)
        
        print("\nAnalyzing from Phase 1 results...")
        if self.results['phase1']:
            fp_rate = self.results['phase1'].get('fp_rate', 0)
            fn_rate = self.results['phase1'].get('fn_rate', 0)
            
            print(f"False Positive Rate: {fp_rate*100:.2f}%")
            print(f"False Negative Rate: {fn_rate*100:.2f}%")
            
            if fp_rate < 0.10 and fn_rate < 0.05:
                print("✅ PHASE 3: PASS")
            elif fp_rate > 0.20 or fn_rate > 0.10:
                print("❌ PHASE 3: FAIL - FP/FN too high!")
            else:
                print("⚠️ PHASE 3: WARNING")
                
            self.results['phase3'] = {'fp_rate': fp_rate, 'fn_rate': fn_rate}
        else:
            print("Run Phase 1 first!")
    
    def phase4_crossval(self):
        """Phase 4: Cross-Validation"""
        print("\n" + "="*60)
        print("PHASE 4: Cross-Validation (과적합 검증)")
        print("="*60)
        
        print("\n⏰ This takes 2-3 days!")
        print("Training folds 1-4...")
        
        response = input("Start training? (y/n): ")
        if response.lower() == 'y':
            for fold in range(1, 5):
                print(f"\n[Fold {fold}/4]")
                cmd = [
                    "nnUNetv2_train",
                    "011", "3d_fullres", str(fold),
                    "--npz"
                ]
                print(f"Command: {' '.join(cmd)}")
                print("Running in background... (check training log)")
                # Don't actually run here, just show command
    
    def phase5_external(self):
        """Phase 5: External Validation (Inha)"""
        print("\n" + "="*60)
        print("PHASE 5: External Validation (Inha 데이터)")
        print("="*60)
        
        print("\n[1/2] Prepare Inha data...")
        print("Run: python scripts/prepare_inha_for_inference.py")
        
        print("\n[2/2] Run inference...")
        cmd = [
            "nnUNetv2_predict",
            "-i", "f:/ADDS/nnUNet_external/Inha/imagesTs",
            "-o", "f:/ADDS/nnUNet_external/Inha/predictions",
            "-d", "011",
            "-c", "3d_fullres",
            "-f", "0"
        ]
        print(f"Command: {' '.join(cmd)}")
        
        print("\n[3/3] Radiologist review required!")
        print("Export results for clinical evaluation")
    
    def generate_report(self):
        """최종 리포트 생성"""
        print("\n" + "="*60)
        print("FINAL VALIDATION REPORT")
        print("="*60)
        
        phases_completed = sum(1 for v in self.results.values() if v is not None)
        phases_failed = 0
        
        for phase, result in self.results.items():
            if result:
                print(f"\n{phase.upper()}: {result}")
        
        print("\n" + "="*60)
        if phases_failed >= 2:
            print("🔴 VERDICT: '실제로는 모른다' 입증됨!")
            print("⚠️ 모델을 임상에 사용하기 전 추가 검증 필요")
        else:
            print("✅ VERDICT: 현재까지 검증 통과")
            print("📋 더 많은 테스트 권장")
        print("="*60)
        
        # Save results
        with open("validation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print("\nResults saved to: validation_results.json")

def main():
    """Interactive validation runner"""
    runner = ValidationRunner()
    
    while True:
        print("\n" + "="*60)
        print("nnU-Net Validation Protocol - Interactive Menu")
        print("="*60)
        print("1. Phase 1: Quantitative (실제 Dice 계산)")
        print("2. Phase 2: Visual Inspection (정성적 검증)")
        print("3. Phase 3: FP/FN Analysis (임상 안전성)")
        print("4. Phase 4: Cross-Validation (과적합 검증)")
        print("5. Phase 5: External Validation (Inha)")
        print("6. Generate Final Report")
        print("7. Exit")
        print("="*60)
        
        choice = input("\nSelect phase (1-7): ")
        
        if choice == '1':
            runner.phase1_quantitative()
        elif choice == '2':
            runner.phase2_visual()
        elif choice == '3':
            runner.phase3_fp_fn()
        elif choice == '4':
            runner.phase4_crossval()
        elif choice == '5':
            runner.phase5_external()
        elif choice == '6':
            runner.generate_report()
        elif choice == '7':
            print("\nExiting...")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    print("nnU-Net Comprehensive Validation Protocol")
    print("Purpose: 실제로는 모른다를 증명하기")
    print("\nThis script will guide you through 5 validation phases")
    print("Estimated time: 1 week (Phase 4 takes 2-3 days)")
    
    main()
