#!/usr/bin/env python3
"""
ADDS 성능 측정 및 벤치마크 스크립트
Nature Communications 출판을 위한 메트릭 계산
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

class ADDSBenchmark:
    """ADDS 시스템 성능 측정"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_dice(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Dice Similarity Coefficient
        
        Target: >0.85 for Nature-tier
        """
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        dice = 2 * intersection / union
        return dice
    
    def calculate_icc(self, measurements: List[np.ndarray]) -> float:
        """Intraclass Correlation Coefficient
        
        Target: >0.90 for scan-rescan stability
        
        Args:
            measurements: List of repeated measurements (e.g., from different scans)
        """
        # 간단한 ICC(2,1) 구현
        k = len(measurements)
        n = len(measurements[0])
        
        # Grand mean
        grand_mean = np.mean([np.mean(m) for m in measurements])
        
        # Between-subject variance
        subject_means = np.mean(np.array(measurements), axis=0)
        BSS = n * np.sum((subject_means - grand_mean) ** 2)
        
        # Within-subject variance  
        WSS = np.sum([np.sum((m - subject_means) ** 2) for m in measurements])
        
        # ICC calculation
        BMS = BSS / (n - 1)
        WMS = WSS / (n * (k - 1))
        
        icc = (BMS - WMS) / (BMS + (k - 1) * WMS)
        return max(0, min(1, icc))  # Clamp to [0, 1]
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, 
                     bins: int = 10) -> float:
        """Population Stability Index
        
        Target: <0.1 for scanner drift robustness
        
        Measures distribution shift between expected and actual data
        """
        # Create bins
        breakpoints = np.linspace(np.min(expected), np.max(expected), bins + 1)
        
        # Calculate proportions
        expected_counts, _ = np.histogram(expected, bins=breakpoints)
        actual_counts, _ = np.histogram(actual, bins=breakpoints)
        
        expected_prop = expected_counts / len(expected)
        actual_prop = actual_counts / len(actual)
        
        # Avoid division by zero
        expected_prop = np.where(expected_prop == 0, 0.0001, expected_prop)
        actual_prop = np.where(actual_prop == 0, 0.0001, actual_prop)
        
        # PSI calculation
        psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
        
        return psi
    
    def calculate_metrics_batch(self, 
                                predictions: List[np.ndarray],
                                ground_truths: List[np.ndarray]) -> Dict:
        """배치 예측에 대한 전체 메트릭 계산"""
        
        # Dice coefficients
        dice_scores = [
            self.calculate_dice(pred, gt) 
            for pred, gt in zip(predictions, ground_truths)
        ]
        
        results = {
            'dice': {
                'mean': np.mean(dice_scores),
                'std': np.std(dice_scores),
                'min': np.min(dice_scores),
                'max': np.max(dice_scores),
                'target': 0.85,
                'pass': np.mean(dice_scores) > 0.85
            },
            'n_samples': len(predictions)
        }
        
        self.results.update(results)
        return results
    
    def generate_report(self, output_path: str):
        """벤치마크 리포트 생성"""
        
        # JSON 저장
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 콘솔 출력
        print("\n" + "="*60)
        print("ADDS Performance Benchmark Report")
        print("="*60)
        
        if 'dice' in self.results:
            dice = self.results['dice']
            print(f"\nDice Similarity Coefficient:")
            print(f"  Mean: {dice['mean']:.3f} ± {dice['std']:.3f}")
            print(f"  Range: [{dice['min']:.3f}, {dice['max']:.3f}]")
            print(f"  Target: >{dice['target']}")
            print(f"  Status: {'✓ PASS' if dice['pass'] else '✗ FAIL'}")
        
        if 'icc' in self.results:
            print(f"\nIntraclass Correlation:")
            print(f"  ICC: {self.results['icc']:.3f}")
            print(f"  Target: >0.90")
            print(f"  Status: {'✓ PASS' if self.results['icc'] > 0.90 else '✗ FAIL'}")
        
        if 'psi' in self.results:
            print(f"\nPopulation Stability Index:")
            print(f"  PSI: {self.results['psi']:.3f}")
            print(f"  Target: <0.1")
            print(f"  Status: {'✓ PASS' if self.results['psi'] < 0.1 else '✗ FAIL'}")
        
        print(f"\n{'='*60}")
        print(f"Full report saved to: {output_path}")
        print("="*60 + "\n")

# Example usage
if __name__ == '__main__':
    print("ADDS Benchmark Script")
    print("Usage:")
    print("  1. Load your predictions and ground truths")
    print("  2. benchmark = ADDSBenchmark()")
    print("  3. results = benchmark.calculate_metrics_batch(preds, gts)")
    print("  4. benchmark.generate_report('benchmark_report.json')")
    print("\nThis script calculates Nature-tier metrics:")
    print("  - Dice >0.85")
    print("  - ICC >0.90")
    print("  - PSI <0.1")
