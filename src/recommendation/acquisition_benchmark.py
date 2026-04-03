"""
ADDS-Exscientia Integration: Phase 2-1B
A/B Testing Benchmark for Acquisition Functions

벤치마크 목표:
- 5가지 함수 성능 비교
- Simple Regret, Cumulative Regret 측정
- 수렴 속도 분석
- 의학적 시나리오 테스트

Author: ADDS Development Team  
Date: 2026-01-15
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend for saving plots
import matplotlib.pyplot as plt
from typing import List, Dict, Callable, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

try:
    from .acquisition_functions import AcquisitionFunctionRegistry
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from recommendation.acquisition_functions import AcquisitionFunctionRegistry

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    function_name: str
    test_function_name: str
    repeat_id: int
    simple_regret: float  # 최종 regret
    cumulative_regret: float  # 누적 regret
    convergence_iteration: int  # 수렴한 iteration
    best_value_found: float
    iterations: List[float]  # iteration별 best value


class TestFunction:
    """
    최적화 테스트 함수
    
    의료 시나리오를 시뮬레이션:
    - 2D: (약물 A 용량, 약물 B 용량)
    - 목표: 효능 최대화
    """
    
    def __init__(self, name: str, dim: int = 2):
        self.name = name
        self.dim = dim
        self.global_optimum = None
        self.global_optimum_value = None
    
    def __call__(self, x: np.ndarray) -> float:
        """평가"""
        raise NotImplementedError


class DrugCombinationTestFunction(TestFunction):
    """
    약물 조합 시뮬레이션 함수
    
    f(x1, x2) = efficacy - toxicity
    - efficacy: 용량에 비례, saturation curve
    - toxicity: 용량 제곱에 비례
    """
    
    def __init__(self):
        super().__init__("DrugCombination", dim=2)
        
        # 최적점: 적절한 균형
        self.global_optimum = np.array([0.6, 0.7])
        self.global_optimum_value = self(self.global_optimum)
    
    def __call__(self, x: np.ndarray) -> float:
        """
        Args:
            x: [dose1, dose2] (0-1 normalized)
        
        Returns:
            Overall score (higher is better)
        """
        x = np.clip(x, 0, 1)
        
        # Efficacy (Emax model)
        emax = 1.0
        ec50_1, ec50_2 = 0.5, 0.5
        efficacy = (
            emax * x[0] / (ec50_1 + x[0]) +
            emax * x[1] / (ec50_2 + x[1])
        ) / 2.0
        
        # Toxicity (quadratic)
        toxicity = 0.3 * (x[0]**2 + x[1]**2)
        
        # Overall score
        score = efficacy - toxicity
        
        # Add noise (realistic)
        noise = np.random.normal(0, 0.05)
        
        return score + noise


class CancerResponseTestFunction(TestFunction):
    """
    암 반응 시뮬레이션
    
    복잡한 landscape (여러 local minima)
    """
    
    def __init__(self):
        super().__init__("CancerResponse", dim=2)
        self.global_optimum = np.array([0.75, 0.65])
        self.global_optimum_value = self(self.global_optimum)
    
    def __call__(self, x: np.ndarray) -> float:
        x = np.clip(x, 0, 1)
        
        # 복잡한 반응 곡면
        response = (
            np.sin(5 * x[0]) * np.cos(5 * x[1]) +
            2 * np.exp(-((x[0]-0.75)**2 + (x[1]-0.65)**2) / 0.1)
        )
        
        # Toxicity penalty
        toxicity = 0.5 * (x[0]**2 + x[1]**2)
        
        score = response - toxicity
        noise = np.random.normal(0, 0.1)
        
        return score + noise


class AcquisitionBenchmark:
    """
    Acquisition Function A/B 테스트 벤치마크
    
    의료급 검증:
    - 재현성 (random seed)
    - 여러 시나리오 테스트
    - 통계적 유의성
    """
    
    def __init__(
        self,
        test_functions: List[TestFunction],
        output_dir: str = "benchmark_results"
    ):
        """
        Args:
            test_functions: 테스트할 함수 리스트
            output_dir: 결과 저장 디렉토리
        """
        self.test_functions = test_functions
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = []
        
        logger.info(f"Benchmark initialized with {len(test_functions)} test functions")
    
    def run_benchmark(
        self,
        acquisition_functions: List[str],
        n_iterations: int = 20,
        n_repeats: int = 5,
        initial_samples: int = 3
    ) -> List[BenchmarkResult]:
        """
        전체 벤치마크 실행
        
        Args:
            acquisition_functions: 테스트할 함수 이름 리스트
            n_iterations: Active Learning iteration 수
            n_repeats: 반복 횟수 (통계적 유의성)
            initial_samples: 초기 랜덤 샘플 수
        
        Returns:
            결과 리스트
        """
        logger.info(f"Running benchmark: {n_iterations} iterations, {n_repeats} repeats")
        
        registry = AcquisitionFunctionRegistry()
        
        total_runs = len(acquisition_functions) * len(self.test_functions) * n_repeats
        current_run = 0
        
        for acq_name in acquisition_functions:
            for test_func in self.test_functions:
                for repeat in range(n_repeats):
                    current_run += 1
                    logger.info(
                        f"[{current_run}/{total_runs}] "
                        f"{acq_name} on {test_func.name} (repeat {repeat+1})"
                    )
                    
                    result = self._simulate_optimization(
                        acq_name,
                        test_func,
                        registry,
                        n_iterations,
                        initial_samples,
                        random_seed=repeat
                    )
                    
                    self.results.append(BenchmarkResult(
                        function_name=acq_name,
                        test_function_name=test_func.name,
                        repeat_id=repeat,
                        simple_regret=result['simple_regret'],
                        cumulative_regret=result['cumulative_regret'],
                        convergence_iteration=result['convergence_iteration'],
                        best_value_found=result['best_value'],
                        iterations=result['best_per_iteration']
                    ))
        
        logger.info(f"Benchmark complete: {len(self.results)} results")
        return self.results
    
    def _simulate_optimization(
        self,
        acq_name: str,
        test_func: TestFunction,
        registry: AcquisitionFunctionRegistry,
        n_iterations: int,
        initial_samples: int,
        random_seed: int
    ) -> Dict:
        """
        단일 최적화 시뮬레이션
        
        Active Learning 프로세스:
        1. 초기 랜덤 샘플링
        2. GP 학습
        3. Acquisition Function으로 다음 포인트 선택
        4. 평가 및 업데이트
        """
        np.random.seed(random_seed)
        
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        
        # 1. 초기 샘플
        X_train = np.random.rand(initial_samples, test_func.dim)
        y_train = np.array([test_func(x) for x in X_train])
        
        best_so_far = []
        current_best = y_train.max()
        
        # 2. Active Learning Loop
        for iteration in range(n_iterations):
            # GP 학습
            kernel = Matern(nu=2.5, length_scale=1.0)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.1,
                n_restarts_optimizer=5,
                random_state=random_seed
            )
            gp.fit(X_train, y_train)
            
            # 후보 포인트 생성
            n_candidates = 100
            X_candidates = np.random.rand(n_candidates, test_func.dim)
            
            # Acquisition Function 평가
            acq_scores = []
            for x_cand in X_candidates:
                mean, std = gp.predict(x_cand.reshape(1, -1), return_std=True)
                
                score = registry.evaluate(
                    acq_name,
                    mean[0],
                    std[0],
                    best_so_far=current_best,
                    iteration=iteration + 1
                )
                acq_scores.append(score)
            
            # 최고 점수 포인트 선택
            best_idx = np.argmax(acq_scores)
            x_next = X_candidates[best_idx]
            y_next = test_func(x_next)
            
            # 데이터 업데이트
            X_train = np.vstack([X_train, x_next])
            y_train = np.append(y_train, y_next)
            
            # Best tracking
            current_best = max(current_best, y_next)
            best_so_far.append(current_best)
        
        # 3. 결과 계산
        simple_regret = test_func.global_optimum_value - current_best
        cumulative_regret = sum(
            test_func.global_optimum_value - b for b in best_so_far
        )
        
        # 수렴 iteration 찾기 (95% 도달)
        threshold = test_func.global_optimum_value * 0.95
        convergence_iter = n_iterations
        for i, val in enumerate(best_so_far):
            if val >= threshold:
                convergence_iter = i + 1
                break
        
        return {
            'simple_regret': simple_regret,
            'cumulative_regret': cumulative_regret,
            'convergence_iteration': convergence_iter,
            'best_value': current_best,
            'best_per_iteration': best_so_far
        }
    
    def plot_results(self):
        """결과 시각화"""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # 1. Regret curves
        self._plot_regret_curves()
        
        # 2. Box plots
        self._plot_comparison_boxplots()
        
        # 3. Convergence comparison
        self._plot_convergence()
        
        logger.info(f"Plots saved to {self.output_dir}")
    
    def _plot_regret_curves(self):
        """Regret 곡선"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 함수별로 그룹화
        for test_func in self.test_functions:
            ax = axes[0] if test_func.name == self.test_functions[0].name else axes[1]
            
            for acq_name in set(r.function_name for r in self.results):
                # 해당 조합의 모든 반복
                curves = [
                    r.iterations for r in self.results
                    if r.function_name == acq_name and r.test_function_name == test_func.name
                ]
                
                if not curves:
                    continue
                
                # 평균 및 표준편차
                mean_curve = np.mean(curves, axis=0)
                std_curve = np.std(curves, axis=0)
                
                iterations = range(1, len(mean_curve) + 1)
                
                ax.plot(iterations, mean_curve, label=acq_name, linewidth=2)
                ax.fill_between(
                    iterations,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    alpha=0.2
                )
            
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Best Value Found', fontsize=12)
            ax.set_title(f'Convergence - {test_func.name}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'regret_curves.png', dpi=150)
        plt.close()
    
    def _plot_comparison_boxplots(self):
        """Box plot 비교"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Simple Regret
        simple_regrets = {}
        cumulative_regrets = {}
        
        for result in self.results:
            key = result.function_name
            if key not in simple_regrets:
                simple_regrets[key] = []
                cumulative_regrets[key] = []
            
            simple_regrets[key].append(result.simple_regret)
            cumulative_regrets[key].append(result.cumulative_regret)
        
        # Plot
        axes[0].boxplot(
            simple_regrets.values(),
            labels=simple_regrets.keys(),
            patch_artist=True
        )
        axes[0].set_ylabel('Simple Regret', fontsize=12)
        axes[0].set_title('Final Regret Comparison', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        axes[1].boxplot(
            cumulative_regrets.values(),
            labels=cumulative_regrets.keys(),
            patch_artist=True
        )
        axes[1].set_ylabel('Cumulative Regret', fontsize=12)
        axes[1].set_title('Cumulative Regret Comparison', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'boxplot_comparison.png', dpi=150)
        plt.close()
    
    def _plot_convergence(self):
        """수렴 속도 비교"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        convergence_iters = {}
        for result in self.results:
            key = result.function_name
            if key not in convergence_iters:
                convergence_iters[key] = []
            convergence_iters[key].append(result.convergence_iteration)
        
        means = [np.mean(v) for v in convergence_iters.values()]
        stds = [np.std(v) for v in convergence_iters.values()]
        names = list(convergence_iters.keys())
        
        x = range(len(names))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Iterations to Convergence', fontsize=12)
        ax.set_title('Convergence Speed Comparison', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'convergence_comparison.png', dpi=150)
        plt.close()
    
    def print_summary(self):
        """요약 통계 출력"""
        if not self.results:
            print("No results available")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80 + "\n")
        
        # 함수별 평균 성능
        func_stats = {}
        for result in self.results:
            func = result.function_name
            if func not in func_stats:
                func_stats[func] = {
                    'simple_regret': [],
                    'cumulative_regret': [],
                    'convergence': [],
                    'best_value': []
                }
            
            func_stats[func]['simple_regret'].append(result.simple_regret)
            func_stats[func]['cumulative_regret'].append(result.cumulative_regret)
            func_stats[func]['convergence'].append(result.convergence_iteration)
            func_stats[func]['best_value'].append(result.best_value_found)
        
        for func, stats in sorted(func_stats.items()):
            print(f"{func}:")
            print(f"  Simple Regret:     {np.mean(stats['simple_regret']):.4f} ± {np.std(stats['simple_regret']):.4f}")
            print(f"  Cumulative Regret: {np.mean(stats['cumulative_regret']):.4f} ± {np.std(stats['cumulative_regret']):.4f}")
            print(f"  Convergence Iter:  {np.mean(stats['convergence']):.1f} ± {np.std(stats['convergence']):.1f}")
            print(f"  Best Value:        {np.mean(stats['best_value']):.4f} ± {np.std(stats['best_value']):.4f}")
            print()
        
        # 최고 성능 함수
        best_func = min(
            func_stats.items(),
            key=lambda x: np.mean(x[1]['simple_regret'])
        )
        print(f"Best Overall: {best_func[0]} (lowest simple regret)")
        print("="*80 + "\n")


# 실행
if __name__ == "__main__":
    print("\n" + "="*80)
    print("ACQUISITION FUNCTION A/B TESTING BENCHMARK")
    print("="*80 + "\n")
    
    # 테스트 함수
    test_functions = [
        DrugCombinationTestFunction(),
        CancerResponseTestFunction()
    ]
    
    # 벤치마크
    benchmark = AcquisitionBenchmark(
        test_functions=test_functions,
        output_dir="benchmark_results"
    )
    
    # 실행
    acquisition_functions = [
        'expected_improvement',
        'upper_confidence_bound',
        'probability_of_improvement',
        'thompson_sampling',
        'entropy_search_simplified'
    ]
    
    results = benchmark.run_benchmark(
        acquisition_functions=acquisition_functions,
        n_iterations=20,
        n_repeats=3,  # 빠른 테스트용 3회
        initial_samples=3
    )
    
    # 결과
    benchmark.print_summary()
    benchmark.plot_results()
    
    print("\n[OK] Benchmark complete! Check 'benchmark_results/' for plots.")
