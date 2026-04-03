"""
Statistics Service
Real statistical analysis with scipy.stats
Supports: t-test, Mann-Whitney, ANOVA, Kruskal-Wallis, post-hoc tests
"""

import numpy as np
from typing import Dict, List, Optional, Any
from scipy import stats
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


class StatsService:
    """Service for statistical group comparisons"""
    
    SUPPORTED_TESTS = {
        "parametric": ["t-test", "paired-t", "anova"],
        "non-parametric": ["mann-whitney", "kruskal-wallis", "wilcoxon"],
        "corrections": ["bonferroni", "fdr-bh", "holm"]
    }
    
    async def compare(
        self,
        groups: List[List[float]],
        features: List[str],
        test_type: str = "auto",
        correction: str = "bonferroni"
    ) -> Dict[str, Any]:
        """
        Perform statistical group comparison
        
        Args:
            groups: List of data arrays (one per group)
            features: Feature names being compared
            test_type: "auto", "t-test", "anova", "mann-whitney", "kruskal-wallis"
            correction: Multiple testing correction method
            
        Returns:
            Dict with p_values, effect_sizes, test_used, and post_hoc results
        """
        if len(groups) < 2:
            raise ValueError("At least 2 groups required for comparison")
        
        # Convert to numpy arrays and clean NaN
        clean_groups = []
        for g in groups:
            arr = np.array(g, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) < 2:
                raise ValueError(f"Group has fewer than 2 valid observations")
            clean_groups.append(arr)
        
        # Auto-select test type
        if test_type == "auto":
            test_type = self._auto_select_test(clean_groups)
        
        logger.info(f"Running {test_type} test on {len(clean_groups)} groups")
        
        # Run the appropriate test
        if len(clean_groups) == 2:
            result = self._two_group_test(clean_groups[0], clean_groups[1], test_type)
        else:
            result = self._multi_group_test(clean_groups, test_type, correction)
        
        result["test_used"] = test_type
        result["n_groups"] = len(clean_groups)
        result["group_sizes"] = [len(g) for g in clean_groups]
        result["features"] = features
        
        return result
    
    def _auto_select_test(self, groups: List[np.ndarray]) -> str:
        """Auto-select test based on normality and group count"""
        # Test normality for each group (Shapiro-Wilk)
        all_normal = True
        normality_results = []
        
        for i, g in enumerate(groups):
            if len(g) < 3:
                # Too few samples for normality test
                all_normal = False
                normality_results.append({"group": i, "normal": False, "reason": "n < 3"})
                continue
            if len(g) > 5000:
                # Shapiro-Wilk limited to 5000
                g_sample = np.random.choice(g, 5000, replace=False)
            else:
                g_sample = g
            
            stat, p = stats.shapiro(g_sample)
            is_normal = p > 0.05
            if not is_normal:
                all_normal = False
            normality_results.append({
                "group": i, "normal": is_normal,
                "shapiro_stat": float(stat), "shapiro_p": float(p)
            })
        
        if len(groups) == 2:
            return "t-test" if all_normal else "mann-whitney"
        else:
            return "anova" if all_normal else "kruskal-wallis"
    
    def _two_group_test(
        self, a: np.ndarray, b: np.ndarray, test_type: str
    ) -> Dict[str, Any]:
        """Run two-group comparison"""
        result: Dict[str, Any] = {}
        
        if test_type == "t-test":
            # Levene's test for equal variances
            lev_stat, lev_p = stats.levene(a, b)
            equal_var = lev_p > 0.05
            
            stat, p = stats.ttest_ind(a, b, equal_var=equal_var)
            result["statistic"] = float(stat)
            result["p_value"] = float(p)
            result["equal_variance"] = equal_var
            result["levene_p"] = float(lev_p)
            
        elif test_type == "paired-t":
            if len(a) != len(b):
                raise ValueError("Paired t-test requires equal group sizes")
            stat, p = stats.ttest_rel(a, b)
            result["statistic"] = float(stat)
            result["p_value"] = float(p)
            
        elif test_type == "mann-whitney":
            stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            result["statistic"] = float(stat)
            result["p_value"] = float(p)
            # Rank-biserial correlation as effect size
            n1, n2 = len(a), len(b)
            result["rank_biserial"] = float(1 - (2 * stat) / (n1 * n2))
            
        elif test_type == "wilcoxon":
            if len(a) != len(b):
                raise ValueError("Wilcoxon test requires equal group sizes")
            stat, p = stats.wilcoxon(a, b)
            result["statistic"] = float(stat)
            result["p_value"] = float(p)
        else:
            raise ValueError(f"Unsupported two-group test: {test_type}")
        
        # Effect size: Cohen's d
        pooled_std = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + 
                              (len(b) - 1) * np.var(b, ddof=1)) / 
                             (len(a) + len(b) - 2))
        if pooled_std > 0:
            cohens_d = float((np.mean(a) - np.mean(b)) / pooled_std)
        else:
            cohens_d = 0.0
        
        result["effect_sizes"] = {
            "cohens_d": cohens_d,
            "interpretation": self._interpret_cohens_d(abs(cohens_d))
        }
        
        # Descriptive stats
        result["descriptive"] = {
            "group_0": {"mean": float(np.mean(a)), "std": float(np.std(a, ddof=1)),
                        "median": float(np.median(a)), "n": len(a)},
            "group_1": {"mean": float(np.mean(b)), "std": float(np.std(b, ddof=1)),
                        "median": float(np.median(b)), "n": len(b)}
        }
        
        # Significance
        result["significant"] = result["p_value"] < 0.05
        
        return result
    
    def _multi_group_test(
        self, groups: List[np.ndarray], test_type: str, correction: str
    ) -> Dict[str, Any]:
        """Run multi-group comparison with post-hoc"""
        result: Dict[str, Any] = {}
        
        if test_type == "anova":
            stat, p = stats.f_oneway(*groups)
            result["statistic"] = float(stat)
            result["p_value"] = float(p)
            
            # Eta-squared effect size
            grand_mean = np.mean(np.concatenate(groups))
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            ss_total = sum(np.sum((g - grand_mean) ** 2) for g in groups)
            eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
            result["effect_sizes"] = {
                "eta_squared": eta_sq,
                "interpretation": self._interpret_eta_squared(eta_sq)
            }
            
        elif test_type == "kruskal-wallis":
            stat, p = stats.kruskal(*groups)
            result["statistic"] = float(stat)
            result["p_value"] = float(p)
            
            # Epsilon-squared effect size
            N = sum(len(g) for g in groups)
            k = len(groups)
            epsilon_sq = float((stat - k + 1) / (N - k)) if N > k else 0.0
            result["effect_sizes"] = {
                "epsilon_squared": max(0, epsilon_sq),
                "interpretation": self._interpret_eta_squared(max(0, epsilon_sq))
            }
        else:
            raise ValueError(f"Unsupported multi-group test: {test_type}")
        
        result["significant"] = result["p_value"] < 0.05
        
        # Post-hoc pairwise comparisons (only if omnibus is significant)
        if result["significant"]:
            result["post_hoc"] = self._pairwise_posthoc(groups, test_type, correction)
        else:
            result["post_hoc"] = None
        
        # Descriptive stats
        result["descriptive"] = {}
        for i, g in enumerate(groups):
            result["descriptive"][f"group_{i}"] = {
                "mean": float(np.mean(g)), "std": float(np.std(g, ddof=1)),
                "median": float(np.median(g)), "n": len(g)
            }
        
        return result
    
    def _pairwise_posthoc(
        self, groups: List[np.ndarray], test_type: str, correction: str
    ) -> List[Dict]:
        """Pairwise post-hoc comparisons with correction"""
        pairs = list(combinations(range(len(groups)), 2))
        raw_p_values = []
        pair_results = []
        
        for i, j in pairs:
            if test_type == "anova":
                stat, p = stats.ttest_ind(groups[i], groups[j])
            else:
                stat, p = stats.mannwhitneyu(
                    groups[i], groups[j], alternative='two-sided'
                )
            raw_p_values.append(p)
            pair_results.append({
                "group_i": i, "group_j": j,
                "statistic": float(stat), "p_raw": float(p)
            })
        
        # Apply correction
        corrected = self._correct_pvalues(raw_p_values, correction)
        
        for k, pr in enumerate(pair_results):
            pr["p_corrected"] = float(corrected[k])
            pr["significant"] = corrected[k] < 0.05
            pr["correction"] = correction
        
        return pair_results
    
    def _correct_pvalues(self, pvalues: List[float], method: str) -> np.ndarray:
        """Apply multiple testing correction"""
        p = np.array(pvalues)
        n = len(p)
        
        if method == "bonferroni":
            return np.minimum(p * n, 1.0)
        elif method == "holm":
            sorted_idx = np.argsort(p)
            sorted_p = p[sorted_idx]
            corrected = np.zeros(n)
            for i, idx in enumerate(sorted_idx):
                corrected[idx] = sorted_p[i] * (n - i)
            # Enforce monotonicity
            for i in range(1, n):
                idx = sorted_idx[i]
                prev_idx = sorted_idx[i - 1]
                corrected[idx] = max(corrected[idx], corrected[prev_idx])
            return np.minimum(corrected, 1.0)
        elif method == "fdr-bh":
            sorted_idx = np.argsort(p)
            sorted_p = p[sorted_idx]
            corrected = np.zeros(n)
            for i in range(n - 1, -1, -1):
                idx = sorted_idx[i]
                corrected[idx] = sorted_p[i] * n / (i + 1)
            # Enforce monotonicity (reverse)
            for i in range(n - 2, -1, -1):
                idx = sorted_idx[i]
                next_idx = sorted_idx[i + 1]
                corrected[idx] = min(corrected[idx], corrected[next_idx])
            return np.minimum(corrected, 1.0)
        else:
            return p
    
    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def _interpret_eta_squared(eta: float) -> str:
        if eta < 0.01:
            return "negligible"
        elif eta < 0.06:
            return "small"
        elif eta < 0.14:
            return "medium"
        else:
            return "large"
