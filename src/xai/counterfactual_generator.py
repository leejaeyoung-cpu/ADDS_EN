"""
Counterfactual Explanations for ADDS
Shows "what-if" scenarios: how to change predictions through feature modifications
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class CounterfactualGenerator:
    """
    Generate counterfactual explanations for clinical AI predictions
    
    Answers: "What needs to change for different outcome?"
    Example: "If cell density decreased from 120 to 90, ORR would increase from 38% to 62%"
    """
    
    def __init__(
        self,
        feature_names: List[str],
        continuous_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        immutable_features: Optional[List[str]] = None
    ):
        """
        Initialize counterfactual generator
        
        Args:
            feature_names: All feature names
            continuous_features: Features that can change continuously (e.g., cell_density)
            categorical_features: Features with discrete values (e.g., cancer_type)
            immutable_features: Features that cannot change (e.g., age, sex)
        """
        self.feature_names = feature_names
        self.continuous_features = continuous_features or []
        self.categorical_features = categorical_features or []
        self.immutable_features = immutable_features or []
        
        # Mutable features = all features except immutable
        self.mutable_features = [
            f for f in feature_names 
            if f not in immutable_features
        ]
    
    def generate_simple_counterfactual(
        self,
        instance: np.ndarray,
        predict_fn: callable,
        target_class: int,
        max_iterations: int = 1000,
        step_size: float = 0.1,
        proximity_weight: float = 0.5,
        diversity: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactual using gradient-based approach
        
        Args:
            instance: Original patient features
            predict_fn: Model's predict_proba function
            target_class: Desired outcome (0 or 1)
            max_iterations: Max optimization steps
            step_size: Learning rate
            proximity_weight: Weight for proximity loss (vs prediction loss)
            diversity: Number of diverse counterfactuals to generate
            
        Returns:
            List of counterfactual dictionaries
        """
        counterfactuals = []
        
        for div_idx in range(diversity):
            # Start from original instance
            cf = instance.copy()
            
            # Add small random perturbation for diversity
            if div_idx > 0:
                noise = np.random.randn(len(cf)) * 0.1
                cf += noise
            
            best_cf = None
            best_distance = float('inf')
            
            for iter_idx in range(max_iterations):
                # Get current prediction
                pred = predict_fn(cf.reshape(1, -1))[0]
                current_prob = pred[target_class]
                
                # Check if target reached
                if current_prob > 0.5:
                    distance = np.linalg.norm(cf - instance)
                    if distance < best_distance:
                        best_cf = cf.copy()
                        best_distance = distance
                
                # Compute gradient approximation (finite differences)
                gradient = np.zeros_like(cf)
                epsilon = 1e-5
                
                for i, feat_name in enumerate(self.feature_names):
                    # Skip immutable features
                    if feat_name in self.immutable_features:
                        continue
                    
                    # Perturb feature
                    cf_plus = cf.copy()
                    cf_plus[i] += epsilon
                    
                    pred_plus = predict_fn(cf_plus.reshape(1, -1))[0][target_class]
                    gradient[i] = (pred_plus - current_prob) / epsilon
                
                # Update counterfactual (gradient ascent for target class probability)
                cf += step_size * gradient
                
                # Add proximity penalty (stay close to original)
                proximity_grad = -(cf - instance) * proximity_weight
                cf += step_size * proximity_grad
                
                # Project immutable features back
                for i, feat_name in enumerate(self.feature_names):
                    if feat_name in self.immutable_features:
                        cf[i] = instance[i]
            
            # Use best counterfactual found
            if best_cf is not None:
                cf_dict = self._create_cf_dict(instance, best_cf, predict_fn, target_class)
                counterfactuals.append(cf_dict)
        
        return counterfactuals
    
    def _create_cf_dict(
        self,
        original: np.ndarray,
        counterfactual: np.ndarray,
        predict_fn: callable,
        target_class: int
    ) -> Dict[str, Any]:
        """Create counterfactual dictionary with comparison"""
        
        original_pred = predict_fn(original.reshape(1, -1))[0]
        cf_pred = predict_fn(counterfactual.reshape(1, -1))[0]
        
        # Find changed features
        changes = {}
        for i, feat_name in enumerate(self.feature_names):
            if abs(original[i] - counterfactual[i]) > 1e-6:
                changes[feat_name] = {
                    'original': original[i],
                    'counterfactual': counterfactual[i],
                    'change': counterfactual[i] - original[i],
                    'change_pct': ((counterfactual[i] - original[i]) / (abs(original[i]) + 1e-10)) * 100
                }
        
        return {
            'counterfactual': counterfactual,
            'changes': changes,
            'original_prediction': original_pred,
            'cf_prediction': cf_pred,
            'distance': np.linalg.norm(counterfactual - original),
            'num_changes': len(changes)
        }
    
    def explain_counterfactual_clinical(
        self,
        cf_dict: Dict[str, Any],
        class_names: Optional[List[str]] = None,
        outcome_metric: str = "ORR"
    ) -> str:
        """
        Generate clinical explanation of counterfactual
        
        Args:
            cf_dict: Counterfactual dictionary from generate_simple_counterfactual
            class_names: Class labels (e.g., ['Low Response', 'High Response'])
            outcome_metric: Outcome name (e.g., 'ORR', 'PFS')
            
        Returns:
            Clinical explanation string
        """
        class_names = class_names or ['Class 0', 'Class 1']
        
        original_pred = cf_dict['original_prediction']
        cf_pred = cf_dict['cf_prediction']
        changes = cf_dict['changes']
        
        # Format predictions
        orig_class = np.argmax(original_pred)
        orig_prob = original_pred[orig_class]
        cf_class = np.argmax(cf_pred)
        cf_prob = cf_pred[cf_class]
        
        # Build explanation
        lines = [
            "## 🔄 Counterfactual Analysis: 'What-If' Scenario",
            "",
            f"**현재 상태**: {class_names[orig_class]} ({outcome_metric}: {orig_prob:.1%})",
            f"**목표 상태**: {class_names[cf_class]} ({outcome_metric}: {cf_prob:.1%})",
            f"**개선**: {outcome_metric} {orig_prob:.1%} → {cf_prob:.1%} (+{(cf_prob - orig_prob)*100:.1f}%p)",
            "",
            "### 📋 필요한 변화:",
            ""
        ]
        
        # Sort changes by magnitude
        sorted_changes = sorted(
            changes.items(),
            key=lambda x: abs(x[1]['change']),
            reverse=True
        )
        
        for feat_name, change_info in sorted_changes:
            orig = change_info['original']
            cf = change_info['counterfactual']
            delta = change_info['change']
            pct = change_info['change_pct']
            
            direction = "증가" if delta > 0 else "감소"
            arrow = "↑" if delta > 0 else "↓"
            
            lines.append(
                f"- **{feat_name}**: {orig:.2f} → {cf:.2f} "
                f"({arrow} {abs(delta):.2f}, {abs(pct):.1f}% {direction})"
            )
        
        lines.extend([
            "",
            f"**총 변경 특징 수**: {len(changes)}개",
            f"**변화 거리**: {cf_dict['distance']:.3f}",
            "",
            "### 💡 임상 해석:",
            ""
        ])
        
        # Add clinical interpretation
        actionable = [f for f in changes.keys() if f not in self.immutable_features]
        immutable_changed = [f for f in changes.keys() if f in self.immutable_features]
        
        if actionable:
            lines.append(f"조치 가능한 변화 ({len(actionable)}개): {', '.join(actionable)}")
        
        if immutable_changed:
            lines.append(f"⚠️ 변경 불가 특징: {', '.join(immutable_changed)} (참고용)")
        
        return "\n".join(lines)
    
    def find_minimal_changes(
        self,
        instance: np.ndarray,
        predict_fn: callable,
        target_class: int,
        max_features: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Find minimal set of feature changes to flip prediction
        
        Args:
            instance: Original patient
            predict_fn: Model prediction function
            target_class: Desired outcome
            max_features: Maximum number of features to change
            
        Returns:
            Counterfactual with minimal changes or None
        """
        from itertools import combinations
        
        # Try changing 1, 2, 3, ... features
        for num_changes in range(1, max_features + 1):
            # Try all combinations of mutable features
            mutable_indices = [
                i for i, f in enumerate(self.feature_names)
                if f in self.mutable_features
            ]
            
            for feature_combo in combinations(mutable_indices, num_changes):
                # Try different change magnitudes
                for magnitude in [0.1, 0.2, 0.5, 1.0]:
                    for direction in [-1, 1]:
                        cf = instance.copy()
                        
                        # Apply changes
                        for idx in feature_combo:
                            cf[idx] += direction * magnitude
                        
                        # Check if target reached
                        pred = predict_fn(cf.reshape(1, -1))[0]
                        if pred[target_class] > 0.5:
                            return self._create_cf_dict(instance, cf, predict_fn, target_class)
        
        return None
    
    def plot_counterfactual(
        self,
        cf_dict: Dict[str, Any],
        save_path: Optional[Path] = None,
        top_n: int = 10
    ) -> plt.Figure:
        """
        Visualize counterfactual changes
        
        Args:
            cf_dict: Counterfactual dictionary
            save_path: Optional save path
            top_n: Show top N changes
            
        Returns:
            Matplotlib figure
        """
        changes = cf_dict['changes']
        
        # Sort by absolute change
        sorted_changes = sorted(
            changes.items(),
            key=lambda x: abs(x[1]['change']),
            reverse=True
        )[:top_n]
        
        # Prepare data
        features = [f for f, _ in sorted_changes]
        original_vals = [c['original'] for _, c in sorted_changes]
        cf_vals = [c['counterfactual'] for _, c in sorted_changes]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(features))
        width = 0.35
        
        bars1 = ax.barh(x - width/2, original_vals, width, label='Original', alpha=0.7)
        bars2 = ax.barh(x + width/2, cf_vals, width, label='Counterfactual', alpha=0.7)
        
        ax.set_yticks(x)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Value')
        ax.set_title('Counterfactual: Feature Changes Required', fontsize=14, fontweight='bold')
        ax.legend()
        
        # Add change annotations
        for i, (feat, change_info) in enumerate(sorted_changes):
            delta = change_info['change']
            mid_x = (original_vals[i] + cf_vals[i]) / 2
            ax.text(mid_x, i, f'{delta:+.2f}', ha='center', va='center',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def demo_counterfactual():
    """Demonstration of counterfactual generator"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    X = np.random.randn(n_samples, 5)
    feature_names = ['cell_density', 'nucleus_area', 'N/C_ratio', 'PrPc_score', 'age']
    
    # Outcome (high cell_density + high PrPc → low response)
    y = ((X[:, 0] > 0.5) & (X[:, 3] > 0.3)).astype(int)
    y = (~y.astype(bool)).astype(int)  # Invert
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize counterfactual generator
    cf_gen = CounterfactualGenerator(
        feature_names=feature_names,
        continuous_features=['cell_density', 'nucleus_area', 'N/C_ratio', 'PrPc_score'],
        immutable_features=['age']  # Age cannot be changed
    )
    
    # Select a low-response patient
    test_instance = X_test[y_test == 0][0]
    
    print("=" * 70)
    print("Original Patient:")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {test_instance[i]:.2f}")
    
    original_pred = model.predict_proba(test_instance.reshape(1, -1))[0]
    print(f"\nOriginal Prediction: {original_pred[1]:.1%} High Response")
    
    # Generate counterfactual
    print("\nGenerating counterfactual...")
    cfs = cf_gen.generate_simple_counterfactual(
        test_instance,
        model.predict_proba,
        target_class=1,  # Want high response
        diversity=1
    )
    
    if cfs:
        cf = cfs[0]
        explanation = cf_gen.explain_counterfactual_clinical(
            cf,
            class_names=['Low Response', 'High Response']
        )
        
        print("\n" + "=" * 70)
        print(explanation)
        print("=" * 70)
        
        # Visualization
        fig = cf_gen.plot_counterfactual(cf)
        plt.show()
    else:
        print("No counterfactual found")


if __name__ == '__main__':
    demo_counterfactual()
