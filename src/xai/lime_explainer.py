"""
LIME (Local Interpretable Model-agnostic Explanations) for ADDS
Explains individual patient predictions in a clinically interpretable manner
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class LIMEExplainer:
    """
    LIME-based explainer for ADDS predictions
    
    Provides local explanations for individual patient predictions,
    showing which features contributed most to the prediction.
    """
    
    def __init__(
        self,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        mode: str = 'classification'
    ):
        """
        Initialize LIME explainer
        
        Args:
            feature_names: List of feature names (e.g., pathology features, clinical variables)
            class_names: List of class names (e.g., ['Low Response', 'High Response'])
            mode: 'classification' or 'regression'
        """
        self.feature_names = feature_names
        self.class_names = class_names or ['Negative', 'Positive']
        self.mode = mode
        self.explainer = None
        
    def fit(self, X_train: np.ndarray, categorical_features: Optional[List[int]] = None):
        """
        Fit LIME explainer on training data
        
        Args:
            X_train: Training features (N x D array)
            categorical_features: Indices of categorical features
        """
        self.explainer = LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            categorical_features=categorical_features or [],
            mode=self.mode,
            discretize_continuous=True
        )
        
    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: callable,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Dict[str, Any]:
        """
        Explain a single prediction
        
        Args:
            instance: Feature vector for one patient (D-dimensional)
            predict_fn: Model's predict_proba or predict function
            num_features: Number of top features to show
            num_samples: Number of perturbed samples for LIME
            
        Returns:
            Dictionary containing:
                - explanation: LIME explanation object
                - top_features: List of (feature_name, contribution) tuples
                - prediction: Model prediction
                - intercept: Base prediction
        """
        if self.explainer is None:
            raise ValueError("Must call fit() before explain_instance()")
        
        # Generate explanation
        exp = self.explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Extract top features
        if self.mode == 'classification':
            # For binary classification, get explanation for positive class
            top_features = exp.as_list(label=1)
        else:
            top_features = exp.as_list()
        
        # Get prediction
        prediction = predict_fn(instance.reshape(1, -1))
        
        return {
            'explanation': exp,
            'top_features': top_features,
            'prediction': prediction,
            'intercept': exp.intercept[1] if self.mode == 'classification' else exp.intercept,
            'local_pred': exp.local_pred[0] if hasattr(exp, 'local_pred') else None
        }
    
    def explain_clinical(
        self,
        instance: np.ndarray,
        predict_fn: callable,
        feature_values: Optional[Dict[str, float]] = None,
        num_features: int = 10
    ) -> str:
        """
        Generate clinical explanation in natural language
        
        Args:
            instance: Patient feature vector
            predict_fn: Model prediction function
            feature_values: Optional dict of feature name -> actual value for display
            num_features: Number of features to explain
            
        Returns:
            Clinical explanation string
        """
        result = self.explain_instance(instance, predict_fn, num_features)
        
        prediction = result['prediction']
        top_features = result['top_features']
        
        # Format prediction
        if self.mode == 'classification':
            pred_class = self.class_names[1] if prediction[0][1] > 0.5 else self.class_names[0]
            prob = prediction[0][1] if prediction[0][1] > 0.5 else prediction[0][0]
            pred_str = f"{pred_class} (확률: {prob:.1%})"
        else:
            pred_str = f"{prediction[0]:.2f}"
        
        # Build clinical explanation
        explanation_lines = [
            f"**환자 예측**: {pred_str}",
            "",
            "**주요 기여 요인**:",
            ""
        ]
        
        positive_factors = []
        negative_factors = []
        
        for feature_desc, contribution in top_features:
            # Parse feature description (e.g., "cell_density > 120.5")
            if '>' in feature_desc or '<' in feature_desc or '=' in feature_desc:
                feature_name = feature_desc.split()[0]
            else:
                feature_name = feature_desc
            
            # Get actual value if provided
            if feature_values and feature_name in feature_values:
                actual = feature_values[feature_name]
                value_str = f" (실제값: {actual:.2f})"
            else:
                value_str = ""
            
            # Categorize as positive or negative contribution
            if contribution > 0:
                positive_factors.append(f"  ✓ {feature_desc}{value_str} → +{contribution:.3f}")
            else:
                negative_factors.append(f"  ✗ {feature_desc}{value_str} → {contribution:.3f}")
        
        if positive_factors:
            explanation_lines.append("긍정적 영향 (반응률 증가):")
            explanation_lines.extend(positive_factors)
            explanation_lines.append("")
        
        if negative_factors:
            explanation_lines.append("부정적 영향 (반응률 감소):")
            explanation_lines.extend(negative_factors)
        
        return "\n".join(explanation_lines)
    
    def plot_explanation(
        self,
        instance: np.ndarray,
        predict_fn: callable,
        num_features: int = 10,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create visualization of LIME explanation
        
        Args:
            instance: Patient feature vector
            predict_fn: Model prediction function
            num_features: Number of features to show
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        result = self.explain_instance(instance, predict_fn, num_features)
        top_features = result['top_features']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        features = [f.split()[0] for f, _ in top_features][::-1]  # Reverse for bottom-to-top
        contributions = [c for _, c in top_features][::-1]
        colors = ['green' if c > 0 else 'red' for c in contributions]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        ax.barh(y_pos, contributions, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Contribution to Prediction', fontsize=12)
        ax.set_title('LIME Explanation - Feature Contributions', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels
        for i, (feat, contrib) in enumerate(zip(features, contributions)):
            label_x = contrib + (0.01 if contrib > 0 else -0.01)
            ha = 'left' if contrib > 0 else 'right'
            ax.text(label_x, i, f'{contrib:.3f}', va='center', ha=ha, fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_explanations(
        self,
        instances: List[np.ndarray],
        predict_fn: callable,
        labels: Optional[List[str]] = None,
        num_features: int = 10
    ) -> pd.DataFrame:
        """
        Compare LIME explanations for multiple patients
        
        Args:
            instances: List of patient feature vectors
            predict_fn: Model prediction function
            labels: Optional labels for each patient
            num_features: Number of features to compare
            
        Returns:
            DataFrame with comparison of feature contributions
        """
        if labels is None:
            labels = [f"Patient {i+1}" for i in range(len(instances))]
        
        # Get all feature names that appear in any explanation
        all_features = set()
        explanations = []
        
        for instance in instances:
            result = self.explain_instance(instance, predict_fn, num_features)
            explanations.append(result)
            for feat, _ in result['top_features']:
                feature_name = feat.split()[0]
                all_features.add(feature_name)
        
        # Build comparison dataframe
        comparison_data = {feat: [] for feat in all_features}
        comparison_data['Patient'] = labels
        comparison_data['Prediction'] = []
        
        for i, (exp, instance) in enumerate(zip(explanations, instances)):
            pred = predict_fn(instance.reshape(1, -1))
            comparison_data['Prediction'].append(pred[0][1] if len(pred[0]) > 1 else pred[0])
            
            # Create feature -> contribution mapping
            contrib_map = {}
            for feat_desc, contrib in exp['top_features']:
                feat_name = feat_desc.split()[0]
                contrib_map[feat_name] = contrib
            
            # Fill in contributions
            for feat in all_features:
                comparison_data[feat].append(contrib_map.get(feat, 0.0))
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        df = df[['Patient', 'Prediction'] + sorted(all_features)]
        
        return df


def demo_lime_explainer():
    """
    Demonstration of LIME explainer with synthetic data
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic patient data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: [cell_density, nucleus_area, N/C_ratio, cluster_coef, texture_entropy]
    X = np.random.randn(n_samples, 5)
    
    # Outcome: High cell_density + high N/C ratio → low response
    y = (X[:, 0] > 0.5) & (X[:, 2] > 0.3)
    y = (~y).astype(int)  # Invert for positive class
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize LIME
    feature_names = ['cell_density', 'nucleus_area', 'N/C_ratio', 'cluster_coef', 'texture_entropy']
    lime_exp = LIMEExplainer(
        feature_names=feature_names,
        class_names=['Low Response', 'High Response']
    )
    lime_exp.fit(X_train)
    
    # Explain test instance
    test_instance = X_test[0]
    explanation = lime_exp.explain_clinical(
        test_instance,
        model.predict_proba,
        feature_values={name: val for name, val in zip(feature_names, test_instance)}
    )
    
    print("=" * 70)
    print("LIME Explanation Demo")
    print("=" * 70)
    print(explanation)
    print("=" * 70)
    
    # Create visualization
    fig = lime_exp.plot_explanation(test_instance, model.predict_proba)
    plt.show()
    
    # Compare multiple patients
    comparison = lime_exp.compare_explanations(
        X_test[:3],
        model.predict_proba,
        labels=['Patient A', 'Patient B', 'Patient C']
    )
    print("\nComparison of Multiple Patients:")
    print(comparison.to_string())


if __name__ == '__main__':
    demo_lime_explainer()
