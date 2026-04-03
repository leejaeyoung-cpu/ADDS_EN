"""
Model evaluation utilities for ADDS
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..utils import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluate model performance with various metrics and validation strategies
    """
    
    def __init__(self):
        """Initialize evaluator"""
        logger.info("✓ Model evaluator initialized")
    
    def cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        stratified: bool = False,
        shuffle: bool = True,
        random_state: int = 42
    ) -> Dict[str, any]:
        """
        Perform k-fold cross-validation
        
        Args:
            model: Model with fit() and predict() methods
            X: Features
            y: Targets
            n_folds: Number of folds
            stratified: Use stratified k-fold
            shuffle: Shuffle data
            random_state: Random seed
        
        Returns:
            Cross-validation results
        """
        if stratified:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        else:
            kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        
        fold_scores = {
            'mse': [],
            'mae': [],
            'r2': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            fold_scores['mse'].append(mean_squared_error(y_val, y_pred))
            fold_scores['mae'].append(mean_absolute_error(y_val, y_pred))
            fold_scores['r2'].append(r2_score(y_val, y_pred))
            
            logger.info(f"Fold {fold+1}/{n_folds}: MSE={fold_scores['mse'][-1]:.4f}, "
                       f"MAE={fold_scores['mae'][-1]:.4f}, R²={fold_scores['r2'][-1]:.4f}")
        
        # Calculate mean and std
        results = {
            'mse_mean': np.mean(fold_scores['mse']),
            'mse_std': np.std(fold_scores['mse']),
            'mae_mean': np.mean(fold_scores['mae']),
            'mae_std': np.std(fold_scores['mae']),
            'r2_mean': np.mean(fold_scores['r2']),
            'r2_std': np.std(fold_scores['r2']),
            'fold_scores': fold_scores
        }
        
        logger.info(f"✓ Cross-validation completed: MSE={results['mse_mean']:.4f}±{results['mse_std']:.4f}")
        return results
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            confidence_scores: Optional confidence scores
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        metrics['mape'] = mape
        
        # Pearson correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        metrics['correlation'] = correlation
        
        if confidence_scores is not None:
            # Confidence calibration
            metrics['mean_confidence'] = np.mean(confidence_scores)
            
        return metrics
    
    def ablation_study(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Perform ablation study by removing features one at a time
        
        Args:
            model: Model to evaluate
            X: Full feature matrix
            y: Targets
            feature_names: Names of features
            test_size: Test set size
            random_state: Random seed
        
        Returns:
            DataFrame with ablation results
        """
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Baseline with all features
        model.fit(X_train, y_train)
        baseline_score = r2_score(y_test, model.predict(X_test))
        
        logger.info(f"Baseline R² (all features): {baseline_score:.4f}")
        
        results = []
        
        # Test removing each feature
        for i, feature_name in enumerate(feature_names):
            # Create dataset without feature i
            mask = [j != i for j in range(len(feature_names))]
            X_train_ablated = X_train[:, mask]
            X_test_ablated = X_test[:, mask]
            
            # Train and evaluate
            model.fit(X_train_ablated, y_train)
            score = r2_score(y_test, model.predict(X_test_ablated))
            
            # Calculate importance as drop in performance
            importance = baseline_score - score
            
            results.append({
                'feature': feature_name,
                'r2_without': score,
                'importance': importance,
                'performance_drop_%': (importance / baseline_score * 100) if baseline_score != 0 else 0
            })
            
            logger.info(f"Without '{feature_name}': R²={score:.4f}, Drop={importance:.4f}")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('importance', ascending=False)
        
        logger.info("✓ Ablation study completed")
        return results_df
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Path] = None,
        title: str = "Predicted vs Actual"
    ):
        """
        Plot predicted vs actual values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save plot
            title: Plot title
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='Perfect prediction')
        axes[0].set_xlabel('Actual Values', fontsize=12)
        axes[0].set_ylabel('Predicted Values', fontsize=12)
        axes[0].set_title(title, fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        axes[0].text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f}',
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].set_title('Residual Plot', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✓ Plot saved to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: np.ndarray,
        save_path: Optional[Path] = None,
        top_k: int = 20
    ):
        """
        Plot feature importance
        
        Args:
            feature_names: Names of features
            importance_scores: Importance scores
            save_path: Path to save plot
            top_k: Number of top features to show
        """
        # Sort by importance
        indices = np.argsort(importance_scores)[::-1][:top_k]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance_scores[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_k} Feature Importance', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✓ Feature importance plot saved to {save_path}")
        
        plt.close()


class BiologicalValidator:
    """
    Validate model predictions against biological knowledge
    """
    
    def __init__(self, known_interactions: Optional[Dict] = None):
        """
        Initialize biological validator
        
        Args:
            known_interactions: Dictionary of known drug interactions
        """
        self.known_interactions = known_interactions or {}
        logger.info("✓ Biological validator initialized")
    
    def validate_predictions(
        self,
        predictions: List[Dict[str, any]],
        tolerance: float = 0.2
    ) -> Dict[str, any]:
        """
        Validate predictions against known biological interactions
        
        Args:
            predictions: List of prediction dictionaries
            tolerance: Acceptable deviation from known values
        
        Returns:
            Validation report
        """
        validated = 0
        consistent = 0
        inconsistent = []
        
        for pred in predictions:
            drug_a = pred.get('drug_a')
            drug_b = pred.get('drug_b')
            predicted_synergy = pred.get('predicted_synergy')
            
            key = f"{drug_a}+{drug_b}"
            reverse_key = f"{drug_b}+{drug_a}"
            
            known_value = self.known_interactions.get(key) or \
                         self.known_interactions.get(reverse_key)
            
            if known_value is not None:
                validated += 1
                
                if abs(predicted_synergy - known_value) <= tolerance:
                    consistent += 1
                else:
                    inconsistent.append({
                        'combination': key,
                        'predicted': predicted_synergy,
                        'known': known_value,
                        'difference': abs(predicted_synergy - known_value)
                    })
        
        report = {
            'total_predictions': len(predictions),
            'validated_against_known': validated,
            'consistent': consistent,
            'inconsistent': len(inconsistent),
            'consistency_rate': (consistent / validated * 100) if validated > 0 else 0,
            'inconsistent_details': inconsistent[:10]  # Show first 10
        }
        
        logger.info(f"Biological validation: {consistent}/{validated} consistent "
                   f"({report['consistency_rate']:.1f}%)")
        
        return report
