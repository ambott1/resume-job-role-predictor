"""
Model Trainer Component
Handles model training, hyperparameter tuning, and evaluation.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (classification_report, accuracy_score, f1_score, confusion_matrix, 
                            precision_score, recall_score, precision_recall_curve, average_precision_score,
                            roc_curve, auc)
from sklearn.preprocessing import label_binarize

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import save_object, get_artifacts_path, get_project_root


class ModelTrainerConfig:
    """Configuration for model training."""
    def __init__(self):
        artifacts_path = get_artifacts_path()
        self.model_path = artifacts_path / "model.joblib"
        
        # Hyperparameter search space for XGBoost
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
        
        # Use RandomizedSearchCV for efficiency
        self.n_iter = 50  # Number of random combinations to try
        self.cv_folds = 3
        self.random_state = 42
        self.n_jobs = -1  # Use all available cores


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.model = None
        
    def initiate_model_training(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Train the model with hyperparameter tuning and evaluate.
        
        Args:
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Target labels
            
        Returns:
            dict: Dictionary containing model and evaluation metrics
        """
        print("=" * 80)
        print("Starting Model Training")
        print("=" * 80)
        
        try:
            # Combine train and validation for hyperparameter tuning
            X_train_val = np.vstack([X_train, X_val])
            y_train_val = np.hstack([y_train, y_val])
            
            print(f"\nTraining set shape: {X_train_val.shape}")
            print(f"Test set shape: {X_test.shape}")
            print(f"Number of classes: {len(np.unique(y_train_val))}")
            
            # Initialize base model
            print("\n[1/3] Initializing XGBoost Classifier...")
            print("-" * 80)
            
            base_model = XGBClassifier(
                random_state=self.config.random_state,
                eval_metric='mlogloss',
                use_label_encoder=False,
                tree_method='hist'  # Faster training
            )
            
            # Hyperparameter tuning with RandomizedSearchCV
            print("\n[2/3] Performing Hyperparameter Tuning...")
            print("-" * 80)
            print(f"Search method: RandomizedSearchCV")
            print(f"Number of iterations: {self.config.n_iter}")
            print(f"Cross-validation folds: {self.config.cv_folds}")
            print(f"Scoring metric: F1 (weighted)")
            print("\nThis may take several minutes...")
            
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self.config.param_grid,
                n_iter=self.config.n_iter,
                cv=self.config.cv_folds,
                scoring='f1_weighted',
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                verbose=2
            )
            
            # Fit the model
            random_search.fit(X_train_val, y_train_val)
            
            # Get best model
            self.model = random_search.best_estimator_
            
            print("\nHyperparameter tuning completed!")
            print(f"\nBest parameters:")
            for param, value in random_search.best_params_.items():
                print(f"  {param}: {value}")
            print(f"\nBest cross-validation F1 score: {random_search.best_score_:.4f}")
            
            # Evaluate on test set
            print("\n[3/3] Evaluating Model on Test Set...")
            print("-" * 80)
            
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)  # Get prediction probabilities
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
            test_f1_macro = f1_score(y_test, y_pred, average='macro')
            
            print(f"\nTest Set Performance:")
            print(f"  Accuracy: {test_accuracy:.4f}")
            print(f"  F1 Score (Weighted): {test_f1_weighted:.4f}")
            print(f"  F1 Score (Macro): {test_f1_macro:.4f}")
            
            print("\n" + "=" * 80)
            print("Classification Report (Test Set)")
            print("=" * 80)
            
            # Get class names from label encoder if available
            try:
                from src.utils import load_object
                label_encoder = load_object(get_artifacts_path() / "label_encoder.joblib")
                target_names = label_encoder.classes_
            except:
                target_names = None
            
            print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
            
            # Generate and save performance plots
            print("\n" + "-" * 80)
            print("Generating performance visualizations...")
            print("-" * 80)
            self._generate_performance_plots(y_test, y_pred, y_pred_proba, X_test, target_names, 
                                             test_accuracy, test_f1_weighted, test_f1_macro)
            
            # Save the model
            print("\n" + "-" * 80)
            print("Saving trained model...")
            print("-" * 80)
            save_object(self.config.model_path, self.model)
            
            print("\n" + "=" * 80)
            print("Model Training completed successfully!")
            print("=" * 80)
            
            # Return evaluation results
            results = {
                'model': self.model,
                'best_params': random_search.best_params_,
                'cv_f1_score': random_search.best_score_,
                'test_accuracy': test_accuracy,
                'test_f1_weighted': test_f1_weighted,
                'test_f1_macro': test_f1_macro,
                'classification_report': classification_report(y_test, y_pred, target_names=target_names, zero_division=0, output_dict=True)
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Error occurred during model training: {str(e)}")
    
    def _generate_performance_plots(self, y_test, y_pred, y_pred_proba, X_test, target_names=None, 
                                   test_accuracy=None, test_f1_weighted=None, test_f1_macro=None):
        """
        Generate and save performance visualization plots.
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            X_test: Test features (for recalculation if needed)
            target_names: List of class names (optional)
            test_accuracy: Test accuracy score
            test_f1_weighted: Test F1 weighted score
            test_f1_macro: Test F1 macro score
        """
        # Calculate metrics if not provided
        if test_accuracy is None:
            test_accuracy = accuracy_score(y_test, y_pred)
        if test_f1_weighted is None:
            test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
        if test_f1_macro is None:
            test_f1_macro = f1_score(y_test, y_pred, average='macro')
        # Create output directory
        project_root = get_project_root()
        output_dir = project_root / "imgs" / "model" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style (use seaborn default style for compatibility)
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                plt.style.use('default')
        sns.set_palette("husl")
        
        # Get class names
        if target_names is None:
            unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
            target_names = [f"Class {i}" for i in unique_classes]
        
        # 1. Confusion Matrix
        print("  • Generating confusion matrix...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Job Role', fontsize=12, fontweight='bold')
        plt.ylabel('Actual Job Role', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Actual vs Predicted Comparison
        print("  • Generating actual vs predicted comparison...")
        # Create a comparison dataframe
        comparison_df = pd.DataFrame({
            'Actual': [target_names[i] if i < len(target_names) else f"Class {i}" for i in y_test],
            'Predicted': [target_names[i] if i < len(target_names) else f"Class {i}" for i in y_pred]
        })
        
        # Count matches and mismatches
        comparison_df['Match'] = comparison_df['Actual'] == comparison_df['Predicted']
        
        # Create bar chart
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Match vs Mismatch
        match_counts = comparison_df['Match'].value_counts()
        axes[0].bar(['Correct Predictions', 'Incorrect Predictions'], 
                   [match_counts.get(True, 0), match_counts.get(False, 0)],
                   color=['#2ecc71', '#e74c3c'], alpha=0.8)
        axes[0].set_title('Prediction Accuracy Overview', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate([match_counts.get(True, 0), match_counts.get(False, 0)]):
            axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        # Plot 2: Distribution of actual vs predicted
        actual_counts = comparison_df['Actual'].value_counts().sort_index()
        predicted_counts = comparison_df['Predicted'].value_counts().sort_index()
        
        x = np.arange(len(target_names))
        width = 0.35
        
        axes[1].bar(x - width/2, [actual_counts.get(name, 0) for name in target_names], 
                   width, label='Actual', alpha=0.8, color='#3498db')
        axes[1].bar(x + width/2, [predicted_counts.get(name, 0) for name in target_names], 
                   width, label='Predicted', alpha=0.8, color='#9b59b6')
        
        axes[1].set_xlabel('Job Role', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Actual vs Predicted Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(target_names, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "actual_vs_predicted.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Per-Class Performance Metrics
        print("  • Generating per-class performance metrics...")
        report_dict = classification_report(y_test, y_pred, target_names=target_names, 
                                           zero_division=0, output_dict=True)
        
        # Extract per-class metrics
        class_metrics = []
        for name in target_names:
            if name in report_dict:
                class_metrics.append({
                    'Class': name,
                    'Precision': report_dict[name]['precision'],
                    'Recall': report_dict[name]['recall'],
                    'F1-Score': report_dict[name]['f1-score'],
                    'Support': report_dict[name]['support']
                })
        
        metrics_df = pd.DataFrame(class_metrics)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(metrics_df))
        width = 0.25
        
        ax.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8, color='#3498db')
        ax.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8, color='#2ecc71')
        ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8, color='#9b59b6')
        
        ax.set_xlabel('Job Role', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics (Precision, Recall, F1-Score)', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Class'], rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (prec, rec, f1) in enumerate(zip(metrics_df['Precision'], 
                                                 metrics_df['Recall'], 
                                                 metrics_df['F1-Score'])):
            ax.text(i - width, prec + 0.02, f'{prec:.2f}', ha='center', fontsize=8)
            ax.text(i, rec + 0.02, f'{rec:.2f}', ha='center', fontsize=8)
            ax.text(i + width, f1 + 0.02, f'{f1:.2f}', ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / "per_class_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Feature Importance (XGBoost)
        print("  • Generating feature importance plot...")
        try:
            feature_importance = self.model.feature_importances_
            
            # Get top 20 most important features
            top_n = min(20, len(feature_importance))
            top_indices = np.argsort(feature_importance)[-top_n:][::-1]
            top_importance = feature_importance[top_indices]
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(top_n), top_importance, color='#3498db', alpha=0.8)
            plt.yticks(range(top_n), [f'Feature {i}' for i in top_indices])
            plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
            plt.ylabel('Feature Index', fontsize=12, fontweight='bold')
            plt.title(f'Top {top_n} Feature Importances (XGBoost)', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(top_importance):
                plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Warning: Could not generate feature importance plot: {str(e)}")
        
        # 5. Overall Metrics Summary
        print("  • Generating overall metrics summary...")
        overall_metrics = {
            'Accuracy': test_accuracy,
            'F1 (Weighted)': test_f1_weighted,
            'F1 (Macro)': test_f1_macro
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_names = list(overall_metrics.keys())
        metrics_values = list(overall_metrics.values())
        
        bars = ax.bar(metrics_names, metrics_values, color=['#3498db', '#2ecc71', '#9b59b6'], 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Overall Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_dir / "overall_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Precision-Recall Curve (Multi-class)
        print("  • Generating precision-recall curve...")
        try:
            # Binarize labels for multi-class PR curve
            n_classes = len(target_names)
            y_test_binarized = label_binarize(y_test, classes=range(n_classes))
            
            # Calculate PR curve for each class (one-vs-rest)
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Micro-averaged PR curve
            y_test_micro = y_test_binarized.ravel()
            y_pred_proba_micro = y_pred_proba.ravel()
            precision_micro, recall_micro, _ = precision_recall_curve(y_test_micro, y_pred_proba_micro)
            avg_precision_micro = average_precision_score(y_test_micro, y_pred_proba_micro)
            
            ax.plot(recall_micro, precision_micro, 
                   label=f'Micro-average (AP = {avg_precision_micro:.3f})',
                   linewidth=2, color='#e74c3c')
            
            # Per-class PR curves (sample a few classes if too many)
            colors = plt.cm.Set3(np.linspace(0, 1, min(n_classes, 12)))
            for i in range(min(n_classes, 12)):  # Limit to 12 classes for readability
                precision, recall, _ = precision_recall_curve(
                    y_test_binarized[:, i], y_pred_proba[:, i]
                )
                avg_precision = average_precision_score(y_test_binarized[:, i], y_pred_proba[:, i])
                ax.plot(recall, precision, 
                       label=f'{target_names[i]} (AP = {avg_precision:.3f})',
                       linewidth=1.5, alpha=0.7, color=colors[i])
            
            ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
            ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
            ax.set_title('Precision-Recall Curve (Multi-class)', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlim([0.0, 1.05])
            ax.set_ylim([0.0, 1.05])
            
            plt.tight_layout()
            plt.savefig(output_dir / "precision_recall_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Warning: Could not generate precision-recall curve: {str(e)}")
        
        # 7. Prediction Probability Distribution
        print("  • Generating prediction probability distribution...")
        try:
            # Get max probability for each prediction
            max_probs = np.max(y_pred_proba, axis=1)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Distribution of maximum probabilities
            axes[0].hist(max_probs, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
            axes[0].axvline(np.mean(max_probs), color='#e74c3c', linestyle='--', 
                           linewidth=2, label=f'Mean: {np.mean(max_probs):.3f}')
            axes[0].axvline(np.median(max_probs), color='#2ecc71', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(max_probs):.3f}')
            axes[0].set_xlabel('Maximum Prediction Probability', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Frequency', fontsize=12)
            axes[0].set_title('Distribution of Maximum Prediction Probabilities', 
                            fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            # Plot 2: Prediction confidence vs accuracy
            # Group predictions by confidence bins
            bins = np.linspace(0, 1, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            accuracies = []
            counts = []
            
            for i in range(len(bins) - 1):
                mask = (max_probs >= bins[i]) & (max_probs < bins[i+1])
                if np.sum(mask) > 0:
                    correct = (y_test[mask] == y_pred[mask]).sum()
                    acc = correct / np.sum(mask)
                    accuracies.append(acc)
                    counts.append(np.sum(mask))
                else:
                    accuracies.append(0)
                    counts.append(0)
            
            axes[1].plot(bin_centers, accuracies, marker='o', linewidth=2, 
                        markersize=8, color='#9b59b6', label='Accuracy')
            axes[1].axhline(test_accuracy, color='#e74c3c', linestyle='--', 
                          linewidth=2, label=f'Overall Accuracy: {test_accuracy:.3f}')
            axes[1].set_xlabel('Prediction Confidence (Max Probability)', 
                             fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].set_title('Prediction Confidence vs Accuracy', 
                            fontsize=14, fontweight='bold')
            axes[1].set_ylim([0, 1.1])
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            
            # Add count annotations
            for i, (center, acc, count) in enumerate(zip(bin_centers, accuracies, counts)):
                if count > 0:
                    axes[1].annotate(f'n={count}', (center, acc), 
                                   textcoords="offset points", xytext=(0,10), 
                                   ha='center', fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(output_dir / "prediction_probability_distribution.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Warning: Could not generate prediction probability distribution: {str(e)}")
        
        # 8. Actual vs Predicted Scatter Plot
        print("  • Generating actual vs predicted scatter plot...")
        try:
            # For scatter plot, we can use the encoded labels directly since they're already numeric
            # But we need to ensure they're in the range [0, n_classes-1]
            y_test_encoded = y_test.copy()
            y_pred_encoded = y_pred.copy()
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Create scatter plot with jitter for better visibility
            jitter = 0.1
            y_test_jittered = y_test_encoded + np.random.normal(0, jitter, len(y_test_encoded))
            y_pred_jittered = y_pred_encoded + np.random.normal(0, jitter, len(y_pred_encoded))
            
            # Color points by correctness
            correct_mask = y_test_encoded == y_pred_encoded
            ax.scatter(y_test_jittered[correct_mask], y_pred_jittered[correct_mask], 
                      c='#2ecc71', alpha=0.6, s=50, label='Correct Predictions', edgecolors='black', linewidth=0.5)
            ax.scatter(y_test_jittered[~correct_mask], y_pred_jittered[~correct_mask], 
                      c='#e74c3c', alpha=0.6, s=50, label='Incorrect Predictions', edgecolors='black', linewidth=0.5)
            
            # Add diagonal line (perfect predictions)
            ax.plot([0, n_classes-1], [0, n_classes-1], 'k--', linewidth=2, alpha=0.5, label='Perfect Predictions')
            
            ax.set_xlabel('Actual Job Role (Encoded)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Predicted Job Role (Encoded)', fontsize=12, fontweight='bold')
            ax.set_title('Actual vs Predicted Job Roles', fontsize=14, fontweight='bold')
            ax.set_xticks(range(n_classes))
            ax.set_xticklabels(target_names, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(n_classes))
            ax.set_yticklabels(target_names, fontsize=8)
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "actual_vs_predicted_scatter.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Warning: Could not generate actual vs predicted scatter plot: {str(e)}")
        
        print(f"\n  ✓ All performance plots saved to: {output_dir}")


if __name__ == "__main__":
    # Test the model trainer
    from data_ingestion import DataIngestion
    from data_transformation import DataTransformation
    
    print("Testing Model Trainer...")
    
    print("\nStep 1: Data Ingestion")
    ingestion = DataIngestion()
    train_df, val_df, test_df = ingestion.initiate_data_ingestion()
    
    print("\n\nStep 2: Data Transformation")
    transformation = DataTransformation()
    X_train, X_val, X_test, y_train, y_val, y_test = transformation.initiate_data_transformation(
        train_df, val_df, test_df
    )
    
    print("\n\nStep 3: Model Training")
    trainer = ModelTrainer()
    results = trainer.initiate_model_training(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print(f"\n\nTraining completed!")
    print(f"Final test accuracy: {results['test_accuracy']:.4f}")
    print(f"Final test F1 (weighted): {results['test_f1_weighted']:.4f}")



