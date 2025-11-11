"""
Model Trainer Component
Handles model training, hyperparameter tuning, and evaluation.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import save_object, get_artifacts_path


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



