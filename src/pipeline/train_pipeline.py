"""
Training Pipeline
Orchestrates the complete training workflow from data ingestion to model training.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    """Complete training pipeline orchestration."""
    
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        
    def run_pipeline(self):
        """
        Execute the complete training pipeline.
        
        Returns:
            dict: Training results including model and evaluation metrics
        """
        print("\n" + "=" * 80)
        print("STARTING COMPLETE TRAINING PIPELINE")
        print("=" * 80)
        print("\nThis pipeline will:")
        print("  1. Ingest and split the data")
        print("  2. Transform features using embeddings and encodings")
        print("  3. Train an XGBoost model with hyperparameter tuning")
        print("  4. Evaluate the model and save all artifacts")
        print("\n" + "=" * 80)
        
        try:
            # Step 1: Data Ingestion
            print("\n\n")
            train_df, val_df, test_df = self.data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            print("\n\n")
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_transformation.initiate_data_transformation(
                train_df, val_df, test_df
            )
            
            # Step 3: Model Training
            print("\n\n")
            training_results = self.model_trainer.initiate_model_training(
                X_train, X_val, X_test, y_train, y_val, y_test
            )
            
            # Final Summary
            print("\n\n")
            print("=" * 80)
            print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("\nFinal Model Performance:")
            print(f"  • Test Accuracy: {training_results['test_accuracy']:.4f}")
            print(f"  • Test F1 Score (Weighted): {training_results['test_f1_weighted']:.4f}")
            print(f"  • Test F1 Score (Macro): {training_results['test_f1_macro']:.4f}")
            print(f"  • Cross-validation F1 Score: {training_results['cv_f1_score']:.4f}")
            
            print("\n\nSaved Artifacts:")
            print("  • artifacts/model.joblib")
            print("  • artifacts/standard_scaler.joblib")
            print("  • artifacts/qualification_ordinal_encoder.joblib")
            print("  • artifacts/experience_ordinal_encoder.joblib")
            print("  • artifacts/label_encoder.joblib")
            
            print("\n" + "=" * 80)
            print("You can now use the prediction pipeline for inference!")
            print("=" * 80 + "\n")
            
            return training_results
            
        except Exception as e:
            print("\n" + "=" * 80)
            print("ERROR: Training pipeline failed!")
            print("=" * 80)
            raise Exception(f"Training pipeline error: {str(e)}")


def main():
    """Main function to run the training pipeline."""
    pipeline = TrainingPipeline()
    results = pipeline.run_pipeline()
    return results


if __name__ == "__main__":
    main()


