"""
Data Ingestion Component
Handles loading data from CSV and splitting into train/validation/test sets.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import get_datasets_path


class DataIngestionConfig:
    """Configuration for data ingestion."""
    def __init__(self):
        self.dataset_path = get_datasets_path() / "candidate_job_role_dataset.csv"
        self.train_size = 0.7
        self.val_size = 0.15
        self.test_size = 0.15
        self.random_state = 42


class DataIngestion:
    """Handles data loading and splitting."""
    
    def __init__(self):
        self.config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        """
        Load the dataset and split into train, validation, and test sets.
        
        Returns:
            tuple: (train_df, val_df, test_df) - Three pandas DataFrames
            
        Raises:
            Exception: If there's an error during data ingestion
        """
        print("=" * 80)
        print("Starting Data Ingestion")
        print("=" * 80)
        
        try:
            # Load the dataset
            print(f"Loading dataset from: {self.config.dataset_path}")
            df = pd.read_csv(self.config.dataset_path)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            
            # Display basic info
            print(f"\nColumns: {list(df.columns)}")
            print(f"Number of unique job roles: {df['job_role'].nunique()}")
            print(f"\nJob role distribution:")
            print(df['job_role'].value_counts())
            
            # Check for missing values
            print(f"\nMissing values:")
            print(df.isnull().sum())
            
            # Check class distribution and filter out classes with insufficient samples
            class_counts = df['job_role'].value_counts()
            min_samples_required = 3  # Need at least 3 samples for nested stratified splits
            
            insufficient_classes = class_counts[class_counts < min_samples_required]
            if len(insufficient_classes) > 0:
                print(f"\n⚠ WARNING: Removing {len(insufficient_classes)} job role(s) with < {min_samples_required} samples:")
                for job_role, count in insufficient_classes.items():
                    print(f"  - {job_role}: {count} sample(s)")
                
                # Filter out these classes
                df = df[~df['job_role'].isin(insufficient_classes.index)]
                print(f"\n✓ Filtered dataset size: {len(df)} samples")
                print(f"✓ Remaining job roles: {df['job_role'].nunique()}")
            
            # First split: separate test set (15%)
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=df['job_role']
            )
            
            # Second split: separate validation set from remaining data
            # val_size relative to train_val is 0.15/0.85 = ~0.176
            relative_val_size = self.config.val_size / (self.config.train_size + self.config.val_size)
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=relative_val_size,
                random_state=self.config.random_state,
                stratify=train_val_df['job_role']
            )
            
            print(f"\n{'Dataset Split Summary':^80}")
            print("-" * 80)
            print(f"Training set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
            print(f"Validation set size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
            print(f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
            print(f"Total: {len(train_df) + len(val_df) + len(test_df)}")
            print("-" * 80)
            
            print("\nData Ingestion completed successfully!")
            print("=" * 80)
            
            return train_df, val_df, test_df
            
        except Exception as e:
            raise Exception(f"Error occurred during data ingestion: {str(e)}")


if __name__ == "__main__":
    # Test the data ingestion
    ingestion = DataIngestion()
    train_df, val_df, test_df = ingestion.initiate_data_ingestion()
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")



