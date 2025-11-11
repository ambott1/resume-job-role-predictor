"""
Data Transformation Component
Handles feature engineering and data transformation for the job role prediction system.
"""
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sentence_transformers import SentenceTransformer
import gensim.downloader as api

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import save_object, get_artifacts_path


class DataTransformationConfig:
    """Configuration for data transformation."""
    def __init__(self):
        artifacts_path = get_artifacts_path()
        
        # Paths for saving artifacts
        self.qualification_ordinal_encoder_path = artifacts_path / "qualification_ordinal_encoder.joblib"
        self.experience_ordinal_encoder_path = artifacts_path / "experience_ordinal_encoder.joblib"
        self.label_encoder_path = artifacts_path / "label_encoder.joblib"
        self.standard_scaler_path = artifacts_path / "standard_scaler.joblib"
        
        # Ordinal mappings
        self.degree_level_mapping = [['High School', "Bachelor's", "Master's", 'PhD']]
        self.experience_mapping = [['Entry', 'Mid', 'Senior']]
        
        # Model names
        self.sentence_transformer_model = 'all-MiniLM-L6-v2'
        self.word2vec_model = 'word2vec-google-news-300'


class DataTransformation:
    """Handles all feature engineering and data transformation."""
    
    def __init__(self):
        self.config = DataTransformationConfig()
        
        # Initialize models (will be loaded on demand)
        self.sentence_model = None
        self.word2vec_model = None
        
        # Initialize encoders
        self.qualification_ordinal_encoder = None
        self.experience_ordinal_encoder = None
        self.label_encoder = None
        self.scaler = None
        
    def _load_sentence_transformer(self):
        """Load the sentence transformer model."""
        if self.sentence_model is None:
            print(f"Loading Sentence Transformer model: {self.config.sentence_transformer_model}...")
            self.sentence_model = SentenceTransformer(self.config.sentence_transformer_model)
            print("Sentence Transformer model loaded successfully!")
        return self.sentence_model
    
    def _load_word2vec(self):
        """Load the Word2Vec model."""
        if self.word2vec_model is None:
            print(f"Loading Word2Vec model: {self.config.word2vec_model}...")
            print("This may take a few minutes on first run...")
            self.word2vec_model = api.load(self.config.word2vec_model)
            print("Word2Vec model loaded successfully!")
        return self.word2vec_model
    
    def _extract_degree_level(self, qualification):
        """
        Extract the degree level from a qualification string.
        
        Args:
            qualification (str): Full qualification string
            
        Returns:
            str: Degree level (High School, Bachelor's, Master's, or PhD)
        """
        qualification_lower = qualification.lower()
        
        if 'phd' in qualification_lower or 'ph.d' in qualification_lower:
            return 'PhD'
        elif 'master' in qualification_lower:
            return "Master's"
        elif 'bachelor' in qualification_lower:
            return "Bachelor's"
        else:
            return 'High School'
    
    def _encode_qualification_hierarchy(self, df, fit=False):
        """
        Encode the educational hierarchy from qualifications.
        
        Args:
            df (pd.DataFrame): DataFrame with 'qualification' column
            fit (bool): Whether to fit the encoder
            
        Returns:
            np.ndarray: Ordinal encoded degree levels
        """
        # Extract degree levels
        degree_levels = df['qualification'].apply(self._extract_degree_level).values.reshape(-1, 1)
        
        if fit:
            self.qualification_ordinal_encoder = OrdinalEncoder(
                categories=self.config.degree_level_mapping,
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            encoded = self.qualification_ordinal_encoder.fit_transform(degree_levels)
        else:
            encoded = self.qualification_ordinal_encoder.transform(degree_levels)
        
        return encoded
    
    def _encode_qualification_semantic(self, df):
        """
        Generate semantic embeddings for qualifications.
        
        Args:
            df (pd.DataFrame): DataFrame with 'qualification' column
            
        Returns:
            np.ndarray: Semantic embeddings
        """
        model = self._load_sentence_transformer()
        qualifications = df['qualification'].tolist()
        embeddings = model.encode(qualifications, show_progress_bar=True)
        return embeddings
    
    def _encode_experience(self, df, fit=False):
        """
        Encode experience levels.
        
        Args:
            df (pd.DataFrame): DataFrame with 'experience_level' column
            fit (bool): Whether to fit the encoder
            
        Returns:
            np.ndarray: Ordinal encoded experience levels
        """
        experience = df['experience_level'].values.reshape(-1, 1)
        
        if fit:
            self.experience_ordinal_encoder = OrdinalEncoder(
                categories=self.config.experience_mapping,
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            encoded = self.experience_ordinal_encoder.fit_transform(experience)
        else:
            encoded = self.experience_ordinal_encoder.transform(experience)
        
        return encoded
    
    def _generate_skill_vectors(self, df):
        """
        Generate averaged word embeddings for skills.
        
        Args:
            df (pd.DataFrame): DataFrame with 'skills' column
            
        Returns:
            np.ndarray: Averaged skill embeddings
        """
        model = self._load_word2vec()
        skill_vectors = []
        
        print("Generating skill vectors...")
        for idx, skills_str in enumerate(df['skills']):
            # Parse skills (comma-separated)
            skills = [skill.strip() for skill in skills_str.split(',')]
            
            # Get embeddings for each skill
            embeddings = []
            for skill in skills:
                # Try the skill as-is first
                if skill in model:
                    embeddings.append(model[skill])
                else:
                    # Try without spaces (e.g., "Machine Learning" -> "Machine_Learning")
                    skill_normalized = skill.replace(' ', '_')
                    if skill_normalized in model:
                        embeddings.append(model[skill_normalized])
                    else:
                        # Try individual words
                        words = skill.split()
                        for word in words:
                            if word in model:
                                embeddings.append(model[word])
            
            # Average the embeddings
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
            else:
                # If no embeddings found, use zero vector
                avg_embedding = np.zeros(300)
            
            skill_vectors.append(avg_embedding)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} skill vectors...")
        
        print(f"Skill vector generation completed!")
        return np.array(skill_vectors)
    
    def _encode_target(self, df, fit=False):
        """
        Encode target labels (job roles).
        
        Args:
            df (pd.DataFrame): DataFrame with 'job_role' column
            fit (bool): Whether to fit the encoder
            
        Returns:
            np.ndarray: Encoded job roles
        """
        if fit:
            self.label_encoder = LabelEncoder()
            encoded = self.label_encoder.fit_transform(df['job_role'])
            print(f"\nTarget encoding completed!")
            print(f"Number of unique job roles: {len(self.label_encoder.classes_)}")
            print(f"Job roles: {list(self.label_encoder.classes_)}")
        else:
            encoded = self.label_encoder.transform(df['job_role'])
        
        return encoded
    
    def initiate_data_transformation(self, train_df, val_df, test_df):
        """
        Perform complete data transformation pipeline.
        
        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data
            test_df (pd.DataFrame): Test data
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("=" * 80)
        print("Starting Data Transformation")
        print("=" * 80)
        
        try:
            # 1. QUALIFICATION FEATURES (Hierarchy + Semantic)
            print("\n[1/5] Processing Qualification Features...")
            print("-" * 80)
            
            print("Encoding qualification hierarchy...")
            qual_hierarchy_train = self._encode_qualification_hierarchy(train_df, fit=True)
            qual_hierarchy_val = self._encode_qualification_hierarchy(val_df, fit=False)
            qual_hierarchy_test = self._encode_qualification_hierarchy(test_df, fit=False)
            
            print("Generating qualification semantic embeddings...")
            qual_semantic_train = self._encode_qualification_semantic(train_df)
            qual_semantic_val = self._encode_qualification_semantic(val_df)
            qual_semantic_test = self._encode_qualification_semantic(test_df)
            
            print(f"Qualification hierarchy shape: {qual_hierarchy_train.shape}")
            print(f"Qualification semantic shape: {qual_semantic_train.shape}")
            
            # 2. EXPERIENCE FEATURES
            print("\n[2/5] Processing Experience Level Features...")
            print("-" * 80)
            
            exp_train = self._encode_experience(train_df, fit=True)
            exp_val = self._encode_experience(val_df, fit=False)
            exp_test = self._encode_experience(test_df, fit=False)
            
            print(f"Experience level shape: {exp_train.shape}")
            
            # 3. SKILL FEATURES
            print("\n[3/5] Processing Skill Features...")
            print("-" * 80)
            
            skill_train = self._generate_skill_vectors(train_df)
            skill_val = self._generate_skill_vectors(val_df)
            skill_test = self._generate_skill_vectors(test_df)
            
            print(f"Skill vector shape: {skill_train.shape}")
            
            # 4. COMBINE FEATURES
            print("\n[4/5] Combining Features...")
            print("-" * 80)
            
            X_train = np.hstack([
                qual_hierarchy_train,
                qual_semantic_train,
                exp_train,
                skill_train
            ])
            
            X_val = np.hstack([
                qual_hierarchy_val,
                qual_semantic_val,
                exp_val,
                skill_val
            ])
            
            X_test = np.hstack([
                qual_hierarchy_test,
                qual_semantic_test,
                exp_test,
                skill_test
            ])
            
            print(f"Combined feature matrix shape: {X_train.shape}")
            print(f"Feature breakdown:")
            print(f"  - Qualification hierarchy: 1 feature")
            print(f"  - Qualification semantic: {qual_semantic_train.shape[1]} features")
            print(f"  - Experience level: 1 feature")
            print(f"  - Skill vectors: {skill_train.shape[1]} features")
            print(f"  - Total: {X_train.shape[1]} features")
            
            # 5. SCALING
            print("\n[5/5] Applying Standard Scaling...")
            print("-" * 80)
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            print("Scaling completed!")
            
            # ENCODE TARGETS
            print("\nEncoding target labels...")
            print("-" * 80)
            
            y_train = self._encode_target(train_df, fit=True)
            y_val = self._encode_target(val_df, fit=False)
            y_test = self._encode_target(test_df, fit=False)
            
            # SAVE ALL ARTIFACTS
            print("\nSaving transformation artifacts...")
            print("-" * 80)
            
            save_object(self.config.qualification_ordinal_encoder_path, self.qualification_ordinal_encoder)
            save_object(self.config.experience_ordinal_encoder_path, self.experience_ordinal_encoder)
            save_object(self.config.label_encoder_path, self.label_encoder)
            save_object(self.config.standard_scaler_path, self.scaler)
            
            print("\n" + "=" * 80)
            print("Data Transformation completed successfully!")
            print("=" * 80)
            
            return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
            
        except Exception as e:
            raise Exception(f"Error occurred during data transformation: {str(e)}")


if __name__ == "__main__":
    # Test the data transformation
    from data_ingestion import DataIngestion
    
    print("Testing Data Transformation...")
    print("\nStep 1: Data Ingestion")
    ingestion = DataIngestion()
    train_df, val_df, test_df = ingestion.initiate_data_ingestion()
    
    print("\n\nStep 2: Data Transformation")
    transformation = DataTransformation()
    X_train, X_val, X_test, y_train, y_val, y_test = transformation.initiate_data_transformation(
        train_df, val_df, test_df
    )
    
    print(f"\n\nFinal shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")



