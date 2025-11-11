"""
Prediction Pipeline
Handles loading artifacts and making predictions on new data.
"""
import sys
import re
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
import gensim.downloader as api

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import load_object, get_artifacts_path


class PredictionPipeline:
    """Complete prediction pipeline for job role prediction."""
    
    def __init__(self):
        """Initialize and load all required artifacts and models."""
        print("Initializing Prediction Pipeline...")
        
        artifacts_path = get_artifacts_path()
        
        # Load saved artifacts
        print("Loading artifacts...")
        self.model = load_object(artifacts_path / "model.joblib")
        self.scaler = load_object(artifacts_path / "standard_scaler.joblib")
        self.qualification_ordinal_encoder = load_object(artifacts_path / "qualification_ordinal_encoder.joblib")
        self.experience_ordinal_encoder = load_object(artifacts_path / "experience_ordinal_encoder.joblib")
        self.label_encoder = load_object(artifacts_path / "label_encoder.joblib")
        
        # Load pre-trained models
        print("Loading Sentence Transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Loading Word2Vec model...")
        self.word2vec_model = api.load('word2vec-google-news-300')
        
        print("Prediction Pipeline initialized successfully!\n")
        
    def _extract_degree_level(self, qualification):
        """
        Extract the degree level from a qualification string.
        
        Args:
            qualification (str): Full qualification string
            
        Returns:
            str: Degree level
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
    
    def _transform_qualification(self, qualification):
        """
        Transform qualification into features (hierarchy + semantic).
        
        Args:
            qualification (str): Full qualification string
            
        Returns:
            np.ndarray: Combined qualification features
        """
        # Hierarchy encoding
        degree_level = self._extract_degree_level(qualification)
        hierarchy = self.qualification_ordinal_encoder.transform([[degree_level]])
        
        # Semantic embedding
        semantic = self.sentence_model.encode([qualification])
        
        # Combine
        combined = np.hstack([hierarchy, semantic])
        return combined
    
    def _transform_experience(self, experience_level):
        """
        Transform experience level into features.
        
        Args:
            experience_level (str): Experience level (Entry/Mid/Senior)
            
        Returns:
            np.ndarray: Encoded experience level
        """
        return self.experience_ordinal_encoder.transform([[experience_level]])
    
    def _transform_skills(self, skills):
        """
        Transform skills into averaged word embeddings.
        
        Args:
            skills (str or list): Comma-separated string or list of skills
            
        Returns:
            np.ndarray: Averaged skill embedding
        """
        # Parse skills
        if isinstance(skills, str):
            skills_list = [skill.strip() for skill in skills.split(',')]
        else:
            skills_list = skills
        
        # Get embeddings
        embeddings = []
        for skill in skills_list:
            # Try the skill as-is
            if skill in self.word2vec_model:
                embeddings.append(self.word2vec_model[skill])
            else:
                # Try without spaces
                skill_normalized = skill.replace(' ', '_')
                if skill_normalized in self.word2vec_model:
                    embeddings.append(self.word2vec_model[skill_normalized])
                else:
                    # Try individual words
                    words = skill.split()
                    for word in words:
                        if word in self.word2vec_model:
                            embeddings.append(self.word2vec_model[word])
        
        # Average the embeddings
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
        else:
            # If no embeddings found, use zero vector
            avg_embedding = np.zeros(300)
        
        return avg_embedding.reshape(1, -1)
    
    def predict_role(self, skills, qualification, experience_level):
        """
        Predict the job role for a candidate.
        
        Args:
            skills (str or list): Comma-separated string or list of skills
            qualification (str): Educational qualification
            experience_level (str): Experience level (Entry/Mid/Senior)
            
        Returns:
            dict: Prediction results containing predicted role and suitability score
        """
        try:
            # Transform input features
            qual_features = self._transform_qualification(qualification)
            exp_features = self._transform_experience(experience_level)
            skill_features = self._transform_skills(skills)
            
            # Combine all features
            X = np.hstack([qual_features, exp_features, skill_features])
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(probabilities)
            predicted_role = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Get suitability score (highest probability)
            suitability_score = probabilities[predicted_class_idx]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_roles = [(self.label_encoder.inverse_transform([idx])[0], probabilities[idx]) 
                          for idx in top_3_indices]
            
            return {
                'predicted_role': predicted_role,
                'suitability_score': float(suitability_score),
                'top_3_predictions': top_3_roles,
                'all_probabilities': {
                    role: float(prob) 
                    for role, prob in zip(self.label_encoder.classes_, probabilities)
                }
            }
            
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")
    
    def predict_batch(self, candidates_data):
        """
        Predict job roles for multiple candidates.
        
        Args:
            candidates_data (list): List of dictionaries with keys: skills, qualification, experience_level
            
        Returns:
            list: List of prediction results
        """
        results = []
        for i, candidate in enumerate(candidates_data):
            print(f"Processing candidate {i+1}/{len(candidates_data)}...")
            result = self.predict_role(
                candidate['skills'],
                candidate['qualification'],
                candidate['experience_level']
            )
            results.append(result)
        
        return results


def demo_prediction():
    """Demonstrate the prediction pipeline with example candidates."""
    print("=" * 80)
    print("Job Role Prediction Pipeline - Demo")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = PredictionPipeline()
    
    # Example candidates
    candidates = [
        {
            'name': 'Candidate 1',
            'skills': 'Python, TensorFlow, Machine Learning, Deep Learning, NLP',
            'qualification': "Master's in Data Science",
            'experience_level': 'Senior'
        },
        {
            'name': 'Candidate 2',
            'skills': 'React, JavaScript, HTML, CSS, TypeScript',
            'qualification': "Bachelor's in Computer Science",
            'experience_level': 'Mid'
        },
        {
            'name': 'Candidate 3',
            'skills': 'Java, Spring Boot, Hibernate, Microservices, AWS',
            'qualification': "Master's in Software Engineering",
            'experience_level': 'Senior'
        },
        {
            'name': 'Candidate 4',
            'skills': 'Figma, Adobe XD, UI/UX Design, Prototyping',
            'qualification': "Bachelor's in Design",
            'experience_level': 'Entry'
        }
    ]
    
    # Make predictions
    for candidate in candidates:
        print("\n" + "=" * 80)
        print(f"Predicting for: {candidate['name']}")
        print("-" * 80)
        print(f"Skills: {candidate['skills']}")
        print(f"Qualification: {candidate['qualification']}")
        print(f"Experience: {candidate['experience_level']}")
        print("-" * 80)
        
        result = pipeline.predict_role(
            candidate['skills'],
            candidate['qualification'],
            candidate['experience_level']
        )
        
        print(f"\n{'PREDICTION RESULTS':^80}")
        print("=" * 80)
        print(f"Predicted Job Role: {result['predicted_role']}")
        print(f"Suitability Score: {result['suitability_score']:.2%}")
        
        print(f"\nTop 3 Predictions:")
        for i, (role, prob) in enumerate(result['top_3_predictions'], 1):
            print(f"  {i}. {role}: {prob:.2%}")
        
        print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo_prediction()



