"""
Flask Web Application for Tech Job Role Prediction
Provides a user-friendly interface to predict tech job roles based on skills, qualifications, and experience.
"""
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

# Global variable to store prediction pipeline
pipeline = None

def extract_unique_skills():
    """
    Extract all unique skills from the dataset.
    
    Returns:
        list: Sorted list of unique skills
    """
    try:
        dataset_path = Path(__file__).parent / "datasets" / "candidate_job_role_dataset.csv"
        df = pd.read_csv(dataset_path)
        
        # Extract all skills from the skills column
        all_skills = []
        for skills_str in df['skills']:
            # Split by comma and strip whitespace
            skills = [skill.strip() for skill in str(skills_str).split(',')]
            all_skills.extend(skills)
        
        # Get unique skills and sort
        unique_skills = sorted(set(all_skills))
        
        return unique_skills
    except Exception as e:
        print(f"Error extracting skills: {e}")
        return []

def get_pipeline():
    """
    Get or initialize the prediction pipeline.
    
    Returns:
        PredictionPipeline: Initialized prediction pipeline
    """
    global pipeline
    if pipeline is None:
        print("Initializing prediction pipeline...")
        pipeline = PredictionPipeline()
        print("Pipeline initialized successfully!")
    return pipeline

@app.route('/')
def index():
    """
    Render the main page with the prediction form.
    
    Returns:
        str: Rendered HTML template
    """
    skills = extract_unique_skills()
    return render_template('index.html', skills=skills)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    
    Returns:
        JSON: Prediction results including predicted role, suitability score, and top 3 predictions
    """
    try:
        # Get form data
        data = request.get_json()
        
        name = data.get('name', '')
        skills = data.get('skills', [])
        degree_level = data.get('degree_level', '')
        field_of_study = data.get('field_of_study', '')
        experience_level = data.get('experience_level', '')
        
        # Validate input
        if not skills:
            return jsonify({'error': 'Please select at least one skill'}), 400
        if not degree_level:
            return jsonify({'error': 'Please select a qualification level'}), 400
        if not field_of_study:
            return jsonify({'error': 'Please enter your field of study'}), 400
        if not experience_level:
            return jsonify({'error': 'Please select an experience level'}), 400
        
        # Format qualification
        qualification = f"{degree_level} in {field_of_study}"
        
        # Convert skills list to comma-separated string
        skills_str = ', '.join(skills)
        
        # Get prediction pipeline
        pred_pipeline = get_pipeline()
        
        # Make prediction
        result = pred_pipeline.predict_role(
            skills=skills_str,
            qualification=qualification,
            experience_level=experience_level
        )
        
        # Add candidate name to result
        result['candidate_name'] = name
        
        # Format top 3 predictions for response
        result['top_3_predictions'] = [
            {
                'role': role,
                'probability': float(prob)
            }
            for role, prob in result['top_3_predictions']
        ]
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON: Health status
    """
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("=" * 80)
    print("Tech Job Role Prediction Web Application")
    print("=" * 80)
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("\nNote: First prediction may take 5-10 minutes to download Word2Vec model.")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

