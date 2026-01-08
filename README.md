# Tech Job Role Prediction System

A comprehensive machine learning system for predicting **tech industry job roles** based on candidate skills, qualifications, and experience levels using advanced feature engineering with word embeddings, semantic transformers, and XGBoost.

## Project Overview

This system predicts the most suitable **tech job role** for a candidate by analyzing:
- **Skills**: Semantic representation using Word2Vec embeddings
- **Qualifications**: Hybrid approach combining educational hierarchy and semantic field embeddings
- **Experience Level**: Ordinal encoding for career progression

## My Contribution

- Developed the architecture and training pipeline for the XGBoost model. 
- Designed the data pipeline to feed processed feature vectors into the classifier and managed the model serialization.

## Dataset

**Location**: `datasets/candidate_job_role_dataset.csv`

**Statistics**:
- 1,000 candidate profiles from the tech industry
- 20+ unique tech job roles (Data Scientist, Full Stack Developer, DevOps Engineer, etc.)
- 200+ unique technical skills
- Features: candidate_id, skills, qualification, experience_level, job_role

## Project Structure

```
ECS171ResumeAnalysis/
│
├── artifacts/                          # Trained models and encoders
│   ├── model.joblib
│   ├── standard_scaler.joblib
│   ├── qualification_ordinal_encoder.joblib
│   ├── experience_ordinal_encoder.joblib
│   └── label_encoder.joblib
│
├── datasets/                           # Data files
│   └── candidate_job_role_dataset.csv
│
├── imgs/                               # Visualization outputs
│   ├── datasets/
│   │   └── candidate_job_role/        # Data exploration plots
│   │       ├── job_role_distribution.png
│   │       ├── experience_level_distribution.png
│   │       ├── qualifications_distribution.png
│   │       └── skills_analysis.png
│   │
│   └── model/
│       └── performance/                # Model performance plots
│           ├── confusion_matrix.png
│           ├── actual_vs_predicted.png
│           ├── per_class_metrics.png
│           ├── feature_importance.png
│           ├── overall_metrics.png
│           ├── precision_recall_curve.png
│           ├── prediction_probability_distribution.png
│           └── actual_vs_predicted_scatter.png
│
├── notebooks/                          # Jupyter notebooks
│   └── 1-data-exploration.ipynb
│
├── project_report/                     # Project documentation
│   ├── main.tex                        # LaTeX project report source
│   ├── _ECS_171__Project_Report___Team_5.pdf  # Compiled project report
│   ├── ECS_171_Project_Guidelines.pdf
│   └── example_project_report_elsroujiluis_147715_7213196_FACES Emotional Classifier.pdf
│
├── src/                                # Source code
│   ├── components/
│   │   ├── data_ingestion.py          # Data loading and splitting
│   │   ├── data_transformation.py     # Feature engineering
│   │   └── model_trainer.py           # Model training and evaluation
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py          # Complete training workflow
│   │   └── predict_pipeline.py        # Inference pipeline
│   │
│   └── utils.py                       # Utility functions
│
├── templates/                          # Flask templates
│   └── index.html                     # Web interface
│
├── static/                             # Static assets
│   └── styles.css                     # Custom CSS
│
├── app.py                              # Flask web application
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. **Clone the repository** (if applicable)
   ```bash
   cd ECS171ResumeAnalysis
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: First-time installation will download large pre-trained models:
   - Sentence Transformer (all-MiniLM-L6-v2): ~90 MB
   - Word2Vec (word2vec-google-news-300): ~1.5 GB

## Usage

### 1. Web Interface (Recommended)

The easiest way to use the system is through the web interface:

```bash
python3 app.py
```

Then open your browser and navigate to: **http://localhost:5000**

**Features:**
- Modern, responsive UI built with Tailwind CSS and Alpine.js
- Multi-select dropdown with search for 200+ technical skills
- Real-time predictions with confidence scores
- Visual probability charts for top 3 matching roles
- Instant results with detailed analysis

**First-time use:** The initial prediction may take 5-10 minutes to download the Word2Vec model (~1.5 GB). Subsequent predictions are instant.

**Demo:** Fill out the form with:
- Name
- Technical skills (select multiple from dropdown)
- Highest qualification (degree level + field of study)
- Experience level (Entry/Mid/Senior)

The system will predict your best-fit tech role with a confidence score and show the top 3 matching roles.

### 2. Data Exploration (Optional)

Explore the dataset interactively:

```bash
jupyter notebook notebooks/1-data-exploration.ipynb
```

This notebook includes:
- Data profiling and quality checks
- Distribution visualizations (saved to `imgs/datasets/candidate_job_role/`)
- Skills analysis and word clouds
- Class imbalance analysis
- Feature engineering prototypes
- Baseline model testing

**Generated Visualizations:**
- Job Role Distribution
- Experience Level Distribution
- Qualifications Distribution
- Skills Analysis (word clouds, top skills)

### 3. Training the Model

Run the complete training pipeline:

```bash
cd src/pipeline
python3 train_pipeline.py
```

**What happens during training:**
1. **Data Ingestion**: Loads data and splits into train/val/test (70%/15%/15%)
2. **Feature Engineering**:
   - Qualification: Ordinal hierarchy + semantic embeddings (384-dim)
   - Experience: Ordinal encoding (Entry=1, Mid=2, Senior=3)
   - Skills: Averaged Word2Vec embeddings (300-dim)
3. **Model Training**: XGBoost with RandomizedSearchCV (50 iterations, 3-fold CV)
4. **Evaluation**: Classification report on test set
5. **Visualization**: Generates performance plots (confusion matrix, metrics, feature importance)
6. **Artifact Saving**: All models and encoders saved to `artifacts/`

**Expected output:**
```
================================================================================
TRAINING PIPELINE COMPLETED SUCCESSFULLY!
================================================================================

Final Model Performance:
  • Test Accuracy: 0.99+ (varies)
  • Test F1 Score (Weighted): 0.99+ (varies)
  • Test F1 Score (Macro): 0.99+ (varies)
  • Cross-validation F1 Score: 0.99+ (varies)
```

### 4. Making Predictions (Command Line)

#### Option A: Demo Script

Run the demo with example candidates:

```bash
cd src/pipeline
python3 predict_pipeline.py
```

#### Option B: Custom Predictions

Use the prediction pipeline in your code:

```python
from src.pipeline.predict_pipeline import PredictionPipeline

# Initialize pipeline (loads all artifacts)
pipeline = PredictionPipeline()

# Make a prediction
result = pipeline.predict_role(
    skills="Python, TensorFlow, Machine Learning, Deep Learning, NLP",
    qualification="Master's in Data Science",
    experience_level="Senior"
)

print(f"Predicted Role: {result['predicted_role']}")
print(f"Suitability Score: {result['suitability_score']:.2%}")
print(f"Top 3 Predictions: {result['top_3_predictions']}")
```

**Example output:**
```
================================================================================
Predicting for: Sample Candidate
--------------------------------------------------------------------------------
Skills: Python, TensorFlow, Machine Learning, Deep Learning, NLP
Qualification: Master's in Data Science
Experience: Senior
--------------------------------------------------------------------------------

                        PREDICTION RESULTS                        
================================================================================
Predicted Job Role: Data Scientist
Suitability Score: 92.45%

Top 3 Predictions:
  1. Data Scientist: 92.45%
  2. AIML: 4.23%
  3. Data Analyst: 1.87%
================================================================================
```

## Feature Engineering Details

### Hybrid Qualification Encoding

**1. Educational Hierarchy (Ordinal)**
- Maps degree levels: High School(1) → Bachelor's(2) → Master's(3) → PhD(4)
- Captures career progression requirements

**2. Semantic Field Embeddings**
- Uses Sentence Transformers (all-MiniLM-L6-v2)
- Generates 384-dimensional embeddings
- Captures similarity between fields (e.g., "Computer Science" ≈ "Software Engineering")

### Skill Vector Generation

- Uses Word2Vec (word2vec-google-news-300)
- Generates 300-dimensional averaged embeddings
- Handles multi-word skills (e.g., "Machine Learning")
- Captures semantic relationships (e.g., "Python" ≈ "JavaScript")

### Standard Scaling

- Applied to all combined features
- Fitted on training data only
- Ensures consistent feature scales

## Model Architecture

**Algorithm**: XGBoost Classifier
- Multi-class classification (20+ tech job roles)
- Hyperparameter tuning with RandomizedSearchCV
- 50 random combinations tested
- 3-fold cross-validation
- Optimization metric: F1 Score (weighted)

**Hyperparameter Search Space**:
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}
```

## Performance Metrics

The system is evaluated using:
- **Accuracy**: Overall correctness
- **F1 Score (Weighted)**: Accounts for class imbalance
- **F1 Score (Macro)**: Equal weight to all classes
- **Classification Report**: Per-class precision, recall, and F1

Expected performance: **~99% accuracy** on test set

### Performance Visualizations

After training, the following visualizations are automatically generated and saved to `imgs/model/performance/`:

1. **Confusion Matrix** (`confusion_matrix.png`)
   - Shows actual vs predicted job roles for all classes
   - Helps identify which roles are frequently confused

2. **Actual vs Predicted Comparison** (`actual_vs_predicted.png`)
   - Overview of prediction accuracy
   - Distribution comparison between actual and predicted job roles

3. **Per-Class Performance Metrics** (`per_class_metrics.png`)
   - Precision, Recall, and F1-Score for each job role
   - Identifies which roles the model predicts best/worst

4. **Feature Importance** (`feature_importance.png`)
   - Top 20 most important features from XGBoost
   - Shows which features contribute most to predictions

5. **Overall Metrics Summary** (`overall_metrics.png`)
   - Bar chart of overall accuracy and F1 scores
   - Quick visual summary of model performance

6. **Precision-Recall Curve** (`precision_recall_curve.png`)
   - Multi-class precision-recall curves
   - Shows micro-averaged and per-class PR curves with average precision scores
   - Useful for understanding model performance across different thresholds

7. **Prediction Probability Distribution** (`prediction_probability_distribution.png`)
   - Distribution of maximum prediction probabilities
   - Confidence vs accuracy analysis
   - Helps assess model calibration and prediction certainty

8. **Actual vs Predicted Scatter Plot** (`actual_vs_predicted_scatter.png`)
   - Scatter plot showing actual vs predicted job roles
   - Color-coded by prediction correctness
   - Visual representation of prediction patterns and errors

## 🔧 Saved Artifacts

All artifacts are saved in `artifacts/` directory:

| Artifact | Description | Size |
|----------|-------------|------|
| `model.joblib` | Trained XGBoost model | ~5-10 MB |
| `standard_scaler.joblib` | Feature scaler | <1 MB |
| `qualification_ordinal_encoder.joblib` | Degree level encoder | <1 MB |
| `experience_ordinal_encoder.joblib` | Experience encoder | <1 MB |
| `label_encoder.joblib` | Job role encoder | <1 MB |

## Key Insights

From data exploration:
- **Balanced dataset**: ~40 candidates per tech job role
- **Common tech skills**: Python, SQL, JavaScript, AWS, Docker, React, Java, Kubernetes
- **Education**: Majority have Bachelor's or Master's degrees in CS, Software Engineering, or related fields
- **Experience**: Distributed across Entry (0-2 years), Mid (3-5 years), and Senior (6+ years) levels
- **Tech roles**: Data Scientist, Full Stack Developer, DevOps Engineer, Frontend Developer, Backend Developer, and more

## Web Interface Details

The Flask web application (`app.py`) provides:

**Backend:**
- RESTful API endpoints for predictions
- Automatic skill extraction from dataset
- Integration with prediction pipeline
- Error handling and validation

**Frontend:**
- Tailwind CSS for modern, responsive design
- Alpine.js for reactive UI components
- Choices.js for enhanced multi-select dropdown
- Smooth animations and loading states
- Mobile-friendly layout

**Tech Stack:**
- **Framework:** Flask 3.0+
- **Styling:** Tailwind CSS (CDN)
- **Interactivity:** Alpine.js (CDN)
- **Multi-select:** Choices.js (CDN)

## Project Report

The `project_report/` directory contains:
- **main.tex**: LaTeX source code for the project report
- **_ECS_171__Project_Report___Team_5.pdf**: Compiled project report (PDF)
- **ECS_171_Project_Guidelines.pdf**: Official project guidelines and requirements
- **example_project_report_elsroujiluis_147715_7213196_FACES Emotional Classifier.pdf**: Example project report for reference

The `main.tex` file contains the complete LaTeX source code for the project report, including all sections, figures, tables, and bibliography. The compiled PDF report (`_ECS_171__Project_Report___Team_5.pdf`) is the final formatted version ready for submission.

The guidelines and example report provide guidance on project structure, reporting requirements, and formatting standards for the ECS171 course.

## License

This project is part of the ECS171 Machine Learning course at UC Davis.

## Contributors

Team 5 - ECS171 Fall Quarter 2025

## Acknowledgments

- Pre-trained models: Sentence Transformers, Google Word2Vec
- Libraries: scikit-learn, XGBoost, gensim, pandas, Flask
- Frontend: Tailwind CSS, Alpine.js, Choices.js
- Course: ECS171 - Machine Learning, UC Davis
