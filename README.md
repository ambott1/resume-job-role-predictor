# Job Role Prediction System

A comprehensive machine learning system for predicting job roles based on candidate skills, qualifications, and experience levels using advanced feature engineering with word embeddings, semantic transformers, and XGBoost.

## 🎯 Project Overview

This system predicts the most suitable job role for a candidate by analyzing:
- **Skills**: Semantic representation using Word2Vec embeddings
- **Qualifications**: Hybrid approach combining educational hierarchy and semantic field embeddings
- **Experience Level**: Ordinal encoding for career progression

## 📊 Dataset

**Location**: `datasets/candidate_job_role_dataset.csv`

**Statistics**:
- 1,000 candidate profiles
- 25+ unique job roles
- 100+ unique skills
- Features: candidate_id, skills, qualification, experience_level, job_role

## 🏗️ Project Structure

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
├── notebooks/                          # Jupyter notebooks
│   └── 1-data-exploration.ipynb
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
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## 🚀 Getting Started

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

## 📚 Usage

### 1. Data Exploration (Optional)

Explore the dataset interactively:

```bash
jupyter notebook notebooks/1-data-exploration.ipynb
```

This notebook includes:
- Data profiling and quality checks
- Distribution visualizations
- Skills analysis and word clouds
- Class imbalance analysis
- Feature engineering prototypes
- Baseline model testing

### 2. Training the Model

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
5. **Artifact Saving**: All models and encoders saved to `artifacts/`

**Expected output:**
```
================================================================================
TRAINING PIPELINE COMPLETED SUCCESSFULLY!
================================================================================

Final Model Performance:
  • Test Accuracy: 0.85+ (varies)
  • Test F1 Score (Weighted): 0.85+
  • Test F1 Score (Macro): 0.84+
  • Cross-validation F1 Score: 0.83+
```

### 3. Making Predictions

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

## 🔬 Feature Engineering Details

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

## 🎯 Model Architecture

**Algorithm**: XGBoost Classifier
- Multi-class classification (25+ job roles)
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

## 📈 Performance Metrics

The system is evaluated using:
- **Accuracy**: Overall correctness
- **F1 Score (Weighted)**: Accounts for class imbalance
- **F1 Score (Macro)**: Equal weight to all classes
- **Classification Report**: Per-class precision, recall, and F1

Expected performance: **85-90% accuracy** on test set

## 🔧 Saved Artifacts

All artifacts are saved in `artifacts/` directory:

| Artifact | Description | Size |
|----------|-------------|------|
| `model.joblib` | Trained XGBoost model | ~5-10 MB |
| `standard_scaler.joblib` | Feature scaler | <1 MB |
| `qualification_ordinal_encoder.joblib` | Degree level encoder | <1 MB |
| `experience_ordinal_encoder.joblib` | Experience encoder | <1 MB |
| `label_encoder.joblib` | Job role encoder | <1 MB |

## 📊 Key Insights

From data exploration:
- **Balanced dataset**: ~40 candidates per job role
- **Common skills**: Python, SQL, JavaScript, AWS, Docker
- **Education**: Majority have Bachelor's or Master's degrees
- **Experience**: Distributed across Entry, Mid, and Senior levels

## 🔮 Future Enhancements

- [ ] Create web interface for predictions
- [ ] Add model versioning and tracking

## 📝 License

This project is part of the ECS171 Machine Learning course at UC Davis.

## 👥 Contributors

Team 5 - ECS171 Fall Quarter 2025

## 🙏 Acknowledgments

- Pre-trained models: Sentence Transformers, Google Word2Vec
- Libraries: scikit-learn, XGBoost, gensim, pandas
- Course: ECS171 - Machine Learning, UC Davis

