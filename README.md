# Student At-Risk Prediction System

An end-to-end machine learning application that predicts whether a student may be academically at risk based on academic performance and student characteristics.

## Features

- Student at-risk prediction
- Probability-based risk assessment
- SHAP model explainability
- Confusion matrix
- Feature importance
- Model comparison
- Prediction history
- CSV prediction report download
- Dataset and model information
- Streamlit web interface

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Data Preprocessing
6. Model Training
7. Model Evaluation
8. Hyperparameter Tuning
9. SHAP Explainability
10. Prediction

## Dataset

The dataset contains 500 student records.

Features include:

- Attendance
- Internal Marks
- Assignment Marks
- Study Hours
- Previous GPA
- Previous Failures
- Department
- Gender

The target variable is `AtRisk`.

The dataset contains approximately:

- 75% Not At Risk
- 25% At Risk

`FinalMarks` was excluded from the prediction features to avoid target leakage.

## Model

The final application uses Logistic Regression with:

- StandardScaler for numerical features
- OneHotEncoder for categorical features
- Class balancing using `class_weight="balanced"`

## Model Performance

| Metric | Score |
|---|---:|
| F1 Score | 0.678 |
| Precision | 0.588 |
| Recall | 0.800 |
| ROC-AUC | 0.882 |

## Model Explainability

SHAP is used to explain individual predictions and identify which features contribute most to the model's prediction.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- SHAP
- Plotly
- Streamlit
- Joblib
- Git
- GitHub

## Running Locally

Clone the repository:

```bash
git clone https://github.com/jithendranshreyas-maker/student-at-risk-prediction.git
cd student-at-risk-prediction