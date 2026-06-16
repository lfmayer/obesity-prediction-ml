# Obesity Risk Prediction — End-to-End Machine Learning System

A supervised multi-class classification system that identifies obesity risk levels across **7 clinical categories** based on dietary habits, physical activity, and lifestyle data. The trained pipeline is served through a no-code interactive web application built with Streamlit.

**Live demo:** https://fiap-tc4-obesidade-jdhk9vfrpcjj9g5uzwh6xg.streamlit.app/

---

## Problem Statement

Obesity is a complex, multifactorial chronic condition associated with type 2 diabetes, cardiovascular disease, and elevated all-cause mortality. Healthcare teams often lack scalable, data-driven tools to stratify patients by risk level before a formal clinical assessment—leading to delayed interventions and inconsistent screening across practitioners.

## Solution

A reproducible ML pipeline that:
1. Cleans raw survey data and engineers domain-informed features (BMI, Healthy Score, Sedentary Index)
2. Benchmarks three classification algorithms using stratified cross-validation
3. Selects the best pipeline automatically based on macro-averaged F1
4. Exposes predictions through a two-page Streamlit app—individual prediction form and an analytical dashboard—with no coding required from end users

## Results

| Model | CV F1-macro | Test Accuracy | Test F1-macro |
|---|---|---|---|
| Logistic Regression | 0.942 | 94.0% | 0.938 |
| Random Forest | 0.976 | 96.9% | 0.967 |
| **Gradient Boosting** ✓ | **0.971** | **97.1%** | **0.970** |

**Project baseline target:** 75% accuracy · **Achieved:** 97.1%

The winning model (Gradient Boosting) misclassifies only adjacent classes in ambiguous edge cases—for example, Overweight Level I vs. II—which minimizes the clinical risk of a severe misdiagnosis.

## Feature Engineering

Three composite features were derived from the raw survey variables before training:

| Feature | Formula | Rationale |
|---|---|---|
| **BMI** | `Weight / Height²` | Standard clinical indicator of body composition |
| **Healthy Score** | `Vegetable intake + Physical activity + Daily water` | Aggregates positive health behaviors into a single proxy |
| **Sedentary Index** | `Screen time / (Physical activity + 1)` | Captures the ratio of sedentary to active behavior; `+1` prevents division by zero |

Ordinal categorical variables (snacking frequency, alcohol use, transport mode) were manually mapped to integer scales that preserve their natural intensity ordering, rather than applying one-hot encoding, to keep the feature space compact and interpretable.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data processing | pandas, NumPy |
| ML pipeline | scikit-learn (Pipeline, ColumnTransformer, GridSearchCV) |
| Models evaluated | LogisticRegression, RandomForestClassifier, GradientBoostingClassifier |
| Serialization | joblib |
| Web app | Streamlit |
| Visualizations | Plotly Express |
| Deployment | Streamlit Community Cloud |

## Application Pages

**Page 1 — Individual Prediction**
A form-based interface where a clinician fills in a patient's demographic and lifestyle data. The app applies the same preprocessing used at training time (imported directly from `train_model.py` to prevent train-serve skew), then returns:
- The predicted obesity class
- Per-class confidence probabilities
- The computed derived features (BMI, Healthy Score, Sedentary Index) for transparency

**Page 2 — Analytical Dashboard**
An interactive exploratory view of the full dataset with sidebar filters (gender, age range, family history, obesity level). Includes:
- KPI summary cards (patient count, mean BMI, obesity prevalence, family history rate)
- Target class distribution bar chart
- BMI and age boxplots broken down by class and gender
- Physical activity frequency and caloric consumption habits vs. obesity level
- Family history and transport mode cross-tabulations
- Numerical correlation heatmap
- Model comparison table and confusion matrix for the selected best model

## Project Structure

```
obesity-prediction-ml/
├── app.py                        # Streamlit entry point (home page)
├── train_model.py                # Full training pipeline — data loading, feature
│                                 #   engineering, GridSearchCV, model persistence
├── pages/
│   ├── 1_Predicao_Individual.py  # Individual prediction form
│   └── 2_Dashboard_Analitico.py  # Analytical dashboard
├── notebooks/
│   └── TechChallenge4_v2.ipynb   # EDA, modeling rationale, and reproducibility
├── models/
│   ├── best_pipeline.joblib      # Serialized sklearn Pipeline (preprocessor + model)
│   └── metrics_report.json       # Full evaluation results for all candidate models
├── data/
│   └── Obesity.csv               # Raw dataset (2,111 records, 17 columns)
├── requirements.txt
└── runtime.txt
```

## Dataset

- **Source:** UCI Machine Learning Repository — Estimation of Obesity Levels Based on Eating Habits and Physical Conditions
- **Records:** 2,111 individuals from Mexico, Peru, and Colombia
- **Target:** `Obesity` — 7 ordinal classes ranging from `Insufficient_Weight` to `Obesity_Type_III`
- **Input features:** Gender, Age, Height, Weight, family history of overweight, dietary habits (vegetable intake, caloric food consumption, meals per day, snacking, water intake, alcohol), physical activity, screen time, calorie monitoring, and transport mode

## Reproducibility

The complete training run can be re-executed at any time:

```bash
python train_model.py
```

This regenerates `models/best_pipeline.joblib` and `models/metrics_report.json` from scratch. The random seed is fixed (`random_state=42`) and the train/test split is stratified (80/20), so results are deterministic.

## How to Run Locally

```bash
git clone https://github.com/<your-username>/obesity-prediction-ml.git
cd obesity-prediction-ml
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`. The serialized model is already included in the repository under `models/`, so no retraining is needed to use the prediction page.

---

**FIAP PosTech** · Business Analytics & Data-Driven Decision Making  
Phase 4 Tech Challenge · Group 87
