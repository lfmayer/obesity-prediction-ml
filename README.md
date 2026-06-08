# Obesity Risk Prediction · Machine Learning

Predictive classification system that identifies obesity risk levels across 7 categories based on eating habits, physical activity, and lifestyle data. Deployed as an interactive web application via Streamlit.

**Live demo:** https://fiap-tc4-obesidade-jdhk9vfrpcjj9g5uzwh6xg.streamlit.app/

---

## Problem

Healthcare institutions need scalable tools to screen patients for obesity risk before clinical assessment. Manual evaluation is time-consuming and inconsistent across practitioners.

## Solution

A supervised ML pipeline that classifies patients into 7 obesity categories with ~97% accuracy, deployable by medical teams through a no-code web interface.

## Results

| Model | CV F1-macro | Test Accuracy |
|---|---|---|
| Logistic Regression | 0.942 | 94.0% |
| Random Forest | 0.976 | 96.9% |
| **Gradient Boosting** ✓ | **0.971** | **97.1%** |

Target: 75% accuracy · Achieved: **97.1%**

## Tech Stack

- **Python** · pandas · scikit-learn · Streamlit · Plotly
- **ML:** GradientBoostingClassifier with GridSearchCV (5-fold CV)
- **Deploy:** Streamlit Community Cloud

## Features

- Individual prediction form with confidence score per category
- Analytical dashboard with KPIs, distributions, confusion matrix
- Custom feature engineering: BMI, Healthy Score, Sedentary Index

## Project Structure

```
├── app.py                        # Streamlit main app
├── pages/
│   ├── 1_Individual_Prediction.py
│   └── 2_Analytical_Dashboard.py
├── notebooks/
│   └── TechChallenge4_v2.ipynb   # Full EDA + modeling
├── models/
│   └── best_pipeline.joblib
├── data/
│   └── Obesity.csv
└── train_model.py
```

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

**FIAP PosTech** · Business Analytics & Data-Driven Decision  
Phase 4 Tech Challenge · Group 87
