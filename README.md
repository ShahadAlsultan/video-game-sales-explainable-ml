# 🎮 Video Game Sales-Level Classification & Regional Explainable ML Pipeline

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/scikit--learn-1.2.0%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-1.6.0%2B-green.svg)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/SHAP-0.41.0%2B-purple.svg)](https://shap.readthedocs.io/)

> ⚠️ **IMPORTANT FRAMING**: This is a **post-release classification system** using rating, review, and engagement metadata collected *after* a game has shipped. It is **NOT** a pre-launch forecasting tool. 

---

## 📌 Project Overview
This repository contains an end-to-end Machine Learning pipeline developed to classify video games into three distinct commercial success tiers (**Low**, **Medium**, **High**) based on their global and regional sales performance. 

The core emphasis of this project is **Rigorous Benchmarking** and **Explainable AI (XAI)**. It evaluates multiple traditional and ensemble classifiers, performs feature ablation studies to isolate the impact of post-release engagement signals, and leverages **SHAP (SHapley Additive exPlanations)** to dissect model decisions at both a global and regional (NA, EU, JP) level.

### 🎯 Target Definition (Three-Class Problem)
* **Low Sales Tier**: $< 1.0$ Million units sold.
* **Medium Sales Tier**: $1.0 - 5.0$ Million units sold.
* **High Sales Tier**: $> 5.0$ Million units sold.

---

## 🛠️ Repository Structure

```text
```
├── video_game_classifier.py            # Main end-to-end pipeline (Phases 1-8)
├── regional_shap_per_class.py          # Standalone regional class-specific SHAP analysis
├── run_ablation_only.py                # Standalone feature ablation testing script
├── run_all_features_benchmark.py       # Benchmark comparing V1, V2, and V3 feature sets
├── run_sales_features_benchmark.py     # Benchmark testing regional sales leakage (V4/V5)
├── run_feature_selection_benchmark.py  # RF-based SelectFromModel validation
├── run_xgb_feature_selection_benchmark.py # XGBoost-based feature selection validation
├── run_shap_phases.py                  # Standalone runner for fast SHAP analysis execution
├── requirements.txt                    # Project dependencies and constraints
└── SPEC.md                             # Detailed engineering and technical specifications







# Detailed engineering and technical specifications

📊 Pipeline Architecture (Phases)

The main workflow (video_game_classifier.py) orchestrates 8 structured phases:
  🧹 Phase 1: Data Cleaning: Strict handling of missing values, category grouping for long-tail publishers, and robust data type casting.
    📉 Phase 2: Exploratory Data Analysis (EDA): Generation of 10 comprehensive visualization plots exploring distributions, correlations, and genre/platform trends.
 ⚙️ Phase 3: Preprocessing: Stratified train/test split (80/20) and a robust ColumnTransformer scaling numerical features and one-hot encoding categorical attributes.

 🔍 Phase 4: Hyperparameter Tuning: Full GridSearchCV optimized via Macro F1-score across 4 core classifiers (Logistic Regression, Decision Trees, Random Forests, XGBoost).

   📈 Phase 5: Model Evaluation: Metrics breakdown (Accuracy, Precision, Recall, F1 Macro, and OvR ROC-AUC) along with multi-class Confusion Matrices and ROC Curves.

   🔬 Phase 6: Feature Ablation Study: Dynamically quantifies the exact marginal drop in predictive power when isolating post-release rating/count matrices.

   💡 Phase 7: Global Explainability (SHAP): Extracts Tree-based SHAP summary graphs and marginal dependency plots to decode the top global performance signals.

   🌍 Phase 8: Regional Sales Analysis: Isolates models trained strictly on regional sub-markets (North America, Europe, Japan) to map geographic variations in consumer behavior.
