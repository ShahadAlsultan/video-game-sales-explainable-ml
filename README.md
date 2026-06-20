# Video Game Sales-Level Classification & Regional Explainable ML Pipeline

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/scikit--learn-1.2.0%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-1.6.0%2B-green.svg)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/SHAP-0.41.0%2B-purple.svg)](https://shap.readthedocs.io/)

> **IMPORTANT FRAMING**: This is a **post-release classification system** using rating, review, and engagement metadata collected *after* a game has shipped. It is **NOT** a pre-launch forecasting tool. 

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
