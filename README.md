# Video Game Sales-Level Classification

> **This is a post-release classification system.** It uses rating and engagement
> metadata collected after a game has shipped. It is **NOT** a pre-launch
> forecasting tool.

---

## Dataset

Download `Video_Games_Sales_as_at_22_Dec_2016.csv` from Kaggle:

```
https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings
```

Place the CSV in the same directory as `video_game_classifier.py`.

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended. Key version constraints:

| Package | Minimum | Reason |
|---------|---------|--------|
| scikit-learn | 1.2.0 | `OneHotEncoder(sparse_output=False)` |
| xgboost | 1.6.0 | `eval_metric` kwarg + stable API |
| shap | 0.41.0 | `TreeExplainer`, `dependence_plot` stability |

---

## Run

```bash
# Default: looks for dataset in current directory
python video_game_classifier.py

# Explicit path
python video_game_classifier.py path/to/Video_Games_Sales_as_at_22_Dec_2016.csv
```

The script runs all 8 phases end-to-end. Expected wall-clock time: 5–20 minutes
depending on CPU core count (GridSearchCV uses `n_jobs=-1`).

---

## Output Structure

```
outputs/
  eda/
    eda_01_target_distribution.png
    eda_02_games_by_genre.png
    eda_03_games_by_platform.png
    eda_04_sales_level_by_genre.png
    eda_05_sales_level_by_platform.png
    eda_06_critic_score_by_sales_level.png
    eda_07_user_score_by_sales_level.png
    eda_08_correlation_heatmap.png
    eda_09_critic_score_boxplot.png
    eda_10_games_per_year.png
  models/
    best_params.json
    model_comparison.csv
    confusion_matrix_random_forest.png   ← top-2 models by F1 Macro
    confusion_matrix_xgboost.png         ← (actual names depend on results)
    roc_curves.png
  shap/
    shap_summary_plot.png
    shap_dependence_top1_feature.png
  regional/
    regional_shap_comparison.csv
```

---

## Pipeline Summary

| Phase | Description |
|-------|-------------|
| 1 | Data cleaning: drops, type conversions, fixed-threshold Sales_Level target (Low/Medium/High), publisher grouping |
| 2 | EDA: 10 saved plots |
| 3 | Preprocessing: stratified 80/20 split, ColumnTransformer pipeline |
| 4 | Train 4 models (LR, DT baselines; RF + XGBoost with GridSearchCV) |
| 5 | Evaluate all models; select champion by F1 Macro (ROC-AUC tiebreaker) |
| 6 | Ablation: champion with vs. without post-release rating features |
| 7 | SHAP explainability on champion (Version A) |
| 8 | Regional analysis for NA / EU / JP with per-region SHAP top-5 comparison |
