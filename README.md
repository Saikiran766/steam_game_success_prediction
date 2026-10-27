# Steam Games Success Prediction

## 1. Project Overview

This project uses machine learning to predict whether a Steam game is successful based on its available features. It explores the relationship between game characteristics, pricing, release year, and player activity, and compares multiple classification algorithms.

The project includes two experiments:

- **Baseline prediction:** Uses a broader set of game features, including some information that may become available after release.
- **Restricted-feature prediction:** Uses a smaller set of selected features intended to represent information that may be available before or around launch.

The second experiment is a restricted-feature experiment, not yet a fully validated historical simulation of pre-launch prediction.

## 2. Objectives

- Audit and clean the Steam games dataset.
- Define a binary target variable for game success.
- Perform data preprocessing and feature engineering.
- Train and compare Logistic Regression, Random Forest, and XGBoost classifiers.
- Evaluate models using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Investigate feature importance and business insights.
- Tune classification thresholds using validation data.
- Visualize game success trends using Power BI and Tableau Public.

## 3. Dataset

The project uses the FronkonGames Steam Games Dataset, obtained from Hugging Face.

The original dataset contains approximately 125,855 records and 39 columns. After removing a row with a missing game name and applying the project filters, the targeted dataset contains **97,565 games**.

The dataset includes information about game prices, platforms, genres, categories, reviews, player activity, and other game characteristics.

### Target Variable

The target column is `Success`.

A game is labelled successful when the lower bound of its estimated owners is at least 20,000. Games below this threshold are labelled unsuccessful.

This is a project-specific definition of success based on estimated ownership, not a universal measure of commercial success or profitability.

Target distribution:

- **Successful games:** 24,598 (25.21%)
- **Unsuccessful games:** 72,967 (74.79%)

Only games released before January 1, 2025, are included in the targeted dataset.

## 4. Train, Validation, and Test Split

The targeted dataset is divided using a stratified split with random seed 42.

| Dataset | Records |
|---|---:|
| Training | 58,539 |
| Validation | 19,513 |
| Test | 19,513 |
| **Total** | **97,565** |

The validation set is used for model comparison and threshold selection. The test set is reserved for final evaluation.

## 5. Machine Learning Algorithms

The project evaluates three classification algorithms:

1. **Logistic Regression** — a linear classification baseline.
2. **Random Forest** — an ensemble of decision trees.
3. **XGBoost** — a gradient-boosted tree-based model.

The experiments compare models trained using different feature representations and preprocessing approaches.

## 6. Baseline Model Results

The following are the final test-set results for the baseline experiment.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Raw Logistic Regression | 76.63% | 53.56% | 54.90% | 54.22% | 74.98% |
| Raw Random Forest | 87.91% | 83.29% | 65.12% | 73.09% | 89.18% |
| Raw XGBoost | 88.04% | 86.78% | 62.03% | 72.35% | 89.21% |
| Preprocessed Logistic Regression | 82.84% | 70.84% | 54.27% | 61.46% | 80.51% |
| Preprocessed Random Forest | 85.87% | 72.13% | 71.61% | 71.87% | 88.94% |
| **Preprocessed XGBoost** | **88.15%** | **85.38%** | **63.96%** | **73.14%** | **89.57%** |

### Baseline Findings

- Preprocessed XGBoost achieved the highest baseline test accuracy, F1-score, and ROC-AUC.
- Raw XGBoost achieved the highest precision among the six baseline models.
- Preprocessed Random Forest achieved the highest recall.
- Model selection depends on whether the priority is overall classification, identifying successful games, or reducing missed successful games.

These results should be interpreted in light of the available features. Some baseline features may contain post-release information and therefore may not be suitable for genuine pre-launch prediction.

## 7. Restricted-Feature Experiment

A separate experiment uses a limited set of features, including:

- Required age
- Price
- DiscountDLC count
- Windows support
- Mac support
- Linux support
- Achievements

Features such as Peak CCU, reviews, Metacritic score, user score, positive and negative ratings, recommendations, and playtime fields are excluded from this experiment.

The availability of every retained feature at the actual launch date has not yet been independently verified. Therefore, these results should be described as a restricted-feature experiment rather than a fully validated pre-launch prediction system.

### Test Results After Threshold Tuning

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 62.07% | 35.85% | 63.88% | 45.93% | 66.35% |
| Random Forest | 68.60% | 40.73% | 53.92% | 46.41% | 69.13% |
| XGBoost | 66.57% | 38.99% | 57.72% | 46.54% | 70.24% |

The classification thresholds were selected using validation data and then evaluated on the held-out test set.

### Restricted-Feature Findings

- XGBoost achieved the highest F1-score and ROC-AUC in the threshold-tuned test comparison.
- Random Forest achieved the highest accuracy and precision among the three tuned models.
- Logistic Regression achieved the highest recall, identifying a larger proportion of successful games at the selected threshold.
- Performance is lower than in the baseline experiment, highlighting the difficulty of predicting success using a restricted set of features.

ROC-AUC measures the model's ability to rank positive examples above negative examples across thresholds. It does not represent the percentage of predictions that are correct.

## 8. Data Preprocessing and Evaluation

The project includes:

- Dataset auditing and cleaning
- Target creation and filtering
- Feature preparation
- Stratified train, validation, and test splitting
- Numeric and categorical preprocessing
- Model training and comparison
- Held-out test evaluation
- Feature importance analysis
- Business insight generation
- Threshold tuning for the restricted-feature experiment
- Confusion matrix and metric visualizations

Feature importance indicates which features influence a fitted model's predictions. It does not, by itself, establish causation.

## 9. Visualizations and Dashboards

The project includes visualizations for model performance, confusion matrices, and feature importance.

Interactive dashboards were also developed in Power BI and Tableau Public to explore:

- Game success rate by price range
- Game success rate by release year
- Average peak concurrent users by success group

The dashboards use the same project-specific success definition. Release-year comparisons should be interpreted cautiously because newer games have had less time to accumulate owners.

## 10. Project Structure

```text
Steam_Games_Success_Prediction/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── processed/
│   └── ...
│
├── src/
│   ├── 01_load_and_audit.py
│   ├── 02_clean_data.py
│   ├── 03_create_target.py
│   ├── 04_prepare_features.py
│   ├── 05_split_data.py
│   ├── 06_preprocess_data.py
│   ├── 06a_preprocess_baseline.py
│   ├── 07_train_raw_models.py
│   ├── 08_train_preprocessed_models.py
│   ├── 09_compare_models.py
│   ├── 10_evaluate_test_set.py
│   ├── 11_feature_importance.py
│   ├── 12_business_insights.py
│   ├── 13_inspect_dataset_columns.py
│   ├── 14_train_prelaunch_models.py
│   ├── 15_evaluate_prelaunch_test_set.py
│   ├── 16_tune_prelaunch_thresholds.py
│   └── 17_evaluate_tuned_prelaunch_test.py
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── models/
│
├── dashboards/
├── docs/
├── notebooks/
├── requirements.txt
└── README.md
```

The actual contents of the data, results, and dashboard directories may vary depending on which scripts and dashboard files have been saved.

## 11. Installation and Setup

### Prerequisites

- Python
- pip
- Visual Studio Code or another Python-compatible editor

### Step 1: Clone or Download the Project

Download or clone the project repository and open its root directory in a terminal.

### Step 2: Create a Virtual Environment

```powershell
python -m venv .venv
```

### Step 3: Activate the Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Add the Dataset

Place the required dataset in the expected project data directory. Check the scripts for the exact input filenames and paths before running them.

The raw dataset is not necessarily included in a code-only project package.

## 12. Running the Project

Run the scripts from the project root directory, in the intended order.

For example:

```powershell
python src/01_load_and_audit.py
python src/02_clean_data.py
python src/03_create_target.py
python src/04_prepare_features.py
python src/05_split_data.py
python src/06a_preprocess_baseline.py
python src/07_train_raw_models.py
python src/08_train_preprocessed_models.py
python src/09_compare_models.py
python src/10_evaluate_test_set.py
python src/11_feature_importance.py
python src/12_business_insights.py
```

The restricted-feature experiment uses a separate preprocessing and evaluation workflow:

```powershell
python src/06_preprocess_data.py
python src/14_train_prelaunch_models.py
python src/15_evaluate_prelaunch_test_set.py
python src/16_tune_prelaunch_thresholds.py
python src/17_evaluate_tuned_prelaunch_test.py
```

Run the data preparation scripts first because later scripts depend on their generated files. Some scripts may need to be run in a specific order or rerun after changes to preprocessing. Confirm that each expected input file exists before continuing.

## 13. Limitations

- Estimated owners are not exact sales figures or verified revenue.
- The 20,000-owner threshold is a project-defined success criterion.
- The dataset may contain missing, noisy, or inconsistent information.
- Class imbalance can affect accuracy and other evaluation metrics.
- Baseline performance may benefit from post-release information.
- The restricted-feature experiment requires further verification of feature availability at launch.
- Feature importance does not prove that a feature causes game success.
- Release-year dashboard trends may be affected by differences in how long games have been available.
- Model performance on this dataset may not generalize to future games or other platforms.

## 14. Future Improvements

- Verify feature availability using documented historical release-time information.
- Test additional classification algorithms and hyperparameter configurations.
- Investigate class imbalance and probability calibration.
- Evaluate performance across genres, price groups, and release periods.
- Add explainability methods such as SHAP.
- Improve data validation and reproducibility.
- Add automated tests and a documented end-to-end execution pipeline.
- Evaluate whether the success definition should be complemented by alternative ownership thresholds.

## 15. Conclusion

This project compares machine learning approaches for classifying Steam games using estimated ownership as a proxy for success.

In the baseline experiment, preprocessed XGBoost achieved an accuracy of **88.15%**, an F1-score of **73.14%**, and a ROC-AUC of **89.57%** on the held-out test set.

The restricted-feature experiment achieved lower performance, illustrating the additional challenge of predicting game success without relying on a broader set of potentially post-release signals.

Together, the experiments demonstrate the importance of feature selection, preprocessing, threshold selection, and appropriate evaluation when developing machine learning models for business-oriented prediction tasks.
