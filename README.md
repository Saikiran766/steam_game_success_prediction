# Steam Game Success Prediction

## Project Overview

This project develops a binary classification system to predict whether a Steam game can be considered successful using information available about the game and a predefined post-release success label.

The project follows a leakage-aware machine learning workflow covering data auditing, cleaning, target creation, balanced sampling, stratified splitting, training-only preprocessing, model training, validation-based tuning, final test evaluation, and model interpretation.

## Objective

The objective is to classify Steam games into two classes:

- **Success**
- **Not Success**

The target is created from the estimated-owner range after applying a release-maturity filter.

## Dataset

The project uses the **FronkonGames Steam Games Dataset**.

The raw CSV contains Steam game information such as:

- App ID
- Game name
- Release date
- Estimated owners
- Price
- Reviews
- Platforms
- Categories
- Genres
- Tags
- Achievements
- Metacritic information
- Developers and publishers
- Other Steam metadata

The raw dataset is stored at:

```text
data/raw/steam_games.csv
```

The raw file is kept unchanged during the workflow.

## Target Definition

Games released on or after **January 1, 2025** are excluded so that newer games have sufficient time to accumulate ownership.

The operational success rule used in this project is:

```text
Success      : estimated owners lower bound >= 20,000
Not Success  : estimated owners lower bound < 20,000
```

After the maturity filter:

- Mature games retained: **97,565**
- Success: **24,598 (25.21%)**
- Not Success: **72,967 (74.79%)**

The target-source variables are not used as predictive features.

## Dataset Balancing

To create a balanced modeling population, all **24,598 Success** games were retained and **24,598 Not Success** games were randomly sampled using a fixed random seed of 42.

Final balanced dataset:

- Total: **49,196**
- Success: **24,598**
- Not Success: **24,598**

The balanced dataset is stored at:

```text
data/processed/steam_games_balanced.csv
```

This balancing changes the class prevalence relative to the original dataset, so the reported performance should be interpreted as performance on the balanced modeling population.

## Train / Validation / Test Split

The balanced dataset was split using stratified sampling with a fixed random seed of 42:

| Split | Rows |
|---|---:|
| Training | 29,517 |
| Validation | 9,839 |
| Test | 9,840 |

The test set was kept untouched until final evaluation.

Files:

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

## Preprocessing and Feature Engineering

Preprocessing was fitted using the training data and then applied to validation and test data.

The final feature matrix contains **98 features**.

Main transformations include:

- Numeric conversion of applicable numeric columns
- Missing-value handling using training-derived preprocessing
- Numeric scaling
- Top-50 category multi-hot features
- Top-30 genre multi-hot features
- Supported-language count
- Full-audio-language count
- Tag count
- Platform indicators for usable platform fields
- Removal of identifiers and target-source variables
- Exclusion of fields with no usable training information

The fitted preprocessing object is saved as:

```text
results/models/preprocessor.joblib
```

## Models

The project evaluates the following models:

1. Majority-class baseline
2. Logistic Regression
3. Random Forest
4. XGBoost

The primary model-selection metric was **ROC-AUC**.

## Validation Results Before Tuning

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.4999 | 0.4999 | 1.0000 | 0.6666 | 0.5000 |
| Logistic Regression | 0.7096 | 0.7499 | 0.6290 | 0.6841 | 0.7845 |
| Random Forest | 0.7283 | 0.7497 | 0.6855 | 0.7162 | 0.8043 |
| XGBoost | 0.7374 | 0.7744 | 0.6699 | 0.7183 | 0.8134 |

## Hyperparameter Tuning

Tuning was performed using the validation set.

The primary tuning metric was **ROC-AUC**.

Classification thresholds were selected separately on the validation set using F1 score.

Selected configurations:

### Logistic Regression

- C = 0.1
- Threshold = 0.33
- Validation ROC-AUC = 0.7845

### Random Forest

- n_estimators = 400
- max_depth = None
- min_samples_leaf = 2
- max_features = sqrt
- Threshold = 0.40
- Validation ROC-AUC = 0.8128

### XGBoost

- n_estimators = 200
- max_depth = 5
- learning_rate = 0.10
- subsample = 0.8
- colsample_bytree = 0.8
- Threshold = 0.38
- Validation ROC-AUC = 0.8160

## Final Test Results

The final evaluation was performed once on the held-out test set using the tuned models and validation-selected thresholds.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6666 | 0.6208 | 0.8561 | 0.7197 | 0.7864 | 0.8019 |
| Random Forest | 0.7206 | 0.6876 | 0.8087 | 0.7433 | 0.8226 | 0.8338 |
| XGBoost | 0.7186 | 0.6828 | 0.8165 | 0.7437 | 0.8243 | 0.8370 |

The XGBoost model produced the highest test ROC-AUC and PR-AUC among the three trained models, while Random Forest produced a very similar overall F1 score.

## Confusion Matrices

### Logistic Regression

- True Negatives: 2,347
- False Positives: 2,573
- False Negatives: 708
- True Positives: 4,212

### Random Forest

- True Negatives: 3,112
- False Positives: 1,808
- False Negatives: 941
- True Positives: 3,979

### XGBoost

- True Negatives: 3,054
- False Positives: 1,866
- False Negatives: 903
- True Positives: 4,017

Test evaluation files and figures are stored under:

```text
results/metrics/
results/figures/
```

## Model Interpretation

Model interpretation was performed using training and validation data without using the held-out test set for interpretation.

The project includes:

- Logistic Regression coefficient analysis
- Random Forest feature importance
- XGBoost feature importance
- Validation permutation importance
- Validation error analysis by release year

Interpretation outputs are stored in:

```text
results/metrics/
results/figures/
```

Important interpretation files include:

```text
logistic_coefficients.csv
random_forest_feature_importance.csv
xgboost_feature_importance.csv
logistic_regression_permutation_importance.csv
random_forest_permutation_importance.csv
xgboost_permutation_importance.csv
interpretation_summary.txt
```

## Project Structure

```text
Steam_Games_Success_Prediction/
│
├── data/
│   ├── raw/
│   │   └── steam_games.csv
│   ├── cleaned/
│   │   └── steam_games_targeted.csv
│   └── processed/
│       ├── steam_games_balanced.csv
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       ├── train_features.csv
│       ├── validation_features.csv
│       └── test_features.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── 01_load_and_audit.py
│   ├── 02_clean_data.py
│   ├── 03_create_target.py
│   ├── 03_target_maturity_analysis.py
│   ├── 04_balance_dataset.py
│   ├── 05_split_data.py
│   ├── 06_preprocess.py
│   ├── 07_train_models.py
│   ├── 08_tune_models.py
│   ├── 09_evaluate_models.py
│   └── 10_interpret_results.py
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
├── docs/
│   └── methodology.md
│
├── requirements.txt
└── README.md
```

## Reproducibility

The project uses a fixed random seed:

```text
42
```

The seed is used for sampling, splitting, and model-related reproducibility where applicable.

The workflow separates:

- Raw data
- Cleaned/targeted data
- Balanced data
- Train/validation/test data
- Processed feature matrices
- Trained models
- Metrics
- Figures
- Documentation

This makes the experiment easier to reproduce and audit.

## How to Run

From the project root:

```powershell
cd D:\Steam_Games_Success_Prediction
```

Run the scripts in this order:

```powershell
python src/01_load_and_audit.py
python src/02_clean_data.py
python src/03_create_target.py
python src/04_balance_dataset.py
python src/05_split_data.py
python src/06_preprocess.py
python src/07_train_models.py
python src/08_tune_models.py
python src/09_evaluate_models.py
python src/10_interpret_results.py
```

The EDA notebook is:

```text
notebooks/01_eda.ipynb
```

## Limitations

1. The success threshold of 20,000 estimated owners is an operational definition used for this case study and should be interpreted in that context.
2. The maturity cutoff of January 1, 2025 is an operational filtering choice.
3. The modeling dataset was deliberately balanced, so the class distribution does not represent the natural prevalence of successful Steam games in the original dataset.
4. Steam estimated-owner values are provided as ranges rather than exact owner counts.
5. The Windows platform field in the source data did not contain usable values and was therefore not fabricated or used as a predictive feature.
6. Model performance depends on the selected dataset, target definition, feature representation, sampling design, and preprocessing choices.
7. The final metrics should not be interpreted as universal predictions of Steam game success for all future releases.

## Outputs

The main outputs of the project are:

- Cleaned and targeted dataset
- Balanced modeling dataset
- Train/validation/test splits
- Training-derived preprocessing pipeline
- Trained baseline and machine learning models
- Tuned models
- Final test metrics
- Confusion matrices
- ROC curves
- Feature importance results
- Permutation importance results
- Interpretation summary

## Conclusion

This project implements a complete machine learning pipeline for Steam game success classification. It compares a simple baseline with Logistic Regression, Random Forest, and XGBoost models, performs validation-based tuning, evaluates the tuned models on a held-out test set, and provides feature-level interpretation to understand model behavior.
