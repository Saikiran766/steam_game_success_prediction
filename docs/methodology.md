# Methodology

## 1. Problem Definition

The project is formulated as a binary classification problem for predicting Steam game success.

Each row represents one Steam game. The modeling process is designed to use information that is available for the game without using the variables that directly define the final success outcome as predictive inputs.

The two target classes are:

- `Success`
- `Not Success`

The project uses a release-maturity filter so that very recent games are not treated in the same way as older games.

---

## 2. Data Ingestion and Audit

The raw Steam Games dataset was loaded from:

```text
data/raw/steam_games.csv
```

The raw file contains 125,855 rows and 39 columns.

The first audit checked:

- Number of rows and columns
- Column names
- Data types
- Missing values
- Unique values
- Duplicate App IDs
- Numeric and categorical fields
- Basic alignment of the first records
- Dataset memory usage

The raw dataset was kept unchanged.

The audit results were saved in:

```text
results/metrics/data_audit.txt
```

A fixed random seed of `42` was used throughout the workflow where random operations were required.

---

## 3. Data Cleaning

Cleaning was performed before target creation and modeling.

The cleaning process included:

- Removing completely duplicated rows
- Checking for missing App IDs
- Checking for invalid App IDs
- Removing duplicate App IDs where necessary
- Removing the row with a missing game name
- Parsing release dates
- Checking invalid release dates
- Cleaning boolean platform fields
- Checking all-missing columns
- Preserving post-release outcome fields temporarily for target construction

The cleaned dataset contained:

```text
125,854 rows
39 columns
```

The source Windows field did not contain usable values after cleaning. Its values were effectively missing for the dataset, so Windows information was not fabricated.

Mac and Linux platform information remained available.

The cleaned data was subsequently used for target construction.

---

## 4. Target Definition

The target was constructed from the estimated-owner range.

### Release-maturity filter

Games released on or after:

```text
2025-01-01
```

were excluded from target construction.

This removed:

```text
28,289 recent games
```

and retained:

```text
97,565 mature games
```

### Success rule

The lower bound of the estimated-owner range was used.

The operational rule was:

```text
Success      = estimated owners lower bound >= 20,000
Not Success  = estimated owners lower bound < 20,000
```

The target-source variables were not used as predictive features.

### Target distribution before balancing

```text
Success      : 24,598 (25.21%)
Not Success  : 72,967 (74.79%)
```

The targeted dataset was saved as:

```text
data/cleaned/steam_games_targeted.csv
```

The target definition was documented in:

```text
results/metrics/target_definition.txt
```

The 20,000-owner threshold and January 1, 2025 maturity cutoff are operational choices for this case study. They should therefore be interpreted within the scope of this project rather than as universal definitions of Steam game success.

---

## 5. Dataset Balancing

The target distribution after maturity filtering was substantially uneven.

For the modeling dataset, all 24,598 Success games were retained and an equal number of Not Success games were randomly sampled.

The random sampling used:

```text
random_state = 42
```

Final balanced dataset:

```text
Total       : 49,196
Success     : 24,598
Not Success : 24,598
```

The balanced dataset was saved as:

```text
data/processed/steam_games_balanced.csv
```

The sampling procedure creates a 50/50 modeling population. Consequently, the final performance measurements should not be interpreted as performance under the original natural prevalence of successful Steam games.

Balancing details were recorded in:

```text
results/metrics/balancing_summary.txt
```

---

## 6. Train / Validation / Test Split

The balanced dataset was divided using stratified sampling.

The split ratio was:

```text
60% Training
20% Validation
20% Test
```

A fixed random seed of `42` was used.

Final split sizes:

| Split | Rows |
|---|---:|
| Training | 29,517 |
| Validation | 9,839 |
| Test | 9,840 |

Class proportions were preserved across the three subsets.

The files are:

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

The test set was kept separate from model training and hyperparameter tuning and was used for final evaluation.

---

## 7. Exploratory Data Analysis

Exploratory data analysis is represented by:

```text
notebooks/01_eda.ipynb
```

The notebook is designed around the training data so that exploratory analysis does not use the held-out test set for modeling decisions.

The EDA stage examines the training data for:

- Target distribution
- Numeric feature distributions
- Missing-value patterns
- Categorical information
- Platform information
- Price-related information
- Game metadata patterns
- Potentially useful relationships between available features and the target

The EDA notebook should be run from the project root so that its relative paths resolve correctly.

---

## 8. Feature Eligibility and Feature Engineering

Variables that directly define the target or represent post-release outcomes were excluded from the predictive feature set.

Identifiers, raw free-text fields, and target-source information were not used as direct model inputs.

The preprocessing stage created a compact feature representation from the available game metadata.

The final feature matrix contains:

```text
98 features
```

Feature engineering included:

- Numeric conversion of applicable numeric columns
- Training-derived missing-value handling
- Numeric scaling
- Top-50 category multi-hot encoding
- Top-30 genre multi-hot encoding
- Supported-language count
- Full-audio-language count
- Tag count
- Usable platform indicators
- Removal of unusable or inappropriate fields

The very high-cardinality language field was not fully multi-hot encoded. Instead, a count-based representation was used to avoid an extremely large sparse/dense feature matrix.

Developers and publishers were not included in the final feature representation.

The fitted preprocessing object was saved as:

```text
results/models/preprocessor.joblib
```

The processed feature matrices are:

```text
data/processed/train_features.csv
data/processed/validation_features.csv
data/processed/test_features.csv
```

The preprocessing summary is stored in:

```text
results/metrics/preprocessing_summary.txt
```

---

## 9. Training Pipeline

Four modeling approaches were evaluated:

1. Majority-class baseline
2. Logistic Regression
3. Random Forest
4. XGBoost

The primary validation model-selection metric was:

```text
ROC-AUC
```

The baseline establishes the performance of a simple classifier before applying machine learning models.

The trained models were saved under:

```text
results/models/
```

---

## 10. Initial Validation Results

Before tuning, the validation results were:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.4999 | 0.4999 | 1.0000 | 0.6666 | 0.5000 |
| Logistic Regression | 0.7096 | 0.7499 | 0.6290 | 0.6841 | 0.7845 |
| Random Forest | 0.7283 | 0.7497 | 0.6855 | 0.7162 | 0.8043 |
| XGBoost | 0.7374 | 0.7744 | 0.6699 | 0.7183 | 0.8134 |

These results were generated using the validation set while the test set remained unused.

---

## 11. Hyperparameter Tuning

Hyperparameter tuning was performed on the validation set.

The search was deliberately kept relatively small so that the project remained computationally practical.

### Logistic Regression

The tested values included:

```text
C = 0.1, 1, 10
class_weight = None
```

Selected configuration:

```text
C = 0.1
```

Validation ROC-AUC:

```text
0.7845
```

### Random Forest

The search varied:

```text
n_estimators = 200, 400
max_depth = None, 15
min_samples_leaf = 1, 2
max_features = sqrt
class_weight = None
```

Selected configuration:

```text
n_estimators = 400
max_depth = None
min_samples_leaf = 2
max_features = sqrt
```

Validation ROC-AUC:

```text
0.8128
```

### XGBoost

The search varied:

```text
n_estimators = 200, 400
max_depth = 3, 5
learning_rate = 0.05, 0.10
subsample = 0.8
colsample_bytree = 0.8
```

Selected configuration:

```text
n_estimators = 200
max_depth = 5
learning_rate = 0.10
subsample = 0.8
colsample_bytree = 0.8
```

Validation ROC-AUC:

```text
0.8160
```

---

## 12. Classification Threshold Selection

The default probability threshold of 0.50 was not assumed to be optimal for the final classification decision.

Thresholds were selected using the validation set.

The selected thresholds were:

| Model | Validation Threshold |
|---|---:|
| Logistic Regression | 0.33 |
| Random Forest | 0.40 |
| XGBoost | 0.38 |

Hyperparameter selection was based on validation ROC-AUC, while threshold selection used validation F1.

The test set was not used for threshold selection.

---

## 13. Final Test Evaluation

After model selection and threshold selection, the held-out test set was evaluated.

The final test evaluation was performed once.

### Final results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6666 | 0.6208 | 0.8561 | 0.7197 | 0.7864 | 0.8019 |
| Random Forest | 0.7206 | 0.6876 | 0.8087 | 0.7433 | 0.8226 | 0.8338 |
| XGBoost | 0.7186 | 0.6828 | 0.8165 | 0.7437 | 0.8243 | 0.8370 |

The XGBoost model obtained the highest test ROC-AUC and PR-AUC among the three trained machine-learning models. Random Forest produced a very similar F1 value.

The final evaluation files are stored in:

```text
results/metrics/final_test_comparison.csv
results/metrics/final_test_metrics.json
results/metrics/final_test_evaluation_summary.txt
```

The test figures include:

```text
results/figures/test_roc_curves.png
results/figures/logistic_regression_test_confusion_matrix.png
results/figures/random_forest_test_confusion_matrix.png
results/figures/xgboost_test_confusion_matrix.png
```

---

## 14. Model Interpretation

Model interpretation was performed after final model evaluation without retraining the models.

To preserve the held-out test set as an evaluation set, the main interpretation analysis uses training and validation data.

### Permutation importance

Permutation importance was calculated on the validation data using ROC-AUC as the scoring metric.

This measures how much model performance changes when individual features are randomly permuted.

Outputs include:

```text
results/metrics/logistic_regression_permutation_importance.csv
results/metrics/random_forest_permutation_importance.csv
results/metrics/xgboost_permutation_importance.csv
```

### Logistic Regression coefficients

Logistic Regression coefficients were extracted to identify features associated with positive or negative model contributions.

Outputs:

```text
results/metrics/logistic_coefficients.csv
results/metrics/logistic_positive_coefficients.csv
results/metrics/logistic_negative_coefficients.csv
```

### Tree-based feature importance

Feature importance values were extracted from:

- Random Forest
- XGBoost

These are treated as supplementary interpretation measures.

Outputs:

```text
results/metrics/random_forest_feature_importance.csv
results/metrics/xgboost_feature_importance.csv
```

### Error analysis

Validation predictions were also used to examine model performance across release-year segments.

The resulting analysis is stored in:

```text
results/metrics/xgboost_validation_error_by_release_year.csv
```

Interpretation plots are stored in:

```text
results/figures/
```

A readable interpretation report is available at:

```text
results/metrics/interpretation_summary.txt
```

---

## 15. Reproducibility

The project uses a fixed random seed:

```text
42
```

Random sampling and stratified splitting therefore produce reproducible dataset partitions when the same data and software environment are used.

The project separates:

- Raw data
- Cleaned data
- Targeted data
- Balanced data
- Train/validation/test data
- Processed features
- Models
- Metrics
- Figures
- Documentation

This separation makes the workflow easier to reproduce and audit.

---

## 16. Limitations

### Target definition

The 20,000 estimated-owner threshold is an operational choice for this case study. It is not presented as a universal definition of Steam game success.

### Release maturity

The January 1, 2025 maturity cutoff is also an operational choice. Different maturity periods could produce different target distributions.

### Estimated-owner ranges

Steam estimated owners are represented as ranges rather than exact owner counts. The lower-bound rule therefore simplifies the underlying ownership information.

### Balanced sampling

The final modeling population was intentionally balanced at 50% Success and 50% Not Success. Consequently, the final performance metrics do not represent the original natural class prevalence.

### Source platform information

The Windows field did not contain usable values in the source data after cleaning. No Windows values were fabricated.

### Feature representation

The final feature set uses selected metadata representations rather than every available raw column. In particular, high-cardinality text and categorical information was constrained to keep the feature space computationally manageable.

### Generalization

The results are specific to the dataset, target definition, sampling strategy, preprocessing decisions, and models used in this study. They should not be treated as universal predictions for all future Steam releases.

---

## 17. Workflow Summary

The complete implemented workflow is:

```text
Raw Steam Dataset
        |
        v
Data Audit
        |
        v
Data Cleaning
        |
        v
Target Creation
        |
        v
Balanced Dataset
        |
        v
Stratified 60/20/20 Split
        |
        +----------------------+
        |                      |
        v                      |
Training Data             Validation Data
        |                      |
        v                      |
Training-derived             Evaluation/
Preprocessing                Tuning
        |                      |
        +----------+-----------+
                   |
                   v
          Model Training
                   |
                   v
          Hyperparameter Tuning
                   |
                   v
       Validation Threshold Selection
                   |
                   v
          Final Test Evaluation
                   |
                   v
          Model Interpretation
                   |
                   v
          Results and Reporting
```

---

## 18. Re-running the Project

From the project root:

```powershell
cd D:\Steam_Games_Success_Prediction
```

Run:

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

It should be used with the training data for exploratory analysis.

---

## 19. Final Project Status

The implemented pipeline currently contains:

- Data audit
- Data cleaning
- Target construction
- Balanced dataset creation
- Stratified data splitting
- Preprocessing and feature engineering
- Baseline model
- Logistic Regression
- Random Forest
- XGBoost
- Validation-based hyperparameter tuning
- Validation-based threshold selection
- Final held-out test evaluation
- Confusion matrices
- ROC curves
- Permutation importance
- Logistic coefficient analysis
- Tree-based feature importance
- Validation segment-error analysis
- README documentation

The main remaining documentation/environment file is:

```text
requirements.txt
```

The methodology document itself should be saved as:

```text
docs/methodology.md
```
