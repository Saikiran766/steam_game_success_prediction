# Methodology — Steam Games Success Prediction

## 1. Study Overview

This case study investigates whether machine learning models can predict the success of Steam games using game metadata. The workflow covers data auditing, cleaning, target construction, feature engineering, data splitting, model training, evaluation, and business interpretation.

The study also compares a baseline experiment using a broad set of features with a restricted-feature experiment using a smaller set of selected variables.

## 2. Dataset and Scope

The dataset contains Steam game information obtained from the FronkonGames Steam Games Dataset hosted on Hugging Face.

- Original dataset: 125,855 rows and 39 columns.
- After removing one row with a missing game name: 125,854 rows.
- Analysis scope: games released before 1 January 2025.
- Games in the analysis scope: 97,565.

The release-date restriction is intended to provide a more mature observation period for estimating game success. It does not eliminate every possible source of bias in the available data.

## 3. Data Cleaning and Target Construction

The dataset was audited for missing values, data types, and relevant columns. One row with a missing game name was removed.

The binary target variable, `Success`, was constructed from the lower bound of the estimated owners field:

- **Success (1):** estimated lower-bound owners of at least 20,000.
- **Not Success (0):** estimated lower-bound owners below 20,000.

The resulting dataset contained:

| Class | Games | Percentage |
|---|---:|---:|
| Success | 24,598 | 25.21% |
| Not Success | 72,967 | 74.79% |
| Total | 97,565 | 100.00% |

This is an operational definition of success for the case study. It should not be interpreted as an official Steam classification or a direct measure of revenue or profitability.

## 4. Data Splitting

The dataset was divided into training, validation, and test sets using stratified sampling with random seed 42.

| Split | Records |
|---|---:|
| Training | 58,539 |
| Validation | 19,513 |
| Test | 19,513 |
| Total | 97,565 |

Stratification preserves approximately the same target-class proportions across the three splits.

The training set is used to fit models, the validation set supports model selection and threshold tuning, and the test set is reserved for final evaluation.

## 5. Feature Preparation

Feature preparation includes identifying relevant columns, handling categorical and numerical variables, and transforming predictors into model-compatible representations.

The baseline preprocessing workflow produced 415 transformed predictors, in addition to the target column. The preprocessing pipeline was designed to prepare the features for the selected machine learning algorithms.

The restricted-feature experiment uses the following seven variables:

- Required age
- Price
- DiscountDLC count
- Windows
- Mac
- Linux
- Achievements

This experiment investigates performance when the models receive a smaller feature set. It is called a restricted-feature experiment rather than a verified pre-launch prediction experiment, because the availability of every feature at a game's launch has not been established.

## 6. Machine Learning Models

Three classification algorithms were compared:

1. **Logistic Regression** — a linear classification baseline.
2. **Random Forest** — an ensemble of decision trees.
3. **XGBoost** — a gradient-boosted tree model.

The models were evaluated using raw-feature and preprocessed-feature workflows, followed by the restricted-feature experiment.

## 7. Evaluation Metrics

The study uses several complementary classification metrics:

- **Accuracy:** proportion of predictions that are correct.
- **Precision:** proportion of predicted successes that are actual successes.
- **Recall:** proportion of actual successes identified by the model.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** ability to rank positive examples above negative examples.
- **Balanced accuracy:** average of the recall values for the two classes.

Confusion matrices and feature-importance outputs are also used to inspect model behaviour and understand which features contribute to predictions.

Because the target classes are imbalanced, accuracy alone is insufficient for judging model usefulness.

## 8. Baseline Results

The following results were obtained on the held-out test set.

| Workflow | Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Raw features | Logistic Regression | 0.7663 | 0.5356 | 0.5490 | 0.5422 | 0.7498 |
| Raw features | Random Forest | 0.8791 | 0.8329 | 0.6512 | 0.7309 | 0.8918 |
| Raw features | XGBoost | 0.8804 | 0.8678 | 0.6203 | 0.7235 | 0.8921 |
| Preprocessed | Logistic Regression | 0.8284 | 0.7084 | 0.5427 | 0.6146 | 0.8051 |
| Preprocessed | Random Forest | 0.8587 | 0.7213 | 0.7161 | 0.7187 | 0.8894 |
| Preprocessed | XGBoost | 0.8815 | 0.8538 | 0.6396 | 0.7314 | 0.8957 |

In the baseline comparison, preprocessed XGBoost achieved the highest ROC-AUC and F1-score among the six reported workflows. These results describe this dataset and evaluation setup; they do not guarantee similar performance on other datasets or future releases.

## 9. Restricted-Feature Experiment

Decision thresholds were selected using the validation set and evaluated on the held-out test set.

| Model | Threshold | Accuracy | Precision | Recall | F1-score | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.40 | 0.6207 | 0.3585 | 0.6388 | 0.4593 | 0.6267 | 0.6635 |
| Random Forest | 0.50 | 0.6860 | 0.4073 | 0.5392 | 0.4641 | 0.6373 | 0.6913 |
| XGBoost | 0.24 | 0.6657 | 0.3899 | 0.5772 | 0.4654 | 0.6364 | 0.7024 |

The restricted-feature results show a different precision-recall trade-off from the baseline experiment. XGBoost achieved the highest ROC-AUC and F1-score in this comparison, while Logistic Regression had the highest recall.

The selected thresholds reflect the validation-based tuning procedure and should not be interpreted as universally optimal operating points.

## 10. Interpretation and Business Relevance

The analysis can help illustrate how game metadata relates to estimated ownership-based success and how model choice changes the balance between identifying successful games and avoiding false positives.

Potential applications include exploratory market analysis, prioritising games for further investigation, and studying relationships between game characteristics and estimated success.

These results are predictive associations, not evidence that changing a particular game feature will cause higher sales or player adoption.

## 11. Limitations

1. **Estimated ownership:** the target is based on an estimated owner range, not verified sales or revenue.
2. **Class imbalance:** successful games form approximately one-quarter of the analysis dataset.
3. **Feature availability:** the restricted feature set has not been verified as fully available before launch.
4. **Temporal variation:** market conditions, genres, and player preferences change over time.
5. **Potential data leakage:** features that directly encode or closely reflect popularity may make the baseline task easier than a genuine pre-launch forecasting task.
6. **External validity:** performance on this dataset may not generalise to different time periods, platforms, or game populations.

## 12. Reproducibility

The Python scripts in `src/` implement the data preparation, training, evaluation, and interpretation workflow. `requirements.txt` lists the Python dependencies.

To reproduce the experiments, install the dependencies, obtain the dataset, place it in the location expected by the scripts, and execute the scripts in the order documented in the project README.

The dataset and Power BI report are not included in the Git repository because the project configuration excludes large data and `.pbix` files.

## 13. Conclusion

This case study demonstrates a complete data science workflow for Steam game success prediction, from data preparation through comparative modelling and interpretation. The baseline and restricted-feature experiments illustrate how available predictors and decision thresholds affect classification performance.

The findings should be interpreted within the study's target definition, dataset limitations, and feature-availability constraints.
