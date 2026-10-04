# Fit training-only preprocessing and apply the same transformation to
# validation and test data without using the test set during fitting.

from pathlib import Path
import json
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(r"D:\Steam_Games_Success_Prediction")

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

TRAIN_OUTPUT = DATA_DIR / "train_features.csv"
VALIDATION_OUTPUT = DATA_DIR / "validation_features.csv"
TEST_OUTPUT = DATA_DIR / "test_features.csv"

PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
SUMMARY_FILE = METRICS_DIR / "preprocessing_summary.txt"
FEATURE_NAMES_FILE = METRICS_DIR / "feature_names.json"

TOP_K_CATEGORIES = 50
TOP_K_GENRES = 30

METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading train, validation, and test datasets...")

train = pd.read_csv(TRAIN_FILE, low_memory=False)
validation = pd.read_csv(VALIDATION_FILE, low_memory=False)
test = pd.read_csv(TEST_FILE, low_memory=False)

print(f"Training rows: {len(train)}")
print(f"Validation rows: {len(validation)}")
print(f"Test rows: {len(test)}")

TARGET = "Success"

# These fields are either identifiers, raw text, URLs, target-source fields,
# or outcomes that happen after release and therefore cannot be predictors.
DROP_COLUMNS = [
    "AppID", "Name", "Release date", "About the game",
    "Header image", "Website", "Support url", "Support email",
    "Metacritic url", "Notes", "Screenshots", "Movies",
    "Developers", "Publishers", "Categories", "Genres", "Tags",
    "Supported languages", "Full audio languages",
    "Estimated owners", "owners_lower", "owners_upper", "Success Label",
    "Peak CCU", "Reviews", "Positive", "Negative", "Recommendations",
    "Average playtime two weeks", "Median playtime forever",
    "Median playtime two weeks",
    "Windows",
]

def add_rowwise_features(df):
    # Create features using only information available in the same row.
    out = pd.DataFrame(index=df.index)

    if "Release date" in df.columns:
        release = pd.to_datetime(df["Release date"], errors="coerce")
        out["release_year"] = release.dt.year
        out["release_month"] = release.dt.month

    # Explicitly coerce numeric source fields. This prevents a malformed or
    # string-valued source entry from reaching the numeric imputer.
    numeric_source_columns = [
        "Price",
        "Required age",
        "Discount",
        "DLC count",
        "Metacritic score",
        "Achievements",
    ]

    for col in numeric_source_columns:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    if "Price" in out.columns:
        price = out["Price"]
        out["price_log1p"] = np.log1p(price.clip(lower=0))
        out["price_free"] = (price == 0).astype("float")
        out["price_band"] = pd.cut(
            price,
            bins=[-0.01, 0, 5, 15, 30, np.inf],
            labels=["free", "low", "medium", "high", "premium"]
        ).astype("string")

    if "Discount" in out.columns:
        out["discounted"] = (out["Discount"] > 0).astype("float")

    if "DLC count" in out.columns:
        out["dlc_count_log1p"] = np.log1p(
            out["DLC count"].clip(lower=0)
        )

    # Windows is intentionally excluded because the cleaned source has no
    # usable Windows values. Mac and Linux are retained.
    for col in ["Mac", "Linux"]:
        if col in df.columns:
            values = df[col].astype("string").str.lower().str.strip()
            out[f"platform_{col.lower()}"] = values.map({
                "true": 1.0,
                "false": 0.0,
                "1": 1.0,
                "0": 0.0,
            })

    for col, feature_name in [
        ("Supported languages", "language_count"),
        ("Full audio languages", "audio_language_count"),
        ("Tags", "tag_count"),
    ]:
        if col in df.columns:
            def count_items(value):
                if pd.isna(value):
                    return np.nan
                text = str(value).strip()
                if not text:
                    return 0.0
                parts = re.split(r"[,\|;]+", text)
                return float(sum(bool(p.strip()) for p in parts))
            out[feature_name] = df[col].apply(count_items)

    return out

def get_top_labels(series, k):
    counts = {}
    for value in series.dropna().astype(str):
        for part in re.split(r"[,\|;]+", value):
            label = part.strip()
            if label:
                counts[label] = counts.get(label, 0) + 1

    return [
        label
        for label, _ in
        sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:k]
    ]

def multi_hot_top_k(series, labels, prefix):
    result = pd.DataFrame(index=series.index)

    for label in labels:
        column_name = f"{prefix}__{label}"
        result[column_name] = series.fillna("").astype(str).apply(
            lambda value: float(
                label in {
                    part.strip()
                    for part in re.split(r"[,\|;]+", value)
                    if part.strip()
                }
            )
        )

    return result

category_labels = get_top_labels(
    train["Categories"], TOP_K_CATEGORIES
) if "Categories" in train.columns else []

genre_labels = get_top_labels(
    train["Genres"], TOP_K_GENRES
) if "Genres" in train.columns else []

print(
    f"Categories: {len(category_labels)} training labels selected "
    f"(top {TOP_K_CATEGORIES})."
)
print(
    f"Genres: {len(genre_labels)} training labels selected "
    f"(top {TOP_K_GENRES})."
)

train_base = add_rowwise_features(train)
validation_base = add_rowwise_features(validation)
test_base = add_rowwise_features(test)

category_train = multi_hot_top_k(
    train["Categories"], category_labels, "category"
)
category_validation = multi_hot_top_k(
    validation["Categories"], category_labels, "category"
)
category_test = multi_hot_top_k(
    test["Categories"], category_labels, "category"
)

genre_train = multi_hot_top_k(
    train["Genres"], genre_labels, "genre"
)
genre_validation = multi_hot_top_k(
    validation["Genres"], genre_labels, "genre"
)
genre_test = multi_hot_top_k(
    test["Genres"], genre_labels, "genre"
)

train_features_raw = pd.concat(
    [train_base, category_train, genre_train],
    axis=1
)
validation_features_raw = pd.concat(
    [validation_base, category_validation, genre_validation],
    axis=1
)
test_features_raw = pd.concat(
    [test_base, category_test, genre_test],
    axis=1
)

# Remove any accidental columns that are not explicitly represented in the
# feature-building logic above. The intended predictor set is therefore
# controlled and auditable.
feature_columns = list(train_features_raw.columns)

validation_features_raw = validation_features_raw.reindex(
    columns=feature_columns
)
test_features_raw = test_features_raw.reindex(
    columns=feature_columns
)

categorical_columns = [
    col for col in feature_columns
    if (
        train_features_raw[col].dtype.name
        in ["object", "string", "category"]
    )
]

numeric_columns = [
    col for col in feature_columns
    if col not in categorical_columns
]

# Safety check: every numeric feature must actually be numeric before the
# median imputer is fitted.
for col in numeric_columns:
    train_features_raw[col] = pd.to_numeric(
        train_features_raw[col], errors="coerce"
    )
    validation_features_raw[col] = pd.to_numeric(
        validation_features_raw[col], errors="coerce"
    )
    test_features_raw[col] = pd.to_numeric(
        test_features_raw[col], errors="coerce"
    )

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )),
])

transformers = []

if numeric_columns:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_columns)
    )

if categorical_columns:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_columns)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)

X_train = preprocessor.fit_transform(train_features_raw)
X_validation = preprocessor.transform(validation_features_raw)
X_test = preprocessor.transform(test_features_raw)

feature_names = list(preprocessor.get_feature_names_out())

X_train = pd.DataFrame(X_train, columns=feature_names)
X_validation = pd.DataFrame(X_validation, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)

X_train[TARGET] = train[TARGET].values
X_validation[TARGET] = validation[TARGET].values
X_test[TARGET] = test[TARGET].values

# Final leakage check.
LEAKAGE_TERMS = [
    "estimated owners", "owners_lower", "owners_upper",
    "peak ccu", "reviews", "positive", "negative",
    "recommendations", "average playtime", "median playtime",
]

leakage_features = [
    feature for feature in feature_names
    if any(term in feature.lower() for term in LEAKAGE_TERMS)
]

if leakage_features:
    raise ValueError(
        "Potential leakage features detected: "
        + ", ".join(leakage_features)
    )

X_train.to_csv(TRAIN_OUTPUT, index=False)
X_validation.to_csv(VALIDATION_OUTPUT, index=False)
X_test.to_csv(TEST_OUTPUT, index=False)

joblib.dump(preprocessor, PREPROCESSOR_FILE)

FEATURE_NAMES_FILE.write_text(
    json.dumps(feature_names, indent=2),
    encoding="utf-8"
)

summary = [
    "Preprocessing Summary",
    "======================",
    "All data-driven preprocessing is fitted on training data only.",
    "Validation and test data use the fitted training preprocessor.",
    "",
    f"Training rows: {len(train)}",
    f"Validation rows: {len(validation)}",
    f"Test rows: {len(test)}",
    "",
    f"Top Categories used: {len(category_labels)}",
    f"Top Genres used: {len(genre_labels)}",
    "",
    "Windows excluded because the cleaned source contains no usable values.",
    "User score excluded because the training split contains no observed values.",
    "",
    f"Raw feature columns before transformation: {len(feature_columns)}",
    f"Final transformed feature count: {len(feature_names)}",
    "",
    f"Train features: {TRAIN_OUTPUT}",
    f"Validation features: {VALIDATION_OUTPUT}",
    f"Test features: {TEST_OUTPUT}",
    f"Fitted preprocessor: {PREPROCESSOR_FILE}",
    f"Feature names: {FEATURE_NAMES_FILE}",
]

SUMMARY_FILE.write_text("\n".join(summary), encoding="utf-8")

print("\nPreprocessing completed successfully.")
print(f"Final feature count: {len(feature_names)}")
print("\nProcessed datasets saved:")
print(TRAIN_OUTPUT)
print(VALIDATION_OUTPUT)
print(TEST_OUTPUT)
print(f"\nFitted preprocessor:\n{PREPROCESSOR_FILE}")
print(f"\nSummary:\n{SUMMARY_FILE}")
print(f"\nFeature names:\n{FEATURE_NAMES_FILE}")
