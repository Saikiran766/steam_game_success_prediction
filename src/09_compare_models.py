"""
Compare raw-feature and preprocessed-feature validation results.

Inputs:
    results/metrics/raw_model_validation_metrics.csv
    results/metrics/preprocessed_model_validation_metrics.csv

Outputs:
    results/metrics/model_comparison.csv
    results/metrics/model_comparison_summary.txt
    results/figures/model_comparison_f1.png
    results/figures/model_comparison_roc_auc.png
    results/figures/model_comparison_accuracy.png

This script uses validation metrics only. It does not evaluate or select models
using the test set.
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT_DIR / "results" / "metrics"
FIGURES_DIR = ROOT_DIR / "results" / "figures"

RAW_METRICS_FILE = METRICS_DIR / "raw_model_validation_metrics.csv"
PREPROCESSED_METRICS_FILE = METRICS_DIR / "preprocessed_model_validation_metrics.csv"

OUTPUT_CSV = METRICS_DIR / "model_comparison.csv"
OUTPUT_REPORT = METRICS_DIR / "model_comparison_summary.txt"

METRICS_TO_COMPARE = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}


def load_metrics(path, experiment_name):
    if not path.exists():
        print(f"ERROR: Required metrics file not found:\n{path}")
        sys.exit(1)

    frame = pd.read_csv(path)
    required = {"model", *METRICS_TO_COMPARE}
    missing = required - set(frame.columns)
    if missing:
        print(f"ERROR: {path.name} is missing columns: {sorted(missing)}")
        sys.exit(1)

    frame = frame.copy()
    frame["experiment"] = experiment_name
    frame["model"] = frame["model"].astype(str)
    for metric in METRICS_TO_COMPARE:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    if frame[METRICS_TO_COMPARE].isna().any().any():
        print(f"ERROR: Missing or non-numeric metric values found in {path.name}.")
        sys.exit(1)

    return frame[["experiment", "model", *METRICS_TO_COMPARE]]


def save_bar_chart(data, metric, output_path):
    pivot = data.pivot(index="model", columns="experiment", values=metric)
    pivot = pivot.sort_index()

    ax = pivot.plot(kind="bar", figsize=(9, 5), width=0.78)
    ax.set_title(f"Raw vs Preprocessed Models — {METRIC_LABELS[metric]}")
    ax.set_xlabel("Model")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Feature experiment")
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_metrics(RAW_METRICS_FILE, "Raw features")
    preprocessed = load_metrics(PREPROCESSED_METRICS_FILE, "Preprocessed features")
    combined = pd.concat([raw, preprocessed], ignore_index=True)

    # Save long-form comparison so it is easy to load into Power BI/Tableau.
    combined.to_csv(OUTPUT_CSV, index=False)

    # Produce three simple, report-ready comparison plots.
    save_bar_chart(combined, "accuracy", FIGURES_DIR / "model_comparison_accuracy.png")
    save_bar_chart(combined, "f1", FIGURES_DIR / "model_comparison_f1.png")
    save_bar_chart(combined, "roc_auc", FIGURES_DIR / "model_comparison_roc_auc.png")

    report_lines = [
        "STEAM GAMES SUCCESS PREDICTION — MODEL COMPARISON",
        "=" * 62,
        "Source: validation metrics for raw-feature and preprocessed-feature experiments.",
        "The test set was not used for this comparison.",
        "",
        "METRICS BY MODEL AND EXPERIMENT",
        "-" * 62,
    ]

    for model_name in sorted(combined["model"].unique()):
        model_rows = combined[combined["model"] == model_name].set_index("experiment")
        report_lines.append(model_name)
        for experiment_name in ["Raw features", "Preprocessed features"]:
            if experiment_name not in model_rows.index:
                continue
            row = model_rows.loc[experiment_name]
            report_lines.append(
                f"  {experiment_name}: "
                + ", ".join(
                    f"{METRIC_LABELS[metric]}={row[metric]:.4f}"
                    for metric in METRICS_TO_COMPARE
                )
            )

        if {"Raw features", "Preprocessed features"}.issubset(set(model_rows.index)):
            raw_row = model_rows.loc["Raw features"]
            pre_row = model_rows.loc["Preprocessed features"]
            report_lines.append("  Change after preprocessing (percentage points):")
            for metric in METRICS_TO_COMPARE:
                change = (pre_row[metric] - raw_row[metric]) * 100
                report_lines.append(f"    {METRIC_LABELS[metric]}: {change:+.2f}")
        report_lines.append("")

    # Select the best experiment/model by validation F1, not test performance.
    best_row = combined.loc[combined["f1"].idxmax()]
    report_lines.extend([
        "BEST VALIDATION F1 RESULT",
        "-" * 62,
        f"Model: {best_row['model']}",
        f"Experiment: {best_row['experiment']}",
        f"F1-score: {best_row['f1']:.4f}",
        f"ROC-AUC: {best_row['roc_auc']:.4f}",
        "",
        "OUTPUTS",
        "-" * 62,
        f"Comparison CSV: {OUTPUT_CSV.relative_to(ROOT_DIR)}",
        f"F1 chart: {(FIGURES_DIR / 'model_comparison_f1.png').relative_to(ROOT_DIR)}",
        f"ROC-AUC chart: {(FIGURES_DIR / 'model_comparison_roc_auc.png').relative_to(ROOT_DIR)}",
        f"Accuracy chart: {(FIGURES_DIR / 'model_comparison_accuracy.png').relative_to(ROOT_DIR)}",
        "",
        "INTERPRETATION NOTE",
        "-" * 62,
        "These are validation results for model comparison, not final held-out test results.",
        "The raw-feature baseline uses minimal compatibility transformations; it is not completely unprocessed raw data.",
    ])

    OUTPUT_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("Model comparison completed successfully.")
    print("\nValidation results:")
    display_columns = ["experiment", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    print(combined[display_columns].to_string(index=False, formatters={
        metric: "{:.4f}".format for metric in METRICS_TO_COMPARE
    }))
    print(f"\nComparison CSV: {OUTPUT_CSV}")
    print(f"Summary report: {OUTPUT_REPORT}")
    print(f"Charts saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
