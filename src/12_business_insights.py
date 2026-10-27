"""
Corrected descriptive business insights for Steam Games Success Prediction.

Uses:
- Tags for tag/genre-like analysis (not the mislabeled Genres column).
- Categories is excluded from genre analysis because observed values include publisher names.
- Minimum sample size of 100 games for release-year and tag comparisons.
- Median and mean Peak CCU by success group.

Run from the project root:
    python src/12_business_insights.py

Input:
    data/cleaned/steam_games_targeted.csv

Outputs:
    results/metrics/business_insights_summary.txt
    results/metrics/success_rate_by_year.csv
    results/metrics/success_rate_by_price_range.csv
    results/metrics/success_rate_by_tag.csv
    results/metrics/player_activity_by_success.csv
    charts in results/figures/
"""

from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "cleaned" / "steam_games_targeted.csv"
METRICS_DIR = ROOT_DIR / "results" / "metrics"
FIGURES_DIR = ROOT_DIR / "results" / "figures"
MIN_GROUP_SIZE = 100

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def find_column(df, name):
    return {str(c).strip().lower(): c for c in df.columns}.get(name.lower())


def parse_labels(value):
    """Split the dataset's comma-separated label fields."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    # These dataset fields are comma-separated text, not reliably Python lists.
    return [
        part.strip().strip("'\"")
        for part in re.split(r"\s*,\s*", text.strip("[]"))
        if part.strip().strip("'\"")
    ]


def save_bar_chart(data, label_col, value_col, title, ylabel, path,
                   horizontal=False, limit=None):
    plot_data = data.copy()
    if limit:
        plot_data = plot_data.head(limit)
    if plot_data.empty:
        return

    if horizontal:
        plot_data = plot_data.sort_values(value_col, ascending=True)
        fig, ax = plt.subplots(figsize=(11, max(5, min(12, 0.34 * len(plot_data) + 1.5))))
        ax.barh(plot_data[label_col].astype(str), plot_data[value_col])
        ax.set_xlabel(ylabel)
        ax.set_ylabel("")
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(plot_data[label_col].astype(str), plot_data[value_col])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=35)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    if "Success" not in df.columns:
        raise ValueError("Required target column 'Success' is missing.")

    target = df["Success"]
    if target.dtype == object:
        mapped = target.astype(str).str.strip().str.lower().map({
            "success": 1, "successful": 1, "1": 1, "true": 1,
            "not success": 0, "not successful": 0, "0": 0, "false": 0,
        })
        if mapped.notna().all():
            df["_success"] = mapped.astype(int)
        else:
            df["_success"] = pd.to_numeric(target, errors="coerce")
    else:
        df["_success"] = pd.to_numeric(target, errors="coerce")

    df = df[df["_success"].isin([0, 1])].copy()
    df["_success"] = df["_success"].astype(int)
    df["_success_label"] = df["_success"].map({1: "Success", 0: "Not Success"})

    lines = [
        "STEAM GAMES SUCCESS PREDICTION — CORRECTED BUSINESS INSIGHTS",
        "=" * 64,
        f"Rows analyzed: {len(df):,}",
        f"Overall success rate: {df['_success'].mean() * 100:.2f}%",
        "",
        "Important data note:",
        "- The observed Genres values include feature names such as Single-player and Steam Trading Cards.",
        "- The observed Categories values include publisher names such as MangaGamer and 8floor.",
        "- Therefore, Tags is used for tag/genre-like analysis; Categories is not interpreted as game category.",
        "- Results are descriptive associations and do not prove causation.",
        f"- Groups with fewer than {MIN_GROUP_SIZE} games are excluded from year/tag comparisons.",
        "",
    ]

    # Release year with minimum sample size.
    release_col = find_column(df, "Release date")
    if release_col:
        dates = pd.to_datetime(df[release_col], errors="coerce")
        df["_release_year"] = dates.dt.year
        year_rows = df[df["_release_year"].between(1990, 2025, inclusive="both")]
        year_table = (
            year_rows.groupby("_release_year")
            .agg(games=("_success", "size"), success_rate=("_success", "mean"))
            .reset_index()
            .rename(columns={"_release_year": "release_year"})
        )
        year_table = year_table[year_table["games"] >= MIN_GROUP_SIZE].copy()
        year_table["success_rate_percent"] = year_table["success_rate"] * 100
        year_table.to_csv(METRICS_DIR / "success_rate_by_year.csv", index=False)
        save_bar_chart(
            year_table, "release_year", "success_rate_percent",
            f"Success Rate by Release Year (at least {MIN_GROUP_SIZE} games)",
            "Success rate (%)", FIGURES_DIR / "success_rate_by_year.png"
        )
        lines += ["SUCCESS RATE BY RELEASE YEAR (minimum sample size applied)"]
        lines += [
            year_table.sort_values("release_year")
            [["release_year", "games", "success_rate_percent"]]
            .to_string(index=False, float_format=lambda x: f"{x:.2f}")
            if not year_table.empty else "No years met the minimum sample size."
        ]
        lines.append("")

    # Price bands.
    price_col = find_column(df, "Price")
    if price_col:
        df["_price"] = pd.to_numeric(df[price_col], errors="coerce")
        priced = df[df["_price"].notna() & (df["_price"] >= 0)].copy()
        bins = [-0.001, 0, 5, 10, 20, 40, float("inf")]
        labels = ["Free", ">0–5", ">5–10", ">10–20", ">20–40", ">40"]
        priced["_price_range"] = pd.cut(
            priced["_price"], bins=bins, labels=labels, include_lowest=True
        )
        price_table = (
            priced.groupby("_price_range", observed=False)
            .agg(games=("_success", "size"), success_rate=("_success", "mean"))
            .reset_index()
            .rename(columns={"_price_range": "price_range"})
        )
        price_table["success_rate_percent"] = price_table["success_rate"] * 100
        price_table.to_csv(METRICS_DIR / "success_rate_by_price_range.csv", index=False)
        save_bar_chart(
            price_table, "price_range", "success_rate_percent",
            "Success Rate by Price Range", "Success rate (%)",
            FIGURES_DIR / "success_rate_by_price_range.png"
        )
        lines += ["SUCCESS RATE BY PRICE RANGE"]
        lines += [
            price_table[["price_range", "games", "success_rate_percent"]]
            .to_string(index=False, float_format=lambda x: f"{x:.2f}")
        ]
        lines.append("")

    # Tags: do not call these verified official genres; labels can include descriptors/features.
    tags_col = find_column(df, "Tags")
    if tags_col:
        records = []
        for tags, success in df[[tags_col, "_success"]].itertuples(index=False, name=None):
            for tag in set(parse_labels(tags)):
                records.append((tag, int(success)))

        if records:
            tag_rows = pd.DataFrame(records, columns=["tag", "success"])
            tag_table = (
                tag_rows.groupby("tag")
                .agg(games=("success", "size"), success_rate=("success", "mean"))
                .reset_index()
            )
            tag_table = tag_table[tag_table["games"] >= MIN_GROUP_SIZE].copy()
            tag_table["success_rate_percent"] = tag_table["success_rate"] * 100
            tag_table = tag_table.sort_values(
                ["success_rate", "games"], ascending=[False, False]
            )
            tag_table.to_csv(METRICS_DIR / "success_rate_by_tag.csv", index=False)
            save_bar_chart(
                tag_table, "tag", "success_rate_percent",
                f"Top Tags by Success Rate (at least {MIN_GROUP_SIZE} games)",
                "Success rate (%)", FIGURES_DIR / "top_tags_by_success_rate.png",
                horizontal=True, limit=20
            )
            lines += ["TOP TAGS BY SUCCESS RATE (tags may include genres, features, and descriptors)"]
            lines += [
                tag_table.head(15)[["tag", "games", "success_rate_percent"]]
                .to_string(index=False, float_format=lambda x: f"{x:.2f}")
                if not tag_table.empty else "No tags met the minimum sample size."
            ]
            lines.append("")
    else:
        warnings.warn("Tags column not found; skipping tag analysis.")

    # Peak concurrent players by target.
    peak_col = find_column(df, "Peak CCU")
    if peak_col:
        df["_peak_ccu"] = pd.to_numeric(df[peak_col], errors="coerce")
        df.loc[df["_peak_ccu"] < 0, "_peak_ccu"] = float("nan")
        activity = (
            df.groupby("_success_label")
            .agg(
                games=("_success", "size"),
                median_peak_ccu=("_peak_ccu", "median"),
                mean_peak_ccu=("_peak_ccu", "mean"),
            )
            .reset_index()
            .rename(columns={"_success_label": "success_group"})
        )
        activity.to_csv(METRICS_DIR / "player_activity_by_success.csv", index=False)
        save_bar_chart(
            activity.dropna(subset=["median_peak_ccu"]),
            "success_group", "median_peak_ccu",
            "Median Peak Concurrent Players by Success Group",
            "Median Peak CCU", FIGURES_DIR / "player_activity_by_success.png"
        )
        lines += ["PLAYER ACTIVITY BY SUCCESS GROUP"]
        lines += [
            activity.to_string(index=False, float_format=lambda x: f"{x:.2f}")
        ]
        lines.append("")

    summary_path = METRICS_DIR / "business_insights_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Corrected business-insights analysis completed.")
    print(f"Rows analyzed: {len(df):,}")
    print(f"Summary saved to: {summary_path}")
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Charts saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
