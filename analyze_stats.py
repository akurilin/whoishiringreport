#!/usr/bin/env python3
"""Analyze eval stats from JSONL file.

Usage:
    python analyze_stats.py [path/to/eval_stats.jsonl]

If no path provided, defaults to out/eval_stats.jsonl
"""

import sys
from pathlib import Path

import pandas as pd

STATS_FILE = Path(__file__).parent / "out" / "eval_stats.jsonl"


def load_stats(path: Path) -> pd.DataFrame:
    """Load stats from JSONL file."""
    if not path.exists():
        print(f"No stats file found at {path}")
        print("Run 'make eval' first to generate stats.")
        sys.exit(1)

    return pd.read_json(path, lines=True)


def analyze(df: pd.DataFrame):
    """Analyze and print statistics."""
    if df.empty:
        print("No stats to analyze.")
        return

    runs = df["timestamp"].nunique()
    models = df["model"].unique()

    print("=" * 70)
    print("EVAL STATS ANALYSIS")
    print("=" * 70)
    print(f"\nTotal records: {len(df)}")
    print(f"Total runs: {runs}")
    print(f"Models tested: {', '.join(sorted(models))}")

    # Per-model summary
    print("\n" + "-" * 70)
    print("MODEL SUMMARY (all runs combined)")
    print("-" * 70)

    summary = df.groupby("model").agg(
        avg_time=("elapsed_seconds", "mean"),
        success_rate=("success", "mean"),
        avg_tokens=("total_tokens", "mean"),
    )
    summary["success_rate"] *= 100

    print(f"{'Model':<30} {'Avg Time':>10} {'Success':>10} {'Avg Tokens':>12}")
    print("-" * 70)
    for model, row in summary.sort_index().iterrows():
        print(
            f"{model:<30} {row['avg_time']:>9.2f}s "
            f"{row['success_rate']:>9.0f}% {row['avg_tokens']:>12,.0f}"
        )

    # Per-case analysis (latest run)
    latest_ts = df["timestamp"].max()
    latest = df[df["timestamp"] == latest_ts]

    print("\n" + "-" * 70)
    print(f"CASE BREAKDOWN (latest run: {str(latest_ts)[:19]})")
    print("-" * 70)

    for model in sorted(latest["model"].unique()):
        model_df = latest[latest["model"] == model].sort_values(
            "elapsed_seconds", ascending=False
        )
        print(f"\n{model}:")
        print(f"  {'Case':<40} {'Time':>8} {'Roles':>6} {'Tokens':>8} {'Status':>8}")
        print(f"  {'-' * 72}")

        for _, r in model_df.iterrows():
            status = "OK" if r["success"] else (r["error_type"] or "FAIL")
            tokens = f"{r['total_tokens']:,.0f}" if pd.notna(r["total_tokens"]) else "-"
            print(
                f"  {r['case_name']:<40} {r['elapsed_seconds']:>7.2f}s "
                f"{r['role_count']:>6.0f} {tokens:>8} {status:>8}"
            )

    # Slowest cases overall
    print("\n" + "-" * 70)
    print("SLOWEST EXTRACTIONS (top 5)")
    print("-" * 70)
    slowest = df.nlargest(5, "elapsed_seconds")
    for _, r in slowest.iterrows():
        print(f"  {r['elapsed_seconds']:>6.2f}s  {r['model']:<25} {r['case_name']}")

    # Error breakdown
    errors = df[~df["success"]]
    if not errors.empty:
        print("\n" + "-" * 70)
        print("ERROR BREAKDOWN")
        print("-" * 70)
        error_counts = errors["error_type"].value_counts()
        for error_type, count in error_counts.items():
            print(f"  {error_type}: {count}")

    # Token efficiency
    print("\n" + "-" * 70)
    print("TOKEN EFFICIENCY (successful extractions only)")
    print("-" * 70)

    successful = df[df["success"] & df["total_tokens"].notna()]
    if not successful.empty:
        efficiency = successful.groupby("model").apply(
            lambda x: x["total_tokens"].sum() / x["role_count"].sum()
            if x["role_count"].sum() > 0
            else 0,
            include_groups=False,
        )
        for model, tokens_per_role in efficiency.sort_index().items():
            print(f"  {model}: {tokens_per_role:,.0f} tokens/role")

    # Run history
    if runs > 1:
        print("\n" + "-" * 70)
        print("RUN HISTORY")
        print("-" * 70)
        print(f"  {'Timestamp':<26} {'Model':<25} {'Pass':>6} {'Avg Time':>10}")
        print(f"  {'-' * 70}")

        history = df.groupby(["timestamp", "model"]).agg(
            passed=("success", "sum"),
            total=("success", "count"),
            avg_time=("elapsed_seconds", "mean"),
        )
        for (ts, model), row in history.iterrows():
            print(
                f"  {str(ts)[:19]:<26} {model:<25} "
                f"{row['passed']:>.0f}/{row['total']:>3.0f} {row['avg_time']:>9.2f}s"
            )

    print("\n" + "=" * 70)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else STATS_FILE
    df = load_stats(path)
    analyze(df)


if __name__ == "__main__":
    main()
