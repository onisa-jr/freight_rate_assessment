from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from config import DATA_PATH, MLP_MAX_EPOCHS, RESULTS_DIR, SEEDS, TARGET_MODE, USE_QUOTE_SIGNAL
from data import load_data, run_data_checks
from evaluate import evaluate_all, generate_extended_summary, generate_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freight Rate Estimation: Multi-Seed Evaluation Pipeline (Ridge, LightGBM, MLP)"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to training data CSV (default: assesiment/data/train-test.csv)",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Sample N rows for quick smoke runs (default: None, use full dataset)",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["raw", "rate_per_mile", "log"],
        default=TARGET_MODE,
        help="Target formulation to model: raw, rate_per_mile, or log (default: raw)",
    )
    parser.add_argument(
        "--no-quote-signal",
        action="store_true",
        help="Disable quote_signal to test performance without it at prediction time",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="List of random seeds to evaluate (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--mlp-max-epochs",
        type=int,
        default=MLP_MAX_EPOCHS,
        help="Maximum epochs for MLP training (default: 200, use 3-5 for smoke checks)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to save metric CSVs and visualization plots (default: results)",
    )
    parser.add_argument(
        "--skip-data-checks",
        action="store_true",
        help="Skip Step 1 data checks printout (default: False)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_quote_signal = not args.no_quote_signal

    print("=" * 80)
    print("FREIGHT RATE ESTIMATION - MULTI-SEED EVALUATION (Ridge, LightGBM, MLP)")
    print(f"Data path: {args.data_path}")
    print(f"Target mode: {args.target_mode} | USE_QUOTE_SIGNAL: {use_quote_signal}")
    print(f"Seeds: {args.seeds} | Sample rows: {args.sample_rows}")
    print(f"MLP Max Epochs: {args.mlp_max_epochs}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # 1. Load data
    df = load_data(str(args.data_path), parse_dates=True, sample_rows=args.sample_rows, random_state=42)

    # 2. Run Step 1 data checks
    if not args.skip_data_checks:
        run_data_checks(df, use_quote_signal=use_quote_signal)

    # 3. Step 4: Run evaluation across protocols and seeds
    print("\nRunning multi-seed evaluation across Protocol A and Protocol B (Ridge, LightGBM, MLP)...")
    seed_metrics_df, summary_df, avg_importance, scatter_data, mlp_history = evaluate_all(
        df=df,
        seeds=args.seeds,
        target_mode=args.target_mode,
        use_quote_signal=use_quote_signal,
        mlp_max_epochs=args.mlp_max_epochs,
    )

    # 4. Step 5: Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = args.output_dir / "seed_metrics.csv"
    seed_metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\n[Saved] Per-seed metrics saved to {metrics_csv_path}")

    summary_csv_path = args.output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_csv_path)

    # 5. Print Summary Table
    print("\n" + "=" * 80)
    print("SUMMARY METRICS TABLE (across seeds)")
    print("=" * 80)
    pd.set_option("display.max_columns", 25)
    pd.set_option("display.width", 1000)
    print(summary_df.round(3))
    print("=" * 80)

    # 6. Generate and save required plots
    print("\nGenerating visualization plots...")
    generate_plots(
        seed_metrics_df=seed_metrics_df,
        avg_importance=avg_importance,
        scatter_data=scatter_data,
        mlp_history=mlp_history,
        output_dir=args.output_dir,
    )
    print(f"[Saved] Plots saved to {args.output_dir}:")
    print("  - mae_per_seed.png")
    print("  - feature_importance.png")
    print("  - predicted_vs_actual.png")
    print("  - training_curve_mlp.png")

    # 7. Print closing summary
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY (Extended with MLP)")
    print("=" * 80)
    summary_text = generate_extended_summary(summary_df)
    print(summary_text)
    print("=" * 80)


if __name__ == "__main__":
    main()
