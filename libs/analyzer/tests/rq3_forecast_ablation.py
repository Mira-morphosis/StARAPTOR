"""
RQ3 — forecasting ablation study.

Answers RQ3 ("Is Time-Series Forecasting on KPIs, Preprocessed Reviews and Mined
Associations an improvement on KPI-only systems?") through the ablation protocol of the
methodology ("Forecasting Evaluation"): the pipeline is trained and evaluated three times
with a progressively richer feature set, everything else held constant:

  * C1 (baseline): KPI history and calendar features only — a KPI-only forecaster within
    the same framework;
  * C2: C1 plus the daily review aggregates (per-aspect means, positive-review ratio,
    daily review count);
  * C3 (full system): C2 plus the association-rule features mined by the pipeline.

Each configuration undergoes an independent Optuna tuning pass (on the training split
only, so the held-out 30% is never used for training or tuning), is evaluated on the
chronological 70/30 split with MAE, RMSE and MAPE for both the one-day-ahead prediction
and the recursive 30-day forecast, with weekday/weekend breakdown and underestimation
rate, and the improvement over C1 is expressed as relative error reduction. For C3, the
gain-based feature importance additionally reports the share of total gain absorbed by
the rule features and the highest-gain rules.

Outputs are written under out/<app_id>/rq3/.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: render straight to PDF files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from analyzer.matrix_generator import MatrixGenerator
from analyzer.regressor import Regressor, make_asymmetric_capacity_objective
from analyzer.regressor_tuner import _tune_tier

# Feature groups of MatrixGenerator.build_daily_matrix, selected by column prefix.
KPI_CAL_PREFIXES: Tuple[str, ...] = (
    "kpi_", "pct_change", "mean_concurrent",
    "dayofweek", "is_weekend", "dow_", "lag7_weekend_interaction", "weekend_boost_ratio",
)
REVIEW_PREFIXES: Tuple[str, ...] = ("daily_review_count", "avg_", "positive_ratio")
RULE_PREFIX = "rule_lift_"

CONFIGS: Dict[str, Optional[Tuple[str, ...]]] = {
    "C1": KPI_CAL_PREFIXES,                     # KPI history + calendar only (baseline)
    "C2": KPI_CAL_PREFIXES + REVIEW_PREFIXES,   # + daily review aggregates
    "C3": None,                                 # full matrix, + association-rule features
}

# Metrics for which a lower value is better; improvement over C1 is reported as
# relative error reduction for each of them.
ERROR_KEYS = [
    "mae_1step", "rmse_1step", "mape_1step_pct",
    "mae_multistep", "rmse_multistep", "mape_multistep_pct",
    "mape_multistep_weekday_pct", "mape_multistep_weekend_pct",
    "underestimation_pct", "underestimation_weekend_pct",
]


def select_feature_cols(columns: List[str], prefixes: Optional[Tuple[str, ...]]) -> List[str]:
    """
    Selects the columns belonging to a feature group.
    :param columns: All feature-matrix columns.
    :param prefixes: The group's prefixes; None selects everything.
    :return: The selected column names.
    """
    return [c for c in columns if prefixes is None or c.startswith(prefixes)]


def tune_config(
    app_id: int,
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    df_train: pd.DataFrame,
    n_trials: int,
    n_jobs: int,
) -> Tuple[XGBRegressor, Dict[str, float], float, Dict[str, Any]]:
    """
    Runs an independent Optuna tuning pass for one ablation configuration on the training
    split only. Does not touch the stored tuning history, so production parameters are
    left untouched.
    :param app_id: The Steam unique identifier for the application.
    :param X_train: Training feature matrix for this configuration.
    :param Y_train: Training target.
    :param df_train: Training full frame (carries target_y_abs).
    :param n_trials: Optuna trials for this configuration.
    :param n_jobs: Parallel trials (-1 for all cores).
    :return: A tuple (model, decay_params, peak_weight_alpha, tuning_metrics) ready for benchmarking.
    """
    data_tuple = (
        X_train.reset_index(drop=True),
        Y_train.reset_index(drop=True),
        df_train.reset_index(drop=True),
    )
    best_params, tune_metrics, _ = _tune_tier(
        app_id, len(X_train), data_tuple, n_trials=n_trials, n_jobs=n_jobs,
    )

    # _tune_tier returns asymmetric_alpha inside xgb_params and no compiled objective:
    # pop the former and attach the latter, exactly like resolve_params() does.
    raw = best_params["xgb_params"]
    alpha = raw.get("asymmetric_alpha", 5.0)
    xgb_params = {k: v for k, v in raw.items() if k != "asymmetric_alpha"}
    xgb_params["objective"] = make_asymmetric_capacity_objective(alpha=alpha)
    model = XGBRegressor(**xgb_params)

    return model, dict(best_params["decay_params"]), float(best_params.get("peak_weight_alpha", 1.0)), tune_metrics


def gain_importance(
    app_id: int,
    model: XGBRegressor,
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    df_train: pd.DataFrame,
    peak_weight_alpha: float,
) -> pd.DataFrame:
    """
    Fits the configuration's model on the training split and returns its gain-based
    feature importance (total loss improvement attributed to each feature's splits).
    :return: A DataFrame with feature, gain, and gain_share columns, sorted by gain.
    """
    reg = Regressor(app_id=app_id, model=copy.deepcopy(model), capacity_mode=False)
    reg.fit(data_tuple=(X_train, Y_train, df_train), peak_weight_alpha=peak_weight_alpha)
    gain = reg.model.get_booster().get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    # dtype="str" keeps the feature column usable with .str even when gain is empty
    # (an empty DataFrame column would otherwise default to float64).
    df = pd.DataFrame({
        "feature": pd.Series([str(k) for k in gain.keys()], dtype="str"),
        "gain": pd.Series([float(v) for v in gain.values()], dtype="float64"),
    })
    df["gain_share"] = df["gain"] / total
    return df.sort_values("gain", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures (vector PDF, ready for inclusion in the thesis)
# ---------------------------------------------------------------------------

def plot_rq3_figures(
    results: Dict[str, Any],
    per_day_by_config: Dict[str, pd.DataFrame],
    fig_dir: str | Path,
) -> List[str]:
    """
    Renders the RQ3 result figures as PDFs: per-configuration error metrics, relative
    error reduction vs the C1 baseline, the C3 held-out forecast against the actual KPI,
    and the highest-gain association-rule features.
    :param results: The per-config results dict built by run_rq3_ablation().
    :param per_day_by_config: config -> per-day benchmark DataFrame.
    :param fig_dir: Directory the PDF figures are written to.
    :return: The list of written figure paths.
    """
    fig_path = Path(fig_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    written = []
    names = list(CONFIGS.keys())
    colors = {"C1": "gray", "C2": "steelblue", "C3": "seagreen"}

    # 1. Error metrics per configuration.
    metric_keys = ["mape_1step_pct", "mape_multistep_pct", "mape_multistep_weekday_pct",
                   "mape_multistep_weekend_pct", "underestimation_pct"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(metric_keys))
    width = 0.25
    for i, name in enumerate(names):
        vals = [results[name]["metrics"].get(k, 0.0) for k in metric_keys]
        ax.bar(x + (i - 1) * width, vals, width, label=name, color=colors[name])
    ax.set_xticks(x, [k.replace("_pct", " (%)").replace("mape_", "MAPE ").replace("_", " ") for k in metric_keys], rotation=15, ha="right")
    ax.set_ylabel("value (%)")
    ax.set_title("RQ3 — forecast quality per ablation configuration (held-out 30%)")
    ax.legend()
    fig.tight_layout()
    out = fig_path / "rq3_metrics_by_config.pdf"
    fig.savefig(out)
    plt.close(fig)
    written.append(str(out))

    # 2. Relative error reduction vs C1.
    red_keys = ["mape_1step_pct", "mape_multistep_pct", "underestimation_pct"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(red_keys))
    width = 0.35
    for i, name in enumerate(("C2", "C3")):
        vals = [results[name]["relative_error_reduction_vs_C1_pct"].get(k, 0.0) for k in red_keys]
        ax.bar(x + (i - 0.5) * width, vals, width, label=name, color=colors[name])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, [k.replace("_pct", " (%)").replace("mape_", "MAPE ").replace("_", " ") for k in red_keys])
    ax.set_ylabel("relative error reduction vs C1 (%)")
    ax.set_title("RQ3 — improvement over the KPI-only baseline (positive = better)")
    ax.legend()
    fig.tight_layout()
    out = fig_path / "rq3_reduction_vs_C1.pdf"
    fig.savefig(out)
    plt.close(fig)
    written.append(str(out))

    # 3. C3 held-out forecast vs actual.
    df = per_day_by_config.get("C3")
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df["date"], df["actual_kpi"], label="actual", color="black", linewidth=1.5)
        ax.plot(df["date"], df["pred_multistep"], label="30-day recursive forecast", color="seagreen", linewidth=1.2)
        ax.plot(df["date"], df["pred_1step"], label="1-step-ahead", color="steelblue", linewidth=1.0, linestyle=":")
        ax.set_ylabel("mean concurrent players")
        ax.set_title("RQ3 — C3 (full system) on the held-out 30%")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        out = fig_path / "rq3_forecast_vs_actual_C3.pdf"
        fig.savefig(out)
        plt.close(fig)
        written.append(str(out))

    # 4. Highest-gain rule features of C3.
    top_rules = [r for r in results["C3"]["top_rules_by_gain"] if r["gain"] > 0][:10]
    if top_rules:
        labels = [r["feature"].replace("rule_lift_", "").replace("_->", " → ") for r in top_rules][::-1]
        vals = [r["gain_share"] for r in top_rules][::-1]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(np.arange(len(vals)), vals, color="darkred", height=0.65)
        ax.set_yticks(np.arange(len(vals)), labels)
        ax.set_xlabel("share of total gain")
        ax.set_title("RQ3 — top association-rule features by gain (C3)")
        fig.tight_layout()
        out = fig_path / "rq3_top_rules_by_gain.pdf"
        fig.savefig(out)
        plt.close(fig)
        written.append(str(out))

    return written


def run_rq3_ablation(
    app_id: int,
    split_ratio: float = 0.70,
    n_trials: int = 150,
    eval_days: int = 30,
    n_jobs: int = -1,
    out_dir: str = "./out",
    fig_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the C1/C2/C3 ablation for one game.
    :param app_id: The Steam unique identifier for the application.
    :param split_ratio: Chronological fraction used for training/tuning; the rest is held out.
    :param n_trials: Optuna trials per configuration.
    :param eval_days: Horizon (days) of the recursive multi-step evaluation.
    :param n_jobs: Parallel Optuna trials (-1 for all cores).
    :param out_dir: Root output directory; artifacts land in <out_dir>/<app_id>/rq3/.
    :param fig_dir: Directory for the PDF figures; defaults to <out_dir>/<app_id>/rq3/figures/.
    :return: The report dict (also written to report.json).
    """
    out_path = Path(out_dir) / str(app_id) / "rq3"
    out_path.mkdir(parents=True, exist_ok=True)

    generator = MatrixGenerator(app_id)
    X_full, Y, df_full = generator.build_daily_matrix()
    if X_full.empty or len(X_full) < 10:
        raise RuntimeError(f"Not enough daily rows for App ID {app_id} (got {len(X_full)}).")

    train_size = int(len(X_full) * split_ratio)
    X_train_full, Y_train = X_full.iloc[:train_size], Y.iloc[:train_size]
    df_train = df_full.iloc[:train_size]
    print(f"[App {app_id}] Ablation on {len(X_full)} days "
          f"(train {train_size} / held-out {len(X_full) - train_size}), "
          f"{n_trials} tuning trials per configuration.")

    results: Dict[str, Any] = {}
    per_day_by_config: Dict[str, pd.DataFrame] = {}
    for name, prefixes in CONFIGS.items():
        cols = select_feature_cols(list(X_full.columns), prefixes)
        X_cfg = X_full[cols]

        print(f"\n[App {app_id}] === {name}: {len(cols)} features ===")
        model, decay_params, peak_alpha, tune_metrics = tune_config(
            app_id, X_train_full[cols], Y_train, df_train, n_trials, n_jobs,
        )
        print(f"[App {app_id}] {name} tuned (CV MAPE {tune_metrics['mape_multistep_pct']:.2f}%), benchmarking...")

        # capacity_mode=False: the model already carries the tuned asymmetric objective,
        # and fit() must not overwrite it with the default alpha.
        regressor = Regressor(app_id=app_id, model=model, capacity_mode=False)
        metrics, per_day = regressor.benchmark(
            split_ratio=split_ratio,
            peak_weight_alpha=peak_alpha,
            eval_days=eval_days,
            data_tuple=(X_cfg, Y, df_full),
            **decay_params,
        )
        if not metrics:
            raise RuntimeError(f"Benchmark returned no metrics for {name} (App {app_id}).")

        importance = gain_importance(app_id, model, X_train_full[cols], Y_train, df_train, peak_alpha)
        importance.to_csv(out_path / f"feature_importance_{name}.csv", index=False)
        per_day.to_csv(out_path / f"per_day_{name}.csv", index=False)
        per_day_by_config[name] = per_day

        rule_gain = float(importance.loc[importance["feature"].str.startswith(RULE_PREFIX), "gain"].sum())
        total_gain = float(importance["gain"].sum()) or 1.0
        results[name] = {
            "n_features": len(cols),
            "tuning": {k: tune_metrics[k] for k in ("score", "mape_multistep_pct", "mape_1step_pct") if k in tune_metrics},
            "metrics": metrics,
            "rule_feature_gain_share": rule_gain / total_gain,
            "top_rules_by_gain": importance[importance["feature"].str.startswith(RULE_PREFIX)]
                                      .head(20)[["feature", "gain", "gain_share"]]
                                      .to_dict(orient="records"),
        }
        print(f"[App {app_id}] {name}: mape_1step={metrics['mape_1step_pct']:.2f}% "
              f"mape_multistep={metrics['mape_multistep_pct']:.2f}% "
              f"(weekday {metrics['mape_multistep_weekday_pct']:.2f}% / "
              f"weekend {metrics['mape_multistep_weekend_pct']:.2f}%) "
              f"underestimation={metrics['underestimation_pct']:.2f}% "
              f"rule_gain_share={results[name]['rule_feature_gain_share']:.3f}")

    # Relative error reduction with respect to the C1 baseline.
    baseline = results["C1"]["metrics"]
    for name in ("C2", "C3"):
        reductions = {}
        for key in ERROR_KEYS:
            b = baseline.get(key, 0.0)
            v = results[name]["metrics"].get(key, 0.0)
            reductions[key] = ((b - v) / b * 100.0) if b != 0 else 0.0
        results[name]["relative_error_reduction_vs_C1_pct"] = reductions
        print(f"\n[App {app_id}] {name} vs C1 relative error reduction: "
              + ", ".join(f"{k}={reductions[k]:+.2f}%" for k in
                          ("mape_1step_pct", "mape_multistep_pct", "underestimation_pct")))

    report = {
        "rq": "RQ3",
        "app_id": app_id,
        "split_ratio": split_ratio,
        "eval_days": eval_days,
        "n_trials_per_config": n_trials,
        "configs": results,
        "monotonic_improvement_C1_C2_C3": all(
            results["C2"]["metrics"][k] <= results["C1"]["metrics"][k] and
            results["C3"]["metrics"][k] <= results["C2"]["metrics"][k]
            for k in ("mae_multistep", "rmse_multistep", "mape_multistep_pct")
        ),
        "outputs": {name: str(out_path / f"per_day_{name}.csv") for name in CONFIGS},
    }

    figures = plot_rq3_figures(results, per_day_by_config, fig_dir or out_path / "figures")
    report["figures"] = figures
    for f in figures:
        print(f"[App {app_id}] figure written: {f}")

    with open(out_path / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n[App {app_id}] RQ3 report written to {out_path / 'report.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description="RQ3 — forecasting ablation (C1/C2/C3)")
    parser.add_argument("--app-id", type=int, default=int(os.getenv("STARAPTOR_APP_ID", 1086940)))
    parser.add_argument("--split", type=float, default=0.70, help="Chronological train fraction.")
    parser.add_argument("--n-trials", type=int, default=150, help="Optuna trials per configuration.")
    parser.add_argument("--eval-days", type=int, default=30, help="Recursive forecast evaluation horizon.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out", type=str, default="./out")
    parser.add_argument("--fig-dir", type=str, default=None, help="Directory for the PDF figures.")
    args = parser.parse_args()
    run_rq3_ablation(
        app_id=args.app_id,
        split_ratio=args.split,
        n_trials=args.n_trials,
        eval_days=args.eval_days,
        n_jobs=args.n_jobs,
        out_dir=args.out,
        fig_dir=args.fig_dir,
    )


if __name__ == "__main__":
    main()
