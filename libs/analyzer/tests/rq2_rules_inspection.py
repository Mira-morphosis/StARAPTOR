"""
RQ2 — empirical verification of the mined association rules.

Answers RQ2 ("Is Association Mining a solid approach to find review patterns?") by
operationalizing the inspection protocol of the methodology ("Empirical Verification of
Mined Rules"):

  * sampling: for each historical mining window, the top-K rules by lift plus K random
    rules from the remainder (fixed seed) are selected and presented in random order;
  * classification rubric: each sampled rule is assigned to exactly one of four mutually
    exclusive categories — actionable, semantically trivial, taxonomy artifact, unclear —
    recorded in a structured form (rule identifier, window, category, supporting evidence);
  * trace-back verification: every rule classified as actionable is traced back to the
    stored reviews containing its antecedent items, to confirm the association reflects
    actual review content rather than a discretization artifact;
  * cross-window persistence: rules are compared across independent sliding windows; an
    association re-occurring across windows is used as a proxy for robustness.

Quality metrics computed from the annotated form: residual noise rate (fraction of
inspected rules that should have been filtered out), actionability rate (fraction deemed
actionable) and persistence rate (fraction of rules re-occurring across windows).

Usage:
  # 1. (optional) re-mine the full history, then sample the rules to inspect
  python libs/analyzer/tests/rq2_rules_inspection.py --app-id 1086940 --remine
  # 2. fill the category/evidence columns of out/<app_id>/rq2/sample.csv by hand
  # 3. compute the metrics from the annotated form
  python libs/analyzer/tests/rq2_rules_inspection.py --app-id 1086940 \
      --annotations out/1086940/rq2/sample.csv

Outputs are written under out/<app_id>/rq2/.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: render straight to PDF files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyzer.analyzer import Analyzer
from analyzer.data_standardizer import DataStandardizer
from storage.storage_utils import clear_mining_rules, db_connection

CATEGORIES = ["actionable", "semantically_trivial", "taxonomy_artifact", "unclear"]

# Item prefixes whose associations are mediated by the item taxonomy rather than the domain.
STRUCTURAL_PREFIXES = ("lang_", "hw_", "playtime_")

DEFAULT_MINING_PARAMS = dict(
    window_days=14,
    step_days=7,
    min_support=0.02,
    min_confidence=0.5,
    min_lift=1.2,
    deduplicate_mode="both",
)


# ---------------------------------------------------------------------------
# Loading and sampling
# ---------------------------------------------------------------------------

def load_rules(app_id: int) -> pd.DataFrame:
    """
    Loads every mined rule joined with its mining-run (window) metadata.
    :param app_id: The Steam unique identifier for the application.
    :return: A DataFrame with rule and window columns; empty if no runs exist.
    """
    with db_connection(app_id) as conn:
        try:
            return conn.execute("""
                SELECT r.rule_id, r.run_id, r.antecedents, r.consequents,
                       r.support, r.confidence, r.lift,
                       m.timestamp AS window_timestamp, m.window_size_days, m.total_reviews
                FROM association_rules r
                JOIN mining_runs m ON r.run_id = m.run_id
                ORDER BY m.timestamp DESC, r.lift DESC
            """).df()
        except Exception:
            return pd.DataFrame()


def rule_signature(antecedents: str, consequents: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Normalizes a rule into a hashable signature of sorted item sets, used for
    permutation-invariant comparison across windows.
    :param antecedents: The antecedents string ("A AND B").
    :param consequents: The consequents string ("C AND D").
    :return: A tuple (sorted antecedent items, sorted consequent items).
    """
    ant = tuple(sorted(str(antecedents).split(" AND ")))
    cons = tuple(sorted(str(consequents).split(" AND ")))
    return ant, cons


def sample_rules(
    rules: pd.DataFrame,
    top_k: int,
    random_k: int,
    seed: int,
) -> pd.DataFrame:
    """
    Samples, per mining window, the top-K rules by lift plus K random rules from the
    remainder, and shuffles the presentation order so prominence cannot be inferred.
    :param rules: The full rules DataFrame.
    :param top_k: Number of top-lift rules sampled per window.
    :param random_k: Number of uniformly random rules sampled per window.
    :param seed: The fixed sampling seed.
    :return: The sampled rules with sample_group and presentation order columns.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for run_id, group in rules.groupby("run_id"):
        group = group.sort_values("lift", ascending=False)
        top = group.head(top_k)
        rest = group.iloc[len(top):]
        rand = rest.sample(n=min(random_k, len(rest)), random_state=seed) if len(rest) else rest
        part = pd.concat([top.assign(sample_group="top_lift"), rand.assign(sample_group="random")])
        parts.append(part)
    sampled = pd.concat(parts, ignore_index=True) if parts else rules.iloc[0:0].copy()
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # random presentation order
    sampled.insert(0, "presentation_order", range(1, len(sampled) + 1))
    return sampled


def propose_category(antecedents: str, consequents: str) -> str:
    """
    Applies the mechanical part of the fixed decision procedure, as a proposal to be
    confirmed or overridden during manual inspection: rules whose consequent is entailed
    by the antecedent are proposed trivial; rules whose link is mediated only by structural
    items (language, hardware, playtime buckets) are proposed taxonomy artifacts; the rest
    are actionable candidates.
    :param antecedents: The antecedents string.
    :param consequents: The consequents string.
    :return: One of "semantically_trivial", "taxonomy_artifact", "actionable_candidate".
    """
    ant, cons = rule_signature(antecedents, consequents)
    all_items = set(ant) | set(cons)
    if set(cons).issubset(set(ant)):
        return "semantically_trivial"
    if all_items and all(any(item.startswith(p) for p in STRUCTURAL_PREFIXES) for item in all_items):
        return "taxonomy_artifact"
    return "actionable_candidate"


# ---------------------------------------------------------------------------
# Trace-back verification
# ---------------------------------------------------------------------------

def traceback_reviews(
    app_id: int,
    rule: pd.Series,
    max_reviews: int,
    seed: int,
) -> pd.DataFrame:
    """
    Retrieves a sample of the stored non-spam reviews whose transaction contains every
    antecedent item of the rule, so the association can be verified against actual content.
    :param app_id: The Steam unique identifier for the application.
    :param rule: A sampled rule row.
    :param max_reviews: Maximum number of supporting reviews to return.
    :param seed: Sampling seed.
    :return: A DataFrame with the supporting reviews (recommendationid, review text, date).
    """
    ant_items: Set[str] = set(str(rule["antecedents"]).split(" AND "))
    standardizer = DataStandardizer(app_id)
    reviews = standardizer.get_window_review_list(0)  # all non-spam reviews
    supporting = [
        r for r in reviews
        if ant_items.issubset(set(standardizer.transactional(r)))
    ]
    rng = np.random.default_rng(seed)
    if len(supporting) > max_reviews:
        idx = rng.choice(len(supporting), size=max_reviews, replace=False)
        supporting = [supporting[i] for i in idx]
    out = pd.DataFrame(supporting)
    keep = [c for c in ["recommendationid", "review", "language", "timestamp_updated", "voted_up"] if c in out.columns]
    return out[keep] if not out.empty else out


# ---------------------------------------------------------------------------
# Persistence and metrics
# ---------------------------------------------------------------------------

def persistence_rates(rules: pd.DataFrame, sampled: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes the fraction of rules re-occurring across independent windows, both over all
    mined rules and over the inspected sample. A rule "persists" if its normalized
    signature appears in at least two distinct mining runs.
    :param rules: The full rules DataFrame.
    :param sampled: The inspected sample.
    :return: A dict with overall and sampled persistence rates.
    """
    def _rate(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"n_rules": 0, "n_persistent": 0, "persistence_rate": float("nan")}
        sig = df.apply(lambda r: rule_signature(r["antecedents"], r["consequents"]), axis=1)
        runs_per_sig = df.assign(sig=sig).groupby("sig")["run_id"].nunique()
        n_persistent = int((runs_per_sig >= 2).sum())
        return {
            "n_rules": int(len(df)),
            "n_persistent": n_persistent,
            "persistence_rate": n_persistent / len(df) if len(df) else float("nan"),
        }

    return {"overall": _rate(rules), "sampled": _rate(sampled)}


def compute_metrics(annotations: pd.DataFrame, rules: pd.DataFrame, sampled: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes the RQ2 quality metrics from the annotated inspection form.
    :param annotations: The filled sample (must contain a category column with values in CATEGORIES).
    :param rules: The full rules DataFrame.
    :param sampled: The inspected sample.
    :return: A report dict with category counts and the three rates, overall and per window.
    """
    ann = annotations.dropna(subset=["category"])
    ann = ann[ann["category"].isin(CATEGORIES)]
    n = len(ann)

    counts = ann["category"].value_counts().to_dict()
    noise = counts.get("semantically_trivial", 0) + counts.get("taxonomy_artifact", 0)

    per_window = (
        ann.groupby("run_id")["category"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORIES, fill_value=0)
        .to_dict(orient="index")
        if n else {}
    )

    report = {
        "rq": "RQ2",
        "n_inspected": n,
        "category_counts": {c: int(counts.get(c, 0)) for c in CATEGORIES},
        "residual_noise_rate": (noise / n) if n else float("nan"),
        "actionability_rate": (counts.get("actionable", 0) / n) if n else float("nan"),
        "unclear_rate": (counts.get("unclear", 0) / n) if n else float("nan"),
        "persistence": persistence_rates(rules, sampled),
        "per_window_category_counts": per_window,
    }
    return report


# ---------------------------------------------------------------------------
# Figures (vector PDF, ready for inclusion in the thesis)
# ---------------------------------------------------------------------------

def plot_rq2_figures(report: Dict[str, Any], fig_dir: str | Path) -> List[str]:
    """
    Renders the RQ2 result figures as PDFs: the classification-rubric outcome
    (category counts with the residual noise / actionability rates) and the
    cross-window persistence rates.
    :param report: The metrics report dict produced by compute_metrics().
    :param fig_dir: Directory the PDF figures are written to.
    :return: The list of written figure paths.
    """
    fig_path = Path(fig_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    counts = report["category_counts"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    colors = {"actionable": "seagreen", "semantically_trivial": "gray",
              "taxonomy_artifact": "darkorange", "unclear": "lightgray"}
    labels = list(counts.keys())
    axes[0].bar(labels, [counts[c] for c in labels],
                color=[colors[c] for c in labels])
    for i, c in enumerate(labels):
        axes[0].text(i, counts[c], str(counts[c]), ha="center", va="bottom")
    axes[0].set_ylabel("inspected rules")
    axes[0].set_title(f"RQ2 — rubric outcome (n={report['n_inspected']})\n"
                      f"residual noise rate = {report['residual_noise_rate']:.3f}, "
                      f"actionability rate = {report['actionability_rate']:.3f}")
    axes[0].tick_params(axis="x", rotation=20)

    pers = report["persistence"]
    axes[1].bar(["all mined rules", "inspected sample"],
                [pers["overall"]["persistence_rate"], pers["sampled"]["persistence_rate"]],
                color="steelblue")
    axes[1].set_ylabel("persistence rate")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("RQ2 — rules re-occurring across independent windows")

    fig.tight_layout()
    out = fig_path / "rq2_inspection_summary.pdf"
    fig.savefig(out)
    plt.close(fig)
    return [str(out)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_rq2_inspection(
    app_id: int,
    seed: int = 42,
    top_k: int = 10,
    random_k: int = 5,
    remine: bool = False,
    annotations: Optional[str] = None,
    traceback_max_reviews: int = 10,
    out_dir: str = "./out",
    fig_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the RQ2 inspection pipeline for one game.
    :param app_id: The Steam unique identifier for the application.
    :param seed: The fixed sampling seed.
    :param top_k: Top-lift rules sampled per window.
    :param random_k: Random rules sampled per window.
    :param remine: If True, clears the stored rules and re-mines the full history first.
    :param annotations: Optional path to the manually annotated sample CSV; when given,
                        metrics, trace-back outputs and figures are produced.
    :param traceback_max_reviews: Max supporting reviews exported per actionable rule.
    :param out_dir: Root output directory; artifacts land in <out_dir>/<app_id>/rq2/.
    :param fig_dir: Directory for the PDF figures; defaults to <out_dir>/<app_id>/rq2/figures/.
    :return: The report dict (written to report.json when annotations are provided).
    """
    out_path = Path(out_dir) / str(app_id) / "rq2"
    out_path.mkdir(parents=True, exist_ok=True)

    if remine:
        print(f"[App {app_id}] Re-mining full history (window={DEFAULT_MINING_PARAMS['window_days']}d, "
              f"step={DEFAULT_MINING_PARAMS['step_days']}d)...")
        clear_mining_rules(app_id)
        analyzer = Analyzer(app_id)
        run_ids = analyzer.run_historical(**DEFAULT_MINING_PARAMS)
        print(f"[App {app_id}] Re-mining complete: {len(run_ids)} runs saved.")

    rules = load_rules(app_id)
    if rules.empty:
        raise RuntimeError(f"No mined rules found for App ID {app_id}. Run with --remine first.")

    sampled = sample_rules(rules, top_k, random_k, seed)
    sampled["proposed_category"] = [
        propose_category(a, c) for a, c in zip(sampled["antecedents"], sampled["consequents"])
    ]
    # Structured form for manual inspection: fill category + evidence by hand.
    sampled["category"] = ""
    sampled["evidence"] = ""
    sample_file = out_path / "sample.csv"
    sampled.to_csv(sample_file, index=False)
    print(f"[App {app_id}] Inspection sample ({len(sampled)} rules over {rules['run_id'].nunique()} windows, "
          f"seed {seed}) written to {sample_file}")
    print("Fill the 'category' and 'evidence' columns, then re-run with --annotations to compute the metrics.")

    if annotations is None:
        return {"rq": "RQ2", "app_id": app_id, "sample_file": str(sample_file), "n_sampled": int(len(sampled))}

    ann = pd.read_csv(annotations)
    report = compute_metrics(ann, rules, sampled)
    report["app_id"] = app_id
    report["seed"] = seed

    # Trace-back verification for every rule classified as actionable.
    actionable = ann[ann["category"] == "actionable"]
    tb_frames = []
    for _, rule in actionable.iterrows():
        tb = traceback_reviews(app_id, rule, traceback_max_reviews, seed)
        tb.insert(0, "rule_id", rule["rule_id"])
        tb_frames.append(tb)
        print(f"[App {app_id}] trace-back for rule {rule['rule_id']}: {len(tb)} supporting reviews.")
    if tb_frames:
        tb_all = pd.concat(tb_frames, ignore_index=True)
        tb_all.to_csv(out_path / "traceback_actionable_rules.csv", index=False)
        report["traceback_file"] = str(out_path / "traceback_actionable_rules.csv")

    figures = plot_rq2_figures(report, fig_dir or out_path / "figures")
    report["figures"] = figures
    for f in figures:
        print(f"[App {app_id}] figure written: {f}")

    with open(out_path / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n[App {app_id}] RQ2 metrics: residual_noise_rate={report['residual_noise_rate']:.3f} "
          f"actionability_rate={report['actionability_rate']:.3f} "
          f"persistence_rate(sampled)={report['persistence']['sampled']['persistence_rate']:.3f}")
    print(f"[App {app_id}] RQ2 report written to {out_path / 'report.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description="RQ2 — empirical verification of mined rules")
    parser.add_argument("--app-id", type=int, default=int(os.getenv("STARAPTOR_APP_ID", 1086940)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--random-k", type=int, default=5)
    parser.add_argument("--remine", action="store_true", help="Clear stored rules and re-mine the full history first.")
    parser.add_argument("--annotations", type=str, default=None, help="Path to the manually annotated sample CSV.")
    parser.add_argument("--traceback-max-reviews", type=int, default=10)
    parser.add_argument("--out", type=str, default="./out")
    parser.add_argument("--fig-dir", type=str, default=None, help="Directory for the PDF figures.")
    args = parser.parse_args()
    run_rq2_inspection(
        app_id=args.app_id,
        seed=args.seed,
        top_k=args.top_k,
        random_k=args.random_k,
        remine=args.remine,
        annotations=args.annotations,
        traceback_max_reviews=args.traceback_max_reviews,
        out_dir=args.out,
        fig_dir=args.fig_dir,
    )


if __name__ == "__main__":
    main()
