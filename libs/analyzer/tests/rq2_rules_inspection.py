"""
RQ2 — empirical verification of the mined association rules.

Answers RQ2 ("Is Association Mining a solid approach to find review patterns?") by
operationalizing the inspection protocol of the methodology ("Empirical Verification of
Mined Rules"):

  * sampling: for each historical mining window, the top-K rules by lift plus K random
    rules from the remainder (fixed seed) are selected and presented in random order;
  * classification rubric (CHOSEN APPROACH — deterministic and fully reproducible): each
    sampled rule is assigned to exactly one of four mutually exclusive categories —
    actionable, semantically trivial, taxonomy artifact, unclear — by a fixed decision
    procedure (classify_rule): entailed-label links are derived from the data itself
    (measured conditionals >= 0.90 on the non-spam pool), structural items are
    lang_/hw_/playtime_/kpi_/steam_purchase/received_for_free, near-independence is
    lift < 1.5. The rule-based labels are written to sample_classified.csv; a manual
    annotation (sample_annotated.csv / --annotations) is OPTIONAL and used only as a
    validation layer — the report then includes the rule-vs-human agreement;
  * trace-back verification: every rule classified as actionable is traced back to the
    stored reviews containing its antecedent items, to confirm the association reflects
    actual review content rather than a discretization artifact;
  * cross-window persistence: rules are compared across independent sliding windows; an
    association re-occurring across windows is used as a proxy for robustness.

Quality metrics computed from the annotated form: residual noise rate (fraction of
inspected rules that should have been filtered out), actionability rate (fraction deemed
actionable) and persistence rate (fraction of rules re-occurring across windows). Rule
redundancy (near-duplicate rules surviving the miner's deduplication) is NOT a manual
category — it is measured automatically and objectively as redundancy_stats() over the
sampled rules and reported alongside the rates.

Usage:
  # sample the rules, classify them with the deterministic rule-based protocol,
  # compute metrics + trace-back + tables in one pass
  python libs/analyzer/tests/rq2_rules_inspection.py --app-id 1086940
  # optional validation against a manual annotation (agreement is reported):
  python libs/analyzer/tests/rq2_rules_inspection.py --app-id 1086940 \
      --annotations out/1086940/rq2/sample_annotated.csv

The study works on the shared study database in the appid folder,
out/<app_id>/<app_id>.db (override with --db-path) — the one RQ1 stored its most
useful run into — not on the database in the directory defined in .env.

Outputs are written under out/<app_id>/rq2/. RQ2 emits NO PDF figure: its results are
counts and rates, exported as CSV tables (rq2_metrics_table.csv, rq2_per_window_categories.csv)
for the LaTeX tables of the results chapter.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from analyzer.analyzer import Analyzer
from analyzer.data_standardizer import DataStandardizer
from storage.storage_utils import clear_mining_rules, db_connection, get_latest_run_id

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
# Deterministic rule-based classifier (the chosen RQ2 inspection protocol)
# ---------------------------------------------------------------------------
# Fixed, data-derived decision procedure. All thresholds are measured on the non-spam
# review pool of the study database, so the classification is fully reproducible.

LABEL_FIELDS = ["voted_up", "actuallyRecommendsTitle", "containsPositiveAspects",
                "containsNegativeAspects", "containsBugDescription"]
PILLAR_PREFIXES = ["multiplayer", "immersion", "community", "replayability", "story",
                   "monetizationModel", "gameplay", "controls", "graphics", "customization",
                   "audio", "difficulty"]
STRUCT_ALL = ("lang_", "hw_", "playtime_", "kpi_", "steam_purchase", "received_for_free")
STRUCT_DEMO = ("lang_", "hw_", "playtime_")
ENTAILMENT_THRESHOLD = 0.90   # measured P(consequent_item | antecedent_item) counted as entailed
BASE_RATE_THRESHOLD = 0.90    # consequent item base rate counted as near-universal
LOW_LIFT_THRESHOLD = 1.5      # lift below this = near-independence


def label_stats(app_id: int) -> Tuple[Dict[str, float], Dict[str, Set[str]]]:
    """
    Derives, from the stored non-spam reviews, the base rate of every boolean label item
    and the entailment map (antecedent label item -> set of consequent label items with
    measured conditional probability >= ENTAILMENT_THRESHOLD). Both drive classify_rule.
    :param app_id: The Steam unique identifier for the application.
    :return: (base_rate, entailed) dicts keyed by item strings such as "voted_up_True".
    """
    cols = LABEL_FIELDS + ["steam_purchase", "received_for_free", "isUseful"]
    sel = ", ".join(f'"{c}"' for c in cols)
    with db_connection(app_id) as conn:
        rows = conn.execute(f'SELECT {sel} FROM reviews WHERE "isSpam" = FALSE').fetchall()
    df = pd.DataFrame(rows, columns=cols)
    base_rate: Dict[str, float] = {}
    for c in cols:
        base_rate[f"{c}_True"] = float((df[c] == True).mean())
        base_rate[f"{c}_False"] = float((df[c] == False).mean())
    entailed: Dict[str, Set[str]] = {}
    for a in LABEL_FIELDS:
        for va in (True, False):
            sub = df[df[a] == va]
            if sub.empty:
                continue
            for c in LABEL_FIELDS:
                for vc in (True, False):
                    if float((sub[c] == vc).mean()) >= ENTAILMENT_THRESHOLD:
                        entailed.setdefault(f"{a}_{va}", set()).add(f"{c}_{vc}")
    return base_rate, entailed


def _rule_items(s: Any) -> Set[str]:
    return set(str(s).split(" AND "))


def _is_structural(item: str, prefixes: Tuple[str, ...]) -> bool:
    return any(item.startswith(p) for p in prefixes)


def _is_content(item: str) -> bool:
    return any(item.startswith(p + "_") for p in PILLAR_PREFIXES) or item.startswith("containsBugDescription")


def classify_rule(rule: pd.Series, base_rate: Dict[str, float], entailed: Dict[str, Set[str]]) -> Tuple[str, str]:
    """
    The fixed decision procedure of the rule-based protocol. First match wins:
      R1 consequent subset of antecedent            -> semantically_trivial (entailed)
      R2 consequent entailed by an antecedent label item (measured >= 0.90), with the rest
         structural/near-universal/in-antecedent  -> semantically_trivial (label redundancy);
         if the consequent also carries content items, the rule is judged on its content
      R3 all items structural                       -> taxonomy_artifact
      R4 antecedent entirely demographic            -> taxonomy_artifact
      R4b consequent entirely structural            -> taxonomy_artifact
      R5 every consequent item near-universal       -> semantically_trivial (base-rate restatement)
      R6 lift < 1.5                                 -> unclear (near-independence)
      R7 otherwise                                  -> actionable
    :param rule: A sampled rule row (antecedents, consequents, lift).
    :param base_rate: Item base rates from label_stats().
    :param entailed: Entailment map from label_stats().
    :return: (category, human-readable reason).
    """
    ant, cons = _rule_items(rule["antecedents"]), _rule_items(rule["consequents"])
    if cons and cons.issubset(ant):
        return "semantically_trivial", "R1 entailed: consequent subset of antecedent"
    ent: Set[str] = set()
    for x in ant:
        ent |= (entailed.get(x, set()) & cons)
    note = ""
    if ent:
        rest = cons - ent - ant
        content = sorted(i for i in rest if _is_content(i))
        unexplained = [i for i in rest
                       if not _is_content(i) and not _is_structural(i, STRUCT_ALL)
                       and base_rate.get(i, 0.0) < BASE_RATE_THRESHOLD]
        if not content and not unexplained:
            return "semantically_trivial", ("R2 label redundancy: consequent entailed by "
                                            + ", ".join(sorted(ent)))
        note = "R2 note: entailed " + ", ".join(sorted(ent)) + (
            " but consequent carries content: " + ", ".join(content) if content else "")
    if ant | cons and all(_is_structural(i, STRUCT_ALL) for i in (ant | cons)):
        return "taxonomy_artifact", "R3 all items structural"
    if ant and all(_is_structural(i, STRUCT_DEMO) for i in ant):
        return "taxonomy_artifact", "R4 demographic antecedent only"
    if cons and all(_is_structural(i, STRUCT_ALL) for i in cons):
        return "taxonomy_artifact", "R4b structural consequent only"
    if cons and all(base_rate.get(i, 0.0) >= BASE_RATE_THRESHOLD for i in cons):
        return "semantically_trivial", "R5 base-rate restatement (all consequent items near-universal)"
    if rule["lift"] < LOW_LIFT_THRESHOLD:
        return "unclear", f"R6 near-independence (lift={rule['lift']:.2f} < {LOW_LIFT_THRESHOLD})" + (f" [{note}]" if note else "")
    return "actionable", "R7 passes all filters" + (f" [{note}]" if note else "")


# ---------------------------------------------------------------------------
# Trace-back verification
# ---------------------------------------------------------------------------

def traceback_reviews(
    app_id: int,
    rule: pd.Series,
    max_reviews: int,
    seed: int,
    preloaded: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Retrieves a sample of the stored non-spam reviews whose transaction contains every
    antecedent item of the rule, so the association can be verified against actual content.
    :param app_id: The Steam unique identifier for the application.
    :param rule: A sampled rule row.
    :param max_reviews: Maximum number of supporting reviews to return.
    :param seed: Sampling seed.
    :param preloaded: Optional (reviews, transaction_sets) tuple with the full non-spam
                      review pool and its precomputed transaction sets; avoids reloading
                      and re-deriving every review once per rule.
    :return: A DataFrame with the supporting reviews (recommendationid, review text, date).
    """
    ant_items: Set[str] = set(str(rule["antecedents"]).split(" AND "))
    if preloaded is not None:
        reviews, txn_sets = preloaded
        supporting = [r for r, txn in zip(reviews, txn_sets) if ant_items.issubset(txn)]
    else:
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


def redundancy_stats(sampled: pd.DataFrame) -> Dict[str, Any]:
    """
    Objective redundancy measure over the inspected sample (no manual label needed):
    a sampled rule is redundant if its normalized full item-set signature (sorted
    antecedents + consequents, permutation-invariant) appears in more than one sampled
    rule; same-antecedent duplicates are counted separately.
    :param sampled: The inspected sample.
    :return: A dict with n_rules, n_redundant_full / redundancy_rate, and same-antecedent counts.
    """
    if sampled.empty:
        return {"n_rules": 0, "n_redundant_full": 0, "redundancy_rate": float("nan"),
                "n_same_antecedent": 0, "same_antecedent_rate": float("nan")}
    sig = sampled.apply(lambda r: rule_signature(r["antecedents"], r["consequents"]), axis=1)
    full = sig.apply(lambda s: s[0] + s[1])
    ant = sig.apply(lambda s: s[0])
    full_counts = full.value_counts()
    ant_counts = ant.value_counts()
    n = len(sampled)
    n_full = int((full.map(full_counts) > 1).sum())
    n_ant = int((ant.map(ant_counts) > 1).sum())
    return {
        "n_rules": n,
        "n_redundant_full": n_full,
        "redundancy_rate": n_full / n,
        "n_same_antecedent": n_ant,
        "same_antecedent_rate": n_ant / n,
    }


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
        "redundancy": redundancy_stats(sampled),
        "per_window_category_counts": per_window,
    }
    return report


# ---------------------------------------------------------------------------
# Tables (CSV, ready for the LaTeX tables of the thesis)
# ---------------------------------------------------------------------------
# RQ2 emits NO PDF figure: its results are small discrete counts and rates, which a
# table communicates better than a bar chart (thesis vademecum: figures must not be
# abundant). Two tidy CSVs are written instead: the overall metrics and the per-window
# category counts.

def write_rq2_tables(report: Dict[str, Any], out_path: str | Path) -> List[str]:
    """
    Writes the RQ2 result tables as CSVs: rq2_metrics_table.csv (tidy long format:
    category counts and rates, the three quality rates, persistence rates) and
    rq2_per_window_categories.csv (category counts per mining window).
    :param report: The metrics report dict produced by compute_metrics().
    :param out_path: Directory the CSV tables are written to (<out>/<app_id>/rq2/).
    :return: The list of written CSV paths.
    """
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)

    n = report["n_inspected"]
    rows = []
    for cat, count in report["category_counts"].items():
        rows.append({"group": "category", "metric": cat, "value": int(count)})
        rows.append({"group": "category_rate", "metric": cat,
                     "value": (count / n) if n else float("nan")})
    for key in ("residual_noise_rate", "actionability_rate", "unclear_rate"):
        rows.append({"group": "quality_rate", "metric": key, "value": report[key]})
    for scope in ("overall", "sampled"):
        pers = report["persistence"][scope]
        rows.append({"group": "persistence", "metric": f"{scope}_n_rules", "value": pers["n_rules"]})
        rows.append({"group": "persistence", "metric": f"{scope}_n_persistent", "value": pers["n_persistent"]})
        rows.append({"group": "persistence", "metric": f"{scope}_persistence_rate",
                     "value": pers["persistence_rate"]})
    red = report.get("redundancy", {})
    for key in ("n_rules", "n_redundant_full", "redundancy_rate", "n_same_antecedent", "same_antecedent_rate"):
        if key in red:
            rows.append({"group": "redundancy", "metric": key, "value": red[key]})
    metrics_csv = out / "rq2_metrics_table.csv"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)

    win_rows = [
        {"run_id": str(run_id), "category": cat, "count": int(cnt)}
        for run_id, counts in report["per_window_category_counts"].items()
        for cat, cnt in counts.items()
    ]
    per_window_csv = out / "rq2_per_window_categories.csv"
    pd.DataFrame(win_rows).to_csv(per_window_csv, index=False)

    return [str(metrics_csv), str(per_window_csv)]


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
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the RQ2 inspection pipeline for one game.
    :param app_id: The Steam unique identifier for the application.
    :param seed: The fixed sampling seed.
    :param top_k: Top-lift rules sampled per window.
    :param random_k: Random rules sampled per window.
    :param remine: If True, clears the stored rules and re-mines the full history first.
    :param annotations: Optional path to a manually annotated sample CSV, used as a
                        validation layer (rule-vs-human agreement is reported). When None,
                        metrics are computed from the rule-based sample_classified.csv.
    :param traceback_max_reviews: Max supporting reviews exported per actionable rule.
    :param out_dir: Root output directory; artifacts land in <out_dir>/<app_id>/rq2/.
    :param fig_dir: Directory for the PDF figures; defaults to <out_dir>/<app_id>/rq2/figures/.
    :param db_path: The shared study database to work on (default: <out>/<app_id>/<app_id>.db,
                    the appid folder; resolved via the STARAPTOR_DB_PATH env var by
                    storage.storage_utils). RQ2 reads — and with --remine rewrites — the
                    rules of this database, the one RQ1 stored its best run into.
    :return: The report dict (written to report.json when annotations are provided).
    """
    if db_path:
        os.environ["STARAPTOR_DB_PATH"] = str(Path(db_path).resolve())
    out_path = Path(out_dir) / str(app_id) / "rq2"
    out_path.mkdir(parents=True, exist_ok=True)

    rules = load_rules(app_id)
    if remine or rules.empty:
        # --remine forces a re-mine; on a from-scratch study database (no rules yet, e.g.
        # right after RQ1 stored its best run) mining happens automatically so RQ2 can
        # always work from the shared database.
        reason = "--remine" if remine else "no rules stored yet (from-scratch study db)"
        print(f"[App {app_id}] Mining association rules ({reason}): "
              f"window={DEFAULT_MINING_PARAMS['window_days']}d, step={DEFAULT_MINING_PARAMS['step_days']}d...")
        clear_mining_rules(app_id)
        analyzer = Analyzer(app_id)
        run_ids = analyzer.run_historical(**DEFAULT_MINING_PARAMS)
        if not run_ids:
            # Too few reviews for a full historical window (MIN_REVIEWS_THRESHOLD per
            # window): fall back to a single mining pass over the best sliding window,
            # so the study still has rules to sample.
            print(f"[App {app_id}] No historical window reached the review threshold; "
                  f"running a single-window mining pass instead.")
            analyzer.run(**{k: v for k, v in DEFAULT_MINING_PARAMS.items()
                             if k in ("min_support", "min_confidence", "min_lift", "deduplicate_mode")})
            run_ids = [r for r in [get_latest_run_id(app_id)] if r]
        print(f"[App {app_id}] Mining complete: {len(run_ids)} runs saved.")
        rules = load_rules(app_id)

    if rules.empty:
        raise RuntimeError(f"No mined rules found for App ID {app_id} even after mining.")

    sampled = sample_rules(rules, top_k, random_k, seed)
    sampled["proposed_category"] = [
        propose_category(a, c) for a, c in zip(sampled["antecedents"], sampled["consequents"])
    ]
    # Per-rule redundancy flags (same permutation-invariant signature as redundancy_stats):
    # lets the annotator see at a glance which sampled rules are near-duplicates.
    sig = sampled.apply(lambda r: rule_signature(r["antecedents"], r["consequents"]), axis=1)
    full_sig = sig.apply(lambda s: s[0] + s[1])
    sig_counts = full_sig.value_counts()
    sampled["signature_count"] = full_sig.map(sig_counts)
    sampled["redundant"] = sampled["signature_count"] > 1
    # Structured form for optional manual inspection: fill category + evidence by hand.
    sampled["category"] = ""
    sampled["evidence"] = ""
    sample_file = out_path / "sample.csv"
    sampled.to_csv(sample_file, index=False)
    print(f"[App {app_id}] Inspection sample ({len(sampled)} rules over {rules['run_id'].nunique()} windows, "
          f"seed {seed}) written to {sample_file}")

    # Rule-based classification (chosen protocol): deterministic, data-derived, reproducible.
    base_rate, entailed = label_stats(app_id)
    labels = [classify_rule(r, base_rate, entailed) for _, r in sampled.iterrows()]
    classified = sampled.copy()
    classified["category"] = [c for c, _ in labels]
    classified["evidence"] = [w for _, w in labels]
    classified["annotation_source"] = "rule-based"
    classified["human_verified"] = "pending"
    classified_file = out_path / "sample_classified.csv"
    classified.to_csv(classified_file, index=False)
    dist = classified["category"].value_counts().to_dict()
    print(f"[App {app_id}] Rule-based classification written to {classified_file}: {dist}")
    print("Optional: copy sample.csv to sample_annotated.csv, review/fill it by hand, and re-run "
          "with --annotations to validate the rule-based labels (agreement is reported).")

    if annotations is None:
        annotations = str(classified_file)

    ann = pd.read_csv(annotations)
    report = compute_metrics(ann, rules, sampled)
    report["app_id"] = app_id
    report["seed"] = seed
    report["classification"] = {
        "protocol": "rule-based (classify_rule; entailments measured at >= "
                    f"{ENTAILMENT_THRESHOLD}, lift threshold {LOW_LIFT_THRESHOLD})",
        "classified_file": str(classified_file),
        "distribution": {c: int(dist.get(c, 0)) for c in CATEGORIES},
    }
    # Validation layer: if a human-annotated file was used, report rule-vs-human agreement.
    if Path(annotations).resolve() != classified_file.resolve():
        rule_ann = pd.read_csv(classified_file)
        m = rule_ann.merge(ann[["rule_id", "category"]], on="rule_id", suffixes=("_rule", "_human"))
        report["validation"] = {
            "human_file": str(annotations),
            "n_compared": int(len(m)),
            "agreement": float((m["category_rule"] == m["category_human"]).mean()),
            "disagreements": int((m["category_rule"] != m["category_human"]).sum()),
        }
        print(f"[App {app_id}] Validation vs human annotations: "
              f"agreement={report['validation']['agreement']:.3f} "
              f"({report['validation']['disagreements']} disagreements / {len(m)})")

    # Trace-back verification for every rule classified as actionable. The full non-spam
    # review pool is loaded ONCE and shared across rules: reloading it per rule would cost
    # one full table scan per actionable rule.
    actionable = ann[ann["category"] == "actionable"]
    tb_frames = []
    preloaded = None
    if len(actionable):
        print(f"[App {app_id}] Loading non-spam review pool once for {len(actionable)} trace-backs...")
        standardizer = DataStandardizer(app_id)
        reviews = standardizer.get_window_review_list(0)
        txn_sets = [frozenset(standardizer.transactional(r)) for r in reviews]
        preloaded = (reviews, txn_sets)
        print(f"[App {app_id}] Pool loaded: {len(reviews)} reviews, transactions derived.")
    for _, rule in actionable.iterrows():
        tb = traceback_reviews(app_id, rule, traceback_max_reviews, seed, preloaded=preloaded)
        tb.insert(0, "rule_id", rule["rule_id"])
        tb_frames.append(tb)
        print(f"[App {app_id}] trace-back for rule {rule['rule_id']}: {len(tb)} supporting reviews.")
    if tb_frames:
        tb_all = pd.concat(tb_frames, ignore_index=True)
        tb_all.to_csv(out_path / "traceback_actionable_rules.csv", index=False)
        report["traceback_file"] = str(out_path / "traceback_actionable_rules.csv")

    tables = write_rq2_tables(report, out_path)
    report["tables"] = tables
    report["figures"] = []  # RQ2 results are tabular; no PDF figure is emitted
    for t in tables:
        print(f"[App {app_id}] table written: {t}")

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
    parser.add_argument("--db-path", type=str, default=None,
                        help="Shared study database to work on "
                             "(default: <out>/<app_id>/<app_id>.db, the appid folder).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--random-k", type=int, default=5)
    parser.add_argument("--remine", action="store_true", help="Clear stored rules and re-mine the full history first.")
    parser.add_argument("--annotations", type=str, default=None,
                        help="Optional manually annotated sample CSV, used as a validation layer "
                             "against the rule-based classification (agreement is reported). "
                             "Default: the rule-based out/<app_id>/rq2/sample_classified.csv.")
    parser.add_argument("--traceback-max-reviews", type=int, default=10)
    parser.add_argument("--out", type=str, default="./out")
    parser.add_argument("--fig-dir", type=str, default=None, help="Directory for the PDF figures.")
    args = parser.parse_args()
    db_path = args.db_path or str((Path(args.out).resolve() / str(args.app_id) / f"{args.app_id}.db"))
    run_rq2_inspection(
        app_id=args.app_id,
        db_path=db_path,
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
