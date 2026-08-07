import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor

from analyzer.regressor import Regressor, XGB_BEST_PARAMS
from storage.storage_utils import (
    get_params_for_size,
    get_latest_tuning_record,
    list_tuning_history,
    save_tuned_params,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =====================================================================================
# INTERFACCIA DI CONTROLLO — determina se è necessario un nuovo ciclo di tuning.
# Le operazioni di fetch/store sono interamente delegate a storage.storage_utils.
# =====================================================================================

def needs_retuning(
    app_id: int,
    current_size: int,
    overshoot_ratio: float = 1.5,
    staleness_days: Optional[int] = 90,
) -> Tuple[bool, str]:
    """
    Determina se il dataset attuale richiede un nuovo ciclo di tuning.
    Controlla due condizioni indipendenti:
      1. Il dataset attuale supera di 'overshoot_ratio' volte il tier più grande
         mai tunato: i parametri esistenti sono stati ottimizzati su una quantità
         di dati insufficiente rispetto a quella oggi disponibile.
      2. L'ultimo tuning risale a più di 'staleness_days' giorni fa (se abilitato,
         staleness_days=None disabilita questo controllo).
    Ritorna una tupla (bool, motivo) per rendere la decisione ispezionabile/loggabile.
    """
    history = list_tuning_history(app_id)

    if history.empty:
        return True, "Nessun tuning precedente trovato per questo App ID."

    max_tier = int(history["tier_size"].max())
    if current_size > max_tier * overshoot_ratio:
        return True, (
            f"Il dataset attuale ({current_size} righe) supera di oltre "
            f"{overshoot_ratio}x il tier massimo tunato ({max_tier} righe)."
        )

    if staleness_days is not None:
        latest = get_latest_tuning_record(app_id)
        latest_ts = pd.to_datetime(latest["timestamp"])
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.tz_localize("UTC")
        now = pd.Timestamp.now(tz=latest_ts.tzinfo)
        age_days = (now - latest_ts).days
        if age_days > staleness_days:
            return True, f"L'ultimo tuning risale a {age_days} giorni fa (soglia: {staleness_days})."

    return False, "I parametri correnti sono ancora validi, nessun tuning necessario."


# =====================================================================================
# GENERAZIONE TIER E CROSS-VALIDATION WALK-FORWARD
# =====================================================================================

def generate_size_tiers(
    max_observed_size: int,
    base: int = 140,
    growth: float = 1.6,
    cap: int = 5000,
) -> List[int]:
    """
    Genera tier di dimensione dataset con crescita geometrica: densi vicino al
    cold-start (dove la sensibilità agli iperparametri è più alta), radi a
    dimensioni grandi (dove l'effetto marginale di un giorno in più si appiattisce).
    Include sempre max_observed_size come bordo superiore, per non tunare mai
    su dati che non esistono ancora.
    """
    if max_observed_size < base:
        return [max_observed_size]

    tiers = [base]
    while tiers[-1] * growth < min(max_observed_size, cap):
        tiers.append(int(tiers[-1] * growth))
    if tiers[-1] < max_observed_size:
        tiers.append(max_observed_size)
    return tiers


def adaptive_fold_params(n_samples: int) -> Tuple[int, int]:
    """Ritorna (test_size, n_folds) scalati alla dimensione del dataset corrente."""
    test_size = max(7, n_samples // 14)
    min_train = max(40, n_samples // 3)
    max_folds = max(1, (n_samples - min_train) // test_size)
    n_folds = max(3, min(6, max_folds))
    return test_size, n_folds


def _generate_folds(
    n_samples: int,
    n_folds: int,
    test_size: int,
    min_train_size: int = 40,
) -> List[Tuple[int, int, int]]:
    """
    Genera fold walk-forward (train_start=0, train_end, test_end), sempre a
    finestra espandente: il train set parte sempre dall'inizio della serie e
    cresce, non scorre in avanti scartando dati vecchi.
    """
    folds = []
    max_test_end = n_samples
    for i in range(n_folds):
        test_end = max_test_end - i * test_size
        train_end = test_end - test_size
        if train_end < min_train_size:
            break
        folds.append((0, train_end, test_end))
    return list(reversed(folds))


def _run_cv_benchmark(
    regressor: Regressor,
    model: XGBRegressor,
    X: pd.DataFrame,
    Y: pd.Series,
    df_full: pd.DataFrame,
    n_folds: int,
    test_size: int,
    decay_rate: float = 0.95,
) -> Dict[str, float]:
    """
    Esegue il benchmark su più fold walk-forward e aggrega le metriche
    (media e deviazione standard), riusando Regressor.benchmark() esistente
    su ciascuna finestra senza modificarlo.
    """
    folds = _generate_folds(len(X), n_folds=n_folds, test_size=test_size)
    if not folds:
        return {}

    fold_metrics: List[Dict[str, float]] = []

    for train_start, train_end, test_end in folds:
        X_fold = X.iloc[train_start:test_end].reset_index(drop=True)
        Y_fold = Y.iloc[train_start:test_end].reset_index(drop=True)
        df_fold = df_full.iloc[train_start:test_end].reset_index(drop=True)

        split_ratio = train_end / test_end

        fold_regressor = Regressor(app_id=regressor.app_id, model=copy.deepcopy(model))
        metrics, _ = fold_regressor.benchmark(
            model=model,
            split_ratio=split_ratio,
            decay_rate=decay_rate,
            data_tuple=(X_fold, Y_fold, df_fold),
        )
        if metrics:
            fold_metrics.append(metrics)

    if not fold_metrics:
        return {}

    agg: Dict[str, float] = {"n_folds": len(fold_metrics)}
    for key in fold_metrics[0]:
        if isinstance(fold_metrics[0][key], (int, float)):
            vals = [m[key] for m in fold_metrics]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))

    return agg


# =====================================================================================
# ORCHESTRAZIONE — tuning per singolo tier e ciclo completo per App ID.
# =====================================================================================

def _tune_tier(
    app_id: int,
    tier_size: int,
    data_tuple: Tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    n_trials: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """
    Esegue uno studio Optuna completo per un singolo tier di dimensione dataset.
    Ritorna (best_params, metrics_dict, n_folds_usati).
    """
    regressor = Regressor(app_id=app_id)
    test_size, n_folds = adaptive_fold_params(tier_size)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            "random_state": 42,
            "n_jobs": 2,
        }
        model = XGBRegressor(**params)

        agg = _run_cv_benchmark(
            regressor=regressor,
            model=model,
            X=data_tuple[0], Y=data_tuple[1], df_full=data_tuple[2],
            n_folds=n_folds, test_size=test_size,
        )
        if not agg:
            return float("inf")

        mape_ms = agg.get("mape_multistep_pct_mean", float("inf"))
        mape_1s = agg.get("mape_1step_pct_mean", float("inf"))
        mape_ms_std = agg.get("mape_multistep_pct_std", 0.0)

        trial.set_user_attr("mape_multistep_pct", mape_ms)
        trial.set_user_attr("mape_1step_pct", mape_1s)
        trial.set_user_attr("mape_multistep_std", mape_ms_std)

        # Objective Penalizing Underprediction (Asymmetric Peak Coverage)
        # Gives higher priority to multi-step stability and penalizes peak underestimation
        return (0.70 * mape_ms) + (0.30 * mape_1s) + (0.50 * mape_ms_std)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    metrics = {
        "score": best.value,
        "mape_multistep_pct": best.user_attrs["mape_multistep_pct"],
        "mape_1step_pct": best.user_attrs["mape_1step_pct"],
        "mape_multistep_std": best.user_attrs["mape_multistep_std"],
    }

    return best.params, metrics, n_folds


def run_tuning_cycle(
    app_id: int,
    n_trials: int = 150,
    force: bool = False,
    verbose: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Esegue il tuning per tutti i tier non ancora presenti nello storico, in base
    alla dimensione attuale del dataset dell'App ID. Salva ogni risultato in DB.
    """
    regressor = Regressor(app_id=app_id)
    X, Y, df_full = regressor.generator.build_daily_matrix()
    current_size = len(X)

    history = list_tuning_history(app_id)
    # If force=True, ignore history to re-tune existing tiers
    already_tuned = set(history["tier_size"].tolist()) if (not history.empty and not force) else set()

    tiers = generate_size_tiers(max_observed_size=current_size)
    tiers_to_tune = [t for t in tiers if t not in already_tuned]

    if not tiers_to_tune:
        if verbose:
            print(f"[App {app_id}] Nessun nuovo tier da tunare (dataset: {current_size} righe).")
        return {}

    results: Dict[int, Dict[str, Any]] = {}

    for tier_size in tiers_to_tune:
        if tier_size > current_size:
            continue

        data_tuple = (
            X.iloc[-tier_size:].reset_index(drop=True),
            Y.iloc[-tier_size:].reset_index(drop=True),
            df_full.iloc[-tier_size:].reset_index(drop=True),
        )

        if verbose:
            print(f"[App {app_id}] Tuning tier={tier_size} righe ({n_trials} trial, force={force})...")

        params, metrics, n_folds = _tune_tier(app_id, tier_size, data_tuple, n_trials)

        save_tuned_params(
            app_id=app_id,
            tier_size=tier_size,
            dataset_size_at_tuning=current_size,
            params=params,
            metrics=metrics,
            n_trials=n_trials,
            n_folds=n_folds,
        )

        results[tier_size] = {"params": params, "metrics": metrics}

        if verbose:
            print(f"[App {app_id}] Tier {tier_size} completato — "
                  f"MAPE multi-step: {metrics['mape_multistep_pct']:.2f}%, "
                  f"MAPE 1-step: {metrics['mape_1step_pct']:.2f}%")

    return results


def ensure_up_to_date_params(
    app_id: int,
    n_trials: int = 150,
    overshoot_ratio: float = 1.5,
    staleness_days: Optional[int] = 90,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Interfaccia principale: verifica se servono nuovi iperparametri per l'App ID
    dato, esegue il tuning se necessario, e ritorna comunque i parametri correnti
    più adatti alla dimensione attuale del dataset.
    Se non esiste alcun modello tunato in precedenza, il tuning è forzato.
    """
    regressor = Regressor(app_id=app_id)
    X, _, _ = regressor.generator.build_daily_matrix()
    current_size = len(X)

    should_tune, reason = needs_retuning(
        app_id, current_size,
        overshoot_ratio=overshoot_ratio,
        staleness_days=staleness_days,
    )

    if verbose:
        print(f"[App {app_id}] needs_retuning={should_tune} — {reason}")

    if should_tune:
        run_tuning_cycle(app_id, n_trials=n_trials, verbose=verbose)

    params = get_params_for_size(app_id, current_size)
    if params is None:
        raise RuntimeError(
            f"Tuning eseguito ma nessun parametro trovato per App ID {app_id}. "
            "Controllare che build_daily_matrix() produca dati validi."
        )

    return params


def resolve_params(
    app_id: int,
    n_trials: int = 150,
    auto_tune: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Risolve gli iperparametri da usare per un dato App ID, pensata per essere
    chiamata da script di test/benchmark che NON devono innescare un tuning
    costoso a loro insaputa.

    - auto_tune=False (default): usa esclusivamente lo storico già presente in DB.
      Se non esiste alcun tuning storicizzato, ricade su XGB_BEST_PARAMS (nessuna
      chiamata a Optuna, nessuna scrittura su DB).
    - auto_tune=True: delega a ensure_up_to_date_params, che verifica needs_retuning
      ed esegue un ciclo di tuning completo se necessario (può richiedere minuti).
    """
    if auto_tune:
        return ensure_up_to_date_params(app_id, n_trials=n_trials, verbose=verbose)

    regressor = Regressor(app_id=app_id)
    X, _, _ = regressor.generator.build_daily_matrix()
    params = get_params_for_size(app_id, len(X))

    if params is None:
        if verbose:
            print(f"[App {app_id}] Nessun tuning storicizzato trovato — "
                  f"uso i parametri di default (XGB_BEST_PARAMS).")
        return dict(XGB_BEST_PARAMS)

    return params


if __name__ == "__main__":
    ensure_up_to_date_params(app_id=1086940, n_trials=150)