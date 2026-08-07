import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from analyzer.data_standardizer import DataStandardizer
from storage.storage_utils import save_mining_results


def _filter_tautologies(rules_df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove regole circolari o tautologiche."""
    valid_indices = []
    for idx, row in rules_df.iterrows():
        ants = set(row['antecedents'])
        cons = set(row['consequents'])
        if ants != cons:
            valid_indices.append(idx)
    return rules_df.loc[valid_indices].reset_index(drop=True)


def _filter_trivial_hardware_rules(rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove le regole composte ESCLUSIVAMENTE da attributi hardware sia negli antecedenti
    sia nei conseguenti (es. hw_ram_31GB -> hw_vram_16GB), che creano rumore
    senza offrire insight concreti sul gioco.
    """
    if rules_df.empty:
        return rules_df

    def is_hardware_item(item: str) -> bool:
        return item.startswith("hw_")

    valid_indices = []
    for idx, row in rules_df.iterrows():
        ants = set(row['antecedents'])
        cons = set(row['consequents'])

        all_hardware_ants = all(is_hardware_item(i) for i in ants)
        all_hardware_cons = all(is_hardware_item(i) for i in cons)

        # Mantiene la regola se include ALMENO una feature non hardware (es. review, gameplay, KPI)
        if not (all_hardware_ants and all_hardware_cons):
            valid_indices.append(idx)

    return rules_df.loc[valid_indices].reset_index(drop=True)


def _prune_itemset_permutations(rules_df: pd.DataFrame) -> pd.DataFrame:
    """Elimina le permutazioni dello stesso pool di elementi conservando quella a Lift maggiore."""
    if rules_df.empty:
        return rules_df

    sorted_df = rules_df.sort_values(by="lift", ascending=False).copy()
    kept_indices = []
    seen_signatures = set()

    for idx, row in sorted_df.iterrows():
        all_items = sorted(list(row["antecedents"]) + list(row["consequents"]))
        signature = tuple(all_items)
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            kept_indices.append(idx)

    return rules_df.loc[kept_indices].reset_index(drop=True)

def _prune_subsup_redundancies(rules_df: pd.DataFrame, min_improvement: float = 0.05) -> pd.DataFrame:
    """
    Rimuove le regole specializzate (A AND B -> C) se il loro incremento di confidenza
    rispetto alla regola base (A -> C) è inferiore a min_improvement.
    """
    if rules_df.empty:
        return rules_df

    # Lavora temporaneamente coi set per efficienza
    rules_df = rules_df.copy()
    rules_df['ant_set'] = rules_df['antecedents'].apply(lambda x: set(x.split(' AND ')) if isinstance(x, str) else x)
    rules_df['cons_set'] = rules_df['consequents'].apply(lambda x: set(x.split(' AND ')) if isinstance(x, str) else x)

    indices_to_remove = set()
    records = rules_df.to_dict('records')

    for i, spec_rule in enumerate(records):
        for j, base_rule in enumerate(records):
            if i != j and i not in indices_to_remove:
                # Controlla se hanno lo stesso conseguente
                if spec_rule['cons_set'] == base_rule['cons_set']:
                    # Controlla se base_rule è un sottoinsieme stretto di spec_rule
                    if base_rule['ant_set'].issubset(spec_rule['ant_set']) and base_rule['ant_set'] != spec_rule['ant_set']:
                        # Se la regola specializzata NON migliora sufficientemente la confidenza, si marca come ridondante
                        if (spec_rule['confidence'] - base_rule['confidence']) < min_improvement:
                            indices_to_remove.add(i)
                            break

    cleaned_df = rules_df.drop(index=list(indices_to_remove)).drop(columns=['ant_set', 'cons_set'])
    return cleaned_df.reset_index(drop=True)


def _deduplicate_by_antecedent(rules_df: pd.DataFrame) -> pd.DataFrame:
    """Mantiene solo la regola a Lift più alto per ciascun antecedente unico."""
    if rules_df.empty:
        return rules_df

    sorted_df = rules_df.sort_values(by="lift", ascending=False).copy()
    return sorted_df.drop_duplicates(subset=["antecedents"], keep="first").reset_index(drop=True)


def _mine_transactions(
        transactions: list,
        min_support: float,
        min_confidence: float,
        min_lift: float,
        deduplicate_mode: str
) -> pd.DataFrame:
    """
    Motore unico di estrazione: applica FP-Growth e filtraggio su un set di transazioni.
    """
    if not transactions:
        return pd.DataFrame()

        # Rimuove elementi quasi universali (>95%)
    flat_items = [item for t in transactions for item in t]
    item_counts = pd.Series(flat_items).value_counts()
    total_trans = len(transactions)
    universal_items = set(item_counts[item_counts > int(0.95 * total_trans)].index)
    cleaned_transactions = [[item for item in t if item not in universal_items] for t in transactions]

    if not cleaned_transactions:
        return pd.DataFrame()

    # Encoding
    te = TransactionEncoder()
    te_ary = te.fit(cleaned_transactions).transform(cleaned_transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Mining FP-Growth
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if min_lift:
        rules = rules[rules['lift'] >= min_lift]

    # 1. Filtro Tautologie e Correlazioni Puramente Hardware (Rumore Trivial)
    rules = _filter_tautologies(rules)
    rules = _filter_trivial_hardware_rules(rules)

    # 2. Deduplicazione
    if deduplicate_mode in ['itemset', 'both']:
        rules = _prune_itemset_permutations(rules)

    if deduplicate_mode in ['antecedent', 'both']:
        rules = _deduplicate_by_antecedent(rules)

    rules = _prune_subsup_redundancies(rules, min_improvement=0.05)

    # Formattazione finale in stringhe
    rules['antecedents'] = rules['antecedents'].apply(lambda x: " AND ".join(sorted(list(x))))
    rules['consequents'] = rules['consequents'].apply(lambda x: " AND ".join(sorted(list(x))))

    return rules.sort_values(by="lift", ascending=False).reset_index(drop=True)


class Analyzer:
    def __init__(self, app_id: int):
        self.app_id = app_id
        self.standardizer = DataStandardizer(app_id)
        self.metadata = {}
        self.frequent_itemsets = pd.DataFrame()
        self.rules = pd.DataFrame()

    def run(
            self,
            min_support=0.02,
            min_confidence=0.5,
            min_lift=1.2,
            deduplicate_mode='both',
            save_to_db=True
    ):
        """Esecuzione singola standard (es. per il ciclo daily)."""
        transactions, metadata = self.standardizer.get_mining_dataset()
        self.metadata = metadata

        if not transactions or metadata.get("total_reviews", 0) == 0:
            print("No transactions found.")
            return pd.DataFrame()

        self.rules = _mine_transactions(
            transactions, min_support, min_confidence, min_lift, deduplicate_mode
        )

        if save_to_db and not self.rules.empty:
            params = {
                "min_support": min_support,
                "min_confidence": min_confidence,
                "min_lift": min_lift,
                "deduplicate_mode": deduplicate_mode
            }
            run_id = save_mining_results(self.app_id, self.rules, self.metadata, params)
            print(f"Mining run saved successfully under Run ID: {run_id}")

        return self.rules

    def run_historical(
            self,
            window_days: int = 14,
            step_days: int = 7,
            min_support=0.02,
            min_confidence=0.5,
            min_lift=1.2,
            deduplicate_mode='both'
    ) -> list[str]:
        """Esegue il mining per TUTTE le sliding window definite nello storico."""
        historical_datasets = self.standardizer.get_historical_sliding_windows(
            window_days=window_days,
            step_days=step_days
        )
        print(f"[*] Trovate {len(historical_datasets)} sliding window storiche da analizzare.")

        saved_runs = []
        for transactions, metadata in historical_datasets:
            label = metadata.get("window_label", "")
            print(f"--> Mining della finestra {label} ({metadata['total_reviews']} recensioni)...")

            rules = _mine_transactions(
                transactions, min_support, min_confidence, min_lift, deduplicate_mode
            )

            if not rules.empty:
                params = {
                    "min_support": min_support,
                    "min_confidence": min_confidence,
                    "min_lift": min_lift,
                    "deduplicate_mode": deduplicate_mode
                }
                run_id = save_mining_results(self.app_id, rules, metadata, params)
                saved_runs.append(run_id)

        print(f"\n[✓] Mining storico completato: salvate {len(saved_runs)} run nel DB.")
        return saved_runs