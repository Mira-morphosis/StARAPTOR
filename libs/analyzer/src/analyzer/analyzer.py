import pandas as pd
from analyzer.data_standardizer import DataStandardizer
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from storage.storage_utils import save_mining_results


def _filter_tautologies(rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rules whose antecedent and consequent are identical (circular/tautological rules).
    :param rules_df: The rules' DataFrame to filter.
    :return: The filtered DataFrame.
    """
    valid_indices = []
    for idx, row in rules_df.iterrows():
        ants = set(row['antecedents'])
        cons = set(row['consequents'])
        if ants != cons:
            valid_indices.append(idx)
    return rules_df.loc[valid_indices].reset_index(drop=True)


def _filter_trivial_hardware_rules(rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rules made up entirely of hardware attributes on both sides (e.g. hw_ram_31GB -> hw_vram_16GB),
    which are noise rather than useful insight about the game itself.
    :param rules_df: The rules' DataFrame to filter.
    :return: The filtered DataFrame.
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

        # Keep the rule if it includes at least one non-hardware feature (review, gameplay, KPI, ...)
        if not (all_hardware_ants and all_hardware_cons):
            valid_indices.append(idx)

    return rules_df.loc[valid_indices].reset_index(drop=True)


def _prune_itemset_permutations(rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses different antecedent/consequent permutations of the same item pool to the highest-lift one.
    :param rules_df: The rules' DataFrame to prune.
    :return: The pruned DataFrame.
    """
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
    Removes specialized rules (A AND B -> C) whose confidence gain over the base rule (A -> C)
    is smaller than min_improvement.
    :param rules_df: The rules' DataFrame to prune.
    :param min_improvement: The minimum confidence gain required to keep a specialized rule.
    :return: The pruned DataFrame.
    """
    if rules_df.empty:
        return rules_df

    rules_df = rules_df.copy()
    rules_df['ant_set'] = rules_df['antecedents'].apply(lambda x: set(x.split(' AND ')) if isinstance(x, str) else x)
    rules_df['cons_set'] = rules_df['consequents'].apply(lambda x: set(x.split(' AND ')) if isinstance(x, str) else x)

    indices_to_remove = set()
    records = rules_df.to_dict('records')

    for i, spec_rule in enumerate(records):
        for j, base_rule in enumerate(records):
            if i != j and i not in indices_to_remove:
                if spec_rule['cons_set'] == base_rule['cons_set']:
                    if base_rule['ant_set'].issubset(spec_rule['ant_set']) and base_rule['ant_set'] != spec_rule['ant_set']:
                        if (spec_rule['confidence'] - base_rule['confidence']) < min_improvement:
                            indices_to_remove.add(i)
                            break

    cleaned_df = rules_df.drop(index=list(indices_to_remove)).drop(columns=['ant_set', 'cons_set'])
    return cleaned_df.reset_index(drop=True)


def _deduplicate_by_antecedent(rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the highest-lift rule for each unique antecedent.
    :param rules_df: The rules' DataFrame to deduplicate.
    :return: The deduplicated DataFrame.
    """
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
    Runs FP-Growth association mining on a set of transactions and applies noise-filtering/deduplication.

    FP-Growth finds sets of items that show up together often (frequent itemsets),
    then turns them into "if you see A, you probably see B" rules. Support is how
    common an itemset is overall; confidence is how often the rule holds when the antecedent
    is present; lift is how much more likely the consequent is compared to random chance. It's
    the same family of algorithm used for market-basket analysis ("customers who bought X also
    bought Y"), applied here to review attributes instead of products.
    :param transactions: A list of transactions, each a list of string item tags.
    :param min_support: The minimum support threshold for frequent itemsets.
    :param min_confidence: The minimum confidence threshold for generated rules.
    :param min_lift: The minimum lift threshold for generated rules.
    :param deduplicate_mode: One of 'itemset', 'antecedent', 'both', or any other value to skip deduplication.
    :return: A DataFrame of mined association rules, sorted by lift descending.
    """
    if not transactions:
        return pd.DataFrame()

    # Drop near-universal items (present in >95% of transactions): they carry no signal
    flat_items = [item for t in transactions for item in t]
    item_counts = pd.Series(flat_items).value_counts()
    total_trans = len(transactions)
    universal_items = set(item_counts[item_counts > int(0.95 * total_trans)].index)
    cleaned_transactions = [[item for item in t if item not in universal_items] for t in transactions]

    if not cleaned_transactions:
        return pd.DataFrame()

    te = TransactionEncoder()
    te_ary = te.fit(cleaned_transactions).transform(cleaned_transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if min_lift:
        rules = rules[rules['lift'] >= min_lift]

    # Filter out tautologies and pure-hardware noise
    rules = _filter_tautologies(rules)
    rules = _filter_trivial_hardware_rules(rules)

    # Deduplicate
    if deduplicate_mode in ['itemset', 'both']:
        rules = _prune_itemset_permutations(rules)

    if deduplicate_mode in ['antecedent', 'both']:
        rules = _deduplicate_by_antecedent(rules)

    rules = _prune_subsup_redundancies(rules, min_improvement=0.05)

    # Format itemsets as sorted, joined strings for storage
    rules['antecedents'] = rules['antecedents'].apply(lambda x: " AND ".join(sorted(list(x))))
    rules['consequents'] = rules['consequents'].apply(lambda x: " AND ".join(sorted(list(x))))

    return rules.sort_values(by="lift", ascending=False).reset_index(drop=True)


class Analyzer:
    """Runs FP-Growth association mining over a game's reviews, either as a single pass or across historical windows."""

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
        """
        Runs a single mining pass over the current best sliding window (used for the daily cycle).
        :param min_support: The minimum support threshold for frequent itemsets.
        :param min_confidence: The minimum confidence threshold for generated rules.
        :param min_lift: The minimum lift threshold for generated rules.
        :param deduplicate_mode: One of 'itemset', 'antecedent', 'both', or any other value to skip deduplication.
        :param save_to_db: Whether to persist the resulting rules to DuckDB.
        :return: A DataFrame of mined association rules (empty if no transactions were found).
        """
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
        """
        Runs mining over every historical sliding window available, saving each run separately.
        :param window_days: The size in days of each sliding window.
        :param step_days: The number of days to shift back between consecutive windows.
        :param min_support: The minimum support threshold for frequent itemsets.
        :param min_confidence: The minimum confidence threshold for generated rules.
        :param min_lift: The minimum lift threshold for generated rules.
        :param deduplicate_mode: One of 'itemset', 'antecedent', 'both', or any other value to skip deduplication.
        :return: A list of the run_ids that were saved.
        """
        historical_datasets = self.standardizer.get_historical_sliding_windows(
            window_days=window_days,
            step_days=step_days
        )
        print(f"[*] Found {len(historical_datasets)} historical sliding windows to analyze.")

        saved_runs = []
        for transactions, metadata in historical_datasets:
            label = metadata.get("window_label", "")
            print(f"--> Mining window {label} ({metadata['total_reviews']} reviews)...")

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

        print(f"\n[✓] Historical mining complete: saved {len(saved_runs)} runs to the DB.")
        return saved_runs