import os

from analyzer.analyzer import Analyzer


def run_test(app_id: int):
    analyzer: Analyzer = Analyzer(app_id=app_id)

    print("Running FP-Growth on last sliding window with deduplication...")
    # Run with deduplication enabled to prune permutation noise and redundant sentiment variations
    analyzer.run(min_support=0.02, min_confidence=0.5, min_lift=1.2, deduplicate_mode='both')

    if not analyzer.metadata or analyzer.metadata.get("total_reviews", 0) == 0:
        print("No reviews found. Closing...")
        return

    for key, val in analyzer.metadata.items():
        print(f"{key}: {val}")

    print(f"Frequent Itemsets Found: {len(analyzer.frequent_itemsets)}")
    print(f"Total Association Rules Generated (After Deduplication): {len(analyzer.rules)}")

    if analyzer.rules is not None and not analyzer.rules.empty:
        # Ensure the output directory exists
        os.makedirs("./out", exist_ok=True)
        file_path = f"./out/{app_id}.csv"

        # Prepare a clean DataFrame (antecedents and consequents are already formatted as strings)
        export_df = analyzer.rules.copy()

        # Select the desired columns for the CSV file
        columns_to_save = [col for col in ["antecedents", "consequents", "support", "confidence", "lift"] if
                           col in export_df.columns]
        export_df = export_df[columns_to_save]

        # Save to CSV
        export_df.to_csv(file_path, index=False)
        print(f"Rules successfully saved to {file_path}")
    else:
        print("No association rules found to save.")


if __name__ == "__main__":
    run_test(1086940)