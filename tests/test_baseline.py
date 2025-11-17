"""Test baseline: Semantic search only (no improvements)."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bulk_testing import BulkTestRunner, BulkTestConfig
from dataset_adapters import FinanceBenchAdapter


def main():
    print("="*80)
    print("TEST: BASELINE (Semantic Search Only)")
    print("="*80)
    print("\nConfiguration:")
    print("  - Hybrid Search: DISABLED")
    print("  - Metadata Filtering: DISABLED")
    print("  - Reranking: DISABLED")
    print("  - Pure semantic search with OpenAI embeddings")
    print("\n" + "="*80 + "\n")

    # Create config with all improvements disabled
    config = BulkTestConfig(
        dataset_name='financebench',
        top_k_retrieval=10,
        use_hybrid_search=False,
        use_metadata_filtering=False,
        use_reranking=False
    )

    # Create adapter with subset
    adapter = FinanceBenchAdapter(
        subset_csv='data/question_sets/financebench_subset_questions.csv'
    )

    # Run test
    runner = BulkTestRunner(config)
    results_df = runner.run_bulk_test(adapter)
    runner.save_results(results_df, adapter)


if __name__ == "__main__":
    main()
