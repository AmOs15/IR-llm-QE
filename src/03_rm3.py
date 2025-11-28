#!/usr/bin/env python3
"""
RM3 Pseudo-Relevance Feedback Search and Evaluation for MIRACL Japanese

Performs BM25 search with RM3 query expansion on dev queries,
then evaluates the results using Recall@K and nDCG@10 metrics.

RM3 (Relevance Model 3) uses pseudo-relevance feedback to expand queries
by extracting terms from top-ranked documents and interpolating them
with the original query.

Results are saved with timestamps to enable multiple runs.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm


# Official MIRACL Japanese baseline scores for comparison
OFFICIAL_BASELINE = {
    'recall_100': 0.8048,
    'ndcg_10': 0.3689,
}

# RM3 parameters (standard conservative settings)
RM3_FB_DOCS = 10              # Number of feedback documents
RM3_FB_TERMS = 10             # Number of expansion terms
RM3_ORIGINAL_QUERY_WEIGHT = 0.5  # Interpolation coefficient (0.0-1.0)


def load_topics(topics_file: Path) -> Dict[str, str]:
    """
    Load dev topics (queries) from TSV file.

    Args:
        topics_file: Path to topics.dev.tsv

    Returns:
        Dictionary mapping query_id to query text
    """
    topics = {}

    with open(topics_file, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                query_id = parts[0]
                query_text = parts[1]
                topics[query_id] = query_text

    return topics


def load_qrels(qrels_file: Path) -> Dict[str, Dict[str, int]]:
    """
    Load qrels (relevance judgments) from TSV file.

    Args:
        qrels_file: Path to qrels.dev.tsv

    Returns:
        Nested dictionary: {query_id: {doc_id: relevance}}
    """
    qrels = {}

    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                query_id = parts[0]
                doc_id = parts[2]
                relevance = int(parts[3])

                if query_id not in qrels:
                    qrels[query_id] = {}

                qrels[query_id][doc_id] = relevance

    return qrels


def search_queries(
    searcher: LuceneSearcher,
    topics: Dict[str, str],
    top_k: int = 100
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Perform BM25 search with RM3 query expansion for all queries.

    RM3 query expansion is performed automatically by the searcher
    if configured via set_rm3().

    Args:
        searcher: Pyserini LuceneSearcher (with RM3 enabled)
        topics: Dictionary of query_id to query text
        top_k: Number of top documents to retrieve

    Returns:
        Dictionary mapping query_id to list of (doc_id, score) tuples
    """
    results = {}

    print(f"\nSearching {len(topics)} queries...")

    for query_id, query_text in tqdm(topics.items(), desc="Searching"):
        try:
            hits = searcher.search(query_text, k=top_k)

            # Store results as list of (doc_id, score) tuples
            results[query_id] = [(hit.docid, hit.score) for hit in hits]

        except Exception as e:
            print(f"\n⚠ Error searching query {query_id}: {e}")
            results[query_id] = []

    return results


def save_trec_results(
    results: Dict[str, List[Tuple[str, float]]],
    output_file: Path,
    run_name: str
) -> None:
    """
    Save search results in TREC format.

    Format: query_id Q0 doc_id rank score run_name

    Args:
        results: Dictionary of query_id to list of (doc_id, score) tuples
        output_file: Path to output file
        run_name: Name of the run
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for query_id, doc_scores in sorted(results.items()):
            for rank, (doc_id, score) in enumerate(doc_scores, start=1):
                f.write(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{score}\t{run_name}\n")


def calculate_recall_at_k(
    results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int
) -> float:
    """
    Calculate Recall@K metric.

    Args:
        results: Dictionary of query_id to list of (doc_id, score) tuples
        qrels: Dictionary of query_id to dict of doc_id to relevance
        k: Cutoff position

    Returns:
        Recall@K score (macro-averaged across queries)
    """
    recall_scores = []

    for query_id, doc_scores in results.items():
        # Get relevant documents for this query
        relevant_docs = qrels.get(query_id, {})
        if not relevant_docs:
            continue

        # Get top-k retrieved documents
        retrieved_docs = set(doc_id for doc_id, _ in doc_scores[:k])

        # Count how many relevant documents were retrieved
        relevant_retrieved = sum(
            1 for doc_id in relevant_docs
            if doc_id in retrieved_docs and relevant_docs[doc_id] > 0
        )

        # Calculate recall for this query
        num_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
        if num_relevant > 0:
            recall = relevant_retrieved / num_relevant
            recall_scores.append(recall)

    # Return macro-averaged recall
    return sum(recall_scores) / len(recall_scores) if recall_scores else 0.0


def calculate_ndcg_at_k(
    results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int
) -> float:
    """
    Calculate nDCG@K (Normalized Discounted Cumulative Gain).

    Args:
        results: Dictionary of query_id to list of (doc_id, score) tuples
        qrels: Dictionary of query_id to dict of doc_id to relevance
        k: Cutoff position

    Returns:
        nDCG@K score (macro-averaged across queries)
    """
    import math

    ndcg_scores = []

    for query_id, doc_scores in results.items():
        # Get relevant documents for this query
        relevant_docs = qrels.get(query_id, {})
        if not relevant_docs:
            continue

        # Calculate DCG@K
        dcg = 0.0
        for i, (doc_id, _) in enumerate(doc_scores[:k], start=1):
            relevance = relevant_docs.get(doc_id, 0)
            dcg += relevance / math.log2(i + 1)

        # Calculate IDCG@K (ideal DCG)
        ideal_relevances = sorted(relevant_docs.values(), reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevances[:k], start=1):
            idcg += rel / math.log2(i + 1)

        # Calculate nDCG
        if idcg > 0:
            ndcg = dcg / idcg
            ndcg_scores.append(ndcg)

    # Return macro-averaged nDCG
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


def evaluate_results(
    results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]]
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.

    Args:
        results: Dictionary of query_id to list of (doc_id, score) tuples
        qrels: Dictionary of query_id to dict of doc_id to relevance

    Returns:
        Dictionary of metric names to values
    """
    print("\nCalculating evaluation metrics...")

    metrics = {}

    # Calculate Recall@K
    print("  Computing Recall@10...")
    metrics['recall_10'] = calculate_recall_at_k(results, qrels, k=10)

    print("  Computing Recall@100...")
    metrics['recall_100'] = calculate_recall_at_k(results, qrels, k=100)

    # Calculate nDCG@K
    print("  Computing nDCG@10...")
    metrics['ndcg_10'] = calculate_ndcg_at_k(results, qrels, k=10)

    return metrics


def save_evaluation_json(
    metrics: Dict[str, float],
    output_file: Path,
    run_name: str,
    timestamp: str,
    num_queries: int,
    search_params: Dict[str, float]
) -> None:
    """
    Save evaluation results in JSON format.

    Args:
        metrics: Dictionary of metric names to values
        output_file: Path to output JSON file
        run_name: Name of the run
        timestamp: ISO format timestamp
        num_queries: Number of queries processed
        search_params: Search parameters (BM25 + RM3) used
    """
    # Calculate differences from official baseline
    baseline_comparison = {}
    for metric_name, baseline_value in OFFICIAL_BASELINE.items():
        if metric_name in metrics:
            diff = metrics[metric_name] - baseline_value
            baseline_comparison[f"{metric_name}_diff"] = diff

    # Create evaluation report
    report = {
        'run_name': run_name,
        'timestamp': timestamp,
        'query_expansion': 'rm3',
        'metrics': metrics,
        'baseline_comparison': baseline_comparison,
        'parameters': search_params,
        'num_queries': num_queries,
        'official_baseline': OFFICIAL_BASELINE,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def save_evaluation_csv(
    metrics: Dict[str, float],
    output_file: Path
) -> None:
    """
    Save evaluation results in CSV format.

    Args:
        metrics: Dictionary of metric names to values
        output_file: Path to output CSV file
    """
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(['metric', 'value', 'baseline', 'difference'])

        # Write metrics
        for metric_name, value in sorted(metrics.items()):
            baseline_value = OFFICIAL_BASELINE.get(metric_name, None)

            if baseline_value is not None:
                diff = value - baseline_value
                writer.writerow([metric_name, f"{value:.4f}", f"{baseline_value:.4f}", f"{diff:+.4f}"])
            else:
                writer.writerow([metric_name, f"{value:.4f}", '-', '-'])


def print_evaluation_summary(metrics: Dict[str, float]) -> None:
    """
    Print evaluation results to stdout.

    Args:
        metrics: Dictionary of metric names to values
    """
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    for metric_name, value in sorted(metrics.items()):
        baseline_value = OFFICIAL_BASELINE.get(metric_name)

        if baseline_value is not None:
            diff = value - baseline_value
            status = "✓" if diff >= 0 else "✗"
            print(f"  {status} {metric_name:12s}: {value:.4f}  (baseline: {baseline_value:.4f}, diff: {diff:+.4f})")
        else:
            print(f"    {metric_name:12s}: {value:.4f}")


def main():
    """Main function to run RM3 query expansion search and evaluation."""
    print("=" * 60)
    print("RM3 Pseudo-Relevance Feedback Search and Evaluation")
    print("=" * 60)

    # Define paths
    base_dir = Path(__file__).parent.parent
    index_dir = base_dir / "indexes" / "miracl-ja-bm25"
    topics_file = base_dir / "data" / "topics" / "topics.dev.tsv"
    qrels_file = base_dir / "data" / "qrels" / "qrels.dev.tsv"
    results_dir = base_dir / "results"

    # Create timestamp for filenames
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    run_name = f"rm3_{timestamp_str}"

    print(f"\nRun name: {run_name}")
    print(f"Timestamp: {timestamp.isoformat()}")
    print(f"Query expansion: RM3 (Relevance Model 3)")

    # Check prerequisites
    print("\n" + "=" * 60)
    print("Checking Prerequisites")
    print("=" * 60)

    if not index_dir.exists():
        print(f"✗ Index not found at {index_dir}")
        print("\nPlease create the index first:")
        print("  poetry run python src/02_index.py")
        return 1

    if not topics_file.exists():
        print(f"✗ Topics file not found at {topics_file}")
        print("\nPlease download the data first:")
        print("  poetry run python src/01_download.py")
        return 1

    if not qrels_file.exists():
        print(f"✗ Qrels file not found at {qrels_file}")
        print("\nPlease download the data first:")
        print("  poetry run python src/01_download.py")
        return 1

    print(f"✓ Index found: {index_dir}")
    print(f"✓ Topics found: {topics_file}")
    print(f"✓ Qrels found: {qrels_file}")

    # Load data
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)

    print("Loading topics...")
    topics = load_topics(topics_file)
    print(f"✓ Loaded {len(topics)} queries")

    print("Loading qrels...")
    qrels = load_qrels(qrels_file)
    num_qrels = sum(len(docs) for docs in qrels.values())
    print(f"✓ Loaded {num_qrels} relevance judgments for {len(qrels)} queries")

    # Initialize searcher
    print("\n" + "=" * 60)
    print("Initializing BM25 Searcher with RM3")
    print("=" * 60)

    print(f"Loading index from {index_dir}...")
    searcher = LuceneSearcher(str(index_dir))
    searcher.set_language('ja')

    # Set BM25 parameters
    bm25_k1 = 1.2
    bm25_b = 0.75
    searcher.set_bm25(k1=bm25_k1, b=bm25_b)

    # Configure RM3 query expansion
    searcher.set_rm3(
        fb_terms=RM3_FB_TERMS,
        fb_docs=RM3_FB_DOCS,
        original_query_weight=RM3_ORIGINAL_QUERY_WEIGHT
    )

    print(f"✓ Searcher initialized")
    print(f"  BM25 parameters: k1={bm25_k1}, b={bm25_b}")
    print(f"  RM3 parameters: fb_docs={RM3_FB_DOCS}, fb_terms={RM3_FB_TERMS}, original_query_weight={RM3_ORIGINAL_QUERY_WEIGHT}")

    # Perform search
    print("\n" + "=" * 60)
    print("Performing Search with RM3 Query Expansion")
    print("=" * 60)

    top_k = 100
    results = search_queries(searcher, topics, top_k=top_k)

    print(f"✓ Search completed")
    print(f"  Retrieved top-{top_k} documents for {len(results)} queries")
    print(f"  Query expansion: RM3 applied automatically during search")

    # Save search results
    print("\n" + "=" * 60)
    print("Saving Search Results")
    print("=" * 60)

    trec_file = results_dir / f"{run_name}.trec"
    print(f"Saving TREC results to {trec_file}...")
    save_trec_results(results, trec_file, run_name)
    print(f"✓ TREC results saved")

    # Evaluate results
    print("\n" + "=" * 60)
    print("Evaluating Results")
    print("=" * 60)

    metrics = evaluate_results(results, qrels)

    # Save evaluation results
    print("\n" + "=" * 60)
    print("Saving Evaluation Results")
    print("=" * 60)

    # JSON format
    json_file = results_dir / f"eval_{run_name}.json"
    print(f"Saving JSON to {json_file}...")
    save_evaluation_json(
        metrics,
        json_file,
        run_name,
        timestamp.isoformat(),
        len(topics),
        {
            'k1': bm25_k1,
            'b': bm25_b,
            'top_k': top_k,
            'rm3_fb_docs': RM3_FB_DOCS,
            'rm3_fb_terms': RM3_FB_TERMS,
            'rm3_original_query_weight': RM3_ORIGINAL_QUERY_WEIGHT
        }
    )
    print(f"✓ JSON saved")

    # CSV format
    csv_file = results_dir / f"eval_{run_name}.csv"
    print(f"Saving CSV to {csv_file}...")
    save_evaluation_csv(metrics, csv_file)
    print(f"✓ CSV saved")

    # Print summary
    print_evaluation_summary(metrics)

    # Final summary
    print("\n" + "=" * 60)
    print("Files Saved")
    print("=" * 60)
    print(f"  TREC results: {trec_file}")
    print(f"  JSON eval:    {json_file}")
    print(f"  CSV eval:     {csv_file}")

    print("\n" + "=" * 60)
    print("RM3 Query Expansion Run Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
