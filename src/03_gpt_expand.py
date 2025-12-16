#!/usr/bin/env python3
"""
GPT-OSS-20B Query Expansion with BM25 Search and Evaluation

Expands queries using GPT-OSS-20B LLM to generate similar words,
then performs BM25 search and evaluates results using Recall@K and nDCG@10 metrics.

Key features:
- LLM-based query expansion with similar words
- Retry mechanism (max 3 attempts) for LLM failures
- Skipped queries are tracked and saved
- Results saved with timestamps
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

from prompt import PROMPT_SIMPLE_TEMPLATE, PROMPT_COT_AGR_TEMPLATE, PROMPT_COT_VIRTUAL_ANSWER_TEMPLATE


# Official MIRACL Japanese baseline scores for comparison
OFFICIAL_BASELINE = {
    'recall_100': 0.8048,
    'ndcg_10': 0.3689,
}
BASELINE_SCORE ={
    'ndcg_10': 0.2934,
    "recall_10": 0.4055,
    'recall_100': 0.7501
}

# LLM configuration
MAX_RETRIES = 10
LLM_MODEL_NAME = "openai/gpt-oss-20b"

# LLM_SYSTEM_PROMPT = PROMPT_SIMPLE_TEMPLATE
LLM_SYSTEM_PROMPT = PROMPT_COT_AGR_TEMPLATE
# LLM_SYSTEM_PROMPT = PROMPT_COT_VIRTUAL_ANSWER_TEMPLATE


def initialize_llm():
    """
    Initialize the LLM pipeline for query expansion.

    Returns:
        pipeline object or None if initialization fails
    """
    try:
        from transformers import pipeline

        print("Initializing LLM...")
        print(f"  Model: {LLM_MODEL_NAME}")
        print(f"  Quantization: MXFP4 (auto)")

        generator = pipeline(
            "text-generation",
            model=LLM_MODEL_NAME,
            torch_dtype="auto",
            device_map="auto"
        )

        print("✓ LLM initialized successfully")
        return generator

    except Exception as e:
        print(f"✗ Failed to initialize LLM: {e}")
        return None


def parse_llm_output(llm_response: str) -> Optional[List[str]]:
    """
    Parse LLM output to extract word list.

    Expected format: ["word1", "word2", "word3", ...]

    Args:
        llm_response: Raw LLM output text

    Returns:
        List of words if parsing succeeds, None otherwise
    """
    try:
        # Try to find JSON array pattern
        match = re.search(r'\[.*?\]', llm_response, re.DOTALL)
        if not match:
            return None

        json_str = match.group(0)
        word_list = json.loads(json_str)

        # Validate that it's a list of strings
        if isinstance(word_list, list) and all(isinstance(w, str) for w in word_list):
            return word_list

        return None

    except (json.JSONDecodeError, AttributeError):
        return None


def expand_query_with_llm(
    query_text: str,
    generator,
    max_retries: int = MAX_RETRIES
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Expand query using LLM with retry mechanism.

    Args:
        query_text: Original query text
        generator: Transformers pipeline object
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (success: bool, expanded_query: str or None, error_msg: str or None)
    """
    for attempt in range(max_retries):
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": query_text},
            ]

            # Generate response
            outputs = generator(
                messages,
                max_new_tokens=2048,
            )

            # Extract generated text
            generated_text = outputs[0]["generated_text"][-1]["content"]

            text = generated_text
            marker = "assistantfinal"
            if marker in text:
                generated_text = text.split(marker, 1)[1].strip()
            else:
                generated_text = text

            # Parse output
            word_list = parse_llm_output(generated_text)

            if word_list is None:
                error_msg = f"Failed to parse LLM output (attempt {attempt + 1}/{max_retries})"
                if attempt < max_retries - 1:
                    continue
                return (False, None, error_msg)

            # Combine words with spaces
            expanded_query = query_text + " " + " ".join(word_list)

            return (True, expanded_query, None)

        except Exception as e:
            error_msg = f"LLM generation error: {str(e)} (attempt {attempt + 1}/{max_retries})"
            if attempt < max_retries - 1:
                continue
            return (False, query_text, error_msg)

    return (False, query_text, "Max retries exceeded")


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


def search_queries_with_llm(
    searcher: LuceneSearcher,
    topics: Dict[str, str],
    generator,
    top_k: int = 100
) -> Tuple[Dict[str, List[Tuple[str, float]]], List[Tuple[str, str, str]]]:
    """
    Perform BM25 search for all queries with LLM expansion.

    Args:
        searcher: Pyserini LuceneSearcher
        topics: Dictionary of query_id to query text
        generator: LLM pipeline
        top_k: Number of top documents to retrieve

    Returns:
        Tuple of (results dict, skipped queries list)
        - results: {query_id: [(doc_id, score), ...]}
        - skipped: [(query_id, query_text, error_msg), ...]
    """
    results = {}
    skipped_queries = []
    queries = []

    print(f"\nExpanding and searching {len(topics)} queries...")

    for query_id, original_query in tqdm(topics.items(), desc="Processing"):
        try:
            # Expand query with LLM (with retry)
            success, expanded_query, error_msg = expand_query_with_llm(
                original_query,
                generator,
                max_retries=MAX_RETRIES
            )
            queries.append(expanded_query)

            if not success:
                # Skip this query
                skipped_queries.append((query_id, original_query, error_msg))
                continue

            # Perform search with expanded query
            hits = searcher.search(expanded_query, k=top_k)

            # Store results as list of (doc_id, score) tuples
            results[query_id] = [(hit.docid, hit.score) for hit in hits]

        except Exception as e:
            # Unexpected error during search
            skipped_queries.append((query_id, original_query, f"Search error: {str(e)}"))
            continue

    return results, skipped_queries, queries


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


def save_skipped_queries(
    skipped_queries: List[Tuple[str, str, str]],
    output_file: Path,
    run_name: str,
    timestamp: str
) -> None:
    """
    Save skipped queries information to JSON file.

    Args:
        skipped_queries: List of (query_id, query_text, error_msg) tuples
        output_file: Path to output JSON file
        run_name: Name of the run
        timestamp: ISO format timestamp
    """
    skipped_data = {
        'run_name': run_name,
        'timestamp': timestamp,
        'total_skipped': len(skipped_queries),
        'max_retries_per_query': MAX_RETRIES,
        'skipped_details': [
            {
                'query_id': qid,
                'query_text': qtext,
                'error': error,
                'attempts': MAX_RETRIES
            }
            for qid, qtext, error in skipped_queries
        ]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(skipped_data, f, indent=2, ensure_ascii=False)


def save_evaluation_json(
    metrics: Dict[str, float],
    output_file: Path,
    run_name: str,
    timestamp: str,
    num_queries_total: int,
    num_queries_processed: int,
    num_queries_skipped: int,
    skipped_query_ids: List[str],
    bm25_params: Dict[str, float]
) -> None:
    """
    Save evaluation results in JSON format.

    Args:
        metrics: Dictionary of metric names to values
        output_file: Path to output JSON file
        run_name: Name of the run
        timestamp: ISO format timestamp
        num_queries_total: Total number of queries
        num_queries_processed: Number of queries successfully processed
        num_queries_skipped: Number of queries skipped
        skipped_query_ids: List of skipped query IDs
        bm25_params: BM25 parameters used
    """
    # Calculate differences from official baseline
    baseline_comparison = {}
    for metric_name, baseline_value in BASELINE_SCORE.items():
        if metric_name in metrics:
            diff = metrics[metric_name] - baseline_value
            baseline_comparison[f"{metric_name}_diff"] = diff

    # Create evaluation report
    report = {
        'run_name': run_name,
        'timestamp': timestamp,
        'model': LLM_MODEL_NAME,
        'query_expansion': 'llm_similar_words',
        'max_retries': MAX_RETRIES,
        'metrics': metrics,
        'baseline_comparison': baseline_comparison,
        'parameters': {
            'bm25_k1': bm25_params['k1'],
            'bm25_b': bm25_params['b'],
            'top_k': bm25_params['top_k'],
        },
        'num_queries_total': num_queries_total,
        'num_queries_processed': num_queries_processed,
        'num_queries_skipped': num_queries_skipped,
        'skipped_queries': skipped_query_ids,
        'official_baseline': BASELINE_SCORE,
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
            baseline_value = BASELINE_SCORE.get(metric_name, None)

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
        baseline_value = BASELINE_SCORE.get(metric_name)

        if baseline_value is not None:
            diff = value - baseline_value
            status = "✓" if diff >= 0 else "✗"
            print(f"  {status} {metric_name:12s}: {value:.4f}  (baseline: {baseline_value:.4f}, diff: {diff:+.4f})")
        else:
            print(f"    {metric_name:12s}: {value:.4f}")


def print_skipped_summary(
    num_total: int,
    num_processed: int,
    skipped_queries: List[Tuple[str, str, str]]
) -> None:
    """
    Print skipped queries summary to stdout.

    Args:
        num_total: Total number of queries
        num_processed: Number of queries processed
        skipped_queries: List of skipped query tuples
    """
    num_skipped = len(skipped_queries)

    print("\n" + "=" * 60)
    print("Skipped Queries Summary")
    print("=" * 60)
    print(f"Total queries: {num_total}")
    print(f"Processed: {num_processed} ({100*num_processed/num_total:.2f}%)")
    print(f"Skipped: {num_skipped} ({100*num_skipped/num_total:.2f}%)")

    if num_skipped > 0:
        skipped_ids = [qid for qid, _, _ in skipped_queries[:20]]  # Show first 20
        if num_skipped <= 20:
            print(f"\nSkipped query IDs: {', '.join(skipped_ids)}")
        else:
            print(f"\nSkipped query IDs (first 20): {', '.join(skipped_ids)}, ...")


def main():
    """Main function to run LLM-based query expansion, search, and evaluation."""
    print("=" * 60)
    print("GPT-OSS-20B Query Expansion with BM25 Search")
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
    run_name = f"gpt_expand_{timestamp_str}"

    print(f"\nRun name: {run_name}")
    print(f"Timestamp: {timestamp.isoformat()}")
    print(f"Model: {LLM_MODEL_NAME}")
    print(f"Prompt: {LLM_SYSTEM_PROMPT}")
    print(f"Max retries per query: {MAX_RETRIES}")

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

    # Initialize LLM
    print("\n" + "=" * 60)
    print("Initializing LLM")
    print("=" * 60)

    generator = initialize_llm()
    if generator is None:
        print("\n✗ Failed to initialize LLM. Cannot proceed.")
        print("\nMake sure you have installed:")
        print("  poetry add transformers torch accelerate")
        return 1

    # Load data
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)

    print("Loading topics...")
    topics = load_topics(topics_file)
    # Debug:
    # topics = dict(list(topics.items())[:100])
    print(f"✓ Loaded {len(topics)} queries")

    print("Loading qrels...")
    qrels = load_qrels(qrels_file)
    num_qrels = sum(len(docs) for docs in qrels.values())
    print(f"✓ Loaded {num_qrels} relevance judgments for {len(qrels)} queries")

    # Initialize searcher
    print("\n" + "=" * 60)
    print("Initializing BM25 Searcher")
    print("=" * 60)

    print(f"Loading index from {index_dir}...")
    searcher = LuceneSearcher(str(index_dir))
    searcher.set_language('ja')

    # Set BM25 parameters
    bm25_k1 = 1.2
    bm25_b = 0.75
    searcher.set_bm25(k1=bm25_k1, b=bm25_b)

    print(f"✓ Searcher initialized")
    print(f"  BM25 parameters: k1={bm25_k1}, b={bm25_b}")

    # Perform LLM expansion and search
    print("\n" + "=" * 60)
    print("Expanding Queries and Searching")
    print("=" * 60)

    top_k = 100
    results, skipped_queries, queries = search_queries_with_llm(searcher, topics, generator, top_k=top_k)

    print(f"\n✓ Search completed")
    print(f"  Retrieved top-{top_k} documents for {len(results)} queries")
    print(f"  Skipped {len(skipped_queries)} queries")
    print("RESULTS")
    print(queries)

    # Print skipped summary
    print_skipped_summary(len(topics), len(results), skipped_queries)

    # Save skipped queries
    if len(skipped_queries) > 0:
        print("\n" + "=" * 60)
        print("Saving Skipped Queries")
        print("=" * 60)

        skipped_file = results_dir / f"skipped_{run_name}.json"
        print(f"Saving skipped queries to {skipped_file}...")
        save_skipped_queries(skipped_queries, skipped_file, run_name, timestamp.isoformat())
        print(f"✓ Skipped queries saved")

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
    skipped_query_ids = [qid for qid, _, _ in skipped_queries]
    save_evaluation_json(
        metrics,
        json_file,
        run_name,
        timestamp.isoformat(),
        len(topics),
        len(results),
        len(skipped_queries),
        skipped_query_ids,
        {'k1': bm25_k1, 'b': bm25_b, 'top_k': top_k}
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
    if len(skipped_queries) > 0:
        print(f"  Skipped:      {skipped_file}")

    print("\n" + "=" * 60)
    print("GPT Query Expansion Run Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
