#!/usr/bin/env python3
"""
MIRACL Japanese Dataset Downloader

Downloads MIRACL Japanese corpus and dev dataset (queries and qrels)
and saves them locally for BM25 indexing and evaluation.

Dataset details:
- Corpus: 6,953,614 passages from Japanese Wikipedia
- Dev queries: 860 queries
- Dev qrels: 8,354 relevance judgments
"""

import json
import os
from pathlib import Path
from typing import Iterator

import ir_datasets
from tqdm import tqdm


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_corpus(dataset, output_dir: Path) -> int:
    """
    Save corpus documents in JSONL format for Pyserini indexing.

    Each document is saved as a JSON object with 'id', 'title', and 'text' fields.

    Args:
        dataset: ir_datasets dataset object
        output_dir: Directory to save corpus files

    Returns:
        Number of documents saved
    """
    ensure_dir(output_dir)
    corpus_file = output_dir / "corpus.jsonl"

    # Check if corpus already exists
    if corpus_file.exists():
        print(f"✓ Corpus already exists at {corpus_file}")
        # Count existing documents
        with open(corpus_file, 'r', encoding='utf-8') as f:
            doc_count = sum(1 for _ in f)
        print(f"  Found {doc_count:,} documents")
        return doc_count

    print(f"Downloading and saving corpus to {corpus_file}...")
    doc_count = 0

    with open(corpus_file, 'w', encoding='utf-8') as f:
        for doc in tqdm(dataset.docs_iter(), desc="Processing documents", unit="docs"):
            # Convert to Pyserini-compatible JSON format
            doc_json = {
                'id': doc.doc_id,
                'title': doc.title if hasattr(doc, 'title') else '',
                'text': doc.text
            }
            f.write(json.dumps(doc_json, ensure_ascii=False) + '\n')
            doc_count += 1

    print(f"✓ Saved {doc_count:,} documents")
    return doc_count


def save_topics(dataset, output_dir: Path) -> int:
    """
    Save dev queries to TSV file.

    Args:
        dataset: ir_datasets dataset object
        output_dir: Directory to save topics file

    Returns:
        Number of queries saved
    """
    ensure_dir(output_dir)
    topics_file = output_dir / "topics.dev.tsv"

    # Check if topics already exist
    if topics_file.exists():
        print(f"✓ Topics already exist at {topics_file}")
        with open(topics_file, 'r', encoding='utf-8') as f:
            query_count = sum(1 for _ in f) - 1  # Subtract header
        print(f"  Found {query_count:,} queries")
        return query_count

    print(f"Downloading and saving topics to {topics_file}...")
    query_count = 0

    with open(topics_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("query_id\tquery\n")

        for query in tqdm(dataset.queries_iter(), desc="Processing queries", unit="queries"):
            f.write(f"{query.query_id}\t{query.text}\n")
            query_count += 1

    print(f"✓ Saved {query_count:,} queries")
    return query_count


def save_qrels(dataset, output_dir: Path) -> int:
    """
    Save dev qrels (relevance judgments) to TSV file.

    Args:
        dataset: ir_datasets dataset object
        output_dir: Directory to save qrels file

    Returns:
        Number of qrels saved
    """
    ensure_dir(output_dir)
    qrels_file = output_dir / "qrels.dev.tsv"

    # Check if qrels already exist
    if qrels_file.exists():
        print(f"✓ Qrels already exist at {qrels_file}")
        with open(qrels_file, 'r', encoding='utf-8') as f:
            qrel_count = sum(1 for _ in f)
        print(f"  Found {qrel_count:,} relevance judgments")
        return qrel_count

    print(f"Downloading and saving qrels to {qrels_file}...")
    qrel_count = 0

    with open(qrels_file, 'w', encoding='utf-8') as f:
        for qrel in tqdm(dataset.qrels_iter(), desc="Processing qrels", unit="qrels"):
            # Format: query_id iteration doc_id relevance
            f.write(f"{qrel.query_id}\t0\t{qrel.doc_id}\t{qrel.relevance}\n")
            qrel_count += 1

    print(f"✓ Saved {qrel_count:,} relevance judgments")
    return qrel_count


def verify_dataset(doc_count: int, query_count: int, qrel_count: int) -> bool:
    """
    Verify that downloaded dataset matches expected counts.

    Expected values from MIRACL paper:
    - Documents: 6,953,614
    - Dev queries: 860
    - Dev qrels: 8,354

    Args:
        doc_count: Number of documents downloaded
        query_count: Number of queries downloaded
        qrel_count: Number of qrels downloaded

    Returns:
        True if counts match expected values (with tolerance)
    """
    print("\nVerifying dataset integrity...")

    expected = {
        'documents': 6_953_614,
        'queries': 860,
        'qrels': 8_354
    }

    actual = {
        'documents': doc_count,
        'queries': query_count,
        'qrels': qrel_count
    }

    all_match = True
    for key in expected:
        exp = expected[key]
        act = actual[key]
        match = abs(exp - act) / exp < 0.01  # Allow 1% tolerance
        status = "✓" if match else "✗"
        print(f"  {status} {key.capitalize()}: {act:,} (expected: {exp:,})")

        if not match:
            all_match = False

    return all_match


def main():
    """Main function to download MIRACL Japanese dataset."""
    print("=" * 60)
    print("MIRACL Japanese Dataset Downloader")
    print("=" * 60)

    # Define paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    corpus_dir = data_dir / "corpus"
    topics_dir = data_dir / "topics"
    qrels_dir = data_dir / "qrels"

    print(f"\nBase directory: {base_dir}")
    print(f"Data directory: {data_dir}")

    # Load MIRACL Japanese dev dataset
    print("\nLoading MIRACL Japanese dev dataset...")
    try:
        dataset = ir_datasets.load("miracl/ja/dev")
        print("✓ Dataset loaded successfully")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure ir-datasets is installed: pip install ir-datasets")
        print("2. Check your internet connection")
        print("3. Try running: python -c 'import ir_datasets; print(ir_datasets.__version__)'")
        return 1

    # Download and save data
    print("\n" + "=" * 60)
    print("Step 1: Downloading Corpus")
    print("=" * 60)
    doc_count = save_corpus(dataset, corpus_dir)

    print("\n" + "=" * 60)
    print("Step 2: Downloading Topics (Queries)")
    print("=" * 60)
    query_count = save_topics(dataset, topics_dir)

    print("\n" + "=" * 60)
    print("Step 3: Downloading Qrels (Relevance Judgments)")
    print("=" * 60)
    qrel_count = save_qrels(dataset, qrels_dir)

    # Verify dataset
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    is_valid = verify_dataset(doc_count, query_count, qrel_count)

    # Print summary
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"\nData saved to:")
    print(f"  Corpus:  {corpus_dir / 'corpus.jsonl'}")
    print(f"  Topics:  {topics_dir / 'topics.dev.tsv'}")
    print(f"  Qrels:   {qrels_dir / 'qrels.dev.tsv'}")

    if is_valid:
        print("\n✓ Dataset integrity verified")
        print("\nNext steps:")
        print("  1. Create BM25 index: python src/02_index.py")
        print("  2. Run searches: python src/03_search.py")
        print("  3. Evaluate results: python src/04_evaluate.py")
    else:
        print("\n⚠ Warning: Dataset counts don't match expected values")
        print("  This might indicate an incomplete download or dataset update")

    return 0


if __name__ == "__main__":
    exit(main())
