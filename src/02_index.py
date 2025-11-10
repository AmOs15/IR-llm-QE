#!/usr/bin/env python3
"""
BM25 Index Builder for MIRACL Japanese Corpus

Creates a Lucene-based BM25 index using Pyserini for the MIRACL Japanese corpus.
The index uses Japanese language analyzer and standard BM25 parameters (k1=1.2, b=0.75).

Requirements:
- Java JDK 11+ must be installed and JAVA_HOME must be set
- Corpus data must be downloaded first (run 01_download.py)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


def check_java() -> bool:
    """
    Check if Java is installed and accessible.

    Returns:
        True if Java is available, False otherwise
    """
    try:
        result = subprocess.run(
            ['java', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Java version output goes to stderr
        version_output = result.stderr
        print(f"✓ Java detected: {version_output.splitlines()[0]}")
        return True
    except FileNotFoundError:
        print("✗ Java not found")
        return False
    except Exception as e:
        print(f"✗ Error checking Java: {e}")
        return False


def check_corpus(corpus_dir: Path) -> bool:
    """
    Check if corpus data exists.

    Args:
        corpus_dir: Directory containing corpus data

    Returns:
        True if corpus exists, False otherwise
    """
    corpus_file = corpus_dir / "corpus.jsonl"

    if not corpus_file.exists():
        print(f"✗ Corpus not found at {corpus_file}")
        print("\nPlease run the download script first:")
        print("  poetry run python src/01_download.py")
        return False

    # Count documents
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            doc_count = sum(1 for _ in f)
        print(f"✓ Corpus found: {doc_count:,} documents")
        return True
    except Exception as e:
        print(f"✗ Error reading corpus: {e}")
        return False


def check_existing_index(index_dir: Path) -> bool:
    """
    Check if index already exists.

    Args:
        index_dir: Directory where index should be stored

    Returns:
        True if index exists, False otherwise
    """
    # Check for Lucene index files
    if index_dir.exists() and any(index_dir.iterdir()):
        print(f"✓ Index already exists at {index_dir}")
        return True

    return False


def get_index_stats(index_dir: Path) -> dict:
    """
    Get statistics about the index using Pyserini's IndexReader.

    Args:
        index_dir: Directory containing the index

    Returns:
        Dictionary with index statistics
    """
    try:
        # from pyserini.index import IndexReader
        from pyserini.index.lucene import IndexReader


        reader = IndexReader(str(index_dir))

        stats = {
            'num_docs': reader.stats()['documents'],
            'total_terms': reader.stats()['total_terms'],
            'unique_terms': reader.stats()['unique_terms'],
        }

        return stats
    except Exception as e:
        print(f"⚠ Could not read index stats: {e}")
        return {}


def convert_corpus_to_pyserini_format(input_file: Path, output_file: Path) -> int:
    """
    Convert MIRACL corpus format to Pyserini-compatible format.

    MIRACL format: {'id': ..., 'title': ..., 'text': ...}
    Pyserini format: {'id': ..., 'contents': ...}

    Args:
        input_file: Path to original corpus.jsonl
        output_file: Path to output corpus_pyserini.jsonl

    Returns:
        Number of documents converted
    """
    print(f"\nConverting corpus to Pyserini format...")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")

    doc_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="Converting documents", unit="docs"):
            try:
                doc = json.loads(line)

                # Extract fields
                doc_id = doc.get('id', '')
                title = doc.get('title', '')
                text = doc.get('text', '')

                # Combine title and text into contents
                # If title exists, prepend it to the text
                if title:
                    contents = f"{title}\n{text}"
                else:
                    contents = text

                # Create Pyserini-compatible document
                pyserini_doc = {
                    'id': doc_id,
                    'contents': contents
                }

                f_out.write(json.dumps(pyserini_doc, ensure_ascii=False) + '\n')
                doc_count += 1

            except json.JSONDecodeError as e:
                print(f"\n⚠ Error parsing JSON line: {e}")
                continue

    print(f"✓ Converted {doc_count:,} documents")
    return doc_count


def print_index_stats(stats: dict) -> None:
    """
    Print index statistics in a formatted way.

    Args:
        stats: Dictionary containing index statistics
    """
    if not stats:
        return

    print("\nIndex Statistics:")
    print(f"  Documents:    {stats.get('num_docs', 0):,}")
    print(f"  Total terms:  {stats.get('total_terms', 0):,}")
    print(f"  Unique terms: {stats.get('unique_terms', 0):,}")


def create_index(corpus_dir: Path, index_dir: Path, threads: int = 4) -> bool:
    """
    Create BM25 index using Pyserini.

    Args:
        corpus_dir: Directory containing corpus JSONL file
        index_dir: Directory where index will be created
        threads: Number of threads for indexing (default: 4)

    Returns:
        True if indexing succeeded, False otherwise
    """
    print(f"\nCreating BM25 index...")
    print(f"  Input:   {corpus_dir}")
    print(f"  Output:  {index_dir}")
    print(f"  Threads: {threads}")
    print(f"  Language: Japanese (ja)")

    # Ensure index directory exists
    index_dir.mkdir(parents=True, exist_ok=True)

    # Build Pyserini indexing command
    cmd = [
        sys.executable,  # Use current Python interpreter
        '-m', 'pyserini.index.lucene',
        '--collection', 'JsonCollection',
        '--input', str(corpus_dir),
        '--index', str(index_dir),
        '--generator', 'DefaultLuceneDocumentGenerator',
        '--threads', str(threads),
        '--language', 'ja',  # Japanese analyzer
        '--storePositions',
        '--storeDocvectors',
        '--storeRaw',
    ]

    print("\nRunning indexer...")
    print(f"Command: {' '.join(cmd)}")
    print("\nThis may take 30-60 minutes depending on your system...")
    print("-" * 60)

    try:
        # Run indexing command
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            bufsize=1,  # Line buffered
        )

        print("-" * 60)
        print("✓ Indexing completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print(f"✗ Indexing failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print("-" * 60)
        print(f"✗ Error during indexing: {e}")
        return False


def main():
    """Main function to create BM25 index."""
    print("=" * 60)
    print("BM25 Index Builder for MIRACL Japanese")
    print("=" * 60)

    # Define paths
    base_dir = Path(__file__).parent.parent
    corpus_dir = base_dir / "data" / "corpus"
    index_dir = base_dir / "indexes" / "miracl-ja-bm25"

    print(f"\nBase directory: {base_dir}")
    print(f"Corpus directory: {corpus_dir}")
    print(f"Index directory: {index_dir}")

    # Check prerequisites
    print("\n" + "=" * 60)
    print("Checking Prerequisites")
    print("=" * 60)

    # Check Java
    if not check_java():
        print("\n" + "=" * 60)
        print("Setup Required")
        print("=" * 60)
        print("\nJava JDK 11+ is required for Pyserini.")
        print("\nInstallation instructions:")
        print("  macOS:")
        print("    brew install openjdk@11")
        print("    export JAVA_HOME=$(/usr/libexec/java_home -v 11)")
        print("\n  Ubuntu/Debian:")
        print("    sudo apt-get install openjdk-11-jdk")
        print("    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64")
        return 1

    # Check corpus
    if not check_corpus(corpus_dir):
        return 1

    # Convert corpus to Pyserini format
    print("\n" + "=" * 60)
    print("Converting Corpus Format")
    print("=" * 60)

    original_corpus = corpus_dir / "corpus.jsonl"
    pyserini_corpus_dir = base_dir / "data" / "corpus_pyserini"
    pyserini_corpus_file = pyserini_corpus_dir / "corpus.jsonl"

    # Create directory for Pyserini corpus
    pyserini_corpus_dir.mkdir(parents=True, exist_ok=True)

    # Check if converted corpus already exists
    if pyserini_corpus_file.exists():
        print(f"✓ Pyserini-format corpus already exists at {pyserini_corpus_file}")
        with open(pyserini_corpus_file, 'r', encoding='utf-8') as f:
            pyserini_doc_count = sum(1 for _ in f)
        print(f"  Found {pyserini_doc_count:,} documents")
    else:
        # Convert corpus
        doc_count = convert_corpus_to_pyserini_format(original_corpus, pyserini_corpus_file)

        if doc_count == 0:
            print("✗ Failed to convert corpus")
            return 1

    # Check for existing index
    print("\n" + "=" * 60)
    print("Checking Existing Index")
    print("=" * 60)

    if check_existing_index(index_dir):
        print("\nIndex already exists.")

        # Get and display stats
        stats = get_index_stats(index_dir)
        print_index_stats(stats)

        # If index has 0 documents, automatically delete it
        if stats.get('num_docs', 0) == 0:
            print("\n⚠ Index has 0 documents - automatically deleting and recreating...")
            import shutil
            shutil.rmtree(index_dir)
            print("✓ Empty index deleted")
        else:
            print("\nOptions:")
            print("  1. Use existing index (recommended)")
            print("  2. Delete and recreate index")

            response = input("\nDo you want to recreate the index? (y/N): ").strip().lower()

            if response not in ['y', 'yes']:
                print("\n✓ Using existing index")
                print("\nNext steps:")
                print("  1. Run baseline search and evaluation: poetry run python src/03_baseline.py")
                return 0

            print("\nDeleting existing index...")
            import shutil
            shutil.rmtree(index_dir)
            print("✓ Existing index deleted")

    # Create index
    print("\n" + "=" * 60)
    print("Creating Index")
    print("=" * 60)

    success = create_index(pyserini_corpus_dir, index_dir, threads=4)

    if not success:
        print("\n" + "=" * 60)
        print("Indexing Failed")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("  1. Check JAVA_HOME is set correctly")
        print("  2. Ensure enough disk space (~15GB)")
        print("  3. Check corpus file format (JSONL)")
        print("  4. Try with fewer threads: --threads 1")
        return 1

    # Display statistics
    print("\n" + "=" * 60)
    print("Index Created Successfully")
    print("=" * 60)

    stats = get_index_stats(index_dir)
    print_index_stats(stats)

    # Verify document count
    if stats.get('num_docs', 0) > 0:
        expected_docs = 6_953_614
        actual_docs = stats['num_docs']

        if abs(expected_docs - actual_docs) / expected_docs < 0.01:
            print(f"\n✓ Document count verified: {actual_docs:,} (expected: {expected_docs:,})")
        else:
            print(f"\n⚠ Document count mismatch: {actual_docs:,} (expected: {expected_docs:,})")

    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("\nYou can now:")
    print("  1. Run baseline search and evaluation: poetry run python src/03_baseline.py")

    print("\nBM25 Parameters:")
    print("  k1 = 1.2 (term frequency saturation)")
    print("  b  = 0.75 (document length normalization)")

    return 0


if __name__ == "__main__":
    exit(main())
