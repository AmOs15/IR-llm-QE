import csv
import sys
import ast
from pathlib import Path
from collections import defaultdict
import pandas as pd
from pyserini.search.lucene import LuceneSearcher

# ==========================================
# 1. ファイルパス設定
# ==========================================
BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR / "indexes" / "miracl-ja-bm25"

# Run Files
FILE_BASE = BASE_DIR / "results" / "baseline_20251128_124541.trec"
FILE_ZERO = BASE_DIR / "results" / "gpt_expand_20251127_041615.trec"
FILE_COT  = BASE_DIR / "results" / "gpt_expand_20251127_223406.trec"

# Log Files (Lists of strings)
LOG_ZERO = BASE_DIR / "data" / "log_zero.txt"
LOG_COT  = BASE_DIR / "data" / "log_cot.txt"

# Data
TOPICS_FILE = BASE_DIR / "data" / "topics" / "topics.dev.tsv"
QRELS_FILE  = BASE_DIR / "data" / "qrels" / "qrels.dev.tsv"

# ==========================================
# 2. ユーティリティ関数
# ==========================================
def load_run(filepath):
    run_data = defaultdict(dict)
    if not filepath.exists(): return run_data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            run_data[parts[0]][parts[2]] = int(parts[3])
    return run_data

def load_log_text(log_path, topics):
    """ログから拡張語句を抽出 (None対策済み)"""
    expanded_map = {}
    if not log_path.exists(): return expanded_map
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if "[" in content and "]" in content:
                start = content.find("[")
                end = content.rfind("]") + 1
                content = content[start:end]
                expanded_list = ast.literal_eval(content)
            else:
                return {}
    except Exception as e:
        print(f"Log parse error: {e}")
        return {}

    topic_ids = list(topics.keys())
    for i, full_text in enumerate(expanded_list):
        if i < len(topic_ids):
            qid = topic_ids[i]
            original = topics[qid]
            
            if full_text is None:
                expanded_map[qid] = "N/A (Skipped)"
                continue
            
            if not isinstance(full_text, str): full_text = str(full_text)

            if full_text.startswith(original):
                added = full_text[len(original):].strip()
            else:
                added = full_text
            expanded_map[qid] = added
    return expanded_map

def get_snippet(searcher, docid):
    if not searcher: return ""
    try:
        doc = searcher.doc(docid)
        if doc:
            import json
            js = json.loads(doc.raw())
            return f"【{js.get('title','')}】 {js.get('text','')}"[:100] + "..."
    except: pass
    return "Doc not found"

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    print("Loading data...")
    topics = {}
    with open(TOPICS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            if len(row) >= 2: topics[row[0]] = row[1]

    qrels = defaultdict(list)
    with open(QRELS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 4 and int(row[3]) > 0:
                qrels[row[0]].append(row[2])

    runs = {
        "Base": load_run(FILE_BASE),
        "Zero": load_run(FILE_ZERO),
        "CoT":  load_run(FILE_COT)
    }
    
    words_zero = load_log_text(LOG_ZERO, topics)
    words_cot  = load_log_text(LOG_COT, topics)

    searcher = None
    if INDEX_DIR.exists():
        searcher = LuceneSearcher(str(INDEX_DIR))
        searcher.set_language('ja')

    print("Analyzing failures...")
    
    degradations_cot = [] # CoTで悪化したケース
    hard_queries = []     # 全員ダメだったケース

    for qid, doc_ids in qrels.items():
        query_text = topics.get(qid, "")
        target_doc = doc_ids[0]
        
        r_base = runs["Base"].get(qid, {}).get(target_doc, 1001)
        r_zero = runs["Zero"].get(qid, {}).get(target_doc, 1001)
        r_cot  = runs["CoT"].get(qid, {}).get(target_doc, 1001)

        # パターン1: CoTによる改悪 (Degradation)
        # Baseはそこそこ良い(<=20位)のに、CoTで大きく順位を落とした(>20位)
        if r_base <= 20 and r_cot > 20:
            diff = r_cot - r_base # 正の値が大きいほど悪化
            degradations_cot.append({
                "qid": qid,
                "type": "Degradation (CoT)",
                "query": query_text,
                "docid": target_doc,
                "ranks": (r_base, r_zero, r_cot),
                "diff": diff,
                "word_zero": words_zero.get(qid, "N/A"),
                "word_cot": words_cot.get(qid, "N/A")
            })

        # パターン2: 難問 (Hard Queries)
        # どの手法でも 50位以下 (改善の余地ありだが難しい)
        if r_base > 50 and r_zero > 50 and r_cot > 50:
             hard_queries.append({
                "qid": qid,
                "type": "Hard Query",
                "query": query_text,
                "docid": target_doc,
                "ranks": (r_base, r_zero, r_cot),
                "diff": 0,
                "word_zero": words_zero.get(qid, "N/A"),
                "word_cot": words_cot.get(qid, "N/A")
            })

    # ソート
    degradations_cot.sort(key=lambda x: x['diff'], reverse=True) # 悪化幅が大きい順

    # --- 結果表示: CoTによる改悪 ---
    print("\n" + "="*80)
    print("📉 Top CoT Degradations (Base was OK, but CoT failed)")
    print("   Analysis Hint: Look for 'Topic Drift' or 'Noisy Expansion'")
    print("="*80)

    for i, res in enumerate(degradations_cot[:10]):
        base, zero, cot = res['ranks']
        doc_text = get_snippet(searcher, res['docid'])
        print(f"\n[{i+1}] QID: {res['qid']}")
        print(f"Query:    {res['query']}")
        print(f"Ranks:    Base[{base}] -> CoT[{cot}] (Dropped by {res['diff']})")
        print(f"Added(CoT): {res['word_cot']}")
        print(f"Correct Doc: {doc_text}")
        print("-" * 50)

    # --- 結果表示: 難問 ---
    print("\n" + "="*80)
    print("🤔 Hard Queries (All methods failed > 50)")
    print("   Analysis Hint: Is the answer in the index? Is vocabulary completely different?")
    print("="*80)
    
    for i, res in enumerate(hard_queries[:5]):
        base, zero, cot = res['ranks']
        doc_text = get_snippet(searcher, res['docid'])
        print(f"\n[{i+1}] QID: {res['qid']}")
        print(f"Query:    {res['query']}")
        print(f"Ranks:    Base[{base}] -> Zero[{zero}] -> CoT[{cot}]")
        print(f"Added(CoT): {res['word_cot']}")
        print(f"Correct Doc: {doc_text}")
        print("-" * 50)

    # CSV保存 (全結合)
    all_rows = degradations_cot + hard_queries
    if all_rows:
        df = pd.DataFrame(all_rows)
        # 列整理
        df_out = pd.DataFrame({
            'Type': [r['type'] for r in all_rows],
            'QID': [r['qid'] for r in all_rows],
            'Query': [r['query'] for r in all_rows],
            'Rank_Base': [r['ranks'][0] for r in all_rows],
            'Rank_Zero': [r['ranks'][1] for r in all_rows],
            'Rank_CoT': [r['ranks'][2] for r in all_rows],
            'Added_Zero': [r['word_zero'] for r in all_rows],
            'Added_CoT': [r['word_cot'] for r in all_rows],
            'Doc_ID': [r['docid'] for r in all_rows]
        })
        out_path = BASE_DIR / "results" / "analysis_failures.csv"
        df_out.to_csv(out_path, index=False)
        print(f"\nFailure analysis saved to: {out_path}")

if __name__ == "__main__":
    main()
