import csv
import sys
import ast
from pathlib import Path
from collections import defaultdict
import pandas as pd
from pyserini.search.lucene import LuceneSearcher

# ==========================================
# 設定: ファイルパス
# ==========================================
BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR / "indexes" / "miracl-ja-bm25"

# 分析対象のTRECファイル
FILE_BASELINE = BASE_DIR / "results" / "baseline_20251128_124541.trec"
FILE_COT = BASE_DIR / "results" / "gpt_expand_20251127_223406.trec"

# データファイル
TOPICS_FILE = BASE_DIR / "data" / "topics" / "topics.dev.tsv"
QRELS_FILE = BASE_DIR / "data" / "qrels" / "qrels.dev.tsv"

# ★ 作成したログファイルのパスを指定してください
LOG_FILE = BASE_DIR / "data" / "log_cot.txt" 

# ==========================================
# 関数群
# ==========================================

def load_qrels(filepath):
    qrels = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 4: continue
            qid, _, docid, rel = row
            if int(rel) > 0:
                qrels[qid].append(docid)
    return qrels

def load_topics(filepath):
    """
    Topicsを順序保持辞書として読み込む
    （ログのリスト順序と一致させるため）
    """
    topics = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None) # header
        for row in reader:
            if len(row) < 2: continue
            topics[row[0]] = row[1]
    return topics

def load_run(filepath):
    run_data = defaultdict(dict)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            qid = parts[0]
            docid = parts[2]
            rank = int(parts[3])
            run_data[qid][docid] = rank
    return run_data

def load_log_text(log_path, topics):
    """ログファイルから拡張語句を抽出する"""
    expanded_map = {}
    if not log_path.exists():
        return expanded_map

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 念のため改行などを除去してパース
            if "[" in content and "]" in content:
                start = content.find("[")
                end = content.rfind("]") + 1
                content = content[start:end]
                expanded_list = ast.literal_eval(content)
            else:
                return {}
    except Exception as e:
        print(f"Log parse error ({log_path.name}): {e}")
        return {}

    topic_ids = list(topics.keys())
    for i, full_text in enumerate(expanded_list):
        if i < len(topic_ids):
            qid = topic_ids[i]
            original = topics[qid]

            # --- 修正箇所: Noneチェックを追加 ---
            if full_text is None:
                expanded_map[qid] = "N/A (Skipped)"
                continue
            
            if not isinstance(full_text, str):
                # 文字列以外が入っていた場合の安全策
                full_text = str(full_text)
            # -----------------------------------

            # 元クエリを除去して「追加部分」だけ抽出
            if full_text.startswith(original):
                added = full_text[len(original):].strip()
            else:
                added = full_text # マッチしない場合はそのまま
            expanded_map[qid] = added
            
    return expanded_map

def get_doc_content(searcher, docid):
    if searcher is None: return ""
    try:
        doc = searcher.doc(docid)
        if doc:
            import json
            content = json.loads(doc.raw())
            return f"【{content.get('title', '')}】 {content.get('text', '')}"
    except:
        pass
    return ""

# ==========================================
# Main
# ==========================================
def main():
    print("Loading Data...")
    qrels = load_qrels(QRELS_FILE)
    topics = load_topics(TOPICS_FILE)
    
    # 拡張クエリの読み込み
    expanded_map = load_log_text(LOG_FILE, topics)
    print(f"Loaded {len(expanded_map)} expanded queries from log.")

    run_base = load_run(FILE_BASELINE)
    run_cot = load_run(FILE_COT)

    searcher = None
    if INDEX_DIR.exists():
        searcher = LuceneSearcher(str(INDEX_DIR))
        searcher.set_language('ja')

    print("\nAnalyzing Improvements...")
    
    # 分析データの作成
    improvements = []
    
    for qid, doc_ids in qrels.items():
        original_query = topics.get(qid, "")
        added_words = expanded_map.get(qid, "N/A")
        
        for docid in doc_ids:
            rank_base = run_base.get(qid, {}).get(docid, 1001)
            rank_cot = run_cot.get(qid, {}).get(docid, 1001)
            
            # Baselineが圏外(>20位) または 低い順位から、CoTでTop10に入った
            if rank_base > 20 and rank_cot <= 10:
                diff = rank_base - rank_cot
                improvements.append({
                    "qid": qid,
                    "query": original_query,
                    "added_words": added_words, # ★ここが重要
                    "rank_base": rank_base,
                    "rank_cot": rank_cot,
                    "diff": diff,
                    "docid": docid
                })

    # 改善度順にソート
    improvements.sort(key=lambda x: x['diff'], reverse=True)

    # 結果表示
    print("\n" + "="*80)
    print("🏆 Top 10 IMPACTFUL Improvements (なぜ上がったか？)")
    print("="*80)

    for i, item in enumerate(improvements[:10]):
        doc_text = get_doc_content(searcher, item['docid'])
        # 読みやすくトリミング
        doc_snippet = doc_text[:150] + "..." if len(doc_text) > 150 else doc_text
        
        print(f"\n[{i+1}] QID: {item['qid']}")
        print(f"Original: {item['query']}")
        print(f"★ Added:  {item['added_words']}") # これが見たかった情報！
        print(f"Rank:     {item['rank_base']} -> {item['rank_cot']} (Has improved by {item['diff']} ranks)")
        print(f"Doc:      {doc_snippet}")
        print("-" * 50)

    # CSV保存
    df = pd.DataFrame(improvements)
    output_path = BASE_DIR / "results" / "analysis_qualitative.csv"
    df.to_csv(output_path, index=False)
    print(f"\nAnalysis saved to {output_path}")

if __name__ == "__main__":
    main()
