import csv
import sys
import ast
from pathlib import Path
from collections import defaultdict
import pandas as pd
from pyserini.search.lucene import LuceneSearcher

# ==========================================
# 1. ファイルパスの設定
# ==========================================
BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR / "indexes" / "miracl-ja-bm25"

# TRECファイル (検索結果)
FILE_BASE = BASE_DIR / "results" / "baseline_20251128_124541.trec"
FILE_ZERO = BASE_DIR / "results" / "gpt_expand_20251127_041615.trec"
FILE_COT  = BASE_DIR / "results" / "gpt_expand_20251127_223406.trec"

# ログファイル (拡張語句のテキスト) ※用意できた場合のみ指定
LOG_ZERO = BASE_DIR / "data" / "log_zero.txt"  
LOG_COT  = BASE_DIR / "data" / "log_cot.txt"

# データセット
TOPICS_FILE = BASE_DIR / "data" / "topics" / "topics.dev.tsv"
QRELS_FILE  = BASE_DIR / "data" / "qrels" / "qrels.dev.tsv"

# ==========================================
# 2. ユーティリティ関数
# ==========================================
def load_run(filepath):
    """TRECファイルを読み込む {qid: {docid: rank}}"""
    run_data = defaultdict(dict)
    if not filepath.exists():
        print(f"Warning: File not found {filepath}")
        return run_data
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            run_data[parts[0]][parts[2]] = int(parts[3])
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

def get_snippet(searcher, docid):
    """文書の先頭を取得"""
    if not searcher: return ""
    try:
        doc = searcher.doc(docid)
        if doc:
            import json
            js = json.loads(doc.raw())
            return f"【{js.get('title','')}】 {js.get('text','')}"[:120] + "..."
    except:
        pass
    return "Doc not found"

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    print("Loading data...")
    
    # クエリと正解データの読み込み
    topics = {} # {qid: text} (順序保持)
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

    # 検索結果の読み込み
    runs = {
        "Base": load_run(FILE_BASE),
        "Zero": load_run(FILE_ZERO),
        "CoT":  load_run(FILE_COT)
    }

    # ログ(拡張語句)の読み込み
    words_zero = load_log_text(LOG_ZERO, topics)
    words_cot  = load_log_text(LOG_COT, topics)

    # 検索エンジン (本文表示用)
    searcher = None
    if INDEX_DIR.exists():
        searcher = LuceneSearcher(str(INDEX_DIR))
        searcher.set_language('ja')

    print("Analyzing differences...")
    
    results = []
    
    for qid, doc_ids in qrels.items():
        query_text = topics.get(qid, "")
        
        # 最初の正解文書だけで評価（簡易化）
        target_doc = doc_ids[0]
        
        r_base = runs["Base"].get(qid, {}).get(target_doc, 1001)
        r_zero = runs["Zero"].get(qid, {}).get(target_doc, 1001)
        r_cot  = runs["CoT"].get(qid, {}).get(target_doc, 1001)

        # 判定ロジック:
        # Zero-shotでは失敗 (>15位) したが、CoTでは成功 (<=10位) したケース
        if r_zero > 15 and r_cot <= 10:
            diff = r_zero - r_cot
            results.append({
                "qid": qid,
                "query": query_text,
                "docid": target_doc,
                "ranks": (r_base, r_zero, r_cot),
                "diff": diff,
                "word_zero": words_zero.get(qid, "N/A"),
                "word_cot": words_cot.get(qid, "N/A")
            })

    # 改善度順にソート
    results.sort(key=lambda x: x['diff'], reverse=True)

    print("\n" + "="*80)
    print("🔥 CoT Wins: Zero-shot vs CoT Comparison")
    print("   (Cases where simple expansion failed, but CoT succeeded)")
    print("="*80)

    for i, res in enumerate(results[:10]): # Top 10を表示
        base, zero, cot = res['ranks']
        doc_text = get_snippet(searcher, res['docid'])
        
        print(f"\n[{i+1}] QID: {res['qid']}")
        print(f"Query:    {res['query']}")
        print(f"Ranks:    Base[{base}] -> Zero[{zero}] -> CoT[{cot}] (Diff: {res['diff']})")
        print(f"Add(Zero): {res['word_zero']}")
        print(f"Add(CoT):  {res['word_cot']}")
        print(f"Doc:      {doc_text}")
        print("-" * 50)

    # CSV出力
    if results:
        df = pd.DataFrame(results)
        # カラム名をわかりやすく整理
        df_out = pd.DataFrame({
            'QID': [r['qid'] for r in results],
            'Query': [r['query'] for r in results],
            'Rank_Base': [r['ranks'][0] for r in results],
            'Rank_Zero': [r['ranks'][1] for r in results],
            'Rank_CoT': [r['ranks'][2] for r in results],
            'Added_Zero': [r['word_zero'] for r in results],
            'Added_CoT': [r['word_cot'] for r in results],
            'Doc_ID': [r['docid'] for r in results]
        })
        out_path = BASE_DIR / "results" / "analysis_cot_effect.csv"
        df_out.to_csv(out_path, index=False)
        print(f"\nDetailed CSV saved to: {out_path}")
    else:
        print("該当するクエリが見つかりませんでした。条件（r_zero > 15など）を緩和してみてください。")

if __name__ == "__main__":
    main()
