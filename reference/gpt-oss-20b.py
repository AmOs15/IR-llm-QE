from transformers import pipeline

generator = pipeline(
  "text-generation",
  model="openai/gpt-oss-20b",
  torch_dtype="auto",
  device_map="auto"
)
# Memo: gpt-oss-20bを使用
# MXFP4 の量子化モデルを実行

policy ="""あなたは日本語における類似語を出力するシステムです。

指定する文章における各トークンにおける類似語を追加して出力してください。
最終出力形式は
["単語", "単語",,,]
の形式を必ず守ってください。
"""
# target = """["ダン" , "ダニエル", "ジャドソン", "キャラハン", "出身", "どこ"]"""
# Memo: ["ダン","Dan","ダニエル","Daniel","ジャドソン","Jason","キャラハン","Charhan","出身","出身地","どこ","どこ"]
target = "日本テレビ系列『ZIP!』の初代の司会は誰ですか？"
# Memo: ["日本テレビ系列","日本テレビ","ZIP!","ZIP","初代","初期","司会","ホスト","誰","何人","です","です","か","か"]'
messages = [
    {"role": "system", "content": policy},
    {"role": "user", "content": target},
]

outputs = generator(
  messages, 
  max_new_tokens=2048, 
  # temperature=0.7
)
print(outputs)
print()
print(outputs[0]["generated_text"][-1])