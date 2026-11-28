# ファイル要約エージェント

「README.md を要約して」「このフォルダには何がある？」と聞くと、Claudeが自分でファイルを探して読み、要約するAIエージェントのサンプルです。

## 仕組み

```
「このフォルダのメモを要約して」
  → Claudeが判断「まず一覧を見よう」   → list_files(".") を実行
  → Claudeが判断「memo.txt を読もう」  → read_file("memo.txt", 1) を実行
  → (長いファイルなら続きを読む)       → read_file("memo.txt", 501) を実行
  → Claudeが判断「もう要約できる」     → 要約を回答
```

- `list_files` / `read_file` … ふつうのPython関数（ツール）
- `TOOLS` … ツールの説明書。Claudeはこれを読んで使うツールを選ぶ
- `ask()` … ツールを実行してClaudeに結果を返すループ

安全のため、読めるのは起動時に指定したフォルダの中だけです。テキストファイル（.txt / .md / .py / .csv など）に対応しています。

## 使い方

1行ずつ実行してください。

```bash
cd file_summary_agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-ここに本物のキー"
```

```bash
python file_summary_agent.py ../weather_agent "weather_agent.py を要約して"
python file_summary_agent.py ~/Documents
```

1つ目の引数が対象フォルダ、2つ目が質問です。質問を省くと対話モードになります（exit で終了）。
`python` が見つからない場合は `python3` に変えてください。
