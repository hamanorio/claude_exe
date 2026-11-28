# 天気エージェント

「明日の大阪は傘いる？」のように聞くと、Claudeが自分で天気APIを調べて答えるAIエージェントのサンプルです。

## 仕組み

```
質問 → Claudeが判断「まず場所を調べよう」 → search_location("大阪") を実行
     → Claudeが判断「次は天気を調べよう」 → get_weather(34.69, 135.50, 2) を実行
     → Claudeが判断「もう答えられる」     → 回答
```

- `search_location` / `get_weather` / `suggest_outfit` … ふつうのPython関数（ツール）
  - `suggest_outfit` は気温・降水確率・風速から服装と持ち物を提案します（「明日何着ていけばいい？」など）
- `TOOLS` … ツールの説明書。Claudeはこれを読んで使うツールを選ぶ
- `ask()` … ツールを実行してClaudeに結果を返すループ

天気データは [Open-Meteo](https://open-meteo.com/)（無料・APIキー不要）を使っています。

## 使い方

```bash
cd weather_agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY="あなたのAPIキー"   # https://console.anthropic.com/ で取得

python weather_agent.py "東京の今の天気は？"   # 1回だけ質問
python weather_agent.py                         # 対話モード（exit で終了）
```

## 改造のヒント

- ツールを追加する：関数を書き、`TOOL_FUNCTIONS` と `TOOLS` に登録するだけ
  （例：週間の気温グラフ作成など）
- 口調や回答の形式を変える：`SYSTEM_PROMPT` を書き換える
