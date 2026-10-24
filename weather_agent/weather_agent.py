"""天気を調べて答えるAIエージェント

仕組み:
  1. ツール(関数)を用意する      … search_location / get_weather
  2. LLM(Claude)がどのツールを呼ぶか判断する
  3. プログラムがツールを実行し、結果をClaudeに返す
  4. Claudeが「もう答えられる」と判断するまで 2〜3 を繰り返す

天気データは Open-Meteo (https://open-meteo.com/) を使用。APIキー不要。
"""

import json
import sys
import urllib.parse
import urllib.request

import anthropic

MODEL = "claude-opus-5-5"

# WMO天気コード → 日本語
WEATHER_CODES = {
    0: "快晴", 1: "晴れ", 2: "一部曇り", 3: "曇り",
    45: "霧", 48: "霧(着氷性)",
    51: "弱い霧雨", 53: "霧雨", 55: "強い霧雨",
    61: "弱い雨", 63: "雨", 65: "強い雨",
    66: "弱い着氷性の雨", 67: "強い着氷性の雨",
    71: "弱い雪", 73: "雪", 75: "強い雪", 77: "霧雪",
    80: "弱いにわか雨", 81: "にわか雨", 82: "激しいにわか雨",
    85: "弱いにわか雪", 86: "強いにわか雪",
    95: "雷雨", 96: "雹を伴う雷雨", 99: "強い雹を伴う雷雨",
}


# ---------------------------------------------------------------------------
# 1. ツール(関数)本体
# ---------------------------------------------------------------------------

def _fetch_json(url: str, params: dict) -> dict:
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=15) as res:
        return json.load(res)


def search_location(name: str) -> dict:
    """地名から緯度・経度を調べる"""
    data = _fetch_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": name, "count": 3, "language": "ja"},
    )
    results = data.get("results")
    if not results:
        raise ValueError(f"「{name}」という地名が見つかりませんでした")
    return {
        "candidates": [
            {
                "name": r["name"],
                "country": r.get("country"),
                "region": r.get("admin1"),
                "latitude": r["latitude"],
                "longitude": r["longitude"],
            }
            for r in results
        ]
    }


def get_weather(latitude: float, longitude: float, days: int = 3) -> dict:
    """緯度・経度から現在の天気と数日分の予報を取得する"""
    days = max(1, min(int(days), 7))
    data = _fetch_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max",
            "timezone": "auto",
            "forecast_days": days,
        },
    )
    current = data["current"]
    daily = data["daily"]
    return {
        "current": {
            "time": current["time"],
            "weather": WEATHER_CODES.get(current["weather_code"], "不明"),
            "temperature_c": current["temperature_2m"],
            "humidity_percent": current["relative_humidity_2m"],
            "wind_speed_kmh": current["wind_speed_10m"],
        },
        "forecast": [
            {
                "date": daily["time"][i],
                "weather": WEATHER_CODES.get(daily["weather_code"][i], "不明"),
                "max_c": daily["temperature_2m_max"][i],
                "min_c": daily["temperature_2m_min"][i],
                "precipitation_probability_percent": daily["precipitation_probability_max"][i],
                "max_wind_speed_kmh": daily["wind_speed_10m_max"][i],
            }
            for i in range(len(daily["time"]))
        ],
    }


def suggest_outfit(max_c: float, min_c: float, precipitation_probability_percent: float,
                   wind_speed_kmh: float) -> dict:
    """気温・降水確率・風速から服装と持ち物を提案する"""
    # 1日の平均的な体感で服装を決め、朝晩の冷え込みは羽織りもので補う
    avg = (max_c + min_c) / 2
    if wind_speed_kmh >= 30:
        avg -= 2  # 強い風は体感温度を下げる

    if avg >= 25:
        clothes = ["半袖", "通気性のよい薄手の服"]
    elif avg >= 20:
        clothes = ["半袖または薄手の長袖"]
    elif avg >= 16:
        clothes = ["長袖シャツ", "薄手のカーディガン"]
    elif avg >= 12:
        clothes = ["長袖", "セーターまたはパーカー", "薄手のジャケット"]
    elif avg >= 7:
        clothes = ["ニット", "コート"]
    else:
        clothes = ["厚手のコートまたはダウン", "マフラー", "手袋"]

    items = []
    if precipitation_probability_percent >= 50:
        items.append("傘")
    elif precipitation_probability_percent >= 30:
        items.append("折りたたみ傘")
    if max_c - min_c >= 10:
        items.append("脱ぎ着しやすい羽織りもの(寒暖差が大きい)")
    if max_c >= 30:
        items.extend(["帽子", "飲み物(熱中症対策)"])
    if wind_speed_kmh >= 30:
        items.append("風を通しにくい上着")

    return {"clothes": clothes, "items": items}


TOOL_FUNCTIONS = {
    "search_location": search_location,
    "get_weather": get_weather,
    "suggest_outfit": suggest_outfit,
}


# ---------------------------------------------------------------------------
# 2. ツールの説明書(Claudeはこれを読んで、どのツールを使うか判断する)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_location",
        "description": "地名(都市名など)から緯度・経度を検索する。天気を調べる前に、場所の座標を知るために使う。",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "地名。例: 東京, 大阪, Paris"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_weather",
        "description": "緯度・経度を指定して、現在の天気と最大7日間の天気予報を取得する。",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "緯度"},
                "longitude": {"type": "number", "description": "経度"},
                "days": {"type": "integer", "description": "予報の日数(1〜7)。"},
            },
            "required": ["latitude", "longitude", "days"],
            "additionalProperties": False,
        },
    },
    {
        "name": "suggest_outfit",
        "description": (
            "その日の最高・最低気温、降水確率、風速から、おすすめの服装と持ち物(傘など)を提案する。"
            "get_weather で天気を調べた後、服装や持ち物を聞かれたときに使う。"
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "max_c": {"type": "number", "description": "最高気温(℃)"},
                "min_c": {"type": "number", "description": "最低気温(℃)"},
                "precipitation_probability_percent": {"type": "number", "description": "降水確率(%)"},
                "wind_speed_kmh": {"type": "number", "description": "風速(km/h)。分からなければ 0"},
            },
            "required": ["max_c", "min_c", "precipitation_probability_percent", "wind_speed_kmh"],
            "additionalProperties": False,
        },
    },
]

SYSTEM_PROMPT = (
    "あなたは天気を案内するアシスタントです。"
    "天気の質問には、必ずツールで最新データを調べてから答えてください。"
    "場所の座標が分からないときは search_location で調べ、その後 get_weather を使います。"
    "服装や持ち物を聞かれたら、get_weather の結果を使って suggest_outfit で提案を作ります。"
    "回答は日本語で、簡潔に、服装や傘の要否などのひとことアドバイスを添えてください。"
)


# ---------------------------------------------------------------------------
# 3. エージェントのループ
# ---------------------------------------------------------------------------

def run_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """ツールを実行し、(結果のJSON文字列, エラーかどうか) を返す"""
    try:
        result = TOOL_FUNCTIONS[name](**tool_input)
        return json.dumps(result, ensure_ascii=False), False
    except Exception as e:  # エラーもClaudeに伝えれば、別の方法を考えてくれる
        return f"エラー: {e}", True


def ask(client: anthropic.Anthropic, messages: list) -> str:
    """messages に質問が入った状態で呼ぶ。Claudeが答えを出すまでツール実行を繰り返す。"""
    while True:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "low"},
            # 安全フィルタで断られた場合に、別モデルで自動的にやり直す
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            return "この質問にはお答えできませんでした。"

        if response.stop_reason != "tool_use":
            # Claudeが「もう答えられる」と判断した
            return "".join(b.text for b in response.content if b.type == "text")

        # Claudeが指定したツールを実行し、結果をまとめて返す
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [ツール実行] {block.name}({json.dumps(block.input, ensure_ascii=False)})")
                content, is_error = run_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                })
        messages.append({"role": "user", "content": tool_results})


def main() -> None:
    client = anthropic.Anthropic()  # APIキーは環境変数 ANTHROPIC_API_KEY から読み込まれる
    messages: list = []

    # コマンド引数があれば1回だけ答えて終了、なければ対話モード
    if len(sys.argv) > 1:
        messages.append({"role": "user", "content": " ".join(sys.argv[1:])})
        print(ask(client, messages))
        return

    print("天気エージェントです。質問をどうぞ(終了: exit)")
    while True:
        try:
            question = input("\nあなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit", "終了"):
            break
        if not question:
            continue
        messages.append({"role": "user", "content": question})
        print(f"\nエージェント> {ask(client, messages)}")


if __name__ == "__main__":
    main()
