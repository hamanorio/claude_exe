"""OpenAI の音声認識API(gpt-transcribe など)で文字起こしする。アプリ(app.py)と評価(eval/eval_asr.py)で共用。

注意: 音声が OpenAI のサーバーに送られる。実在する患者の録音には使わないこと(テスト用の録音のみ)。

Whisper と違い、このAPIは文字起こしの本文だけを返す(区間ごとの時刻・信頼度は返さない)。
アプリでは本文を文ごとに分け、時刻は文字数の割合から見積もった「おおよその値」を付ける。
"""

import os
import re

DEFAULT_MODEL = "gpt-transcribe"
MAX_FILE_BYTES = 25 * 1024 * 1024  # APIが受け付ける1ファイルの上限
MAX_KEYWORDS = 100  # keywords に渡す用語の上限(長すぎるとエラーや精度低下のおそれ)


def hint_keywords(medical_terms: str) -> list:
    """サイドバーの医学用語ヒント(「、」や改行で区切られた語)を keywords 用のリストにする。"""
    words = [w.strip() for w in re.split(r"[、,，\n]", medical_terms or "")]
    seen, out = set(), []
    for w in words:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:MAX_KEYWORDS]


def transcribe_text(audio_path: str, medical_terms: str = "", model: str = DEFAULT_MODEL, api_key: str = "") -> str:
    from openai import OpenAI

    size = os.path.getsize(audio_path)
    if size > MAX_FILE_BYTES:
        raise RuntimeError(
            f"音声ファイルが大きすぎます({size / 1024 / 1024:.1f}MB)。OpenAIのAPIは25MBまでです。"
            "短く切るか、圧縮(例: ffmpeg -i 元.m4a -ac 1 -b:a 48k 圧縮.m4a)してください。"
        )
    client = OpenAI(api_key=api_key or None)  # 空なら環境変数 OPENAI_API_KEY を使う
    kwargs = {
        "model": model,
        "language": "ja",
        "response_format": "json",
        # 長い録音でも途中が抜けにくいよう、サーバー側で無音の位置で区切ってから認識させる
        "chunking_strategy": "auto",
        "prompt": "これは医師と患者の診察会話です。",
    }
    keywords = hint_keywords(medical_terms)
    if keywords and model == "gpt-transcribe":
        kwargs["keywords"] = keywords  # gpt-transcribe だけが対応
    elif keywords:
        kwargs["prompt"] += "次のような医学用語が含まれます: " + "、".join(keywords)
    with open(audio_path, "rb") as fh:
        result = client.audio.transcriptions.create(file=fh, **kwargs)
    return result.text.strip()


def to_segments(text: str, duration: float) -> list:
    """本文を文ごとに分け、Whisperと同じ形のセグメントにする。
    時刻は「その文が本文の何文字目にあるか」の割合から見積もる(話す速さが一定と仮定した概算)。
    信頼度は返らないので avg_logprob は None。
    """
    parts = [p.strip() for p in re.split(r"(?<=[。？?！!])", text) if p.strip()]
    total = sum(len(p) for p in parts) or 1
    segments, pos = [], 0
    for p in parts:
        start = duration * pos / total
        pos += len(p)
        segments.append({"start": start, "end": duration * pos / total, "text": p, "avg_logprob": None, "approx_time": True})
    return segments
