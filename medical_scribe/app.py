"""
診察音声 → カルテ下書き プロトタイプ

機能:
1. 音声ファイルをアップロードしてWhisperで文字起こし(セグメント単位のタイムスタンプ・信頼度付き)
2. 信頼度の低いセグメントを色分け表示し、クリックでその部分の音声を再生
3. テキストの内容(質問/症状の訴え等のパターン)から医師/患者の発言を推測する、
   テキストベースの話者推定(音声の話者分離なしで試す簡易な方法)
4. LLMによる文脈補正をかけ、話者推定後のテキストとの差分をハイライト表示
   (黄色=既存箇所の修正、赤=LLMが新たに追加した可能性のある箇所)
5. 信頼度スコアでは検知できない「文法的には自然だが非定型的な表現」を、
   別の視点のLLM呼び出しで検出(例:「右の奥が痛む」のような曖昧な身体部位表現)
6. (オプション)pyannote.audioによる音声レベルの話者分離(医師/患者の発言区別、SPEAKER_00/01のラベル付け)
7. カルテ下書きから重要所見を構造化チェックリストとして抽出し、側性(右/左/不明)等を
   自由文に埋め込まず、医師が個別に確認・選択する形で表示
8. ②③④⑤のLLM処理をOpenAI APIの代わりにローカルのSwallowモデルで実行する選択肢
   (実在患者データを扱う際の越境移転の問題を避けるため)

使い方(Google Colab):
    !pip install streamlit openai-whisper openai pyannote.audio transformers accelerate
    !streamlit run app.py &
    !npx localtunnel --port 8501
    (表示されたURLを開く。パスワードを聞かれたら `!wget -q -O - ipv4.icanhazip.com` の出力を使う)

話者分離を使うには、事前に以下が必要です:
    1. https://huggingface.co/pyannote/speaker-diarization-3.1 の利用規約に同意
    2. https://huggingface.co/pyannote/segmentation-3.0 の利用規約に同意
    3. Hugging Faceのアクセストークン(Read権限)を発行し、サイドバーに入力

ローカルLLM(Swallow)を使う場合の注意:
    - Whisper(ASR)とSwallow(8B)を同時にGPUに載せるため、VRAM使用量に注意
    - 初回はモデルのダウンロードに時間がかかる
    - ベースのInstructモデル(ファインチューニング前)を推奨。鑑別診断用にファインチューニング
      したモデルは、汎用的な補正・抽出タスクには偏りが出る可能性があるため

使い方(ローカル環境):
    pip install streamlit openai-whisper openai pyannote.audio transformers accelerate
    streamlit run app.py

使い方(Mac / Apple Silicon、メモリ16GB想定):
    Swallow 8Bをbf16のままtransformersで載せると約16GBでメモリが足りないため、
    4bit量子化(GGUF)したモデルをOllamaで動かし、文字起こしはmlx-whisperを使う。
    1. Ollamaをインストール(https://ollama.com)
    2. Swallow 8B InstructのGGUF(Q4_K_M推奨)をHugging Faceで探し、Ollamaに取り込む
         ollama pull hf.co/<ユーザー名>/<GGUFリポジトリ名>:Q4_K_M
       またはダウンロードしたGGUFから:
         echo "FROM ./ファイル名.gguf" > Modelfile && ollama create swallow-8b -f Modelfile
    3. pip install streamlit mlx-whisper openai pykakasi
       (日本語特化の kotoba-whisper も使う場合: pip install transformers torch)
    4. streamlit run app.py
       サイドバーで「Whisperエンジン = mlx-whisper」「LLM = ローカル(Ollama・Mac向け)」を選び、
       Ollamaのモデル名(`ollama list` で表示される名前)を入力する
"""

import difflib
import json
import os
import platform
import re
import tempfile
import urllib.request

import streamlit as st
from openai import OpenAI

# torch / transformers / whisper は使うバックエンドを選んだときだけ読み込む
# (Mac + Ollama + mlx-whisper の構成ではインストール不要にするため)

OLLAMA_URL = "http://localhost:11434/api/chat"
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

st.set_page_config(page_title="診察音声→カルテ下書き プロトタイプ", layout="wide")


@st.cache_resource
def load_local_llm(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def generate_with_local_llm(prompt: str, model_path: str, max_new_tokens: int = 1024) -> str:
    """ローカルのSwallowモデルで生成する。クラウドAPIに患者データを送らずに済む。"""
    import torch

    model, tokenizer = load_local_llm(model_path)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs.shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def unload_ollama(model_name: str) -> None:
    """Ollamaに読み込まれているモデルをメモリから外す(次に使うときは自動で再読み込みされる)。
    16GBのMacでは、文字起こし中にSwallow(約5GB)が載ったままだとメモリが足りなくなるため。
    """
    try:
        req = urllib.request.Request(
            OLLAMA_URL.replace("/api/chat", "/api/generate"),
            data=json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass  # Ollamaが起動していない等。文字起こし自体には影響しない


OLLAMA_MAX_TOKENS = 2048  # 1回の生成の上限。ローカルLLMが同じ文を繰り返し続けて終わらないのを防ぐ


def generate_with_ollama(prompt: str, model_name: str, json_mode: bool = False) -> str:
    """Mac上のOllama(量子化済みSwallow)で生成する。データはMacの外に出ない。
    生成結果を少しずつ受け取る(ストリーミング)ので、全体に時間がかかっても途中で
    タイムアウトしない(タイムアウトは「300秒間なにも返ってこない」場合だけ)。
    """
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "temperature": 0,
            "repeat_penalty": 1.15,
            "num_ctx": 8192,
            "num_predict": OLLAMA_MAX_TOKENS,
        },
    }
    if json_mode:
        payload["format"] = "json"
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    parts = []
    with urllib.request.urlopen(req, timeout=300) as res:
        for line in res:
            if not line.strip():
                continue
            msg = json.loads(line.decode("utf-8"))
            parts.append(msg.get("message", {}).get("content", ""))
            if msg.get("done"):
                break
    return "".join(parts).strip()


def extract_json_block(text: str) -> dict:
    """ローカルLLMはOpenAIのJSONモードのような強制力が無いため、
    出力の中からJSONオブジェクトらしき部分を正規表現で探して解析する。
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def run_llm(prompt: str, backend: str, api_key: str, local_model_path: str) -> str:
    if backend == "ローカル(Swallow)":
        return generate_with_local_llm(prompt, local_model_path)
    if backend == "ローカル(Ollama・Mac向け)":
        return generate_with_ollama(prompt, local_model_path)
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


MLX_WHISPER_REPOS = {
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


@st.cache_resource
def load_whisper_model(model_size: str):
    import whisper

    if not hasattr(whisper, "load_model"):
        # 同名の別パッケージ(pip の "whisper")が入っていると、この状態になる
        raise RuntimeError(
            "openai-whisper が見つかりません。Macではサイドバーで「mlx-whisper」を選んでください。"
            "openai-whisper を使う場合は `pip uninstall whisper && pip install openai-whisper` を実行してください。"
        )
    return whisper.load_model(model_size)


KOTOBA_DEFAULT_MODEL = "kotoba-tech/kotoba-whisper-v2.0"


@st.cache_resource
def load_kotoba_pipeline(model_id: str):
    import torch
    from transformers import pipeline

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    elif torch.cuda.is_available():
        device, dtype = "cuda:0", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    return pipeline("automatic-speech-recognition", model=model_id, torch_dtype=dtype, device=device)


def load_audio_16k(audio_path: str):
    """ffmpegでファイルを直接読み、16kHzモノラルの波形にする。
    transformersに「ファイルのパス」を渡すと内部でパイプ経由でffmpegに流すため、
    m4a(iPhone・Macのボイスメモ等)のように末尾に目次情報がある形式を読めないことがある。
    """
    import subprocess

    import numpy as np

    cmd = ["ffmpeg", "-nostdin", "-i", audio_path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", errors="ignore").strip().splitlines()[-3:]
        raise RuntimeError("ffmpegで音声を読み込めませんでした: " + " / ".join(err))
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


def transcribe_kotoba(audio_path: str, model_id: str) -> dict:
    """日本語に特化して学習されたWhisper系モデル(kotoba-whisper)で文字起こしする。
    出力をopenai-whisperと同じ形({"text", "segments"})にそろえる。
    このモデルは信頼度(avg_logprob)を返さないので、セグメントの信頼度は「なし」になる。
    医学用語ヒント(initial_prompt)は使わない(用語の補正は①-2の音照合で行う)。
    """
    pipe = load_kotoba_pipeline(model_id)
    out = pipe(
        {"raw": load_audio_16k(audio_path), "sampling_rate": 16000},
        chunk_length_s=15,
        # 16GBのMacでは一度に複数区間を処理するとメモリを使い切りやすいので1区間ずつにする
        batch_size=1,
        return_timestamps=True,
        generate_kwargs={"language": "ja", "task": "transcribe"},
    )
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()  # 次の処理(Swallow等)のためにGPUメモリを返す
    except Exception:
        pass
    segments, last_end = [], 0.0
    for chunk in out.get("chunks", []):
        start, end = chunk.get("timestamp", (None, None))
        start = last_end if start is None else start
        end = start if end is None else end
        text = chunk.get("text", "").strip()
        if text:
            segments.append({"start": start, "end": end, "text": text, "avg_logprob": None})
        last_end = end
    return {"text": "".join(seg["text"] for seg in segments), "segments": segments}


def transcribe(audio_path: str, medical_terms: str, model_size: str, engine: str = "openai-whisper"):
    initial_prompt = f"これは医師と患者の診察会話です。次のような医学用語が含まれます: {medical_terms}"
    if engine == "kotoba-whisper(日本語特化)":
        return transcribe_kotoba(audio_path, model_size)
    if engine == "mlx-whisper":
        # Apple SiliconのGPUで動く。出力形式(segments/avg_logprob等)はopenai-whisperと同じ
        import mlx_whisper

        return mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=MLX_WHISPER_REPOS[model_size],
            language="ja",
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
        )
    model = load_whisper_model(model_size)
    result = model.transcribe(
        audio_path,
        language="ja",
        initial_prompt=initial_prompt,
        condition_on_previous_text=False,
    )
    return result


def split_utterances(segments) -> list:
    """Whisperのセグメント(無音で区切られた単位)を、さらに句点・疑問符で文に分ける。
    話者の交代は文の途中では起きにくいので、この単位ごとに話者を割り当てる。
    """
    units = []
    for seg in segments:
        for part in re.split(r"(?<=[。？?！!])", seg["text"]):
            part = part.strip()
            if part:
                units.append(part)
    return units


def infer_speakers_from_text(segments, backend: str, api_key: str, local_model_path: str) -> str:
    """音声レベルの話者分離を使わず、発言内容の文脈だけから医師/患者を推測する。
    質問・所見の提示(医師)と症状の訴え(患者)というパターンの違いを手がかりにする。

    LLMには番号付きの文を渡し、番号ごとの話者だけを答えさせる。本文はプログラム側で
    組み立て直すので、LLMが文の区切りを勝手に変えたり、本文を書き換えたりできない。
    """
    units = split_utterances(segments)
    all_labels = {}
    # 長い会話は SPEAKER_CHUNK 文ずつに分けて推定する(ローカルLLMの入力・出力の上限と速度のため)。
    # 区切り目の判断が崩れないよう、直前の数文とその推定結果を文脈として添える。
    for start in range(0, len(units), SPEAKER_CHUNK):
        chunk = units[start : start + SPEAKER_CHUNK]
        context = "\n".join(
            f"{all_labels.get(j + 1, '不明')}：{units[j]}" for j in range(max(0, start - 3), start)
        )
        all_labels.update(
            label_speaker_chunk(chunk, start, context, backend, api_key, local_model_path)
        )

    lines = []
    for i, unit in enumerate(units):
        speaker = all_labels.get(i + 1, "不明")
        if lines and lines[-1][0] == speaker:
            lines[-1][1] += unit
        else:
            lines.append([speaker, unit])
    return "\n".join(f"{sp}：{text}" for sp, text in lines)


SPEAKER_CHUNK = 40  # 話者推定で一度にLLMへ渡す文の数
CHUNK_CHARS = 1200  # ③④⑤で一度にLLMへ渡す文字数の目安
MAX_CANDIDATE_TERMS = 300  # ③の指示文に載せる用語リストの上限(長すぎると遅くなるため)
MAX_CORRECTION_LEN = 15  # ③で1件として受け付ける「元の表記」の最大文字数(語句単位に限る)
CORRECTION_CHUNK_CHARS = 2500  # ③(修正箇所の一覧だけを出力させる)で一度に渡す文字数


def label_speaker_chunk(chunk, offset, context, backend, api_key, local_model_path) -> dict:
    numbered = "\n".join(f"{offset + i + 1}. {u}" for i, u in enumerate(chunk))
    context_block = f"【直前の会話(参考。回答は不要)】\n{context}\n\n" if context else ""
    prompt = f"""以下は、医師と患者の診察会話の文字起こしを、文ごとに番号を付けて並べたものです。
各文が「医師」と「患者」のどちらの発言かを、前後の流れから判断してください。

判断の手がかり:
- 医師: 質問する、診察・検査の説明をする、所見や検査結果を述べる、それらへの短いコメント(「高いですね」等)
- 患者: 質問に答える、自分の症状・生活・気持ちを話す
- 「〜なんですけど」「〜があります」のように自分の体のことを話している文は患者です
- 話者は1文ごとに交代するとは限りません。同じ人が続けて何文も話すことはよくあります。
  特に医師は、質問の後に「〜とか、〜とか」と例を挙げて質問を続けたり、
  挨拶の後に続けて質問したり、測定値を言った後に感想を続けたりします。

【例】
1. おはようございます。
2. 今日はどうしましたか？
3. 咳が止まらなくて。
4. 熱はありますか？
5. 寒気がするとか、だるいとか。
6. 昨日から少し寒気があります。
7. 体温を測りますね38度2分です
8. 高めですね
回答:
1: 医師
2: 医師
3: 患者
4: 医師
5: 医師
6: 患者
7: 医師
8: 医師

出力は1行に1つ、「番号: 医師」または「番号: 患者」の形式だけで、全ての番号について答えてください。本文は書かないでください。

{context_block}【文字起こし】
{numbered}

【回答】
"""
    output = run_llm(prompt, backend, api_key, local_model_path)
    valid = range(offset + 1, offset + len(chunk) + 1)
    labels = {}
    for m in re.finditer(r"(\d+)\s*[.:：、)]\s*(医師|患者)", output):
        if int(m.group(1)) in valid:
            labels[int(m.group(1))] = m.group(2)
    return labels


def split_into_chunks(text: str, max_chars: int = CHUNK_CHARS) -> list:
    """長い会話を、行(話者ごとの発言)の区切りで max_chars 程度ずつに分ける。
    1行が長すぎる場合は句点で分ける(分けた文同士は改行を入れずにつなぐ)。"""
    pieces = []  # (文字列, 前に改行が必要か)
    for line in text.splitlines():
        if len(line) <= max_chars:
            pieces.append((line, True))
        else:
            sents = [p for p in re.split(r"(?<=[。！？!?])", line) if p]
            pieces += [(p, i == 0) for i, p in enumerate(sents)]
    chunks, cur = [], ""
    for piece, newline in pieces:
        if cur and len(cur) + len(piece) > max_chars:
            chunks.append(cur)
            cur = ""
        cur = piece if not cur else cur + ("\n" if newline else "") + piece
    if cur:
        chunks.append(cur)
    return chunks


def correct_with_llm(
    raw_text, backend, api_key, local_model_path, medical_terms="", progress=None, candidate_terms=()
):
    """長い会話は分割して補正し、つなぎ直す。progress(i, n) で進み具合を知らせる。"""
    # 出力は修正箇所の一覧だけで短いので、③は大きめの単位で分けて呼び出し回数を減らす
    chunks = split_into_chunks(raw_text, max_chars=CORRECTION_CHUNK_CHARS)
    out = []
    for i, c in enumerate(chunks):
        if progress:
            progress(i, len(chunks))
        out.append(correct_chunk_with_llm(c, backend, api_key, local_model_path, medical_terms, candidate_terms))
    if progress:
        progress(len(chunks), len(chunks))
    return "\n".join(out)


def correct_chunk_with_llm(
    raw_text: str, backend: str, api_key: str, local_model_path: str, medical_terms: str = "", candidate_terms=()
) -> str:
    """誤変換の箇所だけを「誤 → 正」の形でLLMに答えさせ、プログラム側で本文に当てはめる。
    全文を書き直させる方式より出力が短いので、ローカルLLMでも数倍速い。
    また、本文に実在する語しか置き換えないので、LLMが文を削ったり書き足したりできない。
    """
    prompt = f"""以下は、医師と患者の診察会話を音声認識で文字起こししたテキストです。
音声認識特有の誤変換(医学用語が、似た音の別の言葉や漢字になっている等)を探してください。

誤変換の典型例(同じ音・似た音の別の漢字や単語になっている):
- 「関節」と「間接」、「意志」と「医師」のような同音異義語の取り違え
- 医学用語が、意味の通らない一般語の組み合わせになっているもの
  (例:「心房最同」→「心房細動」、「指示異常賞」→「脂質異常症」)

この会話には、次のような用語が出てくる可能性があります(医師が事前に指定したもの):
{medical_terms}

誤変換は、次の医学用語のどれかが別の字や似た音の語になっていることが多いです(修正後の候補):
{"、".join(list(candidate_terms)[:MAX_CANDIDATE_TERMS])}

【例】
文字起こし:
患者：咳が出て、竹が絡みます
医師：胸の音を聞きますね、指定ラ音があります。善則の既往はありますか
誤変換の一覧:
竹 → 痰
指定ラ音 → 湿性ラ音
善則 → 喘息

答え方:
- 見つけた誤変換を1行に1つ、「元の表記 → 修正後の表記」の形で書いてください。
- 元の表記は、テキスト中の文字をそのまま(一字一句同じに)書き写してください。
- 聞き取れなかった内容を推測で補うことや、言い回しを整えることはしないでください。
- 確信が持てない箇所は書かないでください。
- 句読点(、。？)の追加や修正は不要です。文全体を書き写さず、間違っている語だけを書いてください。
- 誤変換が無ければ「なし」とだけ書いてください。

【文字起こし】
{raw_text}

【誤変換の一覧】
"""
    output = run_llm(prompt, backend, api_key, local_model_path)
    return apply_llm_corrections(raw_text, output, frozenset(candidate_terms))


def apply_llm_corrections(text: str, llm_output: str, protected_terms=frozenset()) -> str:
    """LLMが挙げた「誤 → 正」のうち、本文に実在する語の置き換えだけを適用する。
    話者ラベルや改行には触れない。次の提案は誤変換の修正ではないので適用しない:
    - 修正後が空(削除)
    - 元の語をそのまま含む書き足し(例:「甲状腺」→「甲状腺の薬」)や、その逆の削り
    - 元の語が用語リストにある正しい医学用語(例:「甲状腺」を別の語にする)
    書き足しを本文中の同じ語すべてに当てはめると、会話全体が壊れるため。"""
    for line in llm_output.splitlines():
        line = re.sub(r"^\s*(?:[-・*]|\d+[.)、])\s*", "", line).strip()
        parts = re.split(r"\s*(?:→|->|=>|⇒)\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        wrong, right = (p.strip().strip("「」『』\"'") for p in parts)
        if not wrong or not right or wrong == right or wrong not in text:
            continue
        if wrong in ("医師", "患者") or "：" in wrong or ":" in wrong:
            continue  # 話者ラベルは書き換えさせない
        if is_punctuation_only(wrong, right) or len(wrong) > MAX_CORRECTION_LEN:
            continue  # 句読点だけの修正や、文ごとの書き直しは誤変換の修正ではないので使わない
        if wrong in right or right in wrong:
            continue  # 書き足し・削りは誤変換の修正ではない
        if wrong in protected_terms:
            continue  # 正しい医学用語は書き換えない
        text = text.replace(wrong, right)
    return text


def parse_correction_dict(raw: str) -> list:
    """「誤→正」を1行に1つ書いた辞書を読み取る。"""
    rules = []
    for line in raw.splitlines():
        parts = re.split(r"\s*(?:→|->|=>)\s*", line.strip(), maxsplit=1)
        if len(parts) == 2 and parts[0] and parts[1]:
            rules.append((parts[0], parts[1]))
    return rules


def apply_correction_dict(text: str, rules: list, applied: dict) -> str:
    """医師が登録した誤変換を機械的に置き換える。LLMと違って、登録した以外の変更は一切しない。"""
    for wrong, right in rules:
        n = text.count(wrong)
        if n:
            text = text.replace(wrong, right)
            applied[f"{wrong}→{right}"] = applied.get(f"{wrong}→{right}", 0) + n
    return text


@st.cache_resource
def load_diarization_pipeline(hf_token: str):
    from pyannote.audio import Pipeline

    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )


def diarize(audio_path: str, hf_token: str):
    pipeline = load_diarization_pipeline(hf_token)
    diarization = pipeline(audio_path)
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))
    return turns


def assign_speakers(segments, turns):
    """各Whisperセグメントに、時間的に最も重なりが大きい話者ラベルを付与する。"""
    labeled = []
    for seg in segments:
        seg_start, seg_end = seg["start"], seg["end"]
        best_speaker, best_overlap = None, 0.0
        for t_start, t_end, speaker in turns:
            overlap = min(seg_end, t_end) - max(seg_start, t_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        labeled.append({**seg, "speaker": best_speaker or "不明"})
    return labeled


SPEAKER_COLORS = {
    "SPEAKER_00": "#1f77b4",
    "SPEAKER_01": "#d62728",
    "SPEAKER_02": "#2ca02c",
}


def speaker_badge(speaker: str) -> str:
    color = SPEAKER_COLORS.get(speaker, "#888888")
    return f"<span style='background-color:{color}; color:white; padding:2px 8px; border-radius:4px; font-size:0.85em;'>{speaker}</span>"


def confidence_badge(avg_logprob) -> str:
    if avg_logprob is None:
        return "⚪ 信頼度なし"
    if avg_logprob < -1.0:
        return "🔴 低信頼"
    elif avg_logprob < -0.5:
        return "🟡 中信頼"
    else:
        return "🟢 高信頼"


def check_vague_terms(text: str, backend: str, api_key: str, local_model_path: str) -> str:
    """長い会話は分割してチェックし、指摘をまとめる。"""
    flags = [check_vague_terms_chunk(c, backend, api_key, local_model_path) for c in split_into_chunks(text)]
    flags = [f for f in flags if f and not f.startswith(NO_FLAG_PHRASE)]
    return "\n\n".join(flags) if flags else NO_FLAG_PHRASE + "。"


def check_vague_terms_chunk(text: str, backend: str, api_key: str, local_model_path: str) -> str:
    prompt = f"""以下はカルテ下書きです。
次の2種類の箇所を探して、音声認識の誤りの可能性があるとして指摘してください。

1. 意味の通らない語句: 診察の会話として文脈上ありえない単語や、医学用語が崩れたように見える語句
   (例:「心房最同」「左の配に影」のように、一般語の組み合わせとしても医学用語としても不自然なもの)
2. 医師の発言の中の曖昧な部位表現: 診察所見として、身体のどこかが特定できないもの
   (例:医師が「奥のほうに」「あのへんに」と言っていて、部位が分からないもの)

患者が自分の症状を自分の言葉で表した表現(「ズキズキする」「キラキラした光」「重い感じ」など)は、
症状の性質を伝える大切な情報なので、曖昧として指摘しないでください。
会話なので、話し言葉が口語的なのは自然です。言い回しが口語的というだけでは指摘しないでください。
また、医学的に正しく使われている語(例:「発作」「前兆」「圧痛」)を別の語に言い換える提案はしないでください。

文を1つずつ確認し、見落としが無いようにしてください。
問題のある箇所だけを挙げ、問題のない文は書かないでください(「特に問題なし」と1文ずつ書くことはしない)。
断定せず、「要確認」として該当箇所を引用した上で挙げてください。
該当箇所が無い場合は「特に気になる曖昧な表現はありません」とだけ答えてください。

【カルテ下書き】
{text}

【要確認フラグ】
"""
    return drop_contradicting_none(run_llm(prompt, backend, api_key, local_model_path))


NO_FLAG_PHRASE = "特に気になる曖昧な表現はありません"


def drop_contradicting_none(text: str) -> str:
    """指摘を挙げた後に「特に気になる表現はありません」と付け足す矛盾した出力を整える。"""
    rest = text.replace(NO_FLAG_PHRASE + "。", "").replace(NO_FLAG_PHRASE, "").strip()
    return rest if rest else NO_FLAG_PHRASE + "。"


LATERALITY_OPTIONS = ["不明", "右", "左", "両側", "なし(左右関係なし)"]


def normalize_laterality(value) -> str:
    """LLMの表記ゆれ(「なし」「左右関係なし」「両方」等)を選択肢に揃える。
    ローカルLLMは指定どおりの文字列を返さないことがあり、そのまま「不明」に
    落とすと、左右の概念が無い所見まで「不明」と表示されてしまうため。
    """
    v = str(value or "").strip().replace("（", "(").replace("）", ")")
    if v in LATERALITY_OPTIONS:
        return v
    if "関係" in v or v in ("なし", "無し", "該当なし", "N/A", "-"):
        return "なし(左右関係なし)"
    if "両" in v:
        return "両側"
    if v.startswith("右"):
        return "右"
    if v.startswith("左"):
        return "左"
    return "不明"


# 左右の概念が無いことが明らかな所見。ローカルLLMは指示しても「不明」を返すことが
# あるため、項目名にこれらの語を含む場合はプログラム側で「なし」に揃える。
NO_LATERALITY_KEYWORDS = [
    "動悸", "体重", "発汗", "汗", "暑がり", "寒がり", "食欲", "発熱", "体温", "倦怠",
    "イライラ", "不眠", "月経", "脈", "血圧", "呼吸数", "SpO2", "TSH", "FT3", "FT4",
    "検査", "採血", "血液", "過敏", "頻度", "周期", "関連", "既往",
]


# 左右がありうる部位・所見。LLMが「なし(左右関係なし)」と答えても、医師が選べるよう「不明」に戻す。
LATERAL_KEYWORDS = [
    "震え", "振戦", "麻痺", "しびれ", "痺れ", "痛", "腫", "浮腫", "眼", "目", "耳",
    "手", "足", "腕", "脚", "肺", "乳房", "甲状腺", "関節",
]


def apply_laterality_rules(findings: list) -> list:
    for f in findings:
        item = str(f.get("item", ""))
        if any(k in item for k in NO_LATERALITY_KEYWORDS):
            f["laterality"] = "なし(左右関係なし)"
        elif any(k in item for k in LATERAL_KEYWORDS) and normalize_laterality(f.get("laterality")) == "なし(左右関係なし)":
            f["laterality"] = "不明"
    return findings


def is_paraphrased(item: str, text: str) -> bool:
    """項目名が会話中に見当たらない(LLMが別の用語に言い換えた)かを判定する。
    項目名の漢字・カタカナ2文字の並びが1つも本文に無ければ「言い換え」とみなす。
    例: 患者が「キラキラした光」と言っただけなのに項目名が「光視症」になっている場合。
    """
    chunks = re.findall(r"[\u4e00-\u9fff\u30a0-\u30ffA-Za-z0-9]+", item)
    bigrams = [c[i : i + 2] for c in chunks for i in range(max(1, len(c) - 1))]
    return bool(bigrams) and not any(b in text for b in bigrams)


def extract_structured_findings(text: str, backend: str, api_key: str, local_model_path: str) -> list:
    """長い会話は分割して抽出し、同じ項目名の重複を除いてまとめる。"""
    merged, seen = [], set()
    for c in split_into_chunks(text):
        for f in extract_findings_chunk(c, backend, api_key, local_model_path):
            key = re.sub(r"\s", "", str(f.get("item", "")))
            if key and key not in seen:
                seen.add(key)
                merged.append(f)
    return merged


def extract_findings_chunk(text: str, backend: str, api_key: str, local_model_path: str) -> list:
    """自由文のカルテ下書きから、診断に直結する重要所見を構造化して抽出する。
    側性(左右)や所見の有無を、自由文に埋め込まず個別項目として扱うことで、
    LLMが「不明」を正直に選べるようにし(自由文要約では確認済みの通り機能しなかった)、
    医師が1項目ずつ確認する運用を可能にする。
    """
    prompt = f"""以下はカルテ下書きです。診断・治療方針に直結する重要な所見を構造化して抽出してください。
患者が訴えた症状・変化(食欲・睡眠・排便・気分など)と、医師が述べた身体所見(皮膚・脈・頸部など)と検査結果は、
軽そうに見えても省略せず、1つずつ別の項目として挙げてください。
項目名は会話中の言葉に基づけてください。会話に出ていない診断名や専門用語に置き換えないでください
(例:患者が「胸がしめつけられる」と言っただけなら「胸のしめつけ感」とし、「狭心症」とはしない)。

重要: 「不明」は、その情報が本当にテキスト中に存在しない場合にのみ使ってください。
テキストに明確に書かれている内容は、省略したり「不明」にせず、そのまま正確に抽出してください。
「不明」を安全策として多用することは、情報を省略することになり、推測で埋めることと同じくらい有害です。

"value" フィールドについて:
- テキストに明確な記載がある場合は、その内容をそのまま書いてください(例: 「TSHが著明に低下」→ value: "低下")。
- テキストに記載が無く、本当に判断できない場合のみ "不明" としてください。

"laterality" フィールドについて、まず「この所見はそもそも左右の概念を持つか」を判断してください:
- 疼痛・麻痺・浮腫・感覚障害・腫脹のように、身体の左右どちらかに起こりうる所見 →
  テキストに明記されていれば "右"/"左"/"両側"、明記されていなければ "不明"
- 動悸・体重減少・発熱・血液検査値のように、そもそも左右の概念が無い所見 →
  常に "なし(左右関係なし)" としてください("不明" にしないでください)

confidence フィールドには、その抽出の確信度を "high" または "low" で入れてください。
テキストに明確な記載がある場合は "high" です。表現が曖昧だったり、間接的な推測を含む場合のみ "low" にしてください。

【例】
テキスト: "TSHが著明に低下していて、FT4とFT3はどちらも上昇していた。"
出力の一部: {{"item": "TSH", "laterality": "なし(左右関係なし)", "value": "低下", "confidence": "high"}}

テキスト: "右下肺野に湿性ラ音を聴取した。"
出力の一部: {{"item": "肺野の異常音", "laterality": "右", "value": "湿性ラ音", "confidence": "high"}}

次のJSON形式で出力してください(他のテキストは含めないでください):
{{
  "findings": [
    {{"item": "所見名", "laterality": "右/左/両側/なし(左右関係なし)/不明 のいずれか", "value": "具体的な内容", "confidence": "high または low"}}
  ]
}}

【カルテ下書き】
{text}
"""
    if backend == "ローカル(Swallow)":
        raw_output = generate_with_local_llm(prompt, local_model_path)
        data = extract_json_block(raw_output)
    elif backend == "ローカル(Ollama・Mac向け)":
        # OllamaのJSONモードで出力をJSONに制約する(念のため抽出処理も通す)
        raw_output = generate_with_ollama(prompt, local_model_path, json_mode=True)
        data = extract_json_block(raw_output)
    else:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, AttributeError):
            data = {}
    return apply_laterality_rules(data.get("findings", []))


def highlight_diff_html(raw_text: str, corrected_text: str) -> str:
    sm = difflib.SequenceMatcher(None, raw_text, corrected_text)
    html_parts = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        seg = corrected_text[j1:j2]
        if tag == "equal":
            html_parts.append(seg)
        elif tag == "insert":
            html_parts.append(
                f"<span style='background-color:#ffcccc' title='生の音声認識結果には存在しなかった箇所'>{seg}</span>"
            )
        elif tag == "replace":
            html_parts.append(
                f"<del style='color:#999'>{raw_text[_i1:_i2]}</del>"
                f"<span style='background-color:#fff3b0' title='音声認識結果が修正された箇所'>{seg}</span>"
            )
        elif tag == "delete":
            # LLMが消した箇所。情報の欠落(例:「著明に」が消えて程度が分からなくなる)に気づけるよう表示する
            html_parts.append(
                f"<del style='background-color:#e0e0ff' title='LLMが削除した箇所'>{raw_text[_i1:_i2]}</del>"
            )
    # 改行を表示に反映し、話者ごとの行が1段落につながって読みにくくならないようにする
    return "".join(html_parts).replace("\n", "<br>")


# 音の照合に使う、診療科を問わない基本的な医学用語。サイドバーで追加・ファイル読み込みができる。
DEFAULT_SOUND_TERMS = """診察 問診 触診 聴診 視診 打診 皮膚 発汗 多汗 暑がり 寒がり 動悸 息切れ 倦怠感
食欲 食欲不振 食欲亢進 体重減少 体重増加 発熱 頭痛 腹痛 胸痛 背部痛 腰痛 嘔気 嘔吐 下痢 便秘 血便
浮腫 振戦 しびれ めまい 失神 咳嗽 喀痰 呼吸困難 喘鳴 月経不順 不順 甲状腺 腫大 びまん性 結節
眼球突出 頻脈 徐脈 不整脈 心房細動 血圧 脈拍 呼吸音 心雑音 湿性ラ音 浸潤影 胸水 著明 軽度 中等度 高度
遊離T4 遊離T3 血液検査 採血 尿検査 心電図 圧痛 反跳痛 筋性防御 腹部膨満 黄疸 貧血 蕁麻疹 発疹
呂律 構音障害 顔面神経麻痺 片麻痺 意識障害 片頭痛 緊張型頭痛 群発頭痛 前兆 閃輝暗点 光過敏 音過敏 脂質異常症 糖尿病 高血圧 既往歴 家族歴 服薬 内服 頓服"""

TERM_LIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "term_lists")


def list_term_files() -> list:
    """term_lists フォルダ内の診療科別用語リスト(.txt)の名前一覧。ファイルを足せば選択肢に増える。"""
    if not os.path.isdir(TERM_LIST_DIR):
        return []
    # 「_」で始まるファイル(_除外語.txt 等)は診療科の選択肢に出さない
    return sorted(f[:-4] for f in os.listdir(TERM_LIST_DIR) if f.endswith(".txt") and not f.startswith("_"))


def load_term_file(name: str) -> list:
    path = os.path.join(TERM_LIST_DIR, name + ".txt")
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


_kakasi = None


def to_reading_tokens(text: str) -> list:
    """文字列を(表記, ひらがな読み)のトークン列に分ける。英数字はそのまま読みとして扱う。"""
    global _kakasi
    if _kakasi is None:
        import pykakasi

        _kakasi = pykakasi.kakasi()
    return [(t["orig"], t["hira"]) for t in _kakasi.convert(text) if t["orig"]]


def parse_term_list(raw: str) -> list:
    terms = re.split(r"[\s、,，・/]+", raw)
    return list(dict.fromkeys(t.strip() for t in terms if t.strip()))


@st.cache_resource
def build_sound_index(terms: tuple):
    """用語の読みを2文字ずつに分けた索引を作る。大きな用語リストでも照合が速くなるように。"""
    entries, index = [], {}
    for term in terms:
        reading = "".join(h for _, h in to_reading_tokens(term))
        if len(reading) < 3:  # 2文字以下の読みは偶然の一致が多すぎるため対象外
            continue
        idx = len(entries)
        entries.append((term, reading))
        for i in range(len(reading) - 1):
            index.setdefault(reading[i : i + 2], set()).add(idx)
    return entries, index


PARTICLE_MID = re.compile(r"[^\u3040-\u309f][はがをにのでともへや][^\u3040-\u309f]")
HIRAGANA_ANY = re.compile(r"[\u3040-\u309f]")
PARTICLE_END = re.compile(r"[はがをにのでともへや]$")
HIRAGANA_END = re.compile(r"[\u3040-\u309f]$")
HIRAGANA_ONLY = re.compile(r"^[\u3040-\u309fー、。,.？！?!\s]*$")


def find_sound_alike(text: str, entries: list, index: dict, term_set: set, max_tokens: int = 6) -> list:
    """読みが用語リストの語に近いのに表記が違う箇所を探す(例:「有利T4」→「遊離T4」)。
    誤りの書き方ではなく正しい用語だけを登録すればよいので、未知の誤変換にも対応できる。
    戻り値: (開始位置, 終了位置, 元の表記, 候補の用語, 類似度) のリスト(重なりなし)
    """
    toks = to_reading_tokens(text)
    starts, pos = [], 0
    for surf, _ in toks:
        found_at = text.find(surf, pos)
        starts.append(found_at if found_at >= 0 else pos)
        pos = starts[-1] + len(surf)

    found = []
    for i in range(len(toks)):
        surface, reading = "", ""
        for j in range(i, min(i + max_tokens, len(toks))):
            surface += toks[j][0]
            reading += toks[j][1]
            # 前後のひらがなを除いた中心部分(「なる症状」→「症状」)も、用語・除外語なら対象外
            core = re.sub(r"^[\u3040-\u309f]+|[\u3040-\u309f]+$", "", surface)
            if len(reading) < 3 or len(surface) < 2 or HIRAGANA_ONLY.match(surface) or surface in term_set or core in term_set:
                continue
            if re.search(r"[：:、。？！?!\s]", surface):
                continue  # 話者ラベルや句読点をまたぐ区切りは対象外
            cands = set()
            for k in range(len(reading) - 1):
                cands |= index.get(reading[k : k + 2], set())
            for idx in cands:
                term, term_reading = entries[idx]
                if abs(len(term_reading) - len(reading)) > 2 or term in surface or surface in term:
                    continue
                # 「脈は」→「脈拍」のように、助詞で終わる区切りを漢字で終わる用語と取り違えない
                if PARTICLE_END.search(surface) and not HIRAGANA_END.search(term):
                    continue
                # 「甲状腺にホルモン」→「甲状腺ホルモン」のように、助詞をはさんだ2語を1語にしない
                if PARTICLE_MID.search(surface) and not HIRAGANA_ANY.search(term):
                    continue
                ratio = difflib.SequenceMatcher(None, reading, term_reading).ratio()
                # 短い語は読みが完全一致する場合のみ(偶然の一致を避ける)
                if ratio >= (1.0 if len(term_reading) <= 3 else 0.75):
                    found.append((starts[i], starts[i] + len(surface), surface, term, ratio))

    # 類似度の高い順に、重ならないものだけを残す(「腸鳴」と「腸鳴に」のような重複を除く)
    found.sort(key=lambda f: (-f[4], f[1] - f[0]))
    chosen, used = [], set()
    for f in found:
        span = set(range(f[0], f[1]))
        # 同じ箇所で同じくらい近い候補が複数あるとき(発見→発汗/発疹)は両方出して医師に選ばせる
        tie = any(c[0] == f[0] and c[1] == f[1] and c[4] == f[4] and c[3] != f[3] for c in chosen)
        if not span & used or tie:
            if not any(c[0] == f[0] and c[1] == f[1] and c[3] == f[3] for c in chosen):
                chosen.append(f)
            used |= span
    return sorted(chosen)


def apply_sound_replacements(text: str, matches: list, accepted_pairs: set) -> str:
    out, last = [], 0
    for start, end, surface, term, _ in matches:
        if (surface, term) in accepted_pairs and start >= last:  # 同じ箇所の別候補は先に採用した方だけ
            out.append(text[last:start])
            out.append(term)
            last = end
    out.append(text[last:])
    return "".join(out)


PUNCTUATION = set("。、，．,.!?！？ 　\n")


def group_changes(old_text: str, new_text: str, max_gap: int = 1) -> list:
    """③の前後のテキストを、採用・却下を選べる「修正」の単位に分ける。
    文字単位の差分は細切れになるため(例:「美満腺の主題」→「びまん性の腫大」が
    「美満腺」「主題」の2つに割れる)、間の一致部分が max_gap 文字以下(既定は1文字。「の」等)なら1つの修正にまとめる。
    戻り値: ("equal", 文字列) または ("change", 元の文字列, 修正後の文字列) のリスト
    """
    ops = difflib.SequenceMatcher(None, old_text, new_text, autojunk=False).get_opcodes()
    parts = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            parts.append(["equal", old_text[i1:i2]])
        elif parts and parts[-1][0] == "change":
            parts[-1][1] += old_text[i1:i2]
            parts[-1][2] += new_text[j1:j2]
        elif len(parts) >= 2 and parts[-2][0] == "change" and len(parts[-1][1]) <= max_gap:
            gap = parts.pop()[1]
            parts[-1][1] += gap + old_text[i1:i2]
            parts[-1][2] += gap + new_text[j1:j2]
        else:
            parts.append(["change", old_text[i1:i2], new_text[j1:j2]])
    return [tuple(p) for p in parts]


def is_punctuation_only(old: str, new: str) -> bool:
    strip = lambda t: "".join(c for c in t if c not in PUNCTUATION)
    return strip(old) == strip(new)


def build_reviewed_text(parts: list, accepted: dict) -> str:
    """採用された修正だけを反映したテキストを組み立てる。採用されていない修正は元のまま。"""
    out = []
    for idx, part in enumerate(parts):
        if part[0] == "equal":
            out.append(part[1])
        else:
            out.append(part[2] if accepted.get(idx, False) else part[1])
    return "".join(out)


def md_escape(text: str) -> str:
    text = text.replace("\n", " / ")
    return re.sub(r"([*_~`\\\[\]])", r"\\\1", text)


SOAP_SECTIONS = ["S(患者の訴え)", "O(所見・検査)"]


def guess_soap_section(item: str, value: str, text: str) -> str:
    """項目がSかOかを、会話のどちらの話者の発言に根拠があるかで推定する。
    項目名と内容の2文字の並びが、患者の発言と医師の発言のどちらに多く含まれるかで判断する。
    (例:「ドキドキする感じ」は患者の発言にある→S、「全体的に腫れている」は医師の発言にある→O)
    """
    patient = "".join(m.group(2) for m in re.finditer(r"(患者)\s*[:：](.*)", text))
    doctor = "".join(m.group(2) for m in re.finditer(r"(医師)\s*[:：](.*)", text))
    chunks = re.findall(r"[\u3040-\u30ff\u4e00-\u9fffA-Za-z0-9]+", f"{item} {value}")
    bigrams = {c[i : i + 2] for c in chunks for i in range(max(1, len(c) - 1))}
    p_score = sum(b in patient for b in bigrams)
    d_score = sum(b in doctor for b in bigrams)
    return SOAP_SECTIONS[1] if d_score > p_score else SOAP_SECTIONS[0]


def build_chart_draft(findings: list) -> str:
    """医師が確認した所見だけから、SOAP形式のカルテ下書きを組み立てる。
    S・Oは確認済みの項目を要約として並べる(新たにLLMで要約し直さないので、
    確認していない言い換えや推測は入らない)。A(評価)とP(計画)は医師が書く。
    """

    def line(f):
        lat = f["laterality"]
        lat_txt = "" if lat in ("なし(左右関係なし)", "不明") else f"({lat})"
        return f"・{f['item']}{lat_txt}: {f['value']}"

    s_items = [line(f) for f in findings if f["section"] == SOAP_SECTIONS[0]]
    o_items = [line(f) for f in findings if f["section"] == SOAP_SECTIONS[1]]
    out = ["【S】"] + (s_items or ["・(なし)"])
    out += ["", "【O】"] + (o_items or ["・(なし)"])
    out += ["", "【A】", "(医師が記入)", "", "【P】", "(医師が記入)"]
    return "\n".join(out)


# ---------------- UI ----------------

st.title("🩺 診察音声 → カルテ下書き プロトタイプ")
st.caption("AIの出力をそのまま信じず、怪しい箇所は音声に戻って確認する、という設計の検証用プロトタイプです。")

with st.sidebar:
    st.header("設定")
    whisper_engine = st.radio(
        "音声認識エンジン",
        ["openai-whisper", "mlx-whisper", "kotoba-whisper(日本語特化)"],
        # Apple Silicon の Mac では mlx-whisper を初期選択にする(再起動のたびに選び直さなくて済むように)
        index=1 if IS_APPLE_SILICON else 0,
        help="Mac(Apple Silicon)では mlx-whisper の方が大幅に速く動きます。"
        "kotoba-whisper は日本語の音声で学習し直したモデルで、日本語の誤変換が減る可能性があります"
        "(要 `pip install transformers torch`)。同じ音声で切り替えて①を実行すると比較できます。",
    )
    if whisper_engine == "kotoba-whisper(日本語特化)":
        model_size = st.text_input(
            "kotoba-whisperのモデル",
            KOTOBA_DEFAULT_MODEL,
            help="Hugging Faceのモデル名。初回は自動でダウンロードされます(約1.5GB)。",
        )
        st.caption("※ このエンジンは医学用語ヒントを使いません。用語の補正は①-2の音照合で行います。")
    else:
        whisper_sizes = ["medium", "large-v3", "large-v3-turbo"] if whisper_engine == "mlx-whisper" else ["medium", "large-v3"]
        model_size = st.selectbox("Whisperモデルサイズ", whisper_sizes, index=len(whisper_sizes) - 1)
    medical_terms = st.text_area(
        "医学用語ヒント(initial_prompt用)",
        "呂律、構音障害、顔面神経麻痺、心房細動、不整脈、脂質異常症、浸潤影、湿性ラ音",
        height=100,
    )
    # 「誤→正」の行がヒント欄に入っていると、誤った語(「腸鳴に低下」等)がWhisperへのヒントになり、
    # かえってその誤変換を誘発しうる。ヒントからは除外し、辞書欄に移すよう促す。
    misplaced = [line for line in medical_terms.splitlines() if re.search(r"→|->|=>", line)]
    if misplaced:
        st.warning(
            "医学用語ヒントに「誤→正」の行があります。これらはヒントとしては使わずに除外しました。"
            "下の「誤変換辞書」に移してください: " + " / ".join(misplaced)
        )
        medical_terms = "\n".join(line for line in medical_terms.splitlines() if line not in misplaced)
    correction_dict_raw = st.text_area(
        "誤変換辞書(誤→正 を1行に1つ)",
        "有利T4→遊離T4\n有利T3→遊離T3",
        height=100,
        help="よく起きる音声認識の誤りを登録しておくと、②の前に機械的に置き換えます。"
        "LLMより確実で、登録した以外の変更はしません。",
    )
    term_files = list_term_files()
    selected_term_files = st.multiselect(
        "音照合に使う診療科の用語リスト",
        term_files,
        default=[f for f in term_files if "内科一般" in f],
        help="文字起こしの中で、読みがこれらの語に近いのに表記が違う箇所を修正候補として出します"
        "(例:「有利T4」→「遊離T4」)。リストは medical_scribe/term_lists/ のtxtファイルで、"
        "1行1語で編集・追加できます(新しいファイルを置けば選択肢に増えます)。",
    )
    sound_terms_raw = st.text_area(
        "追加の用語(自分用・任意)",
        "" if term_files else DEFAULT_SOUND_TERMS,
        height=80,
        help="リストに無い用語をここに足せます(空白・読点・改行区切り)。医学用語ヒントの語も自動で含めます。",
    )
    sound_terms_file = st.file_uploader(
        "用語リストのファイル(任意・1行1語のtxt、または1列目が用語のcsv)", type=["txt", "csv"]
    )
    st.divider()
    llm_backend = st.radio(
        "②③④⑤で使うLLM",
        ["OpenAI API (GPT-4o)", "ローカル(Swallow)", "ローカル(Ollama・Mac向け)"],
        # 患者データが外部に送られないよう、初期選択は必ずローカルにする。
        # Apple Silicon の Mac では Ollama、それ以外(Colab等)では transformers の Swallow。
        index=2 if IS_APPLE_SILICON else 1,
        help="実在する患者データを扱う場合は越境移転の問題を避けるため、ローカルのいずれかを推奨します。"
        "Macでは「ローカル(Ollama・Mac向け)」を使ってください。",
    )
    api_key = ""
    local_model_path = ""
    if llm_backend == "OpenAI API (GPT-4o)":
        api_key = st.text_input("OpenAI APIキー", type="password")
    elif llm_backend == "ローカル(Ollama・Mac向け)":
        local_model_path = st.text_input(
            "Ollamaのモデル名",
            "swallow-8b",
            help="`ollama list` で表示される名前。4bit量子化(Q4_K_M)のSwallow 8B Instructを推奨。",
        )
        st.caption("⚠️ 事前にOllamaを起動しておいてください。16GBのMacでは他のアプリを閉じておくと安定します。")
    else:
        local_model_path = st.text_input(
            "Swallowモデルのパス",
            "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
            help="ファインチューニング前のベースInstructモデルを推奨(汎用的な補正・抽出タスクのため)。",
        )
        st.caption("⚠️ 初回実行時はモデルのダウンロード・読み込みに時間がかかります。GPUのメモリ使用量にも注意してください。")
    st.divider()
    enable_diarization = st.checkbox("話者分離を行う(医師/患者の区別)")
    hf_token = st.text_input("Hugging Face トークン(話者分離用)", type="password") if enable_diarization else None

uploaded_file = st.file_uploader("診察音声をアップロード", type=["mp3", "wav", "m4a"])

if uploaded_file:
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        audio_path = tmp.name

    if "play_start" not in st.session_state:
        st.session_state.play_start = 0

    if st.button("① 文字起こしを実行", type="primary"):
        if llm_backend == "ローカル(Ollama・Mac向け)" and local_model_path:
            unload_ollama(local_model_path)
        with st.spinner(f"{whisper_engine}で文字起こし中...(初回はモデルのダウンロードに時間がかかります)"):
            st.session_state.result = transcribe(audio_path, medical_terms, model_size, whisper_engine)
        if whisper_engine == "kotoba-whisper(日本語特化)":
            # 文字起こし用のモデルをメモリから外し、②以降のSwallowに空ける(16GBのMac向け)
            load_kotoba_pipeline.clear()
            import gc

            gc.collect()
            try:
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
        st.session_state.pop("corrected", None)
        st.session_state.pop("speaker_turns", None)
        st.session_state.pop("speaker_labeled_text_box", None)
        st.session_state.pop("chart_draft", None)

        if enable_diarization:
            if not hf_token:
                st.error("話者分離にはHugging Faceトークンが必要です。サイドバーに入力してください。")
            else:
                with st.spinner("話者分離を実行中...(初回はモデルのダウンロードで時間がかかります)"):
                    st.session_state.speaker_turns = diarize(audio_path, hf_token)

    if "result" in st.session_state:
        result = st.session_state.result
        segments = result["segments"]

        if "speaker_turns" in st.session_state:
            segments = assign_speakers(segments, st.session_state.speaker_turns)

        st.subheader("🔊 音声プレイヤー")
        st.audio(uploaded_file, start_time=int(st.session_state.play_start))
        st.caption("下のセグメント一覧の「▶ ここから再生」を押すと、該当箇所からここで再生されます。")

        st.subheader("📝 セグメント別 文字起こし(信頼度付き)")
        if "speaker_turns" not in st.session_state and enable_diarization:
            st.caption("※ 話者分離の結果がありません。①のボタンをもう一度押して実行してください。")
        for i, seg in enumerate(segments):
            badge = confidence_badge(seg.get("avg_logprob"))
            speaker_html = speaker_badge(seg["speaker"]) + "&nbsp;&nbsp;" if "speaker" in seg else ""
            col1, col2 = st.columns([9, 1])
            with col1:
                st.markdown(
                    f"{speaker_html}{badge}&nbsp;&nbsp;`{seg['start']:.1f}s - {seg['end']:.1f}s`&nbsp;&nbsp;{seg['text']}",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("▶ 再生", key=f"play_{i}"):
                    st.session_state.play_start = int(seg["start"])
                    st.rerun()

        correction_rules = parse_correction_dict(correction_dict_raw)
        dict_applied = {}
        dict_segments = [
            {**seg, "text": apply_correction_dict(seg["text"], correction_rules, dict_applied)} for seg in segments
        ]

        st.divider()
        st.subheader("①-2 用語の音照合")
        st.caption(
            "読みが医学用語リストの語に近いのに、表記が違う箇所を修正候補として出します。"
            "チェックを入れたものだけが②以降に反映されます。"
        )
        sound_terms = parse_term_list(sound_terms_raw) + parse_term_list(medical_terms)
        for name in selected_term_files:
            sound_terms += load_term_file(name)
        if sound_terms_file is not None:
            content = sound_terms_file.getvalue().decode("utf-8", errors="ignore")
            for line in content.splitlines():
                first = line.split(",")[0].strip().strip('"')
                if first:
                    sound_terms.append(first)
        sound_terms = tuple(dict.fromkeys(sound_terms))
        try:
            entries, index = build_sound_index(sound_terms)
            # 除外語(会話によく出る一般語)は、置き換え候補の元の語として扱わない
            stop_words = set(load_term_file("_除外語")) if os.path.exists(os.path.join(TERM_LIST_DIR, "_除外語.txt")) else set()
            term_set = set(sound_terms) | stop_words
            seg_matches = [find_sound_alike(seg["text"], entries, index, term_set) for seg in dict_segments]
        except ImportError:
            st.warning("音照合には pykakasi が必要です: `pip install pykakasi`")
            seg_matches = [[] for _ in dict_segments]

        pairs = {}
        for seg, matches in zip(dict_segments, seg_matches):
            for start, end, surface, term, ratio in matches:
                p = pairs.setdefault((surface, term), {"ratio": ratio, "count": 0, "context": ""})
                p["count"] += 1
                if not p["context"]:
                    p["context"] = (seg["text"][max(0, start - 8) : start], seg["text"][end : end + 8])

        with st.expander("音照合の診断情報(うまく候補が出ないとき用)"):
            try:
                import importlib.metadata as _md

                kakasi_ver = _md.version("pykakasi")
            except Exception:
                kakasi_ver = "不明"
            st.write(f"pykakasi: {kakasi_ver} / 用語数: {len(sound_terms)} / 索引に入った用語: {len(entries) if 'entries' in dir() else 0}")
            st.write(f"セグメント数: {len(dict_segments)} / 候補(重複込み): {sum(len(m) for m in seg_matches)}")
            if dict_segments:
                sample = dict_segments[0]["text"][:40]
                try:
                    st.write(f"1番目のセグメント: {sample}")
                    st.write("読み: " + " ".join(f"{o}({h})" for o, h in to_reading_tokens(sample)))
                except ImportError:
                    st.write("pykakasi が読み込めません")

        sound_key = abs(hash(result["text"])) % 10**8
        accepted_pairs = set()
        if not pairs:
            st.info("用語リストと音が近い表記ゆれは見つかりませんでした。")
        else:
            pair_list = sorted(pairs.items(), key=lambda kv: -kv[1]["ratio"])
            if st.button("すべて採用", key="sound_accept_all"):
                for i in range(len(pair_list)):
                    st.session_state[f"snd_{sound_key}_{i}"] = True
            for i, ((surface, term), info) in enumerate(pair_list):
                before, after = info["context"]
                label = (
                    f"…{md_escape(before)} :red[~~{md_escape(surface)}~~] → :green[**{md_escape(term)}**] "
                    f"{md_escape(after)}… (類似度 {info['ratio']:.2f}・{info['count']}箇所)"
                )
                if st.checkbox(label, key=f"snd_{sound_key}_{i}"):
                    accepted_pairs.add((surface, term))
            st.caption(f"採用 {len(accepted_pairs)} 件 / 全 {len(pair_list)} 件")

        dict_segments = [
            {**seg, "text": apply_sound_replacements(seg["text"], matches, accepted_pairs)}
            for seg, matches in zip(dict_segments, seg_matches)
        ]
        raw_text = "".join(seg["text"] for seg in dict_segments)
        backend_ready = bool(api_key) if llm_backend == "OpenAI API (GPT-4o)" else bool(local_model_path)
        backend_error = (
            "サイドバーにOpenAI APIキーを入力してください。"
            if llm_backend == "OpenAI API (GPT-4o)"
            else "サイドバーにSwallowモデルのパス(Ollamaの場合はモデル名)を入力してください。"
        )

        st.divider()
        st.subheader("② 話者推定(テキストベース)")
        st.caption(
            "音声の話者分離(pyannote)を使わず、発言内容の文脈だけから医師/患者を推測します。"
            "質問・所見の提示(医師)と症状の訴え(患者)というパターンの違いを手がかりにします。"
        )

        if st.button("話者を推定する"):
            if not backend_ready:
                st.error(backend_error)
            else:
                with st.spinner("話者を推定中..."):
                    st.session_state.speaker_labeled_text_box = infer_speakers_from_text(
                        dict_segments, llm_backend, api_key, local_model_path
                    )

        if dict_applied:
            st.caption(
                "📖 誤変換辞書で置き換えた箇所: "
                + "、".join(f"{k}({n}箇所)" for k, n in dict_applied.items())
            )

        if "speaker_labeled_text_box" in st.session_state:
            st.text_area(
                "話者推定付きの会話(必要なら手動で修正してください。修正内容は③以降に使われます)",
                height=200,
                key="speaker_labeled_text_box",
            )

        text_for_correction = st.session_state.get("speaker_labeled_text_box", raw_text)

        st.divider()
        st.subheader("③ LLMによる文脈補正")

        st.caption(
            "①-2の音照合で直しきれなかった誤変換をLLMに探させます。"
            "時間がかかるので、①-2で十分直っている場合はスキップできます。"
        )
        if st.button("③をスキップ(LLM補正なしで④へ進む)"):
            st.session_state.corrected = text_for_correction
            st.session_state.correction_base = text_for_correction
            st.session_state.correction_run = st.session_state.get("correction_run", 0) + 1
        if st.button("LLMで補正を実行"):
            if not backend_ready:
                st.error(backend_error)
            else:
                with st.spinner("補正中..."):
                    bar = st.progress(0.0, text="補正中...")
                    st.session_state.corrected = correct_with_llm(
                        text_for_correction,
                        llm_backend,
                        api_key,
                        local_model_path,
                        medical_terms,
                        progress=lambda i, n: bar.progress(i / n, text=f"補正中...({i}/{n} ブロック完了)"),
                        candidate_terms=sound_terms,
                    )
                    bar.empty()
                    st.session_state.correction_base = text_for_correction
                    # 新しい補正結果では採用・却下のチェックをやり直す
                    st.session_state.correction_run = st.session_state.get("correction_run", 0) + 1

        if "corrected" in st.session_state:
            st.markdown(
                "**差分ハイライト** (~~取り消し線~~=LLMが削除・置換した元の文字) "
                "(🟨 黄色=既存箇所の修正　🟥 赤=生の音声認識結果には無かった、LLMが新たに補った可能性のある箇所)"
            )
            html = highlight_diff_html(
                st.session_state.get("correction_base", text_for_correction), st.session_state.corrected
            )
            st.markdown(
                f"<div style='border:1px solid #ddd; padding:12px; border-radius:6px; line-height:1.8;'>{html}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("**修正の採用・却下**")
            st.caption(
                "LLMの修正は、チェックを入れたものだけが反映されます(初期状態はすべて却下=元のまま)。"
                "句読点だけの変更は自動で採用しています。"
            )
            base_text = st.session_state.get("correction_base", text_for_correction)
            parts = group_changes(base_text, st.session_state.corrected)
            run = st.session_state.get("correction_run", 0)
            change_ids = [i for i, p in enumerate(parts) if p[0] == "change" and not is_punctuation_only(p[1], p[2])]
            punct_ids = [i for i, p in enumerate(parts) if p[0] == "change" and is_punctuation_only(p[1], p[2])]

            if change_ids:
                bcols = st.columns([1, 1, 4])
                if bcols[0].button("すべて採用"):
                    for i in change_ids:
                        st.session_state[f"chg_{run}_{i}"] = True
                if bcols[1].button("すべて却下"):
                    for i in change_ids:
                        st.session_state[f"chg_{run}_{i}"] = False

            accepted = {i: True for i in punct_ids}
            for i in change_ids:
                _, old, new = parts[i]
                before = "".join(p[1] for p in parts[:i])[-12:]
                after = "".join(p[1] for p in parts[i + 1 :])[:12]
                old_md = f":red[~~{md_escape(old)}~~]" if old.strip() else ":gray[(なし)]"
                new_md = f":green[**{md_escape(new)}**]" if new.strip() else ":red[**(削除)**]"
                label = f"…{md_escape(before)} {old_md} → {new_md} {md_escape(after)}…"
                if not new.strip():
                    # 削除は情報の欠落につながる(例:「著明に」が消えて程度が分からなくなる)ため目立たせる
                    label = ":red[**⚠削除(情報が消えます)**] " + label
                accepted[i] = st.checkbox(label, key=f"chg_{run}_{i}")

            if not change_ids:
                st.info("句読点以外の修正はありませんでした。")
            else:
                n_ok = sum(1 for i in change_ids if accepted[i])
                st.caption(f"採用 {n_ok} 件 / 全 {len(change_ids)} 件(句読点のみの変更 {len(punct_ids)} 件は自動採用)")

            reviewed_text = build_reviewed_text(parts, accepted)

            st.divider()
            st.subheader("④ 不自然な表現のチェック")
            st.caption(
                "信頼度スコアでは検知できない「文法的には自然だが、医学記録としては非定型的な表現」"
                "(例:「右の奥が痛む」等)を、別の視点でLLMにチェックさせます。"
            )

            if st.button("不自然な表現をチェック"):
                if not backend_ready:
                    st.error(backend_error)
                else:
                    with st.spinner("チェック中..."):
                        st.session_state.vague_flags = check_vague_terms(
                            reviewed_text, llm_backend, api_key, local_model_path
                        )

            if "vague_flags" in st.session_state:
                st.warning(st.session_state.vague_flags)

            st.divider()
            st.subheader("⑤ 構造化チェックリスト")
            st.caption(
                "側性(右/左)や所見の有無のような、診断に直結する重要事項を自由文から抜き出し、"
                "個別に確認します。「不明」を選択肢として用意することで、自由文要約では防げなかった"
                "推測による埋め合わせ(confabulation)を避けます。"
            )

            if st.button("重要所見を抽出する"):
                if not backend_ready:
                    st.error(backend_error)
                else:
                    with st.spinner("抽出中..."):
                        st.session_state.findings = extract_structured_findings(
                            reviewed_text, llm_backend, api_key, local_model_path
                        )
                        st.session_state.findings_source = reviewed_text
                        # 抽出し直したら、前回の入力・チェック状態を引き継がない
                        st.session_state.findings_run = st.session_state.get("findings_run", 0) + 1

            if "findings" in st.session_state:
                if not st.session_state.findings:
                    st.info("抽出できる重要所見が見つかりませんでした。")
                all_confirmed = True
                fr = st.session_state.get("findings_run", 0)
                for i, finding in enumerate(st.session_state.findings):
                    conf_icon = "🟢" if finding.get("confidence") == "high" else "🔴"
                    cols = st.columns([3, 2, 2, 3, 1, 1])
                    excluded = st.session_state.get(f"finding_ex_{fr}_{i}", False)
                    with cols[0]:
                        item_name = finding.get("item", "項目不明")
                        # 印は入力欄の「上」に出す(下に出すと次の項目の印に見えてしまうため)
                        flags = f"項目名 {conf_icon}"
                        if is_paraphrased(item_name, st.session_state.get("findings_source", "")):
                            flags += " :orange[⚠会話に無い語]"
                        st.text_input(
                            flags,
                            item_name,
                            key=f"finding_item_{fr}_{i}",
                            disabled=excluded,
                            help="⚠会話に無い語: LLMが会話中の言葉を別の用語に言い換えた項目名です。意味が合うか確認してください。",
                        )
                    with cols[1]:
                        source = st.session_state.get("findings_source", "")
                        guessed = guess_soap_section(item_name, finding.get("value", ""), source)
                        st.selectbox(
                            "区分",
                            SOAP_SECTIONS,
                            index=SOAP_SECTIONS.index(guessed),
                            key=f"finding_sec_{fr}_{i}",
                            disabled=excluded,
                            help="カルテ下書きのS(患者の訴え)とO(所見・検査)のどちらに入れるか。会話のどちらの発言に根拠があるかで推定しています。",
                        )
                    with cols[2]:
                        laterality = normalize_laterality(finding.get("laterality"))
                        st.selectbox(
                            "側性",
                            LATERALITY_OPTIONS,
                            index=LATERALITY_OPTIONS.index(laterality),
                            key=f"finding_lat_{fr}_{i}",
                            disabled=excluded,
                        )
                    with cols[3]:
                        st.text_input(
                            "内容",
                            finding.get("value", "不明"),
                            key=f"finding_val_{fr}_{i}",
                            disabled=excluded,
                        )
                    with cols[4]:
                        confirmed = st.checkbox("確認済", key=f"finding_ok_{fr}_{i}", disabled=excluded)
                    with cols[5]:
                        st.checkbox("除外", key=f"finding_ex_{fr}_{i}", help="誤って抽出された項目をカルテに含めない")
                    all_confirmed = all_confirmed and (confirmed or excluded)

                if st.session_state.findings and not all_confirmed:
                    st.caption("⚠️ すべての項目に「確認済」か「除外」のチェックが入るまで、下の確定ボタンは有効になりません。")

            st.divider()
            st.subheader("⑥ 最終確認・編集")
            final_text = st.text_area(
                "補正後テキスト(採用した修正のみ反映。確定前に必ず目視確認・編集してください)",
                reviewed_text,
                height=200,
            )

            findings_confirmed = (
                "findings" not in st.session_state
                or not st.session_state.findings
                or all(
                    st.session_state.get(f"finding_ok_{fr_}_{i}", False)
                    or st.session_state.get(f"finding_ex_{fr_}_{i}", False)
                    for fr_ in [st.session_state.get("findings_run", 0)]
                    for i in range(len(st.session_state.findings))
                )
            )
            if st.button("この内容でカルテ下書きを作成する", disabled=not findings_confirmed, type="primary"):
                fr = st.session_state.get("findings_run", 0)
                confirmed_findings = [
                    {
                        "item": st.session_state.get(f"finding_item_{fr}_{i}", f.get("item", "")),
                        "laterality": st.session_state.get(f"finding_lat_{fr}_{i}", "不明"),
                        "value": st.session_state.get(f"finding_val_{fr}_{i}", f.get("value", "")),
                        "section": st.session_state.get(f"finding_sec_{fr}_{i}", SOAP_SECTIONS[0]),
                    }
                    for i, f in enumerate(st.session_state.get("findings", []))
                    if not st.session_state.get(f"finding_ex_{fr}_{i}", False)
                ]
                st.session_state.chart_draft = build_chart_draft(confirmed_findings)

            if "chart_draft" in st.session_state:
                st.divider()
                st.subheader("⑦ カルテ下書き(SOAP)")
                st.caption(
                    "⑤で確認した所見を、区分(S/O)ごとに要約として並べたものです"
                    "(確認していない言い換えが入らないよう、LLMで要約し直してはいません)。"
                    "A(評価)・P(計画)は医師が記入してください。"
                )
                draft = st.text_area("カルテ下書き", st.session_state.chart_draft, height=400)
                st.download_button(
                    "テキストファイルとして保存",
                    draft,
                    file_name="karte_draft.txt",
                    mime="text/plain",
                )
else:
    st.info("まずは音声ファイル(mp3/wav/m4a)をアップロードしてください。")
