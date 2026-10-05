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
    3. pip install streamlit mlx-whisper openai
    4. streamlit run app.py
       サイドバーで「Whisperエンジン = mlx-whisper」「LLM = ローカル(Ollama・Mac向け)」を選び、
       Ollamaのモデル名(`ollama list` で表示される名前)を入力する
"""

import difflib
import json
import re
import tempfile
import urllib.request

import streamlit as st
from openai import OpenAI

# torch / transformers / whisper は使うバックエンドを選んだときだけ読み込む
# (Mac + Ollama + mlx-whisper の構成ではインストール不要にするため)

OLLAMA_URL = "http://localhost:11434/api/chat"

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


def generate_with_ollama(prompt: str, model_name: str, json_mode: bool = False) -> str:
    """Mac上のOllama(量子化済みSwallow)で生成する。データはMacの外に出ない。"""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "repeat_penalty": 1.15, "num_ctx": 8192},
    }
    if json_mode:
        payload["format"] = "json"
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as res:
        body = json.loads(res.read().decode("utf-8"))
    return body["message"]["content"].strip()


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

    return whisper.load_model(model_size)


def transcribe(audio_path: str, medical_terms: str, model_size: str, engine: str = "openai-whisper"):
    initial_prompt = f"これは医師と患者の診察会話です。次のような医学用語が含まれます: {medical_terms}"
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


def infer_speakers_from_text(raw_text: str, backend: str, api_key: str, local_model_path: str) -> str:
    """音声レベルの話者分離を使わず、発言内容の文脈だけから医師/患者を推測する。
    質問・所見の提示(医師)と症状の訴え(患者)というパターンの違いを手がかりにする。
    """
    prompt = f"""以下は、医師と患者の診察会話の文字起こしです。話者のラベルはついていません。
文脈(質問している/答えている、症状を訴えている/所見を述べている等)から、各発言が「医師」か「患者」のどちらの発言かを推測し、
発言ごとに分けて以下の形式で出力してください。
テキストの内容自体は変更せず、話者の割り当てと改行のみ行ってください。

【文字起こし】
{raw_text}

【話者推定付きの会話】
"""
    return run_llm(prompt, backend, api_key, local_model_path)


def correct_with_llm(raw_text: str, backend: str, api_key: str, local_model_path: str) -> str:
    prompt = f"""以下は、医師と患者の診察会話を音声認識(Whisper)で文字起こししたテキストです。
話者ラベル(医師:/患者:)が付いている場合は、そのラベルと発言の区切りを維持してください。
音声認識特有の誤変換(医学用語が似た音の別の言葉に変換されている等)が含まれている可能性があります。
文脈から医学的に正しいと考えられる形に修正してください。
ただし、聞き取れなかった可能性がある情報を、典型的な症例パターンから推測して新たに追加することは絶対にしないでください。
意味が不明瞭、または欠落している可能性がある箇所は、無理に埋めず、そのまま残してください。

【音声認識結果】
{raw_text}

【修正後のテキスト】
"""
    return run_llm(prompt, backend, api_key, local_model_path)


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


def confidence_badge(avg_logprob: float) -> str:
    if avg_logprob < -1.0:
        return "🔴 低信頼"
    elif avg_logprob < -0.5:
        return "🟡 中信頼"
    else:
        return "🟢 高信頼"


def check_vague_terms(text: str, backend: str, api_key: str, local_model_path: str) -> str:
    prompt = f"""以下はカルテ下書きです。
身体の部位や症状の表現の中で、医学的な記録としては不自然に曖昧・口語的な表現
(例:「奥」「なんか」「あのへん」「変な感じ」等、標準的な解剖学的・臨床的用語になっていないもの)
が使われている箇所があれば、音声認識の誤りの可能性があるとして指摘してください。
断定せず、「要確認」として該当箇所を引用した上で挙げてください。
該当箇所が無い場合は「特に気になる曖昧な表現はありません」とだけ答えてください。

【カルテ下書き】
{text}

【要確認フラグ】
"""
    return run_llm(prompt, backend, api_key, local_model_path)


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


def extract_structured_findings(text: str, backend: str, api_key: str, local_model_path: str) -> list:
    """自由文のカルテ下書きから、診断に直結する重要所見を構造化して抽出する。
    側性(左右)や所見の有無を、自由文に埋め込まず個別項目として扱うことで、
    LLMが「不明」を正直に選べるようにし(自由文要約では確認済みの通り機能しなかった)、
    医師が1項目ずつ確認する運用を可能にする。
    """
    prompt = f"""以下はカルテ下書きです。診断・治療方針に直結する重要な所見を構造化して抽出してください。

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
    return data.get("findings", [])


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
                f"<span style='background-color:#fff3b0' title='音声認識結果が修正された箇所'>{seg}</span>"
            )
        # tag == "delete" は補正後テキストには現れないので無視
    return "".join(html_parts)


# ---------------- UI ----------------

st.title("🩺 診察音声 → カルテ下書き プロトタイプ")
st.caption("AIの出力をそのまま信じず、怪しい箇所は音声に戻って確認する、という設計の検証用プロトタイプです。")

with st.sidebar:
    st.header("設定")
    whisper_engine = st.radio(
        "Whisperエンジン",
        ["openai-whisper", "mlx-whisper"],
        help="Mac(Apple Silicon)では mlx-whisper の方が大幅に速く動きます。",
    )
    whisper_sizes = ["medium", "large-v3", "large-v3-turbo"] if whisper_engine == "mlx-whisper" else ["medium", "large-v3"]
    model_size = st.selectbox("Whisperモデルサイズ", whisper_sizes, index=len(whisper_sizes) - 1)
    medical_terms = st.text_area(
        "医学用語ヒント(initial_prompt用)",
        "呂律、構音障害、顔面神経麻痺、心房細動、不整脈、脂質異常症、浸潤影、湿性ラ音",
        height=100,
    )
    st.divider()
    llm_backend = st.radio(
        "②③④⑤で使うLLM",
        ["OpenAI API (GPT-4o)", "ローカル(Swallow)", "ローカル(Ollama・Mac向け)"],
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
        with st.spinner("Whisperで文字起こし中..."):
            st.session_state.result = transcribe(audio_path, medical_terms, model_size, whisper_engine)
        st.session_state.pop("corrected", None)
        st.session_state.pop("speaker_turns", None)

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
            badge = confidence_badge(seg["avg_logprob"])
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

        raw_text = result["text"]
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
                    st.session_state.speaker_labeled_text = infer_speakers_from_text(
                        raw_text, llm_backend, api_key, local_model_path
                    )

        if "speaker_labeled_text" in st.session_state:
            st.text_area(
                "話者推定付きの会話(必要なら手動で修正してください)",
                st.session_state.speaker_labeled_text,
                height=200,
                key="speaker_labeled_text_box",
            )

        text_for_correction = st.session_state.get("speaker_labeled_text", raw_text)

        st.divider()
        st.subheader("③ LLMによる文脈補正")

        if st.button("LLMで補正を実行"):
            if not backend_ready:
                st.error(backend_error)
            else:
                with st.spinner("補正中..."):
                    st.session_state.corrected = correct_with_llm(
                        text_for_correction, llm_backend, api_key, local_model_path
                    )

        if "corrected" in st.session_state:
            st.markdown(
                "**差分ハイライト** "
                "(🟨 黄色=既存箇所の修正　🟥 赤=生の音声認識結果には無かった、LLMが新たに補った可能性のある箇所)"
            )
            html = highlight_diff_html(text_for_correction, st.session_state.corrected)
            st.markdown(
                f"<div style='border:1px solid #ddd; padding:12px; border-radius:6px; line-height:1.8;'>{html}</div>",
                unsafe_allow_html=True,
            )

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
                            st.session_state.corrected, llm_backend, api_key, local_model_path
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
                            st.session_state.corrected, llm_backend, api_key, local_model_path
                        )

            if "findings" in st.session_state:
                if not st.session_state.findings:
                    st.info("抽出できる重要所見が見つかりませんでした。")
                all_confirmed = True
                for i, finding in enumerate(st.session_state.findings):
                    conf_icon = "🟢" if finding.get("confidence") == "high" else "🔴"
                    cols = st.columns([3, 2, 3, 1])
                    with cols[0]:
                        st.markdown(f"**{finding.get('item', '項目不明')}** {conf_icon}")
                    with cols[1]:
                        laterality = normalize_laterality(finding.get("laterality"))
                        st.selectbox(
                            "側性",
                            LATERALITY_OPTIONS,
                            index=LATERALITY_OPTIONS.index(laterality),
                            key=f"finding_lat_{i}",
                            label_visibility="collapsed",
                        )
                    with cols[2]:
                        st.text_input(
                            "内容",
                            finding.get("value", "不明"),
                            key=f"finding_val_{i}",
                            label_visibility="collapsed",
                        )
                    with cols[3]:
                        confirmed = st.checkbox("確認済", key=f"finding_ok_{i}")
                        all_confirmed = all_confirmed and confirmed

                if st.session_state.findings and not all_confirmed:
                    st.caption("⚠️ すべての項目の「確認済」にチェックが入るまで、下の確定ボタンは有効になりません。")

            st.divider()
            st.subheader("⑥ 最終確認・編集")
            final_text = st.text_area(
                "補正後テキスト(確定前に必ず目視確認・編集してください)",
                st.session_state.corrected,
                height=200,
            )

            findings_confirmed = (
                "findings" not in st.session_state
                or not st.session_state.findings
                or all(
                    st.session_state.get(f"finding_ok_{i}", False)
                    for i in range(len(st.session_state.findings))
                )
            )
            st.button(
                "この内容でカルテに確定する(プロトタイプでは保存処理なし)",
                disabled=not findings_confirmed,
            )
else:
    st.info("まずは音声ファイル(mp3/wav/m4a)をアップロードしてください。")
