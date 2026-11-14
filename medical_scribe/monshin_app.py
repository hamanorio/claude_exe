"""
問診アシスタント(医師が診察中に使う)

医師が患者さんから聞いた答えを入力すると、AIが鑑別診断と次に聞くとよい質問を示します。
これを繰り返して診断を絞り込み、最後に問診のまとめを作ります。
LLMはMac上のOllama(初期値 gpt-oss:20b)で動かすので、入力した内容はMacの外に出ません。

使い方:
    cd medical_scribe
    streamlit run monshin_app.py
"""

import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import monshin  # noqa: E402

st.set_page_config(page_title="問診アシスタント", page_icon="🩺", layout="centered")

LEVEL_COLOR = {"高": "red", "中": "orange", "低": "gray"}


def get_llm():
    model = st.session_state.model
    # Qwen3 は考える量を段階で指定できないので、考えずに速く答えさせる
    think = "false" if model.lower().startswith("qwen3") else st.session_state.think
    return monshin.ollama_llm(model, think=think)


def run_suggest() -> None:
    """次の質問を提案させる。失敗したら画面にエラーを出す。"""
    start = time.time()
    try:
        with st.spinner(f"{st.session_state.model}が考えています..."):
            st.session_state.suggestion = monshin.suggest_next(
                st.session_state.case, get_llm(), st.session_state.n_questions
            )
        st.session_state.suggest_seconds = time.time() - start
    except Exception as e:
        st.session_state.suggestion = None
        st.error(
            f"{st.session_state.model}を呼び出せませんでした({e})。Ollamaが起動しているか、"
            "サイドバーのモデル名が `ollama list` の表示と同じか確認してください。"
        )


def record_answers(rnd: int, suggestion) -> int:
    """入力された答えを記録する。記録した数を返す。空欄の質問は記録しない(未確認のまま)。"""
    n = 0
    case = st.session_state.case
    for i, q in enumerate((suggestion or {}).get("questions", [])):
        answer = st.session_state.get(f"ans_{rnd}_{i}", "")
        if answer.strip():
            label = q["text"] if q["kind"] == "質問" else f"(診察){q['text']}"
            monshin.add_exchange(case, label, answer)
            n += 1
    own_q = st.session_state.get(f"own_q_{rnd}", "")
    own_a = st.session_state.get(f"own_a_{rnd}", "")
    if own_q.strip() and own_a.strip():
        monshin.add_exchange(case, own_q, own_a, source="医師")
        n += 1
    elif own_a.strip():
        monshin.add_exchange(case, "(医師のメモ)", own_a, source="医師")
        n += 1
    return n


# ---------------- サイドバー ----------------
with st.sidebar:
    st.header("設定")
    st.text_input(
        "モデル(Ollama)",
        "gpt-oss:20b",
        key="model",
        help="`ollama list` で表示される名前。NEJM症例の比較で最も成績がよかった gpt-oss:20b を初期値にしています。",
    )
    st.selectbox(
        "考える量",
        ["low", "medium", "high"],
        index=1,
        key="think",
        help="gpt-oss が答える前に考える量。多いほど鑑別が丁寧になりますが、時間がかかります。"
        "Qwen3 を選んだ場合は、この設定にかかわらず考えずに答えます。",
    )
    st.slider("1回に提案する質問の数", 1, 5, 3, key="n_questions")
    st.divider()
    if st.button("新しい患者を始める"):
        for k in list(st.session_state.keys()):
            if k not in ("model", "think", "n_questions"):
                del st.session_state[k]
        st.rerun()

# ---------------- 本体 ----------------
st.title("🩺 問診アシスタント")
st.caption(
    "AIの提案は診断ではなく、問診を進めるための参考情報です。質問の取捨選択と最終的な判断は医師が行ってください。"
)

if "case" not in st.session_state:
    st.subheader("① 患者さんの基本情報と主訴")
    with st.form("start"):
        patient = st.text_input("年齢・性別・わかっている既往など", placeholder="例: 45歳女性。高血圧で内服中")
        chief = st.text_area("主訴(患者さんの言葉で)", placeholder="例: 3日前からの発熱とのどの痛み", height=80)
        started = st.form_submit_button("問診を始める")
    if started:
        if not chief.strip():
            st.error("主訴を入力してください。")
        else:
            st.session_state.case = monshin.new_case(patient, chief)
            st.session_state.round = 0
            run_suggest()
            st.rerun()
    st.stop()

case = st.session_state.case
rnd = st.session_state.get("round", 0)
suggestion = st.session_state.get("suggestion")

# 危険な兆候(キーワード・LLMを使わない)は、常にいちばん上に出す
flags = monshin.red_flags(case)
if flags:
    st.error("⚠ 危険な兆候の可能性(キーワードで検出・誤検出もあり): " + "、".join(flags))

st.markdown(f"**患者**: {case['patient'] or '(未入力)'}　**主訴**: {case['chief_complaint']}")
st.caption(f"これまでに記録した答え: {len(case['exchanges'])}件")

if suggestion is None:
    if st.button("もう一度AIに提案させる"):
        run_suggest()
        st.rerun()
else:
    st.subheader("② 今考えられる鑑別診断")
    if suggestion["ddx"]:
        st.markdown(
            "　".join(
                f":{LEVEL_COLOR.get(d['level'], 'gray')}[**{d['name']}**" + (f"({d['level']})" if d["level"] else "") + "]"
                for d in suggestion["ddx"]
            )
        )
    else:
        st.caption("鑑別を読み取れませんでした(答え直しも試しました)。質問への答えが増えると出やすくなります。")
    if suggestion["must_not_miss"]:
        st.warning("見逃し注意: " + "、".join(suggestion["must_not_miss"]))

    st.subheader("③ 次に聞くとよいこと")
    st.caption("患者さんに聞いて、答えを入力してください。聞かなかった質問は空欄のままで構いません(未確認として扱います)。")
    if suggestion.get("fallback"):
        st.caption("AIが質問を出せなかったので、どの主訴にも使える基本の質問を出しています。")
    for i, q in enumerate(suggestion["questions"]):
        tag = "🩺 診察" if q["kind"] == "診察" else "💬 質問"
        st.markdown(f"{tag} **{q['text']}**")
        if q["purpose"]:
            st.caption(f"見分けたいこと: {q['purpose']}")
        st.text_input("答え", key=f"ans_{rnd}_{i}", label_visibility="collapsed", placeholder="答え(未確認なら空欄)")

    with st.expander("ご自身で聞いたこと・分かったことを追加"):
        st.text_input("質問(任意)", key=f"own_q_{rnd}", placeholder="例: 周りに同じ症状の人はいますか")
        st.text_area("答え・所見", key=f"own_a_{rnd}", placeholder="例: 職場で同じ症状の人が2人いる", height=70)

    cols = st.columns(2)
    if cols[0].button("答えを記録して次の提案へ", type="primary"):
        if record_answers(rnd, suggestion) == 0:
            st.warning("答えが入力されていません。少なくとも1つ入力するか、「まとめる」を押してください。")
        else:
            st.session_state.round = rnd + 1
            st.session_state.pop("summary", None)
            run_suggest()
            st.rerun()
    if cols[1].button("ここまでをまとめる"):
        record_answers(rnd, suggestion)
        st.session_state.round = rnd + 1
        try:
            with st.spinner("まとめています..."):
                st.session_state.summary = monshin.summarize(case, get_llm())
        except Exception as e:
            st.error(f"まとめの作成に失敗しました({e})。")
        st.rerun()

    if "suggest_seconds" in st.session_state:
        st.caption(f"提案にかかった時間: {st.session_state.suggest_seconds:.0f}秒")
    with st.expander("AIの出力全文"):
        st.text(suggestion.get("raw", ""))

with st.expander(f"問診の記録({len(case['exchanges'])}件)", expanded=False):
    if case["exchanges"]:
        for i, e in enumerate(case["exchanges"], 1):
            st.markdown(f"{i}. {e['question']}　→ **{e['answer']}**" + ("　:gray[(医師)]" if e["source"] == "医師" else ""))
    else:
        st.caption("まだありません。")

if "summary" in st.session_state:
    st.divider()
    st.subheader("④ 問診のまとめ")
    st.markdown(st.session_state.summary)
    st.download_button(
        "まとめをテキストで保存",
        monshin.case_as_text(case, st.session_state.summary),
        file_name="問診のまとめ.txt",
        mime="text/plain",
    )
    st.caption("保存したファイルには問診の内容が含まれます。院内の取り扱いルールに沿って保管してください。")
