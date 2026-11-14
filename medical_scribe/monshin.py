"""
問診アシスタントの処理部分(Streamlitに依存しない)。

医師が診察中に使う: 医師が患者さんから聞いた答えを入力すると、LLMが
  - 今考えられる鑑別診断(可能性の高さ付き)
  - 見逃してはいけない疾患
  - 次に聞くとよい質問(どの疾患を見分けるためか付き)
を返す。これを繰り返し、最後に問診のまとめと鑑別・次の診察や検査を出す。

- 診断を下すものではなく、医師の問診を補助する参考情報。
- 危険な兆候は、LLMとは別にキーワードでも確認する(ddx_agents.rule_red_flags)。
- LLMの呼び出しは llm(prompt, system) の形で受け取るので、画面からも評価スクリプトからも使える。
"""

import json
import re
import urllib.error
import urllib.request

import ddx_agents

OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM = (
    "あなたは外来で医師の問診を補助するAIです。医師が患者さんから聞いた情報をもとに、"
    "鑑別診断を考え、診断を絞り込むために次に聞くべきことを提案します。"
    "最終的な判断は医師が行います。"
)


# ---------- 会話の記録 ----------


def new_case(patient: str, chief_complaint: str) -> dict:
    return {"patient": patient.strip(), "chief_complaint": chief_complaint.strip(), "exchanges": []}


def add_exchange(case: dict, question: str, answer: str, source: str = "AIの提案") -> None:
    """医師が聞いた質問と患者さんの答えを記録する。答えが空なら記録しない。"""
    if question.strip() and answer.strip():
        case["exchanges"].append({"question": question.strip(), "answer": answer.strip(), "source": source})


def history_text(case: dict) -> str:
    """LLMに渡す、これまでに分かったこと。"""
    lines = []
    if case["patient"]:
        lines.append(f"患者: {case['patient']}")
    lines.append(f"主訴: {case['chief_complaint']}")
    for i, e in enumerate(case["exchanges"], 1):
        lines.append(f"[{i}] 質問: {e['question']}\n    答え: {e['answer']}")
    return "\n".join(lines)


def red_flags(case: dict) -> list:
    """キーワードによる危険な兆候(LLMを使わない)。答えと主訴だけを見る(質問文は見ない)。"""
    text = "\n".join([case["chief_complaint"]] + [e["answer"] for e in case["exchanges"]])
    return ddx_agents.rule_red_flags(text)


# ---------- 次の質問の提案 ----------


def next_step_prompt(case: dict, n_questions: int = 3) -> str:
    asked = "\n".join(f"- {e['question']}" for e in case["exchanges"] if e["question"] != "(医師のメモ)")
    asked = asked or "(まだありません)"
    return f"""次の患者さんの問診を進めています。これまでに分かったことから、鑑別診断と、次に聞くべき質問を考えてください。

出力は次の形だけにしてください(前置きや説明、見出しは不要です)。
鑑別: 疾患名(高)、疾患名(中)、疾患名(低)
見逃し注意: 可能性は低くても見逃すと危険な疾患を「、」区切りで。なければ「なし」
質問: 患者さんへの質問文｜この質問で見分けたいこと
質問: 患者さんへの質問文｜この質問で見分けたいこと

決まりごと:
- 鑑別は可能性の高い順に3〜5個。(高)(中)(低)で可能性の程度を付けてください。
- 質問は最大{n_questions}個。診断を最も大きく絞り込めるもの、危険な疾患を除外するものを優先してください。
- 質問は、医師がそのまま患者さんに読み上げられる、短く分かりやすい話し言葉にしてください。1つの質問で聞くことは1つにしてください。
- すでに聞いた質問は繰り返さないでください。
- 身体診察で確かめるべきことがあれば、質問の代わりに「診察: 確かめる所見｜見分けたいこと」と書いてかまいません。
- 書かれていない症状や所見を、あるものとして扱わないでください。

【これまでに分かったこと】
{history_text(case)}

【すでに聞いた質問】
{asked}
"""


def _split_items(text: str) -> list:
    return [x.strip(" *・") for x in re.split(r"[、,，]", text) if x.strip(" *・")]


def parse_next_step(text: str) -> dict:
    """提案の出力を読み取る。返り値: {"ddx": [{"name", "level"}], "must_not_miss": [...], "questions": [{"kind","text","purpose"}]}"""
    ddx, must_not_miss, questions = [], [], []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*・]|\d+[.)．])\s*", "", line.replace("**", "").strip())
        m = re.match(r"(鑑別|見逃し注意|質問|診察)\s*[:：]\s*(.+)", line)
        if not m:
            continue
        label, body = m.group(1), m.group(2).strip()
        if label == "鑑別":
            for item in _split_items(body):
                lv = re.search(r"[（(]\s*(高|中|低)\s*[)）]", item)
                name = re.sub(r"[（(]\s*(高|中|低)\s*[)）]", "", item).strip()
                if name:
                    ddx.append({"name": name, "level": lv.group(1) if lv else ""})
        elif label == "見逃し注意":
            must_not_miss = [x for x in _split_items(body) if x not in ("なし", "特になし")]
        else:
            text_, _, purpose = body.partition("｜") if "｜" in body else body.partition("|")
            if text_.strip():
                questions.append({"kind": label, "text": text_.strip(), "purpose": purpose.strip()})
    return {"ddx": ddx, "must_not_miss": must_not_miss, "questions": questions}


def suggest_next(case: dict, llm, n_questions: int = 3) -> dict:
    raw = llm(next_step_prompt(case, n_questions), SYSTEM)
    result = parse_next_step(raw)
    result["raw"] = raw
    return result


# ---------- まとめ ----------


def summary_prompt(case: dict) -> str:
    return f"""次の問診の記録をもとに、医師向けのまとめを書いてください。

次の4つの見出しを付けて、日本語で書いてください。
## 問診のまとめ
(主訴と現病歴を、発症時期・経過・性状・部位・程度・増悪寛解因子・随伴症状の順に、記録にある範囲で簡潔に。既往歴・内服薬・アレルギーなども記録にあれば書く)
## 鑑別診断
(可能性の高い順に3〜5個。それぞれ「支持する所見」と「合わない所見・まだ確かめていないこと」を添える)
## 見逃してはいけない疾患
(可能性が低くても除外すべき疾患と、除外に必要なこと)
## 次に行う診察・検査
(結果によって診断や方針が変わるものを優先して)

記録に書かれていない症状や所見を作らないでください。分からないことは「未確認」と書いてください。

【問診の記録】
{history_text(case)}
"""


def summarize(case: dict, llm) -> str:
    return llm(summary_prompt(case), SYSTEM).strip()


def case_as_text(case: dict, summary: str = "") -> str:
    """保存用のテキスト(問診の記録とまとめ)。"""
    lines = ["# 問診の記録", history_text(case)]
    flags = red_flags(case)
    if flags:
        lines += ["", "危険な兆候(キーワードで検出): " + "、".join(flags)]
    if summary:
        lines += ["", summary]
    lines += ["", "※AIによる参考情報です。診断は医師が行ってください。"]
    return "\n".join(lines)


# ---------- Ollama ----------


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.split("<think>")[0].strip()


def ollama_llm(model: str, think: str = "low", num_ctx: int = 16384, max_tokens: int = 8192, timeout: int = 300):
    """llm(prompt, system) を返す。データはMacの外に出ない。
    gpt-oss は考える量(low/medium/high)、Qwen3 は true/false を think で指定する。"""

    def call(prompt: str, system: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": True,
            "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": max_tokens},
        }
        if think:
            payload["think"] = {"true": True, "false": False}.get(think, think)
        req = urllib.request.Request(
            OLLAMA_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        parts = []
        try:
            # 少しずつ受け取るので、考える時間が長くても途中でタイムアウトしない
            with urllib.request.urlopen(req, timeout=timeout) as res:
                for line in res:
                    if not line.strip():
                        continue
                    msg = json.loads(line.decode("utf-8"))
                    parts.append(msg.get("message", {}).get("content", ""))
                    if msg.get("done"):
                        break
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
        return strip_thinking("".join(parts))

    return call
