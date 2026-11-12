"""
逐次診断(研究用の試作): 病歴の前半だけを見た「診断医AI」が、質問や検査を「情報係AI」に
依頼しながら、少しずつ情報を集めて診断に迫る。

- 情報係AIは症例の全文を持ち、聞かれたことだけを、全文に書かれている範囲で答える。
  書かれていなければ「記載なし」と答え、診断名や解釈は教えない。
- 診断医AIは1回ごとに「質問」「検査」「診断」のどれか1つを選ぶ。上限回数に達したら診断させる。
- 何回で診断したか、検査を何件依頼したかも記録する(正確さに加えて効率も比べるため)。

LLMの呼び出しは引数 llm(prompt, system) で受け取るので、評価スクリプトからも使える。
"""

import re

DOCTOR_SYSTEM = "あなたは外来で診断を担当する医師です。限られた情報から、必要な情報を選んで集め、診断に迫ります。"

GATEKEEPER_SYSTEM = (
    "あなたは症例の記録を管理する係です。医師に聞かれたことに、記録に書かれている内容だけで答えます。"
)


def doctor_prompt(initial: str, exchanges: list, turn: int, max_turns: int) -> str:
    history = "\n".join(
        f"[{i + 1}] {e['kind']}: {e['request']}\n    → {e['answer']}" for i, e in enumerate(exchanges)
    )
    last = turn >= max_turns
    if last:
        action = "これが最後の回です。今ある情報で、最も可能性が高い診断を1つ「診断: 病名」で答えてください。"
    else:
        action = f"""次のどれか1つを行ってください(残り{max_turns - turn + 1}回)。
- 追加で知りたい病歴や身体所見があれば「質問: 知りたいこと」
- 必要な検査があれば「検査: 検査名、検査名」(1回に3つまで)
- 診断に十分な情報が得られたら「診断: 病名」
検査は、結果によって診断や方針が変わりうるものだけを選び、不要な検査は避けてください。"""
    return f"""{action}

その前に1行目に「鑑別: 疾患名、疾患名、疾患名」として、今考えている上位3つの疾患を書いてください。
出力はこの2行だけにしてください。

【最初に得られた情報】
{initial}

【これまでの質問・検査と結果】
{history or "(まだありません)"}
"""


def gatekeeper_prompt(full_case: str, kind: str, request: str) -> str:
    return f"""医師から次の{kind}の依頼がありました。症例の記録に書かれている内容だけで答えてください。

- 依頼された項目ごとに、記録に書かれている結果や所見をそのまま簡潔に答えてください。
- 記録に書かれていない項目は「記載なし」と答えてください。推測で作らないでください。
- 診断名、診断の推測、所見の解釈は答えないでください(医師が自分で考えます)。

【医師の依頼】
{kind}: {request}

【症例の記録(医師には見えていません)】
{full_case}
"""


def parse_doctor(text: str) -> dict:
    """診断医AIの出力から、鑑別と行動(質問・検査・診断)を読み取る。"""
    ddx_m = re.search(r"鑑別\s*[:：]\s*(.+)", text)
    ddx = [d.strip(" *") for d in re.split(r"[、,，]", ddx_m.group(1))] if ddx_m else []
    for kind in ("診断", "検査", "質問"):
        m = re.search(rf"(?:^|\n)\s*[*・-]*\s*{kind}\s*[:：]\s*(.+)", text)
        if m:
            return {"kind": kind, "content": m.group(1).strip(" *"), "ddx": [d for d in ddx if d]}
    return {"kind": "", "content": "", "ddx": [d for d in ddx if d]}


def run_sequential(initial: str, full_case: str, doctor_llm, gatekeeper_llm, max_turns: int = 8, progress=None) -> dict:
    exchanges, ddx, final = [], [], ""
    for turn in range(1, max_turns + 1):
        if progress:
            progress(turn, max_turns)
        raw = doctor_llm(doctor_prompt(initial, exchanges, turn, max_turns), DOCTOR_SYSTEM)
        action = parse_doctor(raw)
        ddx = action["ddx"] or ddx
        if action["kind"] == "診断":
            final = action["content"]
            break
        if action["kind"] in ("検査", "質問"):
            answer = gatekeeper_llm(gatekeeper_prompt(full_case, action["kind"], action["content"]), GATEKEEPER_SYSTEM)
            exchanges.append({"kind": action["kind"], "request": action["content"], "answer": answer.strip()})
        else:
            # 形式どおりに答えなかった回は、そのまま次の回へ(回数は消費する)
            exchanges.append({"kind": "不明", "request": raw.strip()[:200], "answer": "(形式どおりの依頼ではありませんでした)"})
    if not final and ddx:
        final = ddx[0]  # 上限まで診断しなかったときは、最後の鑑別の1位を診断とみなす
    return {
        "initial": initial,
        "exchanges": exchanges,
        "ddx": ddx,
        "final": final,
        "n_questions": sum(e["kind"] == "質問" for e in exchanges),
        "n_tests": sum(len([t for t in re.split(r"[、,，]", e["request"]) if t.strip()]) for e in exchanges if e["kind"] == "検査"),
        "turns": len(exchanges) + (1 if final else 0),
    }


def sequential_as_text(result: dict) -> str:
    """評価スクリプト(eval_ddx.py)の採点で読める形にする。"""
    lines = ["## 鑑別診断"] + [f"{i + 1}. {d}" for i, d in enumerate(result["ddx"])]
    lines += ["", "## 最終診断", result["final"], ""]
    lines += [
        "## 検討の記録",
        f"質問 {result['n_questions']}回・検査 {result['n_tests']}件・{result['turns']}手で診断",
        "【最初に得られた情報】",
        result["initial"],
    ]
    for i, e in enumerate(result["exchanges"], 1):
        lines += [f"[{i}] {e['kind']}: {e['request']}", f"    → {e['answer']}"]
    return "\n".join(lines)
