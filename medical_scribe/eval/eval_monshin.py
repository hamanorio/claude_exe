"""
問診アシスタント(monshin.py)の評価スクリプト(Mac上のOllamaで実行。データは外部に送らない)

架空の症例(monshin_cases.json)を「患者役」のLLMに持たせ、問診アシスタントの質問に、
症例の記録に書かれている範囲で患者として答えさせます。決めた回数だけ問診を進めたあと、
まとめを作らせ、正しい診断が鑑別に挙がったかを数えます。

使い方:
  python3 eval_monshin.py --out monshin_eval.csv
  python3 eval_monshin.py --rounds 4 --think medium --out monshin_eval.csv

  指示文を直す前と比べる(--compare に、直す前のコミットを指定):
  python3 eval_monshin.py --compare 7047152 --out monshin_compare.csv

数えるもの(症例ごと・指示文ごと):
  - 最後の提案の鑑別に正解があるか
  - まとめの「鑑別診断」「見逃してはいけない疾患」に正解があるか
  - 正解が初めて鑑別に挙がったのは何回目の提案か
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import monshin  # noqa: E402

PATIENT_SYSTEM = "あなたは外来を受診した患者さんの役を演じます。自分の症状について、医師の質問に答えます。"


def patient_prompt(case: dict, question: str) -> str:
    return f"""あなたは次の患者さんです。医師の質問に、患者さんとして短く答えてください。

決まりごと:
- 答えは、下の【患者さんの情報】に書かれている内容だけを使ってください。
- 書かれていないことを聞かれたら、「特にないです」「わかりません」のように答えてください。作り話はしないでください。
- 病名や医学的な解釈は言わないでください(患者さんは自分の病名を知りません)。
- 聞かれたことだけに答え、聞かれていないことは話さないでください。
- 診察(身体所見の確認)を求められたら、【患者さんの情報】にある所見をそのまま答え、書かれていなければ「特に異常なし」と答えてください。

【患者さんの情報】
{case['patient']}。主訴: {case['chief_complaint']}
{case['record']}

【医師の質問】
{question}
"""


def load_monshin_at(commit: str):
    """指定したコミットの monshin.py を読み込む(指示文を直す前と比べるため)。"""
    src = subprocess.run(
        ["git", "show", f"{commit}:medical_scribe/monshin.py"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    module = types.ModuleType(f"monshin_{commit}")
    module.__file__ = os.path.join(ROOT, "monshin.py")
    exec(compile(src, f"monshin@{commit}", "exec"), module.__dict__)
    return module


def normalize(text: str) -> str:
    return re.sub(r"[\s・･()（）「」、。,]", "", text)


def has_gold(text: str, gold: list) -> bool:
    t = normalize(text)
    return any(normalize(g) in t for g in gold)


def section(summary: str, names: tuple) -> str:
    """まとめから、指定した見出しの部分だけを取り出す。"""
    out, keep = [], False
    for line in summary.splitlines():
        if line.lstrip().startswith("#"):
            keep = any(n in line for n in names)
            continue
        if keep:
            out.append(line)
    return "\n".join(out)


def run_case(mod, case: dict, doctor_llm, patient_llm, rounds: int, n_questions: int) -> dict:
    c = mod.new_case(case["patient"], case["chief_complaint"])
    first_hit, last_ddx, log = None, [], []
    for r in range(1, rounds + 1):
        sug = mod.suggest_next(c, doctor_llm, n_questions)
        last_ddx = [d["name"] for d in sug["ddx"]]
        if first_hit is None and has_gold("、".join(last_ddx), case["gold"]):
            first_hit = r
        log.append(f"[{r}回目] 鑑別: {'、'.join(last_ddx)} / 見逃し注意: {'、'.join(sug['must_not_miss'])}")
        for q in sug["questions"]:
            label = q["text"] if q["kind"] == "質問" else f"(診察){q['text']}"
            answer = patient_llm(patient_prompt(case, label), PATIENT_SYSTEM).strip()
            mod.add_exchange(c, label, answer)
            log.append(f"  Q: {label}\n  A: {answer}")
    summary = mod.summarize(c, doctor_llm)
    return {
        "last_ddx": "、".join(last_ddx),
        "gold_in_last_ddx": has_gold("、".join(last_ddx), case["gold"]),
        "gold_in_summary_ddx": has_gold(section(summary, ("鑑別",)), case["gold"]),
        "gold_in_summary_any": has_gold(section(summary, ("鑑別", "見逃")), case["gold"]),
        "first_hit_round": first_hit or "",
        "log": "\n".join(log),
        "summary": summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(HERE, "monshin_cases.json"))
    ap.add_argument("--model", default="gpt-oss:20b", help="問診アシスタントに使うモデル")
    ap.add_argument("--patient-model", default="", help="患者役のモデル(省略時は --model と同じ)")
    ap.add_argument("--think", default="medium", help="gpt-oss の考える量(low/medium/high)")
    ap.add_argument("--rounds", type=int, default=3, help="まとめる前に、提案→答えを何回繰り返すか")
    ap.add_argument("--n-questions", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None, help="この症例だけ評価する(症例のid)")
    ap.add_argument("--compare", default="", help="直す前のコミット。指定すると、そのときの指示文と今の指示文の両方で評価する")
    ap.add_argument("--out", default="monshin_eval.csv")
    args = ap.parse_args()

    cases = json.load(open(args.cases, encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    think_doc = "false" if args.model.lower().startswith("qwen3") else args.think
    doctor_llm = monshin.ollama_llm(args.model, think=think_doc)
    pmodel = args.patient_model or args.model
    patient_llm = monshin.ollama_llm(pmodel, think="false" if pmodel.lower().startswith("qwen3") else "low")

    versions = [("今の指示文", monshin)]
    if args.compare:
        versions.insert(0, (f"直す前({args.compare})", load_monshin_at(args.compare)))

    fields = ["version", "case", "gold", "last_ddx", "gold_in_last_ddx", "gold_in_summary_ddx",
              "gold_in_summary_any", "first_hit_round", "seconds", "log", "summary"]
    rows = []
    with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for vname, mod in versions:
            for case in cases:
                print(f"[{vname}] {case['id']} …", end="", flush=True)
                start = time.time()
                try:
                    res = run_case(mod, case, doctor_llm, patient_llm, args.rounds, args.n_questions)
                except Exception as e:  # 1件の失敗で全体を止めない
                    print(f" エラー: {e}")
                    continue
                row = {"version": vname, "case": case["id"], "gold": case["gold"][0],
                       "seconds": round(time.time() - start), **res}
                writer.writerow(row)
                fh.flush()
                rows.append(row)
                mark = "○" if row["gold_in_summary_any"] else "×"
                print(f" {row['seconds']}秒 まとめに正解:{mark} 最後の鑑別: {row['last_ddx']}")

    print("\n=== 集計 ===")
    for vname, _ in versions:
        rs = [r for r in rows if r["version"] == vname]
        if not rs:
            continue
        n = len(rs)
        print(
            f"{vname}: 最後の提案の鑑別に正解 {sum(r['gold_in_last_ddx'] for r in rs)}/{n}"
            f"  まとめの鑑別に正解 {sum(r['gold_in_summary_ddx'] for r in rs)}/{n}"
            f"  まとめ(鑑別か見逃し注意)に正解 {sum(r['gold_in_summary_any'] for r in rs)}/{n}"
        )
    print(f"\n問診のやり取りとまとめの全文は {args.out} に保存しました。")


if __name__ == "__main__":
    main()
