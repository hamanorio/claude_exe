"""
鑑別診断モデルの比較評価スクリプト(Mac上のOllamaで実行。データは外部に送らない)

NEJM形式の症例データ(JSON: [{"病歴": ..., "最終診断": ..., "鑑別診断": ...}, ...])から
症例を選び、複数のOllamaモデルに同じ病歴を渡して、次の2つを自動で数えます。
  - 最終診断が正解と一致したか
  - 正解の診断名が、出力のどこか(鑑別診断の中など)に含まれていたか
診断名の表記ゆれ(バセドウ病/Graves病 など)は自動判定しきれないので、
全出力を CSV に保存し、医師が「要目視」の行を確認する前提です。

使い方:
  python3 eval_ddx.py --data NEJM_dataset6.json --train 学習に使ったデータ.json \
      --models swallow-8b-nejm swallow-8b --n 20 --out result.csv

  --train を指定すると、学習データと同じ病歴の症例を評価から除外します。
  学習に使っていない短い症例として test_cases_20.json(20症例)を同梱しています。
  途中で止めても、同じ --out で再実行すれば終わった症例は飛ばして続きから実行します。
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
SYSTEM_PROMPT = "あなたは医師を支援する医療AIです。病歴を読み、鑑別診断を挙げ、検証的推論を行い最終診断を考えてください。"
FIELDS = ["case_id", "model", "gold", "predicted_final", "final_match", "gold_in_output", "seconds", "output"]


def load_cases(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):  # {"data": [...]} のような形にも対応
        data = next((v for v in data.values() if isinstance(v, list)), [])
    cases = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        if row.get("病歴") and row.get("最終診断"):
            cases.append({"case_id": i, "病歴": row["病歴"].strip(), "最終診断": row["最終診断"].strip()})
        elif isinstance(row.get("messages"), list):
            # 学習用の会話形式({"messages": [system, user, assistant]})にも対応する
            msgs = {m.get("role"): m.get("content", "") for m in row["messages"] if isinstance(m, dict)}
            history = re.sub(r"^\s*'?病歴'?\s*[:：]\s*", "", msgs.get("user", "")).strip()
            gold = re.search(r"'?最終診断'?\s*[:：]\s*(.+)", msgs.get("assistant", ""))
            if history and gold:
                cases.append({"case_id": i, "病歴": history, "最終診断": gold.group(1).strip()})
    return cases


def ask_ollama(model: str, history: str, timeout: int = 900, num_ctx: int = 8192) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"'病歴': {history}"},  # 学習データと同じ形式
        ],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": 1536},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))["message"]["content"].strip()


def normalize(text: str) -> str:
    """比較用に、空白・記号・括弧内の補足・「〜の疑い」などを取り除く。"""
    text = re.sub(r"[（(][^）)]*[）)]", "", text)
    text = re.sub(r"[\s*#'\"「」『』：:。、,.・\-—]", "", text)
    text = re.sub(r"(の疑い|疑い|と考えられる|である)$", "", text)
    return text.lower()


def extract_final(output: str) -> str:
    """出力から最終診断の部分を取り出す。'最終診断': X / **最終診断:** X / 見出しの次の行 に対応。"""
    m = re.search(r"最終診断[^\n:：]*[:：]?\s*(.*)", output)
    if not m:
        return ""
    rest = m.group(1).strip(" *'\"")
    if rest:
        return rest.splitlines()[0].strip(" *'\"")
    after = output[m.end():].strip().splitlines()
    for line in after:
        line = line.strip(" *'\"-・")
        if line:
            return line
    return ""


def is_match(gold: str, predicted: str) -> bool:
    g, p = normalize(gold), normalize(predicted)
    if not g or not p:
        return False
    # 予測が正解を含む(「バセドウ病」⊂「バセドウ病による甲状腺中毒症」)なら一致。
    # 予測が正解の一部だけ(「肺炎」⊂「市中肺炎」)の場合は、短すぎると曖昧なので目視に回す
    return g in p or (p in g and len(p) >= 0.6 * len(g))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="評価に使う症例JSON")
    ap.add_argument("--train", nargs="*", default=[], help="学習に使ったJSON(同じ病歴の症例を除外する)")
    ap.add_argument("--models", nargs="+", required=True, help="比較するOllamaのモデル名")
    ap.add_argument("--n", type=int, default=20, help="評価する症例数")
    ap.add_argument("--seed", type=int, default=0, help="症例の選び方(同じ値なら同じ症例)")
    ap.add_argument("--max-chars", type=int, default=3000, help="これより長い病歴は除外(16GBのMacで遅くなりすぎないように)")
    ap.add_argument("--num-ctx", type=int, default=8192, help="長い症例(NEJMの症例記録など)を使うときは16384などに増やす")
    ap.add_argument("--out", default="eval_result.csv")
    args = ap.parse_args()

    cases = load_cases(args.data)
    train_histories = set()
    for path in args.train:
        train_histories |= {c["病歴"] for c in load_cases(path)}
    pool = [c for c in cases if c["病歴"] not in train_histories and len(c["病歴"]) <= args.max_chars]
    print(f"症例: 全{len(cases)}件 → 学習データ・長すぎる症例を除いて{len(pool)}件")
    if not pool:
        sys.exit("評価できる症例がありません。--data と --train の組み合わせを確認してください。")
    random.Random(args.seed).shuffle(pool)
    selected = pool[: args.n]

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8-sig") as fh:
            done = {(r["case_id"], r["model"]) for r in csv.DictReader(fh)}
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        # モデルの入れ替えが少なくなるよう、モデルごとにまとめて実行する
        for model in args.models:
            for k, case in enumerate(selected, 1):
                if (str(case["case_id"]), model) in done:
                    continue
                print(f"[{model}] {k}/{len(selected)} 症例{case['case_id']} …", end="", flush=True)
                start = time.time()
                try:
                    output = ask_ollama(model, case["病歴"], num_ctx=args.num_ctx)
                except Exception as e:  # 1件の失敗で全体を止めない
                    output = f"(エラー: {e})"
                final = extract_final(output)
                row = {
                    "case_id": case["case_id"],
                    "model": model,
                    "gold": case["最終診断"],
                    "predicted_final": final,
                    "final_match": "○" if is_match(case["最終診断"], final) else "要目視",
                    "gold_in_output": "○" if normalize(case["最終診断"]) in normalize(output) else "要目視",
                    "seconds": round(time.time() - start, 1),
                    "output": output,
                }
                writer.writerow(row)
                fh.flush()
                print(f" {row['seconds']}秒 最終診断:{row['final_match']}")

    summarize(args.out, args.models)


def summarize(path: str, models: list) -> None:
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    print("\n=== 集計(自動判定。表記ゆれは「要目視」に含まれるので、CSVで確認してください) ===")
    for model in models:
        rs = [r for r in rows if r["model"] == model]
        if not rs:
            continue
        final_ok = sum(r["final_match"] == "○" for r in rs)
        in_out = sum(r["gold_in_output"] == "○" for r in rs)
        avg = sum(float(r["seconds"]) for r in rs) / len(rs)
        print(f"{model}: 最終診断一致 {final_ok}/{len(rs)}  正解が出力内にあり {in_out}/{len(rs)}  平均{avg:.0f}秒/件")


if __name__ == "__main__":
    main()
