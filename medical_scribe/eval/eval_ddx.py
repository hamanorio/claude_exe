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

  OpenAIのモデルと比べるときは、モデル名の頭に openai: を付けます(APIキーは環境変数 OPENAI_API_KEY)。
    export OPENAI_API_KEY=sk-...
    python3 eval_ddx.py --data 症例.json --models openai:gpt-5.6-sol swallow-8b-nejm --n 13 --out result.csv
  考える型のモデル(Qwen3, GPT-OSS)は、考える分も出力の上限に含まれるので上限を増やします。
    python3 eval_ddx.py --data 症例.json --models qwen3:14b --num-predict 8192 --num-ctx 16384 --out result.csv
    python3 eval_ddx.py --data 症例.json --models gpt-oss:20b --think low --num-predict 8192 --num-ctx 16384 --out result.csv
  ※ openai: のモデルには症例の文章が OpenAI(米国)に送られます。公開された症例や架空の症例だけに使い、
    実在の患者さんの情報は送らないでください。

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
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
OPENAI_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
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


def ask_openai(model: str, history: str, timeout: int = 900) -> str:
    """OpenAIのAPIで同じ症例を解かせる(比較用)。推論型のモデルは temperature を指定できないので指定しない。"""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("環境変数 OPENAI_API_KEY が設定されていません")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"'病歴': {history}"},
        ],
        # 推論型のモデルは、内部の思考にもこの上限を使うので大きめにする
        "max_completion_tokens": 16000,
    }
    req = urllib.request.Request(
        OPENAI_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            data = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
    return (data["choices"][0]["message"].get("content") or "").strip()


# アプリ(app.py)と同じ条件で比べるため、考える型のモデルは初期設定で考える量を減らす
# (Qwen3は考えない、GPT-OSSは最小)。--think を指定した場合はそちらを使う
AUTO_THINK = {"qwen3": "false", "gpt-oss": "low"}


def think_for(model: str, think: str) -> str:
    if think != "auto":
        return think
    return next((v for prefix, v in AUTO_THINK.items() if model.lower().startswith(prefix)), "")


def ask_model(model: str, history: str, args) -> str:
    if model.startswith("openai:"):
        return ask_openai(model[len("openai:") :], history)
    return ask_ollama(
        model, history, num_ctx=args.num_ctx, num_predict=args.num_predict, think=think_for(model, args.think)
    )


def strip_thinking(text: str) -> str:
    """考える型のモデル(Qwen3など)が出力に混ぜる <think>…</think> を取り除き、答えの部分だけにする。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:  # 考えている途中で出力の上限に達した
        text = text.split("<think>")[0]
    return text.strip()


def ask_ollama(
    model: str, history: str, timeout: int = 3600, num_ctx: int = 8192, num_predict: int = 1536, think: str = ""
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"'病歴': {history}"},  # 学習データと同じ形式
        ],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    if think:
        # 考える型のモデルだけに指定する(対応していないモデルに送るとエラーになる)。
        # Qwen3 は true/false、GPT-OSS は low/medium/high で考える量を指定する
        payload["think"] = {"true": True, "false": False}.get(think, think)
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            message = json.loads(res.read().decode("utf-8"))["message"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
    # 新しいOllamaでは、考えた過程は message["thinking"] に分かれて返るので content だけを使う
    return strip_thinking(message.get("content", ""))


def normalize(text: str) -> str:
    """比較用に、空白・記号・括弧内の補足・「〜の疑い」などを取り除く。"""
    text = re.sub(r"[（(][^）)]*[）)]", "", text)
    text = re.sub(r"[\s*#'\"「」『』：:。、,.・\-—]", "", text)
    text = re.sub(r"(の疑い|疑い|と考えられる|である)$", "", text)
    return text.lower()


def _clean(line: str) -> str:
    """行頭の見出し記号・番号(## 3. / **/ - / 【)と、行末の記号を取り除く。"""
    line = re.sub(r"^[\s#*'\"\-・>【\[]*(?:[0-9０-９]+[.．)）]\s*)?[\s*'\"【]*", "", line)
    return re.sub(r"[\s*'\"】\]：:。、]+$", "", line)


def _pick(text: str) -> str:
    """太字(**X**)があればその中身を、なければ文をそのまま返す。"""
    bold = re.search(r"\*\*([^*]+?)\*\*", text)
    picked = bold.group(1) if bold else text
    picked = re.sub(r"(が|は)?(最も)?(可能性の高い|考えられる)診断.*$", "", picked)
    return _clean(picked)


def extract_final(output: str) -> str:
    """出力から最終診断を取り出す。

    本文中の「…最終診断を考えます」のような言及ではなく、行頭が「最終診断」で始まる
    見出し(## 最終診断 / **最終診断:** / '最終診断': X / 3. 最終診断)のうち最後のものを使う。
    見出しの後ろが「以下の診断が考えられます」のような前置きなら、次の行を診断とみなす。
    """
    lines = output.splitlines()
    heads = [i for i, ln in enumerate(lines) if _clean(ln).startswith("最終診断")]
    if not heads:
        return ""
    i = heads[-1]
    rest = re.sub(r"^最終診断[^:：]*[:：]?[\s*'\"】]*", "", _clean(lines[i]), count=1).strip()
    candidates = [rest] if rest else []
    candidates += [ln for ln in lines[i + 1 :] if _clean(ln)]
    for cand in candidates[:4]:
        cand = cand.strip()
        if re.search(r"(以下|次の)[^。]*(考えられ|挙げ|示し|まとめ)|[:：]\s*$", cand) and "**" not in cand:
            continue  # 前置きの文は飛ばす
        return _pick(cand)
    return _pick(candidates[0]) if candidates else ""


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
    ap.add_argument(
        "--num-predict",
        type=int,
        default=1536,
        help="1回の出力の上限(トークン)。考える型のモデル(Qwen3, GPT-OSS)は考える分も含むので8192などに増やす",
    )
    ap.add_argument(
        "--think",
        default="auto",
        choices=["auto", "", "true", "false", "low", "medium", "high"],
        help="考える型のモデルの考える量。auto(初期値)はアプリと同じく Qwen3=false、GPT-OSS=low。"
        "Qwen3は true/false、GPT-OSSは low/medium/high で指定でき、\"\" はモデル任せ",
    )
    ap.add_argument("--out", default="eval_result.csv")
    args = ap.parse_args()

    if any(m.startswith("openai:") for m in args.models):
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("openai: のモデルを使うには、先に export OPENAI_API_KEY=... を実行してください。")
        print("※ openai: のモデルには症例の文章がOpenAIに送られます(実在の患者情報は含めないでください)")
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
            # エラーで終わった症例は「済み」にせず、再実行したときにやり直す
            done = {(r["case_id"], r["model"]) for r in csv.DictReader(fh) if not r["output"].startswith("(エラー:")}
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
                    output = ask_model(model, case["病歴"], args)
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


def latest_rows(rows: list) -> list:
    """同じ症例・同じモデルの行が複数あれば(エラー後のやり直しなど)、最後の行だけを残す。"""
    latest = {}
    for r in rows:
        latest[(r["case_id"], r["model"])] = r
    return list(latest.values())


def summarize(path: str, models: list) -> None:
    with open(path, encoding="utf-8-sig") as fh:
        rows = latest_rows(list(csv.DictReader(fh)))
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
