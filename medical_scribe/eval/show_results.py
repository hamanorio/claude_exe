"""
eval_ddx.py の結果(CSV)をターミナルで見やすく表示する。

  python3 show_results.py eval_new.csv            症例ごとに、正解と各モデルの最終診断を並べて表示
  python3 show_results.py eval_new.csv --case 12  症例12について、各モデルの出力全文を表示
"""

import argparse
import csv
from collections import OrderedDict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--case", help="この症例番号の出力全文を表示する")
    args = ap.parse_args()

    with open(args.csv_path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    models = list(OrderedDict.fromkeys(r["model"] for r in rows))

    if args.case is not None:
        for r in rows:
            if r["case_id"] == args.case:
                print(f"\n===== 症例{r['case_id']} / {r['model']} / 正解: {r['gold']} =====")
                print(r["output"])
        return

    by_case = OrderedDict()
    for r in rows:
        by_case.setdefault(r["case_id"], {})[r["model"]] = r
    for case_id, per_model in by_case.items():
        gold = next(iter(per_model.values()))["gold"]
        print(f"\n症例{case_id}  正解: {gold}")
        for m in models:
            r = per_model.get(m)
            if r:
                mark = "○" if r["final_match"] == "○" else "？"
                in_out = "(出力内に正解あり)" if r["gold_in_output"] == "○" else ""
                print(f"  {mark} {m}: {r['predicted_final'] or '(最終診断を読み取れず)'} {in_out}")

    print("\n=== 集計(○=自動で一致と判定。？は表記ゆれの可能性があるので目で確認) ===")
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        ok = sum(r["final_match"] == "○" for r in rs)
        inside = sum(r["gold_in_output"] == "○" for r in rs)
        print(f"{m}: 最終診断一致 {ok}/{len(rs)}  正解が出力内にあり {inside}/{len(rs)}")
    print("\n出力全文を見るには: python3 show_results.py", args.csv_path, "--case 症例番号")


if __name__ == "__main__":
    main()
