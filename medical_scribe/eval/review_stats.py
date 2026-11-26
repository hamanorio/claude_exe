"""
医師の確認記録(review_logs/)を集計するスクリプト。

  python3 review_stats.py                 # 集計を表示
  python3 review_stats.py --min-count 2   # 2回以上出た候補だけ並べる
  python3 review_stats.py --export-refs ../eval/asr_review   # ⑥の確定文を正解データとして書き出す

分かること:
  - ①-2・③・④の候補が、それぞれ何割採用されたか
  - いつも却下される①-2の候補(元の語を term_lists/_除外語.txt に足す候補)
  - いつも採用される候補(誤変換辞書や用語リストに足す候補)
  - ⑤で医師が直した所見の欄、除外した所見
  - ⑥で医師が手で直した箇所(①〜④で拾えていない誤り)

--export-refs で書き出す正解は、①の文字起こしを医師が確認・修正したものです。ただし②の話者ラベルや
③④のLLMの修正(採用したもの)を含み、録音そのものを聞き直して作った正解ではありません。
評価に使うときは、そのことを踏まえてください。録音は書き出しません(記録に録音は含まれません)。
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import review_log  # noqa: E402


def rate(ok: int, total: int) -> str:
    return f"{ok}/{total}({ok / total:.0%})" if total else "0/0"


def tally(records, key, label_of):
    """候補を「元→直し」ごとにまとめ、採用・却下の回数を数える。"""
    counts = collections.defaultdict(lambda: [0, 0, ""])  # 採用, 却下, 文脈の例
    for rec in records:
        for c in rec.get(key) or []:
            if c.get("accepted") is None:
                continue
            row = counts[label_of(c)]
            row[0 if c["accepted"] else 1] += 1
            if not row[2]:
                row[2] = f"…{c.get('before', '')}[{label_of(c)}]{c.get('after', '')}…"
    return counts


def print_section(title, counts, min_count):
    acc = sum(r[0] for r in counts.values())
    total = sum(r[0] + r[1] for r in counts.values())
    print(f"\n=== {title}: 採用 {rate(acc, total)} ===")
    if not total:
        return
    always_no = sorted(((k, r) for k, r in counts.items() if r[0] == 0 and r[1] >= min_count), key=lambda kv: -kv[1][1])
    always_yes = sorted(((k, r) for k, r in counts.items() if r[1] == 0 and r[0] >= min_count), key=lambda kv: -kv[1][0])
    mixed = sorted(((k, r) for k, r in counts.items() if r[0] and r[1]), key=lambda kv: -(kv[1][0] + kv[1][1]))
    if always_no:
        print("  いつも却下:")
        for k, r in always_no[:30]:
            print(f"    {k}  却下{r[1]}回  例 {r[2]}")
    if always_yes:
        print("  いつも採用:")
        for k, r in always_yes[:30]:
            print(f"    {k}  採用{r[0]}回")
    if mixed:
        print("  採用と却下が分かれる:")
        for k, r in mixed[:20]:
            print(f"    {k}  採用{r[0]}回・却下{r[1]}回")


def export_refs(records, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for rec in records:
        sid = rec["session_id"]
        with open(os.path.join(out_dir, f"{sid}.txt"), "w", encoding="utf-8") as fh:
            fh.write(rec.get("final_text", ""))
        with open(os.path.join(out_dir, f"{sid}__asr.txt"), "w", encoding="utf-8") as fh:
            fh.write(rec.get("asr_text", ""))
    print(f"\n{len(records)}件を書き出しました: {out_dir}(<id>.txt=医師の確定文 / <id>__asr.txt=①の文字起こし)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=review_log.LOG_DIR, help="確認記録のフォルダ")
    ap.add_argument("--min-count", type=int, default=1, help="「いつも却下/採用」に並べる最少回数")
    ap.add_argument("--export-refs", default="", help="⑥の確定文と①の文字起こしを書き出すフォルダ")
    args = ap.parse_args()

    records = review_log.load_all(args.dir)
    if not records:
        print(f"確認記録がありません: {args.dir}\nアプリで「カルテ下書きを作成する」まで進むと記録されます。")
        return
    print(f"確認記録 {len(records)}件({records[0]['saved_at'][:10]} 〜 {records[-1]['saved_at'][:10]})")
    engines = collections.Counter(f"{r.get('asr', {}).get('engine', '?')} {r.get('asr', {}).get('model', '')}" for r in records)
    print("文字起こし: " + "、".join(f"{k} {n}件" for k, n in engines.most_common()))

    print_section("①-2 用語の音照合", tally(records, "sound_match", lambda c: f"{c['surface']}→{c['term']}"), args.min_count)
    ran = [r for r in records if r.get("correction_ran")]
    print(f"\n(③を実行した記録 {len(ran)}件 / スキップ {len(records) - len(ran)}件)")
    print_section("③ LLMによる文脈補正", tally(ran, "correction", lambda c: f"{c['old'] or '(なし)'}→{c['new'] or '(削除)'}"), args.min_count)
    print_section("④ 不自然な表現", tally(records, "vague", lambda c: f"{c['phrase']}→{c['suggestion']}"), args.min_count)

    findings = [f for r in records for f in r.get("findings") or []]
    if findings:
        excluded = [f for f in findings if f["reviewed"].get("excluded")]
        edited = [f for f in findings if f["changed"] and not f["reviewed"].get("excluded")]
        fields = collections.Counter(k for f in edited for k in f["changed"])
        print(f"\n=== ⑤ 構造化チェックリスト: 所見 {len(findings)}件 / 除外 {len(excluded)}件 / 直した {len(edited)}件 ===")
        if fields:
            names = {"item": "項目名", "value": "内容", "laterality": "側性"}
            print("  直した欄: " + "、".join(f"{names.get(k, k)} {n}件" for k, n in fields.most_common()))
        for f in excluded[:15]:
            print(f"  除外: {f['extracted'].get('item', '')} = {f['extracted'].get('value', '')}")
        for f in edited[:15]:
            e, r = f["extracted"], f["reviewed"]
            print(f"  直した: {e.get('item', '')}={e.get('value', '')}({e.get('laterality', '')}) → {r.get('item', '')}={r.get('value', '')}({r.get('laterality', '')})")

    edits = [e for r in records for e in r.get("final_edits") or []]
    print(f"\n=== ⑥ 医師が手で直した箇所: {len(edits)}件(①〜④で拾えていない誤り) ===")
    for e in edits[:40]:
        print(f"  …{e['before']}[{e['old'] or '(なし)'}→{e['new'] or '(削除)'}]{e['after']}…")

    if args.export_refs:
        export_refs(records, args.export_refs)


if __name__ == "__main__":
    main()
