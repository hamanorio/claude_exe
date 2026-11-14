"""
問診アシスタント(monshin.py)の、LLMの出力の読み取りと記録のテスト。

  python3 test_monshin.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import monshin  # noqa: E402


def check(name, got, expected, failures):
    if got != expected:
        failures.append(f"{name}: 期待 {expected} / 実際 {got}")


def main() -> None:
    failures = []
    # 太字・番号・箇条書きの記号・全角括弧・半角の区切りが混ざっても読める
    out = (
        "前置きの文です。\n**鑑別:** 溶連菌性咽頭炎(高)、伝染性単核球症（中）、ウイルス性咽頭炎\n"
        "見逃し注意: 扁桃周囲膿瘍、急性喉頭蓋炎\n1. 質問: 咳はありますか｜ウイルス性との区別\n"
        "- 質問: つばが飲み込めますか | 喉頭蓋炎の除外\n診察: 前頸部リンパ節｜溶連菌"
    )
    r = monshin.parse_next_step(out)
    check("鑑別", [(d["name"], d["level"]) for d in r["ddx"]],
          [("溶連菌性咽頭炎", "高"), ("伝染性単核球症", "中"), ("ウイルス性咽頭炎", "")], failures)
    check("見逃し注意", r["must_not_miss"], ["扁桃周囲膿瘍", "急性喉頭蓋炎"], failures)
    check("質問", [(q["kind"], q["text"], q["purpose"]) for q in r["questions"]],
          [("質問", "咳はありますか", "ウイルス性との区別"), ("質問", "つばが飲み込めますか", "喉頭蓋炎の除外"),
           ("診察", "前頸部リンパ節", "溶連菌")], failures)
    check("見逃し注意なし", monshin.parse_next_step("見逃し注意: なし")["must_not_miss"], [], failures)

    # 答えが空の質問は記録しない。危険な兆候は答えから拾い、質問文からは拾わない
    case = monshin.new_case("70歳男性", "めまい")
    monshin.add_exchange(case, "呂律が回らないことはありますか", "")
    check("空欄は記録しない", len(case["exchanges"]), 0, failures)
    monshin.add_exchange(case, "呂律が回らないことはありますか", "いいえ、ありません")
    check("否定の答えは拾わない", monshin.red_flags(case), [], failures)
    monshin.add_exchange(case, "手足の動きはどうですか", "右手に力が入らない")
    check("症状の答えは拾う", any("脳卒中" in f for f in monshin.red_flags(case)), True, failures)
    check("医師のメモは「すでに聞いた質問」に入れない",
          "(医師のメモ)" in monshin.next_step_prompt({**case, "exchanges": case["exchanges"] + [
              {"question": "(医師のメモ)", "answer": "x", "source": "医師"}]}).split("【すでに聞いた質問】")[1], False, failures)

    for f in failures:
        print("NG", f)
    print(f"{9 - len(failures)}/9 件OK")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
