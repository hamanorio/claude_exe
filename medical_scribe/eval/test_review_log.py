"""
確認記録(review_log.py)と集計(review_stats.py)の確認用テスト。作り物の記録を一時フォルダに書いて確かめる。

  python3 test_review_log.py
"""

import contextlib
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import review_log  # noqa: E402
import review_stats  # noqa: E402


def make_record(accept_shujo: bool) -> dict:
    extracted = [{"item": "動悸", "value": "あり", "laterality": "なし"}, {"item": "頭痛", "value": "あり", "laterality": "不明"}]
    reviewed = [
        {"item": "動悸", "value": "3日前から", "laterality": "なし", "excluded": False},
        {"item": "頭痛", "value": "あり", "laterality": "不明", "excluded": True},
    ]
    checked = "医師: 甲状腺の腫大がありますね"
    final = "医師: 甲状腺の腫大が軽度ありますね"
    return {
        "asr": {"engine": "mlx-whisper", "model": "large-v3-turbo"},
        "asr_text": "甲状腺の主題がありますね",
        "sound_match": [
            {"surface": "主題", "term": "腫大", "ratio": 1.0, "count": 1, "before": "甲状腺の", "after": "が", "accepted": accept_shujo},
            {"surface": "定期検査", "term": "髄液検査", "ratio": 0.8, "count": 1, "before": "", "after": "", "accepted": False},
        ],
        "correction_ran": True,
        "correction": [{"old": "献身", "new": "健診", "before": "", "after": "で", "accepted": True}],
        "vague": [{"phrase": "右の奥", "suggestion": "右下腹部", "reason": "", "count": 1, "accepted": False}],
        "findings": review_log.finding_changes(extracted, reviewed),
        "checked_text": checked,
        "final_text": final,
        "final_edits": review_log.text_edits(checked, final),
    }


def main():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ⑥の手直しの一覧
    edits = review_log.text_edits("甲状腺の腫大がありますね", "甲状腺の腫大が軽度ありますね")
    check(len(edits) == 1 and edits[0]["old"] == "" and edits[0]["new"] == "軽度", f"text_edits: {edits}")
    check(review_log.text_edits("同じ", "同じ") == [], "text_edits: 変更なしなら空")

    # ⑤の直した欄
    rows = make_record(True)["findings"]
    check(rows[0]["changed"] == ["value"], f"finding_changes: {rows[0]['changed']}")
    check(rows[1]["changed"] == [] and rows[1]["reviewed"]["excluded"], "finding_changes: 除外")

    # 録音の識別子はファイル名でなく中身から(同じ中身なら同じ)
    check(review_log.audio_id(b"abc") == review_log.audio_id(b"abc") != review_log.audio_id(b"abd"), "audio_id")

    with tempfile.TemporaryDirectory() as d:
        # 同じ録音で2回確定したら1ファイルに上書き
        review_log.save("s1", make_record(False), d)
        review_log.save("s1", make_record(True), d)
        review_log.save("s2", make_record(False), d)
        recs = review_log.load_all(d)
        check(len(recs) == 2, f"save/load: {len(recs)}件")
        check(recs[0]["sound_match"][0]["accepted"] is True, "save: 上書きされていない")
        check(not any(n.endswith(".tmp") for n in os.listdir(d)), "save: 書きかけのファイルが残っている")

        counts = review_stats.tally(recs, "sound_match", lambda c: f"{c['surface']}→{c['term']}")
        check(counts["主題→腫大"][:2] == [1, 1], f"tally: {counts['主題→腫大']}")
        check(counts["定期検査→髄液検査"][:2] == [0, 2], f"tally: {counts['定期検査→髄液検査']}")

        out = io.StringIO()
        sys.argv = ["review_stats.py", "--dir", d, "--export-refs", os.path.join(d, "refs")]
        with contextlib.redirect_stdout(out):
            review_stats.main()
        text = out.getvalue()
        check("定期検査→髄液検査  却下2回" in text, "集計: いつも却下が出ない\n" + text)
        check("[(なし)→軽度]" in text, "集計: ⑥の手直しが出ない")
        check(os.path.exists(os.path.join(d, "refs", "s1.txt")), "書き出し: 確定文が無い")

    for f in failures:
        print("NG", f)
    print("OK" if not failures else f"{len(failures)}件 NG")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
