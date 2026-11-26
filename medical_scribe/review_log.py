"""
医師の確認記録: ①-2・③・④の候補を採用したか却下したか、⑤の所見をどう直したか、⑥で文章をどう直したかを、
録音1件ごとにJSONファイルとしてMacの中(review_logs/)に保存する。

使い道:
  - 却下の多い候補の型から、①-2の除外語や照合の規則を作る(eval/review_stats.py で集計)
  - ⑥で医師が直した後の文章を、文字起こしの正解データとして使う

記録には会話の内容(患者さんの情報)が含まれるため、外部には送らない。review_logs/ は git に含めない。
アプリ(app.py)からも集計スクリプトからも使えるよう、Streamlitに依存しない形で置いている。
"""

import datetime
import difflib
import hashlib
import json
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_logs")
VERSION = 1


def audio_id(data: bytes) -> str:
    """録音の中身から作る識別子。ファイル名(患者さんの名前が入りうる)の代わりに使う。"""
    return hashlib.sha256(data).hexdigest()[:16]


def new_session_id(audio_hash: str) -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + audio_hash


def text_edits(before: str, after: str) -> list:
    """⑥で医師が手で直した箇所を「前→後」と前後の文脈の一覧にする。"""
    edits = []
    sm = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        edits.append(
            {
                "old": before[i1:i2],
                "new": after[j1:j2],
                "before": before[max(0, i1 - 12) : i1],
                "after": before[i2 : i2 + 12],
            }
        )
    return edits


def finding_changes(extracted: list, reviewed: list) -> list:
    """⑤の所見ごとに、抽出結果と医師が確定した内容を並べる(直した欄・除外したかが分かる)。"""
    rows = []
    for orig, rev in zip(extracted, reviewed):
        changed = [k for k in ("item", "value", "laterality") if rev.get(k) != orig.get(k)]
        rows.append({"extracted": orig, "reviewed": rev, "changed": changed})
    return rows


def save(session_id: str, record: dict, log_dir: str = LOG_DIR) -> str:
    """同じ録音で何度確定しても1ファイルにまとめる(最後に確定した内容で上書き)。"""
    os.makedirs(log_dir, exist_ok=True)
    record = {"version": VERSION, "session_id": session_id, "saved_at": datetime.datetime.now().isoformat(timespec="seconds"), **record}
    path = os.path.join(log_dir, session_id + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)  # 書きかけのファイルが残らないように
    return path


def load_all(log_dir: str = LOG_DIR) -> list:
    if not os.path.isdir(log_dir):
        return []
    records = []
    for name in sorted(os.listdir(log_dir)):
        if name.endswith(".json"):
            with open(os.path.join(log_dir, name), encoding="utf-8") as fh:
                records.append(json.load(fh))
    return records
