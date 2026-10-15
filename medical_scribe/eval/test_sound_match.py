"""
①-2 用語の音照合(sound_match.py)の確認用テスト。
これまでに見つかった「出てほしい候補」と「出てはいけない候補」を並べてあり、
①-2を直したときに、以前直した誤りが戻っていないかを確かめられる。

  python3 test_sound_match.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sound_match as sm  # noqa: E402

# 候補が出てほしい(文中の誤り → 用語リストの正しい語)
SHOULD_SUGGEST = {
    "周りの人より厚がりになった": ("厚がり", "暑がり"),
    "TSHが腸鳴に低下": ("腸鳴", "著明"),
    "び慢性の腫れ": ("び慢性", "びまん性"),
    "甲状腺の主題があります": ("主題", "腫大"),
    "パセド病です": ("パセド病", "バセドウ病"),
    "放射性要素の治療": ("放射性要素", "放射性ヨウ素"),
    "失意上昇がある": ("失意上昇", "脂質異常症"),
    "美満腺の腫れ": ("美満腺", "びまん性"),
    "有利T4": ("有利T4", "遊離T4"),
    "著名な腫れ": ("著名", "著明"),
    "動機がします": ("動機", "動悸"),
    "新札します": ("新札", "診察"),
    "原料が必要": ("原料", "減量"),
    "生理不純": ("不純", "不順"),
    "不正脈": ("不正脈", "不整脈"),
    "心房再動": ("心房再動", "心房細動"),
    "構音傷害": ("構音傷害", "構音障害"),
    "無果粒球症": ("無果粒球症", "無顆粒球症"),
    "酸素放和度": ("酸素放和度", "酸素飽和度"),
    "TRABが陽性": ("TRAB", "TRAb"),
}

# 候補が出てはいけない(正しい日本語・正しく書かれた用語)
SHOULD_NOT_SUGGEST = [
    "階段を上る", "欲しいです", "5分間", "全員で", "乾燥します", "対応します", "長所です",
    "注意点は", "効果が", "判断します", "検察", "段階で", "中心に", "お話しします", "高熱が",
    "一旦やめて", "服用して", "境界が", "発見が", "便は異変なし", "体重減少が3キロ",
    "心房の動きを見ます", "症状は最近です", "脈は速い", "甲状腺にホルモン", "なる症状",
    "ありがとうございました", "よろしくお願いします", "変わりないですか", "お薬を飲みましたか",
    "検査しましょう", "説明しました",
]


def main() -> None:
    terms = tuple(dict.fromkeys(t for name in sm.list_term_files() for t in sm.load_term_file(name)))
    entries, index = sm.build_sound_index(terms)
    term_set = set(terms) | sm.stop_words()
    failures = []
    for text, expected in SHOULD_SUGGEST.items():
        got = [(m[2], m[3]) for m in sm.find_sound_alike(text, entries, index, term_set)]
        if expected not in got:
            failures.append(f"候補が出ない: 「{text}」 期待 {expected[0]}→{expected[1]} / 実際 {got}")
    for text in SHOULD_NOT_SUGGEST:
        got = [(m[2], m[3]) for m in sm.find_sound_alike(text, entries, index, term_set)]
        if got:
            failures.append(f"誤った候補: 「{text}」 → {got}")
    total = len(SHOULD_SUGGEST) + len(SHOULD_NOT_SUGGEST)
    for f in failures:
        print("NG", f)
    print(f"{total - len(failures)}/{total} 件OK")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
