"""
⑧ チームでの検討の、危険な兆候のキーワード確認(ddx_agents.rule_red_flags)のテスト。
見落とし(拾うべきものを拾わない)が特に危ないので、否定の扱いを中心に確かめる。

  python3 test_red_flags.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ddx_agents as da  # noqa: E402

# 文 → 拾うべき兆候(名前の一部)。空なら何も拾わないのが正しい
CASES = {
    "胸痛はありません。顔面神経麻痺もなく、構音障害も今はありません": [],
    "呂律が回らない": ["脳卒中"],  # 「ない」は症状の一部で、否定ではない
    "言葉が出にくい": ["脳卒中"],
    "右手に力が入らない": ["脳卒中"],
    "胸痛: あり、冷汗: あり": ["急性冠症候群"],
    "胸痛: なし。吐血: あり": ["消化管出血"],
    "顔面神経麻痺もなく、右片麻痺あり": ["脳卒中"],
    "失神はありません。項部硬直: 陰性": [],
    "麻痺は認めない": [],
    "血圧は86の50": ["ショック"],
    "血圧は138の72": [],
    "横になれないほど苦しい": ["呼吸不全"],
    "酸素飽和度は88パーセント": ["呼吸不全"],
    "酸素飽和度は96パーセント": [],
    "チアマゾール内服中に38度の発熱とのどの痛み": ["無顆粒球症"],
}


def main() -> None:
    failures = []
    for text, expected in CASES.items():
        got = da.rule_red_flags(text)
        if len(got) != len(expected) or not all(any(e in g for g in got) for e in expected):
            failures.append(f"「{text}」 期待 {expected} / 実際 {got}")
    for f in failures:
        print("NG", f)
    print(f"{len(CASES) - len(failures)}/{len(CASES)} 件OK")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
