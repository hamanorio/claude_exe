"""
①-2 用語の音照合: 文字起こしの中で、読みが医学用語リストの語に近いのに表記が違う箇所を探す。
アプリ(app.py)と精度評価(eval/eval_asr.py)の両方から使うため、Streamlitに依存しない形で置いている。
"""

import difflib
import os
import re

TERM_LIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "term_lists")


def list_term_files() -> list:
    """term_lists フォルダ内の診療科別用語リスト(.txt)の名前一覧。ファイルを足せば選択肢に増える。"""
    if not os.path.isdir(TERM_LIST_DIR):
        return []
    # 「_」で始まるファイル(_除外語.txt 等)は診療科の選択肢に出さない
    return sorted(f[:-4] for f in os.listdir(TERM_LIST_DIR) if f.endswith(".txt") and not f.startswith("_"))


def load_term_file(name: str) -> list:
    path = os.path.join(TERM_LIST_DIR, name + ".txt")
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


_kakasi = None


def to_reading_tokens(text: str) -> list:
    """文字列を(表記, ひらがな読み)のトークン列に分ける。英数字はそのまま読みとして扱う。"""
    global _kakasi
    if _kakasi is None:
        import pykakasi

        _kakasi = pykakasi.kakasi()
    return [(t["orig"], t["hira"]) for t in _kakasi.convert(text) if t["orig"]]


def parse_term_list(raw: str) -> list:
    terms = re.split(r"[\s、,，・/]+", raw)
    return list(dict.fromkeys(t.strip() for t in terms if t.strip()))


def build_sound_index(terms: tuple):
    """用語の読みを2文字ずつに分けた索引を作る。大きな用語リストでも照合が速くなるように。"""
    entries, index = [], {}
    for term in terms:
        reading = "".join(h for _, h in to_reading_tokens(term))
        if len(reading) < 3:  # 2文字以下の読みは偶然の一致が多すぎるため対象外
            continue
        idx = len(entries)
        entries.append((term, reading))
        for i in range(len(reading) - 1):
            index.setdefault(reading[i : i + 2], set()).add(idx)
    return entries, index


PARTICLE_MID = re.compile(r"[^\u3040-\u309f][はがをにのでともへや][^\u3040-\u309f]")
HIRAGANA_ANY = re.compile(r"[\u3040-\u309f]")
PARTICLE_END = re.compile(r"[はがをにのでともへや]$")
HIRAGANA_END = re.compile(r"[\u3040-\u309f]$")
HIRAGANA_ONLY = re.compile(r"^[\u3040-\u309fー、。,.？！?!\s]*$")


def find_sound_alike(text: str, entries: list, index: dict, term_set: set, max_tokens: int = 6) -> list:
    """読みが用語リストの語に近いのに表記が違う箇所を探す(例:「有利T4」→「遊離T4」)。
    誤りの書き方ではなく正しい用語だけを登録すればよいので、未知の誤変換にも対応できる。
    戻り値: (開始位置, 終了位置, 元の表記, 候補の用語, 類似度) のリスト(重なりなし)
    """
    # すでに正しく書かれている用語・除外語の部分は置き換えない(「体重減少」の「減少」→「減量」のような誤りを防ぐ)
    protected = set()
    for word in term_set:
        if len(word) < 2:
            continue
        at = text.find(word)
        while at >= 0:
            protected.update(range(at, at + len(word)))
            at = text.find(word, at + 1)

    toks = to_reading_tokens(text)
    starts, pos = [], 0
    for surf, _ in toks:
        found_at = text.find(surf, pos)
        starts.append(found_at if found_at >= 0 else pos)
        pos = starts[-1] + len(surf)

    found = []
    for i in range(len(toks)):
        surface, reading = "", ""
        for j in range(i, min(i + max_tokens, len(toks))):
            surface += toks[j][0]
            reading += toks[j][1]
            # 前後のひらがなを除いた中心部分(「なる症状」→「症状」)も、用語・除外語なら対象外
            core = re.sub(r"^[\u3040-\u309f]+|[\u3040-\u309f]+$", "", surface)
            if len(reading) < 3 or len(surface) < 2 or HIRAGANA_ONLY.match(surface) or surface in term_set or core in term_set:
                continue
            if re.search(r"[：:、。？！?!\s]", surface):
                continue  # 話者ラベルや句読点をまたぐ区切りは対象外
            if protected & set(range(starts[i], starts[i] + len(surface))):
                continue
            cands = set()
            for k in range(len(reading) - 1):
                cands |= index.get(reading[k : k + 2], set())
            for idx in cands:
                term, term_reading = entries[idx]
                if abs(len(term_reading) - len(reading)) > 2 or term in surface or surface in term:
                    continue
                # 「脈は」→「脈拍」のように、助詞で終わる区切りを漢字で終わる用語と取り違えない
                if PARTICLE_END.search(surface) and not HIRAGANA_END.search(term):
                    continue
                # 「甲状腺にホルモン」→「甲状腺ホルモン」のように、助詞をはさんだ2語を1語にしない
                if PARTICLE_MID.search(surface) and not HIRAGANA_ANY.search(term):
                    continue
                ratio = difflib.SequenceMatcher(None, reading, term_reading).ratio()
                # 短い語は読みが完全一致する場合のみ(偶然の一致を避ける)
                if ratio < (1.0 if len(term_reading) <= 3 else 0.75):
                    continue
                # 用語の一部になっている言葉(「心房細動」の「心房」)はそれ自体が正しい語なので、
                # 読みが少し違うだけの別の用語(「心臓」)に置き換えない。読みが完全一致する場合は残す
                if ratio < 1.0 and any(surface in w for w in term_set):
                    continue
                found.append((starts[i], starts[i] + len(surface), surface, term, ratio))

    # 類似度の高い順に、重ならないものだけを残す(「腸鳴」と「腸鳴に」のような重複を除く)
    found.sort(key=lambda f: (-f[4], f[1] - f[0]))
    chosen, used = [], set()
    for f in found:
        span = set(range(f[0], f[1]))
        # 同じ箇所で同じくらい近い候補が複数あるとき(発見→発汗/発疹)は両方出して医師に選ばせる
        tie = any(c[0] == f[0] and c[1] == f[1] and c[4] == f[4] and c[3] != f[3] for c in chosen)
        if not span & used or tie:
            if not any(c[0] == f[0] and c[1] == f[1] and c[3] == f[3] for c in chosen):
                chosen.append(f)
            used |= span
    return sorted(chosen)


def apply_sound_replacements(text: str, matches: list, accepted_pairs: set) -> str:
    out, last = [], 0
    for start, end, surface, term, _ in matches:
        if (surface, term) in accepted_pairs and start >= last:  # 同じ箇所の別候補は先に採用した方だけ
            out.append(text[last:start])
            out.append(term)
            last = end
    out.append(text[last:])
    return "".join(out)


def stop_words() -> set:
    """会話によく出る一般語(_除外語.txt)。置き換え候補の元の語として扱わない。"""
    if os.path.exists(os.path.join(TERM_LIST_DIR, "_除外語.txt")):
        return set(load_term_file("_除外語"))
    return set()
