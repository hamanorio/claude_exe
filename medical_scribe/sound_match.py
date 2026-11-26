"""
①-2 用語の音照合: 文字起こしの中で、読みが医学用語リストの語に近いのに表記が違う箇所を探す。
アプリ(app.py)と精度評価(eval/eval_asr.py)の両方から使うため、Streamlitに依存しない形で置いている。
"""

import difflib
import functools
import os
import re
import unicodedata

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


def strip_voicing(reading: str) -> str:
    """濁点・半濁点を取る(「かんがい」→「かんかい」、「ぱせどう」→「はせとう」)。
    音声認識は清音・濁音・半濁音をよく取り違える(パセドウ↔バセドウ、感慨↔寛解)ので、
    この違いだけの読みは「読みが同じ」とみなす。"""
    return "".join(c for c in unicodedata.normalize("NFD", reading) if c not in "\u3099\u309a")


PARTICLE_MID = re.compile(r"[^\u3040-\u309f][はがをにのでともへや][^\u3040-\u309f]")
HIRAGANA_ANY = re.compile(r"[\u3040-\u309f]")
PARTICLE_START = re.compile(r"[はがをにのでともへや]")
PARTICLE_END = re.compile(r"[はがをにのでともへや]$")
HIRAGANA_END = re.compile(r"[\u3040-\u309f]$")
KANJI = re.compile(r"[\u4e00-\u9fff]")
HIRAGANA_START = re.compile(r"^[\u3040-\u309f]")
HIRAGANA_ONLY = re.compile(r"^[\u3040-\u309fー、。,.？！?!\s]*$")
KANJI_WORD = re.compile(r"^[\u4e00-\u9fff]{2,}$")
# 区切りに入っていたら対象外にする記号(話者ラベル・句読点・括弧・中黒など)
SEPARATORS = re.compile(r"[：:、。，．,.？！?!\s（）()「」『』【】\[\]・…〜~]")


def shared_part_ok(reading: str, term_reading: str) -> bool:
    """読みに共通部分(3文字以上)があるとき、その位置から見て候補として妥当か。
    共通部分があると全体の読みの近さが水増しされるので、次の2つは候補にしない。
      - 共通部分が末尾どうし(「定期検査」→「髄液検査」の けんさ)で、残りの音が近くない。
        末尾の語(検査・療法など)が同じだけで、前の語は全く別の語であることが多い
      - 共通部分が片方では先頭、もう片方では末尾(「コント生活習慣」→「生活習慣病」)
    先頭どうしが共通(「眼球として」→「眼球突出」「パセド病」→「バセドウ病」)なら残す。
    読みは区切り全体で比べる(漢字の一部だけを読み変換すると「ヨウ素」→「ようもと」のように読み違えるため)。"""
    a, b = strip_voicing(reading), strip_voicing(term_reading)
    m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    if m.size < 3:
        return True
    start_a, start_b = m.a == 0, m.b == 0
    end_a, end_b = m.a + m.size == len(a), m.b + m.size == len(b)
    whole = (start_a and end_a) or (start_b and end_b)  # 片方の読みがまるごと共通部分(「失意上昇」→「脂質異常症」)
    if not whole and ((start_a and end_b and not start_b) or (start_b and end_a and not start_a)):
        return False
    if end_a and end_b and not (start_a or start_b):
        rest_a, rest_b = a[: m.a], b[: m.b]
        return difflib.SequenceMatcher(None, rest_a, rest_b).ratio() >= 0.5
    return True


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
        if found_at >= 0:
            starts.append(found_at)
            pos = found_at + len(surf)
        else:
            # 読み変換が元と違う表記を返した(文字化けした「�」など)。位置を進めずにおく。
            # 進めると後ろの語がすべて見つからなくなり、文字の位置が本文の外にずれる
            starts.append(min(pos, len(text)))

    found = []
    for i in range(len(toks)):
        surface, reading = "", ""
        for j in range(i, min(i + max_tokens, len(toks))):
            surface += toks[j][0]
            reading += toks[j][1]
            variants = [(surface, reading)]
            # 読み変換は「厚がりになった」を「厚」「がりになった」のように区切るので、
            # 最後のひらがなのまとまりの一部だけを付けた区切り(「厚がり」)も試す。
            # 1文字だけ付けると「服用し」→「副作用」のように動詞の活用を拾うので、2文字以上に限る
            tok = toks[j][0]
            if j > i and HIRAGANA_ONLY.match(tok) and len(tok) > 2:
                base_s, base_r = surface[: -len(tok)], reading[: -len(toks[j][1])]
                variants += [(base_s + tok[:k], base_r + tok[:k]) for k in range(2, len(tok))]
            # 一部だけ付けたひらがな(「厚がり」の がり)は、用語がその同じかなで終わるか、切った後ろが助詞で
            # 始まる(「眼球として|があります」)場合だけ候補にする。「食欲はむし|ろ」→「食欲不振」
            # 「生活習慣につ|いて」→「生活習慣病」のように、語の途中で切った区切りを別の用語に取り違えないため
            partial = {}
            for surface_v, _ in variants[1:]:
                piece = surface_v[len(surface) - len(tok) :]
                partial[surface_v] = (piece, PARTICLE_START.match(tok[len(piece) :]) is not None)
            for surface_v, reading_v in variants:
                # 前後のひらがなを除いた中心部分(「なる症状」→「症状」)も、用語・除外語なら対象外
                core = re.sub(r"^[\u3040-\u309f]+|[\u3040-\u309f]+$", "", surface_v)
                if len(reading_v) < 3 or len(surface_v) < 2 or HIRAGANA_ONLY.match(surface_v) or surface_v in term_set or core in term_set:
                    continue
                if SEPARATORS.search(surface_v):
                    continue  # 話者ラベル・句読点・括弧をまたぐ区切りは対象外
                if protected & set(range(starts[i], starts[i] + len(surface_v))):
                    continue
                cands = set()
                for k in range(len(reading_v) - 1):
                    cands |= index.get(reading_v[k : k + 2], set())
                for idx in cands:
                    term, term_reading = entries[idx]
                    if surface_v in partial and not (partial[surface_v][1] or term.endswith(partial[surface_v][0])):
                        continue
                    if abs(len(term_reading) - len(reading_v)) > 2 or term in surface_v or surface_v in term:
                        continue
                    # 数字は会話の内容そのもの(「1週間前」→「就寝前」のような置き換えを防ぐ)。
                    # 数字を含む用語(遊離T4 など)への置き換えは残す
                    if re.search(r"[0-9０-９]", surface_v) and not re.search(r"[0-9０-９]", term):
                        continue
                    # 数字の直後から始まる区切り(「1週間前」の「週間前」)は単位なので対象外
                    if 0 < starts[i] <= len(text) and re.match(r"[0-9０-９]", text[starts[i] - 1]) and not re.search(r"[0-9０-９]", term):
                        continue
                    # 「脈は」→「脈拍」のように、助詞で終わる区切りを漢字で終わる用語と取り違えない
                    if PARTICLE_END.search(surface_v) and not HIRAGANA_END.search(term):
                        continue
                    # 「甲状腺にホルモン」→「甲状腺ホルモン」のように、助詞をはさんだ2語を1語にしない
                    # 用語にひらがなが入っていても、その助詞そのものが無ければ同じ(「薬で治療」→「お薬手帳」)
                    mid = PARTICLE_MID.search(surface_v)
                    if mid and mid.group(0)[1] not in term:
                        continue
                    ratio = difflib.SequenceMatcher(None, reading_v, term_reading).ratio()
                    # 読みが同じ(濁点・半濁点の違いだけも含む)なら、以下の「読みが完全一致する場合だけ」の条件を満たす
                    same = ratio == 1.0 or strip_voicing(reading_v) == strip_voicing(term_reading)
                    # 短い語は読みが完全一致する場合のみ(偶然の一致を避ける)
                    if not same and ratio < (1.0 if len(term_reading) <= 3 else 0.75):
                        continue
                    # 「は異変」→「排便」のように、助詞などのひらがなで始まる区切りを漢字の用語にしない
                    if HIRAGANA_START.search(surface_v) and not HIRAGANA_START.search(term):
                        continue
                    # 用語もひらがなで始まる場合でも、先頭の文字が違えば別の語(「に応じ」→「お通じ」)
                    if HIRAGANA_START.search(surface_v) and surface_v[0] != term[0]:
                        continue
                    # 辞書にある普通の単語(読み変換で1語として扱われる「階段」「判断」「乾燥」等)は、
                    # 読みが完全に同じ場合(「動機」→「動悸」)だけ置き換える。読みが少し違うだけでは
                    # 「階段」→「改善」「判断」→「診断」のような誤った候補ばかりになるため
                    if not same and j == i and KANJI.search(surface_v):
                        continue
                    # 送り仮名の付いた単語(「応じ」「話し」)に助詞などが続いただけの区切り(「応じて」)も同じ扱い
                    kanji_toks = [t for t, _ in toks[i : j + 1] if KANJI.search(t)]
                    if not same and len(kanji_toks) == 1 and HIRAGANA_ANY.search(kanji_toks[0]):
                        continue
                    # 漢字2文字以上の普通の単語に活用語尾などのひらがなが続いただけの区切り(「服用し」→「副作用」)も同じ扱い。
                    # 用語がその単語で始まる場合(「眼球として」→「眼球突出」)は、後ろのひらがなが誤りなので残す
                    k0 = kanji_toks[0] if len(kanji_toks) == 1 else ""
                    if not same and KANJI_WORD.match(k0) and surface_v.startswith(k0) and not term.startswith(k0):
                        continue
                    if not same and not shared_part_ok(reading_v, term_reading):
                        continue
                    # 用語の一部になっている言葉(「心房細動」の「心房」)はそれ自体が正しい語なので、
                    # 読みが少し違うだけの別の用語(「心臓」)に置き換えない。読みが完全一致する場合は残す
                    if not same and any(surface_v in w for w in term_set):
                        continue
                    found.append((starts[i], starts[i] + len(surface_v), surface_v, term, ratio))

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


@functools.lru_cache(maxsize=4)
def _term_readings(terms: tuple) -> list:
    """用語ごとの、濁点・半濁点を除いた読み(用語リストが変わらない限り使い回す)。"""
    return [(term, strip_voicing("".join(h for _, h in to_reading_tokens(term)))) for term in terms]


def relevant_terms(text: str, terms, limit: int = 300, min_ratio: float = 0.75) -> list:
    """会話の一部分(text)の中に、読みが近い並びが一続きで出てくる用語だけを選ぶ(③④のLLMに渡す手がかり)。
    用語リストがどれだけ大きくても、その部分に関係しそうな語だけを渡せるので、LLMが速く、迷いにくい。
    濁点・半濁点は除いて比べる(「感慨」の部分から「寛解」を選べるように)。
    """
    terms = tuple(terms)
    if not terms:
        return []
    reading = strip_voicing("".join(h for _, h in to_reading_tokens(text)))
    positions = {}
    for i in range(len(reading) - 1):
        positions.setdefault(reading[i : i + 2], []).append(i)
    scored = []
    for k, (term, tr) in enumerate(_term_readings(terms)):
        if len(tr) < 2:
            continue
        best = 0.0
        # 用語の読みの2文字の組が出てくる場所から、用語と同じ長さの並びを切り出して比べる
        starts = {p - j for j in range(len(tr) - 1) for p in positions.get(tr[j : j + 2], [])}
        for st in starts:
            if st < 0:
                continue
            for length in (len(tr) - 1, len(tr), len(tr) + 1):
                window = reading[st : st + length]
                if window:
                    best = max(best, difflib.SequenceMatcher(None, tr, window).ratio())
            if best == 1.0:
                break
        # 短い読み(3文字以下)は偶然の一致が多いので、完全に同じ並びがある場合だけ
        if best >= (1.0 if len(tr) <= 3 else min_ratio):
            scored.append((-best, -len(tr), k, term))
    scored.sort()
    return [t for *_, t in scored[:limit]]


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
