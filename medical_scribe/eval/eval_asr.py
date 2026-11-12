"""
音声認識(文字起こし)の精度を測るスクリプト(Mac上で実行。音声は外部に送らない)

フォルダに「録音」と「正しい文字起こし」を同じ名前で置きます。
    asr_test/
        case01.m4a   case01.txt   ← txt は人が直した正しい文字起こし
        case02.m4a   case02.txt

測るもの:
  - 文字誤り率(CER): 正解と比べて、何%の文字を間違えたか(置換・脱落・挿入の合計 ÷ 正解の文字数)。
    話者ラベル・句読点・空白は除いて比べます。「ください/下さい」のような表記の違いも誤りに数えます。
  - 医学用語の認識率: 正解に出てくる用語リスト(term_lists)の語のうち、文字起こしにも正しく出た割合。
    取りこぼした用語を一覧にします(今の課題がどの語か分かる)。

使い方:
  python3 eval_asr.py --dir asr_test --engines mlx:large-v3-turbo mlx:large-v3 kotoba
  python3 eval_asr.py --dir asr_test --engines mlx:large-v3-turbo --hint "甲状腺、バセドウ病、動悸"
  python3 eval_asr.py --dir asr_test --engines mlx:large-v3-turbo --reuse --sound-match

  --sound-match  アプリの①-2(用語の音照合)の候補を「すべて採用した場合」の成績も出します。
                 候補のうち正解と合うものの数も数えるので、①-2がどれだけ取り戻せて、どれだけ誤った候補を
                 出しているかが分かります。用語リストは term_lists の全ファイル(--term-files で絞れます)。
  マイクの録り比べ: 同じ会話を2台で同時に録り、「会話A__今.m4a」「会話A__外付け.m4a」のように
                 「__」の後ろにマイクの名前を付けると、どちらも正解「会話A.txt」1つで採点します。
  --reuse        前回の文字起こし結果(文字起こし結果フォルダ)があれば使い回します。①-2や採点だけを
                 調整して測り直すときに、時間のかかる文字起こしを省けます。

  正解の txt がまだ無い録音は、--draft を付けると最初のエンジンの結果を txt として書き出します。
  それを開いて正しく直してから、もう一度実行してください(直していない下書きで測ると100点になります)。

エンジンの書き方:
  mlx:medium / mlx:large-v3 / mlx:large-v3-turbo  … mlx-whisper(アプリの初期設定は large-v3-turbo)
  kotoba                                           … kotoba-whisper v2.0(日本語特化)
"""

import argparse
import csv
import os
import re
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
TERM_LIST_DIR = os.path.join(HERE, "..", "term_lists")
sys.path.insert(0, os.path.join(HERE, ".."))  # アプリと同じ音照合(sound_match.py)を使う
AUDIO_EXTS = (".m4a", ".mp3", ".wav", ".mp4", ".mov")
MLX_REPOS = {
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
KOTOBA_MODEL = "kotoba-tech/kotoba-whisper-v2.0"
FIELDS = [
    "case",
    "engine",
    "ref_chars",
    "cer",
    "terms_in_ref",
    "terms_ok",
    "missed_terms",
    "seconds",
    "cer_sound",
    "terms_ok_sound",
    "missed_terms_sound",
    "suggestions",
    "suggestions_ok",
    "wrong_suggestions",
]


# ---------- 文字起こし ----------


def transcribe(engine: str, audio_path: str, hint: str) -> str:
    if engine.startswith("mlx:"):
        import mlx_whisper

        size = engine[len("mlx:") :]
        prompt = f"これは医師と患者の診察会話です。次のような医学用語が含まれます: {hint}" if hint else None
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=MLX_REPOS[size],
            language="ja",
            initial_prompt=prompt,
            condition_on_previous_text=False,
        )
        return result["text"]
    if engine == "kotoba":
        return transcribe_kotoba(audio_path)
    raise ValueError(f"不明なエンジン: {engine}(mlx:large-v3-turbo / kotoba などを指定)")


_kotoba_pipe = None


def transcribe_kotoba(audio_path: str) -> str:
    global _kotoba_pipe
    import subprocess

    import numpy as np

    if _kotoba_pipe is None:
        import torch
        from transformers import pipeline

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _kotoba_pipe = pipeline(
            "automatic-speech-recognition",
            model=KOTOBA_MODEL,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            device=device,
        )
    cmd = ["ffmpeg", "-nostdin", "-i", audio_path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    out = _kotoba_pipe(
        {"raw": audio, "sampling_rate": 16000},
        chunk_length_s=15,
        batch_size=1,
        generate_kwargs={"language": "ja", "task": "transcribe"},
    )
    return out["text"]


# ---------- 採点 ----------


def normalize(text: str) -> str:
    """比較用に、話者ラベル・句読点・記号・空白を除き、全角英数を半角にそろえる。"""
    text = unicodedata.normalize("NFKC", text)
    # 話者ラベルは行頭とは限らない(1行に続けて書かれた文字起こしもある)ので、どこにあっても除く
    text = re.sub(r"(医師|患者|家族|看護師|Dr|Pt|SPEAKER_\d+)\s*[:：]", "", text)
    return re.sub(r"[\s、。，．,.!?！？「」『』()（）\[\]・…ー~〜:：;；\"']", "", text)


def edit_distance(a: str, b: str) -> int:
    """文字単位の編集距離(置換・脱落・挿入の最少回数)。"""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_terms() -> list:
    terms = set()
    if os.path.isdir(TERM_LIST_DIR):
        for name in os.listdir(TERM_LIST_DIR):
            if name.endswith(".txt") and not name.startswith("_"):
                with open(os.path.join(TERM_LIST_DIR, name), encoding="utf-8") as fh:
                    terms |= {normalize(ln) for ln in fh if ln.strip() and not ln.lstrip().startswith("#")}
    return sorted(t for t in terms if len(t) >= 2)


def term_check(ref: str, hyp: str, terms: list):
    """正解に出てくる用語ごとに、文字起こしにも同じ回数以上出たかを数える。"""
    in_ref, ok, missed = 0, 0, []
    for t in terms:
        n_ref = ref.count(t)
        if not n_ref:
            continue
        n_ok = min(n_ref, hyp.count(t))
        in_ref += n_ref
        ok += n_ok
        if n_ok < n_ref:
            missed.append(t if n_ref == 1 else f"{t}({n_ref - n_ok}/{n_ref}回)")
    return in_ref, ok, missed


def apply_sound_match(hyp_text: str, ref: str, term_files: list):
    """アプリの①-2と同じ方法で候補を出し、すべて採用した文字起こしを返す。
    あわせて、候補が正解と合うか(置き換えた語が、前後の文脈ごと正解に出てくるか)を目安として数える。
    """
    import sound_match

    terms = []
    for name in term_files:
        terms += sound_match.load_term_file(name)
    terms = tuple(dict.fromkeys(terms))
    entries, index = sound_match.build_sound_index(terms)
    matches = sound_match.find_sound_alike(hyp_text, entries, index, set(terms) | sound_match.stop_words())
    fixed = sound_match.apply_sound_replacements(hyp_text, matches, {(m[2], m[3]) for m in matches})
    # 正解に足りない用語の回数(正解での回数 − 文字起こしでの回数)。前後の文字も誤っていて
    # 文脈では判定できない候補も、この不足分を埋めるものなら「合う」と数える
    hyp_norm = normalize(hyp_text)
    shortage = {t: ref.count(normalize(t)) - hyp_norm.count(normalize(t)) for t in {m[3] for m in matches}}
    ok, wrong = 0, []
    for start, end, surface, term, _ in matches:
        left, right = hyp_text[max(0, start - 3) : start], hyp_text[end : end + 3]
        if normalize(left + term) in ref or normalize(term + right) in ref or shortage[term] > 0:
            ok += 1
            shortage[term] -= 1
        else:
            wrong.append(f"{surface}→{term}")
    return fixed, len(matches), ok, wrong


def read_text(path: str) -> str:
    """正解の txt を読む。テキストエディットやWindowsで保存すると UTF-8 以外(Shift_JIS 等)に
    なることがあるので、日本語でよく使われる文字コードを順に試す。"""
    with open(path, "rb") as fh:
        data = fh.read()
    if data.lstrip().startswith(b"{\\rtf"):
        sys.exit(
            f"{path} はリッチテキスト(RTF)で保存されています。テキストエディットで開き、"
            "「フォーマット」→「標準テキストにする」を選んでから保存し直してください。"
        )
    for encoding in ("utf-8-sig", "cp932", "euc_jp", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    sys.exit(f"{path} の文字コードを判別できませんでした。UTF-8で保存し直してください。")


def diff_lines(ref: str, hyp: str, context: int = 6) -> list:
    """正解と文字起こしが食い違う箇所を「正解 / 文字起こし」の形で並べる(前後の文字も付ける)。"""
    import difflib

    lines = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        before, after = ref[max(0, i1 - context) : i1], ref[i2 : i2 + context]
        lines.append(f"…{before}【{ref[i1:i2] or '(なし)'}】{after}…  ←  文字起こし:【{hyp[j1:j2] or '(なし)'}】")
    return lines


# ---------- 本体 ----------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="録音と正解txtを置いたフォルダ")
    ap.add_argument("--engines", nargs="+", default=["mlx:large-v3-turbo"])
    ap.add_argument("--hint", default="", help="Whisperに渡す医学用語のヒント(アプリのサイドバーと同じもの)")
    ap.add_argument("--draft", action="store_true", help="正解txtが無い録音は、最初のエンジンの結果を下書きとして書き出す")
    ap.add_argument("--sound-match", action="store_true", help="①-2の音照合の候補をすべて採用した場合の成績も出す")
    ap.add_argument("--term-files", nargs="*", default=None, help="①-2に使う用語リスト(例: 01_内科一般 05_内分泌代謝)。省略時は全部")
    ap.add_argument("--reuse", action="store_true", help="前回の文字起こし結果があれば使い回す(文字起こしを省く)")
    ap.add_argument("--out", default="asr_result.csv")
    args = ap.parse_args()
    if args.sound_match and args.term_files is None:
        import sound_match

        args.term_files = sound_match.list_term_files()

    audios = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(AUDIO_EXTS))
    if not audios:
        sys.exit(f"{args.dir} に録音ファイル({' / '.join(AUDIO_EXTS)})がありません。")
    terms = load_terms()
    hyp_dir = os.path.join(args.dir, "文字起こし結果")
    os.makedirs(hyp_dir, exist_ok=True)

    rows, drafted = [], set()
    for audio in audios:
        case = os.path.splitext(audio)[0]
        ref_path = os.path.join(args.dir, case + ".txt")
        # 「会話A__今.m4a」「会話A__外付け.m4a」のように同じ会話を別のマイクで録ったものは、
        # 正解「会話A.txt」を共通で使う(マイクの録り比べ用)
        if "__" in case and not os.path.exists(ref_path):
            ref_path = os.path.join(args.dir, case.split("__")[0] + ".txt")
        # 今回の実行で下書きを作った正解は、まだ直されていないので採点に使わない
        has_ref = os.path.exists(ref_path) and ref_path not in drafted
        if not has_ref and not args.draft:
            print(f"[{case}] 正解の {case}.txt がないので飛ばします(--draft で下書きを作れます)")
            continue
        for k, engine in enumerate(args.engines):
            hyp_path = os.path.join(hyp_dir, f"{case}__{engine.replace(':', '_')}.txt")
            if args.reuse and os.path.exists(hyp_path):
                with open(hyp_path, encoding="utf-8") as fh:
                    hyp_text = fh.read()
                seconds = ""
                print(f"[{case}] {engine}: 前回の文字起こし結果を使います")
            else:
                print(f"[{case}] {engine} で文字起こし中…", end="", flush=True)
                start = time.time()
                hyp_text = transcribe(engine, os.path.join(args.dir, audio), args.hint)
                seconds = round(time.time() - start, 1)
                print(f" {seconds}秒")
                with open(hyp_path, "w", encoding="utf-8") as fh:
                    fh.write(hyp_text)
            if not has_ref:
                if k == 0 and not os.path.exists(ref_path):
                    with open(ref_path, "w", encoding="utf-8") as fh:
                        fh.write(hyp_text)
                    drafted.add(ref_path)
                    print(f"  → 下書きを {ref_path} に書き出しました。正しく直してから再実行してください。")
                continue
            ref = normalize(read_text(ref_path))
            hyp = normalize(hyp_text)
            in_ref, ok, missed = term_check(ref, hyp, terms)
            # 食い違う箇所の一覧(どの語がどう誤認識されたかを見るため)
            with open(hyp_path.replace(".txt", "__食い違い.txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(diff_lines(ref, hyp)))
            sound = {}
            if args.sound_match:
                fixed, n_sug, n_ok, wrong = apply_sound_match(hyp_text, ref, args.term_files)
                fixed_norm = normalize(fixed)
                _, ok_s, missed_s = term_check(ref, fixed_norm, terms)
                sound = {
                    "cer_sound": round(100 * edit_distance(ref, fixed_norm) / max(1, len(ref)), 1),
                    "terms_ok_sound": ok_s,
                    "missed_terms_sound": "、".join(missed_s),
                    "suggestions": n_sug,
                    "suggestions_ok": n_ok,
                    "wrong_suggestions": "、".join(wrong),
                }
            rows.append(
                {
                    **sound,
                    "case": case,
                    "engine": engine,
                    "ref_chars": len(ref),
                    "cer": round(100 * edit_distance(ref, hyp) / max(1, len(ref)), 1),
                    "terms_in_ref": in_ref,
                    "terms_ok": ok,
                    "missed_terms": "、".join(missed),
                    "seconds": seconds,
                }
            )

    if not rows:
        return
    with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== 症例ごと(文字誤り率は低いほど良い) ===")
    for r in rows:
        print(
            f"{r['case']}  {r['engine']}: 文字誤り率 {r['cer']}%  用語 {r['terms_ok']}/{r['terms_in_ref']}"
            + (f"  {r['seconds']}秒" if r["seconds"] != "" else "")
            + (f"\n    取りこぼした用語: {r['missed_terms']}" if r["missed_terms"] else "")
        )
        if "suggestions" in r:
            print(
                f"    ①-2をすべて採用: 文字誤り率 {r['cer_sound']}%  用語 {r['terms_ok_sound']}/{r['terms_in_ref']}"
                f"  候補 {r['suggestions']}件(正解と合う {r['suggestions_ok']}件)"
                + (f"\n      まだ取りこぼしている用語: {r['missed_terms_sound']}" if r["missed_terms_sound"] else "")
                + (f"\n      正解と合わない候補: {r['wrong_suggestions']}" if r["wrong_suggestions"] else "")
            )
    print("\n=== エンジンごとの合計 ===")
    for engine in args.engines:
        rs = [r for r in rows if r["engine"] == engine]
        if not rs:
            continue
        chars = sum(r["ref_chars"] for r in rs)
        cer = sum(r["cer"] * r["ref_chars"] for r in rs) / max(1, chars)
        t_in = sum(r["terms_in_ref"] for r in rs)
        t_ok = sum(r["terms_ok"] for r in rs)
        rate = f"{100 * t_ok / t_in:.0f}%" if t_in else "-"
        print(f"{engine}: 文字誤り率 {cer:.1f}%  医学用語の認識率 {rate}({t_ok}/{t_in})  {len(rs)}件")
        if all("suggestions" in r for r in rs):
            cer_s = sum(r["cer_sound"] * r["ref_chars"] for r in rs) / max(1, chars)
            ok_s = sum(r["terms_ok_sound"] for r in rs)
            n_sug = sum(r["suggestions"] for r in rs)
            n_ok = sum(r["suggestions_ok"] for r in rs)
            rate_s = f"{100 * ok_s / t_in:.0f}%" if t_in else "-"
            print(
                f"  └ ①-2をすべて採用: 文字誤り率 {cer_s:.1f}%  医学用語の認識率 {rate_s}({ok_s}/{t_in})"
                f"  候補 {n_sug}件のうち正解と合う {n_ok}件"
            )
    print(f"\n詳細: {args.out} / 文字起こし全文と食い違いの一覧(…__食い違い.txt): {hyp_dir}")


if __name__ == "__main__":
    main()
