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
AUDIO_EXTS = (".m4a", ".mp3", ".wav", ".mp4", ".mov")
MLX_REPOS = {
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
KOTOBA_MODEL = "kotoba-tech/kotoba-whisper-v2.0"
FIELDS = ["case", "engine", "ref_chars", "cer", "terms_in_ref", "terms_ok", "missed_terms", "seconds"]


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


# ---------- 本体 ----------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="録音と正解txtを置いたフォルダ")
    ap.add_argument("--engines", nargs="+", default=["mlx:large-v3-turbo"])
    ap.add_argument("--hint", default="", help="Whisperに渡す医学用語のヒント(アプリのサイドバーと同じもの)")
    ap.add_argument("--draft", action="store_true", help="正解txtが無い録音は、最初のエンジンの結果を下書きとして書き出す")
    ap.add_argument("--out", default="asr_result.csv")
    args = ap.parse_args()

    audios = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(AUDIO_EXTS))
    if not audios:
        sys.exit(f"{args.dir} に録音ファイル({' / '.join(AUDIO_EXTS)})がありません。")
    terms = load_terms()
    hyp_dir = os.path.join(args.dir, "文字起こし結果")
    os.makedirs(hyp_dir, exist_ok=True)

    rows = []
    for audio in audios:
        case = os.path.splitext(audio)[0]
        ref_path = os.path.join(args.dir, case + ".txt")
        has_ref = os.path.exists(ref_path)
        if not has_ref and not args.draft:
            print(f"[{case}] 正解の {case}.txt がないので飛ばします(--draft で下書きを作れます)")
            continue
        for k, engine in enumerate(args.engines):
            print(f"[{case}] {engine} で文字起こし中…", end="", flush=True)
            start = time.time()
            hyp_text = transcribe(engine, os.path.join(args.dir, audio), args.hint)
            seconds = round(time.time() - start, 1)
            print(f" {seconds}秒")
            with open(os.path.join(hyp_dir, f"{case}__{engine.replace(':', '_')}.txt"), "w", encoding="utf-8") as fh:
                fh.write(hyp_text)
            if not has_ref:
                if k == 0:
                    with open(ref_path, "w", encoding="utf-8") as fh:
                        fh.write(hyp_text)
                    print(f"  → 下書きを {ref_path} に書き出しました。正しく直してから再実行してください。")
                continue
            with open(ref_path, encoding="utf-8") as fh:
                ref = normalize(fh.read())
            hyp = normalize(hyp_text)
            in_ref, ok, missed = term_check(ref, hyp, terms)
            rows.append(
                {
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
            f"  {r['seconds']}秒" + (f"\n    取りこぼした用語: {r['missed_terms']}" if r["missed_terms"] else "")
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
    print(f"\n詳細: {args.out} / 文字起こし全文: {hyp_dir}")


if __name__ == "__main__":
    main()
