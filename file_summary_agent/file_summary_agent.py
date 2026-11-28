"""ファイルを読んで要約するAIエージェント

仕組み:
  1. ツール(関数)を用意する      … list_files / read_file
  2. LLM(Claude)がどのファイルを読むか判断する
  3. プログラムがツールを実行し、結果をClaudeに返す
  4. Claudeが「もう要約できる」と判断するまで 2〜3 を繰り返す

安全のため、読めるのは起動時に指定したフォルダの中だけです。
"""

import json
import sys
from pathlib import Path

import anthropic

MODEL = "claude-opus-5-5"
MAX_LINES_PER_READ = 500  # 1回に読む最大行数(長いファイルは分けて読む)

# 読んでよいフォルダ(main() で設定)
BASE_DIR = Path.cwd()


def _resolve(path: str) -> Path:
    """BASE_DIR の外のファイルを読ませないためのチェック"""
    target = (BASE_DIR / path).resolve()
    if not target.is_relative_to(BASE_DIR):
        raise PermissionError(f"{path} は読み取り可能なフォルダの外です")
    return target


# ---------------------------------------------------------------------------
# 1. ツール(関数)本体
# ---------------------------------------------------------------------------

def list_files(directory: str) -> dict:
    """フォルダ内のファイルとサブフォルダの一覧を返す"""
    target = _resolve(directory)
    if not target.is_dir():
        raise NotADirectoryError(f"{directory} はフォルダではありません")
    entries = []
    for p in sorted(target.iterdir()):
        if p.name.startswith("."):
            continue  # .git などの隠しファイルは除外
        entries.append({
            "path": str(p.relative_to(BASE_DIR)),
            "type": "folder" if p.is_dir() else "file",
            "size_bytes": None if p.is_dir() else p.stat().st_size,
        })
    return {"entries": entries}


def read_file(path: str, start_line: int) -> dict:
    """テキストファイルを start_line 行目から最大 MAX_LINES_PER_READ 行読む"""
    target = _resolve(path)
    if not target.is_file():
        raise FileNotFoundError(f"{path} が見つかりません")
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        raise ValueError(f"{path} はテキストファイルではないため読めません")

    start = max(start_line, 1)
    end = min(start + MAX_LINES_PER_READ - 1, len(lines))
    return {
        "path": path,
        "total_lines": len(lines),
        "start_line": start,
        "end_line": end,
        # 続きがある場合、Claudeは next_start_line を指定してもう一度読む
        "next_start_line": end + 1 if end < len(lines) else None,
        "content": "\n".join(lines[start - 1:end]),
    }


TOOL_FUNCTIONS = {
    "list_files": list_files,
    "read_file": read_file,
}


# ---------------------------------------------------------------------------
# 2. ツールの説明書(Claudeはこれを読んで、どのツールを使うか判断する)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "list_files",
        "description": "フォルダ内のファイルとサブフォルダの一覧を取得する。どんなファイルがあるか分からないときに使う。",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "フォルダのパス。一番上のフォルダは \".\""},
            },
            "required": ["directory"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": (
            f"テキストファイルの中身を、start_line 行目から最大 {MAX_LINES_PER_READ} 行読む。"
            "結果の next_start_line が null でなければ続きがあるので、その値を start_line にして続きを読む。"
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "ファイルのパス(list_files で得た path)"},
                "start_line": {"type": "integer", "description": "読み始める行番号。最初は 1"},
            },
            "required": ["path", "start_line"],
            "additionalProperties": False,
        },
    },
]

SYSTEM_PROMPT = (
    "あなたはファイルの内容を要約するアシスタントです。"
    "要約を頼まれたら、必ず read_file でファイルを最後まで読んでから答えてください。"
    "ファイル名が分からないときや「フォルダの中身」を聞かれたときは、まず list_files で一覧を確認します。"
    "要約は日本語で、最初に一言でまとめ、そのあと要点を箇条書きにしてください。"
    "内容から読み取れないことは推測で補わないでください。"
)


# ---------------------------------------------------------------------------
# 3. エージェントのループ
# ---------------------------------------------------------------------------

def run_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """ツールを実行し、(結果のJSON文字列, エラーかどうか) を返す"""
    try:
        result = TOOL_FUNCTIONS[name](**tool_input)
        return json.dumps(result, ensure_ascii=False), False
    except Exception as e:  # エラーもClaudeに伝えれば、別の方法を考えてくれる
        return f"エラー: {e}", True


def ask(client: anthropic.Anthropic, messages: list) -> str:
    """messages に質問が入った状態で呼ぶ。Claudeが答えを出すまでツール実行を繰り返す。"""
    while True:
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=64000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "medium"},
            # 安全フィルタで断られた場合に、別モデルで自動的にやり直す
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            response = stream.get_final_message()
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            return "この依頼にはお答えできませんでした。"

        if response.stop_reason != "tool_use":
            # Claudeが「もう答えられる」と判断した
            return "".join(b.text for b in response.content if b.type == "text")

        # Claudeが指定したツールを実行し、結果をまとめて返す
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [ツール実行] {block.name}({json.dumps(block.input, ensure_ascii=False)})")
                content, is_error = run_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                })
        messages.append({"role": "user", "content": tool_results})


def main() -> None:
    global BASE_DIR
    # 使い方: python file_summary_agent.py <フォルダ> ["質問"]
    BASE_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if not BASE_DIR.is_dir():
        sys.exit(f"フォルダが見つかりません: {BASE_DIR}")

    client = anthropic.Anthropic()  # APIキーは環境変数 ANTHROPIC_API_KEY から読み込まれる
    messages: list = []

    # 質問があれば1回だけ答えて終了、なければ対話モード
    if len(sys.argv) > 2:
        messages.append({"role": "user", "content": " ".join(sys.argv[2:])})
        print(ask(client, messages))
        return

    print(f"要約エージェントです。対象フォルダ: {BASE_DIR}")
    print("例:「README.md を要約して」「このフォルダには何がある？」(終了: exit)")
    while True:
        try:
            question = input("\nあなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit", "終了"):
            break
        if not question:
            continue
        messages.append({"role": "user", "content": question})
        print(f"\nエージェント> {ask(client, messages)}")


if __name__ == "__main__":
    main()
