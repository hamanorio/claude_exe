"""
⑧ チームでの検討(試験的): 振り分け係 → 専門医係 → 反論係 → まとめ係 の順に、
同じLLMに役割を変えて検討させ、鑑別診断の参考を作る。

- 診断を下すものではなく、医師が見落としを確かめるための参考情報。
- 危険な兆候は、LLMの判断とは別に、キーワードでも機械的に確認する(LLMの見落としに備える)。
- LLMの呼び出し方は引数 llm(prompt, system) で受け取るので、アプリ(app.py)からも
  評価スクリプト(eval/eval_ddx.py)からも同じ処理を使える。Streamlitには依存しない。
"""

import re

DEPARTMENTS = [
    "循環器",
    "呼吸器",
    "消化器",
    "内分泌・代謝",
    "神経",
    "腎臓・泌尿器",
    "血液",
    "感染症",
    "膠原病・リウマチ",
    "精神・心療内科",
    "婦人科",
    "整形外科",
    "皮膚科",
    "耳鼻咽喉科",
    "眼科",
]

# キーワードによる危険な兆候の確認(見落とし防止のため広めに拾う。誤って拾うこともある)
RED_FLAG_RULES = [
    ("脳卒中の可能性(突然の麻痺・構音障害など)", r"片麻痺|麻痺|呂律|構音障害|言葉が出にく|力が入らな"),
    ("くも膜下出血の可能性(突然の激しい頭痛)", r"(突然|急に).{0,10}(激し|ひど|割れる).{0,5}頭痛|人生最悪の頭痛|項部硬直"),
    ("意識障害・失神", r"意識障害|意識消失|意識がな|失神|気を失"),
    ("急性冠症候群の可能性(胸痛・胸部圧迫感)", r"胸痛|胸部圧迫感|胸が(痛|締め付|しめつけ)"),
    ("大動脈解離の可能性(引き裂かれるような痛み)", r"引き裂|裂けるような"),
    ("消化管出血の可能性", r"吐血|下血|黒色便|タール便|血便"),
    ("呼吸不全の可能性", r"呼吸困難|起座呼吸|横になれな|酸素飽和度\D{0,6}([1-8]\d)(?!\d)"),
    ("ショックの可能性(低血圧)", r"血圧\D{0,6}([4-8]\d)(?!\d)"),
    ("髄膜炎の可能性", r"項部硬直"),
    ("無顆粒球症の可能性(抗甲状腺薬内服中の発熱・咽頭痛)", r"(チアマゾール|プロピルチオウラシル|抗甲状腺薬).*(発熱|のどの痛|咽頭痛)|(発熱|のどの痛|咽頭痛).*(チアマゾール|プロピルチオウラシル|抗甲状腺薬)"),
]


# キーワードと否定の言葉の間が助詞など(「胸痛は」「麻痺も今は」)だけのときに否定とみなす。
# 「呂律が回らない」「力が入らない」のように動詞を挟む場合は症状そのものなので否定にしない
NEGATION = re.compile(r"^[\sはもがのにで、:：今特に]{0,6}(なし|ない|なく|ありません|認めず|認めない|陰性|\(-\))")


def rule_red_flags(history: str) -> list:
    """キーワードで危険な兆候を拾う。直後が否定(「胸痛はありません」「麻痺もなく」)の箇所は除く。"""
    flags = []
    for name, pattern in RED_FLAG_RULES:
        for m in re.finditer(pattern, history):
            if not NEGATION.search(history[m.end() : m.end() + 12]):
                flags.append(name)
                break
    return flags


# ---------- 各係のプロンプト ----------

TRIAGE_SYSTEM = "あなたは総合診療の経験が豊富な医師で、外来患者の振り分けを担当しています。"


def triage_prompt(history: str) -> str:
    return f"""次の病歴を読み、患者の振り分けをしてください。診断名を決める必要はありません。

出力は次の4行だけにしてください(説明や前置きは不要です)。
緊急度: 緊急 / 準緊急 / 通常 のどれか
危険な兆候: 病歴にある危険な兆候を「、」区切りで。なければ「なし」
診療科: 次の中から関係が深い順に最大2つを「、」区切りで({"、".join(DEPARTMENTS)})
理由: 1文で

病歴に書かれていない所見を作らないでください。

【病歴】
{history}
"""


def specialist_system(department: str) -> str:
    return f"あなたは{department}の専門医です。外来の症例検討会で意見を述べます。"


def specialist_prompt(history: str, department: str) -> str:
    return f"""次の病歴について、{department}の専門医として考えられる疾患を挙げてください。

1行に1疾患、可能性が高い順に最大4つを、次の形式で書いてください。前置きや説明は不要です。
疾患名｜支持する所見｜合わない所見(なければ「なし」)｜次に行う検査

支持する所見・合わない所見は、病歴に書かれていることだけを使ってください。
書かれていない検査値や所見を作らないでください。

【病歴】
{history}
"""


CRITIC_SYSTEM = "あなたは症例検討会で、議論の見落としや思い込みを指摘する役の医師です。"


def critic_prompt(history: str, opinions: str) -> str:
    return f"""次の病歴と、専門医たちの意見を読み、問題点を指摘してください。

次の2種類を、1行に1つずつ書いてください。前置きは不要です。
見逃し注意｜疾患名｜理由   (意見に挙がっていないが、見逃すと危険な疾患)
説明不足｜所見｜理由       (どの意見でも説明されていない、病歴の所見)
指摘がなければ「指摘なし」とだけ書いてください。

【病歴】
{history}

【専門医たちの意見】
{opinions}
"""


SUMMARY_SYSTEM = "あなたは症例検討会の司会を務める医師で、議論をまとめます。"


def summary_prompt(history: str, opinions: str, critique: str) -> str:
    return f"""病歴、専門医たちの意見、指摘をふまえて、鑑別診断をまとめてください。

1行に1疾患、可能性が高い順に最大5つを、次の形式で書いてください。前置きや説明は不要です。
順位｜疾患名｜根拠となる所見｜次に行う検査

根拠は病歴に書かれている所見だけを使ってください。見逃すと危険な疾患は、可能性が低くても含めてください。

【病歴】
{history}

【専門医たちの意見】
{opinions}

【指摘】
{critique}
"""


# ---------- 出力の読み取り ----------


def _cells(line: str) -> list:
    line = re.sub(r"^\s*(?:[-*・]|\d+[.)．])\s+", "", line.strip()).strip("|｜ ")
    return [c.strip(" *`") for c in re.split(r"[｜|]", line)]


def parse_triage(text: str, departments_limit: int = 2) -> dict:
    def field(name):
        m = re.search(rf"{name}\s*[:：]\s*(.+)", text)
        return m.group(1).strip() if m else ""

    urgency = next((u for u in ("緊急", "準緊急", "通常") if field("緊急度").startswith(u)), "")
    flags = [f.strip() for f in re.split(r"[、,，]", field("危険な兆候")) if f.strip() and f.strip() != "なし"]
    named = field("診療科")
    # 一覧にある診療科だけを使う(「内分泌」のような書き方の揺れも拾う)
    depts = [d for d in DEPARTMENTS if any(part and part in named for part in re.split(r"[・]", d))]
    depts.sort(key=lambda d: min((named.find(p) for p in d.split("・") if p in named), default=99))
    return {"urgency": urgency, "flags": flags, "departments": depts[:departments_limit], "reason": field("理由")}


def parse_table(text: str, min_cells: int) -> list:
    rows = []
    for line in text.splitlines():
        cells = _cells(line)
        if len(cells) < min_cells or not cells[0] or set(cells[0]) <= set("-: ") or cells[0] in ("疾患名", "順位"):
            continue
        rows.append(cells)
    return rows


# ---------- 本体 ----------


def run_team(history: str, llm, progress=None, departments_limit: int = 2) -> dict:
    """llm(prompt, system) -> str を使って、4つの係の検討を順に行う。
    progress(step_name, i, n) で進み具合を知らせる。途中の出力もすべて返す(医師が確かめられるように)。
    """
    n_steps = 3 + departments_limit

    def step(name, i):
        if progress:
            progress(name, i, n_steps)

    step("振り分け係", 0)
    triage_raw = llm(triage_prompt(history), TRIAGE_SYSTEM)
    triage = parse_triage(triage_raw, departments_limit)
    if not triage["departments"]:
        triage["departments"] = ["総合診療"]  # 診療科を読み取れなかったときは、総合診療医として検討させる

    specialists = []
    for k, dept in enumerate(triage["departments"]):
        step(f"専門医係({dept})", 1 + k)
        raw = llm(specialist_prompt(history, dept), specialist_system(dept))
        specialists.append({"department": dept, "raw": raw, "rows": parse_table(raw, 2)})
    opinions = "\n".join(f"[{s['department']}]\n{s['raw']}" for s in specialists)

    step("反論係", 1 + len(specialists))
    critique_raw = llm(critic_prompt(history, opinions), CRITIC_SYSTEM)
    critique = parse_table(critique_raw, 2)

    step("まとめ係", 2 + len(specialists))
    summary_raw = llm(summary_prompt(history, opinions, critique_raw), SUMMARY_SYSTEM)
    final = parse_table(summary_raw, 2)
    step("完了", n_steps)

    return {
        "rule_flags": rule_red_flags(history),
        "triage": triage,
        "triage_raw": triage_raw,
        "specialists": specialists,
        "critique": critique,
        "critique_raw": critique_raw,
        "final": final,
        "summary_raw": summary_raw,
    }


def final_table(result: dict) -> list:
    """まとめ係の鑑別を、画面の表に出す形(順位・疾患名・根拠・次の検査)にする。"""
    rows = []
    for k, cells in enumerate(result["final"], 1):
        if re.fullmatch(r"\d+", cells[0]):
            rank, cells = cells[0], cells[1:]
        else:
            rank = str(k)
        cells = cells + [""] * (3 - len(cells))
        rows.append({"順位": rank, "疾患名": cells[0], "根拠となる所見": cells[1], "次に行う検査": cells[2]})
    return rows


def specialist_table(specialist: dict) -> list:
    rows = []
    for cells in specialist["rows"]:
        cells = cells + [""] * (4 - len(cells))
        rows.append({"疾患名": cells[0], "支持する所見": cells[1], "合わない所見": cells[2], "次に行う検査": cells[3]})
    return rows


def all_text(result: dict) -> str:
    """検討の出力全部(病歴に無い数値や、触れられていない所見の確認に使う)。"""
    return "\n".join(
        [result["triage_raw"]] + [s["raw"] for s in result["specialists"]] + [result["critique_raw"], result["summary_raw"]]
    )


def team_as_text(result: dict) -> str:
    """評価スクリプト(eval_ddx.py)の採点で読める形(「最終診断」「鑑別診断」の見出し)にする。"""
    names = [row["疾患名"] for row in final_table(result)]
    lines = ["## 鑑別診断"] + [f"{i + 1}. {n}" for i, n in enumerate(names)]
    lines += ["", "## 最終診断", names[0] if names else ""]
    lines += ["", "## 検討の記録", result["triage_raw"]]
    lines += [s["raw"] for s in result["specialists"]] + [result["critique_raw"], result["summary_raw"]]
    return "\n".join(lines)
