"""
問診アシスタントの処理部分(Streamlitに依存しない)。

医師が診察中に使う: 医師が患者さんから聞いた答えを入力すると、LLMが
  - 今考えられる鑑別診断(可能性の高さ付き)
  - 見逃してはいけない疾患
  - 次に聞くとよい質問(どの疾患を見分けるためか付き)
を返す。これを繰り返し、最後に問診のまとめと鑑別・次の診察や検査を出す。

- 診断を下すものではなく、医師の問診を補助する参考情報。
- 危険な兆候は、LLMとは別にキーワードでも確認する(ddx_agents.rule_red_flags)。
- LLMの呼び出しは llm(prompt, system) の形で受け取るので、画面からも評価スクリプトからも使える。
"""

import json
import re
import urllib.error
import urllib.request

import ddx_agents

OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM = (
    "あなたは外来で医師の問診を補助するAIです。利用者は医師で、医師が患者さんから聞いた情報をもとに、"
    "鑑別診断を考え、診断を絞り込むために次に聞くべきことを提案します。"
    "これは医師が自分で判断するための参考情報なので、情報が少ない段階でも、考えられる鑑別を挙げることが適切です。"
    "「情報が足りないので鑑別を挙げられない」とは答えず、今ある情報で考えられる候補と、絞り込むための質問を必ず示してください。"
    "最終的な判断は医師が行います。"
)


# ---------- 会話の記録 ----------


def new_case(patient: str, chief_complaint: str) -> dict:
    return {"patient": patient.strip(), "chief_complaint": chief_complaint.strip(), "exchanges": []}


def add_exchange(case: dict, question: str, answer: str, source: str = "AIの提案") -> None:
    """医師が聞いた質問と患者さんの答えを記録する。答えが空なら記録しない。"""
    if question.strip() and answer.strip():
        case["exchanges"].append({"question": question.strip(), "answer": answer.strip(), "source": source})


def history_text(case: dict) -> str:
    """LLMに渡す、これまでに分かったこと。"""
    lines = []
    if case["patient"]:
        lines.append(f"患者: {case['patient']}")
    lines.append(f"主訴: {case['chief_complaint']}")
    for i, e in enumerate(case["exchanges"], 1):
        lines.append(f"[{i}] 質問: {e['question']}\n    答え: {e['answer']}")
    return "\n".join(lines)


def red_flags(case: dict) -> list:
    """キーワードによる危険な兆候(LLMを使わない)。答えと主訴だけを見る(質問文は見ない)。"""
    text = "\n".join([case["chief_complaint"]] + [e["answer"] for e in case["exchanges"]])
    return ddx_agents.rule_red_flags(text)


# ---------- 次の質問の提案 ----------


# ---------- 主訴ごとの基本の確認項目(LLMを使わない) ----------
# AIが聞き忘れても医師が気づけるよう、決まった表から表示する。
# 各項目は (表示名, 聞いたかどうかを判定するキーワード)。質問か答えにキーワードがあれば「聞いた」とみなす(目安)。
COMPLAINT_CHECKLISTS = {
    "発疹": {
        "triggers": r"発疹|皮疹|湿疹|ブツブツ|ぶつぶつ|赤い斑|かぶれ|じんましん|蕁麻疹|あざ|紫斑",
        "items": [
            ("分布(どこに、どう広がっているか)", r"どこ|部位|分布|広が|場所"),
            ("押して色が消えるか(紫斑かどうか)", r"押|圧迫|消え|紫斑"),
            ("触れて盛り上がっているか", r"盛り上|隆起|平ら|触"),
            ("かゆみ・痛み", r"かゆ|痒|痛"),
            ("水疱・膿疱", r"水疱|水ぶくれ|膿"),
            ("口・目・陰部など粘膜の症状", r"口|目|粘膜|陰部|唇"),
            ("発熱", r"熱"),
            ("腹痛・関節痛", r"腹|おなか|お腹|関節"),
            ("新しい薬・食べ物・接触", r"薬|服用|内服|食べ|洗剤|化粧|靴|接触"),
            ("先行する感染(かぜ・のどの痛み)", r"かぜ|風邪|のど|咽頭|感染"),
        ],
    },
    "胸痛": {
        "triggers": r"胸痛|胸が痛|胸の痛|胸が苦し|胸部|胸が締め|胸が重",
        "items": [
            ("突然か・いつからか", r"突然|急に|いつ"),
            ("労作との関係", r"動|階段|労作|運動|歩"),
            ("痛みの広がり(肩・腕・顎・背中)", r"広が|肩|腕|顎|あご|背中"),
            ("呼吸や体位で変わるか", r"呼吸|息を吸|体位|姿勢|横にな|前かが"),
            ("冷汗", r"冷や汗|冷汗"),
            ("持続時間", r"何分|続|時間"),
            ("危険因子(高血圧・糖尿病・喫煙・脂質)", r"高血圧|糖尿病|喫煙|タバコ|たばこ|コレステロール|脂質"),
            ("足の腫れ・長時間の移動", r"足|ふくらはぎ|飛行機|旅行|移動"),
        ],
    },
    "腹痛": {
        "triggers": r"腹痛|おなか|お腹|腹が痛|みぞおち|下腹",
        "items": [
            ("部位", r"どこ|部位|場所"),
            ("痛む場所の移動", r"移|変わ"),
            ("始まり方(突然か)", r"突然|急に|いつ"),
            ("食事との関係", r"食事|食べ|食後|空腹"),
            ("排便(下痢・便秘・血便)", r"便|下痢|排便"),
            ("吐き気・嘔吐", r"吐"),
            ("発熱", r"熱"),
            ("妊娠の可能性・最終月経", r"生理|月経|妊娠"),
            ("排尿の症状", r"尿|おしっこ|トイレ"),
        ],
    },
    "頭痛": {
        "triggers": r"頭痛|頭が痛|頭の痛",
        "items": [
            ("突然か・今までで最悪か", r"突然|急|最悪|今まで"),
            ("発熱", r"熱"),
            ("首の痛み・こわばり", r"首|うなじ|項部"),
            ("神経の症状(しびれ・麻痺・言葉・見え方)", r"しびれ|麻痺|力が入|ろれつ|呂律|言葉|見え|二重"),
            ("吐き気・嘔吐", r"吐"),
            ("頭を打ったか", r"打|ぶつけ|転|けが"),
            ("以前からの頭痛か", r"以前|いつも|普段|繰り返|前にも"),
            ("飲んでいる薬", r"薬"),
        ],
    },
    "発熱": {
        "triggers": r"発熱|熱が|熱っぽ|高熱|微熱",
        "items": [
            ("いつから・何日続くか", r"いつ|何日|期間|続"),
            ("最高体温", r"何度|最高|体温"),
            ("熱以外の症状がある部位(咳・のど・尿・腹・皮膚・頭)", r"咳|のど|尿|腹|下痢|発疹|皮疹|頭痛"),
            ("寒気・震え", r"寒気|震え|悪寒"),
            ("海外渡航", r"旅行|渡航|海外"),
            ("周りに同じ症状の人", r"周り|家族|職場|学校|同じ症状"),
            ("飲んでいる薬", r"薬"),
        ],
    },
    "呼吸困難": {
        "triggers": r"息苦し|息切れ|呼吸困難|息が苦し|呼吸が苦し|息がしにく",
        "items": [
            ("突然か・いつからか", r"突然|急に|いつ"),
            ("動いたときか、安静時か", r"動|階段|労作|安静"),
            ("横になると苦しいか", r"横にな|寝|夜"),
            ("胸の痛み", r"胸"),
            ("咳・痰・血痰", r"咳|痰"),
            ("発熱", r"熱"),
            ("足のむくみ・ふくらはぎの腫れ", r"足|むくみ|ふくらはぎ"),
            ("持病(喘息・心臓・肺)・喫煙", r"喘息|心臓|肺|タバコ|たばこ|喫煙"),
        ],
    },
    "めまい": {
        "triggers": r"めまい|目が回|ふらつ|ふらふら|くらくら",
        "items": [
            ("回る感じか、ふわふわか、立ちくらみか", r"回|ぐるぐる|ふわふわ|浮|立ちくらみ"),
            ("始まり方(突然か)", r"突然|急に|いつ"),
            ("頭を動かすと起こるか", r"頭を動|寝返り|姿勢|頭の向き|起き上が"),
            ("耳鳴り・聞こえにくさ", r"耳鳴|聞こえ|難聴"),
            ("神経の症状(しびれ・麻痺・ろれつ・物が二重)", r"しびれ|麻痺|ろれつ|呂律|二重|言葉"),
            ("頭痛", r"頭痛|頭が痛"),
            ("気を失ったか", r"気を失|意識|失神"),
        ],
    },
    "咳": {
        "triggers": r"咳|せき",
        "items": [
            ("いつから(何週間か)", r"いつ|何日|週"),
            ("痰・血痰", r"痰"),
            ("発熱", r"熱"),
            ("息苦しさ", r"息"),
            ("ゼーゼー・ヒューヒュー", r"ゼーゼー|ヒューヒュー|喘鳴"),
            ("夜や朝に悪くなるか", r"夜|朝"),
            ("喫煙", r"タバコ|たばこ|喫煙"),
            ("飲んでいる薬", r"薬"),
        ],
    },
}


def checklists(case: dict) -> list:
    """主訴に当てはまる確認リストと、それぞれの項目を聞いたかどうか(キーワードによる目安)。
    返り値: [{"complaint": 名前, "items": [{"label", "asked"}]}]"""
    asked_text = "\n".join(e["question"] + " " + e["answer"] for e in case["exchanges"])
    out = []
    for name, spec in COMPLAINT_CHECKLISTS.items():
        if re.search(spec["triggers"], case["chief_complaint"]):
            items = [{"label": label, "asked": bool(re.search(kw, asked_text))} for label, kw in spec["items"]]
            out.append({"complaint": name, "items": items})
    return out


def unchecked_items(case: dict) -> list:
    return [f"{c['complaint']}: {i['label']}" for c in checklists(case) for i in c["items"] if not i["asked"]]


# 診断の考え方の基本(小さいモデルが、症状を1つずつ別々に説明して、全体をまとめて説明できる疾患を
# 見落とすことがあったため、医師が普段している考え方を明示する)
REASONING_PRINCIPLES = """診断の考え方:
- 症状や所見を1つずつ別々に説明するのではなく、複数の症状・所見をまとめて1つで説明できる疾患を、必ず鑑別に含めて検討してください。
  (例: 発熱と関節痛と皮疹がそろうなら、それぞれ別の病気より、全部を説明できる1つの疾患を先に考える)
- 年齢・性別、症状の始まり方(突然・急性・慢性)、先行する出来事(感染、新しい薬、外傷、旅行など)を手がかりにしてください。
  年齢や性別が分からなければ、早めに確認してください。
- 主訴ごとに、診断を大きく分ける基本の確認を落とさないでください。例えば:
  発疹: 分布、押して色が消えるか(紫斑かどうか)、触れて盛り上がっているか、かゆみ・痛み、粘膜の症状
  胸痛: 突然か、労作との関係、痛みの広がり、呼吸や体位で変わるか、冷汗
  腹痛: 部位と移動、食事・排便との関係、妊娠の可能性、発熱
  頭痛: 突然か(今までで最悪か)、発熱・項部硬直、神経の症状
  発熱: 期間、熱以外の症状のある部位、渡航・接触歴、内服薬
"""


def next_step_prompt(case: dict, n_questions: int = 3) -> str:
    missing = unchecked_items(case)
    unchecked_hint = (
        "- この主訴でまだ確認していない基本項目があります。診断に関係するものは優先して聞いてください: "
        + "、".join(missing)
        if missing
        else ""
    )
    asked = "\n".join(f"- {e['question']}" for e in case["exchanges"] if e["question"] != "(医師のメモ)")
    asked = asked or "(まだありません)"
    return f"""次の患者さんの問診を進めています。これまでに分かったことから、鑑別診断と、次に聞くべき質問を考えてください。

出力は次の形だけにしてください(前置きや説明、見出しは不要です)。
鑑別: 疾患名(高)、疾患名(中)、疾患名(低)
見逃し注意: 可能性は低くても見逃すと危険な疾患を「、」区切りで。なければ「なし」
質問: 患者さんへの質問文｜この質問で見分けたいこと
質問: 患者さんへの質問文｜この質問で見分けたいこと

決まりごと:
- 鑑別は可能性の高い順に3〜5個。(高)(中)(低)で可能性の程度を付けてください。
- 情報が主訴だけでも、その主訴でよくみられる疾患と危険な疾患から鑑別を挙げてください。
  確信が持てないときは(低)を付ければよく、「鑑別を挙げられない」とは書かないでください。
- 質問は必ず1つ以上書いてください。情報が少ないほど、発症時期・経過・部位・随伴症状など基本的な質問が役に立ちます。
- 質問は最大{n_questions}個。診断を最も大きく絞り込めるもの、危険な疾患を除外するものを優先してください。
- 質問は、医師がそのまま患者さんに読み上げられる、短く分かりやすい話し言葉にしてください。1つの質問で聞くことは1つにしてください。
- すでに聞いた質問は繰り返さないでください。
- 身体診察で確かめるべきことがあれば、質問の代わりに「診察: 確かめる所見｜見分けたいこと」と書いてかまいません。
- 書かれていない症状や所見を、あるものとして扱わないでください。
- 疾患名は日本語の正式な病名で書いてください(英語や、英語をカタカナにした表記は使わない)。
{unchecked_hint}
{REASONING_PRINCIPLES}
【答え方の例】(別の患者の例です。形式だけを参考にしてください)
鑑別: 急性胃腸炎(高)、急性虫垂炎(中)、尿路結石(低)
見逃し注意: 異所性妊娠、腸閉塞
質問: 痛みはどこから始まって、今はどこが痛みますか｜虫垂炎の痛みの移動の有無
質問: 最後の生理はいつでしたか｜異所性妊娠の除外
診察: 右下腹部の圧痛と反跳痛｜虫垂炎の可能性

【これまでに分かったこと】
{history_text(case)}

【すでに聞いた質問】
{asked}
"""


def _split_items(text: str) -> list:
    return [x.strip(" *・") for x in re.split(r"[、,，]", text) if x.strip(" *・")]


def parse_next_step(text: str) -> dict:
    """提案の出力を読み取る。返り値: {"ddx": [{"name", "level"}], "must_not_miss": [...], "questions": [{"kind","text","purpose"}]}"""
    ddx, must_not_miss, questions = [], [], []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*・]|\d+[.)．])\s*", "", line.replace("**", "").strip())
        m = re.match(r"(鑑別|見逃し注意|質問|診察)\s*[:：]\s*(.+)", line)
        if not m:
            continue
        label, body = m.group(1), m.group(2).strip()
        if label == "鑑別":
            for item in _split_items(body):
                lv = re.search(r"[（(]\s*(高|中|低)\s*[)）]", item)
                name = re.sub(r"[（(]\s*(高|中|低)\s*[)）]", "", item).strip()
                if name:
                    ddx.append({"name": name, "level": lv.group(1) if lv else ""})
        elif label == "見逃し注意":
            must_not_miss = [x for x in _split_items(body) if x not in ("なし", "特になし")]
        else:
            text_, _, purpose = body.partition("｜") if "｜" in body else body.partition("|")
            if text_.strip():
                questions.append({"kind": label, "text": text_.strip(), "purpose": purpose.strip()})
    return {"ddx": ddx, "must_not_miss": must_not_miss, "questions": questions}


# モデルが質問を出せなかったときに、問診を止めないための定型の質問(OPQRSTなど、どの主訴にも使える基本)
FALLBACK_QUESTIONS = [
    ("いつから、どのように始まりましたか(急に・徐々に)", "発症の様式(急性か慢性か)"),
    ("始まってから、良くなっていますか、悪くなっていますか", "経過"),
    ("どこが、どのように(痛み・違和感など)つらいですか", "部位と性状"),
    ("ほかに一緒に出ている症状はありますか(発熱・吐き気・息切れなど)", "随伴症状"),
    ("持病や、飲んでいる薬、アレルギーはありますか", "既往歴・内服歴・アレルギー"),
]


def retry_prompt(case: dict, n_questions: int, previous: str) -> str:
    return f"""前回の答えから、鑑別と質問を決めた形で読み取れませんでした。
利用者は医師で、これは医師が判断するための参考です。情報が少なくても、考えられる鑑別(確信がなければ(低))と、
絞り込むための質問を挙げてください。次の形だけで答え直してください。

鑑別: 疾患名(高)、疾患名(中)、疾患名(低)
見逃し注意: 疾患名、疾患名(なければ「なし」)
質問: 患者さんへの質問文｜この質問で見分けたいこと
(質問は1〜{n_questions}個)

【前回の答え】
{previous[:1500]}

{next_step_prompt(case, n_questions)}"""


def suggest_next(case: dict, llm, n_questions: int = 3) -> dict:
    """次の質問を提案させる。鑑別か質問が読み取れなければ1回だけ答え直させ、
    それでも質問が無ければ定型の質問を出す(問診が止まらないように)。"""
    raw = llm(next_step_prompt(case, n_questions), SYSTEM)
    result = parse_next_step(raw)
    result["retried"] = False
    if not result["ddx"] or not result["questions"]:
        raw2 = llm(retry_prompt(case, n_questions, raw), SYSTEM)
        result2 = parse_next_step(raw2)
        # 答え直しで読み取れた部分を使い、読み取れなかった部分は1回目のものを残す
        result = {
            "ddx": result2["ddx"] or result["ddx"],
            "must_not_miss": result2["must_not_miss"] or result["must_not_miss"],
            "questions": result2["questions"] or result["questions"],
            "retried": True,
        }
        raw = f"{raw}\n\n--- 答え直し ---\n{raw2}"
    result["fallback"] = False
    if not result["questions"]:
        asked = {e["question"] for e in case["exchanges"]}
        result["questions"] = [
            {"kind": "質問", "text": q, "purpose": p} for q, p in FALLBACK_QUESTIONS if q not in asked
        ][:n_questions]
        result["fallback"] = True
    result["raw"] = raw
    return result


# ---------- まとめ ----------


def summary_prompt(case: dict) -> str:
    return f"""次の問診の記録をもとに、医師向けのまとめを書いてください。

次の4つの見出しを付けて、日本語で書いてください。
## 問診のまとめ
(主訴と現病歴を、発症時期・経過・性状・部位・程度・増悪寛解因子・随伴症状の順に、記録にある範囲で簡潔に。既往歴・内服薬・アレルギーなども記録にあれば書く)
## 鑑別診断
(可能性の高い順に3〜5個。それぞれ「支持する所見」と「合わない所見・まだ確かめていないこと」を添える)
## 見逃してはいけない疾患
(可能性が低くても除外すべき疾患と、除外に必要なこと)
## 次に行う診察・検査
(結果によって診断や方針が変わるものを優先して)

記録に書かれていない症状や所見を作らないでください。分からないことは「未確認」と書いてください。
表やHTMLのタグ(<br>など)は使わず、見出しと箇条書き(「- 」で始まる行)だけで書いてください。
疾患名は日本語の正式な病名で書いてください(英語や、英語をカタカナにした表記は使わない)。

{REASONING_PRINCIPLES}
【問診の記録】
{history_text(case)}
"""


def clean_markdown(text: str) -> str:
    """画面で文字のまま表示されてしまうHTMLのタグを取り除く(指示しても使うことがあるため)。"""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    return re.sub(r"</?(?:b|strong|i|em|p|ul|ol|li|span|div)\b[^>]*>", "", text, flags=re.IGNORECASE)


def summarize(case: dict, llm) -> str:
    return clean_markdown(llm(summary_prompt(case), SYSTEM)).strip()


def case_as_text(case: dict, summary: str = "") -> str:
    """保存用のテキスト(問診の記録とまとめ)。"""
    lines = ["# 問診の記録", history_text(case)]
    flags = red_flags(case)
    if flags:
        lines += ["", "危険な兆候(キーワードで検出): " + "、".join(flags)]
    if summary:
        lines += ["", summary]
    lines += ["", "※AIによる参考情報です。診断は医師が行ってください。"]
    return "\n".join(lines)


# ---------- Ollama ----------


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.split("<think>")[0].strip()


def ollama_llm(model: str, think: str = "low", num_ctx: int = 16384, max_tokens: int = 8192, timeout: int = 300):
    """llm(prompt, system) を返す。データはMacの外に出ない。
    gpt-oss は考える量(low/medium/high)、Qwen3 は true/false を think で指定する。
    gpt-oss は考える分も出力の上限(max_tokens)に含まれるので、考えているうちに上限に達して答えが空になることがある。
    そのときは考える量を low に下げて、1回だけやり直す。"""

    def once(prompt: str, system: str, think_value: str):
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": True,
            "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": max_tokens},
        }
        if think_value:
            payload["think"] = {"true": True, "false": False}.get(think_value, think_value)
        req = urllib.request.Request(
            OLLAMA_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        parts, stats = [], {}
        try:
            # 少しずつ受け取るので、考える時間が長くても途中でタイムアウトしない
            with urllib.request.urlopen(req, timeout=timeout) as res:
                for line in res:
                    if not line.strip():
                        continue
                    msg = json.loads(line.decode("utf-8"))
                    parts.append(msg.get("message", {}).get("content", ""))
                    if msg.get("done"):
                        # 時間の内訳(遅い呼び出しを調べるため)。単位はナノ秒
                        stats = {
                            "load_s": msg.get("load_duration", 0) / 1e9,
                            "prompt_tokens": msg.get("prompt_eval_count", 0),
                            "prompt_s": msg.get("prompt_eval_duration", 0) / 1e9,
                            "output_tokens": msg.get("eval_count", 0),
                            "output_s": msg.get("eval_duration", 0) / 1e9,
                            "done_reason": msg.get("done_reason", ""),
                        }
                        break
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
        return strip_thinking("".join(parts)), stats

    def call(prompt: str, system: str) -> str:
        text, stats = once(prompt, system, think)
        if not text and think in ("medium", "high"):
            # 考えているうちに上限に達して答えが空になった。考える量を減らしてやり直す
            first = stats
            text, stats = once(prompt, system, "low")
            stats = {**stats, "retried_low": True, "first_output_tokens": first.get("output_tokens", 0),
                     "first_output_s": first.get("output_s", 0)}
        call.last_stats = stats
        return text

    call.last_stats = {}
    return call
