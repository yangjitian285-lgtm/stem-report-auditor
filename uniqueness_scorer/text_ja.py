"""日本語テキストの主張分割・骨格抽出・AI-Slop検知・具体性解析モジュール."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# 日本語「量産型AI」特有の定型表現・クリシェ辞書
REPORT_SLOP_PATTERNS: List[Tuple[str, float]] = [
    (r"重要な役割を果た", 1.0),
    (r"不可欠である|不可欠だ|不可欠となって", 1.0),
    (r"と言えるだろう|といえるだろう|と言えるのではないか", 0.9),
    (r"メリットとデメリット|光と影|両面から", 1.1),
    (r"以上のことから|以上のように|まとめると", 0.8),
    (r"多角的な視点|多様な視点|バランスが重要|バランスを取", 1.1),
    (r"近年[、,].*急速に|急速な発展|急速に普及", 0.9),
    (r"まず[、,].*?次に[、,]", 1.0),
    (r"第一に|第二に|第三に", 0.8),
    (r"今後の課題|さらなる発展|期待される|求められる", 0.7),
    (r"大きな可能性を秘め|可能性が広が", 1.0),
    (r"共存していく|共存が求められ", 0.9),
    (r"社会全体に大きな影響|多大な影響を与え", 0.9),
    (r"適切に活用|有効に活用|うまく活用", 0.8),
    (r"一概には言えない|慎重な議論が必要", 0.8),
]

ES_SLOP_PATTERNS: List[Tuple[str, float]] = [
    (r"周囲を巻き込|メンバーを巻き込", 1.2),
    (r"課題に直面|壁にぶつか", 1.0),
    (r"試行錯誤を重ね|粘り強く取り組", 1.0),
    (r"この経験から.*学んだ|この経験を通じて", 0.9),
    (r"貴社に入社した際|御社でも.*活かし|貴社で.*貢献", 0.9),
    (r"傾聴力|主体性|協調性|課題解決力", 0.8),
    (r"円滑なコミュニケーション|密なコミュニケーション", 1.0),
    (r"一丸となって|全員で協力し", 1.0),
    (r"私は.*に力を入れました|私が学生時代に最も力を入れた", 0.9),
    (r"信頼関係を築|win-win|相乗効果", 0.9),
]

# 段落の骨格（Algorithmic Skeleton）を判定する役割タグ
ROLE_PATTERNS: List[Tuple[str, str]] = [
    ("INTRO", r"近年|現在|背景|本レポート|本稿|学生時代|私が.*力を入れた|について論じ|について考察"),
    ("BENEFIT", r"メリット|利点|効果|効率|向上|可能性|強み|成果|改善|プラス"),
    ("RISK", r"デメリット|課題|リスク|懸念|問題点|困難|壁|障壁|マイナス|危機"),
    ("COUNTER", r"しかし|一方で|反論|批判|限界|前提.*疑|通説|逆説|見落とされ|とは限らない"),
    ("EVIDENCE", r"調査|データ|統計|研究|事例|インタビュー|実験|文献|論文|\d+%|\d{4}年"),
    ("ACTION", r"対策|提言|解決策|取り組み|施策|アプローチ|提案|実行|工夫"),
    ("CONCLUSION", r"結論|以上のことから|まとめると|今後|最終的|総じて|活かしたい|貢献したい"),
]

# 反論・批判的思考（クリティカル・シンキング）のマーカー
COUNTERARGUMENT_PATTERNS = [
    r"しかし[、,]",
    r"一方で[、,]",
    r"反面",
    r"とは限らない",
    r"見落とされ",
    r"前提.*崩|前提.*問|前提.*疑",
    r"通説|一般的には.*だが",
    r"反証|例外|限界がある",
    r"トレードオフ|ジレンマ|パラドックス|逆説",
    r"批判的に|再検討",
]

# 一次情報・具体性（数値・年号・固有名詞・引用・実体験）のマーカー
EVIDENCE_PATTERNS = [
    r"\d+(?:\.\d+)?\s*(?:%|％|倍|割|人|社|件|円|万円|億円|時間|日|か月|年)",
    r"(?:19|20)\d{2}年",
    r"「[^」]{2,30}」|『[^』]{2,30}』",
    r"\([A-Za-z][^)]{2,30}\d{4}\)|（[^）]{2,25}\d{4}）",
    r"アンケート|ヒアリング|インタビュー|実測|観察|フィールドワーク|検証|実験",
    r"自身の|私が.*体験|現場で|アルバイト先|ゼミで|研究室で|実際に",
]

# 簡易ストップワード（内容語抽出は漢字・カタカナ・英数字の連続のみを拾うため、
# ここには「漢字/カタカナで書かれるが意味の薄い語」を中心に置く。ひらがな語は抽出されないが互換のため残す）
STOPWORDS = {
    "場合", "以上", "以下", "本稿", "本レポート", "今回", "一方", "他方", "必要", "可能", "重要",
    "様々", "多く", "自分", "我々", "非常", "結果", "全体", "部分", "方法", "状況", "問題",
    "こと", "もの", "ため", "よう", "それ", "これ", "あれ", "ここ", "そこ",
    "いる", "ある", "なる", "する", "できる", "れる", "られる", "ない", "ます",
    "です", "ある", "おり", "また", "および", "さらに", "しかし", "ただし",
    "について", "において", "に対して", "によって", "として", "という", "といった",
    "など", "から", "まで", "より", "ほど", "だけ", "ばかり", "ながら",
}


@dataclass
class StructureAnalysis:
    """文章の構造・AI臭・具体性の解析結果."""

    sentences: List[str]
    paragraphs: List[str]
    skeleton: List[str]
    slop_hits: List[str]
    slop_density: float  # 0.0 (クリーン) 〜 1.0 (AIテンプレ過密)
    burstiness: float  # 文長変動係数（CV）
    specificity_score: float  # 0.0 〜 1.0
    evidence_count: int
    counterarg_count: int


def split_paragraphs(text: str) -> List[str]:
    """空行または改行で段落に分割する."""
    raw = re.split(r"\n\s*\n+", text.strip())
    paragraphs = [p.strip() for p in raw if p.strip()]
    if len(paragraphs) <= 1 and "\n" in text.strip():
        paragraphs = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return paragraphs


def split_sentences(text: str) -> List[str]:
    """日本語の句点・改行で文（主張候補）に分割する."""
    normalized = re.sub(r"\r\n?", "\n", text.strip())
    chunks = re.split(r"(?<=[。！？!?])\s*|\n+", normalized)
    sentences: List[str] = []
    for c in chunks:
        s = c.strip()
        # 見出し記号や箇条書き記号を軽く除去して意味内容を保持
        s_clean = re.sub(r"^[#\-・●■◆\d\.\)\s]+", "", s).strip()
        if len(s_clean) >= 8:
            sentences.append(s_clean)
    return sentences


def extract_claims(text: str, max_claims: int = 24) -> List[str]:
    """レポートから原子的な主張（Claim）単位を抽出する."""
    sentences = split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []
    if len(sentences) <= max_claims:
        return sentences
    # 長文の場合は隣接する短文をまとめて主張チャンク化
    merged: List[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) < 95:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                merged.append(buf)
            buf = s
    if buf:
        merged.append(buf)
    return merged[:max_claims]


def extract_content_tokens(text: str) -> List[str]:
    """外部形態素解析器なしで日本語の内容語（漢字熟語・カタカナ語・英単語・数値）を抽出."""
    # 漢字熟語、カタカナ語、英単語・数値を抽出（ひらがなは機能語が多いため対象外）
    tokens = re.findall(
        r"[一-龥々]{2,}|[ァ-ヶー]{2,}|[A-Za-z][A-Za-z0-9\-_]{1,}|\d+(?:\.\d+)?%?",
        text,
    )
    # 1文字漢字でも重要概念になりうるため2-gramも補助併用（embedder側）
    filtered = [t.lower() for t in tokens if t.lower() not in STOPWORDS]
    return filtered


def char_ngrams(text: str, n_range: Tuple[int, int] = (2, 3)) -> List[str]:
    """記号・空白を除いた文字n-gramを抽出."""
    cleaned = re.sub(r"[\s、。,.!！?？「」『』（）()\[\]【】・\-_:/\n\r]", "", text.lower())
    grams: List[str] = []
    for n in range(n_range[0], n_range[1] + 1):
        if len(cleaned) >= n:
            for i in range(len(cleaned) - n + 1):
                grams.append(cleaned[i : i + n])
    return grams


def tag_paragraph_role(paragraph: str, idx: int, total: int) -> str:
    """段落の論理的役割（骨格タグ）を推定する."""
    scores: Dict[str, int] = {}
    for role, pattern in ROLE_PATTERNS:
        hits = len(re.findall(pattern, paragraph))
        if hits > 0:
            scores[role] = hits
    if not scores:
        if idx == 0:
            return "INTRO"
        if idx == total - 1:
            return "CONCLUSION"
        return "BODY"
    # 位置による微調整
    if idx == 0 and "INTRO" in scores:
        scores["INTRO"] += 1
    if idx == total - 1 and "CONCLUSION" in scores:
        scores["CONCLUSION"] += 1
    return max(scores.items(), key=lambda kv: kv[1])[0]


def normalized_levenshtein(seq_a: Sequence[str], seq_b: Sequence[str]) -> float:
    """2つの骨格タグ列間の正規化編集距離 (0.0=完全一致 〜 1.0=完全に異なる)."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m] / max(n, m)


def analyze_structure(text: str, mode: str = "report") -> StructureAnalysis:
    """文章の構造・AI臭・バースト性・具体性を総合解析する."""
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)
    if len(paragraphs) <= 1 and len(sentences) >= 4:
        # 改行がない長文は4分割して疑似段落構造を抽出
        step = max(1, math.ceil(len(sentences) / 4))
        paragraphs = [
            "".join(sentences[i : i + step]) for i in range(0, len(sentences), step)
        ]

    skeleton = [
        tag_paragraph_role(p, i, len(paragraphs)) for i, p in enumerate(paragraphs)
    ]

    # AI-Slop表現の検出
    patterns = list(REPORT_SLOP_PATTERNS)
    if mode == "es":
        patterns.extend(ES_SLOP_PATTERNS)
    else:
        # レポートモードでもES系の一部は軽く参照
        patterns.extend([(p, w * 0.4) for p, w in ES_SLOP_PATTERNS[:4]])

    slop_hits: List[str] = []
    slop_weight_sum = 0.0
    for pat, weight in patterns:
        matches = re.findall(pat, text)
        if matches:
            slop_hits.append(matches[0] if isinstance(matches[0], str) else str(matches[0]))
            slop_weight_sum += weight * min(len(matches), 3)

    # 文数で正規化し、0〜1へ飽和変換
    denom = max(len(sentences), 4)
    slop_rate = slop_weight_sum / denom
    slop_density = 1.0 - math.exp(-1.4 * slop_rate)

    # 文長のバースト性（変動係数 CV）
    lengths = [len(s) for s in sentences] if sentences else [len(text)]
    mean_len = sum(lengths) / max(len(lengths), 1)
    if len(lengths) >= 2 and mean_len > 0:
        var_len = sum((x - mean_len) ** 2 for x in lengths) / len(lengths)
        burstiness = math.sqrt(var_len) / mean_len
    else:
        burstiness = 0.0

    # 具体性・一次情報・反証マーカー
    evidence_count = sum(len(re.findall(p, text)) for p in EVIDENCE_PATTERNS)
    counterarg_count = sum(len(re.findall(p, text)) for p in COUNTERARGUMENT_PATTERNS)

    # マーカーの「個数」ではなく「マーカーを含む文の割合」で評価する。
    # 旧方式は1文に「しかし・年号・「」・調査」を詰め込むだけで満点近くになった（スコアハック）。
    # さらに反証マーカーは、マーカー語を除いた内容語が3語以上ある文だけを数える（空疎な「しかし、」を除外）。
    n_sent = max(len(sentences), 1)
    ev_sents = sum(1 for s_ in sentences if any(re.search(p, s_) for p in EVIDENCE_PATTERNS))
    ca_sents = 0
    for s_ in sentences:
        if any(re.search(p, s_) for p in COUNTERARGUMENT_PATTERNS):
            stripped = re.sub("|".join(COUNTERARGUMENT_PATTERNS), "", s_)
            if len(set(extract_content_tokens(stripped))) >= 3:
                ca_sents += 1
    length_conf = min(1.0, n_sent / 6.0)  # 文数が少ない短文は割合が振れやすいので信頼度で割り引く
    ev_norm = min(1.0, (ev_sents / n_sent) / 0.6) * (0.6 + 0.4 * length_conf)
    ca_norm = min(1.0, (ca_sents / n_sent) / 0.5) * (0.6 + 0.4 * length_conf)
    # 語彙のユニーク比（内容語ベース）
    content_tokens = extract_content_tokens(text)
    ttr = len(set(content_tokens)) / max(len(content_tokens), 1)
    ttr_norm = min(1.0, max(0.0, (ttr - 0.35) / 0.45))

    specificity_score = 0.50 * ev_norm + 0.30 * ca_norm + 0.20 * ttr_norm

    return StructureAnalysis(
        sentences=sentences,
        paragraphs=paragraphs,
        skeleton=skeleton,
        slop_hits=list(dict.fromkeys(slop_hits)),
        slop_density=float(min(1.0, max(0.0, slop_density))),
        burstiness=float(burstiness),
        specificity_score=float(min(1.0, max(0.0, specificity_score))),
        evidence_count=evidence_count,
        counterarg_count=counterarg_count,
    )


# ---------------------------------------------------------------------------
# 文章品質（独自性とは別軸）：論理のつながり・要検証の記述
# ---------------------------------------------------------------------------
_OPENING_CONNECTIVES = re.compile(
    r"^(?:したがって|従って|つまり|そのため|このため|さらに|また[、,]|一方で?|他方|このように|その結果|そこで|以上|だからこそ|結論として|とはいえ|ただし)"
)


def _cohesion_tokens(sentence: str) -> set:
    toks = set(extract_content_tokens(sentence))
    # 漢字・カタカナの2文字連続も補助的に使う（形態素解析なしでの語の重なり）
    core = re.sub(r"[^一-龥々ァ-ヶーA-Za-z0-9]", "", sentence)
    toks.update(core[i : i + 2] for i in range(len(core) - 1))
    return toks


def coherence_analysis(sentences: Sequence[str]) -> Dict[str, float]:
    """隣接文どうしの語の重なり（結束性）と、冒頭の接続詞の不自然さから論理のつながりを推定する.

    - adjacency_ratio: 隣接文の重なり ÷ 離れた文どうしの重なり。筋の通った文章は1より大きく、
      文をランダムに並べ替えると1付近に下がる。
    - opening_connective: 1文目が「したがって」「つまり」等で始まっていれば1（並べ替えや切り貼りの兆候）。
    - coherence_score: 0〜1（独自性スコアには含めず、品質指標として別表示）。
    """
    n = len(sentences)
    if n < 3:
        return {"coherence_score": 0.5, "adjacency_ratio": 1.0, "opening_connective": 0.0}
    sets = [_cohesion_tokens(s) for s in sentences]

    def jac(a: set, b: set) -> float:
        return len(a & b) / max(len(a | b), 1)

    adj = [jac(sets[i], sets[i + 1]) for i in range(n - 1)]
    far = [jac(sets[i], sets[j]) for i in range(n) for j in range(i + 2, n)]
    adj_mean = sum(adj) / len(adj)
    far_mean = sum(far) / len(far) if far else adj_mean
    ratio = adj_mean / max(far_mean, 1e-3)
    opening = 1.0 if _OPENING_CONNECTIVES.match(sentences[0].strip()) else 0.0
    # 接続詞で始まる文の直前が無関係な文になっている割合（局所的な飛躍）
    jumps = 0
    conn_sents = 0
    for i in range(1, n):
        if _OPENING_CONNECTIVES.match(sentences[i].strip()):
            conn_sents += 1
            if adj[i - 1] < far_mean * 0.8:
                jumps += 1
    jump_rate = jumps / conn_sents if conn_sents else 0.0
    score = 0.55 + 0.30 * math.tanh(ratio - 1.0) - 0.30 * opening - 0.20 * jump_rate
    return {
        "coherence_score": float(min(1.0, max(0.0, score))),
        "adjacency_ratio": float(round(ratio, 4)),
        "opening_connective": opening,
        "connective_jump_rate": float(round(jump_rate, 4)),
    }


_FIRST_PERSON_RESEARCH = re.compile(
    r"(?:私|筆者|自分)(?:が|は|自身).{0,40}?(?:ヒアリング|アンケート|調査|インタビュー|測定|実測|分析|集計)"
    r"|(?:ヒアリング|アンケート|インタビュー|聞き取り)(?:した|を行った|を実施)"
)
_STAT_CLAIM = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|割|倍|万人|億円|万円)")
_CITATION = re.compile(r"[（(][^）)]{0,30}(?:19|20)\d{2}[^）)]{0,10}[）)]|\[\d+\]|出典|による(?:と|調査)|白書|統計[（(]|報告書")


def verification_needed(sentences: Sequence[str]) -> List[Dict[str, str]]:
    """本人が裏付けを示すべき記述（一次調査の主張、出典のない統計値）を列挙する.

    ツールは内容の真偽を判定できないため、捏造された数値でも具体性として加点されてしまう。
    そこで加点とは別に「提出前に根拠を確認すべき文」を明示する。
    """
    out: List[Dict[str, str]] = []
    for s_ in sentences:
        if _FIRST_PERSON_RESEARCH.search(s_):
            out.append({"text": s_, "reason": "一次調査の主張：調査記録（日時・対象・方法・人数）を手元に残し、必要なら付録に示してください"})
        elif _STAT_CLAIM.search(s_) and not _CITATION.search(s_):
            out.append({"text": s_, "reason": "出典のない統計値：出典（著者・年・資料名）を明記してください"})
    return out
