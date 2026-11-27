"""理系レポート 1ファイル丸投げ決定論的・数理監査エンジン (STEM Report Auditor).

Word(.docx) / PDF / テキストから自動抽出した本文＋構造化テーブルに対し、
LLMのハルシネーションが一切入らない「決定論的な数理・物理・整合監査」と
「脱・量産型AI考察スコア」を統合実行する。

監査項目：
1. 表内計算の自動検算（sin²θ vs 2θ、a = d√(h²+k²+l²)、D = Mw/Mn、平均値列）
2. 表間・表と本文の矛盾検知（同一試料の数値食い違い、ミラー指数(100) vs (110)、平均値検算）
3. 有効数字のばらつき・理系誤記（解説→回折、NMr→NMR、PH→pH等）・未完/テンプレ残骸検知
4. 実験データの系統バイアス・単調トレンド・線形外挿（cos²θ→0 等）のゼロ設定自動発見
5. 理系考察の6軸脱・量産型AIスコア（StemUniquenessScorer統合）
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .doc_parser import ParsedDocument, ParsedTable, answer_paragraphs, parse_report_file, parse_sci_number
from .domain_rules import (
    DomainProfile,
    detect_profiles,
    is_stem_document,
    load_user_rules,
    merged_terms,
    sample_patterns,
    stem_signal_score,
)
from .embeddings import EmbeddingBackend, LocalHybridEmbedder
from .table_checker import check_table
from .stem_engine import (
    DataSeriesInput,
    NumericAuditResult,
    StemScoringResult,
    StemUniquenessScorer,
    audit_numeric_data,
)


# 理系レポートで頻出する表記ミス・変換誤記辞書（分野共通のものだけ）
# ※ \b は日本語の文字に隣接すると機能しないため、英字の前後否定先読みで単語境界を表す
SCIENTIFIC_TYPO_RULES: List[Tuple[str, str, str]] = [
    (r"解説が起こる|X線解説|ブラッグ解説|解説線|解説角|解説図形|解説ピーク", "「回折」の変換ミス（解説→回折）", "ERROR"),
    (r"(?<![A-Za-z])(?:NMr|nmr|Nmr)(?![A-Za-z])", "「NMR」の大文字表記ゆれ", "WARN"),
    (r"(?<![A-Za-z])PH(?![A-Za-z])", "「pH」の表記ミス（pは小文字、Hは大文字）", "WARN"),
    (r"(?<![A-Za-z])hlk(?![A-Za-z])", "ミラー指数「hkl」の誤記（hlk→hkl）", "WARN"),
    (r"参考文(?!献)", "「参考文献」の脱字", "WARN"),
    (r"原子間距離\s*[)）]?\s*$|d\s*[=＝]\s*原子間距離", "Braggの d は「面間隔」（原子間距離ではない）", "WARN"),
]

# 削除し忘れやすい大学レポートテンプレート指示文パターン
TEMPLATE_RESIDUE_PATTERNS: List[str] = [
    r"赤字の部分はすべて削除|赤字で示しています",
    r"青字の注意書きは削除",
    r"実験の目的を簡潔に記述する",
    r"必要に応じて図表を適切に利用する",
    r"客観的な実験事実のみを記し",
    r"実験テキストの内容をもとに",
    r"必要に応じて、実験についての感想等を記載してください",
    r"^\([12]\)\s*[・\.…]{3,}",
]

# 理系レポートの典型的な量産AI考察ベースライン（母集団未指定時の汎用比較用）
DEFAULT_GENERIC_STEM_COHORT: List[str] = [
    (
        "本実験により得られた測定値は理論値および文献値と概ね一致しており、実験方法の妥当性が確認された。"
        "わずかな誤差が生じた原因として、測定時の読み取り誤差やヒューマンエラー、室温などの実験環境の影響、装置の較正不足が考えられる。"
        "今後は測定回数を増やし、より高精度な機器を用いて慎重に操作することで誤差を低減できると考えられる。"
    ),
    (
        "実験結果は概ね良好であり、理論通りの傾向を示すことが確認できた。"
        "理論値との間に若干の差が見られた要因としては、手技による操作ミス、測定器の精度限界、温度変化などの環境要因が挙げられる。"
        "今後の課題として、試行回数を増やして平均値を取るとともに、より丁寧な操作を心がけることが求められる。"
    ),
    (
        "データ解析の結果、算出された値は文献値と誤差の範囲内でほぼ一致し、一定の成果が得られた。"
        "多少の誤差が生じた背景には、目視による読み取りのばらつきや装置の精度、実験室環境の変化による影響が推測される。"
        "さらなる検討が必要であり、次回はより厳密な条件管理のもとで詳細な解析を行いたい。"
    ),
    (
        "測定結果と理論値を比較したところ、大きな乖離はなく目的を達成できたと言える。"
        "微小な差の原因には、機器の校正誤差、試料調製時の秤量ミス、温度や湿度といった外部環境の影響が考えられる。"
        "今後はより精度の高い装置を使用し、測定回数を増やすことで結果の信頼性を高めることが望ましい。"
    ),
]


@dataclass
class AuditFinding:
    """決定論的監査で検出された1件の指摘事項."""

    severity: str  # "CRITICAL", "WARNING", "INFO", "PASS"
    category: str  # "CALC_CHECK", "CROSS_CONSISTENCY", "SIG_FIG", "TYPO", "TEMPLATE", "NUMERIC_DISCOVERY"
    location: str
    message: str
    suggestion: str = ""


@dataclass
class FullStemAuditReport:
    """1ファイル丸投げ理系レポート監査の統合出力."""

    source_path: str
    file_type: str
    deterministic_score: Optional[float]  # 0〜100。実行できた検査が0件なら None（N/A）
    discussion_score: Optional[float]  # 0〜100。非理系・未回答なら None
    overall_score: Optional[float]  # 統合スコア（算出対象が無ければ None）
    overall_grade: str
    extracted_tables_count: int
    findings: List[AuditFinding]
    auto_discovered_audits: List[NumericAuditResult]
    auto_discovered_series: List[Dict[str, Any]]
    stem_discussion_result: Optional[StemScoringResult]
    priority_checklist: List[str]
    is_stem_report: bool = True
    stem_signal: float = 0.0
    detected_profiles: List[str] = field(default_factory=list)
    checks_run: int = 0
    unanswered: bool = False
    answer_chars: int = 0
    discussion_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _half_ulp(raw: str) -> float:
    """記載値の最終桁の半分（丸め幅）。例: '0.2019' -> 0.00005, '3.73*10^4' -> 0.005e4."""
    s = raw.strip().replace(",", "")
    m = re.fullmatch(r"([+-]?\d+(?:\.(\d+))?)\s*(?:[\*×xX]\s*10\^?\s*\(?([+-]?\d+)\)?|[eE]([+-]?\d+))?", s)
    if not m:
        return 0.0
    dec = len(m.group(2)) if m.group(2) else 0
    exp = int(m.group(3) or m.group(4) or 0)
    return 0.5 * (10.0 ** (-dec)) * (10.0 ** exp)


def _sig_digits(raw: str) -> int:
    """小数表記の有効数字の桁数（先頭の0は数えない）."""
    s = raw.strip().lstrip("+-").replace(",", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?", s):
        return 0
    digits = s.replace(".", "").lstrip("0")
    return len(digits)


def _count_decimal_places(raw: str) -> Optional[int]:
    s = raw.strip()
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
        return None
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def _audit_intra_table_arithmetic(tables: Sequence[ParsedTable]) -> List[AuditFinding]:
    """Pass 1: 表内の数式・計算・平均値の自動検算."""
    findings: List[AuditFinding] = []
    for tbl in tables:
        h_low = [h.lower().replace(" ", "") for h in tbl.headers]

        # 1) XRD: 2θ と sin^2θ の検算
        idx_2t = next((i for i, h in enumerate(h_low) if "2θ" in h), None)
        idx_s2 = next(
            (i for i, h in enumerate(h_low) if ("sin^2" in h or "sin²" in h) and "/" not in h and "*" not in h),
            None,
        )
        if idx_2t is not None and idx_s2 is not None:
            checked = 0
            bad = 0
            for r_i, row in enumerate(tbl.rows):
                t2 = parse_sci_number(row[idx_2t]) if idx_2t < len(row) else None
                s2 = parse_sci_number(row[idx_s2]) if idx_s2 < len(row) else None
                if t2 is not None and s2 is not None:
                    checked += 1
                    expected = math.sin(math.radians(t2 / 2.0)) ** 2
                    # 許容誤差＝記載桁の丸め幅＋2θの丸めの伝播（有効数字から決める）
                    tol = _half_ulp(row[idx_s2]) + abs(0.5 * math.sin(math.radians(t2)) * math.pi / 180.0) * _half_ulp(row[idx_2t])
                    tol = max(tol * 1.05, 1e-6)
                    if abs(expected - s2) > tol:
                        bad += 1
                        findings.append(
                            AuditFinding(
                                severity="CRITICAL",
                                category="CALC_CHECK",
                                location=f"{tbl.caption} 行{r_i+1}",
                                message=f"2θ={t2}° に対する sin²θ の計算不一致（記載={s2} vs 正解={expected:.4f}）",
                                suggestion="角度の度/ラジアン変換または2θ→θの半分化を再確認してください。",
                            )
                        )
            if checked >= 2 and bad == 0:
                findings.append(
                    AuditFinding(
                        severity="PASS",
                        category="CALC_CHECK",
                        location=tbl.caption,
                        message=f"2θ → sin²θ の三角関数検算：全{checked}行が記載桁の丸め幅の範囲内で一致",
                    )
                )

        # 2) XRD: a = d * sqrt(h²+k²+l²) の検算
        sig_excess_rows: List[int] = []
        idx_hkl2 = next((i for i, h in enumerate(h_low) if "h²" in h or "h^2" in h), None)
        idx_d = next((i for i, h in enumerate(h_low) if re.fullmatch(r"d(?:\([^)]*\))?", h)), None)
        idx_a = next((i for i, h in enumerate(h_low) if re.fullmatch(r"a(?:\([^)]*\))?", h)), None)
        if idx_hkl2 is not None and idx_d is not None and idx_a is not None:
            checked = 0
            bad = 0
            for r_i, row in enumerate(tbl.rows):
                n2 = parse_sci_number(row[idx_hkl2]) if idx_hkl2 < len(row) else None
                dv = parse_sci_number(row[idx_d]) if idx_d < len(row) else None
                av = parse_sci_number(row[idx_a]) if idx_a < len(row) else None
                if n2 and dv and av:
                    checked += 1
                    expected_a = dv * math.sqrt(n2)
                    tol_a = (math.sqrt(n2) * _half_ulp(row[idx_d]) + _half_ulp(row[idx_a])) * 1.05
                    if _count_decimal_places(row[idx_a]) is not None and _count_decimal_places(row[idx_d]) is not None:
                        if _sig_digits(row[idx_a]) > _sig_digits(row[idx_d]):
                            sig_excess_rows.append(r_i + 1)
                    if abs(expected_a - av) > tol_a:
                        bad += 1
                        findings.append(
                            AuditFinding(
                                severity="WARNING",
                                category="CALC_CHECK",
                                location=f"{tbl.caption} 行{r_i+1}",
                                message=f"a = d√(h²+k²+l²) の検算差（記載 a={av}, d√N={expected_a:.5f}）",
                                suggestion="中間丸め（dの桁落ち）による誤差か計算ミスかを確認してください。",
                            )
                        )
            if checked >= 2 and bad == 0:
                findings.append(
                    AuditFinding(
                        severity="PASS",
                        category="CALC_CHECK",
                        location=tbl.caption,
                        message=f"格子定数 a = d√(h²+k²+l²) の検算：全{checked}行がdの丸め幅の範囲内で一致",
                    )
                )
            if sig_excess_rows:
                findings.append(
                    AuditFinding(
                        severity="WARNING",
                        category="SIG_FIG",
                        location=f"{tbl.caption} 行{', '.join(map(str, sig_excess_rows))}",
                        message="格子定数 a の有効数字が、元にした面間隔 d の有効数字より多く記載されています。",
                        suggestion="a は d の有効桁数に合わせて丸めるか、d をより多い桁で示してください（文献値との差の議論がその桁に依存していないか確認）。",
                    )
                )

        # 3) GPC: D = Mw / Mn の検算
        idx_mn = next((i for i, h in enumerate(h_low) if h.startswith("mn")), None)
        idx_mw = next((i for i, h in enumerate(h_low) if h.startswith("mw")), None)
        idx_disp = next((i for i, h in enumerate(h_low) if "mw/mn" in h or h.startswith("d(")), None)
        if idx_mn is not None and idx_mw is not None and idx_disp is not None:
            for r_i, row in enumerate(tbl.rows):
                mn = parse_sci_number(row[idx_mn]) if idx_mn < len(row) else None
                mw = parse_sci_number(row[idx_mw]) if idx_mw < len(row) else None
                dv = parse_sci_number(row[idx_disp]) if idx_disp < len(row) else None
                if mn and mw and dv:
                    exp_d = mw / mn
                    tol_d = _half_ulp(row[idx_disp]) + exp_d * (
                        _half_ulp(row[idx_mw]) / max(abs(mw), 1e-12) + _half_ulp(row[idx_mn]) / max(abs(mn), 1e-12)
                    )
                    if abs(exp_d - dv) > tol_d * 1.05 + 1e-9:
                        findings.append(
                            AuditFinding(
                                severity="CRITICAL",
                                category="CALC_CHECK",
                                location=f"{tbl.caption} 行{r_i+1}",
                                message=f"分散度 D=Mw/Mn の計算不一致（記載={dv}, Mw/Mn={exp_d:.2f}）",
                                suggestion="表中の Mn, Mw, D の対応を再確認してください。",
                            )
                        )

        # 4) 平均値列の検算（1回目, 2回目, 3回目 -> 平均値）
        trial_cols = [i for i, h in enumerate(h_low) if re.search(r"\d回目", h)]
        avg_col = next((i for i, h in enumerate(h_low) if "平均" in h), None)
        if len(trial_cols) >= 2 and avg_col is not None:
            for r_i, row in enumerate(tbl.rows):
                vals = [parse_sci_number(row[c]) for c in trial_cols if c < len(row)]
                vals_ok = [v for v in vals if v is not None]
                avg_v = parse_sci_number(row[avg_col]) if avg_col < len(row) else None
                if len(vals_ok) >= 2 and avg_v is not None:
                    exp_avg = sum(vals_ok) / len(vals_ok)
                    tol_avg = _half_ulp(row[avg_col]) + max(_half_ulp(row[c]) for c in trial_cols if c < len(row))
                    if abs(exp_avg - avg_v) > tol_avg * 1.05 + 1e-9:
                        findings.append(
                            AuditFinding(
                                severity="WARNING",
                                category="CALC_CHECK",
                                location=f"{tbl.caption} 行{r_i+1}",
                                message=f"平均値の検算差（記載={avg_v}, 実計算={exp_avg:.2f}）",
                            )
                        )

        # 5) 表内のミラー指数 (hkl) と h²+k²+l² の一致
        idx_hkl = next((i for i, h in enumerate(h_low) if "(hkl)" in h or h == "hkl"), None)
        if idx_hkl is not None and idx_hkl2 is not None:
            for r_i, row in enumerate(tbl.rows):
                m_hkl = re.search(r"\((\d)(\d)(\d)\)", row[idx_hkl]) if idx_hkl < len(row) else None
                n2 = parse_sci_number(row[idx_hkl2]) if idx_hkl2 < len(row) else None
                if m_hkl and n2 is not None:
                    calc_n2 = int(m_hkl.group(1)) ** 2 + int(m_hkl.group(2)) ** 2 + int(m_hkl.group(3)) ** 2
                    if abs(calc_n2 - n2) > 0.1:
                        findings.append(
                            AuditFinding(
                                severity="CRITICAL",
                                category="CALC_CHECK",
                                location=f"{tbl.caption} 行{r_i+1}",
                                message=f"ミラー指数 {m_hkl.group(0)} の h²+k²+l²={calc_n2} が記載値 {n2} と不一致",
                            )
                        )

    return findings


def _nearest_hkl(hkl: str, candidates: set) -> str:
    def n2(s: str) -> int:
        return sum(int(c) ** 2 for c in s[1:4])

    target = n2(hkl)
    return min(sorted(candidates), key=lambda c: (abs(n2(c) - target), c))


def _audit_cross_consistency(doc: ParsedDocument, profiles: Sequence[DomainProfile] = ()) -> List[AuditFinding]:
    """Pass 2: 表間および表と本文の数値・指数・サンプル名矛盾を検知."""
    findings: List[AuditFinding] = []

    # 1) ミラー指数 (hkl) の本文 vs 表の整合性（例: 本文に(100)とあるが表は(110)）
    table_hkls: set[str] = set()
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row:
                for m in re.findall(r"\(\d{3}\)", cell):
                    table_hkls.add(m)

    if table_hkls:
        body_hkls = set(re.findall(r"\(\d{3}\)", doc.body_text))
        extra_in_body = body_hkls - table_hkls
        for hkl in sorted(extra_in_body):
            # BCC記述があり、h+k+l が奇数なら消滅則違反も併記
            digits = [int(c) for c in hkl[1:4]]
            bcc_violation = ("BCC" in doc.body_text) and (sum(digits) % 2 == 1)
            extra_note = "（しかもBCCの消滅則 h+k+l=偶数 に違反）" if bcc_violation else ""
            findings.append(
                AuditFinding(
                    severity="CRITICAL",
                    category="CROSS_CONSISTENCY",
                    location="本文 vs 結晶構造表",
                    message=f"本文にミラー指数 {hkl} が記載されていますが、表には {sorted(table_hkls)} しか存在しません{extra_note}。",
                    suggestion=f"本文の {hkl} を表と一致する指数（例：h²+k²+l² が最も近い {_nearest_hkl(hkl, table_hkls)}）へ修正してください。",
                )
            )

    # 2) 同一試料名×同一指標（Mn, Mw, D など）の表間・本文間の数値食い違い検知
    sample_metrics: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
    sample_pat = re.compile("|".join(f"(?:{p})" for p in sample_patterns(profiles)))

    for tbl in doc.tables:
        # 縦型2列テーブル（例: 表8: Mn | 3.73*10^4, Mw | 6.08*10^4）で直前文脈に試料名がある場合
        ctx_sample = sample_pat.search(tbl.context_before)
        if len(tbl.headers) == 2 and ctx_sample:
            sname = re.sub(r"\s+", "", ctx_sample.group(0)).upper().replace("：", ":")
            for r in [tbl.headers] + tbl.rows:
                if len(r) >= 2:
                    mkey = r[0].strip().split("(")[0].upper()
                    val = parse_sci_number(r[1])
                    if mkey in ("MN", "MW", "D", "MW/MN") and val is not None:
                        sample_metrics.setdefault((sname, mkey), []).append((tbl.caption, val))
        else:
            # 横型テーブル（行に試料名、列に Mn, Mw, D）
            h_keys = [h.strip().split("(")[0].upper() for h in tbl.headers]
            for row in tbl.rows:
                row_s = sample_pat.search(" ".join(row))
                if not row_s:
                    continue
                sname = re.sub(r"\s+", "", row_s.group(0)).upper().replace("：", ":")
                for c_i, hk in enumerate(h_keys):
                    if hk in ("MN", "MW", "D", "MW/MN") and c_i < len(row):
                        val = parse_sci_number(row[c_i])
                        if val is not None:
                            sample_metrics.setdefault((sname, hk), []).append((tbl.caption, val))

    # 表間比較で10%以上の乖離があればCRITICALとして報告
    for (sname, mkey), entries in sample_metrics.items():
        if len(entries) >= 2:
            vals = [v for _, v in entries]
            vmin, vmax = min(vals), max(vals)
            if vmax > 0 and (vmax - vmin) / vmax > 0.10:
                detail = " vs ".join(f"{loc}={v:.4g}" for loc, v in entries)
                findings.append(
                    AuditFinding(
                        severity="CRITICAL",
                        category="CROSS_CONSISTENCY",
                        location=f"試料 {sname} / 指標 {mkey}",
                        message=f"同一試料・同一指標の数値が表の間で大きく矛盾しています（{detail}、差 {(vmax-vmin)/vmax*100:.1f}%）。",
                        suggestion="解析区間や単位・試料ラベルの取り違えを確認し、レポート全体で数値を統一してください。",
                    )
                )

    # 3) 同一レポート内で「MAA40:BMA60」と「MAA60:BMA40」など反転した試料名表記の注意喚起
    all_samples = sorted(
        set(re.sub(r"\s+", "", m).upper().replace("：", ":") for m in sample_pat.findall(doc.full_text))
    )
    for s in all_samples:
        m_parts = re.fullmatch(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", s)
        if m_parts and m_parts.group(2) != m_parts.group(4):
            swapped = f"{m_parts.group(1)}{m_parts.group(4)}:{m_parts.group(3)}{m_parts.group(2)}"
            if swapped in all_samples and s < swapped:
                findings.append(
                    AuditFinding(
                        severity="WARNING",
                        category="CROSS_CONSISTENCY",
                        location="レポート全体（試料表記）",
                        message=f"組成比が反転した試料表記（{s} と {swapped}）が同一レポート内に混在しています。",
                        suggestion="別試料なのか記載ミスなのかを本文・表タイトルで明確に確認してください。",
                    )
                )

    # 4) 本文中の「平均値は X」「文献値 Y より約 Z% 小さい/大きい」の自動検算
    for tbl in doc.tables:
        for c_i, hdr in enumerate(tbl.headers):
            nums = [v for v in tbl.col_floats(c_i) if v is not None]
            if len(nums) >= 3 and re.match(r"a(?:\(|$)", hdr.lower().replace(" ", "")):
                calc_mean = sum(nums) / len(nums)
                m_avg = re.search(r"格子定数\s*a?\s*の平均値は\s*([0-9]+\.[0-9]+)", doc.body_text)
                if m_avg:
                    rep_mean = float(m_avg.group(1))
                    if abs(rep_mean - calc_mean) / max(abs(calc_mean), 1e-9) < 0.002:
                        findings.append(
                            AuditFinding(
                                severity="PASS",
                                category="CROSS_CONSISTENCY",
                                location=f"本文 vs {tbl.caption}",
                                message=f"本文の格子定数平均値 ({rep_mean}) と表の平均値 ({calc_mean:.5f}) が正確に一致しています。",
                            )
                        )
                    else:
                        findings.append(
                            AuditFinding(
                                severity="CRITICAL",
                                category="CROSS_CONSISTENCY",
                                location=f"本文 vs {tbl.caption}",
                                message=f"本文の平均値 ({rep_mean}) と表の平均値 ({calc_mean:.5f}) が不一致です。",
                            )
                        )
    return findings


def _audit_sigfigs_typos_and_templates(doc: ParsedDocument, user_rules: Optional[Dict[str, List]] = None) -> List[AuditFinding]:
    """Pass 3: 有効数字・理系誤記・テンプレ残骸・書きかけ文の自動検知."""
    findings: List[AuditFinding] = []

    # 1) 表の同一列における小数桁（有効数字）のばらつき検知
    for tbl in doc.tables:
        for c_i, hdr in enumerate(tbl.headers):
            if any(k in hdr for k in ["順番", "No", "h²", "h^2", "倍率"]):
                continue
            raw_vals = tbl.col_values(c_i)
            decs = [_count_decimal_places(v) for v in raw_vals]
            decs_valid = [d for d in decs if d is not None]
            if len(decs_valid) >= 3 and (max(decs_valid) - min(decs_valid)) >= 2:
                findings.append(
                    AuditFinding(
                        severity="WARNING",
                        category="SIG_FIG",
                        location=f"{tbl.caption} 列「{hdr}」",
                        message=f"同一列内で小数点以下の桁数が不統一です（記載例: {', '.join(raw_vals[:4])}）。",
                        suggestion="測定器の精度・有効数字に合わせて列内の小数桁（例：1 → 1.000）を統一してください。",
                    )
                )

    # 2) 理系レポート特有の誤記・表記ゆれ
    typo_rules = list(SCIENTIFIC_TYPO_RULES) + [
        (re.escape(t[0]), t[1], t[2]) for t in (user_rules or {}).get("typos", [])
    ]
    for pat, msg, sev in typo_rules:
        m = re.search(pat, doc.full_text)
        if m:
            findings.append(
                AuditFinding(
                    severity="CRITICAL" if sev == "ERROR" else ("WARNING" if sev == "WARN" else "INFO"),
                    category="TYPO",
                    location=f"語句「{m.group(0)}」",
                    message=msg,
                    suggestion="提出前に該当語句を正しい学術表記へ修正してください。",
                )
            )

    # 3) 赤字・青字のテンプレート注意書きの消し忘れ検知
    if doc.colored_template_lines:
        sample = doc.colored_template_lines[0][:45]
        findings.append(
            AuditFinding(
                severity="WARNING",
                category="TEMPLATE",
                location="Word書式（色付き文字）",
                message=f"赤字/青字のテンプレート指示文と思われる箇所が {len(doc.colored_template_lines)} 件残存しています（例：「{sample}…」）。",
                suggestion="提出規定に従い、赤字・青字の案内文を削除してから提出してください。",
            )
        )
    else:
        for pat in list(TEMPLATE_RESIDUE_PATTERNS) + [re.escape(r) for r in (user_rules or {}).get("template_residue", [])]:
            m = re.search(pat, doc.full_text, re.MULTILINE)
            if m:
                findings.append(
                    AuditFinding(
                        severity="WARNING",
                        category="TEMPLATE",
                        location="本文テンプレート残存",
                        message=f"レポート様式の指示文が未削除のまま残っています（「{m.group(0)}」）。",
                        suggestion="様式テンプレートの案内文・プレースホルダを削除してください。",
                    )
                )
                break

    # 4) 設問番号の重複（例：2問目も「（１）」になっている）
    numbered: Dict[str, int] = {}
    for p in doc.paragraphs:
        m_num = re.match(r"^\s*[（(]\s*([0-9０-９]+)\s*[)）]", p)
        if m_num:
            key = m_num.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
            numbered[key] = numbered.get(key, 0) + 1
    for key, cnt in numbered.items():
        if cnt >= 2:
            findings.append(
                AuditFinding(
                    severity="WARNING",
                    category="NUMBERING",
                    location=f"設問番号（{key}）",
                    message=f"設問番号「（{key}）」が {cnt} 回使われています。番号の付け間違いの可能性があります。",
                    suggestion="2問目以降の番号を（２）（３）…に修正してください。",
                )
            )

    # 5) 未記入の番号・書きかけの考察文の検知
    for idx, p in enumerate(doc.paragraphs):
        if re.fullmatch(r"[1-9１-９]+[\.．。]", p.strip()):
            findings.append(
                AuditFinding(
                    severity="CRITICAL",
                    category="TEMPLATE",
                    location=f"段落#{idx+1}（「{p.strip()}」）",
                    message=f"番号「{p.strip()}」の後に内容が未記入の空項目があります。",
                    suggestion="未回答の設問がないか確認し、本文を記入するか不要な番号を削除してください。",
                )
            )
        if re.search(r"以下の点で.*限界がある。?$|以下の理由が挙げられる。?$", p.strip()):
            next_p = doc.paragraphs[idx + 1].strip() if idx + 1 < len(doc.paragraphs) else ""
            if not next_p or re.match(r"^(?:課題|参考文|\d+[\.．])", next_p):
                findings.append(
                    AuditFinding(
                        severity="CRITICAL",
                        category="TEMPLATE",
                        location=f"段落#{idx+1}",
                        message=f"「{p.strip()[-28:]}」と予告した直後に内容が続かず、考察が途中で切れています。",
                        suggestion="予告した具体的な限界点・理由を箇条書きまたは段落で補完してください。",
                    )
                )

    return findings


def _auto_discover_data_series(doc: ParsedDocument, profiles: Sequence[DomainProfile] = ()) -> List[DataSeriesInput]:
    """Pass 4: 抽出テーブルと本文中の文献値から自動で DataSeriesInput を構築する."""
    discovered: List[DataSeriesInput] = []
    prof_keys = {p.key for p in profiles}
    prof_by_key = {p.key: p for p in profiles}

    # 1) 格子定数 a テーブルの自動検出（2θ があれば cos²θ を独立変数として自動生成）
    m_lit_a = re.search(r"文献値\s*([0-9]+\.[0-9]+)\s*(nm|Å|A)?", doc.full_text)
    lit_a = float(m_lit_a.group(1)) if m_lit_a else None
    lit_unit = (m_lit_a.group(2) or "nm") if m_lit_a else "nm"

    for tbl in doc.tables:
        h_low = [h.lower().replace(" ", "") for h in tbl.headers]
        idx_a = next((i for i, h in enumerate(h_low) if re.fullmatch(r"a(?:\([^)]*\))?", h)), None)
        idx_2t = next((i for i, h in enumerate(h_low) if "2θ" in h), None)
        if idx_a is not None:
            a_vals = [v for v in tbl.col_floats(idx_a) if v is not None]
            if len(a_vals) >= 3 and lit_a is not None:
                indep: Optional[List[float]] = None
                indep_name = ""
                if idx_2t is not None:
                    t2_vals = [v for v in tbl.col_floats(idx_2t) if v is not None]
                    if len(t2_vals) == len(a_vals):
                        indep = [round(math.cos(math.radians(t / 2.0)) ** 2, 4) for t in t2_vals]
                        indep_name = "cos²θ"
                assumptions = list(prof_by_key["xrd_lattice"].default_assumptions) if "xrd_lattice" in prof_keys else []
                discovered.append(
                    DataSeriesInput(
                        name=f"格子定数 a（{tbl.caption}）",
                        observed=a_vals,
                        reference=lit_a,
                        unit=lit_unit,
                        independent_var=indep,
                        independent_name=indep_name,
                        assumptions=assumptions,
                    )
                )

    # 2) 本文中の「%未反応... = XX.X%」の自動抽出（基準値0・非負量なので符号の偏りは判定しない）
    unreacted = [
        float(m)
        for m in re.findall(r"%未反応[A-Za-z]+\s*=\s*[^=\n]*?=\s*([0-9]+(?:\.[0-9]+)?)\s*%", doc.full_text)
    ]
    if len(unreacted) >= 2:
        assumptions = (
            list(prof_by_key["radical_copolymer"].default_assumptions) if "radical_copolymer" in prof_keys else []
        )
        discovered.append(
            DataSeriesInput(
                name="未反応モノマー率（本文算出値）",
                observed=unreacted,
                reference=0.0,
                unit="%",
                assumptions=assumptions,
            )
        )

    # 3) 汎用フォールバック：本文に「文献値/理論値」があり、同オーダーの数値列が表にあれば自動紐付け
    if not discovered:
        m_ref = re.search(r"(?:文献値|理論値)[^\d]{0,12}([0-9]+(?:\.[0-9]+)?)", doc.full_text)
        if m_ref:
            ref_val = float(m_ref.group(1))
            for tbl in doc.tables:
                for c_i, hdr in enumerate(tbl.headers):
                    nums = [v for v in tbl.col_floats(c_i) if v is not None]
                    if len(nums) >= 3 and ref_val > 0:
                        mean_v = sum(nums) / len(nums)
                        if 0.5 * ref_val <= mean_v <= 1.5 * ref_val:
                            discovered.append(
                                DataSeriesInput(
                                    name=f"{hdr}（{tbl.caption}）",
                                    observed=nums,
                                    reference=ref_val,
                                )
                            )
                            break

    return discovered


# 決定論的チェックとして「実行できた」ことを示すカテゴリ
_CHECK_CATEGORIES = {"CALC_CHECK", "CROSS_CONSISTENCY", "SIG_FIG", "NUMBERING", "GENERIC_CALC"}


def _audit_generic_tables(tables: Sequence[ParsedTable], existing: Sequence[AuditFinding]) -> List[AuditFinding]:
    """分野を問わない汎用検算（列間の計算関係を推定し、外れたセルを指摘）."""
    taken = {f.location for f in existing if f.category == "CALC_CHECK" and f.severity in ("CRITICAL", "WARNING")}
    out: List[AuditFinding] = []
    for tbl in tables:
        if len(tbl.rows) < 3:
            continue
        res = check_table(tbl.headers, tbl.rows, tbl.caption)
        for tf in res.findings:
            loc = tbl.caption + (f" 行{tf.row}" if tf.row else "") + (f"（{tf.column}）" if tf.column and not tf.row else "")
            if tf.row and f"{tbl.caption} 行{tf.row}" in taken:
                continue  # 分野別の専用検算ですでに指摘済み
            out.append(AuditFinding(
                severity=tf.severity,
                category="GENERIC_CALC",
                location=loc,
                message=tf.message,
                suggestion=tf.suggestion,
            ))
    return out
UNANSWERED_MIN_CHARS = 150


def _deterministic_score(findings: Sequence[AuditFinding], n_checks: int) -> Optional[float]:
    """減点を逓減させた決定論的品質点。検査が1件も実行できなければ None（N/A）.

    旧方式（CRITICAL -15 / WARNING -6 の単純減点）は数件で0点に張り付き改善が見えなかったため、
    100 × exp(-(0.18×CRITICAL + 0.07×WARNING)) とする。
    """
    if n_checks == 0:
        return None
    n_crit = sum(1 for f in findings if f.severity == "CRITICAL")
    n_warn = sum(1 for f in findings if f.severity == "WARNING" and f.category != "NUMERIC_DISCOVERY")
    return float(100.0 * math.exp(-(0.18 * n_crit + 0.07 * n_warn)))


def audit_stem_document(
    file_path: str,
    experiment_prompt: Optional[str] = None,
    reference_cohort: Optional[Sequence[str]] = None,
    backend: Optional[EmbeddingBackend] = None,
    user_rules_path: Optional[str] = None,
) -> FullStemAuditReport:
    """Word/PDF/テキストファイルを1パスで完全監査するメインエントリポイント."""
    doc = parse_report_file(file_path)
    user_rules = load_user_rules(user_rules_path)
    profiles = detect_profiles(doc.full_text)
    signal = stem_signal_score(doc.body_text, len(doc.tables))
    is_stem = is_stem_document(doc.body_text, len(doc.tables))

    findings: List[AuditFinding] = []

    # 0) 未回答（設問だけ・白紙）の検知
    answers = answer_paragraphs(doc.paragraphs)
    answer_chars = sum(len(a) for a in answers)
    unanswered = answer_chars < UNANSWERED_MIN_CHARS
    if unanswered:
        findings.append(
            AuditFinding(
                severity="CRITICAL",
                category="UNANSWERED",
                location="レポート全体",
                message=f"設問・指示文を除いた回答本文が {answer_chars} 字しかありません。未回答（設問のみ）の可能性があります。",
                suggestion="各設問への回答を記入してから再監査してください。",
            )
        )

    findings.extend(_audit_intra_table_arithmetic(doc.tables))
    findings.extend(_audit_generic_tables(doc.tables, findings))
    findings.extend(_audit_cross_consistency(doc, profiles))
    findings.extend(_audit_sigfigs_typos_and_templates(doc, user_rules))

    # 実験データの自動発見＆系統バイアス・外挿解析
    series_list = _auto_discover_data_series(doc, profiles)
    auto_audits: List[NumericAuditResult] = []
    for ds in series_list:
        na = audit_numeric_data(ds)
        auto_audits.append(na)
        if na.has_strong_systematic_bias:
            findings.append(
                AuditFinding(
                    severity="WARNING",
                    category="NUMERIC_DISCOVERY",
                    location=ds.name,
                    message=na.findings[1] if len(na.findings) > 1 else na.findings[0],
                    suggestion="系統バイアスを生む物理・装置メカニズムを考察に明記してください。",
                )
            )
        if na.trend_r2 is not None and na.trend_r2 >= 0.80:
            sugg = "単純平均で終わらせず、外挿値とその不確かさ（信頼区間）を考察へ加えると加点対象になります。"
            if na.reference_within_ci95:
                sugg += "外挿後の残差は誤差範囲内なので、残差から新しい機構を断定しないでください。"
            findings.append(
                AuditFinding(
                    severity="INFO",
                    category="NUMERIC_DISCOVERY",
                    location=ds.name,
                    message=na.findings[-1],
                    suggestion=sugg,
                )
            )

    # 理系考察スコア：理系実験レポートで、かつ回答がある場合のみ
    disc_res: Optional[StemScoringResult] = None
    discussion_note = ""
    if unanswered:
        discussion_note = "未回答のため考察スコアは算出していません。"
    elif not is_stem:
        discussion_note = (
            f"理系実験レポートではないと判定（理系シグナル {signal:.2f}）したため、理系考察スコアは対象外です。"
            "自由記述・文系レポートの独自性は `python cli.py score` で採点してください。"
        )
    elif not doc.discussion_text.strip():
        discussion_note = "考察セクションを抽出できなかったため、考察スコアは算出していません。"
    else:
        prompt_eff = experiment_prompt or (
            doc.paragraphs[0] + " " + doc.paragraphs[1] if len(doc.paragraphs) >= 2 else Path(file_path).stem
        )
        cohort_eff = (
            list(reference_cohort) if reference_cohort and len(reference_cohort) >= 3 else DEFAULT_GENERIC_STEM_COHORT
        )
        primary_ds = series_list[0] if series_list else None
        stem_scorer = StemUniquenessScorer(backend=backend or LocalHybridEmbedder())
        disc_res = stem_scorer.score(
            experiment_prompt=prompt_eff,
            discussion_text=doc.discussion_text,
            reference_cohort=cohort_eff,
            data_series=primary_ds,
            extra_mechanism_terms=merged_terms(profiles, "mechanism_terms"),
            extra_quant_terms=merged_terms(profiles, "quant_terms"),
        )

    n_numbered = len(re.findall(r"(?m)^\s*[（(]\s*[0-9０-９]+\s*[)）]", "\n".join(doc.paragraphs)))
    n_checks = (
        sum(1 for f in findings if f.category in _CHECK_CATEGORIES)
        + len(series_list)
        + len(doc.tables)
        + (1 if n_numbered >= 2 else 0)
    )
    det_score = _deterministic_score(findings, n_checks)
    n_crit = sum(1 for f in findings if f.severity == "CRITICAL")

    parts = []
    if det_score is not None:
        parts.append((0.45, det_score))
    if disc_res is not None:
        parts.append((0.55, disc_res.final_score))
    overall: Optional[float]
    if parts:
        w = sum(p[0] for p in parts)
        overall = round(sum(p[0] * p[1] for p in parts) / w, 1)
    else:
        overall = None

    n_warn_all = sum(1 for f in findings if f.severity == "WARNING")
    if not is_stem and not unanswered:
        # 理系考察の対象外：総合点は出さず、形式チェックの結果だけを示す（少数の検査で「S」と誤解させない）
        overall = None
    if unanswered:
        grade = "未回答 (設問のみ・要記入)"
    elif not is_stem:
        grade = f"形式チェックのみ (CRITICAL {n_crit}件 / WARNING {n_warn_all}件・理系考察は対象外)"
    elif overall is None:
        grade = "N/A (自動検査の対象となる表・数値・考察がありません)"
    elif overall >= 80 and n_crit == 0:
        grade = "S (提出準備万端・加点レベル)"
    elif overall >= 65 and n_crit == 0:
        grade = "A (高品質・微調整でS狙い)"
    elif overall >= 50:
        grade = "B (要修正：矛盾または考察深掘り不足)"
    elif overall >= 35:
        grade = "C (提出前修正必須：重大指摘あり)"
    else:
        grade = "D (再構築推奨)"

    # 優先修正チェックリスト（重大度順）
    checklist: List[str] = []
    for sev in ("CRITICAL", "WARNING"):
        for f in findings:
            if f.severity == sev:
                checklist.append(f"[{f.severity} / {f.location}] {f.message}" + (f" → {f.suggestion}" if f.suggestion else ""))
    if disc_res is not None:
        for s in disc_res.scientific_suggestions:
            checklist.append(f"[考察改善] {s}")
    elif discussion_note:
        checklist.append(f"[INFO] {discussion_note}")

    return FullStemAuditReport(
        source_path=doc.source_path,
        file_type=doc.file_type,
        deterministic_score=round(det_score, 1) if det_score is not None else None,
        discussion_score=disc_res.final_score if disc_res is not None else None,
        overall_score=overall,
        overall_grade=grade,
        extracted_tables_count=len(doc.tables),
        findings=findings,
        auto_discovered_audits=auto_audits,
        auto_discovered_series=[asdict(ds) for ds in series_list],
        stem_discussion_result=disc_res,
        priority_checklist=checklist[:10],
        is_stem_report=is_stem,
        stem_signal=round(signal, 3),
        detected_profiles=[p.label for p in profiles],
        checks_run=n_checks,
        unanswered=unanswered,
        answer_chars=answer_chars,
        discussion_note=discussion_note,
    )
