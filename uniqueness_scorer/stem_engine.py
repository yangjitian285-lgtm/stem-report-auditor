"""理系計算・実験レポート特化「脱・量産型考察」数理スコアリングエンジン.

文系・感想レポートと異なり、理系レポートでは「自分の体験談」を足して差別化できない。
代わりに、教員・TAがS評価と量産型C評価を分ける以下の6つの客観的・数理的軸で診断する：
1. 脱・量産型AI考察（Claim単位被り率 ＋ 限界Vendi増分 ΔVS）
2. 脱・理系逃げフレーズ（読み取り誤差・環境要因・概ね一致・測定回数増加の定型文検知）
3. 誤差方向（バイアス符号）＆系統トレンド整合（実測データの統計解析と文章の物理整合チェック）
4. オーダー評価・誤差伝播（数値・%寄与・標準偏差・感度・外挿による定量的裏付け）
5. 物理メカニズム・理論近似の破綻特定（数式前提の崩壊箇所・派生仮説・文献比較）
6. 結果再述からの脱却度（単なる「〜が得られた」の再述 vs 因果・機構・仮説の説明比率）
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embeddings import EmbeddingBackend, LocalHybridEmbedder
from .math_engine import ClaimDiagnostic, marginal_vendi_gain
from .text_ja import extract_claims, extract_content_tokens, split_sentences


# 理系実験・計算レポートの「量産型AI 逃げフレーズ」辞書
STEM_SLOP_PATTERNS: List[Tuple[str, float, str]] = [
    (r"読み取り誤差|目視.*誤差|ヒューマンエラー|人的ミス|操作ミス|手技.*誤差", 1.3, "偶然/人的誤差への安易な帰着"),
    (r"実験環境.*影響|環境要因|室温.*変化|温度変化.*影響|湿度.*影響", 1.1, "未定量の環境要因逃げ"),
    (r"装置の?較正|校正不足|機器の?精度|装置の?限界|測定器.*誤差", 1.0, "未定量の装置誤差逃げ"),
    (r"概ね一致|ほぼ一致|よく一致|妥当性が確認|良好な結果|理論通り", 1.2, "無検証の『概ね一致』着地"),
    (r"測定回数を増や|試行回数を増や|より高精度な機器|慎重に操作|丁寧に行う", 1.3, "量産型の今後の課題テンプレ"),
    (r"若干の差|わずかな誤差|多少の誤差|誤差の範囲内", 1.0, "定量的根拠のない誤差矮小化"),
    (r"さらなる検討が必要|今後の課題である|より詳細な解析が求められ", 0.9, "抽象的な結び"),
]

# 定量的オーダー評価・誤差伝播・統計処理のマーカー
QUANT_ERROR_PATTERNS = [
    # ±付きの不確かさ表記、または符号付き百分率（「2-3」のような範囲・ハイフンは数えない）
    r"±\s*\d+(?:\.\d+)?|(?<![\w.])[+\-−]\d+(?:\.\d+)?\s*(?:%|％)",
    r"\d+(?:\.\d+)?\s*(?:%|％|nm|mm|μm|pm|mN/m|℃|kV|mA|g/mol|mol|ppm|倍|桁)|\d+(?:\.\d+)?\s*K(?![A-Za-z])",
    r"誤差伝播|偏微分|感度|寄与率|オーダー|標準偏差|不確かさ|信頼区間|相関係数|決定係数|R\^?2|R²|r\s*=",
    r"外挿|回帰|線形フィット|最小二乗|切片|傾き",
    r"文献値|理論値.*比|相対誤差|系統誤差|偶然誤差|バイアス",
]

# 物理メカニズム・理論前提・近似破綻のマーカー
MECHANISM_PATTERNS = [
    r"近似|仮定|理想|無視|成り立たない|破綻|逸脱|前提",
    r"系統誤差|偶然誤差|系統的|単調|比例|依存|相関|シフト|偏り",
    r"メカニズム|機構|起因|由来|引き起こ|抑制|促進|競合|平衡|速度論|熱力学",
    r"仮説|検証.*実験|再実験|比較.*条件|対照実験|文献.*比較|\[\d+\]",
]

# 説明的・因果的表現（結果再述ではない考察文の判定）
EXPLANATORY_PATTERNS = [
    r"なぜなら|ためである|に起因|によると考え|が原因|によって生じ|から説明できる",
    r"仮説|示唆され|推定される|解釈できる|矛盾|整合|裏付け",
    r"もし|仮に|〜であれば|検証するためには|比較すると|対照的に",
    r"支配的|寄与|主因|副次的|無視できない|トレードオフ|限界がある",
]

# 単なる結果再述の表現
RESTATEMENT_PATTERNS = [
    r"が得られた|を確認した|が確認できた|と観測された|を示した|となった|であった$",
    r"表\s*\d+|図\s*\d+.*示す|平均値は.*となり",
]

# 誤差の方向（正・負）およびランダム vs 系統の言及パターン
NEG_BIAS_WORDS = r"小さ[いく]|低[いく]|下回|負の|減少|縮小|マイナス|下振れ|短[いく]"
POS_BIAS_WORDS = r"大き[いく]|高[いく]|上回|正の|増加|膨張|プラス|上振れ|長[いく]"
SYSTEMATIC_WORDS = r"系統|単調|直線|外挿|比例|シフト|バイアス|一定方向|傾き|増加するにつれ|大きくなるにつれ|に依存して|とともに"
# 誤差の向きを述べた文と判定するための「基準値」語（方向語単独ではほぼ全文書に出るため）
REFERENCE_WORDS = r"文献値|理論値|基準値|真値|期待値|計算値|標準値|参照値|仕込み"
RANDOM_ONLY_WORDS = r"読み取り誤差|ヒューマンエラー|人的ミス|偶然誤差|ランダム誤差|測定.*ばらつき"


@dataclass
class DataSeriesInput:
    """実測値・理論値/文献値・独立変数の数値データ入力."""

    name: str
    observed: List[float]
    reference: Any  # float (定数文献値) または List[float] (各点理論値)
    unit: str = ""
    independent_var: Optional[List[float]] = None
    independent_name: str = ""
    assumptions: List[str] = field(default_factory=list)


@dataclass
class NumericAuditResult:
    """数値データから自動算出した統計・バイアス・トレンド解析結果."""

    has_data: bool = False
    mean_observed: float = 0.0
    mean_reference: float = 0.0
    mean_rel_error_pct: float = 0.0
    bias_direction: str = "NONE"  # "NEGATIVE", "POSITIVE", "MIXED"
    one_sided_ratio: float = 0.0  # 全データ中、同符号の割合
    has_strong_systematic_bias: bool = False
    trend_r: Optional[float] = None
    trend_r2: Optional[float] = None
    extrapolated_at_zero: Optional[float] = None
    extrapolated_rel_error_pct: Optional[float] = None
    error_reduction_pct: Optional[float] = None
    # 外挿切片の不確かさ（最小二乗の標準誤差と95%信頼区間）
    n_points: int = 0
    intercept_se: Optional[float] = None
    intercept_ci95: Optional[List[float]] = None
    reference_within_ci95: Optional[bool] = None
    bias_check_skipped_reason: str = ""
    findings: List[str] = field(default_factory=list)


@dataclass
class StemScoringResult:
    """理系計算・実験レポート特化スコアリングの完全出力."""

    final_score: float
    grade: str
    raw_depth_originality: float
    relevance_gate: float
    metrics: Dict[str, float]
    raw_stats: Dict[str, Any]
    numeric_audit: NumericAuditResult
    claims: List[ClaimDiagnostic]
    slop_hits: List[Dict[str, str]]
    scientific_suggestions: List[str]
    pca_projection: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 両側95%のt分布臨界値（自由度1〜30）。31以上は正規近似1.96
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t95(dof: int) -> float:
    if dof <= 0:
        return float("inf")
    return _T95[dof - 1] if dof <= len(_T95) else 1.96


def linear_fit_with_uncertainty(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """最小二乗直線 y = a + b x と切片・傾きの標準誤差を返す."""
    n = len(x)
    X = np.vstack([np.ones(n), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    s2 = float(resid @ resid) / dof if dof > 0 else float("nan")
    cov = s2 * np.linalg.inv(X.T @ X) if dof > 0 else np.full((2, 2), np.nan)
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "intercept_se": float(np.sqrt(cov[0, 0])) if dof > 0 else float("nan"),
        "slope_se": float(np.sqrt(cov[1, 1])) if dof > 0 else float("nan"),
        "dof": dof,
    }


def audit_numeric_data(data: Optional[DataSeriesInput]) -> NumericAuditResult:
    """実測値と理論/文献値からバイアス方向・系統トレンド・ゼロ外挿を自動解析する."""
    if data is None or not data.observed:
        return NumericAuditResult(has_data=False)

    obs = np.asarray(data.observed, dtype=np.float64)
    if isinstance(data.reference, (list, tuple, np.ndarray)):
        ref = np.asarray(data.reference, dtype=np.float64)
    else:
        ref = np.full_like(obs, float(data.reference))

    residuals = obs - ref
    zero_ref = bool(np.all(np.abs(ref) < 1e-12))
    if zero_ref:
        rel_errs = residuals  # 基準値0のときは絶対偏差をそのまま保持
    else:
        rel_errs = residuals / np.where(np.abs(ref) > 1e-12, np.abs(ref), 1.0) * 100.0
    mean_obs = float(np.mean(obs))
    mean_ref = float(np.mean(ref))
    mean_rel = float(np.mean(rel_errs))

    n_neg = int(np.sum(residuals < -1e-12))
    n_pos = int(np.sum(residuals > 1e-12))
    n_tot = len(obs)
    one_sided = max(n_neg, n_pos) / max(n_tot, 1)

    if n_neg == n_tot and n_tot >= 2:
        bias_dir = "NEGATIVE"
    elif n_pos == n_tot and n_tot >= 2:
        bias_dir = "POSITIVE"
    elif n_neg > n_pos and one_sided >= 0.75:
        bias_dir = "NEGATIVE"
    elif n_pos > n_neg and one_sided >= 0.75:
        bias_dir = "POSITIVE"
    else:
        bias_dir = "MIXED"

    strong_bias = (one_sided >= 0.80 and n_tot >= 3) or (
        one_sided == 1.0 and n_tot >= 2 and abs(mean_rel) >= 0.1
    )
    skip_reason = ""
    # 基準値0で全点が非負（収率・未反応率・濃度など定義上0以上の量）なら、
    # 「全点が正側」は物理的に自明で系統誤差の証拠にならない
    if zero_ref and np.all(obs >= 0):
        strong_bias = False
        skip_reason = "基準値0の非負量のため、符号の偏りは系統誤差の判定に使えません"

    findings: List[str] = []
    unit_s = f" {data.unit}" if data.unit else ""
    if zero_ref:
        findings.append(
            f"{data.name}: 実測平均 {mean_obs:.5g}{unit_s} vs 理想基準 {mean_ref:.5g}{unit_s} "
            f"（平均残差 {mean_obs - mean_ref:+.3g}{unit_s}、負側 {n_neg}/{n_tot}点・正側 {n_pos}/{n_tot}点）"
        )
    else:
        findings.append(
            f"{data.name}: 実測平均 {mean_obs:.5g}{unit_s} vs 基準値 {mean_ref:.5g}{unit_s} "
            f"（平均相対誤差 {mean_rel:+.3f}%、負側 {n_neg}/{n_tot}点・正側 {n_pos}/{n_tot}点）"
        )
    if skip_reason:
        findings.append(f"（{skip_reason}）")
    if strong_bias:
        dir_jp = "負側（系統的に小さい）" if bias_dir == "NEGATIVE" else "正側（系統的に大きい）"
        findings.append(
            f"【系統バイアス検出】全{n_tot}点中{max(n_neg, n_pos)}点が{dir_jp}に偏っています。"
            "ランダム誤差（読み取り誤差等）のみでは説明できず、物理的・装置的な系統誤差要因の特定が不可欠です。"
        )

    trend_r: Optional[float] = None
    trend_r2: Optional[float] = None
    extrap_zero: Optional[float] = None
    extrap_rel: Optional[float] = None
    err_red: Optional[float] = None
    icpt_se: Optional[float] = None
    icpt_ci: Optional[List[float]] = None
    ref_in_ci: Optional[bool] = None

    if data.independent_var is not None and len(data.independent_var) == n_tot and n_tot >= 3:
        x = np.asarray(data.independent_var, dtype=np.float64)
        if np.std(x) > 1e-12 and np.std(obs) > 1e-12:
            r = float(np.corrcoef(x, obs)[0, 1])
            r2 = r * r
            fit = linear_fit_with_uncertainty(x, obs)
            intercept = fit["intercept"]
            trend_r = round(r, 4)
            trend_r2 = round(r2, 4)
            extrap_zero = float(intercept)
            if fit["dof"] > 0 and math.isfinite(fit["intercept_se"]):
                icpt_se = fit["intercept_se"]
                half = t95(fit["dof"]) * icpt_se
                icpt_ci = [intercept - half, intercept + half]
            # 基準値がスカラー（一定文献値）の場合、x→0外挿による誤差低減率を評価
            if not isinstance(data.reference, (list, tuple, np.ndarray)):
                extrap_rel = float((intercept - float(data.reference)) / abs(float(data.reference)) * 100.0)
                if abs(mean_rel) > 1e-9:
                    err_red = float((1.0 - abs(extrap_rel) / abs(mean_rel)) * 100.0)
                ref_f = float(data.reference)
                if icpt_ci is not None:
                    ref_in_ci = bool(icpt_ci[0] <= ref_f <= icpt_ci[1])
                if r2 >= 0.80:
                    msg = (
                        f"【強い変数依存トレンド検出】{data.independent_name or '独立変数'} に対して線形相関 "
                        f"r={r:+.3f} (R²={r2:.3f}, n={n_tot}) が存在します。x→0 へ線形外挿すると "
                        f"{intercept:.5g}{unit_s}（基準値比 {extrap_rel:+.3f}%）"
                    )
                    if icpt_ci is not None:
                        msg += (
                            f"、切片の標準誤差 {icpt_se:.2g}{unit_s}、95%信頼区間 "
                            f"[{icpt_ci[0]:.5g}, {icpt_ci[1]:.5g}]{unit_s}"
                        )
                    msg += f" となり、単純平均の誤差を約{err_red:.1f}%削減できます。"
                    if ref_in_ci:
                        msg += (
                            "ただし基準値は信頼区間内にあるため、外挿後に残る差は統計的に有意ではありません"
                            "（残差から新たな機構を主張するのは過剰解釈です）。"
                        )
                    elif ref_in_ci is False:
                        msg += "基準値は信頼区間の外にあり、外挿後も有意な差が残っています。"
                    if n_tot <= 4:
                        msg += f"点数が{n_tot}点と少ないため、外挿値の信頼性には限界があります。"
                    findings.append(msg)

    return NumericAuditResult(
        has_data=True,
        mean_observed=round(mean_obs, 6),
        mean_reference=round(mean_ref, 6),
        mean_rel_error_pct=round(mean_rel, 4),
        bias_direction=bias_dir,
        one_sided_ratio=round(one_sided, 4),
        has_strong_systematic_bias=strong_bias,
        trend_r=trend_r,
        trend_r2=trend_r2,
        extrapolated_at_zero=round(extrap_zero, 6) if extrap_zero is not None else None,
        extrapolated_rel_error_pct=round(extrap_rel, 4) if extrap_rel is not None else None,
        error_reduction_pct=round(err_red, 1) if err_red is not None else None,
        n_points=n_tot,
        intercept_se=round(icpt_se, 8) if icpt_se is not None else None,
        intercept_ci95=[round(v, 8) for v in icpt_ci] if icpt_ci is not None else None,
        reference_within_ci95=ref_in_ci,
        bias_check_skipped_reason=skip_reason,
        findings=findings,
    )


class StemUniquenessScorer:
    """理系計算・実験レポートの考察に特化した多角的数理スコアラー."""

    def __init__(self, backend: Optional[EmbeddingBackend] = None) -> None:
        self.backend = backend or LocalHybridEmbedder()

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _chunk_embed_doc(self, text: str, claim_vecs: np.ndarray) -> np.ndarray:
        """主張ベクトルの平均による長さ不変な文書埋め込み."""
        if claim_vecs.shape[0] == 0:
            return self.backend.embed([text])[0]
        mean_v = np.mean(claim_vecs, axis=0)
        norm = float(np.linalg.norm(mean_v))
        return mean_v / norm if norm > 1e-12 else mean_v

    def score(
        self,
        experiment_prompt: str,
        discussion_text: str,
        reference_cohort: Sequence[str],
        data_series: Optional[DataSeriesInput] = None,
        extra_mechanism_terms: Optional[Sequence[str]] = None,
        extra_quant_terms: Optional[Sequence[str]] = None,
    ) -> StemScoringResult:
        """理系考察テキストと任意の実測データを統合採点する.

        extra_mechanism_terms / extra_quant_terms には、分野別プロファイル（domain_rules）の
        専門語を渡す。本体は分野に依存しない汎用パターンだけで採点する。
        """
        mech_patterns = list(MECHANISM_PATTERNS)
        if extra_mechanism_terms:
            mech_patterns.append("|".join(re.escape(t) for t in extra_mechanism_terms))
        quant_patterns = list(QUANT_ERROR_PATTERNS)
        if extra_quant_terms:
            quant_patterns.append("|".join(re.escape(t) for t in extra_quant_terms))
        cohort = list(reference_cohort)
        draft_claims = extract_claims(discussion_text, max_claims=30)
        cohort_claims_per_doc = [extract_claims(doc, max_claims=25) for doc in cohort]

        # 全テキストを一括埋め込み（TF-IDF整合＋API効率）
        all_texts: List[str] = [experiment_prompt, discussion_text] + cohort
        doc_end = len(all_texts)
        d_start = doc_end
        all_texts.extend(draft_claims)
        d_end = len(all_texts)
        c_spans: List[Tuple[int, int]] = []
        for clist in cohort_claims_per_doc:
            st = len(all_texts)
            all_texts.extend(clist)
            c_spans.append((st, len(all_texts)))

        all_vecs = self.backend.embed(all_texts)
        prompt_vec = all_vecs[0]
        draft_claim_vecs = all_vecs[d_start:d_end]
        cohort_claim_vecs = [all_vecs[st:ed] for st, ed in c_spans]

        # 長さバイアスを抑えるチャンク平均＋文書全体埋め込みのハイブリッド
        draft_vec_raw = all_vecs[1]
        draft_vec_chunk = self._chunk_embed_doc(discussion_text, draft_claim_vecs)
        draft_vec = 0.5 * draft_vec_raw + 0.5 * draft_vec_chunk
        draft_vec = draft_vec / max(float(np.linalg.norm(draft_vec)), 1e-12)

        cohort_vecs_list: List[np.ndarray] = []
        for idx, doc in enumerate(cohort):
            cv_raw = all_vecs[2 + idx]
            cv_chk = self._chunk_embed_doc(doc, cohort_claim_vecs[idx])
            cv = 0.5 * cv_raw + 0.5 * cv_chk
            cohort_vecs_list.append(cv / max(float(np.linalg.norm(cv)), 1e-12))
        cohort_vecs = np.vstack(cohort_vecs_list)

        # ----------------------------------------------------------------------
        # 1) 脱・量産型AI考察（Claim単位被り率 ＋ 限界Vendi増分 ΔVS）
        # ----------------------------------------------------------------------
        delta_vs, vs_before, _ = marginal_vendi_gain(cohort_vecs, draft_vec)
        loo_gains = [
            marginal_vendi_gain(np.delete(cohort_vecs, i, axis=0), cohort_vecs[i])[0]
            for i in range(cohort_vecs.shape[0])
        ]
        base_g_mean = float(np.mean(loo_gains))
        base_g_std = max(float(np.std(loo_gains)), 0.03)
        vendi_z = (delta_vs - base_g_mean) / base_g_std
        # 絶対ΔVSと相対zの併用（未知ドメインでの疎ベクトル分散に頑健化）
        vendi_rel = self._sigmoid(1.0 * (vendi_z - 0.4))
        vendi_abs = min(1.0, delta_vs / 0.45)
        vendi_score_norm = 0.55 * vendi_rel + 0.45 * vendi_abs

        claim_diags: List[ClaimDiagnostic] = []
        overlaps: List[float] = []
        for idx, (c_txt, c_vec) in enumerate(zip(draft_claims, draft_claim_vecs)):
            probs: List[float] = []
            best_s = -1.0
            best_ex = ""
            for doc_cl, doc_vs in zip(cohort_claims_per_doc, cohort_claim_vecs):
                if doc_vs.shape[0] == 0:
                    probs.append(0.0)
                    continue
                sims = np.clip(doc_vs @ c_vec, -1.0, 1.0)
                mi = int(np.argmax(sims))
                ms = float(sims[mi])
                if ms > best_s:
                    best_s = ms
                    best_ex = doc_cl[mi]
                probs.append(self._sigmoid(13.0 * (ms - 0.36)))
            ov = float(np.mean(probs))
            overlaps.append(ov)
            status = "RED" if ov >= 0.60 else ("YELLOW" if ov >= 0.32 else "GREEN")
            # 自分の測定値を2つ以上含む文は、その学生固有のデータ記述なので「丸被り」扱いしない
            if status == "RED" and len(re.findall(r"\d+\.\d+|\d{2,}", c_txt)) >= 2:
                status = "YELLOW"
            claim_diags.append(
                ClaimDiagnostic(
                    index=idx + 1,
                    text=c_txt,
                    cohort_overlap_rate=round(ov, 4),
                    max_similarity=round(max(0.0, best_s), 4),
                    matched_example=best_ex,
                    status=status,
                )
            )
        mean_claim_ov = float(np.mean(overlaps)) if overlaps else 0.5
        claim_nov_norm = max(0.0, min(1.0, 1.0 - mean_claim_ov))
        dehomog_score = 0.60 * claim_nov_norm + 0.40 * vendi_score_norm

        # ----------------------------------------------------------------------
        # 2) 脱・理系逃げフレーズスコア（Anti-STEM-Slop）
        # ----------------------------------------------------------------------
        sentences = split_sentences(discussion_text)
        slop_hits: List[Dict[str, str]] = []
        slop_penalty_sum = 0.0
        for pat, w, cat in STEM_SLOP_PATTERNS:
            for sent in sentences:
                m = re.search(pat, sent)
                if m:
                    # 同じ文に具体的数値・%・オーダー評価が添えられていればペナルティを軽減
                    has_quant_in_sent = bool(
                        re.search(r"\d+(?:\.\d+)?\s*(?:%|％|nm|mm|K|℃|倍|桁)|オーダー|寄与", sent)
                    )
                    eff_w = w * (0.35 if has_quant_in_sent else 1.0)
                    slop_penalty_sum += eff_w
                    slop_hits.append(
                        {
                            "phrase": m.group(0),
                            "category": cat,
                            "quantified": "定量化あり" if has_quant_in_sent else "未定量（逃げ表現）",
                        }
                    )
        # 重複phrase整理
        slop_rate = slop_penalty_sum / max(len(sentences), 3)
        anti_stem_slop = math.exp(-1.55 * slop_rate)

        # ----------------------------------------------------------------------
        # 3) 数値データ自動監査 ＆ 誤差方向・系統トレンド整合スコア
        # ----------------------------------------------------------------------
        num_audit = audit_numeric_data(data_series)
        # 方向語は「文献値/理論値」等の基準語と同じ文に出たときだけ、誤差の向きの議論とみなす
        ref_sents = [s_ for s_ in sentences if re.search(REFERENCE_WORDS, s_)]
        has_neg = any(re.search(NEG_BIAS_WORDS, s_) for s_ in ref_sents)
        has_pos = any(re.search(POS_BIAS_WORDS, s_) for s_ in ref_sents)
        has_sys = bool(re.search(SYSTEMATIC_WORDS, discussion_text))
        has_rand_only = bool(re.search(RANDOM_ONLY_WORDS, discussion_text)) and not has_sys
        has_extrap = bool(re.search(r"外挿|切片|回帰|線形|Bradley|Nelson|cos\^?2|傾き", discussion_text))

        bias_contradiction = False
        if num_audit.has_data:
            score_b = 0.20  # ベースライン
            # 方向性（符号）を正しく認識しているか
            if num_audit.bias_direction == "NEGATIVE" and has_neg:
                score_b += 0.25
            elif num_audit.bias_direction == "POSITIVE" and has_pos:
                score_b += 0.25
            elif num_audit.bias_direction == "MIXED" and (has_neg and has_pos or "逆転" in discussion_text):
                score_b += 0.30

            # 強い系統バイアスがあるのにランダム誤差だけで逃げたら物理的矛盾ペナルティ
            if num_audit.has_strong_systematic_bias:
                if has_sys:
                    score_b += 0.35
                elif has_rand_only:
                    bias_contradiction = True
                    score_b = max(0.05, score_b - 0.30)
            else:
                if has_sys:
                    score_b += 0.25

            # 強い線形トレンドがある場合に外挿/トレンドに踏み込んでいるか
            if num_audit.trend_r2 is not None and num_audit.trend_r2 >= 0.80:
                if has_extrap:
                    score_b += 0.20
                else:
                    score_b *= 0.82
            else:
                if has_extrap:
                    score_b += 0.15
            bias_consistency = min(1.0, max(0.0, score_b))
        else:
            # 数値データ未添付時はテキスト内の系統/方向性議論で評価
            bias_consistency = min(
                1.0,
                0.20
                + (0.30 if (has_neg or has_pos) else 0.0)
                + (0.35 if has_sys else 0.0)
                + (0.15 if has_extrap else 0.0)
                - (0.20 if has_rand_only else 0.0),
            )
            bias_consistency = max(0.0, bias_consistency)

        # ----------------------------------------------------------------------
        # 4) オーダー評価・誤差伝播・定量裏付けスコア
        # ----------------------------------------------------------------------
        quant_hits = sum(len(re.findall(p, discussion_text)) for p in quant_patterns)
        has_error_budget = bool(
            re.search(r"誤差伝播|偏微分|オーダー|寄与|説明できる|削減|外挿.*(?:nm|%|％)|[±]\s*\d", discussion_text)
        )
        char_scale = max(len(discussion_text) / 350.0, 1.0)
        q_density = 1.0 - math.exp(-0.35 * (quant_hits / char_scale))
        quant_rigor = min(1.0, 0.65 * q_density + (0.35 if has_error_budget else 0.0))

        # ----------------------------------------------------------------------
        # 5) 物理メカニズム・理論近似の破綻特定スコア
        # ----------------------------------------------------------------------
        mech_hits = sum(len(re.findall(p, discussion_text)) for p in mech_patterns)
        mech_norm = 1.0 - math.exp(-0.32 * (mech_hits / char_scale))
        # 指定された理論前提キーワードへの言及ボーナス
        assumption_cov = 0.0
        if data_series and data_series.assumptions:
            matched_a = sum(1 for a in data_series.assumptions if a in discussion_text)
            assumption_cov = matched_a / len(data_series.assumptions)
            mechanism_score = min(1.0, 0.72 * mech_norm + 0.28 * assumption_cov)
        else:
            mechanism_score = mech_norm

        # ----------------------------------------------------------------------
        # 6) 結果再述からの脱却度（Beyond-Restatement Depth）
        # ----------------------------------------------------------------------
        exp_sents = sum(
            1 for s in sentences if any(re.search(p, s) for p in EXPLANATORY_PATTERNS)
        )
        rest_sents = sum(
            1 for s in sentences if any(re.search(p, s) for p in RESTATEMENT_PATTERNS)
        )
        n_sents = max(len(sentences), 1)
        exp_ratio = exp_sents / n_sents
        # 説明文が45%以上で満点に近づく。さらに全段が結果再述だけなら低く抑える
        beyond_restatement = min(1.0, exp_ratio / 0.48)
        if exp_sents == 0 and rest_sents > 0:
            beyond_restatement = 0.08
        elif len(sentences) <= 2 and len(discussion_text) < 120:
            beyond_restatement *= 0.45

        # ----------------------------------------------------------------------
        # Gate) 理系課題適合・文章品質ゲート G(x|q)
        # ----------------------------------------------------------------------
        draft_prompt_sim = float(np.clip(draft_vec @ prompt_vec, -1.0, 1.0))
        cohort_prompt_sim = float(np.mean(np.clip(cohort_vecs @ prompt_vec, -1.0, 1.0)))
        mean_cohort_sim = float(np.mean(np.clip(cohort_vecs @ draft_vec, -1.0, 1.0)))
        rel_ratio = draft_prompt_sim / max(cohort_prompt_sim, 0.12)
        sem_rel = self._sigmoid(8.5 * (rel_ratio - 0.28))

        prompt_terms = set(extract_content_tokens(experiment_prompt))
        if data_series:
            prompt_terms.update(extract_content_tokens(data_series.name))
            for a in data_series.assumptions:
                prompt_terms.update(extract_content_tokens(a))
        draft_low = discussion_text.lower()
        lex_rel = (
            min(1.0, sum(1 for t in prompt_terms if t in draft_low) / max(len(prompt_terms) * 0.20, 1.0))
            if prompt_terms
            else 1.0
        )
        # 同じ実験の母集団との最低限の専門用語共有＋高度な専門語追加で希釈されない語彙アンカー
        cohort_anchor = self._sigmoid(10.0 * (mean_cohort_sim - 0.14))
        topic_gate = max(
            0.55 * sem_rel + 0.45 * lex_rel,
            0.96 * lex_rel,
            0.85 * cohort_anchor,
        )
        if draft_prompt_sim < 0.07 and lex_rel < 0.15 and mean_cohort_sim < 0.11:
            topic_gate *= 0.12

        len_ok = min(1.0, len(discussion_text.strip()) / 140.0)
        sent_ok = min(1.0, len(sentences) / 2.5)
        gate = float(min(1.0, max(0.0, topic_gate * (0.55 * len_ok + 0.45 * sent_ok))))

        # ----------------------------------------------------------------------
        # 総合スコア合成
        # ----------------------------------------------------------------------
        raw_total = (
            0.20 * dehomog_score
            + 0.16 * anti_stem_slop
            + 0.18 * bias_consistency
            + 0.16 * quant_rigor
            + 0.16 * mechanism_score
            + 0.14 * beyond_restatement
        )
        final_score = 100.0 * (gate ** 1.4) * raw_total

        if gate < 0.45:
            grade = "GATE_FAIL (論点ズレ/考察不足)"
        elif final_score >= 76:
            grade = "S (加点レベル・発展的考察)"
        elif final_score >= 62:
            grade = "A (論理的・定量的考察)"
        elif final_score >= 46:
            grade = "B (標準・一部定型/再述)"
        elif final_score >= 30:
            grade = "C (量産型AI寄り・要深掘り)"
        else:
            grade = "D (量産型テンプレ/結果再述のみ)"

        # 2D PCA可視化座標
        all_p = np.vstack([cohort_vecs, draft_vec.reshape(1, -1)])
        mu_p = np.mean(cohort_vecs, axis=0)
        cent_p = all_p - mu_p
        _, s_vals, vt = np.linalg.svd(cent_p, full_matrices=False)
        comps = vt[:2] if vt.shape[0] >= 2 else np.vstack([vt, np.zeros((2 - vt.shape[0], vt.shape[1]))])
        coords = cent_p @ comps.T
        exp_var = (s_vals[:2] ** 2) / max(float(np.sum(s_vals**2)), 1e-12)
        pca_proj = {
            "cohort_points": [
                {"id": f"AI-{i+1}", "x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4)}
                for i in range(cohort_vecs.shape[0])
            ],
            "draft_point": {
                "id": "YOUR_DISCUSSION",
                "x": round(float(coords[-1, 0]), 4),
                "y": round(float(coords[-1, 1]), 4),
            },
            "explained_variance": [round(float(v), 4) for v in exp_var],
        }

        # 具体的な理系ピボット提案の自動生成
        suggestions: List[str] = []
        if bias_contradiction:
            suggestions.append(
                "【物理的矛盾の警告】実測値は一方向に系統バイアスを持っていますが、考察がランダム/読み取り誤差のみに依存しています。一定方向のズレを生む系統誤差メカニズムに書き換えてください。"
            )
        if num_audit.trend_r2 is not None and num_audit.trend_r2 >= 0.80 and not has_extrap:
            suggestions.append(
                f"【データに埋もれた高得点ポイント】{data_series.independent_name if data_series else '独立変数'}に対して強い線形トレンド(R²={num_audit.trend_r2:.3f})があります。"
                f"単純平均ではなく x→0 へ外挿した値（{num_audit.extrapolated_at_zero}、誤差{num_audit.extrapolated_rel_error_pct:+.3f}%）を算出し、"
                "系統誤差除去後の残差の物理的原因を考察すると加点レベルの独自考察になります。"
            )
        if beyond_restatement < 0.45:
            suggestions.append(
                "【結果の再述から『なぜ』への転換】現在の文章は測定結果の言い換え比率が高い状態です。「なぜその符号・大きさになったのか」という物理・化学機構と検証仮説を最低2文追加してください。"
            )
        unquant_slops = [h for h in slop_hits if h["quantified"] != "定量化あり"]
        if unquant_slops:
            phrases = "」「".join(dict.fromkeys(h["phrase"] for h in unquant_slops[:3]))
            suggestions.append(
                f"【理系AI逃げフレーズの駆逐】「{phrases}」が定量評価なしで出現しています。その要因で生じるズレが理論値比で何%（何桁）に相当するかオーダー見積もりを添えてください。"
            )
        if quant_rigor < 0.55:
            suggestions.append(
                "【オーダー評価・誤差伝播の追加】誤差要因を挙げる際は、使用した理論式の偏微分や感度見積もり（例：パラメータが±Δだけ動くと結果が何%変わるか）を入れて主因と副因を切り分けてください。"
            )
        red_claims = [c for c in claim_diags if c.status == "RED"]
        if red_claims:
            top_r = sorted(red_claims, key=lambda c: c.cohort_overlap_rate, reverse=True)[0]
            suggestions.append(
                f"【量産型AIとの丸被り箇所】主張#{top_r.index}（被り推定 {top_r.cohort_overlap_rate*100:.0f}%：「{top_r.text[:35]}…」）は他学生のAI考察と重複しています。理論式の前提条件の破綻や次回の検証実験設計へ差し替えてください。"
            )

        return StemScoringResult(
            final_score=round(float(final_score), 2),
            grade=grade,
            raw_depth_originality=round(float(raw_total * 100.0), 2),
            relevance_gate=round(float(gate), 4),
            metrics={
                "dehomog_score": round(dehomog_score * 100.0, 1),
                "anti_stem_slop_score": round(anti_stem_slop * 100.0, 1),
                "bias_trend_consistency": round(bias_consistency * 100.0, 1),
                "quant_error_rigor": round(quant_rigor * 100.0, 1),
                "mechanism_depth": round(mechanism_score * 100.0, 1),
                "beyond_restatement": round(beyond_restatement * 100.0, 1),
            },
            raw_stats={
                "delta_vendi_score": round(delta_vs, 4),
                "cohort_vendi_score": round(vs_before, 4),
                "mean_claim_overlap_rate": round(mean_claim_ov, 4),
                "unquantified_slop_count": len(unquant_slops),
                "quant_marker_count": quant_hits,
                "mechanism_marker_count": mech_hits,
                "explanatory_sentence_ratio": round(exp_ratio, 3),
                "bias_contradiction": bias_contradiction,
                "draft_prompt_sim": round(draft_prompt_sim, 4),
                "mean_cohort_sim": round(mean_cohort_sim, 4),
            },
            numeric_audit=num_audit,
            claims=claim_diags,
            slop_hits=slop_hits,
            scientific_suggestions=suggestions[:5],
            pca_projection=pca_proj,
        )
