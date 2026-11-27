"""「脱・量産型AI」多角的数理スコアリングエンジン.

以下の6つの独立した数理軸＋課題適合・品質ゲートを統合する：
1. Marginal Vendi Score Gain (ΔVS): 類似度カーネル固有値のフォン・ノイマンエントロピー増分
2. Regularized Mahalanobis Distance (D_M): 量産型AI多様体の主成分分散を割り引いた直交逸脱度
3. Nearest-Neighbor Cluster Escape (1 - ρ_max): サブクラスタ丸被り検知
4. Claim-Level Novelty (Information Gain): 主張単位のクラス内被り率からの独自性
5. Algorithmic Skeleton & Anti-Slop: 段落骨格の編集距離＋日本語AI常套句密度＋文長バースト性
6. Specificity & Critical Rigor: 一次情報・具体数値・引用・反論構造の密度
Gate: Relevance & Coherence Gate G(x|q): 課題適合度と論理妥当性による乗算ゲート
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embeddings import EmbeddingBackend, LocalHybridEmbedder
from .text_ja import (
    StructureAnalysis,
    analyze_structure,
    coherence_analysis,
    extract_claims,
    extract_content_tokens,
    normalized_levenshtein,
    verification_needed,
)


def vendi_score(similarity_matrix: np.ndarray, q: float = 1.0) -> float:
    """Vendi Score (Friedman & Dieng, TMLR 2023) を計算する.

    半正定値類似度行列 K (対角成分1, K_ij >= 0) に対し、
    K / n の固有値 λ_i から exp(-sum λ_i ln λ_i)（実質ユニーク要素数）を返す。
    """
    n = similarity_matrix.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    # 対称化＋数値誤差対策で半正定値へクリップ
    K = 0.5 * (similarity_matrix + similarity_matrix.T)
    eigvals = np.linalg.eigvalsh(K / float(n))
    eigvals = eigvals[eigvals > 1e-12]
    eigvals = eigvals / np.sum(eigvals)
    if abs(q - 1.0) < 1e-9:
        entropy = -float(np.sum(eigvals * np.log(eigvals)))
        return float(math.exp(entropy))
    return float(np.sum(eigvals**q) ** (1.0 / (1.0 - q)))


def marginal_vendi_gain(
    cohort_vecs: np.ndarray,
    draft_vec: np.ndarray,
    kernel_power: float = 2.0,
) -> Tuple[float, float, float]:
    """量産型母集団にドラフトを追加した際の限界Vendiスコア増分 ΔVS を計算する.

    コサイン類似度 s ∈ [-1, 1] を正のカーネル k(u,v) = ((1 + s)/2)^p へ写像して
    PSD性を保ちつつ解像度を高める。
    """
    n = cohort_vecs.shape[0]
    if n == 0:
        return 1.0, 0.0, 1.0
    sim_base = np.clip(cohort_vecs @ cohort_vecs.T, -1.0, 1.0)
    K_base = np.power(0.5 * (1.0 + sim_base), kernel_power)
    np.fill_diagonal(K_base, 1.0)
    vs_before = vendi_score(K_base)

    all_vecs = np.vstack([cohort_vecs, draft_vec.reshape(1, -1)])
    sim_all = np.clip(all_vecs @ all_vecs.T, -1.0, 1.0)
    K_all = np.power(0.5 * (1.0 + sim_all), kernel_power)
    np.fill_diagonal(K_all, 1.0)
    vs_after = vendi_score(K_all)

    delta_vs = max(0.0, vs_after - vs_before)
    return float(delta_vs), float(vs_before), float(vs_after)


@dataclass
class ScoringConfig:
    """多角的スコアリングの重み・ハイパーパラメータ設定."""

    mode: str = "report"  # "report" (大学レポート) or "es" (就活ES)
    # 6指標の重み（合計1.0）
    w_vendi: float = 0.22
    w_mahalanobis: float = 0.16
    w_knn: float = 0.14
    w_claim: float = 0.24
    w_skeleton: float = 0.12
    w_specificity: float = 0.12
    # 品質ゲート指数 α（大きいほど論点ズレを厳しく減点）
    gate_alpha: float = 1.6
    # Claim一致のしきい値（ソフトマッチング中心）
    claim_match_threshold: float = 0.42
    claim_match_sharpness: float = 12.0
    # マハラノビス正則化係数 λ (Ledoit-Wolf 収縮比)
    mahalanobis_shrinkage: float = 0.25
    # 推奨最小文字数
    min_chars: int = 180


@dataclass
class ClaimDiagnostic:
    """主張（文）単位の被り率診断."""

    index: int
    text: str
    cohort_overlap_rate: float  # 0.0 〜 1.0（量産型AIの何%がこの主張を含むか）
    max_similarity: float
    matched_example: str
    status: str  # "RED" (量産型丸被り), "YELLOW" (要注意), "GREEN" (独自視点)


@dataclass
class ScoringResult:
    """多角的独自性スコアリングの完全出力."""

    final_score: float  # 0 〜 100（独自性×課題適合ゲート。後方互換のため残す）
    grade: str  # 独自性の判定（ゲート前の純粋独自性で判定）。ゲート不合格時のみ GATE_FAIL
    raw_originality: float  # ゲート前の純粋独自性 (0 〜 100)
    relevance_gate: float  # 0.0 〜 1.0
    # 6つの数理サブスコア (各 0 〜 100)
    metrics: Dict[str, float]
    # 論文・解釈用の生統計量
    raw_stats: Dict[str, float]
    # 主張単位ヒートマップ診断
    claims: List[ClaimDiagnostic]
    # 構造・AI臭解析
    skeleton: List[str]
    cohort_dominant_skeleton: List[str]
    slop_hits: List[str]
    # 脱・量産型ピボット提案
    orthogonal_suggestions: List[str]
    # 2D PCA可視化座標
    pca_projection: Dict[str, Any] = field(default_factory=dict)
    # 「独自性」とは別軸の品質指標（合計点にはまとめない）
    uniqueness_score: float = 0.0
    quality_scores: Dict[str, float] = field(default_factory=dict)
    quality_flags: List[str] = field(default_factory=list)
    # 本人が根拠を確認すべき記述（ツールは真偽を判定できない）
    verification_needed: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class UniquenessScorer:
    """課題文と仮想・量産型AI母集団を用いてドラフトの独自性を数理評価するコアエンジン."""

    def __init__(
        self,
        backend: Optional[EmbeddingBackend] = None,
        config: Optional[ScoringConfig] = None,
    ) -> None:
        self.backend = backend or LocalHybridEmbedder()
        self.config = config or ScoringConfig()

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _compute_mahalanobis(
        self,
        cohort_vecs: np.ndarray,
        draft_vec: np.ndarray,
    ) -> Tuple[float, float, float]:
        """正則化マハラノビス距離とLeave-One-Out (LOO) 基準のzスコアを返す.

        母集団内の各点も必ず標本外（Leave-One-Out）として距離計算することで、
        高次元空間（D >> M）での直交部分空間バイアスを完全に除去する。
        """
        m, d = cohort_vecs.shape
        lam = self.config.mahalanobis_shrinkage

        def _fit_and_dist(train_vecs: np.ndarray, target_vec: np.ndarray) -> float:
            mt = train_vecs.shape[0]
            mu_t = np.mean(train_vecs, axis=0)
            centered = train_vecs - mu_t
            trace_s = float(np.sum(centered * centered)) / max(mt, 1)
            iso = max(trace_s / max(d, 1), 1e-6)
            alpha = max(lam * iso, 1e-6)
            scale = (1.0 - lam) / max(mt, 1)

            # Woodbury 恒等式で (αI + scale * X^T X)^{-1} (v - μ) を厳密計算
            gram = centered @ centered.T
            A = (alpha / max(scale, 1e-12)) * np.eye(mt) + gram
            diff = target_vec - mu_t
            xv = centered @ diff
            sol = np.linalg.solve(A, xv)
            inv_diff = (diff - centered.T @ sol) / alpha
            quad = max(0.0, float(diff @ inv_diff))
            return math.sqrt(quad)

        # 母集団のLOO距離分布（すべて標本外なのでdraft_vecと幾何的に完全対等）
        loo_dists = np.array(
            [_fit_and_dist(np.delete(cohort_vecs, i, axis=0), cohort_vecs[i]) for i in range(m)],
            dtype=np.float64,
        )
        draft_dist = _fit_and_dist(cohort_vecs, draft_vec)
        mean_d = float(np.mean(loo_dists))
        std_d = max(float(np.std(loo_dists)), mean_d * 0.08, 1e-4)
        z_score = (draft_dist - mean_d) / std_d
        # z=0（量産型平均）で低く、z>=2.5で高く飽和
        maha_score = self._sigmoid(1.05 * (z_score - 1.1))
        return float(draft_dist), float(z_score), float(maha_score)

    def _compute_knn_escape(
        self,
        cohort_vecs: np.ndarray,
        draft_vec: np.ndarray,
    ) -> Tuple[float, float, float]:
        """最近傍コサイン類似度 ρ_max と上位k近傍平均からクラスタ脱出度を計算する."""
        sims = np.clip(cohort_vecs @ draft_vec, -1.0, 1.0)
        rho_max = float(np.max(sims))
        k = min(3, len(sims))
        rho_knn = float(np.mean(np.sort(sims)[-k:]))

        # 母集団同士の最近傍類似度（キャリブレーション基準）
        c_sim = np.clip(cohort_vecs @ cohort_vecs.T, -1.0, 1.0)
        np.fill_diagonal(c_sim, -1.0)
        cohort_nn_mean = float(np.mean(np.max(c_sim, axis=1)))
        cohort_nn_std = float(np.std(np.max(c_sim, axis=1))) + 0.03

        # 母集団内の典型的な最近傍距離と比べてどれだけ離れているか
        escape_z = (cohort_nn_mean - rho_max) / cohort_nn_std
        # 絶対類似度も加味（rho_max が 0.85 以上なら確実に丸被り）
        abs_escape = max(0.0, min(1.0, (0.92 - rho_max) / 0.55))
        rel_escape = self._sigmoid(1.0 * (escape_z - 0.5))
        knn_score = 0.55 * rel_escape + 0.45 * abs_escape
        return rho_max, rho_knn, float(knn_score)

    def _compute_claim_diagnostics(
        self,
        draft_claims: Sequence[str],
        cohort_claims_per_doc: Sequence[Sequence[str]],
        draft_claim_vecs: np.ndarray,
        cohort_claim_vecs_per_doc: Sequence[np.ndarray],
    ) -> Tuple[List[ClaimDiagnostic], float, float]:
        """主張単位の「クラス内被り率」と全体Claim独自性スコアを計算する."""
        m_docs = len(cohort_claims_per_doc)
        diagnostics: List[ClaimDiagnostic] = []
        if len(draft_claims) == 0 or m_docs == 0:
            return diagnostics, 0.5, 0.5

        thresh = self.config.claim_match_threshold
        sharp = self.config.claim_match_sharpness

        overlap_rates: List[float] = []
        for idx, (c_text, c_vec) in enumerate(zip(draft_claims, draft_claim_vecs)):
            doc_covered_probs: List[float] = []
            best_sim = -1.0
            best_example = ""
            for doc_claims, doc_vecs in zip(cohort_claims_per_doc, cohort_claim_vecs_per_doc):
                if doc_vecs.shape[0] == 0:
                    doc_covered_probs.append(0.0)
                    continue
                sims = np.clip(doc_vecs @ c_vec, -1.0, 1.0)
                max_idx = int(np.argmax(sims))
                max_s = float(sims[max_idx])
                if max_s > best_sim:
                    best_sim = max_s
                    best_example = doc_claims[max_idx]
                # ソフトステップ関数で「この量産型レポートが同主張を含む確率」を算出
                prob = self._sigmoid(sharp * (max_s - thresh))
                doc_covered_probs.append(prob)

            overlap_rate = float(np.mean(doc_covered_probs))
            overlap_rates.append(overlap_rate)

            if overlap_rate >= 0.65:
                status = "RED"
            elif overlap_rate >= 0.35:
                status = "YELLOW"
            else:
                status = "GREEN"

            diagnostics.append(
                ClaimDiagnostic(
                    index=idx + 1,
                    text=c_text,
                    cohort_overlap_rate=round(overlap_rate, 4),
                    max_similarity=round(max(0.0, best_sim), 4),
                    matched_example=best_example,
                    status=status,
                )
            )

        mean_overlap = float(np.mean(overlap_rates))
        # 内容の薄い短文（「しかし、この通説は疑わしい。」等）は独自主張としての重みを下げる。
        # 中身のない文を足すだけで独自性が上がるスコアハックへの対策。
        weights = np.array(
            [min(1.0, len(set(extract_content_tokens(t))) / 6.0) for t in draft_claims], dtype=np.float64
        )
        weights = np.maximum(weights, 0.15)
        novs = 1.0 - np.asarray(overlap_rates, dtype=np.float64)
        w_mean_nov = float(np.sum(weights * novs) / np.sum(weights))
        # 上位独自主張の比率も加味（導入1文が被っていても本論が独自なら評価）
        order = np.argsort(-(novs * weights))
        k_top = max(1, math.ceil(len(order) * 0.6))
        top = order[:k_top]
        top_nov = float(np.sum(weights[top] * novs[top]) / np.sum(weights[top]))
        claim_novelty = 0.55 * w_mean_nov + 0.45 * top_nov
        return diagnostics, float(mean_overlap), float(min(1.0, max(0.0, claim_novelty)))

    def _compute_skeleton_score(
        self,
        draft_struct: StructureAnalysis,
        cohort_structs: Sequence[StructureAnalysis],
    ) -> Tuple[float, List[str], float]:
        """骨格のテンプレート逸脱度＋脱AI-Slop＋文長バースト性スコアを統合する."""
        if not cohort_structs:
            return 0.5, [], 0.5

        dists = [
            normalized_levenshtein(draft_struct.skeleton, cs.skeleton)
            for cs in cohort_structs
        ]
        mean_skel_dist = float(np.mean(dists))
        # 最頻出の量産型骨格を特定
        skel_counts: Dict[Tuple[str, ...], int] = {}
        for cs in cohort_structs:
            key = tuple(cs.skeleton)
            skel_counts[key] = skel_counts.get(key, 0) + 1
        dominant_skel = list(max(skel_counts.items(), key=lambda kv: kv[1])[0])

        # 文長バースト性（AIは0.20前後で均一、人間・良質文は0.35〜0.65）
        burst_score = min(1.0, max(0.0, (draft_struct.burstiness - 0.15) / 0.40))
        anti_slop = 1.0 - draft_struct.slop_density

        skeleton_score = 0.40 * min(1.0, mean_skel_dist * 1.4) + 0.42 * anti_slop + 0.18 * burst_score
        return float(min(1.0, max(0.0, skeleton_score))), dominant_skel, mean_skel_dist

    def _compute_relevance_gate(
        self,
        assignment_prompt: str,
        draft: str,
        prompt_vec: np.ndarray,
        draft_vec: np.ndarray,
        cohort_vecs: np.ndarray,
        draft_struct: StructureAnalysis,
        rubric_keywords: Optional[Sequence[str]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """論点ズレ・支離滅裂な文章によるスコアハックを防ぐ品質ゲート G(x|q) ∈ [0,1].

        設問文をそのままオウム返ししない独自表現を不当に減点しないよう、
        一定以上の課題適合が確認できれば滑らかに1.0へ飽和させ、明らかな論点逸脱だけを強く遮断する。
        """
        # 1) 課題文との意味的関連度（母集団の平均関連度を基準にキャリブレーション）
        draft_prompt_sim = float(np.clip(draft_vec @ prompt_vec, -1.0, 1.0))
        cohort_prompt_sims = np.clip(cohort_vecs @ prompt_vec, -1.0, 1.0)
        cohort_ref_sim = float(np.mean(cohort_prompt_sims))
        mean_cohort_sim = float(np.mean(np.clip(cohort_vecs @ draft_vec, -1.0, 1.0)))

        rel_ratio = draft_prompt_sim / max(cohort_ref_sim, 0.12)
        # 0.35以上で急速に合格圏へ立ち上がり、0.55以上でほぼ1.0に飽和
        semantic_rel = self._sigmoid(9.0 * (rel_ratio - 0.32))

        # 2) 課題キーワード・ルーブリック要素の語彙アンカー適合
        prompt_terms = set(extract_content_tokens(assignment_prompt))
        if rubric_keywords:
            prompt_terms.update(t.lower() for t in rubric_keywords)
        draft_low = draft.lower()
        if prompt_terms:
            matched_terms = sum(1 for t in prompt_terms if t in draft_low)
            term_coverage = matched_terms / len(prompt_terms)
            lexical_rel = min(1.0, term_coverage / 0.22)
        else:
            lexical_rel = 1.0

        # 3) ESモードなど「体験談の構造」を問う課題向けの構造適合アンカー
        # ただし最低限の意味的接続（draft_prompt_sim または mean_cohort_sim）がある場合のみ有効
        es_struct_bonus = 0.0
        if self.config.mode == "es" and (draft_prompt_sim >= 0.08 or mean_cohort_sim >= 0.12):
            has_roles = set(draft_struct.skeleton)
            expected_dims = 0
            if any(k in draft_low for k in ["大学", "学生", "ゼミ", "バイト", "アルバイト", "サークル", "部活", "研究", "活動", "立ち上げ", "取り組"]):
                expected_dims += 1
            if any(k in draft_low for k in ["課題", "壁", "問題", "赤字", "反対", "限界", "困難", "失敗", "低迷"]):
                expected_dims += 1
            if any(k in draft_low for k in ["分析", "転換", "工夫", "見直", "実行", "改善", "導入", "撤退", "変え"]):
                expected_dims += 1
            if any(k in draft_low for k in ["結果", "成果", "黒字", "向上", "達成", "痛感", "学ん", "学び"]):
                expected_dims += 1
            es_struct_bonus = expected_dims / 4.0
            if len(has_roles) >= 2:
                es_struct_bonus = min(1.0, es_struct_bonus + 0.1)

        # 4) 文量・構造の妥当性（極端な短文や単語羅列を防ぐ）
        char_len = len(draft.strip())
        length_ok = min(1.0, char_len / float(self.config.min_chars))
        sent_ok = min(1.0, len(draft_struct.sentences) / 3.0)
        jp_ratio = len([c for c in draft if "\u3040" <= c <= "\u309f"]) / max(char_len, 1)
        fluency_ok = 1.0 if jp_ratio >= 0.18 else max(0.2, jp_ratio / 0.18)

        base_topic_gate = 0.65 * semantic_rel + 0.35 * lexical_rel
        topic_gate = max(base_topic_gate, 0.88 * es_struct_bonus + 0.12 * semantic_rel)

        # 課題関連が致命的に低い（完全な別トピック）場合は確実に遮断
        if draft_prompt_sim < 0.08 and lexical_rel < 0.15 and mean_cohort_sim < 0.12:
            topic_gate *= 0.15

        form_gate = 0.45 * length_ok + 0.30 * sent_ok + 0.25 * fluency_ok
        gate = float(min(1.0, max(0.0, topic_gate * form_gate)))

        details = {
            "draft_prompt_sim": round(draft_prompt_sim, 4),
            "cohort_prompt_sim": round(cohort_ref_sim, 4),
            "mean_cohort_sim": round(mean_cohort_sim, 4),
            "lexical_coverage": round(lexical_rel, 4),
            "form_adequacy": round(form_gate, 4),
        }
        return gate, details

    def _compute_pca_2d(
        self,
        cohort_vecs: np.ndarray,
        draft_vec: np.ndarray,
    ) -> Dict[str, Any]:
        """母集団＋ドラフトをNumPyのSVDで2次元PCA射影（可視化用）."""
        all_vecs = np.vstack([cohort_vecs, draft_vec.reshape(1, -1)])
        mu = np.mean(cohort_vecs, axis=0)
        centered = all_vecs - mu
        # SVDで上位2主成分を取得
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        comps = vt[:2] if vt.shape[0] >= 2 else np.vstack([vt, np.zeros((2 - vt.shape[0], vt.shape[1]))])
        coords = centered @ comps.T
        explained = (s[:2] ** 2) / max(float(np.sum(s**2)), 1e-12)
        return {
            "cohort_points": [
                {"id": f"AI-{i+1}", "x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4)}
                for i in range(cohort_vecs.shape[0])
            ],
            "draft_point": {
                "id": "YOUR_DRAFT",
                "x": round(float(coords[-1, 0]), 4),
                "y": round(float(coords[-1, 1]), 4),
            },
            "explained_variance": [round(float(v), 4) for v in explained],
        }

    def _build_suggestions(
        self,
        assignment_prompt: str,
        claims: Sequence[ClaimDiagnostic],
        draft_struct: StructureAnalysis,
        relevance_gate: float,
    ) -> List[str]:
        """数理診断に基づき、量産型AIから抜け出す具体的ピボット案を生成する."""
        suggestions: List[str] = []

        if relevance_gate < 0.55:
            suggestions.append(
                "【最優先：課題適合の改善】課題文の主要論点との距離が離れすぎています。独自性を狙う前に、課題が問うている対象・条件に明確に接続する導入と結論を置いてください。"
            )

        red_claims = [c for c in claims if c.status == "RED"]
        if red_claims:
            top_red = sorted(red_claims, key=lambda c: c.cohort_overlap_rate, reverse=True)[0]
            pct = int(round(top_red.cohort_overlap_rate * 100))
            suggestions.append(
                f"【飽和論点の差し替え】主張#{top_red.index}（「{top_red.text[:36]}…」）は量産型AIの約{pct}%と重複しています。一般論の代わりに「例外となる現場・対象」や「あえて逆説的なコスト」へ論点をずらしてください。"
            )

        # LLMバックエンドがあれば未踏の直交切り口も取得
        if red_claims:
            llm_angles = self.backend.suggest_angles_llm(
                assignment_prompt,
                [c.text for c in red_claims],
                mode=self.config.mode,
            )
            if llm_angles:
                for ang in llm_angles:
                    suggestions.append(f"【未踏の直交視点案】{ang}")

        if draft_struct.evidence_count < 2:
            suggestions.append(
                "【一次情報・具体データの追加】固有名詞・統計数値・実体験（ゼミ/バイト/調査データ）が不足しています。"
                "出典を示せる年次データや、実際に経験・確認した事例を1つ深掘りしてください。"
                "※実在しない調査・数値・体験を書くことは捏造（研究不正）です。"
            )

        if draft_struct.counterarg_count == 0:
            suggestions.append(
                "【反証・トレードオフの導入】量産型AIは「メリット＋デメリット→バランスが大事」で着地しがちです。「通説が成り立たない条件」や「誰が損をするか（利害対立）」を1段落設けてください。"
            )

        if draft_struct.slop_hits:
            sample_hits = "」「".join(draft_struct.slop_hits[:3])
            suggestions.append(
                f"【AI常套句の削減】「{sample_hits}」といった量産型フレーズが検出されました。抽象的な結びを削除し、自身の具体的な判断基準・検証仮説で言い換えてください。"
            )

        return suggestions[:5]

    @staticmethod
    def _grade_from_score(score: float, gate: float) -> str:
        """独自性の判定。score にはゲート前の純粋独自性を渡す（ゲート落ちを『丸被り』と誤表示しない）."""
        if gate < 0.45:
            return "GATE_FAIL (論点ズレ/品質不足)"
        if score >= 78:
            return "S (突出した独自性)"
        if score >= 64:
            return "A (高い独自性)"
        if score >= 48:
            return "B (標準・一部量産型)"
        if score >= 32:
            return "C (量産型寄り・要改善)"
        return "D (量産型AIと丸被り)"

    def score(
        self,
        assignment_prompt: str,
        draft: str,
        reference_cohort: Optional[Sequence[str]] = None,
        n_synthetic: int = 12,
        rubric_keywords: Optional[Sequence[str]] = None,
    ) -> ScoringResult:
        """ドラフトの多角的独自性スコアを算出する."""
        if reference_cohort is None or len(reference_cohort) < 3:
            cohort = self.backend.generate_cohort(
                assignment_prompt=assignment_prompt,
                n_samples=n_synthetic,
                mode=self.config.mode,
            )
        else:
            cohort = list(reference_cohort)

        # 文書・主張を一括埋め込み（ローカルTF-IDF/概念射影の一貫性＆API回数最小化）
        draft_claims = extract_claims(draft)
        cohort_claims_per_doc = [extract_claims(doc) for doc in cohort]

        all_texts: List[str] = [assignment_prompt, draft] + cohort
        doc_end = len(all_texts)

        draft_claim_start = doc_end
        all_texts.extend(draft_claims)
        draft_claim_end = len(all_texts)

        cohort_claim_spans: List[Tuple[int, int]] = []
        for c_list in cohort_claims_per_doc:
            st = len(all_texts)
            all_texts.extend(c_list)
            cohort_claim_spans.append((st, len(all_texts)))

        all_vecs = self.backend.embed(all_texts)
        prompt_vec = all_vecs[0]
        draft_vec = all_vecs[1]
        cohort_vecs = all_vecs[2:doc_end]
        draft_claim_vecs = all_vecs[draft_claim_start:draft_claim_end]
        cohort_claim_vecs_per_doc = [all_vecs[st:ed] for st, ed in cohort_claim_spans]

        # 1) Marginal Vendi Score Gain (ΔVS)
        delta_vs, vs_before, vs_after = marginal_vendi_gain(cohort_vecs, draft_vec)
        # 量産型AIの1本が追加する平均ΔVS（Leave-one-out）で相対校正
        loo_gains: List[float] = []
        for i in range(cohort_vecs.shape[0]):
            others = np.delete(cohort_vecs, i, axis=0)
            g_i, _, _ = marginal_vendi_gain(others, cohort_vecs[i])
            loo_gains.append(g_i)
        base_gain_mean = float(np.mean(loo_gains))
        base_gain_std = float(np.std(loo_gains)) + 0.015
        vendi_z = (delta_vs - base_gain_mean) / base_gain_std
        vendi_norm = self._sigmoid(1.05 * (vendi_z - 0.8))

        # 2) Regularized Mahalanobis Distance
        maha_dist, maha_z, maha_norm = self._compute_mahalanobis(cohort_vecs, draft_vec)

        # 3) Nearest-Neighbor Cluster Escape
        rho_max, rho_knn, knn_norm = self._compute_knn_escape(cohort_vecs, draft_vec)

        # 4) Claim-Level Novelty (Information Gain)
        claim_diags, mean_claim_overlap, claim_norm = self._compute_claim_diagnostics(
            draft_claims,
            cohort_claims_per_doc,
            draft_claim_vecs,
            cohort_claim_vecs_per_doc,
        )

        # 5) Algorithmic Skeleton & Anti-Slop
        draft_struct = analyze_structure(draft, mode=self.config.mode)
        cohort_structs = [analyze_structure(doc, mode=self.config.mode) for doc in cohort]
        skeleton_norm, dominant_skeleton, mean_skel_dist = self._compute_skeleton_score(
            draft_struct, cohort_structs
        )

        # 6) Specificity & Critical Rigor
        spec_norm = draft_struct.specificity_score

        # Gate) Relevance & Quality Gate
        gate, gate_details = self._compute_relevance_gate(
            assignment_prompt=assignment_prompt,
            draft=draft,
            prompt_vec=prompt_vec,
            draft_vec=draft_vec,
            cohort_vecs=cohort_vecs,
            draft_struct=draft_struct,
            rubric_keywords=rubric_keywords,
        )

        cfg = self.config
        raw_orig = (
            cfg.w_vendi * vendi_norm
            + cfg.w_mahalanobis * maha_norm
            + cfg.w_knn * knn_norm
            + cfg.w_claim * claim_norm
            + cfg.w_skeleton * skeleton_norm
            + cfg.w_specificity * spec_norm
        )
        final_score = 100.0 * (gate ** cfg.gate_alpha) * raw_orig
        grade = self._grade_from_score(raw_orig * 100.0, gate)

        # 品質（独自性とは別軸）
        coh = coherence_analysis(draft_struct.sentences)
        verify = verification_needed(draft_struct.sentences)
        quality_scores = {
            "task_relevance": round(gate * 100.0, 1),
            "form_adequacy": round(gate_details.get("form_adequacy", 0.0) * 100.0, 1),
            "coherence": round(coh["coherence_score"] * 100.0, 1),
            "adjacency_ratio": coh["adjacency_ratio"],
        }
        quality_flags: List[str] = []
        if 0.45 <= gate < 0.7:
            quality_flags.append("課題適合が低めです（独自性とは別問題）。設問の問いに正面から答えているか確認してください。")
        if coh["opening_connective"]:
            quality_flags.append("1文目が「したがって」「つまり」等の接続詞で始まっています。文の順序や切り貼りを確認してください。")
        if coh["coherence_score"] < 0.45:
            quality_flags.append("隣り合う文の話題のつながりが弱い可能性があります（論理の流れを確認）。")
        if verify:
            quality_flags.append(f"根拠の確認が必要な記述が {len(verify)} 件あります（verification_needed を参照）。")

        pca_proj = self._compute_pca_2d(cohort_vecs, draft_vec)
        suggestions = self._build_suggestions(
            assignment_prompt, claim_diags, draft_struct, gate
        )

        return ScoringResult(
            final_score=round(float(final_score), 2),
            grade=grade,
            raw_originality=round(float(raw_orig * 100.0), 2),
            relevance_gate=round(float(gate), 4),
            metrics={
                "vendi_gain_score": round(vendi_norm * 100.0, 1),
                "mahalanobis_score": round(maha_norm * 100.0, 1),
                "knn_escape_score": round(knn_norm * 100.0, 1),
                "claim_novelty_score": round(claim_norm * 100.0, 1),
                "skeleton_antislop_score": round(skeleton_norm * 100.0, 1),
                "specificity_score": round(spec_norm * 100.0, 1),
            },
            raw_stats={
                "delta_vendi_score": round(delta_vs, 4),
                "cohort_vendi_score": round(vs_before, 4),
                "cohort_loo_vendi_mean": round(base_gain_mean, 4),
                "vendi_z_score": round(vendi_z, 3),
                "mahalanobis_distance": round(maha_dist, 4),
                "mahalanobis_z_score": round(maha_z, 3),
                "max_cosine_similarity": round(rho_max, 4),
                "top3_knn_similarity": round(rho_knn, 4),
                "mean_claim_overlap_rate": round(mean_claim_overlap, 4),
                "skeleton_edit_distance": round(mean_skel_dist, 4),
                "slop_density": round(draft_struct.slop_density, 4),
                "burstiness_cv": round(draft_struct.burstiness, 4),
                "evidence_count": float(draft_struct.evidence_count),
                "counterarg_count": float(draft_struct.counterarg_count),
                **gate_details,
            },
            claims=claim_diags,
            skeleton=draft_struct.skeleton,
            cohort_dominant_skeleton=dominant_skeleton,
            slop_hits=draft_struct.slop_hits,
            orthogonal_suggestions=suggestions,
            pca_projection=pca_proj,
            uniqueness_score=round(float(raw_orig * 100.0), 2),
            quality_scores=quality_scores,
            quality_flags=quality_flags,
            verification_needed=verify,
        )
