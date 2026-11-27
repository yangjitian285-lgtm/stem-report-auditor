"""実レポート3本に対するデモ（旧 test_real_reports.py）.

経済事情論・知的財産権の実レポートへモデルを適用して挙動を確認する。
注意：ローカル埋め込みの概念辞書はこの2分野を含まないため、距離系の指標は信頼性が低い
（評価結果の解釈は README の「検証」節を参照）。データは private_data/ から読み込む。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from uniqueness_scorer.embeddings import LocalHybridEmbedder
from uniqueness_scorer.math_engine import ScoringConfig, UniquenessScorer
from uniqueness_scorer.visualizer import render_single_report_html


# 実レポート本文は個人情報・未公開の提出物を含むため、コードには埋め込まず
# private_data/real_reports.json（git管理外）から読み込む。
_DATA_PATH = Path(__file__).resolve().parent / "private_data" / "real_reports.json"


def _load_data() -> Dict[str, object]:
    if not _DATA_PATH.is_file():
        print(f"[SKIP] {_DATA_PATH} がありません（実レポートは個人データのため同梱していません）。")
        raise SystemExit(0)
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


OUT_DIR = Path(__file__).resolve().parent / "outputs"


def run_case(
    title: str,
    prompt: str,
    draft: str,
    cohort: List[str],
    html_name: str,
) -> None:
    scorer = UniquenessScorer(
        backend=LocalHybridEmbedder(),
        config=ScoringConfig(mode="report"),
    )
    res = scorer.score(
        assignment_prompt=prompt,
        draft=draft,
        reference_cohort=cohort,
    )
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / html_name).write_text(
        render_single_report_html(title=title, assignment_prompt=prompt, result=res),
        encoding="utf-8",
    )

    print(f"\n==================== {title} ====================")
    print(f"文字数           : {len(draft)}字")
    print(f"総合独自性スコア : {res.final_score:.1f} / 100  [{res.grade}]")
    print(
        f"純独自性={res.raw_originality:.1f} | 適合ゲートG={res.relevance_gate:.2f} | "
        f"ΔVS=+{res.raw_stats['delta_vendi_score']:.4f} | "
        f"MahaZ={res.raw_stats['mahalanobis_z_score']:+.2f}σ | "
        f"主張平均被り={res.raw_stats['mean_claim_overlap_rate']*100:.1f}%"
    )
    print("6軸スコア:")
    for k, v in res.metrics.items():
        print(f"  - {k:24s}: {v:5.1f}")
    print(f"骨格: {' -> '.join(res.skeleton)}  (量産型最頻: {' -> '.join(res.cohort_dominant_skeleton)})")
    print(f"検出AI常套句: {res.slop_hits if res.slop_hits else 'なし'}")
    print("主張ヒートマップ（上位/特徴抜粋）:")
    for c in res.claims:
        print(f"  [#{c.index:02d} {c.status:6s} 被り{c.cohort_overlap_rate*100:4.0f}%] {c.text[:55]}")
    print("ピボット提案:")
    for s in res.orthogonal_suggestions:
        print(f"  * {s}")
    print(f"HTML出力 -> {OUT_DIR / html_name}")


def main() -> None:
    d = _load_data()
    run_case(
        "実レポート①：経済事情論 期末レポート",
        d["KEIZAI_PROMPT"],
        d["KEIZAI_REAL_REPORT"],
        d["KEIZAI_COHORT"],
        "real_test_keizai.html",
    )
    run_case(
        "実レポート②：MOT知的財産権 設問1（請求項の限定）",
        d["MOT_Q1_PROMPT"],
        d["MOT_Q1_REAL_REPORT"],
        d["MOT_Q1_COHORT"],
        "real_test_mot_q1.html",
    )
    run_case(
        "実レポート③：MOT知的財産権 設問2（講義の気づき・活用）",
        d["MOT_Q2_PROMPT"],
        d["MOT_Q2_REAL_REPORT"],
        d["MOT_Q2_COHORT"],
        "real_test_mot_q2.html",
    )


if __name__ == "__main__":
    main()
