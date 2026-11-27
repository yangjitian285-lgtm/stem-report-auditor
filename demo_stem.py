"""100%架空データによる理系レポート監査＆脱・量産型考察デモスクリプト.

GitHubからクローンした直後に追加データ・APIキーなしで動作確認できる。
"""

from __future__ import annotations

from uniqueness_scorer.embeddings import LocalHybridEmbedder
from uniqueness_scorer.stem_engine import DataSeriesInput, StemUniquenessScorer


DEMO_PROMPT = (
    "単振り子の周期測定実験から重力加速度gを算出し、"
    "理論値9.80665 m/s^2との比較および系統誤差・近似破綻の要因を考察せよ。"
)

# 振れ角 θ (rad)^2 が大きくなるほど周期が伸び、小角近似で算出した見かけの g が系統的に小さく出る合成データ
DEMO_DATA = DataSeriesInput(
    name="重力加速度 g",
    observed=[9.782, 9.735, 9.660, 9.558],
    reference=9.80665,
    unit="m/s^2",
    independent_var=[0.02, 0.06, 0.12, 0.20],  # θ_0^2 (rad^2)
    independent_name="振れ角二乗 θ0²",
    assumptions=["小角近似 sinθ≈θ", "空気抵抗無視", "糸の質量無視"],
)

DEMO_AI_COHORT = [
    (
        "単振り子の周期から算出した重力加速度の平均値は9.68 m/s^2となり、理論値9.81 m/s^2と概ね一致しており実験の妥当性が確認された。"
        "わずかな誤差の原因として、ストップウォッチの読み取り誤差やヒューマンエラー、室温など実験環境の影響が考えられる。"
        "今後は測定回数を増やし、より高精度な機器を用いて慎重に操作することで誤差を低減できる。"
    ),
    (
        "得られた重力加速度は理論値とほぼ一致し良好な結果が得られた。"
        "若干の差が生じた要因には、目視による計測ミスや手技のばらつき、空気の流れといった環境要因が挙げられる。"
        "今後の課題として試行回数を増やすとともにさらなる検討が必要である。"
    ),
    (
        "本実験により重力加速度を求めたところ、理論値と誤差の範囲内で一致した。"
        "多少の誤差は測定器の精度限界や操作ミスによるものと考えられるため、次回は丁寧に行うことが求められる。"
    ),
]

DEMO_CANDIDATES = [
    (
        "量産型AI考察（典型テンプレ）",
        "算出した重力加速度の平均値は9.684 m/s^2となり、理論値9.807 m/s^2と概ね一致し実験の妥当性が確認された。"
        "わずかな誤差はストップウォッチの読み取り誤差やヒューマンエラー、実験環境の影響によるものと考えられる。"
        "今後は測定回数を増やし慎重に操作することで精度向上が期待される。",
    ),
    (
        "脱・量産型 Sランク考察（外挿＋小角近似破綻＋オーダー評価）",
        "算出した4点の重力加速度gは9.558〜9.782 m/s^2と全点が理論値9.80665 m/s^2より負側に偏っており、ランダム誤差ではなく明確な系統誤差が支配的である。"
        "実際、見かけのgは振れ角二乗θ0²に対して強い負の線形相関（r=-0.999、R²=0.999）を示す。これは理論式が前提とする小角近似 sinθ≈θ が大振幅域で破綻し、楕円積分の一次補正項 T = T0(1 + θ0²/16) によって周期が過大・見かけのgが過小評価されるためである。"
        "そこでθ0²→0へ線形外挿するとg = 9.807 m/s^2（理論値比+0.00%〜0.01%）となり、単純平均の相対誤差（-1.25%）の99%以上が小角近似の破綻で定量的に説明できる。"
        "残存する微小誤差については糸の質量無視や空気抵抗の寄与が考えられるため、次回は振れ角5°以下で糸長Lを変えた対照実験を行い、系統残差を切り分ける検証が有効である。",
    ),
]


def main() -> None:
    scorer = StemUniquenessScorer(backend=LocalHybridEmbedder())
    print("=== 理系レポート 脱・量産型考察スコアラー デモ（合成データ） ===")
    for label, text in DEMO_CANDIDATES:
        res = scorer.score(DEMO_PROMPT, text, DEMO_AI_COHORT, data_series=DEMO_DATA)
        print(f"\n[{label}]")
        print(
            f"  総合スコア: {res.final_score:.1f}/100 ({res.grade}) | "
            f"適合G={res.relevance_gate:.2f} | 物理矛盾={res.raw_stats['bias_contradiction']}"
        )
        print("  6軸:", res.metrics)
        if res.numeric_audit.extrapolated_at_zero is not None:
            print(
                f"  自動外挿: θ0²→0 で {res.numeric_audit.extrapolated_at_zero:.5g} m/s^2 "
                f"(R²={res.numeric_audit.trend_r2}, 誤差削減 {res.numeric_audit.error_reduction_pct}%)"
            )


if __name__ == "__main__":
    main()
