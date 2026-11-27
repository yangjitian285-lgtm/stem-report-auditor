"""スコアハック耐性の回帰テスト.

レビューで見つかった弱点（表層マーカー挿入・文の並べ替え・捏造データ・白紙・別分野）を
固定ケースとして残し、改修で再発しないことを確認する。
"""

import random

import pytest

from uniqueness_scorer.embeddings import LocalHybridEmbedder
from uniqueness_scorer.math_engine import ScoringConfig, UniquenessScorer

MARKER_SENTENCE = "しかし、この通説は前提が疑わしい。2024年の「調査」では例外もある。一方で、"
FABRICATED = "私が2025年に5社12名へヒアリングしたところ、37%が「若手の仕事が減った」と答えた。"


@pytest.fixture(scope="module")
def scorer():
    return UniquenessScorer(LocalHybridEmbedder(), ScoringConfig(mode="report"))


def _score(scorer, bench, text):
    return scorer.score(bench.TASK1_PROMPT, text, bench.TASK1_COHORT)


def test_marker_injection_gain_is_limited(scorer, bench):
    """中身のないマーカー文を1つ足しても、独自性は大きく上がらず判定も変わらない."""
    a = bench.TASK1_TESTS[0]["text"]
    base = _score(scorer, bench, a)
    hacked = _score(scorer, bench, a.replace("以上のことから、", MARKER_SENTENCE))
    assert hacked.uniqueness_score - base.uniqueness_score < 15.0
    assert hacked.grade == base.grade
    # 具体性の軸はマーカーの個数ではなく文の割合で数えるため、満点近くにはならない
    assert hacked.metrics["specificity_score"] < 60.0


def test_shuffled_sentences_lower_coherence(scorer, bench):
    """文をランダムに並べ替えると、品質の『論理のつながり』が明確に下がる."""
    d = bench.TASK1_TESTS[3]["text"]
    sents = [s + "。" for s in d.split("。") if s]
    random.seed(0)
    random.shuffle(sents)
    orig = _score(scorer, bench, d)
    shuf = _score(scorer, bench, "".join(sents))
    assert shuf.quality_scores["coherence"] < orig.quality_scores["coherence"] - 20
    assert any("接続詞" in f or "つながり" in f for f in shuf.quality_flags)


def test_fabricated_survey_is_flagged_for_verification(scorer, bench):
    """ツールは捏造を見抜けないため、一次調査の主張は必ず『要確認』に列挙する."""
    res = _score(scorer, bench, bench.TASK1_TESTS[0]["text"] + FABRICATED)
    texts = [v["text"] for v in res.verification_needed]
    assert any("ヒアリング" in t for t in texts)
    assert any("捏造" in s for s in res.orthogonal_suggestions) or res.verification_needed


def test_offtopic_is_gated(scorer, bench):
    res = _score(scorer, bench, bench.TASK1_TESTS[2]["text"])
    assert res.relevance_gate < 0.45
    assert res.grade.startswith("GATE_FAIL")


def test_low_gate_is_not_labelled_as_copy(bench):
    """課題適合が低いだけの原稿を『量産型AIと丸被り』と表示しない（独自性で判定する）."""
    res = UniquenessScorer(LocalHybridEmbedder(), ScoringConfig(mode="report")).score(
        bench.TASK1_PROMPT, bench.TASK1_TESTS[3]["text"], bench.TASK1_COHORT
    )
    # ゲートを人工的に下げても grade は raw_originality 基準
    assert res.grade == UniquenessScorer._grade_from_score(res.uniqueness_score, max(res.relevance_gate, 0.5))


def test_benchmark_separation_still_holds(scorer, bench):
    """既存ベンチマーク（開発用データ）の分離が崩れていないこと."""
    by = {t["group_key"]: _score(scorer, bench, t["text"]) for t in bench.TASK1_TESTS}
    assert by["D"].final_score - by["A"].final_score >= 25
    assert by["D"].final_score - by["B"].final_score >= 20
    assert by["C"].relevance_gate < 0.45


def test_lexicon_can_be_disabled(bench):
    """概念辞書を切っても動作する（辞書依存度を測るためのスイッチ）."""
    s = UniquenessScorer(LocalHybridEmbedder(use_concept_lexicon=False), ScoringConfig(mode="report"))
    res = s.score(bench.TASK1_PROMPT, bench.TASK1_TESTS[3]["text"], bench.TASK1_COHORT)
    assert 0.0 <= res.final_score <= 100.0
