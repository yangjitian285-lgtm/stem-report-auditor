"""理系レポート監査（決定論的チェック）の回帰テスト."""

from conftest import sample
from uniqueness_scorer.stem_auditor import _half_ulp, _nearest_hkl, _sig_digits, audit_stem_document


def _cats(rep, sev=None):
    return [(f.category, f.message) for f in rep.findings if sev is None or f.severity == sev]


def test_blank_assignment_is_unanswered():
    rep = audit_stem_document(str(sample("nov4_report.docx")))
    assert rep.unanswered
    assert rep.overall_grade.startswith("未回答")
    assert rep.overall_score is None


def test_non_stem_report_gets_no_stem_advice():
    rep = audit_stem_document(str(sample("mot_report.docx")))
    assert not rep.is_stem_report
    assert rep.stem_discussion_result is None
    assert not any("誤差伝播" in c for c in rep.priority_checklist)
    # 設問番号の重複（（１）が2回）を検出
    assert any(f.category == "NUMBERING" for f in rep.findings)


def test_crystal_real_errors_detected():
    rep = audit_stem_document(str(sample("crystal_report.docx")))
    msgs = " ".join(m for _, m in _cats(rep))
    assert "(100)" in msgs and "消滅則" in msgs
    assert "回折" in msgs
    assert "hkl" in msgs  # 日本語に隣接した hlk も検出（\b バグの修正）
    assert "面間隔" in msgs
    assert any(f.category == "SIG_FIG" and "有効数字" in f.message for f in rep.findings)


def test_extrapolation_reports_uncertainty():
    rep = audit_stem_document(str(sample("crystal_report.docx")))
    na = rep.auto_discovered_audits[0]
    assert na.intercept_ci95 is not None and na.intercept_se is not None
    assert na.reference_within_ci95 is True  # 0.28660 は95%信頼区間内 → 残差は有意でない
    assert "有意ではありません" in na.findings[-1]


def test_polymer_nonnegative_rate_not_flagged_as_bias():
    rep = audit_stem_document(str(sample("polymer_theme4.txt")))
    assert not any(
        f.category == "NUMERIC_DISCOVERY" and "未反応" in f.location for f in rep.findings
    )
    assert rep.deterministic_score is not None and rep.deterministic_score > 0  # 0点に張り付かない
    assert "ラジカル重合・共重合" in rep.detected_profiles


def test_rounding_helpers():
    assert abs(_half_ulp("0.2019") - 0.00005) < 1e-12
    assert abs(_half_ulp("3.73*10^4") - 50.0) < 1e-9
    assert _sig_digits("0.28558") == 5 and _sig_digits("0.2019") == 4
    assert _nearest_hkl("(100)", {"(110)", "(200)", "(211)"}) == "(110)"
