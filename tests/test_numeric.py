"""数値データ監査（外挿の不確かさ・系統バイアス判定）の単体テスト."""

import math

from uniqueness_scorer.stem_engine import DataSeriesInput, audit_numeric_data, linear_fit_with_uncertainty

import numpy as np


def test_linear_fit_matches_known_values():
    x = np.array([0.8038, 0.6079, 0.4131, 0.2187])
    y = np.array([0.28558, 0.28571, 0.28600, 0.28623])
    fit = linear_fit_with_uncertainty(x, y)
    assert math.isclose(fit["intercept"], 0.286467, abs_tol=2e-6)
    assert math.isclose(fit["intercept_se"], 6.38e-5, rel_tol=0.02)


def test_significant_residual_is_reported():
    # 切片が基準値から大きく外れる合成データ
    x = [0.8, 0.6, 0.4, 0.2, 0.1]
    y = [1.00 - 0.10 * v + 0.0005 * ((-1) ** i) for i, v in enumerate(x)]
    na = audit_numeric_data(DataSeriesInput(name="q", observed=y, reference=1.05, independent_var=x))
    assert na.reference_within_ci95 is False


def test_zero_reference_nonnegative_skips_bias():
    na = audit_numeric_data(DataSeriesInput(name="未反応率", observed=[10.7, 12.7, 12.3, 15.7], reference=0.0))
    assert not na.has_strong_systematic_bias
    assert na.bias_check_skipped_reason
