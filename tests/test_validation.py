"""検証スクリプトの統計関数の単体テスト."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "validate", Path(__file__).resolve().parent.parent / "validation" / "validate.py"
)
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)


def test_spearman_basic():
    assert abs(validate.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(validate.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9


def test_spearman_ties():
    r = validate.spearman([1, 1, 2, 3], [1, 2, 3, 4])
    assert 0.9 < r < 1.0


def test_dev_split_runs():
    files = sorted((Path(validate.__file__).parent / "dev").glob("*.json"))
    out = validate.evaluate(files, "local", "final_score")
    assert out["n_items"] >= 8
