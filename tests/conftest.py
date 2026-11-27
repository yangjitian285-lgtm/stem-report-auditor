"""pytest 共通設定：プロジェクト直下を import パスに追加し、サンプルファイルの場所を提供する."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "sample_reports"


def sample(name: str) -> Path:
    """sample_reports/ のファイル。個人データのため無い環境ではそのテストをスキップする."""
    p = SAMPLES / name
    if not p.exists():
        pytest.skip(f"{p} がありません（sample_reports は git 管理外）")
    return p


@pytest.fixture(scope="session")
def bench():
    import benchmark

    return benchmark
