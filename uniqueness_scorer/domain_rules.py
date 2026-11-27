"""分野別ルール（ドメインプロファイル）と利用者独自ルールの読み込み.

本体の監査・採点ロジックは分野に依存しない汎用処理だけを持ち、
「X線回折の格子定数」「ラジカル共重合」など特定の実験に固有の用語・前提・試料名規則は
ここに分離する。新しい実験テーマはプロファイルを1つ追加するだけで対応できる。

利用者が自分のレポートで繰り返しやりがちな誤字などは、
カレントディレクトリまたはパッケージ直下の ``user_rules.json`` に書けば読み込まれる（任意）::

    {
      "typos": [["ペンタドロップ", "「ペンダントドロップ」の誤記", "WARN"]],
      "template_residue": ["実験テキストの内容をもとに"]
    }
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class DomainProfile:
    """特定の実験テーマに固有の語彙・前提・試料名規則."""

    key: str
    label: str
    # 文書がこのテーマであると判定するための語（いずれか min_hits 個以上）
    detect_terms: List[str]
    min_hits: int = 2
    # 考察の「物理メカニズム深掘り」として加点する専門語（汎用語に追加）
    mechanism_terms: List[str] = field(default_factory=list)
    # 定量・誤差解析として加点する専門手法名
    quant_terms: List[str] = field(default_factory=list)
    # 自動発見したデータ系列に付与する理論前提（考察で触れているかを評価）
    default_assumptions: List[str] = field(default_factory=list)
    # 試料名の正規表現（表間の数値矛盾・組成比反転の検知に使用）
    sample_name_pattern: Optional[str] = None


# 汎用：どの分野でも使う試料名パターン（例: MAA40:BMA60, PS50:PMMA50）
GENERIC_COPOLYMER_SAMPLE = r"[A-Za-z]{2,6}\s*\d+\s*[:：]\s*[A-Za-z]{2,6}\s*\d+"

PROFILES: List[DomainProfile] = [
    DomainProfile(
        key="xrd_lattice",
        label="X線回折・格子定数",
        detect_terms=["2θ", "回折", "格子定数", "ミラー指数", "デバイ", "シェラー", "Bragg", "ブラッグ", "消滅則", "sin^2θ", "sin²θ"],
        min_hits=2,
        mechanism_terms=["固溶", "格子歪み", "転位", "結晶子", "吸収", "試料位置", "ゼロ点", "偏心", "残留応力", "置換型", "侵入型"],
        quant_terms=["Bradley", "Nelson", "Riley", "Scherrer", "Williamson", "Hall", "cos²θ", "cos^2θ"],
        default_assumptions=["試料位置ずれなし", "X線吸収無視", "単相"],
    ),
    DomainProfile(
        key="radical_copolymer",
        label="ラジカル重合・共重合",
        detect_terms=["共重合", "ラジカル", "Mn", "Mw", "分散度", "GPC", "モノマー", "開始剤", "再沈殿", "NMR"],
        min_hits=3,
        mechanism_terms=["停止反応", "不均化", "再結合", "連鎖移動", "反応性比", "共沸", "ゲル効果", "会合", "解離", "ミセル", "静電反発", "組成ドリフト"],
        quant_terms=["Mayo", "Lewis", "Fineman", "Ross", "Kelen", "Tüdős"],
        default_assumptions=["完全転化", "再沈殿による完全分離"],
        sample_name_pattern=GENERIC_COPOLYMER_SAMPLE,
    ),
]


def detect_profiles(text: str) -> List[DomainProfile]:
    """本文から当てはまるドメインプロファイルを返す（複数可）."""
    hits: List[DomainProfile] = []
    for prof in PROFILES:
        n = sum(1 for t in prof.detect_terms if t in text)
        if n >= prof.min_hits:
            hits.append(prof)
    return hits


def merged_terms(profiles: Sequence[DomainProfile], attr: str) -> List[str]:
    out: List[str] = []
    for p in profiles:
        for t in getattr(p, attr):
            if t not in out:
                out.append(t)
    return out


def sample_patterns(profiles: Sequence[DomainProfile]) -> List[str]:
    pats = [p.sample_name_pattern for p in profiles if p.sample_name_pattern]
    return pats or [GENERIC_COPOLYMER_SAMPLE]


# ---------------------------------------------------------------------------
# 分野判定（理系実験レポートか、文系・自由記述か）
# ---------------------------------------------------------------------------
_STEM_SIGNALS = [
    r"\d+(?:\.\d+)?\s*(?:nm|mm|μm|cm|mol|g/mol|mL|ml|℃|K|Pa|kPa|MPa|V|mA|A|W|J|Hz|mN/m|ppm|°)",
    r"実験|測定|試料|理論値|文献値|誤差|算出|計算|測定値|装置|試薬|サンプル",
    r"表\s*[\d１-９]|図\s*[\d１-９]",
    r"[=＝]\s*[-\d.]+",
    r"sin|cos|log|ln|√|∝|±",
]


def stem_signal_score(text: str, n_tables: int = 0) -> float:
    """理系実験レポートらしさ（0〜1）を返す."""
    if not text:
        return 0.0
    scale = max(len(text) / 800.0, 1.0)
    hits = sum(len(re.findall(p, text)) for p in _STEM_SIGNALS)
    density = hits / scale
    score = 1.0 - pow(2.718281828, -0.12 * density)
    if n_tables >= 1:
        score = min(1.0, score + 0.25)
    return float(score)


def is_stem_document(text: str, n_tables: int = 0, threshold: float = 0.55) -> bool:
    return stem_signal_score(text, n_tables) >= threshold


# ---------------------------------------------------------------------------
# 利用者独自ルール
# ---------------------------------------------------------------------------
def load_user_rules(extra_path: Optional[str] = None) -> Dict[str, List]:
    """user_rules.json（任意）を読み込む。見つからなければ空のルールを返す."""
    candidates: List[Path] = []
    if extra_path:
        candidates.append(Path(extra_path))
    env = os.environ.get("UNIQ_USER_RULES")
    if env:
        candidates.append(Path(env))
    candidates.append(Path.cwd() / "user_rules.json")
    candidates.append(Path(__file__).resolve().parent.parent / "user_rules.json")
    for c in candidates:
        try:
            if c.is_file():
                data = json.loads(c.read_text(encoding="utf-8"))
                typos: List[Tuple[str, str, str]] = [
                    (str(t[0]), str(t[1]), str(t[2]) if len(t) > 2 else "WARN")
                    for t in data.get("typos", [])
                    if isinstance(t, (list, tuple)) and len(t) >= 2
                ]
                residue = [str(r) for r in data.get("template_residue", [])]
                return {"typos": typos, "template_residue": residue, "source": [str(c)]}
        except (OSError, ValueError):
            continue
    return {"typos": [], "template_residue": [], "source": []}
