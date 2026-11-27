"""ホールドアウト検証：ツールのスコアと人間の評価の順位相関（Spearman）を測る.

使い方::

    python validation/validate.py                 # validation/holdout/*.json を評価
    python validation/validate.py --split dev     # 開発用データ（チューニングに使ってよい）
    python validation/validate.py --provider st   # 埋め込みを変えて比較

データ形式（1課題 = 1ファイル, JSON）::

    {
      "task": "課題文",
      "mode": "report" | "es",
      "domain": "経済" など（集計用ラベル・任意）,
      "cohort": ["比較用の量産型AI回答", ...],        # 3本以上。省略時は --provider の LLM で生成
      "items": [
        {"id": "s01", "text": "学生の回答", "human_score": 72, "rater": "TA-A"},
        ...
      ]
    }

ルール（README「検証」節も参照）
- holdout/ のデータは、辞書・正規表現・重み・しきい値を決めるときに **見てはいけない**。
  見た時点でそのファイルは dev/ に移す。
- human_score は教員・TA・友人など、ツールの作者以外が付けた点数（0〜100、順位が分かれば尺度は自由）。
- 1課題あたり最低10本、複数分野をそろえると相関の信頼区間が狭くなる。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from uniqueness_scorer.embeddings import create_backend  # noqa: E402
from uniqueness_scorer.math_engine import ScoringConfig, UniquenessScorer  # noqa: E402


def _rank(values: Sequence[float]) -> List[float]:
    """同順位は平均順位にする."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def bootstrap_ci(x: Sequence[float], y: Sequence[float], n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(x)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        r = spearman([x[i] for i in idx], [y[i] for i in idx])
        if r == r:  # NaN除外
            vals.append(r)
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def evaluate(files: Sequence[Path], provider: str, metric: str) -> Dict[str, object]:
    backend = create_backend(provider)
    all_tool: List[float] = []
    all_human: List[float] = []
    per_task: List[Dict[str, object]] = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        scorer = UniquenessScorer(backend=backend, config=ScoringConfig(mode=data.get("mode", "report")))
        tool_scores: List[float] = []
        human: List[float] = []
        for it in data["items"]:
            res = scorer.score(data["task"], it["text"], reference_cohort=data.get("cohort"))
            tool_scores.append(float(getattr(res, metric)))
            human.append(float(it["human_score"]))
        rho = spearman(tool_scores, human)
        per_task.append({"file": f.name, "domain": data.get("domain", ""), "n": len(human), "spearman": rho})
        # 課題ごとに順位化してから全体プール（課題間の点数尺度の違いを消す）
        all_tool.extend(_rank(tool_scores))
        all_human.extend(_rank(human))
    rho_all = spearman(all_tool, all_human)
    lo, hi = bootstrap_ci(all_tool, all_human)
    return {"per_task": per_task, "pooled_spearman": rho_all, "ci95": [lo, hi], "n_items": len(all_human)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["holdout", "dev"], default="holdout")
    ap.add_argument("--provider", default="local")
    ap.add_argument(
        "--metric", default="final_score", choices=["final_score", "uniqueness_score", "raw_originality"],
        help="人間評価と比較するツール側の値",
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    files = sorted((Path(__file__).resolve().parent / args.split).glob("*.json"))
    if not files:
        print(f"validation/{args.split}/ に評価データ（*.json）がありません。docstring の形式で追加してください。")
        sys.exit(0)
    out = evaluate(files, args.provider, args.metric)
    print(f"=== {args.split} 検証（provider={args.provider}, metric={args.metric}） ===")
    for t in out["per_task"]:
        print(f"  {t['file']:36s} {t['domain']:10s} n={t['n']:3d}  Spearman ρ = {t['spearman']:+.3f}")
    lo, hi = out["ci95"]
    print(f"全体（課題内順位をプール）: ρ = {out['pooled_spearman']:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  n={out['n_items']}")
    if args.split == "dev":
        print("※ dev はチューニングに使ったデータです。性能の主張には holdout の結果を使ってください。")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
