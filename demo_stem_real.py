"""実理系レポート2本に対する理系特化モデルのデモ（旧 test_stem_real.py。データは private_data/ から読み込む）.

- Case 1: 結晶構造解析（XRD デバイ・シェラー環、実測格子定数 a vs cos²θ & 純鉄文献値 0.28660 nm）
- Case 2: 学生実験Ⅱ テーマ4（ラジカル共重合 P(MAA-r-BMA)、NMR組成比・GPC分散度・未反応モノマー）
各ケースで：
  1) 量産型AI考察（他学生がAIに丸投げして出す典型）
  2) ユーザーの実提出レポート本文そのまま
  3) 論点ズレ珍文（ゲート検証）
  4) 脱・量産型 Sランク改善考察（データ監査・系統誤差外挿・物理機構・検証実験案）
の4つを並べて数理分離精度と自動提案の実用性を検証し、stem_benchmark_report.html を出力する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from uniqueness_scorer.embeddings import LocalHybridEmbedder
from uniqueness_scorer.mcp_server import handle_jsonrpc_message
from uniqueness_scorer.stem_engine import DataSeriesInput, StemUniquenessScorer
from uniqueness_scorer.visualizer import render_stem_benchmark_html


# 実提出レポート本文を含むため、データは private_data/stem_real_cases.json（git管理外）から読み込む。
_DATA_PATH = Path(__file__).resolve().parent / "private_data" / "stem_real_cases.json"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _load_data() -> Dict[str, Any]:
    if not _DATA_PATH.is_file():
        print(f"[SKIP] {_DATA_PATH} がありません（実レポートは個人データのため同梱していません）。")
        raise SystemExit(0)
    raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    for k, v in list(raw.items()):
        if isinstance(v, dict) and v.get("__dataclass__") == "DataSeriesInput":
            raw[k] = DataSeriesInput(**{kk: vv for kk, vv in v.items() if kk != "__dataclass__"})
    return raw


def run_stem_case(
    title: str,
    prompt: str,
    cohort: List[str],
    tests: List[Dict[str, str]],
    data_series: DataSeriesInput,
) -> Dict[str, Any]:
    scorer = StemUniquenessScorer(backend=LocalHybridEmbedder())
    items: List[Dict[str, Any]] = []

    print(f"\n==================== {title} ====================")
    print(
        f"{'バージョン':36s} | {'総合':>5s} | {'脱量':>4s} | {'反逃':>4s} | {'符号':>4s} | "
        f"{'定量':>4s} | {'機構':>4s} | {'脱再':>4s} | {'適合G':>5s} | 判定"
    )
    print("-" * 116)
    for t in tests:
        res = scorer.score(prompt, t["text"], cohort, data_series=data_series)
        items.append({**t, "result": res})
        m = res.metrics
        print(
            f"{t['name']:36s} | {res.final_score:5.1f} | {m['dehomog_score']:4.0f} | "
            f"{m['anti_stem_slop_score']:4.0f} | {m['bias_trend_consistency']:4.0f} | "
            f"{m['quant_error_rigor']:4.0f} | {m['mechanism_depth']:4.0f} | "
            f"{m['beyond_restatement']:4.0f} | {res.relevance_gate:5.2f} | {res.grade}"
        )

    by_tag = {it["tag"]: it["result"] for it in items}
    print("\n[実提出レポートに対する数値自動監査]")
    for f in by_tag["REAL"].numeric_audit.findings:
        print(f"  * {f}")
    print("[実提出レポートに対する自動ピボット提案]")
    for s in by_tag["REAL"].scientific_suggestions:
        print(f"  * {s}")
    if by_tag["AI"].raw_stats["bias_contradiction"]:
        print("[量産型AI考察の物理的矛盾検知] -> 系統バイアスをランダム誤差のみで説明した矛盾を自動検知 (PASS)")

    passed = (
        by_tag["S_PLUS"].final_score > by_tag["REAL"].final_score + 20.0
        and by_tag["S_PLUS"].final_score > by_tag["AI"].final_score + 35.0
        and by_tag["OFF"].final_score < 5.0
    )
    print(f"[分離検証] S_PLUS - AI = +{by_tag['S_PLUS'].final_score - by_tag['AI'].final_score:.1f}pt | "
          f"S_PLUS - REAL = +{by_tag['S_PLUS'].final_score - by_tag['REAL'].final_score:.1f}pt | "
          f"OFF = {by_tag['OFF'].final_score:.1f}pt -> {'PASS' if passed else 'CHECK'}")
    return {
        "title": title,
        "prompt": prompt,
        "items": items,
        "data_series": data_series,
        "passed": passed,
    }


def main() -> None:
    d = _load_data()
    CRYSTAL_PROMPT, CRYSTAL_AI_COHORT, CRYSTAL_TESTS, CRYSTAL_DATA = (
        d["CRYSTAL_PROMPT"], d["CRYSTAL_AI_COHORT"], d["CRYSTAL_TESTS"], d["CRYSTAL_DATA"]
    )
    POLYMER_PROMPT, POLYMER_AI_COHORT, POLYMER_TESTS, POLYMER_DATA = (
        d["POLYMER_PROMPT"], d["POLYMER_AI_COHORT"], d["POLYMER_TESTS"], d["POLYMER_DATA"]
    )
    s1 = run_stem_case(
        "Case 1：結晶構造解析（XRD デバイ・シェラー環・格子定数）",
        CRYSTAL_PROMPT,
        CRYSTAL_AI_COHORT,
        CRYSTAL_TESTS,
        CRYSTAL_DATA,
    )
    s2 = run_stem_case(
        "Case 2：学生実験Ⅱ テーマ4（ラジカル共重合 P(MAA-r-BMA)）",
        POLYMER_PROMPT,
        POLYMER_AI_COHORT,
        POLYMER_TESTS,
        POLYMER_DATA,
    )

    # MCP score_stem_report 疎通テスト
    resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {
                "name": "score_stem_report",
                "arguments": {
                    "experiment_prompt": CRYSTAL_PROMPT,
                    "discussion_text": CRYSTAL_TESTS[1]["text"],  # 実提出レポート
                    "reference_cohort": CRYSTAL_AI_COHORT,
                    "data_series": {
                        "name": CRYSTAL_DATA.name,
                        "observed": CRYSTAL_DATA.observed,
                        "reference": CRYSTAL_DATA.reference,
                        "unit": CRYSTAL_DATA.unit,
                        "independent_var": CRYSTAL_DATA.independent_var,
                        "independent_name": CRYSTAL_DATA.independent_name,
                        "assumptions": CRYSTAL_DATA.assumptions,
                    },
                    "provider": "local",
                },
            },
        }
    )
    assert resp and not resp["result"]["isError"]
    mcp_payload = json.loads(resp["result"]["content"][0]["text"])
    print("\n==================== MCP score_stem_report 疎通検証 ====================")
    print(
        f"MCP返却OK: score={mcp_payload['final_score']} | grade={mcp_payload['grade']} | "
        f"外挿値={mcp_payload['numeric_audit']['extrapolated_at_zero']} nm "
        f"(誤差削減 {mcp_payload['numeric_audit']['error_reduction_pct']}%)"
    )

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "stem_benchmark_report.html"
    out_path.write_text(render_stem_benchmark_html([s1, s2]), encoding="utf-8")
    print(f"\nHTML可視化レポート出力 -> {out_path.resolve()}")
    print(f"全体検証ステータス: {'ALL PASS' if (s1['passed'] and s2['passed']) else 'NEEDS TUNING'}")


if __name__ == "__main__":
    main()
