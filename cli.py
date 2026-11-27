"""「脱・量産型AI」独自性スコアラー CLI エントリポイント."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from uniqueness_scorer.embeddings import create_backend
from uniqueness_scorer.math_engine import ScoringConfig, UniquenessScorer
from uniqueness_scorer.mcp_server import run_stdio_server
from uniqueness_scorer.stem_auditor import audit_stem_document
from uniqueness_scorer.stem_engine import DataSeriesInput, StemUniquenessScorer
from uniqueness_scorer.visualizer import render_full_audit_html, render_single_report_html


def _read_text(path_or_text: str) -> str:
    p = Path(path_or_text)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    return path_or_text


def _load_cohort(cohort_path: Optional[str]) -> Optional[List[str]]:
    if not cohort_path:
        return None
    p = Path(cohort_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError("cohort JSON must be a list of strings.")


def cmd_score(args: argparse.Namespace) -> None:
    task = _read_text(args.task)
    draft = _read_text(args.draft)
    cohort = _load_cohort(args.cohort)
    backend = create_backend(args.provider)
    scorer = UniquenessScorer(backend=backend, config=ScoringConfig(mode=args.mode))
    res = scorer.score(
        assignment_prompt=task,
        draft=draft,
        reference_cohort=cohort,
        n_synthetic=args.n_samples,
    )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(res.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.html_out:
        html_str = render_single_report_html(
            title=args.title or "提出原稿",
            assignment_prompt=task,
            result=res,
        )
        Path(args.html_out).write_text(html_str, encoding="utf-8")

    print(f"=== 「脱・量産型AI」独自性診断 ({args.mode}) ===")
    print(f"総合独自性スコア : {res.final_score:.1f} / 100  [{res.grade}]")
    print(
        f"純粋独自性       : {res.raw_originality:.1f}  |  課題適合ゲート G: {res.relevance_gate:.2f}"
    )
    print(
        f"ΔVendi Score     : +{res.raw_stats['delta_vendi_score']:.4f}  |  "
        f"Mahalanobis Z: {res.raw_stats['mahalanobis_z_score']:+.2f}σ  |  "
        f"主張平均被り率: {res.raw_stats['mean_claim_overlap_rate']*100:.1f}%"
    )
    print("\n--- 6軸サブスコア ---")
    for k, v in res.metrics.items():
        print(f"  {k:26s}: {v:5.1f}")
    print("\n--- 主張ヒートマップ ---")
    for c in res.claims:
        print(
            f"  [#{c.index:02d} {c.status:6s} 被り{c.cohort_overlap_rate*100:4.0f}%] {c.text[:60]}"
        )
    print("\n--- 品質（独自性とは別軸） ---")
    for k, v in res.quality_scores.items():
        print(f"  {k:26s}: {v}")
    for f in res.quality_flags:
        print(f"  ! {f}")
    if res.verification_needed:
        print("\n--- 根拠の確認が必要な記述（ツールは真偽を判定できません） ---")
        for v in res.verification_needed:
            print(f"  ? {v['text'][:60]}  … {v['reason']}")
    print("\n--- 脱・量産型ピボット提案 ---")
    for s in res.orthogonal_suggestions:
        print(f"  * {s}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="「脱・量産型AI」独自性スコアラー CLI / MCP Server"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="原稿の独自性をスコアリング")
    p_score.add_argument("--task", required=True, help="課題文のファイルパスまたは文字列")
    p_score.add_argument("--draft", required=True, help="原稿のファイルパスまたは文字列")
    p_score.add_argument("--cohort", default=None, help="比較用AI母集団JSONファイル（任意）")
    p_score.add_argument("--mode", choices=["report", "es"], default="report")
    p_score.add_argument(
        "--provider", choices=["auto", "local", "st", "openai", "gemini"], default="auto"
    )
    p_score.add_argument("--n-samples", type=int, default=12)
    p_score.add_argument("--title", default="")
    p_score.add_argument("--json-out", default=None)
    p_score.add_argument("--html-out", default=None)
    p_score.set_defaults(func=cmd_score)

    p_mcp = sub.add_parser("mcp", help="MCPサーバー(stdio JSON-RPC)を起動")
    p_mcp.set_defaults(func=lambda _args: run_stdio_server())

    def cmd_stem(a: argparse.Namespace) -> None:
        task = _read_text(a.task)
        draft = _read_text(a.draft)
        cohort = _load_cohort(a.cohort)
        backend = create_backend(a.provider)
        if cohort is None:
            cohort = backend.generate_cohort(task, n_samples=a.n_samples, mode="report")
        ds = None
        if a.data:
            raw = json.loads(Path(a.data).read_text(encoding="utf-8"))
            ds = DataSeriesInput(**raw)
        scorer = StemUniquenessScorer(backend=backend)
        res = scorer.score(task, draft, cohort, data_series=ds)
        if a.json_out:
            Path(a.json_out).write_text(
                json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(f"=== 理系計算・実験レポート 脱量産型診断 ===")
        print(f"総合スコア : {res.final_score:.1f} / 100  [{res.grade}]  (適合G={res.relevance_gate:.2f})")
        for k, v in res.metrics.items():
            print(f"  {k:24s}: {v:5.1f}")
        if res.numeric_audit.has_data:
            print("\n--- 数値データ自動監査 ---")
            for f in res.numeric_audit.findings:
                print(f"  * {f}")
        print("\n--- 科学的ピボット提案 ---")
        for s in res.scientific_suggestions:
            print(f"  * {s}")

    p_stem = sub.add_parser("stem", help="理系計算・実験レポートの考察を診断")
    p_stem.add_argument("--task", required=True)
    p_stem.add_argument("--draft", required=True)
    p_stem.add_argument("--cohort", default=None)
    p_stem.add_argument("--data", default=None, help="実測/理論データJSON（任意）")
    p_stem.add_argument("--provider", choices=["auto", "local", "st", "openai", "gemini"], default="auto")
    p_stem.add_argument("--n-samples", type=int, default=10)
    p_stem.add_argument("--json-out", default=None)
    p_stem.set_defaults(func=cmd_stem)

    def cmd_audit(a: argparse.Namespace) -> None:
        backend = create_backend(a.provider)
        cohort = _load_cohort(a.cohort)
        rep = audit_stem_document(
            file_path=a.file,
            experiment_prompt=a.task,
            reference_cohort=cohort,
            backend=backend,
        )
        if a.json_out:
            Path(a.json_out).write_text(
                json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if a.html_out:
            Path(a.html_out).write_text(render_full_audit_html(rep), encoding="utf-8")

        def _f(v):
            return "N/A" if v is None else f"{v:.1f}"

        print(f"=== 理系レポート 1ファイル完全監査: {rep.source_path} ===")
        print(f"総合監査スコア : {_f(rep.overall_score)} / 100  [{rep.overall_grade}]")
        disc_grade = rep.stem_discussion_result.grade if rep.stem_discussion_result else "対象外"
        print(
            f"  決定論的品質 : {_f(rep.deterministic_score)}（実行した検査 {rep.checks_run} 件）  |  "
            f"理系考察スコア : {_f(rep.discussion_score)} ({disc_grade})  |  "
            f"抽出表数 : {rep.extracted_tables_count}"
        )
        print(
            f"  分野判定 : {'理系実験' if rep.is_stem_report else '非理系/自由記述'}（シグナル {rep.stem_signal:.2f}）"
            + (f"  |  プロファイル : {', '.join(rep.detected_profiles)}" if rep.detected_profiles else "")
        )
        if rep.discussion_note:
            print(f"  ※ {rep.discussion_note}")
        print("\n--- 決定論的監査ログ ---")
        for f in rep.findings:
            print(f"  [{f.severity:8s}] ({f.category} / {f.location}) {f.message}")
        print("\n--- 提出前 優先修正チェックリスト ---")
        for idx, item in enumerate(rep.priority_checklist, 1):
            print(f"  {idx}. {item}")

    p_audit = sub.add_parser("audit", help="Word/PDF理系レポートを1ファイル丸投げで完全監査")
    p_audit.add_argument("file", help="監査するレポートファイル (.docx / .pdf / .txt / .md)")
    p_audit.add_argument("--task", default=None, help="任意の実験テーマ・課題文")
    p_audit.add_argument("--cohort", default=None, help="任意の比較用AI考察JSON")
    p_audit.add_argument("--provider", choices=["auto", "local", "st", "openai", "gemini"], default="local")
    p_audit.add_argument("--json-out", default=None)
    p_audit.add_argument("--html-out", default=None)
    p_audit.set_defaults(func=cmd_audit)

    def cmd_table(a: argparse.Namespace) -> None:
        from uniqueness_scorer.table_checker import check_table_text

        text = Path(a.file).read_text(encoding=a.encoding)
        res = check_table_text(text, caption=Path(a.file).name)
        if a.json_out:
            Path(a.json_out).write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"=== 汎用表検算: {a.file}（数値列 {len(res.numeric_columns)} / データ行 {res.data_rows}） ===")
        if res.relations:
            print("推定した計算関係:")
            for r in res.relations:
                print(f"  {r.target} = {r.formula}   [{r.matched_rows}/{r.checked_rows} 行一致]")
        else:
            print("列どうしの計算関係は推定できませんでした（測定値だけの表か、行数が少なすぎます）。")
        issues = [f for f in res.findings if f.severity != "PASS"]
        print(f"指摘 {len(issues)} 件:")
        for f in issues:
            where = (f"行{f.row} " if f.row else "") + (f.column or "")
            print(f"  [{f.severity:8s}] {where}: {f.message}")
            if f.suggestion:
                print(f"             → {f.suggestion}")

    p_table = sub.add_parser("table", help="CSV/TSV/Markdownの表を分野を問わず汎用検算")
    p_table.add_argument("file", help="表ファイル（1行目が見出し）")
    p_table.add_argument("--encoding", default="utf-8")
    p_table.add_argument("--json-out", default=None)
    p_table.set_defaults(func=cmd_table)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
