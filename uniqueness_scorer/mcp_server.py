"""外部SDK不要・JSON-RPC 2.0 (stdio) 準拠のMCPサーバー実装.

Claude Desktop / Cursor / Antigravity などのAIクライアントから直接以下のツールを呼び出せる：
- score_report_uniqueness: レポート/ESの多角的独自性スコアリング
- diagnose_claims_heatmap: 主張単位の被り率と書き換え対象箇所のピンポイント抽出
- compare_draft_revisions: 改訂前後のスコア・Vendi増分・主張被り率の定量比較
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from .docx_patcher import patch_docx_safe
from .embeddings import create_backend
from .math_engine import ScoringConfig, UniquenessScorer
from .stem_auditor import audit_stem_document
from .stem_engine import DataSeriesInput, StemUniquenessScorer


SERVER_INFO = {
    "name": "de-homogenizer-uniqueness-scorer",
    "version": "0.1.0",
}

PROTOCOL_VERSION = "2024-11-05"

TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "name": "score_report_uniqueness",
        "description": (
            "学生のレポートまたは就活ESの下書きを、仮想・量産型AI母集団（または提供された比較群）と比較し、"
            "Vendi Score増分・正則化マハラノビス距離・最近傍クラスタ脱出度・主張単位被り率・骨格/AI臭・具体性・"
            "課題適合ゲートの多角的数理モデルで0〜100点採点し、脱・量産型の改善提案を返します。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "assignment_prompt": {
                    "type": "string",
                    "description": "レポートの課題文またはES設問",
                },
                "draft": {
                    "type": "string",
                    "description": "評価したい学生の原稿テキスト",
                },
                "reference_cohort": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "比較対象となる量産型AIレポート群（3本以上推奨）。"
                        "省略時は OPENAI_API_KEY / GEMINI_API_KEY 経由で自動生成します。"
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["report", "es"],
                    "default": "report",
                    "description": "大学レポート(report)か就活ES(es)か",
                },
                "provider": {
                    "type": "string",
                    "enum": ["auto", "local", "st", "openai", "gemini"],
                    "default": "auto",
                    "description": "埋め込み・母集団生成バックエンド",
                },
                "rubric_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "評価ルーブリック上の必須キーワード（任意）",
                },
            },
            "required": ["assignment_prompt", "draft"],
        },
    },
    {
        "name": "diagnose_claims_heatmap",
        "description": (
            "原稿を主張（文）単位に分解し、量産型AI母集団との被り率（RED/YELLOW/GREEN）と"
            "最も類似した量産型主張・具体的な差し替え提案だけを簡潔に返します。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "assignment_prompt": {"type": "string"},
                "draft": {"type": "string"},
                "reference_cohort": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "mode": {"type": "string", "enum": ["report", "es"], "default": "report"},
                "provider": {
                    "type": "string",
                    "enum": ["auto", "local", "st", "openai", "gemini"],
                    "default": "auto",
                },
            },
            "required": ["assignment_prompt", "draft"],
        },
    },
    {
        "name": "compare_draft_revisions",
        "description": (
            "改訂前の原稿(before_draft)と改訂後の原稿(after_draft)を同一の量産型AI母集団に対して採点し、"
            "総合スコア・限界Vendi増分・主張被り率・課題適合ゲートの改善幅を定量比較します。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "assignment_prompt": {"type": "string"},
                "before_draft": {"type": "string"},
                "after_draft": {"type": "string"},
                "reference_cohort": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "mode": {"type": "string", "enum": ["report", "es"], "default": "report"},
                "provider": {
                    "type": "string",
                    "enum": ["auto", "local", "st", "openai", "gemini"],
                    "default": "auto",
                },
            },
            "required": ["assignment_prompt", "before_draft", "after_draft"],
        },
    },
    {
        "name": "score_stem_report",
        "description": (
            "理系の計算・実験レポートの考察に特化した脱・量産型スコアラー。"
            "実測値・理論/文献値・独立変数（任意）を自動統計監査し、系統バイアスや線形外挿トレンドと"
            "考察文章の物理的整合性、理系AI逃げフレーズ（読み取り誤差・概ね一致等）、オーダー評価、"
            "物理メカニズム深掘り、結果再述脱却度の6軸で採点して科学的改善案を返します。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "experiment_prompt": {"type": "string", "description": "実験テーマ・考察課題"},
                "discussion_text": {"type": "string", "description": "学生の考察・結論本文"},
                "reference_cohort": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "比較用の量産型AI考察群（3本以上推奨）",
                },
                "data_series": {
                    "type": "object",
                    "description": "任意：実測値と理論値/文献値の数値データ",
                    "properties": {
                        "name": {"type": "string"},
                        "observed": {"type": "array", "items": {"type": "number"}},
                        "reference": {},
                        "unit": {"type": "string"},
                        "independent_var": {"type": "array", "items": {"type": "number"}},
                        "independent_name": {"type": "string"},
                        "assumptions": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "provider": {
                    "type": "string",
                    "enum": ["auto", "local", "st", "openai", "gemini"],
                    "default": "local",
                },
            },
            "required": ["experiment_prompt", "discussion_text"],
        },
    },
    {
        "name": "audit_stem_file",
        "description": (
            "Word(.docx) / PDF / テキストの理系レポートファイルを1ファイルそのまま監査します。"
            "①表内計算の自動検算（sin²θ、格子定数、Mw/Mn、平均値）、"
            "②表間・表と本文の数値/ミラー指数矛盾・有効数字・誤記・テンプレ消し忘れ検知、"
            "③実験データの系統バイアス＆線形外挿（cos²θ→0等）の自動発見、"
            "④理系考察の6軸脱・量産型AIスコアを一括返却します。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "監査するレポートファイルのパス (.docx/.pdf/.txt/.md)"},
                "experiment_prompt": {"type": "string", "description": "任意の実験テーマ・課題文"},
                "reference_cohort": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "任意の比較用AI考察群（省略時は汎用理系量産AIベースラインを使用）",
                },
                "provider": {
                    "type": "string",
                    "enum": ["auto", "local", "st", "openai", "gemini"],
                    "default": "local",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "patch_docx_safe",
        "description": (
            "元のWord(.docx)ファイルを絶対に上書きせず、図表・書式を保持したコピー版(*_revised.docx)に対して"
            "ユーザー承認済みの置換・段落挿入・テンプレ指示文削除を安全適用し、直後に再監査(Before/After比較)を返します。"
            "【学術倫理】挿入する段落は、学生本人が書いた文章か本人が内容を確認・承認したものに限ってください。"
            "実在しないデータ・調査・文献を書き加えてはいけません。挿入段落は既定で黄色ハイライトされ、"
            "提出前に本人が自分の言葉で書き直す対象として残ります。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_path": {"type": "string", "description": "元の.docxファイルパス"},
                "output_path": {"type": "string", "description": "出力先.docxパス（省略時は *_revised.docx）"},
                "replacements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string"},
                            "replacement": {"type": "string"},
                        },
                        "required": ["target", "replacement"],
                    },
                    "description": "誤記修正や語句置換のリスト（分割ランをまたいでも安全に置換）",
                },
                "insertions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "after_text": {"type": "string", "description": "挿入位置の基準となる既存段落内の部分文字列"},
                            "new_paragraph": {"type": "string", "description": "直後に追加する新しい考察段落"},
                        },
                        "required": ["after_text", "new_paragraph"],
                    },
                },
                "delete_containing": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "この文字列を含むテンプレート注意書き段落を削除",
                },
                "auto_reaudit": {"type": "boolean", "default": True},
                "highlight_insertions": {
                    "type": "boolean",
                    "default": True,
                    "description": "挿入段落を黄色ハイライトして、本人の確認・書き直し対象として明示する",
                },
            },
            "required": ["source_path"],
        },
    },
]


# 各ツールの provider 既定値（inputSchema の default と一致させる）
_DEFAULT_PROVIDER = {
    "score_report_uniqueness": "auto",
    "diagnose_claims_heatmap": "auto",
    "compare_draft_revisions": "auto",
    "score_stem_report": "local",
    "audit_stem_file": "local",
}


def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """ツール呼び出しをディスパッチして構造化結果を返す."""
    mode = arguments.get("mode", "report")

    # Word修正ツールは埋め込みバックエンドを使わないので、APIキーの有無に関係なく動かす
    if name == "patch_docx_safe":
        res = patch_docx_safe(
            source_path=arguments["source_path"],
            output_path=arguments.get("output_path"),
            replacements=arguments.get("replacements"),
            insertions=arguments.get("insertions"),
            delete_containing=arguments.get("delete_containing"),
            auto_reaudit=arguments.get("auto_reaudit", True),
            highlight_insertions=arguments.get("highlight_insertions", True),
        )
        return res.to_dict()

    if name not in _DEFAULT_PROVIDER:
        raise ValueError(f"Unknown tool: {name}")
    provider = arguments.get("provider") or _DEFAULT_PROVIDER[name]
    backend = create_backend(provider)
    scorer = UniquenessScorer(backend=backend, config=ScoringConfig(mode=mode))

    if name == "score_report_uniqueness":
        res = scorer.score(
            assignment_prompt=arguments["assignment_prompt"],
            draft=arguments["draft"],
            reference_cohort=arguments.get("reference_cohort"),
            rubric_keywords=arguments.get("rubric_keywords"),
        )
        return res.to_dict()

    if name == "diagnose_claims_heatmap":
        res = scorer.score(
            assignment_prompt=arguments["assignment_prompt"],
            draft=arguments["draft"],
            reference_cohort=arguments.get("reference_cohort"),
        )
        return {
            "final_score": res.final_score,
            "grade": res.grade,
            "relevance_gate": res.relevance_gate,
            "claims": [
                {
                    "index": c.index,
                    "status": c.status,
                    "overlap_pct": round(c.cohort_overlap_rate * 100.0, 1),
                    "text": c.text,
                    "closest_ai_claim": c.matched_example,
                }
                for c in res.claims
            ],
            "slop_hits": res.slop_hits,
            "orthogonal_suggestions": res.orthogonal_suggestions,
            "quality_scores": res.quality_scores,
            "quality_flags": res.quality_flags,
            "verification_needed": res.verification_needed,
        }

    if name == "compare_draft_revisions":
        cohort = arguments.get("reference_cohort")
        if cohort is None or len(cohort) < 3:
            cohort = backend.generate_cohort(arguments["assignment_prompt"], mode=mode)
        r_before = scorer.score(
            assignment_prompt=arguments["assignment_prompt"],
            draft=arguments["before_draft"],
            reference_cohort=cohort,
        )
        r_after = scorer.score(
            assignment_prompt=arguments["assignment_prompt"],
            draft=arguments["after_draft"],
            reference_cohort=cohort,
        )
        return {
            "before": {
                "final_score": r_before.final_score,
                "grade": r_before.grade,
                "relevance_gate": r_before.relevance_gate,
                "delta_vendi": r_before.raw_stats["delta_vendi_score"],
                "mean_claim_overlap_pct": round(
                    r_before.raw_stats["mean_claim_overlap_rate"] * 100.0, 1
                ),
            },
            "after": {
                "final_score": r_after.final_score,
                "grade": r_after.grade,
                "relevance_gate": r_after.relevance_gate,
                "delta_vendi": r_after.raw_stats["delta_vendi_score"],
                "mean_claim_overlap_pct": round(
                    r_after.raw_stats["mean_claim_overlap_rate"] * 100.0, 1
                ),
            },
            "delta": {
                "score_change": round(r_after.final_score - r_before.final_score, 2),
                "vendi_change": round(
                    r_after.raw_stats["delta_vendi_score"]
                    - r_before.raw_stats["delta_vendi_score"],
                    4,
                ),
                "claim_overlap_change_pct": round(
                    (
                        r_after.raw_stats["mean_claim_overlap_rate"]
                        - r_before.raw_stats["mean_claim_overlap_rate"]
                    )
                    * 100.0,
                    1,
                ),
            },
            "remaining_suggestions": r_after.orthogonal_suggestions,
        }

    if name == "score_stem_report":
        cohort = arguments.get("reference_cohort")
        if cohort is None or len(cohort) < 3:
            cohort = backend.generate_cohort(arguments["experiment_prompt"], mode="report")
        ds_raw = arguments.get("data_series")
        ds = None
        if isinstance(ds_raw, dict) and ds_raw.get("observed"):
            ds = DataSeriesInput(
                name=ds_raw.get("name", "測定量"),
                observed=[float(x) for x in ds_raw["observed"]],
                reference=ds_raw.get("reference", 0.0),
                unit=ds_raw.get("unit", ""),
                independent_var=(
                    [float(x) for x in ds_raw["independent_var"]]
                    if ds_raw.get("independent_var")
                    else None
                ),
                independent_name=ds_raw.get("independent_name", ""),
                assumptions=list(ds_raw.get("assumptions") or []),
            )
        stem_scorer = StemUniquenessScorer(backend=backend)
        res = stem_scorer.score(
            experiment_prompt=arguments["experiment_prompt"],
            discussion_text=arguments["discussion_text"],
            reference_cohort=cohort,
            data_series=ds,
        )
        return res.to_dict()

    if name == "audit_stem_file":
        report = audit_stem_document(
            file_path=arguments["file_path"],
            experiment_prompt=arguments.get("experiment_prompt"),
            reference_cohort=arguments.get("reference_cohort"),
            backend=backend,
        )
        return report.to_dict()

    raise ValueError(f"Unknown tool: {name}")


def handle_jsonrpc_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """単一のJSON-RPC 2.0リクエストを処理する."""
    method = msg.get("method")
    req_id = msg.get("id")
    params = msg.get("params") or {}

    # Notification（idなし）は応答不要
    if req_id is None:
        return None

    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                },
            }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS_SPEC}}
        if method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments") or {}
            payload = execute_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(payload, ensure_ascii=False, indent=2),
                        }
                    ],
                    "isError": False,
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    except Exception as exc:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            },
        }


def run_stdio_server() -> None:
    """標準入出力（行区切りJSON-RPC）でMCPサーバーを起動する."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            sys.stdout.write(json.dumps(err, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue

        resp = handle_jsonrpc_message(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
