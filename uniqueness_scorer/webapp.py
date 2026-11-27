"""ブラウザで使うローカルアプリ（追加インストール不要・外部送信なし）.

    python app.py            # 既定ブラウザが開く（http://127.0.0.1:8765）

Python 標準ライブラリの http.server だけで動く。待ち受けは 127.0.0.1 のみで、
アップロードされたファイルは一時フォルダで処理して、終わったら消す。
"""

from __future__ import annotations

import base64
import json
import os
import socket
import tempfile
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

UI_PATH = Path(__file__).with_name("webapp_ui.html")
MAX_BODY = 40 * 1024 * 1024  # 40MB
ALLOWED_SUFFIX = {".docx", ".pdf", ".txt", ".md"}


# ---------------------------------------------------------------------------
# 各機能（JSON を受けて JSON を返す）
# ---------------------------------------------------------------------------
def api_audit(payload: Dict[str, Any]) -> Dict[str, Any]:
    from .stem_auditor import audit_stem_document
    from .visualizer import render_full_audit_html

    name = Path(str(payload.get("filename") or "report.docx")).name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIX:
        raise ValueError(f"対応しているのは {', '.join(sorted(ALLOWED_SUFFIX))} です（{suffix or '拡張子なし'} は読めません）。")
    data = base64.b64decode(payload.get("content_b64") or "")
    if not data:
        raise ValueError("ファイルが空です。")
    with tempfile.TemporaryDirectory(prefix="stem_audit_") as td:
        path = Path(td) / ("upload" + suffix)
        path.write_bytes(data)
        rep = audit_stem_document(str(path), experiment_prompt=(payload.get("task") or None))
        html = render_full_audit_html(rep)
    findings = [
        {"severity": f.severity, "category": f.category, "location": f.location, "message": f.message, "suggestion": f.suggestion}
        for f in rep.findings
    ]
    return {
        "filename": name,
        "overall_score": rep.overall_score,
        "overall_grade": rep.overall_grade,
        "deterministic_score": rep.deterministic_score,
        "discussion_score": rep.discussion_score,
        "checks_run": rep.checks_run,
        "tables": rep.extracted_tables_count,
        "is_stem_report": rep.is_stem_report,
        "detected_profiles": rep.detected_profiles,
        "discussion_note": rep.discussion_note,
        "unanswered": rep.unanswered,
        "findings": findings,
        "checklist": rep.priority_checklist,
        "report_html": html.replace(str(path), name),
    }


def api_table(payload: Dict[str, Any]) -> Dict[str, Any]:
    from .table_checker import check_table, parse_table_text

    text = str(payload.get("text") or "")
    headers, rows = parse_table_text(text)
    if not headers or len(rows) < 2:
        raise ValueError("表を読み取れませんでした。1行目に見出し、2行目以降にデータを貼り付けてください（Excel からコピーしてそのまま貼れます）。")
    res = check_table(headers, rows, payload.get("caption") or "表")
    out = res.to_dict()
    out["headers"] = headers
    out["rows"] = rows
    return out


def api_score(payload: Dict[str, Any]) -> Dict[str, Any]:
    from .embeddings import create_backend
    from .math_engine import ScoringConfig, UniquenessScorer
    from .visualizer import render_single_report_html

    task = str(payload.get("task") or "").strip()
    draft = str(payload.get("draft") or "").strip()
    cohort_text = str(payload.get("cohort") or "")
    cohort = [c.strip() for c in cohort_text.replace("\r\n", "\n").split("\n---\n") if c.strip()]
    if len(cohort) < 3:
        # 「---」が無ければ空行2つ以上で区切る
        import re

        cohort = [c.strip() for c in re.split(r"\n\s*\n\s*\n", cohort_text) if c.strip()]
    if not task or not draft:
        raise ValueError("課題文と原稿の両方を入れてください。")
    provider = str(payload.get("provider") or "local")
    if provider not in {"local", "st"}:
        provider = "local"  # アプリからは無料・ローカルのものだけ
    if len(cohort) < 3:
        raise ValueError("比較用のAI回答を3つ以上入れてください（回答と回答の間に「---」だけの行を入れます）。")
    scorer = UniquenessScorer(backend=create_backend(provider), config=ScoringConfig(mode=str(payload.get("mode") or "report")))
    res = scorer.score(assignment_prompt=task, draft=draft, reference_cohort=cohort)
    return {
        "final_score": res.final_score,
        "uniqueness_score": res.uniqueness_score,
        "grade": res.grade,
        "relevance_gate": res.relevance_gate,
        "metrics": res.metrics,
        "claims": [{"index": c.index, "text": c.text, "status": c.status, "overlap": c.cohort_overlap_rate} for c in res.claims],
        "quality_scores": res.quality_scores,
        "quality_flags": res.quality_flags,
        "verification_needed": res.verification_needed,
        "suggestions": res.orthogonal_suggestions,
        "cohort_size": len(cohort),
        "report_html": render_single_report_html(title="提出原稿", assignment_prompt=task, result=res),
    }


ROUTES = {"/api/audit": api_audit, "/api/table": api_table, "/api/score": api_score}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    server_version = "STEMReportAuditor/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # 端末を静かに保つ
        pass

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in {"127.0.0.1", "localhost"}

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._send(403, b"forbidden", "text/plain")
        if self.path in ("/", "/index.html"):
            return self._send(200, UI_PATH.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/ping":
            return self._send(200, b'{"ok":true}', "application/json")
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._send(403, b"forbidden", "text/plain")
        fn = ROUTES.get(self.path)
        if fn is None:
            return self._send(404, b"not found", "text/plain")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._json(413, {"error": "ファイルが大きすぎます（40MBまで）。"})
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = fn(payload)
            self._json(200, result)
        except ValueError as e:
            self._json(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"error": f"処理中にエラーが起きました：{type(e).__name__}: {e}"})

    def _json(self, code: int, obj: Dict[str, Any]) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), "application/json; charset=utf-8")


def _free_port(preferred: int) -> int:
    for port in [preferred] + list(range(preferred + 1, preferred + 20)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 0


def make_server(port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", _free_port(port)), _Handler)


def run_app(port: int = 8765, open_browser: bool = True) -> None:
    srv = make_server(port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print("=" * 56)
    print(" STEM Report Auditor を起動しました")
    print(f"  ブラウザで開く: {url}")
    print("  終了するには、このウィンドウを閉じるか Ctrl+C を押してください。")
    print("  ※ ファイルはこのパソコンの中だけで処理し、外部には送信しません。")
    print("=" * 56)
    if open_browser and not os.environ.get("UNIQ_NO_BROWSER"):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
