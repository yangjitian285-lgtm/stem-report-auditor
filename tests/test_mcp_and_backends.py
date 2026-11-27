"""MCPサーバーとバックエンド選択の回帰テスト."""

import json
import sys
import types

import numpy as np

from uniqueness_scorer import embeddings
from uniqueness_scorer.mcp_server import TOOLS_SPEC, handle_jsonrpc_message


def test_placeholder_key_falls_back_to_local(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "YOUR_KEY_OR_OMIT_FOR_LOCAL")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(embeddings.create_backend("auto"), embeddings.LocalHybridEmbedder)


def test_mcp_initialize_and_list():
    r = handle_jsonrpc_message({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert "protocolVersion" in r["result"]
    r = handle_jsonrpc_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in r["result"]["tools"]}
    assert {"audit_stem_file", "patch_docx_safe", "score_report_uniqueness"} <= names


def test_patch_tool_works_without_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-invalid-but-present")
    r = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "patch_docx_safe", "arguments": {"source_path": str(tmp_path / "none.docx")}},
        }
    )
    # バックエンド生成で落ちるのではなく、ファイル不在のエラーになること
    assert r["result"]["isError"]
    assert "not found" in r["result"]["content"][0]["text"].lower()


def test_schema_defaults_match_code():
    from uniqueness_scorer.mcp_server import _DEFAULT_PROVIDER

    for tool in TOOLS_SPEC:
        prov = tool["inputSchema"]["properties"].get("provider")
        if prov and "default" in prov:
            assert _DEFAULT_PROVIDER[tool["name"]] == prov["default"], tool["name"]


def test_sentence_transformer_backend_with_stub(monkeypatch):
    """sentence-transformers が無い環境でも、スタブでインターフェースを検証する."""

    class FakeST:
        def __init__(self, name, device=None):
            self.name = name

        def encode(self, inputs, **kw):
            assert all(i.startswith("query: ") for i in inputs)
            rng = np.random.default_rng(0)
            return rng.normal(size=(len(inputs), 8))

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    be = embeddings.create_backend("st")
    v = be.embed(["あ", "い"])
    assert v.shape == (2, 8)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
