"""ローカルアプリ（http.server）の動作確認."""

import json
import threading
import urllib.request

import pytest

from uniqueness_scorer.webapp import make_server


@pytest.fixture()
def server():
    srv = make_server(18765)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_index_and_table_api(server):
    with urllib.request.urlopen(server + "/") as r:
        assert "STEM Report Auditor" in r.read().decode("utf-8")
    text = "V (V)\tI (mA)\tP (mW)\n1.00\t10.2\t10.2\n2.00\t20.5\t41.0\n3.00\t30.1\t90.3\n4.00\t40.8\t136\n5.00\t50.9\t255"
    code, j = _post(server + "/api/table", {"text": text})
    assert code == 200 and j["headers"][0] == "V (V)"
    assert any(f["row"] == 4 and f["severity"] in ("WARNING", "CRITICAL") for f in j["findings"])


def test_bad_inputs_return_readable_errors(server):
    code, j = _post(server + "/api/table", {"text": "a"})
    assert code == 400 and "表" in j["error"]
    code, j = _post(server + "/api/audit", {"filename": "x.exe", "content_b64": "AA=="})
    assert code == 400
    code, j = _post(server + "/api/score", {"task": "t", "draft": "d", "cohort": "x"})
    assert code == 400 and "3つ以上" in j["error"]
