"""STEM Report Auditor をアプリとして起動する（ブラウザが自動で開きます）.

    python app.py              # Windows は「起動.bat」をダブルクリックでもOK
    python app.py --port 9000  # ポートを変える
"""

import argparse

from uniqueness_scorer.webapp import run_app

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="STEM Report Auditor（ローカルアプリ）")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    a = ap.parse_args()
    run_app(port=a.port, open_browser=not a.no_browser)
