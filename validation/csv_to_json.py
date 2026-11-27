"""rating_template.csv 形式（task_id,item_id,human_score,rater,text）を validate.py 用のJSONへ変換する.

使い方: python validation/csv_to_json.py ratings.csv --task-file task01.txt --task-id task01 --out validation/holdout/task01.json
"""

import argparse
import csv
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--task-file", required=True, help="課題文のテキストファイル")
ap.add_argument("--task-id", required=True)
ap.add_argument("--mode", default="report", choices=["report", "es"])
ap.add_argument("--domain", default="")
ap.add_argument("--cohort", default=None, help="比較用AI回答のJSON配列ファイル（任意）")
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = [r for r in csv.DictReader(open(a.csv, encoding="utf-8-sig")) if r["task_id"] == a.task_id]
data = {
    "task": Path(a.task_file).read_text(encoding="utf-8").strip(),
    "mode": a.mode,
    "domain": a.domain,
    "items": [
        {"id": r["item_id"], "text": r["text"], "human_score": float(r["human_score"]), "rater": r.get("rater", "")}
        for r in rows
    ],
}
if a.cohort:
    data["cohort"] = json.loads(Path(a.cohort).read_text(encoding="utf-8"))
Path(a.out).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(rows)} 件を {a.out} に書き出しました")
