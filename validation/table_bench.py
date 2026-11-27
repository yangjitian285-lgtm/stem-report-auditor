"""汎用表検算のベンチマーク：誤りを仕込んだ理系実験の表で再現率・誤検知率を測る.

    python validation/table_bench.py --generate      # 表と正解キーを作る（乱数シード固定）
    python validation/table_bench.py                 # ツールで採点

出力
- validation/table_bench/tables.md   … 表だけ（正解は書かない。LLM に見せる用）
- validation/table_bench/key.json    … 正解キー（どの表のどの行・どの種類の誤りか）
- validation/table_bench/tool_result.json … ツールの採点結果

注意：表のテンプレート（実験の種類と計算式）はツールの作者が作っている。
ツールが知らない形の式（例：f = ab/(a+b)）も意図的に含めているが、
実レポートの分布とは違うので、ここでの数値は「上限に近い目安」として扱うこと。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from uniqueness_scorer.table_checker import check_table  # noqa: E402

OUT = Path(__file__).resolve().parent / "table_bench"
DEV_SEED = 1001  # 開発（チューニング）用
TEST_SEED = 20261127  # 2回目の開発用（LLM との比較に使った。誤検出の分析に使ったので最終評価には使わない）
FINAL_SEED = 777  # 3回目の開発用（初回 88.6%。ここで見つかったバグ2件を直したので最終評価には使わない）
HOLDOUT_SEED = 9090  # 最終評価用（すべての修正の後に一度だけ実行。以後この結果でチューニングしない）
EXCEL_SEED = 4242  # Excel風（丸め前の値から計算）の誤りなしの表：誤検知率の確認用
EXCEL_ERR_SEED = 5151  # Excel風の表に誤りを入れたもの


EXCEL_STYLE = False  # True: 計算列を「丸める前の値」から計算する（Excel で作った表の再現）


class Num(str):
    """表示文字列と、丸める前の値を両方もつ."""

    def __new__(cls, text: str, precise: float):
        obj = super().__new__(cls, text)
        obj.precise = precise
        return obj


def val(x) -> float:
    if EXCEL_STYLE and isinstance(x, Num):
        return x.precise
    return float(x)


def fmt(v: float, d: int) -> str:
    return Num(f"{v:.{d}f}", v)


def sig(v: float, n: int) -> str:
    """有効数字 n 桁で表記（小数表記）."""
    if v == 0:
        return Num("0", 0.0)
    d = max(0, n - 1 - int(math.floor(math.log10(abs(v)))))
    return Num(f"{v:.{d}f}", v)


# ---------------------------------------------------------------------------
# 実験テンプレート：(説明, 見出し, 行リスト, 計算列の列番号, 入力列の列番号)
# 計算列は、学生と同じく「表に書いた丸め後の値」から計算して丸める。
# ---------------------------------------------------------------------------
def t_ohm(r: random.Random):
    R0 = r.uniform(47, 220)
    rows = []
    for v in [1, 2, 3, 4, 5, 6][: r.randint(4, 6)]:
        V = fmt(v + r.gauss(0, 0.01), 2)
        I = fmt(val(V) / R0 * 1000 * (1 + r.gauss(0, 0.01)), 2)
        Rv = sig(val(V) / val(I) * 1000, 3)
        P = sig(val(V) * val(I), 3)
        rows.append([V, I, Rv, P])
    return "オームの法則：電圧と電流を測定し抵抗と消費電力を求めた", ["V (V)", "I (mA)", "R (Ω)", "P (mW)"], rows, [2, 3], [0, 1]


def t_pendulum(r):
    g = 9.80
    rows = []
    for L in [0.200, 0.300, 0.400, 0.500, 0.600, 0.700][: r.randint(4, 6)]:
        T = fmt(2 * math.pi * math.sqrt(L / g) * (1 + r.gauss(0, 0.004)), 3)
        T2 = fmt(val(T) ** 2, 3)
        gv = fmt(4 * math.pi ** 2 * L / val(T2), 2)
        rows.append([fmt(L, 3), T, T2, gv])
    return "単振り子：糸の長さと周期から重力加速度を求めた", ["L (m)", "T (s)", "T² (s²)", "g (m/s²)"], rows, [2, 3], [0, 1]


def t_density(r):
    rho = r.choice([2.70, 7.87, 8.96, 1.18])
    rows = []
    for _ in range(r.randint(4, 6)):
        Vv = fmt(r.uniform(2, 15), 2)
        m = fmt(val(Vv) * rho * (1 + r.gauss(0, 0.01)), 2)
        rows.append([m, Vv, fmt(val(m) / val(Vv), 2)])
    return "密度測定：試料の質量と体積から密度を求めた", ["m (g)", "V (cm³)", "ρ (g/cm³)"], rows, [2], [0, 1]


def t_spring(r):
    k = r.uniform(10, 40)
    rows = []
    for m in [50, 100, 150, 200, 250, 300][: r.randint(4, 6)]:
        F = fmt(m * 9.80 / 1000, 3)
        x = fmt(val(F) / k * 100 * (1 + r.gauss(0, 0.01)), 2)
        kv = fmt(val(F) / (val(x) / 100), 1)
        rows.append([str(m), F, x, kv])
    return "ばね定数：おもりの質量 m から力 F = mg（g = 9.80）を求め、伸び x からばね定数を求めた", ["m (g)", "F (N)", "x (cm)", "k (N/m)"], rows, [1, 3], [0, 2]


def t_temp(r):
    rows = []
    for _ in range(r.randint(4, 6)):
        t = fmt(r.uniform(5, 95), 1)
        rows.append([t, fmt(val(t) + 273.15, 2)])
    return "温度の換算：セルシウス温度を熱力学温度に換算した", ["t (℃)", "T (K)"], rows, [1], [0]


def t_absorb(r):
    eps = r.uniform(0.8, 1.6)
    rows = []
    for c in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60][: r.randint(4, 6)]:
        A_true = eps * c * (1 + r.gauss(0, 0.01))
        Tp = fmt(100 * 10 ** (-A_true), 1)
        A = fmt(-math.log10(val(Tp) / 100), 3)
        rows.append([fmt(c, 2), Tp, A])
    return "吸光度：透過率 T(%) から吸光度 A = −log₁₀(T/100) を求めた", ["c (mmol/L)", "透過率 T (%)", "吸光度 A"], rows, [2], [1]


def t_titration(r):
    rows = []
    for s in ["A", "B", "C", "D", "E"][: r.randint(4, 5)]:
        base = r.uniform(8, 22)
        tr = [round(base + r.gauss(0, 0.08), 2) for _ in range(3)]
        rows.append([s] + [fmt(x, 2) for x in tr] + [fmt(sum(tr) / 3, 2)])
    return "中和滴定：各試料を3回滴定し、滴下量の平均を求めた", ["試料", "1回目 (mL)", "2回目 (mL)", "3回目 (mL)", "平均 (mL)"], rows, [4], [1, 2, 3]


def t_freefall(r):
    rows = []
    for h in [0.500, 0.800, 1.000, 1.200, 1.500, 1.800][: r.randint(4, 6)]:
        t = fmt(math.sqrt(2 * h / 9.80) * (1 + r.gauss(0, 0.004)), 3)
        rows.append([fmt(h, 3), t, fmt(2 * h / val(t) ** 2, 2)])
    return "自由落下：落下距離 h と落下時間 t から g = 2h/t² を求めた", ["h (m)", "t (s)", "g (m/s²)"], rows, [2], [0, 1]


def t_xrd(r):
    lam = 0.15406
    a = r.choice([0.2866, 0.3615, 0.4078, 0.4050])
    hkls = [2, 4, 6, 8, 10, 12] if a == 0.2866 else [3, 4, 8, 11, 12, 16]
    rows = []
    for n2 in hkls[: r.randint(4, 6)]:
        s2 = (lam / (2 * a)) ** 2 * n2
        if s2 >= 0.98:
            continue
        tt = fmt(2 * math.degrees(math.asin(math.sqrt(s2))), 2)
        s2v = fmt(math.sin(math.radians(val(tt) / 2)) ** 2, 4)
        d = fmt(lam / (2 * math.sin(math.radians(val(tt) / 2))), 4)
        rows.append([tt, s2v, d])
    return "X線回折：回折角 2θ から sin²θ と面間隔 d（λ = 0.15406 nm）を求めた", ["2θ (°)", "sin²θ", "d (nm)"], rows, [1, 2], [0]


def t_refraction(r):
    n = r.uniform(1.45, 1.55)
    rows = []
    for i in [10, 20, 30, 40, 50, 60][: r.randint(4, 6)]:
        rr = fmt(math.degrees(math.asin(math.sin(math.radians(i)) / n)) + r.gauss(0, 0.2), 1)
        si = fmt(math.sin(math.radians(i)), 3)
        sr = fmt(math.sin(math.radians(val(rr))), 3)
        rows.append([str(i), rr, si, sr, fmt(val(si) / val(sr), 3)])
    return "屈折率：入射角 i と屈折角 r から n = sin i / sin r を求めた", ["i (°)", "r (°)", "sin i", "sin r", "n"], rows, [2, 3, 4], [0, 1]


def t_kinetic(r):
    rows = []
    for _ in range(r.randint(4, 6)):
        m = fmt(r.uniform(0.1, 2.0), 3)
        v = fmt(r.uniform(0.5, 5.0), 2)
        rows.append([m, v, sig(0.5 * val(m) * val(v) ** 2, 3)])
    return "運動エネルギー：質量と速さから K = mv²/2 を求めた", ["m (kg)", "v (m/s)", "K (J)"], rows, [2], [0, 1]


def t_percent_err(r):
    rows = []
    for k in range(r.randint(4, 6)):
        g = fmt(9.80 * (1 + r.gauss(0, 0.012)), 2)
        rows.append([str(k + 1), g, fmt(abs(val(g) - 9.80) / 9.80 * 100, 2)])
    return "重力加速度の測定値と文献値 9.80 m/s² との誤差率", ["回", "g (m/s²)", "誤差率 (%)"], rows, [2], [1]


def t_heat(r):
    rows = []
    for _ in range(r.randint(4, 6)):
        m = fmt(r.uniform(50, 200), 1)
        dT = fmt(r.uniform(2, 15), 1)
        rows.append([m, dT, sig(val(m) * 4.18 * val(dT), 4)])
    return "熱量：水の質量と温度上昇から Q = mcΔT（c = 4.18 J/(g·K)）を求めた", ["m (g)", "ΔT (K)", "Q (J)"], rows, [2], [0, 1]


def t_lens(r):
    f = r.uniform(8, 15)
    rows = []
    for a in [20.0, 25.0, 30.0, 35.0, 40.0, 50.0][: r.randint(4, 6)]:
        b = fmt(1 / (1 / f - 1 / a) * (1 + r.gauss(0, 0.005)), 1)
        fv = fmt(a * val(b) / (a + val(b)), 2)
        rows.append([fmt(a, 1), b, fv])
    return "凸レンズ：物体距離 a と像距離 b から焦点距離 f = ab/(a+b) を求めた", ["a (cm)", "b (cm)", "f (cm)"], rows, [2], [0, 1]


def t_unit(r):
    rows = []
    for _ in range(r.randint(4, 6)):
        d = fmt(r.uniform(0.1, 0.5), 4)
        rows.append([d, fmt(val(d) * 10, 3)])
    return "面間隔の単位換算（nm → Å）", ["d (nm)", "d (Å)"], rows, [1], [0]


def t_summary(r):
    vals = [round(r.uniform(9.6, 10.0), 3) for _ in range(r.randint(5, 7))]
    rows = [[str(i + 1), fmt(v, 3)] for i, v in enumerate(vals)]
    rows.append(["平均", fmt(statistics.mean(vals), 3)])
    rows.append(["標準偏差", fmt(statistics.stdev(vals), 3)])
    return "繰り返し測定：同じ試料の長さを繰り返し測り、平均と標準偏差（n−1）を求めた", ["回", "長さ (mm)"], rows, [1], [1]


def t_yield(r):
    theo = round(r.uniform(2.0, 5.0), 3)
    rows = []
    for s in ["1", "2", "3", "4", "5"][: r.randint(4, 5)]:
        m = fmt(theo * r.uniform(0.4, 0.95), 3)
        rows.append([s, m, fmt(val(m) / theo * 100, 1)])
    return f"合成の収率：理論収量 {theo} g に対する収率", ["実験", "収量 (g)", "収率 (%)"], rows, [2], [1]


TEMPLATES: List[Callable] = [t_ohm, t_pendulum, t_density, t_spring, t_temp, t_absorb, t_titration, t_freefall,
                             t_xrd, t_refraction, t_kinetic, t_percent_err, t_heat, t_lens, t_unit, t_summary, t_yield]


# ---------------------------------------------------------------------------
# 誤りの注入
# ---------------------------------------------------------------------------
def _swap_digits(s: str, r: random.Random) -> Optional[str]:
    idx = [i for i in range(len(s) - 1) if s[i].isdigit() and s[i + 1].isdigit() and s[i] != s[i + 1]]
    if not idx:
        return None
    i = r.choice(idx)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def _bump_digit(s: str, r: random.Random, amount: int) -> str:
    """最後から2番目の桁あたりを amount だけずらす（計算ミス相当）."""
    d = len(s.split(".")[1]) if "." in s else 0
    v = float(s) + amount * (10 ** (-d)) * r.choice([1, -1])
    return f"{v:.{d}f}"


def inject(name: str, headers, rows, calc_cols, input_cols, r: random.Random):
    """1つの誤りを入れて (rows, key) を返す。key は正解情報."""
    kinds = ["calc", "transpose", "input", "unit10", "lastdigit"]
    if name == "t_summary":
        kinds = ["summary_mean", "summary_sd"]
    if name == "t_refraction":
        kinds = kinds + ["deg_rad"]
    if name == "t_xrd":
        kinds = kinds + ["two_theta"]
    kind = r.choice(kinds)
    rows = [list(x) for x in rows]
    if kind == "summary_mean":
        ri = len(rows) - 2
        rows[ri][1] = _bump_digit(rows[ri][1], r, 7)
        return rows, {"kind": kind, "row": ri + 1, "col": headers[1]}
    if kind == "summary_sd":
        ri = len(rows) - 1
        vals = [float(x[1]) for x in rows[:-2]]
        rows[ri][1] = fmt(statistics.pstdev(vals) * 0.8, 3)  # 誤った式
        return rows, {"kind": kind, "row": ri + 1, "col": headers[1]}
    if kind == "deg_rad":
        for x in rows:
            x[2] = fmt(math.sin(float(x[0])), 3)
            x[3] = fmt(math.sin(float(x[1])), 3)
            x[4] = fmt(float(x[2]) / float(x[3]), 3) if float(x[3]) != 0 else "—"
        return rows, {"kind": kind, "row": None, "col": headers[2]}
    if kind == "two_theta":
        for x in rows:
            x[1] = fmt(math.sin(math.radians(float(x[0]))) ** 2, 4)
        return rows, {"kind": kind, "row": None, "col": headers[1]}
    ri = r.randrange(len(rows))
    if kind == "input":
        ci = r.choice(input_cols)
        new = _swap_digits(rows[ri][ci], r) or _bump_digit(rows[ri][ci], r, 3)
    else:
        ci = r.choice(calc_cols)
        old = rows[ri][ci]
        if kind == "calc":
            new = _bump_digit(old, r, r.choice([3, 4, 6, 9]))
        elif kind == "transpose":
            new = _swap_digits(old, r) or _bump_digit(old, r, 5)
        elif kind == "unit10":
            d = len(old.split(".")[1]) if "." in old else 0
            new = f"{float(old) * r.choice([10, 0.1]):.{d}f}"
        else:  # lastdigit: 最終桁が2つずれる（丸め以上のずれ）
            new = _bump_digit(old, r, 2)
    if new == rows[ri][ci]:
        new = _bump_digit(rows[ri][ci], r, 5)
    rows[ri][ci] = new
    return rows, {"kind": kind, "row": ri + 1, "col": headers[ci]}


def generate(n_tables: int = 68, seed: int = TEST_SEED, error_rate: float = 0.65) -> None:
    r = random.Random(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    tables, key = [], []
    for i in range(n_tables):
        tmpl = TEMPLATES[i % len(TEMPLATES)]
        desc, headers, rows, calc_cols, input_cols = tmpl(r)
        if r.random() < error_rate:
            rows, k = inject(tmpl.__name__, headers, rows, calc_cols, input_cols, r)
        else:
            k = {"kind": "clean", "row": None, "col": None}
        k.update({"id": i + 1, "template": tmpl.__name__})
        tables.append({"id": i + 1, "desc": desc, "headers": headers, "rows": rows})
        key.append(k)
    # 表の並びを乱す（テンプレートの順番から正解を推測できないように）
    order = list(range(n_tables))
    r.shuffle(order)
    tables = [dict(tables[j], id=n + 1) for n, j in enumerate(order)]
    key = [dict(key[j], id=n + 1) for n, j in enumerate(order)]
    md = ["# 表の検算ベンチマーク（誤りを含む表と含まない表が混在）\n",
          "各表について、計算・転記・換算・集計の誤りがあれば「表番号・行番号・列」を、なければ「誤りなし」と答えてください。"
          "行番号は見出しを除いたデータ行の番号（1始まり）です。\n"]
    for t in tables:
        md.append(f"\n## 表{t['id']}  {t['desc']}\n")
        md.append("| " + " | ".join(t["headers"]) + " |")
        md.append("|" + "---|" * len(t["headers"]))
        for row in t["rows"]:
            md.append("| " + " | ".join(row) + " |")
    (OUT / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "tables.json").write_text(json.dumps(tables, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{n_tables} 表を書き出しました: {OUT}")


# ---------------------------------------------------------------------------
# 採点
# ---------------------------------------------------------------------------
def score_predictions(preds: Dict[int, List[Dict]], key: List[Dict], ids: Optional[List[int]] = None) -> Dict:
    """preds: 表id -> [{'row': int|None, 'col': str|None}, ...]（WARNING 以上の指摘だけ）."""
    tp = fn = fp_tables = tn = 0
    wrong_loc = 0
    per_kind: Dict[str, List[int]] = {}
    rows_out = []
    for k in key:
        if ids is not None and k["id"] not in ids:
            continue
        p = preds.get(k["id"], [])
        if k["kind"] == "clean":
            if p:
                fp_tables += 1
            else:
                tn += 1
            rows_out.append((k["id"], k["kind"], "FP" if p else "TN"))
            continue
        if k["row"] is None:
            # 列全体の誤り：列単位の指摘があれば検出とみなす
            hit = any(x.get("row") is None for x in p)
        else:
            hit = any(x.get("row") == k["row"] for x in p)
        extra = [x for x in p if k["row"] is not None and x.get("row") not in (None, k["row"])]
        if extra:
            wrong_loc += 1
        per_kind.setdefault(k["kind"], [0, 0])
        per_kind[k["kind"]][1] += 1
        if hit:
            tp += 1
            per_kind[k["kind"]][0] += 1
        else:
            fn += 1
        rows_out.append((k["id"], k["kind"], "TP" if hit else "FN"))
    n_err = tp + fn
    n_clean = tn + fp_tables
    return {
        "errors": n_err,
        "clean": n_clean,
        "recall": round(tp / n_err, 3) if n_err else None,
        "false_alarm_rate_clean_tables": round(fp_tables / n_clean, 3) if n_clean else None,
        "tables_with_extra_wrong_row_flags": wrong_loc,
        "per_kind": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_kind.items())},
        "detail": rows_out,
    }


def run_tool(min_severity: str = "WARNING") -> Dict[int, List[Dict]]:
    tables = json.loads((OUT / "tables.json").read_text(encoding="utf-8"))
    order = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
    preds: Dict[int, List[Dict]] = {}
    for t in tables:
        res = check_table(t["headers"], t["rows"], f"表{t['id']}")
        preds[t["id"]] = [
            {"row": f.row, "col": f.column, "severity": f.severity, "kind": f.kind, "message": f.message}
            for f in res.findings
            if f.severity in order and order[f.severity] >= order[min_severity]
        ]
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--split", default="holdout", choices=["dev", "test", "final", "holdout", "excel", "excel_err"])
    ap.add_argument("--details", action="store_true", help="表ごとの判定も表示")
    ap.add_argument("--min-severity", default="WARNING", choices=["INFO", "WARNING", "CRITICAL"])
    ap.add_argument("--llm", default=None, help="LLM の回答 JSON（表id -> 指摘リスト）を同じ基準で採点")
    a = ap.parse_args()
    global OUT, EXCEL_STYLE
    OUT = OUT.parent / ("table_bench" if a.split == "test" else f"table_bench_{a.split}")
    if a.generate or not (OUT / "key.json").exists():
        if a.split == "excel":
            EXCEL_STYLE = True
            generate(seed=EXCEL_SEED, error_rate=0.0)
        elif a.split == "excel_err":
            EXCEL_STYLE = True
            generate(seed=EXCEL_ERR_SEED)
        else:
            generate(seed={"test": TEST_SEED, "dev": DEV_SEED, "final": FINAL_SEED, "holdout": HOLDOUT_SEED}[a.split])
    key = json.loads((OUT / "key.json").read_text(encoding="utf-8"))
    preds = run_tool(a.min_severity)
    res = score_predictions(preds, key)
    (OUT / "tool_result.json").write_text(json.dumps({"summary": res, "predictions": preds}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ツール:", json.dumps({k: v for k, v in res.items() if k != "detail"}, ensure_ascii=False))
    if a.details:
        for tid, kind, verdict in res["detail"]:
            if verdict in ("FN", "FP"):
                k = key[tid - 1]
                print(f"  表{tid} {k['template']} {kind} 正解行={k['row']} 列={k['col']} → {verdict}; 指摘={[(x['row'], x['column'] if 'column' in x else x['col'], x['severity']) for x in preds[tid]]}")
    if a.llm:
        llm = {int(k): v for k, v in json.loads(Path(a.llm).read_text(encoding="utf-8")).items()}
        ids = sorted(llm)
        print("ツール（同じ表のみ）:", json.dumps({k: v for k, v in score_predictions(preds, key, ids).items() if k != "detail"}, ensure_ascii=False))
        print("LLM:", json.dumps({k: v for k, v in score_predictions(llm, key, ids).items() if k != "detail"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
