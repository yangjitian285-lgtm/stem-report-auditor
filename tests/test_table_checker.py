"""汎用表検算（分野を問わない列間の関係推定）の回帰テスト."""

import math
import statistics

from uniqueness_scorer.mcp_server import execute_tool
from uniqueness_scorer.table_checker import check_table, check_table_text, parse_cell


def _issues(res, min_sev=("WARNING", "CRITICAL")):
    return [f for f in res.findings if f.severity in min_sev]


def test_parse_cell_rounding_width():
    c = parse_cell("0.2019")
    assert abs(c.value - 0.2019) < 1e-12 and abs(c.half_ulp - 0.00005) < 1e-12
    c = parse_cell("3.73×10^4")
    assert abs(c.value - 37300) < 1e-6 and abs(c.half_ulp - 50) < 1e-6
    assert parse_cell("2-3") is None
    assert parse_cell("試料A") is None


def test_ohm_table_wrong_row_detected():
    V = [1.00, 2.00, 3.00, 4.00, 5.00]
    I = [10.2, 20.5, 30.1, 40.8, 50.9]
    rows = [[f"{v:.2f}", f"{c:.1f}", f"{v / c * 1000:.1f}"] for v, c in zip(V, I)]
    rows[4][2] = "89.2"  # 正しくは 98.2
    res = check_table(["V (V)", "I (mA)", "R (Ω)"], rows)
    assert any(r.target == "R (Ω)" and "V" in r.formula and "I" in r.formula for r in res.relations)
    bad = _issues(res)
    assert bad and all(f.row == 5 for f in bad)


def test_clean_tables_have_no_warnings():
    # 振り子・吸光度・温度換算・誤差率（すべて正しい表）
    L = [0.200, 0.300, 0.400, 0.500, 0.600]
    T = [0.898, 1.102, 1.269, 1.421, 1.556]
    rows = []
    for l, t in zip(L, T):
        t2 = f"{t * t:.3f}"
        rows.append([f"{l:.3f}", f"{t:.3f}", t2, f"{4 * math.pi ** 2 * l / float(t2):.2f}"])
    assert not _issues(check_table(["L (m)", "T (s)", "T² (s²)", "g (m/s²)"], rows))

    rows = [[f"{c:.2f}", f"{tp:.1f}", f"{-math.log10(tp / 100):.3f}"] for c, tp in [(0.1, 70.8), (0.2, 49.2), (0.3, 35.4), (0.4, 24.9), (0.5, 17.3)]]
    assert not _issues(check_table(["c (mmol/L)", "T (%)", "A"], rows))

    rows = [[f"{t:.1f}", f"{t + 273.15:.2f}"] for t in (20.0, 35.5, 50.1, 80.0, 12.3)]
    assert not _issues(check_table(["t (℃)", "T (K)"], rows))


def test_offset_and_percent_error_relations():
    rows = [[f"{t:.1f}", f"{t + 273.15:.2f}"] for t in (20.0, 35.5, 50.1, 80.0, 12.3)]
    rows[2][1] = "323.35"  # 正しくは 323.25
    res = check_table(["t (℃)", "T (K)"], rows)
    assert any("273.15" in r.formula for r in res.relations)
    assert any(f.row == 3 for f in _issues(res))

    vals = [9.71, 9.95, 9.62, 9.88, 9.75]
    rows = [[f"{v:.2f}", f"{abs(v - 9.80) / 9.80 * 100:.2f}"] for v in vals]
    rows[3][1] = "0.28"
    res = check_table(["g (m/s²)", "誤差率 (%)"], rows)
    assert any("9.8" in r.formula for r in res.relations)
    assert any(f.row == 4 for f in _issues(res))


def test_mean_column_and_summary_rows():
    data = [[12.31, 12.52, 12.40], [15.11, 15.02, 15.30], [9.81, 9.92, 10.10], [20.21, 20.40, 20.02], [17.5, 17.62, 17.41]]
    rows = [[s] + [f"{x:.2f}" for x in tr] + [f"{sum(tr) / 3:.2f}"] for s, tr in zip("ABCDE", data)]
    rows[2][4] = "10.21"  # 正しくは 9.94
    res = check_table(["試料", "1回目", "2回目", "3回目", "平均"], rows)
    assert any(f.row == 3 and f.column == "平均" for f in _issues(res))

    vals = [9.812, 9.734, 9.901, 9.655, 9.777]
    rows = [[str(i + 1), f"{v:.3f}"] for i, v in enumerate(vals)]
    rows.append(["平均", f"{statistics.mean(vals):.3f}"])
    rows.append(["標準偏差", f"{statistics.pstdev(vals) * 0.8:.3f}"])  # 誤った式
    res = check_table(["回", "長さ (mm)"], rows)
    kinds = {(f.kind, f.row) for f in _issues(res)}
    assert ("SUMMARY_ROW", 7) in kinds and ("SUMMARY_ROW", 6) not in kinds


def test_systematic_angle_mistakes():
    rows = [[str(x), f"{math.sin(x):.4f}"] for x in (10, 20, 30, 40, 50)]
    res = check_table(["θ (°)", "sin θ"], rows)
    assert any(f.kind == "DEG_RAD" for f in res.findings)

    rows = [[f"{x:.1f}", f"{math.sin(math.radians(x)) ** 2:.4f}"] for x in (38.4, 44.6, 64.9, 78.0, 82.2)]
    res = check_table(["2θ (°)", "sin²θ"], rows)
    assert any(f.kind == "TWO_THETA" for f in res.findings)


def test_unit_conversion_slip():
    rows = [["12.5", "0.125"], ["33.1", "0.331"], ["7.8", "0.078"], ["51.0", "0.510"]]
    res = check_table(["長さ (mm)", "長さ (m)"], rows)
    assert any(f.kind == "UNIT" for f in res.findings)


def test_last_digit_slip_in_hand_computed_table_but_not_excel_rounding():
    # 表示値から手計算した表：1行だけ最終桁が2ずれている
    pairs = [(0.174, 0.118), (0.342, 0.232), (0.500, 0.334), (0.643, 0.432), (0.766, 0.517)]
    rows = [[f"{a:.3f}", f"{b:.3f}", f"{a / b:.3f}"] for a, b in pairs]
    rows[0][2] = "1.473"  # 表示値からは 1.475
    res = check_table(["sin i", "sin r", "n"], rows)
    assert any(f.row == 1 for f in _issues(res))

    # 最終桁が1つだけずれる（丸め前の値で計算した表では正常）なら指摘しない
    rows[0][2] = "1.474"
    res = check_table(["sin i", "sin r", "n"], rows)
    assert not any(f.row == 1 for f in _issues(res))


def test_measured_only_table_has_no_relation():
    rows = [["A", "12.3", "7.1"], ["B", "15.8", "6.2"], ["C", "9.4", "8.8"], ["D", "11.1", "5.0"]]
    res = check_table(["試料", "質量 (g)", "時間 (s)"], rows)
    assert not res.relations and not _issues(res)


def test_text_formats_and_mcp_tool():
    md = """| V (V) | I (mA) | P (mW) |
|---|---|---|
| 1.00 | 10.2 | 10.2 |
| 2.00 | 20.5 | 41.0 |
| 3.00 | 30.1 | 90.3 |
| 4.00 | 40.8 | 136 |
| 5.00 | 50.9 | 255 |"""
    res = check_table_text(md)
    assert any(f.row == 4 for f in _issues(res))
    csv_text = "V (V),I (mA),P (mW)\n1.00,10.2,10.2\n2.00,20.5,41.0\n3.00,30.1,90.3\n4.00,40.8,136\n5.00,50.9,255\n"
    out = execute_tool("check_table_generic", {"table_text": csv_text})
    assert any(f["row"] == 4 for f in out["findings"] if f["severity"] in ("WARNING", "CRITICAL"))
