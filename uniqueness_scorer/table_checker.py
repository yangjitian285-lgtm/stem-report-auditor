"""分野を問わない汎用の表検算エンジン.

どの理系実験の表でも使えるよう、列名や実験テーマに頼らず「表の数値そのもの」から
列どうしの計算関係を推定し、その関係から外れるセルを指摘する。

考え方
------
1. 各セルを「記載値 ± 最終桁の半分（丸め幅）」の区間として読む。
2. 列 C について、他の列から作れる候補式（C = A/B, C = k·A, C = sin²(A/2),
   C = 列の平均・合計・標準偏差, C = 100|A−B|/B …）を総当たりし、
   丸め幅を伝播させた許容範囲に入る行を数える。
3. 「偶然に一致する確率」（許容幅 ÷ 列の値の広がり）を一致した行で掛け合わせ、
   十分小さい（偶然では説明できない）ときだけ、その式を「この表の計算関係」と認める。
   定数 k を推定した式は、推定に使った自由度の分だけ証拠を差し引く。
4. 認めた式に合わない行を「計算ミス・転記ミスの候補」として報告する。
   ずれが丸め幅の数倍以内なら中間丸めの可能性として軽く扱う。
5. 「平均」「合計」「標準偏差」などの集計行、列内の桁数の不統一、
   列名の単位と換算係数の食い違い、度とラジアンの混同、2θ をそのまま使った三角関数も調べる。

限界
----
- 列全体が一貫して間違っている（全行で同じ誤った式を使った）場合、多数決では検出できない。
  度/ラジアンと 2θ/θ の取り違えだけは専用に検出する。
- 表に現れない定数（文献値・装置定数）を2つ以上含む式は推定しない。
- 測定値どうしの相関（物理法則に近い関係）は、丸め幅の範囲で一致しない限り関係とみなさない。
"""

from __future__ import annotations

import csv
import io
import itertools
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# セルの読み取り
# ---------------------------------------------------------------------------
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_NUM_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))"  # 仮数
    r"(?:\s*(?:[\*×xX・]\s*10\s*\^?\s*\(?([+-]?\d+)\)?|[eE]([+-]?\d+)))?"  # 指数
    r"\s*(?:±\s*[\d.]+)?"  # ±誤差（無視）
    r"\s*([%‰]|[A-Za-zμµÅ°℃Ω/·\^\-\d]{0,8})?\s*$"  # セル内の単位
)


@dataclass
class Cell:
    value: float
    half_ulp: float
    raw: str
    decimals: Optional[int]  # 小数点以下の桁数（指数表記は None）
    sig: int  # 有効数字の桁数（推定）


def parse_cell(text: str) -> Optional[Cell]:
    """セル文字列を数値＋丸め幅に変換する。数値でなければ None."""
    if text is None:
        return None
    s = str(text).strip().translate(_SUP)
    s = s.replace(",", "").replace("−", "-").replace("－", "-").replace("–", "-").replace("＋", "+")
    s = s.replace("．", ".").translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if not s or s in {"-", "--", "—", "―", "/"}:
        return None
    m = _NUM_RE.match(s)
    if not m:
        return None
    mant, e1, e2, unit = m.group(1), m.group(2), m.group(3), m.group(4) or ""
    # 「2-3」「1/2」などは数値ではない
    if unit and re.fullmatch(r"[\-/]\d+.*", unit):
        return None
    exp = int(e1 or e2 or 0)
    try:
        val = float(mant) * (10.0 ** exp)
    except ValueError:
        return None
    dec = len(mant.split(".")[1]) if "." in mant else 0
    half = 0.5 * (10.0 ** (-dec)) * (10.0 ** exp)
    digits = mant.lstrip("+-").replace(".", "").lstrip("0")
    sig = max(len(digits), 1)
    if unit == "%":
        pass  # % はそのまま数値として扱う（列名側で単位を判断）
    return Cell(value=val, half_ulp=half, raw=str(text).strip(), decimals=None if (e1 or e2) else dec, sig=sig)


# ---------------------------------------------------------------------------
# 単位
# ---------------------------------------------------------------------------
_PREFIX = {"p": 1e-12, "n": 1e-9, "μ": 1e-6, "µ": 1e-6, "u": 1e-6, "m": 1e-3, "c": 1e-2, "d": 1e-1, "k": 1e3, "M": 1e6, "G": 1e9}
_BASE = {"m": "m", "g": "g", "s": "s", "L": "L", "l": "L", "mol": "mol", "Pa": "Pa", "N": "N", "J": "J", "W": "W", "V": "V", "A": "A", "Hz": "Hz", "Ω": "Ω"}
_SPECIAL = {"Å": ("m", 1e-10), "min": ("s", 60.0), "h": ("s", 3600.0), "kgf": ("N", 9.80665)}


def header_unit(header: str) -> Optional[str]:
    h = header.strip()
    m = re.search(r"[\(\[（［]\s*([^\)\]）］]{1,10})\s*[\)\]）］]\s*$", h)
    if m:
        return m.group(1).strip()
    m = re.search(r"\s*[/／]\s*([^/／\s]{1,10})\s*$", h)
    if m and (m.group(0).startswith(" ") or unit_factor(m.group(1))):
        return m.group(1).strip()
    return None


def unit_factor(unit: Optional[str]) -> Optional[Tuple[str, float]]:
    """'nm' -> ('m', 1e-9). 解釈できなければ None."""
    if not unit:
        return None
    u = unit.strip()
    if u in _SPECIAL:
        return _SPECIAL[u]
    if u in _BASE:
        return (_BASE[u], 1.0)
    if len(u) >= 2 and u[0] in _PREFIX and u[1:] in _BASE:
        return (_BASE[u[1:]], _PREFIX[u[0]])
    return None


_DEG_HINT = re.compile(r"°|deg|角度|2θ|θ|φ|角")
_RAD_HINT = re.compile(r"rad")


# ---------------------------------------------------------------------------
# 結果
# ---------------------------------------------------------------------------
@dataclass
class TableFinding:
    severity: str  # CRITICAL / WARNING / INFO / PASS
    kind: str  # RELATION_OUTLIER, SUMMARY_ROW, DIGITS, UNIT, DEG_RAD, TWO_THETA, RELATION
    row: Optional[int]  # 1始まりのデータ行番号（表の見出しを除く）
    column: Optional[str]
    message: str
    suggestion: str = ""


@dataclass
class Relation:
    target: str
    formula: str
    matched_rows: int
    checked_rows: int
    evidence_log10: float
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class TableCheckResult:
    caption: str
    numeric_columns: List[str]
    data_rows: int
    summary_rows: List[str]
    relations: List[Relation]
    findings: List[TableFinding]

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 候補式
# ---------------------------------------------------------------------------
def _safe(f: Callable[..., float]) -> Callable[..., Optional[float]]:
    def g(*a: float) -> Optional[float]:
        try:
            v = f(*a)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        if v is None or isinstance(v, complex) or math.isnan(v) or math.isinf(v):
            return None
        return v

    return g


_R = math.radians
UNARY: List[Tuple[str, str, Callable[[float], float], int]] = [
    # (key, 表示テンプレート, 関数, 複雑さ)
    ("id", "{A}", lambda a: a, 0),
    ("sq", "({A})²", lambda a: a * a, 1),
    ("sqrt", "√{A}", lambda a: math.sqrt(a), 1),
    ("inv", "1/{A}", lambda a: 1.0 / a, 1),
    ("ln", "ln {A}", lambda a: math.log(a), 1),
    ("log10", "log₁₀ {A}", lambda a: math.log10(a), 1),
    ("exp", "exp({A})", lambda a: math.exp(a), 2),
    ("sin_d", "sin {A}°", lambda a: math.sin(_R(a)), 1),
    ("cos_d", "cos {A}°", lambda a: math.cos(_R(a)), 1),
    ("tan_d", "tan {A}°", lambda a: math.tan(_R(a)), 1),
    ("sin2_d", "sin²{A}°", lambda a: math.sin(_R(a)) ** 2, 1),
    ("cos2_d", "cos²{A}°", lambda a: math.cos(_R(a)) ** 2, 1),
    ("sinhalf_d", "sin({A}/2)°", lambda a: math.sin(_R(a / 2.0)), 1),
    ("sin2half_d", "sin²({A}/2)°", lambda a: math.sin(_R(a / 2.0)) ** 2, 1),
    ("cos2half_d", "cos²({A}/2)°", lambda a: math.cos(_R(a / 2.0)) ** 2, 1),
    ("sin_r", "sin {A}(rad)", lambda a: math.sin(a), 2),
    ("cos_r", "cos {A}(rad)", lambda a: math.cos(a), 2),
    ("tan_r", "tan {A}(rad)", lambda a: math.tan(a), 2),
    ("sin2_r", "sin²{A}(rad)", lambda a: math.sin(a) ** 2, 2),
    ("pow10", "10^{A}", lambda a: 10.0 ** a, 2),
    ("inv_sqrt", "1/√{A}", lambda a: 1.0 / math.sqrt(a), 1),
    ("inv_sq", "1/{A}²", lambda a: 1.0 / (a * a), 1),
    ("inv_sin_d", "1/sin {A}°", lambda a: 1.0 / math.sin(_R(a)), 1),
    ("inv_sinhalf_d", "1/sin({A}/2)°", lambda a: 1.0 / math.sin(_R(a / 2.0)), 1),
]
UNARY = [(k, t, _safe(f), c) for k, t, f, c in UNARY]

BINARY: List[Tuple[str, str, Callable[[float, float], float], int, bool]] = [
    # (key, 表示, 関数, 複雑さ, 定数倍kを推定するか)
    ("add", "{A} + {B}", lambda a, b: a + b, 0, False),
    ("sub", "{A} − {B}", lambda a, b: a - b, 0, False),
    ("absdiff", "|{A} − {B}|", lambda a, b: abs(a - b), 1, False),
    ("mul", "{A}·{B}", lambda a, b: a * b, 0, True),
    ("div", "{A}/{B}", lambda a, b: a / b, 0, True),
    ("relerr", "|{A} − {B}|/{B}", lambda a, b: abs(a - b) / abs(b), 1, True),
    ("relsigned", "({A} − {B})/{B}", lambda a, b: (a - b) / b, 1, True),
    ("div_sq", "{A}/({B})²", lambda a, b: a / (b * b), 1, True),
    ("mul_sq", "{A}·({B})²", lambda a, b: a * b * b, 1, True),
    ("mul_sqrt", "{A}·√{B}", lambda a, b: a * math.sqrt(b), 1, True),
    ("div_sqrt", "{A}/√{B}", lambda a, b: a / math.sqrt(b), 1, True),
    ("hypot", "√({A}² + {B}²)", lambda a, b: math.hypot(a, b), 2, False),
    ("harm", "{A}·{B}/({A} + {B})", lambda a, b: a * b / (a + b), 2, False),
]
BINARY = [(k, t, _safe(f), c, kf) for k, t, f, c, kf in BINARY]

# 「きれいな」定数：推定した k がこの値と区間で一致すればこの値に丸める
NICE_K: List[Tuple[float, str]] = [(1.0, "1"), (-1.0, "−1"), (2.0, "2"), (0.5, "1/2"), (100.0, "100"), (-100.0, "−100"),
                                   (math.pi, "π"), (2 * math.pi, "2π"), (4 * math.pi ** 2, "4π²"), (1 / math.pi, "1/π"),
                                   (180 / math.pi, "180/π"), (math.pi / 180, "π/180"), (9.80665, "g(9.80665)"), (9.81, "9.81"),
                                   (9.8, "9.8"), (8.314, "R(8.314)"), (8.314462618, "R"), (60.0, "60"), (1 / 60, "1/60"),
                                   (3600.0, "3600"), (1 / 3600, "1/3600"), (4.0, "4"), (0.25, "1/4"), (3.0, "3"), (1 / 3, "1/3")]
NICE_K += [(10.0 ** n, f"10^{n}") for n in range(-12, 13) if n not in (0, 2)]
# 重力加速度・水の比熱など、実験でよく使う定数の 10^±3 倍（g→kg, J→kJ の換算込み）
for _base, _nm in ((9.8, "9.8"), (9.81, "9.81"), (9.80665, "9.80665"), (4.18, "4.18"), (4.184, "4.184")):
    for _e in (-3, 3):
        NICE_K.append((_base * 10.0 ** _e, f"{_nm}×10^{_e}"))
NICE_K += [(4.18, "4.18"), (4.184, "4.184")]
NICE_OFFSET: List[Tuple[float, str]] = [(273.15, "273.15"), (-273.15, "−273.15"), (273.0, "273"), (-273.0, "−273"),
                                        (2.0, "2"), (100.0, "100"), (-100.0, "−100"), (1.0, "1"), (-1.0, "−1"), (0.0, "0")]

_SUMMARY_LABELS = [
    ("mean", re.compile(r"平均|mean|average|^ave\.?$|^avg\.?$", re.I)),
    ("sum", re.compile(r"合計|総和|総計|^計$|total|^sum$", re.I)),
    ("sd", re.compile(r"標準偏差|^s\.?d\.?$|^σ|std", re.I)),
    ("se", re.compile(r"標準誤差|^s\.?e\.?$|^sem$", re.I)),
    ("max", re.compile(r"最大|^max", re.I)),
    ("min", re.compile(r"最小|^min", re.I)),
]

ACCEPT_LOG10 = -3.0  # 偶然一致の確率がこれ未満なら関係とみなす
MIN_MATCH_FRAC = 0.6
TOL_SLACK = 1.05


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------
def _interval(f: Callable[..., Optional[float]], cells: Sequence[Cell]) -> Optional[Tuple[float, float, float]]:
    """丸め幅の範囲で f の値がとりうる区間 (center, lo, hi)。角の点と中心で評価."""
    center = f(*[c.value for c in cells])
    if center is None:
        return None
    vals = [center]
    for signs in itertools.product((-1.0, 1.0), repeat=len(cells)):
        v = f(*[c.value + s * c.half_ulp for c, s in zip(cells, signs)])
        if v is None:
            return None
        vals.append(v)
    # 区間内に極値をもつ関数（sin² 等）の取りこぼしを避けるため中点方向も評価
    for i in range(len(cells)):
        for s in (-0.5, 0.5):
            args = [c.value for c in cells]
            args[i] += s * cells[i].half_ulp
            v = f(*args)
            if v is not None:
                vals.append(v)
    return center, min(vals), max(vals)


def _max_stab(intervals: Sequence[Tuple[float, float, int]]) -> Tuple[int, float, float]:
    """区間 (lo, hi, row_id) の中で最も多くの行が重なる点の範囲を返す (count, lo, hi)."""
    events: List[Tuple[float, int, int]] = []
    for lo, hi, rid in intervals:
        events.append((lo, 0, rid))  # 開始を先に
        events.append((hi, 1, rid))
    events.sort()
    active: Dict[int, int] = {}
    best, best_lo, best_hi = 0, 0.0, 0.0
    for i, (x, typ, rid) in enumerate(events):
        if typ == 0:
            active[rid] = active.get(rid, 0) + 1
            cnt = len(active)
            if cnt > best:
                best = cnt
                best_lo = x
                # 次のイベント位置まで
                best_hi = events[i + 1][0] if i + 1 < len(events) else x
        else:
            active[rid] -= 1
            if active[rid] == 0:
                del active[rid]
    return best, best_lo, best_hi


def _snap(lo: float, hi: float, table: Sequence[Tuple[float, str]]) -> Optional[Tuple[float, str]]:
    for v, name in table:
        if lo - 1e-12 * max(1.0, abs(v)) <= v <= hi + 1e-12 * max(1.0, abs(v)):
            return v, name
    return None


def _shortest_decimal(lo: float, hi: float) -> Optional[float]:
    """区間 [lo, hi] に入る、桁数が最も少ない10進数（文献値・基準値は丸い値が多い）."""
    mid = (lo + hi) / 2.0
    for d in range(-3, 8):
        c = round(mid, d)
        if lo <= c <= hi:
            return c
    return None


def short_name(header: str) -> str:
    """式の表示用に列名から単位を外す: 'd (nm)' -> 'd'."""
    h = re.sub(r"\s*[\(\[（［][^\)\]）］]{1,10}[\)\]）］]\s*$", "", header.strip())
    m = re.search(r"\s*[/／]\s*([^/／\s]{1,10})\s*$", h)
    if m and (m.group(0).startswith(" ") or unit_factor(m.group(1)) or m.group(1) in ("%", "°", "℃")):
        h = h[: m.start()]
    return h or header


def _wrap(name: str) -> str:
    return f"({name})" if re.search(r"[+\-−*/·×\s]", name) and not (name.startswith("(") and name.endswith(")")) else name


def _fmt_k(k: float, name: Optional[str]) -> str:
    if name:
        return name
    return f"{k:.4g}"


_INDEX_HEADER = re.compile(r"^(no\.?|番号|#|回|回数|試行|run|trial|サンプル|試料|測定)\s*(no\.?|番号)?$", re.I)


def _is_index_column(values: Sequence[Optional[Cell]], header: str = "", first: bool = False) -> bool:
    """1,2,3,... のような通し番号列（列名が番号らしいか、先頭列のとき）."""
    v = [c.value for c in values if c is not None]
    if len(v) < 3 or any(c.decimals != 0 for c in values if c is not None):
        return False  # 「1.00, 2.00 …」のように小数で書かれた列は番号ではなく設定値
    seq = all(abs(x - round(x)) < 1e-9 for x in v) and all(abs((v[i + 1] - v[i]) - 1.0) < 1e-9 for i in range(len(v) - 1))
    return seq and (first or bool(_INDEX_HEADER.match(header.strip())))


def _mark_set_points(col: Sequence[Optional[Cell]], leftmost: bool = False) -> None:
    """等間隔の設定値（10, 20, 30 °／0.10, 0.20 mol/L など）は丸め値ではなく厳密な設定値として扱う."""
    cells = [c for c in col if c is not None]
    if len(cells) < 3 or any(c.decimals is None for c in cells):
        return
    if not leftmost and any(c.decimals != 0 for c in cells):
        return  # 小数の等間隔は、表の左端（独立変数の位置）にあるときだけ設定値とみなす
    v = [c.value for c in cells]
    steps = {round(v[i + 1] - v[i], 9) for i in range(len(v) - 1)}
    if len(steps) == 1 and next(iter(steps)) != 0:
        for c in cells:
            c.half_ulp = 0.0


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
class _Grid:
    def __init__(self, headers: Sequence[str], rows: Sequence[Sequence[str]]):
        width = max([len(headers)] + [len(r) for r in rows]) if rows else len(headers)
        self.headers = [str(h).strip() if i < len(headers) else f"列{i+1}" for i, h in enumerate(list(headers) + [""] * (width - len(headers)))]
        self.headers = [h if h else f"列{i+1}" for i, h in enumerate(self.headers)]
        self.raw = [list(r) + [""] * (width - len(r)) for r in rows]
        self.cells: List[List[Optional[Cell]]] = [[parse_cell(x) for x in r] for r in self.raw]
        self.width = width
        # 集計行の判定（数値でない最初のセルのラベル）
        self.summary: Dict[int, str] = {}
        for ri, r in enumerate(self.raw):
            label = next((str(x).strip() for x, c in zip(r, self.cells[ri]) if c is None and str(x).strip()), "")
            for kind, pat in _SUMMARY_LABELS:
                if label and pat.search(label):
                    self.summary[ri] = kind
                    break
        self.data_idx = [ri for ri in range(len(self.raw)) if ri not in self.summary]
        # 数値列：データ行の6割以上が数値
        self.numeric_cols: List[int] = []
        for ci in range(width):
            col = [self.cells[ri][ci] for ri in self.data_idx]
            n_num = sum(1 for c in col if c is not None)
            n_nonempty = sum(1 for ri in self.data_idx if str(self.raw[ri][ci]).strip())
            if n_num >= 2 and n_num >= 0.6 * max(n_nonempty, 1):
                self.numeric_cols.append(ci)
        self.index_cols = {ci for ci in self.numeric_cols
                           if _is_index_column(self.col(ci), self.headers[ci], first=(ci == min(self.numeric_cols + [ci]) and ci == 0))}
        non_index = [ci for ci in self.numeric_cols if ci not in self.index_cols]
        for ci in non_index:
            _mark_set_points(self.col(ci), leftmost=(ci == non_index[0]))

    def col(self, ci: int) -> List[Optional[Cell]]:
        return [self.cells[ri][ci] for ri in self.data_idx]


@dataclass
class _Hyp:
    target: int
    inputs: Tuple[int, ...]
    formula: str
    complexity: int
    matched: List[int]  # data_idx 上の位置
    checked: List[int]
    evidence: float  # log10
    predicted: Dict[int, Tuple[float, float]]  # 位置 -> (予測値, 許容幅)
    params: Dict[str, float]
    func_key: str = ""
    free_params: int = 0
    strict: List[int] = field(default_factory=list)  # 表示値そのものから計算した値と最終桁まで一致する行
    exact_const: bool = True  # 定数がきれいな値（または丸い基準値）で、表示値からの厳密な再計算ができる
    strict_evidence: float = 0.0  # 最終桁まで一致した行だけで計算した偶然一致の確率（log10）


def _evaluate(grid: _Grid, target: int, inputs: Tuple[int, ...], g: Callable[..., Optional[float]],
              template: str, complexity: int, mode: str, func_key: str) -> List[_Hyp]:
    """mode: 'fixed' | 'scale'(C=k·g) | 'offset'(C=±g+b).

    scale / offset では「きれいな定数で最も多く一致する候補」と「定数を自由に推定した候補」の両方を返す。
    """
    tcol = grid.col(target)
    icols = [grid.col(i) for i in inputs]
    n = len(tcol)
    vals_c = [c.value for c in tcol if c is not None]
    if len(vals_c) < 3:
        return []
    spread = max(vals_c) - min(vals_c)
    if spread <= 0:
        return []
    rows: List[Tuple[int, Cell, Tuple[float, float, float]]] = []
    for p in range(n):
        c = tcol[p]
        ins = [col[p] for col in icols]
        if c is None or any(x is None for x in ins):
            continue
        iv = _interval(g, ins)  # type: ignore[arg-type]
        if iv is None:
            continue
        rows.append((p, c, iv))
    n_valid_inputs = sum(1 for p in range(n) if tcol[p] is not None and all(col[p] is not None for col in icols))
    if len(rows) < 3 or len(rows) < 0.6 * max(n_valid_inputs, 1):
        return []

    settings: List[Tuple[float, float, Optional[str], Optional[str], int]] = []  # (k, b, k_name, b_name, free)
    if mode == "fixed":
        settings.append((1.0, 0.0, None, None, 0))
    elif mode == "scale":
        ivs = []
        for p, c, (gc, glo, ghi) in rows:
            if glo <= 0 <= ghi:
                continue
            cands = [(c.value + s1 * c.half_ulp) / gg for s1 in (-1, 1) for gg in (glo, ghi)]
            ivs.append((min(cands), max(cands), p))
        if len(ivs) < 3:
            return []
        # きれいな定数ごとに、その値を含む行の数を数える
        best_nice = None
        for v, name in NICE_K:
            cnt = sum(1 for lo, hi, _ in ivs if lo - 1e-12 * abs(v) <= v <= hi + 1e-12 * abs(v))
            if cnt >= 2 and (best_nice is None or cnt > best_nice[0]):
                best_nice = (cnt, v, name)
        if best_nice:
            settings.append((best_nice[1], 0.0, best_nice[2], None, 0))
        cnt, lo, hi = _max_stab(ivs)
        if not (best_nice and lo <= best_nice[1] <= hi) and abs((lo + hi) / 2.0) > 1e-300:
            settings.append(((lo + hi) / 2.0, 0.0, None, None, 1))
    else:  # offset
        ivs = []
        sign = -1.0 if func_key.startswith("neg_") else 1.0
        for p, c, (gc, glo, ghi) in rows:
            cands = [c.value + s1 * c.half_ulp - sign * gg for s1 in (-1, 1) for gg in (glo, ghi)]
            ivs.append((min(cands), max(cands), p))
        best_nice = None
        for v, name in NICE_OFFSET:
            cnt = sum(1 for lo, hi, _ in ivs if lo - 1e-12 <= v <= hi + 1e-12)
            if cnt >= 2 and (best_nice is None or cnt > best_nice[0]):
                best_nice = (cnt, v, name)
        if best_nice:
            settings.append((sign, best_nice[1], None, best_nice[2], 0))
        cnt, lo, hi = _max_stab(ivs)
        if not (best_nice and lo <= best_nice[1] <= hi):
            b = _shortest_decimal(lo, hi)
            settings.append((sign, (lo + hi) / 2.0 if b is None else b, None, None, 1))

    out: List[_Hyp] = []
    names = {"A": _wrap(short_name(grid.headers[inputs[0]])), "B": _wrap(short_name(grid.headers[inputs[1]])) if len(inputs) > 1 else ""}
    body = template if func_key.startswith("run_") else template.format(**names)
    for k, b, k_name, b_name, free_params in settings:
        matched: List[int] = []
        strict: List[int] = []
        strict_ev = 0.0
        pvals: List[float] = []
        predicted: Dict[int, Tuple[float, float]] = {}
        for p, c, (gc, glo, ghi) in rows:
            pred = k * gc + b
            # 表示された入力値そのものから計算したとき、最終桁まで一致するか（手計算の表はこちら）
            if not free_params and abs(pred - c.value) <= c.half_ulp * TOL_SLACK + 1e-9 * max(1.0, abs(pred)):
                strict.append(p)
                strict_ev += math.log10(max(min(1.0, 2.0 * c.half_ulp * TOL_SLACK / spread), 1e-300))
            # 予測値の区間幅（入力の丸め）＋記載値の丸め幅
            tol = abs(k) * max(ghi - gc, gc - glo) + c.half_ulp
            tol = tol * TOL_SLACK + 1e-12 * max(1.0, abs(pred))
            if free_params:
                tol += abs(pred) * 1e-6  # 推定した定数そのものの不確かさ
            predicted[p] = (pred, tol)
            if abs(pred - c.value) <= tol:
                matched.append(p)
                pvals.append(min(1.0, 2.0 * tol / spread))
        if not matched:
            continue
        pvals.sort()
        # 自由パラメータの数だけ、最も情報量の大きい（p が小さい）行を証拠から外す
        used = pvals[free_params:]
        evidence = sum(math.log10(max(x, 1e-300)) for x in used) if used else 0.0
        k_str = _fmt_k(k, k_name)
        if mode == "scale":
            if k_name == "1":
                formula = body
            elif body.startswith("1/"):
                formula = f"{k_str}/{body[2:]}"
            elif " " not in body and "+" not in body and "−" not in body:
                formula = f"{k_str}·{body}"
            else:
                formula = f"{k_str}·({body})"
        elif mode == "offset":
            lead = "−" if k < 0 else ""
            if b == 0:
                tail = ""
            elif b > 0:
                tail = f" + {_fmt_k(b, b_name)}"
            else:
                tail = f" − {_fmt_k(-b, None if b_name is None else b_name.lstrip('−'))}"
            formula = f"{lead}{body}{tail}"
        else:
            formula = body
        cx = complexity
        if func_key == "id" and mode == "scale" and k_name == "1":
            cx += 3  # 別の列をそのまま写しただけ、は計算関係としては弱い仮説
        out.append(_Hyp(target, inputs, formula, cx + 2 * free_params, matched, [r[0] for r in rows], evidence, predicted,
                        {"k": k, "b": b}, func_key, free_params, strict, not free_params, strict_ev))
    return out


_ANGLE_HINT = re.compile(r"°|deg|rad|角|θ|φ|ψ|angle", re.I)
_TRIG_KEYS = {"sin_d", "cos_d", "tan_d", "sin2_d", "cos2_d", "sinhalf_d", "sin2half_d", "cos2half_d",
              "sin_r", "cos_r", "tan_r", "sin2_r", "inv_sin_d", "inv_sinhalf_d"}


def _candidate_hypotheses(grid: _Grid, target: int) -> List[_Hyp]:
    out: List[_Hyp] = []
    others = [c for c in grid.numeric_cols if c != target and c not in grid.index_cols]
    for a in others:
        angle = bool(_ANGLE_HINT.search(grid.headers[a]))
        for key, tmpl, f, cx in UNARY:
            hs = _evaluate(grid, target, (a,), f, tmpl, cx, "scale", key)
            if key in _TRIG_KEYS and not angle:
                # 角度らしくない列の三角関数は、定数倍なし（k=1）のときだけ候補にする
                # （sin x ≈ x の小角近似で、比例関係を三角関数と誤認しないため）
                hs = [h for h in hs if h.free_params == 0 and abs(h.params["k"] - 1.0) < 1e-12]
            out.extend(hs)
            if key in ("id", "log10", "ln", "inv"):
                for sign_key in (key, "neg_" + key):
                    out.extend(_evaluate(grid, target, (a,), f, tmpl, cx + 1, "offset", sign_key))
        # 定数の参照値に対する相対誤差（%）：C = k·|A − b|/b、b は表外の文献値
        h = _relerr_const(grid, target, a)
        if h:
            out.append(h)
    for a, b in itertools.permutations(others, 2):
        for key, tmpl, f, cx, kfree in BINARY:
            if key in ("add", "mul", "hypot", "absdiff", "harm") and a > b:
                continue  # 可換
            out.extend(_evaluate(grid, target, (a, b), f, tmpl, cx, "scale" if kfree else "fixed", key))
    # 隣り合う数値列のまとまり（1回目・2回目・3回目 …）の平均・合計・標準偏差
    cols = [c for c in grid.numeric_cols if c != target and c not in grid.index_cols]
    for i in range(len(cols)):
        for j in range(i + 1, min(len(cols), i + 12)):
            run = cols[i: j + 1]
            if any(run[t + 1] - run[t] != 1 for t in range(len(run) - 1)):
                break
            label = f"{short_name(grid.headers[run[0]])}〜{short_name(grid.headers[run[-1]])}"
            for key, name, f in (
                ("run_mean", f"平均({label})", lambda *v: sum(v) / len(v)),
                ("run_sum", f"合計({label})", lambda *v: sum(v)),
                ("run_sd1", f"標本標準偏差({label})", lambda *v: statistics.stdev(v) if len(v) > 1 else None),
                ("run_sd0", f"母標準偏差({label})", lambda *v: statistics.pstdev(v)),
            ):
                out.extend(_evaluate(grid, target, tuple(run), _safe(f), name, 1 if key in ("run_mean", "run_sum") else 2, "fixed", key))
    return out


def _relerr_const(grid: _Grid, target: int, a: int) -> Optional[_Hyp]:
    tcol, acol = grid.col(target), grid.col(a)
    vals = [c.value for c in tcol if c is not None]
    if len(vals) < 3 or max(vals) - min(vals) <= 0:
        return None
    spread = max(vals) - min(vals)
    best = None
    for scale, sname in ((100.0, "100"), (1.0, "1")):
        ivs = []
        for p, (c, x) in enumerate(zip(tcol, acol)):
            if c is None or x is None:
                continue
            for sgn in (1.0, -1.0):
                cands = []
                for s1 in (-1, 1):
                    for s2 in (-1, 1):
                        den = 1.0 + sgn * (c.value + s1 * c.half_ulp) / scale
                        if den <= 0:
                            continue
                        cands.append((x.value + s2 * x.half_ulp) / den)
                if len(cands) == 4:
                    ivs.append((min(cands), max(cands), p))
        if len(ivs) < 3:
            continue
        cnt, lo, hi = _max_stab(ivs)
        # 表示値どうしで最終桁まで合う基準値（手計算の表）があれば、そちらを優先する
        sivs = []
        for p, (c, x) in enumerate(zip(tcol, acol)):
            if c is None or x is None:
                continue
            for sgn in (1.0, -1.0):
                cands = [x.value / (1.0 + sgn * (c.value + s1 * c.half_ulp) / scale) for s1 in (-1, 1)
                         if 1.0 + sgn * (c.value + s1 * c.half_ulp) / scale > 0]
                if len(cands) == 2:
                    sivs.append((min(cands), max(cands), p))
        scnt, slo, shi = _max_stab(sivs) if len(sivs) >= 3 else (0, 0.0, 0.0)
        n_rows = sum(1 for c, x in zip(tcol, acol) if c is not None and x is not None)
        ssd = _shortest_decimal(slo, shi) if scnt >= max(3, n_rows - 1) else None
        sd = ssd if ssd is not None else _shortest_decimal(lo, hi)
        if sd is not None:
            bref, lo, hi = sd, sd, sd
        else:
            bref = (lo + hi) / 2.0
        if bref <= 0:
            continue
        matched, pvals, predicted, checked, strict = [], [], {}, [], []
        strict_ev = 0.0
        for p, (c, x) in enumerate(zip(tcol, acol)):
            if c is None or x is None:
                continue
            checked.append(p)
            pred = scale * abs(x.value - bref) / bref
            signed = scale * (x.value - bref) / bref
            if c.value < 0:
                pred = signed
            tol = (scale * x.half_ulp / bref + c.half_ulp) * TOL_SLACK + scale * abs(x.value) * (hi - lo) / (bref * bref)
            predicted[p] = (pred, tol)
            if sd is not None and abs(pred - c.value) <= c.half_ulp * TOL_SLACK + 1e-9:
                strict.append(p)
                strict_ev += math.log10(max(min(1.0, 2.0 * c.half_ulp * TOL_SLACK / spread), 1e-300))
            if abs(pred - c.value) <= tol:
                matched.append(p)
                pvals.append(min(1.0, 2 * tol / spread))
        if len(matched) < 3:
            continue
        pvals.sort()
        ev = sum(math.log10(max(x, 1e-300)) for x in pvals[1:])
        h = _Hyp(target, (a,), f"{sname}·|{short_name(grid.headers[a])} − {bref:.6g}|/{bref:.6g}（基準値 {bref:.6g} を推定）",
                 3, matched, checked, ev, predicted, {"ref": bref, "scale": scale}, "relerr_const", 1, strict, sd is not None,
                 strict_ev - (max(math.log10(max(min(1.0, 2.0 * tcol[strict[0]].half_ulp / spread), 1e-300)), -300) if strict else 0.0))
        if best is None or (len(h.matched), -h.evidence) > (len(best.matched), -best.evidence):
            best = h
    return best


def _accept(h: _Hyp) -> bool:
    n_chk = len(h.checked)
    if h.free_params and h.func_key.startswith("neg_") or (h.free_params and h.params.get("b", 0) != 0):
        # 表外の定数を足し引きする式は、相関した測定列どうしで偶然当てはまりやすい
        min_match, frac = 4, 0.75
    elif h.free_params:
        min_match, frac = 3, 0.75
    else:
        min_match, frac = 3, MIN_MATCH_FRAC
    # 表示値から最終桁まで一致する行が3行以上あれば、その一致だけで偶然を否定できることがある
    strict_ok = len(h.strict) >= 3 and h.strict_evidence <= ACCEPT_LOG10
    return (
        n_chk >= 3
        and len(h.matched) >= min_match
        and len(h.matched) >= frac * n_chk
        and (h.evidence <= ACCEPT_LOG10 or strict_ok)
    )


def _accept_weak(h: _Hyp) -> bool:
    n_chk = len(h.checked)
    k = h.params.get("k", 1.0)
    simple = h.free_params == 0 and (
        h.func_key in ("div", "mul", "div_sq", "mul_sq", "relerr", "relsigned", "harm")
        or (h.func_key in ("id", "inv", "sq") and abs(abs(k) - 1.0) > 1e-9)
    )
    return simple and n_chk >= 4 and len(h.matched) >= 0.75 * n_chk and len(h.matched) >= 3


_HINTS = [
    (re.compile(r"平均|mean|average|ave", re.I), ("run_mean",)),
    (re.compile(r"合計|総和|sum|total", re.I), ("run_sum",)),
    (re.compile(r"標準偏差|s\.?d\.?|σ|std", re.I), ("run_sd1", "run_sd0")),
    (re.compile(r"誤差|error|ずれ|偏差", re.I), ("relerr", "relsigned", "relerr_const", "absdiff", "sub")),
]


def _apply_header_hints(header: str, hyps: List[_Hyp]) -> None:
    for pat, keys in _HINTS:
        if pat.search(header):
            for h in hyps:
                if h.func_key in keys:
                    h.complexity -= 3


def _severity_for(dev: float, tol: float) -> str:
    """許容幅は入力の丸めをすでに含むので、1.25倍を超えたら表の値どうしで説明できないずれ."""
    ratio = dev / max(tol, 1e-300)
    if ratio > 5:
        return "CRITICAL"
    if ratio > 1.25:
        return "WARNING"
    return "INFO"


def _rank(h: _Hyp) -> Tuple[float, int, float]:
    # 自由パラメータ1つは一致1行ぶんの価値として差し引く
    return (-(len(h.matched) - h.free_params), h.complexity, h.evidence)


def check_table(headers: Sequence[str], rows: Sequence[Sequence[str]], caption: str = "表") -> TableCheckResult:
    """表（見出し＋行）を汎用検算する."""
    grid = _Grid(headers, rows)
    findings: List[TableFinding] = []
    relations: List[Relation] = []
    data_rowno = {p: grid.data_idx[p] + 1 for p in range(len(grid.data_idx))}

    # ---- 1) 列どうしの計算関係（右の列ほど計算結果である可能性が高いので右から）
    used_sets: List[frozenset] = []
    all_hyps: Dict[int, List[_Hyp]] = {}
    chosen: List[Tuple[_Hyp, bool, List[_Hyp]]] = []
    ranked: Dict[int, Tuple[List[_Hyp], List[_Hyp]]] = {}
    order = [t for t in sorted(grid.numeric_cols, reverse=True) if t not in grid.index_cols]
    for target0 in order:
        cand = _candidate_hypotheses(grid, target0)
        _apply_header_hints(grid.headers[target0], cand)
        for h in cand:
            h.complexity += max(0, len(h.inputs) - 1) if not h.func_key.startswith("run_") else 0
        all_hyps[target0] = cand
        strong = sorted([h for h in cand if _accept(h)], key=_rank)
        # 結果の列がほぼ一定（g, 抵抗, 密度など）だと偶然一致の確率を小さく見積もれない。
        # その場合は「きれいな定数」で大半の行が一致する単純な式だけ、確度の低い関係として採用する。
        weak_list = sorted([h for h in cand if not _accept(h) and _accept_weak(h)], key=_rank)
        ranked[target0] = (strong, weak_list)

    n_taken: Dict[int, int] = {t: 0 for t in order}
    # 1巡目：各列の主たる関係を1つずつ。2巡目：独立な2つ目の関係（P = V·I と P = R·I² のような従属関係より後に回す）
    for pass_no in (1, 2):
        for target0 in order:
            strong, weak_list = ranked[target0]
            pool_list = [(x, False) for x in strong] + ([(x, True) for x in weak_list] if pass_no == 1 else [])
            for h, weak in pool_list:
                if n_taken[target0] >= pass_no:
                    break
                cols_set = frozenset((target0,) + h.inputs)
                # 同じ関係の並べ替え（C=A/B と A=B·C）や、既出の関係を含むだけの式は数えない
                if any(cols_set <= u or cols_set >= u for u in used_sets):
                    continue
                # 既出の2つの関係から導ける式（P = V·I と R = V/I があるときの P = R·I² など）も数えない
                if any(cols_set <= (u1 | u2) for u1, u2 in itertools.combinations(used_sets, 2) if len(u1 & u2) >= 1):
                    continue
                if any(target0 in u for u in used_sets):
                    # すでに別の関係に出てくる列（例：T² の元になった T）に新しい関係を足すのは、
                    # 定数なしの厳密な関数（k=1）か、一致がとても厳密か、全行が一致するときだけ
                    # （物理法則と定数の偶然の一致を避ける）
                    exact_fn = h.free_params == 0 and abs(h.params.get("k", 1.0) - 1.0) < 1e-12 and h.params.get("b", 0.0) == 0.0
                    n_used = max(1, len(h.matched) - h.free_params)
                    tight = h.evidence / n_used <= -2.5  # 1行あたり、列の広がりの0.3%以内で一致
                    if not exact_fn and not tight and len(h.matched) < len(h.checked):
                        continue
                used_sets.append(cols_set)
                n_taken[target0] += 1
                best, pool = h, strong
                # 表示は「より右の列＝計算結果」の向きにそろえる（g = 4π²L/T² のように）
                for t2 in sorted(cols_set, reverse=True):
                    if t2 <= target0:
                        break
                    same = [x for x in all_hyps.get(t2, []) if frozenset((t2,) + x.inputs) == cols_set and len(x.matched) >= len(h.matched)]
                    if same:
                        same.sort(key=lambda x: (x.complexity, -len(x.matched)))
                        ev = h.evidence
                        best = same[0]
                        best.evidence = ev
                        pool = ranked.get(t2, ([], []))[0]
                        break
                chosen.append((best, weak, pool))

    for best, weak, pool in chosen:
        target = best.target
        relations.append(Relation(grid.headers[target], best.formula + ("（確度低め）" if weak else ""), len(best.matched), len(best.checked),
                                  round(best.evidence, 1), {k: round(v, 10) for k, v in best.params.items()}))
        best_cols = set((target,) + best.inputs)
        for p in best.checked:
            if p in best.matched:
                continue
            pred, tol = best.predicted[p]
            cell = grid.col(target)[p]
            dev = abs(pred - cell.value)
            sev = _severity_for(dev, tol)
            # 表示値から計算した表（一致した行がすべて最終桁まで一致）で、最終桁が1.5以上ずれていれば誤り
            strict_table = best.exact_const and len(best.strict) >= 3 and len(best.strict) == len(best.matched)
            if sev == "INFO" and strict_table and dev >= 3.0 * cell.half_ulp:
                sev = "WARNING"
            if weak and sev == "CRITICAL":
                sev = "WARNING"
            # 別の（同程度に確からしい）関係がこの行を説明できるなら、その関係に出てこない列が怪しい
            alt = next((x for x in pool if x is not best and p in x.matched and (not strict_table or p in x.strict)
                        and (len(x.matched) - x.free_params) >= (len(best.matched) - best.free_params) - 1), None)
            suspects: List[str] = []
            if alt is not None:
                suspects = [grid.headers[c] for c in sorted(best_cols - set((alt.target,) + alt.inputs))]
                if not suspects and sev != "INFO":
                    sev = "INFO"
            in_names = "・".join(grid.headers[i] for i in best.inputs)
            if sev == "INFO":
                sugg = "ずれは丸め幅を少し超える程度です。表に載せていない途中の値を丸めて使った可能性があります。"
            elif suspects:
                sugg = (f"この行は「{short_name(grid.headers[alt.target])} = {alt.formula}」とは合うので、"
                        f"{'・'.join(suspects)} の値（転記・計算）が最も疑わしいです。")
            else:
                sugg = f"この行の {grid.headers[target]} または {in_names} の計算・転記を確認してください。"
            findings.append(TableFinding(
                sev, "RELATION_OUTLIER", data_rowno[p], grid.headers[target],
                f"他の{len(best.matched)}行は「{short_name(grid.headers[target])} = {best.formula}」で一致しますが、"
                f"この行は記載 {cell.raw} に対し計算値 {pred:.6g}（許容 ±{tol:.2g}）です。",
                sugg,
            ))
        # 表示値から計算した表（他の行はすべて最終桁まで一致）なのに、1行だけ最終桁がずれている
        loose_only = [p for p in best.matched if p not in best.strict]
        # 同じ列について別の関係（例：T² 列経由の g）がこの行を最終桁まで説明するなら、計算経路の違いにすぎない
        others_strict = {q for (o, _w, _pl) in chosen if o is not best and o.target == target for q in o.strict}
        loose_only = [p for p in loose_only if p not in others_strict]
        if (not weak and best.exact_const and len(best.strict) >= 3 and len(loose_only) == 1):
            p = loose_only[0]
            pred, _tol = best.predicted[p]
            cell = grid.col(target)[p]
            # Excel など丸め前の値で計算した表では、表示値から再計算すると最終桁が1つ程度ずれるのは正常。
            # 最終桁で1.5以上（丸め幅の3倍以上）ずれるときだけ指摘する。
            if abs(pred - cell.value) < 3.0 * cell.half_ulp:
                loose_only = []
        if (not weak and best.exact_const and len(best.strict) >= 3 and len(loose_only) == 1):
            p = loose_only[0]
            pred, _tol = best.predicted[p]
            cell = grid.col(target)[p]
            findings.append(TableFinding(
                "WARNING", "LAST_DIGIT", data_rowno[p], grid.headers[target],
                f"他の{len(best.strict)}行は表に書かれた値から「{short_name(grid.headers[target])} = {best.formula}」で計算すると"
                f"最終桁まで一致しますが、この行は記載 {cell.raw} に対し {pred:.6g} です。",
                "この行だけ丸め前の値を使ったのでなければ、計算ミスか転記ミスです。",
            ))
        # 単位換算の食い違い
        if best.func_key == "id" and len(best.inputs) == 1:
            ua, ut = unit_factor(header_unit(grid.headers[best.inputs[0]])), unit_factor(header_unit(grid.headers[target]))
            if ua and ut and ua[0] == ut[0]:
                expected = ua[1] / ut[1]
                k = best.params.get("k", 1.0)
                if abs(k - expected) > 1e-6 * abs(expected) and abs(math.log10(abs(k) / expected)) > 0.05:
                    findings.append(TableFinding(
                        "CRITICAL", "UNIT", None, grid.headers[target],
                        f"{grid.headers[best.inputs[0]]} → {grid.headers[target]} の換算係数が {k:.4g} 倍になっています"
                        f"（列名の単位どおりなら {expected:.4g} 倍）。",
                        "単位の接頭語（m, μ, n, Å など）と換算の桁を確認してください。",
                    ))
        # 度とラジアン・2θ の取り違え（列全体が一貫して誤っている場合）
        in_h = grid.headers[best.inputs[0]]
        if best.func_key.endswith("_r") and _DEG_HINT.search(in_h) and not _RAD_HINT.search(in_h):
            findings.append(TableFinding(
                "CRITICAL", "DEG_RAD", None, grid.headers[target],
                f"{grid.headers[target]} は「{best.formula}」、つまり {in_h} をラジアンとして計算した値と一致します。列名は度のようです。",
                "関数電卓・Excel の角度モード（DEG/RAD）を確認してください。Excel の SIN は RADIANS() で変換が必要です。",
            ))
        if best.func_key in ("sin_d", "cos_d", "tan_d", "sin2_d", "cos2_d") and "2θ" in in_h.replace(" ", "") and "2θ" not in grid.headers[target].replace(" ", ""):
            findings.append(TableFinding(
                "WARNING", "TWO_THETA", None, grid.headers[target],
                f"{grid.headers[target]} は 2θ をそのまま使った「{best.formula}」と一致します。θ = 2θ/2 を使うべき式ではありませんか。",
                "ブラッグの式では θ（2θ の半分）を使います。",
            ))
    if relations:
        findings.append(TableFinding(
            "PASS", "RELATION", None, None,
            "推定した計算関係：" + " ／ ".join(f"{short_name(r.target)} = {r.formula}（{r.matched_rows}/{r.checked_rows}行一致）" for r in relations),
        ))

    # ---- 2) 集計行（平均・合計・標準偏差 …）
    for ri, kind in grid.summary.items():
        for ci in grid.numeric_cols:
            cell = grid.cells[ri][ci]
            col = [c for c in grid.col(ci) if c is not None]
            if cell is None or len(col) < 2:
                continue
            vals = [c.value for c in col]
            h_in = max(c.half_ulp for c in col)
            cands: Dict[str, float] = {}
            if kind == "mean":
                cands["平均"] = sum(vals) / len(vals)
                tol = h_in
            elif kind == "sum":
                cands["合計"] = sum(vals)
                tol = sum(c.half_ulp for c in col)
            elif kind in ("sd", "se"):
                s1, s0 = statistics.stdev(vals), statistics.pstdev(vals)
                if kind == "se":
                    s1, s0 = s1 / math.sqrt(len(vals)), s0 / math.sqrt(len(vals))
                cands["標本(n−1)"] = s1
                cands["母(n)"] = s0
                tol = h_in * 2.0
            elif kind == "max":
                cands["最大"] = max(vals)
                tol = 0.0
            else:
                cands["最小"] = min(vals)
                tol = 0.0
            tol = (tol + cell.half_ulp) * TOL_SLACK + 1e-12 * max(1.0, abs(cell.value))
            ok = {name: v for name, v in cands.items() if abs(v - cell.value) <= tol}
            label = {"mean": "平均", "sum": "合計", "sd": "標準偏差", "se": "標準誤差", "max": "最大", "min": "最小"}[kind]
            if ok:
                if kind in ("sd", "se") and len(ok) == 1 and len(col) <= 10:
                    which = next(iter(ok))
                    findings.append(TableFinding(
                        "INFO", "SUMMARY_ROW", ri + 1, grid.headers[ci],
                        f"{label}の行は{which}の式で一致します。",
                        "n が小さいと n と n−1 で値が大きく変わります。どちらで計算したかを本文に書いてください。",
                    ))
                continue
            v_show = " / ".join(f"{name} {v:.6g}" for name, v in cands.items())
            dev = min(abs(v - cell.value) for v in cands.values())
            findings.append(TableFinding(
                _severity_for(dev, tol), "SUMMARY_ROW", ri + 1, grid.headers[ci],
                f"{label}の行の {grid.headers[ci]} は記載 {cell.raw} ですが、データ{len(col)}行から計算すると {v_show} です。",
                "集計に含める行（外れ値の除外など）と計算式を確認してください。",
            ))

    # ---- 3) 列内の桁数の不統一（同じ測定・計算なのに小数点以下の桁がばらばら）
    for ci in grid.numeric_cols:
        if ci in grid.index_cols:
            continue
        decs = [c.decimals for c in grid.col(ci) if c is not None and c.decimals]
        if len(decs) < 3:
            continue
        mode = max(set(decs), key=decs.count)
        odd = [data_rowno[p] for p, c in enumerate(grid.col(ci)) if c is not None and c.decimals and c.decimals != mode]
        sigs = [c.sig for c in grid.col(ci) if c is not None]
        # 有効数字が揃っていて小数桁だけ違う（桁の大きく違う値）場合は正常
        if odd and len(set(sigs)) > 1 and len(odd) <= max(1, len(decs) // 3):
            findings.append(TableFinding(
                "INFO", "DIGITS", None, grid.headers[ci],
                f"{grid.headers[ci]} 列の小数点以下の桁数が行{', '.join(map(str, odd))}だけ他（{mode}桁）と違います。",
                "末尾の0の書き落とし、または丸め忘れでないか確認してください。",
            ))

    return TableCheckResult(
        caption=caption,
        numeric_columns=[grid.headers[c] for c in grid.numeric_cols],
        data_rows=len(grid.data_idx),
        summary_rows=[f"行{ri+1}:{k}" for ri, k in grid.summary.items()],
        relations=relations,
        findings=findings,
    )


# ---------------------------------------------------------------------------
# テキスト（CSV / TSV / Markdown / スペース区切り）からの読み込み
# ---------------------------------------------------------------------------
def parse_table_text(text: str) -> Tuple[List[str], List[List[str]]]:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return [], []
    if lines[0].strip().startswith("|"):
        rows = []
        for ln in lines:
            if re.fullmatch(r"\s*\|?\s*:?-{2,}.*", ln):
                continue
            rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
        return rows[0], rows[1:]
    sample = "\n".join(lines[:5])
    commas = [ln.count(",") for ln in lines]
    if "\t" in sample:
        delim = "\t"
    elif commas[0] >= 1 and len(set(commas)) == 1:
        delim = ","
    else:
        rows = [re.split(r"\s{2,}|\s+", ln.strip()) for ln in lines]
        return rows[0], rows[1:]
    rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delim))
    return [c.strip() for c in rows[0]], [[c.strip() for c in r] for r in rows[1:]]


def check_table_text(text: str, caption: str = "表") -> TableCheckResult:
    headers, rows = parse_table_text(text)
    return check_table(headers, rows, caption)
