"""Word(.docx) / PDF / Markdown / Text から段落・構造化表・テンプレ残骸を自動抽出するパーサー.

外部パッケージなし（Python標準ライブラリのみ）で .docx の段落・表・文字色（赤字/青字のテンプレ指示文）を
厳密に構造化抽出する。PDFは pypdf / pymupdf(fitz) / pdfplumber / pdftotext があれば自動使用する。
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


@dataclass
class ParsedTable:
    """レポートから抽出された構造化テーブル."""

    index: int
    caption: str
    headers: List[str]
    rows: List[List[str]]
    context_before: str = ""

    def col_values(self, col_idx: int) -> List[str]:
        return [r[col_idx] if col_idx < len(r) else "" for r in self.rows]

    def col_floats(self, col_idx: int) -> List[Optional[float]]:
        return [parse_sci_number(v) for v in self.col_values(col_idx)]


@dataclass
class ParsedDocument:
    """構造化抽出されたレポート全体."""

    source_path: str
    file_type: str
    paragraphs: List[str]
    tables: List[ParsedTable]
    full_text: str
    body_text: str  # 表セルを除いた本文テキスト
    discussion_text: str  # 自動抽出した考察・結論セクション
    colored_template_lines: List[str] = field(default_factory=list)


def parse_sci_number(text: str) -> Optional[float]:
    """表セルや本文中の数値（指数表記 3.73*10^4, 1.19×10^4, 3.73e4 含む）をfloatへ変換."""
    s = text.strip().replace(",", "").replace("−", "-").replace("－", "-")
    if not s:
        return None
    # 3.73*10^4 / 3.73×10^4 / 3.73 x 10^4
    m_sci = re.fullmatch(
        r"([+-]?\d+(?:\.\d+)?)\s*[\*×xX]\s*10\^?\s*\(?([+-]?\d+)\)?",
        s,
    )
    if m_sci:
        return float(m_sci.group(1)) * (10.0 ** int(m_sci.group(2)))
    m_num = re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", s)
    if m_num:
        return float(m_num.group(0))
    return None


def _is_rubric_table(headers: List[str], rows: List[List[str]]) -> bool:
    """表紙の大学採点ルーブリック表（A 書式/形式, B 原理...）をデータ表と区別する."""
    joined = " ".join(headers + [c for r in rows[:4] for c in r])
    return ("初歩的" in joined and "模範的" in joined) or ("書式/形式" in joined and "考察・結" in joined)


# 設問文・指示文らしい段落（考察として採点しない）
_PROMPT_LIKE_RE = re.compile(
    r"(?:ください|下さい|せよ|なさい|しなさい|述べよ|求めよ|示せ|答えよ|論ぜよ|記せ)[。．.！!]*\s*(?:[（(]\s*\d+\s*点\s*[)）])?\s*$"
    r"|[（(]\s*\d+\s*点\s*[)）]\s*$"
    r"|^(?:レポート課題|課題[:：]|提出期限|学籍番号|学生番号|氏名)"
)


def is_prompt_like(paragraph: str) -> bool:
    """設問文・指示文・表紙の記入欄など、学生の回答ではない段落か判定する."""
    p = paragraph.strip()
    if not p:
        return True
    if _PROMPT_LIKE_RE.search(p):
        return True
    # 「1. 〜を説明してください。」のような番号付き設問
    if re.match(r"^[\(（]?[0-9０-９一二三四五]+[\)）.．、]", p) and re.search(r"ください|せよ|なさい|求めよ", p):
        return True
    return False


def answer_paragraphs(paragraphs: List[str]) -> List[str]:
    """設問・指示文・見出し程度の短文を除いた、学生自身の回答段落を返す."""
    out: List[str] = []
    for p in paragraphs:
        t = p.strip()
        if len(t) < 15 or is_prompt_like(t):
            continue
        if any(k in t for k in ["赤字で示しています", "青字の注意書き", "最大10点加点", "削除してください", "記載してください"]):
            continue
        out.append(t)
    return out


def _extract_discussion_section(paragraphs: List[str]) -> str:
    """段落列から『考察・結論・まとめ』に該当する部分を自動抽出する."""
    disc_lines: List[str] = []
    in_disc = False
    for p in paragraphs:
        low = p.strip()
        if not low:
            continue
        # 参考文献に入ったら終了
        if re.match(r"^(?:\d+[\.\s　]*)?参考文", low):
            if in_disc:
                break
            continue
        # 表紙・ルーブリック説明をスキップ
        if any(k in low for k in ["このほか", "最大10点加点", "赤字で示しています", "青字の注意書き"]):
            continue
        # 考察・結論・まとめ開始トリガー
        if re.match(
            r"^(?:[345３４５][\.\s　]*|4\.\d+[\.\s　]*)?(?:実験結果および考察|考察|結論|まとめ|課題\s*[1-5１-５])",
            low,
        ):
            in_disc = True
        if in_disc:
            # 設問文そのものや短い表ラベルを除外しつつ考察を蓄積
            if len(low) >= 22 and not re.match(r"^(?:表|図)\s*[\d１-９]+$", low) and not is_prompt_like(low):
                disc_lines.append(low)

    if len("".join(disc_lines)) >= 120:
        return "\n".join(disc_lines)

    # フォールバック：後半45%の長め段落を考察とみなす
    body_long = [p for p in answer_paragraphs(paragraphs) if len(p) >= 25]
    if not body_long:
        return ""
    half = max(0, len(body_long) // 2)
    return "\n".join(body_long[half:])


def _parse_docx(path: Path) -> ParsedDocument:
    """Word(.docx)のword/document.xmlから段落・表・赤/青色テンプレ指示文を順序保持で抽出."""
    paragraphs: List[str] = []
    tables: List[ParsedTable] = []
    colored_templates: List[str] = []

    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))

    body = root.find(f"{W_NS}body")
    if body is None:
        return ParsedDocument(str(path), "docx", [], [], "", "", "")

    def _para_text_and_color(p_elem: ET.Element) -> Tuple[str, bool]:
        txt = "".join(t.text or "" for t in p_elem.iter(f"{W_NS}t")).strip()
        is_colored = False
        for rpr in p_elem.iter(f"{W_NS}rPr"):
            col = rpr.find(f"{W_NS}color")
            if col is not None:
                val = (col.attrib.get(f"{W_NS}val") or "").upper()
                # 赤系または青系（テンプレートの削除指示文によく使われる色）
                if val and val not in ("000000", "AUTO", "FFFFFF"):
                    if val.startswith(("FF00", "C000", "ED", "0000FF", "0070C0", "2F5496", "4472C4")):
                        is_colored = True
        return txt, is_colored

    table_counter = 0
    last_nonempty_para = ""

    for child in list(body):
        if child.tag == f"{W_NS}p":
            txt, is_col = _para_text_and_color(child)
            if txt:
                paragraphs.append(txt)
                last_nonempty_para = txt
                if is_col and len(txt) >= 12:
                    colored_templates.append(txt)
        elif child.tag == f"{W_NS}tbl":
            grid: List[List[str]] = []
            for tr in child.findall(f"{W_NS}tr"):
                row_cells: List[str] = []
                for tc in tr.findall(f"{W_NS}tc"):
                    cell_parts = [
                        "".join(t.text or "" for t in p.iter(f"{W_NS}t")).strip()
                        for p in tc.findall(f"{W_NS}p")
                    ]
                    row_cells.append(" ".join(x for x in cell_parts if x).strip())
                if any(c for c in row_cells):
                    grid.append(row_cells)
            if not grid:
                continue
            headers = grid[0]
            rows = grid[1:]
            if _is_rubric_table(headers, rows):
                continue
            # 1列のみの空枠や表紙テーブルは除外
            if len(headers) <= 1 or not rows:
                continue
            table_counter += 1
            caption = last_nonempty_para if re.search(r"表\s*[\d１-９]", last_nonempty_para) else f"表(自動#{table_counter})"
            tables.append(
                ParsedTable(
                    index=table_counter,
                    caption=caption,
                    headers=headers,
                    rows=rows,
                    context_before=last_nonempty_para,
                )
            )

    body_text = "\n".join(paragraphs)
    table_text_lines = [
        " | ".join(r) for tbl in tables for r in ([tbl.headers] + tbl.rows)
    ]
    full_text = body_text + ("\n" + "\n".join(table_text_lines) if table_text_lines else "")
    discussion = _extract_discussion_section(paragraphs)
    return ParsedDocument(
        source_path=str(path),
        file_type="docx",
        paragraphs=paragraphs,
        tables=tables,
        full_text=full_text,
        body_text=body_text,
        discussion_text=discussion,
        colored_template_lines=colored_templates,
    )


def _extract_pdf_raw_text(path: Path) -> str:
    """利用可能なPDFライブラリまたはpdftotextから生テキストを抽出する."""
    if importlib.util.find_spec("pypdf"):
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if importlib.util.find_spec("fitz"):
        import fitz  # type: ignore

        doc = fitz.open(str(path))
        return "\n".join(page.get_text("text") for page in doc)
    if importlib.util.find_spec("pdfplumber"):
        import pdfplumber  # type: ignore

        with pdfplumber.open(str(path)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    if shutil.which("pdftotext"):
        proc = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout
    raise RuntimeError(
        "PDF読み取りライブラリ（pypdf / pymupdf / pdfplumber）または pdftotext が見つかりません。"
        "pip install pypdf を実行するか、.docx / .txt 形式で渡してください。"
    )


def _extract_tables_from_plain_text(lines: List[str]) -> List[ParsedTable]:
    """PDF/テキスト中の『表N』直後の表ブロックをヒューリスティックに構造化する."""
    tables: List[ParsedTable] = []
    i = 0
    idx = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r"^表\s*[\d１-９]+$", line):
            caption = line
            ctx = lines[i - 1].strip() if i > 0 else ""
            j = i + 1
            raw_rows: List[List[str]] = []
            while j < len(lines):
                cand = lines[j].strip()
                if not cand:
                    if raw_rows:
                        break
                    j += 1
                    continue
                if re.match(r"^(?:図|表|課題|〇|○|\d+[\.\s])", cand) and raw_rows:
                    break
                cols = re.split(r"\s{2,}|\t|\s*\|\s*", cand)
                if len(cols) == 1:
                    # 単一スペース区切りの数値表（例: "MAA40:BMA60 8758 18988 2.17"）
                    toks = cand.split()
                    if len(toks) >= 2 and any(parse_sci_number(t) is not None for t in toks):
                        cols = toks
                    elif not raw_rows and len(toks) >= 2 and len(cand) <= 45:
                        cols = toks
                if len(cols) >= 2:
                    raw_rows.append(cols)
                    j += 1
                else:
                    if raw_rows:
                        break
                    j += 1
            if len(raw_rows) >= 2:
                idx += 1
                tables.append(
                    ParsedTable(
                        index=idx,
                        caption=caption,
                        headers=raw_rows[0],
                        rows=raw_rows[1:],
                        context_before=ctx,
                    )
                )
            i = j
            continue
        i += 1
    return tables


def parse_report_file(file_path: str) -> ParsedDocument:
    """Word(.docx) / PDF(.pdf) / Text(.txt, .md) ファイルを統一構造へパースする."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = path.suffix.lower()
    if ext == ".docx":
        return _parse_docx(path)
    if ext == ".pdf":
        raw = _extract_pdf_raw_text(path)
    else:
        raw = path.read_text(encoding="utf-8")

    lines = [ln.rstrip() for ln in raw.splitlines()]
    paragraphs = [ln.strip() for ln in lines if ln.strip()]
    tables = _extract_tables_from_plain_text(lines)
    discussion = _extract_discussion_section(paragraphs)
    return ParsedDocument(
        source_path=str(path),
        file_type=ext.lstrip(".") or "txt",
        paragraphs=paragraphs,
        tables=tables,
        full_text=raw,
        body_text="\n".join(paragraphs),
        discussion_text=discussion,
        colored_template_lines=[],
    )
