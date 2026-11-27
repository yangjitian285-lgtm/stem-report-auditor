"""Word(.docx)の図表・書式・名前空間を保持したまま安全にコピー修正＆再監査するパッチャー.

安全設計：
- 元ファイルは絶対に上書きしない（常に別ファイル *_revised.docx へ出力）
- Word特有の分割ラン（<w:r><w:t>の途中分割）をまたいだ文字列置換に対応
- 本文だけでなくヘッダー・フッター・脚注・文末脚注の置換にも対応
- 画像(word/media/*)・表・スタイル等のZIPエントリをそのままバイト保持
- ElementTree再保存で消える名前空間宣言（mc:Ignorable が参照する w15 等）を
  元のルート開始タグごと復元し、Wordの「読み取れない内容」エラーを防止
- 挿入段落は見出し・番号付き・セクション区切り・赤字テンプレの書式を引き継がず、本文書式で作成
- 挿入段落は既定で黄色ハイライトし、提出前に本人が読み直し・自分の言葉で書き直す対象として明示
- 修正後に自動で audit_stem_document を実行し、Before→Afterのスコア改善幅を返却
"""

from __future__ import annotations

import copy
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from .stem_auditor import audit_stem_document


W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

# 置換対象とするパーツ（本文＋ヘッダー/フッター/脚注/文末脚注）
_TEXT_PART_RE = re.compile(r"^word/(document|header\d*|footer\d*|footnotes|endnotes)\.xml$")

# 挿入段落にコピーしない段落書式・文字書式
_PPR_DROP = {"pStyle", "numPr", "sectPr", "outlineLvl", "pageBreakBefore", "framePr"}
_RPR_DROP = {"color", "highlight", "shd", "strike", "dstrike", "u", "vanish", "rStyle"}

INTEGRITY_NOTICE = (
    "挿入した段落は黄色ハイライトで示しています。提出前に必ず本人が内容を検証し、"
    "自分の言葉で書き直してからハイライトを外してください。実在しないデータ・調査・文献を書き加えることは研究不正にあたります。"
)


@dataclass
class PatchOperationResult:
    """パッチ適用結果のサマリー."""

    source_path: str
    output_path: str
    applied_replacements: List[Dict[str, Any]]
    applied_insertions: List[Dict[str, Any]]
    deleted_paragraphs: List[str]
    not_found_targets: List[str]
    before_audit: Optional[Dict[str, Any]] = None
    after_audit: Optional[Dict[str, Any]] = None
    score_delta: Optional[Dict[str, float]] = None
    integrity_notice: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# XMLシリアライズ（名前空間・宣言の保全）
# ---------------------------------------------------------------------------
def _register_original_namespaces(xml_bytes: bytes) -> None:
    """元XMLで使われている全名前空間プレフィックスをETへ登録する（ns0等への改名防止）."""
    seen: Dict[str, str] = {}
    for _, elem in ET.iterparse(io.BytesIO(xml_bytes), events=("start-ns",)):
        prefix, uri = elem
        if prefix not in seen:
            seen[prefix] = uri
            try:
                ET.register_namespace(prefix, uri)
            except ValueError:
                pass


_ROOT_TAG_RE = re.compile(rb"<(?!\?)(?!!)[^>]*>", re.DOTALL)
_DECL_RE = re.compile(rb"^\s*<\?xml[^>]*\?>", re.DOTALL)


def _serialize_preserving_root(root: ET.Element, original_bytes: bytes) -> bytes:
    """ETで再シリアライズし、ルート開始タグとXML宣言を元のバイト列に戻す.

    ElementTree は「本文で使われていない名前空間」の宣言を出力しないため、
    mc:Ignorable="w14 w15 ..." が未宣言の接頭辞を参照する壊れたXMLになりうる。
    元のルート開始タグ（全宣言を含む）をそのまま戻すことでこれを防ぐ。
    """
    body = ET.tostring(root, encoding="utf-8")
    # ET が付けたXML宣言を除去
    body = _DECL_RE.sub(b"", body, count=1).lstrip()
    m_new = _ROOT_TAG_RE.search(body)
    m_old = _ROOT_TAG_RE.search(_DECL_RE.sub(b"", original_bytes, count=1))
    if m_new and m_old:
        old_tag = m_old.group(0)
        new_tag = m_new.group(0)
        # 元では子要素で局所宣言されていた名前空間を ET がルートへ集約することがあるため、
        # 元のルートにない宣言は追加で引き継ぐ（未束縛プレフィックスを防ぐ）
        old_decls = set(re.findall(rb'xmlns(?::[\w\-.]+)?="[^"]*"', old_tag))
        old_prefixes = {d.split(b"=")[0] for d in old_decls}
        extra = [
            d for d in re.findall(rb'xmlns(?::[\w\-.]+)?="[^"]*"', new_tag)
            if d.split(b"=")[0] not in old_prefixes
        ]
        if extra:
            close = b"/>" if old_tag.endswith(b"/>") else b">"
            old_tag = old_tag[: -len(close)].rstrip() + b" " + b" ".join(extra) + close
        if new_tag.endswith(b"/>") and not old_tag.endswith(b"/>"):
            old_tag = old_tag[:-1] + b"/>"
        body = body[: m_new.start()] + old_tag + body[m_new.end():]
    m_decl = _DECL_RE.search(original_bytes)
    decl = m_decl.group(0).strip() if m_decl else b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    return decl + b"\r\n" + body


def _missing_ignorable_prefixes(xml_bytes: bytes) -> List[str]:
    """mc:Ignorable が参照しているのに宣言されていない接頭辞を返す（自己検査用）."""
    m_root = _ROOT_TAG_RE.search(_DECL_RE.sub(b"", xml_bytes, count=1))
    if not m_root:
        return []
    tag = m_root.group(0).decode("utf-8", "replace")
    declared = set(re.findall(r'xmlns:([\w\-]+)=', tag))
    m_ign = re.search(r'Ignorable="([^"]*)"', tag)
    if not m_ign:
        return []
    return [p for p in m_ign.group(1).split() if p not in declared]


# ---------------------------------------------------------------------------
# スタイル判定
# ---------------------------------------------------------------------------
def _heading_style_ids(styles_xml: Optional[bytes]) -> Set[str]:
    """styles.xml から見出し・タイトル系の段落スタイルIDを抽出する."""
    ids: Set[str] = set()
    if not styles_xml:
        return ids
    try:
        st_root = ET.fromstring(styles_xml)
    except ET.ParseError:
        return ids
    for st in st_root.iter(f"{W_NS}style"):
        if st.attrib.get(f"{W_NS}type") != "paragraph":
            continue
        sid = st.attrib.get(f"{W_NS}styleId", "")
        name_el = st.find(f"{W_NS}name")
        name = (name_el.attrib.get(f"{W_NS}val", "") if name_el is not None else "").lower()
        ppr = st.find(f"{W_NS}pPr")
        has_outline = ppr is not None and ppr.find(f"{W_NS}outlineLvl") is not None
        if has_outline or name.startswith("heading") or name in ("title", "subtitle") or "見出し" in name:
            ids.add(sid)
    return ids


def _para_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.iter(f"{W_NS}t"))


def _is_heading_para(p: ET.Element, heading_ids: Set[str]) -> bool:
    ppr = p.find(f"{W_NS}pPr")
    if ppr is None:
        return False
    if ppr.find(f"{W_NS}outlineLvl") is not None:
        return True
    ps = ppr.find(f"{W_NS}pStyle")
    return ps is not None and ps.attrib.get(f"{W_NS}val", "") in heading_ids


def _is_colored_template(p: ET.Element) -> bool:
    for rpr in p.iter(f"{W_NS}rPr"):
        col = rpr.find(f"{W_NS}color")
        if col is not None:
            val = (col.attrib.get(f"{W_NS}val") or "").upper()
            if val and val not in ("000000", "AUTO"):
                return True
    return False


def _pick_body_template(children: Sequence[ET.Element], idx: int, heading_ids: Set[str]) -> Optional[ET.Element]:
    """挿入位置の近くから『本文らしい段落』（見出しでも赤字テンプレでもない）を探す."""
    order = list(range(idx + 1, len(children))) + list(range(idx, -1, -1))
    for j in order:
        c = children[j]
        if c.tag != f"{W_NS}p":
            continue
        if not _para_text(c).strip():
            continue
        if _is_heading_para(c, heading_ids) or _is_colored_template(c):
            continue
        return c
    return None


def _build_new_paragraph(
    template_p: Optional[ET.Element],
    text: str,
    highlight: bool = True,
) -> ET.Element:
    """本文段落の書式を（見出し・番号・セクション区切り・色を除いて）継承した新段落を生成."""
    new_p = ET.Element(f"{W_NS}p")
    if template_p is not None:
        ppr = template_p.find(f"{W_NS}pPr")
        if ppr is not None:
            ppr_c = copy.deepcopy(ppr)
            for child in list(ppr_c):
                local = child.tag.split("}")[-1]
                if local in _PPR_DROP or local == "rPr":
                    ppr_c.remove(child)
            if len(ppr_c):
                new_p.append(ppr_c)

    new_r = ET.SubElement(new_p, f"{W_NS}r")
    rpr_new: Optional[ET.Element] = None
    if template_p is not None:
        first_r = template_p.find(f"{W_NS}r")
        if first_r is not None and first_r.find(f"{W_NS}rPr") is not None:
            rpr_new = copy.deepcopy(first_r.find(f"{W_NS}rPr"))
            for child in list(rpr_new):
                if child.tag.split("}")[-1] in _RPR_DROP:
                    rpr_new.remove(child)
    if highlight:
        if rpr_new is None:
            rpr_new = ET.Element(f"{W_NS}rPr")
        hl = ET.SubElement(rpr_new, f"{W_NS}highlight")
        hl.set(f"{W_NS}val", "yellow")
    if rpr_new is not None and len(rpr_new):
        new_r.append(rpr_new)

    new_t = ET.SubElement(new_r, f"{W_NS}t")
    new_t.text = text
    if text.startswith(" ") or text.endswith(" "):
        new_t.set(XML_SPACE, "preserve")
    return new_p


# ---------------------------------------------------------------------------
# 置換
# ---------------------------------------------------------------------------
def _replace_in_paragraph(p_elem: ET.Element, target: str, replacement: str) -> int:
    """単一<w:p>内で分割<w:t>ランをまたいで target を replacement に置換する."""
    t_nodes = list(p_elem.iter(f"{W_NS}t"))
    if not t_nodes:
        return 0

    # まず単一<w:t>内で完結している場合はそのまま置換（ラン書式を100%保持）
    single_hits = 0
    for tn in t_nodes:
        txt = tn.text or ""
        if target in txt:
            cnt = txt.count(target)
            tn.text = txt.replace(target, replacement)
            if tn.text.startswith(" ") or tn.text.endswith(" "):
                tn.set(XML_SPACE, "preserve")
            single_hits += cnt
    if single_hits > 0:
        return single_hits

    # 分割ランをまたぐ場合：文字オフセット対応で最小限の<w:t>だけ書き換える
    count = 0
    guard = 0
    while guard < 1000:
        guard += 1
        texts = [tn.text or "" for tn in t_nodes]
        full = "".join(texts)
        pos = full.find(target)
        if pos < 0:
            break
        end = pos + len(target)

        cur = 0
        replaced_inserted = False
        for tn, txt in zip(t_nodes, texts):
            nxt = cur + len(txt)
            if nxt <= pos or cur >= end:
                cur = nxt
                continue
            local_s = max(0, pos - cur)
            local_e = min(len(txt), end - cur)
            prefix = txt[:local_s]
            suffix = txt[local_e:]
            if not replaced_inserted:
                tn.text = prefix + replacement + suffix
                replaced_inserted = True
            else:
                tn.text = prefix + suffix
            if tn.text and (tn.text.startswith(" ") or tn.text.endswith(" ")):
                tn.set(XML_SPACE, "preserve")
            cur = nxt
        count += 1
        if target in replacement:
            break  # 置換後文字列に target が含まれる場合の無限ループ防止
    return count


def patch_docx_safe(
    source_path: str,
    output_path: Optional[str] = None,
    replacements: Optional[Sequence[Dict[str, str]]] = None,
    insertions: Optional[Sequence[Dict[str, str]]] = None,
    delete_containing: Optional[Sequence[str]] = None,
    auto_reaudit: bool = True,
    highlight_insertions: bool = True,
) -> PatchOperationResult:
    """元.docxを保護しつつコピー先へ安全パッチを適用し、任意でBefore/After再監査を行う."""
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    if src.suffix.lower() != ".docx":
        raise ValueError("patch_docx_safe supports .docx files only.")

    out = Path(output_path) if output_path else src.with_name(f"{src.stem}_revised.docx")
    if out.suffix.lower() != ".docx":
        raise ValueError("output_path must end with .docx")
    if out.resolve() == src.resolve():
        raise ValueError(
            "安全保護のため元ファイルへの直接上書きは禁止されています。別の output_path を指定してください。"
        )

    before_rep = audit_stem_document(str(src)) if auto_reaudit else None

    applied_reps: List[Dict[str, Any]] = []
    applied_ins: List[Dict[str, Any]] = []
    deleted_paras: List[str] = []
    not_found: List[str] = []
    warnings: List[str] = []
    new_parts: Dict[str, bytes] = {}

    with zipfile.ZipFile(src, "r") as zin:
        names = zin.namelist()
        styles_xml = zin.read("word/styles.xml") if "word/styles.xml" in names else None
        heading_ids = _heading_style_ids(styles_xml)

        part_names = [n for n in names if _TEXT_PART_RE.match(n)]
        part_names.sort(key=lambda n: (n != "word/document.xml", n))
        rep_hits: Dict[str, int] = {}

        for part in part_names:
            raw = zin.read(part)
            _register_original_namespaces(raw)
            root = ET.fromstring(raw)
            changed = False

            # 1) テキスト置換（段落・表セル・ヘッダー等すべての<w:p>）
            all_paras = list(root.iter(f"{W_NS}p"))
            for rep in replacements or []:
                tgt = rep.get("target", "")
                val = rep.get("replacement", "")
                if not tgt:
                    continue
                hits = sum(_replace_in_paragraph(p, tgt, val) for p in all_paras)
                if hits:
                    rep_hits[tgt] = rep_hits.get(tgt, 0) + hits
                    changed = True

            if part == "word/document.xml":
                body = root.find(f"{W_NS}body")
                if body is None:
                    raise ValueError("Invalid docx: missing w:body element.")
                parent_map = {c: p for p in root.iter() for c in p}

                # 2) 段落削除（本文直下＋表セル内。セル最後の段落は空にしてセル構造を保つ）
                if delete_containing:
                    keys = [k for k in delete_containing if k]
                    for p in list(root.iter(f"{W_NS}p")):
                        p_txt = _para_text(p).strip()
                        if not p_txt or not any(k in p_txt for k in keys):
                            continue
                        parent = parent_map.get(p)
                        if parent is None:
                            continue
                        if p.find(f".//{W_NS}sectPr") is not None:
                            warnings.append(f"セクション区切りを含む段落は削除せず本文のみ消去: {p_txt[:30]}")
                            for t in p.iter(f"{W_NS}t"):
                                t.text = ""
                        elif parent.tag == f"{W_NS}tc" and len(parent.findall(f"{W_NS}p")) <= 1:
                            for t in p.iter(f"{W_NS}t"):
                                t.text = ""
                        else:
                            parent.remove(p)
                        deleted_paras.append(p_txt[:60])
                        changed = True

                # 3) アンカー段落の直後へ新段落を挿入
                for ins in insertions or []:
                    anchor_key = ins.get("after_text", "")
                    new_txt = ins.get("new_paragraph", "").strip()
                    if not anchor_key or not new_txt:
                        continue
                    inserted = False
                    children = list(body)
                    for idx, child in enumerate(children):
                        if child.tag != f"{W_NS}p":
                            continue
                        if anchor_key in _para_text(child):
                            template = _pick_body_template(children, idx, heading_ids)
                            new_p = _build_new_paragraph(template, new_txt, highlight=highlight_insertions)
                            body.insert(idx + 1, new_p)
                            applied_ins.append(
                                {
                                    "after_text": anchor_key,
                                    "new_paragraph_preview": new_txt[:60],
                                    "highlighted": highlight_insertions,
                                }
                            )
                            inserted = True
                            changed = True
                            break
                    if not inserted:
                        not_found.append(f"insert_after:{anchor_key}")

            if changed:
                new_bytes = _serialize_preserving_root(root, raw)
                missing = _missing_ignorable_prefixes(new_bytes)
                if missing:
                    raise RuntimeError(f"{part}: 名前空間の復元に失敗しました（未宣言: {missing}）。出力を中止します。")
                new_parts[part] = new_bytes

        for rep in replacements or []:
            tgt = rep.get("target", "")
            if not tgt:
                continue
            if rep_hits.get(tgt):
                applied_reps.append({"target": tgt, "replacement": rep.get("replacement", ""), "count": rep_hits[tgt]})
            else:
                not_found.append(f"replace:{tgt}")

        # 全ZIPエントリを無劣化コピーし、変更したパーツだけ差し替える
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename in new_parts:
                    zout.writestr(item, new_parts[item.filename])
                else:
                    zout.writestr(item, zin.read(item.filename))

    after_rep = audit_stem_document(str(out)) if auto_reaudit else None
    before_summary = None
    after_summary = None
    delta = None
    if before_rep and after_rep:

        def _summ(rep_obj: Any) -> Dict[str, Any]:
            return {
                "overall_score": rep_obj.overall_score,
                "deterministic_score": rep_obj.deterministic_score,
                "discussion_score": rep_obj.discussion_score,
                "grade": rep_obj.overall_grade,
                "critical_count": sum(1 for f in rep_obj.findings if f.severity == "CRITICAL"),
                "warning_count": sum(1 for f in rep_obj.findings if f.severity == "WARNING"),
            }

        before_summary = _summ(before_rep)
        after_summary = {**_summ(after_rep), "remaining_checklist": after_rep.priority_checklist[:5]}

        def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
            if a is None or b is None:
                return None
            return round(b - a, 1)

        delta = {
            "overall_change": _diff(before_rep.overall_score, after_rep.overall_score),
            "deterministic_change": _diff(before_rep.deterministic_score, after_rep.deterministic_score),
            "discussion_change": _diff(before_rep.discussion_score, after_rep.discussion_score),
        }

    return PatchOperationResult(
        source_path=str(src),
        output_path=str(out),
        applied_replacements=applied_reps,
        applied_insertions=applied_ins,
        deleted_paragraphs=deleted_paras,
        not_found_targets=not_found,
        before_audit=before_summary,
        after_audit=after_summary,
        score_delta=delta,
        integrity_notice=INTEGRITY_NOTICE if applied_ins else "",
        warnings=warnings,
    )
