"""Word パッチャーの回帰テスト（名前空間の保全・本文書式での挿入・ハイライト）."""

import re
import zipfile

import pytest

from conftest import sample
from uniqueness_scorer.docx_patcher import _missing_ignorable_prefixes, patch_docx_safe


@pytest.fixture()
def patched(tmp_path):
    out = tmp_path / "out.docx"
    res = patch_docx_safe(
        str(sample("crystal_report.docx")),
        str(out),
        replacements=[{"target": "(100)", "replacement": "(110)"}],
        insertions=[{"after_text": "格子定数aの平均値", "new_paragraph": "テスト挿入段落。"}],
        auto_reaudit=False,
    )
    return res, out


def test_ignorable_namespaces_preserved(patched):
    _, out = patched
    with zipfile.ZipFile(out) as z:
        for name in z.namelist():
            if re.match(r"word/(document|header\d*|footer\d*|footnotes|endnotes)\.xml$", name):
                assert _missing_ignorable_prefixes(z.read(name)) == [], name


def test_xml_is_wellformed_with_strict_parser(patched):
    lxml = pytest.importorskip("lxml.etree")
    _, out = patched
    with zipfile.ZipFile(out) as z:
        for name in z.namelist():
            if name.endswith(".xml"):
                lxml.fromstring(z.read(name))


def test_inserted_paragraph_is_body_style_and_highlighted(patched):
    docx = pytest.importorskip("docx")
    res, out = patched
    assert res.applied_insertions and res.integrity_notice
    d = docx.Document(str(out))
    para = next(p for p in d.paragraphs if "テスト挿入段落" in p.text)
    assert not para.style.name.lower().startswith("heading")
    assert all(r.font.highlight_color is not None for r in para.runs)


def test_refuses_overwrite(tmp_path):
    src = sample("crystal_report.docx")
    with pytest.raises(ValueError):
        patch_docx_safe(str(src), str(src), auto_reaudit=False)
