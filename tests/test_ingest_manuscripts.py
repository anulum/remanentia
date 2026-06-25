# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — manuscript ingest tests

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from types import SimpleNamespace
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "ingest_manuscripts.py"
SPEC = importlib.util.spec_from_file_location("ingest_manuscripts", MODULE_PATH)
assert SPEC is not None
ingest_manuscripts = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["ingest_manuscripts"] = ingest_manuscripts
SPEC.loader.exec_module(ingest_manuscripts)


def _write_docx(path: Path, text: str) -> None:
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>{text}</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)


def _write_docx_with_controls(path: Path) -> None:
    document = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Alpha</w:t><w:tab/><w:t>Beta</w:t><w:br/><w:t>Gamma</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)


def _write_odt(path: Path, text: str) -> None:
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
 <office:body><office:text><text:p>{text}</text:p></office:text></office:body>
</office:document-content>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("content.xml", content)


def test_extract_docx_text_reads_word_document(tmp_path: Path) -> None:
    docx = tmp_path / "sample.docx"
    _write_docx(docx, "Sentient control paragraph")

    text = ingest_manuscripts.extract_docx_text(docx)

    assert "Sentient control paragraph" in text


def test_extract_docx_text_preserves_tabs_and_breaks(tmp_path: Path) -> None:
    docx = tmp_path / "controls.docx"
    _write_docx_with_controls(docx)

    text = ingest_manuscripts.extract_docx_text(docx)

    assert "Alpha\tBeta\nGamma" in text


def test_extract_odt_text_reads_content_xml(tmp_path: Path) -> None:
    odt = tmp_path / "sample.odt"
    _write_odt(odt, "Open document manuscript paragraph")

    assert "Open document manuscript paragraph" in ingest_manuscripts.extract_odt_text(odt)


def test_extract_odt_text_missing_content_returns_empty(tmp_path: Path) -> None:
    odt = tmp_path / "empty.odt"
    with zipfile.ZipFile(odt, "w") as archive:
        archive.writestr("meta.xml", "<meta />")

    assert ingest_manuscripts.extract_odt_text(odt) == ""


def test_read_text_strips_html_markup(tmp_path: Path) -> None:
    html = tmp_path / "page.html"
    html.write_text(
        "<html><body><h1>Title</h1><p>Paragraph text</p></body></html>", encoding="utf-8"
    )

    assert ingest_manuscripts._read_text(html) == "Title\nParagraph text"


def test_extract_pdf_text_success_and_failure(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")

    monkeypatch.setattr(
        ingest_manuscripts.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"PDF text", stderr=b""),
    )
    assert ingest_manuscripts.extract_pdf_text(pdf, "pdftotext", 1.0) == "PDF text"

    monkeypatch.setattr(
        ingest_manuscripts.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=b"", stderr=b"bad pdf"),
    )
    with pytest.raises(RuntimeError, match="bad pdf"):
        ingest_manuscripts.extract_pdf_text(pdf, "pdftotext", 1.0)


def test_chunks_prefers_paragraph_boundary_and_writes_metadata(tmp_path: Path) -> None:
    text = "A" * 20 + "\n\n" + "B" * 20
    chunks = list(ingest_manuscripts._chunks(text, max_chars=25))
    assert chunks == ["A" * 20, "B" * 20]

    output = tmp_path / "texts"
    output.mkdir()
    paths = ingest_manuscripts._write_chunks(
        "Manuscript content with sufficient body.",
        tmp_path / "source.md",
        Path("source.md"),
        "a" * 64,
        output,
    )
    written = Path(paths[0]).read_text(encoding="utf-8")
    assert "source_path:" in written
    assert "relative_path: source.md" in written


def test_chunks_falls_back_to_hard_split_when_no_boundaries() -> None:
    chunks = list(ingest_manuscripts._chunks("A" * 30, max_chars=10))

    assert chunks == ["A" * 10, "A" * 10, "A" * 10]


def test_kind_archive_detection_and_sort_key(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "ARCHIVE").mkdir(parents=True)
    archive_file = root / "ARCHIVE" / "old.md"
    active_file = root / "active.md"
    archive_file.write_text("old", encoding="utf-8")
    active_file.write_text("active", encoding="utf-8")

    assert ingest_manuscripts._kind(".jpg") == "media"
    assert ingest_manuscripts._kind(".zip") == "archive"
    assert ingest_manuscripts._kind(".weird") == "unknown"
    assert ingest_manuscripts._is_archive_path(Path("ARCHIVE/old.md")) is True
    assert ingest_manuscripts._sort_key(root, active_file, False) < ingest_manuscripts._sort_key(
        root, archive_file, False
    )


def test_extract_text_dispatches_supported_formats(tmp_path: Path, monkeypatch) -> None:
    text_file = tmp_path / "text.txt"
    text_file.write_text("Text manuscript paragraph.\n" * 4, encoding="utf-8")
    docx = tmp_path / "paper.docx"
    _write_docx(docx, "Docx paragraph for dispatch")
    odt = tmp_path / "paper.odt"
    _write_odt(odt, "Odt paragraph for dispatch")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(
        ingest_manuscripts,
        "extract_pdf_text",
        lambda path, pdftotext_bin, timeout: "PDF paragraph for dispatch",
    )

    assert "Text manuscript" in ingest_manuscripts._extract_text(text_file, ".txt", None, 1.0)
    assert "Docx paragraph" in ingest_manuscripts._extract_text(docx, ".docx", None, 1.0)
    assert "Odt paragraph" in ingest_manuscripts._extract_text(odt, ".odt", None, 1.0)
    assert (
        ingest_manuscripts._extract_text(pdf, ".pdf", "pdftotext", 1.0)
        == "PDF paragraph for dispatch"
    )
    assert ingest_manuscripts._extract_text(tmp_path / "image.jpg", ".jpg", None, 1.0) == ""
    with pytest.raises(RuntimeError, match="pdftotext unavailable"):
        ingest_manuscripts._extract_text(pdf, ".pdf", None, 1.0)


def test_normalise_tree_writes_chunks_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "01_MANUSCRIPTS"
    output = tmp_path / "memory" / "manuscripts"
    root.mkdir()
    (root / "paper.md").write_text("A measured manuscript paragraph.\n" * 4, encoding="utf-8")
    (root / "duplicate.md").write_text("A measured manuscript paragraph.\n" * 4, encoding="utf-8")
    _write_docx(
        root / "paper.docx",
        "Docx manuscript paragraph for retrieval with enough context to index.",
    )

    summary = ingest_manuscripts.normalise_tree(root, output, pdf_enabled=False)

    assert summary["processed"] == 3
    assert summary["counts"]["indexed"] == 2
    assert summary["counts"]["duplicate"] == 1
    records = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    indexed = [record for record in records if record["status"] == "indexed"]
    assert all(record["output_paths"] for record in indexed)
    output_paths = [path for record in indexed for path in record["output_paths"]]
    assert any(
        "Docx manuscript paragraph" in Path(path).read_text(encoding="utf-8")
        for path in output_paths
    )


def test_normalise_tree_records_metadata_only_and_errors(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "01_MANUSCRIPTS"
    output = tmp_path / "memory" / "manuscripts"
    (root / "ARCHIVE").mkdir(parents=True)
    (root / "ARCHIVE" / "old.md").write_text(
        "Archived text that is metadata only.", encoding="utf-8"
    )
    (root / "image.jpg").write_bytes(b"jpeg")
    (root / "unknown.bin").write_bytes(b"bin")
    (root / "paper.pdf").write_bytes(b"%PDF")
    (root / "bad.docx").write_bytes(b"not a zip")
    (root / "short.txt").write_text("tiny", encoding="utf-8")

    monkeypatch.setattr(ingest_manuscripts.shutil, "which", lambda name: None)

    summary = ingest_manuscripts.normalise_tree(root, output, pdf_enabled=True, progress_every=2)

    assert summary["counts"]["archive_metadata_only"] == 1
    assert summary["counts"]["metadata_only"] == 2
    assert summary["counts"]["pdf_metadata_only"] == 1
    assert summary["counts"]["error"] == 1
    assert summary["counts"]["too_short"] == 1


def test_normalise_tree_extracts_pdf_and_respects_limit(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "01_MANUSCRIPTS"
    output = tmp_path / "memory" / "manuscripts"
    root.mkdir()
    (root / "paper.pdf").write_bytes(b"%PDF")
    (root / "later.md").write_text("Later manuscript paragraph.\n" * 4, encoding="utf-8")
    monkeypatch.setattr(ingest_manuscripts.shutil, "which", lambda name: "pdftotext")
    monkeypatch.setattr(
        ingest_manuscripts,
        "extract_pdf_text",
        lambda path, pdftotext_bin, timeout: "PDF manuscript paragraph.\n" * 4,
    )

    summary = ingest_manuscripts.normalise_tree(root, output, limit=1)

    assert summary["processed"] == 1
    assert summary["counts"]["indexed"] == 1


def test_normalise_tree_treats_extensionless_version_as_text(tmp_path: Path) -> None:
    root = tmp_path / "01_MANUSCRIPTS"
    output = tmp_path / "memory" / "manuscripts"
    root.mkdir()
    (root / "VERSION").write_text(
        "Version note with enough manuscript context.\n" * 4, encoding="utf-8"
    )

    summary = ingest_manuscripts.normalise_tree(root, output, pdf_enabled=False)

    assert summary["counts"]["indexed"] == 1
    record = json.loads((output / "manifest.jsonl").read_text(encoding="utf-8"))
    assert record["extension"] == ".txt"


def test_main_prints_summary(tmp_path: Path, capsys) -> None:
    root = tmp_path / "root"
    output = tmp_path / "out"
    root.mkdir()
    (root / "paper.md").write_text("A measured manuscript paragraph.\n" * 4, encoding="utf-8")

    code = ingest_manuscripts.main(
        ["--root", str(root), "--output", str(output), "--no-pdf", "--progress-every", "0"]
    )

    captured = capsys.readouterr().out
    assert code == 0
    assert '"processed": 1' in captured
