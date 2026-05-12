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
from pathlib import Path

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


def test_extract_docx_text_reads_word_document(tmp_path: Path) -> None:
    docx = tmp_path / "sample.docx"
    _write_docx(docx, "Sentient control paragraph")

    text = ingest_manuscripts.extract_docx_text(docx)

    assert "Sentient control paragraph" in text


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
