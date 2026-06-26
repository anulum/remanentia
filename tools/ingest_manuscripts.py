# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — manuscript corpus normaliser

"""Normalise manuscript corpora into Remanentia memory chunks."""

from __future__ import annotations

import argparse
import hashlib
import html.parser
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from defusedxml import ElementTree  # type: ignore[import-untyped]

BASE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.environ.get("REMANENTIA_MANUSCRIPTS_ROOT", BASE / "manuscripts"))
DEFAULT_OUTPUT = BASE / "memory" / "manuscripts"

TEXT_EXTENSIONS = {
    ".bat",
    ".bib",
    ".csv",
    ".htm",
    ".html",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".ps1",
    ".sh",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}
DOCUMENT_EXTENSIONS = {".docx", ".odt", ".pdf"}
MEDIA_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mp3",
    ".png",
    ".tif",
    ".tiff",
    ".wav",
    ".webp",
}
ARCHIVE_EXTENSIONS = {".7z", ".gz", ".rar", ".tar", ".tgz", ".zip"}
CHUNK_CHARS = 750_000
MIN_TEXT_CHARS = 50


@dataclass
class FileRecord:
    relpath: str
    path: str
    extension: str
    size: int
    sha256: str
    kind: str
    status: str
    output_paths: list[str] = field(default_factory=list)
    duplicate_of: str = ""
    error: str = ""


class _HTMLTextParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(relpath: Path, digest: str) -> str:
    stem = "".join(ch if ch.isalnum() else "-" for ch in relpath.as_posix().lower())
    stem = "-".join(part for part in stem.split("-") if part)
    return f"{stem[:140]}-{digest[:12]}"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".html", ".htm"}:
        parser = _HTMLTextParser()
        parser.feed(text)
        return parser.text()
    return text


def extract_docx_text(path: Path) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for member in (
            "word/document.xml",
            "word/footnotes.xml",
            "word/endnotes.xml",
            "word/comments.xml",
        ):
            if member not in archive.namelist():
                continue
            root = ElementTree.fromstring(archive.read(member))
            for para in root.iter():
                if _local_name(para.tag) != "p":
                    continue
                para_parts: list[str] = []
                for node in para.iter():
                    name = _local_name(node.tag)
                    if name == "t" and node.text:
                        para_parts.append(node.text)
                    elif name == "tab":
                        para_parts.append("\t")
                    elif name in {"br", "cr"}:
                        para_parts.append("\n")
                paragraph = "".join(para_parts).strip()
                if paragraph:
                    parts.append(paragraph)
    return "\n\n".join(parts)


def extract_odt_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        if "content.xml" not in archive.namelist():
            return ""
        root = ElementTree.fromstring(archive.read("content.xml"))
    parts: list[str] = []
    for para in root.iter():
        if _local_name(para.tag) != "p":
            continue
        paragraph = "".join(para.itertext()).strip()
        if paragraph:
            parts.append(paragraph)
    return "\n\n".join(parts)


def extract_pdf_text(path: Path, pdftotext_bin: str, timeout: float) -> str:
    result = subprocess.run(
        [pdftotext_bin, "-enc", "UTF-8", "-q", str(path), "-"],
        check=False,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"pdftotext exited with {result.returncode}")
    return result.stdout.decode("utf-8", errors="replace")


def _chunks(text: str, max_chars: int = CHUNK_CHARS) -> Iterable[str]:
    text = text.strip()
    while len(text) > max_chars:
        split_at = text.rfind("\n\n", 0, max_chars)
        if split_at < max_chars // 2:
            split_at = text.rfind("\n", 0, max_chars)
        if split_at < max_chars // 2:
            split_at = max_chars
        yield text[:split_at].strip()
        text = text[split_at:].strip()
    if text:
        yield text


def _write_chunks(
    text: str,
    source_path: Path,
    relpath: Path,
    digest: str,
    output_texts: Path,
) -> list[str]:
    slug = _slug(relpath, digest)
    output_paths: list[str] = []
    for index, chunk in enumerate(_chunks(text), start=1):
        output = output_texts / f"{slug}.part{index:03d}.md"
        header = (
            "---\n"
            f"source_path: {source_path.as_posix()}\n"
            f"relative_path: {relpath.as_posix()}\n"
            f"sha256: {digest}\n"
            f"part: {index}\n"
            "---\n\n"
        )
        output.write_text(header + chunk + "\n", encoding="utf-8")
        output_paths.append(str(output))
    return output_paths


def _kind(extension: str) -> str:
    if extension in TEXT_EXTENSIONS:
        return "text"
    if extension in DOCUMENT_EXTENSIONS:
        return extension.lstrip(".")
    if extension in MEDIA_EXTENSIONS:
        return "media"
    if extension in ARCHIVE_EXTENSIONS:
        return "archive"
    return "unknown"


def _is_archive_path(relpath: Path) -> bool:
    return any("ARCHIVE" in part.upper() for part in relpath.parts)


def _metadata_digest(relpath: Path, stat_result: os.stat_result) -> str:
    payload = f"{relpath.as_posix()}:{stat_result.st_size}:{stat_result.st_mtime_ns}".encode()
    return hashlib.sha256(payload).hexdigest()


def _sort_key(root: Path, path: Path, archive_text: bool) -> tuple[int, int, str]:
    relpath = path.relative_to(root)
    archive_rank = 0 if archive_text or not _is_archive_path(relpath) else 1
    kind = _kind(path.suffix.lower())
    kind_rank = {
        "text": 0,
        "docx": 1,
        "odt": 1,
        "pdf": 2,
        "media": 3,
        "archive": 4,
        "unknown": 5,
    }.get(kind, 5)
    return archive_rank, kind_rank, relpath.as_posix().lower()


def _extract_text(path: Path, extension: str, pdftotext_bin: str | None, pdf_timeout: float) -> str:
    if extension in TEXT_EXTENSIONS:
        return _read_text(path)
    if extension == ".docx":
        return extract_docx_text(path)
    if extension == ".odt":
        return extract_odt_text(path)
    if extension == ".pdf":
        if not pdftotext_bin:
            raise RuntimeError("pdftotext unavailable")
        return extract_pdf_text(path, pdftotext_bin, pdf_timeout)
    return ""


def normalise_tree(
    root: Path,
    output: Path,
    *,
    pdf_enabled: bool = True,
    pdf_timeout: float = 20.0,
    archive_text: bool = False,
    progress_every: int = 100,
    limit: int | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    output_texts = output / "texts"
    output.mkdir(parents=True, exist_ok=True)
    output_texts.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.jsonl"

    pdftotext_bin = shutil.which("pdftotext") if pdf_enabled else None
    seen: dict[str, str] = {}
    counts: dict[str, int] = {}
    total = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        files = sorted(
            (p for p in root.rglob("*") if p.is_file()),
            key=lambda candidate: _sort_key(root, candidate, archive_text),
        )
        print(json.dumps({"discovered": len(files)}), flush=True)
        for path in files:
            if limit is not None and total >= limit:
                break
            total += 1
            extension = path.suffix.lower()
            if not extension and path.name.upper() == "VERSION":
                extension = ".txt"
            kind = _kind(extension)
            relpath = path.relative_to(root)
            in_archive = _is_archive_path(relpath)
            stat_result = path.stat()
            metadata_only = (
                (in_archive and not archive_text)
                or kind in {"media", "archive", "unknown"}
                or (kind == "pdf" and not pdftotext_bin)
            )
            digest = _metadata_digest(relpath, stat_result) if metadata_only else _sha256(path)
            record = FileRecord(
                relpath=relpath.as_posix(),
                path=str(path),
                extension=extension,
                size=stat_result.st_size,
                sha256=digest,
                kind=kind,
                status="pending",
            )

            if not metadata_only and digest in seen:
                record.status = "duplicate"
                record.duplicate_of = seen[digest]
            elif in_archive and not archive_text:
                record.status = "archive_metadata_only"
            elif kind == "pdf" and not pdftotext_bin:
                record.status = "pdf_metadata_only"
            elif kind in {"media", "archive", "unknown"}:
                record.status = "metadata_only"
            else:
                try:
                    text = _extract_text(path, extension, pdftotext_bin, pdf_timeout)
                    if len(text.strip()) < MIN_TEXT_CHARS:
                        record.status = "too_short"
                    else:
                        record.output_paths = _write_chunks(
                            text, path, relpath, digest, output_texts
                        )
                        record.status = "indexed"
                    seen[digest] = relpath.as_posix()
                except Exception as exc:
                    record.status = "error"
                    record.error = str(exc)[:500]
                    seen[digest] = relpath.as_posix()

            counts[record.status] = counts.get(record.status, 0) + 1
            manifest.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            if progress_every > 0 and total % progress_every == 0:
                print(json.dumps({"processed": total, "counts": counts}), flush=True)

    summary = {
        "root": str(root),
        "output": str(output),
        "manifest": str(manifest_path),
        "processed": total,
        "counts": counts,
        "unique_sha256": len(seen),
        "pdf_enabled": pdf_enabled,
        "pdftotext": pdftotext_bin or "",
        "archive_text": archive_text,
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--pdf-timeout", type=float, default=20.0)
    parser.add_argument("--include-archive-text", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    summary = normalise_tree(
        args.root,
        args.output,
        pdf_enabled=not args.no_pdf,
        pdf_timeout=args.pdf_timeout,
        archive_text=args.include_archive_text,
        progress_every=args.progress_every,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
