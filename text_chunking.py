# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Text and source-code chunking

"""Split source documents into bounded retrieval units."""

from __future__ import annotations

import ast
import re

MAX_TEXT_PARAGRAPH_CHARS = 10_000
MAX_FALLBACK_TEXT_CHARS = 2_000
MAX_CODE_CHUNK_CHARS = 1000
MAX_CODE_CHUNKS = 200


def split_paragraphs(text: str, is_code: bool = False) -> list[str]:
    """Split prose or source code into searchable units."""
    if is_code:
        return split_code(text)

    paragraphs = []
    for block in text.split("\n\n"):
        stripped = block.strip()
        if len(stripped) < 30:
            continue
        if len(stripped) <= 200:
            paragraphs.append(stripped[:MAX_TEXT_PARAGRAPH_CHARS])
            continue

        sentences = split_sentences(stripped)
        if len(sentences) <= 2:
            paragraphs.append(stripped[:MAX_TEXT_PARAGRAPH_CHARS])
            continue

        for index in range(len(sentences)):
            start = max(0, index - 1)
            end = min(len(sentences), index + 2)
            window = " ".join(sentences[start:end])
            if len(window) > 30:
                paragraphs.append(window[:MAX_TEXT_PARAGRAPH_CHARS])

    if not paragraphs and len(text.strip()) > 30:
        paragraphs.append(text.strip()[:MAX_FALLBACK_TEXT_CHARS])
    return paragraphs


def split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries and discard tiny fragments."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [part.strip() for part in parts if len(part.strip()) > 10]


def split_code(text: str) -> list[str]:
    """Split source code into bounded function and class blocks."""
    python_chunks = split_python_code(text)
    if python_chunks:
        return python_chunks[:MAX_CODE_CHUNKS]

    chunks = []
    doc_match = re.search(r'"""(.*?)"""', text, re.DOTALL)
    if doc_match and doc_match.start() < 500:
        chunks.append(doc_match.group(1).strip()[:MAX_CODE_CHUNK_CHARS])

    for match in re.finditer(
        r"^((?:def|class|fn|pub fn|impl)\s+\w+.*?)(?=\n(?:def |class |fn |pub fn |impl |\Z))",
        text,
        re.MULTILINE | re.DOTALL,
    ):
        block = match.group(1).strip()
        if len(block) > 30:
            chunks.append(block[:MAX_CODE_CHUNK_CHARS])

    if not chunks:
        chunks.extend(
            block.strip()[:MAX_CODE_CHUNK_CHARS]
            for block in text.split("\n\n")
            if len(block.strip()) > 30
        )
    return chunks[:MAX_CODE_CHUNKS]


def split_python_code(text: str) -> list[str]:
    """Split Python into module, class, function, and method chunks."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines()
    chunks = []
    module_doc = ast.get_docstring(tree)
    if module_doc:
        chunks.append(module_doc.strip()[:MAX_CODE_CHUNK_CHARS])

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            block = _extract_python_block(lines, node)
            if len(block) > 30:
                chunks.append(block[:MAX_CODE_CHUNK_CHARS])
            continue
        if not isinstance(node, ast.ClassDef):
            continue

        class_header = f"class {node.name}:"
        class_doc = ast.get_docstring(node)
        if class_doc:
            chunks.append(f'{class_header}\n"""{class_doc.strip()}"""'[:MAX_CODE_CHUNK_CHARS])
        else:
            class_block = _extract_python_block(lines, node)
            if len(class_block) > 30:
                chunks.append(class_block[:MAX_CODE_CHUNK_CHARS])

        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_block = _extract_python_block(lines, child)
                if len(method_block) > 30:
                    chunks.append(f"{class_header}\n{method_block}"[:MAX_CODE_CHUNK_CHARS])

    return chunks[:MAX_CODE_CHUNKS]


def _extract_python_block(lines: list[str], node: ast.AST) -> str:
    start = max(getattr(node, "lineno", 1) - 1, 0)
    end = max(getattr(node, "end_lineno", start + 1), start + 1)
    return "\n".join(lines[start:end]).strip()
