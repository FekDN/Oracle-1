# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""Persistent local document library for Oracle-1 ingestion.

The queue is intentionally separate from the graph.  The graph answers
"which concepts and relationships exist?"; the queue answers "which source
documents were already converted into observations?"  Both a content digest
and bibliographic identity (embedded title + author) are retained, so moving
or renaming a document cannot make it silently enter the graph again.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET


SUPPORTED_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json",
    ".xml", ".html", ".htm", ".xhtml", ".fb2", ".epub", ".rtf", ".doc",
    ".docx", ".odt", ".pdf",
}
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    return re.sub(r"\s+", " ", value).strip()


def _clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\x00", " ").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", value)).strip()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _xml_text(xml_bytes: bytes) -> str:
    try:
        root = ET.fromstring(xml_bytes)
        return _clean_text(" ".join(part.strip() for part in root.itertext() if part and part.strip()))
    except ET.ParseError:
        decoded, _encoding = _decode_text_bytes(
            xml_bytes, _declared_charset(xml_bytes))
        decoded = re.sub(r"<[^>]+>", " ", decoded)
        return _clean_text(unescape(decoded))


def _element_text(root: ET.Element, local_name: str) -> str:
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1].lower() == local_name.lower() and element.text:
            return _clean_text(element.text)
    return ""


def _all_element_text(root: ET.Element, local_name: str) -> List[str]:
    result = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1].lower() == local_name.lower() and element.text:
            value = _clean_text(element.text)
            if value:
                result.append(value)
    return result


def _zip_read(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        return archive.read(name)
    except KeyError:
        return b""


def _decode_text_bytes(raw: bytes, declared_encoding: Optional[str] = None) -> Tuple[str, str]:
    """Decode text from BOMs, declared codecs, UTF-8, and common legacy codecs."""
    bom_encodings = (
        (codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16"),
    )
    for marker, encoding in bom_encodings:
        if raw.startswith(marker):
            return raw.decode(encoding), encoding

    tried: Set[str] = set()
    if declared_encoding:
        try:
            encoding = codecs.lookup(declared_encoding).name
            tried.add(encoding)
            return raw.decode(encoding), encoding
        except (LookupError, UnicodeDecodeError):
            pass

    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    detected = []
    try:
        from charset_normalizer import from_bytes
        detected = [match for match in from_bytes(raw) if match.encoding and match.chaos < 0.2]
    except (ImportError, LookupError, ValueError):
        pass

    candidates: List[Tuple[str, str]] = [
        (str(match), match.encoding) for match in detected
    ]
    for candidate in ("cp1251", "koi8-r", "cp866", "cp1252", "latin-1"):
        try:
            encoding = codecs.lookup(candidate).name
            if encoding in tried:
                continue
            tried.add(encoding)
            candidates.append((raw.decode(encoding), encoding))
        except (LookupError, UnicodeDecodeError):
            continue
    russian_candidates = [
        (text, encoding) for text, encoding in candidates
        if _russian_text_score(text) > 0.25
    ]
    if russian_candidates:
        return max(russian_candidates, key=lambda item: _russian_text_score(item[0]))
    if candidates:
        return candidates[0]
    raise UnicodeError("Unable to decode document with supported text encodings")


def _russian_text_score(text: str) -> float:
    letters = [char.lower() for char in text if char.isalpha()]
    if not letters:
        return 0.0
    cyrillic = [char for char in letters if "\u0400" <= char <= "\u052f"]
    ratio = len(cyrillic) / len(letters)
    if ratio < 0.08:
        return 0.0
    common_bigrams = {
        "ст", "но", "то", "на", "ен", "ов", "ни", "ра", "во", "ко",
        "пр", "ро", "ос", "по", "го", "ал", "ер", "ли", "ре", "че",
        "ри", "ет", "ла", "ве", "ть", "ан", "од", "ор", "ом", "ит",
    }
    bigrams = list(zip(cyrillic, cyrillic[1:]))
    common_ratio = sum(a + b in common_bigrams for a, b in bigrams) / max(1, len(bigrams))
    return ratio * 0.3 + common_ratio


def _read_text_file(path: Path) -> str:
    text, _encoding = _decode_text_bytes(path.read_bytes())
    return text


def _declared_charset(raw: bytes) -> Optional[str]:
    """HTML/XML documents may declare their own encoding — trust that over
    guessing, when present. The declaration itself is always ASCII-safe
    (charset names are ASCII), so scanning the raw bytes with a Latin-1
    decode (never raises) before the real encoding is known is safe."""
    head = raw[:4096].decode("latin-1", errors="replace")
    match = re.search(r'<meta[^>]+charset=["\']?\s*([\w-]+)', head, re.I)
    if not match:
        match = re.search(r'encoding=["\']([\w-]+)["\']', head, re.I)
    if not match:
        return None
    name = match.group(1).strip().lower()
    try:
        "x".encode(name)
        return name
    except LookupError:
        return None


def _extract_docx(path: Path) -> Tuple[str, str, List[str], Dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        document = _zip_read(archive, "word/document.xml")
        core = _zip_read(archive, "docProps/core.xml")
    text = _xml_text(document)
    title, authors = "", []
    if core:
        try:
            root = ET.fromstring(core)
            title = _element_text(root, "title")
            authors = _all_element_text(root, "creator") or _all_element_text(root, "lastModifiedBy")
        except ET.ParseError:
            pass
    return text, title, authors, {"container": "docx"}


def _extract_odt(path: Path) -> Tuple[str, str, List[str], Dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        content = _zip_read(archive, "content.xml")
        meta = _zip_read(archive, "meta.xml")
    text = _xml_text(content)
    title, authors = "", []
    if meta:
        try:
            root = ET.fromstring(meta)
            title = _element_text(root, "title")
            authors = _all_element_text(root, "initial-creator") or _all_element_text(root, "creator")
        except ET.ParseError:
            pass
    return text, title, authors, {"container": "odt"}


def _extract_fb2(path: Path) -> Tuple[str, str, List[str], Dict[str, Any]]:
    raw = path.read_bytes()
    try:
        root = ET.fromstring(raw)
        title = _element_text(root, "book-title")
        authors: List[str] = []
        for author in root.iter():
            if author.tag.rsplit("}", 1)[-1] != "author":
                continue
            pieces = [
                _element_text(author, "first-name"), _element_text(author, "middle-name"),
                _element_text(author, "last-name"), _element_text(author, "nickname"),
            ]
            value = " ".join(part for part in pieces if part).strip()
            if value:
                authors.append(value)
        return _clean_text(" ".join(part for part in root.itertext() if part)), title, authors, {"container": "fb2"}
    except ET.ParseError:
        return _xml_text(raw), "", [], {"container": "fb2", "warning": "Malformed XML"}


def _extract_epub(path: Path) -> Tuple[str, str, List[str], Dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        opf_name = next((name for name in names if name.lower().endswith(".opf")), "")
        opf = _zip_read(archive, opf_name) if opf_name else b""
        xhtml = [name for name in names if name.lower().endswith((".xhtml", ".html", ".htm"))]
        text = "\n".join(_xml_text(_zip_read(archive, name)) for name in xhtml)
    title, authors = "", []
    if opf:
        try:
            root = ET.fromstring(opf)
            title = _element_text(root, "title")
            authors = _all_element_text(root, "creator")
        except ET.ParseError:
            pass
    return _clean_text(text), title, authors, {"container": "epub"}


def _extract_html_or_xml(path: Path) -> Tuple[str, str, List[str], Dict[str, Any]]:
    raw = path.read_bytes()
    decoded, encoding = _decode_text_bytes(raw, _declared_charset(raw))
    title = ""
    author = ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", decoded, re.I | re.S)
    if title_match:
        title = _clean_text(re.sub(r"<[^>]+>", " ", unescape(title_match.group(1))))
    author_match = re.search(
        r"<meta[^>]+(?:name|property)=[\"'](?:author|dc\.creator)[\"'][^>]+content=[\"']([^\"']+)",
        decoded, re.I,
    )
    if author_match:
        author = _clean_text(author_match.group(1))
    if path.suffix.lower() == ".xml":
        try:
            root = ET.fromstring(raw)
            title = title or _element_text(root, "title")
            if not author:
                authors = _all_element_text(root, "author") or _all_element_text(root, "creator")
            else:
                authors = [author]
            return _xml_text(raw), title, authors, {"container": "xml", "encoding": encoding}
        except ET.ParseError:
            pass
    no_script = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", decoded, flags=re.I | re.S)
    text = _clean_text(unescape(re.sub(r"<[^>]+>", " ", no_script)))
    return text, title, [author] if author else [], {
        "container": "html", "encoding": encoding,
    }


def _extract_rtf(path: Path) -> Tuple[str, str, List[str], Dict[str, Any]]:
    raw = path.read_bytes()
    codepage = re.search(rb"\\ansicpg(\d+)", raw)
    declared = f"cp{codepage.group(1).decode('ascii')}" if codepage else None
    decoded, encoding = _decode_text_bytes(raw, declared)

    def decode_hex_run(match: re.Match) -> str:
        hex_bytes = re.findall(r"\\'([0-9a-fA-F]{2})", match.group(0))
        escaped = bytes.fromhex(" ".join(hex_bytes))
        return escaped.decode(encoding, errors="replace")

    decoded = re.sub(r"(?:\\'[0-9a-fA-F]{2})+", decode_hex_run, decoded)

    def decode_unicode_run(match: re.Match) -> str:
        units = [int(value) & 0xFFFF for value in re.findall(r"\\u(-?\d+)\??", match.group(0))]
        raw_units = b"".join(unit.to_bytes(2, "little") for unit in units)
        return raw_units.decode("utf-16le", errors="replace")

    decoded = re.sub(r"(?:\\u-?\d+\??)+", decode_unicode_run, decoded)
    decoded = re.sub(r"\\par[d]?\b", "\n", decoded)
    decoded = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", decoded)
    return _clean_text(re.sub(r"[{}]", " ", decoded)), "", [], {
        "container": "rtf", "quality": "best_effort", "encoding": encoding,
    }


def _extract_doc_best_effort(path: Path) -> Tuple[str, str, List[str], Dict[str, Any]]:
    """Read legacy binary .doc without pretending it is a reliable parser.

    ``antiword`` or LibreOffice is used when present.  On a minimal system we
    retain only printable UTF-16/Latin runs and explicitly label the result as
    best-effort instead of silently parsing opaque binary data as evidence.
    """
    for executable in ("antiword", "catdoc"):
        try:
            completed = subprocess.run([executable, str(path)], capture_output=True,
                                       text=True, encoding="utf-8", errors="replace", timeout=90)
            if completed.returncode == 0 and completed.stdout.strip():
                return _clean_text(completed.stdout), "", [], {"container": "doc", "quality": "native_tool"}
        except (OSError, subprocess.SubprocessError):
            continue
    raw = path.read_bytes()
    unicode_runs = re.findall(r"(?:[\x20-\x7e]\x00){8,}", raw)
    latin_runs = re.findall(rb"[\x20-\x7e]{24,}", raw)
    pieces = [run.decode("utf-16le", errors="ignore") for run in unicode_runs]
    pieces.extend(run.decode("cp1252", errors="ignore") for run in latin_runs)
    return _clean_text("\n".join(pieces)), "", [], {
        "container": "doc", "quality": "best_effort",
        "warning": "Legacy .doc parsed without antiword/catdoc; inspect extracted text before trusting ingestion.",
    }


def _extract_pdf_with_bundled_python(path: Path, document_python: Optional[str]) -> Tuple[str, str, List[str], Dict[str, Any]]:
    python = document_python or os.environ.get("ORACLE_DOCUMENT_PYTHON") or sys.executable
    code = r'''
import json, sys
from pypdf import PdfReader
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
p = sys.argv[1]
r = PdfReader(p)
m = r.metadata or {}
def clean(v): return str(v or '').strip()
authors = [clean(m.get('/Author'))] if clean(m.get('/Author')) else []
text = '\n'.join((page.extract_text() or '') for page in r.pages)
print(json.dumps({'text': text, 'title': clean(m.get('/Title')), 'authors': authors,
                  'pages': len(r.pages)}, ensure_ascii=False))
'''
    completed = subprocess.run(
        [python, "-c", code, str(path)], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "PDF extraction failed")[-1000:])
    data = json.loads(completed.stdout)
    return _clean_text(data.get("text", "")), _clean_text(data.get("title", "")), [
        _clean_text(author) for author in data.get("authors", []) if _clean_text(author)
    ], {"container": "pdf", "pages": data.get("pages", 0)}


@dataclass
class ExtractedDocument:
    path: Path
    text: str
    title: str
    authors: List[str]
    content_hash: str
    file_hash: str
    bibliographic_key: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    warning: str = ""

    @property
    def strong_bibliographic_identity(self) -> bool:
        return bool(self.title.strip() and self.authors)


def extract_document(path: Path, document_python: Optional[str] = None,
                     max_chars: int = 0) -> ExtractedDocument:
    """Extract one supported document and preserve source metadata for dedupe."""
    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document format: {suffix or '(none)'}")
    if path.stat().st_size > 250 * 1024 * 1024:
        raise ValueError("Document exceeds the 250 MB ingestion safety limit")

    if suffix in TEXT_SUFFIXES:
        text, encoding = _decode_text_bytes(path.read_bytes())
        title, authors, metadata = "", [], {"container": "text", "encoding": encoding}
    elif suffix == ".docx":
        text, title, authors, metadata = _extract_docx(path)
    elif suffix == ".odt":
        text, title, authors, metadata = _extract_odt(path)
    elif suffix == ".fb2":
        text, title, authors, metadata = _extract_fb2(path)
    elif suffix == ".epub":
        text, title, authors, metadata = _extract_epub(path)
    elif suffix == ".pdf":
        text, title, authors, metadata = _extract_pdf_with_bundled_python(path, document_python)
    elif suffix in {".html", ".htm", ".xhtml", ".xml"}:
        text, title, authors, metadata = _extract_html_or_xml(path)
    elif suffix == ".rtf":
        text, title, authors, metadata = _extract_rtf(path)
    else:  # .doc
        text, title, authors, metadata = _extract_doc_best_effort(path)

    text = _clean_text(text)
    # A zero limit means "whole document".  Oracle's extraction is based on
    # document-level context; silently cutting a book before the LLM sees it
    # would make distant causes and effects look unrelated.  Deployments with
    # a known hard context limit may opt in to a positive cap, which is
    # recorded in metadata rather than hidden.
    if max_chars and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
        metadata["truncated_to_chars"] = max_chars
    if len(text) < 40:
        raise ValueError("Extracted text is too short; likely scanned, encrypted, or unsupported")
    title = _clean_text(title)
    metadata["embedded_title_available"] = bool(title)
    title = title or path.stem
    authors = list(dict.fromkeys(_clean_text(author) for author in authors if _clean_text(author)))
    content_hash = _hash(_normalise(text))
    biblio = _hash(_normalise(title) + "\x1f" + "\x1f".join(sorted(_normalise(author) for author in authors))) if authors else ""
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    warning = str(metadata.get("warning", ""))
    return ExtractedDocument(path, text, title, authors, content_hash, file_hash, biblio, metadata, warning)


class DocumentQueue:
    """Atomic JSON queue/ledger for a single persistent Oracle library."""

    FORMAT_VERSION = 1

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.payload = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"format_version": self.FORMAT_VERSION, "updated_at": utc_now(), "documents": {}}
        with self.path.open(encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError(f"Unsupported or invalid document queue: {self.path}")
        payload.setdefault("documents", {})
        return payload

    def save(self) -> None:
        self.payload["updated_at"] = utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, self.path)

    def _biblio_match(self, document: ExtractedDocument) -> Optional[Dict[str, Any]]:
        if not document.strong_bibliographic_identity:
            return None
        for record in self.payload["documents"].values():
            if record.get("bibliographic_key") == document.bibliographic_key and record.get("state") in {"ingested", "ingested_empty"}:
                return record
        return None

    def select_new(
        self, root: Path, processed_document_ids: Optional[Set[str]] = None,
        allow_revisions: bool = False, retry_failed: bool = True,
        document_python: Optional[str] = None, max_chars: int = 0,
    ) -> Tuple[List[ExtractedDocument], Dict[str, int]]:
        root = root.resolve()
        if not root.is_dir():
            raise ValueError(f"Corpus directory does not exist: {root}")
        processed_document_ids = processed_document_ids or set()
        selected: List[ExtractedDocument] = []
        selected_content_hashes: Set[str] = set()
        stats = {"seen": 0, "queued": 0, "recovered_interrupted": 0,
             "duplicate_content": 0, "duplicate_bibliographic": 0,
             "unsupported": 0, "failed_extract": 0, "already_in_graph": 0}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            stats["seen"] += 1
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                stats["unsupported"] += 1
                continue
            try:
                document = extract_document(path, document_python=document_python, max_chars=max_chars)
            except Exception as exc:
                key = "unreadable:" + _hash(str(path.resolve()))
                self.payload["documents"][key] = {
                    "state": "failed_extract", "path": str(path), "last_error": str(exc),
                    "updated_at": utc_now(),
                }
                stats["failed_extract"] += 1
                continue
            unreadable_key = "unreadable:" + _hash(str(path.resolve()))
            unreadable = self.payload["documents"].get(unreadable_key)
            if (unreadable and unreadable.get("state") == "failed_extract"
                    and os.path.normcase(os.path.normpath(unreadable.get("path", "")))
                    == os.path.normcase(os.path.normpath(str(path)))):
                del self.payload["documents"][unreadable_key]
            current = self.payload["documents"].get(document.content_hash)
            if document.content_hash in processed_document_ids:
                stats["already_in_graph"] += 1
                self._upsert(document, "ingested", "reconciled_from_graph")
                continue
            if document.content_hash in selected_content_hashes:
                stats["duplicate_content"] += 1
                continue
            if current and current.get("state") in {"ingested", "ingested_empty"}:
                self._upsert(document, current.get("state", "ingested"), "duplicate_content")
                stats["duplicate_content"] += 1
                continue
            interrupted = bool(current and current.get("state") in {"processing", "queued"})
            if current and current.get("state") == "failed" and not retry_failed:
                stats["duplicate_content"] += 1
                continue
            duplicate = self._biblio_match(document)
            if duplicate and duplicate.get("content_hash") != document.content_hash and not allow_revisions:
                self._upsert(document, "duplicate_bibliographic", "same_embedded_title_and_author")
                stats["duplicate_bibliographic"] += 1
                continue
            self._upsert(document, "queued", "recovered_interrupted" if interrupted else "new_or_revision")
            selected.append(document)
            selected_content_hashes.add(document.content_hash)
            stats["queued"] += 1
            if interrupted:
                stats["recovered_interrupted"] += 1
        self.save()
        return selected, stats

    def _upsert(self, document: ExtractedDocument, state: str, reason: str) -> None:
        previous = self.payload["documents"].get(document.content_hash, {})
        paths = list(dict.fromkeys(previous.get("paths", []) + [str(document.path)]))
        self.payload["documents"][document.content_hash] = {
            **previous,
            "content_hash": document.content_hash,
            "file_hash": document.file_hash,
            "bibliographic_key": document.bibliographic_key,
            "title": document.title,
            "authors": document.authors,
            "paths": paths,
            "format": document.path.suffix.lower().lstrip("."),
            "metadata": document.metadata,
            "state": state,
            "reason": reason,
            "updated_at": utc_now(),
        }

    def mark(self, content_hashes: Iterable[str], state: str, reason: str = "") -> None:
        for content_hash in content_hashes:
            record = self.payload["documents"].get(content_hash)
            if record is None:
                continue
            record["state"] = state
            record["reason"] = reason
            record["updated_at"] = utc_now()
        self.save()

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.payload["documents"].values():
            state = str(record.get("state", "unknown"))
            counts[state] = counts.get(state, 0) + 1
        return counts
