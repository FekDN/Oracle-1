"""Build an auditable citation-only graph preview from local PDF references.

This is a seed for extending a corpus, not a substitute for article analysis:
each edge is created only from a DOI hyperlink (or printed DOI) in a PDF's
References section. OpenAlex is used for bibliographic labels and publication
years; it does not provide semantic/causal edge judgments here.

Example:
    python reference_graph_preview.py --corpus-dir "C:\\Users\\F\\Desktop\\зубы"
    python visualizer_server.py --snapshot artifacts/reference_preview/graph_snapshot.pkl --port 50510
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from pypdf import PdfReader

from oracle1 import (
    KnowledgeEdge,
    KnowledgeNode,
    RuntimeGraph,
    _year_to_ts,
    save_graph_snapshot,
)


logger = logging.getLogger("oracle1.references")
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s<>\"{}|\\^`]+", re.IGNORECASE)
REFERENCE_HEADING = re.compile(
    r"(?i)^\s*(?:references|bibliography|works cited|literature cited)"
    r"(?:\s+(?:and\s+notes|continued))?(?:\s+\d{1,3}[.)](?:\s+.*)?)?\s*$"
)


def normalize_doi(value: str) -> str:
    value = urllib.parse.unquote(str(value or "")).strip()
    if "://" in value:
        parsed = urllib.parse.urlparse(value)
        query = urllib.parse.parse_qs(parsed.query)
        doi_value = next((v[0] for k, v in query.items()
                          if k.lower() == "doi" and v), "")
        if doi_value:
            value = doi_value
        elif "doi.org" in parsed.netloc.lower():
            value = parsed.path.lstrip("/")
    value = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", value, flags=re.I)
    value = value.split("?", 1)[0].strip()
    value = value.split("#", 1)[0].strip()
    value = re.sub(r"(?i)&(?:amp;)?(?:domain|date_stamp|locale)=[^&]*.*$", "", value)
    value = re.sub(r"(?i)/full$", "", value)
    return value.rstrip("/.,;:)]}").lower()


def dois_from_uri(uri: str) -> List[str]:
    decoded = urllib.parse.unquote(uri or "")
    candidates = [match.group(0) for match in DOI_PATTERN.finditer(decoded)]
    if "doi=" in decoded.lower():
        query = urllib.parse.urlparse(decoded).query
        for key, values in urllib.parse.parse_qs(query).items():
            if key.lower() == "doi":
                candidates.extend(values)
    return list(dict.fromkeys(
        doi for candidate in candidates
        if (doi := normalize_doi(candidate)) and DOI_PATTERN.fullmatch(doi)
    ))


def page_links(page: Any) -> Iterable[str]:
    """Yield URI annotations while tolerating malformed PDF annotations."""
    try:
        annotations = page.get("/Annots") or []
    except Exception:
        return
    for annotation_ref in annotations:
        try:
            annotation = annotation_ref.get_object()
            action = annotation.get("/A")
            uri = action.get("/URI") if action else None
            if uri:
                yield str(uri)
        except Exception:
            continue


def pdf_identity(path: Path, reader: PdfReader, page_texts: List[str]) -> Dict[str, Any]:
    first_pages = "\n".join(page_texts[:2])
    own_dois: List[str] = []
    for page in reader.pages[:2]:
        for uri in page_links(page):
            own_dois.extend(dois_from_uri(uri))
    own_dois.extend(normalize_doi(x) for x in DOI_PATTERN.findall(first_pages))
    own_dois = list(dict.fromkeys(doi for doi in own_dois if doi))

    metadata = reader.metadata or {}
    title = str(metadata.get("/Title") or "").strip()
    if not title or title.lower() in {path.name.lower(), path.stem.lower()}:
        # If the PDF producer omitted /Title, use a plausible first-page title
        # line. OpenAlex metadata replaces this label whenever the PDF DOI resolves.
        title = ""
        for line in first_pages.splitlines():
            candidate = " ".join(line.split()).strip(" \t·•")
            if len(candidate) >= 25 and not candidate.lower().startswith((
                "doi:", "received:", "accepted:", "published:", "copyright",
                "abstract", "www.", "http:", "https:",
            )):
                title = candidate[:240]
                break
    title = title or path.stem

    year_match = re.search(r"\b(?:19|20)\d{2}\b", first_pages)
    year = int(year_match.group(0)) if year_match else None
    return {
        "doi": own_dois[0] if own_dois else "",
        "title": title,
        "year": year,
    }


def scan_pdf(path: Path) -> Dict[str, Any]:
    reader = PdfReader(str(path))
    page_texts = [(page.extract_text() or "") for page in reader.pages]
    identity = pdf_identity(path, reader, page_texts)
    ref_pages = [i for i, text in enumerate(page_texts)
                 if any(REFERENCE_HEADING.match(line.strip())
                        for line in text.splitlines())]
    if not ref_pages:
        return {
            "path": str(path.resolve()), **identity,
            "reference_start_page": None, "references": [],
            "warning": "No recognizable References heading; skipped citation links.",
        }

    start = min(ref_pages)
    cited: Dict[str, Set[int]] = defaultdict(set)
    for page_index in range(start, len(reader.pages)):
        page = reader.pages[page_index]
        for uri in page_links(page):
            for doi in dois_from_uri(uri):
                if doi != identity["doi"]:
                    cited[doi].add(page_index + 1)
        # Some PDF generators print DOI text without making it a link.
        for raw in DOI_PATTERN.findall(page_texts[page_index]):
            doi = normalize_doi(raw)
            if doi and doi != identity["doi"]:
                cited[doi].add(page_index + 1)

    return {
        "path": str(path.resolve()), **identity,
        "reference_start_page": start + 1,
        "references": [
            {"doi": doi, "pages": sorted(pages)}
            for doi, pages in sorted(cited.items())
        ],
        "warning": "",
    }


def fetch_openalex(doi_values: List[str], cache_path: Path, refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """Resolve DOI labels in batches of <=100 using OpenAlex's DOI filter."""
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists() and not refresh:
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cache = loaded
        except (OSError, ValueError):
            cache = {}

    missing = [doi for doi in doi_values if doi not in cache]
    for offset in range(0, len(missing), 100):
        chunk = missing[offset:offset + 100]
        filter_value = "doi:" + "|".join(f"https://doi.org/{doi}" for doi in chunk)
        query = urllib.parse.urlencode({
            "filter": filter_value,
            "per_page": "100",
            "select": "id,doi,display_name,publication_year,publication_date,type,open_access,best_oa_location",
        })
        request = urllib.request.Request(
            f"https://api.openalex.org/works?{query}",
            headers={"User-Agent": "Oracle1ReferencePreview/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"OpenAlex DOI lookup failed for batch {offset // 100 + 1}: {exc}"
            ) from exc
        for work in payload.get("results", []):
            doi = normalize_doi(work.get("doi", ""))
            if doi:
                cache[doi] = {
                    "title": work.get("display_name") or doi,
                    "year": work.get("publication_year"),
                    "publication_date": work.get("publication_date"),
                    "type": work.get("type"),
                    "open_access": work.get("open_access"),
                    "best_oa_location": work.get("best_oa_location"),
                    "openalex_url": work.get("id"),
                }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("resolved DOI metadata: %d/%d", min(offset + len(chunk), len(missing)), len(missing))

    return cache


def stable_id(doi: str) -> str:
    return "paper_" + hashlib.sha1(doi.encode("utf-8")).hexdigest()[:18]


def make_graph(documents: List[Dict[str, Any]], metadata: Dict[str, Dict[str, Any]]) -> Tuple[RuntimeGraph, Dict[str, Any]]:
    graph = RuntimeGraph()
    source_nodes: List[Dict[str, Any]] = []
    citation_pairs: Set[Tuple[str, str]] = set()
    source_ids_by_file: Dict[str, str] = {}

    def add_paper_node(doi: str, title: str, year: Optional[int], provenance: str) -> str:
        node_id = stable_id(doi) if doi else "paper_" + hashlib.sha1(provenance.encode("utf-8")).hexdigest()[:18]
        if node_id not in graph.nodes:
            timestamp = _year_to_ts(int(year)) if year else 0.0
            node = KnowledgeNode(
                id=node_id, text=title[:500], full_text="", node_type="research",
                domain="academic_research", entity_type="research_frontier",
                origin_kind="combinatorial", provenance=provenance,
                timestamp=timestamp, publication_date=str(year) if year else None,
            )
            node.__dict__["doi"] = doi or None
            node.__dict__["citation_preview"] = True
            graph.add_node(node)
        return node_id

    for doc in documents:
        if not doc.get("references"):
            continue
        doi = doc.get("doi", "")
        meta = metadata.get(doi, {})
        title = meta.get("title") or doc.get("title") or Path(doc["path"]).stem
        year = meta.get("year") or doc.get("year")
        source_id = add_paper_node(doi, title, year, Path(doc["path"]).name)
        source_ids_by_file[doc["path"]] = source_id
        source_nodes.append({"id": source_id, "doi": doi, "title": title,
                             "year": year, "file": Path(doc["path"]).name})

    for doc in documents:
        source_id = source_ids_by_file.get(doc["path"])
        if not source_id:
            continue
        source_node = graph.nodes[source_id]
        source_year = _year_to_ts(int(source_node.publication_date)) if source_node.publication_date else 0.0
        for ref in doc.get("references", []):
            doi = ref["doi"]
            if doi == doc.get("doi"):
                continue
            # An unresolved DOI-looking fragment (often a PDF line-wrap) is
            # kept in the manifest for review but must not become a graph node
            # or edge that looks authoritative.
            if metadata and doi not in metadata:
                continue
            meta = metadata.get(doi, {})
            title = meta.get("title") or doi
            year = meta.get("year")
            target_id = add_paper_node(doi, title, year, "OpenAlex DOI metadata")
            key = (source_id, target_id)
            if key in citation_pairs:
                continue
            citation_pairs.add(key)
            pages = ", ".join(map(str, ref.get("pages", [])))
            edge = KnowledgeEdge(
                id="cite_" + hashlib.sha1((doi + doc["path"]).encode("utf-8")).hexdigest()[:18],
                source_id=source_id, target_id=target_id,
                citation_link=1.0, confidence=1.0,
                relationship_type="cites", relation_type="cites",
                discovery_method="pdf_reference_doi",
                evidence=[f"DOI {doi} linked in {Path(doc['path']).name}, page(s) {pages}"],
                timestamp=source_year,
            )
            edge.compute_total_weight()
            graph.add_edge(edge)

    manifest = {
        "format": "oracle1_reference_preview_v1",
        "visualizer_mode": "citation_preview",
        "description": "Citation-only graph. DOI-backed reference edges are source evidence; no semantic LLM analysis was applied.",
        "source_documents": source_nodes,
        "doi_metadata_records": len(metadata),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "distinct_reference_dois": len({
            ref["doi"] for doc in documents for ref in doc.get("references", [])
        }),
        "unresolved_reference_dois": sorted({
            ref["doi"] for doc in documents for ref in doc.get("references", [])
            if ref["doi"] not in metadata
        }),
        "citation_sources": dict(Counter(
            graph.nodes[src].text for src, _ in citation_pairs
        )),
    }
    return graph, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/reference_preview"))
    parser.add_argument("--offline", action="store_true",
                        help="Do not resolve titles/years; DOI itself labels each referenced work.")
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument("--max-dois", type=int, default=0,
                        help="Optional deterministic cap for quick previews (0 means all).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = args.corpus_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"corpus directory does not exist: {root}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    documents = []
    for path in sorted(root.rglob("*.pdf")):
        try:
            record = scan_pdf(path)
            documents.append(record)
            logger.info("%s: %d DOI references%s", path.name, len(record["references"]),
                        " (no recognizable References section)" if record.get("warning") else "")
        except Exception as exc:
            logger.warning("skip %s: %s", path.name, exc)

    reference_dois = {ref["doi"] for doc in documents for ref in doc.get("references", [])}
    source_dois = {doc["doi"] for doc in documents if doc.get("doi")}
    if args.max_dois > 0:
        capped = set(sorted(reference_dois)[:args.max_dois])
        documents = [{**doc, "references": [r for r in doc.get("references", []) if r["doi"] in capped]}
                     for doc in documents]
        reference_dois = capped
    all_dois = sorted(reference_dois | source_dois)

    cache_path = output / "openalex_metadata.json"
    metadata = {} if args.offline else fetch_openalex(all_dois, cache_path, refresh=args.refresh_metadata)
    graph, manifest = make_graph(documents, metadata)
    manifest["corpus_dir"] = str(root)
    manifest["references_found"] = sum(len(doc.get("references", [])) for doc in documents)
    manifest["resolved_reference_edges"] = len(graph.edges)
    resolved_ref_dois = reference_dois - set(manifest["unresolved_reference_dois"])
    manifest["resolved_reference_dois"] = len(resolved_ref_dois)
    manifest["papers_with_reference_edges"] = sum(
        any(ref["doi"] in resolved_ref_dois for ref in doc.get("references", []))
        for doc in documents
    )
    manifest["documents"] = [{
        "file": Path(doc["path"]).name,
        "doi": doc.get("doi"), "title": doc.get("title"), "year": doc.get("year"),
        "reference_start_page": doc.get("reference_start_page"),
        "reference_doi_count": len(doc.get("references", [])),
        "warning": doc.get("warning", ""),
    } for doc in documents]
    reference_rows = []
    for doc in documents:
        for ref in doc.get("references", []):
            doi = ref["doi"]
            meta = metadata.get(doi, {})
            oa = meta.get("open_access") or {}
            location = meta.get("best_oa_location") or {}
            reference_rows.append({
                "source_file": Path(doc["path"]).name,
                "source_doi": doc.get("doi") or None,
                "doi": doi,
                "resolved": doi in metadata,
                "title": meta.get("title") or None,
                "year": meta.get("year"),
                "reference_pages": ref.get("pages", []),
                "is_open_access": oa.get("is_oa"),
                "open_access_status": oa.get("oa_status"),
                "open_access_landing_page": location.get("landing_page_url") or oa.get("oa_url"),
                "open_access_pdf_url": location.get("pdf_url"),
                "license": location.get("license"),
            })
    manifest["references"] = reference_rows
    manifest["open_access_pdf_candidates"] = sum(
        bool(item["resolved"] and item["is_open_access"] and item["open_access_pdf_url"])
        for item in reference_rows
    )

    snapshot_path = output / "graph_snapshot.pkl"
    save_graph_snapshot(snapshot_path, {"graph": graph})
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("saved %d nodes / %d DOI citation edges to %s",
                len(graph.nodes), len(graph.edges), snapshot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
