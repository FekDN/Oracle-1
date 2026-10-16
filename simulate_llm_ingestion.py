"""Run Oracle-1's real ingestion/update pipeline with a deterministic LLM simulator.

The simulator derives its structured responses from literal phrases in each
article's extracted text. It is useful for exercising parsing, validation,
canonicalization, consolidation, queueing, and RuntimeGraph behavior without an
LLM endpoint. Its generated semantics are synthetic and must not be treated as
research conclusions.

Example:
    python simulate_llm_ingestion.py --limit 24
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

import graph_update
from corpus_library import DocumentQueue
from oracle1 import CONFIG, ResponseValidator, graph_snapshot_summary, save_graph_snapshot
from oracle1_modules.label_status import parse_label_status


logger = logging.getLogger("oracle1.simulated_ingestion")
USER_AGENT = "Oracle1 local research prototype"

# Deliberately small ontology: exact occurrences are candidate concepts, not
# automatic findings. A candidate still needs a source-grounded result and a
# substantive role in the article before it becomes a node observation.
CONCEPTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("dental pulp stem cells", "research_frontier", ("dental pulp stem cells", "dental pulp stem cell", "dpscs", "dpsc")),
    ("dental pulp cells", "research_frontier", ("dental pulp cells",)),
    ("mesenchymal stem cells", "research_frontier", ("mesenchymal stem cells", "mscs")),
    ("adipose-derived stem cells", "research_frontier", ("adipose-derived stem cells", "adipose derived stem cells")),
    ("bone marrow stem cells", "research_frontier", ("bone marrow stem cells",)),
    ("periodontal ligament stem cells", "research_frontier", ("periodontal ligament stem cells",)),
    ("stem cell differentiation", "research_frontier", ("stem cell differentiation", "stem cells differentiate")),
    ("neuronal differentiation", "research_frontier", ("neuronal differentiation", "neural differentiation")),
    ("odontoblast differentiation", "research_frontier", ("odontoblast differentiation", "odontoblastic differentiation", "odontogenic differentiation")),
    ("dentin-pulp complex", "convergence_node", ("dentin-pulp complex", "dentin pulp complex", "pulp-dentin complex", "pulp dentin complex")),
    ("new bone formation", "research_frontier", ("new bone formation", "formation of new bone", "bone formation")),
    ("dental pulp regeneration", "breakthrough_event", ("dental pulp regeneration", "pulp regeneration")),
    ("dentin regeneration", "research_frontier", ("dentin regeneration",)),
    ("tissue engineering", "challenger_tech", ("tissue engineering",)),
    ("regenerative medicine", "research_frontier", ("regenerative medicine",)),
    ("extracellular matrix", "enabler_component", ("extracellular matrix",)),
    ("biomaterial scaffold", "enabler_component", ("biomaterial scaffold", "biomaterial scaffolds", "scaffold")),
    ("Wnt signaling", "research_frontier", ("wnt signaling", "wnt/β-catenin", "wnt/beta-catenin")),
    ("TGF-beta signaling", "research_frontier", ("tgf-β signaling", "tgf-β", "tgf-beta signaling", "tgf-beta")),
    ("BMP signaling", "research_frontier", ("bmp signaling", "bone morphogenetic protein")),
    ("platelet-derived growth factor", "enabler_component", ("platelet-derived growth factor", "pdgf-bb", "pdgf bb")),
    ("nerve growth factor", "enabler_component", ("nerve growth factor", "ngf")),
    ("microRNA regulation", "research_frontier", ("microrna", "micro-rna", "mirna")),
    ("cell proliferation", "research_frontier", ("cell proliferation",)),
    ("cell migration", "research_frontier", ("cell migration",)),
    ("angiogenesis", "research_frontier", ("angiogenesis",)),
    ("inflammation", "side_effect_risk", ("inflammation",)),
    ("cell culture", "measurement_tool", ("cell culture",)),
    ("in vitro model", "measurement_tool", ("in vitro model", "in-vitro model")),
    ("in vivo model", "measurement_tool", ("in vivo model", "in-vivo model")),
    ("clinical trial", "measurement_tool", ("clinical trial", "clinical trials")),
    ("gene expression", "research_frontier", ("gene expression",)),
    ("stem cell niche", "research_frontier", ("stem cell niche",)),
    ("dental pulp", "research_frontier", ("dental pulp",)),
    ("dentin", "research_frontier", ("dentin",)),
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Duplicate-filter audit failed: {message}")


def audit_duplicate_filtering(output_dir: Path) -> Dict[str, Any]:
    """Exercise DocumentQueue's content, graph-receipt, and bibliography filters.

    This uses tiny synthetic text/HTML fixtures in a temporary directory. It
    runs the real DocumentQueue.select_new()/mark() methods without touching
    the article corpus or the persistent graph.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="duplicate-filter-audit-", dir=output_dir) as temporary:
        root = Path(temporary)

        content_corpus = root / "content"
        content_corpus.mkdir()
        exact_text = (
            "A repeated document body used to test Oracle queue filtering. "
            "The content hash should be identical even when the filename differs."
        )
        (content_corpus / "a.txt").write_text(exact_text, encoding="utf-8")
        (content_corpus / "b.txt").write_text(exact_text, encoding="utf-8")
        content_queue_path = root / "content_queue.json"
        content_queue = DocumentQueue(content_queue_path)
        selected, first_scan = content_queue.select_new(content_corpus, processed_document_ids=set())
        _require(len(selected) == 1, f"identical files selected {len(selected)} times, expected once")
        _require(first_scan["queued"] == 1 and first_scan["duplicate_content"] == 1,
                 f"unexpected first content scan stats: {first_scan}")
        content_hash = selected[0].content_hash
        content_queue.mark([content_hash], "ingested", "duplicate_filter_audit")

        replay_selected, replay_scan = DocumentQueue(content_queue_path).select_new(
            content_corpus, processed_document_ids=set())
        _require(not replay_selected, "an already-ingested content hash was queued again")
        _require(replay_scan["duplicate_content"] == 2,
                 f"repeat scan did not filter both copies: {replay_scan}")

        graph_selected, graph_scan = DocumentQueue(content_queue_path).select_new(
            content_corpus, processed_document_ids={content_hash})
        _require(not graph_selected and graph_scan["already_in_graph"] == 2,
                 f"graph receipt did not suppress both copies: {graph_scan}")

        biblio_corpus = root / "bibliography"
        biblio_corpus.mkdir()
        original = biblio_corpus / "a_original.html"
        revision = biblio_corpus / "b_revision.html"
        head = ('<html><head><title>Queue Audit Study</title>'
                '<meta name="author" content="Ada Example"></head><body>')
        original.write_text(
            head + "Original article body with enough text for extraction and identity checks.</body></html>",
            encoding="utf-8")
        biblio_queue_path = root / "bibliography_queue.json"
        biblio_queue = DocumentQueue(biblio_queue_path)
        original_selected, original_scan = biblio_queue.select_new(
            biblio_corpus, processed_document_ids=set())
        _require(len(original_selected) == 1 and original_scan["queued"] == 1,
                 f"bibliography fixture was not queued: {original_scan}")
        original_hash = original_selected[0].content_hash
        biblio_queue.mark([original_hash], "ingested", "duplicate_filter_audit")
        revision.write_text(
            head + "Revised article body with different content but the same title and author identity.</body></html>",
            encoding="utf-8")
        revision_selected, revision_scan = DocumentQueue(biblio_queue_path).select_new(
            biblio_corpus, processed_document_ids=set())
        _require(not revision_selected and revision_scan["duplicate_bibliographic"] == 1,
                 f"same-title/same-author revision was not filtered: {revision_scan}")

        retry_corpus = root / "retry"
        retry_corpus.mkdir()
        retry_path = retry_corpus / "transient_failure.txt"
        retry_path.write_text(
            "A temporary ingestion failure fixture with enough text for queue extraction and retry checks.",
            encoding="utf-8")
        retry_queue_path = root / "retry_queue.json"
        retry_queue = DocumentQueue(retry_queue_path)
        failed_candidate, _ = retry_queue.select_new(retry_corpus, processed_document_ids=set())
        _require(len(failed_candidate) == 1, "retry fixture was not queued initially")
        failed_hash = failed_candidate[0].content_hash
        retry_queue.mark([failed_hash], "failed", "simulated_first_attempt_failure")
        blocked_retry, blocked_stats = DocumentQueue(retry_queue_path).select_new(
            retry_corpus, processed_document_ids=set(), retry_failed=False)
        _require(not blocked_retry and blocked_stats["duplicate_content"] == 1,
                 f"failed document was retried while retry_failed=False: {blocked_stats}")
        allowed_retry, allowed_stats = DocumentQueue(retry_queue_path).select_new(
            retry_corpus, processed_document_ids=set(), retry_failed=True)
        _require(len(allowed_retry) == 1 and allowed_stats["queued"] == 1,
                 f"failed document did not requeue when retry_failed=True: {allowed_stats}")

        return {
            "passed": True,
            "scope": "real DocumentQueue filters over temporary synthetic fixtures; no graph/corpus mutations",
            "identical_content": {
                "initial_queued": first_scan["queued"],
                "initial_duplicate_copy_filtered": first_scan["duplicate_content"],
                "repeat_scan_filtered": replay_scan["duplicate_content"],
                "processed_graph_receipts_filtered": graph_scan["already_in_graph"],
            },
            "bibliographic_revision": {
                "same_title_author_different_content_filtered": revision_scan["duplicate_bibliographic"],
                "original_ingested_before_revision_scan": True,
            },
            "failed_document_retry": {
                "retry_disabled_filtered": blocked_stats["duplicate_content"],
                "retry_enabled_queued": allowed_stats["queued"],
            },
        }


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
    return value[:40] or "source_concept"


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    words = re.split(r"[\s-]+", phrase.strip())
    return re.compile(r"(?<!\w)" + r"[\s\-–—]+".join(map(re.escape, words)) + r"(?!\w)", re.IGNORECASE)


def _content_before_references(text: str) -> str:
    # Keep the abstract and substantive article sections. Background prose is
    # useful to a real model for interpretation, but a mention there is not
    # this deterministic fixture's node-level finding. Bibliography items
    # must not become concepts or evidence for relationships.
    # Restore PDF words split at a line-end hyphen before section, concept,
    # citation, and sentence detection.
    text = re.sub(r"(?<=\w)-\s+(?=[a-z])", "", text)
    match = re.search(r"(?im)^\s*(references|bibliography|literature cited)\s*$", text)
    content = text[:match.start()] if match and match.start() > 1500 else text
    abstract = re.search(r"(?im)^\s*abstract\s*:?[ \t]*", content)
    if abstract:
        intro = re.search(
            r"(?im)^\s*(?:\d+[.)]?\s*)?(?:introduction|background)\s*$",
            content[abstract.end():],
        )
        if intro:
            abstract_text = content[abstract.end():abstract.end() + intro.start()]
            section_start = abstract.end() + intro.end()
            next_section = re.search(
                r"(?im)^\s*(?:\d+[.)]?\s*)?(?:materials?\s+and\s+methods?|methods?|"
                r"results?|findings?|discussion|conclusions?|summary)\s*$",
                content[section_start:],
            )
            if next_section:
                return abstract_text + "\n" + content[section_start + next_section.start():]
            return abstract_text
    introduction = re.search(r"(?im)^\s*(?:\d+[.)]?\s*)?(?:introduction|background)\s*$", content)
    if introduction:
        next_section = re.search(
            r"(?im)^\s*(?:\d+[.)]?\s*)?(?:materials?\s+and\s+methods?|methods?|"
            r"results?|findings?|discussion|conclusions?|summary)\s*$",
            content[introduction.end():],
        )
        if next_section:
            section_start = introduction.end() + next_section.start()
            return content[:introduction.start()] + "\n" + content[section_start:]
        return content[:introduction.start()]
    # Some journal PDFs omit an "Introduction" heading. When their abstract
    # ends with a standard conclusion sentence and a later Methods/Results
    # heading exists, skip the unheaded background between those sections.
    abstract_end = re.search(r"(?is)\bin conclusion\b.{0,700}?[.!?](?=\s|$)", content[:7000])
    if abstract_end:
        next_section = re.search(
            r"(?im)^\s*(?:\d+[.)]?\s*)?(?:materials?\s+and\s+methods?|methods?|"
            r"results?|findings?|discussion|conclusions?|summary)\s*$",
            content[abstract_end.end():],
        )
        if next_section:
            section_start = abstract_end.end() + next_section.start()
            return content[:abstract_end.end()] + "\n" + content[section_start:]
    return content


def _sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
            if len(part.strip()) >= 35]


def _is_citation_or_table_fragment(text: str) -> bool:
    """Avoid treating cited-study summaries and PDF table reflow as findings."""
    return bool(re.search(
        r"(?i)(?:^\s*(?:\(\s*\d{1,3}\s*\)|\[\s*\d{1,3}\s*\])\s*|"
        r"\bet\s+al\.?\b|\by\s+cols\.?\b|\bprevious(?:ly)?\s+(?:study|research|work)\b|"
        r"\b(?:19|20)\d{2}\s*;\s*\d+(?:\(\d+\))?\s*:\s*\d+\b|"
        r"\btable\s+\d+\b|\b(?:cryopreserved tissues|tissue sources|"
        r"freezing method|storage|records found|search syntaxes)\b)",
        text,
    ))


def _is_inside_reference_cluster(text: str, position: int) -> bool:
    """Detect numbered bibliography blocks interleaved by PDF column order."""
    context = text[max(0, position - 900):min(len(text), position + 900)]
    numbered_entries = re.findall(
        r"(?i)(?<!\w)\d{1,2}\.\s+[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){1,3}\b",
        context,
    )
    journal_years = re.findall(r"\b(?:19|20)\d{2}\s*;\s*\d+(?:\(\d+\))?\s*:\s*\d+\b", context)
    return len(numbered_entries) >= 3 or len(journal_years) >= 3


def _has_inline_citation(text: str) -> bool:
    """Avoid using a cited prior-study sentence as this paper's node result."""
    return bool(re.search(
        r"(?:\[\s*\d{1,3}(?:\s*[-,;]\s*\d{1,3})*\s*\]|"
        r"\(\s*\d{1,3}(?:\s*[-,;]\s*\d{1,3})*\s*\)|"
        r"\(\s*[A-Z][A-Za-z’'-]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z’'-]+))?,?\s*"
        r"(?:19|20)\d{2}[a-z]?\s*\))",
        text,
    ))


def _relation_match(text: str):
    return re.search(
        r"(?i)\b(?:induc(?:e|es|ed)|caus(?:e|es|ed)|promot(?:e|es|ed)|"
        r"enhanc(?:e|es|ed)|inhibit(?:s|ed)?|prevent(?:s|ed)?|enabl(?:e|es|ed)|"
        r"contribut(?:e|es|ed)\s+to|lead(?:s|ing)?\s+to|result(?:s|ed|ing)?\s+in|"
        r"associat(?:e|es|ed)\s+with|correlat(?:e|es|ed)\s+with|"
        r"increas(?:e|es|ed|ing)|reduc(?:e|es|ed|ing)|improv(?:e|es|ed)|"
        r"differentiat(?:e|es|ed|ing)\s+into|derived\s+from|secreted\s+by)\b",
        text,
    )


def _is_substantive_relation_sentence(text: str) -> bool:
    """Keep only explicit result-bearing relations, not aim/co-occurrence edges."""
    if (_is_citation_or_table_fragment(text) or _has_inline_citation(text)
            or re.search(
                r"(?i)\b(?:aim|objective|purpose|designed to|planned to|protocol|"
                r"we sought to|we aimed to|raised the possibility|may potentially|"
                r"could potentially|might potentially)\b", text)):
        return False
    relation = _relation_match(text)
    result = re.search(
        r"(?i)\b(?:result(?:s|ed)?|confirm(?:s|ed)?|show(?:s|ed|n)?|"
        r"demonstrat(?:e|es|ed)|observ(?:e|es|ed)|measur(?:e|es|ed)|"
        r"significant(?:ly)?|higher|lower|increased|decreased|reduced|improved|"
        r"formation|regeneration|proliferation|viability|survival)\b|"
        r"\b\d+(?:\.\d+)?\s*%",
        text,
    )
    return bool(relation and result)


def _supporting_quote(sentence: str, anchor: str = "", max_words: int = 35) -> str:
    """Return a source-grounded excerpt with the concept in its evidence context.

    Synthetic extraction used to quote only the matched keyword, which made
    the demo look as if a label itself proved the claim. Keep a bounded window
    around the concept so nearby methods/results remain visible.
    """
    spans = list(re.finditer(r"\S+", sentence))
    if len(spans) <= max_words:
        return sentence.strip()
    anchor_pos = sentence.casefold().find(anchor.casefold()) if anchor else -1
    anchor_word = next((i for i, match in enumerate(spans)
                        if anchor_pos >= 0 and match.start() <= anchor_pos < match.end()), 0)
    start = max(0, anchor_word - max_words // 3)
    start = min(start, max(0, len(spans) - max_words))
    end = min(len(spans), start + max_words)
    return sentence[spans[start].start():spans[end - 1].end()].strip()


def _claim_description(sentence: str, anchor: str, max_words: int = 18) -> str:
    """Use a bounded source-text claim around the concept instead of a bare topic."""
    spans = list(re.finditer(r"\S+", sentence))
    if not spans:
        return ""
    match = _phrase_pattern(anchor).search(sentence) if anchor else None
    anchor_pos = match.start() if match else 0
    anchor_word = next((index for index, span in enumerate(spans)
                        if span.start() <= anchor_pos < span.end()), 0)
    width = min(max_words, len(spans))
    start = max(0, min(anchor_word - width // 2, len(spans) - width))
    end = start + width
    return sentence[spans[start].start():spans[end - 1].end()].strip()


def _simulated_label_status(sentence: str) -> str:
    """Assign a conservative synthetic status from wording in one sentence."""
    if re.search(
            r"(?i)\b(?:refut(?:e|es|ed)|contradict(?:s|ed)?|failed to replicate|"
            r"did not (?:show|support|confirm|demonstrate)|no evidence of|"
            r"could not (?:confirm|replicate)|unable to replicate|"
            r"not (?:independently )?(?:replicated|confirmed)|does not replicate|"
            r"not statistically significant)\b",
            sentence):
        return "refutation"
    if re.search(
            r"(?i)\b(?:hypothes(?:is|ized)|theoris(?:e|ed)|theoriz(?:e|ed)|"
            r"propos(?:e|es|ed)|may|might|could|potentially|model predicts?)\b",
            sentence):
        return "theory"
    if re.search(
            r"(?i)\b(?:formal proof|mathematically proved|proved theorem|"
            r"independently replicated|independently confirmed|"
            r"conclusively proved|conclusively demonstrated)\b", sentence):
        return "proof"
    if re.search(
            r"(?i)\b(?:show(?:s|ed|n)?|demonstrat(?:e|es|ed)|observ(?:e|es|ed)|"
            r"measur(?:e|es|ed)|found|result(?:s|ed)?|increased|decreased|"
            r"reduced|improved|significant(?:ly)?)\b|\b\d+(?:\.\d+)?\s*%",
            sentence):
        return "fact"
    return "idea_concept"


def _simulated_node_is_valid(item: Dict[str, Any]) -> bool:
    """Keep the simulator's responses within the same evidence gates as real output."""
    if not ResponseValidator._is_substantive_node_label(item.get("text", "")):
        return False
    if item.get("edge_endpoint_only"):
        return True
    evidence = item.get("evidence", []) or []
    if isinstance(evidence, str):
        evidence = [evidence]
    status = parse_label_status(item.get("label_status"))
    status_evidence = str(item.get("status_evidence") or "")
    return bool(
        evidence
        and status is not None
        and status_evidence in evidence
        and ResponseValidator._quote_has_supporting_context(
            str(evidence[0]), item.get("text", ""))
        and ResponseValidator._status_evidence_is_sufficient(status, status_evidence)
    )


def _source_year(text: str, fallback: Optional[int]) -> Optional[int]:
    head = text[:7000]
    patterns = (
        r"(?i)(?:published(?: online)?|publication date|received|accepted|copyright|©|the author\(s\))[^\d]{0,40}(19\d{2}|20\d{2})",
        r"(?i)\b(19\d{2}|20\d{2})\b",
    )
    for pattern in patterns:
        found = re.search(pattern, head)
        if found:
            year = int(found.group(1))
            if 1850 <= year <= 2035:
                return year
    return int(fallback) if fallback else None


def _safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


class SimulatedLLM:
    """Prompt-compatible deterministic stand-in used only for this run."""

    model = "oracle1-simulated-unbounded"

    def __init__(self, paper_metadata: Dict[str, Dict[str, Any]]):
        self.paper_metadata = paper_metadata
        self.calls: Dict[str, int] = {}

    def _count(self, kind: str) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1

    def complete(self, prompt: str) -> str:
        if "═══ STAGE 1" in prompt and '"document_metadata"' in prompt:
            self._count("unified_extraction")
            return _safe_json(self._extract_response(prompt))
        if "Available domains:" in prompt and "Available groups:" in prompt:
            self._count("concept_classification")
            return _safe_json(self._classify_response(prompt))
        if "--- Edge " in prompt and '"edge_id"' in prompt:
            self._count("edge_validation")
            edge_ids = re.findall(r"--- Edge \d+ \(id=([^\)]+)\) ---", prompt)
            return _safe_json({"results": [
                {"edge_id": edge_id, "plausibility": 0.82, "verdict": "keep",
                 "rationale": "Synthetic co-occurrence edge has source-grounded evidence."}
                for edge_id in edge_ids
            ]})
        if '"results"' in prompt and '"item_id"' in prompt:
            self._count("pass_d")
            return _safe_json({"results": []})
        # Fail loudly on a newly introduced LLM call path. Silent generic
        # responses would make a pipeline change look tested when it was not.
        raise RuntimeError("SimulatedLLM has no response fixture for this prompt type")

    def _extract_response(self, prompt: str) -> Dict[str, Any]:
        marker = "\nTEXT:\n"
        if marker not in prompt:
            raise ValueError("Unified prompt has no TEXT section")
        text = prompt.split(marker, 1)[1]
        end = text.find("\n════════════════════════════════════════")
        if end >= 0:
            text = text[:end]
        text = text.strip()
        body = re.sub(r"\s+", " ", _content_before_references(text)).strip()
        source_match = re.search(r'^DOCUMENT: "([^"]+)"', prompt, re.MULTILINE)
        source = source_match.group(1) if source_match else ""
        metadata = self.paper_metadata.get(Path(source).name, {})
        year = _source_year(text, metadata.get("year"))
        title = str(metadata.get("title") or "").strip() or None
        if title and re.search(r"(?i)\bprotocol\b", title):
            # A protocol records planned design and endpoints, not outcomes
            # about the entities it intends to use. The real validator also
            # rejects protocol-derived node observations; avoid fabricating
            # them in the stand-in response in the first place.
            return {
                "document_metadata": {
                    "title": title, "authors": [], "source_type": "unknown",
                    "publication_date": str(year) if year else None,
                    "date_evidence": None,
                },
                "domain": "biomedical research", "nodes": [],
                "edges": [], "temporal_zones": [],
            }

        found: List[Dict[str, Any]] = []
        for label, entity_type, variants in CONCEPTS:
            patterns = [(variant, _phrase_pattern(variant)) for variant in variants]
            hits = [(variant, match) for variant, pattern in patterns
                    for match in pattern.finditer(body)]
            if not hits:
                continue
            sentences = _sentences(body)
            candidates = []
            for sentence in sentences:
                sentence_pos = body.find(sentence)
                if (_is_citation_or_table_fragment(sentence) or _has_inline_citation(sentence)
                        or _is_inside_reference_cluster(body, max(0, sentence_pos))):
                    continue
                for variant, pattern in patterns:
                    match = pattern.search(sentence)
                    if not match:
                        continue
                    quote = _supporting_quote(sentence, match.group(0))
                    if (ResponseValidator._quote_has_supporting_context(quote, label)
                            and (ResponseValidator._label_is_claim_subject(sentence, label)
                                 or ResponseValidator._label_is_result_object(sentence, label))
                            and not ResponseValidator._quote_repeats_source_title(quote, title or "")):
                        words = ResponseValidator._evidence_tokens(quote)
                        fact_score = 4 * bool(re.search(
                            r"(?i)\b(?:\d+(?:\.\d+)?\s*%|higher|lower|similar|most|least|"
                            r"compared|increased|decreased|reduced|improved)\b", quote))
                        candidates.append((fact_score + len(set(words)), -body.find(sentence), sentence,
                                           quote, match.start(), match.group(0)))
            if candidates:
                _, _, sentence, quote, hit_offset, anchor = max(candidates)
                first_pos = max(0, body.find(sentence)) + hit_offset
            else:
                # A paragraph-level window can recover support when PDF text
                # extraction split a sentence at a line or column boundary.
                quote = sentence = ""
                anchor = ""
                first_pos = -1
                for _, candidate_hit in hits:
                    if _is_inside_reference_cluster(body, candidate_hit.start()):
                        continue
                    window_start = max(0, candidate_hit.start() - 180)
                    window_end = min(len(body), candidate_hit.end() + 240)
                    # Keep complete words at the clipped edges. A cut such as
                    # "lize mesenchymal stem cells" hid the governing verb
                    # "utilize" and made a background mention look like a
                    # source finding.
                    while window_start > 0 and not body[window_start - 1].isspace():
                        window_start -= 1
                    while window_end < len(body) and not body[window_end].isspace():
                        window_end += 1
                    window = body[window_start:window_end].strip()
                    if (_is_citation_or_table_fragment(window)
                            or _has_inline_citation(window)):
                        continue
                    candidate_quote = _supporting_quote(window, candidate_hit.group(0))
                    if (ResponseValidator._quote_has_supporting_context(candidate_quote, label)
                            and (ResponseValidator._label_is_claim_subject(window, label)
                                 or ResponseValidator._label_is_result_object(window, label))
                            and not ResponseValidator._quote_repeats_source_title(candidate_quote, title or "")):
                        sentence, quote = window, candidate_quote
                        anchor = candidate_hit.group(0)
                        first_pos = max(0, body.find(window)) + candidate_hit.start()
                        break
                if first_pos < 0:
                    continue
            found.append({
                "id": _slug(label), "label": label, "entity_type": entity_type,
                "text": _claim_description(quote, anchor or label),
                # This stand-in does not perform semantic analysis. Keep the
                # field empty instead of mislabelling a copied source passage
                # as an analytical explanation in the UI.
                "full_context": "",
                "evidence": [quote],
                "label_status": _simulated_label_status(quote),
                "status_evidence": quote,
                "contradicted_claims": [],
                "first_pos": first_pos,
                "evidence_sentence": sentence,
            })
        found.sort(key=lambda item: item["first_pos"])
        found = [item for item in found
                 if ResponseValidator._is_substantive_node_label(item.get("text", ""))]

        # A concept that is only the subject/source endpoint of a result
        # relation gets no node-level observation from this article. Keep an
        # endpoint reference so the explicit edge can still be assembled.
        for item in found:
            sentence = item.get("evidence_sentence", "")
            relation = _relation_match(sentence) if sentence else None
            if not relation or not _is_substantive_relation_sentence(sentence):
                continue
            present = []
            for candidate in found:
                variants = next(v for label, _, v in CONCEPTS
                                if label == candidate["label"])
                positions = [match.start() for variant in variants
                             for match in _phrase_pattern(variant).finditer(sentence)]
                if positions:
                    present.append((min(positions), candidate["id"]))
            left = [node_id for position, node_id in present if position < relation.start()]
            right = [node_id for position, node_id in present if position > relation.end()]
            if left and right and item["id"] == left[-1]:
                item["edge_endpoint_only"] = True
                item["evidence"] = []
                item["full_context"] = ""

        # If the lexicon finds no domain phrase, use a grounded title/first
        # sentence as one node so the real validator and date gate still run.
        if not found:
            sentence = next(iter(_sentences(body)), "")
            if sentence:
                label = (title or sentence).strip()
                quote = _supporting_quote(sentence)
                if (ResponseValidator._quote_has_supporting_context(quote, label)
                        and not ResponseValidator._quote_repeats_source_title(quote, title or "")):
                    found.append({
                        "id": _slug(label), "label": label,
                        "entity_type": "research_frontier",
                        "text": _claim_description(quote, label),
                        "full_context": "",
                        "evidence": [quote],
                        "label_status": _simulated_label_status(quote),
                        "status_evidence": quote,
                        "contradicted_claims": [],
                        "first_pos": body.find(quote),
                        "evidence_sentence": sentence,
                    })
        found = [item for item in found if _simulated_node_is_valid(item)][:16]

        nodes = []
        for item in found:
            nodes.append({
                "id": item["id"], "text": item["text"],
                "full_context": item["full_context"],
                "entity_type": item["entity_type"], "origin_kind": "combinatorial",
                "domain": "biomedical research", "node_type": "research",
                "scientific_score": 5.0, "investment_score": 5.0,
                "social_score": 5.0, "maturity_score": 5.0,
                "strategic_value": 5.0, "efficiency_plateau": 5.0,
                "dual_use_risk": 0.0, "legal_risk_score": 0.0,
                "export_control_risk": 0.0, "publication_year": year,
                "solves_limitations": [], "requires_node_ids": [],
                "evidence": item["evidence"],
                "label_status": item.get("label_status", "idea_concept"),
                "status_evidence": item.get("status_evidence", ""),
                "contradicted_claims": item.get("contradicted_claims", []),
                "edge_endpoint_only": bool(item.get("edge_endpoint_only")),
                "physical_axes": {
                    "resource_intensity": 0.5, "scale_feasibility": 0.5,
                    "constraint_proximity": 0.5, "reversibility": 0.5,
                    "parallel_deployability": 0.5, "environmental_coupling": 0.5,
                    "cross_scale_stability": 0.5, "synthesis_complexity": 0.5,
                    "longevity": 0.5, "cascadability": 0.5,
                    "rationale": "Synthetic neutral midpoint; no physical assessment is simulated.",
                },
                "secondary_contexts": [],
            })

        # Co-occurrence is not a relationship. Emit an edge only when the
        # same source sentence states a result-bearing relation between two
        # concepts. Aims, cited background, and bibliography/table fragments
        # are not edge evidence.
        edge_pairs = set()
        edges = []
        for sentence in _sentences(body):
            if not _is_substantive_relation_sentence(sentence):
                continue
            relation = _relation_match(sentence)
            if relation is None:
                continue
            present = []
            for item in found:
                variants = next(v for label, _, v in CONCEPTS if label == item["label"])
                matches = [match.start() for variant in variants
                           for match in _phrase_pattern(variant).finditer(sentence)]
                if matches:
                    present.append((min(matches), item["id"]))
            present.sort()
            left = [node_id for position, node_id in present if position < relation.start()]
            right = [node_id for position, node_id in present if position > relation.end()]
            # The two endpoints must flank the asserted predicate. Merely
            # mentioning a clinical trial before a clause about MSCs does not
            # make "clinical trial -> MSCs" a supported relationship.
            if not left or not right:
                continue
            source_id, target_id = left[-1], right[0]
            key = (source_id, target_id)
            if source_id != target_id and key not in edge_pairs:
                edge_pairs.add(key)
                edges.append({
                    "source": source_id, "target": target_id,
                    "relationship_type": "related",
                    "semantic_similarity": 0.55, "temporal_proximity": 0.6,
                    "limitation_resolution": 0.0, "citation_link": 0.0,
                    "investment_correlation": 0.0, "social_correlation": 0.0,
                    "inhibitory_force": 0.0, "confidence": 0.62,
                    "evidence": [sentence[:800]],
                })
            if len(edges) >= 32:
                break

        return {
            "document_metadata": {
                "title": title, "authors": [], "source_type": "unknown",
                "publication_date": str(year) if year else None,
                "date_evidence": None,
            },
            "domain": "biomedical research", "nodes": nodes,
            "edges": edges, "temporal_zones": [],
        }

    @staticmethod
    def _classify_response(prompt: str) -> Dict[str, Any]:
        def names(label: str) -> List[str]:
            match = re.search(rf"^{re.escape(label)}:\s*(.+)$", prompt, re.MULTILINE)
            return [x.strip() for x in match.group(1).split(",") if x.strip()] if match else []
        domains = names("Available domains")
        groups = names("Available groups")
        entity_types = names("Available entity types")
        choose = lambda values, hints: next(
            (x for x in values if any(h in x.casefold() for h in hints)),
            values[0] if values else "unknown",
        )
        return {
            "domain": choose(domains, ("medical", "biolog", "health", "science")),
            "group": choose(groups, ("biolog", "medical", "research", "science")),
            "entity_type": choose(entity_types, ("research_frontier", "research", "technology")),
            "confidence": 0.84,
            "reasoning": "Deterministic biomedical fallback for the simulation run.",
            "aliases": [],
        }


def _doi_filename(doi: str) -> str:
    return "doi_" + re.sub(r"[^A-Za-z0-9]+", "_", doi).strip("_") + ".pdf"


def _download_one(item: Dict[str, Any], corpus: Path, max_bytes: int) -> Dict[str, Any]:
    doi = str(item["doi"])
    path = corpus / _doi_filename(doi)
    if path.is_file() and path.stat().st_size > 100 and path.read_bytes()[:5] == b"%PDF-":
        return {"doi": doi, "path": path.name, "status": "reused", "bytes": path.stat().st_size}
    temporary = path.with_suffix(".pdf.part")
    try:
        with requests.get(item["open_access_pdf_url"], headers={"User-Agent": USER_AGENT},
                          stream=True, timeout=(20, 90), allow_redirects=True) as response:
            if response.status_code != 200:
                return {"doi": doi, "status": "http_error", "http_status": response.status_code}
            content_type = response.headers.get("Content-Type", "").casefold()
            chunks = []
            size = 0
            for chunk in response.iter_content(1024 * 256):
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    return {"doi": doi, "status": "too_large", "bytes": size}
                chunks.append(chunk)
        data = b"".join(chunks)
        if not data.startswith(b"%PDF-") and "application/pdf" not in content_type:
            return {"doi": doi, "status": "not_pdf", "content_type": content_type}
        if not data.startswith(b"%PDF-"):
            return {"doi": doi, "status": "invalid_pdf_header", "content_type": content_type}
        temporary.write_bytes(data)
        temporary.replace(path)
        return {"doi": doi, "path": path.name, "status": "downloaded", "bytes": size}
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return {"doi": doi, "status": "download_error", "error": str(exc)[:300]}


def _rank_candidates(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for row in manifest.get("references", []):
        counts[row.get("doi", "")] = counts.get(row.get("doi", ""), 0) + 1
    unique: Dict[str, Dict[str, Any]] = {}
    for row in manifest.get("references", []):
        doi = row.get("doi")
        url = row.get("open_access_pdf_url")
        if not (doi and row.get("resolved") and row.get("is_open_access") and url):
            continue
        if doi not in unique:
            unique[doi] = dict(row)
    priorities = re.compile(r"dental|dentin|pulp|odont|endodont|stem cell|regenerat|scaffold|mesenchymal|growth factor", re.I)
    return sorted(unique.values(), key=lambda row: (
        0 if priorities.search(str(row.get("title", ""))) else 1,
        -counts.get(row.get("doi", ""), 0),
        str(row.get("title", "")).casefold(),
    ))


def download_corpus(manifest: Dict[str, Any], corpus: Path, limit: int,
                    max_pdf_mb: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    corpus.mkdir(parents=True, exist_ok=True)
    candidates = _rank_candidates(manifest)[:limit]
    max_bytes = max_pdf_mb * 1024 * 1024
    logger.info("[simulation] trying %d distinct DOI-backed OA PDF candidates", len(candidates))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = list(executor.map(lambda item: _download_one(item, corpus, max_bytes), candidates))
    metadata_by_file: Dict[str, Dict[str, Any]] = {}
    candidate_by_doi = {x["doi"]: x for x in candidates}
    for outcome in outcomes:
        if outcome.get("path"):
            item = candidate_by_doi[outcome["doi"]]
            metadata_by_file[outcome["path"]] = {
                "path": outcome["path"], "doi": outcome["doi"], "title": item.get("title"),
                "year": item.get("year"), "source_file": item.get("source_file"),
                "pdf_url": item.get("open_access_pdf_url"),
            }
    return outcomes, list(metadata_by_file.values())


def run(args: argparse.Namespace) -> Dict[str, Any]:
    manifest_path = args.reference_manifest.resolve()
    reference_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus = args.corpus_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        outcomes, downloaded_meta = download_corpus(
            reference_manifest, corpus, args.limit, args.max_pdf_mb)
        (output / "downloads.json").write_text(
            json.dumps(outcomes, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        downloaded_meta = []
        downloads_path = output / "downloads.json"
        outcomes = (json.loads(downloads_path.read_text(encoding="utf-8"))
                    if downloads_path.exists() else [])

    # Keep bookkeeping outside the corpus: DocumentQueue recursively treats
    # supported sidecar formats as input documents too.
    sidecar = output / "papers_manifest.json"
    if downloaded_meta:
        sidecar.write_text(json.dumps(downloaded_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    elif sidecar.exists():
        downloaded_meta = json.loads(sidecar.read_text(encoding="utf-8"))

    metadata_by_file = {item["path"]: item for item in downloaded_meta if item.get("path")}
    if not metadata_by_file:
        # Existing corpus-only mode: source year is parsed from PDF text.
        metadata_by_file = {}

    # The stand-in has no context limit. Give DocumentChunker an explicit,
    # generous budget so it exercises whole-document ingestion instead of
    # accidentally hitting the conservative no-model fallback.
    CONFIG["llm_context_window_default"] = max(2_000_000, CONFIG.get("llm_context_window_default", 128_000))
    llm = SimulatedLLM(metadata_by_file)
    system = graph_update.build_symbolic_system(llm)
    graph_path = output / "graph_snapshot.pkl"
    queue_path = output / "document_queue.json"
    queue = DocumentQueue(queue_path)
    if args.reset and graph_path.exists():
        graph_path.unlink()
    if args.reset and queue_path.exists():
        queue_path.unlink()
    if graph_path.exists():
        from oracle1 import load_graph_snapshot
        load_graph_snapshot(graph_path, system)

    ingest_args = argparse.Namespace(
        corpus_dir=str(corpus), document_queue=str(queue_path),
        graph_snapshot=str(graph_path),
        allow_revisions=False, retry_failed_documents=args.retry_failed,
        document_python="", max_document_chars=0,
        ingestion_rate_limit_rpm=3600, ingestion_max_errors=args.max_errors,
        checkpoint_documents=max(1, args.checkpoint_documents),
    )
    started = time.time()
    graph_update.ingest_documents(system, ingest_args)
    graph_update.run_graph_restructuring(system, max(0, args.graph_steps))
    save_graph_snapshot(graph_path, system)
    graph = system["graph"]
    before_replay = {
        "nodes": len(graph.nodes), "edges": len(graph.edges),
        "llm_calls": dict(llm.calls),
    }
    _, _, replay_stats, replay_documents = graph_update.collect_documents(ingest_args, graph)
    if replay_documents:
        raise RuntimeError(
            f"Idempotency check selected {len(replay_documents)} previously ingested document(s)")
    graph_update.ingest_documents(system, ingest_args)
    if (len(graph.nodes), len(graph.edges), llm.calls) != (
            before_replay["nodes"], before_replay["edges"], before_replay["llm_calls"]):
        raise RuntimeError("Repeated corpus ingestion changed graph contents or called the simulator")
    repeat_ingestion_check = {
        "passed": True,
        "documents_seen": replay_stats.get("seen", 0),
        "already_in_graph": replay_stats.get("already_in_graph", 0),
        "selected_for_reingestion": len(replay_documents),
        "graph_unchanged": True,
        "llm_calls_unchanged": True,
    }
    duplicate_filter_audit = audit_duplicate_filtering(output)
    (output / "duplicate_filter_audit.json").write_text(
        json.dumps({"repeat_ingestion": repeat_ingestion_check,
                    "queue_filters": duplicate_filter_audit},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = round(time.time() - started, 2)

    bad_edges = [edge.id for edge in graph.edges.values()
                 if edge.source_id not in graph.nodes or edge.target_id not in graph.nodes]
    summary = graph_snapshot_summary(system)
    queue_counts = DocumentQueue(queue_path).counts()
    visualizer_manifest = {
        "format": "oracle1_simulated_llm_preview_v1",
        "visualizer_mode": "simulated_llm_response",
        "label": "Synthetic LLM-shaped responses; not real LLM analysis or findings",
        "description": "Deterministic, source-text-based response simulation used to exercise Oracle-1 ingestion and graph algorithms.",
        "nodes": len(graph.nodes), "edges": len(graph.edges),
        "source_manifest": str(manifest_path),
        "corpus_dir": str(corpus),
        "warning": "Do not interpret simulated nodes or edges as scientific conclusions.",
    }
    (output / "manifest.json").write_text(
        json.dumps(visualizer_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "run_mode": "simulated_llm_response",
        "warning": "Synthetic LLM-shaped responses; generated concepts/edges are not scientific findings.",
        "algorithm_path": [
            "DocumentQueue.select_new", "IngestionPipeline.ingest",
            "ResponseValidator", "CanonicalNodeAssembler", "GraphConsolidator",
            "RuntimeGraph", "run_graph_restructuring",
        ],
        "reference_manifest": str(manifest_path),
        "corpus_dir": str(corpus), "output_dir": str(output),
        "download_counts": {
            "attempted": len(outcomes),
            "downloaded_or_reused": sum(x.get("status") in {"downloaded", "reused"} for x in outcomes),
            "failed_or_skipped": sum(x.get("status") not in {"downloaded", "reused"} for x in outcomes),
        },
        "queue_counts": queue_counts,
        "graph_nodes": len(graph.nodes), "graph_edges": len(graph.edges),
        "orphan_edges": len(bad_edges),
        "llm_simulator_calls": llm.calls,
        "repeat_ingestion_check": repeat_ingestion_check,
        "duplicate_filter_audit": duplicate_filter_audit,
        "graph_steps": max(0, args.graph_steps), "elapsed_seconds": elapsed,
        "snapshot": str(graph_path), "snapshot_summary": summary,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if bad_edges:
        raise RuntimeError(f"Graph contains {len(bad_edges)} orphan edge(s); see {output / 'run_manifest.json'}")
    logger.info("[simulation] complete: %d docs, %d nodes, %d edges in %.2fs",
                sum(queue_counts.get(key, 0) for key in ("ingested", "ingested_empty")),
                len(graph.nodes), len(graph.edges), elapsed)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path,
                        default=Path("artifacts/reference_preview/manifest.json"))
    parser.add_argument("--corpus-dir", type=Path,
                        default=Path("artifacts/reference_simulated_corpus"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/simulated_llm_run"))
    parser.add_argument("--limit", type=int, default=24,
                        help="Maximum distinct reference PDFs to try, ranked for Oracle1's source domain.")
    parser.add_argument("--max-pdf-mb", type=int, default=20)
    parser.add_argument("--skip-download", action="store_true",
                        help="Process PDFs already present in --corpus-dir.")
    parser.add_argument("--reset", action="store_true",
                        help="Start a fresh isolated queue and graph snapshot in --output-dir.")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--checkpoint-documents", type=int, default=10)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--graph-steps", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    try:
        report = run(parse_args())
        print(json.dumps({
            key: report[key] for key in (
                "run_mode", "download_counts", "queue_counts", "graph_nodes",
                "graph_edges", "orphan_edges", "llm_simulator_calls",
                "elapsed_seconds", "snapshot",
            )
        }, ensure_ascii=False, indent=2))
    except Exception:
        logger.exception("Simulated Oracle-1 run failed")
        raise SystemExit(1)
