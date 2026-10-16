# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""Deduplication indexes, ingestion pipeline, and batch processing."""
from __future__ import annotations

from ._shared import *
from .canonical import CanonicalNodeAssembler, ConceptClassifier
from .canonical_keys import DOMAIN_TABLE, ENTITY_TYPE_TABLE
from .config import CONFIG
from .entities import KnowledgeEdge, NodeFeatureBuilder
from .json_utils import _recover_partial_array
from .label_status import LabelStatus, label_status_name, parse_label_status
from .time_utils import _ts_to_year, _year_to_ts
from .validation import (
    DocumentChunk,
    DocumentChunker,
    EdgeSemanticValidator,
    INGESTION_SOURCE_TYPES,
    IngestionPromptBuilder,
    LLMAnnotator,
    LLMResponse,
    NodeEdgeAssembler,
    ResponseValidator,
)

class FuzzyMatcher:
    """
    Normalises and compares entity names across languages, transliterations,
    abbreviations and spelling variants — without external dependencies.

    Scoring: weighted combination of four signals.
      exact_norm  — normalised string equality          (weight 1.0)
      prefix4     — first-4-char prefix match           (weight 0.6)
      token_jaccard — token-set overlap                 (weight 0.5)
      year_overlap — year-hint proximity                (weight 0.3)

    Typical use: pre-filter bucket candidates before LLM Pass D.
    """

    # Common articles / stopwords across several languages to strip
    _ARTICLES = re.compile(
        r"^(the|a|an|der|die|das|des|dem|den|ein|eine|einer|"
        r"le|la|les|l'|un|une|des|"
        r"el|la|los|las|un|una|"
        r"il|lo|la|gli|le|un|uno|una|"
        r"o|a|os|as|um|uma)\s+",
        re.IGNORECASE,
    )
    # Cyrillic → Latin transliteration via dict.
    # Multi-char digraphs properly disambiguated to avoid collision
    # (previously ж→z collided with з→z; ч→c collided with ц→c).
    # After transliteration the string may be longer but token-Jaccard
    # still works correctly on the expanded form.
    _TRANSLIT_MAP: Dict[str, str] = {
        'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ж':'zh','з':'z',
        'и':'i','й':'j','к':'k','л':'l','м':'m','н':'n','о':'o','п':'p',
        'р':'r','с':'s','т':'t','у':'u','ф':'f','х':'kh','ц':'ts','ч':'ch',
        'ш':'sh','щ':'shch','ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya',
        # Upper-case variants
        'А':'A','Б':'B','В':'V','Г':'G','Д':'D','Е':'E','Ж':'Zh','З':'Z',
        'И':'I','Й':'J','К':'K','Л':'L','М':'M','Н':'N','О':'O','П':'P',
        'Р':'R','С':'S','Т':'T','У':'U','Ф':'F','Х':'Kh','Ц':'Ts','Ч':'Ch',
        'Ш':'Sh','Щ':'Shch','Ъ':'','Ы':'Y','Ь':'','Э':'E','Ю':'Yu','Я':'Ya',
    }

    @classmethod
    def _do_translit(cls, s: str) -> str:
        """Character-by-character Cyrillic → Latin using _TRANSLIT_MAP."""
        return "".join(cls._TRANSLIT_MAP.get(c, c) for c in s)

    @classmethod
    def normalise(cls, text: str) -> str:
        """Lower-case, unicode NFKD, strip articles, collapse spaces."""
        import unicodedata
        s = unicodedata.normalize("NFKD", text)
        # Transliterate Cyrillic → Latin (multi-char digraphs, no collisions)
        s = cls._do_translit(s)
        # Drop combining diacritics
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower().strip()
        s = cls._ARTICLES.sub("", s)
        s = re.sub(r"[\s\-_/.,;:!?\"'()]+", " ", s).strip()
        return s

    @classmethod
    def prefix4(cls, a: str, b: str) -> bool:
        """True if first 4 normalised chars match."""
        na, nb = cls.normalise(a)[:4], cls.normalise(b)[:4]
        return bool(na) and na == nb

    @classmethod
    def token_jaccard(cls, a: str, b: str) -> float:
        """Jaccard similarity on token sets of normalised strings."""
        ta = set(cls.normalise(a).split())
        tb = set(cls.normalise(b).split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    @classmethod
    def year_score(cls, year_hint: Optional[int],
                   node_timestamp: Optional[float]) -> float:
        """1.0 if within 5 years, decays to 0.0 at 50+ years difference."""
        if year_hint is None or node_timestamp is None:
            return 0.5  # neutral when unknown
        node_year = int(node_timestamp / 31_536_000) + 1970  # epoch → year approx
        delta = abs(year_hint - node_year)
        return max(0.0, 1.0 - delta / 50.0)

    @classmethod
    def score(cls, query_text: str, candidate_text: str,
              query_year: Optional[int] = None,
              candidate_ts: Optional[float] = None) -> float:
        """Combined fuzzy score in [0, 1]."""
        nq = cls.normalise(query_text)
        nc = cls.normalise(candidate_text)
        exact  = 1.0 if nq == nc else 0.0
        pre    = 0.6 if cls.prefix4(query_text, candidate_text) else 0.0
        jac    = 0.5 * cls.token_jaccard(query_text, candidate_text)
        yr     = 0.3 * cls.year_score(query_year, candidate_ts)
        raw    = exact + pre + jac + yr          # max ≈ 2.4
        return min(1.0, raw / 2.4)


# ── Layer 6b: Bucket Index ────────────────────────────────────────────────────

class BucketIndex:
    """
    Hierarchical node index keyed by (domain_id, entity_type_id).

    Replaces CrossDocumentLinker's flat embedding list with a two-level
    structure that reduces candidate search from O(N) to O(N/B) where
    B = number of non-empty buckets (typically thousands).

    Each bucket stores:
      - A list of lightweight NodeRecord tuples for fuzzy/temporal pre-filter
      - Optionally a FAISS IndexFlatIP for ANN search within the bucket

    Merge / link thresholds are preserved from CONFIG for backward compat.
    """

    MERGE_THRESHOLD  = CONFIG["ingestion_merge_threshold"]
    FAMILY_THRESHOLD = CONFIG["ingestion_family_threshold"]
    LINK_THRESHOLD   = CONFIG["ingestion_link_threshold"]
    MAX_CANDIDATES   = CONFIG["bucket_max_candidates"]   # top-k sent to Pass-D LLM

    @staticmethod
    def _semantic_embedding(embedding: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Select the stable text section for semantic matching.

        Nodes can hold base graph features (413 dimensions) or the same vector
        extended with ten physical axes (423).  Those axes are essential to
        Oracle's graph calculations, but they must not leak into document
        identity/linking where mixed dimensions would make a dot product fail.
        """
        if embedding is None:
            return None
        raw = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if raw.size < NodeFeatureBuilder.TEXT_DIM:
            return None
        vec = raw[:NodeFeatureBuilder.TEXT_DIM].copy()
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-12 else None

    @dataclass
    class _NodeRecord:
        node_id:     str
        text:        str
        domain:      str
        entity_type: str
        timestamp:   Optional[float]
        embedding:   Optional[np.ndarray]

    def __init__(self, existing_nodes: Optional[List] = None):
        # bucket_key → list of _NodeRecord
        self._buckets: Dict[Tuple[int, int], List["BucketIndex._NodeRecord"]] = \
            defaultdict(list)
        # bucket_key → np.ndarray matrix (stacked embeddings) for ANN
        self._matrices: Dict[Tuple[int, int], np.ndarray] = {}
        # flat list for legacy merge detection (same-doc primary nodes)
        self._flat: List[Tuple[str, np.ndarray]] = []

        if existing_nodes:
            for n in existing_nodes:
                self.register_node_obj(n)

    # ── Registration ─────────────────────────────────────────────

    def register_node_obj(self, node) -> None:
        """Register a KnowledgeNode (or compatible object) into the index."""
        domain_id      = DOMAIN_TABLE.get_id(getattr(node, "domain", "unknown"))
        entity_type_id = ENTITY_TYPE_TABLE.get_id(
            getattr(node, "entity_type", "unknown"))
        bkey = (domain_id, entity_type_id)
        rec  = BucketIndex._NodeRecord(
            node_id     = node.id,
            text        = getattr(node, "text", ""),
            domain      = getattr(node, "domain", ""),
            entity_type = getattr(node, "entity_type", ""),
            timestamp   = getattr(node, "timestamp", None),
            embedding   = self._semantic_embedding(getattr(node, "embedding", None)),
        )
        self._buckets[bkey].append(rec)
        # Invalidate cached matrix for this bucket
        self._matrices.pop(bkey, None)
        # Flat list for merge detection
        if rec.embedding is not None:
            self._flat.append((rec.node_id, rec.embedding))

    def register_node(self, node_id: str, embedding: np.ndarray,
                       domain: str = "unknown",
                       entity_type: str = "unknown",
                       text: str = "",
                       timestamp: Optional[float] = None) -> None:
        """Register by explicit fields (used after assembly)."""
        domain_id      = DOMAIN_TABLE.get_id(domain)
        entity_type_id = ENTITY_TYPE_TABLE.get_id(entity_type)
        bkey = (domain_id, entity_type_id)
        rec  = BucketIndex._NodeRecord(
            node_id=node_id, text=text, domain=domain,
            entity_type=entity_type, timestamp=timestamp,
            embedding=self._semantic_embedding(embedding),
        )
        self._buckets[bkey].append(rec)
        self._matrices.pop(bkey, None)
        if rec.embedding is not None:
            self._flat.append((node_id, rec.embedding))

    # ── Candidate retrieval ───────────────────────────────────────

    def get_candidates(self,
                        ctx_text:        str,
                        ctx_domain:      str,
                        ctx_entity_type: str,
                        ctx_year:        Optional[int]      = None,
                        query_embedding: Optional[np.ndarray] = None,
                        top_k:           int                = MAX_CANDIDATES,
                        ) -> List[Dict]:
        """
        Return up to top_k candidate dicts from the matching bucket,
        ranked by combined (embedding ANN + fuzzy string + temporal) score.

        Signature is domain-agnostic: caller supplies the secondary context's
        domain and entity_type (which may differ from the primary node's).

        Domain registration: unknown domain/entity_type strings are
        auto-registered so they receive their own bucket ID rather than
        collapsing into the global bucket-0 "trash" bucket.
        """
        # Auto-register to avoid all unknowns collapsing into (0, 0)
        query_embedding = self._semantic_embedding(query_embedding)
        domain_id      = DOMAIN_TABLE.register(ctx_domain) if ctx_domain and ctx_domain != "unknown" \
                         else DOMAIN_TABLE.get_id(ctx_domain)
        entity_type_id = ENTITY_TYPE_TABLE.register(ctx_entity_type) if ctx_entity_type and ctx_entity_type != "unknown" \
                         else ENTITY_TYPE_TABLE.get_id(ctx_entity_type)
        bkey           = (domain_id, entity_type_id)
        bucket         = list(self._buckets.get(bkey, []))

        if not bucket:
            # Fallback 1: same domain, any entity_type
            for (did, _eid), recs in self._buckets.items():
                if did == domain_id:
                    bucket.extend(recs)

        if not bucket:
            # Fallback 2: embedding-only global search over top scored records.
            # Caps at 4 * top_k to stay bounded regardless of graph size.
            if query_embedding is not None:
                qn = np.linalg.norm(query_embedding)
                if qn > 1e-8:
                    scored_global: List[Tuple[float, "BucketIndex._NodeRecord"]] = []
                    cap = max(top_k * 4, 80)
                    for recs in self._buckets.values():
                        for rec in recs:
                            if rec.embedding is None:
                                continue
                            rn = np.linalg.norm(rec.embedding)
                            if rn < 1e-8:
                                continue
                            sim = float(np.dot(query_embedding, rec.embedding) / (qn * rn))
                            scored_global.append((sim, rec))
                    scored_global.sort(key=lambda x: x[0], reverse=True)
                    bucket = [rec for _, rec in scored_global[:cap]]

        if not bucket:
            return []

        scored: List[Tuple[float, "BucketIndex._NodeRecord"]] = []
        for rec in bucket:
            # Signal A: embedding cosine similarity (uses cached matrix if built)
            emb_sim = 0.0
            if query_embedding is not None and rec.embedding is not None:
                qn = np.linalg.norm(query_embedding)
                rn = np.linalg.norm(rec.embedding)
                if qn > 1e-8 and rn > 1e-8:
                    emb_sim = float(
                        np.dot(query_embedding, rec.embedding) / (qn * rn))

            # Signal B: fuzzy string match
            fuzzy_sim = FuzzyMatcher.score(
                ctx_text, rec.text,
                query_year=ctx_year,
                candidate_ts=rec.timestamp,
            )

            # Combined score (embedding weighted higher when available)
            if query_embedding is not None and rec.embedding is not None:
                combined = 0.6 * emb_sim + 0.4 * fuzzy_sim
            else:
                combined = fuzzy_sim

            scored.append((combined, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, rec in scored[:top_k]:
            results.append({
                "id":          rec.node_id,
                "text":        rec.text,
                "domain":      rec.domain,
                "entity_type": rec.entity_type,
                "timestamp":   rec.timestamp,
                "embedding":   rec.embedding,
                "_sim":        round(sim, 4),
            })
        return results

    # ── Legacy merge/link detection (same-domain primary nodes) ──

    # Hard limit: even with millions of registered nodes the flat scan is
    # bounded.  Increase if precision matters more than ingestion speed.
    _FLAT_SCAN_CAP = 50_000

    def find_connections(self, new_nodes: List) -> Tuple[List[Tuple], List[Dict]]:
        """
        Backward-compatible method used for same-document primary nodes.
        Detects merge candidates and semantic cross-doc edges via flat scan.
        For primary nodes this is acceptable (small N per document).

        When _flat grows beyond _FLAT_SCAN_CAP, scan is limited to the
        most-recently-registered nodes to bound latency.
        """
        merge_pairs: List[Tuple[str, str]] = []
        cross_edges: List[Dict]            = []
        # Bound the flat list so we don't do O(N_total) at scale
        flat_view = (self._flat[-self._FLAT_SCAN_CAP:]
                     if len(self._flat) > self._FLAT_SCAN_CAP
                     else self._flat)
        for node in new_nodes:
            if node.embedding is None:
                continue
            new_emb  = self._semantic_embedding(node.embedding)
            if new_emb is None:
                continue
            new_norm = np.linalg.norm(new_emb)
            if new_norm < 1e-8:
                continue
            sims: List[Tuple[float, str]] = []
            for eid, eemb in flat_view:
                if eid == node.id:
                    continue
                enorm = np.linalg.norm(eemb)
                if enorm < 1e-8:
                    continue
                sims.append((float(np.dot(new_emb, eemb) / (new_norm * enorm)), eid))
            sims.sort(reverse=True)
            for sim, eid in sims[:10]:
                if sim >= self.MERGE_THRESHOLD:
                    merge_pairs.append((node.id, eid))
                    break
                elif sim >= self.FAMILY_THRESHOLD:
                    cross_edges.append({
                        "source": node.id, "target": eid,
                        "relationship_type": "semantic_cross_doc",
                        "semantic_similarity": sim, "temporal_proximity": 0.3,
                        "limitation_resolution": 0.0, "citation_link": 0.0,
                        "investment_correlation": 0.0, "social_correlation": 0.0,
                        "inhibitory_force": 0.0, "confidence": sim,
                        "evidence": ["cross-document semantic match"],
                        # Near-miss just below MERGE_THRESHOLD: almost
                        # certainly the same underlying idea restated, not a
                        # loose cross-reference. Kept as a separate node
                        # (unique phrasing/detail preserved) rather than
                        # merged, but tagged so whatever applies this
                        # IngestionResult to a RuntimeGraph can call
                        # graph.union_family(source, target) — see
                        # RuntimeGraph.family_epistemic_mass for why this
                        # matters (an "iceberg" of near-duplicates should
                        # read as one aggregate mass, not N fragments).
                        "is_family": True,
                    })
                elif sim >= self.LINK_THRESHOLD:
                    cross_edges.append({
                        "source": node.id, "target": eid,
                        "relationship_type": "semantic_cross_doc",
                        "semantic_similarity": sim, "temporal_proximity": 0.3,
                        "limitation_resolution": 0.0, "citation_link": 0.0,
                        "investment_correlation": 0.0, "social_correlation": 0.0,
                        "inhibitory_force": 0.0, "confidence": sim,
                        "evidence": ["cross-document semantic match"],
                    })
        return merge_pairs, cross_edges


# ── Backward-compatible alias ─────────────────────────────────────────────────
CrossDocumentLinker = BucketIndex


# ── Layer 7: Orchestrator ─────────────────────────────────────────────────────

@dataclass
class IngestionResult:
    """Final result of processing a single document."""
    source:            str
    # Inferred by the extraction LLM from the full document.  These are
    # evidence attributes, unlike a filename or filesystem timestamp.
    source_type:        str = "unknown"
    document_timestamp: float = 0.0
    document_metadata:  Dict[str, Any] = field(default_factory=dict)
    nodes:             List           = field(default_factory=list)
    edges:             List           = field(default_factory=list)
    zones:             List           = field(default_factory=list)
    containment_edges: List           = field(default_factory=list)
    cross_doc_edges:   List           = field(default_factory=list)
    influence_observations: List[Dict[str, Any]] = field(default_factory=list)
    # Nodes that carry no date (date_unknown=True) but matched an existing graph
    # node via BucketIndex.  These must NOT be created as new graph nodes —
    # they are enrichment observations and should only go through
    # GraphConsolidator.observe_node(node, canonical_id=<matched_id>).
    enrichment_nodes:  List[Tuple]    = field(default_factory=list)  # [(node, matched_id), ...]
    merge_suggestions: List[Tuple]    = field(default_factory=list)
    stats:             Dict[str, Any] = field(default_factory=dict)
    warnings:          List[str]      = field(default_factory=list)
    errors:            List[str]      = field(default_factory=list)

    def ready_to_commit(self) -> bool:
        return (
            bool(self.nodes or self.influence_observations)
            and not self.errors
            and self.stats.get("llm_failures", 0) == 0
            and self.stats.get("llm_incomplete_calls", 0) == 0
        )


class IngestionPipeline:
    """
    Main orchestrator:
        document → chunk → PassA(entities + secondary_contexts)
                         → PassB(physical)
                         → PassC(edges within document)
                         → PassD(secondary_contexts → existing graph nodes)
                         → validate → assemble → cross-link → merge

    Pass D resolves secondary contexts (external influences with potentially
    different domain/type) against existing graph nodes via BucketIndex
    routing + FuzzyMatcher pre-filter + one LLM call per context.
    Replaces O(N) global cosine scan for cross-domain edge discovery.

    Usage:
        pipeline = IngestionPipeline(llm_client=my_llm, embedder=my_embedder,
                                     graph=runtime_graph)
        result   = pipeline.ingest(text, source, timestamp, source_type="arxiv")
        load_result_into_graph(result, graph)
    """

    def __init__(self, llm_client, embedder=None,
                 existing_nodes: Optional[List] = None,
                 chunk_size: int = None,
                 graph=None,
                 classifier: Optional["ConceptClassifier"] = None):
        # Manifest §2: size the ingestion budget to the model that will
        # actually read the document, so whole-document ingestion is the
        # default rather than something callers have to opt into by hand.
        # An explicit chunk_size still wins (e.g. tests that want to force
        # the fallback split path); otherwise infer from llm_client.model.
        model_name = getattr(llm_client, "model", None)
        self.chunker        = DocumentChunker(chunk_size=chunk_size, model_name=model_name)
        self.prompts        = IngestionPromptBuilder()
        self.annotator      = LLMAnnotator(llm_client)
        self.validator      = ResponseValidator()
        self.assembler      = NodeEdgeAssembler(embedder=embedder)  # edges/zones only — see its docstring
        # TODO #19: node.id = 12-byte CanonicalKey (hex) unconditionally now
        # — the "ingestion_use_canonical_key" flag and its MD5(source+text)
        # fallback are gone, so this is never None. classifier defaults to a
        # fresh ConceptClassifier wired to the module-level LookupTable/
        # AliasTable/SemanticHasher singletons (DOMAIN_TABLE etc.) — a caller
        # that already has one running across a longer-lived process (so the
        # dynamically-learned domain/group/entity_type vocabulary keeps
        # accumulating instead of resetting) can pass it in instead.
        # self.llm_client (== the same client passed to LLMAnnotator above)
        # is reused as the classifier's llm_fallback — no second client.
        self.canonical_assembler = CanonicalNodeAssembler(
            classifier=classifier or ConceptClassifier(),
            embedder=embedder, llm_fallback=llm_client,
        )
        self.linker         = BucketIndex(existing_nodes or [])
        self.edge_validator = EdgeSemanticValidator(self.annotator)
        self._graph         = graph
        if graph is not None and not existing_nodes:
            try:
                for node in graph.nodes.values():
                    self.linker.register_node_obj(node)
                logger.info(
                    f"[IngestionPipeline] BucketIndex pre-populated: "
                    f"{len(graph.nodes)} nodes"
                )
            except Exception as e:
                logger.warning(f"[IngestionPipeline] BucketIndex pre-pop failed: {e}")

    @staticmethod
    def _document_observation_metadata(raw: Any) -> Tuple[str, float, Dict[str, Any]]:
        """Validate document metadata inferred by the LLM.

        A local file's mtime is intentionally excluded: it says when a copy
        arrived in the library, not when the evidence was produced.  Unknown
        dates remain ``0`` rather than being fabricated.
        """
        metadata = dict(raw) if isinstance(raw, dict) else {}
        source_type = str(metadata.get("source_type", "unknown")).strip().lower()
        if source_type not in INGESTION_SOURCE_TYPES:
            source_type = "unknown"
        title = metadata.get("title")
        metadata["title"] = str(title).strip()[:500] if isinstance(title, str) and title.strip() else None
        authors = metadata.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        metadata["authors"] = [str(author).strip()[:200] for author in authors
                                if isinstance(author, str) and author.strip()][:30] \
            if isinstance(authors, list) else []
        for field_name in ("publication", "forum_topic", "source_url", "date_evidence"):
            value = metadata.get(field_name)
            metadata[field_name] = str(value).strip()[:1000] if isinstance(value, str) and value.strip() else None
        if not metadata["publication"]:
            for alias in ("venue", "journal", "publisher", "platform", "forum_name"):
                value = metadata.get(alias)
                if isinstance(value, str) and value.strip():
                    metadata["publication"] = value.strip()[:1000]
                    break
        date_value = str(metadata.get("publication_date", "")).strip()
        timestamp = 0.0
        if re.fullmatch(r"\d{4}", date_value):
            timestamp = _year_to_ts(int(date_value))
        elif date_value:
            try:
                timestamp = datetime.fromisoformat(date_value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                # Do not convert a vague LLM answer into a pseudo-date.
                metadata["publication_date"] = None
        metadata["source_type"] = source_type
        metadata["observation_timestamp"] = timestamp
        return source_type, timestamp, metadata

    @staticmethod
    def _merge_document_source_metadata(
        inferred: Dict[str, Any], provided: Optional[Dict[str, Any]], source: str,
    ) -> Dict[str, Any]:
        """Fill LLM-extracted bibliographic fields from trusted file metadata."""
        metadata = dict(inferred)
        provided = provided if isinstance(provided, dict) else {}
        if not metadata.get("title") and provided.get("document_title"):
            metadata["title"] = str(provided["document_title"]).strip()[:500] or None
        if not metadata.get("authors") and isinstance(provided.get("document_authors"), list):
            metadata["authors"] = [str(author).strip()[:200]
                                    for author in provided["document_authors"]
                                    if isinstance(author, str) and author.strip()][:30]
        if not metadata.get("source_url") and re.match(r"^https?://", source, re.IGNORECASE):
            metadata["source_url"] = source[:1000]
        metadata["document_format"] = provided.get("document_format")
        metadata["document_container"] = provided.get("document_container")
        metadata["document_content_hash"] = provided.get("document_content_hash")
        return metadata

    def ingest(self, text: str, source: str, timestamp: float = 0.0,
               source_type: str = "unknown", domain_hint: str = "",
               source_metadata: Optional[Dict[str, Any]] = None) -> IngestionResult:
        result    = IngestionResult(source=source)
        t_start   = time.time()
        calls_before = self.annotator.calls_made
        failures_before = self.annotator.failed_calls
        incomplete_before = self.annotator.incomplete_calls
        classifier = self.canonical_assembler.classifier
        classifier_calls_before = classifier.llm_fallback_calls
        classifier_failures_before = classifier.llm_fallback_failures
        classifier_incomplete_before = classifier.llm_fallback_incomplete_calls
        llm_calls = 0
        chunks    = self.chunker.chunk(text, source)
        logger.info(f"[Ingestion] '{source}': {len(chunks)} chunk(s)")
        all_chunk_results: List[Dict] = []
        # Only ever populated across chunks of the SAME document — the rare
        # DocumentChunker fallback-split path (see its docstring; the
        # default is one chunk, so this list stays empty in the common
        # case). Gives each later chunk a compact "already extracted"
        # registry so it can correctly reference an earlier chunk's node in
        # requires_node_ids/solves_limitations/edges instead of either
        # re-extracting it as new or being structurally unable to point at
        # it at all — see IngestionPromptBuilder._render_prior_nodes_block
        # for why this is the practical analogue to KV-cache reuse across
        # chunks rather than an attempt at the real (backend-level) thing.
        prior_nodes: List[Dict[str, str]] = []
        max_carry = CONFIG.get("ingestion_carryforward_max_nodes", 200)
        for chunk in chunks:
            # source_type/domain_hint are legacy call parameters deliberately
            # ignored.  The metadata pass inside _process_chunk owns them.
            cr = self._process_chunk(
                chunk, source, timestamp, prior_nodes=prior_nodes,
                source_metadata=source_metadata,
            )
            llm_calls += cr.get("llm_calls", 0)
            result.warnings.extend(cr.get("warnings", []))
            result.errors.extend(cr.get("errors", []))
            result.nodes.extend(cr.get("nodes", []))
            result.edges.extend(cr.get("edges", []))
            result.zones.extend(cr.get("zones", []))
            result.containment_edges.extend(cr.get("containment_edges", []))
            result.cross_doc_edges.extend(cr.get("pass_d_edges", []))
            result.influence_observations.extend(cr.get("influence_observations", []))
            result.enrichment_nodes.extend(cr.get("enrichment_nodes", []))
            if not result.document_metadata or cr.get("document_metadata", {}).get("publication_date"):
                result.document_metadata = dict(cr.get("document_metadata", {}))
                result.source_type = cr.get("source_type", "unknown")
                result.document_timestamp = float(cr.get("document_timestamp", 0.0) or 0.0)
            all_chunk_results.append(cr)
            if len(chunks) > 1:
                prior_nodes.extend({"id": n.id, "text": n.text}
                                    for n in cr.get("nodes", []))
                if len(prior_nodes) > max_carry:
                    prior_nodes = prior_nodes[-max_carry:]
        if len(chunks) > 1:
            mr = self._merge_chunks(all_chunk_results, result)
            result.edges.extend(mr.get("cross_chunk_edges", []))
        valid_ids   = {n.id for n in result.nodes}
        node_id_map = {n.id: n.id for n in result.nodes}
        merge_pairs, cross_edges_dicts = self.linker.find_connections(result.nodes)
        result.merge_suggestions = merge_pairs
        for ed in cross_edges_dicts:
            if ed["source"] in valid_ids and ed["target"] in valid_ids:
                try:
                    result.cross_doc_edges.append(
                        self.assembler.assemble_edge(
                            ed, source, result.document_timestamp, node_id_map,
                            node_lookup={node.id: node for node in result.nodes},
                            source_trust=INGESTION_SOURCE_TYPES.get(
                                result.source_type, INGESTION_SOURCE_TYPES["unknown"])["trust"],
                            source_identity=str((source_metadata or {}).get(
                                "document_content_hash") or source),
                            source_metadata=result.document_metadata))
                except Exception as e:
                    result.warnings.append(f"Cross-doc edge assembly failed: {e}")
        for node in result.nodes:
            if node.embedding is not None:
                self.linker.register_node(
                    node.id, node.embedding,
                    domain=getattr(node, "domain", "unknown"),
                    entity_type=getattr(node, "entity_type", "unknown"),
                    text=getattr(node, "text", ""),
                    timestamp=getattr(node, "timestamp", None),
                )
        result.stats = {
            "chunks": len(chunks), "nodes": len(result.nodes),
            "enrichment_nodes": len(result.enrichment_nodes),
            "influence_observations": len(result.influence_observations),
            "edges": len(result.edges), "zones": len(result.zones),
            "cross_doc_edges": len(result.cross_doc_edges),
            "merge_suggestions": len(result.merge_suggestions),
            "llm_calls": (self.annotator.calls_made - calls_before
                          + classifier.llm_fallback_calls - classifier_calls_before),
            "llm_failures": (self.annotator.failed_calls - failures_before
                             + classifier.llm_fallback_failures - classifier_failures_before),
            "llm_incomplete_calls": (self.annotator.incomplete_calls - incomplete_before
                                     + classifier.llm_fallback_incomplete_calls
                                     - classifier_incomplete_before),
            "total_time_s": round(time.time() - t_start, 2),
            "warnings": len(result.warnings), "errors": len(result.errors),
        }
        logger.info(f"[Ingestion] Done: {result.stats}")
        return result

    # ── Pass D: secondary contexts → existing graph nodes ─────────────────────

    def _find_existing_match(self, node) -> Optional[str]:
        """Return the graph node ID that best matches *node*, or None.

        Uses BucketIndex (embedding ANN + FuzzyMatcher) — the same mechanism
        as CrossDocumentLinker.find_connections() but for a single node.
        A match is accepted only at MERGE_THRESHOLD (0.92 by default) to
        ensure we don't erroneously treat a merely similar node as the same one.
        Called exclusively for date_unknown nodes to decide:
          - match found  → legitimate enrichment of an existing node
          - no match     → reject (no date AND no existing target to enrich)
        """
        if node.embedding is None:
            # No embedding → fall back to graph direct lookup by ID (same source)
            if self._graph is not None and self._graph.get_node(node.id) is not None:
                return node.id
            return None

        emb  = node.embedding
        norm = float(np.linalg.norm(emb))
        if norm < 1e-8:
            return None

        candidates = self.linker.get_candidates(
            ctx_text        = node.text,
            ctx_domain      = node.domain,
            ctx_entity_type = node.entity_type,
            query_embedding = emb,
            top_k           = 5,
        )
        for cand in candidates:
            if cand.get("_sim", 0.0) >= BucketIndex.MERGE_THRESHOLD:
                return cand["id"]

        # Direct graph lookup by stable ID as last resort (same source re-ingestion)
        if self._graph is not None and self._graph.get_node(node.id) is not None:
            return node.id

        return None

    def _pass_d_gather_candidates(self, ctx: Dict, embedder=None) -> List[Dict]:
        """
        Bucket routing + ranked candidate retrieval for one secondary
        context — Stages 1+2 of the original per-context Pass D design,
        factored out so both the fast-path check and the batched LLM call
        in _run_pass_d_batch can share it:
          1. Route to bucket by (ctx.domain, ctx.entity_type) — O(1).
          2. Retrieve top-N candidates via BucketIndex (embedding ANN +
             FuzzyMatcher) for the context's own text AND every alias it
             was given, merged and deduplicated by node_id keeping the
             highest similarity seen — O(bucket_size), not O(N_total).
        """
        ctx_text   = ctx.get("text", "").strip()
        ctx_domain = ctx.get("domain", "unknown")
        ctx_etype  = ctx.get("entity_type", "unknown")
        ctx_year   = ctx.get("year_hint")
        ctx_aliases: List[str] = [a for a in ctx.get("aliases", []) if a and a != ctx_text]
        if not ctx_text:
            return []

        query_emb: Optional[np.ndarray] = None
        if embedder is not None:
            try:
                query_emb = np.array(embedder.encode(ctx_text), dtype=np.float32)
            except Exception:
                pass

        seen_sims: Dict[str, float] = {}
        seen_records: Dict[str, Dict] = {}

        def _absorb(cands: List[Dict]) -> None:
            for c in cands:
                nid = c["id"]
                sim = c.get("_sim", 0.0)
                if nid not in seen_sims or sim > seen_sims[nid]:
                    seen_sims[nid]    = sim
                    seen_records[nid] = c

        _absorb(self.linker.get_candidates(
            ctx_text=ctx_text, ctx_domain=ctx_domain, ctx_entity_type=ctx_etype,
            ctx_year=ctx_year, query_embedding=query_emb,
            top_k=BucketIndex.MAX_CANDIDATES,
        ))
        if ctx.get("role") == "contradicts" and ctx_etype != "unknown":
            _absorb(self.linker.get_candidates(
                ctx_text=ctx_text, ctx_domain=ctx_domain, ctx_entity_type="unknown",
                ctx_year=ctx_year, query_embedding=query_emb,
                top_k=BucketIndex.MAX_CANDIDATES,
            ))
        for alias in ctx_aliases:
            alias_emb: Optional[np.ndarray] = None
            if embedder is not None:
                try:
                    alias_emb = np.array(embedder.encode(alias), dtype=np.float32)
                except Exception:
                    pass
            _absorb(self.linker.get_candidates(
                ctx_text=alias, ctx_domain=ctx_domain, ctx_entity_type=ctx_etype,
                ctx_year=ctx_year, query_embedding=alias_emb,
                top_k=BucketIndex.MAX_CANDIDATES,
            ))

        return sorted(seen_records.values(), key=lambda c: c.get("_sim", 0.0),
                      reverse=True)[:BucketIndex.MAX_CANDIDATES]

    def _run_pass_d_batch(self,
                           node_pairs: List[Tuple[Dict, "KnowledgeNode"]],
                           source: str,
                           timestamp: float,
                           embedder=None,
                           influence_observations: Optional[List[Dict]] = None,
                           source_identity: Optional[str] = None,
                           source_text: str = "",
                           source_metadata: Optional[Dict[str, Any]] = None
                           ) -> Tuple[List[KnowledgeEdge], List[Dict], List[str], int]:
        """
        Replaces the old one-LLM-call-per-secondary-context design with:

          1. Fast-path — a context with zero bucket candidates is skipped
             with NO LLM call: there is nothing to arbitrate, "no match" is
             the only possible conclusion.
          2. Optional fast-path — CONFIG["pass_d_auto_accept_threshold"]
             (off by default) auto-accepts a candidate whose similarity
             already clears it, the same threshold
             BucketIndex.find_connections already uses for same-document
             duplicates with zero LLM calls. Off by default: it skips the
             chronology/plausibility check pure similarity cannot make.
          3. Everything else goes into batches of
             CONFIG["pass_d_batch_size"] items per call instead of one
             call per item.
          4. The plausibility check that used to be a separate Pass D-V
             LLM pass is answered by the SAME call, via "plausibility"/
             "verdict" fields on each result — no second pass over the
             same edges.

        Correctness against cross-item confusion (the reason this returns
        edges built strictly from an item_id-keyed lookup, never from list
        position): every item sent gets a globally unique item_id
        ("<assembled_node.id>::<ctx index>"), builds off `node_pairs`
        rather than re-zipping two separately-filtered lists (see the
        comment at its construction site in _process_chunk for the bug
        that used to cause), and every result is matched back to its item
        BY THAT ID. An LLM reordering, dropping, or duplicating results
        within a batch cannot silently attach one item's context to
        another item's node — unknown or duplicate item_ids in the
        response are logged and ignored rather than guessed at.

        Returns (edges, warnings, llm_calls_made).
        """
        warnings: List[str] = []
        items: List[Dict] = []
        signal_context_roles = {
            "cites", "uses_as_data_source", "funded_by",
            "popularized_by", "depicted_in",
        }
        for raw_node, assembled_node in node_pairs:
            contexts = list(raw_node.get("secondary_contexts", []) or [])
            for claim in raw_node.get("contradicted_claims", []) or []:
                if isinstance(claim, str):
                    claim = {"text": claim}
                if isinstance(claim, dict):
                    contexts.append({**claim, "role": "contradicts"})
            for ctx_idx, ctx in enumerate(contexts):
                if str(ctx.get("role", "")).strip().casefold() in signal_context_roles:
                    continue
                ctx_text = ctx.get("text", "").strip()
                if not ctx_text:
                    continue
                candidates = self._pass_d_gather_candidates(ctx, embedder=embedder)
                if not candidates:
                    continue  # fast-path: nothing to arbitrate, no LLM call, no edge
                items.append({
                    "item_id": f"{assembled_node.id}::{ctx_idx}",
                    "node": raw_node, "ctx": ctx, "candidates": candidates,
                    "assembled_node": assembled_node,
                    "assembled_node_id": assembled_node.id,
                    "ctx_text": ctx_text,
                })

        for signal_index, signal in enumerate(influence_observations or []):
            ctx = {
                "text": str(signal.get("target_text", "")).strip(),
                "domain": signal.get("target_domain", "unknown"),
                "entity_type": signal.get("target_entity_type", "unknown"),
                "year_hint": signal.get("year_hint"),
                "aliases": list(signal.get("aliases", []) or []),
                "role": f"signal:{signal.get('signal_type', '')}",
                "evidence": signal.get("evidence", ""),
            }
            ctx_text = ctx["text"]
            if not ctx_text:
                continue
            candidates = self._pass_d_gather_candidates(ctx, embedder=embedder)
            if not candidates:
                warnings.append(
                    f"Influence signal '{ctx.get('role')}': no existing concept matched '{ctx_text}'")
                continue
            items.append({
                "item_id": f"influence::{signal_index}",
                "node": {"id": f"influence::{signal_index}", "text": ctx_text,
                         "domain": ctx["domain"], "entity_type": ctx["entity_type"]},
                "ctx": ctx,
                "candidates": candidates,
                "assembled_node": None,
                "assembled_node_id": None,
                "ctx_text": ctx_text,
                "is_signal": True,
                "signal": dict(signal),
            })

        edges: List[KnowledgeEdge] = []
        signals: List[Dict] = []
        llm_calls = 0
        if not items:
            return edges, signals, warnings, llm_calls

        auto_accept = CONFIG["pass_d_auto_accept_threshold"]
        remaining: List[Dict] = []
        for it in items:
            top = it["candidates"][0]
            if it.get("is_signal") and top.get("_sim", 0.0) >= auto_accept:
                signals.append({
                    **it["signal"], "target_node_id": top["id"], "source": source,
                    "source_identity": str(it["signal"].get("source_identity") or source),
                    "timestamp": float(timestamp or 0.0),
                    "trust": float(INGESTION_SOURCE_TYPES.get(
                        it["signal"].get("source_type", "unknown"),
                        INGESTION_SOURCE_TYPES["unknown"])["trust"]),
                })
                continue
            remaining.append(it)

        batch_size = CONFIG["pass_d_batch_size"]
        min_score  = CONFIG["edge_validation_min_score"]
        gate_edges = CONFIG.get("edge_validation_enabled", True)
        for i in range(0, len(remaining), batch_size):
            batch = remaining[i:i + batch_size]
            sent_ids = {it["item_id"] for it in batch}
            prompt = self.prompts.build_pass_d_batch_prompt(batch)
            try:
                resp = self.annotator.call(prompt, pass_type="pass_d")
                llm_calls += 1
            except Exception as e:
                warnings.append(f"PassD batch failed: {e}")
                continue

            results: List[Dict] = []
            if resp.success and resp.parsed:
                results = resp.parsed.get("results", []) or []
            elif resp.raw:
                results = _recover_partial_array(resp.raw, "results")
                if results:
                    warnings.append(
                        f"PassD batch: recovered {len(results)}/{len(batch)} "
                        "result(s) from a malformed response"
                    )

            results_by_id: Dict[str, Dict] = {}
            for r in results:
                rid = r.get("item_id")
                if rid not in sent_ids:
                    warnings.append(
                        f"PassD batch: ignored result for unknown item_id "
                        f"'{rid}' (not part of this batch — possible LLM confusion)"
                    )
                    continue
                if rid in results_by_id:
                    warnings.append(
                        f"PassD batch: duplicate result for item_id '{rid}' "
                        "— keeping the first, ignoring the rest"
                    )
                    continue
                results_by_id[rid] = r

            for it in batch:
                r = results_by_id.get(it["item_id"])
                if r is None:
                    warnings.append(
                        f"PassD batch: no result for '{it['ctx_text']}' "
                        f"(node {it['assembled_node_id']}) — LLM failure or truncation"
                    )
                    self.annotator.incomplete_calls += 1
                    continue
                relationship_type = str(r.get("relationship_type", "related")).strip().lower()
                if relationship_type in {"cites", "funded_by", "popularized_by", "depicted_in"}:
                    warnings.append(
                        f"PassD: '{relationship_type}' is recorded as a source signal, not as a graph edge")
                    continue
                is_grounded_refutation_candidate = relationship_type == "contradicts"
                is_status_relation = relationship_type in {"supports", "confirms", "proves"}
                if (gate_edges and not it.get("is_signal")
                        and not is_grounded_refutation_candidate and not is_status_relation
                        and (r.get("verdict") == "drop"
                             or float(r.get("plausibility", 1.0)) < min_score)):
                    continue
                match_id = r.get("match")
                if not match_id or match_id in ("null", "none"):
                    continue
                # Never trust a match_id the LLM invented outside what it was
                # actually given for THIS item — same guard the old
                # per-context design had, checked against this item's own
                # candidate list specifically (never another item's list,
                # which the batching makes newly possible to confuse).
                valid_match = any(c["id"] == match_id for c in it["candidates"])
                if not valid_match and self._graph is not None:
                    try:
                        valid_match = self._graph.get_node(match_id) is not None
                    except Exception:
                        pass
                if not valid_match:
                    warnings.append(
                        f"PassD batch: match_id '{match_id}' for "
                        f"'{it['ctx_text']}' is not in that item's own "
                        "candidate list or the graph — ignored"
                    )
                    continue
                if it.get("is_signal"):
                    signals.append({
                        **it["signal"], "target_node_id": match_id, "source": source,
                        "source_identity": str(it["signal"].get("source_identity") or source),
                        "timestamp": float(timestamp or 0.0),
                        "trust": float(INGESTION_SOURCE_TYPES.get(
                            it["signal"].get("source_type", "unknown"),
                            INGESTION_SOURCE_TYPES["unknown"])["trust"]),
                    })
                    continue
                edge_evidence = r.get("evidence", []) or []
                if isinstance(edge_evidence, str):
                    edge_evidence = [edge_evidence]
                source_quotes = [str(q) for q in (it["node"].get("evidence", []) or [])]
                if is_status_relation:
                    source_status = parse_label_status(it["node"].get("label_status"))
                    candidate = next((c for c in it["candidates"]
                                      if c.get("id") == match_id), {})
                    if source_status not in {LabelStatus.FACT, LabelStatus.PROOF}:
                        warnings.append(
                            f"PassD: {relationship_type} edge from "
                            f"'{it['assembled_node_id']}' requires a fact/proof source label")
                        continue
                    if relationship_type == "proves" and source_status != LabelStatus.PROOF:
                        warnings.append(
                            f"PassD: proves edge from '{it['assembled_node_id']}' "
                            "requires a proof-status source label")
                        continue
                    edge_evidence = [
                        str(quote) for quote in edge_evidence
                        if isinstance(quote, str) and quote in source_quotes
                        and self.validator._quote_supports_claim(
                            quote, str(candidate.get("text", "")))
                    ]
                    if not edge_evidence:
                        warnings.append(
                            f"PassD: {relationship_type} edge from "
                            f"'{it['assembled_node_id']}' has no exact quote supporting "
                            f"candidate '{match_id}'")
                        continue
                if relationship_type == "contradicts":
                    if parse_label_status(it["node"].get("label_status")) != LabelStatus.REFUTATION:
                        warnings.append(
                            f"PassD: contradiction edge from '{it['assembled_node_id']}' "
                            "was rejected because its label is not classified as refutation")
                        continue
                    grounded_quotes = [str(q) for q in edge_evidence
                                       if isinstance(q, str) and q in source_quotes]
                    if not grounded_quotes:
                        warnings.append(
                            f"PassD: contradiction edge from '{it['assembled_node_id']}' "
                            "was rejected because no edge quote matched its source evidence")
                        continue
                    edge_evidence = grounded_quotes[:3]
                else:
                    grounded_quotes = [str(q) for q in edge_evidence
                                       if isinstance(q, str) and q in source_quotes
                                       and self.validator._quote_is_grounded(q, source_text)]
                    if not grounded_quotes:
                        warnings.append(
                            f"PassD: {relationship_type} edge from '{it['assembled_node_id']}' "
                            "has no exact source quote stating the relationship")
                        continue
                    edge_evidence = grounded_quotes[:3]
                parameter_observations, parameter_warnings = (
                    self.validator._validate_parameter_observations(
                        r.get("parameter_observations", []), source_text=source_text,
                        required_quotes=edge_evidence,
                    ))
                warnings.extend(
                    f"PassD item {it['item_id']}: {warning}"
                    for warning in parameter_warnings)
                target_candidate = next((c for c in it["candidates"]
                                         if c.get("id") == match_id), {})
                edge = self.assembler.assemble_edge(
                    {"source": it["assembled_node_id"], "target": match_id,
                     "relationship_type": relationship_type,
                     "evidence": edge_evidence,
                     "parameter_observations": parameter_observations},
                    source, timestamp,
                    {it["assembled_node_id"]: it["assembled_node_id"],
                     match_id: match_id},
                    node_lookup={
                        it["assembled_node_id"]: it["assembled_node"],
                        match_id: (self._graph.get_node(match_id)
                                   if self._graph is not None else target_candidate),
                    },
                    source_trust=float(getattr(
                        it["assembled_node"], "__dict__", {}).get("source_trust", 0.5)),
                    source_identity=source_identity or source,
                    source_metadata=source_metadata,
                )
                if relationship_type == "contradicts" or is_status_relation:
                    source_status = parse_label_status(it["node"].get("label_status"))
                    edge.__dict__["source_label_status_code"] = int(source_status)
                    edge.__dict__["source_observation"] = {
                        "source_node_id": it["assembled_node_id"],
                        "source_node_text": str(it["node"].get("text", "")),
                        "timestamp": float(timestamp or 0.0),
                        "label_status_code": int(source_status),
                        "label_status": label_status_name(source_status),
                        "status_evidence": str(it["node"].get("status_evidence", "")),
                        "evidence": list(source_quotes),
                        "context": str(it["node"].get("full_context", "")),
                        "source_metadata": dict(getattr(
                            it["assembled_node"], "__dict__", {}).get("source_metadata", {})),
                    }
                edges.append(edge)

        return edges, signals, warnings, llm_calls

    def _process_chunk(self, chunk: DocumentChunk, source: str,
                        technical_timestamp: float = 0.0,
                        prior_nodes: Optional[List[Dict]] = None,
                        source_metadata: Optional[Dict[str, Any]] = None) -> Dict:
        out = {"nodes": [], "edges": [], "zones": [], "containment_edges": [],
               "warnings": [], "errors": [], "llm_calls": 0, "raw_nodes": [],
               "pass_d_edges": [], "influence_observations": [], "enrichment_nodes": [],
               "source_type": "unknown", "document_timestamp": 0.0,
               "document_metadata": {}}

        unified = CONFIG.get("ingestion_unified_extraction", True)
        prior_ids = {n["id"] for n in (prior_nodes or [])}

        # ── Pass A (unified mode: +B+C in the SAME response) ────────────────────
        # See CONFIG["ingestion_unified_extraction"] for why this is
        # switchable rather than a straight replacement: fewer calls but a
        # longer single generation, a trade not yet measured on real
        # documents. Everything downstream of this branch — validation,
        # node assembly, dependency remap, edge/zone assembly, Pass D — is
        # the SAME code either way; only how raw_nodes/physical/edges are
        # obtained differs.
        if unified:
            ra = self.annotator.call(
                self.prompts.build_unified_prompt(chunk, prior_nodes=prior_nodes),
                pass_type="unified")
            out["llm_calls"] += 1
            if (not ra.success or not ra.parsed) and ra.raw:
                recovered_nodes = _recover_partial_array(ra.raw, "nodes")
                if recovered_nodes:
                    out["warnings"].append(
                        f"Chunk {chunk.chunk_index}: unified extraction response "
                        f"was malformed — recovered {len(recovered_nodes)} node(s) "
                        "from a partial parse. edges/temporal_zones from this "
                        "response are discarded rather than guessed at (the "
                        "prompt orders \"nodes\" before them for exactly this "
                        "reason — the truncation lands in the cheaper, "
                        "re-derivable tail more often than not)."
                    )
                    ra = LLMResponse(
                        raw=ra.raw,
                        parsed={"document_metadata": {}, "domain": "",
                                "nodes": recovered_nodes, "edges": [],
                                "temporal_zones": []},
                        success=True, attempts=ra.attempts,
                    )
        else:
            ra = self.annotator.call(
                self.prompts.build_entity_prompt(chunk, prior_nodes=prior_nodes),
                pass_type="entities")
            out["llm_calls"] += 1

        if not ra.success or not ra.parsed:
            out["errors"].append(f"Chunk {chunk.chunk_index}: entity extraction failed")
            return out
        source_type, timestamp, document_metadata = self._document_observation_metadata(
            ra.parsed.get("document_metadata"))
        document_metadata = self._merge_document_source_metadata(
            document_metadata, source_metadata, source)
        trust = INGESTION_SOURCE_TYPES.get(
            source_type, INGESTION_SOURCE_TYPES["unknown"])["trust"]
        out["source_type"] = source_type
        out["document_timestamp"] = timestamp
        out["document_metadata"] = document_metadata
        if timestamp == 0.0:
            out["warnings"].append(
                "Document date not found in content; no filesystem timestamp was used."
            )
        # source_text enables ResponseValidator's evidence-grounding check —
        # every "evidence"/secondary_context quote is verified against THIS
        # text (what the model actually saw), not the whole document, since
        # a multi-chunk document only ever shows the model one chunk at a time.
        val_n = self.validator.validate_nodes(ra.parsed, source_text=chunk.text)
        out["warnings"].extend(val_n.warnings)
        out["errors"].extend(val_n.errors)
        val_i = self.validator.validate_influence_observations(ra.parsed, source_text=chunk.text)
        out["warnings"].extend(val_i.warnings)
        raw_influences = val_i.fixed.get("influence_observations", [])
        raw_nodes = val_n.fixed.get("nodes", [])
        context_signal_types = {
            "cites": "citation",
            "uses_as_data_source": "data_use",
            "funded_by": "investment",
            "popularized_by": "social_attention",
            "depicted_in": "fictional_hype",
        }
        for raw_node in raw_nodes:
            for context in raw_node.get("secondary_contexts", []) or []:
                signal_type = context_signal_types.get(
                    str(context.get("role", "")).strip().casefold())
                if not signal_type:
                    continue
                candidate = {
                    "target_text": context.get("text", ""),
                    "target_domain": context.get("domain", raw_node.get("domain", "unknown")),
                    "target_entity_type": context.get(
                        "entity_type", raw_node.get("entity_type", "unknown")),
                    "year_hint": context.get("year_hint"),
                    "aliases": context.get("aliases", []),
                    "signal_type": signal_type,
                    "evidence": context.get("evidence", ""),
                }
                checked = self.validator.validate_influence_observations(
                    {"influence_observations": [candidate]}, source_text=chunk.text)
                out["warnings"].extend(checked.warnings)
                raw_influences.extend(checked.fixed.get("influence_observations", []))
        unique_influences = {}
        for item in raw_influences:
            key = (str(item.get("target_text", "")).casefold(),
                   str(item.get("signal_type", "")).casefold(),
                   str(item.get("evidence", "")))
            unique_influences[key] = item
        raw_influences = list(unique_influences.values())
        if not raw_nodes and not raw_influences:
            out["warnings"].append(f"Chunk {chunk.chunk_index}: no valid nodes extracted")
            return out
        slug_map: Dict[str, str] = {}

        # ── Physical axes: inline per node in unified mode, a separate Pass B
        # call otherwise. Reshaped into the same {"physical_scores": {id:
        # axes}} structure validate_physical already expects either way, so
        # node assembly below never needs to know which mode produced it.
        if not raw_nodes:
            physical_map = {}
        elif unified:
            physical_raw = {"physical_scores":
                             {n["id"]: n.get("physical_axes", {}) for n in raw_nodes}}
            val_p = self.validator.validate_physical(physical_raw, {n["id"] for n in raw_nodes})
            out["warnings"].extend(val_p.warnings)
            physical_map = val_p.fixed.get("physical_scores", {})
        else:
            rb = self.annotator.call(
                self.prompts.build_physical_prompt(raw_nodes, chunk.text),
                pass_type="physical")
            out["llm_calls"] += 1
            physical_map = {}
            if rb.success and rb.parsed:
                val_p = self.validator.validate_physical(rb.parsed, {n["id"] for n in raw_nodes})
                out["warnings"].extend(val_p.warnings)
                physical_map = val_p.fixed.get("physical_scores", {})

        # ── Assemble nodes ────────────────────────────────────────────────────
        assembled_nodes: List = []
        # (raw_item, assembled_node) pairs, built HERE — at the point where
        # both objects are simultaneously in scope — for every node that
        # enters the graph in ANY form (normal or enrichment). The previous
        # design reconstructed this pairing after the loop by zip()-ing
        # out["raw_nodes"] against assembled_nodes; those two lists silently
        # drift out of step the moment ANY node in the chunk takes the
        # enrichment branch below (it appends to assembled_nodes but not to
        # out["raw_nodes"]), so zip() would pair every subsequent raw item
        # with the WRONG assembled node — Pass D would then read one node's
        # secondary_contexts but attach the resulting edge to a different
        # node entirely. Building the pair inline makes that misalignment
        # structurally impossible rather than merely unlikely.
        node_pairs: List[Tuple[Dict, "KnowledgeNode"]] = []
        for item in raw_nodes:
            try:
                edge_endpoint_only = bool(item.get("edge_endpoint_only"))
                # TODO #19: node.id is the CanonicalKey (hex) unconditionally
                # now — the MD5(source+text) path has been removed, not just
                # branched around.
                node = self.canonical_assembler.assemble_node_with_canonical_key(
                    item, source, timestamp, source_type=source_type,
                    source_trust=trust, physical_scores=physical_map.get(item["id"]),
                    record_observation=not edge_endpoint_only)
                evidence = item.get("evidence", [])
                if isinstance(evidence, str):
                    evidence = [evidence]
                evidence = [str(quote)[:1000] for quote in evidence
                            if isinstance(quote, str) and quote.strip()][:5] \
                    if isinstance(evidence, list) else []
                source_url = document_metadata.get("source_url")
                if not source_url and re.match(r"^https?://", source, re.IGNORECASE):
                    source_url = source
                node.__dict__["source_evidence"] = evidence
                node.__dict__["source_context"] = str(item.get("full_context", ""))[:2000]
                node.__dict__["source_trust"] = float(trust)
                node_status = (
                    parse_label_status(item.get("label_status_code"))
                    or parse_label_status(item.get("label_status"))
                    or LabelStatus.IDEA_CONCEPT
                )
                node.__dict__["label_status_code"] = int(node_status)
                node.__dict__["label_status"] = label_status_name(
                    node.__dict__["label_status_code"])
                node.__dict__["status_evidence"] = str(
                    item.get("status_evidence") or "")[:1000]
                if edge_endpoint_only:
                    node.__dict__["edge_endpoint_only"] = True
                node.__dict__["source_metadata"] = {
                    "title": document_metadata.get("title"),
                    "authors": list(document_metadata.get("authors", [])),
                    "publication": document_metadata.get("publication"),
                    "forum_topic": document_metadata.get("forum_topic"),
                    "source_type": source_type,
                    "publication_date": document_metadata.get("publication_date"),
                    "date_evidence": document_metadata.get("date_evidence"),
                    "source_url": source_url,
                    "document_format": document_metadata.get("document_format"),
                    "document_container": document_metadata.get("document_container"),
                }
                node.__dict__["source_observations"] = [{
                    "source": str(source),
                    "timestamp": float(node.timestamp) if node.timestamp else None,
                    "year": _ts_to_year(node.timestamp) if node.timestamp else None,
                    "trust": float(trust),
                    "title": document_metadata.get("title"),
                    "authors": list(document_metadata.get("authors", [])),
                    "publication": document_metadata.get("publication"),
                    "forum_topic": document_metadata.get("forum_topic"),
                    "source_type": source_type,
                    "publication_date": document_metadata.get("publication_date"),
                    "date_evidence": document_metadata.get("date_evidence"),
                    "source_url": source_url,
                    "source_metadata": dict(node.__dict__["source_metadata"]),
                    "evidence": list(evidence),
                    "context": node.__dict__["source_context"],
                    "label_status_code": node.__dict__["label_status_code"],
                    "label_status": node.__dict__["label_status"],
                    "status_evidence": node.__dict__["status_evidence"],
                }]
                canonical_observations = self.canonical_assembler.observations.get(node.id, [])
                if canonical_observations and not edge_endpoint_only:
                    canonical_observation = canonical_observations[-1]
                    canonical_observation.delta_fields["source_evidence"] = evidence
                    canonical_observation.delta_fields["source_context"] = node.__dict__["source_context"]
                    canonical_observation.delta_fields["source_metadata"] = dict(node.__dict__["source_metadata"])
                    canonical_observation.delta_fields["label_status_code"] = node.__dict__["label_status_code"]
                    canonical_observation.delta_fields["label_status"] = node.__dict__["label_status"]
                    canonical_observation.delta_fields["status_evidence"] = node.__dict__["status_evidence"]
                slug_map[item["id"]] = node.id

                # ── Date-unknown gate ─────────────────────────────────────────
                # A node without a date (date_unknown=True) is not allowed to
                # enter the graph as a *new* node — the graph is a temporal
                # knowledge structure and undated new nodes break ordering,
                # AncestryEnforcer, and all time-based pressure calculations.
                #
                # Exception: if the node matches an existing graph node via
                # BucketIndex (sim >= MERGE_THRESHOLD), it is legitimate
                # enrichment data.  It goes into out["enrichment_nodes"] as a
                # (node, matched_graph_id) pair; load_result_into_graph will
                # route it to GraphConsolidator instead of add_node_checked().
                if node.__dict__.get("date_unknown"):
                    matched_id = self._find_existing_match(node)
                    if matched_id is not None:
                        out["enrichment_nodes"].append((node, matched_id))
                        out["warnings"].append(
                            f"Node '{node.id}' ('{node.text[:50]}'): no date — "
                            f"treated as enrichment of existing '{matched_id}'"
                        )
                        # This node's own cid_hex (node.id) never becomes a
                        # graph.nodes key — it is routed to GraphConsolidator
                        # against matched_id instead (see load_result_into_graph).
                        # Any sibling node in this chunk whose requires_node_ids/
                        # solves_limitations references this slug must resolve
                        # to matched_id, not to a dangling id that will never
                        # exist in the graph. Overwrites the assemble-time
                        # mapping set just above.
                        slug_map[item["id"]] = matched_id
                        # Still register slug for edge assembly so edges within
                        # the chunk that reference this node remain valid.
                        assembled_nodes.append(node)
                        node_pairs.append((item, node))
                    else:
                        out["warnings"].append(
                            f"Node '{node.id}' ('{node.text[:50]}'): "
                            "no date and no match in graph — creation rejected. "
                            "Provide publication_year to create this node."
                        )
                    continue  # do NOT add to out["nodes"] in either case

                out["nodes"].append(node)
                out["raw_nodes"].append(item)
                assembled_nodes.append(node)
                node_pairs.append((item, node))
            except Exception as e:
                out["errors"].append(f"Node assembly failed for '{item['id']}': {e}")

        # ── Resolve intra-document dependency references ───────────────────────
        # requires_node_ids / solves_limitations / enables_node_ids are emitted
        # by the LLM using the SAME per-document slugs as item["id"] (the
        # extraction prompt is explicit that these ids are "internal to this
        # document" — see the prompt's SECONDARY CONTEXT RULES). They are
        # never the canonical node.id (12-byte CanonicalKey hex) that
        # assemble_node_with_canonical_key() just assigned above — until now
        # nothing translated one into the other, so every non-empty entry was
        # a dangling slug matching no key in RuntimeGraph.nodes, exactly the
        # failure Pass C's assemble_edge(..., slug_map) already guards against
        # for regular edges. Run over assembled_nodes (not just out["nodes"])
        # so enrichment nodes' dependency lists are also clean before
        # GraphConsolidator UNION_LISTs them into the matched node's record —
        # otherwise a dangling slug becomes permanent the moment it's merged
        # into a long-lived canonical node.
        #
        # prior_ids are already real canonical node.id values (they came from
        # an EARLIER chunk's out["nodes"], not this chunk's slug namespace —
        # see the carry-forward registry built in ingest()), so they resolve
        # as-is rather than through slug_map.
        dependency_fields = ("requires_node_ids", "solves_limitations", "enables_node_ids")
        for node in assembled_nodes:
            for field_name in dependency_fields:
                raw_ids = getattr(node, field_name, None)
                if not raw_ids:
                    continue
                resolved, dropped = [], []
                for raw_id in raw_ids:
                    mapped = slug_map.get(raw_id) or (raw_id if raw_id in prior_ids else None)
                    (resolved if mapped else dropped).append(mapped or raw_id)
                setattr(node, field_name, resolved)
                if dropped:
                    out["warnings"].append(
                        f"Node '{node.id}': {field_name} referenced "
                        f"{len(dropped)} unresolved slug(s) {dropped} — dropped "
                        "(referenced node failed assembly, was rejected by the "
                        "date-unknown gate, or was never extracted)"
                    )

        # ── Pass C: edges within document ───────────────────────────────────────
        if not raw_nodes:
            rc_parsed, rc_ok = None, False
        elif unified:
            rc_parsed, rc_ok = ra.parsed, True
        else:
            rc = self.annotator.call(
                self.prompts.build_edge_prompt(raw_nodes, chunk.text),
                pass_type="edges")
            out["llm_calls"] += 1
            rc_parsed, rc_ok = rc.parsed, rc.success
        if rc_ok and rc_parsed:
            edge_data = dict(rc_parsed)
            edge_data.setdefault("nodes", raw_nodes)
            val_e = self.validator.validate_edges(
                edge_data, set(slug_map.keys()) | prior_ids, source_text=chunk.text)
            out["warnings"].extend(val_e.warnings)
            for ed in val_e.fixed.get("edges", []):
                try:
                    edge_node_lookup = {}
                    for raw_node, assembled_node in node_pairs:
                        edge_node_lookup[raw_node.get("id")] = assembled_node
                        edge_node_lookup[assembled_node.id] = assembled_node
                    if self._graph is not None:
                        for node_id in (ed.get("source"), ed.get("target")):
                            mapped_id = slug_map.get(node_id, node_id)
                            existing_node = self._graph.get_node(mapped_id)
                            if existing_node is not None:
                                edge_node_lookup[node_id] = existing_node
                                edge_node_lookup[mapped_id] = existing_node
                    edge = self.assembler.assemble_edge(
                        ed, source, timestamp, slug_map,
                        node_lookup=edge_node_lookup, source_trust=trust,
                        source_identity=str(document_metadata.get("document_content_hash") or source),
                        source_metadata=document_metadata)
                    source_node = next(
                        (candidate for candidate in assembled_nodes
                         if candidate.id == edge.source_id), None)
                    source_private = getattr(source_node, "__dict__", {}) if source_node else {}
                    if source_private.get("label_status_code") is not None:
                        edge.__dict__["source_label_status_code"] = int(
                            source_private["label_status_code"])
                        edge.__dict__["source_observation"] = {
                            "source_node_id": source_node.id,
                            "source_node_text": source_node.text,
                            "timestamp": float(source_node.timestamp or timestamp),
                            "trust": float(source_private.get("source_trust", trust)),
                            "label_status_code": source_private["label_status_code"],
                            "label_status": source_private.get("label_status"),
                            "status_evidence": source_private.get("status_evidence", ""),
                            "evidence": list(source_private.get("source_evidence", [])),
                            "source_metadata": dict(source_private.get("source_metadata", {})),
                        }
                    out["edges"].append(edge)
                except Exception as e:
                    out["edges_errors"] = out.get("edge_errors", [])
                    out["errors"].append(f"Edge assembly failed: {e}")
            valid_stable = {n.id for n in out["nodes"]} | prior_ids
            for zi in val_e.fixed.get("temporal_zones", []):
                try:
                    zone, zone_node = self.assembler.assemble_zone(
                        zi, timestamp, valid_stable, slug_map)
                    out["zones"].append(zone)
                    out["nodes"].append(zone_node)
                    for member_id in zone.contained_node_ids:
                        out["containment_edges"].append(KnowledgeEdge(
                            id=str(uuid.uuid4())[:8],
                            source_id=zone.id, target_id=member_id,
                            relationship_type="contains", is_containment_edge=True,
                            timestamp=timestamp, confidence=1.0))
                except Exception as e:
                    out["errors"].append(f"Zone assembly failed: {e}")

        # ── Pass C-V: semantic validation of intra-doc edges (optional) ────────
        if "intra_doc" in CONFIG.get("edge_validation_passes", []):
            node_lookup = {n.id: n for n in out["nodes"]}
            kept, dropped = self.edge_validator.validate_batch(
                out["edges"], node_lookup)
            if dropped:
                out["warnings"].append(
                    f"Chunk {chunk.chunk_index}: EdgeValidator dropped "
                    f"{len(dropped)} intra-doc edge(s)"
                )
            out["edges"] = kept

        # ── Pass D: secondary contexts → existing graph nodes (batched) ────────
        # Built from node_pairs (see its construction above for why, not from
        # zip(out["raw_nodes"], assembled_nodes)).
        try:
            pass_d_edges, pass_d_signals, pass_d_warnings, pass_d_calls = self._run_pass_d_batch(
                node_pairs=node_pairs, source=source, timestamp=timestamp,
                embedder=self.assembler.embedder,
                influence_observations=[
                    {**item, "source_type": source_type,
                     "source_identity": str(document_metadata.get("document_content_hash") or source)}
                    for item in raw_influences],
                source_identity=str(document_metadata.get("document_content_hash") or source),
                source_text=chunk.text,
                source_metadata=document_metadata,
            )
            out["pass_d_edges"].extend(pass_d_edges)
            out["influence_observations"].extend(pass_d_signals)
            out["warnings"].extend(pass_d_warnings)
            out["llm_calls"] += pass_d_calls
        except Exception as e:
            out["warnings"].append(f"PassD batch failed: {e}")

        # Pass D-V is gone as a separate pass: _run_pass_d_batch's own batched
        # call already returns "plausibility"/"verdict" for each match and
        # filters on it inline (see CONFIG["edge_validation_enabled"]) — a
        # second EdgeSemanticValidator.validate_batch call here would be
        # revalidating the same decision at full extra cost, exactly the
        # redundancy this whole redesign was meant to remove.

        out["slug_map"] = dict(slug_map)
        return out

    def _merge_chunks(self, chunk_results: List[Dict],
                       result: IngestionResult) -> Dict:
        result_nodes = {node.id: node for node in result.nodes}
        summaries = []
        for index, chunk_result in enumerate(chunk_results):
            nodes = []
            slug_map = chunk_result.get("slug_map", {})
            for raw_node in chunk_result.get("raw_nodes", []):
                canonical_id = slug_map.get(raw_node.get("id"), raw_node.get("id"))
                node = result_nodes.get(canonical_id)
                if node is None:
                    continue
                private = getattr(node, "__dict__", {})
                nodes.append({
                    "id": canonical_id,
                    "text": node.text,
                    "label_status": private.get("label_status"),
                    "evidence": list(private.get("source_evidence", [])),
                })
            summaries.append({"chunk": index, "nodes": nodes})
        response = self.annotator.call(
            self.prompts.build_merge_prompt(summaries), pass_type="merge")
        if not response.success or not response.parsed:
            return {"merged": [], "cross_chunk_edges": []}
        raw_xedges = response.parsed.get("cross_chunk_edges", [])
        valid_ids = set(result_nodes)
        cross_edges = []
        for ed in raw_xedges:
            src_id = ed.get("source", "")
            tgt_id = ed.get("target", "")
            if src_id in valid_ids and tgt_id in valid_ids and src_id != tgt_id:
                try:
                    relationship_type = str(ed.get("relationship_type", "related")).strip().lower()
                    if relationship_type in {"cites", "funded_by", "popularized_by", "depicted_in"}:
                        continue
                    evidence = ed.get("evidence", []) or []
                    if isinstance(evidence, str):
                        evidence = [evidence]
                    source_node = result_nodes[src_id]
                    source_private = getattr(source_node, "__dict__", {})
                    source_status = parse_label_status(
                        source_private.get("label_status_code"))
                    source_quotes = source_private.get("source_evidence", []) or []
                    target_node = result_nodes[tgt_id]
                    if relationship_type in {"supports", "confirms", "proves"}:
                        if source_status not in {LabelStatus.FACT, LabelStatus.PROOF}:
                            continue
                        if relationship_type == "proves" and source_status != LabelStatus.PROOF:
                            continue
                        evidence = [quote for quote in evidence
                                    if isinstance(quote, str) and quote in source_quotes
                                    and self.validator._quote_supports_claim(
                                        quote, target_node.text)]
                        if not evidence:
                            continue
                    elif relationship_type == "contradicts":
                        if source_status != LabelStatus.REFUTATION:
                            continue
                        evidence = [quote for quote in evidence
                                    if isinstance(quote, str) and quote in source_quotes]
                        if not evidence:
                            continue
                    else:
                        evidence = [quote for quote in evidence
                                    if isinstance(quote, str) and quote in source_quotes
                                    and self.validator._quote_is_grounded(quote, "\n".join(source_quotes))]
                        if not evidence:
                            continue
                    source_identity = str(result.document_metadata.get(
                        "document_content_hash") or result.source)
                    edge = self.assembler.assemble_edge(
                        {"source": src_id, "target": tgt_id,
                         "relationship_type": relationship_type, "evidence": evidence},
                        result.source, result.document_timestamp,
                        {src_id: src_id, tgt_id: tgt_id},
                        node_lookup={src_id: source_node, tgt_id: target_node},
                        source_trust=INGESTION_SOURCE_TYPES.get(
                            result.source_type, INGESTION_SOURCE_TYPES["unknown"])["trust"],
                        source_identity=source_identity,
                        source_metadata=result.document_metadata)
                    if relationship_type == "contradicts" or relationship_type in {
                            "supports", "confirms", "proves"}:
                        edge.__dict__["source_label_status_code"] = int(source_status)
                        edge.__dict__["source_observation"] = {
                            "source_node_id": source_node.id,
                            "source_node_text": source_node.text,
                            "timestamp": float(source_node.timestamp or 0.0),
                            "trust": float(source_private.get("source_trust", 0.5)),
                            "label_status_code": int(source_status),
                            "label_status": label_status_name(source_status),
                            "status_evidence": source_private.get("status_evidence", ""),
                            "evidence": list(source_quotes),
                            "source_metadata": dict(source_private.get("source_metadata", {})),
                        }
                    edge.compute_total_weight()
                    cross_edges.append(edge)
                except Exception:
                    pass
        return {"merged": response.parsed.get("merge_groups", []),
                "cross_chunk_edges": cross_edges}

@dataclass
class BatchDocument:
    text:        str
    source:      str
    # Kept at zero for library ingestion.  Historical time and source class
    # are inferred from the document by IngestionPipeline, never supplied by
    # the UI or the filesystem.
    timestamp:   float = 0.0
    metadata:    Dict[str, Any] = field(default_factory=dict)


class BatchIngestionProcessor:
    """
    Processes a queue of documents with rate-limiting on LLM calls.
    Stops after max_errors consecutive failures.
    """
    def __init__(self, pipeline: IngestionPipeline,
                 rate_limit_rpm: int = 60, max_errors: int = 5):
        self.pipeline    = pipeline
        self.min_delay_s = 60.0 / max(rate_limit_rpm, 1)
        self.max_errors  = max_errors

    def process(self, documents: List[BatchDocument],
                progress_callback=None) -> List[IngestionResult]:
        results:      List[IngestionResult] = []
        error_streak: int   = 0
        last_call_t:  float = 0.0
        for i, doc in enumerate(documents):
            elapsed = time.time() - last_call_t
            if elapsed < self.min_delay_s:
                time.sleep(self.min_delay_s - elapsed)
            logger.info(f"[Batch] Processing {i+1}/{len(documents)}: {doc.source}")
            try:
                result = self.pipeline.ingest(
                    text=doc.text, source=doc.source, timestamp=doc.timestamp,
                    source_metadata=doc.metadata)
                results.append(result)
                error_streak = 0
                last_call_t  = time.time()
                if progress_callback:
                    progress_callback(i + 1, len(documents), result)
            except Exception as e:
                logger.error(f"[Batch] Failed on '{doc.source}': {e}")
                error_streak += 1
                if error_streak >= self.max_errors:
                    logger.critical(f"[Batch] {self.max_errors} consecutive errors — stopping")
                    break
        logger.info(f"[Batch] Complete: {len(results)}/{len(documents)} docs, "
                    f"{sum(len(r.nodes) for r in results)} nodes, "
                    f"{sum(len(r.edges)+len(r.cross_doc_edges) for r in results)} edges")
        return results


def load_result_into_graph(result: IngestionResult, graph,
                            consolidator=None,
                            edge_validator=None) -> Dict[str, int]:
    """Load an IngestionResult into a RuntimeGraph with ancestry invariant enforcement.

    Every non-primordial node must arrive with at least one incoming
    non-containment edge.  When the document doesn't declare one, the
    AncestryEnforcer synthesises a DERIVED_FROM edge to the best-matching
    predecessor already in the graph.

    Parameters
    ----------
    result : IngestionResult
    graph  : RuntimeGraph
    consolidator : GraphConsolidator, optional
        If supplied, enrichment_nodes (date_unknown observations that matched
        an existing graph node) are routed through it rather than creating
        new graph nodes.
    edge_validator : EdgeSemanticValidator, optional
        If supplied AND ``CONFIG["edge_validation_passes"]`` contains
        ``"graph_load"``, all non-containment edges are passed through the
        LLM plausibility filter as a final gate before ``add_edge()``.
        This is the most conservative option — it fires even for edges
        synthesised by AncestryEnforcer.  Containment edges are always
        exempt (zone membership is structural, not semantic).

    Returns
    -------
    dict with keys: nodes, edges, zones, synthesised_ancestry,
    enrichment_routed, validation_dropped.
    """
    added = {"nodes": 0, "edges": 0, "zones": 0,
             "synthesised_ancestry": 0, "enrichment_routed": 0,
             "validation_dropped": 0}

    all_edges = result.edges + result.containment_edges + result.cross_doc_edges

    for node in result.nodes:
        # Pass the full edge batch so the enforcer can see edges arriving
        # alongside this node in the same IngestionResult (avoids false
        # positives when source and target land in the same document).
        anchor = graph.add_node_checked(node, batch_edges=all_edges)
        added["nodes"] += 1
        if anchor is not None:
            # anchor was already written to the graph by add_node_checked()
            added["synthesised_ancestry"] += 1
            added["edges"] += 1

    # ── graph_load validation pass (optional) ────────────────────────────────
    # Runs BEFORE add_edge() so bad edges never enter the graph at all.
    # Only non-containment edges are checked; containment (zone membership)
    # edges are structural and always trusted.
    if (edge_validator is not None
            and "graph_load" in CONFIG.get("edge_validation_passes", [])):
        non_containment = [e for e in all_edges if not e.is_containment_edge]
        containment     = [e for e in all_edges if e.is_containment_edge]
        # Build node lookup from both the result and the live graph
        node_lookup: Dict[str, "KnowledgeNode"] = {
            n.id: n for n in result.nodes}
        for edge in non_containment:
            for nid in (edge.source_id, edge.target_id):
                if nid not in node_lookup:
                    gn = graph.get_node(nid)
                    if gn:
                        node_lookup[gn.id] = gn
        kept, dropped = edge_validator.validate_batch(non_containment, node_lookup)
        all_edges = kept + containment
        added["validation_dropped"] += len(dropped)
        if dropped:
            logger.info(
                f"[load_result] graph_load pass dropped "
                f"{len(dropped)} edge(s) before add_edge()"
            )

    for edge in all_edges:
        if graph.get_node(edge.source_id) and graph.get_node(edge.target_id):
            if str(edge.relationship_type).casefold() == "contradicts":
                source_node = graph.get_node(edge.source_id)
                source_private = getattr(source_node, "__dict__", {})
                source_observation = getattr(edge, "__dict__", {}).get(
                    "source_observation", {})
                if not isinstance(source_observation, dict) or not source_observation:
                    source_observation = {
                        "source_node_id": source_node.id,
                        "source_node_text": source_node.text,
                        "timestamp": float(source_node.timestamp or edge.timestamp),
                        "trust": float(source_private.get("source_trust", 0.5)),
                        "label_status_code": source_private.get("label_status_code"),
                        "label_status": source_private.get("label_status"),
                        "status_evidence": source_private.get("status_evidence", ""),
                        "evidence": list(source_private.get("source_evidence", [])),
                        "source_metadata": dict(source_private.get("source_metadata", {})),
                    }
                edge.__dict__["source_observations"] = [{
                    "source": result.source,
                    "timestamp": float(edge.timestamp or source_node.timestamp),
                    "trust": float(source_private.get("source_trust", 0.5)),
                    "confidence": float(edge.confidence),
                    "relationship_type": edge.relationship_type,
                    "source_label_status_code": getattr(edge, "__dict__", {}).get(
                        "source_label_status_code", source_private.get("label_status_code")),
                    "source_observation": source_observation,
                    "evidence": list(edge.evidence or []),
                }]
            graph.add_edge(edge)
            added["edges"] += 1

    if any(str(edge.relationship_type).casefold() == "contradicts" for edge in all_edges):
        from .consolidation import GraphConsolidator
        GraphConsolidator._rebuild_label_support(graph)

    # NOTE: result.zones only ever fed the "zones" count here — the
    # TemporalZone objects themselves were never registered into
    # graph.zones. Their zone_node (a KnowledgeNode) and containment edges
    # do get added above via result.nodes / all_edges, but without
    # graph.add_zone() the zone never exists as a queryable field: nothing
    # in RuntimeGraph.zones_covering_node(), measured_zone_multiplier(), or
    # get_all_zones() could ever see it. add_zone() also triggers an
    # immediate geometric membership sweep (Manifest §1), which is what
    # actually attaches acceleration_multiplier to the nodes the zone
    # covers.
    for zone in result.zones:
        graph.add_zone(zone)
    added["zones"] = len(result.zones)

    # Route enrichment_nodes through consolidator.
    # These are date_unknown observations that matched an existing graph node.
    # They must NOT be added as new graph nodes — only their field values are
    # merged into the matched node's record.
    if consolidator is not None and result.enrichment_nodes:
        trust = 0.6  # slightly lower trust for dateless enrichment observations
        for enr_node, matched_id in result.enrichment_nodes:
            try:
                consolidator.observe_node(
                    enr_node,
                    source=result.source,
                    timestamp=graph.get_node(matched_id).timestamp
                              if graph.get_node(matched_id) else 0.0,
                    trust=trust,
                    canonical_id=matched_id,
                )
                added["enrichment_routed"] += 1
            except Exception as e:
                logger.warning(
                    f"[load_result] enrichment routing failed for "
                    f"'{enr_node.id}' → '{matched_id}': {e}"
                )

    if added["synthesised_ancestry"]:
        logger.info(
            f"[Ancestry] {added['synthesised_ancestry']} lineage edge(s) "
            "synthesised during ingestion to satisfy ancestry invariant"
        )
    if added["enrichment_routed"]:
        logger.info(
            f"[Enrichment] {added['enrichment_routed']} dateless observation(s) "
            "routed to consolidator for existing nodes"
        )
    return added


def merge_nodes_in_graph(merge_pairs: List[Tuple[str, str]], graph) -> int:
    """
    Applies merge suggestions from CrossDocumentLinker.
    Keeps the node with the higher scientific_score and re-points all edges.
    """
    merged = 0
    for new_id, existing_id in merge_pairs:
        new_node = graph.get_node(new_id)
        existing = graph.get_node(existing_id)
        if not new_node or not existing:
            continue
        keeper_id  = (existing_id if existing.scientific_score >= new_node.scientific_score
                      else new_id)
        discard_id = new_id if keeper_id == existing_id else existing_id
        for edge in list(graph.edges.values()):
            if edge.source_id == discard_id:
                edge.source_id = keeper_id
            if edge.target_id == discard_id:
                edge.target_id = keeper_id
        keeper  = graph.get_node(keeper_id)
        discard = graph.get_node(discard_id)
        keeper.scientific_score = max(keeper.scientific_score, discard.scientific_score)
        keeper.investment_score = max(keeper.investment_score, discard.investment_score)
        keeper.strategic_value  = max(keeper.strategic_value,  discard.strategic_value)
        keeper.evidence = (getattr(keeper, "evidence", []) + getattr(discard, "evidence", []))
        graph.nodes.pop(discard_id, None)
        merged += 1
        logger.info(f"[Merge] {discard_id} -> {keeper_id}")
    return merged

__all__ = ['FuzzyMatcher', 'BucketIndex', 'CrossDocumentLinker', 'IngestionResult', 'IngestionPipeline', 'BatchDocument', 'BatchIngestionProcessor', 'load_result_into_graph', 'merge_nodes_in_graph']
