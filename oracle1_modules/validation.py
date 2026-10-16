"""Document chunking, prompts, annotation, and response validation."""
from __future__ import annotations

from ._shared import *
from .config import CONFIG, PHYSICAL_AXIS_ORDER
from .entities import KnowledgeEdge, KnowledgeNode, TemporalZone
from .edge_signals import derive_pair_metrics
from .json_utils import _parse_json_safe
from .label_status import LabelStatus, label_status_name, parse_label_status
from .llm import LocalLLMClient
from .time_utils import _year_to_ts

# ══════════════════════════════════════════════════════════════════════════════
#  INGESTION PIPELINE
#  Transforms raw documents into KnowledgeNode / KnowledgeEdge / TemporalZone objects.
#
#  Pipeline: DocumentSizer → IngestionPromptBuilder → LLMAnnotator →
#            ResponseValidator → NodeEdgeAssembler → CrossDocumentLinker →
#            IngestionPipeline → BatchIngestionProcessor
#
#  Fixes over MotherAnnotator:
#    1. Documents loaded whole    → context-window-sized budget, not a fixed
#                                    chunker (Manifest §2, "Никакого
#                                    DocumentChunker"); see
#                                    estimate_ingestion_char_budget() and
#                                    DocumentChunker below — splitting is now
#                                    a rare fallback, not the standard path.
#    2. One prompt for everything → 3 independent passes (entities / physical / edges)
#    3. No validation             → ResponseValidator
#    4. Unstable IDs              → deterministic MD5(source + entity_text)
#    5. Isolated documents        → CrossDocumentLinker via embeddings
#
#  Manifest §2 rationale for point 1: sending the whole document into a
#  single LLM pass lets the model see macro-connections between the first
#  and last paragraph of an article, and avoids re-extracting the same
#  entity once per fragment (which used to require CrossDocumentLinker-style
#  reconciliation *within a single document*, not just across documents).
#  Splitting is only correct once a document genuinely exceeds what the
#  active model can hold in context; everything below sizes against the
#  *actual* configured model instead of an arbitrary constant.
# ══════════════════════════════════════════════════════════════════════════════

INGESTION_ENTITY_TYPES: List[str] = [
    "physical_limit", "incumbent_tech", "challenger_tech", "enabler_component",
    "measurement_tool", "side_effect_risk", "regulatory_constraint",
    "cultural_phantom", "convergence_node", "breakthrough_event",
    "research_frontier", "economic_barrier",
]

INGESTION_SOURCE_TYPES: Dict[str, Dict[str, float]] = {
    k: {"trust": v, "citation_weight": {"arxiv": 0.9, "patent": 0.5,
        "nature": 1.0, "science": 1.0, "preprint": 0.6, "news": 0.1,
        "blog": 0.05, "report": 0.4, "book": 0.7, "forum": 0.1,
        "conference": 0.75, "unknown": 0.3}.get(k, 0.3)}
    for k, v in CONFIG["ingestion_source_trust"].items()
}


# ── Layer 1: Chunker ─────────────────────────────────────────────────────────

# Conservative characters-per-token ratio for mixed English/Russian technical
# prose (arXiv/patent/news text tends to run ~3.3-4.2 chars/token under
# typical BPE tokenizers; 3.5 undershoots slightly on purpose so the char
# budget stays inside the real token budget rather than risk overflow).
_CHARS_PER_TOKEN = 3.5
# Calibrated against English/Latin-script text (a common rule of thumb, not
# measured against this project's actual tokenizer). Most BPE tokenizers
# split Cyrillic less efficiently per character than Latin script, so for
# heavily Russian documents this ratio likely UNDERSTATES real token count,
# meaning the reservation below is less conservative than it looks for that
# case. Recalibrate against your actual model's tokenizer on a sample of
# real Russian documents before trusting this near the context ceiling —
# this project's own docs are explicitly bilingual (see profile), so this
# is not a hypothetical edge case.


def estimate_ingestion_char_budget(model_name: Optional[str] = None) -> int:
    """
    Manifest §2, "Никакого DocumentChunker": the character budget a document
    gets *before* it is considered for splitting, derived from the real
    context window of the model that will actually read it — not a fixed
    constant tuned for a small-context model.

    Reserves headroom for:
      - the prompt scaffolding itself — measured from the ACTUAL template
        text (IngestionPromptBuilder.UNIFIED_EXTRACTION_PROMPT or
        .ENTITY_EXTRACTION_PROMPT, whichever CONFIG["ingestion_unified_extraction"]
        will actually send), not a second hand-maintained guess. A guess is
        exactly how this reservation drifted out of sync the last time the
        prompt text changed: the unified prompt merges three passes' worth
        of rules into one template and runs measurably longer than any
        single old-mode pass's own template, and a flat reservation
        calibrated against the old, shorter template silently understated
        how much headroom the unified prompt actually needs.
      - the model's own completion — LLMAnnotator.MAX_TOKENS_PASS is the
        single source of truth for that number; reading it here rather
        than duplicating it is what keeps this in sync automatically the
        next time a pass's token budget changes.

    Unrecognised model names fall back to CONFIG["llm_context_window_default"]
    rather than a small value, because the manifest's premise is that
    Oracle-1 runs on a large-context model by design — an unknown model
    should not silently regress ingestion back to aggressive chunking.
    """
    windows = CONFIG.get("llm_context_windows", {})
    context_tokens = windows.get(model_name) if model_name else None
    if context_tokens is None:
        context_tokens = CONFIG.get("llm_context_window_default", 128_000)

    pass_type = "unified" if CONFIG.get("ingestion_unified_extraction", True) else "entities"
    completion_tokens = LLMAnnotator.MAX_TOKENS_PASS.get(pass_type, 4096)
    template = (IngestionPromptBuilder.UNIFIED_EXTRACTION_PROMPT if pass_type == "unified"
                else IngestionPromptBuilder.ENTITY_EXTRACTION_PROMPT)
    scaffolding_tokens = int(len(template) / _CHARS_PER_TOKEN)
    reserved_tokens = scaffolding_tokens + completion_tokens
    usable_tokens = max(context_tokens - reserved_tokens, 4_000)
    return int(usable_tokens * _CHARS_PER_TOKEN)


@dataclass
class DocumentChunk:
    chunk_id:     str
    source:       str
    text:         str
    chunk_index:  int
    total_chunks: int
    start_char:   int
    end_char:     int
    metadata:     Dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """
    Splits a document into overlapping chunks on paragraph boundaries — but
    only when it has to.

    Manifest §2: by default `chunk_size` is sized against the *active LLM's*
    real context window via estimate_ingestion_char_budget(), not against a
    small fixed constant. For the 128k+-token local models Oracle-1 is
    configured to use, that budget comfortably covers the vast majority of
    real ingested documents (articles, papers, patents), so
    DocumentChunker.chunk() returns a single whole-document chunk in the
    common case — the LLM sees the entire document in one pass, with no
    macro-connections lost across a chunk boundary and no risk of
    re-extracting the same entity twice per document. Splitting remains
    available as a fallback for the rare document that genuinely exceeds
    even that budget (e.g. a book-length source), so ingestion degrades
    gracefully instead of failing outright — but this is now the exception,
    not the pipeline's default behaviour, and is logged as such.
    """

    def __init__(self, chunk_size: int = None, overlap: int = None,
                 model_name: Optional[str] = None,
                 allow_fallback_split: bool = False):
        if chunk_size is not None:
            self.chunk_size = chunk_size
        elif model_name is not None:
            self.chunk_size = estimate_ingestion_char_budget(model_name)
        else:
            # No model context to size against — fall back to the
            # conservative floor rather than guess.
            self.chunk_size = CONFIG["ingestion_chunk_size"]
        self.overlap    = overlap    or CONFIG["ingestion_chunk_overlap"]
        self.model_name = model_name
        self.allow_fallback_split = allow_fallback_split

    @classmethod
    def for_model(cls, model_name: Optional[str], overlap: int = None) -> "DocumentChunker":
        """Construct a chunker sized to `model_name`'s real context window."""
        return cls(chunk_size=estimate_ingestion_char_budget(model_name),
                    overlap=overlap, model_name=model_name)

    def chunk(self, text: str, source: str) -> List[DocumentChunk]:
        if len(text) <= self.chunk_size:
            return [DocumentChunk(
                chunk_id=self._make_id(source, 0), source=source, text=text,
                chunk_index=0, total_chunks=1, start_char=0, end_char=len(text),
            )]
        # Oracle reasons over a document as one semantic object.  Fragmenting
        # it can erase a causal relation between its beginning and end, so the
        # default is deliberately to reject an over-context document instead
        # of silently changing its meaning.  A deployment may explicitly opt
        # into the old paragraph fallback for operational triage only.
        if not self.allow_fallback_split:
            raise ValueError(
                f"Document '{source}' is {len(text)} chars, exceeding the active "
                f"LLM whole-document budget ({self.chunk_size}). Choose a larger-context "
                "model or explicitly enable a lossy fallback split."
            )
        # Explicit, exceptional fallback path.
        logger.warning(
            "DocumentChunker: '%s' is %d chars, over the %d-char budget "
            "(%s) — falling back to paragraph-boundary splitting. This is "
            "the Manifest §2 fallback path, not the default: macro-links "
            "between the first and last paragraph may be lost for this "
            "document.", source, len(text), self.chunk_size,
            self.model_name or "unknown model",
        )
        paragraphs = re.split(r'\n{2,}', text)
        chunks: List[DocumentChunk] = []
        current_buf: List[str] = []
        current_len = 0
        char_pos    = 0
        chunk_start = 0
        for para in paragraphs:
            para_len = len(para) + 2
            if current_len + para_len > self.chunk_size and current_buf:
                chunk_text = "\n\n".join(current_buf)
                chunks.append(DocumentChunk(
                    chunk_id=self._make_id(source, len(chunks)), source=source,
                    text=chunk_text, chunk_index=len(chunks), total_chunks=-1,
                    start_char=chunk_start, end_char=chunk_start + len(chunk_text),
                ))
                overlap_buf: List[str] = []
                overlap_len = 0
                for p in reversed(current_buf):
                    if overlap_len + len(p) > self.overlap:
                        break
                    overlap_buf.insert(0, p)
                    overlap_len += len(p) + 2
                current_buf = overlap_buf
                current_len = overlap_len
                chunk_start = char_pos - overlap_len
            current_buf.append(para)
            current_len += para_len
            char_pos    += para_len
        if current_buf:
            chunk_text = "\n\n".join(current_buf)
            chunks.append(DocumentChunk(
                chunk_id=self._make_id(source, len(chunks)), source=source,
                text=chunk_text, chunk_index=len(chunks), total_chunks=-1,
                start_char=chunk_start, end_char=chunk_start + len(chunk_text),
            ))
        for c in chunks:
            c.total_chunks = len(chunks)
        return chunks

    @staticmethod
    def _make_id(source: str, idx: int) -> str:
        return hashlib.md5(f"{source}::{idx}".encode()).hexdigest()[:8]


# ── Layer 2: Prompt builder ──────────────────────────────────────────────────

class IngestionPromptBuilder:
    """
    Four independent passes per chunk:
        Pass A — entity extraction (nodes) + secondary context extraction
        Pass B — physical-axis scoring for each node
        Pass C — relationship extraction (edges between nodes in THIS document)
        Pass D — cross-graph matching: secondary contexts → existing graph nodes
    """

    ENTITY_EXTRACTION_PROMPT = """\
You are an expert knowledge graph constructor specialising in technology and science.

TASK: Extract all significant concepts, technologies, limits, barriers, and events
from the text below. Each extracted item becomes a NODE in a knowledge graph.

DOCUMENT: "{source}" (the complete document, not a manually classified sample)
{prior_nodes_block}
TEXT:
{text}

EXTRACTION RULES:
* First infer the document's source_type, title/authors when present, and
  publication/observation date from its own content. Never use filename or
  filesystem modification time as historical evidence. If a fact is absent,
  return null or "unknown" rather than inventing it.
* Infer each node's domain from evidence in the document. There is no caller
  supplied domain hint.
* Extract 5-25 nodes. Prefer specificity over generality.
* Create a node-level observation only for an object or result the article
  substantively analyzes. A concept appearing only in the title, keywords,
  introduction/background, or a summary of cited work is not a finding about
  that concept. If the text only states a relation between established
  concepts, record it as an edge and do not turn it into a new observation
  about either endpoint.
  A study protocol or planned method is not an observed result about the
  entities it proposes to use.
* Assign a stable slug ID: snake_case, max 40 chars (e.g. "crispr_cas9_editing").
* entity_type must be one of: {entity_types}
* origin_kind must be one of: "fundamental" | "combinatorial".
  - "fundamental": something nature already contains and humans only *discover* as
    measurement/theory improves — a physical constant, a material property, a
    biological mechanism, a law or limit. Not a matter of choice or engineering
    effort; its timing is not predictable from investment or demand.
  - "combinatorial": an engineered recombination of already-known fundamentals —
    almost all products, techniques, and applications. Its timing tends to track
    investment, convergence pressure, and adjacent-domain maturity.
  - Default to "combinatorial" when unsure; "fundamental" is the rarer case.
* Scores are 0.0-10.0 unless noted. evidence: 1-3 verbatim passages from
  the article body, max 35 words each. Each quote must contain the actual
  method, result, measured observation, or formula that supports this node.
  Do not quote the node label, a fragment of its name, the article title,
  a heading, a table/figure label, or an isolated keyword occurrence. A
  keyword being present is not evidence that the node is correct. If the
  body contains no substantive supporting passage, omit the node. Keep
  author/contact details, affiliations, issue/DOI data, and keywords out of
  evidence and full_context. Background statements about prior work do not
  count as this article's finding unless the article explicitly analyzes or
  synthesizes them as a result.
* Every node text must be a concise, source-specific claim, not a bare topic
  such as "clinical trial" or "in vitro model". For each claim, assign
  label_status using ONLY its status_evidence passage from this document.
  status_evidence must be one of the verbatim evidence quotes. Use exactly
  one value: "idea_concept", "theory", "fact", "proof", or "refutation".
  Idea/concept means a named or proposed possibility without a tested result.
  Theory means an explanatory model or hypothesis. Fact means a directly
  reported observation or measured result. Proof means the text supplies a
  formal derivation/proof or explicitly replicated/independently confirmed
  evidence for the claim. Refutation means the passage directly challenges
  or invalidates a specific claim. A measurement or formula is not proof by
  itself unless the passage uses it to establish the claim. Do not infer a
  status from popularity, source prestige, or general knowledge. If the
  passage does not support the claim and its status, omit the node.
* For a refutation label, return contradicted_claims containing only claims
  explicitly challenged by this source passage. Copy identifying text
  verbatim from this document; use [] if no target claim is named.
* full_context is a concise factual interpretation of the evidence, not a
  copy of the quote or bibliographic/page metadata. Add only facts supported
  by the evidence; leave it empty when no explanation is justified.

SECONDARY CONTEXT RULES:
* For each node, extract secondary_contexts: external entities or earlier
  claims explicitly named as influencing, enabling, constraining, supporting,
  confirming, proving, or contradicting this node.
* These are NOT the same as solves_limitations/requires_node_ids (which are
  internal to this document). secondary_contexts can point to an earlier
  claim or outside entity that may already exist in the graph.
* For each secondary context provide:
  - text: the entity name as mentioned in the source (raw, do not normalise)
  - role: how it relates ("influenced_by"|"enabled_by"|"constrained_by"|
          "co-evolved_with"|"funded_by"|"competed_with"|"derived_from"|
          "supports"|"confirms"|"proves"|"cites"|"uses_as_data_source"|
          "popularized_by"|"depicted_in")
  - domain: best-guess domain from the known domain list
  - entity_type: best-guess entity_type
  - year_hint: approximate year if mentioned, else null
  - aliases: alternative names/spellings you know for this entity (0-3)
  - evidence: one verbatim passage showing the relation or source influence

NON-CAUSAL SOURCE INFLUENCE:
* Also return influence_observations when this document cites a named
  concept, uses its published data as an input, reports investment in it,
  reports public/media/fictional attention around it, or documents an
  explicit unmet need or demand for it. This is a source observation about
  an existing concept, not a new scientific claim and not a causal edge.
  Include the exact passage as evidence. Do not return a numeric strength;
  the application accumulates grounded observations.
* influence_observations can be present even when no new scientific node is
  justified. Use only the fixed signal types in the JSON schema.

OUTPUT FORMAT (ONLY valid JSON, NO markdown fences):
{{
  "document_metadata": {{
    "title": "<title as stated, or null>",
    "authors": ["<author as stated>"],
    "publication": "<journal, conference, publisher, forum, or website if stated, else null>",
    "forum_topic": "<forum thread/topic if stated, else null>",
    "source_url": "<URL or source identifier explicitly present in TEXT, else null>",
    "source_type": "<arxiv|blog|book|conference|forum|nature|news|patent|preprint|report|science|unknown>",
    "publication_date": "<YYYY-MM-DD or YYYY, null if absent>",
    "date_evidence": "<short quote or null>"
  }},
  "domain": "<primary domain inferred from the document>",
  "influence_observations": [
    {{"target_text": "<named concept used, cited, funded, or discussed>",
      "target_domain": "<domain>", "target_entity_type": "<entity_type>",
      "year_hint": <int or null>, "aliases": ["<source-given alias>"],
      "signal_type": "<citation|data_use|investment|social_attention|fictional_hype|unmet_need>",
      "evidence": "<verbatim passage establishing this use or influence>",
      "parameter_observations": [
        {{"factor": "<limitation_resolution|citation_link|investment_correlation|social_correlation|inhibitory_force>",
          "value": <number copied from source>, "unit": "<percent|fraction|ratio|raw>",
          "denominator": <number copied from source for unit ratio, otherwise null>,
          "evidence": "<verbatim quote containing the number and its meaning>"}}
      ]}}
  ],
  "nodes": [
    {{
      "id": "<slug>",
      "text": "<concise functional description, 8-18 words>",
      "full_context": "<1-2 sentence explanation>",
      "entity_type": "<one of the allowed types>",
      "origin_kind": "<fundamental|combinatorial>",
      "domain": "<sub-domain>", "node_type": "research",
      "scientific_score": <0-10>, "investment_score": <0-10>,
      "social_score": <0-10>, "maturity_score": <0-10>,
      "strategic_value": <0-10>, "efficiency_plateau": <0-10>,
      "dual_use_risk": <0-10>, "legal_risk_score": <0-10>,
      "export_control_risk": <0-10>, "publication_year": <int or null>,
      "solves_limitations": ["<id>"], "requires_node_ids": ["<id>"],
      "evidence": ["<quote1>", "<quote2>"],
      "label_status": "<idea_concept|theory|fact|proof|refutation>",
      "status_evidence": "<one verbatim quote from evidence that determines label_status>",
      "contradicted_claims": [
        {{"text": "<verbatim claim or label the passage directly disputes>",
         "domain": "<domain>", "entity_type": "<entity_type>",
         "year_hint": <int or null>, "aliases": ["<source-given alias>"]}}
      ],
      "secondary_contexts": [
        {{
          "text": "<entity name as in source>",
          "role": "<influenced_by|enabled_by|constrained_by|co-evolved_with|funded_by|competed_with|derived_from|supports|confirms|proves|cites|uses_as_data_source|popularized_by|depicted_in>",
          "domain": "<domain>",
          "entity_type": "<entity_type>",
          "year_hint": <int or null>,
          "aliases": ["<alt name 1>", "<alt name 2>"],
          "evidence": "<verbatim quote supporting this relation or influence>"
        }}
      ]
    }}
  ]
}}
"""

    PHYSICAL_SUBSTRATE_PROMPT = """\
You are a physicist assessing technology constraints.

For each node, rate 10 constraint axes on 0.0-1.0.
HIGH means: resource_intensity=enormous resources; constraint_proximity=near limit (bad);
synthesis_complexity=hard to make; scale_feasibility=easy to scale; longevity=long life;
cascadability=enables breakthroughs; reversibility=easily reversible;
parallel_deployability=many instances; environmental_coupling=env-dependent;
cross_scale_stability=works at all scales.

CONTEXT: {context_excerpt}
NODES: {nodes_list}

OUTPUT (ONLY valid JSON):
{{
  "physical_scores": {{
    "<node_id>": {{
      "resource_intensity": <0-1>, "scale_feasibility": <0-1>,
      "constraint_proximity": <0-1>, "reversibility": <0-1>,
      "parallel_deployability": <0-1>, "environmental_coupling": <0-1>,
      "cross_scale_stability": <0-1>, "synthesis_complexity": <0-1>,
      "longevity": <0-1>, "cascadability": <0-1>,
      "rationale": "<one sentence>"
    }}
  }}
}}
"""

    EDGE_EXTRACTION_PROMPT = """\
You are a knowledge graph architect specialising in causal relationships.

TASK: Identify relationships between these nodes. Only assert edges explicitly supported.

EXTRACTED NODES:
{nodes_summary}

ORIGINAL TEXT:
{text}

Do not return numeric edge scores. The application derives pair similarity
and time proximity from endpoint data. Other numeric factors may only be
supported by parameter_observations that copy a number and its meaning from
the source. Never estimate a score from prose, relation type, popularity, or
the fact that a source mentions a concept. Use unit "percent" or "fraction"
only when that normalized measurement is explicit in the source. If TEXT
gives a numerator and denominator, return unit "ratio" with both quoted
numbers so the application can divide them deterministically. Use "raw"
for counts, money, or measurements without a source-defined scale. Raw
measurements are retained as evidence and are not converted to factor scores.
relationship_type: "enables"|"resolves_limit"|"competes_with"|"requires"|
                   "precedes"|"co-evolves_with"|"inhibits"|
                   "supports"|"confirms"|"proves"|"contradicts"|"related"
Bibliography references, data reuse, investment, and public or fictional
attention belong in influence_observations; they are not graph edges.
For "contradicts", source must be the claim/refutation supported by the
evidence passage, target must be the earlier or opposing claim, and the edge
must preserve both nodes. Do not infer contradiction from different values
unless they address the same conditions and cannot both be true.
Use "supports" when a quoted result adds evidence for the target claim,
"confirms" when the source reports an independent confirmation, and "proves"
only for a formal proof or explicit replicated demonstration. These edge
types require a verbatim source quote that identifies the target claim. They
can strengthen its later label status only from a source label marked fact or
proof; co-occurrence, citations, and popularity do not change status.

OUTPUT (ONLY valid JSON):
{{
  "edges": [{{
    "source": "<id>", "target": "<id>", "relationship_type": "<type>",
    "evidence": ["<quote stating the relationship itself>"],
    "parameter_observations": [
      {{"factor": "<limitation_resolution|citation_link|investment_correlation|social_correlation|inhibitory_force>",
        "value": <number copied from source>, "unit": "<percent|fraction|ratio|raw>",
        "denominator": <number copied from source for unit ratio, otherwise null>,
        "evidence": "<verbatim quote stating the measurement and what it measures>"}}
    ]
  }}],
  "temporal_zones": [{{
    "description": "<period>", "domain_focus": "<domain>",
    "start_year": <year the acceleration begins, e.g. 1939>,
    "end_year": <year it ends, or null if still ongoing/unbounded>,
    "zone_multiplier": <1-5>, "contained_node_ids": ["<id>"],
    "evidence": ["<quote>"]
  }}]
}}

A temporal zone is a region in (domain, time) — start_year/end_year must
bound the actual period of acceleration described by the evidence (e.g.
"Manhattan Project" -> 1942-1946), not left open-ended by default. Only
omit end_year (null) when the text genuinely gives no indication the
acceleration has ended (e.g. a boom still in progress at time of writing).
"""

    MERGE_PROMPT = """\
You are merging entity extractions from multiple chunks of the same document.
Identify which nodes refer to the same concept and should be merged.

CHUNK SUMMARIES:
{chunk_summaries}

For cross_chunk_edges, the evidence field must be a verbatim quote from the
source node's own evidence. Use relationship_type "contradicts" only when
the source node has label_status "refutation" and its quote directly
disputes the target claim under comparable conditions. Keep both nodes.

OUTPUT (ONLY valid JSON):
{{
  "merge_groups": [{{"canonical_id": "<id>", "merged_ids": ["<id1>"], "reason": "<why>"}}],
  "cross_chunk_edges": [{{
    "source": "<id>", "target": "<id>", "relationship_type": "<type>",
    "evidence": ["<verbatim quote from source node evidence>"]
  }}]
}}
"""

    UNIFIED_EXTRACTION_PROMPT = """\
You are analysing one document for a knowledge graph, in three sequential
stages within THIS single response. Complete each stage IN FULL for every
node before moving to the next stage — do not interleave them, and do not
start Stage 3 until every node from Stage 1 already has its Stage 2 data.

DOCUMENT: "{source}" (the complete document, not a manually classified sample)

TEXT:
{text}

════════════════════════════════════════════════════════════════════════
RULE THAT OVERRIDES EVERYTHING BELOW: use ONLY what this TEXT states or
directly, unambiguously implies. Never use outside knowledge, training
data, or plausible-sounding invention to fill a field — not a fact, not a
name, not a number, not an example. If TEXT does not support a field,
output null, "unknown", or an empty list for it. Every "evidence" entry
must be copied character-for-character from TEXT above — it will be
checked programmatically against the source, and a node whose evidence
cannot be found verbatim in TEXT will be discarded entirely, so a
fabricated quote costs you the whole node, not just that field.
════════════════════════════════════════════════════════════════════════
{prior_nodes_block}
═══ STAGE 1 — EXTRACTION (act as an expert knowledge graph constructor) ═══
Extract all significant concepts, technologies, limits, barriers, and
events from TEXT. Each extracted item becomes a NODE.
* First infer the document's source_type, title/authors when present, and
  publication/observation date from TEXT itself. Never use filename or
  filesystem modification time as historical evidence. If a fact is
  absent, return null or "unknown" rather than inventing it.
* Infer each node's domain from evidence in TEXT. There is no caller
  supplied domain hint.
* Extract 5-25 nodes actually discussed in TEXT. Prefer specificity over
  generality. Do not add a well-known node just because the topic usually
  calls for one — only nodes TEXT actually discusses.
* Create a node-level observation only for an object or result the article
  substantively analyzes. A concept appearing only in the title, keywords,
  introduction/background, or a summary of cited work is not a finding about
  that concept. If the text only states a relation between established
  concepts, record it as an edge and do not turn it into a new observation
  about either endpoint.
  A study protocol or planned method is not an observed result about the
  entities it proposes to use.
* Assign a stable slug ID: snake_case, max 40 chars (e.g.
  "crispr_cas9_editing"). IDs must be unique within THIS response and are
  reused verbatim by solves_limitations, requires_node_ids, and the edges
  in Stage 3 below — character-for-character, never re-derived or
  reworded later.
* entity_type must be one of: {entity_types}
* origin_kind must be one of: "fundamental" | "combinatorial".
  - "fundamental": something nature already contains and humans only
    *discover* as measurement/theory improves — a physical constant, a
    material property, a biological mechanism, a law or limit. Not a
    matter of choice or engineering effort; its timing is not predictable
    from investment or demand.
  - "combinatorial": an engineered recombination of already-known
    fundamentals — almost all products, techniques, and applications. Its
    timing tends to track investment, convergence pressure, and adjacent-
    domain maturity.
  - Default to "combinatorial" when unsure; "fundamental" is the rarer case.
* Scores are 0.0-10.0 unless noted. For every node, evidence must be a
  verbatim passage that substantively supports the node's text/full_context:
  a reported result, experiment or measurement, method, formula, or an
  explicit claim made by the source. Do NOT use only the node name, a title,
  a keyword occurrence, a background/literature-review mention, or a passage
  that merely mentions the topic. Use 1-3 passages, up to 35 words each,
  copied exactly from TEXT. If TEXT only names the topic without a finding
  about it, do not emit that node. Do not use author/contact details,
  affiliations, issue/DOI data, or keywords as evidence or full_context.
* full_context is a concise factual interpretation of the evidence, not a
  copied quote or source metadata. It may contain only facts supported by the
  evidence; leave it empty when no such explanation is justified.
* Every node text must be a concise, source-specific claim, not a bare topic.
  Assign label_status using only its status_evidence quote. Use exactly one
  value: "idea_concept", "theory", "fact", "proof", or "refutation".
  Idea/concept is a named or proposed possibility without a tested result;
  theory is an explanatory model or hypothesis; fact is a directly reported
  observation or measured result; proof requires a formal derivation/proof or
  explicit replication/independent confirmation; refutation directly
  challenges a specific claim. A measurement or formula is not proof by itself.
  Never infer status from popularity, source prestige, or general knowledge.
  Omit a node when its quote does not support both its claim and status.
* For a refutation label, return contradicted_claims containing only claims
  explicitly challenged by this source passage. Copy identifying text
  verbatim from TEXT; use [] if no target claim is named.
* secondary_contexts: external entities or earlier claims explicitly named
  as influencing, enabling, constraining, supporting, confirming, proving,
  or contradicting this node, ONLY if TEXT actually names them. NOT the same
  as solves_limitations/requires_node_ids (which
  reference OTHER nodes in THIS SAME response's "nodes" list, not external
  entities). secondary_contexts may already exist in the knowledge graph
  under different names or languages — a later, separate step resolves
  that; here you only report what TEXT names. For each provide:
  - text: the entity name exactly as it appears in TEXT (raw, do not
    normalise or translate — this will also be checked against TEXT)
  - role: how it relates ("influenced_by"|"enabled_by"|"constrained_by"|
          "co-evolved_with"|"funded_by"|"competed_with"|"derived_from"|
          "supports"|"confirms"|"proves")
  - domain: best-guess domain from the known domain list
  - entity_type: best-guess entity_type
  - year_hint: approximate year if TEXT mentions one, else null
  - aliases: alternative names/spellings for this SAME entity if TEXT
    itself gives more than one form (0-3); do not supply a translation or
    alias from outside knowledge that TEXT does not itself contain
  - evidence: a verbatim quote from TEXT that supports this role
* influence_observations: also report a named concept when this document
  cites it, uses its published data as an input, reports investment in it,
  describes public/media/fictional attention around it, or explicitly
  describes an unmet need or demand for it. This records a source influence
  only; it does not create a scientific node or causal edge. Include a
  verbatim evidence passage. Do not provide a numeric score.
  Emit this list even when the document adds no new scientific finding.
* Edge parameter_observations are measurements, not judgments. Return one
  only when TEXT explicitly gives the number for the named factor. Copy the
  number and unit exactly. Percentages and source-defined fractions can be
  normalized by the application. If TEXT gives numerator and denominator,
  return unit "ratio" and copy both numbers; raw counts and money remain raw
  evidence unless TEXT also supplies a denominator or scale.

═══ STAGE 2 — PHYSICAL CONSTRAINTS (act as a physicist assessing technology constraints) ═══
For EACH node from Stage 1, in the SAME node object, add a "physical_axes"
object rating 10 constraint axes on 0.0-1.0, based only on what TEXT
supports — a genuinely unaddressed axis should be 0.5 (uninformative
midpoint), not a guess dressed up as a measurement.
HIGH means: resource_intensity=enormous resources; constraint_proximity=
near limit (bad); synthesis_complexity=hard to make; scale_feasibility=
easy to scale; longevity=long life; cascadability=enables breakthroughs;
reversibility=easily reversible; parallel_deployability=many instances;
environmental_coupling=env-dependent; cross_scale_stability=works at all
scales.

═══ STAGE 3 — RELATIONSHIPS (act as a knowledge graph architect specialising in causal relationships) ═══
Using ONLY the node ids assigned in Stage 1 (never a new id, never an id
from outside knowledge), identify relationships between them. Only assert
an edge TEXT explicitly supports. Mere co-occurrence, a shared keyword,
proximity in a sentence, or a bibliography/reference citation is not an
edge. Cite a passage that states the relationship itself. If the passage
only relates two established concepts, keep the information on the edge and
do not turn it into a node observation.
Do not return numeric edge scores. Pair similarity and time proximity are
derived from endpoint data. Other numeric factors require an explicit
parameter_observation with a quoted number and its meaning; qualitative
mentions are retained as source observations but do not become scores.
relationship_type: "enables"|"resolves_limit"|"competes_with"|"requires"|
                   "precedes"|"co-evolves_with"|"inhibits"|
                   "supports"|"confirms"|"proves"|"contradicts"|"related"
For "contradicts", source must be the claim/refutation supported by the
evidence passage, target must be the earlier or opposing claim, and the edge
must preserve both nodes. Do not infer contradiction from different values
unless they address the same conditions and cannot both be true.
Also identify temporal_zones — a region in (domain, time) where TEXT
describes activity clearly accelerating. start_year/end_year must bound
the actual period of acceleration described by the evidence (e.g.
"Manhattan Project" -> 1942-1946), not left open-ended by default. Only
omit end_year (null) when TEXT genuinely gives no indication the
acceleration has ended.

═══ OUTPUT FORMAT RULES ═══
* Output ONLY valid JSON below. No markdown fences, no commentary before
  or after, no text outside the JSON object.
* Emit "nodes" completely BEFORE "edges" and "temporal_zones" — if you run
  low on space, a complete "nodes" array with empty "edges"/
  "temporal_zones" is far more useful than a truncated node, because a
  node cut off mid-object is discarded entirely on this end.
* Close every node object fully (every field below present, including
  "physical_axes") before starting the next node — never leave one
  half-written to save space; a half-written node is discarded the same
  way a truncated one is.
* Do not explain or justify your answers outside the specific fields
  provided for that purpose (evidence, rationale) — no preamble, no
  summary, no meta-commentary about the task.

{{
  "document_metadata": {{
    "title": "<title as stated, or null>",
    "authors": ["<author as stated>"],
    "publication": "<journal, conference, publisher, forum, or website if stated, else null>",
    "forum_topic": "<forum thread/topic if stated, else null>",
    "source_url": "<URL or source identifier explicitly present in TEXT, else null>",
    "source_type": "<arxiv|blog|book|conference|forum|nature|news|patent|preprint|report|science|unknown>",
    "publication_date": "<YYYY-MM-DD or YYYY, null if absent>",
    "date_evidence": "<short quote or null>"
  }},
  "domain": "<primary domain inferred from TEXT>",
  "influence_observations": [
    {{"target_text": "<named concept used, cited, funded, or discussed>",
      "target_domain": "<domain>", "target_entity_type": "<entity_type>",
      "year_hint": <int or null>, "aliases": ["<source-given alias>"],
      "signal_type": "<citation|data_use|investment|social_attention|fictional_hype|unmet_need>",
      "evidence": "<verbatim passage establishing this use or influence>",
      "parameter_observations": [
        {{"factor": "<limitation_resolution|citation_link|investment_correlation|social_correlation|inhibitory_force>",
          "value": <number copied from source>, "unit": "<percent|fraction|ratio|raw>",
          "denominator": <number copied from source for unit ratio, otherwise null>,
          "evidence": "<verbatim quote containing the number and its meaning>"}}
      ]}}
  ],
  "nodes": [
    {{
      "id": "<slug>",
      "text": "<concise functional description, 8-18 words>",
      "full_context": "<1-2 sentence explanation>",
      "entity_type": "<one of the allowed types>",
      "origin_kind": "<fundamental|combinatorial>",
      "domain": "<sub-domain>", "node_type": "research",
      "scientific_score": <0-10>, "investment_score": <0-10>,
      "social_score": <0-10>, "maturity_score": <0-10>,
      "strategic_value": <0-10>, "efficiency_plateau": <0-10>,
      "dual_use_risk": <0-10>, "legal_risk_score": <0-10>,
      "export_control_risk": <0-10>, "publication_year": <int or null>,
      "solves_limitations": ["<id>"], "requires_node_ids": ["<id>"],
      "evidence": ["<verbatim quote 1>", "<verbatim quote 2>"],
      "label_status": "<idea_concept|theory|fact|proof|refutation>",
      "status_evidence": "<one verbatim quote from evidence that determines label_status>",
      "contradicted_claims": [
        {{"text": "<verbatim claim or label the passage directly disputes>",
         "domain": "<domain>", "entity_type": "<entity_type>",
         "year_hint": <int or null>, "aliases": ["<source-given alias>"]}}
      ],
      "physical_axes": {{
        "resource_intensity": <0-1>, "scale_feasibility": <0-1>,
        "constraint_proximity": <0-1>, "reversibility": <0-1>,
        "parallel_deployability": <0-1>, "environmental_coupling": <0-1>,
        "cross_scale_stability": <0-1>, "synthesis_complexity": <0-1>,
        "longevity": <0-1>, "cascadability": <0-1>,
        "rationale": "<one sentence>"
      }},
      "secondary_contexts": [
        {{
          "text": "<entity name exactly as in TEXT>",
          "role": "<influenced_by|enabled_by|constrained_by|co-evolved_with|funded_by|competed_with|derived_from|supports|confirms|proves|cites|uses_as_data_source|popularized_by|depicted_in>",
          "domain": "<domain>",
          "entity_type": "<entity_type>",
          "year_hint": <int or null>,
          "aliases": ["<alt name 1>", "<alt name 2>"],
          "evidence": "<verbatim quote supporting this relation or influence>"
        }}
      ]
    }}
  ],
  "edges": [{{
    "source": "<id from Stage 1>", "target": "<id from Stage 1>",
    "relationship_type": "<type>",
    "evidence": ["<verbatim quote>"],
    "parameter_observations": [
        {{"factor": "<limitation_resolution|citation_link|investment_correlation|social_correlation|inhibitory_force>",
        "value": <number copied from source>, "unit": "<percent|fraction|ratio|raw>",
        "denominator": <number copied from source for unit ratio, otherwise null>,
        "evidence": "<verbatim quote stating the measurement and what it measures>"}}
    ]
  }}],
  "temporal_zones": [{{
    "description": "<period>", "domain_focus": "<domain>",
    "start_year": <year the acceleration begins, e.g. 1939>,
    "end_year": <year it ends, or null if still ongoing/unbounded>,
    "zone_multiplier": <1-5>, "contained_node_ids": ["<id from Stage 1>"],
    "evidence": ["<verbatim quote>"]
  }}]
}}
"""

    PASS_D_BATCH_PROMPT = """\
You are a knowledge graph linker. Each ITEM below is either a new node's
reference to an external entity or a source passage's explicit reference to
a claim it disputes. The role field distinguishes these cases.

Process every item INDEPENDENTLY — nothing in one item's decision should
be influenced by another item's node, context, or candidates, even when
several items in this batch share the same new node or the same
candidate pool. Do not let items "borrow" a match from each other.

ITEMS:
{items_block}

For EACH item:
1. For a normal secondary context, decide whether a candidate is the same
   real-world entity (consider language, transliteration, abbreviation, or
   spelling variants). For role="contradicts", choose only a candidate
   representing the explicitly disputed claim; do not substitute a merely
   related concept. Only choose a candidate id listed for THAT item.
2. If role starts with "signal:", this is a source-influence observation:
   find the matching existing concept, but do not create an edge. Return
   signal_type copied from the item and an evidence quote copied exactly
   from SOURCE EVIDENCE.
3. For a non-signal item, if a match is found, output its graph_node_id and
   relationship_type. Do not estimate numeric scores. Return a
   parameter_observation only when SOURCE EVIDENCE states a numeric
   measurement for that factor; quote the number and its meaning exactly.
   Percentages, explicitly normalized fractions, and ratios with a quoted
   denominator can be normalized by the application. Counts and money remain
   raw unless a denominator or scale is given in SOURCE EVIDENCE.
4. If no candidate matches, output "match": null.
5. For non-signal items, assess whether the RESULTING connection (if any)
   would be semantically plausible and factually defensible — apply the
   same drop rules a human curator would:
   - fictional <-> real boundary crossing asserted as direct causal influence
   - chronological impossibility (the secondary context's year_hint is
     after the candidate's, with a relationship implying the earlier
     caused or enabled the later)
   - epistemic domain mismatch (fan speculation or informal commentary
     asserted as a causal link into peer-reviewed work, rather than merely
     "commentary on" it)
   Set "verdict" to "keep" or "drop" accordingly, "plausibility" 0.0-1.0,
   and a one-sentence "rationale" (max 25 words, your own reasoning — this
   one field is not required to be a verbatim quote).
6. For role="supports", "confirms", or "proves", select that same
   relationship_type only when SOURCE EVIDENCE directly supports the candidate
   claim, and include an exact quote. A fact/proof source is required for the
   target's evidence status to be strengthened.
7. Use relationship_type "contradicts" only when the NEW NODE is explicitly
   classified as "refutation" and its source evidence directly disputes the
   candidate claim under comparable conditions. The edge points from the new
   refutation node to the candidate claim. Return an evidence quote copied
   exactly from SOURCE EVIDENCE; never delete or merge either node.

OUTPUT (ONLY valid JSON, NO markdown fences, exactly one result per
item_id listed above, any order, item_id copied back exactly as given):
{{
  "results": [
    {{
      "item_id": "<copied exactly from the item above>",
      "match": "<graph_node_id from that item's own candidate list, or null>",
      "signal_type": "<copied for signal items, otherwise null>",
      "confidence": <0.0-1.0>,
      "relationship_type": "<enables|resolves_limit|competes_with|requires|precedes|cites|co-evolves_with|inhibits|supports|confirms|proves|contradicts|influenced_by|funded_by|derived_from|related>",
      "plausibility": <0.0-1.0>,
      "verdict": "<keep|drop>",
      "rationale": "<one sentence, max 25 words>",
      "evidence": ["<verbatim quote from SOURCE EVIDENCE>"],
      "parameter_observations": [
        {{"factor": "<limitation_resolution|citation_link|investment_correlation|social_correlation|inhibitory_force>",
          "value": <number copied from SOURCE EVIDENCE>, "unit": "<percent|fraction|ratio|raw>",
          "denominator": <number copied from SOURCE EVIDENCE for ratio, otherwise null>,
          "evidence": "<verbatim quote with the number and its meaning>"}}
      ]
    }}
  ]
}}
"""

    @staticmethod
    def _render_prior_nodes_block(prior_nodes: Optional[List[Dict]]) -> str:
        """Renders the "already extracted from earlier in this document"
        registry inserted into a later chunk's prompt when a document had
        to be split — DocumentChunker's explicit-opt-in fallback path (see
        its docstring); the default single-chunk case never populates this.

        This is the practical, backend-independent analogue to KV-cache
        reuse across chunks of the same document: literal token-level KV
        reuse is a property of the INFERENCE SERVER (vLLM/SGLang/llama.cpp
        with prefix caching enabled), not something client code can force,
        and re-sending each earlier chunk's full text just to get that
        reuse would only pay off if the backend actually has it turned on.
        This instead gives the model a small, bounded, ALWAYS-useful
        registry of prior node ids+text (capped by
        CONFIG["ingestion_carryforward_max_nodes"]), so requires_node_ids/
        solves_limitations/edges can correctly reference something
        extracted a chunk ago, and the model does not re-extract it as if
        it were new — the actual correctness problem multi-chunk
        extraction has, independent of whether any particular backend
        happens to cache prefixes.
        """
        if not prior_nodes:
            return ""
        lines = "\n".join(f'  - id="{n["id"]}": {n["text"]}' for n in prior_nodes)
        return (
            "\nALREADY-EXTRACTED NODES FROM EARLIER IN THIS DOCUMENT (a "
            "different chunk of the same document — you have NOT been "
            "shown their surrounding text again here):\n"
            f"{lines}\n"
            "You may reference these EXACT ids in solves_limitations/"
            "requires_node_ids/edges if THIS chunk's text builds on one "
            "of them. Do NOT re-extract any of them as a new node here, "
            "and do NOT invent a different id for the same concept if it "
            "is already listed above.\n"
        )

    def build_unified_prompt(self, chunk: "DocumentChunk",
                              prior_nodes: Optional[List[Dict]] = None) -> str:
        return self.UNIFIED_EXTRACTION_PROMPT.format(
            source=chunk.source, text=chunk.text,
            entity_types=", ".join(INGESTION_ENTITY_TYPES),
            prior_nodes_block=self._render_prior_nodes_block(prior_nodes),
        )

    def _pass_d_item_block(self, item_id: str, node: Dict, ctx: Dict,
                            candidates: List[Dict]) -> str:
        if candidates:
            cand_lines = "\n".join(
                f'    [{i+1}] id={c["id"]} text="{c["text"]}" '
                f'domain={c.get("domain","?")} '
                f'entity_type={c.get("entity_type","?")} '
                f'sim={c.get("_sim", 0.0):.3f}'
                for i, c in enumerate(candidates)
            )
        else:
            cand_lines = "    (no candidates found in bucket)"
        aliases = ", ".join(ctx.get("aliases", [])) or "none"
        signal_evidence = str(ctx.get("evidence", ""))
        signal_block = (
            f'  SIGNAL TYPE: {str(ctx.get("role", "")).removeprefix("signal:")}\n'
            f'  SOURCE EVIDENCE: "{signal_evidence}"\n'
            "  This updates an existing concept's accumulated signal only; create no edge.\n"
            if str(ctx.get("role", "")).startswith("signal:") else ""
        )
        return (
            f'ITEM item_id="{item_id}"\n'
            f'  NEW NODE: id={node.get("id","")} text="{node.get("text","")}" '
            f'domain={node.get("domain","")} entity_type={node.get("entity_type","")}\n'
            f'  LABEL STATUS: {node.get("label_status", "idea_concept")}\n'
            f'  SOURCE EVIDENCE: {json.dumps(node.get("evidence", []), ensure_ascii=False)}\n'
            f'  SECONDARY CONTEXT: raw text="{ctx.get("text","")}" '
            f'role={ctx.get("role","related")} domain={ctx.get("domain","")} '
            f'entity_type={ctx.get("entity_type","")} '
            f'year_hint={ctx.get("year_hint","unknown")} aliases={aliases}\n'
            f'{signal_block}'
            f'  GRAPH CANDIDATES (ranked by embedding similarity):\n{cand_lines}'
        )

    def build_pass_d_batch_prompt(self, items: List[Dict]) -> str:
        """items: [{"item_id":..., "node":raw_node_dict, "ctx":ctx_dict,
        "candidates":[...]}]"""
        blocks = [
            self._pass_d_item_block(it["item_id"], it["node"], it["ctx"], it["candidates"])
            for it in items
        ]
        return self.PASS_D_BATCH_PROMPT.format(items_block="\n\n".join(blocks))

    def build_entity_prompt(self, chunk: "DocumentChunk",
                             prior_nodes: Optional[List[Dict]] = None) -> str:
        return self.ENTITY_EXTRACTION_PROMPT.format(
            source=chunk.source, text=chunk.text,
            entity_types=", ".join(INGESTION_ENTITY_TYPES),
            prior_nodes_block=self._render_prior_nodes_block(prior_nodes),
        )

    def build_physical_prompt(self, nodes: List[Dict], context_excerpt: str) -> str:
        nodes_list = "\n".join(
            f'  {n["id"]}: {n["text"]} [{n.get("entity_type", "")}]' for n in nodes)
        return self.PHYSICAL_SUBSTRATE_PROMPT.format(
            context_excerpt=context_excerpt, nodes_list=nodes_list)

    def build_edge_prompt(self, nodes: List[Dict], text: str) -> str:
        nodes_summary = "\n".join(f'  {n["id"]}: {n["text"]}' for n in nodes)
        return self.EDGE_EXTRACTION_PROMPT.format(
            nodes_summary=nodes_summary, text=text)

    def build_merge_prompt(self, chunk_summaries: List[Dict]) -> str:
        return self.MERGE_PROMPT.format(
            chunk_summaries=json.dumps(chunk_summaries, ensure_ascii=False, indent=2))


# ── Layer 3: LLM Annotator ────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    raw:       str
    parsed:    Optional[Dict]
    success:   bool
    error:     Optional[str] = None
    attempts:  int           = 1
    latency_s: float         = 0.0


class LLMAnnotator:
    """
    Wrapper around llm_client with retry logic, JSON cleanup.
    """
    MAX_RETRIES     = CONFIG["llm_annotator_max_retries"]
    RETRY_DELAY_S   = CONFIG["llm_annotator_retry_delay_s"]
    MAX_TOKENS_PASS = {"entities": 4096, "physical": 2048, "edges": 3072,
                        "merge": 2048, "unified": 8192, "pass_d": 4096}

    def __init__(self, llm_client, model: Optional[str] = None):
        # choose the model from argument or fall back to config default
        self.llm   = llm_client
        self.model = model or CONFIG.get("llm_default_model") or CONFIG["llm_local_models"][0]
        self.calls_made = 0
        self.failed_calls = 0
        self.incomplete_calls = 0
        retries_env = os.environ.get("ORACLE_LLM_MAX_RETRIES", "").strip()
        try:
            self.max_retries = max(0, int(retries_env)) if retries_env else self.MAX_RETRIES
        except ValueError as exc:
            raise ValueError("ORACLE_LLM_MAX_RETRIES must be an integer") from exc

    def call(self, prompt: str, pass_type: str = "entities") -> LLMResponse:
        self.calls_made += 1
        max_tokens = self.MAX_TOKENS_PASS.get(pass_type, 3000)
        last_raw   = ""
        last_error = "All retry attempts failed"
        active_prompt = prompt
        for attempt in range(1, self.max_retries + 2):
            t0 = time.time()
            try:
                raw      = self._call_llm(active_prompt, max_tokens)
                last_raw = raw  # kept even if the parse below fails, so a
                                 # caller can still attempt partial-array
                                 # recovery on the final failed attempt —
                                 # previously this was discarded entirely
                                 # and _recover_partial_array would always
                                 # see an empty string on total failure.
                parsed = self._parse_json(raw)
                return LLMResponse(raw=raw, parsed=parsed, success=True,
                                   attempts=attempt, latency_s=time.time() - t0)
            except json.JSONDecodeError as e:
                logger.warning(f"[LLMAnnotator] JSON parse failed (attempt {attempt}): {e}")
                last_error = str(e)
                if attempt <= self.max_retries:
                    active_prompt = (
                        f"{prompt}\n\nFORMAT REPAIR: Your previous response was not valid JSON. "
                        "Repeat the same task and return exactly one complete JSON object. "
                        "Do not include acknowledgements, explanations, Markdown fences, or "
                        "text before or after the JSON. Ensure every key/value and array item "
                        "is separated by a comma and all strings and containers are closed. "
                        f"Parser error: {e}"
                    )
                    time.sleep(self.RETRY_DELAY_S * attempt)
            except Exception as e:
                logger.error(f"[LLMAnnotator] LLM call failed (attempt {attempt}): {e}")
                last_error = str(e)
                if attempt <= self.max_retries:
                    time.sleep(self.RETRY_DELAY_S * attempt)
        # last_raw stays "" only when _call_llm itself never returned text on
        # any attempt (a transport-level failure) — nothing to recover then
        # regardless of what a caller tries.
        self.failed_calls += 1
        return LLMResponse(raw=last_raw, parsed=None, success=False,
                           error=last_error, attempts=self.max_retries + 1)

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """Call LLM with support for Anthropic SDK, OpenAI SDK, and local models."""
        # Local models (LocalLLMClient, etc.) — simple interface
        if isinstance(self.llm, LocalLLMClient) or not hasattr(self.llm, "messages") and not hasattr(self.llm, "chat"):
            return self.llm.complete(prompt)
        
        # Anthropic SDK
        if hasattr(self.llm, "messages"):
            resp = self.llm.messages.create(
                model=self.model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            return resp.content[0].text
        
        # OpenAI SDK
        if hasattr(self.llm, "chat"):
            resp = self.llm.chat.completions.create(
                model=self.model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content
        
        # Fallback
        return self.llm.complete(prompt)

    @staticmethod
    def _parse_json(raw: str) -> Dict:
        return _parse_json_safe(raw)


# ── Layer 4: Validator ────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid:    bool
    warnings: List[str] = field(default_factory=list)
    errors:   List[str] = field(default_factory=list)
    fixed:    Dict      = field(default_factory=dict)


class ResponseValidator:
    """
    Validates and fixes LLM output:
      - numeric field ranges (0-10 and 0-1)
      - edge references to existing nodes
      - duplicate IDs and required fields
    """
    SCORE_FIELDS_10 = [
        "scientific_score", "investment_score", "social_score",
        "maturity_score", "strategic_value", "efficiency_plateau",
        "dual_use_risk", "legal_risk_score", "export_control_risk",
    ]
    SCORE_FIELDS_01 = [
        "semantic_similarity", "temporal_proximity", "limitation_resolution",
        "citation_link", "investment_correlation", "social_correlation",
        "inhibitory_force", "confidence",
    ] + PHYSICAL_AXIS_ORDER

    EVIDENCE_STOP_WORDS = frozenset({
        "a", "an", "the", "of", "in", "on", "at", "by", "to", "for", "from",
        "with", "without", "as", "and", "or", "but", "that", "which", "who",
        "this", "these", "those", "than", "then", "it", "they", "their", "we",
        "our", "was", "were", "is", "are", "be", "been", "being", "into",
        "between", "after", "before", "during", "via", "per", "using", "used",
        "use", "also", "within", "over", "under", "such", "study", "studies",
        "данный", "данные", "это", "этот", "эта", "эти", "и", "в", "во", "на",
        "к", "ко", "с", "со", "по", "из", "для", "от", "до", "при", "как",
        "что", "который", "которая", "которые", "был", "была", "были", "быть",
        "является", "являются",
    })

    @staticmethod
    def _quote_is_grounded(quote: str, source_text: str) -> bool:
        """Normalised substring check: does this quote actually occur in
        the source? Collapses whitespace and case so a reflowed line break
        or a smart-quote the LLM normalised doesn't register as a false
        hallucination flag — the bar is "this wording is really in the
        text", not byte-exact reproduction. Not used to detect plagiarism
        going the other way; used only to catch the LLM inventing a claim
        that TEXT never made.
        """
        if not quote or not source_text:
            return False
        def normalize(value: str) -> str:
            value = html_lib.unescape(value)
            value = re.sub(r"(?<=\w)-\s+(?=[a-z])", "", value)
            return re.sub(r"\s+", " ", value).strip().casefold()
        norm_quote = normalize(quote)
        norm_source = normalize(source_text)
        return bool(norm_quote) and norm_quote in norm_source

    @staticmethod
    def _clean_evidence_quote(quote: Any) -> str:
        """Decode presentation artifacts without inventing source wording."""
        value = html_lib.unescape(str(quote or ""))
        # Rejoin words split by a line-end hyphen in PDF text extraction.
        value = re.sub(r"(?<=\w)-\s+(?=[a-z])", "", value)
        value = re.sub(r"\*\*(.+?)\*\*", r"\1", value)
        value = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", value)
        # An unmatched pair is a common truncated Markdown artifact (e.g. a
        # model response cut off after **in vitro); remove only the pair.
        value = value.replace("**", "")
        value = re.sub(r"`+([^`]+)`+", r"\1", value)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _evidence_tokens(value: str) -> List[str]:
        return re.findall(r"[^\W_]+(?:['’][^\W_]+)*", value.casefold(), flags=re.UNICODE)

    @staticmethod
    def _is_substantive_node_label(value: str) -> bool:
        """Require a claim-like label instead of a bare source topic."""
        text = str(value or "").strip()
        if re.search(r"(?:=|≤|≥|±|×|÷|≠|≈|→|⇌)", text):
            return True
        tokens = [token for token in ResponseValidator._evidence_tokens(text)
                  if token not in ResponseValidator.EVIDENCE_STOP_WORDS]
        if len(tokens) < 4:
            return False
        return bool(re.search(
            r"(?i)\b(?:show(?:s|ed|n|ing)?|demonstrat(?:e|es|ed|ing)|"
            r"reveal(?:s|ed|ing)?|indicat(?:e|es|ed|ing)|find(?:s|ings|ing)?|"
            r"observ(?:e|es|ed|ing)|measur(?:e|es|ed|ing)|evaluat(?:e|es|ed|ing)|"
            r"analyz(?:e|es|ed|ing)|analys(?:e|es|ed|ing)|express(?:es|ed|ing)|"
            r"increas(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|reduc(?:e|es|ed|ing)|"
            r"improv(?:e|es|ed|ing)|induc(?:e|es|ed|ing)|inhibit(?:s|ed|ing)?|"
            r"prevent(?:s|ed|ing)?|caus(?:e|es|ed|ing)|associat(?:e|es|ed|ing)|"
            r"correlat(?:e|es|ed|ing)|support(?:s|ed|ing)?|confirm(?:s|ed|ing)?|"
            r"replicat(?:e|es|ed|ing)|refut(?:e|es|ed|ing)|contradict(?:s|ed|ing)?|"
            r"propos(?:e|es|ed|ing)|hypothesiz(?:e|es|ed|ing)|predict(?:s|ed|ing)?|"
            r"explain(?:s|ed|ing)?|defin(?:e|es|ed|ing)|contain(?:s|ed|ing)?|"
            r"compris(?:e|es|ed|ing)|consist(?:s|ed|ing)?|remain(?:s|ed|ing)?|"
            r"in vitro .*\b(?:increase|decrease|expression|viability|formation)\b|"
            r"показал(?:и|о|а)?|демонстрир(?:ует|уют|овал|овали)|"
            r"выявил(?:и|о|а)?|наблюдал(?:и|о|а)?|измерил(?:и|о|а)?|"
            r"увеличил(?:и|о|а)?|снизил(?:и|о|а)?|сократил(?:и|о|а)?|"
            r"улучшил(?:и|о|а)?|индуцировал(?:и|о|а)?|подтвердил(?:и|о|а)?|"
            r"опроверг(?:ли|нуто|ла)?|предложил(?:и|о|а)?|предсказывает|"
            r"объясняет|определяет|содержит|состоит|связан(?:а|о|ы)?|"
            r"увеличивается|снижается)\b",
            text,
        ))

    @staticmethod
    def _quote_supports_claim(quote: str, claim: str) -> bool:
        """Check that a source passage identifies and reports a target claim."""
        cleaned = ResponseValidator._clean_evidence_quote(quote)
        if (not cleaned or ResponseValidator._evidence_has_metadata(cleaned)
                or ResponseValidator._evidence_is_cited_summary(cleaned)):
            return False
        claim_tokens = set(ResponseValidator._evidence_tokens(claim)) \
            - ResponseValidator.EVIDENCE_STOP_WORDS
        quote_tokens = set(ResponseValidator._evidence_tokens(cleaned)) \
            - ResponseValidator.EVIDENCE_STOP_WORDS
        if len(claim_tokens) < 4 or not quote_tokens:
            return False
        overlap = len(claim_tokens & quote_tokens) / len(claim_tokens)
        has_result = bool(re.search(
            r"(?i)\b(?:show(?:s|ed|n|ing)?|demonstrat(?:e|es|ed|ing)|"
            r"confirm(?:s|ed|ing)?|replicat(?:e|es|ed|ing)|prove(?:s|d|n|ing)?|"
            r"support(?:s|ed|ing)?|find(?:s|ings|ing)?|observ(?:e|es|ed|ing)|"
            r"measur(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|"
            r"reduc(?:e|es|ed|ing)|improv(?:e|es|ed|ing)|induc(?:e|es|ed|ing)|"
            r"caus(?:e|es|ed|ing)|correlat(?:e|es|ed|ing)|"
            r"показал(?:и|о|а)?|подтвердил(?:и|о|а)?|доказал(?:и|о|а)?|"
            r"выявил(?:и|о|а)?|наблюдал(?:и|о|а)?|измерил(?:и|о|а)?|"
            r"увеличил(?:и|о|а)?|снизил(?:и|о|а)?|опроверг(?:ли|нуто|ла)?)\b|"
            r"(?:=|≤|≥|±|×|÷|≠|≈|\b\d+(?:\.\d+)?\s*%)",
            cleaned,
        ))
        return overlap >= 0.4 and has_result

    @staticmethod
    def _status_evidence_is_sufficient(status: LabelStatus, quote: str) -> bool:
        """Reject proof/fact/refutation labels that their own quote does not justify."""
        text = ResponseValidator._clean_evidence_quote(quote)
        if status == LabelStatus.IDEA_CONCEPT:
            return True
        if status == LabelStatus.THEORY:
            return bool(re.search(
                r"(?i)\b(?:hypothes(?:is|ize|ized|es)|theor(?:y|ize|ized)|"
                r"propos(?:e|es|ed)|suggest(?:s|ed|ing)?|predict(?:s|ed|ing)?|"
                r"model(?:s|ed|ling)?|may|might|could|potentially|"
                r"объясня(?:ет|ют)|предполага(?:ет|ют)|гипотез(?:а|у|ы)|"
                r"предложил(?:и|а|о)?|предсказыва(?:ет|ют)|может|возможно)\b",
                text,
            ))
        if status == LabelStatus.PROOF:
            return bool(re.search(
                r"(?i)\b(?:formal proof|mathematical proof|proved theorem|"
                r"proof of|proven|proved|independently replicated|"
                r"independently confirmed|replicated independently|"
                r"reproduced independently|confirmed in (?:an? )?independent|"
                r"replicated in (?:an? )?independent|"
                r"independent (?:laboratory|lab|team|group) "
                r"(?:replicated|confirmed|reproduced)|"
                r"(?:replicated|confirmed|reproduced) by an? independent "
                r"(?:laboratory|lab|team|group)|"
                r"доказал(?:и|а|о)?|доказательств(?:о|а)|"
                r"независим(?:о|ая|ое|ые) подтвержден|"
                r"независим(?:о|ая|ое|ые) воспроизвед|"
                r"повторно подтвержден|реплицирован)\b",
                text,
            ))
        if status == LabelStatus.FACT:
            return bool(re.search(
                r"(?i)\b(?:show(?:s|ed|n|ing)?|demonstrat(?:e|es|ed|ing)|"
                r"reveal(?:s|ed|ing)?|observ(?:e|es|ed|ing)|measur(?:e|es|ed|ing)|"
                r"find(?:s|ings|ing)?|report(?:s|ed|ing)?|detect(?:s|ed|ing)?|"
                r"express(?:es|ed|ing)|increas(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|"
                r"reduc(?:e|es|ed|ing)|improv(?:e|es|ed|ing)|induc(?:e|es|ed|ing)|"
                r"result(?:s|ed|ing)?|was observed|were observed|"
                r"was measured|were measured|"
                r"показал(?:и|о|а)?|демонстрир(?:ует|уют|овал|овали)|"
                r"наблюдал(?:и|о|а)?|измерил(?:и|о|а)?|выявил(?:и|о|а)?|"
                r"обнаружил(?:и|о|а)?|увеличил(?:и|о|а)?|снизил(?:и|о|а)?|"
                r"составил(?:и|о|а)?|экспрессир(?:ует|уют|овал|овали))\b|"
                r"\b\d+(?:\.\d+)?\s*%\s+of\b",
                text,
            ))
        if status == LabelStatus.REFUTATION:
            return bool(re.search(
                r"(?i)\b(?:refut(?:e|es|ed|ing)|contradict(?:s|ed|ing)?|"
                r"disprov(?:e|es|ed|ing)|falsif(?:y|ies|ied)|"
                r"failed to replicate|did not (?:show|support|confirm|demonstrate)|"
                r"no evidence of|could not (?:confirm|replicate)|"
                r"unable to replicate|not (?:independently )?(?:replicated|confirmed)|"
                r"does not replicate|not statistically significant|"
                r"не подтвердил(?:и|а|о)?|не показал(?:и|а|о)?|"
                r"не удалось подтвердить|не воспроизвел(?:и|а|о)?|"
                r"опроверг(?:ли|ла|нуто)?|противоречит|не обнаружил(?:и|а|о)?)\b",
                text,
            ))
        return False

    @staticmethod
    def _quote_repeats_source_title(quote: str, source_title: str) -> bool:
        if not quote or not source_title:
            return False
        quote_tokens = ResponseValidator._evidence_tokens(
            ResponseValidator._clean_evidence_quote(quote))
        title_tokens = ResponseValidator._evidence_tokens(
            ResponseValidator._clean_evidence_quote(source_title))
        if len(quote_tokens) < 4 or len(quote_tokens) > len(title_tokens):
            # PDF extraction often appends the first sentence to a repeated
            # title. Catch the full title embedded in that longer passage.
            width = len(title_tokens)
            if width >= 8 and len(quote_tokens) >= width and any(
                    quote_tokens[i:i + width] == title_tokens
                    for i in range(len(quote_tokens) - width + 1)):
                return True
        if len(title_tokens) >= 8:
            # PDF column extraction can preserve only a shortened/reordered
            # article heading. High coverage of title-specific terms still
            # identifies front matter, even when the paragraph continues.
            title_terms = {token[:5] if len(token) >= 7 else token.rstrip("s")
                           for token in title_tokens
                           if token not in ResponseValidator.EVIDENCE_STOP_WORDS}
            quote_terms = {token[:5] if len(token) >= 7 else token.rstrip("s")
                           for token in quote_tokens}
            if title_terms and len(title_terms & quote_terms) / len(title_terms) >= 0.62:
                return True
        if len(quote_tokens) > len(title_tokens):
            return False
        width = len(quote_tokens)
        return any(title_tokens[i:i + width] == quote_tokens
                   for i in range(len(title_tokens) - width + 1))

    @staticmethod
    def _quote_only_repeats_label(quote: str, label: str) -> bool:
        """Reject a label, or a fragment composed only of label words."""
        quote_tokens = ResponseValidator._evidence_tokens(
            ResponseValidator._clean_evidence_quote(quote))
        label_tokens = set(ResponseValidator._evidence_tokens(label))
        return bool(quote_tokens) and set(quote_tokens).issubset(label_tokens)

    @staticmethod
    def _evidence_has_metadata(quote: str) -> bool:
        """Reject bibliographic/front-matter text masquerading as evidence."""
        return bool(re.search(
            r"(?i)(?:\bkey\s*words?\s*:|\bkeywords?\s*:|\bissn\b|\bisbn\b|"
            r"\be-?mail\s*:|\bcorrespondence\s*:|\bphone\s*:|\btel\s*:|"
            r"\bdoi\s*:|\bpublication\s+types?\s*:|\breceived\s*:|"
            r"\baccepted\s*:|\baffiliations?\s*:|\baddress\s*:|"
            r"\babbreviations\s*:|\backnowledg(?:e)?ments?\b|\bauthor contributions\b|"
            r"\bfunding\s*:|\bconflict of interest\b|\bpage\s+\d+\s+of\s+\d+\b|"
            r"\b(?:vol(?:ume)?\.?\s*)\d+\b|"
            r"\b(?:department|college|school|faculty|university|institute)\s+of\s+[A-Z]|"
            r"https?://|\b10\.\d{4,9}/[\w.()/-]+|\b[\w.+-]+@[\w.-]+\.[A-Z]{2,}\b|"
            r"\+?\d[\d\s().-]{7,}\d)", quote,
        ))

    @staticmethod
    def _evidence_is_cited_summary(quote: str) -> bool:
        """A cited prior-study result is not this article's own observation."""
        return bool(re.match(
            r"(?is)^\s*(?:\(\s*\d{1,3}\s*\)|\[\s*\d{1,3}\s*\]|"
            r"[A-Z][\w’'-]+\s+et\s+al\.?)\s+"
            r"(?:(?:has|have|had)\s+)?"
            r"(?:show(?:s|ed|n)?|demonstrat(?:e|es|ed)|find(?:s|ings|found)?|"
            r"report(?:s|ed)?|observ(?:e|es|ed))\b",
            quote,
        ))

    @staticmethod
    def _label_is_anchored(quote: str, label: str) -> bool:
        """Require the evidence to identify the concept attached to it.

        A conventional initialism formed from a multi-word label is accepted
        too (for example, MSC for mesenchymal stem cells).
        """
        quote_tokens = ResponseValidator._evidence_tokens(quote)
        label_tokens = [token for token in ResponseValidator._evidence_tokens(label)
                        if token not in ResponseValidator.EVIDENCE_STOP_WORDS]
        if not quote_tokens or not label_tokens:
            return False
        quote_set = set(quote_tokens)

        def present(label_token: str) -> bool:
            if label_token in quote_set:
                return True
            prefix = label_token[:5] if len(label_token) >= 7 else label_token.rstrip("s")
            return any(token.startswith(prefix) for token in quote_set)

        if all(present(token) for token in label_tokens):
            return True
        if len(label_tokens) >= 2:
            acronym = "".join(token[0] for token in label_tokens)
            if acronym and any(token.rstrip("s") == acronym for token in quote_tokens):
                return True
        return False

    @staticmethod
    def _label_is_claim_subject(quote: str, label: str) -> bool:
        """Reject evidence where the label is only a comparison/relation target."""
        cleaned = ResponseValidator._clean_evidence_quote(quote)
        label_tokens = [token for token in ResponseValidator._evidence_tokens(label)
                        if token not in ResponseValidator.EVIDENCE_STOP_WORDS]
        if not label_tokens:
            return False
        phrase = r"(?<!\w)" + r"[\s\-–—]+".join(map(re.escape, label_tokens)) + r"(?!\w)"
        matches = list(re.finditer(phrase, cleaned, flags=re.IGNORECASE))
        if not matches and len(label_tokens) >= 2:
            acronym = "".join(token[0] for token in label_tokens)
            matches = list(re.finditer(
                rf"(?<!\w){re.escape(acronym)}s?(?!\w)", cleaned, flags=re.IGNORECASE))
        if not matches:
            return False
        start = matches[0].start()
        prefix = cleaned[:start]
        # Restrict syntax checks to the label's sentence so an earlier
        # statement in a multi-sentence quote does not determine its role.
        sentence_start = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?")) + 1
        prefix = prefix[sentence_start:]
        if re.search(
                r"(?i)\b(?:than|compared\s+to|relative\s+to|versus|vs\.?|such\s+as)\b[^.!?]*$",
                prefix):
            return False
        if re.search(
                r"(?i)\b(?:of|from|with|by|for|in|among|between|against|via|using|including)\s*$",
                prefix):
            return False
        if re.search(
                r"(?i)\b(?:using|applying|including|involving|targeting|treating|testing|"
                r"evaluating|studying|investigating|comparing|selecting|transplanting|"
                r"utiliz(?:e|es|ed|ing)|utilis(?:e|es|ed|ing)|employ(?:s|ed|ing)?)\s*$",
                prefix):
            return False
        if re.search(
                r"(?i)\b(?:consist(?:s|ed|ing)?\s+of|compris(?:e|es|ed|ing)|"
                r"compos(?:e|es|ed|ing)\s+of|include(?:s|d|ing)?|"
                r"contain(?:s|ed|ing)?)\b[^.!?]*$",
                prefix,
        ):
            return False
        # A few PDF text layers render headings/abstracts with one space
        # between every glyph. Match these relation cues without rewriting
        # or displaying the source passage.
        def spaced_phrase(phrase: str) -> str:
            words = phrase.split()
            return r"(?<!\w)" + r"\s+".join(
                r"\s*".join(map(re.escape, word)) for word in words
            ) + r"(?!\w)"
        for relation in (
            "consisting of", "comprising", "composed of", "using", "applying",
            "including", "involving", "targeting", "treating", "testing",
            "evaluating", "studying", "investigating", "comparing", "selecting",
            "transplanting", "utilize", "utilizes", "utilized", "utilizing",
            "utilise", "utilises", "utilised", "utilising", "employs", "employed",
        ):
            if re.search(spaced_phrase(relation) + r"[^.!?]*$", prefix, re.IGNORECASE):
                return False
        # If the reported action occurs before the label, the sentence is
        # generally describing an effect on/relationship to the concept,
        # not a result about the concept as the observed subject.
        predicate_before_label = re.search(
            r"(?i)\b(?:show(?:s|ed|n)?|demonstrat(?:e|es|ed)|reveal(?:s|ed)?|"
            r"find(?:s|ings|found)?|observ(?:e|es|ed)|measur(?:e|es|ed)|"
            r"evaluat(?:e|es|ed)|analyz(?:e|es|ed)|analys(?:e|es|ed)|"
            r"express(?:es|ed|ing)|use(?:s|d)?|treat(?:s|ed)?|"
            r"increase(?:s|d)?|reduc(?:e|es|ed)|improv(?:e|es|ed)|"
            r"stud(?:y|ies|ied)|investigat(?:e|es|ed)|compare(?:s|d)?|"
            r"is|are|was|were|has|have|had)\b",
            prefix,
        )
        return predicate_before_label is None

    @staticmethod
    def _label_is_result_object(quote: str, label: str) -> bool:
        """Allow an explicitly reported outcome phrase after a causal verb.

        A finding can name the outcome of an intervention even when the
        concept is the grammatical object (for example, MSCs induced new
        bone formation). This supports an outcome node and a separately
        grounded relationship edge without treating every relation target
        as a node fact.
        """
        cleaned = ResponseValidator._clean_evidence_quote(quote)
        content = [token for token in ResponseValidator._evidence_tokens(label)
                   if token not in ResponseValidator.EVIDENCE_STOP_WORDS]
        if not content:
            return False
        token_matches = list(re.finditer(r"[^\W_]+", cleaned, flags=re.UNICODE))
        positions = {match.group(0).casefold(): match.start() for match in token_matches}
        found = [positions[token] for token in content if token in positions]
        if len(found) != len(content):
            return False
        start = min(found)
        end = max(found)
        if end - start > 90:
            return False
        sentence_start = max(cleaned.rfind(".", 0, start), cleaned.rfind("!", 0, start),
                             cleaned.rfind("?", 0, start)) + 1
        prefix = cleaned[sentence_start:start]
        return bool(re.search(
            r"(?i)\b(?:induc(?:e|es|ed)|caus(?:e|es|ed)|promot(?:e|es|ed)|"
            r"enhanc(?:e|es|ed)|enabl(?:e|es|ed)|prevent(?:s|ed)?|"
            r"increas(?:e|es|ed|ing)|reduc(?:e|es|ed|ing)|improv(?:e|es|ed)|"
            r"lead(?:s|ing)?\s+to|result(?:s|ed|ing)?\s+in)\b[^.!?]{0,100}$",
            prefix,
        ))

    @staticmethod
    def _quote_has_supporting_context(quote: str, label: str) -> bool:
        """Require source wording beyond a bare label or keyword occurrence.

        This is a conservative quality gate, not a semantic entailment model:
        it rejects metadata and mere mentions, requires the label to occur in
        the passage, and requires a concrete result, method, comparison, or
        measurement cue. Passage length alone is never evidence.
        """
        quote = ResponseValidator._clean_evidence_quote(quote)
        tokens = ResponseValidator._evidence_tokens(quote)
        if not tokens:
            return False
        if (ResponseValidator._evidence_has_metadata(quote)
                or ResponseValidator._evidence_is_cited_summary(quote)):
            return False
        label_content = set(ResponseValidator._evidence_tokens(label)) - ResponseValidator.EVIDENCE_STOP_WORDS
        content = [token for token in tokens if token not in ResponseValidator.EVIDENCE_STOP_WORDS]
        independent = {token for token in content if token not in label_content}
        claim_signal = bool(re.search(
            r"(?i)\b(?:show(?:s|ed|n)?|demonstrat(?:e|es|ed|ing)|reveal(?:s|ed)?|"
            r"indicat(?:e|es|ed)|find(?:s|ings)?|observ(?:e|ed|es)|measur(?:e|ed|es)|"
            r"evaluat(?:e|ed|es)|analyz(?:e|ed|es)|analys(?:e|ed|es)|cultivat(?:e|ed|es)|"
            r"simulat(?:e|ed|es|ing)|deriv(?:e|ed|es)|isolat(?:e|ed|es)|"
            r"test(?:s|ed|ing)?|assess(?:es|ed|ing)?|quantif(?:y|ied|ies)|exhibit(?:s|ed)|"
            r"express(?:es|ed)|promot(?:e|ed|es)|enhanc(?:e|ed|es)|produc(?:e|ed|es)|"
            r"treat(?:ed|s)?|increas(?:e|ed|es|ing)|reduc(?:e|ed|es|ing)|induc(?:e|ed|es)|"
            r"associat(?:e|ed|es)|correlat(?:e|ed|es)|improv(?:e|ed|es)|remain(?:ed|s)?|"
            r"maintain(?:s|ed|ing)?|possess(?:es|ed)?|"
            r"result(?:s|ed|ing)?|support(?:s|ed)?|prevent(?:s|ed)?|cause(?:s|d)?|"
            r"свидетельств|показал(?:и|о)?|выявил(?:и|о)?|обнаружил(?:и|о)?|"
            r"измерил(?:и|о)?|увеличил(?:и|о)?|снизил(?:и|о)?|сравнил(?:и|о)?|"
            r"культивировал(?:и|о)?|наблюдал(?:и|о)?|привел(?:и|о)?|привёл(?:и|о)?|"
            r"моделировал(?:и|о)?|оценил(?:и|о)?|изучил(?:и|о)?|выразил(?:и|о)?)\b",
            quote,
        ))
        formula_signal = bool(re.search(
            r"(?:=|≤|≥|±|×|÷|≠|≈|\b\d+(?:\.\d+)?\s*%|"
            r"\b(?:higher|lower|greater|less|similar|most|least|compared|versus|vs\.?|"
            r"increased|decreased|reduced|improved|worse|лучше|хуже|больше|меньше)\b)",
            quote, re.IGNORECASE,
        ))
        # Claim labels use several source-specific terms and a predicate. For
        # these, require the evidence to share the claim's key wording while
        # contributing at least two additional evidence terms. This keeps a
        # quoted explanation distinct from the label itself.
        if len(label_content) >= 4:
            label_claim = ResponseValidator._is_substantive_node_label(label)
            claim_overlap = len(label_content & set(content)) / max(1, len(label_content))
            claim_signal = bool(re.search(
                r"(?i)\b(?:show(?:s|ed|n|ing)?|demonstrat(?:e|es|ed|ing)|"
                r"reveal(?:s|ed|ing)?|indicat(?:e|es|ed|ing)|find(?:s|ings|ing)?|"
                r"observ(?:e|es|ed|ing)|measur(?:e|es|ed|ing)|evaluat(?:e|es|ed|ing)|"
                r"express(?:es|ed|ing)|increas(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|"
                r"reduc(?:e|es|ed|ing)|improv(?:e|es|ed|ing)|induc(?:e|es|ed|ing)|"
                r"inhibit(?:s|ed|ing)?|prevent(?:s|ed|ing)?|caus(?:e|es|ed|ing)|"
                r"associat(?:e|es|ed|ing)|correlat(?:e|es|ed|ing)|support(?:s|ed|ing)?|"
                r"confirm(?:s|ed|ing)?|replicat(?:e|es|ed|ing)|refut(?:e|es|ed|ing)|"
                r"contradict(?:s|ed|ing)?|propos(?:e|es|ed|ing)|predict(?:s|ed|ing)?|"
                r"explain(?:s|ed|ing)?|defin(?:e|es|ed|ing)|contain(?:s|ed|ing)?|"
                r"remain(?:s|ed|ing)?|свидетельств|показал(?:и|о|а)?|"
                r"выявил(?:и|о|а)?|обнаружил(?:и|о|а)?|измерил(?:и|о|а)?|"
                r"увеличил(?:и|о|а)?|снизил(?:и|о|а)?|подтвердил(?:и|о|а)?|"
                r"опроверг(?:ли|нуто|ла)?|предложил(?:и|о|а)?|определяет|содержит)\b",
                quote,
            ))
            return (label_claim and claim_overlap >= 0.5
                    and len(independent) >= 2 and len(content) >= 6
                    and (claim_signal or formula_signal))

        if (ResponseValidator._quote_only_repeats_label(quote, label)
                or not ResponseValidator._label_is_anchored(quote, label)
                or not (ResponseValidator._label_is_claim_subject(quote, label)
                        or ResponseValidator._label_is_result_object(quote, label))):
            return False
        method_only = bool(re.search(
            r"(?i)\b(?:used|utilized|cultured|washed|resuspended|passaged|incubated|"
            r"exposed|seeded|isolated|transplanted|administered|implanted|"
            r"objective was|aimed to|designed to|protocol)\b",
            quote,
        ))
        outcome_signal = bool(re.search(
            r"(?i)\b(?:show(?:s|ed|n)?|demonstrat(?:e|es|ed)|reveal(?:s|ed)?|"
            r"find(?:s|ings|found)?|observ(?:e|es|ed)|measur(?:e|es|ed)|"
            r"higher|lower|similar|greater|less|increased|decreased|reduced|improved|"
            r"significant(?:ly)?|viability|survival|proliferation|efficacy|"
            r"immunogenic|formation|regeneration|differentiation|express(?:es|ed|ing)|"
            r"induc(?:e|es|ed)|inhibit(?:s|ed)?|prevent(?:s|ed)?|cause(?:s|d)?|"
            r"promot(?:e|es|ed)|enhanc(?:e|es|ed)|support(?:s|ed)?)\b",
            quote,
        ))
        speculative_background = bool(re.search(
            r"(?i)\b(?:raised the possibility|could be valuable|may be valuable|"
            r"might be valuable|it is possible|could potentially|may potentially)\b",
            quote,
        ))
        # A long paragraph with unrelated words used to pass here. Require
        # both an identified concept and an evidence-bearing predicate or
        # quantitative/comparative result, with independent factual content.
        return (not speculative_background and not (method_only and not outcome_signal)
                and len(independent) >= 2
                and len(content) >= 4 and (claim_signal or formula_signal))

    @staticmethod
    def _analysis_context_is_supported(context: Any, label: str,
                                       evidence: List[str]) -> bool:
        """Keep only a factual paraphrase distinct from the source quote."""
        cleaned = ResponseValidator._clean_evidence_quote(context)
        if (not cleaned or ResponseValidator._evidence_has_metadata(cleaned)
                or not ResponseValidator._quote_has_supporting_context(cleaned, label)):
            return False
        context_tokens = set(ResponseValidator._evidence_tokens(cleaned)) - ResponseValidator.EVIDENCE_STOP_WORDS
        for quote in evidence:
            quote_tokens = set(ResponseValidator._evidence_tokens(
                ResponseValidator._clean_evidence_quote(quote))) - ResponseValidator.EVIDENCE_STOP_WORDS
            if not context_tokens or not quote_tokens:
                continue
            overlap = len(context_tokens & quote_tokens) / max(len(context_tokens), len(quote_tokens))
            if overlap >= 0.85:
                return False  # duplicate source wording is not analysis context
            if len(context_tokens & quote_tokens) < 2:
                return False  # avoid analysis detached from its evidence
        return True

    def validate_nodes(self, data: Dict, source_text: str = "") -> ValidationResult:
        result    = ValidationResult(valid=True)
        seen_ids: set = set()
        fixed_nodes   = []
        for i, node in enumerate(data.get("nodes", [])):
            if not node.get("id"):
                node["id"] = f"auto_{i:04d}"
                result.warnings.append(f"Node {i}: missing id assigned {node['id']}")
            if not node.get("text"):
                result.errors.append(f"Node {node['id']}: missing text — skipped")
                continue

            # ── Evidence must be grounded and support more than the label ──
            # The prompt instructs verbatim quoting; this is the algorithmic
            # check that it actually happened, not a request for the model
            # to police itself. A node whose evidence quotes NONE appear in
            # the actual chunk text is a real hallucination signal — a fact
            # pulled from outside knowledge rather than this document — and
            # gets rejected outright rather than kept with a caveat.
            evidence = node.get("evidence", []) or []
            if isinstance(evidence, str):
                evidence = [evidence]
            elif not isinstance(evidence, list):
                evidence = []
            evidence = [self._clean_evidence_quote(quote) for quote in evidence
                        if isinstance(quote, str) and quote.strip()]
            evidence = [quote for quote in evidence if quote]
            edge_endpoint_only = bool(node.get("edge_endpoint_only"))
            if (not edge_endpoint_only
                    and not self._is_substantive_node_label(node.get("text", ""))):
                result.errors.append(
                    f"Node {node['id']}: text must describe a source-specific claim, "
                    "not an isolated topic label — rejected")
                continue
            if edge_endpoint_only:
                # A relationship endpoint is a graph concept, not a node
                # observation. Keep it in the response only when a grounded
                # edge passage explicitly names it and states a relation.
                endpoint_supported = False
                relation_cue = re.compile(
                    r"(?i)\b(?:induc(?:e|es|ed)|caus(?:e|es|ed)|promot(?:e|es|ed)|"
                    r"enhanc(?:e|es|ed)|inhibit(?:s|ed)?|prevent(?:s|ed)?|"
                    r"enabl(?:e|es|ed)|contribut(?:e|es|ed)\s+to|lead(?:s|ing)?\s+to|"
                    r"result(?:s|ed|ing)?\s+in|associat(?:e|es|ed)\s+with|"
                    r"correlat(?:e|es|ed)\s+with|increas(?:e|es|ed|ing)|"
                    r"reduc(?:e|es|ed|ing)|improv(?:e|es|ed))\b")
                for edge in data.get("edges", []) or []:
                    if node.get("id") not in (edge.get("source"), edge.get("target")):
                        continue
                    edge_evidence = edge.get("evidence", []) or []
                    if isinstance(edge_evidence, str):
                        edge_evidence = [edge_evidence]
                    for quote in edge_evidence:
                        if not isinstance(quote, str) or not source_text:
                            continue
                        quote = self._clean_evidence_quote(quote)
                        if (self._quote_is_grounded(quote, source_text)
                                and self._label_is_anchored(quote, node.get("text", ""))
                                and relation_cue.search(quote)
                                and not self._evidence_has_metadata(quote)
                                and not self._evidence_is_cited_summary(quote)):
                            endpoint_supported = True
                            break
                    if endpoint_supported:
                        break
                if not endpoint_supported:
                    result.errors.append(
                        f"Node {node['id']}: edge-only endpoint lacks a grounded "
                        "relationship passage — rejected")
                    continue
                node["evidence"] = []
                node["full_context"] = ""
                node["secondary_contexts"] = []
                evidence = []
            elif source_text and not evidence:
                result.errors.append(
                    f"Node {node['id']}: no supporting evidence quote was supplied — rejected"
                )
                continue
            if evidence:
                if source_text:
                    grounded = [q for q in evidence if self._quote_is_grounded(q, source_text)]
                    ungrounded = [q for q in evidence if q not in grounded]
                    if not grounded:
                        result.errors.append(
                            f"Node {node['id']}: none of its {len(evidence)} evidence "
                            "quote(s) appear verbatim in the source text — rejected "
                            "as likely hallucinated"
                        )
                        continue
                    if ungrounded:
                        result.warnings.append(
                            f"Node {node['id']}: {len(ungrounded)}/{len(evidence)} "
                            f"evidence quote(s) not found verbatim in source, dropped: "
                            f"{ungrounded}"
                        )
                else:
                    grounded = evidence
                document_metadata = data.get("document_metadata", {})
                source_title = (str(document_metadata.get("title") or "")
                                if isinstance(document_metadata, dict) else "")
                study_protocol = bool(re.search(r"(?i)\bprotocol\b", source_title))
                unsupported = [q for q in grounded if (
                    not self._quote_has_supporting_context(q, node.get("text", ""))
                    or study_protocol
                    or self._quote_repeats_source_title(q, source_title)
                )]
                grounded = [q for q in grounded if q not in unsupported]
                if unsupported:
                    result.warnings.append(
                        f"Node {node['id']}: {len(unsupported)} evidence quote(s) lacked "
                        "supporting context or repeated the article title, and were dropped"
                    )
                if not grounded:
                    result.errors.append(
                        f"Node {node['id']}: no substantive source passage remains after "
                        "grounding and context checks — rejected as unsupported"
                    )
                    continue
                node["evidence"] = grounded
                analysis_context = node.get("full_context", "")
                if analysis_context and not self._analysis_context_is_supported(
                        analysis_context, node.get("text", ""), grounded):
                    node["full_context"] = ""

            if not edge_endpoint_only:
                status = parse_label_status(node.get("label_status"))
                status_evidence = self._clean_evidence_quote(
                    str(node.get("status_evidence") or ""))
                if status is None:
                    result.errors.append(
                        f"Node {node['id']}: label_status must be one of "
                        "idea_concept, theory, fact, proof, refutation — rejected")
                    continue
                if not status_evidence or status_evidence not in node.get("evidence", []):
                    result.errors.append(
                        f"Node {node['id']}: status_evidence must be one of its "
                        "grounded evidence quotes — rejected")
                    continue
                if source_text and not self._quote_is_grounded(status_evidence, source_text):
                    result.errors.append(
                        f"Node {node['id']}: status_evidence is not verbatim source text — rejected")
                    continue
                if not self._status_evidence_is_sufficient(status, status_evidence):
                    result.errors.append(
                        f"Node {node['id']}: its status is not supported by the quoted passage — rejected")
                    continue
                node["label_status"] = label_status_name(status)
                node["label_status_code"] = int(status)
                node["status_evidence"] = status_evidence

            # ── Same check for secondary_contexts' raw text ─────────────────
            # These are meant to be names/phrases lifted from the document
            # too ("raw text as mentioned in the source, do not normalise").
            # An ungrounded one costs a real Pass D arbitration call over a
            # reference the document never actually made — drop it here,
            # before it ever reaches Pass D, rather than the node itself
            # (a made-up external reference is a smaller failure than a
            # made-up node, so only that one context is removed).
            secondary_contexts = node.get("secondary_contexts", []) or []
            if source_text and secondary_contexts:
                kept_ctx, dropped_ctx = [], []
                for ctx in secondary_contexts:
                    ctx_text = ctx.get("text", "")
                    if self._quote_is_grounded(ctx_text, source_text):
                        kept_ctx.append(ctx)
                    else:
                        dropped_ctx.append(ctx_text)
                if dropped_ctx:
                    result.warnings.append(
                        f"Node {node['id']}: {len(dropped_ctx)} secondary_context(s) "
                        f"not found verbatim in source, dropped: {dropped_ctx}"
                    )
                node["secondary_contexts"] = kept_ctx

            contradicted_claims = node.get("contradicted_claims", []) or []
            if not isinstance(contradicted_claims, list):
                contradicted_claims = [contradicted_claims]
            claim_status = parse_label_status(node.get("label_status"))
            kept_claims, dropped_claims = [], []
            if claim_status == LabelStatus.REFUTATION and source_text:
                for claim in contradicted_claims:
                    if isinstance(claim, str):
                        claim = {"text": claim}
                    if not isinstance(claim, dict):
                        dropped_claims.append(str(claim))
                        continue
                    claim_text = str(claim.get("text", "")).strip()
                    if not claim_text or not self._quote_is_grounded(claim_text, source_text):
                        dropped_claims.append(claim_text or repr(claim))
                        continue
                    fixed_claim = dict(claim)
                    fixed_claim["text"] = claim_text
                    aliases = fixed_claim.get("aliases", []) or []
                    if isinstance(aliases, str):
                        aliases = [aliases]
                    fixed_claim["aliases"] = [
                        str(alias) for alias in aliases
                        if isinstance(alias, str) and self._quote_is_grounded(alias, source_text)
                    ][:3]
                    fixed_claim["role"] = "contradicts"
                    kept_claims.append(fixed_claim)
            elif contradicted_claims:
                dropped_claims = [
                    str(claim.get("text", claim)) if isinstance(claim, dict) else str(claim)
                    for claim in contradicted_claims
                ]
            if dropped_claims:
                result.warnings.append(
                    f"Node {node['id']}: contradicted_claims without a grounded refutation "
                    f"were dropped: {dropped_claims}"
                )
            node["contradicted_claims"] = kept_claims

            if node["id"] in seen_ids:
                node["id"] = f"{node['id']}_{i}"
                result.warnings.append(f"Duplicate id resolved to {node['id']}")
            seen_ids.add(node["id"])
            for f in self.SCORE_FIELDS_10:
                v = node.get(f)
                if v is None:
                    node[f] = 5.0
                    result.warnings.append(f"Node {node['id']}.{f}: missing → 5.0")
                else:
                    try:
                        node[f] = float(np.clip(float(v), 0.0, 10.0))
                    except (ValueError, TypeError):
                        node[f] = 5.0
                        result.warnings.append(f"Node {node['id']}.{f}: non-numeric → 5.0")
            if node.get("entity_type") not in INGESTION_ENTITY_TYPES:
                node["entity_type"] = "research_frontier"
            fixed_nodes.append(node)
        result.fixed["nodes"] = fixed_nodes
        return result

    def validate_physical(self, data: Dict, valid_node_ids: set) -> ValidationResult:
        result = ValidationResult(valid=True)
        fixed  = {}
        for node_id, axes in data.get("physical_scores", {}).items():
            if node_id not in valid_node_ids:
                result.warnings.append(f"Physical scores for unknown '{node_id}' — skipped")
                continue
            fixed_axes = {}
            for ax in PHYSICAL_AXIS_ORDER:
                v = axes.get(ax)
                try:
                    fixed_axes[ax] = float(np.clip(float(v), 0.0, 1.0))
                except (ValueError, TypeError):
                    fixed_axes[ax] = 0.5
            fixed[node_id] = fixed_axes
        result.fixed["physical_scores"] = fixed
        return result

    def validate_edges(self, data: Dict, valid_node_ids: set, source_text: str = "") -> ValidationResult:
        result      = ValidationResult(valid=True)
        fixed_edges = []
        source_nodes = {node.get("id"): node for node in data.get("nodes", [])
                        if isinstance(node, dict) and node.get("id")}
        node_statuses = {node_id: (parse_label_status(node.get("label_status_code"))
                                   or parse_label_status(node.get("label_status")))
                         for node_id, node in source_nodes.items()}
        for i, edge in enumerate(data.get("edges", [])):
            src, tgt = edge.get("source"), edge.get("target")
            if src not in valid_node_ids:
                result.warnings.append(f"Edge {i}: unknown source '{src}' — skipped")
                continue
            if tgt not in valid_node_ids:
                result.warnings.append(f"Edge {i}: unknown target '{tgt}' — skipped")
                continue
            if src == tgt:
                result.warnings.append(f"Edge {i}: self-loop on '{src}' — skipped")
                continue
            relationship_type = str(edge.get("relationship_type", "related")).strip().lower()
            if relationship_type in {"cites", "funded_by", "popularized_by", "depicted_in"}:
                result.warnings.append(
                    f"Edge {i} ({src}->{tgt}): '{relationship_type}' is recorded as source influence, not as a graph relation")
                continue
            if relationship_type == "contradicts" and node_statuses.get(src) != LabelStatus.REFUTATION:
                result.warnings.append(
                    f"Edge {i}: contradicts source '{src}' is not classified as a refutation — skipped")
                continue
            if relationship_type in {"supports", "confirms", "proves"}:
                source_status = node_statuses.get(src)
                source_claim = source_nodes.get(src, {})
                target_claim = source_nodes.get(tgt, {})
                if source_status not in {LabelStatus.FACT, LabelStatus.PROOF}:
                    result.warnings.append(
                        f"Edge {i}: {relationship_type} requires a fact/proof source label — skipped")
                    continue
                if relationship_type == "proves" and source_status != LabelStatus.PROOF:
                    result.warnings.append(
                        f"Edge {i}: proves requires a proof-status source label — skipped")
                    continue
                source_quotes = source_claim.get("evidence", []) or []
                if isinstance(source_quotes, str):
                    source_quotes = [source_quotes]
                edge_quotes = edge.get("evidence", []) or []
                if isinstance(edge_quotes, str):
                    edge_quotes = [edge_quotes]
                grounded_source_quotes = [
                    self._clean_evidence_quote(str(quote))
                    for quote in source_quotes if isinstance(quote, str)
                ]
                edge_quotes = [self._clean_evidence_quote(str(quote))
                               for quote in edge_quotes if isinstance(quote, str)]
                supported_quotes = [quote for quote in edge_quotes
                                    if quote in grounded_source_quotes
                                    and self._quote_supports_claim(
                                        quote, str(target_claim.get("text", "")))]
                if not supported_quotes:
                    result.warnings.append(
                        f"Edge {i} ({src}->{tgt}): {relationship_type} needs a source-label quote "
                        "that directly supports the target claim — skipped")
                    continue
                edge["evidence"] = supported_quotes
            # ── Anti-hallucination: same verbatim-grounding check as nodes ──
            evidence = edge.get("evidence", []) or []
            if isinstance(evidence, str):
                evidence = [evidence]
            if source_text and evidence:
                grounded = [q for q in evidence
                            if self.__class__._quote_is_grounded(q, source_text)]
                if not grounded:
                    result.warnings.append(
                        f"Edge {i} ({src}->{tgt}): none of its evidence quote(s) "
                        "appear verbatim in the source — rejected as likely "
                        "hallucinated"
                    )
                    continue
                edge["evidence"] = grounded
            elif relationship_type == "contradicts":
                result.warnings.append(
                    f"Edge {i} ({src}->{tgt}): contradiction requires a verbatim source quote — skipped")
                continue
            if relationship_type == "contradicts" and source_text and not edge.get("evidence"):
                result.warnings.append(
                    f"Edge {i} ({src}->{tgt}): contradiction quote was not grounded — skipped")
                continue
            edge_evidence = edge.get("evidence", []) or []
            if isinstance(edge_evidence, str):
                edge_evidence = [edge_evidence]
            if edge_evidence:
                parameter_observations, parameter_warnings = self._validate_parameter_observations(
                    edge.get("parameter_observations", []), source_text=source_text,
                    required_quotes=edge_evidence,
                )
            else:
                parameter_observations = []
                parameter_warnings = ([
                    "parameter measurements require a grounded quote for the edge relation"
                ] if edge.get("parameter_observations") else [])
            result.warnings.extend(
                f"Edge {i} ({src}->{tgt}): {warning}" for warning in parameter_warnings)
            # LLM-provided edge scores are intentionally ignored. Pair
            # similarity and temporal proximity are derived from endpoint
            # data by NodeEdgeAssembler; evidence-sensitive factors are
            # accumulated from grounded, source-attributed observations.
            edge.update({
                "semantic_similarity": 0.0,
                "temporal_proximity": 0.0,
                "limitation_resolution": 0.0,
                "citation_link": 0.0,
                "investment_correlation": 0.0,
                "social_correlation": 0.0,
                "inhibitory_force": 0.0,
                "confidence": 1.0 if edge.get("evidence") else 0.0,
            })
            edge["factor_evidence"] = {}
            edge["parameter_observations"] = parameter_observations
            fixed_edges.append(edge)
        result.fixed["edges"]          = fixed_edges
        result.fixed["temporal_zones"] = data.get("temporal_zones", [])
        return result

    def validate_influence_observations(self, data: Dict,
                                        source_text: str = "") -> ValidationResult:
        """Keep only source-grounded, typed signals about existing concepts."""
        result = ValidationResult(valid=True)
        allowed = {"citation", "data_use", "investment", "social_attention",
                   "fictional_hype", "unmet_need"}
        fixed = []
        cue_patterns = {
            "citation": r"(?i)\b(cit(?:e|es|ed|ing|ation|ations)|refer(?:s|red|ence|ences)|bibliograph)\b|ссылк\w*|цитир\w*|цитат\w*",
            "data_use": r"(?i)\b(data|dataset|database|results?|measurements?)\b.{0,50}\b(from|of|reported|published|provided)\b|\b(using|used|based on|adopted|reanalysed|reanalyzed)\b.{0,60}\b(data|dataset|database|results?|measurements?)\b|данн\w*.{0,60}(из|использ)|на основе.{0,60}(данн\w*|результат\w*)",
            "investment": r"(?i)\b(invest(?:s|ed|ment|ments|ing)?|fund(?:s|ed|ing)?|financ(?:e|ed|ing)|grant|venture capital|back(?:s|ed|ing))\b|инвест\w*|финанс\w*|вложен\w*|грант\w*",
            "social_attention": r"(?i)\b(popular|popularity|hype|attention|public interest|media coverage|discussion|discussed|forum|social media|mentions?)\b|популяр\w*|хайп\w*|вниман\w*|обсужд\w*|медиа|форум\w*|упомин\w*|общественн\w* интерес\w*|общество.{0,20}интерес\w*|интерес\w*.{0,20}общество",
            "fictional_hype": r"(?i)\b(fiction|fictional|book|novel|film|movie|story|plot|character|science fiction)\b|фантаст\w*|роман\w*|книг\w*|фильм\w*|сюжет\w*|персонаж\w*",
            "unmet_need": r"(?i)\b(unmet need|needs?|needed|need for|demand for|demand|required|requirement|shortage|degradation|obsolescence|growing (?:data )?volume|storage capacity)\b|необходимост\w*|потребност\w*|нужд\w*|спрос\w*|дефицит\w*|деградац\w*|износ\w*|устарев\w*|растущ\w* объем\w*",
        }
        for index, item in enumerate(data.get("influence_observations", []) or []):
            if not isinstance(item, dict):
                continue
            signal_type = str(item.get("signal_type", "")).strip().casefold()
            target_text = str(item.get("target_text", "")).strip()
            evidence = self._clean_evidence_quote(str(item.get("evidence", "")))
            if signal_type not in allowed or not target_text or not evidence:
                result.warnings.append(f"Influence observation {index}: missing target, type, or evidence")
                continue
            if source_text and not self._quote_is_grounded(evidence, source_text):
                result.warnings.append(f"Influence observation {index}: evidence is not verbatim source text")
                continue
            if source_text and not self._label_is_anchored(evidence, target_text):
                result.warnings.append(
                    f"Influence observation {index}: the evidence does not identify target '{target_text}'")
                continue
            if not re.search(cue_patterns[signal_type], evidence):
                result.warnings.append(
                    f"Influence observation {index}: quote does not support signal type '{signal_type}'")
                continue
            if self._evidence_has_metadata(evidence):
                result.warnings.append(f"Influence observation {index}: bibliographic metadata is not evidence")
                continue
            item = dict(item)
            item["target_text"] = target_text[:500]
            item["signal_type"] = signal_type
            item["evidence"] = evidence[:1000]
            parameter_observations, parameter_warnings = self._validate_parameter_observations(
                item.get("parameter_observations", []), source_text=source_text,
                anchor_text=target_text,
            )
            result.warnings.extend(
                f"Influence observation {index}: {warning}" for warning in parameter_warnings)
            item["parameter_observations"] = parameter_observations
            item["aliases"] = [str(alias)[:250] for alias in (item.get("aliases", []) or [])
                               if isinstance(alias, str) and alias.strip()][:5]
            fixed.append(item)
        result.fixed["influence_observations"] = fixed
        return result

    def _validate_parameter_observations(
            self, items: Any, source_text: str = "", anchor_text: str = "",
            required_quotes: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Validate source-quoted numeric measurements for an edge factor."""
        allowed_factors = {
            "limitation_resolution", "citation_link", "investment_correlation",
            "social_correlation", "inhibitory_force",
        }
        factor_cues = {
            "limitation_resolution": r"(?i)limit|constraint|bottleneck|resolve|reduc|improv|mitigat|огранич|барьер|сниж|устран|решен",
            "citation_link": r"(?i)citat|cit(?:e|es|ed|ing)|referenc|cited by|ссылк|цитир|цитат",
            "investment_correlation": r"(?i)invest|fund|financ|capital|grant|investment|инвест|финанс|вложен|грант",
            "social_correlation": r"(?i)social|public|popular|attention|media|discussion|forum|mention|audience|belief|view|engagement|обществ|популяр|вниман|медиа|обсужд|форум|упомин|аудитор|верован|просмотр",
            "inhibitory_force": r"(?i)inhibit|suppress|block|impair|decreas|reduce|negative effect|препят|подав|блокир|сниж|уменьш|негатив",
        }
        warnings: List[str] = []
        fixed: List[Dict[str, Any]] = []
        seen = set()
        if not isinstance(items, list):
            return fixed, warnings
        required = [self._clean_evidence_quote(str(q)) for q in (required_quotes or [])
                    if isinstance(q, str) and q.strip()]
        number_pattern = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            factor = str(item.get("factor", "")).strip().casefold()
            evidence = self._clean_evidence_quote(str(item.get("evidence", "")))
            unit = str(item.get("unit", "")).strip().casefold()
            if factor not in allowed_factors or unit not in {"percent", "fraction", "ratio", "raw"}:
                warnings.append(f"measurement {index}: unknown factor or unit")
                continue
            if not evidence:
                warnings.append(f"measurement {index}: missing numeric evidence quote")
                continue
            if source_text and not self._quote_is_grounded(evidence, source_text):
                warnings.append(f"measurement {index}: quote is not verbatim source text")
                continue
            if required and not any(evidence in quote for quote in required):
                warnings.append(f"measurement {index}: quote is not part of the edge evidence")
                continue
            if anchor_text and not self._label_is_anchored(evidence, anchor_text):
                warnings.append(f"measurement {index}: quote does not identify its target concept")
                continue
            if not re.search(factor_cues[factor], evidence):
                warnings.append(f"measurement {index}: quote does not state the '{factor}' context")
                continue
            try:
                value = float(item.get("value"))
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                warnings.append(f"measurement {index}: value must be a finite non-negative number")
                continue
            matching_tokens = []
            for token_match in number_pattern.finditer(evidence):
                token = token_match.group(0)
                try:
                    quoted = float(token.replace(",", "."))
                    if math.isclose(value, quoted, rel_tol=1e-6, abs_tol=1e-8):
                        matching_tokens.append(token_match)
                except ValueError:
                    continue
            if not matching_tokens:
                warnings.append(f"measurement {index}: reported value is absent from its quote")
                continue
            normalized = None
            denominator = None
            if unit == "percent":
                explicit_percent = any(re.match(
                    r"(?i)\s*(?:%|percent(?:age)?\b|per cent\b|процент\w*)",
                    evidence[match.end():match.end() + 16],
                ) for match in matching_tokens)
                if value > 100.0 or not explicit_percent:
                    warnings.append(f"measurement {index}: percent value needs an explicit percent marker")
                    continue
                normalized = value / 100.0
            elif unit == "fraction":
                if value > 1.0:
                    warnings.append(f"measurement {index}: fraction must be between 0 and 1")
                    continue
                if not re.search(
                        r"(?i)fraction|proportion|ratio|correlation|coefficient|probability|likelihood|index|score|доля|отношени|коэффициент|вероятност|индекс|оценк", evidence):
                    warnings.append(f"measurement {index}: quote does not identify a normalized fraction or score")
                    continue
                normalized = value
            elif unit == "ratio":
                try:
                    denominator = float(item.get("denominator"))
                    if (not math.isfinite(denominator) or denominator <= 0.0
                            or value > denominator):
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    warnings.append(f"measurement {index}: ratio needs a positive quoted denominator")
                    continue
                denominator_matches = any(
                    math.isclose(denominator, float(token.group(0).replace(",", ".")),
                                 rel_tol=1e-6, abs_tol=1e-8)
                    for token in number_pattern.finditer(evidence)
                )
                if not denominator_matches:
                    warnings.append(f"measurement {index}: denominator is absent from its quote")
                    continue
                if not re.search(r"(?i)\bof\b|\bout of\b|\bamong\b|\bfrom\b|/|из|от|среди", evidence):
                    warnings.append(f"measurement {index}: quote does not state a numerator/denominator relation")
                    continue
                normalized = value / denominator
            record = {
                "factor": factor, "raw_value": value, "unit": unit,
                "denominator": denominator,
                "normalized_value": normalized, "evidence": evidence[:1000],
            }
            key = (factor, value, denominator, unit, evidence)
            if key not in seen:
                fixed.append(record)
                seen.add(key)
        return fixed, warnings


# ── Layer 5b: Edge Semantic Validator (optional LLM pass) ────────────────────
#
# Enabled via CONFIG["edge_validation_enabled"] = True.
#
# The validator batches proposed edges and asks the LLM a single structured
# question: is each connection semantically plausible given the two node
# descriptions, their domains, entity types, and the stated relationship?
#
# Classic example it catches:
#   source: "1984 (Orwell novel)"  entity_type=fictional_event  domain=literature
#   target: "Room 101 CIA torture programme"  domain=military_intelligence
#   relationship: DERIVED_FROM
#   => LLM: plausibility 0.12 -- fictional political event != real science
#
# Each edge gets a plausibility score [0-1] and a one-sentence rationale.
# Edges below CONFIG["edge_validation_min_score"] are dropped and logged.
# ─────────────────────────────────────────────────────────────────────────────

class EdgeSemanticValidator:
    """LLM-based plausibility filter for proposed KnowledgeEdge connections.

    Usage (inside IngestionPipeline)::

        validator = EdgeSemanticValidator(llm_annotator)
        kept, dropped = validator.validate_batch(edges, node_lookup)

    *node_lookup* is a Dict[str, KnowledgeNode] — both endpoints must be present.
    Returns ``(kept_edges, dropped_edges)``.  Dropped edges carry a
    ``__validation_rationale`` key in their ``__dict__`` for diagnostics.

    When ``CONFIG["edge_validation_enabled"]`` is False the class is
    instantiated but ``validate_batch()`` is a no-op that returns
    ``(edges, [])`` immediately with zero LLM overhead.
    """

    EDGE_VALIDATION_PROMPT = """\
You are an expert knowledge-graph curator validating proposed edges in a \
scientific / technological / historical knowledge graph.

Your task: for each proposed edge decide whether the connection is \
semantically plausible and factually defensible given the two node \
descriptions and the stated relationship type.

━━━ DROP RULES (any one is sufficient to drop) ━━━
1. FICTIONAL → REAL  boundary crossing
   Fictional characters, events, or organisations (literature, films, games)
   linked to real-world science, politics, or history as a direct causal
   influence.  Example: "Room 101 from 1984 (Orwell)" → "CIA interrogation
   doctrine" with relationship DERIVED_FROM.
2. CHRONOLOGICAL IMPOSSIBILITY
   The source node's year is AFTER the target's year and the relationship
   implies the source caused or enabled the target.
3. EPISTEMIC DOMAIN MISMATCH without documented bridge
   Connecting social media content / fan speculation directly to
   peer-reviewed discoveries as a causal link (not as "commentary on").
4. CONTRADICTORY RELATIONSHIP TYPE
   Relationship type is inconsistent with node types (e.g. a person node
   DERIVED_FROM a physical law).
5. CONFIDENCE < 0.15 AND semantic_similarity < 0.20
   Near-zero strength with no supporting evidence — likely a hallucinated edge.

━━━ KEEP RULES ━━━
- Both nodes real AND share the same epistemic register (both scientific,
  both political, both fictional).
- Relationship is historically or scientifically documented.
- Cross-domain connections are acceptable when the relationship type is
  "influenced_by", "commentary_on", "inspired_by" — not "DERIVED_FROM",
  "ENABLES", or "CAUSED_BY".

━━━ SCORING GUIDE ━━━
  1.0  — obvious, well-documented connection
  0.7  — plausible, no strong counter-evidence
  0.4  — uncertain / weak but possible (threshold by default)
  0.2  — unlikely, domain mismatch or weak evidence
  0.0  — clearly wrong (fictional→real causal, impossible chronology)

EDGES TO EVALUATE:
{edges_block}

OUTPUT (ONLY valid JSON, NO markdown fences, no extra keys):
{{
  "results": [
    {{
      "edge_id": "<id>",
      "plausibility": <0.0-1.0>,
      "verdict": "<keep or drop>",
      "rationale": "<one sentence explaining the decision, max 25 words>"
    }}
  ]
}}

IMPORTANT: verdict must be exactly "keep" or "drop" — never any other value.\
"""

    MAX_TOKENS = CONFIG["edge_validator_max_tokens"]

    def __init__(self, annotator):
        self.annotator  = annotator
        self.enabled    = CONFIG.get("edge_validation_enabled", False)
        self.min_score  = CONFIG.get("edge_validation_min_score", 0.40)
        self.batch_size = CONFIG.get("edge_validation_batch_size", 12)
        self.fail_open  = CONFIG.get("edge_validation_fail_open", True)

    # ── Public API ──────────────────────────────────────────────────────────

    def validate_batch(self, edges, node_lookup):
        """Filter *edges* through the LLM plausibility check.

        Returns ``(kept, dropped)``.  When validation is disabled returns
        ``(edges, [])`` immediately with zero LLM overhead.
        """
        if not self.enabled or not edges:
            return list(edges), []

        kept, dropped = [], []
        for i in range(0, len(edges), self.batch_size):
            batch = edges[i : i + self.batch_size]
            b_kept, b_dropped = self._validate_batch_chunk(batch, node_lookup)
            kept.extend(b_kept)
            dropped.extend(b_dropped)

        if dropped:
            logger.info(
                f"[EdgeValidator] Dropped {len(dropped)}/{len(edges)} edge(s) "
                f"(min_score={self.min_score}):"
            )
            for e in dropped:
                rationale = e.__dict__.get("__validation_rationale", "no rationale")
                logger.info(
                    f"  x {e.source_id} -> {e.target_id} "
                    f"[{e.relationship_type}] -- {rationale}"
                )
        return kept, dropped

    # ── Internals ───────────────────────────────────────────────────────────

    def _validate_batch_chunk(self, edges, node_lookup):
        """Send one LLM call for a single batch of <= batch_size edges."""
        edge_lines = []
        for idx, edge in enumerate(edges):
            src = node_lookup.get(edge.source_id)
            tgt = node_lookup.get(edge.target_id)
            edge_lines.append(self._edge_line(edge, src, tgt, idx))

        prompt = self.EDGE_VALIDATION_PROMPT.format(
            edges_block="\n\n".join(edge_lines)
        )

        response = self.annotator.call(prompt, pass_type="edge_validation")
        if not response.success or not response.parsed:
            logger.warning(
                "[EdgeValidator] LLM call failed -- applying "
                f"{'fail-open (keeping all)' if self.fail_open else 'fail-closed (dropping all)'}"
            )
            if self.fail_open:
                return list(edges), []
            for e in edges:
                e.__dict__["__validation_rationale"] = "LLM validation failed (fail-closed)"
            return [], list(edges)

        results_map = {
            item.get("edge_id", ""): item
            for item in response.parsed.get("results", [])
            if item.get("edge_id")
        }
        if any(edge.id not in results_map for edge in edges):
            self.annotator.incomplete_calls += 1

        kept, dropped = [], []
        for edge in edges:
            if str(getattr(edge, "relationship_type", "")).casefold() == "contradicts":
                source_node = node_lookup.get(edge.source_id)
                source_private = getattr(source_node, "__dict__", {}) if source_node else {}
                status = parse_label_status(source_private.get("label_status_code"))
                source_quotes = source_private.get("source_evidence", []) or []
                if isinstance(source_quotes, str):
                    source_quotes = [source_quotes]
                source_observations = source_private.get("source_observations", []) or []
                if not isinstance(source_observations, list):
                    source_observations = []
                for point in source_observations:
                    if (isinstance(point, dict)
                            and parse_label_status(point.get("label_status_code")) == LabelStatus.REFUTATION):
                        point_quotes = point.get("evidence", []) or []
                        source_quotes.extend([point_quotes] if isinstance(point_quotes, str)
                                             else point_quotes)
                edge_quotes = getattr(edge, "evidence", []) or []
                if isinstance(edge_quotes, str):
                    edge_quotes = [edge_quotes]
                if status == LabelStatus.REFUTATION or any(
                        isinstance(point, dict) and parse_label_status(
                            point.get("label_status_code")) == LabelStatus.REFUTATION
                        for point in source_observations):
                    if any(quote in source_quotes for quote in edge_quotes):
                        edge.__dict__["__validation_score"] = 1.0
                        edge.__dict__["__validation_rationale"] = (
                            "Kept source-grounded contradiction from a refutation label.")
                        kept.append(edge)
                        continue
            result = results_map.get(edge.id)
            if result is None:
                (kept if self.fail_open else dropped).append(edge)
                if not self.fail_open:
                    edge.__dict__["__validation_rationale"] = "no LLM result (fail-closed)"
                continue

            score     = float(result.get("plausibility", 1.0))
            rationale = result.get("rationale", "")
            verdict   = result.get("verdict", "keep")

            edge.__dict__["__validation_score"]     = round(score, 3)
            edge.__dict__["__validation_rationale"] = rationale

            if score >= self.min_score and verdict != "drop":
                kept.append(edge)
            else:
                dropped.append(edge)

        return kept, dropped

    @staticmethod
    def _edge_line(edge, src, tgt, idx):
        """Render one edge as a structured text block for the LLM prompt."""
        def _node_desc(n, eid):
            if n is None:
                return f'id="{eid}"  [node data unavailable]'
            year = ""
            if n.timestamp and n.timestamp != 0.0:
                try:
                    year = f"  year~{int(n.timestamp / (365.25*24*3600)) + 1970}"
                except Exception:
                    pass
            return (
                f'id="{n.id}"\n'
                f'  text: "{n.text[:120]}"\n'
                f'  domain: {n.domain}  entity_type: {n.entity_type}'
                f'  source_type: {getattr(n, "source_type", "unknown")}{year}'
            )

        return (
            f"--- Edge {idx + 1} (id={edge.id}) ---\n"
            f"SOURCE NODE:\n{_node_desc(src, edge.source_id)}\n"
            f"TARGET NODE:\n{_node_desc(tgt, edge.target_id)}\n"
            f"RELATIONSHIP: {edge.relationship_type}\n"
            f"  semantic_similarity={edge.semantic_similarity:.2f}  "
            f"confidence={edge.confidence:.2f}  "
            f"temporal_proximity={edge.temporal_proximity:.2f}\n"
            f"  evidence: {'; '.join(edge.evidence[:2]) if edge.evidence else 'none'}"
        )


# Register edge_validation as a known pass_type in LLMAnnotator token budget.
LLMAnnotator.MAX_TOKENS_PASS["edge_validation"] = 1024


# ── Layer 5: Object assembler ─────────────────────────────────────────────────

class NodeEdgeAssembler:
    """
    Assembles KnowledgeEdge / TemporalZone from validated dicts.

    TODO #19: node assembly (make_stable_id() / assemble_node(), the
    MD5(source + "::" + text)[:12] identity scheme) removed from this class
    entirely — it was the "remnant of the old implementation" the manifest
    warned about, kept alive only behind the now-deleted
    "ingestion_use_canonical_key" flag. Node identity has exactly one path
    now: CanonicalNodeAssembler.assemble_node_with_canonical_key(). This
    class still owns assemble_edge()/assemble_zone() — edges/zones don't
    carry an identity scheme of their own (their ids are plain uuid4 slugs,
    never MD5), so there is nothing here for CanonicalKey to replace or
    collide with.
    """

    def __init__(self, embedder=None):
        self.embedder = embedder

    def assemble_edge(self, item: Dict, source: str, timestamp: float,
                       node_id_map: Dict[str, str], node_lookup: Optional[Dict[str, Any]] = None,
                       source_trust: float = 0.5,
                       source_identity: Optional[str] = None,
                       source_metadata: Optional[Dict[str, Any]] = None) -> KnowledgeEdge:
        src_id = node_id_map.get(item["source"], item["source"])
        tgt_id = node_id_map.get(item["target"], item["target"])
        node_lookup = node_lookup or {}
        source_node = node_lookup.get(item["source"]) or node_lookup.get(src_id)
        target_node = node_lookup.get(item["target"]) or node_lookup.get(tgt_id)
        def node_value(node, name, default=None):
            return node.get(name, default) if isinstance(node, dict) else getattr(node, name, default)
        pair_metrics = derive_pair_metrics(
            source_node, target_node,
            source_text=str(node_value(source_node, "text", item.get("source_text", ""))),
            target_text=str(node_value(target_node, "text", item.get("target_text", ""))),
            source_timestamp=float(node_value(source_node, "timestamp", 0.0) or 0.0),
            target_timestamp=float(node_value(target_node, "timestamp", 0.0) or 0.0),
        )
        evidence = item.get("evidence", []) or []
        if isinstance(evidence, str):
            evidence = [evidence]
        parameter_observations = list(item.get("parameter_observations", []) or [])
        factor_evidence = {}
        edge = KnowledgeEdge(
            id=str(uuid.uuid4())[:8], source_id=src_id, target_id=tgt_id,
            semantic_similarity=pair_metrics["semantic_similarity"],
            temporal_proximity=pair_metrics["temporal_proximity"],
            limitation_resolution=0.0,
            citation_link=0.0,
            investment_correlation=0.0,
            social_correlation=0.0,
            inhibitory_force=0.0,
            confidence=1.0 if evidence else 0.0,
            relationship_type=item.get("relationship_type", "related"),
            evidence=evidence, timestamp=timestamp,
        )
        edge.compute_total_weight()
        edge.__dict__["factor_evidence"] = factor_evidence
        edge.__dict__["source_identity"] = source_identity or source
        edge.__dict__["source_metadata"] = dict(source_metadata or {})
        edge.__dict__["parameter_observations"] = parameter_observations
        edge.__dict__["source_observations"] = [{
            "source": source,
            "source_identity": source_identity or source,
            "timestamp": float(timestamp) if timestamp else None,
            "trust": float(np.clip(source_trust, 0.0, 1.0)),
            "relationship_type": edge.relationship_type,
            "semantic_similarity": pair_metrics["semantic_similarity"],
            "temporal_proximity": pair_metrics["temporal_proximity"],
            "factor_evidence": factor_evidence,
            "parameter_observations": parameter_observations,
            "source_metadata": dict(source_metadata or {}),
            "evidence": [str(quote)[:1000] for quote in evidence
                         if isinstance(quote, str) and quote.strip()][:5],
        }]
        if item.get("is_family"):
            # See BucketIndex.find_connections: a near-miss just below
            # MERGE_THRESHOLD, tagged so apply_to_graph can register it as
            # an "iceberg" family rather than an ordinary cross-doc edge.
            edge.__dict__["is_family"] = True
        return edge

    def assemble_zone(self, item: Dict, timestamp: float,
                       valid_node_ids: set,
                       node_id_map: Dict[str, str]) -> Tuple[TemporalZone, KnowledgeNode]:
        zone_id   = str(uuid.uuid4())[:8]
        contained = [node_id_map.get(nid, nid)
                     for nid in item.get("contained_node_ids", [])
                     if node_id_map.get(nid, nid) in valid_node_ids]

        # A zone is a bounded region in time (Manifest §1: zones are
        # "fields", covers() tests [start_timestamp, end_timestamp]) — an
        # unconditional end_timestamp=inf made every zone permanently
        # cover any future node in its domain, which isn't what "Cold War"
        # or "Manhattan Project" mean. Prefer the LLM's start_year/end_year;
        # fall back to the ingestion timestamp (start) and "still open"
        # (end) only when the model genuinely didn't supply one.
        start_year = item.get("start_year")
        end_year   = item.get("end_year")
        start_ts = _year_to_ts(start_year) if start_year is not None else timestamp
        end_ts   = _year_to_ts(end_year)   if end_year   is not None else float("inf")
        if end_ts < start_ts:
            # Malformed LLM output (end before start) — treat as unbounded
            # rather than silently producing a zone that can never cover
            # anything.
            end_ts = float("inf")

        zone = TemporalZone(
            id=zone_id, description=item.get("description", ""),
            start_timestamp=start_ts, end_timestamp=end_ts,
            domain_focus=item.get("domain_focus", ""),
            zone_multiplier=float(np.clip(item.get("zone_multiplier", 1.0), 1.0, 5.0)),
            contained_node_ids=contained, evidence=item.get("evidence", []),
        )
        zone_node = KnowledgeNode(
            id=zone_id, text=zone.description, node_type="temporal_zone",
            domain=zone.domain_focus, entity_type="temporal_zone",
            timestamp=timestamp, is_temporal_zone=True,
            zone_multiplier=zone.zone_multiplier, contained_node_ids=contained,
        )
        return zone, zone_node


# ── Layer 6: Fuzzy Matcher ────────────────────────────────────────────────────

__all__ = ['INGESTION_ENTITY_TYPES', 'INGESTION_SOURCE_TYPES', '_CHARS_PER_TOKEN', 'estimate_ingestion_char_budget', 'DocumentChunk', 'DocumentChunker', 'IngestionPromptBuilder', 'LLMResponse', 'LLMAnnotator', 'ValidationResult', 'ResponseValidator', 'EdgeSemanticValidator', 'NodeEdgeAssembler']
