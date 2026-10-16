"""LLM client implementations and client selection."""
from __future__ import annotations

from ._shared import *
from .config import CONFIG

# ── Helpers for LLM initialization based on CONFIG ──────────────────────────
class LocalLLMClient:
    """Small common interface for a locally hosted text model.

    Oracle uses an LLM only as a *sensor* that turns a document into a
    structured observation.  The graph, the provenance and the forecast are
    deliberately kept outside the LLM.  Keeping this interface tiny also makes
    it possible to use Ollama, vLLM, LM Studio or another OpenAI-compatible
    local server without adding a provider SDK to the core package.
    """
    def __init__(self, model_name: str):
        self.model = model_name

    def complete(self, prompt: str) -> str:
        raise NotImplementedError(
            f"Local LLM backend for '{self.model}' is not configured. "
            "Set ORACLE_LLM_BACKEND=openai_compatible together with "
            "ORACLE_LLM_ENDPOINT, or explicitly select the heuristic "
            "smoke-test backend."
        )

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.model}>"


class OpenAICompatibleLLMClient(LocalLLMClient):
    """Client for an OpenAI-compatible *local* chat-completions endpoint.

    ``ORACLE_LLM_ENDPOINT`` is the complete endpoint URL, for example
    ``http://127.0.0.1:11434/v1/chat/completions`` for an OpenAI-compatible
    Ollama proxy, or the equivalent URL exposed by vLLM / LM Studio.  An API
    key is optional and is read only from the process environment, never from
    a graph snapshot or run configuration.

    Ollama specifically: its OpenAI-compatible /v1/chat/completions endpoint
    has no way to set the context window per request — Ollama's own docs say
    so outright ("The OpenAI API does not have a way of setting the context
    size for a model"), and a nested options.num_ctx sent there is silently
    dropped (verified against Ollama v0.30.11, tracked upstream as issue
    #11409) — every call quietly runs at Ollama's own default instead of
    whatever this pipeline computed. Ollama's NATIVE /api/chat endpoint DOES
    honor options.num_ctx, and this client's response parsing already
    accepts that endpoint's reply shape as a fallback (see below) — so for
    Ollama, point ORACLE_LLM_ENDPOINT at .../api/chat, not .../v1/chat/
    completions, and pass num_ctx via extra_options (get_llm_client() wires
    this from ORACLE_LLM_NUM_CTX). Baking num_ctx into a Modelfile
    (``ollama create`` with ``PARAMETER num_ctx <n>``) is the belt-and-
    suspenders alternative if you'd rather keep hitting /v1/.
    """
    def __init__(self, model_name: str, endpoint: str,
                 api_key: Optional[str] = None, timeout_s: Optional[float] = None,
                 extra_options: Optional[Dict[str, Any]] = None):
        super().__init__(model_name)
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.extra_options = extra_options or {}

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            # Ollama's native /api/chat streams by default; an OpenAI-
            # compatible server's own default is already non-streaming, so
            # this is a no-op there and load-bearing only for Ollama.
            # complete()'s response parsing below assumes one full body.
            "stream": False,
        }
        if self.extra_options:
            # Ollama-native generation options (num_ctx, num_predict, ...) —
            # see the class docstring: only takes effect against Ollama's
            # /api/chat, not /v1/chat/completions.
            payload["options"] = dict(self.extra_options)
        # Strict JSON mode is supported by many OpenAI-compatible servers but
        # not all of them.  It is opt-in so a plain Ollama/vLLM deployment
        # remains usable; LLMAnnotator still validates and parses the response.
        if os.environ.get("ORACLE_LLM_JSON_MODE", "").lower() in {"1", "true", "strict"}:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.endpoint, data=body,
                                         headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Local LLM HTTP {exc.code} at {self.endpoint}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach local LLM at {self.endpoint}: {exc.reason}"
            ) from exc

        try:
            data = json.loads(raw)
            choices = data.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
            # Ollama's native /api/chat response shape, supported as a
            # convenience when that endpoint is supplied directly.
            message = data.get("message") or {}
            if isinstance(message.get("content"), str):
                return message["content"]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Local LLM returned invalid JSON") from exc
        raise RuntimeError("Local LLM response contains no assistant content")


class HeuristicLLMClient(LocalLLMClient):
    """Deterministic ingestion fallback for wiring and smoke tests only.

    This is intentionally conservative and visibly labelled as heuristic.  It
    extracts sentence-level candidates so the complete graph/training/UI path
    can be exercised offline, but it is not a replacement for evidence-aware
    extraction by a local model and must never be interpreted as a forecast.
    """
    _STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "into", "are",
        "was", "were", "has", "have", "will", "using", "used", "also", "than",
        "как", "что", "это", "для", "или", "при", "также", "между", "которые",
    }

    def __init__(self):
        super().__init__("heuristic-smoke-test")

    @staticmethod
    def _slug(text: str, idx: int) -> str:
        words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text.lower())
        useful = [word for word in words if len(word) > 2][:5]
        stem = "_".join(useful) or "concept"
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]
        return f"{stem[:28]}_{idx}_{digest}"

    @classmethod
    def _domain(cls, text: str) -> str:
        lowered = text.lower()
        rules = (
            ("artificial_intelligence", ("neural", "ai", "machine learning", "модель", "нейросет")),
            ("materials_science", ("material", "battery", "superconduct", "alloy", "материал", "батаре")),
            ("physics", ("photon", "quantum", "laser", "optical", "квант", "лазер", "фотон")),
            ("biology", ("protein", "cell", "gene", "crispr", "биолог", "ген", "клетк")),
            ("energy", ("energy", "fusion", "solar", "grid", "энерг", "термояд", "топлив")),
        )
        for domain, needles in rules:
            if any(needle in lowered for needle in needles):
                return domain
        return "technology"

    @staticmethod
    def _score(text: str, positive: Tuple[str, ...], baseline: float = 4.0) -> float:
        lowered = text.lower()
        return min(9.0, baseline + 1.25 * sum(term in lowered for term in positive))

    @classmethod
    def _nodes_from_text(cls, text: str, domain: str,
                          with_physical_axes: bool = False) -> List[Dict[str, Any]]:
        """Shared node-building logic behind both _entity_response (old
        3-call mode) and _unified_response (unified mode) — kept as one
        implementation so the two modes' heuristic fallback stays
        consistent rather than drifting apart the way the LLM prompts
        themselves briefly did (see estimate_ingestion_char_budget's
        docstring for that episode).
        """
        sentences = [
            re.sub(r"\s+", " ", sentence).strip(" -•\t")
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        ]
        candidates = [sentence for sentence in sentences if len(sentence) >= 35]
        if not candidates and text:
            candidates = [text[:500]]
        unique: List[str] = []
        seen: Set[str] = set()
        for sentence in candidates:
            normal = re.sub(r"\W+", " ", sentence.lower()).strip()
            if normal and normal not in seen:
                seen.add(normal)
                unique.append(sentence)
            if len(unique) >= 8:
                break
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
        year = int(year_match.group(1)) if year_match else None

        nodes: List[Dict[str, Any]] = []
        for idx, sentence in enumerate(unique):
            words = sentence.split()
            node: Dict[str, Any] = {
                "id": cls._slug(sentence, idx),
                "text": " ".join(words[:18]).rstrip(" ,;:"),
                "full_context": sentence[:600],
                "entity_type": "physical_limit" if any(
                    term in sentence.lower() for term in
                    ("limit", "barrier", "bottleneck", "constraint", "огранич", "барьер")
                ) else "research_frontier",
                "origin_kind": "fundamental" if any(
                    term in sentence.lower() for term in
                    ("law", "constant", "mechanism", "property", "закон", "свойство", "механизм")
                ) else "combinatorial",
                "domain": cls._domain(sentence) if cls._domain(sentence) != "technology" else domain,
                "node_type": "research",
                "scientific_score": cls._score(sentence, ("study", "experiment", "research", "paper", "исследован", "эксперимент")),
                "investment_score": cls._score(sentence, ("investment", "funding", "capital", "market", "инвест", "финанс"), 3.0),
                "social_score": cls._score(sentence, ("public", "adoption", "society", "consumer", "обще", "рынок"), 3.0),
                "maturity_score": cls._score(sentence, ("prototype", "demonstrat", "deploy", "pilot", "прототип", "демонстр"), 3.5),
                "strategic_value": cls._score(sentence, ("critical", "strategic", "security", "важн", "стратег"), 4.0),
                "efficiency_plateau": 4.0,
                "dual_use_risk": 2.0,
                "legal_risk_score": 2.0,
                "export_control_risk": 2.0,
                "publication_year": year,
                "solves_limitations": [],
                "requires_node_ids": [],
                # A verbatim, already-whitespace-normalised substring of the
                # SAME text this was split from — passes ResponseValidator's
                # evidence-grounding check the same way a real quote would,
                # since that check normalises both sides identically before
                # comparing.
                "evidence": [sentence[:160]],
                "secondary_contexts": [],
            }
            if with_physical_axes:
                node["physical_axes"] = {
                    "resource_intensity": 0.5, "scale_feasibility": 0.5,
                    "constraint_proximity": 0.5, "reversibility": 0.5,
                    "parallel_deployability": 0.5, "environmental_coupling": 0.5,
                    "cross_scale_stability": 0.5, "synthesis_complexity": 0.5,
                    "longevity": 0.5, "cascadability": 0.5,
                    "rationale": "heuristic mode — neutral midpoint, not a measurement",
                }
            nodes.append(node)
        return nodes, year, year_match

    @classmethod
    def _entity_response(cls, prompt: str) -> Dict[str, Any]:
        match = re.search(r"\nTEXT:\s*(.*?)\n\nEXTRACTION RULES:", prompt, re.S)
        text = match.group(1).strip() if match else ""
        domain = cls._domain(text)
        nodes, year, year_match = cls._nodes_from_text(text, domain)
        # This is only a deterministic smoke-test sensor.  It still follows
        # the same contract as a real LLM: metadata comes from document text,
        # never from path/mtime, and absent evidence stays absent.
        return {
            "document_metadata": {
                "title": None, "authors": [], "source_type": "unknown",
                "publication_date": str(year) if year is not None else None,
                "date_evidence": year_match.group(0) if year_match else None,
            },
            "domain": domain, "nodes": nodes,
        }

    @classmethod
    def _unified_response(cls, prompt: str) -> Dict[str, Any]:
        """Heuristic stand-in for the unified Pass A+B+C prompt: same node
        extraction as _entity_response, plus inline physical_axes on each
        node (neutral 0.5 midpoints — a heuristic has no basis to claim a
        real physical measurement) and simple adjacency edges between
        consecutive extracted nodes, so the unified code path can be
        smoke-tested exactly as the old 3-call path already was.
        """
        match = re.search(
            r"\nTEXT:\s*(.*?)\n═+\nRULE THAT OVERRIDES", prompt, re.S)
        text = match.group(1).strip() if match else ""
        domain = cls._domain(text)
        nodes, year, year_match = cls._nodes_from_text(text, domain, with_physical_axes=True)
        edges = [
            {
                "source": a["id"], "target": b["id"], "relationship_type": "related",
                "semantic_similarity": 0.35, "temporal_proximity": 0.5,
                "limitation_resolution": 0.0, "citation_link": 0.0,
                "investment_correlation": 0.0, "social_correlation": 0.0,
                "inhibitory_force": 0.0, "confidence": 0.2,
                "evidence": [a["evidence"][0]],
            }
            for a, b in zip(nodes, nodes[1:])
        ]
        return {
            "document_metadata": {
                "title": None, "authors": [], "source_type": "unknown",
                "publication_date": str(year) if year is not None else None,
                "date_evidence": year_match.group(0) if year_match else None,
            },
            "domain": domain, "nodes": nodes,
            "edges": edges, "temporal_zones": [],
        }

    @staticmethod
    def _physical_response(prompt: str) -> Dict[str, Any]:
        ids = re.findall(r"^\s*([\w-]+):.*?\[[^\]]*\]", prompt, re.M)
        axes = {
            "resource_intensity": 0.5, "scale_feasibility": 0.5,
            "constraint_proximity": 0.5, "reversibility": 0.5,
            "parallel_deployability": 0.5, "environmental_coupling": 0.5,
            "cross_scale_stability": 0.5, "synthesis_complexity": 0.5,
            "longevity": 0.5, "cascadability": 0.5,
        }
        return {"physical_scores": {node_id: dict(axes) for node_id in ids}}

    @staticmethod
    def _edge_response(prompt: str) -> Dict[str, Any]:
        ids = re.findall(r"^\s*([\w-]+):", prompt.split("ORIGINAL TEXT:", 1)[0], re.M)
        edges: List[Dict[str, Any]] = []
        for source, target in zip(ids, ids[1:]):
            edges.append({
                "source": source, "target": target, "relationship_type": "related",
                "semantic_similarity": 0.35, "temporal_proximity": 0.5,
                "limitation_resolution": 0.0, "citation_link": 0.0,
                "investment_correlation": 0.0, "social_correlation": 0.0,
                "inhibitory_force": 0.0, "confidence": 0.2,
                "evidence": ["Heuristic adjacency; requires review"],
            })
        return {"edges": edges, "temporal_zones": []}

    @staticmethod
    def _pass_d_batch_response(prompt: str) -> Dict[str, Any]:
        """Heuristic stand-in for the batched Pass D prompt: echoes back
        every item_id it can find with a neutral "no match" verdict — a
        heuristic has no embedding-based candidate ranking to arbitrate
        with, so the honest answer is "cannot decide", not a fabricated
        match. Returning the item_id-keyed shape (rather than an empty
        {}) still lets the real reconciliation logic in _run_pass_d_batch
        run its normal course during a smoke test instead of every item
        showing up as "no result — LLM failure".
        """
        item_ids = re.findall(r'ITEM item_id="([^"]+)"', prompt)
        return {"results": [
            {"item_id": iid, "match": None, "confidence": 0.0,
             "relationship_type": "related", "semantic_similarity": 0.0,
             "temporal_proximity": 0.0, "plausibility": 1.0, "verdict": "keep",
             "rationale": "heuristic mode — no arbitration performed"}
            for iid in item_ids
        ]}

    @classmethod
    def _classification_response(cls, prompt: str) -> Dict[str, Any]:
        """Return a deterministic, explicitly heuristic classifier response.

        ConceptClassifier uses its LLM fallback whenever no trained model is
        available.  Supporting this prompt is necessary for an offline
        ingestion run to exercise canonical-key assembly and downstream graph
        logic instead of silently falling back to the all-unknown key.
        """
        term_match = re.search(r"^TERM:\s*(.*?)\s*$", prompt, re.M)
        context_match = re.search(
            r"^CONTEXT:\s*(.*?)\nSOURCE TYPE:", prompt, re.M | re.S)
        term = term_match.group(1).strip() if term_match else ""
        context = context_match.group(1).strip() if context_match else ""
        corpus = f"{term} {context}".lower()

        domains_match = re.search(
            r"Available domains:\s*(.*?)\nAvailable groups:", prompt, re.S)
        groups_match = re.search(
            r"Available groups:\s*(.*?)\nAvailable entity types:", prompt, re.S)
        types_match = re.search(
            r"Available entity types:\s*(.*?)\n\nRules:", prompt, re.S)
        domains = [v.strip() for v in domains_match.group(1).split(",") if v.strip()] if domains_match else ["unknown"]
        groups = [v.strip() for v in groups_match.group(1).split(",") if v.strip()] if groups_match else ["unknown"]
        entity_types = [v.strip() for v in types_match.group(1).split(",") if v.strip()] if types_match else ["unknown"]

        def choose(preferred: List[str], available: List[str]) -> str:
            normalized = {value.lower(): value for value in available}
            return next((normalized[value] for value in preferred
                         if value.lower() in normalized),
                        normalized.get("unknown", available[0] if available else "unknown"))

        hinted_domain = cls._domain(corpus)
        if any(word in corpus for word in ("tooth", "dental", "periodontal", "endodont", "oral", "patient", "clinical")):
            hinted_domain = "medicine"
        domain = choose([hinted_domain, "biology", "medicine", "biotechnology", "unknown"], domains)

        if any(word in corpus for word in ("barrier", "limitation", "constraint", "bottleneck", "limit")):
            group = choose(["barrier_constraint", "biological_system", "applied_technology"], groups)
            entity_type = choose(["physical_limit", "economic_barrier", "research_frontier"], entity_types)
        elif any(word in corpus for word in ("cell", "tissue", "stem", "regenerat", "gene", "protein")):
            group = choose(["biological_system", "applied_technology"], groups)
            entity_type = choose(["research_frontier", "experimental_method"], entity_types)
        else:
            group = choose(["applied_technology", "fundamental_principle", "theoretical_framework"], groups)
            entity_type = choose(["research_frontier", "experimental_method", "functional_spec"], entity_types)

        return {
            "domain": domain,
            "group": group,
            "entity_type": entity_type,
            "confidence": 0.55,
            "reasoning": "Heuristic offline classification for pipeline testing; not an LLM judgment.",
            "aliases": [],
        }

    @staticmethod
    def _edge_validation_response(prompt: str) -> Dict[str, Any]:
        """Return a complete neutral plausibility judgment for every edge."""
        edge_ids = re.findall(r"--- Edge \d+ \(id=([^)]+)\) ---", prompt)
        return {"results": [
            {"edge_id": edge_id, "plausibility": 0.7, "verdict": "keep",
             "rationale": "Heuristic smoke-test judgment; semantic validation is simulated."}
            for edge_id in edge_ids
        ]}

    def complete(self, prompt: str) -> str:
        if "analysing one document for a knowledge graph" in prompt:
            data = self._unified_response(prompt)
        elif "TASK: Extract all significant concepts" in prompt:
            data = self._entity_response(prompt)
        elif "For each node, rate 10 constraint axes" in prompt:
            data = self._physical_response(prompt)
        elif "TASK: Identify relationships between these nodes" in prompt:
            data = self._edge_response(prompt)
        elif "merging entity extractions" in prompt:
            data = {"merge_groups": [], "cross_chunk_edges": []}
        elif "Each ITEM below is a new node's reference" in prompt:
            data = self._pass_d_batch_response(prompt)
        elif "You are a knowledge graph classifier." in prompt:
            data = self._classification_response(prompt)
        elif "You are an expert knowledge-graph curator validating proposed edges" in prompt:
            data = self._edge_validation_response(prompt)
        else:
            # Includes optional edge validation.  Returning an empty object is
            # safe: validation then retains no unreviewed heuristic judgement.
            data = {}
        return json.dumps(data, ensure_ascii=False)


def _check_ollama_context_window(endpoint: str, model_name: str, expected_num_ctx: int) -> None:
    """
    Best-effort sanity check for the single most common way an Ollama-backed
    run silently gets worse results than expected: every context-window
    calculation in this pipeline (estimate_ingestion_char_budget and
    everything downstream of it) is meaningless if Ollama is actually
    running the model at its own default context instead of the num_ctx
    we asked for — and it fails SILENTLY (truncated input, not an error).

    Queries Ollama's POST /api/show for `model_name` and checks:
      - `parameters` (a newline-separated string) for an explicit
        "num_ctx <n>" line — this is what a Modelfile baked in, if any.
      - `model_info["<family>.context_length"]` (key name varies by model
        architecture, e.g. "qwen2.context_length") — the model's own
        architectural ceiling, independent of what's configured.
    Logs a prominent warning (never raises) when:
      - /api/show is unreachable or this doesn't look like an Ollama
        server — the check just can't run, said once at debug level.
      - no explicit num_ctx is baked in AND the endpoint is /v1/... — that
        combination means Ollama is DEFINITELY not honoring
        expected_num_ctx (see OpenAICompatibleLLMClient's docstring).
      - a baked-in num_ctx doesn't match expected_num_ctx.
      - expected_num_ctx exceeds the model's own architectural ceiling.
    """
    m = re.match(r"^(https?://[^/]+)", endpoint)
    if not m:
        return
    base = m.group(1)
    try:
        req = urllib.request.Request(
            f"{base}/api/show",
            data=json.dumps({"model": model_name}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            info = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug(f"[LLM] could not query {base}/api/show to verify num_ctx: {exc}")
        return

    baked_in = None
    for line in (info.get("parameters") or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "num_ctx":
            try:
                baked_in = int(parts[1])
            except ValueError:
                pass

    ceiling = None
    for key, val in (info.get("model_info") or {}).items():
        if key.endswith(".context_length"):
            try:
                ceiling = int(val)
            except (TypeError, ValueError):
                pass

    if baked_in is None and "/v1" in endpoint:
        logger.warning(
            f"[LLM] '{model_name}' has no num_ctx baked into its Modelfile, and "
            f"ORACLE_LLM_ENDPOINT ({endpoint}) is the OpenAI-compatible /v1 "
            f"endpoint, which Ollama does NOT let you set context size on "
            f"per request (Ollama's own docs; also tracked as ollama/ollama#11409). "
            f"Every ingestion call will silently run at Ollama's own default "
            f"context instead of the {expected_num_ctx} tokens this pipeline "
            f"computed chunk sizes for — expect large documents to be silently "
            f"truncated, not an error. Fix: point ORACLE_LLM_ENDPOINT at "
            f"'{base}/api/chat' (native, honors options.num_ctx — already sent "
            f"via ORACLE_LLM_NUM_CTX) or bake it in with "
            f"'ollama create {model_name} -f Modelfile' (FROM {model_name} / "
            f"PARAMETER num_ctx {expected_num_ctx})."
        )
    elif baked_in is not None and baked_in != expected_num_ctx:
        logger.warning(
            f"[LLM] '{model_name}' has num_ctx={baked_in} baked into its "
            f"Modelfile, but this run expects {expected_num_ctx} (from "
            f"ORACLE_LLM_NUM_CTX / CONFIG['llm_context_windows']). Chunk "
            f"sizing follows the CONFIG value, not what Ollama will actually "
            f"run with — the two need to match."
        )
    if ceiling is not None and expected_num_ctx > ceiling:
        logger.warning(
            f"[LLM] requested context {expected_num_ctx} exceeds '{model_name}'s "
            f"own architectural ceiling ({ceiling}) reported by Ollama — this "
            f"will not work regardless of num_ctx configuration."
        )


def get_llm_client(model_name: Optional[str] = None):
    """Return a configured local extraction client, or explain how to enable it.

    Environment configuration is intentionally process-local so credentials
    never end up in snapshots/checkpoints:

    * ``ORACLE_LLM_BACKEND=openai_compatible`` and ``ORACLE_LLM_ENDPOINT``
      enable a real local OpenAI-compatible server.
    * ``ORACLE_LLM_BACKEND=heuristic`` enables only the labelled offline
      smoke-test extractor.
    * with no backend configured, ingestion is disabled rather than silently
      fabricating observations.
    * ``ORACLE_LLM_NUM_CTX``, if set, is sent as Ollama's native
      ``options.num_ctx`` on every request — see
      OpenAICompatibleLLMClient's docstring for why this only takes
      effect against Ollama's /api/chat, not /v1/chat/completions, and
      why get_llm_client() then verifies the server actually honors it
      rather than trusting the setting silently.
    """
    backend = os.environ.get("ORACLE_LLM_BACKEND", "").strip().lower()
    name = (model_name or os.environ.get("ORACLE_LLM_MODEL")
            or CONFIG.get("llm_default_model"))
    if backend == "heuristic":
        logger.warning("[LLM] heuristic smoke-test extractor enabled; results are not evidence")
        return HeuristicLLMClient()
    if backend in {"openai_compatible", "openai-compatible", "openai"}:
        endpoint = os.environ.get("ORACLE_LLM_ENDPOINT", "").strip()
        if not endpoint:
            raise ValueError(
                "ORACLE_LLM_BACKEND=openai_compatible requires ORACLE_LLM_ENDPOINT "
                "(for example http://127.0.0.1:11434/v1/chat/completions)"
            )
        timeout_raw = os.environ.get("ORACLE_LLM_TIMEOUT_S", "").strip()
        if not timeout_raw or timeout_raw.lower() in {"off", "none", "0"}:
            timeout_s = None
        else:
            try:
                timeout_s = float(timeout_raw)
            except ValueError as exc:
                raise ValueError("ORACLE_LLM_TIMEOUT_S must be a number of seconds or disabled") from exc
            if not math.isfinite(timeout_s) or timeout_s <= 0:
                raise ValueError("ORACLE_LLM_TIMEOUT_S must be positive when enabled")
        extra_options: Dict[str, Any] = {}
        num_ctx_env = os.environ.get("ORACLE_LLM_NUM_CTX", "").strip()
        if num_ctx_env:
            try:
                extra_options["num_ctx"] = int(num_ctx_env)
            except ValueError:
                raise ValueError(f"ORACLE_LLM_NUM_CTX must be an integer, got {num_ctx_env!r}")
            _check_ollama_context_window(endpoint, name, extra_options["num_ctx"])
        return OpenAICompatibleLLMClient(
            name, endpoint, os.environ.get("ORACLE_LLM_API_KEY") or None, timeout_s,
            extra_options=extra_options or None)
    if backend in {"", "disabled", "none"}:
        raise ValueError(
            "No local LLM extraction backend configured. Set ORACLE_LLM_BACKEND="
            "openai_compatible with ORACLE_LLM_ENDPOINT, or explicitly use "
            "ORACLE_LLM_BACKEND=heuristic for an offline smoke test."
        )
    raise ValueError(f"Unknown ORACLE_LLM_BACKEND={backend!r}")

__all__ = ['LocalLLMClient', 'OpenAICompatibleLLMClient', 'HeuristicLLMClient', '_check_ollama_context_window', 'get_llm_client']
