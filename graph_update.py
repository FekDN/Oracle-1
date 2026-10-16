"""Ingest local documents and restructure the Oracle graph without neural training."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from corpus_library import DocumentQueue
from oracle1 import (
    BatchDocument,
    BatchIngestionProcessor,
    ConceptClassifier,
    ConvergencePressureField,
    DynamicsOrchestrator,
    GraphConsolidator,
    HashingTextEmbedder,
    IngestionPipeline,
    Oracle1Extended,
    PhantomNodeGenerator,
    RuntimeGraph,
    TopologyInstabilityOrchestrator,
    get_llm_client,
    load_graph_snapshot,
    print_conflict_report,
    save_graph_snapshot,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("oracle1.graph_update")


class _SymbolicModel:
    latent_dim = 512
    neural_enabled = False
    priority_head = None
    edge_pred_head = None


def build_symbolic_system(llm_client: Any) -> Dict[str, Any]:
    graph = RuntimeGraph()
    model = _SymbolicModel()
    pressure = ConvergencePressureField()
    phantom_gen = PhantomNodeGenerator(neural_enabled=False)
    extended = Oracle1Extended(model, graph, phantom_gen=phantom_gen)
    dynamics = DynamicsOrchestrator(
        graph, pressure, phantom_gen, llm_client=llm_client, neural_enabled=False)
    topology = TopologyInstabilityOrchestrator(graph, pressure, phantom_gen=phantom_gen)
    return {
        "graph": graph, "model": model, "extended": extended,
        "dynamics": dynamics, "topology": topology,
        "pressure_field": pressure, "phantom_gen": phantom_gen,
        "embedder": HashingTextEmbedder(), "llm_client": llm_client,
        "concept_classifier": ConceptClassifier(use_bert=False),
    }


def collect_documents(args: argparse.Namespace, graph: RuntimeGraph):
    corpus_path = Path(args.corpus_dir).expanduser().resolve()
    if not corpus_path.is_dir():
        raise ValueError(f"Corpus directory does not exist: {corpus_path}")
    queue = DocumentQueue(Path(args.document_queue))
    selected, queue_stats = queue.select_new(
        corpus_path,
        processed_document_ids=graph.processed_document_ids,
        allow_revisions=args.allow_revisions,
        retry_failed=args.retry_failed_documents,
        document_python=args.document_python or None,
        max_chars=args.max_document_chars,
    )
    documents: List[BatchDocument] = []
    for item in selected:
        try:
            source = str(item.path.relative_to(corpus_path))
        except ValueError:
            source = str(item.path)
        documents.append(BatchDocument(
            text=item.text,
            source=source,
            timestamp=0.0,
            metadata={
                "document_content_hash": item.content_hash,
                "document_warning": item.warning,
                "document_title": (
                    item.title if item.metadata.get("embedded_title_available") else None
                ),
                "document_authors": list(item.authors),
                "document_format": item.path.suffix.lower().lstrip("."),
                "document_container": item.metadata.get("container"),
            },
        ))
        if item.warning:
            logger.warning("[ingest] %s: %s", source, item.warning)
    return corpus_path, queue, queue_stats, documents


def ingest_documents(system: Dict[str, Any], args: argparse.Namespace) -> None:
    graph = system["graph"]
    corpus_path, queue, queue_stats, documents = collect_documents(args, graph)
    logger.info("[ingest] corpus=%s selected=%d queue=%s",
                corpus_path, len(documents), queue_stats)
    if not documents:
        failed_extract = queue_stats.get("failed_extract", 0)
        if failed_extract:
            raise RuntimeError(
                f"No documents could be extracted ({failed_extract} failed); "
                f"see the document queue at {args.document_queue}"
            )
        return
    llm_client = system.get("llm_client")
    if llm_client is None:
        raise RuntimeError(
            "No extraction LLM is configured. Set ORACLE_LLM_BACKEND and its local endpoint.")

    pipeline = IngestionPipeline(
        llm_client=llm_client,
        embedder=system["embedder"],
        graph=graph,
        classifier=system["concept_classifier"],
    )
    processor = BatchIngestionProcessor(
        pipeline, rate_limit_rpm=args.ingestion_rate_limit_rpm,
        max_errors=args.ingestion_max_errors)
    pending_receipts: List[str] = []
    pending_states: Dict[str, str] = {}
    consecutive_errors = 0
    last_attempt = 0.0
    checkpoint_documents = max(1, args.checkpoint_documents)

    def checkpoint() -> None:
        if not pending_receipts:
            return
        graph.processed_document_ids.update(pending_receipts)
        save_graph_snapshot(args.graph_snapshot, system)
        for state in ("ingested", "ingested_empty"):
            hashes = [content_id for content_id in pending_receipts
                      if pending_states[content_id] == state]
            queue.mark(hashes, state, "observations_checkpointed")
        pending_receipts.clear()
        pending_states.clear()

    for index, document in enumerate(documents, start=1):
        elapsed = time.time() - last_attempt
        minimum_delay = 60.0 / max(args.ingestion_rate_limit_rpm, 1)
        if last_attempt and elapsed < minimum_delay:
            time.sleep(minimum_delay - elapsed)
        results = processor.process([document])
        last_attempt = time.time()
        content_id = str(document.metadata.get("document_content_hash", ""))
        if not results:
            queue.mark([content_id], "failed", "ingestion_exception_or_interrupted_batch")
            consecutive_errors += 1
            if consecutive_errors >= args.ingestion_max_errors:
                logger.error("[ingest] stopping after %d consecutive document failures",
                             consecutive_errors)
                break
            continue

        result = results[0]
        if not result.ready_to_commit():
            queue.mark([content_id], "failed", "llm_incomplete_or_no_nodes")
            consecutive_errors += 1
            if consecutive_errors >= args.ingestion_max_errors:
                logger.error("[ingest] stopping after %d consecutive document failures",
                             consecutive_errors)
                break
            continue

        consecutive_errors = 0
        consolidator = GraphConsolidator()
        consolidator.observe_from_ingestion_result(
            result, source_type=getattr(result, "source_type", "unknown"))
        stats = consolidator.apply_to_graph(graph)
        redirects = consolidator._id_redirects

        def node_is_committed(node_id: str) -> bool:
            visited = set()
            while node_id in redirects and node_id not in visited:
                visited.add(node_id)
                node_id = redirects[node_id]
            return graph.get_node(node_id) is not None

        if any(not node_is_committed(node.id) for node in result.nodes):
            queue.mark([content_id], "failed", "result_nodes_not_present_after_consolidation")
            consecutive_errors += 1
            if consecutive_errors >= args.ingestion_max_errors:
                logger.error("[ingest] stopping after %d consecutive document failures",
                             consecutive_errors)
                break
            continue
        zones_added = 0
        seen_zone_ids = set()
        for zone in result.zones:
            if zone.id not in seen_zone_ids and zone.id not in graph.zones:
                zones_added += 1
            seen_zone_ids.add(zone.id)
            graph.add_zone(zone)
        state = "ingested"
        pending_receipts.append(content_id)
        pending_states[content_id] = state
        print_conflict_report(consolidator)
        logger.info("[ingest] %d/%d %s: committed %d nodes, %d edges, %d zones; graph=%d/%d",
                    index, len(documents), result.source, stats.nodes_after,
                    stats.edges_after, zones_added, len(graph.nodes), len(graph.edges))
        if len(pending_receipts) >= checkpoint_documents:
            checkpoint()

    checkpoint()


def run_graph_restructuring(system: Dict[str, Any], steps: int) -> None:
    for index in range(steps):
        timestamp = time.time()
        extended = system["extended"].update_epoch(timestamp, node_latents=None)
        dynamics = system["dynamics"].run_epoch(timestamp)
        topology = system["topology"].run_epoch(timestamp)
        logger.info(
            "[graph] step=%d/%d nodes=%d edges=%d virtual=%d phantoms=%d voids=%d ruptures=%d",
            index + 1, steps, len(system["graph"].nodes), len(system["graph"].edges),
            dynamics.get("total_virtual_nodes", 0), extended.get("n_phantoms", 0),
            dynamics.get("new_structural_voids", 0), topology.get("rupture_count", 0),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symbolic Oracle-1 graph update")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--document_queue", required=True)
    parser.add_argument("--graph_snapshot", required=True)
    parser.add_argument("--document_python", default="")
    parser.add_argument("--max_document_chars", type=int, default=0)
    parser.add_argument("--allow_revisions", action="store_true")
    parser.add_argument("--retry_failed_documents", action="store_true")
    parser.add_argument("--graph_steps", type=int, default=2)
    parser.add_argument("--ingestion_rate_limit_rpm", type=int, default=60)
    parser.add_argument("--ingestion_max_errors", type=int, default=5)
    parser.add_argument("--checkpoint_documents", type=int, default=10)
    parser.add_argument("--disable_edge_validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_document_chars < 0:
        raise SystemExit("--max_document_chars must be zero or positive")
    if args.disable_edge_validation:
        from oracle1 import CONFIG
        CONFIG["edge_validation_enabled"] = False

    try:
        llm_client = get_llm_client()
        system = build_symbolic_system(llm_client)
        if Path(args.graph_snapshot).is_file():
            load_graph_snapshot(args.graph_snapshot, system)
        ingest_documents(system, args)
        run_graph_restructuring(system, max(0, args.graph_steps))
        save_graph_snapshot(args.graph_snapshot, system)
        logger.info("Graph update complete: %d nodes / %d edges",
                    len(system["graph"].nodes), len(system["graph"].edges))
    except Exception:
        logger.exception("Symbolic graph update failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
