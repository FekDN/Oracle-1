# Oracle-1: Concept and Current Prototype

> **Status and scope:** This README describes both the intended Oracle-1 design and the current Python prototype. Statements about the target system are design goals unless the text identifies them as implemented. The repository currently provides a local, single-process research prototype; it is not a live service covering humanity's knowledge. Forecast scores are heuristic diagnostics, not calibrated probabilities.

### Local Dashboard Setup

Run `python dashboard.py` with 64-bit Python 3.11 or 3.12. On first launch, the dashboard creates a project-local `.venv`, installs the pinned CPU dependencies from `requirements.txt`, checks imports and `pip check`, then relaunches inside that environment. Internet access is required for the initial install; the CPU PyTorch package is about 200 MB. The system Python environment is not modified.

The managed environment intentionally excludes optional compiled PyG extensions (`pyg-lib`, `torch-scatter`, and related wheels) because they must match the exact PyTorch build. Neural training runs on CPU; use the `.venv` interpreter in VS Code (`.venv\Scripts\python.exe`).

### Why Oracle-1 Is Needed

The technological future of humanity develops chaotically — we cannot see which ideas and discoveries are already near convergence.
Most breakthroughs occur by chance, even though the necessary knowledge already exists across multiple domains.
We continuously miss weak signals and tipping points that determine the emergence of new technologies.
Humanity expends enormous resources on directions that prove dead ends or are significantly delayed.
A critical tool is needed to help researchers examine technological development systematically. Whether Oracle-1 can predict it accurately remains an empirical question.

### What Oracle-1 Is

Oracle-1 is a research prototype and proposed system for representing knowledge as a time-aware, multidimensional graph and exploring whether its structure can reveal candidate convergences.
The long-term goal is to connect scientific articles, patents, investment records, and social signals in a map of possible technological trajectories. The current repository does not yet collect millions of records or operate as a real-time global service.

### How Oracle-1 Works

The target design calls for continuously collecting data from many sources. The current prototype ingests local documents through an explicit queue and a configured LLM; it does not include hundreds of live source connectors.

Each graph node represents a concept or other classified knowledge entity, not its source document. One document can contribute several nodes, and multiple documents can contribute observations to the same canonical node. A node can be represented as a point in time or, when dated observations support it, as an interval with a density profile. The profile is rebuilt from recorded source-time observations; it does not establish that the sources are independent studies. Temporal Nonlinearity and `ConvergenceAccelerometer` use this profile as a signal, not as proof of independent replication.

Edges store separate components such as semantic similarity, temporal proximity, limitation resolution, citation, investment, social correlation, and inhibitory force. The graph can combine these components into a traversal weight. The availability and evidential basis of a component depend on the ingested data and the path that created the edge; a stored edge or score is not automatically a verified factual claim. Some edges are explicitly synthetic structural links, described below.

The design aims to identify candidate emerging clusters and missing connections. The code includes recursive scenario generation, virtual-node synthesis, and intervention analysis, but it does not run thousands of competing autonomous graph variants or continuously calibrate itself against live outcomes. Historical training objectives and backtesting utilities exist; validated predictive accuracy has not been established.

### Practical Applications

The intended applications include exploring candidate trajectories over a chosen time horizon, finding possible bottlenecks, comparing cross-domain paths, and simulating interventions. Current scores and rankings are heuristic; they should not be read as calibrated probabilities, reliable timing estimates, or proven recommendations.
The long-term system is intended to show which edges may converge on a missing or incomplete node and what parameters a candidate node might need. The prototype can produce candidate gaps and scenarios, but these remain model outputs for human review; they are not demonstrated inventions or reliable forecasts.

### Seven Foundational Principles (Design Assumptions)

1. **Convergence Determinism (Principle of Inevitability)**
   Breakthroughs are not acts of random genius but inevitable points where existing knowledge converges. **Assumption:** if one person does not invent a technology, another inevitably would, because the knowledge graph has already “collapsed” at that point. This assumption governs **combinatorial** nodes — engineered recombinations of already-known building blocks, which is what almost all products and techniques are, and whose timing tracks investment and convergence pressure the way the rest of this section describes. A smaller class of nodes is **fundamental**: something nature already contains and humans only discover as measurement or theory improves — a constant, a material property, a mechanism. A fundamental node's timing is not a convergence event and does not respond to funding the way a combinatorial one does (the Intervention Optimizer dampens investment effect on these nodes accordingly, see below); once discovered, though, it tends to move a whole cluster of downstream combinatorial technologies at once rather than one neighbor at a time — which the cascade wave models as a conductance boost specifically on hops that cross a domain boundary.

2. **Topological Pressure (Theory of Vacuums)**
   Missing links in a technological chain create a “structural vacuum” that attracts energy (money, talent, ideas). **Assumption:** characteristics of not-yet-existent inventions (**Phantom Nodes**) can be predicted by measuring “tension” around the empty space in the graph.

3. **Physical Substrate (Sieve of Reality)**
   Ideas exist in language but are realized in matter. **Assumption:** every technology has a “physical mass” (energy intensity, resource rarity, synthesis complexity), enabling the filtering of fantasy from feasible reality based on fundamental laws of nature.

4. **Institutional Friction (Theory of Resistance)**
   No breakthrough technology emerges in a vacuum; it always threatens someone’s resources. **Assumption:** resistance from established industries (**Inhibitory Edges**) is measurable and predictably slows progress. Ignoring this friction renders predictions inaccurate. This resistance is not only industrial: peer review, grant committees, and publication metrics are themselves gatekeeping mechanisms that protect careers built on the current consensus, which is why unconventional hypotheses tend to be underfunded long before any data disproves them. The same conservatism also filters out genuine noise and pseudoscience, so Oracle-1 treats it as a real cost to be weighed and overcome, not an illegitimate force to be dismissed outright.

5. **Temporal Nonlinearity (Temporal Zones)**
   Time in science flows at varying speeds. **Assumption:** there exist “acceleration zones” (arms races, pandemics) where event density is 5–10 times higher than normal. Oracle detects these zones to identify where the future will arrive faster.

6. **Equifinality (Multiple Paths)**
   Multiple independent trajectories can lead to the same goal in the graph. **Assumption:** the more alternative paths lead to a breakthrough (redundancy), the higher the probability of its realization, even if 90% of attempts are blocked.

7. **Ontological Gap (Phase Transition)**
   Knowledge development is not always linear. **Assumption:** when accumulated tension in the graph exceeds a threshold, a “rupture” occurs — old physical and economic constraints vanish, and the knowledge system instantly restructures for a new reality (e.g., the world before and after the Internet).

**Note on single-domain data.** A single-domain dataset — for example, papers about cold fusion — is useful for testing document extraction, graph construction, and within-domain dynamics. It cannot evaluate the cross-domain Topological Pressure hypothesis on its own. The design expects that investment, publication, and social-attention signals from multiple relevant fields may help rank candidate gaps. For example, an investment imbalance could motivate checking whether a cost, materials, or confinement constraint is missing. The imbalance does not establish that a particular gap exists or that a predicted node would resolve it. A broader cross-domain corpus is necessary to evaluate this hypothesis, but is not sufficient evidence that it is correct.

**Note on epistemic plugs and cultural sacralization.** Institutional Friction (Principle 4) is not only exerted by incumbents defending a market; it is also exerted by placeholder variables a theory carries to survive contact with anomalous data instead of revising the underlying model — dark matter reconciling galaxy rotation curves with visible mass, endosymbiosis bridging the jump from prokaryote to eukaryote, or the unproven assumption that mathematics derived inside a human brain describes the rest of the universe. Such a plug begins as an honest admission of a gap. Once it crosses from working hypothesis into textbooks, popular science, and film, however, it accumulates a graph density that looks like convergence but is largely the same original claim being repeated rather than independently re-derived — while the institutional apparatus around it (funded departments, built instruments, defended dissertations) grows to match. This is a failure mode Oracle-1's weighting still has to guard against, not one already solved: as implemented, an idea-node's velocity is a log-compressed count of recent mentions weighted by source credibility, and its source diversity counts how many source *categories* (forum, press, journal…) are represented — neither checks whether those mentions are independent observations or the same claim echoing through one category, so a viral repost pattern and genuine independent replication are not yet distinguished. When a plug's unexplained anomalies outgrow the evidence actually behind it, that imbalance is itself worth surfacing — not as proof the plug is false, but as a candidate for the restructuring described in Principle 7. Oracle-1 is not exempt from the friction it measures: a prediction that displaces a culturally entrenched node will accumulate Inhibitory Edges from the same institutions and narratives it is scoring, most visibly as its methodology gets dismissed as an unauditable “black box” rather than engaged with on the merits. A slow adoption curve for a given prediction therefore says something about the resistance around that node, not about whether the prediction was structurally correct.

**Note on physical grounding.** As implemented, `PhysicalSubstrateEncoder` does not check a node against a known physical bound — it asks an LLM to rate ten abstract axes (resource intensity, synthesis complexity, and so on) from a text description, and `feasibility_score` is a weighted combination of three of those axes. This is a plausibility heuristic, not a verified physics check: nothing in the current pipeline does dimensional analysis, looks up a table of physical constants, or tests a claim against a named limit such as Carnot efficiency, the Landauer bound, or Shannon capacity. A confidently-worded but physically absurd node can score as feasible today for the same reason an LLM can hallucinate confidently elsewhere — the sieve is only as sharp as the language model doing the rating. Grounding `constraint_proximity` and `resource_intensity` in retrieved numeric bounds rather than an unaided LLM estimate is future work, not a solved part of the sieve.

**Note on the rupture threshold.** `PhaseTransitionDetector` currently classifies a rupture locally: a single node's pressure, conflict, and connectivity-entropy each crossing a fixed, hand-tuned threshold (`PHASE_THRESHOLDS`). This is a per-node trigger, not a structural one — it says nothing about whether the graph *as a whole* has reorganized. A more principled version, not yet implemented, would track the spectral gap of the graph Laplacian in the node's neighborhood: a sharp drop in that gap is the standard percolation-theory signature of previously separate clusters fusing into one, and would let a rupture be read off an actual topology change rather than inferred from three scalar thresholds crossing together.

**Note on coordinated low-trust clusters.** Source credibility currently enters the graph as a static per-category multiplier (`ingestion_source_trust`: Nature at 0.95, a blog at 0.30) applied once at ingestion — it says nothing about how a cluster of low-trust sources cites itself. A large body of mutually-citing but institutionally disconnected material can still accumulate velocity and apparent cross-source corroboration the same way genuine convergence does, because nothing currently checks whether a cluster's internal density is backed by any path back to a trusted root node. A TrustRank-style propagation — trust flowing outward from verified sources through the citation graph, rather than assigned once per document — would let such a cluster be flagged or quarantined instead of scored as a breakthrough. Not yet built.

**Conclusion:** Oracle-1 posits that **the future is a geometric property of the present**. It can be computed rather than guessed by identifying points of maximum tension in the current knowledge structure.

### How Oracle Differs from LLMs

1. **Prediction Object**

   * LLM — next word/token.
   * Oracle — future connection or node in the knowledge graph.

2. **Mathematical Objective**

   * LLM — minimize language error (cross-entropy).
   * Oracle — aims to reduce structural conflict and measure pressure within the knowledge graph; the current system does not demonstrate that this objective improves predictions.

3. **Nature of Model**

   * LLM — statistical language approximation.
   * Oracle — a proposed dynamic topological model of knowledge evolution, represented by prototype graph and model components.

4. **Source of Truth**

   * LLM — distribution of past texts.
   * Oracle — source evidence and heuristic estimates of physical, economic, and temporal constraints. Those estimates do not directly reveal reality and are not all independently verified.

5. **Handling the Unknown**

   * LLM reproduces patterns in its training and input text.
   * Oracle aims to rank candidate gaps in the ingested graph. It cannot establish that a missing node is structurally necessary or inevitable.

6. **Relation to Error**

   * LLM may convincingly generate the impossible.
   * Oracle aims to screen implausible candidates. In the current code, physical feasibility is an LLM-based heuristic rather than a check against verified physical bounds (see the physical-grounding note below).

7. **Temporal Model**

   * LLM — static corpus model.
   * Oracle — graph-dynamics features such as connection accumulation, conflict, and local phase-transition indicators.

8. **Functional Role**

   * LLM — text generation and analysis tool.
   * Oracle — intended as a decision-support tool for comparing candidate directions; current results are not validated recommendations.

9. **Type of Forecast**

   * LLM — “what is likely to be said.”
   * Oracle — candidate trajectories and estimated timing. “Inevitability” is a design assumption, not a validated guarantee.

10. **Level of Abstraction**

* LLM — linguistic space.
* Oracle — structural dynamics of the knowledge system.

11. **Interpretability (White Box vs. Black Box)**

* LLM — opaque weight matrices; cannot definitively explain why a specific prediction was made.
* Oracle — source-backed observations retain provenance and supporting quotes where available. Generated forecasts and virtual nodes carry their generation path or trajectory, but they are not source evidence. `AncestryEnforcer` can add a synthetic `DERIVED_FROM` edge to anchor a node structurally; this edge is not a relationship extracted from a source and must not be presented as one.

As a design metaphor, an LLM is a gear advancing one token at a time, while Oracle is a mesh of connected gears whose interactions represent graph dynamics. The metaphor describes the intended model, not a measured property of the current prototype.

Oracle uses LLMs as an auxiliary tool to transform source documents into structured features. The graph and dynamics modules then operate on those records. Calibration against independent outcomes remains separate work; an LLM is not Oracle itself. In the current pipeline, LLMs can perform:

* Entity extraction.
* Relation discovery from text.
* Semantic normalization.
* Concept clustering.
* Initial historical data annotation.

The system operates on graph records assembled from text and other evidence. The document remains a source record; it is not itself made into a knowledge node. Nodes represent extracted concepts, constraints, or other classified entities, and may collect observations from multiple documents.

Neural model classes can process graph features and generate candidate predictions. Their presence in the code does not mean that a useful trained model is available: weights, training data, and independent evaluation are separate requirements. In particular, the local graph-update workflow uses a symbolic model with neural processing disabled.

### Current Document-to-Graph Flow

The local ingestion path makes several distinctions that are important when interpreting a graph:

1. **Source library and duplicate handling.** `DocumentQueue` tracks source files separately from graph nodes. It hashes normalized extracted text to identify repeated content and, when the file contains embedded title and author metadata, can also flag a matching bibliographic identity. Bibliographic matching is a queue heuristic, not universal publication-identity resolution; revisions can be allowed explicitly. A processed-document receipt is saved with the graph snapshot so a restart does not normally apply the same document twice. It does not yet compare semantically similar full articles or detect translations and revised editions. File modification time is not treated as the date of the research claim.

2. **LLM extraction and evidence checks.** `IngestionPipeline` extracts document metadata, concepts, physical attributes, within-document relations, and candidate links to existing graph concepts. `ResponseValidator` checks required fields, evidence quotes, and relation endpoints. A node quote must occur in the normalized source text and contain more than a bare label, title, or metadata fragment; a relation endpoint must be supported by a passage that states a relation. These are conservative text heuristics, not a semantic entailment proof. Unsupported or incomplete extraction can fail the document instead of adding its proposed nodes.

3. **Dates and canonical concepts.** Publication dates are taken from document evidence and extracted metadata; unknown dates are not replaced with file timestamps. `CanonicalNodeAssembler` assigns a canonical concept ID, while `BucketIndex` proposes matches and cross-document links. A `CanonicalKey` is 12 bytes, but bytewise distance between keys is not a reliable semantic-similarity measure: the key mixes categorical IDs with a six-byte semantic code, and the code may be a SHA fallback when product quantization is unavailable. Near-duplicate candidates should be selected by text or embedding similarity, then checked against the source documents. An undated concept is treated as an enrichment of a confirmed existing concept or rejected as a new dated observation; it should not acquire a fabricated historical date.

4. **Consolidation and provenance.** `GraphConsolidator` combines observations using source-type trust weights, keeps source-level observation records and reports conflicts. Temporal intervals and density are rebuilt from dated observations. Trust is currently a static source-category multiplier; the code does not determine whether two sources are institutionally independent or whether one repeats the other. The density curve is therefore a weighted observation history, not a count of independently replicated findings. Source records retain evidence and bibliographic metadata, but the epistemic classification is not attached to each extracted label and its source text: `epistemic_level` is a node-wide scalar, and some paths assign `PROVEN_DISCOVERY` from source category, entity type, or score. This is not evidence that every claim represented by that node is a fact. The intended unit is each dated label/claim from its own source text, classified as idea/concept, theory, fact, evidence, or refutation. The timeline should retain those dated classifications and evidence, then derive how the support for that label changes over time. New evidence or a refutation may strengthen or weaken its current support without erasing earlier history. Popularity and journal prestige alone must not change a claim's classification.

5. **Evidence edges, structural edges, and forecasts.** Edges extracted from a source and backed by its text should be interpreted separately from generated structure. `AncestryEnforcer` may synthesize a modest `DERIVED_FROM` edge to give a node a structural predecessor; its marker is `discovery_method="ancestry_synthesised"`, and its placeholder evidence is not a source quotation. Forecast scenarios, phantom nodes, and virtual nodes are candidate outputs with generation paths; they are not source-backed facts and should not be mixed into an evidence graph without an explicit label.

   The current `PhantomNodeGenerator` is a structural-gap detector, not a technology synthesizer. It searches a bounded pool around changed nodes (two graph hops plus a capped same-topic group/domain pool), ranks unconnected pairs by their existing void pressure, and requires source-quoted fact/proof support at both endpoints or at one endpoint paired with a limited share of theory/idea nodes. Social-readiness observations—such as fictional depictions, public or forum interest, and documented unmet needs—are retained in candidate data when available and deduplicated by document identity. Their absence does not block a candidate or change a claim's fact/theory status. `unmet_need` is validated from its source quote and contributes to the observed social factor. The pressure threshold and structural-gap score rank candidates; they are not probabilities or measurements of a proposed edge. The generator does not yet extract a durable inventory of problem statements and solution mechanisms or compare those across unrelated documents. That cross-document matching must use evidence from both documents and remain a forecast until confirmed. The live visualizer therefore draws `candidate_bridge` paths as dashed forecasts with no edge factor values or source observations. A source-backed limitation-to-solution edge must come from an analyzed document that explicitly states the relation.

   Citations, reuse of published data, investment reports, public discussion, fictional depictions, and explicitly documented unmet needs are recorded as source-influence observations on an already matched concept. They do not by themselves create a scientific node or a causal edge, and they do not promote a theory to fact. Their qualitative effect on existing incident edges is counted by unique document identity and expressed as a share of the analyzed corpus for citation, investment, or social factors; replaying one document does not add another count. When a source explicitly reports a normalized numeric measurement for an edge factor, that quoted value replaces that document's binary signal contribution. A ratio is normalized only when both quoted numbers and their numerator/denominator relation are present; raw counts and monetary values otherwise remain attached to their source quote. The LLM does not invent seven edge scores from prose; semantic and temporal components are derived from endpoint data, and other scores require the evidence rules above. Thus many papers, books, or discussions about a theoretical concept can increase its observed corpus prevalence without changing its per-source epistemic status.

   The optional `reference_graph_preview.py` mode builds a citation-only preview from DOI-backed entries in document References sections. A preview edge records a bibliographic citation; it does not show that the cited work supports a particular claim or that the two concept nodes have a semantic, causal, or technical relationship. The preview is not a full Oracle-1 analysis result.

6. **Historical slices.** `RuntimeGraph.as_of()` can build a graph slice at a past cutoff and trims recorded temporal and epistemic histories. Some aggregate scores that lack a single reversible history remain at their last-known value. Historical slices are useful for replay, but those fields can still contain hindsight and should not be treated as a fully reconstructed past state.

Oracle-1's graph and model components include:

1. **Multi-Component Edge Attention**
   Implemented in `MultiComponentEdgeAttention`. Seven parallel streams analyze each edge (semantics, time, problem-solving, citations, money, society, inhibitors).
   Each stream has a learnable weight matrix; the model dynamically prioritizes which stream to trust per node.
   The streams retain distinct edge components rather than requiring one blended semantic score. Some graph dynamics recompute derived priorities and weights as the graph changes. Source prevalence can be represented when observations are ingested, but the current system does not verify that corroborating sources are independent. A high `limitation_resolution` value can connect nodes with little shared vocabulary; this is a candidate relation score, not by itself proof that one node resolves the other's limitation.

2. **Contextual Priority Head**
   Implemented in `ContextualPriorityHead`. Modulates attention based on “graph vibration”:

   * **Void Pressure:** Attention rises exponentially where unresolved problems, expectations, or capital accumulate.
   * **Sleeping Edge Effect:** Old ideas near “hot zones” are instantly highlighted.
   * **Inhibitory Brake:** Nodes surrounded by incumbents slow signal propagation.

3. **Structural Feedback Loop**
   Implemented in `StructuralDependencyUpdater`. Adjusts node roles dynamically: “source” vs. “sink” and recalibrates attention to transmit signals from critical nodes efficiently.

4. **Retrospective Loss**
   A retrospective loss (`RetrospectiveConvergenceLoss`) is available for training when historical targets are supplied. It does not by itself provide historical labels, prove that a model has learned, or establish future accuracy.

**Oracle vs. LLM Attention:**

* LLM: “Which word should I focus on?”
* Oracle: “Which connection in human civilization’s structure is critical for future emergence?”

Oracle-1 is intended to combine scientific and non-scientific signals. The code has fields and mechanisms for the following; their presence does not mean that live connectors or comprehensive data for each signal are included:

* `cultural_phantom + sentiment_fiction_score`
* `social_gravity + forum_post_count + social_correlation`
* `investment_correlation + excess_investment_pressure`
* `inhibitory_force` from incumbents
* Temporal zones and acceleration multipliers
* Epistemic levels from folklore to proven discovery
* Cross-domain from myths, fiction, news, memes

Speculative fiction can offer candidate effects or combinations that may suggest edges or missing nodes. This is a design hypothesis, not an established statistical result. Such descriptions are not evidence that a proposed mechanism is real; in the current prototype, any physical plausibility screen is an LLM-based heuristic rather than a verified physics check. A proposed effect and its fictional mechanism should therefore remain separate annotations.

Reducing or removing these signals yields an ordinary scientific network, not Oracle-1.

### The "What-If" Intervention Optimizer

Oracle-1 also includes a strategic sandbox. `InterventionOptimizer` applies hypothetical interventions and uses `CascadeAnalyzer` to estimate downstream effects across domains. These results depend on the prototype's graph scores and heuristics; they are not empirically calibrated impact estimates. The code supports five intervention types:

* **Investment Injection** — simulating funding at a given scale, its effect on readiness and maturity scaled down by the node's own `feasibility_score`, so money alone cannot buy past a synthesis-complexity limit that was never a funding problem to begin with.
* **Regulatory Removal** — artificially lowering `inhibitory_force` to see how fast a technology would grow absent incumbent resistance.
* **Breakthrough Adjacent** — simulating sudden maturation in a parallel domain (e.g., an optical-modulator breakthrough resolving a storage bottleneck).
* **Talent Influx** and **Geopolitical Acceleration** — social and zone-level catalysts.

`CascadeAnalyzer` can also superpose several simultaneous interventions and surface where their waves interfere constructively: a node reached by more than one intervention's wave is flagged as a resonance point, not merely reported twice.

**Note on suppression vs. acquisition.** `inhibitory_force` is currently a single unsigned value in [0,1]: how strongly an incumbent resists a node, with nothing recording *how*. An incumbent buying a threatening startup to absorb and scale it (Facebook/Oculus) and one lobbying to delay or bury it (incumbent resistance to a competing technology) both look identical to the model today — high investment correlation, high inhibitory force — even though their effect on the node's trajectory is opposite. Separating the two would need a signed polarity term, learned from whether an incumbent's investment edges have historically preceded a target node's growth or its stagnation, rather than folding both into one magnitude. Not yet implemented.

`OptimalInterventionFinder` uses beam search to rank combinations for a stated goal and optional budget. Its objective is a model score, not a calibrated probability, and the result is a candidate for analysis rather than a validated prescription.

### High-Level System Overview

1. **Foundation and Scalability**

   * `CanonicalKey`: 12-byte node IDs encoding domain, group, type, and semantic hash. This gives the identifier format a large namespace; it does not mean the in-memory graph can store or process billions of nodes.
   * `AncestryEnforcer`: Adds a marked synthetic predecessor edge to eligible nodes that otherwise have no incoming edge; primordial roots and axioms are exempt.
   * `LookupTable`: Global registry of knowledge domains and entity types; normalizes related but domain-specific concepts.
   * `BucketIndex` and `StructuralSparsityRegulator`: `BucketIndex` narrows candidate matching using embedding and domain/type buckets; the regulator limits forecast branching and some dense-neighbor expansion. These are prototype optimizations, not a benchmark demonstrating billion-node performance.

2. **Epistemology Layer**

   * `IngestionPipeline`: Multi-stage document processing for concepts, evidence, physical attributes, and secondary contexts.
   * `PhysicalSubstrateEncoder` and `ResponseValidator`: Produce heuristic physical-axis scores and check extracted evidence; neither is a verified physics solver or semantic entailment engine.
   * `SignificanceProcessor`: Builds node-level influence timelines and scores from recorded source observations, using configured source-category trust. It does not yet classify each extracted label from that observation's source text or maintain a time-varying support state per label.

3. **Neural Core**

   * `Multi-Component Edge Attention`: Model code for processing seven separate edge components; useful learned behavior depends on training and evaluation.
   * `ContextualPriorityHead`: Amplifies attention toward unresolved clusters.
   * `StructuralDependencyUpdater`: Dynamically updates node influence (donor vs. acceptor).

4. **Physics and Dynamics**

   * `PhysicalSubstrateEncoder`: Produces LLM-based plausibility scores; it does not verify physical bounds.
   * `DormancyTracker`: Reactivates old knowledge in response to modern catalysts.
   * `ConvergenceAccelerometer`: Measures convergence speed and acceleration of nodes.

5. **Singularity Topology**

   * `OntologicalTensionField`: Computes accumulated tension per graph location.
   * `PhaseTransitionDetector`: Applies local rupture indicators; virtual-node generation is a separate forecast output.
   * `PhantomNodeGenerator`: Finds structurally pressured pairs of evidence-supported nodes and records a generic bridge candidate; it does not generate a named technical solution or compare explicit limitations with solution mechanisms.

6. **Future Synthesis**

   * `RecursiveForecastingLoop`: Builds recursive forecast scenarios from candidate nodes and edges.
   * `VirtualNodeSynthesizer`: Stores candidate future nodes in a separate virtual-node collection for inspection.

7. **Consolidation Layer**

   * `GraphConsolidator`: Merges source observations into canonical nodes using field-specific consolidation rules and source trust weights.

Oracle-1's design treats graph structure and temporal density as model inputs alongside extracted facts. This is a modeling assumption; the current prototype does not establish that graph geometry predicts real-world outcomes.

**Temporal Geometry**

* `Temporal Zones`: Time-and-domain regions with acceleration multipliers (e.g., a modeled Manhattan Project era or AI boom period).
* `DormancyTracker`: Tracks inactive nodes and can reactivate them when neighboring conditions change.
* Historical replay: `RuntimeGraph.as_of()` and temporal-window snapshots can construct past graph slices for analysis; see the limitations above before using them for backtests.

**Topological Architecture**

* `First-Class Constraints`: The schema and extraction prompts can represent physical limits and bottlenecks as independent `physical_limit` nodes. Whether a limit is extracted correctly and whether a proposed `limitation_resolution` edge is valid still depend on source evidence and validation; the graph representation alone does not establish either.
* `Hierarchical Containment`: Nodes contain nested subnodes, distributing stress proportionally.
* `Structural Voids`: Scores candidate gaps between graph regions and can propose bridge nodes for inspection.
* `Cross-Domain Isomorphism Amplifier`: Scores candidate transfers between domains with matching graph structure; a match is a hypothesis to investigate, not evidence that a solution transfers.

**Fields of Influence**

* `Convergence Pressure Field`: Computes modeled convergence pressure; intersections are candidate high-pressure regions, not established singularities.
* `Inhibitory Drag`: Represents an estimated resistance signal that can damp modeled propagation; it does not directly measure actual technology adoption.
* `Epistemic Mass`: A modeled node score used by graph dynamics; it does not measure actual research allocation unless that allocation is represented by suitable source data.

**Phase Mechanics**

* `Ontological Tension`: A local score derived from pressure, conflict, and connectivity signals.
* `Rupture Trigger`: Current code flags local threshold conditions from pressure, conflict, and connectivity signals. This does not establish that the graph topology as a whole has reorganized; the threshold limitation is described above.

### Implementation Status

The repository contains a runnable local prototype, not merely a reference sketch. It includes a document queue and extraction pipeline, evidence validation, canonical concept IDs, graph consolidation, persistent snapshots, graph dynamics, forecast and intervention components, and a local dashboard. This is still a research prototype: the existence of a class or algorithm in the code is not evidence that its predictions are accurate or that its assumptions have been validated.

Illustrative documents such as `Convergence_Analysis.md` are examples or concept material, not output from an Oracle-1 run. Run outputs should be identified by their run artifacts and metadata.

**Implemented in the current repository:**

* Local document ingestion through a configured LLM, with source metadata, evidence checks, duplicate handling, and graph snapshots.
* In-memory `RuntimeGraph` operations, temporal and provenance records, canonical matching, and field-specific consolidation.
* Heuristic graph dynamics, scenario generation, virtual/phantom nodes, and intervention analysis.
* Neural model and training code. The local `graph_update.py` workflow is symbolic and disables neural processing; running ingestion or graph restructuring through it does not train a predictive model.

**Not established or not built:**

* Operational storage for billions of graph nodes. Automatic source collection is outside the current debugging scope: documents are intentionally selected and submitted for local ingestion. Broad live connectors would be a separate future deployment feature, not a prerequisite for debugging Oracle-1.
* Distributed graph storage, topology sharding, or cross-machine ripple propagation. These remain target-architecture ideas.
* LLM-assisted comparison of selected source documents when similar concepts or semantic IDs suggest a possible duplicate. The intended debug workflow is to retrieve each stored source (when available), compare article text, and run a marked A/B LLM comparison only when article identity is above a configured threshold (initially 50%). The LLM should return one of `duplicate`, `alternate wording or translation`, `supplement`, or `different data`. For duplicates and translations, reuse existing graph nodes and add the new source name/provenance rather than creating duplicate nodes. For supplements, retain new evidence on matched concepts and add only genuinely new concepts. For different data, preserve both observations; create a contradiction edge only when the claims actually conflict. Current deduplication does none of this: it uses normalized content and limited bibliographic identity.
* Independent-source verification, TrustRank-style trust propagation, or automatic detection of copied and echoing claims.
* Per-label epistemic classification and temporal support. For each label/claim extracted from an article, the LLM must use that article's text and supporting passage to classify it as idea/concept, theory, fact, evidence, or refutation. The dated history belongs to that label/claim, not to the whole canonical node; later evidence can change its support going forward while preserving the earlier timeline. Current `source_observations` retain per-source evidence, but there is no such per-label classification or support curve. The node-wide `epistemic_level` and `InfluenceTimeline` cannot substitute for it.
* Claim-level contradiction handling: never delete either claim merely because it is disputed; retain both with their source text and evidence, and add an explicit contradiction relation when the claims conflict. A later refutation must update the affected label's temporal support and flow into dependent graph calculations. Current conflict reports surface some field disagreements but do not implement this belief-revision process or an explicit `contradicts` relation in extraction.
* Reliable evidence-based status rules. The current code can assign `PROVEN_DISCOVERY` from journal name, entity type, or score, and `InfluenceTimeline` can derive proven status from those inputs. That is unsafe: popularity or journal prestige alone must never make a label such as dark matter a proven fact. Source credibility, the claim's type, and evidence supporting that specific claim must be separate inputs; downstream strength must come from dated evidence and refutations for the individual label.
* A physics solver grounded in verified numeric limits, a graph-wide spectral rupture detector, or calibrated probability and timing estimates.
* Demonstrated accuracy from independent temporal backtests or continuous evaluation against live outcomes.

At the intended scale — many live sources and potentially billions of nodes — a single in-memory process would not suffice. The proposed architecture is distributed and would propagate updates only through graph regions materially affected by a change. This causal-gating design has not been implemented or benchmarked.

**Topology sharding** is also a proposal, not a current deployment feature. One idea is to group nodes by primary domain and keep common same-domain edges local while updating cross-domain edges asynchronously. The performance and correctness tradeoffs of this approach remain open engineering questions.

Small or single-domain corpora are useful for validating extraction, deduplication, graph rendering, and local algorithm behavior. They are not enough to evaluate the broader cross-domain convergence hypothesis. A pilot intended to evaluate that hypothesis would need connected evidence across the relevant domains, with source provenance and independent temporal evaluation.

---

Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)

---
