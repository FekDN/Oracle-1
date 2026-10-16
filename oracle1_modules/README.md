# Oracle-1 Python modules

`oracle1.py` is a compatibility facade. Existing scripts can keep importing
from `oracle1`; new implementation code should import from the module that owns
the relevant behavior.

| Module | Responsibility |
| --- | --- |
| `config`, `llm` | Runtime settings, derived constants, and LLM clients |
| `entities`, `canonical_keys`, `canonical` | Graph records, stable identifiers, and concept assembly |
| `graph`, `time_utils` | Runtime graph operations and platform-independent date conversion |
| `graph_dynamics`, `topology_dynamics`, `forecasting` | Local pressure, topology signals, and future scenarios |
| `validation`, `ingestion`, `consolidation` | Document processing, evidence validation, graph updates, and merging |
| `neural`, `tensorization`, `training` | Model layers, graph-to-tensor conversion, and training routines |
| `significance`, `json_utils` | Epistemic scoring and tolerant model-response parsing |
| `orchestrators`, `system` | Coordination and full-system construction |
| `snapshots`, `examples`, `interventions`, `visualizer` | Persistence, sample histories, intervention analysis, and live graph queries |

Implementation modules import shared primitives from `_shared.py` and use
relative imports for their dependencies. Keep the dependency direction
acyclic; put cross-cutting composition in `system.py` rather than importing the
system factory into lower-level modules.

Write new function and class comments and docstrings in English.
