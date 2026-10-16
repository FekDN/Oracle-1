"""Canonical key, alias, lookup-table, and semantic-hash primitives."""
from __future__ import annotations

from ._shared import *

# ══════════════════════════════════════════════════════════════════════════════
# INLINED MODULE: oracle1_canonical_key.py
# Canonical Key System + Domain Classifier
# ══════════════════════════════════════════════════════════════════════════════

# Optional dependencies used by this module
try:
    from torch.utils.data import DataLoader, Dataset as TorchDataset
except ImportError:
    DataLoader = None       # type: ignore[assignment,misc]
    TorchDataset = None     # type: ignore[assignment,misc]

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("[CanonicalKey] transformers not available — using hash-only embedder")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("[CanonicalKey] faiss not available — brute-force ANN fallback")


# ══════════════════════════════════════════════════════════════════
#  LOOKUP TABLES  (3 основные + Alias)
#  Пополняются LLM и вручную. Хранятся в памяти + сериализуются.
# ══════════════════════════════════════════════════════════════════

# Начальные записи — seed. При старте системы LLM дополняет их.
_DEFAULT_DOMAINS: Dict[int, str] = {
    0x0000: "unknown",
    0x0001: "physics",
    0x0002: "quantum_mechanics",
    0x0003: "optics",
    0x0004: "chemistry",
    0x0005: "biology",
    0x0006: "medicine",
    0x0007: "military_technology",
    0x0008: "electronics",
    0x0009: "materials_science",
    0x000A: "computer_science",
    0x000B: "mathematics",
    0x000C: "economics",
    0x000D: "fiction",
    0x000E: "mythology_folklore",
    0x000F: "social_science",
    0x0010: "energy",
    0x0011: "space_technology",
    0x0012: "biotechnology",
    0x0013: "nanotechnology",
    0x0014: "quantum_electronics",
    0x0015: "military_finance",
    0x0016: "industrial_manufacturing",
    0x0017: "environmental_science",
    0x0018: "neuroscience",
    0x0019: "artificial_intelligence",
}

_DEFAULT_GROUPS: Dict[int, str] = {
    0x0000: "unknown",
    0x0001: "fundamental_principle",
    0x0002: "applied_technology",
    0x0003: "cultural_artifact",
    0x0004: "economic_force",
    0x0005: "biological_system",
    0x0006: "social_phenomenon",
    0x0007: "military_system",
    0x0008: "industrial_process",
    0x0009: "measurement_instrument",
    0x000A: "theoretical_framework",
    0x000B: "regulatory_construct",
    0x000C: "fictional_concept",
    0x000D: "historical_event",
    0x000E: "speculative_technology",
    0x000F: "barrier_constraint",
}

_DEFAULT_ENTITY_TYPES: Dict[int, str] = {
    0x0000: "unknown",
    0x0001: "physical_principle",
    0x0002: "breakthrough_node",
    0x0003: "cultural_phantom",
    0x0004: "dormant_principle",
    0x0005: "functional_spec",
    0x0006: "resource_source",
    0x0007: "industrial_substrate",
    0x0008: "technical_mastery",
    0x0009: "unifying_theory",
    0x000A: "research_frontier",
    0x000B: "incumbent_tech",
    0x000C: "challenger_tech",
    0x000D: "enabler_component",
    0x000E: "measurement_tool",
    0x000F: "side_effect_risk",
    0x0010: "regulatory_constraint",
    0x0011: "convergence_node",
    0x0012: "breakthrough_event",
    0x0013: "economic_barrier",
    0x0014: "physical_limit",
    0x0015: "experimental_method",
}


class LookupTable:
    """
    Двунаправленная таблица id ↔ name.
    Потокобезопасна для чтения; запись однопоточная.
    Пополняется LLM через register() или вручную.
    """

    def __init__(self, name: str, initial: Dict[int, str]):
        self.name = name
        self._id2name: Dict[int, str] = dict(initial)
        self._name2id: Dict[str, int] = {v: k for k, v in initial.items()}
        self._next_id: int = max(initial.keys(), default=0) + 1

    # ── Основной API ────────────────────────────────────────────

    def get_id(self, name: str) -> int:
        """Вернуть ID по имени. 0x0000 если не найдено."""
        return self._name2id.get(name.lower().strip(), 0x0000)

    def get_name(self, id_: int) -> str:
        """Вернуть имя по ID."""
        return self._id2name.get(id_, "unknown")

    def register(self, name: str) -> int:
        """
        Зарегистрировать новое имя и вернуть его ID.
        Если имя уже есть — вернуть существующий ID.
        """
        key = name.lower().strip()
        if key in self._name2id:
            return self._name2id[key]
        new_id = self._next_id
        self._next_id += 1
        self._id2name[new_id] = key
        self._name2id[key] = new_id
        logger.info(f"[LookupTable:{self.name}] registered '{key}' → 0x{new_id:04X}")
        return new_id

    def all_names(self) -> List[str]:
        return list(self._name2id.keys())

    def __len__(self) -> int:
        return len(self._id2name)

    # ── Сериализация ────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {"name": self.name,
                "id2name": {str(k): v for k, v in self._id2name.items()},
                "next_id": self._next_id}

    @classmethod
    def from_dict(cls, d: Dict) -> "LookupTable":
        id2name = {int(k): v for k, v in d["id2name"].items()}
        obj = cls(d["name"], id2name)
        obj._next_id = d["next_id"]
        return obj

    def restore_from_dict(self, d: Dict) -> None:
        """
        Like from_dict(), but mutates this instance in place instead of
        constructing a new one. Needed because ConceptClassifier.__init__'s
        domain_table/group_table/entity_table/alias_table parameters default
        to the module-level DOMAIN_TABLE/GROUP_TABLE/ENTITY_TYPE_TABLE/
        ALIAS_TABLE singletons — and Python default-argument values are
        bound ONCE, at function-definition (module-import) time. Rebinding
        the module-level name (as the old load_lookup_tables() did, via
        `global DOMAIN_TABLE = LookupTable.from_dict(...)`) does NOT change
        what that already-baked-in default value points to, so every
        ConceptClassifier constructed without an explicit domain_table= —
        which is the normal case — would silently keep using the stale,
        freshly-seeded (_DEFAULT_DOMAINS-only) table forever, never seeing
        anything restored after the fact. Mutating the SAME object every
        holder already points to sidesteps the whole problem.
        """
        self.name          = d["name"]
        self._id2name      = {int(k): v for k, v in d["id2name"].items()}
        self._name2id      = {v: k for k, v in self._id2name.items()}
        self._next_id      = d["next_id"]


# ── Глобальные таблицы (синглтоны) ──────────────────────────────

DOMAIN_TABLE      = LookupTable("domain",      _DEFAULT_DOMAINS)
GROUP_TABLE       = LookupTable("group",        _DEFAULT_GROUPS)
ENTITY_TYPE_TABLE = LookupTable("entity_type",  _DEFAULT_ENTITY_TYPES)


def save_lookup_tables(path: str) -> None:
    data = {
        "domain":      DOMAIN_TABLE.to_dict(),
        "group":       GROUP_TABLE.to_dict(),
        "entity_type": ENTITY_TYPE_TABLE.to_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"[LookupTables] saved to {path}")


def load_lookup_tables(path: str) -> None:
    """Mutates DOMAIN_TABLE/GROUP_TABLE/ENTITY_TYPE_TABLE in place (see
    LookupTable.restore_from_dict for why this must NOT rebind the module-
    level names — any ConceptClassifier already constructed, or about to be
    constructed with its default arguments, holds/will hold a reference to
    these exact objects, captured at class-definition time)."""
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    DOMAIN_TABLE.restore_from_dict(data["domain"])
    GROUP_TABLE.restore_from_dict(data["group"])
    ENTITY_TYPE_TABLE.restore_from_dict(data["entity_type"])
    logger.info(f"[LookupTables] loaded from {path}")


# ══════════════════════════════════════════════════════════════════
#  ALIAS TABLE
#  Строка (любой язык/написание) → Canonical ID
#  Хранится отдельно; в production — Redis
# ══════════════════════════════════════════════════════════════════

class AliasTable:
    """
    Разрешает синонимы и варианты написания к единому canonical_id.

    ВАЖНО: один термин в РАЗНЫХ доменах → РАЗНЫЕ canonical_id.
    Поэтому ключ хранится как (alias, domain_hint) → canonical_id.

    Если domain_hint не указан (None) — используется первый
    зарегистрированный для данного alias (точный match).

    "лазер" (physics) → ключ physics
    "лазер" (fiction) → ключ fiction    ← другой!
    "LASER" (physics) → ключ physics    ← тот же что "лазер" physics

    При ingestion:
      1. Нормализуем термин (lower + strip).
      2. Ищем в alias_table по (term, domain).
         – Нашли → концепт уже существует → merge наблюдений.
         – Не нашли → создаём новый Canonical ID → регистрируем alias.
      3. LLM может явно указать синонимы → bulk_register().
    """

    def __init__(self):
        # (alias_str, domain_str_or_None) → canonical_id bytes(12)
        self._map: Dict[Tuple[str, Optional[str]], bytes] = {}
        # alias_str → set of domains (для cross-domain lookup)
        self._alias_domains: Dict[str, List[str]] = defaultdict(list)

    def register(self, alias: str, canonical_id: bytes,
                 domain: Optional[str] = None) -> None:
        alias_key = alias.lower().strip()
        domain_key = domain.lower().strip() if domain else None
        map_key = (alias_key, domain_key)
        if map_key in self._map and self._map[map_key] != canonical_id:
            # Конфликт в одном домене — предупреждение, оставляем оригинал
            logger.warning(
                f"[AliasTable] conflict: '{alias_key}'@{domain_key} "
                f"was {self._map[map_key].hex()} "
                f"now {canonical_id.hex()} — keeping original"
            )
            return
        self._map[map_key] = canonical_id
        if domain_key and domain_key not in self._alias_domains[alias_key]:
            self._alias_domains[alias_key].append(domain_key)

    def bulk_register(self, aliases: List[str], canonical_id: bytes,
                      domain: Optional[str] = None) -> None:
        for a in aliases:
            self.register(a, canonical_id, domain=domain)

    def lookup(self, alias: str,
               domain: Optional[str] = None) -> Optional[bytes]:
        """
        Поиск canonical_id по alias + domain.
        Если domain указан — точный поиск.
        Если domain=None — ищем без домена (точный alias в любом контексте).
        """
        alias_key  = alias.lower().strip()
        domain_key = domain.lower().strip() if domain else None
        return self._map.get((alias_key, domain_key))

    def all_domains_for(self, alias: str) -> List[str]:
        """Все домены, в которых зарегистрирован этот alias."""
        return self._alias_domains.get(alias.lower().strip(), [])

    def __len__(self) -> int:
        return len(self._map)

    def to_dict(self) -> Dict[str, str]:
        # Сериализуем ключ как "alias|||domain" (domain может быть None → "")
        return {
            f"{alias}|||{domain or ''}": cid.hex()
            for (alias, domain), cid in self._map.items()
        }

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "AliasTable":
        obj = cls()
        for compound_key, hex_val in d.items():
            if "|||" in compound_key:
                alias, domain_part = compound_key.split("|||", 1)
                domain = domain_part if domain_part else None
            else:
                alias, domain = compound_key, None
            cid = bytes.fromhex(hex_val)
            obj._map[(alias, domain)] = cid
            if domain and domain not in obj._alias_domains[alias]:
                obj._alias_domains[alias].append(domain)
        return obj

    def restore_from_dict(self, d: Dict[str, str]) -> None:
        """In-place counterpart to from_dict() — see LookupTable.
        restore_from_dict for why ConceptClassifier's alias_table=ALIAS_TABLE
        default argument needs the SAME object mutated, not a new one built
        and left unreferenced by anything already holding the old one."""
        self._map.clear()
        self._alias_domains.clear()
        for compound_key, hex_val in d.items():
            if "|||" in compound_key:
                alias, domain_part = compound_key.split("|||", 1)
                domain = domain_part if domain_part else None
            else:
                alias, domain = compound_key, None
            self._map[(alias, domain)] = bytes.fromhex(hex_val)
            if domain and domain not in self._alias_domains[alias]:
                self._alias_domains[alias].append(domain)


ALIAS_TABLE = AliasTable()


# ══════════════════════════════════════════════════════════════════
#  SEMANTIC HASH  (6 байт)
#  Квантизация эмбеддинга → детерминированный хэш
#  Два семантически близких вектора → одинаковый или соседний хэш
# ══════════════════════════════════════════════════════════════════

class SemanticHasher:
    """
    Product Quantization (PQ) → 6-байтный семантический хэш.

    При наличии FAISS:
      – Обучает PQ-кодировщик на накопленных эмбеддингах.
      – Выдаёт 6-байтный код (48 бит).

    Без FAISS (fallback):
      – SHA-256 от нормализованного вектора.
      – Коллизии выше, но работает без зависимостей.

    После обучения PQ один и тот же концепт в разных формулировках
    попадает в один центроид → один хэш → один canonical_id.
    """

    PQ_BYTES   = 6    # 48 бит кода
    PQ_SUBVECS = 6    # количество субпространств (должно делить embedding_dim)
    MIN_TRAIN  = 256  # минимум векторов для обучения PQ

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self._pq           = None   # faiss.ProductQuantizer после обучения
        self._trained      = False
        self._train_buffer: List[np.ndarray] = []

    # ── Основной API ────────────────────────────────────────────

    def hash(self, embedding: np.ndarray) -> bytes:
        """
        Вычислить 6-байтный семантический хэш для эмбеддинга.
        Автоматически обучает PQ при достаточном буфере.
        """
        vec = self._normalize(embedding)
        self._train_buffer.append(vec)

        if FAISS_AVAILABLE and not self._trained:
            if len(self._train_buffer) >= self.MIN_TRAIN:
                self._train_pq()

        if FAISS_AVAILABLE and self._trained:
            return self._pq_hash(vec)
        else:
            return self._fallback_hash(vec)

    def hash_text(self, text: str, embedder=None) -> bytes:
        """Хэш от текста. Если embedder=None — чистый SHA-256."""
        if embedder is not None:
            try:
                emb = embedder.encode(text)
                return self.hash(np.array(emb, dtype=np.float32))
            except Exception as e:
                logger.warning(f"[SemanticHasher] embedder failed: {e}")
        # Fallback: хэш от нормализованного текста
        normalized = text.lower().strip()
        digest = hashlib.sha256(normalized.encode()).digest()
        return digest[:self.PQ_BYTES]

    # ── Обучение PQ ─────────────────────────────────────────────

    def _train_pq(self) -> None:
        if not FAISS_AVAILABLE:
            return
        matrix = np.stack(self._train_buffer).astype(np.float32)
        # subvector size: embedding_dim / n_subvecs
        sub_dim = self.embedding_dim // self.PQ_SUBVECS
        if self.embedding_dim % self.PQ_SUBVECS != 0:
            logger.warning(
                f"[SemanticHasher] embedding_dim={self.embedding_dim} "
                f"не делится на PQ_SUBVECS={self.PQ_SUBVECS} — fallback"
            )
            return
        # 256 центроидов на субпространство → 8 бит на субпространство
        self._pq = faiss.ProductQuantizer(self.embedding_dim, self.PQ_SUBVECS, 8)
        self._pq.train(matrix)
        self._trained = True
        logger.info(
            f"[SemanticHasher] PQ trained on {len(self._train_buffer)} vectors "
            f"({self.PQ_SUBVECS} subvecs × 8 bits = {self.PQ_BYTES} bytes)"
        )

    def force_train(self, embeddings: np.ndarray) -> None:
        """Явно обучить PQ на переданных векторах (для инициализации)."""
        self._train_buffer = [self._normalize(e) for e in embeddings]
        if len(self._train_buffer) >= self.MIN_TRAIN:
            self._train_pq()

    # ── Внутренние методы ───────────────────────────────────────

    def _pq_hash(self, vec: np.ndarray) -> bytes:
        codes = np.zeros((1, self.PQ_SUBVECS), dtype=np.uint8)
        self._pq.compute_codes(vec.reshape(1, -1), codes)
        # codes[0] → PQ_SUBVECS байт → берём первые PQ_BYTES
        raw = bytes(codes[0].tolist())
        return raw[:self.PQ_BYTES]

    @staticmethod
    def _fallback_hash(vec: np.ndarray) -> bytes:
        """SHA-256 от квантизованного вектора (без FAISS)."""
        quantized = (vec * 127).astype(np.int8)
        digest = hashlib.sha256(quantized.tobytes()).digest()
        return digest[:6]

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        v = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12)


# Глобальный хэшер
SEMANTIC_HASHER = SemanticHasher(embedding_dim=384)


# ══════════════════════════════════════════════════════════════════
#  CANONICAL KEY  (12 байт)
#  Основная точка входа: текст + контекст → bytes(12)
# ══════════════════════════════════════════════════════════════════

@dataclass
class CanonicalKey:
    """
    12-байтный ключ концепта.

    domain_id     : uint16  — область знаний
    group_id      : uint16  — типовая группа
    entity_type_id: uint16  — тип сущности
    sem_hash      : bytes6  — семантический хэш эмбеддинга

    Важно: один термин "лазер" в разных доменах → разные domain_id
    → разные ключи. Это позволяет агрегировать cross-domain
    присутствие концепта и измерять "необходимость" технологии.
    """

    domain_id:      int
    group_id:       int
    entity_type_id: int
    sem_hash:       bytes   # ровно 6 байт

    # ── Сериализация ────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """Упаковать в 12 байт: 3 × uint16 (big-endian) + 6 байт хэша."""
        header = struct.pack(">HHH", self.domain_id, self.group_id,
                             self.entity_type_id)
        return header + self.sem_hash

    @classmethod
    def from_bytes(cls, raw: bytes) -> "CanonicalKey":
        assert len(raw) == 12, f"Expected 12 bytes, got {len(raw)}"
        d, g, e = struct.unpack(">HHH", raw[:6])
        return cls(domain_id=d, group_id=g, entity_type_id=e,
                   sem_hash=raw[6:])

    def hex(self) -> str:
        return self.to_bytes().hex()

    def __hash__(self):
        return hash(self.to_bytes())

    def __eq__(self, other):
        if isinstance(other, CanonicalKey):
            return self.to_bytes() == other.to_bytes()
        return False

    # ── Читаемое представление ──────────────────────────────────

    def describe(self) -> str:
        d = DOMAIN_TABLE.get_name(self.domain_id)
        g = GROUP_TABLE.get_name(self.group_id)
        e = ENTITY_TYPE_TABLE.get_name(self.entity_type_id)
        h = self.sem_hash.hex()
        return f"[{d} / {g} / {e}] sem:{h}"

    # ── Измерение "необходимости" концепта ─────────────────────

    @staticmethod
    def cross_domain_presence(
        term: str,
        alias_table: AliasTable,
        graph_nodes: Dict[str, Any],   # {canonical_id_hex: node}
    ) -> Dict[str, int]:
        """
        Посчитать, в скольких доменах встречается термин.
        Возвращает {domain_name: count}.

        Высокий cross-domain score = высокая "необходимость"
        технологии (архетип Oracle-1).
        """
        counts: Dict[str, int] = defaultdict(int)
        # Собираем все canonical_id, где alias совпадает семантически
        # (упрощённо: по exact alias match — в production через FAISS)
        cid_bytes = alias_table.lookup(term)
        if cid_bytes is None:
            return dict(counts)

        # Ищем узлы с таким же sem_hash (последние 6 байт)
        target_sem = cid_bytes[6:]
        for hex_id, node in graph_nodes.items():
            try:
                raw = bytes.fromhex(hex_id)
                if len(raw) != 12:
                    continue
                if raw[6:] == target_sem:
                    d_id = struct.unpack(">H", raw[0:2])[0]
                    domain_name = DOMAIN_TABLE.get_name(d_id)
                    counts[domain_name] += 1
            except Exception:
                continue
        return dict(counts)

    @staticmethod
    def necessity_score(cross_domain_counts: Dict[str, int]) -> float:
        """
        Индекс необходимости: log(1 + sum) × кол-во доменов.
        Максимум при широком и глубоком cross-domain присутствии.
        """
        if not cross_domain_counts:
            return 0.0
        total     = sum(cross_domain_counts.values())
        n_domains = len(cross_domain_counts)
        return float(math.log1p(total) * n_domains)


# ══════════════════════════════════════════════════════════════════
#  OBSERVATION RECORD
#  Лёгкая запись наблюдения — время принадлежит ему, не концепту
# ══════════════════════════════════════════════════════════════════

@dataclass
class ObservationRecord:
    """
    Одно наблюдение концепта из одного источника.
    canonical_id → FK на CanonicalKey.
    Много ObservationRecord → один CanonicalKey.
    """
    canonical_id:  bytes            # 12-байтный ключ
    obs_id:        str              # UUID наблюдения
    timestamp:     float            # когда опубликован источник
    source:        str              # URL / название
    source_type:   str              # arxiv / forum / news / ...
    trust:         float            # 0.0–1.0
    delta_fields:  Dict[str, Any]   # только изменённые поля узла

    @classmethod
    def create(cls, canonical_id: bytes, source: str, timestamp: float,
               source_type: str, trust: float,
               delta_fields: Optional[Dict] = None) -> "ObservationRecord":
        return cls(
            canonical_id=canonical_id,
            obs_id=str(uuid.uuid4()),
            timestamp=timestamp,
            source=source,
            source_type=source_type,
            trust=trust,
            delta_fields=delta_fields or {},
        )

__all__ = ['DataLoader', 'TorchDataset', 'TRANSFORMERS_AVAILABLE', 'FAISS_AVAILABLE', '_DEFAULT_DOMAINS', '_DEFAULT_GROUPS', '_DEFAULT_ENTITY_TYPES', 'LookupTable', 'DOMAIN_TABLE', 'GROUP_TABLE', 'ENTITY_TYPE_TABLE', 'save_lookup_tables', 'load_lookup_tables', 'AliasTable', 'ALIAS_TABLE', 'SemanticHasher', 'SEMANTIC_HASHER', 'CanonicalKey', 'ObservationRecord']
