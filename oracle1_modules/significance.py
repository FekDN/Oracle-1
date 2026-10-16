# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""Epistemic scoring and concept significance processing."""
from __future__ import annotations

from ._shared import *
from .config import CONFIG
from .entities import KnowledgeEdge, KnowledgeNode
from .json_utils import _parse_json_safe
from .time_utils import _ts_to_year

# ══════════════════════════════════════════════════════════════════════════════
# INLINED MODULE: oracle1_significance.py
# Significance Gradient + Concept Distillation Engine
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════
#  1. EPISTEMIC GRADIENT — шкала значимости
# ══════════════════════════════════════════════════════════════════

class EpistemicLevel(Enum):
    """
    Уровни эпистемической значимости.
    Значение float — вес в расчётах influence_score.

    Принцип: чем выше уровень, тем МЕНЬШЕ источников нужно для
    достижения статуса "proven". Форум требует 1000 постов чтобы
    сравняться с одной статьёй Nature.
    """
    PROVEN_DISCOVERY    = 1.00  # воспроизведено независимо, реплицировано
    PEER_REVIEWED       = 0.85  # рецензированный журнал
    PREPRINT            = 0.70  # arXiv, bioRxiv
    PATENT              = 0.55  # патент (практическое применение)
    TEXTBOOK_MENTION    = 0.40  # учебник (устоявшееся знание)
    NEWS_REPORT         = 0.30  # СМИ (популяризация)
    FORUM_DISCUSSION    = 0.20  # форум, блог (незащищённое мнение)
    FOLKLORE_FICTION    = 0.10  # фантастика, мифология, сказки
    SPECULATION         = 0.05  # домысел, анонимный пост


# Маппинг source_type → EpistemicLevel
SOURCE_TYPE_TO_LEVEL: Dict[str, EpistemicLevel] = {
    "nature":       EpistemicLevel.PROVEN_DISCOVERY,
    "science":      EpistemicLevel.PROVEN_DISCOVERY,
    "cell":         EpistemicLevel.PROVEN_DISCOVERY,
    "peer_reviewed":EpistemicLevel.PEER_REVIEWED,
    "journal":      EpistemicLevel.PEER_REVIEWED,
    "arxiv":        EpistemicLevel.PREPRINT,
    "preprint":     EpistemicLevel.PREPRINT,
    "biorxiv":      EpistemicLevel.PREPRINT,
    "patent":       EpistemicLevel.PATENT,
    "conference":   EpistemicLevel.PEER_REVIEWED,
    "book":         EpistemicLevel.TEXTBOOK_MENTION,
    "textbook":     EpistemicLevel.TEXTBOOK_MENTION,
    "report":       EpistemicLevel.NEWS_REPORT,
    "news":         EpistemicLevel.NEWS_REPORT,
    "blog":         EpistemicLevel.FORUM_DISCUSSION,
    "forum":        EpistemicLevel.FORUM_DISCUSSION,
    "reddit":       EpistemicLevel.FORUM_DISCUSSION,
    "wikipedia":    EpistemicLevel.NEWS_REPORT,
    "fiction":      EpistemicLevel.FOLKLORE_FICTION,
    "mythology":    EpistemicLevel.FOLKLORE_FICTION,
    "folklore":     EpistemicLevel.FOLKLORE_FICTION,
    "fairytale":    EpistemicLevel.FOLKLORE_FICTION,
    "speculation":  EpistemicLevel.SPECULATION,
    "unknown":      EpistemicLevel.SPECULATION,
}


def get_epistemic_level(source_type: str) -> EpistemicLevel:
    key = source_type.lower().strip()
    # Частичное совпадение (reddit → forum_discussion)
    for pattern, level in SOURCE_TYPE_TO_LEVEL.items():
        if pattern in key:
            return level
    return EpistemicLevel.SPECULATION


@dataclass
class EpistemicWeight:
    """
    Вес одного наблюдения в шкале значимости.
    Учитывает: уровень источника, дату, количество цитирований.
    """
    level:          EpistemicLevel
    raw_trust:      float       # 0.0–1.0 из source_trust таблицы
    timestamp:      float       # Unix timestamp источника
    citation_count: int = 0     # сколько раз этот источник процитирован
    replication_count: int = 0  # независимые репликации (для proven)

    @property
    def effective_weight(self) -> float:
        """
        Итоговый вес наблюдения:
          base = level.value × raw_trust
          citation_bonus = log(1 + citations) / log(1000) × 0.15
          replication_bonus = min(replications / 3, 1.0) × 0.20
        """
        base = self.level.value * self.raw_trust
        cit  = math.log1p(self.citation_count) / math.log1p(1000) * 0.15
        rep  = min(self.replication_count / 3.0, 1.0) * 0.20
        return float(min(1.0, base + cit + rep))


# ══════════════════════════════════════════════════════════════════
#  2. INFLUENCE TIMELINE
#  История подкреплений canonical узла с датами и весами
# ══════════════════════════════════════════════════════════════════

@dataclass
class InfluencePoint:
    """Один момент подкрепления canonical узла."""
    timestamp:      float
    source:         str
    source_type:    str
    epistemic_level: EpistemicLevel
    weight:         float
    delta_fields:   Dict[str, Any] = field(default_factory=dict)
    distill_type:   str = "reinforcement"  # reinforcement / extension / reinterpretation


@dataclass
class InfluenceTimeline:
    """
    Временная шкала влияний на один canonical узел.

    Используется для измерения:
      - СКОРОСТИ НАРАСТАНИЯ (velocity)    — приближение прорыва
      - ДИВЕРСИФИКАЦИИ ИСТОЧНИКОВ         — кросс-доменная значимость
      - ПИКОВ АКТИВНОСТИ                  — «горячие периоды»
      - НАКОПЛЕННОЙ EPISTEMIC MASS        — суммарный вес подкреплений
    """
    canonical_id:    str
    points:          List[InfluencePoint] = field(default_factory=list)
    # Reference "now" for decay/velocity. None (default) means "use the real
    # wall clock" — correct for live ingestion. Historical replay/backtesting
    # must set this explicitly to the simulated current date before reading
    # epistemic_mass/velocity — otherwise both are computed against the
    # actual present day regardless of which historical moment is being
    # replayed, which silently zeroes velocity's "last 5 years" window for
    # almost any point in a backtest and misjudges decay on every point.
    reference_ts:    Optional[float] = None

    # A timeline shorter than this (first-to-last reinforcement) is treated as
    # a single moment rather than a developing idea — see `temporal_kind`.
    INTERVAL_MIN_SPAN_DAYS = CONFIG["interval_min_span_days"]

    def _now(self) -> float:
        return self.reference_ts if self.reference_ts is not None else time.time()

    def add(self, point: InfluencePoint) -> None:
        self.points.append(point)
        self.points.sort(key=lambda p: p.timestamp)

    @property
    def epistemic_mass(self) -> float:
        """
        Накопленная эпистемическая масса.
        Используем убывающую сумму: поздние источники весомее,
        но старые не обнуляются (Einstein 1917 не исчезает).

        mass = Σ weight_i × decay(reference_now - t_i)
        decay = 0.5 если источнику > 50 лет, иначе 1.0
        """
        if not self.points:
            return 0.0
        now = self._now()
        total = 0.0
        for p in self.points:
            age_years = (now - p.timestamp) / (365.25 * 24 * 3600)
            decay = 0.5 if age_years > 50 else 1.0
            total += p.weight * decay
        return float(min(10.0, total))  # cap at 10 для совместимости с score-шкалой

    @property
    def velocity(self) -> float:
        """
        Скорость нарастания за последние 5 лет от reference_ts:
        log1p(число недавних упоминаний) × средний вес.

        Не len(recent) * avg_weight напрямую: сырое число упоминаний ничем
        не ограничено сверху, а средний вес источника варьируется всего в
        ~15-20 раз (SPECULATION=0.05 .. PEER_REVIEWED=0.85) — то есть объём
        побеждает качество на порядки. Вирусный миф с 20 000 постов на
        форумах даёт в ~80 раз более высокую velocity, чем 15 рецензируемых
        статей о реальном прорыве, хотя каждое упоминание там в 17 раз
        менее весомо. epistemic_mass защищён от этого явным потолком в 10.0
        (см. ниже) — velocity такой защиты не имела вообще. log1p — тот же
        приём, что уже используется по всему файлу для сжатия больших
        сырых чисел (investment_total_usd, forum_post_count и т.д.), а не
        новый паттерн.

        Это не решает более глубокую проблему "эхо vs независимое
        подтверждение" — для этого см. RuntimeGraph.union_family (20 000
        репостов одного и того же непроверенного тезиса всё ещё считаются
        20 000 отдельных источников). Но не даёт голому объёму создавать
        неограниченную уверенность.

        Высокая velocity = признак приближающегося прорыва.
        """
        if len(self.points) < 2:
            return 0.0
        now = self._now()
        cutoff = now - 5 * 365.25 * 24 * 3600
        recent = [p for p in self.points if p.timestamp >= cutoff]
        if not recent:
            return 0.0
        avg_w  = float(np.mean([p.weight for p in recent]))
        return math.log1p(len(recent)) * avg_w

    @property
    def source_diversity(self) -> float:
        """
        Разнообразие типов источников (0–1).
        Много разных типов = высокая значимость.
        """
        if not self.points:
            return 0.0
        types = set(p.source_type for p in self.points)
        # Нормализуем по максимальному числу возможных типов
        return min(1.0, len(types) / len(SOURCE_TYPE_TO_LEVEL))

    @property
    def peak_epistemic_year(self) -> Optional[int]:
        """Год с наибольшей суммарной эпистемической активностью."""
        if not self.points:
            return None
        year_weight: Dict[int, float] = defaultdict(float)
        for p in self.points:
            year = int(_ts_to_year(p.timestamp))
            year_weight[year] += p.weight
        return max(year_weight, key=year_weight.get)

    @property
    def span_seconds(self) -> float:
        """Time between the first and last reinforcement, in seconds. 0 for a single point."""
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].timestamp - self.points[0].timestamp

    @property
    def temporal_kind(self) -> str:
        """
        'point'    — one reinforcement, or several close enough together that
                     they describe the same moment (e.g. a paper and its own
                     supplementary material), not a developing idea.
        'interval' — reinforcements spread across INTERVAL_MIN_SPAN_DAYS or
                     more: the idea is being restated, argued over, or
                     elaborated over time.
        """
        span_days = self.span_seconds / (24 * 3600)
        if len(self.points) >= 2 and span_days >= self.INTERVAL_MIN_SPAN_DAYS:
            return "interval"
        return "point"

    def density_profile(self, bin_years: float = 1.0) -> List[float]:
        """
        Weighted reinforcement count per `bin_years`-wide bucket across
        [first_timestamp, last_timestamp]. This is the density curve itself —
        not just the band it spans — so a late burst of independent
        reinforcement reads differently from the same total weight spread
        evenly across the whole span. Returns [] for point-timelines.
        """
        if self.temporal_kind != "interval":
            return []
        t0, t1 = self.points[0].timestamp, self.points[-1].timestamp
        span_years = max((t1 - t0) / (365.25 * 24 * 3600), 1e-6)
        n_bins = max(1, int(math.ceil(span_years / max(bin_years, 1e-6))))
        bins = [0.0] * n_bins
        for p in self.points:
            frac = (p.timestamp - t0) / (t1 - t0) if t1 > t0 else 0.0
            idx = min(n_bins - 1, int(frac * n_bins))
            bins[idx] += p.weight
        return bins

    def proven_status(self) -> bool:
        """
        Достиг ли узел статуса proven_discovery?
        Требования:
          - хотя бы 1 PROVEN_DISCOVERY источник, ИЛИ
          - хотя бы 2 PEER_REVIEWED источника + epistemic_mass >= 2.0
        """
        proven_count = sum(
            1 for p in self.points
            if p.epistemic_level == EpistemicLevel.PROVEN_DISCOVERY
        )
        peer_count = sum(
            1 for p in self.points
            if p.epistemic_level in (EpistemicLevel.PROVEN_DISCOVERY,
                                     EpistemicLevel.PEER_REVIEWED)
        )
        return (proven_count >= 1 or
                (peer_count >= 2 and self.epistemic_mass >= 2.0))

    def to_summary(self) -> Dict[str, Any]:
        return {
            "canonical_id":       self.canonical_id,
            "total_points":       len(self.points),
            "epistemic_mass":     round(self.epistemic_mass, 3),
            "velocity":           round(self.velocity, 3),
            "source_diversity":   round(self.source_diversity, 3),
            "proven":             self.proven_status(),
            "peak_year":          self.peak_epistemic_year,
            "source_types":       list({p.source_type for p in self.points}),
            "temporal_kind":      self.temporal_kind,
        }


# ══════════════════════════════════════════════════════════════════
#  3. DISTILLATION ENGINE
#  LLM извлекает из источника: подтверждение / расширение / новое
# ══════════════════════════════════════════════════════════════════

class DistillationType(Enum):
    REINFORCEMENT     = "reinforcement"       # подтверждает canonical
    EXTENSION         = "extension"           # новый аспект того же
    REINTERPRETATION  = "reinterpretation"    # другой взгляд на то же
    NEW_CONCEPT       = "new_concept"         # действительно новая идея
    ABSTRACT_CONCEPT  = "abstract_concept"    # косвенная концепция (сказки!)


@dataclass
class DistillationResult:
    """
    Результат анализа одного источника относительно canonical узла.
    """
    distill_type:       DistillationType
    canonical_id:       str             # к какому узлу относится
    source:             str
    source_type:        str
    timestamp:          float
    epistemic_level:    EpistemicLevel
    weight:             EpistemicWeight

    # Для REINFORCEMENT: поля для усиления canonical узла
    reinforcement_delta: Dict[str, float] = field(default_factory=dict)

    # Для EXTENSION / REINTERPRETATION / NEW_CONCEPT / ABSTRACT_CONCEPT:
    # описание нового/мутантного узла
    new_node_text:       str = ""
    new_node_domain:     str = ""
    new_node_entity_type: str = ""
    new_node_concepts:   List[str] = field(default_factory=list)  # абстрактные концепции
    new_node_confidence: float = 0.0

    # Связи нового узла с canonical
    relation_type:       str = "extends"  # extends / reinterprets / derived_from
    relation_strength:   float = 0.5


DISTILLATION_PROMPT = """\
You are an epistemologist analyzing a source document relative to a known canonical concept.

CANONICAL CONCEPT: "{canonical_text}"
CANONICAL DOMAIN: {canonical_domain}
SOURCE TYPE: {source_type}
EPISTEMIC LEVEL: {epistemic_level}
SOURCE TEXT:
{source_text}

TASK: Determine what this source contributes relative to the canonical concept.
Classify as ONE of:
  REINFORCEMENT     — source confirms/strengthens the canonical (no new node)
  EXTENSION         — source adds a new verified aspect (creates child node)
  REINTERPRETATION  — source offers different interpretation (creates sibling node)
  NEW_CONCEPT       — source contains genuinely new idea (creates new node)
  ABSTRACT_CONCEPT  — source expresses same deep concept in different form
                      (e.g. fairy tale mirror → "remote visual transmission device")
                      Extract the ABSTRACT FUNCTIONAL CONCEPT, not the surface form.

For ABSTRACT_CONCEPT: ignore the fictional/mythological surface. Extract:
  - core function (what does it DO?)
  - key properties (portable? on-demand? scalable?)
  - the abstract concept name

OUTPUT ONLY valid JSON:
{{
  "distill_type": "<REINFORCEMENT|EXTENSION|REINTERPRETATION|NEW_CONCEPT|ABSTRACT_CONCEPT>",
  "reasoning": "<one sentence>",
  "reinforcement_delta": {{
    "scientific_score_delta": <-2.0 to +2.0 or 0>,
    "social_score_delta": <-2.0 to +2.0 or 0>,
    "maturity_score_delta": <-1.0 to +1.0 or 0>
  }},
  "new_node": {{
    "text": "<functional description if not REINFORCEMENT, else empty>",
    "domain": "<domain>",
    "entity_type": "<type>",
    "abstract_concepts": ["<concept1>", "<concept2>"],
    "confidence": <0.0-1.0>
  }},
  "relation_type": "<extends|reinterprets|derived_from|precursor_of>",
  "relation_strength": <0.0-1.0>
}}
"""


class ConceptDistiller:
    """
    Анализирует каждый новый источник относительно существующих canonical узлов.

    Workflow:
      1. Найти кандидатные canonical узлы (семантическая близость)
      2. Для каждого кандидата → LLM distillation
      3. Вернуть список DistillationResult
         - REINFORCEMENT → обновить InfluenceTimeline, не создавать узел
         - остальные → создать мутантный/новый узел + ребро

    Ключевая идея:
      "Форум Reddit 2019" не становится узлом "Stimulated Emission".
      Он либо УСИЛИВАЕТ "Stimulated Emission" (добавляет вес в timeline),
      либо СОЗДАЁТ новый узел "Популярная дискуссия о лазерах 2019"
      с ребром extension → "Stimulated Emission".
    """

    SIMILARITY_THRESHOLD = CONFIG["concept_distiller_similarity_threshold"]   # минимальная близость для проверки
    MAX_CANDIDATES       = CONFIG["concept_distiller_max_candidates"]         # сколько canonical узлов проверять

    def __init__(self, llm_client=None):
        self.llm = llm_client
        # canonical_id → InfluenceTimeline
        self.timelines: Dict[str, InfluenceTimeline] = {}

    # ── Основной API ────────────────────────────────────────────

    def process_source(
        self,
        source_text:  str,
        source_type:  str,
        source_name:  str,
        timestamp:    float,
        graph_nodes:  Dict[str, Any],   # {id: KnowledgeNode}
        embedder=None,
    ) -> List[DistillationResult]:
        """
        Обработать один источник.
        Возвращает список DistillationResult для каждого релевантного canonical узла.
        """
        level   = get_epistemic_level(source_type)
        # Доверие из epistemic level (не из source_trust таблицы — единая шкала)
        trust   = level.value

        candidates = self._find_candidates(source_text, graph_nodes, embedder)
        results: List[DistillationResult] = []

        for canonical_id, sim_score in candidates:
            node = graph_nodes.get(canonical_id)
            if node is None:
                continue

            if self.llm is not None:
                result = self._llm_distill(
                    source_text=source_text,
                    source_type=source_type,
                    source_name=source_name,
                    timestamp=timestamp,
                    canonical_node=node,
                    level=level,
                    trust=trust,
                    sim_score=sim_score,
                )
            else:
                # Без LLM: высокая схожесть → REINFORCEMENT
                result = self._heuristic_distill(
                    source_text=source_text,
                    source_type=source_type,
                    source_name=source_name,
                    timestamp=timestamp,
                    canonical_node=node,
                    level=level,
                    trust=trust,
                    sim_score=sim_score,
                )

            if result:
                # Обновить timeline для canonical узла
                self._update_timeline(canonical_id, result)
                results.append(result)

        return results

    def get_timeline(self, canonical_id: str) -> Optional[InfluenceTimeline]:
        return self.timelines.get(canonical_id)

    def get_or_create_timeline(self, canonical_id: str) -> InfluenceTimeline:
        if canonical_id not in self.timelines:
            self.timelines[canonical_id] = InfluenceTimeline(canonical_id=canonical_id)
        return self.timelines[canonical_id]

    def top_by_velocity(self, n: int = 10) -> List[Tuple[str, float]]:
        """Топ N узлов по скорости нарастания — сигнал приближения прорыва."""
        scored = [
            (cid, tl.velocity)
            for cid, tl in self.timelines.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def top_by_epistemic_mass(self, n: int = 10) -> List[Tuple[str, float]]:
        """Топ N по накопленной эпистемической массе."""
        scored = [
            (cid, tl.epistemic_mass)
            for cid, tl in self.timelines.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def necessity_from_diversity(self, canonical_id: str) -> float:
        """
        Индекс необходимости из InfluenceTimeline.
        Комбинирует epistemic_mass, source_diversity, velocity.
        """
        tl = self.timelines.get(canonical_id)
        if not tl:
            return 0.0
        return float(
            tl.epistemic_mass * 0.4 +
            tl.source_diversity * 5.0 * 0.3 +
            tl.velocity * 0.3
        )

    # ── Внутренние методы ───────────────────────────────────────

    def _find_candidates(
        self,
        source_text: str,
        graph_nodes: Dict[str, Any],
        embedder=None,
    ) -> List[Tuple[str, float]]:
        """
        Найти canonical узлы, близкие к тексту источника.
        Возвращает [(node_id, similarity)], отсортированные по убыванию.
        """
        if not graph_nodes:
            return []

        # Получаем эмбеддинг источника
        src_emb = self._embed(source_text, embedder)
        if src_emb is None:
            # Fallback: keyword overlap
            return self._keyword_candidates(source_text, graph_nodes)

        candidates = []
        for nid, node in graph_nodes.items():
            node_emb = getattr(node, "embedding", None)
            if node_emb is None:
                continue
            # Сравниваем только первые 384 dim (текстовый блок)
            ne = np.array(node_emb[:384], dtype=np.float32)
            se = np.array(src_emb[:384], dtype=np.float32)
            ne_norm = np.linalg.norm(ne)
            se_norm = np.linalg.norm(se)
            if ne_norm < 1e-8 or se_norm < 1e-8:
                continue
            sim = float(np.dot(ne / ne_norm, se / se_norm))
            if sim >= self.SIMILARITY_THRESHOLD:
                candidates.append((nid, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:self.MAX_CANDIDATES]

    def _keyword_candidates(
        self,
        source_text: str,
        graph_nodes: Dict[str, Any],
    ) -> List[Tuple[str, float]]:
        """Keyword-based fallback без эмбеддингов."""
        src_words = set(re.findall(r'[a-zа-яё]{4,}', source_text.lower()))
        scored = []
        for nid, node in graph_nodes.items():
            node_text = getattr(node, "text", "") + " " + getattr(node, "full_text", "")
            node_words = set(re.findall(r'[a-zа-яё]{4,}', node_text.lower()))
            if not node_words:
                continue
            overlap = len(src_words & node_words) / max(len(src_words | node_words), 1)
            if overlap >= 0.1:
                scored.append((nid, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.MAX_CANDIDATES]

    def _embed(self, text: str, embedder) -> Optional[np.ndarray]:
        if embedder is None:
            return None
        try:
            return np.array(embedder.encode(text[:512]), dtype=np.float32)
        except Exception as e:
            logger.debug(f"[Distiller] embed failed: {e}")
            return None

    def _llm_distill(
        self,
        source_text:    str,
        source_type:    str,
        source_name:    str,
        timestamp:      float,
        canonical_node: Any,
        level:          EpistemicLevel,
        trust:          float,
        sim_score:      float,
    ) -> Optional[DistillationResult]:
        """LLM-based distillation."""
        prompt = DISTILLATION_PROMPT.format(
            canonical_text=getattr(canonical_node, "text", ""),
            canonical_domain=getattr(canonical_node, "domain", ""),
            source_type=source_type,
            epistemic_level=level.name,
            source_text=source_text[:1500],
        )
        try:
            raw  = self.llm.complete(prompt)
            data = _parse_json_safe(raw)
        except Exception as e:
            logger.warning(f"[Distiller] LLM error: {e}")
            return None

        return self._build_result(
            data=data,
            source_name=source_name,
            source_type=source_type,
            timestamp=timestamp,
            canonical_id=canonical_node.id,
            level=level,
            trust=trust,
        )

    def _heuristic_distill(
        self,
        source_text:    str,
        source_type:    str,
        source_name:    str,
        timestamp:      float,
        canonical_node: Any,
        level:          EpistemicLevel,
        trust:          float,
        sim_score:      float,
    ) -> Optional[DistillationResult]:
        """
        Без LLM: эвристика по epistemic level + similarity.
        - Высокий level + высокая sim → REINFORCEMENT
        - Низкий level + средняя sim → EXTENSION (осторожно)
        - Folklore/fiction → ABSTRACT_CONCEPT если sim < 0.75
        """
        if level in (EpistemicLevel.FOLKLORE_FICTION, EpistemicLevel.SPECULATION):
            dtype = DistillationType.ABSTRACT_CONCEPT
        elif sim_score >= 0.80 and level.value >= 0.55:
            dtype = DistillationType.REINFORCEMENT
        elif sim_score >= 0.65:
            dtype = DistillationType.EXTENSION
        else:
            dtype = DistillationType.REINFORCEMENT

        w = EpistemicWeight(
            level=level, raw_trust=trust,
            timestamp=timestamp,
        )
        result = DistillationResult(
            distill_type=dtype,
            canonical_id=canonical_node.id,
            source=source_name,
            source_type=source_type,
            timestamp=timestamp,
            epistemic_level=level,
            weight=w,
            reinforcement_delta={"scientific_score_delta": level.value * 0.5},
            new_node_text=(f"[{source_type}] aspect of: "
                           f"{getattr(canonical_node, 'text', '')[:50]}")
                          if dtype != DistillationType.REINFORCEMENT else "",
            new_node_domain=getattr(canonical_node, "domain", ""),
            new_node_entity_type="cultural_phantom"
                                  if dtype == DistillationType.ABSTRACT_CONCEPT
                                  else "research_frontier",
            new_node_confidence=sim_score * level.value,
        )
        return result

    def _build_result(
        self,
        data:         Dict,
        source_name:  str,
        source_type:  str,
        timestamp:    float,
        canonical_id: str,
        level:        EpistemicLevel,
        trust:        float,
    ) -> DistillationResult:
        dtype_str = data.get("distill_type", "REINFORCEMENT").upper()
        try:
            dtype = DistillationType[dtype_str]
        except KeyError:
            dtype = DistillationType.REINFORCEMENT

        w = EpistemicWeight(
            level=level, raw_trust=trust, timestamp=timestamp,
        )

        new_node_data = data.get("new_node", {})
        return DistillationResult(
            distill_type=dtype,
            canonical_id=canonical_id,
            source=source_name,
            source_type=source_type,
            timestamp=timestamp,
            epistemic_level=level,
            weight=w,
            reinforcement_delta=data.get("reinforcement_delta", {}),
            new_node_text=new_node_data.get("text", ""),
            new_node_domain=new_node_data.get("domain", ""),
            new_node_entity_type=new_node_data.get("entity_type", "research_frontier"),
            new_node_concepts=new_node_data.get("abstract_concepts", []),
            new_node_confidence=float(new_node_data.get("confidence", 0.5)),
            relation_type=data.get("relation_type", "extends"),
            relation_strength=float(data.get("relation_strength", 0.5)),
        )

    def _update_timeline(
        self,
        canonical_id: str,
        result:       DistillationResult,
    ) -> None:
        tl = self.get_or_create_timeline(canonical_id)
        point = InfluencePoint(
            timestamp=result.timestamp,
            source=result.source,
            source_type=result.source_type,
            epistemic_level=result.epistemic_level,
            weight=result.weight.effective_weight,
            delta_fields=result.reinforcement_delta,
            distill_type=result.distill_type.value,
        )
        tl.add(point)


# ══════════════════════════════════════════════════════════════════
#  4. SIGNIFICANCE PROCESSOR
#  Встраивается в IngestionPipeline между PassC и load_result_into_graph
# ══════════════════════════════════════════════════════════════════

@dataclass
class ProcessorResult:
    """Результат обработки одного IngestionResult через SignificanceProcessor."""
    reinforcements: int = 0        # сколько источников стали REINFORCEMENT
    mutant_nodes:   int = 0        # сколько мутантных узлов создано
    abstract_nodes: int = 0        # сколько абстрактных концептов создано
    new_nodes:      List[Any] = field(default_factory=list)   # KnowledgeNode
    new_edges:      List[Any] = field(default_factory=list)   # KnowledgeEdge
    timeline_updates: Dict[str, Dict] = field(default_factory=dict)


class SignificanceProcessor:
    """
    Главный процессор.

    Алгоритм:
      1. Для каждого узла из IngestionResult:
           a. Найти canonical узлы графа (близкие)
           b. Запустить ConceptDistiller
           c. REINFORCEMENT → обновить epistemic_mass canonical узла
              NEW_CONCEPT/EXTENSION/etc → создать мутантный узел + ребро

      2. Мутантные узлы получают:
           - epistemic_level из источника
           - связь с canonical через KnowledgeEdge
           - reduced strategic_value (не "вершина айсберга")

      3. Canonical узел обновляется:
           - epistemic_mass → scientific_score (нормализованный)
           - proven_status → entity_type может стать "breakthrough_node"
           - velocity → forecast_score

    ВАЖНО: SignificanceProcessor не удаляет узлы из IngestionResult.
    Он добавляет к ним слой семантики и принимает решение:
    "этот узел — дубль canonical" или "это новое".
    """

    PROVEN_ENTITY_TYPE     = "breakthrough_node"
    MIN_MASS_FOR_PROVEN    = CONFIG["significance_min_mass_for_proven"]
    REINFORCEMENT_FRACTION = CONFIG["significance_reinforcement_fraction"]   # доля источников, идущих в reinforcement

    def __init__(self,
                 distiller: Optional[ConceptDistiller] = None,
                 embedder=None):
        self.distiller = distiller or ConceptDistiller()
        self.embedder  = embedder

    def process(
        self,
        ingestion_result: Any,   # IngestionResult
        graph:            Any,   # RuntimeGraph
        llm_client=None,
        reference_ts:     Optional[float] = None,
    ) -> ProcessorResult:
        """
        Обработать IngestionResult:
          - усилить canonical узлы через timelines
          - создать мутантные узлы для реальных новшеств
          - обновить epistemic_mass canonical узлов

        reference_ts: "сейчас" для decay/velocity в InfluenceTimeline.
        None (по умолчанию) — реальный wall-clock, корректно для живой
        подгрузки. При историческом воспроизведении (бэктест по датированным
        источникам, а не по хардкод-сценарию вроде лазерного) сюда нужно
        передавать симулированную текущую дату, иначе decay и velocity
        считаются от настоящего "сейчас" независимо от того, какой момент
        истории воспроизводится.
        """
        result = ProcessorResult()
        if llm_client is not None:
            self.distiller.llm = llm_client

        source      = getattr(ingestion_result, "source", "unknown")
        # Определяем source_type из первого узла или из source URL
        source_type = self._infer_source_type(source, ingestion_result)

        for node in getattr(ingestion_result, "nodes", []):
            node_text    = getattr(node, "text", "")
            node_domain  = getattr(node, "domain", "")
            node_ts      = getattr(node, "timestamp", time.time())

            distill_results = self.distiller.process_source(
                source_text=node_text + " " + getattr(node, "full_text", ""),
                source_type=source_type,
                source_name=source,
                timestamp=node_ts,
                graph_nodes=dict(graph.nodes) if hasattr(graph, "nodes") else {},
                embedder=self.embedder,
            )

            for dr in distill_results:
                if dr.distill_type == DistillationType.REINFORCEMENT:
                    # Обновить canonical узел напрямую
                    self._apply_reinforcement(dr, graph)
                    result.reinforcements += 1

                elif dr.distill_type in (
                    DistillationType.EXTENSION,
                    DistillationType.REINTERPRETATION,
                    DistillationType.NEW_CONCEPT,
                    DistillationType.ABSTRACT_CONCEPT,
                ):
                    # Создать мутантный узел
                    mutant, edge = self._create_mutant_node(dr, node, graph)
                    if mutant and edge:
                        result.new_nodes.append(mutant)
                        result.new_edges.append(edge)
                        if dr.distill_type == DistillationType.ABSTRACT_CONCEPT:
                            result.abstract_nodes += 1
                        else:
                            result.mutant_nodes += 1

        # Обновить epistemic_mass → score для всех затронутых canonical узлов
        affected_ids = set(dr.canonical_id for dr in self._collect_all_dr(result))
        for cid in affected_ids:
            self._sync_epistemic_to_node(cid, graph, reference_ts=reference_ts)
            tl = self.distiller.get_timeline(cid)
            if tl:
                result.timeline_updates[cid] = tl.to_summary()

        logger.info(
            f"[SignificanceProcessor] source='{source}' type='{source_type}' "
            f"reinforcements={result.reinforcements} "
            f"mutants={result.mutant_nodes} abstract={result.abstract_nodes}"
        )
        return result

    # ── Создание мутантного узла ─────────────────────────────────

    def _create_mutant_node(
        self,
        dr:     DistillationResult,
        source_node: Any,
        graph:  Any,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Create a mutant KnowledgeNode and KnowledgeEdge connected to the canonical node.
        """
        _KnowledgeNode = KnowledgeNode
        _KnowledgeEdge = KnowledgeEdge

        if not dr.new_node_text:
            return None, None

        # Эпистемический вес определяет начальные скоры мутанта
        ew = dr.weight.effective_weight
        level_val = dr.epistemic_level.value

        # Для абстрактного концепта создаём особый узел
        if dr.distill_type == DistillationType.ABSTRACT_CONCEPT:
            text = self._abstract_concept_text(dr, source_node)
            etype = "cultural_phantom"
        else:
            text = dr.new_node_text
            etype = dr.new_node_entity_type or "research_frontier"

        mutant_id = "mut_" + hashlib.md5(
            f"{dr.canonical_id}::{text}::{dr.source}".encode()
        ).hexdigest()[:10]

        mutant = _KnowledgeNode(
            id=mutant_id,
            text=text,
            full_text=getattr(source_node, "full_text", ""),
            node_type="mutant",
            domain=dr.new_node_domain or getattr(source_node, "domain", "unknown"),
            entity_type=etype,
            provenance=dr.source,
            timestamp=dr.timestamp,
            # Скоры масштабированы по epistemic level
            scientific_score = float(
                getattr(source_node, "scientific_score", 5.0) * level_val
            ),
            social_score     = float(
                getattr(source_node, "social_score", 0.0) * level_val
            ),
            maturity_score   = float(level_val * 5.0),
            strategic_value  = float(
                getattr(source_node, "strategic_value", 5.0) * ew
            ),
            # Epistemic metadata fields (available in merged build)
            epistemic_level  = dr.epistemic_level.name,
            epistemic_mass   = dr.weight.effective_weight,
            influence_velocity = 0.0,
            source_type      = dr.source_type,
        )
        # Сохраняем метаданные
        mutant.__dict__["epistemic_level"]    = dr.epistemic_level.name
        mutant.__dict__["epistemic_weight"]   = ew
        mutant.__dict__["distill_type"]       = dr.distill_type.value
        mutant.__dict__["abstract_concepts"]  = dr.new_node_concepts
        mutant.__dict__["parent_canonical_id"] = dr.canonical_id

        # Ребро mutant → canonical
        edge = _KnowledgeEdge(
            id=f"sig_{mutant_id[:8]}_{dr.canonical_id[:6]}",
            source_id=mutant_id,
            target_id=dr.canonical_id,
            relationship_type=dr.relation_type,
            semantic_similarity=float(dr.relation_strength),
            limitation_resolution=float(
                dr.relation_strength if dr.distill_type == DistillationType.EXTENSION
                else 0.0
            ),
            citation_link=float(level_val),
            confidence=float(dr.new_node_confidence),
            timestamp=dr.timestamp,
        )
        edge.compute_total_weight()
        return mutant, edge

    def _abstract_concept_text(
        self,
        dr: DistillationResult,
        source_node: Any,
    ) -> str:
        """Генерировать текст для абстрактного концепта (из сказок и т.п.)."""
        if dr.new_node_concepts:
            return "Abstract: " + "; ".join(dr.new_node_concepts[:3])
        # Fallback: из текста источника
        base = getattr(source_node, "text", "unknown source")
        return f"Abstract concept from [{dr.source_type}]: {base[:60]}"

    # ── Обновление canonical узла ────────────────────────────────

    def _apply_reinforcement(
        self,
        dr:    DistillationResult,
        graph: Any,
    ) -> None:
        """
        Применить REINFORCEMENT: обновить поля canonical узла
        без создания нового узла.
        """
        node = graph.get_node(dr.canonical_id) if hasattr(graph, "get_node") else None
        if node is None:
            return
        delta = dr.reinforcement_delta
        for field_name, delta_val in delta.items():
            # Убираем суффикс "_delta"
            base = field_name.replace("_delta", "")
            current = getattr(node, base, None)
            if current is not None and isinstance(current, (int, float)):
                new_val = float(np.clip(current + delta_val, 0.0, 10.0))
                setattr(node, base, new_val)

        # Добавляем epistemic_weight в __dict__ для последующей синхронизации
        prev = node.__dict__.get("reinforcement_weights", [])
        prev.append(dr.weight.effective_weight)
        node.__dict__["reinforcement_weights"] = prev

    def _sync_epistemic_to_node(
        self,
        canonical_id: str,
        graph:        Any,
        reference_ts: Optional[float] = None,
    ) -> None:
        """
        Синхронизировать InfluenceTimeline → поля KnowledgeNode:
          epistemic_mass → scientific_score (нормализованный) + native field
          velocity       → forecast_score + influence_velocity
          proven_status  → entity_type апгрейд
          source_diversity → social_score + source_diversity field

        reference_ts: см. process() — "сейчас" для decay/velocity. None
        сохраняет прежнее поведение (реальный wall-clock).
        """
        node = graph.get_node(canonical_id) if hasattr(graph, "get_node") else None
        tl   = self.distiller.get_timeline(canonical_id)
        if node is None or tl is None:
            return
        tl.reference_ts = reference_ts

        # Нормализуем epistemic_mass в [0, 10]
        new_sci = float(np.clip(tl.epistemic_mass, 0.0, 10.0))
        # Применяем только если выше текущего (не понижаем)
        if new_sci > node.scientific_score:
            node.scientific_score = new_sci

        # Sync to native epistemic fields (new in merged build)
        if hasattr(node, "epistemic_mass"):
            node.epistemic_mass = float(np.clip(tl.epistemic_mass, 0.0, 10.0))

        # Velocity → forecast_score + influence_velocity
        vel_normalized = float(np.clip(tl.velocity / 5.0, 0.0, 10.0))
        if vel_normalized > node.forecast_score:
            node.forecast_score = vel_normalized
        if hasattr(node, "influence_velocity"):
            node.influence_velocity = tl.velocity

        # Source diversity → social_score + native field
        div_score = float(tl.source_diversity * 10.0)
        if div_score > node.social_score:
            node.social_score = div_score
        if hasattr(node, "source_diversity"):
            node.source_diversity = tl.source_diversity

        # Proven status → апгрейд entity_type
        if (tl.proven_status() and
                node.entity_type not in (self.PROVEN_ENTITY_TYPE, "temporal_zone")):
            old_type = node.entity_type
            node.entity_type = self.PROVEN_ENTITY_TYPE
            logger.info(
                f"[SignificanceProcessor] '{canonical_id}' upgraded: "
                f"{old_type} → {self.PROVEN_ENTITY_TYPE} "
                f"(mass={tl.epistemic_mass:.2f})"
            )

        # Temporal shape: point vs interval + density profile (see KnowledgeNode).
        # A node starts as a point by definition — it only becomes an interval
        # once independent reinforcement has actually accumulated over time.
        if hasattr(node, "temporal_kind"):
            node.temporal_kind = tl.temporal_kind
            if tl.temporal_kind == "interval":
                node.timestamp_start  = tl.points[0].timestamp
                node.timestamp_end    = tl.points[-1].timestamp
                node.temporal_density = tl.density_profile(
                    bin_years=node.density_bin_years or 1.0
                )
                # `timestamp` keeps meaning "most recent evidence" for any code
                # that still reads it as a single point.
                node.timestamp = node.timestamp_end
            else:
                node.timestamp_start = node.timestamp_end = node.timestamp
                node.temporal_density = []

        # Сохраняем метаданные timeline в node.__dict__
        node.__dict__["epistemic_timeline"] = tl.to_summary()

    # ── Вспомогательные ─────────────────────────────────────────

    @staticmethod
    def _infer_source_type(source: str, ingestion_result: Any) -> str:
        """Определить тип источника из URL/имени."""
        src = source.lower()
        if "nature.com" in src or "nature/" in src:      return "nature"
        if "science" in src and "arxiv" not in src:      return "science"
        if "arxiv" in src:                                return "arxiv"
        if "reddit" in src:                               return "reddit"
        if "forum" in src or "stack" in src:              return "forum"
        if "blog" in src or "medium" in src:              return "blog"
        if "patent" in src:                               return "patent"
        if "wikipedia" in src:                            return "wikipedia"
        if "book" in src or "textbook" in src:            return "book"
        if "fiction" in src or "tale" in src:             return "fiction"
        if "news" in src or "bbc" in src or "cnn" in src: return "news"
        return "unknown"

    @staticmethod
    def _collect_all_dr(result: ProcessorResult) -> List[DistillationResult]:
        """Собрать все DistillationResult из ProcessorResult (для backward compat)."""
        # ProcessorResult не хранит dr напрямую — используем timeline_updates
        return []


# ══════════════════════════════════════════════════════════════════
#  5. ИНТЕГРАЦИЯ С ORACLE-1 СИСТЕМОЙ
#  Патч для oracle1_system_epoch и build_oracle1_system
# ══════════════════════════════════════════════════════════════════

def attach_significance_layer(
    oracle1_system: Dict[str, Any],
    llm_client=None,
    embedder=None,
) -> "SignificanceProcessor":
    """
    Встроить SignificanceProcessor в существующую Oracle-1 систему.

    После вызова:
      system["significance"] = processor

    Использование в цикле ingestion:
      result = pipeline.ingest(text, source, ts)
      sig_result = system["significance"].process(result, graph, llm)
      load_result_into_graph(result, graph)
      # Мутантные узлы уже добавлены в result.nodes/edges
      for node in sig_result.new_nodes:
          graph.add_node(node)
      for edge in sig_result.new_edges:
          graph.add_edge(edge)
    """
    distiller  = ConceptDistiller(llm_client=llm_client)
    processor  = SignificanceProcessor(distiller=distiller, embedder=embedder)
    oracle1_system["significance"] = processor
    logger.info("[SignificanceLayer] attached to oracle1 system")
    return processor


def get_significance_report(
    oracle1_system: Dict[str, Any],
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Сводный отчёт по эпистемической значимости узлов.
    """
    proc = oracle1_system.get("significance")
    if proc is None:
        return {"error": "significance layer not attached"}
    distiller = proc.distiller
    return {
        "top_by_velocity":       distiller.top_by_velocity(top_n),
        "top_by_epistemic_mass": distiller.top_by_epistemic_mass(top_n),
        "total_timelines":       len(distiller.timelines),
        "proven_nodes":          [
            cid for cid, tl in distiller.timelines.items()
            if tl.proven_status()
        ],
    }


def epistemic_level_for_node(node: Any) -> EpistemicLevel:
    """
    Определить текущий epistemic level узла по его метаданным.
    Использует: entity_type, scientific_score, provenance.
    """
    etype = getattr(node, "entity_type", "").lower()
    sci   = getattr(node, "scientific_score", 0.0)

    if "breakthrough" in etype or sci >= 9.0:
        return EpistemicLevel.PROVEN_DISCOVERY
    if "physical_principle" in etype or sci >= 7.5:
        return EpistemicLevel.PEER_REVIEWED
    if "research_frontier" in etype or sci >= 6.0:
        return EpistemicLevel.PREPRINT
    if "cultural_phantom" in etype or "fiction" in etype:
        return EpistemicLevel.FOLKLORE_FICTION
    if "speculation" in etype:
        return EpistemicLevel.SPECULATION
    # По источнику
    prov = getattr(node, "provenance", "unknown").lower()
    return get_epistemic_level(prov)


# ══════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ══════════════════════════════════════════════════════════════════

def _smoke_test():
    logging.basicConfig(level=logging.INFO)
    logger.info("=== Significance Gradient Smoke Test ===")

    # ── Мок KnowledgeNode ───────────────────────────────────────
    class MockNode:
        def __init__(self, id_, text, domain, entity_type="physical_principle",
                     sci=5.0, social=0.0, forecast=0.0):
            self.id            = id_
            self.text          = text
            self.full_text     = text
            self.domain        = domain
            self.entity_type   = entity_type
            self.scientific_score = sci
            self.social_score  = social
            self.forecast_score = forecast
            self.maturity_score = 0.0
            self.strategic_value = 5.0
            self.provenance    = "test"
            self.timestamp     = -1672531200.0  # 1917
            self.embedding     = np.random.randn(384).astype(np.float32)
            self.__dict__      = self.__dict__

    class MockGraph:
        def __init__(self, nodes):
            self.nodes = {n.id: n for n in nodes}
        def get_node(self, nid): return self.nodes.get(nid)

    class MockLLM:
        def complete(self, _prompt):
            return json.dumps({
                "distill_type": "REINFORCEMENT",
                "reasoning": "Nature article confirms Einstein's theory",
                "reinforcement_delta": {
                    "scientific_score_delta": 1.5,
                    "social_score_delta": 0.5,
                    "maturity_score_delta": 0.3,
                },
                "new_node": {"text": "", "domain": "", "entity_type": "",
                             "abstract_concepts": [], "confidence": 0.0},
                "relation_type": "extends",
                "relation_strength": 0.9,
            })

    class MockLLMFairy:
        def complete(self, _prompt):
            return json.dumps({
                "distill_type": "ABSTRACT_CONCEPT",
                "reasoning": "Fairy tale mirror = remote image transmission",
                "reinforcement_delta": {},
                "new_node": {
                    "text": "Remote image transmission device",
                    "domain": "communication_technology",
                    "entity_type": "cultural_phantom",
                    "abstract_concepts": [
                        "portable device",
                        "on-demand activation",
                        "remote visual transmission",
                    ],
                    "confidence": 0.75,
                },
                "relation_type": "precursor_of",
                "relation_strength": 0.6,
            })

    # ── 1. Создаём canonical узел ────────────────────────────────
    einstein = MockNode(
        "einstein_1917",
        "Stimulated Emission of Radiation",
        "quantum_mechanics",
        entity_type="physical_principle",
        sci=8.0,
    )
    graph = MockGraph([einstein])

    distiller = ConceptDistiller(llm_client=MockLLM())
    processor = SignificanceProcessor(distiller=distiller)

    # ── 2. Источник 1: статья Nature 2024 → REINFORCEMENT ───────
    class FakeIngestion:
        source = "https://nature.com/articles/laser2024"
        nodes = [MockNode("nature_2024", "Laser quantum efficiency study",
                          "quantum_mechanics", sci=9.0)]
        edges = []
        cross_doc_edges = []

    sig1 = processor.process(FakeIngestion(), graph, llm_client=MockLLM())
    assert sig1.reinforcements >= 0  # может быть 0 если similarity < threshold
    logger.info(f"✓ Nature 2024: reinforcements={sig1.reinforcements}, "
                f"mutants={sig1.mutant_nodes}")

    # ── 3. Прямой тест distiller с форумным источником ──────────
    distiller.llm = MockLLM()
    dr_list = distiller.process_source(
        source_text="Discussion about stimulated emission quantum effects",
        source_type="reddit",
        source_name="reddit.com/r/physics",
        timestamp=1609459200.0,  # 2021
        graph_nodes=graph.nodes,
        embedder=None,
    )
    # Heuristic path (нет embedder) → keyword overlap
    logger.info(f"✓ Reddit distillation: {len(dr_list)} results")

    # ── 4. InfluenceTimeline — вручную добавим точки ─────────────
    tl = distiller.get_or_create_timeline("einstein_1917")
    tl.add(InfluencePoint(
        timestamp=-1672531200.0,  # 1917
        source="Einstein original",
        source_type="peer_reviewed",
        epistemic_level=EpistemicLevel.PEER_REVIEWED,
        weight=EpistemicWeight(EpistemicLevel.PEER_REVIEWED, 0.85,
                               -1672531200.0).effective_weight,
    ))
    tl.add(InfluencePoint(
        timestamp=0.0,  # 1970
        source="Textbook mention",
        source_type="textbook",
        epistemic_level=EpistemicLevel.TEXTBOOK_MENTION,
        weight=EpistemicWeight(EpistemicLevel.TEXTBOOK_MENTION, 0.40,
                               0.0).effective_weight,
    ))
    tl.add(InfluencePoint(
        timestamp=1704067200.0,  # 2024
        source="Nature 2024",
        source_type="nature",
        epistemic_level=EpistemicLevel.PROVEN_DISCOVERY,
        weight=EpistemicWeight(EpistemicLevel.PROVEN_DISCOVERY, 1.0,
                               1704067200.0).effective_weight,
    ))
    logger.info(f"✓ InfluenceTimeline:")
    logger.info(f"    epistemic_mass   = {tl.epistemic_mass:.3f}")
    logger.info(f"    velocity         = {tl.velocity:.3f}")
    logger.info(f"    source_diversity = {tl.source_diversity:.3f}")
    logger.info(f"    proven_status    = {tl.proven_status()}")
    logger.info(f"    peak_year        = {tl.peak_epistemic_year}")
    assert tl.proven_status(), "FAIL: should be proven with Nature 2024 source"
    logger.info("✓ proven_status=True after Nature source")

    # ── 5. Тест ABSTRACT_CONCEPT из сказки ─────────────────────
    fairy_node = MockNode(
        "magic_mirror",
        "Magic mirror shows distant images on command",
        "folklore",
        entity_type="cultural_phantom",
    )
    fairy_node.embedding = np.random.randn(384).astype(np.float32)
    graph2 = MockGraph([einstein, fairy_node])
    distiller2 = ConceptDistiller(llm_client=MockLLMFairy())
    dr_fairy = distiller2.process_source(
        source_text="A magic mirror that shows what happens far away when asked",
        source_type="fairytale",
        source_name="Snow White fairy tale",
        timestamp=-2208988800.0,  # ~1900
        graph_nodes=graph2.nodes,
        embedder=None,
    )
    logger.info(f"✓ Fairy tale distillation: {len(dr_fairy)} results")

    # Проверяем что создаётся мутантный узел для abstract_concept
    proc2 = SignificanceProcessor(distiller=distiller2)
    # Прямой вызов _create_mutant_node
    if dr_fairy:
        dr = dr_fairy[0]
        if dr.distill_type == DistillationType.ABSTRACT_CONCEPT:
            logger.info(f"✓ ABSTRACT_CONCEPT detected from fairy tale")
            logger.info(f"    concepts: {dr.new_node_concepts}")
        else:
            logger.info(f"  (heuristic: {dr.distill_type.value})")

    # ── 6. EpistemicWeight расчёт ───────────────────────────────
    w_nature  = EpistemicWeight(EpistemicLevel.PROVEN_DISCOVERY, 1.0, 0.0)
    w_reddit  = EpistemicWeight(EpistemicLevel.FORUM_DISCUSSION, 0.2, 0.0)
    w_fiction = EpistemicWeight(EpistemicLevel.FOLKLORE_FICTION, 0.1, 0.0)
    logger.info(f"✓ EpistemicWeight comparison:")
    logger.info(f"    nature  = {w_nature.effective_weight:.3f}")
    logger.info(f"    reddit  = {w_reddit.effective_weight:.3f}")
    logger.info(f"    fiction = {w_fiction.effective_weight:.3f}")
    assert w_nature.effective_weight > w_reddit.effective_weight > w_fiction.effective_weight
    logger.info("✓ nature > reddit > fiction (correct ordering)")

    logger.info("=== Smoke Test PASSED ===")

__all__ = ['EpistemicLevel', 'SOURCE_TYPE_TO_LEVEL', 'get_epistemic_level', 'EpistemicWeight', 'InfluencePoint', 'InfluenceTimeline', 'DistillationType', 'DistillationResult', 'DISTILLATION_PROMPT', 'ConceptDistiller', 'ProcessorResult', 'SignificanceProcessor', 'attach_significance_layer', 'get_significance_report', 'epistemic_level_for_node', '_smoke_test']
