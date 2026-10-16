"""Contextual concept classification and canonical node assembly."""
from __future__ import annotations

from ._shared import *
from .canonical_keys import (
    ALIAS_TABLE,
    AliasTable,
    CanonicalKey,
    DOMAIN_TABLE,
    DataLoader,
    ENTITY_TYPE_TABLE,
    GROUP_TABLE,
    LookupTable,
    ObservationRecord,
    SEMANTIC_HASHER,
    SemanticHasher,
    TRANSFORMERS_AVAILABLE,
    TorchDataset,
    load_lookup_tables,
    save_lookup_tables,
)
from .config import CONFIG, PHYSICAL_AXIS_ORDER
from .entities import KnowledgeNode, NodeFeatureBuilder, PhysicalSubstrateEncoder
from .json_utils import _parse_llm_json
from .time_utils import _year_to_ts

# ══════════════════════════════════════════════════════════════════
#  CONCEPT CLASSIFIER
#  Классифицирует термин → (domain_id, group_id, entity_type_id)
#
#  Три режима работы:
#    1. Cold start: только LLM, все ответы → training_data
#    2. Warm:       DeBERTa/BERT fast path + LLM slow path
#    3. Hot:        DeBERTa покрывает 95%+ случаев
#
#  Также используется для ПРЕДСКАЗАНИЯ домена нового концепта
#  без LLM на основе контекста (inference mode).
# ══════════════════════════════════════════════════════════════════

@dataclass
class ClassificationSample:
    """Обучающий пример для классификатора."""
    text:          str
    context:       str
    source_type:   str
    domain_name:   str
    group_name:    str
    entity_type:   str
    confidence:    float = 1.0   # 1.0 = LLM-ground-truth, <1.0 = predicted


class _SimpleTextEncoder:
    """
    Минимальный TF-IDF / bag-of-ngrams энкодер без трансформеров.
    Используется если transformers недоступны.
    """
    def __init__(self, max_features: int = 4096):
        self.max_features = max_features
        self._vocab: Dict[str, int] = {}
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        counts: Counter = Counter()
        for t in texts:
            for w in self._tokenize(t):
                counts[w] += 1
        top = [w for w, _ in counts.most_common(self.max_features)]
        self._vocab = {w: i for i, w in enumerate(top)}
        self._fitted = True

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.max_features, dtype=np.float32)
        tokens = self._tokenize(text)
        for w in tokens:
            idx = self._vocab.get(w)
            if idx is not None:
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        words = re.findall(r'[a-zа-яё]{3,}', text.lower())
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        return words + bigrams


class ConceptClassifierModel(nn.Module if TORCH_AVAILABLE else object):
    """
    Лёгкая классификационная голова поверх текстового энкодера.
    При наличии transformers — использует CLS-токен BERT/DeBERTa.
    Без них — TF-IDF → Linear → softmax.

    Три независимые головы:
        domain_head      → n_domains классов
        group_head       → n_groups классов
        entity_type_head → n_types классов
    """

    def __init__(self, encoder_dim: int,
                 n_domains: int, n_groups: int, n_types: int,
                 hidden: int = 256, dropout: float = 0.1):
        if TORCH_AVAILABLE:
            super().__init__()
        self.encoder_dim = encoder_dim

        if TORCH_AVAILABLE:
            self.shared = nn.Sequential(
                nn.Linear(encoder_dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
            )
            half = hidden // 2
            self.domain_head      = nn.Linear(half, n_domains)
            self.group_head       = nn.Linear(half, n_groups)
            self.entity_type_head = nn.Linear(half, n_types)
        else:
            # stubs
            self.shared = self.domain_head = None
            self.group_head = self.entity_type_head = None

    def forward(self, x):
        if not TORCH_AVAILABLE:
            return None, None, None
        shared = self.shared(x)
        return (self.domain_head(shared),
                self.group_head(shared),
                self.entity_type_head(shared))


class _ClassifierDataset(TorchDataset if TORCH_AVAILABLE else object):
    def __init__(self, samples: List[ClassificationSample],
                 encoder, domain_table: LookupTable,
                 group_table: LookupTable, entity_table: LookupTable):
        self.samples      = samples
        self.encoder      = encoder
        self.domain_table = domain_table
        self.group_table  = group_table
        self.entity_table = entity_table

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        combined = f"{s.text} [SEP] {s.context} [SRC] {s.source_type}"
        vec = torch.tensor(self.encoder.encode(combined), dtype=torch.float32)
        d   = self.domain_table.get_id(s.domain_name)
        g   = self.group_table.get_id(s.group_name)
        e   = self.entity_table.get_id(s.entity_type)
        return vec, torch.tensor(d), torch.tensor(g), torch.tensor(e)


class ConceptClassifier:
    """
    Главный класс классификации и предсказания доменов.

    Использование:
        clf = ConceptClassifier(alias_table=AliasTable())

        # При cold start:
        result = clf.classify("лазер", context="...", source_type="arxiv",
                               llm_fallback=my_llm)
        # → ClassifyResult(domain="optics", group="fundamental_principle", ...)

        # После накопления данных — обучение:
        clf.train(epochs=5)

        # Предсказание для нового концепта (без LLM):
        pred = clf.predict("новый термин", context="...", source_type="forum")
        # → ClassifyResult с confidence

        # Измерение необходимости технологии:
        score = clf.necessity_score("лазер", graph_nodes)
    """

    CONFIDENCE_THRESHOLD = CONFIG["classifier_confidence_threshold"]
    RETRAIN_EVERY        = CONFIG["classifier_retrain_every"]
    LLM_CLASSIFY_PROMPT  = """\
You are a knowledge graph classifier. Given a concept term with context,
output the most appropriate classification.

TERM: {term}
CONTEXT: {context}
SOURCE TYPE: {source_type}

Available domains:      {domains}
Available groups:       {groups}
Available entity types: {entity_types}

Rules:
- Same term in different contexts → different domains (e.g. "laser" in physics vs fiction)
- Choose the most specific applicable domain
- If unsure → prefer broader categories
- Synonyms/aliases field: list 2-5 alternative names for this exact concept

OUTPUT ONLY valid JSON:
{{
  "domain":      "<exact name from domains list>",
  "group":       "<exact name from groups list>",
  "entity_type": "<exact name from entity_types list>",
  "confidence":  <0.0-1.0>,
  "reasoning":   "<one sentence>",
  "aliases":     ["<alias1>", "<alias2>"]
}}
"""

    def __init__(self,
                 domain_table: LookupTable      = DOMAIN_TABLE,
                 group_table: LookupTable       = GROUP_TABLE,
                 entity_table: LookupTable      = ENTITY_TYPE_TABLE,
                 alias_table: AliasTable        = ALIAS_TABLE,
                 hasher: SemanticHasher         = SEMANTIC_HASHER,
                 model_path: Optional[str]      = None,
                 encoder_name: str              = "sentence-transformers/all-MiniLM-L6-v2",
                 use_bert: bool                  = True):

        self.domain_table  = domain_table
        self.group_table   = group_table
        self.entity_table  = entity_table
        self.alias_table   = alias_table
        self.hasher        = hasher

        self.training_data: List[ClassificationSample] = []
        self._llm_calls_since_retrain = 0
        self.llm_fallback_calls = 0
        self.llm_fallback_failures = 0
        self.llm_fallback_incomplete_calls = 0

        # Энкодер текста
        if TRANSFORMERS_AVAILABLE and use_bert:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(encoder_name)
                self._bert       = AutoModel.from_pretrained(encoder_name)
                self._bert.eval()
                self._encoder_dim = self._bert.config.hidden_size
                self._use_bert    = True
                logger.info(f"[Classifier] using BERT encoder: {encoder_name}")
            except Exception as e:
                logger.warning(f"[Classifier] BERT load failed ({e}) → TF-IDF")
                self._use_bert    = False
                self._simple_enc  = _SimpleTextEncoder()
                self._encoder_dim = 4096
        else:
            self._use_bert    = False
            self._simple_enc  = _SimpleTextEncoder()
            self._encoder_dim = 4096
            if TRANSFORMERS_AVAILABLE and not use_bert:
                logger.info("[Classifier] transformer encoder disabled; using lightweight text features")

        # Нейросетевая голова классификатора
        self._model: Optional[ConceptClassifierModel] = None
        self._model_ready = False

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    # ══════════════════════════════════════════════════════════════
    #  Основной API
    # ══════════════════════════════════════════════════════════════

    @dataclass
    class ClassifyResult:
        domain:         str
        group:          str
        entity_type:    str
        confidence:     float
        canonical_key:  bytes           # 12-байтный ключ
        aliases:        List[str]       # синонимы от LLM
        via_llm:        bool = False    # True если использован LLM

        def describe(self) -> str:
            ck = CanonicalKey.from_bytes(self.canonical_key)
            return (f"domain={self.domain} / group={self.group} / "
                    f"entity_type={self.entity_type} | "
                    f"conf={self.confidence:.2f} | key={ck.hex()} "
                    f"{'[LLM]' if self.via_llm else '[model]'}")

    def classify(self,
                 term: str,
                 context: str = "",
                 source_type: str = "unknown",
                 embedder=None,
                 llm_fallback=None) -> "ConceptClassifier.ClassifyResult":
        """
        Классифицировать концепт.

        1. Проверяем alias_table (если уже видели термин).
        2. Fast path: нейросетевой классификатор (если обучен).
        3. Slow path: LLM (если confidence низкий или модель не обучена).
        4. Регистрируем результат → canonical_key → alias_table.
        """
        # ── Шаг 1: alias lookup (domain-aware) ───────────────────
        # Сначала пытаемся найти по домену из source_type (приближение),
        # затем — без домена. Так "лазер" в физике и в фантастике
        # дают РАЗНЫЕ ключи, а "LASER" в той же физике — тот же ключ.
        existing = self.alias_table.lookup(term, domain=source_type)
        # ``source_type`` (nature/arxiv/unknown/...) is not normally the same
        # namespace as the classifier's subject domain (biology/physics/...).
        # Reuse a unique known domain only when no contextual classifier is
        # available. With an active model or LLM, let context disambiguate
        # a term that may have a second meaning.
        if existing is None:
            known_domains = self.alias_table.all_domains_for(term)
            if len(known_domains) == 1 and llm_fallback is None and not self._model_ready:
                existing = self.alias_table.lookup(term, domain=known_domains[0])
        if existing is not None:
            ck = CanonicalKey.from_bytes(existing)
            return ConceptClassifier.ClassifyResult(
                domain      = self.domain_table.get_name(ck.domain_id),
                group       = self.group_table.get_name(ck.group_id),
                entity_type = self.entity_table.get_name(ck.entity_type_id),
                confidence  = 1.0,
                canonical_key = existing,
                aliases     = [term],
                via_llm     = False,
            )

        # ── Шаг 2: fast path (нейросеть) ────────────────────────
        if self._model_ready:
            result = self._model_classify(term, context, source_type)
            if result and result.confidence >= self.CONFIDENCE_THRESHOLD:
                self._register_result(term, result, embedder)
                return result

        # ── Шаг 3: slow path (LLM) ──────────────────────────────
        if llm_fallback is not None:
            result = self._llm_classify(term, context, source_type, llm_fallback)
            if result:
                # Добавляем в обучающую выборку
                sample = ClassificationSample(
                    text=term, context=context, source_type=source_type,
                    domain_name=result.domain, group_name=result.group,
                    entity_type=result.entity_type, confidence=result.confidence,
                )
                self.training_data.append(sample)
                self._llm_calls_since_retrain += 1

                # Автоматическое переобучение
                if self._llm_calls_since_retrain >= self.RETRAIN_EVERY:
                    logger.info("[Classifier] auto-retrain triggered")
                    self.train(epochs=3)
                    self._llm_calls_since_retrain = 0

                self._register_result(term, result, embedder)
                return result

        # ── Шаг 4: fallback (неизвестный домен) ─────────────────
        return self._unknown_result(term, context, source_type, embedder)

    def predict(self, term: str,
                context: str = "",
                source_type: str = "unknown") -> "ConceptClassifier.ClassifyResult":
        """
        Предсказание без LLM. Используется для:
          – batch-классификации новых узлов
          – предсказания домена виртуальных/phantom-узлов
          – оценки cross-domain необходимости концепта
        """
        if self._model_ready:
            result = self._model_classify(term, context, source_type)
            if result:
                return result
        return self._unknown_result(term, context, source_type, embedder=None)

    def necessity_score(self,
                        term: str,
                        graph_nodes: Dict[str, Any]) -> float:
        """
        Измерить "необходимость" концепта: насколько широко
        он представлен в разных доменах графа.

        Высокий балл = концепт нужен везде = высокая
        вероятность breakthrough (ключевая метрика Oracle-1).
        """
        counts = CanonicalKey.cross_domain_presence(
            term, self.alias_table, graph_nodes)
        score = CanonicalKey.necessity_score(counts)
        logger.debug(
            f"[Classifier] necessity('{term}') = {score:.3f} "
            f"across {len(counts)} domains: {counts}"
        )
        return score

    # ══════════════════════════════════════════════════════════════
    #  Обучение
    # ══════════════════════════════════════════════════════════════

    def train(self,
              epochs: int = 10,
              batch_size: int = 32,
              lr: float = 2e-4,
              save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Обучить/дообучить классификатор на накопленных примерах.
        Возвращает {domain_acc, group_acc, entity_acc, val_domain_acc, …}.

        Улучшения по сравнению с v1:
          • 85/15 train/val split — видим реальную обобщающую способность.
          • Веса старой модели СОХРАНЯЮТСЯ при росте таблиц (expand_heads):
            новые классы инициализируются нулями, старые не сбрасываются.
          • label_smoothing в CrossEntropyLoss — снижает overfit на редкие классы.
          • Confidence вычисляется как среднее (не произведение) трёх max-prob,
            что даёт адекватный сигнал при >10 классах в каждой голове.
        """
        if not TORCH_AVAILABLE:
            logger.warning("[Classifier.train] PyTorch unavailable — skip")
            return {}

        if len(self.training_data) < 10:
            logger.warning(
                f"[Classifier.train] too few samples ({len(self.training_data)}) — skip"
            )
            return {}

        logger.info(
            f"[Classifier.train] starting: {len(self.training_data)} samples, "
            f"{epochs} epochs"
        )

        # ── Шаг 1: подготовка энкодера ───────────────────────────
        encoder = self._get_encoder()
        if not self._use_bert:
            texts = [f"{s.text} {s.context}" for s in self.training_data]
            self._simple_enc.fit(texts)

        # ── Шаг 2: train/val split (85/15) ──────────────────────
        val_frac  = CONFIG.get("classifier_val_split", 0.15)
        n_val     = max(1, int(len(self.training_data) * val_frac))
        # Deterministic shuffle based on text hash so results are reproducible
        sorted_data = sorted(self.training_data,
                             key=lambda s: hashlib.md5(s.text.encode()).hexdigest())
        val_data   = sorted_data[:n_val]
        train_data = sorted_data[n_val:]

        # ── Шаг 3: инициализация/расширение модели ──────────────
        n_d = len(self.domain_table)
        n_g = len(self.group_table)
        n_e = len(self.entity_table)

        new_model = ConceptClassifierModel(
            encoder_dim=self._encoder_dim,
            n_domains=n_d, n_groups=n_g, n_types=n_e,
        )

        # Preserve old weights when the table has grown (new classes appear)
        if (self._model is not None
                and CONFIG.get("classifier_max_head_expand", True)):
            self._expand_heads(new_model)

        self._model = new_model

        # ── Шаг 4: DataLoader ────────────────────────────────────
        train_ds = _ClassifierDataset(
            train_data, encoder,
            self.domain_table, self.group_table, self.entity_table,
        )
        val_ds = _ClassifierDataset(
            val_data, encoder,
            self.domain_table, self.group_table, self.entity_table,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, drop_last=False)

        optimizer  = torch.optim.Adam(self._model.parameters(), lr=lr)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
        smoothing  = CONFIG.get("classifier_label_smoothing", 0.05)
        criterion  = nn.CrossEntropyLoss(label_smoothing=smoothing)

        # ── Шаг 5: обучение ──────────────────────────────────────
        metrics: Dict[str, float] = {}
        for epoch in range(epochs):
            self._model.train()
            total_loss = 0.0
            correct    = [0, 0, 0]
            total_     = 0

            for batch_x, batch_d, batch_g, batch_e in train_loader:
                optimizer.zero_grad()
                logits_d, logits_g, logits_e = self._model(batch_x)
                loss = (criterion(logits_d, batch_d) +
                        criterion(logits_g, batch_g) +
                        criterion(logits_e, batch_e))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_     += batch_x.size(0)
                correct[0] += (logits_d.argmax(1) == batch_d).sum().item()
                correct[1] += (logits_g.argmax(1) == batch_g).sum().item()
                correct[2] += (logits_e.argmax(1) == batch_e).sum().item()

            scheduler.step()
            acc_d = correct[0] / max(total_, 1)
            acc_g = correct[1] / max(total_, 1)
            acc_e = correct[2] / max(total_, 1)
            avg_l = total_loss / max(len(train_loader), 1)

            # ── Валидация ────────────────────────────────────────
            val_d, val_g, val_e = self._eval_loader(val_loader)

            logger.info(
                f"[Classifier.train] epoch {epoch+1}/{epochs} "
                f"loss={avg_l:.4f} "
                f"train d/g/e={acc_d:.3f}/{acc_g:.3f}/{acc_e:.3f} "
                f"val d/g/e={val_d:.3f}/{val_g:.3f}/{val_e:.3f}"
            )
            metrics = {
                "domain_acc": acc_d, "group_acc": acc_g, "entity_acc": acc_e,
                "val_domain_acc": val_d, "val_group_acc": val_g,
                "val_entity_acc": val_e, "loss": avg_l,
            }

        self._model.eval()
        self._model_ready = True
        logger.info("[Classifier.train] complete → model_ready=True")

        if save_path:
            self._save_model(save_path)

        return metrics

    def _eval_loader(self, loader) -> Tuple[float, float, float]:
        """Evaluate accuracy on a DataLoader (no grad)."""
        if loader is None or not TORCH_AVAILABLE or self._model is None:
            return 0.0, 0.0, 0.0
        self._model.eval()
        correct = [0, 0, 0]
        total   = 0
        with torch.no_grad():
            for batch_x, batch_d, batch_g, batch_e in loader:
                ld, lg, le = self._model(batch_x)
                correct[0] += (ld.argmax(1) == batch_d).sum().item()
                correct[1] += (lg.argmax(1) == batch_g).sum().item()
                correct[2] += (le.argmax(1) == batch_e).sum().item()
                total      += batch_x.size(0)
        n = max(total, 1)
        self._model.train()
        return correct[0] / n, correct[1] / n, correct[2] / n

    def _expand_heads(self, new_model: "ConceptClassifierModel") -> None:
        """Copy old head weights into new_model for classes that already exist.

        Called when domain/group/entity tables grew since last train().
        New class rows are left at their random init values; existing rows
        get the pre-trained weights so the model is not reset from scratch.
        """
        if not TORCH_AVAILABLE or self._model is None:
            return
        head_pairs = [
            (self._model.domain_head,      new_model.domain_head),
            (self._model.group_head,       new_model.group_head),
            (self._model.entity_type_head, new_model.entity_type_head),
        ]
        for old_head, new_head in head_pairs:
            if old_head is None or new_head is None:
                continue
            old_w = old_head.weight.data   # (old_classes, hidden)
            new_w = new_head.weight.data   # (new_classes, hidden)
            n_copy = min(old_w.size(0), new_w.size(0))
            new_w[:n_copy, :] = old_w[:n_copy, :]
            if old_head.bias is not None and new_head.bias is not None:
                old_b = old_head.bias.data
                new_b = new_head.bias.data
                new_b[:min(old_b.size(0), new_b.size(0))] = \
                    old_b[:min(old_b.size(0), new_b.size(0))]
        # Shared layers: copy directly (dimensions unchanged)
        try:
            new_model.shared.load_state_dict(
                self._model.shared.state_dict(), strict=False)
        except Exception:
            pass  # mismatched shared dim on encoder change — fine, skip

    # ══════════════════════════════════════════════════════════════
    #  Внутренние методы
    # ══════════════════════════════════════════════════════════════

    def _model_classify(self, term: str, context: str,
                         source_type: str
                         ) -> Optional["ConceptClassifier.ClassifyResult"]:
        if not TORCH_AVAILABLE or self._model is None:
            return None
        encoder = self._get_encoder()
        text    = f"{term} [SEP] {context} [SRC] {source_type}"
        vec     = torch.tensor(encoder.encode(text),
                               dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits_d, logits_g, logits_e = self._model(vec)
        probs_d  = F.softmax(logits_d, dim=-1)[0]
        probs_g  = F.softmax(logits_g, dim=-1)[0]
        probs_e  = F.softmax(logits_e, dim=-1)[0]
        # Arithmetic mean of per-head max-probs gives a readable signal even
        # with many classes; geometric product collapses to near-zero at scale.
        conf = float((probs_d.max() + probs_g.max() + probs_e.max()) / 3.0)

        d_id = int(probs_d.argmax().item())
        g_id = int(probs_g.argmax().item())
        e_id = int(probs_e.argmax().item())

        sem  = self.hasher.hash_text(f"{term} {context}",
                                     self._get_embedder())
        ck   = CanonicalKey(domain_id=d_id, group_id=g_id,
                            entity_type_id=e_id, sem_hash=sem)

        return ConceptClassifier.ClassifyResult(
            domain      = self.domain_table.get_name(d_id),
            group       = self.group_table.get_name(g_id),
            entity_type = self.entity_table.get_name(e_id),
            confidence  = conf,
            canonical_key = ck.to_bytes(),
            aliases     = [],
            via_llm     = False,
        )

    def _llm_classify(self, term: str, context: str,
                       source_type: str,
                       llm_client) -> Optional["ConceptClassifier.ClassifyResult"]:
        self.llm_fallback_calls += 1
        prompt = self.LLM_CLASSIFY_PROMPT.format(
            term=term, context=context[:500], source_type=source_type,
            domains=", ".join(self.domain_table.all_names()[:40]),
            groups=", ".join(self.group_table.all_names()[:20]),
            entity_types=", ".join(self.entity_table.all_names()[:25]),
        )
        try:
            raw  = llm_client.complete(prompt)
            data = _parse_llm_json(raw)
        except Exception as e:
            self.llm_fallback_failures += 1
            logger.warning(f"[Classifier._llm_classify] LLM error: {e}")
            return None

        if not isinstance(data, dict) or any(
                not isinstance(data.get(field), str) or not data[field].strip()
                for field in ("domain", "group", "entity_type")):
            self.llm_fallback_incomplete_calls += 1
            return None

        domain_name = data.get("domain", "unknown")
        group_name  = data.get("group", "unknown")
        etype_name  = data.get("entity_type", "unknown")
        confidence  = float(data.get("confidence", 0.5))
        aliases     = data.get("aliases", [])

        # Авто-регистрация новых значений в таблицах
        d_id = self.domain_table.register(domain_name)
        g_id = self.group_table.register(group_name)
        e_id = self.entity_table.register(etype_name)

        sem = self.hasher.hash_text(f"{term} {context}",
                                    self._get_embedder())
        ck  = CanonicalKey(domain_id=d_id, group_id=g_id,
                           entity_type_id=e_id, sem_hash=sem)
        key_bytes = ck.to_bytes()

        # Регистрируем синонимы с привязкой к домену
        all_aliases = [term] + aliases
        self.alias_table.bulk_register(all_aliases, key_bytes, domain=domain_name)

        return ConceptClassifier.ClassifyResult(
            domain=domain_name, group=group_name, entity_type=etype_name,
            confidence=confidence, canonical_key=key_bytes,
            aliases=aliases, via_llm=True,
        )

    def _unknown_result(self, term: str, context: str,
                         source_type: str,
                         embedder=None) -> "ConceptClassifier.ClassifyResult":
        sem = self.hasher.hash_text(f"{term} {context}", embedder)
        ck  = CanonicalKey(domain_id=0, group_id=0,
                           entity_type_id=0, sem_hash=sem)
        return ConceptClassifier.ClassifyResult(
            domain="unknown", group="unknown", entity_type="unknown",
            confidence=0.0, canonical_key=ck.to_bytes(),
            aliases=[], via_llm=False,
        )

    def _register_result(self, term: str,
                          result: "ConceptClassifier.ClassifyResult",
                          embedder=None) -> None:
        """Зарегистрировать alias с привязкой к домену."""
        self.alias_table.register(term, result.canonical_key,
                                  domain=result.domain)
        for a in result.aliases:
            self.alias_table.register(a, result.canonical_key,
                                      domain=result.domain)

    def _get_encoder(self):
        """Вернуть вызываемый энкодер (BERT или TF-IDF)."""
        if self._use_bert:
            return _BertEncoderWrapper(self._tokenizer, self._bert)
        return self._simple_enc

    def _get_embedder(self):
        """Вернуть объект с методом .encode() для SemanticHasher."""
        if self._use_bert:
            return _BertEncoderWrapper(self._tokenizer, self._bert)
        return None

    # ── Сохранение/загрузка ─────────────────────────────────────

    def _save_model(self, path: str) -> None:
        if self._model is None:
            return
        torch.save({
            "model_state":       self._model.state_dict(),
            "encoder_dim":       self._encoder_dim,
            "use_bert":          self._use_bert,
            "n_domains":         len(self.domain_table),
            "n_groups":          len(self.group_table),
            "n_types":           len(self.entity_table),
            "training_data_len": len(self.training_data),
        }, path)
        logger.info(f"[Classifier] model saved → {path}")

    def _load_model(self, path: str) -> None:
        # weights_only=True avoids arbitrary code execution via pickle
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self._model = ConceptClassifierModel(
            encoder_dim=ckpt["encoder_dim"],
            n_domains=ckpt["n_domains"],
            n_groups=ckpt["n_groups"],
            n_types=ckpt["n_types"],
        )
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()
        self._model_ready = True
        logger.info(f"[Classifier] model loaded from {path}")

    def save_state(self, directory: str) -> None:
        """Сохранить полное состояние: модель + таблицы + alias."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        save_lookup_tables(str(d / "lookup_tables.json"))
        with open(d / "alias_table.json", "w", encoding="utf-8") as f:
            json.dump(self.alias_table.to_dict(), f, ensure_ascii=False, indent=2)
        with open(d / "training_data.pkl", "wb") as f:
            pickle.dump(self.training_data, f)
        if self._model and TORCH_AVAILABLE:
            self._save_model(str(d / "classifier_model.pt"))
        logger.info(f"[Classifier] full state saved → {directory}")

    def load_state(self, directory: str) -> None:
        """Загрузить полное состояние."""
        d = Path(directory)
        tables_path = d / "lookup_tables.json"
        if tables_path.exists():
            load_lookup_tables(str(tables_path))
        alias_path = d / "alias_table.json"
        if alias_path.exists():
            with open(alias_path, encoding="utf-8-sig") as f:
                self.alias_table = AliasTable.from_dict(json.load(f))
        data_path = d / "training_data.pkl"
        if data_path.exists():
            with open(data_path, "rb") as f:
                self.training_data = pickle.load(f)
        model_path = d / "classifier_model.pt"
        if model_path.exists() and TORCH_AVAILABLE:
            self._load_model(str(model_path))
        logger.info(f"[Classifier] full state loaded from {directory}")


# ── Вспомогательный класс для BERT ──────────────────────────────

class _BertEncoderWrapper:
    """Тонкая обёртка: BERT → encode(text) → np.ndarray."""

    def __init__(self, tokenizer, model):
        self._tokenizer = tokenizer
        self._model     = model

    def encode(self, text: str) -> np.ndarray:
        if not TORCH_AVAILABLE:
            return np.zeros(768, dtype=np.float32)
        inputs = self._tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            out = self._model(**inputs)
        cls = out.last_hidden_state[:, 0, :].squeeze(0).numpy()
        return cls.astype(np.float32)


# ── Утилита разбора JSON из LLM ─────────────────────────────────

def make_canonical_id(text: str,
                       context: str,
                       source_type: str,
                       source: str,
                       classifier: ConceptClassifier,
                       embedder=None,
                       llm_fallback=None) -> Tuple[bytes, "ConceptClassifier.ClassifyResult"]:
    """
    Единственная точка входа для получения id узла.

    Возвращает:
        canonical_id (bytes, 12 байт) — ключ узла
        ClassifyResult — домен, группа, тип, confidence

    Вызывается из IngestionPipeline при сборке каждого узла.
    """
    result = classifier.classify(
        term=text,
        context=context,
        source_type=source_type,
        embedder=embedder,
        llm_fallback=llm_fallback,
    )
    # Если alias уже зарегистрирован — вернём существующий ключ
    # (CrossDocumentLinker обработает merge)
    return result.canonical_key, result


# ══════════════════════════════════════════════════════════════════
#  ОБНОВЛЁННЫЙ NodeEdgeAssembler с поддержкой canonical key
# ══════════════════════════════════════════════════════════════════

class CanonicalNodeAssembler:
    """
    Единственный сборщик узлов. Идентичность — через ConceptClassifier
    (CanonicalKey), MD5(source::text) в кодовой базе больше не существует.

    Ключевые отличия:
      – ID = 12-байтный canonical key (не MD5 строки)
      – ObservationRecord создаётся для каждого нового источника
      – necessity_score доступен сразу при сборке
      – cross-domain агрегация встроена
    """

    def __init__(self,
                 classifier: ConceptClassifier,
                 embedder=None,
                 llm_fallback=None):
        self.classifier   = classifier
        self.embedder     = embedder
        self.llm_fallback = llm_fallback
        # Хранилище наблюдений: canonical_id_hex → List[ObservationRecord]
        self.observations: Dict[str, List[ObservationRecord]] = defaultdict(list)

    def assemble_node_with_canonical_key(
        self,
        item: Dict,
        source: str,
        timestamp: float,
        source_type: str = "unknown",
        source_trust: float = 0.7,
        physical_scores: Optional[Dict[str, float]] = None,
        record_observation: bool = True,
    ):
        """
        Собрать KnowledgeNode с canonical ID.
        В объединённой сборке KnowledgeNode и утилиты доступны напрямую.
        В автономном режиме импортируются из oracle1.
        """
        _KnowledgeNode = KnowledgeNode
        _NodeFeatureBuilder = NodeFeatureBuilder
        _PhysSubEncoder = PhysicalSubstrateEncoder
        _PhysAxisOrder = PHYSICAL_AXIS_ORDER

        text    = item.get("text", "")
        context = item.get("full_context", "")

        # ── Классификация и получение canonical ID ───────────────
        cid_bytes, clf_result = make_canonical_id(
            text=text,
            context=context,
            source_type=source_type,
            source=source,
            classifier=self.classifier,
            embedder=self.embedder,
            llm_fallback=self.llm_fallback,
        )
        cid_hex = cid_bytes.hex()

        # ── Разрешение временнОй позиции узла ─────────────────────
        # node.timestamp = the document's own timestamp, always — see the
        # matching (corrected) comment in NodeEdgeAssembler.assemble_node().
        # This mirrored that function's PRE-CORRECTION logic on the first
        # pass of wiring CanonicalNodeAssembler in, deriving node_timestamp
        # from publication_year (the concept's historical origin). That was
        # wrong there too, for the same reason: a node is an idea/concept,
        # not a technology with one fixed invention date — it's a line
        # stretched across time, advancing with each new observation, not a
        # point frozen at its origin. publication_year keeps its own,
        # separate role — a data-quality/specificity signal (feeds
        # date_unknown / temporal_kind="axiom" below), not a timestamp
        # source.
        pub_year = item.get("publication_year")
        pub_year_valid = False
        try:
            pub_timestamp = _year_to_ts(int(pub_year)) if pub_year is not None else 0.0
            pub_year_valid = pub_year is not None
        except (ValueError, TypeError):
            pub_timestamp = 0.0
        node_timestamp = float(timestamp or pub_timestamp)
        # Bug fix (found via test_canonical_key.py, same bug mirrored below
        # in NodeEdgeAssembler.assemble_node()): date_unknown must reflect
        # whether the CONCEPT has a known historical origin (pub_year), not
        # whether node_timestamp happens to be positive. node_timestamp <= 0
        # was a correct proxy back when node_timestamp WAS derived from
        # pub_timestamp — now that node_timestamp is always the document's
        # own timestamp (see the comment above), it is virtually always
        # positive regardless of pub_year, so this condition never fired in
        # practice. That silently disabled the "date-unknown gate" in
        # IngestionPipeline.ingest() (undated concepts should be merged into
        # an existing match or rejected, never created as fresh nodes) for
        # every node assembled through this path. A second attempt using
        # `pub_timestamp <= 0.0` was ALSO wrong: _year_to_ts() legitimately
        # returns a negative number for any pre-1970 year (e.g. 1960),
        # which would have misclassified a valid historical origin as
        # unknown.
        date_unknown = not pub_year_valid

        # ── Создание ObservationRecord ────────────────────────────
        if record_observation:
            obs = ObservationRecord.create(
                canonical_id=cid_bytes,
                source=source,
                timestamp=node_timestamp,
                source_type=source_type,
                trust=source_trust,
                delta_fields={k: v for k, v in item.items()
                              if k not in ("text", "full_context", "id")},
            )
            self.observations[cid_hex].append(obs)

        # ── Сборка KnowledgeNode ──────────────────────────────────
        node = _KnowledgeNode(
            id=cid_hex,
            text=text,
            full_text=context,
            node_type=item.get("node_type", "research"),
            domain=clf_result.domain,
            entity_type=clf_result.entity_type,
            provenance=source,
            timestamp=node_timestamp,
            publication_date=(str(item["publication_year"])
                              if item.get("publication_year") else None),
            scientific_score=float(item.get("scientific_score", 5.0)) * source_trust,
            investment_score=float(item.get("investment_score", 0.0)),
            social_score=float(item.get("social_score", 0.0)),
            maturity_score=float(item.get("maturity_score", 0.0)),
            strategic_value=float(item.get("strategic_value", 5.0)),
            efficiency_plateau=float(item.get("efficiency_plateau", 0.0)),
            dual_use_risk=float(item.get("dual_use_risk", 0.0)),
            legal_risk_score=float(item.get("legal_risk_score", 0.0)),
            export_control_risk=float(item.get("export_control_risk", 0.0)),
            solves_limitations=item.get("solves_limitations", []),
            requires_node_ids=item.get("requires_node_ids", []),
            enables_node_ids=item.get("enables_node_ids", []),
        )
        if date_unknown:
            node.__dict__["date_unknown"] = True
            # Parity with NodeEdgeAssembler.assemble_node() — see its
            # comment for why: opts genuinely origin-less concepts (myths,
            # timeless axioms) out of time-window/binning logic that
            # assumes a meaningful origin point.
            node.temporal_kind = "axiom"

        # Прикрепляем метаданные классификации в __dict__ (legacy) и native fields
        node.__dict__["canonical_key"]     = cid_bytes
        node.__dict__["clf_confidence"]    = clf_result.confidence
        node.__dict__["clf_via_llm"]       = clf_result.via_llm
        node.__dict__["observation_count"] = len(self.observations[cid_hex])
        # Native fields (available in merged oracle1 build)
        if hasattr(node, "observation_count"):
            node.observation_count = len(self.observations[cid_hex])
        if hasattr(node, "source_type"):
            node.source_type = source_type

        # ── Эмбеддинг и физические оси ───────────────────────────
        if self.embedder is not None:
            try:
                feat_builder = _NodeFeatureBuilder(self.embedder)
                phys_encoder = _PhysSubEncoder()
                base_vec     = feat_builder.build(node)
                if physical_scores:
                    clean_phys = {
                        ax: float(np.clip(physical_scores.get(ax, 0.5), 0.0, 1.0))
                        for ax in _PhysAxisOrder
                    }
                    node.embedding = phys_encoder.extend_node_features(
                        base_vec, clean_phys)
                    node.__dict__["raw_physical_scores"] = clean_phys
                    phys_vec = phys_encoder.build_physical_section(clean_phys)
                    node.__dict__["feasibility"] = phys_encoder.feasibility_score(
                        phys_vec)
                else:
                    node.embedding = base_vec

                # Обновляем семантический хэш с реальным эмбеддингом
                if node.embedding is not None:
                    sem = self.classifier.hasher.hash(node.embedding[:384])
                    ck  = CanonicalKey.from_bytes(cid_bytes)
                    ck.sem_hash = sem
                    node.__dict__["canonical_key"] = ck.to_bytes()

            except Exception as e:
                logger.warning(
                    f"[CanonicalAssembler] feature build failed for '{cid_hex}': {e}"
                )

        return node

    def get_necessity_score(self, term: str,
                             graph_nodes: Dict[str, Any]) -> float:
        """
        Измерить cross-domain необходимость термина.
        Прямой вызов к classifier.necessity_score().
        """
        return self.classifier.necessity_score(term, graph_nodes)

    def get_observations(self, canonical_id_hex: str) -> List[ObservationRecord]:
        return self.observations.get(canonical_id_hex, [])

    def observation_summary(self, canonical_id_hex: str) -> Dict[str, Any]:
        """Агрегированная статистика по всем наблюдениям концепта."""
        obs_list = self.observations.get(canonical_id_hex, [])
        if not obs_list:
            return {}
        trusts      = [o.trust for o in obs_list]
        timestamps  = [o.timestamp for o in obs_list]
        source_types = [o.source_type for o in obs_list]
        type_counts: Dict[str, int] = defaultdict(int)
        for st in source_types:
            type_counts[st] += 1
        return {
            "n_observations":  len(obs_list),
            "mean_trust":      float(np.mean(trusts)),
            "first_seen":      min(timestamps),
            "last_seen":       max(timestamps),
            "source_types":    dict(type_counts),
            "time_span_years": (max(timestamps) - min(timestamps)) / 31536000.0,
        }


# ══════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ══════════════════════════════════════════════════════════════════

def _smoke_test_canonical_key():
    logging.basicConfig(level=logging.INFO)
    logger.info("=== Canonical Key Smoke Test ===")

    clf = ConceptClassifier(alias_table=AliasTable())

    # Симулируем LLM-клиент: возвращает фиксированный JSON
    class MockLLM:
        _responses = {
            "лазер_physics": '{"domain":"optics","group":"fundamental_principle","entity_type":"physical_principle","confidence":0.95,"reasoning":"Laser is a core optical device","aliases":["laser","LASER","Light Amplification Stimulated Emission"]}',
            "лазер_fiction": '{"domain":"fiction","group":"fictional_concept","entity_type":"cultural_phantom","confidence":0.90,"reasoning":"Death ray in sci-fi","aliases":["death ray","laser weapon","heat ray"]}',
            "лазер_medicine":'{"domain":"medicine","group":"applied_technology","entity_type":"enabler_component","confidence":0.88,"reasoning":"Medical laser therapy","aliases":["medical laser","laser therapy","surgical laser"]}',
        }
        def __init__(self, key): self._key = key
        def complete(self, _prompt): return self._responses[self._key]

    # Один термин — три домена
    cases = [
        ("лазер", "когерентное излучение, инверсная населённость, квантовый переход", "arxiv",   MockLLM("лазер_physics")),
        ("лазер", "лучевое оружие, смерть врагов, фантастический роман",             "fiction",  MockLLM("лазер_fiction")),
        ("лазер", "хирургия сетчатки, офтальмология, клинические испытания",          "medical",  MockLLM("лазер_medicine")),
    ]

    results = []
    for term, ctx, stype, llm in cases:
        r = clf.classify(term, context=ctx, source_type=stype, llm_fallback=llm)
        logger.info(f"  [{stype:8s}] {r.describe()}")
        results.append(r)

    # Проверяем что все три — РАЗНЫЕ canonical_id (разные домены)
    keys = [r.canonical_key for r in results]
    assert len(set(k.hex() for k in keys)) == 3, \
        "FAIL: один термин в разных доменах должен давать разные ключи!"
    logger.info("✓ Разные домены → разные canonical keys")

    # Синоним должен давать ТОТ ЖЕ ключ что и оригинал в том же домене
    # "LASER" идёт через тот же LLM → тот же домен → bulk_register добавит alias
    # "Light Amplification..." тоже был в aliases первого вызова → уже зарегистрирован
    alias_direct = clf.alias_table.lookup("Light Amplification Stimulated Emission",
                                          domain="optics")
    assert alias_direct == keys[0], \
        f"FAIL: alias из LLM должен давать тот же ключ!\n" \
        f"  alias_lookup → {alias_direct.hex() if alias_direct else None}\n" \
        f"  'лазер'      → {keys[0].hex()}"
    logger.info("✓ LLM-alias 'Light Amplification Stimulated Emission' → тот же ключ (physics)")

    # Необходимость: 3 домена → высокий балл
    mock_nodes = {k.hex(): None for k in keys}
    score = clf.necessity_score("лазер", mock_nodes)
    logger.info(f"✓ Necessity score('лазер') = {score:.3f} (3 домена)")

    # Обучение классификатора (если PyTorch доступен)
    if TORCH_AVAILABLE and len(clf.training_data) >= 3:
        logger.info("Training classifier on 3 LLM examples...")
        metrics = clf.train(epochs=2)
        logger.info(f"✓ Train metrics: {metrics}")

        # Предсказание без LLM
        pred = clf.predict("laser", context="quantum optics stimulated emission")
        logger.info(f"✓ Predict (no LLM): {pred.describe()}")

    # CanonicalKey bytes/hex roundtrip
    ck = CanonicalKey.from_bytes(keys[0])
    assert ck.to_bytes() == keys[0], "FAIL: bytes roundtrip"
    logger.info(f"✓ CanonicalKey roundtrip: {ck.describe()}")

    logger.info("=== Smoke Test PASSED ===")


# standalone smoke test removed in merged build
#     _smoke_test_canonical_key()

__all__ = ['ClassificationSample', '_SimpleTextEncoder', 'ConceptClassifierModel', '_ClassifierDataset', 'ConceptClassifier', '_BertEncoderWrapper', 'make_canonical_id', 'CanonicalNodeAssembler', '_smoke_test_canonical_key']
