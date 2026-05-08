 <alphaxiv-thinking-title title="Пишу инструкцию для ИИ-агента" />

## Инструкция для ИИ-ассистента: контекст дипломной работы

Ниже — структурированный бриф, который можно скопировать и вставить в контекстное окно любого ИИ-агента (ChatGPT, Claude, GitHub Copilot и т.д.), чтобы он понимал специфику задачи.

---

## 1. Общая постановка задачи

**Тема:** «Разработка методов адаптивной токенизации с динамическим словарем для анализа вокальной коммуникации летучих мышей»

**Объект:** Вокализации летучих мышей (конкретно — работа с данными фруктовых летучих мышей, аналогичными датасету в статье Assom 2024/2025).

**Цель:** Создать метод токенизации, где словарь вокальных единиц не фиксирован однократной кластеризацией, а может динамически адаптироваться: пополняться, объединять похожие токены, расщеплять неоднородные, формировать составные токены из частых последовательностей.

**Зачем:** Традиционная фиксированная токенизация (кластеризация сегментов) дает нестабильные оценки репертуара (7 типов vs 14 vs 27 в зависимости от конфигурации), плохо переносится между эмиттерами и не учитывает последовательностную структуру коммуникации.

---

## 2. Базовая работа (что берем за основу)

**Статья:** «Associative Syntax and Maximal Repetitions reveal context-dependent complexity in fruit bat communication» (Luigi Assom, 2024/2025, arXiv:2512.01033)

**Что в ней делают:**
- Берут аудиозаписи вокализаций летучих мышей
- Сегментируют на слоги/вокальные единицы
- Mel-спектрограммы → dimensionality reduction (UMAP) → HDBSCAN clustering
- Кластеры = типы вокальных единиц (токены)
- Анализируют последовательности: синтаксис (associative vs compositional), Maximal Repeats (MR), графы переходов, классификация контекста

**Проблема этой работы:** Словарь токенов фиксирован после кластеризации. Нет механизма его адаптации под данные, нет учета sequence statistics при формировании токенов, нестабильность репертуарной оценки.

**Метрики из этой работы (важно):**
- Silhouette Score (качество кластеров)
- ARI/NMI сравнение с acoustic proxy (DTW+MFCC+Agglomerative)
- Classification F1 по контекстам (isolation, fighting, mating, etc.)
- Maximal Repeats (MR) — длины повторяющихся паттернов, распределения
- Network metrics: small-world (σ, ω), плотность, кластеризация
- Assignment rate (доля точек не в шуме)

---

## 3. Наша идея (что пытаемся сделать)

**Концепция:** Предложить метод, где токенизация — это не одноразовый этап, а **динамический процесс**.

**Возможные компоненты адаптивности:**
1. **Adaptive vocabulary size:** Словарь может расти (новые токены), сокращаться (слияние), уточняться (расщепление)
2. **Sequence-aware:** Токены могут формироваться/объединяться на основе статистики последовательностей (BPE-подобное слияние, mutual information)
3. **Iterative refinement:** Как в BEATs — постепенное улучшение tokenizer на основе улучшающихся representations
4. **Multi-granularity:** Иерархия уровней — микротокены и макротокены (составные единицы)

**Что делает метод "адаптивным":**
- Не просто data-driven (как clustering), а **dynamically adjustable** по правилам
- Может реагировать на structure данных (sequence patterns), не только на acoustic similarity
- Может менять гранулярность в зависимости от контекста/статистики

---

## 4. Связанные работы для контекста

**Bioacoustics baseline:**
- Assom 2024: основной baseline
- «Exploring bat song syllable representations in self-supervised audio encoders» (arXiv:2409.12634): для выбора эмбеддингов
- «Identifying birdsong syllables without labelled data» (arXiv:2509.18412): unsupervised syllable discovery

**Discrete tokens / Acoustic unit discovery:**
- «Discrete Audio Tokens: More Than a Survey!» (arXiv:2506.10274): обзор токенизации в аудио
- «How Should We Extract Discrete Audio Tokens from Self-Supervised Models?» (arXiv:2406.10735): практические методы токенизации
- «Unsupervised word segmentation and lexicon discovery using acoustic word embeddings» (arXiv:1603.02845): идеи лексикона и сегментации

**Iterative tokenization (ключевой источник идей):**
- BEATs: Audio Pre-Training with Acoustic Tokenizers (arXiv:2212.09058): итеративное улучшение tokenizer, self-distilled tokens, semantic acoustic tokens

**Cross-modal / SSL для bioacoustics:**
- animal2vec, MeerKAT (arXiv:2406.01253): SSL для биоакустики

---

## 5. Технические детали (что можем использовать)

**Представления сигналов:**
- Mel-спектрограммы (baseline)
- SSL embeddings (HuBERT, Wav2Vec 2.0, BEATs, YAMNet, VGGish)
- Self-supervised representations trained on animal vocalizations или general audio

**Методы кластеризации/токенизации:**
- HDBSCAN (baseline)
- Online clustering / pseudo-labeling
- Vector Quantization (VQ-VAE style)
- Differentiable k-means

**Sequence modeling:**
- BPE/WordPiece-like merging для акустических токенов
- Mutual information между соседними токенами
- Language modeling на токен-последовательностях

**Dynamic vocabulary механизмы:**
- Add: новый токен при outlier с достаточной поддержкой
- Merge: слияние близких токенов по criterion (distance, co-occurrence, MI)
- Split: расщепление при высокой внутрикластерной дисперсии
- Prune: удаление редких токенов
- Merge frequent pairs: BPE-подобное на акустических токенах

---

## 6. Оценка (как проверить что получилось лучше)

**По качеству токенов:**
- Silhouette Score (не ниже baseline)
- ARI/NMI с acoustic proxy (выше — лучше)
- Stability (согласованность между запусками)
- Assignment rate (доля assigned vs noise)

**По полезности для анализа коммуникации:**
- Classification F1 по контекстам (выше — лучше)
- Тест пермутаций: если делаем sequence-aware, можно проверить влияние порядка
- Maximal Repeats: распределения длин, доля длинных MRs
- Network metrics: small-world properties в графах переходов

**По свойствам словаря:**
- Размер словаря (не должен взрываться)
- Переиспользуемость токенов между эмиттерами
- Сжимаемость последовательностей (cross-entropy language model)

---

## 7. Что мы НЕ делаем (границы)

- НЕ используем ground-truth метки контекста при построении токенов (это было бы cheating для downstream классификации)
- НЕ делаем end-to-end supervised classification (токенизация должна быть unsupervised/semi-supervised)
- НЕ претендуем на发现的 "истинные слоги" летучих мышей — мы предлагаем лучший способ их аппроксимации через дискретные единицы

---

## 8. Что агент должен знать (summary для быстрого старта)

**Когда просите помощь у ИИ, указывайте:**

1. **Контекст:** Работаем с bat vocalizations, graded signals, нужна adaptive tokenization
2. **Baseline:** Assom 2024 — mel + UMAP + HDBSCAN, фиксированные кластеры
3. **Идея:** Динамический словарь — может меняться, sequence-aware, possibly iterative
4. **Источник инспирации:** BEATs (iterative tokenizer refinement), BPE (merge frequent pairs), maybe DP-means or online clustering
5. **Метрики:** Silhouette, ARI/NMI, classification F1, Maximal Repeats, network properties, stability
6. **Ограничение:** Unsupervised или self-supervised, не используем контекстные метки для токенизации

---

## 9. Примеры запросов к агенту (как использовать эту инструкцию)

**Пример 1:**  
«Я работаю над adaptive tokenization для bat vocalizations. Базовый метод — это кластеризация HDBSCAN поверх UMAP на мел-спектрограммах (статья Assom 2024). Я хочу сделать словарь динамическим — чтобы он мог объединять частые пары токенов. Вдохновляюсь BEATs (iterative tokenizer refinement) и BPE. Как бы ты предложил реализовать token merging на основе mutual information или частоты биграмм?»

**Пример 2:**  
«Помоги с evaluation метриками для adaptive tokenization. У меня есть baseline с Silhouette ~0.5 и 7 типов токенов. Я хочу показать, что моя адаптивная версия лучше. Какие метрики помимо Silhouette и ARI/NMI использовать, чтобы показать преимущество динамического словаря для анализа коммуникации?»

**Пример 3:**  
«Мне нужен код для online clustering с add/merge/split логикой. Сигналы — это эмбеддинги аудиосегментов из летучих мышей. Начальный словарь из HDBSCAN. Хочу, чтобы новые сегменты могли создавать новые токены, а близкие токены объединялись. Есть ли готовые реализации или примеры такого adaptive vocabulary?»
