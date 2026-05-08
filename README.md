# Контекстно-зависимая адаптивная токенизация для биоакустики

Воспроизводимый программный пакет к магистерской ВКР *«Контекстно-зависимая
адаптивная токенизация с динамическим словарём для анализа вокальной
коммуникации животных»*, ИТМО, 2026.

Корпус — публичный набор вокализаций египетской фруктовой летучей мыши
(*Rousettus aegyptiacus*), [Prat и соавт., 2017](https://www.nature.com/articles/sdata2017143).

## Основной результат

Per-context DP-GMM с полной ковариационной матрицей на UMAP-8D с
равномерным априорным распределением классифицирует тип коммуникации на
контрольной выборке особей с **macro F1 = 0.313 ± 0.005** против
**0.237 ± 0.018** у опорного пайплайна Assom 2025 (выигрыш +0.076,
95%-доверительный интервал [+0.036, +0.116]; 5 разбиений по особям, 30
обучающих / 11 контрольных эмиттеров).

## Структура репозитория

```
src/                          — Python-пакет с реализацией
  data.py                       загрузка корпуса, сегментация, ресемплирование
  features.py                   мел-фронтенд + BEATs/AVES обёртки
  tokenizer.py                  AdaptiveTokenizer (BPE + HDBSCAN seed)
  per_context_tokenizer.py      контекстно-зависимое семейство DP-GMM/k-means
  sequence.py                   построение последовательностей, RF, HP1
  eval.py                       silhouette / ARI / NMI / MR / network metrics
  proxy_labels.py               DTW + Ward proxy-метки
  beats/                        vendored Microsoft BEATs (см. ниже)
  adaptive_tokenizer/           алгоритм адаптивной токенизации
notebooks/                    — рабочие ноутбуки по экспериментам
scripts/                      — скрипты воспроизведения экспериментов главы 3
tests/                        — модульные тесты
requirements.txt              — список зависимостей Python
environment.yml               — описание conda-окружения animal-comm
```

## Запуск

```bash
# вариант 1: conda
conda env create -f environment.yml
conda activate animal-comm

# вариант 2: venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Загрузка корпуса (~60 ГБ, FigShare):

```bash
python scripts/download_figshare_fruitbat.py
python scripts/unzip_figshare_fruitbat.py
```

Воспроизведение основного результата (5 разбиений по особям,
рекомендуемая конфигурация — DP-GMM full UMAP-8D + равномерный приор):

```bash
python scripts/per_class_f1_chart_uniform.py
# результат: docs/thesis/figures/per_class_f1_uniform.{png,pdf,csv}
# macro F1 = 0.313 ± 0.005, weighted F1 = 0.470 ± 0.038
```

## Окружение и ресурсы

- Python 3.11, основные зависимости: `numpy`, `pandas`, `scikit-learn`,
  `umap-learn`, `hdbscan`, `librosa`, `tensorflow>=2.13`,
  `torch`+`torchaudio`, `safetensors`. GitHub-only зависимости:
  [`avgn`](https://github.com/timsainb/avgn_paper),
  [`vocalseg`](https://github.com/timsainb/vocalization-segmentation).
- Железо: Apple M1 Pro, 16 ГБ RAM (CPU-режим). Полный прогон 5-сидового
  основного эксперимента — около 25 минут на CPU.
- Объёмные производные артефакты (UMAP-эмбеддинги, HDBSCAN-разметки,
  закешированные BEATs/AVES-эмбеддинги) не хранятся в репозитории —
  лежат на внешнем SSD по пути `/Volumes/T7/cache/assom_paper_repro/`.
  Скрипты в `scripts/` пересоздают их на лету при первом запуске.

## Связанные репозитории (как ссылки, не вендорятся)

В репозитории присутствуют ссылки на чужие кодовые базы, которые
использовались как источники методики или данные. Ниже их публичные
адреса; локально они подгружаются как git submodules / отдельные
клоны и в этот репозиторий не включены:

- **[Assom — `decodingNonHumanCommunication`](https://github.com/luigassom/decodingNonHumanCommunication)** —
  опорный пайплайн настоящей работы (Assom 2025, arXiv:2512.01033).
  Параметры и расхождения текста с кодом подробно разобраны в §3.1
  ВКР.
- **[NatureLM-audio](https://github.com/earth-species-project/NatureLM-audio)** —
  audio-language foundation model (Robinson и соавт., 2024,
  arXiv:2411.07186); используется только её BEATs-кодировщик (после
  слияния в `beats_encoder_merged.pt`).
- **[FruitBat-vocalizations-open-data](https://figshare.com/articles/dataset/An_annotated_dataset_of_Egyptian_fruit_bat_vocalizations/5288381)** —
  публичный корпус Prat и соавт. 2017 (FigShare).
- **[Microsoft BEATs (`unilm/beats`)](https://github.com/microsoft/unilm/tree/master/beats)** —
  каталог `src/beats/` в этом репозитории — точная копия с
  патченными относительными импортами (см. CLAUDE.md). Не
  переоформляется в пакет, чтобы сохранить совместимость с весами.

## Раскрытие AI-инструментов

При подготовке настоящей работы автор использовал генеративные модели
ИИ (Claude, GPT-4) для черновой генерации фрагментов кода, поиска
литературы и редактуры текста. Все приведённые в работе численные
результаты получены повторным запуском скриптов и не воспроизводятся
непосредственно из выдачи языковых моделей. Подробнее — во введении ВКР
(раздел «Использование инструментов искусственного интеллекта»).

## Лицензия

MIT, см. [LICENSE](LICENSE).
