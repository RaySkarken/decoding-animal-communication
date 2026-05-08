# Главный эксперимент: контекстно-зависимая адаптивная токенизация

Этот раздел описывает воспроизведение центрального эксперимента диссертации.

## Что здесь

- **`per_context_main_experiment.ipynb`** — исполняемый ноутбук, воспроизводящий все таблицы и рисунки главы 3 с нуля. 22 ячейки кода, все выходы сохранены внутри `.ipynb`.
- **`../src/per_context_tokenizer.py`** — импортируемый модуль с реализацией метода:
  - `DPGMMTokenizer`, `HDBSCANTokenizer`, `KMeansTokenizer` — три варианта инструмента построения словаря.
  - `PerContextFamily` — обёртка с правилом Байеса $\hat c = \arg\max_c \log p_c(x) + \log p(c)$ и опциональным `prior_counts` для корректного подсчёта prior на уровне единицы классификации (вокализации).
- **`../scripts/build_main_notebook.py`** — генератор ноутбука из Python-кода.
- **`../scripts/make_thesis_figures_v2.py`** — генератор PDF-фигур для главы 3 из CSV/JSON результатов.

## Как воспроизвести с нуля

### Требования

- Python 3.13 в виртуальном окружении `.venv/` (или `animal-comm` conda).
- Основные зависимости: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`, `joblib`, `hdbscan`, `jupyter`, `nbformat`.
- Кэшированное состояние baseline-пайплайна: `/Volumes/T7/cache/assom_paper_repro/ablation_state.joblib` (~1.5 ГБ), воспроизводится из `../notebooks/assom_paper_reproduction.ipynb`.

### Шаг 1. Выполнить baseline-воспроизведение

Если `ablation_state.joblib` отсутствует:

```bash
cd /path/to/AnimalCommunication
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/assom_paper_reproduction.ipynb
```

Это сохранит мел-спектрограммы, UMAP-эмбеддинг, HDBSCAN+NCA-разметку и прочее.

### Шаг 2. Выполнить главный эксперимент

```bash
cd /path/to/AnimalCommunication
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=2400 \
    notebooks/per_context_main_experiment.ipynb
```

Время выполнения: ~15–25 минут на 5 сидов × 3 tokenizer-варианта + HDP-approx ablation + prior ablation.

### Шаг 3. Переместить выходные файлы и сгенерировать фигуры

Nbconvert выполняет ноутбук из директории `notebooks/`, поэтому относительные пути CSV падают в `notebooks/docs/thesis/figures/`. После выполнения:

```bash
cp -f notebooks/docs/thesis/figures/*.csv notebooks/docs/thesis/figures/*.json \
    docs/thesis/figures/
rm -rf notebooks/docs

python3 scripts/make_thesis_figures_v2.py
```

Фигуры сохранятся в `docs/thesis/latex/images/*.pdf`.

### Шаг 4. Пересобрать PDF диссертации

```bash
cd docs/thesis/latex
latexmk -xelatex -interaction=nonstopmode main.tex
```

Результат: `main.pdf`. Рекомендуется также скопировать в удобочитаемое место:

```bash
cp main.pdf ../Thesis_full_draft.pdf
```

## Ключевые результаты ноутбука (для справки)

Все цифры — 5 независимых разбиений корпуса по особям (30 train / 11 test):

| Метод | $F_1$ weighted (mean ± std) | $\Delta$ vs baseline | 95%-ДИ |
|---|---|---|---|
| **Per-context DP-GMM** | **0.433 ± 0.084** | **+0.087 ± 0.038** | [+0.012, +0.163] |
| Per-context HDBSCAN | 0.404 ± 0.076 | +0.059 ± 0.020 | [+0.018, +0.099] |
| Per-context k-means | 0.439 ± 0.081 | +0.093 ± 0.030 | [+0.033, +0.154] |
| Assom + RF (baseline) | 0.346 ± 0.075 | — | — |

Для всех трёх вариантов 95%-ДИ выигрыша над baseline строго положителен.

HDP-approx (частичное объединение атомов) проигрывает полностью раздельным словарям на всех 5 сидах, средняя разница +0.140.

## Важный методологический момент

Априорное распределение $p(c)$ в правиле Байеса должно оцениваться как доля **вокализаций** контекста $c$ в обучающей выборке, а не как доля **сегментов**. Единицей классификации является вокализация, поэтому prior оценивается на том же уровне. Использование prior по сегментам снижает $F_1$ примерно на 0.05 и делает выигрыш над baseline статистически незначимым. Модуль `PerContextFamily` принимает опциональный параметр `prior_counts` именно для передачи правильного prior.

## Файлы результатов

- `../docs/thesis/figures/main_experiment_5seeds.csv` — «сырые» F1 по сидам и методам.
- `../docs/thesis/figures/main_experiment_summary.json` — агрегированная сводка (среднее, стандартное отклонение, доверительные интервалы, HDP-approx аблация, prior-аблация, размеры словарей).
- `../docs/thesis/figures/per_class_seed0.csv` — per-class $F_1$ для seed 0.
- `../docs/thesis/latex/images/*.pdf` — семь рисунков главы 3.
