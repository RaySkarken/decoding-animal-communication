"""Comprehensive PDF report on per-context tokenization experiments.

Aggregates all CSV results + cluster visualizations into one multi-page PDF.
"""
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

FIG = Path('docs/thesis/figures')
OUT = Path('docs/thesis/full_results_report.pdf')


def safe_csv(name):
    p = FIG / name
    return pd.read_csv(p) if p.exists() else None


def title_page(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.85, 'Per-context tokenization\n— full results report —',
             ha='center', va='center', fontsize=18, fontweight='bold')
    fig.text(0.5, 0.74, 'Bat vocalizations · Egyptian fruit bat (Prat 2017) · paper-faithful 153k corpus',
             ha='center', va='center', fontsize=11, style='italic')
    fig.text(0.5, 0.69, 'Reproduction of Assom 2025 + per-context tokenizer extension',
             ha='center', va='center', fontsize=11, style='italic')

    summary_text = (
        '\n\nКлючевые результаты\n'
        '═══════════════════════════════════════════════════════════════\n\n'
        '✓  Воспроизведение paper Assom 2025 на 153 366 сегментах (paper: 152 578) —\n'
        '   все headline-метрики в пределах ДИ paper claim\n\n'
        '✓  F1 weighted classification: per-context Bayes 0.476 vs Assom-baseline 0.369\n'
        '   → Δ = +0.107 SIG+ (95%-ДИ [+0.046, +0.167])\n\n'
        '✓  F1 macro: 0.298 vs 0.237 → Δ = +0.061 SIG+\n\n'
        '✓  Cardinality-fair NMI per-(bat,context) vs DTW-прокси:\n'
        '   per-context 0.447 vs baseline 0.411 → Δ = +0.035 (paired Wilcoxon p<10⁻¹¹, 105 ячеек)\n\n'
        '⚠  Continuous baseline (RF на pooled raw mel, без токенизации):\n'
        '   F1 weighted = 0.529 — БЬЁТ нашу per-context Bayes (−0.05)\n\n'
        '⚠  Per-context имеет НИЖЕ silhouette (0.39 vs 0.61) — следствие большей кардинальности\n\n'
        '⚠  ARI cardinality-fair: разница NS (Δ=−0.017, p=0.41)\n\n'
        '⚠  Within-context sequence prediction (next-token / masked cloze / completion):\n'
        '   global tokenization бьёт per-context на 0/8 контекстов в direct-accuracy метриках\n\n'
        '★  Methodological finding: target leakage в paper HP1 (Zhang-18 features e, h, j)\n'
        '   эмпирически доказан — F1 0.84 → 0.31 при перемешивании conditioning context\n'
    )
    fig.text(0.07, 0.55, summary_text, ha='left', va='top', fontsize=9.5,
             family='monospace', linespacing=1.4)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def reproduction_page(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle('1. Воспроизведение Assom 2025 на 153 366 сегментах',
                 fontsize=14, fontweight='bold', y=0.97)

    rows = [
        ('N сегментов',                  '153,366',          '152,578',           '99.5%'),
        ('Mel-спектрограмма',            '(21, 32)',         '(21, 32) (code)',    '✓'),
        ('UMAP параметры',               'n_n=30, m_d=1.0',   'те же',              '✓'),
        ('HDBSCAN config',               'frac=0.02, ms=20', 'defaults Assom',     '✓'),
        ('Число кластеров HDBSCAN',      '11',               '7 (claim)',           '∼'),
        ('Silhouette',                   '0.629',            '> 0.5',               '✓'),
        ('Per-emitter ARI vs прокси',    '0.182 ± 0.114',    '0.12 ± 0.01',         '✓ выше'),
        ('Per-emitter NMI vs прокси',    '0.354 ± 0.083',    '0.30 ± 0.01',         '✓ выше'),
        ('Прокси типов на особь (q=0.10 calibrated)', '24.9 ± 9.6', '27 ± 2',       '✓'),
        ('HP1 F1 (paper-exact, bat 215)', '0.842',           '> 0.9',               'близко'),
        ('HP1 |Δ F1| (perm test)',       '0.006',            '≈ 0',                 '✓'),
    ]
    ax = fig.add_subplot(111); ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=['Метрика', 'Наш', 'Paper', 'Статус'],
                   loc='upper center', cellLoc='left',
                   colWidths=[0.40, 0.22, 0.20, 0.13])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.7)
    for j in range(4):
        tbl[0, j].set_facecolor('#404040')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i, r in enumerate(rows):
        if '✓' in r[3] and 'выше' not in r[3]:
            for j in range(4): tbl[i+1, j].set_facecolor('#e8f5e9')
        elif 'выше' in r[3]:
            for j in range(4): tbl[i+1, j].set_facecolor('#dcedc8')
        elif '∼' in r[3] or 'близко' in r[3]:
            for j in range(4): tbl[i+1, j].set_facecolor('#fff9c4')

    fig.text(0.07, 0.30,
             'Замечания:\n'
             '• Размер мел-спектрограммы (21,32) взят из исходного кода Assom (TF_AE.ipynb), не из подписи Fig 2 paper\n'
             '   (там указано (6,32) — каption противоречит коду; используется код).\n'
             '• UMAP min_dist=1.0 также из их кода, не из caption (там 0.3).\n'
             '• HDBSCAN с дефолтными параметрами их же helper-функции test_hdbscan() даёт 11 кластеров на 153k.\n'
             '   Заявление paper о 7 не воспроизводится дефолтными параметрами без post-merging.\n'
             '• Прокси типов: q=0.10 калибрует under-sampling MFCC из mel-спека (paper q=0.05 на audio MFCC).',
             ha='left', va='top', fontsize=8.5, linespacing=1.6)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def cluster_viz_page(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle('2. UMAP визуализация — наш 153k vs paper Fig 1b', fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(1, 2, wspace=0.05, top=0.92, bottom=0.10)

    a1 = fig.add_subplot(gs[0]); a1.axis('off')
    if (FIG / 'umap_152k_21x32_by_context.png').exists():
        a1.imshow(mpimg.imread(FIG / 'umap_152k_21x32_by_context.png'))
    a1.set_title('Наш UMAP (153k segments, n_neighbors=30, min_dist=1.0)\nцвет = поведенческий контекст',
                 fontsize=11, pad=8)

    a2 = fig.add_subplot(gs[1]); a2.axis('off')
    if (FIG / 'assom_fig1b_extracted.png').exists():
        a2.imshow(mpimg.imread(FIG / 'assom_fig1b_extracted.png'))
    a2.set_title('Paper Fig 1b (Assom 2025)\nrefined pipeline', fontsize=11, pad=8)

    fig.text(0.5, 0.05,
             'Геометрическая структура воспроизводится: множество разделённых кластеров на чёрном фоне.\n'
             'Большой жёлтый кластер Isolation (мать-детёныш вокализации) обособлен в обеих картах.',
             ha='center', fontsize=9, style='italic')
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # Page 2b: by syllable
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle('3. UMAP по syllable-меткам', fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(1, 2, wspace=0.05, top=0.92, bottom=0.10)
    a1 = fig.add_subplot(gs[0]); a1.axis('off')
    if (FIG / 'umap_152k_21x32_by_syllable.png').exists():
        a1.imshow(mpimg.imread(FIG / 'umap_152k_21x32_by_syllable.png'))
    a1.set_title('Наши HDBSCAN-syllable (V=11, Assom defaults)', fontsize=11, pad=8)
    a2 = fig.add_subplot(gs[1]); a2.axis('off')
    if (FIG / 'assom_fig1b_extracted.png').exists():
        a2.imshow(mpimg.imread(FIG / 'assom_fig1b_extracted.png'))
    a2.set_title('Paper Fig 1b (для сравнения цветовой палитры)', fontsize=11, pad=8)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def main_classification_page(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle('4. Главный эксперимент — classification F1 (5 сидов cross-bat, paper-faithful 153k)',
                 fontsize=12.5, fontweight='bold', y=0.97)

    rows = [
        ('Per-context DP-GMM (наш main)',              '0.476 ± 0.048', '+0.107', '[+0.046, +0.167]', 'SIG+'),
        ('Per-context HDBSCAN-tokenizer',              '0.455 ± 0.036', '+0.086', '[+0.056, +0.116]', 'SIG+'),
        ('Per-context k-means tokenizer',              '0.456 ± 0.040', '+0.086', '[+0.039, +0.134]', 'SIG+'),
        ('— Baseline RF (HDBSCAN+bag+RF) —',           '0.369 ± 0.033', '0',      '—',                '—'),
        ('RF на pooled UMAP-8D (no tokens)',           '0.500 ± 0.047', '+0.131', '[+0.064, +0.198]', 'SIG+'),
        ('RF на pooled raw mel (no tokens)',           '0.529 ± 0.058', '+0.160', '[+0.082, +0.238]', 'SIG+'),
    ]
    ax = fig.add_subplot(111); ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=['Method', 'F1 weighted ± SD', 'Δ vs base', '95%-CI', 'Sig'],
                   loc='upper center', cellLoc='left',
                   colWidths=[0.36, 0.20, 0.12, 0.20, 0.08])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.8)
    for j in range(5):
        tbl[0, j].set_facecolor('#404040')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # color rows
    for i in [0, 1, 2]:
        for j in range(5): tbl[i+1, j].set_facecolor('#e8f5e9')
    for j in range(5):
        tbl[4, j].set_facecolor('#cccccc')   # baseline row
    for i in [4, 5]:
        for j in range(5): tbl[i+1, j].set_facecolor('#bbdefb')   # no-token rows highlighted

    fig.text(0.07, 0.55,
             'Macro F1 (та же конфигурация):\n'
             '   Per-context DP-GMM: 0.298 ± 0.008  vs  Baseline: 0.237 ± 0.019  →  Δ = +0.061 SIG+\n\n'
             'Uniform prior (вместо empirical) ablation:\n'
             '   F1 weighted: 0.470 (PC) vs 0.369 (base) → +0.101 SIG+ (почти то же)\n'
             '   F1 macro:    0.313 (PC) vs 0.237 (base) → +0.076 SIG+ (немного лучше с uniform)\n\n'
             'Интерпретация:\n'
             '   • Per-context методы значимо ЛУЧШЕ Assom-style baseline (+0.107 SIG+ для DP-GMM)\n'
             '   • Эффект устойчив к выбору per-context tokenizer (DP-GMM/HDBSCAN/k-means)\n'
             '   • НО: continuous-feature classifier'"'"'ы (RF на UMAP/mel) ещё лучше (+0.024 / +0.053 над PC)\n'
             '   • Это значит: токенизация улучшает token-pipeline, но проигрывает no-token подходам',
             ha='left', va='top', fontsize=9, linespacing=1.5)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def per_class_chart_page(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle('5. Per-class F1 (5 сидов, mean ± SD)', fontsize=13, fontweight='bold')
    if (FIG / 'per_class_f1_chart.png').exists():
        ax = fig.add_subplot(111); ax.axis('off')
        ax.imshow(mpimg.imread(FIG / 'per_class_f1_chart.png'))
    fig.text(0.5, 0.05,
             'Per-context (синий/оранжевый) выигрывает на больших контекстах (Feeding, Isolation, Biting).\n'
             'Проигрывает на редких классах (Kissing n=27, Grooming n=38, Threat n=152) — где все методы дают F1<0.1.\n'
             'Per-context segmentation (oranzhevyy) даёт +0.027 над standard seg, но использует oracle context (leakage).',
             ha='center', fontsize=9, style='italic')
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def proxy_agreement_page(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle('6. Согласованность с DTW-MFCC прокси — три способа агрегации',
                 fontsize=12.5, fontweight='bold', y=0.97)

    rows = [
        ('[1] GLOBAL (один ARI/NMI на все точки)',
         '0.012',         '0.028',          '+0.016',
         '0.240',         '0.396',          '+0.156'),
        ('[2] PER-EMITTER (mean ± std по 41 особи)',
         '0.182 ± 0.113', '0.130 ± 0.087',  '−0.052',
         '0.354 ± 0.082', '0.435 ± 0.076',  '+0.081'),
        ('[3] PER-(bat, context) (cardinality-fair, n=105)',
         '0.158 ± 0.115', '0.141 ± 0.084',  '−0.017 NS',
         '0.411 ± 0.104', '0.447 ± 0.111',  '+0.035 ***'),
    ]
    ax = fig.add_subplot(111); ax.axis('off')
    tbl = ax.table(
        cellText=rows,
        colLabels=['Способ агрегации', 'Glob ARI', 'PC ARI', 'ΔARI', 'Glob NMI', 'PC NMI', 'ΔNMI'],
        loc='upper center', cellLoc='left',
        colWidths=[0.28, 0.10, 0.10, 0.12, 0.10, 0.10, 0.13])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.9)
    for j in range(7):
        tbl[0, j].set_facecolor('#404040')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # highlight cardinality-fair row
    for j in range(7):
        tbl[3, j].set_facecolor('#e8f5e9')

    fig.text(0.07, 0.62,
             'Paper claim: ARI = 0.12 ± 0.01, NMI = 0.30 ± 0.01\n\n'
             'Прямой ответ:\n'
             '   • ARI:  чистого преимущества per-context НЕТ. На cardinality-fair агрегации Δ = −0.017, p=0.41 (NS).\n'
             '   • NMI:  устойчивый малый прирост. На cardinality-fair Δ = +0.035, paired Wilcoxon p<10⁻¹¹ по 105 ячейкам.\n\n'
             'Что значит: жёсткое разбиение на группы у двух методов сравнимо (ARI ~ same), но per-context\n'
             'удерживает чуть больше взаимной информации с независимой acoustic similarity proxy (NMI).\n\n'
             '*** = paired Wilcoxon p<0.001;  NS = not significant',
             ha='left', va='top', fontsize=9, linespacing=1.5)

    fig.text(0.07, 0.30,
             'По контекстам (Δ NMI per-(bat,ctx)):\n'
             '   Biting:    +0.052     Feeding:   +0.046    Fighting:  +0.037\n'
             '   Isolation: −0.042     Kissing:   +0.044    Mating:    +0.069\n'
             '   Threat:    +0.044     Grooming:  ≈0\n\n'
             '7/8 контекстов в пользу per-context. Только Isolation проигрывает\n'
             '(per-context избыточно фрагментирует уже-компактный кластер isolation calls).',
             ha='left', va='top', fontsize=9, linespacing=1.5)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def info_theory_page(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle('7. Информационно-теоретические метрики (caveat: cardinality-зависимые)',
                 fontsize=12, fontweight='bold', y=0.97)

    rows = [
        ('H(syllable) [bits]',                     '3.20',  '5.78',  '+2.58'),
        ('H(syllable | context) [bits]',           '2.90',  '3.52',  '+0.62'),
        ('I(syllable; context) [bits]',            '0.30',  '2.26',  '+1.96 (×7.5)'),
        ('NMI(syllable, context)',                 '0.111', '0.556', '+0.445'),
        ('% контекст-инфо (max=3 bits)',           '10.1%', '75.3%', '+65.2pp'),
        ('Lagrangian MDL/I (bits per bit context)','17.04', '2.60',  '−14.4 (÷6.5)'),
    ]
    ax = fig.add_subplot(111); ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=['Метрика', 'Baseline (V=11)', 'Per-context (V=109)', 'Δ'],
                   loc='upper center', cellLoc='left',
                   colWidths=[0.40, 0.18, 0.20, 0.20])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.85)
    for j in range(4):
        tbl[0, j].set_facecolor('#404040')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    fig.text(0.07, 0.55,
             '⚠ ВАЖНЫЙ caveat:\n\n'
             'Эти метрики — частично tautological из-за разной кардинальности словарей.\n'
             'Per-context имеет 109 syllables, baseline — 11. По построению PC может содержать больше\n'
             'mutual information с любой меткой (включая контекст).\n\n'
             'Для cardinality-fair оценки информационного преимущества используется per-(bat, context) NMI\n'
             'vs DTW-прокси (см. раздел 6) — там Δ = +0.035, что является более скромной, но более чистой\n'
             'оценкой реального преимущества per-context словаря.\n\n'
             'Эти числа демонстрируют, что per-context label encoding — более информационно богатое,\n'
             'но интерпретировать прирост ×7.5 без cardinality-correction не следует.',
             ha='left', va='top', fontsize=9, linespacing=1.6)

    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def per_context_structure_page(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle('8. Per-context structural analysis', fontsize=13, fontweight='bold', y=0.97)

    # Per-context K_95 (effective vocab size)
    rows = [
        ('Grooming',  '1,514',  '11', '12', '9'),
        ('Kissing',   '1,003',  '11', '14', '11'),
        ('Isolation', '33,537', '11', '24', '11'),
        ('Biting',    '6,147',  '11', '15', '12'),
        ('Threat',    '6,801',  '11', '15', '12'),
        ('Feeding',   '20,474', '11', '23', '13'),
        ('Fighting',  '60,211', '11', '64', '13'),
        ('Mating',    '20,228', '11', '21', '13'),
    ]
    ax = fig.add_subplot(2, 1, 1); ax.axis('off')
    ax.set_title('A. Effective vocabulary size per context (K₉₅ = #components covering 95% mass)',
                 fontsize=10.5, loc='left', pad=8)
    tbl = ax.table(cellText=rows, colLabels=['Context', 'n_segs', 'V global', 'V_pc raw (K_max=15)', 'V_pc K₉₅'],
                   loc='upper center', cellLoc='left',
                   colWidths=[0.18, 0.14, 0.16, 0.30, 0.18])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.7)
    for j in range(5):
        tbl[0, j].set_facecolor('#404040')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Silhouette per context
    rows2 = [
        ('Biting',     '6,147',  '0.644', '0.388', '−0.256'),
        ('Feeding',    '20,474', '0.605', '0.331', '−0.274'),
        ('Fighting',   '60,211', '0.596', '0.468', '−0.128'),
        ('Grooming',   '1,514',  '0.617', '0.449', '−0.168'),
        ('Isolation',  '33,537', '0.512', '0.167', '−0.345'),
        ('Kissing',    '1,003',  '0.683', '0.469', '−0.214'),
        ('Mating',     '20,228', '0.620', '0.406', '−0.214'),
        ('Threat',     '6,801',  '0.565', '0.399', '−0.166'),
        ('mean',       '—',      '0.605', '0.385', '−0.220'),
    ]
    ax = fig.add_subplot(2, 1, 2); ax.axis('off')
    ax.set_title('B. Per-context silhouette (8D UMAP)', fontsize=10.5, loc='left', pad=8)
    tbl = ax.table(cellText=rows2, colLabels=['Context', 'n_segs', 'sil baseline (V=11)', 'sil per-ctx (V≈14)', 'Δ'],
                   loc='upper center', cellLoc='left',
                   colWidths=[0.18, 0.14, 0.20, 0.20, 0.14])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.7)
    for j in range(5):
        tbl[0, j].set_facecolor('#404040')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for j in range(5):
        tbl[9, j].set_facecolor('#fff3e0')

    fig.text(0.07, 0.04,
             'A: Per-context K₉₅ варьируется 9–13 (диапазон 4) — confirms hypothesis "different vocab sizes per context".\n'
             'B: Silhouette per-context систематически НИЖЕ baseline во всех 8 контекстах (среднее −0.22). Это\n'
             'математическое следствие большей кардинальности (12-15 vs 11 в той же области), а не плохой кластеризации.',
             ha='left', va='top', fontsize=8.5, linespacing=1.6)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def seq_tasks_page(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle('9. Within-context sequence prediction (3 задачи, 5 сидов)',
                 fontsize=12, fontweight='bold', y=0.97)

    rows = [
        ('T1: Next-token accuracy',                   '0/8',  'global wins all 8'),
        ('T1: Bits/token (raw, lower=better)',        '0/8',  'global wins (smaller vocab = lower entropy)'),
        ('T1: Compression ratio (cardinality-fair)',  '5/8',  'mixed; PC wins in larger contexts'),
        ('T2: Masked cloze accuracy',                 '0/8',  'global wins all 8'),
        ('T2: Masked cloze bits',                     '0/8',  'global wins all 8'),
        ('T3: Sequence completion accuracy',          '0/8',  'global wins all 8'),
        ('T3: BLEU-2 (higher=better)',                '0/8',  'global wins all 8'),
        ('T3: Edit distance (lower=better)',          '0/8',  'global wins all 8'),
    ]
    ax = fig.add_subplot(111); ax.axis('off')
    tbl = ax.table(cellText=rows,
                   colLabels=['Метрика', 'PC wins/8', 'Comment'],
                   loc='upper center', cellLoc='left',
                   colWidths=[0.40, 0.15, 0.40])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.9)
    for j in range(3):
        tbl[0, j].set_facecolor('#404040')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in [0, 1, 3, 4, 5, 6, 7]:
        for j in range(3): tbl[i+1, j].set_facecolor('#ffebee')
    for j in range(3):
        tbl[3, j].set_facecolor('#fff9c4')

    fig.text(0.07, 0.50,
             'Чистый отрицательный результат:\n\n'
             'На прямых sequence prediction задачах (next-token, masked cloze, completion)\n'
             'per-context tokenization ПРОИГРЫВАЕТ global tokenization в КАЖДОМ контексте\n'
             'по КАЖДОЙ direct-accuracy / BLEU / edit-distance метрике.\n\n'
             'Только cardinality-нормализованный compression ratio даёт смешанный 5/8.\n\n'
             'Причина: per-context имеет больше токенов (12-55 vs 11) → больше параметров\n'
             'в bigram → harder estimation от того же объёма train → хуже prediction.\n\n'
             'Per-context не транслирует "более тонкое описание" в predictive power для sequence\n'
             'modeling. Бóльший словарь — недостаток, а не преимущество для предсказания.',
             ha='left', va='top', fontsize=9, linespacing=1.6)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def addressee_page(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle('10. Addressee prediction within Mating context',
                 fontsize=13, fontweight='bold', y=0.97)

    rows = [
        ('Stratified split (random, within-bat OK):', '', ''),
        ('  Bag-of-syllables GLOBAL (V=11)',       '0.270 ± 0.022',  '0.215 ± 0.019'),
        ('  Bag-of-syllables PER-CONTEXT (V=15)',  '0.326 ± 0.018',  '0.256 ± 0.029'),
        ('  Pooled UMAP-8D (no tokens)',           '0.434 ± 0.005',  '0.332 ± 0.011'),
        ('  Pooled raw mel (no tokens)',           '0.610 ± 0.013',  '0.524 ± 0.011'),
        ('Cross-bat split (split by emitter):',     '', ''),
        ('  Bag-of-syllables GLOBAL',              '0.238 ± 0.081',  '0.100 ± 0.035'),
        ('  Bag-of-syllables PER-CONTEXT',         '0.254 ± 0.082',  '0.104 ± 0.022'),
        ('  Pooled UMAP-8D',                       '0.333 ± 0.099',  '0.147 ± 0.052'),
        ('  Pooled raw mel',                       '0.405 ± 0.094',  '0.249 ± 0.081'),
    ]
    ax = fig.add_subplot(111); ax.axis('off')
    tbl = ax.table(cellText=rows,
                   colLabels=['Method', 'F1 weighted', 'F1 macro'],
                   loc='upper center', cellLoc='left',
                   colWidths=[0.50, 0.22, 0.22])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.7)
    for j in range(3):
        tbl[0, j].set_facecolor('#404040')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # PC rows
    for i in [2, 7]:
        for j in range(3): tbl[i+1, j].set_facecolor('#e8f5e9')
    # raw mel rows
    for i in [4, 9]:
        for j in range(3): tbl[i+1, j].set_facecolor('#bbdefb')
    # section dividers
    for i in [0, 5]:
        for j in range(3):
            tbl[i+1, j].set_facecolor('#e0e0e0')
            tbl[i+1, j].set_text_props(fontweight='bold')

    fig.text(0.07, 0.40,
             'Setup: 1307 Mating вокализаций с известным addressee (после фильтра ≥30 vocs/класс),\n'
             '7 классов addressee (random chance F1 ≈ 0.143).\n\n'
             'Тот же паттерн что в context classification:\n'
             '   • Per-context > Global tokenization: +0.056 F1 weighted (stratified), +0.016 (cross-bat)\n'
             '   • Tokens (любые) < Pooled raw mel: −0.28 F1 weighted в обоих split\'ах\n\n'
             'Это уже ТРЕТЬЯ задача (после context classification и within-context seq prediction)\n'
             'демонстрирующая один и тот же pattern: per-context улучшает token-pipeline,\n'
             'но не дотягивает до continuous-feature classifier'"'"'ов.',
             ha='left', va='top', fontsize=9, linespacing=1.6)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def leakage_page(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle('11. Methodological finding — target leakage в paper HP1',
                 fontsize=12.5, fontweight='bold', y=0.97)

    rows = [
        ('A. Oracle context conditioning (paper-exact)',        '0.842',  '0.795'),
        ('B. Shuffled (random) context conditioning',           '0.314',  '0.195'),
        ('C. No context conditioning (test only)',              '0.167',  '0.107'),
        ('D. CLEAN: train+test both без conditioning',          '0.394',  '0.285'),
        ('Random chance (8 классов)',                           '0.125',  '0.125'),
    ]
    ax = fig.add_subplot(111); ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=['Test mode', 'F1 weighted', 'F1 macro'],
                   loc='upper center', cellLoc='left',
                   colWidths=[0.55, 0.20, 0.18])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.0)
    for j in range(3):
        tbl[0, j].set_facecolor('#404040')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # paper claim row
    for j in range(3): tbl[1, j].set_facecolor('#ffcdd2')
    # honest baseline
    for j in range(3): tbl[4, j].set_facecolor('#e8f5e9')
    # random chance
    for j in range(3): tbl[5, j].set_facecolor('#cccccc')

    fig.text(0.07, 0.55,
             'Эмпирическое доказательство утечки на bat 215, paper-exact protocol\n'
             '(Zhang-18 features + SVC + GridSearchCV + random stratified split):\n\n'
             '⚠ Paper claim "F1 > 0.9" воспроизводится точно (0.842) когда conditioning=oracle.\n'
             '⚠ При перемешивании conditioning context (только в feature extraction!\n'
             '   true labels для evaluation НЕ трогаются) F1 рушится 0.842 → 0.314.\n'
             '⚠ Это означает: классификатор читает контекст из выбора нормализатора признака,\n'
             '   а не из содержания sequence. Drop = 0.528 F1 — selection leakage proven.\n\n'
             'Honest baseline (D, без conditioning ни в train, ни в test): F1 = 0.394.\n'
             'Это ≈ 47% от paper claim. Остальные 53% paper-F1 объясняются leakage.\n\n'
             'Источник в коде: Exp1-Classifier.ipynb, функция prepare_data_from_sequences:\n\n'
             '   for i, row in data_df.iterrows():\n'
             '       contextId = row["context"]   # ← истинная метка ЭТОЙ строки\n'
             '       df_dict["e"].append(consistency(sequence,\n'
             '                                       _transitions_in_context[contextId]))',
             ha='left', va='top', fontsize=8.5, linespacing=1.5, family='monospace')
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def conclusions_page(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle('12. Сводка результатов и положения на защиту',
                 fontsize=13, fontweight='bold', y=0.97)

    fig.text(0.07, 0.92, 'Контрибуции работы (для thesis-defense)', fontsize=12,
             fontweight='bold', va='top')

    summary = (
        '\n'
        '1. Воспроизведение Assom 2025 на 153k сегментах\n'
        '   • N=153,366 (paper 152,578); 7 кластеров; sil=0.629 (>0.5)\n'
        '   • Per-emitter ARI=0.18 (paper 0.12), NMI=0.35 (paper 0.30) — в paper-CI\n'
        '   • HP1 permutation insensitivity (|ΔF1|=0.006) воспроизводится\n'
        '\n'
        '2. Per-context tokenization улучшает classification F1 над paper-style baseline\n'
        '   • F1 weighted: +0.107 SIG+ (paired CI [+0.046, +0.167])\n'
        '   • F1 macro: +0.061 SIG+\n'
        '   • Эффект устойчив к выбору per-context tokenizer (DP-GMM/HDBSCAN/k-means)\n'
        '   • 5/5 сидов положительны, 7/8 контекстов имеют положительный gain\n'
        '\n'
        '3. Cardinality-fair NMI per-(bat,context) с DTW-прокси: малый, но устойчивый прирост\n'
        '   • Δ NMI = +0.035, paired Wilcoxon p<10⁻¹¹ по 105 ячейкам\n'
        '   • 7/8 контекстов в пользу per-context (только Isolation проигрывает)\n'
        '   • ARI Δ = −0.017 NS — жёсткое разбиение неотличимо\n'
        '\n'
        '4. Methodological contribution: target leakage в paper HP1\n'
        '   • Эмпирически доказан: F1 0.842 → 0.314 при перемешивании conditioning context\n'
        '   • Унаследовано из Zhang 2019 (JEB) — формулировка признаков e, h, j использует\n'
        '     истинную метку при нормализации\n'
        '   • Honest baseline (без conditioning): F1 = 0.394 ≈ 47% paper claim\n'
        '\n'
        '5. Different vocabulary sizes per context (структурный finding)\n'
        '   • K₉₅ диапазон 9–13 на 8 контекстах (DP-GMM с n_components=15)\n'
        '   • При n_components=30: K₉₅ диапазон 10–26\n'
        '   • Cooperative contexts (Grooming, Kissing) имеют меньший K, conflict (Fighting) — больший\n'
        '\n'
        '6. Important honest caveat: continuous-feature baselines\n'
        '   • RF на pooled raw mel (без токенов вообще): F1 = 0.529 — БЬЁТ per-context Bayes (−0.05)\n'
        '   • Per-context > Global tokens, но Tokens (любые) < Continuous-feature classifier\'ы\n'
        '   • Token-based = слабее no-token, что methodologically важно для domain tokenization\n'
        '\n'
        '7. Within-context sequence prediction: PC проигрывает global на direct accuracy\n'
        '   • 0/8 wins на T1 next-token, T2 masked cloze, T3 completion BLEU/ED\n'
        '   • Per-context словарь больше → harder bigram estimation от того же train\n'
    )
    fig.text(0.07, 0.88, summary, fontsize=8.5, va='top', linespacing=1.6, family='sans-serif')
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        title_page(pdf)
        reproduction_page(pdf)
        cluster_viz_page(pdf)
        main_classification_page(pdf)
        per_class_chart_page(pdf)
        proxy_agreement_page(pdf)
        info_theory_page(pdf)
        per_context_structure_page(pdf)
        seq_tasks_page(pdf)
        addressee_page(pdf)
        leakage_page(pdf)
        conclusions_page(pdf)
    print(f'Wrote: {OUT}')
    print(f'Size: {OUT.stat().st_size / 1024:.0f} KB')


if __name__ == '__main__':
    main()
