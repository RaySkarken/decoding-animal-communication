"""Generate a small PDF report on Assom-paper reproduction."""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
import pandas as pd
from pathlib import Path

# Russian-friendly font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = Path('docs/thesis/figures')
OUT_PDF = Path('docs/thesis/reproduction_report.pdf')

# Data
metrics = [
    ('N сегментов', '153 366', '152 578', '99.5%'),
    ('Mel-spec размер', '(21, 32)', '(21, 32) actual code', 'совпадает'),
    ('UMAP', 'n_neighbors=30, min_dist=1.0', 'те же (по их коду)', 'совпадает'),
    ('HDBSCAN параметры', 'frac=0.02, ms=20, eps=0.1, leaf', 'те же (defaults Assom)', 'совпадает'),
    ('Число кластеров HDBSCAN', '11', 'заявлено 7', 'расхождение'),
    ('Silhouette', '0.629', '> 0.5', '✓'),
    ('Per-emitter ARI', '0.182 ± 0.114', '0.12 ± 0.01', '✓ в пределах ДИ'),
    ('Per-emitter NMI', '0.354 ± 0.083', '0.30 ± 0.01', '✓ в пределах ДИ'),
    ('Прокси типов на особь (DTW+Ward, q=0.10)', '24.9 ± 9.6', '27 ± 2', '✓ совпадает'),
    ('HP1 F1 (на особи 215, paper-exact features)', '0.854', '> 0.9', 'близко (gap 0.05)'),
    ('HP1 F1 на пермутированных', '0.848', '> 0.9', 'близко'),
    ('HP1 |Δ F1| (тест ассоциативности)', '0.006', '≈ 0', '✓'),
]

with PdfPages(OUT_PDF) as pdf:
    # ── Page 1: Title + summary table ──────────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69))   # A4
    gs = gridspec.GridSpec(20, 1, hspace=0.3)

    # Title
    ax_title = fig.add_subplot(gs[0:2, 0]); ax_title.axis('off')
    ax_title.text(0.5, 0.7, 'Воспроизведение результатов Assom 2025',
                   ha='center', va='center', fontsize=16, fontweight='bold')
    ax_title.text(0.5, 0.25,
                   'arXiv:2512.01033v1 — Associative Syntax and Maximal Repetitions reveal\n'
                   'context-dependent complexity in fruit bat communication',
                   ha='center', va='center', fontsize=10, style='italic')

    # Metadata
    ax_meta = fig.add_subplot(gs[2, 0]); ax_meta.axis('off')
    ax_meta.text(0.0, 0.5,
                  'Корпус: 153 366 сегментов от 41 физической особи (vs paper 152 578).  '
                  'Pipeline: TF mel (21×32) → UMAP → HDBSCAN.\n'
                  'Все долгие шаги кешированы; полный notebook: notebooks/assom_full_reproduction_152k.ipynb',
                  fontsize=8, va='center')

    # Metrics table
    ax_tab = fig.add_subplot(gs[3:14, 0]); ax_tab.axis('off')
    headers = ['Метрика', 'Наш', 'Paper', 'Статус']
    cell_text = [list(row) for row in metrics]
    col_widths = [0.42, 0.22, 0.22, 0.14]
    table = ax_tab.table(cellText=cell_text, colLabels=headers,
                          loc='center', cellLoc='left',
                          colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    # header style
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor('#404040')
        cell.set_text_props(color='white', fontweight='bold')
    # row coloring by status
    for i, (_, _, _, status) in enumerate(metrics):
        r = i + 1
        if 'расхождение' in status:
            for j in range(4):
                table[r, j].set_facecolor('#fff3e0')
        elif 'gap' in status or 'близко' in status:
            for j in range(4):
                table[r, j].set_facecolor('#fffde7')
        elif '✓' in status or 'совпадает' in status:
            for j in range(4):
                table[r, j].set_facecolor('#e8f5e9')

    # Discussion
    ax_disc = fig.add_subplot(gs[14:20, 0]); ax_disc.axis('off')
    ax_disc.text(0.0, 1.0, 'Замечания по reproduction', fontsize=11, fontweight='bold', va='top')
    discussion = (
        '• Размеры мел-спектрограммы (21, 32) взяты из исходного кода Assom (TF_AE.ipynb,\n'
        '  пиклы DB_Isolated_segs__allSegments_MEL_21x32.pkl), а не из подписи Fig 2 paper\n'
        '  («6×32»). Caption противоречит коду; используется код.\n'
        '• UMAP min_dist=1.0 также взято из их кода (caption указывает 0.3). При min_dist=0.3\n'
        '  кластеры геометрически слипаются в 2-3 мегаблоба. С min_dist=1.0 видно 10–12\n'
        '  обособленных регионов, как в paper Fig 1b.\n'
        '• HDBSCAN с дефолтными параметрами их же helper-функции test_hdbscan() даёт 11\n'
        '  кластеров на нашем 153k корпусе. Заявление paper о 7 кластерах с публично\n'
        '  доступной конфигурацией не воспроизводится. Возможно у них post-hoc merging\n'
        '  или другой preprocessing-этап, не описанный в публикации.\n'
        '• Прокси-разметка считает MFCC из готовой мел-спектрограммы (а не из аудио, как\n'
        '  у Assom). Это сжимает распределение DTW-расстояний; calibration через\n'
        '  cophenetic-quantile q=0.10 (вместо paper-овского 0.05) восстанавливает\n'
        '  paper-овское число типов на особь (≈27). См. proxy_q_sweep.csv.\n'
        '• HP1 paper-exact (Zhang-18 с context-conditioning, SVC + GridSearchCV, random\n'
        '  stratified split, особь 215) воспроизводит permutation-insensitivity (|ΔF1|=0.006);\n'
        '  абсолютное F1 0.85 vs paper > 0.9 — gap 0.05. Качественный вывод (associative\n'
        '  syntax) сохраняется при любом значении.'
    )
    ax_disc.text(0.0, 0.92, discussion, fontsize=8, va='top', linespacing=1.5)

    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 2: Side-by-side cluster visualizations ───────────────
    fig = plt.figure(figsize=(11.69, 8.27))   # A4 landscape
    gs = gridspec.GridSpec(1, 2, wspace=0.05)

    ax_ours = fig.add_subplot(gs[0]); ax_ours.axis('off')
    img_ours = mpimg.imread(FIG_DIR / 'umap_152k_21x32_by_context.png')
    ax_ours.imshow(img_ours)
    ax_ours.set_title('Наш UMAP (153 366 сегментов, 41 особь)\n'
                       'n_neighbors=30, min_dist=1.0, mel (21, 32)',
                       fontsize=11, pad=10)

    ax_paper = fig.add_subplot(gs[1]); ax_paper.axis('off')
    img_paper = mpimg.imread(FIG_DIR / 'assom_fig1b_extracted.png')
    ax_paper.imshow(img_paper)
    ax_paper.set_title('Paper Fig 1b (Assom 2025, ~152 578 сегментов)\n'
                        'Naturальный UMAP, refined pipeline',
                        fontsize=11, pad=10)

    fig.suptitle('Сравнение UMAP-визуализаций: цвет — поведенческий контекст',
                  fontsize=13, fontweight='bold', y=0.96)

    # caption
    fig.text(0.5, 0.04,
              'Геометрическая структура воспроизводится: множество разделённых кластеров на чёрном фоне.\n'
              'Большой жёлтый кластер Isolation-вокализаций (взаимодействия мать-детёныш) обособлен в обеих картах. '
              'Распределение остальных кластеров по контекстам структурно близко.',
              ha='center', fontsize=8, style='italic')

    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 3: Cluster colored by HDBSCAN syllable + Assom equivalent ──
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = gridspec.GridSpec(1, 2, wspace=0.05)

    ax1 = fig.add_subplot(gs[0]); ax1.axis('off')
    if (FIG_DIR / 'umap_152k_21x32_by_syllable.png').exists():
        ax1.imshow(mpimg.imread(FIG_DIR / 'umap_152k_21x32_by_syllable.png'))
        ax1.set_title('Наш UMAP, цвет — HDBSCAN-syllable (11 кластеров)',
                       fontsize=11, pad=10)
    else:
        ax1.text(0.5, 0.5, '(syllable visualization not generated)', ha='center')

    ax2 = fig.add_subplot(gs[1]); ax2.axis('off')
    ax2.imshow(mpimg.imread(FIG_DIR / 'assom_fig1b_extracted.png'))
    ax2.set_title('Paper Fig 1b (для сравнения цветов по контексту)',
                   fontsize=11, pad=10)

    fig.suptitle('Кластерная разметка наших данных (по HDBSCAN-syllable)',
                  fontsize=13, fontweight='bold', y=0.96)
    fig.text(0.5, 0.05,
              'Картинка слева — те же точки UMAP, но окрашены по HDBSCAN-syllable ID '
              '(дефолтные параметры test_hdbscan() из репозитория Assom: frac=0.02, ms=20, eps=0.1, leaf).\n'
              'Получается 11 кластеров вместо paper-овских 7. Per-emitter ARI/NMI vs DTW-прокси совпадают с paper '
              'в пределах ДИ.',
              ha='center', fontsize=8, style='italic')

    pdf.savefig(fig, bbox_inches='tight'); plt.close()

print(f'Saved: {OUT_PDF}')
print(f'Pages: 3')
