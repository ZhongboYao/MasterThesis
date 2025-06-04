import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
import matplotlib.cm as cm
from collections import Counter
from typing import List, Dict, Tuple
import json

def comparison_star_plot(
    feature_accuracies_list: List[Dict[str, float]],
    output_path: str,
    legend_labels: List[str] = None,
    class_support: Dict[str, float] = None,
    legend_pos: Tuple[float, float] = (-0.2, -0.1),
    figsize = (15, 10),
    title = 'Radar Comparison of ChatGPT-4o Weighted Recall in Dental Lesion Diagnosis'
):
    """
    Plot multiple feature weighted recalls on a single star diagram to compare performances,
    overlaying the class-support (imbalance) as a gray background.
    The “Diagnosis” axis is drawn in bold, and all feature keys are remapped to more
    descriptive labels for display.

    If `feature_accuracies_list` is empty, this will draw _only_ the gray background polygon
    (i.e. the majority‐class support) and return immediately.
    """
    new_keywords = [
        '1. Upper/\nLower/\nBoth Jaws',
        '2. Lesion \nLocation',
        '3. Relationship to the \nSurrounding Teeth',
        '4. Number of Lesions',
        '5. Lesion Size',
        '6. Central or \nPeripheral Lesion',
        '7. Lesion \nContour',
        '8. Loculation',
        '9. Relative \nRadiolucency',
        '10. Multiple \nTooth Involvement',
        '11. Bony Cortex \nExpansion',
        '12. Root \nResorption',
        '13. Tooth \nDisplacement\n or Impaction',
        '14. Diagnosis'
    ]

    if legend_labels is None:
        legend_labels = [f'Model {i+1}' for i in range(len(feature_accuracies_list))]

    orig_features = list(feature_accuracies_list[0].keys()) if feature_accuracies_list else list(class_support.keys())
    assert len(orig_features) == len(new_keywords), "Feature count must match new_keywords length"

    features = new_keywords
    num_vars = len(features)
    diag_idx = features.index('14. Diagnosis')
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.tick_params(axis='x', pad=15)
    # for lbl in ax.get_xticklabels():
    #     if lbl.get_text().startswith('14. Diagnosis'):
    #         lbl.set_fontweight('bold')
    #         lbl.set_color('C3')

    ax.set_rlabel_position(30)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], alpha=0.1)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # --- background support (majority‐class polygon) ---
    if class_support is not None:
        support_vals = [class_support[orig] for orig in orig_features]
        support_vals += support_vals[:1]
        ax.fill(
            angles,
            support_vals,
            color='gray',
            alpha=0.3,
            linewidth=0,
            label='Majority-Class'
        )

    # If no model curves, finalize here
    if not feature_accuracies_list:
        # annotate each support value
        for angle, support in zip(angles[:-1], support_vals[:-1]):
            # offset just inside the ring
            offset = 0.02 if support < 0.95 else -0.02
            # alternate small rotation so labels don't collide
            text_angle = angle + (3 * np.pi/180)
            ax.text(
                text_angle,
                support + offset,
                f"{support:.2f}",
                size=10,
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.1, edgecolor='none', pad=1)
            )

        plt.legend(loc='lower left', bbox_to_anchor=legend_pos, fontsize=12)
        plt.title(
            'Majority-Class Baseline Weighted Recall',
            pad=30,
            fontsize=14
        )
        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
        plt.savefig(output_path)
        plt.close(fig)
        return

    # --- models’ curves and annotations ---
    for i, feat_dict in enumerate(feature_accuracies_list):
        values = [feat_dict[orig] for orig in orig_features]
        values += values[:1]

        ax.plot(
            angles,
            values,
            linewidth=2 if i == 0 else 1.5,
            linestyle='solid',
            label=f"{legend_labels[i]}"
        )
        ax.fill(angles, values, alpha=0.25)

        for angle, value in zip(angles[:-1], values[:-1]):
            offset = 0.02 if value < 0.95 else -0.02
            text_angle = angle + (3 * np.pi/180 if i % 2 == 0 else -3 * np.pi/180)
            ax.text(
                text_angle,
                value + offset,
                f"{value:.2f}",
                size=10,
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.1, edgecolor='none', pad=1)
            )

    # bold the diagnosis spoke
    # ax.plot(
    #     [angles[diag_idx], angles[diag_idx]],
    #     [0, 1],
    #     linewidth=1,
    #     color='C3',
    #     zorder=1
    # )

    plt.legend(loc='lower left', bbox_to_anchor=legend_pos, fontsize=12)
    plt.title(
        title,
        pad=30,
        fontsize=14
    )
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    plt.savefig(output_path)
    plt.close(fig)

def feature_distribution_plot(case_path: str):
    """
    Illustrate the feature distribution across all the cases.
    """
    cases = pd.read_csv(case_path)
    excluded_features = ["What is the sex of your patient?", "What is the sex of your patient?", "What is the age of your patient?", "Is there pain or paresthesia? "]

    for idx, row in cases.iterrows():
        feature_name = row.iloc[0]

        if feature_name in excluded_features:
            continue
        
        data_to_plot = row.iloc[1:]
        
        value_counts = data_to_plot.value_counts()
        
        x_labels = value_counts.index.astype(str)
        y_values = value_counts.values
        
        plt.figure(figsize=(13, 8))
        plt.bar(x_labels, y_values)

        plt.title(f"{feature_name}")
        plt.xlabel("Unique Values")
        plt.ylabel("Count")
        
        plt.tight_layout()
        plt.show()

def create_star_variance(
    dict_list: list,
    file_path: str,
    class_support: dict = None,
    figsize = (12, 10)
) -> None:
    """
    Plot a spider (radar) diagram comparing feature accuracies across multiple experiments.
    Each experiment is a dict mapping feature names to accuracy values. This function:
      1. Aggregates per-feature means and variances
      2. Plots the mean polygon + fill
      3. Adds error bars (stddev) at each spoke, colored by variance
      4. (Optional) Overlays class-support as a gray background
      5. Remaps feature keys to numbered, multi-line labels for display
    """

    # --- 1) Determine feature order and labels mapping -----------------------

    # assume every dict in dict_list has the same keys, in the same order:
    orig_features = list(dict_list[0].keys())
    num_vars = len(orig_features)

    # our human-readable, numbered labels:
    new_labels = [
        '1. Upper/\nLower/\nBoth Jaws',
        '2. Lesion \nLocation',
        '3. Relationship to the \nSurrounding Teeth',
        '4. Number of Lesions',
        '5. Lesion Size',
        '6. Central or \nPeripheral Lesion',
        '7. Lesion \nContour',
        '8. Loculation',
        '9. Relative \nRadiolucency',
        '10. Multiple \nTooth Involvement',
        '11. Bony Cortex \nExpansion',
        '12. Root \nResorption',
        '13. Tooth \nDisplacement\n or Impaction',
        '14. Diagnosis'
    ]
    assert len(new_labels) == num_vars, "Feature count must match label list length"
    # map orig→display
    label_map = dict(zip(orig_features, new_labels))

    # --- 2) Aggregate means & variances per feature ---------------------------

    feature_data = {f: [] for f in orig_features}
    for exp in dict_list:
        for f in orig_features:
            feature_data[f].append(exp[f])

    means = [np.mean(feature_data[f]) for f in orig_features]
    variances = [np.var(feature_data[f]) for f in orig_features]
    std_devs = [np.sqrt(v) for v in variances]

    # close the loop for radar
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    means_loop = means + means[:1]

    # --- 3) Plot setup --------------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    # ticks and labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label_map[f] for f in orig_features])
    ax.tick_params(axis='x', pad=15)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], alpha=0.1)
    ax.set_ylim(0, 1)

    # --- 4) Optional class-support background -------------------------------

    if class_support is not None:
        support_vals = [class_support.get(f, 0.0) for f in orig_features]
        support_loop = support_vals + support_vals[:1]
        ax.fill(
            angles,
            support_loop,
            color='gray',
            alpha=0.3,
            linewidth=0,
            label='Marjority-Class'
        )

    # --- 5) Plot mean polygon and fill ---------------------------------------

    ax.plot(angles, means_loop, linewidth=2, linestyle='solid', color='orange', label='Mean Accuracy')
    ax.fill(angles, means_loop, alpha=0.25, color='orange')

    # --- 6) Error bars + variance color --------------------------------------

    # normalization and colormap for variances
    norm = plt.Normalize(min(variances), max(variances))
    cmap = cm.get_cmap('viridis')

    for i, (angle, m, sd, var) in enumerate(zip(angles[:-1], means, std_devs, variances)):
        ax.errorbar(
            angle,
            m,
            yerr=sd,
            fmt='o',
            markersize=8,
            color=cmap(norm(var)),
            ecolor='red',
            capsize=4,
            linestyle='None'
        )
        # annotate mean value
        offset = 0.08 if m < 0.9 else -0.08
        text_angle = angle + 3 * (np.pi / 180)
        ax.text(
            text_angle,
            m + offset,
            f"{m:.2f}",
            ha='center',
            va='center',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
        )

    # --- 7) Colorbar for variance --------------------------------------------

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label('Variance', fontsize=12)

    # --- 8) Final touches ----------------------------------------------------

    ax.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.1))
    ax.grid(True, alpha=0.3)
    plt.title("Feature-wise Weighted Recall and Variance for Zero-Shot ChatGPT-4o in Dental Lesion Diagnosis", fontsize=14, pad=40)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    plt.savefig(file_path)
    plt.close(fig)

def question_lables_distribution(doc_anns_path: str, exp_names: str, question_ind: int, fig_path: str, title=None):
    with open(doc_anns_path, 'r') as f:
        doc_anns = json.load(f)   # List[Dict[str,str]]
    
    doc_question = list(doc_anns[0].items())[4+question_ind][0]
    doc_answers = [case.get(doc_question, "Unanswered") for case in doc_anns]
    freq_doc = Counter(doc_answers)

    p = f"evaluation_results/{exp_names[0]}/gpt_anns.json"
    with open(p, 'r') as f:
        ann = json.load(f)  # List[Dict[str,str]]
    case = json.loads(ann[0])
    gpt_question = list(case.items())[question_ind][0]

    # 2) load each experiment
    freq_exps = []
    for exp in exp_names:
        ans = []
        p = f"evaluation_results/{exp}/gpt_anns.json"
        with open(p, 'r') as f:
            ann = json.load(f)  # List[Dict[str,str]]
        for case in ann:
            case_dict = json.loads(case)
            ans.append(case_dict.get(gpt_question, "Unanswered"))
        freq_exps.append(Counter(ans))

    # 3) collect all labels across doc + exps
    all_labels = set(freq_doc)
    for cnt in freq_exps:
        all_labels |= set(cnt)
    all_labels = sorted(all_labels, key=lambda x: -freq_doc.get(x, 0))  # sort by doc freq

    # 4) prepare data matrix: rows=label, cols=(doc + each exp)
    counts = []
    counts.append([freq_doc.get(lbl, 0) for lbl in all_labels])
    for cnt in freq_exps:
        counts.append([cnt.get(lbl, 0) for lbl in all_labels])

    counts = np.array(counts)   # shape (1 + len(exp_names), n_labels)

    # 5) plot grouped bar
    n_groups = len(all_labels)
    n_series = counts.shape[0]
    ind = np.arange(n_groups)
    width = 0.8 / n_series

    plt.figure(figsize=(max(10, n_groups), 6))
    # ground truth
    plt.bar(ind - width*(n_series-1)/2, counts[0], width,
            color='gray', alpha=0.4, label='Ground Truth')

    # experiments
    cmap = plt.get_cmap('tab10')
    for i, exp in enumerate(exp_names, start=1):
        plt.bar(ind - width*(n_series-1)/2 + i*width, counts[i], width,
                color=cmap(i-1), alpha=0.7, label=exp)

    # annotate
    for series in range(n_series):
        for j, lbl in enumerate(all_labels):
            cnt = counts[series, j]
            if cnt > 0:
                x = ind[j] - width*(n_series-1)/2 + series*width
                plt.text(x, cnt + 0.1, str(cnt),
                         ha='center', va='bottom', fontsize=9)

    # labels & legend
    plt.xticks(ind, all_labels, rotation=30, ha='right')
    plt.ylabel("Count")
    plt.title(title or f"Answer distribution for:\n“{doc_question[:-1]}”")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()