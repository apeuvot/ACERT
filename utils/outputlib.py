import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def WriteConfusionSeaborn(m, labels, outpath):
    '''
    INPUT:
        m: confusion matrix (numpy array)
        labels: List of string, the category name of each entry in m
        outpath: Name for the output png plot
    '''
    fig, ax = plt.subplots()
    inn = m / m.sum(1, keepdims=True)
    ax = sns.heatmap(inn, cmap='Blues', fmt='.2%', xticklabels=labels, yticklabels=labels,
                     annot=True, annot_kws={"size": 12})
    for t in ax.texts:
        t.set_text(t.get_text()[:-1])

    fig.savefig(outpath)
    print(m)
    print(f"Saved figure to {outpath}.")
    plt.close(fig)


def compute_metrics_per_class(confusion, metrics, class_names, excluded_labels=[]):
    """
    Compute per-class and global metrics based on confusion matrix and global metrics array,
    handling division by zero properly without using 1e-8.
    """
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - TP
    FN = confusion.sum(axis=1) - TP
    support = confusion.sum(axis=1)

    results_dict = {}
    for i, label in enumerate(class_names):
        if label in excluded_labels:
            continue
        TP_i = TP[i]
        FP_i = FP[i]
        FN_i = FN[i]
        support_i = support[i]

        recall_i = TP_i / (TP_i + FN_i) if (TP_i + FN_i) != 0 else 0
        precision_i = TP_i / (TP_i + FP_i) if (TP_i + FP_i) != 0 else 0
        f1_i = 2 * precision_i * recall_i / (precision_i + recall_i) if (precision_i + recall_i) != 0 else 0

        results_dict[label] = {
            'UAR [%]': recall_i * 100,   # per-class recall
            'Precision [%]': precision_i * 100,
            'Recall [%]': recall_i * 100,
            'macroF1 [%]': f1_i * 100,
            'support': support_i
        }

    # Global metrics
    results_dict['TOTAL'] = {
        'UAR [%]': np.mean(TP / (TP + FN) * 100),
        'WAR [%]': TP.sum() / confusion.sum() * 100,
        'Precision [%]': np.mean(np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)) * 100,
        'Recall [%]': np.mean(np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)) * 100,
        'macroF1 [%]': np.mean(metrics[2]),
        'weightedF1 [%]': (np.sum(support * np.divide(2 * TP, 2 * TP + FP + FN, out=np.zeros_like(TP, dtype=float), where=(2 * TP + FP + FN) != 0)) / support.sum()) * 100,
        'support': support.sum()
    }

    return results_dict


def print_metrics(results_dict, file=None, extra_text=None):
    """
    Print metrics in a table format, either to console or to a file.
    Can append extra text (e.g., global summary) at the end.
    """
    if isinstance(file, str):
        f = open(file, 'w')
        close_file = True
    else:
        f = file or None
        close_file = False

    def pf(*args, **kwargs):
        print(*args, file=f, **kwargs)

    pf("{:<20} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
        "Label", "UAR [%]", "WAR [%]", "Precision", "Recall", "macroF1 [%]", "weightedF1 [%]"
    ))
    pf("-" * 120)

    for label, metrics_lbl in sorted(results_dict.items()):
        if label != 'TOTAL':
            pf("{:<20} {:<15.4f} {:<15} {:<15.4f} {:<15.4f} {:<15.4f} {:<15} {:<15}".format(
                label,
                metrics_lbl['UAR [%]'],
                "-",  # WAR is a global metric
                metrics_lbl['Precision [%]'],
                metrics_lbl['Recall [%]'],
                metrics_lbl['macroF1 [%]'],
                "-",  # weightedF1 is a global metric
                int(metrics_lbl['support'])
            ))
    pf("-" * 120)

    total = results_dict['TOTAL']
    pf("{:<20} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15}".format(
        "TOTAL",
        total['UAR [%]'],
        total['WAR [%]'],
        total['Precision [%]'],
        total['Recall [%]'],
        total['macroF1 [%]'],
        total['weightedF1 [%]'],
        int(total['support'])
    ))

    pf('\n\n')

    if extra_text:
        pf(extra_text)

    if close_file:
        f.close()
