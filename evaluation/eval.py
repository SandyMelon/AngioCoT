"""Evaluate classification and regression predictions."""

import json
import os
import math
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error
)

# Configuration
INPUT_FILE = "/path/to/your/inference_results.json"

REGRESSION_KEYS = ['lesion_count', 'syntax_score']



def normalize_val(val):
    if val is None:
        return 'parse_error'
    return str(val).strip().lower()


def to_number(val):
    try:
        if val is None:
            return 0.0
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def print_formatted_cm(cm, labels, title):
    print(f"\nConfusion matrix: {title}")

    label_width = max([len(str(l)) for l in labels] + [5]) + 4
    col_width = max([len(str(x)) for x in cm.flatten()] +
                    [len(str(l)) for l in labels] + [5]) + 4

    header = " " * label_width + "|"
    for label in labels:
        header += f"{str(label):>{col_width}}"

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for i, row_label in enumerate(labels):
        row_str = f"{str(row_label):>{label_width}} |"
        for val in cm[i]:
            row_str += f"{val:>{col_width}}"
        print(row_str)

    print("-" * len(header))
    print("Rows: true; columns: predicted.\n")


def evaluate_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading results: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    clf_collectors = defaultdict(lambda: {'y_true': [], 'y_pred': []})
    reg_collectors = defaultdict(lambda: {'y_true': [], 'y_pred': []})

    parse_errors = 0
    total_samples = 0

    for item in data:
        gt = item.get('gt_parsed')
        pred = item.get('pred_parsed')

        if gt is None:
            continue

        total_samples += 1

        if pred is None:
            parse_errors += 1
            pred = {}

        for key, gt_val in gt.items():
            pred_val = pred.get(key)

            if key in REGRESSION_KEYS:
                reg_collectors[key]['y_true'].append(to_number(gt_val))
                reg_collectors[key]['y_pred'].append(to_number(pred_val))
            else:
                clf_collectors[key]['y_true'].append(normalize_val(gt_val))
                clf_collectors[key]['y_pred'].append(normalize_val(pred_val))

    print("\n" + "="*105)
    print(f"{'FULL METRICS REPORT':^105}")
    print("="*105)
    print(f"Valid samples: {total_samples}")
    print(
        f"JSON parse failures: {parse_errors} (missing attributes: incorrect/0)")

    results_list = []

    # Classification metrics
    if clf_collectors:
        print("-" * 105)
        print(f"{'>>> CLASSIFICATION TASKS':<105}")
        print("-" * 105)
        header = f"{'Attribute / Key':<40} | {'Acc':<8} | {'Mac-P':<8} | {'Mac-R':<8} | {'Mac-F1':<8} | {'Count'}"
        print(header)
        print("-" * 105)

        matrix_data = []

        for key, arrays in clf_collectors.items():
            y_true = arrays['y_true']
            y_pred = arrays['y_pred']

            if not y_true:
                continue

            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )

            print(
                f"{key:<40} | {acc*100:6.2f}% | {precision*100:6.2f}% | {recall*100:6.2f}% | {f1*100:6.2f}% | {len(y_true)}")

            unique_labels = sorted(list(set(y_true + y_pred)))
            cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

            results_list.append({
                "attribute": key,
                "type": "classification",
                "accuracy": acc,
                "macro_f1": f1,
                "macro_precision": precision,
                "macro_recall": recall,
                "count": len(y_true)
            })

            matrix_data.append((key, cm, unique_labels))

    # Regression metrics
    if reg_collectors:
        print("\n" + "-" * 105)
        print(f"{'>>> REGRESSION TASKS (Count/Number)':<105}")
        print("-" * 105)
        # MSE, RMSE, MAE
        header_reg = f"{'Attribute / Key':<40} | {'MSE':<10} | {'RMSE':<10} | {'MAE':<10} | {'Count'}"
        print(header_reg)
        print("-" * 105)

        for key, arrays in reg_collectors.items():
            y_true = arrays['y_true']
            y_pred = arrays['y_pred']

            if not y_true:
                continue

            mse = mean_squared_error(y_true, y_pred)
            rmse = math.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)

            print(
                f"{key:<40} | {mse:<10.4f} | {rmse:<10.4f} | {mae:<10.4f} | {len(y_true)}")

            results_list.append({
                "attribute": key,
                "type": "regression",
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "count": len(y_true)
            })

    print("="*105)
    print("Mac-R: check minority class recall.")
    print("Lower RMSE/MAE is better.")
    print("="*105)

    # Confusion matrices
    if clf_collectors:
        print("\n" + "#"*50)
        print(f"{'DETAILED CONFUSION MATRICES (Classification Only)':^50}")
        print("#"*50)

        for key, cm, labels in matrix_data:
            print_formatted_cm(cm, labels, key)

    output_eval_file = os.path.splitext(file_path)[0] + '_full_metrics.json'
    with open(output_eval_file, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=4, ensure_ascii=False)
    print(f"Metrics saved: {output_eval_file}")


if __name__ == "__main__":
    evaluate_metrics(INPUT_FILE)
