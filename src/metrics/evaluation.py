import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def binary_metrics(labels, probs, threshold=0.5):
    """Compute binary detection metrics."""
    preds = [int(p > threshold) for p in probs]
    result = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    if len(set(labels)) > 1:
        result["auroc"] = roc_auc_score(labels, probs)
        result["auprc"] = average_precision_score(labels, probs)
    return result


def type_metrics(labels, preds, num_classes=9):
    """Compute type classification metrics."""
    result = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "per_class_f1": f1_score(labels, preds, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(num_classes))).tolist(),
    }
    return result


def grounding_metrics(all_labels, all_probs, threshold=0.5):
    """Compute token/patch-level grounding metrics (manipulated samples only)."""
    if len(all_labels) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    flat_labels = np.concatenate(all_labels)
    flat_probs = np.concatenate(all_probs)
    flat_preds = (flat_probs > threshold).astype(int)

    result = {
        "precision": precision_score(flat_labels, flat_preds, zero_division=0),
        "recall": recall_score(flat_labels, flat_preds, zero_division=0),
        "f1": f1_score(flat_labels, flat_preds, zero_division=0),
    }
    if len(set(flat_labels)) > 1:
        result["auprc"] = average_precision_score(flat_labels, flat_probs)
    return result


def per_category_metrics(labels, probs, fake_cls_list, threshold=0.5):
    """Report binary detection metrics broken down by fake_cls."""
    results = {}
    categories = set(fake_cls_list)
    for cat in categories:
        mask = [fc == cat for fc in fake_cls_list]
        if sum(mask) == 0:
            continue
        cat_probs = [p for p, m in zip(probs, mask) if m]
        cat_labels = [l for l, m in zip(labels, mask) if m]
        cat_preds = [int(p > threshold) for p in cat_probs]
        results[cat] = {
            "count": len(cat_labels),
            "accuracy": accuracy_score(cat_labels, cat_preds),
            "f1": f1_score(cat_labels, cat_preds, zero_division=0),
        }
        if len(set(cat_labels)) > 1:
            results[cat]["auroc"] = roc_auc_score(cat_labels, cat_probs)
    return results


def find_optimal_threshold(labels, probs):
    """Find binary threshold that maximizes F1 on validation set."""
    best_f1, best_thresh = 0, 0.5
    for thresh_int in range(20, 81):
        thresh = thresh_int / 100
        preds = [int(p > thresh) for p in probs]
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh, best_f1


def joint_grounding_accuracy(img_ious, txt_f1s, img_threshold=0.3, txt_threshold=0.3):
    """For mixed samples, fraction with both image IoU and text F1 above thresholds."""
    if not img_ious:
        return 0.0
    joint_correct = sum(
        1 for iou, f1 in zip(img_ious, txt_f1s)
        if iou > img_threshold and f1 > txt_threshold
    )
    return joint_correct / len(img_ious)
