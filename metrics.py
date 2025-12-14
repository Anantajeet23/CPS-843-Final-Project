import torch

def confusion_matrix(pred, true):
    """
    pred, true: binary tensors (B,H,W) containing {0,1}
    returns TP, FP, TN, FN
    """
    pred = pred.bool()
    true = true.bool()

    TP = (pred & true).sum().item()
    TN = (~pred & ~true).sum().item()
    FP = (pred & ~true).sum().item()
    FN = (~pred & true).sum().item()

    return TP, FP, TN, FN


def compute_metrics(pred, true, eps=1e-7):
    TP, FP, TN, FN = confusion_matrix(pred, true)

    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)          
    specificity = TN / (TN + FP + eps)
    dice = (2 * TP) / (2*TP + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "dice": dice,
        "iou": iou,
    }
