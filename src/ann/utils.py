# src/ann/utils.py
import numpy as np


def recall_at_k(I_pred: np.ndarray, I_gt: np.ndarray, k: int) -> float:
    """
    Compute recall@k.

    I_pred: predicted indices, shape = (nq, k)
    I_gt: ground-truth indices, shape = (nq, >=k)
    """
    nq = I_pred.shape[0]
    total = 0.0

    for i in range(nq):
        pred_set = set(I_pred[i, :k])
        gt_set = set(I_gt[i, :k])
        total += len(pred_set.intersection(gt_set)) / k

    return total / nq
