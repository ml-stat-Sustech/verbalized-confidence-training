import numpy as np
from sklearn.isotonic import IsotonicRegression


def compute_confidence_distribution_metrics(confidence, prefix="confidence"):
    confidence = np.asarray(confidence, dtype=float)
    confidence = confidence[np.isfinite(confidence)]
    if confidence.size == 0:
        return {}

    _, counts = np.unique(confidence, return_counts=True)
    probabilities = counts.astype(float) / confidence.size
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    top_k = min(3, len(counts))
    return {
        f"{prefix}_mean": float(confidence.mean()),
        f"{prefix}_std": float(confidence.std()),
        f"{prefix}_range": float(confidence.max() - confidence.min()),
        f"{prefix}_unique_count": float(len(counts)),
        f"{prefix}_top1_share": float(counts.max() / confidence.size),
        f"{prefix}_top3_share": float(np.sort(counts)[-top_k:].sum() / confidence.size),
        f"{prefix}_extreme_share": float(np.mean((confidence <= 0.01) | (confidence >= 0.98))),
        f"{prefix}_one_or_zero_ratio": float(
            np.mean(np.isclose(confidence, 0.0) | np.isclose(confidence, 1.0))
        ),
        f"{prefix}_entropy": entropy,
        f"{prefix}_effective_values": float(np.exp(entropy)),
    }


def compute_confidence_subgroup_metrics(correctness, confidence):
    """Confidence diversity statistics split by correctness.

    Correctness is thresholded at 0.5. Empty subgroups are omitted.
    """
    correctness = np.asarray(correctness, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    if correctness.shape != confidence.shape:
        raise ValueError("correctness and confidence must have the same shape")

    valid = np.isfinite(correctness) & np.isfinite(confidence)
    correctness, confidence = correctness[valid], confidence[valid]
    metrics = {}
    correct = correctness >= 0.5
    for name, mask in (("correct", correct), ("incorrect", ~correct)):
        values = confidence[mask]
        if values.size:
            metrics.update(compute_confidence_distribution_metrics(values, prefix=f"confidence_{name}"))

    if correct.any() and (~correct).any():
        metrics["confidence_correct_incorrect_mean_gap"] = float(
            confidence[correct].mean() - confidence[~correct].mean()
        )
    return metrics


def compute_excess_aurc(correctness, confidence):
    correctness = np.asarray(correctness, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    valid = np.isfinite(correctness) & np.isfinite(confidence)
    correctness, confidence = correctness[valid], confidence[valid]
    correct = correctness >= 0.5
    if not correct.any() or correct.all():
        return None

    errors = (~correct).astype(float)
    order = np.argsort(-confidence, kind="mergesort")
    sorted_confidence = confidence[order]
    sorted_errors = errors[order]

    risk_sum = 0.0
    errors_before = 0.0
    start = 0
    while start < len(sorted_errors):
        end = start + 1
        while end < len(sorted_errors) and sorted_confidence[end] == sorted_confidence[start]:
            end += 1
        group_errors = sorted_errors[start:end].sum()
        group_size = end - start
        for offset in range(1, group_size + 1):
            expected_errors = errors_before + group_errors * offset / group_size
            risk_sum += expected_errors / (start + offset)
        errors_before += group_errors
        start = end

    aurc = risk_sum / len(errors)
    oracle_errors = np.sort(errors)
    oracle_aurc = np.mean(np.cumsum(oracle_errors) / np.arange(1, len(errors) + 1))
    return float(max(0.0, aurc - oracle_aurc))


def compute_calibration_metrics(correctness, confidence, n_bins=10):
    correctness = np.asarray(correctness, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    if correctness.shape != confidence.shape:
        raise ValueError("correctness and confidence must have the same shape")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    valid = np.isfinite(correctness) & np.isfinite(confidence)
    correctness, confidence = correctness[valid], confidence[valid]
    if correctness.size == 0:
        return {}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    bin_indices = np.digitize(confidence, bin_edges) - 1
    ece = 0.0
    for index in range(n_bins):
        in_bin = bin_indices == index
        if in_bin.any():
            ece += in_bin.mean() * abs(confidence[in_bin].mean() - correctness[in_bin].mean())

    mean_accuracy = correctness.mean()
    mean_confidence = confidence.mean()
    brier_score = float(np.mean(np.square(confidence - correctness)))
    uncertainty = float(np.mean(np.square(correctness - mean_accuracy)))
    calibrated_confidence = IsotonicRegression(y_min=0.0, y_max=1.0).fit_transform(confidence, correctness)
    calibrated_brier = float(np.mean(np.square(calibrated_confidence - correctness)))

    positive = correctness >= 0.5
    macro_ce = None
    if positive.any() and (~positive).any():
        macro_ce = float(0.5 * ((1.0 - confidence[positive]).mean() + confidence[~positive].mean()))

    metrics = {
        "accuracy": float(mean_accuracy),
        "mean_confidence": float(mean_confidence),
        "abs_confidence_accuracy_gap": float(abs(mean_confidence - mean_accuracy)),
        "brier_score": brier_score,
        "brier_skill_score": None if uncertainty == 0.0 else float(1.0 - brier_score / uncertainty),
        "ece": float(ece),
        "macro_ce": macro_ce,
        "corp_mcb": float(max(0.0, brier_score - calibrated_brier)),
        "corp_dsc": float(max(0.0, uncertainty - calibrated_brier)),
        "eaurc": compute_excess_aurc(correctness, confidence),
    }
    metrics.update(compute_confidence_distribution_metrics(confidence))
    metrics.update(compute_confidence_subgroup_metrics(correctness, confidence))
    return metrics


def compute_binary_auroc(correctness, confidence):
    correctness = np.asarray(correctness, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    valid = np.isfinite(correctness) & np.isfinite(confidence)
    correctness, confidence = correctness[valid], confidence[valid]
    positive = correctness >= 0.5
    num_positive = int(positive.sum())
    num_negative = int((~positive).sum())
    if num_positive == 0 or num_negative == 0:
        return None

    order = np.argsort(confidence, kind="mergesort")
    sorted_scores = confidence[order]
    ranks = np.arange(1, len(confidence) + 1, dtype=float)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end

    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = original_ranks[positive].sum()
    auc = (positive_rank_sum - num_positive * (num_positive + 1) / 2) / (num_positive * num_negative)
    return float(auc)
