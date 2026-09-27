from typing import Dict, List, Set, Tuple
import numpy as np


def compute_entity_f05(predicted_ids: Set[str], true_ids: Set[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity:
    - If true_ids is empty (singleton):
        - scores 1.0 if predicted_ids is empty
        - scores 0.0 if predicted_ids is non-empty (false merge penalty)
    - If true_ids is non-empty:
        - if predicted_ids is empty: scores 0.0 (missed match)
        - else computes Precision, Recall, and F_0.5
    """
    if len(true_ids) == 0:
        return 1.0 if len(predicted_ids) == 0 else 0.0

    if len(predicted_ids) == 0:
        return 0.0

    tp = len(predicted_ids.intersection(true_ids))
    fp = len(predicted_ids - true_ids)
    fn = len(true_ids - predicted_ids)

    if tp == 0:
        return 0.0

    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)

    denom = (0.25 * precision) + recall
    if denom == 0.0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return f05


def evaluate_macro_f05(
    predictions: Dict[str, List[str]], ground_truth: Dict[str, Set[str]]
) -> Tuple[float, float, float, float]:
    """
    Evaluates macro-averaged F_0.5 across all Source 1 entities in ground truth.
    Returns:
    - macro_f05: Average F_0.5 across all S1 entities
    - avg_precision: Average precision across non-singleton entities
    - avg_recall: Average recall across non-singleton entities
    - singleton_accuracy: Accuracy on true singleton entities
    """
    f05_scores = []
    precisions = []
    recalls = []
    singleton_results = []

    for s1_id, true_set in ground_truth.items():
        pred_set = set(predictions.get(s1_id, []))
        score = compute_entity_f05(pred_set, true_set)
        f05_scores.append(score)

        if len(true_set) == 0:
            singleton_results.append(1.0 if len(pred_set) == 0 else 0.0)
        else:
            tp = len(pred_set.intersection(true_set))
            prec = float(tp) / len(pred_set) if len(pred_set) > 0 else 0.0
            rec = float(tp) / len(true_set)
            precisions.append(prec)
            recalls.append(rec)

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    avg_prec = float(np.mean(precisions)) if precisions else 0.0
    avg_rec = float(np.mean(recalls)) if recalls else 0.0
    singleton_acc = float(np.mean(singleton_results)) if singleton_results else 1.0

    return macro_f05, avg_prec, avg_rec, singleton_acc


def optimize_thresholds(
    scored_pairs: List[Tuple[str, str, float]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: Set[str],
) -> Tuple[float, float, float]:
    """
    Akshat's Module: Grid Search to find optimal (singleton_thresh, match_thresh)
    maximizing the competition Macro-Averaged F_0.5 metric.
    """
    # Group pairs by s1_id sorted descending by predicted probability
    s1_candidates = {s1: [] for s1 in all_s1_ids}
    for s1_id, tid, prob in scored_pairs:
        if s1_id in s1_candidates:
            s1_candidates[s1_id].append((tid, prob))

    for s1 in s1_candidates:
        s1_candidates[s1].sort(key=lambda x: x[1], reverse=True)

    best_f05 = -1.0
    best_singleton_thresh = 0.60
    best_match_thresh = 0.50

    singleton_grid = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    match_grid = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    for s_thresh in singleton_grid:
        for m_thresh in match_grid:
            if m_thresh > s_thresh:
                continue

            preds = {}
            for s1_id, cands in s1_candidates.items():
                if not cands or cands[0][1] < s_thresh:
                    # High probability threshold not met -> Predict singleton (empty)
                    preds[s1_id] = []
                else:
                    # Accept top candidate
                    chosen = [cands[0][0]]
                    # Accept subsequent candidates only if exceeding match_thresh
                    for tid, p in cands[1:]:
                        if p >= m_thresh and (cands[0][1] - p) < 0.25:
                            chosen.append(tid)
                    preds[s1_id] = chosen

            macro_f05, _, _, _ = evaluate_macro_f05(preds, ground_truth)
            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_singleton_thresh = s_thresh
                best_match_thresh = m_thresh

    return best_singleton_thresh, best_match_thresh, best_f05
