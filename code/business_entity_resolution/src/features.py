import difflib
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

# Check if rapidfuzz is available for C++ acceleration
try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler, Levenshtein
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute standard Jaccard set similarity."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection) / float(union) if union > 0 else 0.0


def char_ngrams(text: str, n: int = 3) -> set:
    """Extract character n-grams from string."""
    text_clean = "".join(text.split())
    if len(text_clean) < n:
        return {text_clean} if text_clean else set()
    return {text_clean[i:i+n] for i in range(len(text_clean) - n + 1)}


def compute_levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalized edit similarity [0, 1]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if RAPIDFUZZ_AVAILABLE:
        return float(Levenshtein.normalized_similarity(s1, s2))
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def compute_jarowinkler_similarity(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity [0, 1]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if RAPIDFUZZ_AVAILABLE:
        return float(JaroWinkler.similarity(s1, s2))
    # Fallback to difflib ratio if rapidfuzz not present
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def compute_token_sort_ratio(s1: str, s2: str) -> float:
    """Token sort ratio [0, 1]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if RAPIDFUZZ_AVAILABLE:
        return float(fuzz.token_sort_ratio(s1, s2)) / 100.0
    
    t1 = " ".join(sorted(s1.split()))
    t2 = " ".join(sorted(s2.split()))
    return difflib.SequenceMatcher(None, t1, t2).ratio()


def compute_token_set_ratio(s1: str, s2: str) -> float:
    """Token set ratio [0, 1]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if RAPIDFUZZ_AVAILABLE:
        return float(fuzz.token_set_ratio(s1, s2)) / 100.0
    
    set1 = set(s1.split())
    set2 = set(s2.split())
    common = " ".join(sorted(set1.intersection(set2)))
    diff1 = " ".join(sorted(set1 - set2))
    diff2 = " ".join(sorted(set2 - set1))
    
    p1 = f"{common} {diff1}".strip()
    p2 = f"{common} {diff2}".strip()
    return max(
        difflib.SequenceMatcher(None, common, p1).ratio(),
        difflib.SequenceMatcher(None, common, p2).ratio(),
        difflib.SequenceMatcher(None, p1, p2).ratio()
    )


class FeatureExtractor:
    """
    Khushi's Module: Feature Engineering & String Similarity Engine.
    
    Transforms candidate pairs (S1, S2/S3) into rich numerical feature vectors
    capturing name similarity, address similarity, numeric overlap, and context.
    """

    FEATURE_NAMES = [
        "name_exact_match",
        "name_levenshtein",
        "name_jarowinkler",
        "name_token_sort",
        "name_token_set",
        "name_word_jaccard",
        "name_char3_jaccard",
        "name_first_token_match",
        "name_len_diff",
        "name_len_ratio",
        "addr_exact_match",
        "addr_word_jaccard",
        "addr_char4_jaccard",
        "addr_token_sort",
        "addr_num_overlap_count",
        "addr_num_jaccard",
        "addr_has_shared_num",
        "combined_word_jaccard",
        "is_source2",
        "is_source3",
        "cand_rank",
    ]

    def extract_pair_features(
        self, s1_rec: dict, target_rec: dict, cand_rank: int
    ) -> List[float]:
        """Extracts dense feature vector for a single (S1, S2/S3) pair."""
        # 1. Name features
        name1 = s1_rec["clean_name"]
        name2 = target_rec["clean_name"]
        name_exact = 1.0 if name1 == name2 and len(name1) > 0 else 0.0
        name_lev = compute_levenshtein_ratio(name1, name2)
        name_jw = compute_jarowinkler_similarity(name1, name2)
        name_ts = compute_token_sort_ratio(name1, name2)
        name_tset = compute_token_set_ratio(name1, name2)
        name_jacc = jaccard_similarity(s1_rec["name_tokens"], target_rec["name_tokens"])
        name_c3_jacc = jaccard_similarity(
            char_ngrams(name1, 3), char_ngrams(name2, 3)
        )

        # First token comparison (often brand name)
        t1_first = name1.split()[0] if name1.split() else ""
        t2_first = name2.split()[0] if name2.split() else ""
        first_token_match = 1.0 if t1_first and t1_first == t2_first else 0.0

        len1, len2 = len(name1), len(name2)
        len_diff = abs(len1 - len2)
        len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

        # 2. Address features
        addr1 = s1_rec["clean_addr"]
        addr2 = target_rec["clean_addr"]
        addr_exact = 1.0 if addr1 == addr2 and len(addr1) > 0 else 0.0
        addr_jacc = jaccard_similarity(s1_rec["addr_tokens"], target_rec["addr_tokens"])
        addr_c4_jacc = jaccard_similarity(
            char_ngrams(addr1, 4), char_ngrams(addr2, 4)
        )
        addr_ts = compute_token_sort_ratio(addr1, addr2)

        # Numbers / postal codes overlap
        nums1 = s1_rec["addr_numbers"]
        nums2 = target_rec["addr_numbers"]
        num_overlap_cnt = float(len(nums1.intersection(nums2)))
        num_jacc = jaccard_similarity(nums1, nums2)
        has_shared_num = 1.0 if num_overlap_cnt > 0 else 0.0

        # 3. Combined features
        combined_jacc = jaccard_similarity(s1_rec["full_tokens"], target_rec["full_tokens"])

        # 4. Source indicators
        tid = target_rec["entity_id"]
        is_s2 = 1.0 if tid.startswith("S2-") else 0.0
        is_s3 = 1.0 if tid.startswith("S3-") else 0.0

        return [
            name_exact,
            name_lev,
            name_jw,
            name_ts,
            name_tset,
            name_jacc,
            name_c3_jacc,
            first_token_match,
            float(len_diff),
            float(len_ratio),
            addr_exact,
            addr_jacc,
            addr_c4_jacc,
            addr_ts,
            num_overlap_cnt,
            num_jacc,
            has_shared_num,
            combined_jacc,
            is_s2,
            is_s3,
            float(cand_rank),
        ]

    def build_feature_matrix(
        self,
        candidate_dict: Dict[str, List[str]],
        s1_records: Dict[str, dict],
        target_records: Dict[str, dict],
        ground_truth: Dict[str, Set[str]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, List[Tuple[str, str]]]:
        """
        Builds feature DataFrame for all candidate pairs.
        If ground_truth is provided, computes binary matching labels y.
        """
        rows = []
        labels = []
        pairs = []

        for s1_id, cand_list in candidate_dict.items():
            s1_rec = s1_records[s1_id]
            true_matches = ground_truth.get(s1_id, set()) if ground_truth is not None else None

            for rank, tid in enumerate(cand_list, start=1):
                if tid not in target_records:
                    continue
                target_rec = target_records[tid]
                feat_vec = self.extract_pair_features(s1_rec, target_rec, rank)
                rows.append(feat_vec)
                pairs.append((s1_id, tid))

                if true_matches is not None:
                    label = 1 if tid in true_matches else 0
                    labels.append(label)

        X = pd.DataFrame(rows, columns=self.FEATURE_NAMES)
        y = np.array(labels, dtype=int) if ground_truth is not None else None
        return X, y, pairs
