import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd

from .config import Config
from .preprocess import clean_business_name, clean_address


class CandidateGenerator:
    """
    Riya's Module: Multi-Method Scalable Blocking & Candidate Generation.
    
    Generates high-recall, ultra-compact candidate sets per Source 1 entity:
    1. Country Partitioning (open-set: works for US, India, France, etc.)
    2. Informative Name Token Inverted Index (IDF weighted)
    3. Character 3-Gram / Prefix Index for typo tolerance
    4. Numeric Address Token Inverted Index
    5. Union, Fast Jaccard Pre-Scoring, and Top-K Pruning
    """

    def __init__(self, config: Config = Config()):
        self.config = config

    def _prepare_records(self, df: pd.DataFrame) -> Dict[str, dict]:
        """Preprocesses all records and extracts tokens & numbers."""
        records = {}
        
        eids = df["entity_id"].astype(str).tolist()
        names = df.get("business_name", pd.Series([""] * len(df))).fillna("").astype(str).tolist()
        addrs = df.get("business_address", pd.Series([""] * len(df))).fillna("").astype(str).tolist()
        countries = df.get("country", pd.Series([""] * len(df))).fillna("").astype(str).str.strip().str.upper().tolist()

        for eid, name_raw, addr_raw, country in zip(eids, names, addrs, countries):
            eid = eid.strip()
            clean_name, name_tokens = clean_business_name(name_raw)
            clean_addr, addr_tokens, addr_numbers = clean_address(addr_raw)

            # Character 3-grams for fuzzy matching
            name_char_3grams = set()
            for tok in name_tokens:
                if len(tok) >= 3:
                    for i in range(len(tok) - 2):
                        name_char_3grams.add(tok[i:i+3])

            records[eid] = {
                "entity_id": eid,
                "country": country,
                "clean_name": clean_name,
                "name_tokens": set(name_tokens),
                "name_char_3grams": name_char_3grams,
                "clean_addr": clean_addr,
                "addr_tokens": set(addr_tokens),
                "addr_numbers": addr_numbers,
                "full_tokens": set(name_tokens).union(set(addr_tokens)),
            }
        return records

    def build_candidate_pairs(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame,
    ) -> Tuple[Dict[str, List[str]], Dict[str, dict], Dict[str, dict]]:
        """
        Executes multi-method blocking and returns:
        - candidate_dict: {s1_id: [candidate_s2_or_s3_ids, ...]}
        - s1_records: preprocessed S1 records dict
        - target_records: preprocessed S2 & S3 records dict
        """
        s1_records = self._prepare_records(df_s1)
        s2_records = self._prepare_records(df_s2)
        s3_records = self._prepare_records(df_s3)

        # Merge S2 and S3 target records efficiently
        target_records = s2_records
        target_records.update(s3_records)
        del s3_records

        # 1. Country Partitioning
        targets_by_country: Dict[str, List[str]] = defaultdict(list)
        for tid, trec in target_records.items():
            targets_by_country[trec["country"]].append(tid)

        # 2. Build multi-inverted indexes per country for S2 + S3
        # Token Inverted Index & Char 3-gram Inverted Index
        country_token_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        country_ngram_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        country_num_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        for country, tids in targets_by_country.items():
            token_doc_counts = defaultdict(int)
            total_country_targets = max(1, len(tids))

            for tid in tids:
                trec = target_records[tid]
                for tok in trec["name_tokens"]:
                    if len(tok) >= self.config.MIN_TOKEN_LEN_FOR_INDEX:
                        token_doc_counts[tok] += 1
                        country_token_index[country][tok].append(tid)

                for ng in trec["name_char_3grams"]:
                    country_ngram_index[country][ng].append(tid)

                for num in trec["addr_numbers"]:
                    country_num_index[country][num].append(tid)

            # Filter high frequency stopwords / overly generic words
            max_allowed_freq = max(5, int(total_country_targets * self.config.MAX_TOKEN_INDEX_FREQ))
            for tok, cnt in list(token_doc_counts.items()):
                if cnt > max_allowed_freq:
                    # Prune over-generalized token from index to avoid explosion
                    del country_token_index[country][tok]

        # 3. Candidate Retrieval and Scoring for each S1 entity
        candidate_dict: Dict[str, List[str]] = {}

        for s1_id, s1_rec in s1_records.items():
            country = s1_rec["country"]
            cand_scores: Dict[str, float] = defaultdict(float)

            token_idx = country_token_index.get(country, {})
            ngram_idx = country_ngram_index.get(country, {})
            num_idx = country_num_index.get(country, {})

            # A. Informative Name Token Retrieval
            s1_name_tokens = s1_rec["name_tokens"]
            for tok in s1_name_tokens:
                if tok in token_idx:
                    matched_targets = token_idx[tok]
                    # Inverse document weight
                    weight = 1.0 / (1.0 + math.log(1 + len(matched_targets)))
                    for tid in matched_targets:
                        cand_scores[tid] += 2.0 * weight

            # B. Character 3-Gram Retrieval (Typo / Spelling noise)
            s1_ngrams = s1_rec["name_char_3grams"]
            for ng in s1_ngrams:
                if ng in ngram_idx:
                    matched_targets = ngram_idx[ng]
                    if len(matched_targets) < 50:  # Only for sufficiently discriminative n-grams
                        for tid in matched_targets:
                            cand_scores[tid] += 0.2

            # C. Address Number / ZIP overlap
            s1_numbers = s1_rec["addr_numbers"]
            for num in s1_numbers:
                if num in num_idx:
                    matched_targets = num_idx[num]
                    if len(matched_targets) < 30:
                        for tid in matched_targets:
                            cand_scores[tid] += 1.5

            # D. Fast Jaccard Verification & Filtering
            scored_candidates = []
            s1_full_tok = s1_rec["full_tokens"]
            s1_name_tok = s1_rec["name_tokens"]

            for tid, base_score in cand_scores.items():
                trec = target_records[tid]
                t_name_tok = trec["name_tokens"]
                t_full_tok = trec["full_tokens"]

                # Quick token Jaccard on names
                intersection_len = len(s1_name_tok.intersection(t_name_tok))
                union_len = len(s1_name_tok.union(t_name_tok))
                name_jaccard = intersection_len / union_len if union_len > 0 else 0.0

                # Full Jaccard on name + address
                full_inter = len(s1_full_tok.intersection(t_full_tok))
                full_union = len(s1_full_tok.union(t_full_tok))
                full_jaccard = full_inter / full_union if full_union > 0 else 0.0

                # Composite fast score
                fast_score = base_score + (name_jaccard * 4.0) + (full_jaccard * 2.0)

                # Prune extremely weak matches
                if fast_score >= self.config.JACCARD_PRUNING_THRESHOLD:
                    scored_candidates.append((tid, fast_score))

            # Sort descending by fast score and pick top K
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_cands = [c[0] for c in scored_candidates[: self.config.TOP_CANDIDATES_PER_S1]]

            candidate_dict[s1_id] = top_cands

        return candidate_dict, s1_records, target_records

    def write_candidate_pairs_tsv(
        self, candidate_dict: Dict[str, List[str]], output_path: str
    ):
        """
        Saves candidate_pairs.tsv in the exact competition format:
        source1_entity_id\tcandidate_entity_ids
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in sorted(candidate_dict.keys()):
                cand_list = candidate_dict[s1_id]
                cand_str = ",".join(cand_list)
                f.write(f"{s1_id}\t{cand_str}\n")
