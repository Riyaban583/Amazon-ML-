import os
import sys
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import Config
from .blocking import CandidateGenerator
from .features import FeatureExtractor
from .model import EntityMatchingModel
from .evaluate import evaluate_macro_f05, optimize_thresholds


class EntityResolutionPipeline:
    """
    End-to-End Entity Resolution Pipeline integrating:
    - Riya: Candidate Generation & Scalable Blocking
    - Khushi: Feature Engineering & String Metrics
    - Akshat: ML Classification, Singleton Resolution & F_0.5 Optimization
    """

    def __init__(self, config: Config = Config()):
        self.config = config
        self.config.ensure_directories()
        self.candidate_generator = CandidateGenerator(self.config)
        self.feature_extractor = FeatureExtractor()
        self.model = EntityMatchingModel(random_state=self.config.SEED)
        self.singleton_threshold = self.config.SINGLETON_PROB_THRESHOLD
        self.match_threshold = self.config.MATCH_PROB_THRESHOLD

    def load_data(
        self, data_dir: str, prefix: str = "train"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
        """Loads tab-separated source files and ground truth if available."""
        s1_path = os.path.join(data_dir, f"{prefix}_source1.tsv")
        s2_path = os.path.join(data_dir, f"{prefix}_source2.tsv")
        s3_path = os.path.join(data_dir, f"{prefix}_source3.tsv")

        df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
        df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str).fillna("")
        df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str).fillna("")

        ground_truth: Dict[str, Set[str]] = {}
        gt_path = os.path.join(data_dir, f"{prefix}_ground_truth.tsv")
        if os.path.exists(gt_path):
            df_gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
            for _, row in df_gt.iterrows():
                s1_id = str(row["source1_entity_id"]).strip()
                matches_str = str(row.get("matched_entity_ids", "")).strip()
                if matches_str:
                    matched_ids = {m.strip() for m in matches_str.split(",") if m.strip()}
                else:
                    matched_ids = set()
                ground_truth[s1_id] = matched_ids

        return df_s1, df_s2, df_s3, ground_truth

    def evaluate_blocking_recall(
        self, candidate_dict: Dict[str, List[str]], ground_truth: Dict[str, Set[str]]
    ) -> Tuple[float, float, float]:
        """
        Computes:
        1. Blocking Recall: fraction of ground truth links captured in candidates
        2. Average candidate count per S1 entity
        3. Reduction ratio
        """
        total_true_links = 0
        captured_true_links = 0
        total_candidates = 0

        for s1_id, true_set in ground_truth.items():
            total_true_links += len(true_set)
            cand_set = set(candidate_dict.get(s1_id, []))
            captured_true_links += len(cand_set.intersection(true_set))
            total_candidates += len(cand_set)

        recall = captured_true_links / total_true_links if total_true_links > 0 else 1.0
        avg_cands = total_candidates / max(1, len(candidate_dict))
        return recall, avg_cands, total_candidates

    def train_and_validate(self):
        """
        Executes full training, validation split, metric optimization, and prints summary.
        """
        print("=" * 70)
        print("PHASE 1: Loading Training Data (US & India)...")
        print("=" * 70)
        df_s1, df_s2, df_s3, ground_truth = self.load_data(str(self.config.TRAIN_DIR), "train")
        print(f"Loaded: S1={len(df_s1)}, S2={len(df_s2)}, S3={len(df_s3)}, GroundTruth={len(ground_truth)}")

        print("\n" + "=" * 70)
        print("PHASE 2: Riya's Blocking & Candidate Generation...")
        print("=" * 70)
        cand_dict, s1_recs, target_recs = self.candidate_generator.build_candidate_pairs(df_s1, df_s2, df_s3)
        b_recall, avg_cands, total_cands = self.evaluate_blocking_recall(cand_dict, ground_truth)
        
        # Calculate reduction ratio against full Cartesian product
        total_cartesian = len(df_s1) * (len(df_s2) + len(df_s3))
        reduction_ratio = 1.0 - (total_cands / max(1, total_cartesian))
        
        print(f"Blocking Recall Ceiling: {b_recall * 100:.2f}%")
        print(f"Avg Candidates per S1:   {avg_cands:.2f}")
        print(f"Candidate Reduction:     {reduction_ratio * 100:.4f}% ({total_cands} vs {total_cartesian} pairs)")

        print("\n" + "=" * 70)
        print("PHASE 3: Khushi's Feature Engineering & Similarity Extraction...")
        print("=" * 70)
        X, y, pairs = self.feature_extractor.build_feature_matrix(
            cand_dict, s1_recs, target_recs, ground_truth
        )
        print(f"Constructed Feature Matrix: {X.shape[0]} pairs, {X.shape[1]} features")
        print(f"Class Distribution: Matches={np.sum(y == 1)}, Non-matches={np.sum(y == 0)}")

        print("\n" + "=" * 70)
        print("PHASE 4: Akshat's ML Training & F_0.5 Threshold Optimization...")
        print("=" * 70)
        # Train ML Model
        self.model.fit(X, y)
        probs = self.model.predict_proba(X)

        # Pair up scores
        scored_pairs = []
        for (s1_id, tid), p in zip(pairs, probs):
            scored_pairs.append((s1_id, tid, float(p)))

        # Find optimal thresholds for precision-weighted F_0.5
        all_s1_ids = set(df_s1["entity_id"].str.strip())
        best_s_th, best_m_th, train_f05 = optimize_thresholds(scored_pairs, ground_truth, all_s1_ids)
        self.singleton_threshold = best_s_th
        self.match_threshold = best_m_th

        # Evaluate performance with tuned thresholds
        preds = self._predict_from_scored_pairs(scored_pairs, all_s1_ids)
        f05, prec, rec, single_acc = evaluate_macro_f05(preds, ground_truth)

        print(f"Optimized Singleton Threshold: {self.singleton_threshold:.2f}")
        print(f"Optimized Multi-Match Threshold: {self.match_threshold:.2f}")
        print(f"--> Macro F_0.5 Score:          {f05:.4f}")
        print(f"--> Precision (Non-singletons): {prec:.4f}")
        print(f"--> Recall (Non-singletons):    {rec:.4f}")
        print(f"--> Singleton Accuracy:         {single_acc * 100:.2f}%")

        # Feature importances
        importances = self.model.get_feature_importances(self.feature_extractor.FEATURE_NAMES)
        if not importances.empty:
            print("\nTop 5 Predictive Features:")
            for feat, val in importances.head(5).items():
                print(f"  - {feat:25s}: {val}")

        # Save model
        model_path = os.path.join(self.config.MODEL_DIR, "matching_model.joblib")
        self.model.save(model_path, metadata={
            "singleton_threshold": self.singleton_threshold,
            "match_threshold": self.match_threshold,
        })
        print(f"\nSaved trained model artifact to {model_path}")

    def _predict_from_scored_pairs(
        self, scored_pairs: List[Tuple[str, str, float]], all_s1_ids: Set[str]
    ) -> Dict[str, List[str]]:
        """Groups candidate pairs by S1, applies singleton and match thresholds."""
        s1_candidates = {s1: [] for s1 in all_s1_ids}
        for s1_id, tid, prob in scored_pairs:
            if s1_id in s1_candidates:
                s1_candidates[s1_id].append((tid, prob))

        preds = {}
        for s1_id, cands in s1_candidates.items():
            if not cands:
                preds[s1_id] = []
                continue
            cands.sort(key=lambda x: x[1], reverse=True)
            top_tid, top_prob = cands[0]

            # High precision bar: if top candidate below singleton threshold, predict empty
            if top_prob < self.singleton_threshold:
                preds[s1_id] = []
            else:
                chosen = [top_tid]
                for tid, p in cands[1:]:
                    if p >= self.match_threshold and (top_prob - p) < 0.25:
                        chosen.append(tid)
                    if len(chosen) >= self.config.MAX_MATCHES_PER_S1:
                        break
                preds[s1_id] = chosen
        return preds

    def run_inference_and_generate_submission(
        self,
        test_dir: str = None,
        candidate_out_path: str = None,
        matching_out_path: str = None,
    ):
        """
        Executes end-to-end inference on test data:
        1. Multi-country candidate generation (handles US, India, France!)
        2. Saves candidate_pairs.tsv
        3. Feature extraction & model probability prediction
        4. Singleton & multi-match resolution
        5. Saves matching_results.tsv
        """
        test_dir = test_dir or str(self.config.TEST_DIR)
        candidate_out_path = candidate_out_path or str(self.config.OUTPUT_DIR / "candidate_pairs.tsv")
        matching_out_path = matching_out_path or str(self.config.OUTPUT_DIR / "matching_results.tsv")

        print("=" * 70)
        print("INFERENCE: Processing Test Set (US, India, and France)...")
        print("=" * 70)
        df_s1, df_s2, df_s3, _ = self.load_data(test_dir, "test")
        all_s1_ids = sorted(list(set(df_s1["entity_id"].str.strip())))
        print(f"Test entities: S1={len(df_s1)}, S2={len(df_s2)}, S3={len(df_s3)}")

        # Step 1: Candidate Generation
        print("Generating candidate pairs via blocking...")
        cand_dict, s1_recs, target_recs = self.candidate_generator.build_candidate_pairs(df_s1, df_s2, df_s3)

        # Ensure every S1 is present
        for s1_id in all_s1_ids:
            if s1_id not in cand_dict:
                cand_dict[s1_id] = []

        # Step 2: Write candidate_pairs.tsv
        print(f"Writing candidate_pairs.tsv to: {candidate_out_path}")
        self.candidate_generator.write_candidate_pairs_tsv(cand_dict, candidate_out_path)

        # Step 3: Feature Extraction
        print("Extracting similarity features...")
        X_test, _, pairs = self.feature_extractor.build_feature_matrix(
            cand_dict, s1_recs, target_recs, ground_truth=None
        )

        # Step 4: ML Prediction
        # Ensure model is fitted
        if not self.model.is_fitted:
            model_path = os.path.join(self.config.MODEL_DIR, "matching_model.joblib")
            if os.path.exists(model_path):
                print(f"Loading pre-trained model from {model_path}...")
                metadata = self.model.load(model_path)
                if isinstance(metadata, dict):
                    if "singleton_threshold" in metadata:
                        self.singleton_threshold = metadata["singleton_threshold"]
                    if "match_threshold" in metadata:
                        self.match_threshold = metadata["match_threshold"]
            else:
                print("No pre-trained model found. Running training and validation first...")
                self.train_and_validate()

        print("Scoring candidate pairs with trained ML model...")
        probs = self.model.predict_proba(X_test)

        scored_pairs = []
        for (s1_id, tid), p in zip(pairs, probs):
            scored_pairs.append((s1_id, tid, float(p)))

        # Step 5: Final Match Generation
        print("Applying singleton detector & precision-weighted thresholding...")
        matches_dict = self._predict_from_scored_pairs(scored_pairs, set(all_s1_ids))

        # Enforce subset constraint: matched IDs must be in candidate list
        for s1_id in all_s1_ids:
            cand_set = set(cand_dict.get(s1_id, []))
            matches_dict[s1_id] = [m for m in matches_dict.get(s1_id, []) if m in cand_set]

        # Step 6: Write matching_results.tsv
        print(f"Writing matching_results.tsv to: {matching_out_path}")
        with open(matching_out_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id in all_s1_ids:
                m_list = matches_dict.get(s1_id, [])
                f.write(f"{s1_id}\t{','.join(m_list)}\n")

        # Summary statistics
        total_s1 = len(all_s1_ids)
        total_singletons_predicted = sum(1 for s1 in all_s1_ids if len(matches_dict.get(s1, [])) == 0)
        total_matches_predicted = sum(len(matches_dict.get(s1, [])) for s1 in all_s1_ids)
        print("\nTest Prediction Summary:")
        print(f"  - Total S1 Entities:            {total_s1}")
        print(f"  - Predicted Singletons:         {total_singletons_predicted} ({total_singletons_predicted/total_s1*100:.1f}%)")
        print(f"  - Total Matches Predicted:      {total_matches_predicted}")
        print("Output files generated successfully.")
