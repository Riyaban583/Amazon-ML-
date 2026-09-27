import argparse
import os
import sys
import subprocess

# Ensure src can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution"))

from src.config import Config
from src.pipeline import EntityResolutionPipeline


def main():
    parser = argparse.ArgumentParser(description="Run Amazon Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["all", "train", "predict"], default="all",
                        help="Execution mode: all (train + predict + validate), train only, or predict only")
    parser.add_argument("--train-dir", default=None, help="Path to train dataset directory")
    parser.add_argument("--test-dir", default=None, help="Path to test dataset directory")
    args = parser.parse_args()

    config = Config()
    if args.train_dir:
        config.TRAIN_DIR = args.train_dir
    if args.test_dir:
        config.TEST_DIR = args.test_dir

    pipeline = EntityResolutionPipeline(config)

    # 1. Train & Validate
    if args.mode in ["all", "train"]:
        pipeline.train_and_validate()

    # 2. Predict on Test
    if args.mode in ["all", "predict"]:
        cand_out = str(config.OUTPUT_DIR / "candidate_pairs.tsv")
        match_out = str(config.OUTPUT_DIR / "matching_results.tsv")
        pipeline.run_inference_and_generate_submission(
            test_dir=str(config.TEST_DIR),
            candidate_out_path=cand_out,
            matching_out_path=match_out,
        )

        # 3. Automatic Validation
        print("\n" + "=" * 70)
        print("PHASE 5: Validating Generated Submission TSVs...")
        print("=" * 70)
        validator_path = os.path.join(os.path.dirname(__file__), "utils", "validate_submission.py")
        cmd = [
            sys.executable,
            validator_path,
            "--matching", match_out,
            "--candidate", cand_out,
            "--test-dir", str(config.TEST_DIR),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode == 0:
            print(">>> ALL SUBMISSION CHECKS PASSED SUCCESSFULLY (Exit 0) <<<")
        else:
            print(">>> VALIDATION WARNINGS DETECTED <<<")


if __name__ == "__main__":
    main()
