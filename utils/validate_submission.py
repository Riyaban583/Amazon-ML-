import argparse
import csv
import os
import sys
from typing import Dict, List, Set, Tuple


def read_tsv_rows(filepath: str) -> Tuple[List[str], List[List[str]]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    rows = []
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"File is empty: {filepath}")
        for line_num, row in enumerate(reader, start=2):
            rows.append(row)
    return header, rows


def validate_submission(matching_file: str, candidate_file: str, test_dir: str) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    # 1. Check test source files exist
    s1_file = os.path.join(test_dir, "test_source1.tsv")
    s2_file = os.path.join(test_dir, "test_source2.tsv")
    s3_file = os.path.join(test_dir, "test_source3.tsv")

    for fpath in [s1_file, s2_file, s3_file]:
        if not os.path.exists(fpath):
            issues.append(f"Missing required test file: {fpath}")
    if issues:
        return False, issues

    # Load valid entity IDs
    def load_entity_ids(path: str) -> Set[str]:
        ids = set()
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # header
            for row in reader:
                if row and len(row) > 0:
                    ids.add(row[0].strip())
        return ids

    valid_s1 = load_entity_ids(s1_file)
    valid_s2 = load_entity_ids(s2_file)
    valid_s3 = load_entity_ids(s3_file)
    valid_s2_s3 = valid_s2.union(valid_s3)

    num_expected_s1 = len(valid_s1)

    # 2. Validate candidate_pairs.tsv
    cand_dict: Dict[str, List[str]] = {}
    if not os.path.exists(candidate_file):
        issues.append(f"candidate_pairs file does not exist: {candidate_file}")
    else:
        try:
            c_header, c_rows = read_tsv_rows(candidate_file)
            if c_header != ["source1_entity_id", "candidate_entity_ids"]:
                issues.append(
                    f"candidate_pairs.tsv header must be ['source1_entity_id', 'candidate_entity_ids'], got {c_header}"
                )

            seen_c_s1 = set()
            for idx, r in enumerate(c_rows, start=2):
                if len(r) == 0:
                    continue
                s1_id = r[0].strip()
                cand_str = r[1].strip() if len(r) > 1 else ""

                if s1_id in seen_c_s1:
                    issues.append(f"candidate_pairs.tsv: Duplicate source1_entity_id '{s1_id}' at line {idx}")
                seen_c_s1.add(s1_id)

                if s1_id not in valid_s1:
                    issues.append(f"candidate_pairs.tsv: Unknown source1_entity_id '{s1_id}' at line {idx}")

                cand_ids = [c.strip() for c in cand_str.split(",") if c.strip()]
                # Check duplicates in ID list
                if len(cand_ids) != len(set(cand_ids)):
                    issues.append(f"candidate_pairs.tsv: Duplicate candidate IDs in list for '{s1_id}' at line {idx}")

                for cid in cand_ids:
                    if cid.startswith("S1-"):
                        issues.append(f"candidate_pairs.tsv: Self-match candidate '{cid}' for '{s1_id}' at line {idx}")
                    elif cid not in valid_s2_s3:
                        issues.append(f"candidate_pairs.tsv: Candidate ID '{cid}' does not exist in test Source 2 or 3")

                cand_dict[s1_id] = cand_ids

            missing_cand_s1 = valid_s1 - seen_c_s1
            if missing_cand_s1:
                sample_missing = list(missing_cand_s1)[:5]
                issues.append(
                    f"candidate_pairs.tsv: Missing {len(missing_cand_s1)} Source 1 entities (e.g. {sample_missing})"
                )

        except Exception as e:
            issues.append(f"Error parsing candidate_pairs.tsv: {str(e)}")

    # 3. Validate matching_results.tsv
    if not os.path.exists(matching_file):
        issues.append(f"matching_results file does not exist: {matching_file}")
    else:
        try:
            m_header, m_rows = read_tsv_rows(matching_file)
            if m_header != ["source1_entity_id", "matched_entity_ids"]:
                issues.append(
                    f"matching_results.tsv header must be ['source1_entity_id', 'matched_entity_ids'], got {m_header}"
                )

            seen_m_s1 = set()
            for idx, r in enumerate(m_rows, start=2):
                if len(r) == 0:
                    continue
                s1_id = r[0].strip()
                match_str = r[1].strip() if len(r) > 1 else ""

                if s1_id in seen_m_s1:
                    issues.append(f"matching_results.tsv: Duplicate source1_entity_id '{s1_id}' at line {idx}")
                seen_m_s1.add(s1_id)

                if s1_id not in valid_s1:
                    issues.append(f"matching_results.tsv: Unknown source1_entity_id '{s1_id}' at line {idx}")

                match_ids = [m.strip() for m in match_str.split(",") if m.strip()]
                # Check duplicates in ID list
                if len(match_ids) != len(set(match_ids)):
                    issues.append(f"matching_results.tsv: Duplicate matched IDs in list for '{s1_id}' at line {idx}")

                for mid in match_ids:
                    if mid.startswith("S1-"):
                        issues.append(f"matching_results.tsv: Self-match '{mid}' for '{s1_id}' at line {idx}")
                    elif mid not in valid_s2_s3:
                        issues.append(f"matching_results.tsv: Matched ID '{mid}' does not exist in test Source 2 or 3")

                    # Check subset constraint: every matched ID should be in candidates
                    if s1_id in cand_dict and mid not in cand_dict[s1_id]:
                        issues.append(
                            f"matching_results.tsv: Matched ID '{mid}' for '{s1_id}' was not present in candidate_pairs.tsv"
                        )

            missing_m_s1 = valid_s1 - seen_m_s1
            if missing_m_s1:
                sample_missing = list(missing_m_s1)[:5]
                issues.append(
                    f"matching_results.tsv: Missing {len(missing_m_s1)} Source 1 entities (e.g. {sample_missing})"
                )

        except Exception as e:
            issues.append(f"Error parsing matching_results.tsv: {str(e)}")

    passed = len(issues) == 0
    return passed, issues


def main():
    parser = argparse.ArgumentParser(description="Validate submission TSVs for Amazon Entity Resolution Challenge.")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Directory containing test_source1.tsv, etc.")

    args = parser.parse_args()

    passed, issues = validate_submission(args.matching, args.candidate, args.test_dir)
    if passed:
        print("PASS")
        sys.exit(0)
    else:
        print(f"FAILED with {len(issues)} issue(s):")
        for i, issue in enumerate(issues, start=1):
            print(f"  {i}. {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()
