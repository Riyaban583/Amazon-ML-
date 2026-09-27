import argparse
import os
import zipfile


def package_submission(team_name: str, base_dir: str):
    zip_filename = f"{team_name}_submission.zip"
    zip_filepath = os.path.join(base_dir, zip_filename)

    output_dir = os.path.join(base_dir, "output")
    code_dir = os.path.join(base_dir, "code", "business_entity_resolution")
    doc_path = os.path.join(base_dir, "Documentation_template.md")

    # Verify required components
    required_files = [
        os.path.join(output_dir, "matching_results.tsv"),
        os.path.join(output_dir, "candidate_pairs.tsv"),
        doc_path,
        os.path.join(code_dir, "README.md"),
        os.path.join(code_dir, "requirements.txt"),
    ]

    for rf in required_files:
        if not os.path.exists(rf):
            raise FileNotFoundError(f"Missing required component for submission package: {rf}")

    print(f"Creating submission archive: {zip_filename}...")
    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. output/
        zf.write(os.path.join(output_dir, "matching_results.tsv"), "output/matching_results.tsv")
        zf.write(os.path.join(output_dir, "candidate_pairs.tsv"), "output/candidate_pairs.tsv")

        # 2. Documentation_template.md
        zf.write(doc_path, "Documentation_template.md")

        # 3. code/business_entity_resolution/
        for root, dirs, files in os.walk(code_dir):
            # Exclude __pycache__ or temporary files
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith(".pyc") or f.endswith(".pyo"):
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, base_dir)
                zf.write(full_path, rel_path)

    print(f"Submission package successfully created at: {zip_filepath}")
    print("\nPackage Contents:")
    with zipfile.ZipFile(zip_filepath, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename:45s} ({info.file_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package submission zip archive.")
    parser.add_argument("--team-name", default="AmazonER_Team", help="Name of team for submission archive")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    package_submission(args.team_name, base)
