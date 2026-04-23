"""
Post-run cache union utility.

Usage (merge master + all experiment caches matched by glob):
python ARPO/verl_arpo_entropy/recipe/echo/merge_search_cache_postrun.py \
  --master-cache /scratch/user/saratb_tamu.edu/research/arpo-lens/ARPO/search_cache/search_cache.json \
  --run-cache-inputs "/scratch/user/saratb_tamu.edu/research/arpo-lens/ARPO/search_cache/echo3B*.json" \
  --output-cache /scratch/user/saratb_tamu.edu/research/arpo-lens/ARPO/search_cache/search_cache_union_candidate.json

Usage (merge multiple explicit files and patterns in one run):
python ARPO/verl_arpo_entropy/recipe/echo/merge_search_cache_postrun.py \
  --master-cache /scratch/user/saratb_tamu.edu/research/arpo-lens/ARPO/search_cache/search_cache.json \
  --run-cache-inputs \
    "/scratch/user/saratb_tamu.edu/research/arpo-lens/ARPO/search_cache/echo3B_hl_ll_sel_low.json" \
    "/scratch/user/saratb_tamu.edu/research/arpo-lens/ARPO/search_cache/echo3B_ll_hl_sel_low.json" \
    "/scratch/user/saratb_tamu.edu/research/arpo-lens/ARPO/search_cache/echo3B*.json" \
  --output-cache /scratch/user/saratb_tamu.edu/research/arpo-lens/ARPO/search_cache/search_cache_union_candidate.json
"""

import argparse
import glob
import json
from pathlib import Path

from tqdm.auto import tqdm


def load_cache(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Cache file must contain a JSON object: {path}")
    return data


def collect_run_cache_paths(inputs: list[str], master_cache_path: Path, output_cache_path: Path) -> list[Path]:
    candidate_paths: set[Path] = set()
    for run_cache_input in inputs:
        matched_paths = [Path(p) for p in glob.glob(run_cache_input)]
        if matched_paths:
            candidate_paths.update(matched_paths)
        else:
            candidate_paths.add(Path(run_cache_input))

    run_cache_paths = sorted(
        p
        for p in candidate_paths
        if p.is_file() and p.resolve() != master_cache_path.resolve() and p.resolve() != output_cache_path.resolve()
    )
    return run_cache_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a post-run union cache file from master + run-local caches."
    )
    parser.add_argument(
        "--master-cache",
        type=Path,
        required=True,
        help="Path to the shared master cache JSON used as the base.",
    )
    parser.add_argument(
        "--run-cache-inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more run-cache inputs. Each input can be a glob pattern or an explicit file path.",
    )
    parser.add_argument(
        "--output-cache",
        type=Path,
        required=True,
        help="Path for the merged output cache JSON; this file is written instead of the master cache.",
    )
    args = parser.parse_args()

    master_cache_path = args.master_cache
    output_cache_path = args.output_cache

    master_cache = load_cache(master_cache_path)
    merged_cache = dict(master_cache)

    run_cache_paths = collect_run_cache_paths(args.run_cache_inputs, master_cache_path, output_cache_path)

    for run_cache_path in tqdm(run_cache_paths, desc="Merging run caches", unit="file"):
        run_cache = load_cache(run_cache_path)
        merged_cache.update(run_cache)

    output_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with output_cache_path.open("w", encoding="utf-8") as f:
        json.dump(merged_cache, f, ensure_ascii=False, indent=2)

    print(f"Master entries: {len(master_cache)}")
    print(f"Run cache files merged: {len(run_cache_paths)}")
    print(f"Union entries written: {len(merged_cache)}")
    print(f"Output file: {output_cache_path}")


if __name__ == "__main__":
    main()
