"""
Post-run cache union utility.

Merge policy: master is authoritative for any key whose value is a "valid"
(non-empty, non-sentinel) string. Run caches can only (a) add new keys or
(b) replace master entries whose value is a known miss/timeout sentinel
(e.g. "No search results found."). This prevents fresh runs -- whose results
may drift across time -- from clobbering curated master results.

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


DEFAULT_INVALID_VALUE_SENTINELS: tuple[str, ...] = ("No search results found.",)


def load_cache(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Cache file must contain a JSON object: {path}")
    return data


def is_valid_value(value: object, invalid_sentinels: frozenset) -> bool:
    # A cache value is "valid" iff it is a non-empty string that is not one of
    # the miss/timeout sentinels the search tool writes on failure. Only valid
    # master values are protected from being overwritten by run caches.
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return bool(stripped) and stripped not in invalid_sentinels


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
    parser.add_argument(
        "--invalid-sentinels",
        type=str,
        nargs="*",
        default=list(DEFAULT_INVALID_VALUE_SENTINELS),
        help=(
            "Cache values treated as miss/timeout sentinels. Master entries "
            "holding one of these are NOT protected and may be replaced by "
            "fresh run-cache data. Defaults to the sentinel the search tool "
            "writes on empty API response."
        ),
    )
    args = parser.parse_args()

    master_cache_path = args.master_cache
    output_cache_path = args.output_cache
    invalid_sentinels = frozenset(args.invalid_sentinels)

    master_cache = load_cache(master_cache_path)
    merged_cache = dict(master_cache)
    # Keys whose master value is trusted (valid, non-sentinel). Run caches are
    # forbidden from overwriting these, even if they carry conflicting values.
    protected_keys = {k for k, v in master_cache.items() if is_valid_value(v, invalid_sentinels)}

    run_cache_paths = collect_run_cache_paths(args.run_cache_inputs, master_cache_path, output_cache_path)

    protected_skips = 0
    for run_cache_path in tqdm(run_cache_paths, desc="Merging run caches", unit="file"):
        run_cache = load_cache(run_cache_path)
        for k, v in run_cache.items():
            if k in protected_keys:
                protected_skips += 1
                continue
            merged_cache[k] = v

    output_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with output_cache_path.open("w", encoding="utf-8") as f:
        json.dump(merged_cache, f, ensure_ascii=False, indent=2)

    # End-state diff: unambiguous counts regardless of run-cache write order.
    master_sentinel_keys = set(master_cache) - protected_keys
    upgraded_from_sentinel = sum(
        1 for k in master_sentinel_keys if is_valid_value(merged_cache[k], invalid_sentinels)
    )
    new_keys_added = len(merged_cache) - len(master_cache)

    print(f"Master entries: {len(master_cache)}")
    print(f"Master entries protected (valid, not overwritable): {len(protected_keys)}")
    print(f"Master sentinel entries (eligible for upgrade): {len(master_sentinel_keys)}")
    print(f"Run cache files merged: {len(run_cache_paths)}")
    print(f"Protected-key overwrites skipped: {protected_skips}")
    print(f"Master sentinels upgraded to valid run value: {upgraded_from_sentinel}")
    print(f"New keys added from run caches: {new_keys_added}")
    print(f"Union entries written: {len(merged_cache)}")
    print(f"Output file: {output_cache_path}")


if __name__ == "__main__":
    main()
