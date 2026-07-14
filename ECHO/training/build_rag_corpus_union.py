"""
Naive union of all ECHO/search_cache/*.json into one FAISS-ready corpus.

Last-write-wins via dict.update in sorted path order. Skips .lock / .tmp
(only *.json), and excludes the output file from the merge inputs.

Usage:
python ECHO/training/build_rag_corpus_union.py
"""

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "ECHO" / "search_cache"
DEFAULT_OUTPUT = DEFAULT_CACHE_DIR / "search_cache_union_rag.json"


def load_cache(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Cache file must contain a JSON object: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Naive-union all search_cache/*.json into one RAG corpus JSON."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory of query→result JSON caches.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output union corpus path.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    output_path = args.output.resolve()

    source_paths = sorted(
        p
        for p in cache_dir.glob("*.json")
        if p.is_file() and p.resolve() != output_path
    )

    union: dict = {}
    per_file_counts: list[tuple[str, int]] = []
    for path in tqdm(source_paths, desc="Unioning search caches", unit="file"):
        cache = load_cache(path)
        per_file_counts.append((path.name, len(cache)))
        union.update(cache)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(union, f, ensure_ascii=False)

    print(f"Cache dir: {cache_dir}")
    print(f"Source files: {len(source_paths)}")
    for name, n in per_file_counts:
        print(f"  {name}: {n} keys")
    print(f"Union entries written: {len(union)}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
