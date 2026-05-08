"""
FR-CC-04 — Empty Customer Pool → Empty Cluster List
Real-image test: call greedy clustering with an empty customer_data list.
Must return [] without any exception.

Expected:
  clusters == []
  No exception raised

Run:
  python test_fr_cc_04.py
  (no images needed — pure logic test)
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from _helpers import greedy_cluster, report


def run():
    print("=" * 60)
    print("FR-CC-04  Empty Customer Pool → Empty Cluster List")
    print("=" * 60)
    print("  (pure logic test — no face images needed)")

    clusters = greedy_cluster([], cluster_threshold=0.60)
    print(f"\n  Result : {clusters!r}")

    passed = True
    passed &= report("FR-CC-04-a", "greedy_cluster([]) returns []",
                     clusters == [],
                     f"got {clusters!r}")
    passed &= report("FR-CC-04-b", "no exception raised",
                     True, "reached end without exception")
    print()
    return passed


if __name__ == "__main__":
    run()
