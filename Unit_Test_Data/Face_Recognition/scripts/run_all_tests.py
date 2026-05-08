"""
Master Test Runner — FR Extended Real-Image Tests
Runs all 25 test scripts and prints a summary table.

Usage:
  # Step 1 (once): prepare test data from real footage
  python 00_extract_video_data.py

  # Step 2: run all tests
  python run_all_tests.py

  # Or run a single group:
  python run_all_tests.py --group FR-EX
  python run_all_tests.py --group FR-PD
  python run_all_tests.py --group FR-SM
  python run_all_tests.py --group FR-CC
"""

import sys
import argparse
import importlib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Map test_id → module name (without .py)
TESTS = [
    ("FR-EX-01", "test_fr_ex_01"),
    ("FR-EX-02", "test_fr_ex_02"),
    ("FR-EX-03", "test_fr_ex_03"),
    ("FR-EX-04", "test_fr_ex_04"),
    ("FR-EX-05", "test_fr_ex_05"),
    ("FR-PD-01", "test_fr_pd_01"),
    ("FR-PD-02", "test_fr_pd_02"),
    ("FR-PD-03", "test_fr_pd_03"),
    ("FR-PD-04", "test_fr_pd_04"),
    ("FR-PD-05", "test_fr_pd_05"),
    ("FR-PD-06", "test_fr_pd_06"),
    ("FR-SM-01", "test_fr_sm_01"),
    ("FR-SM-02", "test_fr_sm_02"),
    ("FR-SM-03", "test_fr_sm_03"),
    ("FR-SM-04", "test_fr_sm_04"),
    ("FR-SM-05", "test_fr_sm_05"),
    ("FR-SM-06", "test_fr_sm_06"),
    ("FR-SM-07", "test_fr_sm_07"),
    ("FR-CC-01", "test_fr_cc_01"),
    ("FR-CC-02", "test_fr_cc_02"),
    ("FR-CC-03", "test_fr_cc_03"),
    ("FR-CC-04", "test_fr_cc_04"),
    ("FR-CC-06", "test_fr_cc_06"),
    ("FR-CC-07", "test_fr_cc_07"),
    ("FR-CC-08", "test_fr_cc_08"),
]


def main():
    parser = argparse.ArgumentParser(description="Run all FR extended real-image tests.")
    parser.add_argument("--group", default=None,
                        help="Run only tests from a specific group (e.g. FR-EX, FR-SM)")
    args = parser.parse_args()

    to_run = TESTS
    if args.group:
        prefix = args.group.upper()
        to_run = [(tid, mod) for tid, mod in TESTS if tid.startswith(prefix)]
        if not to_run:
            print(f"No tests found for group '{args.group}'")
            sys.exit(1)

    results = []  # (test_id, verdict)  where verdict in PASS/FAIL/SKIP/ERROR

    for test_id, module_name in to_run:
        try:
            mod = importlib.import_module(module_name)
            outcome = mod.run()
            if outcome is None:
                verdict = "SKIP"
            elif outcome:
                verdict = "PASS"
            else:
                verdict = "FAIL"
        except Exception as exc:
            verdict = "ERROR"
            print(f"  [ERROR] {test_id}: {exc}")
        results.append((test_id, verdict))

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}
    for test_id, verdict in results:
        tag = f"[{verdict}]"
        print(f"  {tag:7s}  {test_id}")
        counts[verdict] += 1

    print("-" * 50)
    print(f"  PASS={counts['PASS']}  FAIL={counts['FAIL']}  "
          f"SKIP={counts['SKIP']}  ERROR={counts['ERROR']}")
    print(f"  Total: {len(results)} tests")

    if counts["FAIL"] > 0 or counts["ERROR"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
