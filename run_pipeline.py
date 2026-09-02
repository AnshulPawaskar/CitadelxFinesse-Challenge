"""Single entry point to run the entire offline research/backtest/reporting pipeline in order.

Assumes historical OHLCV data has already been extracted (see main.py, which drives the live
5Paisa extraction — a separate, deliberately independent concern from this offline pipeline).

Usage:
    python run_pipeline.py                 # run every stage
    python run_pipeline.py --only features experiments   # run just these stages
    python run_pipeline.py --skip experiments             # run everything except this stage
    python run_pipeline.py --list           # list available stage names and exit
"""
import argparse
import time
import traceback

STAGES = [
    ("data_audit", "src.data.validator", "run_full_audit"),
    ("features", "src.features.feature_pipeline", "run_feature_pipeline"),
    ("momentum_baseline", "src.research.momentum_baseline", "run"),
    ("model_comparison", "src.research.model_comparison", "run"),
    ("feature_correlation", "src.research.feature_correlation", "run"),
    ("correlation_selection", "src.research.correlation_selection_comparison", "run"),
    ("weighting_comparison", "src.research.weighting_comparison", "run"),
    ("experiments", "src.research.experiments", "run_experiment_matrix"),
    ("sensitivity", "src.research.sensitivity", "run"),
    ("competition_backtest", "src.competition.final_submission", "run"),
    ("reports", "src.reporting.build_report", "run"),
    ("final_report", "src.reporting.final_report", "build_final_submission_report"),
    ("oracle_comparison", "src.research.oracle_comparison_2026h1", "run"),
]


def run_stage(name: str, module_path: str, func_name: str) -> tuple[bool, float]:
    import importlib
    print(f"\n{'=' * 80}\n>>> {name}\n{'=' * 80}")
    start = time.time()
    try:
        module = importlib.import_module(module_path)
        getattr(module, func_name)()
        elapsed = time.time() - start
        print(f"<<< {name} completed in {elapsed:.1f}s")
        return True, elapsed
    except Exception:
        elapsed = time.time() - start
        print(f"<<< {name} FAILED after {elapsed:.1f}s")
        traceback.print_exc()
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the full research/backtest/reporting pipeline.")
    parser.add_argument("--only", nargs="+", metavar="STAGE", help="Run only these stages")
    parser.add_argument("--skip", nargs="+", metavar="STAGE", help="Run every stage except these")
    parser.add_argument("--list", action="store_true", help="List stage names and exit")
    args = parser.parse_args()

    stage_names = [name for name, _, _ in STAGES]
    if args.list:
        print("Available stages:\n  " + "\n  ".join(stage_names))
        return

    selected = STAGES
    if args.only:
        unknown = set(args.only) - set(stage_names)
        if unknown:
            raise SystemExit(f"Unknown stage(s): {sorted(unknown)}. Use --list to see valid names.")
        selected = [s for s in STAGES if s[0] in args.only]
    elif args.skip:
        unknown = set(args.skip) - set(stage_names)
        if unknown:
            raise SystemExit(f"Unknown stage(s): {sorted(unknown)}. Use --list to see valid names.")
        selected = [s for s in STAGES if s[0] not in args.skip]

    results = []
    pipeline_start = time.time()
    for name, module_path, func_name in selected:
        ok, elapsed = run_stage(name, module_path, func_name)
        results.append((name, ok, elapsed))

    print(f"\n{'=' * 80}\nPIPELINE SUMMARY\n{'=' * 80}")
    for name, ok, elapsed in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status:6}] {name:24} {elapsed:6.1f}s")
    total = time.time() - pipeline_start
    n_failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\nTotal: {total:.1f}s | {len(results) - n_failed}/{len(results)} stages succeeded")

    if n_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
