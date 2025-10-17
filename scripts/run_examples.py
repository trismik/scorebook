#!/usr/bin/env python
"""Script to run all examples with toggleable execution.

Toggle the boolean flags below to enable/disable specific examples.
By default, examples 6, 8, and 9 are disabled.

Usage:
    python scripts/run_examples.py           # Run all enabled examples
    python scripts/run_examples.py --dry-run # Preview which examples will run
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict

# =============================================================================
# Configuration: Toggle these boolean flags to enable/disable examples
# =============================================================================

EXAMPLE_FLAGS = {
    "example_1_simple_evaluation.py": True,
    "example_2.1_evaluation_datasets.py": True,
    "example_2.2_evaluation_datasets.py": True,
    "example_3_inference_pipelines.py": True,
    "example_4_batch_inference.py": True,
    "example_5_cloud_inference.py": True,
    "example_6_cloud_batch_inference.py": False,
    "example_7_hyperparameters.py": True,
    "example_8_uploading_results.py": False,  # Disabled by default
    "example_9_adaptive_evaluation.py": False,  # Disabled by default
}

# =============================================================================
# Runner Implementation
# =============================================================================


def run_example(example_path: Path) -> bool:
    """Run a single example and return True if successful, False otherwise."""
    print(f"\n{'=' * 80}")
    print(f"Running: {example_path.name}")
    print(f"{'=' * 80}\n")

    try:
        result = subprocess.run(
            [sys.executable, str(example_path)],
            cwd=example_path.parent.parent,  # Run from project root
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per example
        )

        if result.returncode == 0:
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            print(f"\n✓ {example_path.name} completed successfully")
            return True
        else:
            print(result.stdout)
            print("STDERR:", result.stderr)
            print(f"\n✗ {example_path.name} failed with exit code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n✗ {example_path.name} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"\n✗ {example_path.name} raised an exception: {e}")
        return False


def main() -> None:
    """Run examples with configurable execution."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Scorebook examples with toggleable execution")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which examples will run without executing them",
    )
    args = parser.parse_args()

    # Find the examples directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    examples_dir = project_root / "examples"

    if not examples_dir.exists():
        print(f"Error: Examples directory not found at {examples_dir}")
        sys.exit(1)

    # Collect enabled examples
    enabled_examples = []
    disabled_examples = []

    for example_name, enabled in EXAMPLE_FLAGS.items():
        example_path = examples_dir / example_name
        if not example_path.exists():
            print(f"Warning: Example file not found: {example_name}")
            continue

        if enabled:
            enabled_examples.append((example_name, example_path))
        else:
            disabled_examples.append(example_name)

    # Print summary
    print("\n" + "=" * 80)
    print("EXAMPLE RUNNER")
    print("=" * 80)
    print(f"\nEnabled examples: {len(enabled_examples)}")
    for name, _ in enabled_examples:
        print(f"  ✓ {name}")

    if disabled_examples:
        print(f"\nDisabled examples: {len(disabled_examples)}")
        for name in disabled_examples:
            print(f"  ✗ {name}")

    if not enabled_examples:
        print("\nNo examples enabled. Edit EXAMPLE_FLAGS in this script to enable examples.")
        sys.exit(0)

    # If dry-run, exit here
    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN MODE - Not executing examples")
        print("=" * 80)
        print("\nTo run these examples, execute without --dry-run flag.")
        sys.exit(0)

    # Run enabled examples
    print("\n" + "=" * 80)
    print("RUNNING EXAMPLES")
    print("=" * 80)

    results: Dict[str, bool] = {}
    for example_name, example_path in enabled_examples:
        success = run_example(example_path)
        results[example_name] = success

    # Print final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    print(f"\nTotal: {len(results)} examples run")
    print(f"Passed: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\nSuccessful examples:")
        for name in successful:
            print(f"  ✓ {name}")

    if failed:
        print("\nFailed examples:")
        for name in failed:
            print(f"  ✗ {name}")

    # Exit with error code if any examples failed
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
