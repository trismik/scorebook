#!/usr/bin/env python
"""Script to run all examples with toggleable execution.

Toggle the boolean flags below to enable/disable specific examples.
By default, examples requiring API keys or credentials are disabled.

Usage:
    python scripts/run_examples.py           # Run all enabled examples
    python scripts/run_examples.py --dry-run # Preview which examples will run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

# =============================================================================
# Configuration: Toggle these boolean flags to enable/disable examples
# =============================================================================

EXAMPLE_FLAGS = {
    # 1-score: Basic scoring examples
    "1-score/1-scoring_model_accuracy.py": True,
    "1-score/2-scoring_model_bleu.py": True,
    "1-score/3-scoring_model_f1.py": True,
    "1-score/4-scoring_model_rouge.py": True,
    "1-score/5-scoring_model_exact_match.py": True,
    # 2-evaluate: Evaluation examples
    "2-evaluate/1-evaluating_local_models.py": True,
    "2-evaluate/2-evaluating_local_models_with_batching.py": True,
    "2-evaluate/3-evaluating_cloud_models.py": False,  # Requires OpenAI API key
    "2-evaluate/4-evaluating_cloud_models_with_batching.py": False,  # Requires OpenAI API key
    "2-evaluate/5-hyperparameter_sweeps.py": True,
    "2-evaluate/6-inference_pipelines.py": True,
    # 3-evaluation_datasets: Dataset loading examples
    "3-evaluation_datasets/1-evaluation_datasets_from_files.py": True,
    "3-evaluation_datasets/2-evaluation_datasets_from_huggingface.py": False,  # Requires OpenAI
    # Requires OpenAI
    "3-evaluation_datasets/3-evaluation_datasets_from_huggingface_with_yaml_configs.py": False,
    # 4-adaptive_evaluations: Adaptive evaluation examples
    "4-adaptive_evaluations/1-adaptive_evaluation.py": False,  # Requires Trismik + OpenAI
    "4-adaptive_evaluations/2-adaptive_dataset_splits.py": False,  # Requires Trismik + OpenAI
    # 5-upload_results: Result upload examples
    "5-upload_results/1-uploading_score_results.py": False,  # Requires Trismik API key
    "5-upload_results/2-uploading_evaluate_results.py": False,  # Requires Trismik API key
    "5-upload_results/3-uploading_your_results.py": False,  # Requires Trismik API key
    # 6-providers: Cloud provider examples
    "6-providers/aws/batch_example.py": False,  # Requires AWS credentials
    "6-providers/portkey/batch_example.py": False,  # Requires Portkey API key
    "6-providers/portkey/messages_example.py": False,  # Requires Portkey API key
    "6-providers/vertex/batch_example.py": False,  # Requires GCP credentials
    "6-providers/vertex/messages_example.py": False,  # Requires GCP credentials
}

# =============================================================================
# Runner Implementation
# =============================================================================


def run_example(example_path: Path, project_root: Path) -> bool:
    """Run a single example and return True if successful, False otherwise."""
    print(f"\n{'=' * 80}")
    examples_base = project_root / "tutorials" / "examples"
    print(f"Running: {example_path.relative_to(examples_base)}")
    print(f"{'=' * 80}\n")

    try:
        # Set PYTHONPATH to include project root so examples can import tutorials
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            [sys.executable, str(example_path)],
            cwd=project_root,  # Run from project root
            env=env,
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
            print(f"\n✗ {example_path.name} failed with code {result.returncode}")
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
    examples_dir = project_root / "tutorials" / "examples"

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
        print("\nNo examples enabled.")
        print("Edit EXAMPLE_FLAGS in this script to enable examples.")
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
        success = run_example(example_path, project_root)
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
