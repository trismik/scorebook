#!/usr/bin/env python
"""Script to run all examples with toggleable execution.

By default, only examples that don't require external API keys or credentials are run.
Use command-line flags to include additional example categories.

Usage:
    python scripts/run_examples.py                # Run default (safe) examples
    python scripts/run_examples.py --dry-run      # Preview which examples will run
    python scripts/run_examples.py --all          # Run ALL examples
    python scripts/run_examples.py --openai       # Include OpenAI examples
    python scripts/run_examples.py --trismik      # Include Trismik examples
    python scripts/run_examples.py --aws          # Include AWS provider examples
    python scripts/run_examples.py --gcp          # Include GCP/Vertex examples
    python scripts/run_examples.py --portkey      # Include Portkey examples
    python scripts/run_examples.py --providers    # Include all cloud providers (aws, gcp, portkey)

    # Combine flags as needed:
    python scripts/run_examples.py --openai --trismik
    python scripts/run_examples.py --providers --openai
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

# =============================================================================
# Example Configuration with Requirements
# =============================================================================


@dataclass
class Example:
    """Represents an example script with its requirements."""

    path: str
    requires: Set[str]  # Set of requirement tags: "openai", "trismik", "aws", "gcp", "portkey"

    @property
    def is_default(self) -> bool:
        """Returns True if this example has no external requirements."""
        return len(self.requires) == 0


# Define all examples with their requirements
EXAMPLES: List[Example] = [
    # 1-score: Basic scoring examples (no external requirements)
    Example("1-score/1-scoring_model_accuracy.py", set()),
    Example("1-score/2-scoring_model_bleu.py", set()),
    Example("1-score/3-scoring_model_f1.py", set()),
    Example("1-score/4-scoring_model_rouge.py", set()),
    Example("1-score/5-scoring_model_exact_match.py", set()),
    # 2-evaluate: Evaluation examples
    Example("2-evaluate/1-evaluating_local_models.py", set()),
    Example("2-evaluate/2-evaluating_local_models_with_batching.py", set()),
    Example("2-evaluate/3-evaluating_cloud_models.py", {"openai"}),
    Example("2-evaluate/4-evaluating_cloud_models_with_batching.py", {"openai"}),
    Example("2-evaluate/5-hyperparameter_sweeps.py", set()),
    Example("2-evaluate/6-inference_pipelines.py", set()),
    # 3-evaluation_datasets: Dataset loading examples
    Example("3-evaluation_datasets/1-evaluation_datasets_from_files.py", set()),
    Example("3-evaluation_datasets/2-evaluation_datasets_from_huggingface.py", {"openai"}),
    Example(
        "3-evaluation_datasets/3-evaluation_datasets_from_huggingface_with_yaml_configs.py",
        {"openai"},
    ),
    # 4-adaptive_evaluations: Adaptive evaluation examples (require both Trismik and OpenAI)
    Example("4-adaptive_evaluations/1-adaptive_evaluation.py", {"trismik", "openai"}),
    Example("4-adaptive_evaluations/2-adaptive_dataset_splits.py", {"trismik", "openai"}),
    # 5-upload_results: Result upload examples (require Trismik)
    Example("5-upload_results/1-uploading_score_results.py", {"trismik"}),
    Example("5-upload_results/2-uploading_evaluate_results.py", {"trismik"}),
    Example("5-upload_results/3-uploading_your_results.py", {"trismik"}),
    # 6-providers: Cloud provider examples
    Example("6-providers/aws/batch_example.py", {"aws"}),
    Example("6-providers/portkey/batch_example.py", {"portkey"}),
    Example("6-providers/portkey/messages_example.py", {"portkey"}),
    Example("6-providers/vertex/batch_example.py", {"gcp"}),
    Example("6-providers/vertex/messages_example.py", {"gcp"}),
]

# All known requirement tags for validation
ALL_REQUIREMENT_TAGS = {"openai", "trismik", "aws", "gcp", "portkey"}

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


def filter_examples(
    examples: List[Example], enabled_tags: Set[str], run_all: bool
) -> List[Example]:
    """Filter examples based on enabled requirement tags.

    Args:
        examples: List of all examples
        enabled_tags: Set of requirement tags that are enabled (e.g., {"openai", "aws"})
        run_all: If True, return all examples regardless of requirements

    Returns:
        List of examples that should be run
    """
    if run_all:
        return examples

    filtered = []
    for example in examples:
        # Include if: no requirements OR all requirements are in enabled_tags
        if example.is_default or example.requires.issubset(enabled_tags):
            filtered.append(example)
    return filtered


def format_requirements(requires: Set[str]) -> str:
    """Format requirements as a readable string."""
    if not requires:
        return ""
    return f" [requires: {', '.join(sorted(requires))}]"


def main() -> None:
    """Run examples with configurable execution."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run Scorebook examples with toggleable execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     Run default examples (no API keys required)
  %(prog)s --all               Run ALL examples
  %(prog)s --openai            Include examples requiring OpenAI API key
  %(prog)s --trismik           Include examples requiring Trismik API key
  %(prog)s --aws               Include AWS provider examples
  %(prog)s --gcp               Include GCP/Vertex provider examples
  %(prog)s --portkey           Include Portkey provider examples
  %(prog)s --providers         Include all cloud provider examples (aws, gcp, portkey)
  %(prog)s --openai --trismik  Combine multiple flags
  %(prog)s --dry-run --all     Preview all examples without running
        """,
    )

    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which examples will run without executing them",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run ALL examples, including those requiring API keys/credentials",
    )

    # API key options
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Include examples requiring OpenAI API key",
    )
    parser.add_argument(
        "--trismik",
        action="store_true",
        help="Include examples requiring Trismik API key",
    )

    # Cloud provider options
    parser.add_argument(
        "--aws",
        action="store_true",
        help="Include AWS Bedrock provider examples",
    )
    parser.add_argument(
        "--gcp",
        action="store_true",
        help="Include GCP Vertex AI provider examples",
    )
    parser.add_argument(
        "--portkey",
        action="store_true",
        help="Include Portkey provider examples",
    )
    parser.add_argument(
        "--providers",
        action="store_true",
        help="Include ALL cloud provider examples (aws, gcp, portkey)",
    )

    args = parser.parse_args()

    # Build the set of enabled requirement tags
    enabled_tags: Set[str] = set()

    if args.openai:
        enabled_tags.add("openai")
    if args.trismik:
        enabled_tags.add("trismik")
    if args.aws:
        enabled_tags.add("aws")
    if args.gcp:
        enabled_tags.add("gcp")
    if args.portkey:
        enabled_tags.add("portkey")
    if args.providers:
        enabled_tags.update({"aws", "gcp", "portkey"})

    # Find the examples directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    examples_dir = project_root / "tutorials" / "examples"

    if not examples_dir.exists():
        print(f"Error: Examples directory not found at {examples_dir}")
        sys.exit(1)

    # Filter examples based on enabled tags
    filtered_examples = filter_examples(EXAMPLES, enabled_tags, args.all)

    # Validate example paths and separate enabled/skipped
    enabled_examples: List[tuple[str, Path, Set[str]]] = []
    skipped_examples: List[tuple[str, Set[str]]] = []
    missing_examples: List[str] = []

    for example in EXAMPLES:
        example_path = examples_dir / example.path
        if not example_path.exists():
            missing_examples.append(example.path)
            continue

        if example in filtered_examples:
            enabled_examples.append((example.path, example_path, example.requires))
        else:
            skipped_examples.append((example.path, example.requires))

    # Print summary
    print("\n" + "=" * 80)
    print("EXAMPLE RUNNER")
    print("=" * 80)

    # Show active filters
    if args.all:
        print("\nMode: Running ALL examples")
    elif enabled_tags:
        print(f"\nEnabled categories: {', '.join(sorted(enabled_tags))}")
    else:
        print("\nMode: Default (examples with no API requirements)")

    print(f"\nExamples to run: {len(enabled_examples)}")
    for name, _, requires in enabled_examples:
        print(f"  ✓ {name}{format_requirements(requires)}")

    if skipped_examples:
        print(f"\nSkipped examples: {len(skipped_examples)}")
        for name, requires in skipped_examples:
            print(f"  ○ {name}{format_requirements(requires)}")

    if missing_examples:
        print(f"\nWarning - Missing files: {len(missing_examples)}")
        for name in missing_examples:
            print(f"  ✗ {name}")

    if not enabled_examples:
        print("\nNo examples to run.")
        print("Use --all to run all examples, or enable specific categories.")
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
    for example_name, example_path, _ in enabled_examples:
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
