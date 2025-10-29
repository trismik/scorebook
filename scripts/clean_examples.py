#!/usr/bin/env python
"""Clean up logs and results directories from tutorial examples.

This script removes all logs/ and results/ directories from tutorials/examples
to clean up after running examples.

Usage:
    python scripts/clean_examples.py           # Clean all logs and results
    python scripts/clean_examples.py --dry-run # Preview what will be deleted
"""

import argparse
import shutil
from pathlib import Path


def find_cleanup_dirs(examples_dir: Path) -> tuple[list[Path], list[Path]]:
    """Find all logs and results directories in examples.

    Args:
        examples_dir: Path to the examples directory

    Returns:
        Tuple of (logs_dirs, results_dirs) lists
    """
    logs_dirs = list(examples_dir.rglob("logs"))
    results_dirs = list(examples_dir.rglob("results"))

    return logs_dirs, results_dirs


def remove_directory(dir_path: Path) -> bool:
    """Remove a directory and return True if successful.

    Args:
        dir_path: Path to directory to remove

    Returns:
        True if removed successfully, False otherwise
    """
    try:
        shutil.rmtree(dir_path)
        return True
    except Exception as e:
        print(f"  Error removing {dir_path}: {e}")
        return False


def main() -> None:
    """Clean up logs and results directories from examples."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Clean up logs and results directories from tutorial examples"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what will be deleted without actually deleting",
    )
    args = parser.parse_args()

    # Find the examples directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    examples_dir = project_root / "tutorials" / "examples"

    if not examples_dir.exists():
        print(f"Error: Examples directory not found at {examples_dir}")
        return

    # Find all logs and results directories
    logs_dirs, results_dirs = find_cleanup_dirs(examples_dir)

    total_dirs = len(logs_dirs) + len(results_dirs)

    if total_dirs == 0:
        print("No logs or results directories found. Nothing to clean.")
        return

    # Print summary
    print("\n" + "=" * 80)
    print("EXAMPLE CLEANUP")
    print("=" * 80)
    print(f"\nFound {total_dirs} directories to clean:")
    print(f"  • {len(logs_dirs)} logs/ directories")
    print(f"  • {len(results_dirs)} results/ directories")

    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN MODE - No files will be deleted")
        print("=" * 80)

        if logs_dirs:
            print("\nLogs directories that would be removed:")
            for dir_path in sorted(logs_dirs):
                rel_path = dir_path.relative_to(examples_dir)
                print(f"  • {rel_path}")

        if results_dirs:
            print("\nResults directories that would be removed:")
            for dir_path in sorted(results_dirs):
                rel_path = dir_path.relative_to(examples_dir)
                print(f"  • {rel_path}")

        print("\nTo actually delete these directories, run without --dry-run flag.")
        return

    # Confirm deletion
    print("\n" + "=" * 80)
    print("WARNING: This will permanently delete all logs and results!")
    print("=" * 80)
    response = input("\nProceed with deletion? [y/N]: ")

    if response.lower() != "y":
        print("Cleanup cancelled.")
        return

    # Delete directories
    print("\n" + "=" * 80)
    print("CLEANING UP")
    print("=" * 80)

    success_count = 0
    fail_count = 0

    if logs_dirs:
        print("\nRemoving logs directories...")
        for dir_path in logs_dirs:
            rel_path = dir_path.relative_to(examples_dir)
            if remove_directory(dir_path):
                print(f"  ✓ Removed {rel_path}")
                success_count += 1
            else:
                fail_count += 1

    if results_dirs:
        print("\nRemoving results directories...")
        for dir_path in results_dirs:
            rel_path = dir_path.relative_to(examples_dir)
            if remove_directory(dir_path):
                print(f"  ✓ Removed {rel_path}")
                success_count += 1
            else:
                fail_count += 1

    # Print final summary
    print("\n" + "=" * 80)
    print("CLEANUP COMPLETE")
    print("=" * 80)
    print(f"\nSuccessfully removed: {success_count} directories")
    if fail_count > 0:
        print(f"Failed to remove: {fail_count} directories")


if __name__ == "__main__":
    main()
