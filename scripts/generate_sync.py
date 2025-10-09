#!/usr/bin/env python3
"""Script to run unasync transformation based on pyproject.toml config."""

from pathlib import Path

import tomlkit
from unasync import Rule, unasync_files


def main() -> None:
    """Run unasync transformation."""
    # Load configuration from pyproject.toml
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "r") as f:
        config = tomlkit.load(f)

    unasync_config = config.get("tool", {}).get("unasync", {})
    rules_list = unasync_config.get("rules", [])

    if not rules_list:
        print("No unasync rules found in pyproject.toml")
        return

    # Process each rule
    for rule_config in rules_list:
        fromdir = rule_config.get("fromdir", "")
        todir = rule_config.get("todir", "")
        replacements = rule_config.get("replacements", {})
        exclude = rule_config.get("exclude", [])

        print(f"Transforming {fromdir} -> {todir}")

        # Create Rule object
        rule = Rule(
            fromdir=fromdir,
            todir=todir,
            additional_replacements=replacements,
        )

        # Find all Python files in fromdir
        from_path = Path(fromdir)
        if not from_path.exists():
            print(f"  Source directory does not exist: {fromdir}")
            continue

        all_py_files = [str(f) for f in from_path.rglob("*.py")]

        # Filter out excluded files
        py_files = []
        for file in all_py_files:
            file_path = Path(file)
            should_exclude = False
            for pattern in exclude:
                # Convert glob pattern to pathlib matching
                clean_pattern = pattern.strip("**/")
                if file_path.name == clean_pattern or file_path.match(pattern):
                    should_exclude = True
                    print(f"  Excluding: {file}")
                    break
            if not should_exclude:
                py_files.append(file)

        if not py_files:
            print(f"  No Python files found in {fromdir}")
            continue

        # Ensure target directory exists
        Path(todir).mkdir(parents=True, exist_ok=True)

        # Run transformation
        unasync_files(py_files, [rule])
        print(f"✅ Transformation complete: {len(py_files)} files -> {todir}")

        # Rename generated files to remove _async suffix
        target_path = Path(todir)
        generated_files = list(target_path.glob("*.py"))

        for gen_file in generated_files:
            if "_async" in gen_file.name:
                # Rename file to remove _async suffix (just remove it, don't replace with _sync)
                new_name = gen_file.name.replace("_async", "")
                new_path = gen_file.parent / new_name
                gen_file.rename(new_path)

                # Post-process: fix imports that need contextlib.nullcontext
                content = new_path.read_text()

                # Fix nullcontext import
                content = content.replace(
                    "from scorebook.utils import nullcontext, evaluation_progress",
                    "from contextlib import nullcontext\n"
                    "from scorebook.utils import evaluation_progress",
                )
                content = content.replace(
                    "from scorebook.utils import evaluation_progress, nullcontext",
                    "from contextlib import nullcontext\n"
                    "from scorebook.utils import evaluation_progress",
                )

                # Fix asyncio.gather to sequential execution
                content = content.replace(
                    "    run_results = asyncio.gather(*[worker(run) for run in runs])",
                    "    run_results = [worker(run) for run in runs]",
                )

                # Fix docstring
                content = content.replace(
                    '    """Run evaluation in parallel."""',
                    '    """Run evaluation sequentially."""',
                )

                # Fix comment
                content = content.replace(
                    "    # Execute all runs concurrently",
                    "    # Execute all runs sequentially",
                )

                # Remove unused asyncio import
                lines = content.split("\n")
                filtered_lines = [line for line in lines if line.strip() != "import asyncio"]
                content = "\n".join(filtered_lines)
                new_path.write_text(content)

                print(f"   Generated: {new_path} (renamed from {gen_file.name})")
            else:
                print(f"   Generated: {gen_file}")


if __name__ == "__main__":
    main()
