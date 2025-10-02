#!/usr/bin/env python3
"""Script to run unasync transformation based on pyproject.toml config."""

from pathlib import Path

import tomlkit
from unasync import Rule, unasync_files


def main() -> None:
    """Run unasync transformation."""
    # Load configuration from pyproject.toml
    pyproject_path = Path(__file__).parent / "pyproject.toml"
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
                # Rename file to remove _async suffix
                new_name = gen_file.name.replace("_async", "_sync")
                new_path = gen_file.parent / new_name
                gen_file.rename(new_path)

                # Post-process the renamed file to fix remaining async patterns
                content = new_path.read_text()
                content = content.replace(
                    "asyncio.gather(*[worker(run) for run in runs])",
                    "[worker(run) for run in runs]",
                )
                content = content.replace("import asyncio", "")
                # Fix duplicate TrismikClient imports
                content = content.replace(
                    "from trismik import TrismikClient, TrismikClient",
                    "from trismik import TrismikClient",
                )
                new_path.write_text(content)

                print(f"   Generated: {new_path} (renamed from {gen_file.name})")
            else:
                print(f"   Generated: {gen_file}")


if __name__ == "__main__":
    main()
