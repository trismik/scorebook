#!/usr/bin/env python3
"""Generate pydoc-markdown documentation for the scorebook package.

Creates individual markdown files for each module with Docusaurus-style front matter.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_python_modules(src_path: str = "src/scorebook") -> List[str]:
    """Find all Python modules in the scorebook package."""
    modules = []
    src_dir = Path(src_path)

    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            # For __init__.py files, use the parent directory as module name
            if py_file.parent != src_dir:  # Skip root __init__.py
                rel_path = py_file.parent.relative_to(src_dir.parent)
                module_name = str(rel_path).replace("/", ".")
                modules.append(module_name)
        else:
            # For regular .py files, use the file path as module name
            rel_path = py_file.relative_to(src_dir.parent)
            module_name = str(rel_path.with_suffix("")).replace("/", ".")
            modules.append(module_name)

    return sorted(set(modules))


def add_docusaurus_frontmatter(content: str, title: str, id_slug: str) -> str:
    """Add Docusaurus front matter to markdown content."""
    frontmatter = f"""---
id: {id_slug}
title: {title}
sidebar_label: {title.split('.')[-1]}
---

"""
    return frontmatter + content


def generate_module_docs(
    module_name: str, output_dir: Path
) -> Tuple[Optional[Path], Optional[str]]:
    """Generate documentation for a single module with Docusaurus front matter."""
    config = {
        "loaders": [{"type": "python", "search_path": ["src"], "modules": [module_name]}],
        "processors": [
            {"type": "filter", "skip_empty_modules": True},
            {"type": "smart"},
            {"type": "crossref"},
        ],
        "renderer": {"type": "markdown"},
    }

    # Create directory structure based on module path
    # Remove 'scorebook.' prefix and create path
    relative_module = module_name.replace("scorebook.", "")
    module_parts = relative_module.split(".")

    # Create nested directory structure
    current_dir = output_dir
    for part in module_parts[:-1]:  # All parts except the last (filename)
        current_dir = current_dir / part
        current_dir.mkdir(exist_ok=True)

    # Use the last part as filename (or 'index' for package __init__.py files)
    filename = f"{module_parts[-1]}.md"
    output_file = current_dir / filename

    # Create clean ID for Docusaurus (still use underscores for uniqueness)
    clean_id = module_name.replace(".", "_")

    try:
        # Run pydoc-markdown with the config
        result = subprocess.run(
            ["poetry", "run", "pydoc-markdown"],
            input=json.dumps(config),
            text=True,
            capture_output=True,
            check=True,
        )

        # Add Docusaurus front matter to the content
        content_with_frontmatter = add_docusaurus_frontmatter(result.stdout, module_name, clean_id)

        # Write output to file
        with open(output_file, "w") as f:
            f.write(content_with_frontmatter)

        print(f"✓ Generated: {output_file}")
        return output_file, clean_id

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate docs for {module_name}")
        if e.stderr:
            print(f"  Error: {e.stderr}")
        return None, None


def generate_sidebar_config(successful_docs: List[str], output_dir: Path) -> None:
    """Generate a Docusaurus sidebar configuration with nested structure."""

    def build_nested_sidebar(docs_list: List[str]) -> List[Dict[str, Any]]:
        """Build nested sidebar structure from flat list of module IDs."""
        sidebar_items: List[Any] = []
        categories: Dict[str, List[str]] = {}

        for doc_id in docs_list:
            # Remove scorebook_ prefix and split
            clean_id = doc_id.replace("scorebook_", "")
            parts = clean_id.split("_")

            if len(parts) == 1:
                # Top-level module
                sidebar_items.append(doc_id)
            else:
                # Nested module - group by first part
                category = parts[0]
                if category not in categories:
                    categories[category] = []
                categories[category].append(doc_id)

        # Add categorized items
        for category, items in sorted(categories.items()):
            sidebar_items.append(
                {"type": "category", "label": category.title(), "items": sorted(items)}
            )

        return sidebar_items

    sidebar_config = {
        "api": [
            {
                "type": "category",
                "label": "API Reference",
                "items": build_nested_sidebar(successful_docs),
            }
        ]
    }

    sidebar_file = output_dir / "sidebar.json"
    with open(sidebar_file, "w") as f:
        json.dump(sidebar_config, f, indent=2)

    print(f"✓ Generated sidebar: {sidebar_file}")


def main() -> None:
    """Generate all module documentation."""
    # Clean up old documentation
    output_dir = Path("docs/pydocs")
    if output_dir.exists():
        print("🧹 Cleaning up old documentation...")
        shutil.rmtree(output_dir)

    # Create fresh output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Finding Python modules in scorebook...")
    modules = find_python_modules()

    if not modules:
        print("❌ No modules found!")
        return

    print(f"📝 Found {len(modules)} modules")
    print(f"📚 Generating documentation in {output_dir}...")

    success_count = 0
    successful_docs = []

    for module in modules:
        output_file, clean_id = generate_module_docs(module, output_dir)
        if output_file and clean_id:
            success_count += 1
            successful_docs.append(clean_id)

    # Generate sidebar configuration
    if success_count > 0:
        generate_sidebar_config(successful_docs, output_dir)

    print(f"\n✅ Generated documentation for {success_count}/{len(modules)} modules")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print("💡 Files maintain source directory structure with Docusaurus front matter")


if __name__ == "__main__":
    main()
