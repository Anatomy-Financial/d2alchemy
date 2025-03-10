import argparse
import importlib.util
import os
import sys
from typing import Any, List, Optional, Tuple

from sqlalchemy_parser import (
    create_d2_diagram_from_sqlalchemy_models,
    find_sqlalchemy_models_in_directory,
)

SKIP_DIRECTORIES = ["venv", "env", "site-packages", "dist-packages"]


def import_module_from_path(module_path: str) -> Tuple[Optional[Any], str]:
    """
    Import a module from a file path

    Args:
        module_path: Path to the Python module file

    Returns:
        A tuple of (module object, module name) or (None, error message)
    """
    try:
        module_name = os.path.basename(module_path)
        if module_name.endswith(".py"):
            module_name = module_name[:-3]

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if not spec or not spec.loader:
            return None, f"Could not load module spec from {module_path}"

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return module, module_name
    except Exception as e:
        return None, str(e)


def main() -> None:
    """Main function to run the script"""
    parser = argparse.ArgumentParser(
        description="Generate D2 diagrams from SQLAlchemy models"
    )
    parser.add_argument(
        "--output",
        default="sqlalchemy_erd",
        help="Output file name without extension (default: sqlalchemy_erd)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--path",
        default=os.getcwd(),
        help="Path to search for SQLAlchemy models (default: current directory)",
    )

    args = parser.parse_args()
    verbose = args.verbose
    path = args.path

    print("Starting SQLAlchemy to D2 diagram generation")
    print(f"Using path: {path}")

    # Check if path exists
    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist")
        sys.exit(1)

    if not os.path.isdir(path):
        print(f"Error: Path '{path}' is not a directory")
        sys.exit(1)

    # Find all SQLAlchemy models in the directory
    print(f"Searching for SQLAlchemy models in: {path}")
    all_models = find_sqlalchemy_models_in_directory(path, verbose)

    if not all_models:
        print(
            "\nNo SQLAlchemy models found. Make sure the project's dependencies are installed."
        )
        sys.exit(0)

    print(f"Creating D2 diagram from {len(all_models)} models")
    # Create D2 diagram
    diagram = create_d2_diagram_from_sqlalchemy_models(all_models)

    # Write diagram to file
    output_file = f"{args.output}.d2"
    with open(output_file, "w") as f:
        f.write(str(diagram))

    print(f"D2 diagram file created: {output_file}")

    # Print instructions on how to generate the diagram
    print("\nTo generate the diagram:")
    print("------------------------")
    print(f"  d2 --layout elk {output_file} {args.output}.svg")
    print("\nIf D2 is not installed:")
    print("----------------------")
    print("  Visit: https://d2lang.com/tour/install")
    print("")


if __name__ == "__main__":
    main()
