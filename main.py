import argparse
import ast
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from py_d2 import D2Diagram, D2Shape, D2Style
from py_d2.sql_table import SQLConstraint, SQLTable, create_foreign_key_connection

"""
Recursively find all SQLAlchemy models in a given directory/project
and extract their column information with constraints.
"""


@dataclass
class ColumnConstraint:
    """Represents a column constraint in a SQLAlchemy model."""

    is_primary_key: bool = False
    is_unique: bool = False
    is_nullable: bool = True
    foreign_key: Optional[str] = None


@dataclass
class Column:
    """Represents a column in a SQLAlchemy model."""

    name: str
    type: str
    type_args: List[Any] = field(default_factory=list)
    constraint: ColumnConstraint = field(default_factory=ColumnConstraint)

    def __str__(self) -> str:
        constraints = []
        if self.constraint.is_primary_key:
            constraints.append("PK")
        if self.constraint.is_unique:
            constraints.append("UNIQUE")
        if not self.constraint.is_nullable:
            constraints.append("NOT NULL")
        if self.constraint.foreign_key:
            constraints.append(f"FK → {self.constraint.foreign_key}")

        type_str = self.type
        if self.type_args:
            type_str += f"({', '.join(str(arg) for arg in self.type_args)})"

        constraint_str = f" [{', '.join(constraints)}]" if constraints else ""
        return f"{self.name}: {type_str}{constraint_str}"


@dataclass
class Table:
    """Represents a SQLAlchemy model with its columns and file location."""

    name: str
    file_path: str
    columns: List[Column] = field(default_factory=list)

    def __str__(self) -> str:
        columns_str = "\n  ".join(str(col) for col in self.columns)
        return f"Table: {self.name} ({self.file_path})\n  {columns_str}"

    def to_d2_table(self) -> "SQLTable":
        """Convert this Table to a py-d2 SQLTable representation.

        Returns:
            A py-d2 SQLTable instance representing this table
        """

        # Create a SQLTable with the table name
        d2_table = SQLTable(self.name)

        # Add each column with its type and constraints
        for column in self.columns:
            type_str = column.type
            if column.type_args:
                # Format type arguments appropriately for SQL type syntax
                if column.type.lower() in ["varchar", "char", "string"]:
                    type_str += f"({column.type_args[0]})"
                elif column.type.lower() in ["decimal", "numeric"]:
                    if len(column.type_args) >= 2:
                        type_str += f"({column.type_args[0]},{column.type_args[1]})"

            if column.constraint.is_nullable:
                type_str += "?"

            constraints = []
            if column.constraint.is_primary_key:
                constraints.append(SQLConstraint.PRIMARY_KEY)
            if column.constraint.foreign_key:
                constraints.append(SQLConstraint.FOREIGN_KEY)
            if column.constraint.is_unique:
                constraints.append(SQLConstraint.UNIQUE)

            d2_table.add_field(column.name, type_str, constraints)

        return d2_table

    @staticmethod
    def create_d2_diagram(
        tables: List["Table"],
        group_prefixes: List[str] = [],
    ) -> Tuple["D2Diagram", List[Any]]:
        """Create a py-d2 diagram from a list of Tables.

        Args:
            tables: List of Table objects
            group_prefixes: List of prefixes to group tables by

        Returns:
            Tuple containing (D2Diagram, list of foreign key connections)
        """

        # Create a new diagram
        diagram = D2Diagram()

        prefix_containers = {}
        no_prefix_container = None

        # Only create containers if group_prefixes is provided
        if group_prefixes:
            for prefix in group_prefixes:
                print(f"Adding prefix container: {prefix}")
                prefix_container = D2Shape(prefix)
                diagram.add_shape(prefix_container)
                prefix_containers[prefix] = prefix_container

            # Create a container for tables without a prefix
            no_prefix_container = D2Shape(
                "Other",
                style=D2Style(
                    fill="white",
                ),
                near="bottom-center",
            )
            diagram.add_shape(no_prefix_container)

        # Add all tables to the diagram
        d2_tables = {}
        # Create a mapping from both CamelCase and snake_case names to the table objects
        table_name_mapping = {}
        # Track which prefix each table belongs to (if any)
        table_prefix_mapping: Dict[str, Optional[str]] = {}

        for table in tables:
            d2_table = table.to_d2_table()
            d2_tables[table.name] = d2_table

            # Add both CamelCase and snake_case variants to the mapping
            table_name_mapping[table.name] = table.name  # Original name
            # Convert CamelCase to snake_case (e.g., DocumentSet -> document_set)
            snake_case = "".join(
                ["_" + c.lower() if c.isupper() else c for c in table.name]
            ).lstrip("_")
            table_name_mapping[snake_case] = table.name

            # Only check for prefixes if group_prefixes is provided
            table_prefix = None
            if group_prefixes:
                table_prefixes = [
                    prefix for prefix in group_prefixes if table.name.startswith(prefix)
                ]

                if len(table_prefixes) > 1:
                    raise ValueError(
                        f"Table {table.name} has multiple prefixes: {table_prefixes}. "
                        "Please specify a single prefix for each table."
                    )

                # Store the prefix for this table (or None if it doesn't have one)
                table_prefix = table_prefixes[0] if table_prefixes else None

            table_prefix_mapping[table.name] = table_prefix

            # Add the table to the appropriate container or directly to the diagram
            if group_prefixes:
                if table_prefix:
                    prefix_containers[table_prefix].add_shape(d2_table)
                else:
                    # Add tables without a prefix to the no_prefix_container
                    no_prefix_container.add_shape(d2_table)
            else:
                # If no groups are provided, add directly to the diagram
                diagram.add_shape(d2_table)

        # Create foreign key connections
        connections = []
        for table in tables:
            for column in table.columns:
                if column.constraint.foreign_key:
                    # Parse the foreign key reference
                    try:
                        ref_table_raw, ref_column = column.constraint.foreign_key.split(
                            "."
                        )

                        # Try to find the actual table name (handling both CamelCase and snake_case)
                        if ref_table_raw in table_name_mapping:
                            ref_table = table_name_mapping[ref_table_raw]
                        else:
                            # Try to convert CamelCase to snake_case or vice versa
                            # This is a fallback if the direct mapping doesn't work
                            ref_table = ref_table_raw
                            print(
                                f"Note: Using original table name reference: {ref_table}"
                            )

                        # Get the source and target table prefixes (if any)
                        source_prefix = table_prefix_mapping.get(table.name)
                        target_prefix = table_prefix_mapping.get(ref_table)

                        # Create fully qualified table names with prefixes if needed
                        source_qualified = (
                            f"{source_prefix}.{table.name}"
                            if source_prefix
                            else table.name
                        )
                        target_qualified = (
                            f"{target_prefix}.{ref_table}"
                            if target_prefix
                            else ref_table
                        )

                        fk = create_foreign_key_connection(
                            source_qualified, column.name, target_qualified, ref_column
                        )
                        connections.append(fk)
                        diagram.add_connection(fk)
                    except ValueError:
                        print(
                            f"Warning: Invalid foreign key format in {table.name}.{column.name}: {column.constraint.foreign_key}"
                        )

        return diagram, connections


class SQLAlchemyModelVisitor(ast.NodeVisitor):
    """
    AST visitor that identifies SQLAlchemy model classes and their columns.
    """

    def __init__(self) -> None:
        self.models: Dict[str, List[Column]] = {}
        # Common SQLAlchemy base classes to look for
        self.sqlalchemy_bases: Set[str] = {
            "Base",
            "db.Model",
            "Model",
            "DeclarativeBase",
        }
        self.current_class: str = ""

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions and check if they're SQLAlchemy models."""
        prev_class = self.current_class

        for base in node.bases:
            base_name = self._get_base_name(base)
            if base_name in self.sqlalchemy_bases:
                self.current_class = node.name
                self.models[node.name] = []
                break

        # Continue traversing the AST
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignments to find column definitions."""
        if not self.current_class:
            return

        # Check if this is a column assignment
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            column_name = node.targets[0].id

            # Check if the value is a Column
            if self._is_column_definition(node.value):
                column = self._extract_column_info(column_name, node.value)
                if column:
                    self.models[self.current_class].append(column)

        self.generic_visit(node)

    def _is_column_definition(self, node: ast.expr) -> bool:
        """Check if the node is a Column definition."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "Column":
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "Column":
                return True
        return False

    def _extract_column_info(
        self, column_name: str, node: ast.Call
    ) -> Optional[Column]:
        """Extract column information from a Column definition."""
        column_type = "Unknown"
        type_args: List[Any] = []
        constraint = ColumnConstraint()

        # Extract column type and attributes
        if node.args:
            # First arg is usually the column type
            first_arg = node.args[0]

            # Handle simple type names (e.g., String)
            if isinstance(first_arg, ast.Name):
                column_type = first_arg.id
            # Handle attribute access types (e.g., db.String)
            elif isinstance(first_arg, ast.Attribute):
                column_type = first_arg.attr
            # Handle types with parameters like String(50)
            elif isinstance(first_arg, ast.Call):
                if isinstance(first_arg.func, ast.Name):
                    column_type = first_arg.func.id
                    if first_arg.args:
                        for arg in first_arg.args:
                            if isinstance(arg, ast.Constant):
                                type_args.append(arg.value)
                # Handle types with parameters like db.String(50)
                elif isinstance(first_arg.func, ast.Attribute):
                    column_type = first_arg.func.attr
                    if first_arg.args:
                        for arg in first_arg.args:
                            if isinstance(arg, ast.Constant):
                                type_args.append(arg.value)

            # Look for ForeignKey in the arguments
            for arg in node.args:
                if isinstance(arg, ast.Call) and (
                    (isinstance(arg.func, ast.Name) and arg.func.id == "ForeignKey")
                    or (
                        isinstance(arg.func, ast.Attribute)
                        and arg.func.attr == "ForeignKey"
                    )
                ):
                    if arg.args and isinstance(arg.args[0], ast.Constant):
                        constraint.foreign_key = arg.args[0].value

        # Extract column attributes (primary_key, nullable, etc.)
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Constant):
                value = keyword.value.value
                if keyword.arg == "primary_key":
                    constraint.is_primary_key = bool(value)
                elif keyword.arg == "unique":
                    constraint.is_unique = bool(value)
                elif keyword.arg == "nullable":
                    constraint.is_nullable = bool(value)

        return Column(
            name=column_name,
            type=column_type,
            type_args=type_args,
            constraint=constraint,
        )

    def _get_base_name(self, node: ast.expr) -> str:
        """Extract the base class name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
        return ""


def find_sqlalchemy_models(directory: str) -> List[Table]:
    """
    Find all SQLAlchemy models in a given directory/project and extract column information.

    Args:
        directory: Path to the directory to search

    Returns:
        List of Table objects representing SQLAlchemy models
    """
    tables: List[Table] = []
    directory_path = Path(directory)

    # Walk through all Python files in the directory
    for root, dirs, files in os.walk(directory_path, topdown=True):
        # Skip virtual environment directories
        dirs[:] = [d for d in dirs if d != "venv" and d != ".venv"]

        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                try:
                    # Parse the Python file
                    with open(file_path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read())

                    # Visit the AST to find models and their columns
                    visitor = SQLAlchemyModelVisitor()
                    visitor.visit(tree)

                    # Add found models with their columns and file path
                    for model_name, columns in visitor.models.items():
                        tables.append(
                            Table(
                                name=model_name,
                                file_path=str(file_path.relative_to(directory_path)),
                                columns=columns,
                            )
                        )

                except (SyntaxError, UnicodeDecodeError, PermissionError) as e:
                    print(f"Error parsing {file_path}: {e}")

    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract SQLAlchemy models from a Python project and generate a schema diagram.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -d ./my_project                # Scan a specific directory
  python main.py --groups User,Post,Comment     # Group tables by prefixes
  python main.py -o my_schema                   # Custom output filename
  python main.py --skip-svg                     # Generate only D2 file, no SVG
        """,
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=".",
        help="Directory to scan for SQLAlchemy models (default: current directory)",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="",
        help="Groups the tables by shared prefixes. Comma-separated list.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="sqlalchemy_schema",
        help="Output filename (without extension) for the diagram (default: sqlalchemy_schema)",
    )
    parser.add_argument(
        "--skip-svg", action="store_true", help="Skip generating SVG diagram"
    )

    # Parse arguments
    args = parser.parse_args()

    # Find SQLAlchemy models in the specified directory
    tables: List[Table] = find_sqlalchemy_models(args.directory)

    # Print tables with proper formatting
    for table in tables:
        print(table)
        print()

    # Convert to a py-d2 diagram
    group_prefixes = args.groups.split(",") if args.groups else []
    diagram, connections = Table.create_d2_diagram(tables, group_prefixes)

    # Write the diagram to a file
    file_name = args.output
    with open(f"{file_name}.d2", "w") as f:
        f.write("direction: right\n")
        f.write(str(diagram))
    print(f"D2 diagram written to: {os.path.abspath(f'{file_name}.d2')}")

    # Generate SVG (if d2 is installed and not skipped)
    if not args.skip_svg:
        try:
            subprocess.run(
                ["d2", "--layout", "elk", f"{file_name}.d2", f"{file_name}.svg"],
                check=True,
            )
            print(f"SVG diagram generated: {os.path.abspath(f'{file_name}.svg')}")
        except Exception as e:
            print(f"Error generating SVG: {e}")
