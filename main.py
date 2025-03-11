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

DEFAULT_GROUP_PREFIX = "Other"


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
    enum_info: Optional[Dict[str, str]] = None

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
        show_enums_key: bool = False,
        enum_definitions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple["D2Diagram", List[Any]]:
        """Create a py-d2 diagram from a list of Tables.

        Args:
            tables: List of Table objects
            group_prefixes: List of prefixes to group tables by
            show_enums_key: Whether to show a key for enum values
            enum_definitions: Dictionary of enum class definitions with all values

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
                DEFAULT_GROUP_PREFIX,
                label="",
                style=D2Style(
                    fill="white",
                    stroke="white",
                ),
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
                    table_prefix_mapping[table.name] = DEFAULT_GROUP_PREFIX
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
                        ref_parts = column.constraint.foreign_key.split(".")
                        if len(ref_parts) != 2:
                            raise ValueError("Invalid foreign key format")

                        ref_table_raw, ref_column = ref_parts

                        # Try to find the actual table name
                        if ref_table_raw in table_name_mapping:
                            ref_table = table_name_mapping[ref_table_raw]
                        else:
                            # Fallback to original name
                            ref_table = ref_table_raw
                            print(
                                f"Note: Using original table name reference: {ref_table}"
                            )

                        # Get prefixes
                        source_prefix = table_prefix_mapping.get(table.name)
                        target_prefix = table_prefix_mapping.get(ref_table)

                        # Create qualified table names
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

                        print(
                            f"Creating foreign key connection: {source_qualified}.{column.name} → {target_qualified}.{ref_column}"
                        )

                        # Create and add the foreign key connection
                        fk = create_foreign_key_connection(
                            source_qualified, column.name, target_qualified, ref_column
                        )
                        connections.append(fk)
                        diagram.add_connection(fk)
                    except ValueError:
                        print(
                            f"Warning: Invalid foreign key format in {table.name}."
                            f"{column.name}: {column.constraint.foreign_key}"
                        )

        # Add enum key if requested
        if show_enums_key:
            # Collect all enums from the tables' columns
            collected_enums: Dict[str, Dict[str, str]] = {}
            enum_classes_to_find: Set[str] = set()

            for table in tables:
                for column in table.columns:
                    if column.enum_info and column.type.lower() in [
                        "integer",
                        "int",
                        "bigint",
                        "smallint",
                    ]:
                        enum_class = column.enum_info.get("enum_class")
                        enum_value = column.enum_info.get("enum_value")

                        if enum_class and enum_value:
                            if enum_class not in collected_enums:
                                collected_enums[enum_class] = {}
                                enum_classes_to_find.add(enum_class)

                            # Add this enum value to the collection with reference to where it's used
                            collected_enums[enum_class][
                                enum_value
                            ] = f"{table.name}.{column.name}"

            # If we found any enums, create a key in the bottom right
            if collected_enums or (enum_definitions and enum_definitions):
                # Create a container for the enum key
                enum_key = D2Shape(
                    "enums_key",
                    label="Enum Key",
                    near="bottom-center",
                    style=D2Style(
                        fill="'#f5f5f5'",
                        stroke="'#333333'",
                        stroke_width=1,
                    ),
                )

                # Process each enum class
                for enum_class, values in collected_enums.items():
                    enum_class_shape = SQLTable(
                        f"{enum_class}",
                        label=enum_class,
                        style=D2Style(
                            stroke_width=1,
                        ),
                    )
                    enum_key.add_shape(enum_class_shape)

                    # Check if we're missing the enum definition
                    if not enum_definitions or enum_class not in enum_definitions:
                        raise ValueError(
                            f"Enum class {enum_class} not found in enum definitions"
                        )

                    # Add complete enum values from the definition
                    all_values = enum_definitions[enum_class]
                    for key, value in all_values.items():
                        # Format the value as a string, escaping any special characters
                        value_str = str(value)
                        # d2 keyword
                        if key == "MULTIPLE":
                            key = "MULTIPLE_"
                        # Make sure the value string is D2 syntax compatible
                        label = value_str.replace(":", "=").replace("\n", " ")

                        value_shape = D2Shape(key, label=label)
                        enum_class_shape.add_shape(value_shape)

                diagram.add_shape(enum_key)

                print(
                    f"Added enum key with {len(collected_enums or enum_definitions or {})} enum classes"
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
        enum_info = None

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
            if keyword.arg == "primary_key" and isinstance(keyword.value, ast.Constant):
                constraint.is_primary_key = bool(keyword.value.value)
            elif keyword.arg == "unique" and isinstance(keyword.value, ast.Constant):
                constraint.is_unique = bool(keyword.value.value)
            elif keyword.arg == "nullable" and isinstance(keyword.value, ast.Constant):
                constraint.is_nullable = bool(keyword.value.value)

            # Extract enum information from default values
            elif keyword.arg == "default":
                # Check for patterns like MyEnum.MyValue.value (for integer columns)
                if column_type.lower() in ["integer", "int", "bigint", "smallint"]:
                    # Try to detect enum pattern: MyEnum.MyValue.value
                    if (
                        isinstance(keyword.value, ast.Attribute)
                        and keyword.value.attr == "value"
                    ):
                        # Case: MyEnum.MyValue.value
                        if (
                            isinstance(keyword.value.value, ast.Attribute)
                            and hasattr(keyword.value.value, "value")
                            and hasattr(keyword.value.value.value, "id")
                        ):
                            enum_class = keyword.value.value.value.id
                            enum_value = keyword.value.value.attr

                            enum_info = {
                                "enum_class": enum_class,
                                "enum_value": enum_value,
                            }
                        # Try another pattern: module.MyEnum.MyValue.value
                        elif isinstance(
                            keyword.value.value, ast.Attribute
                        ) and isinstance(keyword.value.value.value, ast.Attribute):
                            try:
                                # Extract parts from the attribute chain
                                module_or_class = keyword.value.value.value.value.id
                                enum_class = keyword.value.value.value.attr
                                enum_value = keyword.value.value.attr

                                # Use either module_or_class.enum_class or just enum_class as the class name
                                # depending on your preference
                                enum_info = {
                                    "enum_class": f"{enum_class}",
                                    "enum_value": enum_value,
                                }
                            except AttributeError:
                                # Skip if we can't properly extract the values
                                pass

        return Column(
            name=column_name,
            type=column_type,
            type_args=type_args,
            constraint=constraint,
            enum_info=enum_info,
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


def find_enum_definitions(
    directory: str, enum_classes: Set[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Find and parse enum class definitions to extract all key-value pairs.

    Args:
        directory: Path to the directory to search
        enum_classes: Set of enum class names to find

    Returns:
        Dictionary mapping enum class names to their key-value pairs
    """
    enum_definitions: Dict[str, Dict[str, Any]] = {}
    directory_path = Path(directory)

    # Skip if no enum classes to find
    if not enum_classes:
        return enum_definitions

    print(f"Searching for enum class definitions: {', '.join(enum_classes)}")

    class EnumClassVisitor(ast.NodeVisitor):
        """AST visitor to find enum class definitions and extract values."""

        def __init__(self, enum_classes: Set[str]):
            self.enum_classes = enum_classes
            self.found_enums: Dict[str, Dict[str, Any]] = {}
            # Track imported names and their actual module/source
            self.imports: Dict[str, str] = {}

        def visit_Import(self, node: ast.Import) -> None:
            """Track import statements for looking up enums in other modules."""
            for alias in node.names:
                self.imports[alias.asname or alias.name] = alias.name
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            """Track from ... import statements for looking up enums in other modules."""
            if node.module:
                for alias in node.names:
                    # Store as module.name if possible
                    self.imports[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            # Check if this is an Enum class we're looking for
            if node.name in self.enum_classes:
                # Check if it's likely an Enum by examining base classes
                is_enum = False
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_name = base.id
                        # Check direct base class name or imported base
                        if base_name in [
                            "Enum",
                            "IntEnum",
                            "StrEnum",
                        ] or self.imports.get(base_name, "").endswith(
                            ("Enum", "IntEnum", "StrEnum")
                        ):
                            is_enum = True
                            break
                    elif isinstance(base, ast.Attribute) and base.attr in [
                        "Enum",
                        "IntEnum",
                        "StrEnum",
                    ]:
                        is_enum = True
                        break

                if is_enum:
                    # Found an enum class, extract its values
                    enum_values = {}
                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    name = target.id
                                    # Extract value
                                    if isinstance(item.value, ast.Constant):
                                        enum_values[name] = item.value.value
                                    # Handle auto-increment case
                                    elif (
                                        isinstance(item.value, ast.Call)
                                        and isinstance(item.value.func, ast.Name)
                                        and item.value.func.id == "auto"
                                    ):
                                        enum_values[name] = "auto()"
                                    # Handle binary operations like 1 << 0, etc.
                                    elif isinstance(item.value, ast.BinOp):
                                        # Try to extract a string representation of the operation
                                        try:
                                            if isinstance(
                                                item.value.left, ast.Constant
                                            ) and isinstance(
                                                item.value.right, ast.Constant
                                            ):
                                                left_val = item.value.left.value
                                                right_val = item.value.right.value

                                                # Handle common binary operations
                                                if isinstance(
                                                    item.value.op, ast.LShift
                                                ):  # <<
                                                    enum_values[name] = (
                                                        f"{left_val} << {right_val} = {left_val << right_val}"
                                                    )
                                                elif isinstance(
                                                    item.value.op, ast.RShift
                                                ):  # >>
                                                    enum_values[name] = (
                                                        f"{left_val} >> {right_val} = {left_val >> right_val}"
                                                    )
                                                elif isinstance(
                                                    item.value.op, ast.BitOr
                                                ):  # |
                                                    enum_values[name] = (
                                                        f"{left_val} | {right_val} = {left_val | right_val}"
                                                    )
                                                elif isinstance(
                                                    item.value.op, ast.BitAnd
                                                ):  # &
                                                    enum_values[name] = (
                                                        f"{left_val} & {right_val} = {left_val & right_val}"
                                                    )
                                                elif isinstance(
                                                    item.value.op, ast.BitXor
                                                ):  # ^
                                                    enum_values[name] = (
                                                        f"{left_val} ^ {right_val} = {left_val ^ right_val}"
                                                    )
                                                elif isinstance(
                                                    item.value.op, ast.Add
                                                ):  # +
                                                    enum_values[name] = (
                                                        f"{left_val} + {right_val} = {left_val + right_val}"
                                                    )
                                                elif isinstance(
                                                    item.value.op, ast.Sub
                                                ):  # -
                                                    enum_values[name] = (
                                                        f"{left_val} - {right_val} = {left_val - right_val}"
                                                    )
                                                elif isinstance(
                                                    item.value.op, ast.Mult
                                                ):  # *
                                                    enum_values[name] = (
                                                        f"{left_val} * {right_val} = {left_val * right_val}"
                                                    )
                                                else:
                                                    enum_values[name] = "BinOp"
                                            else:
                                                enum_values[name] = "BinOp"
                                        except Exception:
                                            enum_values[name] = "BinOp"
                                    # Handle other common patterns
                                    elif (
                                        isinstance(item.value, ast.Attribute)
                                        and getattr(item.value, "attr", "") == "value"
                                    ):
                                        # Handle references to other enum values like EnumClass.VALUE.value
                                        if (
                                            isinstance(item.value.value, ast.Attribute)
                                            and hasattr(item.value.value, "value")
                                            and hasattr(item.value.value.value, "id")
                                        ):
                                            other_enum = item.value.value.value.id
                                            other_value = item.value.value.attr
                                            enum_values[name] = (
                                                f"{other_enum}.{other_value}.value"
                                            )
                                        else:
                                            enum_values[name] = "Enum.value"

                    if enum_values:
                        self.found_enums[node.name] = enum_values

            # Continue the visit for other nodes
            self.generic_visit(node)

    # Walk through all Python files in the directory
    for root, dirs, files in os.walk(directory_path, topdown=True):
        # Skip virtual environment directories
        dirs[:] = [d for d in dirs if d != "venv" and d != ".venv"]

        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file

                try:
                    with open(file_path, "r") as f:
                        file_content = f.read()

                    try:
                        tree = ast.parse(file_content, filename=str(file_path))

                        # Visit the AST with our visitor
                        visitor = EnumClassVisitor(enum_classes)
                        visitor.visit(tree)

                        # Merge the found enums into our result
                        enum_definitions.update(visitor.found_enums)

                        # If we've found all the enums we're looking for, we can stop
                        if all(
                            enum_class in enum_definitions
                            for enum_class in enum_classes
                        ):
                            break

                    except SyntaxError:
                        print(f"Syntax error in file: {file_path}")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    # Report what we found
    for enum_class in enum_classes:
        if enum_class in enum_definitions:
            print(
                f"Found enum class {enum_class} with {len(enum_definitions[enum_class])} values"
            )
        else:
            print(f"Could not find enum class {enum_class}")

    return enum_definitions


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
        "--enums-key",
        action="store_true",
        default=False,
        help="Add a key to the diagram to show enums values.",
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

    # If enum key is requested, find all enum classes used in columns
    enum_definitions = None
    if args.enums_key:
        # Collect all enum class names from column defaults
        enum_classes_to_find: Set[str] = set()
        for table in tables:
            for column in table.columns:
                if column.enum_info and "enum_class" in column.enum_info:
                    enum_classes_to_find.add(column.enum_info["enum_class"])

        # Find definitions for these enum classes
        if enum_classes_to_find:
            enum_definitions = find_enum_definitions(
                args.directory, enum_classes_to_find
            )

    # Create the diagram
    diagram, connections = Table.create_d2_diagram(
        tables, group_prefixes, args.enums_key, enum_definitions
    )

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
