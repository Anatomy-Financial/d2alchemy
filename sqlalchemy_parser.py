import importlib
import inspect
import os
import sys
from typing import Any, List, Optional, Set, Tuple, Type

from py_d2.diagram import D2Diagram
from py_d2.sql_table import SQLConstraint, SQLTable, create_foreign_key_connection
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.declarative import DeclarativeMeta


def get_column_type(column: Any) -> str:
    """
    Get the SQL type of a SQLAlchemy column

    Args:
        column: A SQLAlchemy Column object

    Returns:
        A string representation of the column type
    """
    type_obj = column.type
    return type_obj.__class__.__name__.lower()


def get_column_constraint(column: Any) -> Optional[SQLConstraint]:
    """
    Get the constraint of a SQLAlchemy column

    Args:
        column: A SQLAlchemy Column object

    Returns:
        A SQLConstraint enum value or None
    """
    if column.primary_key:
        return SQLConstraint.PRIMARY_KEY
    elif column.foreign_keys:
        return SQLConstraint.FOREIGN_KEY
    elif column.unique:
        return SQLConstraint.UNIQUE
    return None


def extract_foreign_keys(model_class: Type[Any]) -> List[Tuple[str, str, str, str]]:
    """
    Extract foreign key relationships from a SQLAlchemy model

    Args:
        model_class: A SQLAlchemy model class

    Returns:
        A list of tuples (source_table, source_field, target_table, target_field)
    """
    foreign_keys = []
    table_name = model_class.__tablename__

    # Get SQLAlchemy inspector for the model
    mapper = sa_inspect(model_class)

    # Extract foreign keys from columns
    for column_name, column_prop in mapper.columns.items():
        for fk in column_prop.foreign_keys:
            target_table, target_field = fk.target_fullname.split(".")
            foreign_keys.append((table_name, column_name, target_table, target_field))

    return foreign_keys


def sqlalchemy_model_to_sql_table(model_class: Type[Any]) -> SQLTable:
    """
    Convert a SQLAlchemy model to a SQLTable object

    Args:
        model_class: A SQLAlchemy model class

    Returns:
        A SQLTable object
    """
    table_name = model_class.__tablename__
    sql_table = SQLTable(table_name)

    # Get SQLAlchemy inspector for the model
    mapper = sa_inspect(model_class)

    # Add columns to the SQL table
    for column_name, column_prop in mapper.columns.items():
        column_type = get_column_type(column_prop)
        constraint = get_column_constraint(column_prop)
        sql_table.add_field(column_name, column_type, constraint)

    return sql_table


def is_sqlalchemy_model(obj: Any) -> bool:
    """
    Check if an object is a SQLAlchemy model class

    Args:
        obj: Any Python object

    Returns:
        True if the object is a SQLAlchemy model class, False otherwise
    """
    return (
        inspect.isclass(obj)
        and isinstance(obj, DeclarativeMeta)
        and hasattr(obj, "__tablename__")
    )


def find_sqlalchemy_models_in_directory(directory: str, verbose: bool = False) -> List[Type[Any]]:
    """
    Find all SQLAlchemy models in all Python files in a directory and its subdirectories

    Args:
        directory: The directory to search
        verbose: Whether to print verbose debug information

    Returns:
        A list of SQLAlchemy model classes
    """
    models = []
    visited_files = set()
    
    # Directories to skip
    SKIP_DIRECTORIES = [
        "venv", ".venv", "env", "site-packages", "dist-packages", 
        "__pycache__", ".git", ".github", "tests", "testing"
    ]
    
    if verbose:
        print(f"Searching for SQLAlchemy models in directory: {directory}")
    
    # Ensure the directory is in the Python path
    directory_abs = os.path.abspath(directory)
    if directory_abs not in sys.path:
        sys.path.insert(0, directory_abs)
        if verbose:
            print(f"Added {directory_abs} to Python path")
    
    # Find all Python files in the directory
    for root, dirs, files in os.walk(directory):
        # Skip directories that match our skip patterns
        dirs[:] = [d for d in dirs if not any(skip in d for skip in SKIP_DIRECTORIES)]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Skip files in directories we want to ignore
                if any(skip in file_path for skip in SKIP_DIRECTORIES):
                    if verbose:
                        print(f"Skipping file in excluded directory: {file_path}")
                    continue
                
                if file_path in visited_files:
                    continue
                
                visited_files.add(file_path)
                
                # Convert file path to module path
                rel_path = os.path.relpath(file_path, directory_abs)
                module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
                
                if verbose:
                    print(f"Processing file: {file_path} as module: {module_name}")
                
                try:
                    # Try to import the module
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec is None or spec.loader is None:
                        if verbose:
                            print(f"Could not create spec for {file_path}")
                        continue
                        
                    module = importlib.util.module_from_spec(spec)
                    
                    # Add the module to sys.modules to handle imports within the module
                    sys.modules[module_name] = module
                    
                    # Use a timeout to prevent hanging on problematic modules
                    try:
                        spec.loader.exec_module(module)
                    except Exception as e:
                        if verbose:
                            print(f"Error executing module {file_path}: {e}")
                        continue
                    
                    # Find SQLAlchemy models in the module
                    for name, obj in inspect.getmembers(module):
                        if verbose and inspect.isclass(obj):
                            # Print details about each class to help diagnose why it's not being recognized
                            is_declarative = isinstance(obj, DeclarativeMeta)
                            has_tablename = hasattr(obj, "__tablename__")
                            print(
                                f"  Class: {name}, DeclarativeMeta: {is_declarative}, has_tablename: {has_tablename}"
                            )

                        if is_sqlalchemy_model(obj):
                            if verbose:
                                print(f"Found SQLAlchemy model: {name} in {file_path}")
                            models.append(obj)
                            
                except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError, KeyError) as e:
                    if verbose:
                        print(f"Error importing {file_path}: {e}")
                    continue
    
    if verbose:
        print(f"Found {len(models)} SQLAlchemy models in {len(visited_files)} files")
        if len(models) == 0:
            print("No SQLAlchemy models found. Make sure the directory contains SQLAlchemy models.")
    
    return models


def create_d2_diagram_from_sqlalchemy_models(models: List[Type[Any]]) -> D2Diagram:
    """
    Create a D2 diagram from a list of SQLAlchemy models

    Args:
        models: A list of SQLAlchemy model classes

    Returns:
        A D2Diagram object
    """
    diagram = D2Diagram()
    processed_tables: Set[str] = set()
    foreign_keys = []

    # Create SQL tables
    for model in models:
        if not hasattr(model, "__tablename__"):
            continue

        if model.__tablename__ in processed_tables:
            continue

        sql_table = sqlalchemy_model_to_sql_table(model)
        diagram.add_shape(sql_table)
        processed_tables.add(model.__tablename__)

        # Collect foreign keys
        foreign_keys.extend(extract_foreign_keys(model))

    # Create connections
    for source_table, source_field, target_table, target_field in foreign_keys:
        connection = create_foreign_key_connection(
            source_table, source_field, target_table, target_field
        )
        diagram.add_connection(connection)

    return diagram