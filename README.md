# sqlalchemy-d2

A Python library that converts SQLAlchemy Models into a D2 file for visualization and ERD generation.

## Installation

1. Make sure you have [D2](https://d2lang.com/tour/install) installed on your system.
2. Clone the repository:

```bash
git clone https://github.com/yourusername/sqlalchemy-d2.git
cd sqlalchemy-d2
```

## Usage

### Direct Script Execution

You can run the script directly without installation:

```bash
# Using the run.py wrapper script
./run.py --path /path/to/your/models --output my_erd --format svg

# Or with uv
uv run run.py --path /path/to/your/models --output my_erd --format svg
```

#### Options

- `--path`: Path to the Python module or directory containing SQLAlchemy models (default: current directory)
- `--output`: Output file name without extension (default: sqlalchemy_erd)
- `--layout`: D2 layout engine (default: elk)
- `--format`: Output format (svg, png, pdf) (default: svg)

### Python API

```python
from sqlalchemy_parser import find_sqlalchemy_models, create_d2_diagram_from_sqlalchemy_models

# Find SQLAlchemy models in a module
models = find_sqlalchemy_models('your_module_name')

# Create a D2 diagram
diagram = create_d2_diagram_from_sqlalchemy_models(models)

# Write the diagram to a file
with open("my_erd.d2", "w") as f:
    f.write(str(diagram))
```

## Example

Create a file with SQLAlchemy models (see example_models.py) and run:

```bash
./run.py --path example_models.py --output my_erd
```

## Requirements

- Python 3.7+
- SQLAlchemy 2.0+
- D2 (for rendering diagrams)

## Development

To set up the development environment:

```bash
# Using uv
uv pip install -e .
```