# sqlalchemy-d2

A Python library that converts SQLAlchemy Models into a D2 file for visualization and ERD generation.

## Installation

1. Make sure you have [D2](https://d2lang.com/tour/install) installed on your system.
2. Clone the repository:

```bash
git clone https://github.com/Anatomy-Financial/sqlalchemy-d2.git
cd sqlalchemy-d2
```

```bash
uv venv
uv pip install -r requirements
```

## Usage

```bash
uv run main.py --directory /path/to/your/models
```

Full usage:

```bash
uv run main.py  --help
```