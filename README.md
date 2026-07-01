# NeuroAgent

An intelligent agent platform that combines LangGraph-based reasoning with machine learning pipelines to deliver adaptive, data-driven decision support.

> **Note:** Always activate `venv/` before running anything:
> ```bash
> # Windows
> venv\Scripts\activate
>
> # macOS / Linux
> source venv/bin/activate
> ```

## Project Structure

```
NeuroAgent/
├── config/           # Configuration files (YAML)
├── data/
│   ├── raw/          # Raw input data (not tracked)
│   └── processed/    # Cleaned/processed data (not tracked)
├── src/              # Core source modules
├── platform_core/    # Platform infrastructure code
├── agent/            # LangGraph agent definitions
├── tracking/         # Experiment and metric tracking
└── tests/            # Pytest test suite
```

## Getting Started

```bash
# 1. Clone the repo
git clone <repo-url>
cd NeuroAgent

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests
pytest --collect-only
```
