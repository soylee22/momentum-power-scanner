set shell := ["bash", "-cu"]

default:
    @just --list

# Set up local venv
install:
    uv venv
    uv pip install -e ".[dev]"

# Full weekly run end-to-end
scan:
    uv run python -m momentum.scanner

# Dry run (no email send)
scan-dry:
    DIGEST_DRY_RUN=1 uv run python -m momentum.scanner

# Just refresh the universe cache
universe:
    uv run python -m momentum.universe

# Render dashboard from latest snapshot
dashboard:
    uv run python -m momentum.dashboard

# Build digest HTML from latest snapshot (no send)
digest-preview:
    DIGEST_DRY_RUN=1 uv run python -m momentum.digest

# Run tests
test:
    uv run pytest tests/ -v

# Tail logs
clean-cache:
    rm -rf data/cache/*.parquet
