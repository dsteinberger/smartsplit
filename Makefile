UV := uv run --extra dev

.PHONY: help install test lint format check run watch clean docker-build docker-up docker-down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (dev)
	uv pip install -e ".[dev]"

test: ## Run tests
	$(UV) pytest tests/ -v

test-quick: ## Run tests (quiet, no verbose)
	$(UV) pytest tests/ -q

lint: ## Lint with ruff
	$(UV) ruff check smartsplit/ tests/ scripts/

format: ## Format with ruff
	$(UV) ruff format smartsplit/ tests/ scripts/

check: lint ## Lint + format check + tests
	$(UV) ruff format smartsplit/ tests/ scripts/ --check
	$(UV) pytest tests/ -q

run: ## Start SmartSplit server
	uv run python -m smartsplit

watch: ## Run provider watch locally
	uv run python scripts/provider_watch.py

docker-build: ## Build Docker image
	docker build -t smartsplit .

docker-up: ## Start with Docker Compose
	docker compose up -d

docker-down: ## Stop Docker Compose
	docker compose down

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache __pycache__ dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
