UV := uv run --extra dev

.PHONY: help install install-proxy test lint format check run proxy setup-claude watch clean env \
        docker-build docker-up docker-down docker-build-proxy docker-up-proxy docker-down-proxy

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Dev ──────────────────────────────────────────────────────

install: ## Install dependencies (dev)
	uv pip install -e ".[dev]"

install-proxy: ## Install with proxy support (Python >= 3.12)
	uv pip install -e ".[dev,proxy]"

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

# ── Run ──────────────────────────────────────────────────────

run: ## Start SmartSplit server (API mode)
	uv run python -m smartsplit

proxy: ## Start SmartSplit in HTTPS proxy mode (for Claude Code)
	uv run --extra proxy smartsplit --mode proxy

setup-claude: ## One-time Claude Code setup (generates certs)
	uv run --extra proxy smartsplit setup-claude

watch: ## Run provider watch locally
	uv run python scripts/provider_watch.py

# ── Docker (API mode) ───────────────────────────────────────

docker-build: ## Build Docker image (API mode)
	docker build -t smartsplit .

docker-up: ## Start with Docker Compose (API mode)
	docker compose up -d

docker-down: ## Stop Docker Compose
	docker compose down

# ── Docker (proxy mode) ─────────────────────────────────────

docker-build-proxy: ## Build Docker image (proxy mode)
	docker build -f Dockerfile.proxy -t smartsplit-proxy .

docker-up-proxy: ## Start with Docker Compose (proxy mode)
	docker compose --profile proxy up -d proxy

docker-down-proxy: ## Stop proxy service
	docker compose --profile proxy down

# ── Setup ────────────────────────────────────────────────────

env: ## Create .env from template (won't overwrite existing)
	@if [ -f .env ]; then echo "\033[33m.env already exists — skipping (delete it first to recreate)\033[0m"; else cp .env.example .env && echo "\033[32m.env created — edit it with your API keys\033[0m"; fi

# ── Clean ────────────────────────────────────────────────────

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache __pycache__ dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
