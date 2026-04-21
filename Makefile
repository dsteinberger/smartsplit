UV := uv run --extra dev

ifdef DEBUG
  export LOG_LEVEL=DEBUG
endif

.PHONY: help install install-proxy test lint format check run run-api run-proxy setup-claude watch clean env \
        build up up-api up-proxy down restart-proxy rebuild-proxy

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

run: ## Start SmartSplit (API + proxy unified — default) — DEBUG=1 for verbose
	uv run smartsplit

run-api: ## Start SmartSplit API only — DEBUG=1 for verbose
	uv run smartsplit --api-only

run-proxy: ## Start SmartSplit HTTPS proxy only — DEBUG=1 for verbose
	uv run smartsplit --proxy-only

setup-claude: ## One-time Claude Code setup (generates certs)
	uv run --extra proxy smartsplit setup-claude

watch: ## Run provider watch locally
	uv run python scripts/provider_watch.py

# ── Docker (unified / API only / proxy only — même image) ───

build: ## Docker — build image
	docker build -t smartsplit .

up: ## Docker — start unified (API + proxy) — DEBUG=1 for verbose
	docker compose up -d smartsplit

up-api: ## Docker — start API only
	docker compose --profile api up -d api

up-proxy: ## Docker — start proxy only (HTTPS interception for Claude Code)
	docker compose --profile proxy up -d proxy

down: ## Docker — stop all services (unified + api + proxy)
	docker compose --profile api --profile proxy down

restart-proxy: ## Docker — recreate proxy (fixes silent port-binding failures)
	docker compose --profile proxy up -d --force-recreate proxy

rebuild-proxy: ## Docker — rebuild image and recreate proxy (apply code changes)
	docker compose --profile proxy up -d --build --force-recreate proxy

# ── Setup ────────────────────────────────────────────────────

env: ## Create .env from template (won't overwrite existing)
	@if [ -f .env ]; then echo "\033[33m.env already exists — skipping (delete it first to recreate)\033[0m"; else cp .env.example .env && echo "\033[32m.env created — edit it with your API keys\033[0m"; fi

# ── Clean ────────────────────────────────────────────────────

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache __pycache__ dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
