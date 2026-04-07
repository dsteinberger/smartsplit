"""Quota tracker — monitors usage, availability, and cost savings."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from smartsplit.models import ProviderType, SavingsReport

if TYPE_CHECKING:
    from smartsplit.config import ProviderConfig

logger = logging.getLogger("smartsplit.quota")

_FALLBACK_TOKENS_PER_CALL = 500
_INPUT_TOKEN_SAFETY_MARGIN = 1.2
_COST_PER_M_TOKENS = 3.0


def estimate_tokens(prompt: str) -> int:
    """Estimate token count from prompt text using chars÷4 heuristic with safety margin.

    Conservative: 1.2x multiplier accounts for tokenizer variance.
    Falls back to 500 for empty prompts.
    """
    if not prompt:
        return _FALLBACK_TOKENS_PER_CALL
    return max(int(len(prompt) / 4 * _INPUT_TOKEN_SAFETY_MARGIN), _FALLBACK_TOKENS_PER_CALL)


_DAY_SECONDS = 86_400
_DEFAULT_RPD = 1_000


def _extract_rpd(pconfig: ProviderConfig) -> int:
    """Get the effective requests-per-day limit from a provider config."""
    limits = pconfig.limits
    if "rpd" in limits:
        return limits["rpd"]
    if "monthly" in limits:
        return limits["monthly"] // 30
    if pconfig.type == ProviderType.PAID:
        return 999_999
    return _DEFAULT_RPD


_SAVE_INTERVAL = 30  # seconds between disk writes


class QuotaTracker:
    """Tracks API usage and computes availability scores and savings."""

    def __init__(
        self,
        provider_configs: dict[str, ProviderConfig] | None = None,
        persistence_path: str | None = None,
    ) -> None:
        self._usage: dict[str, dict[str, int | float | dict[str, int]]] = {}
        self._savings: dict[str, int] = {"free_calls": 0, "paid_calls": 0, "estimated_tokens_saved": 0}
        self._dirty = False
        self._last_save: float = 0.0
        self._lock = asyncio.Lock()

        # Build limits from config (single source of truth)
        self._limits: dict[str, int] = {}
        if provider_configs:
            for name, pconfig in provider_configs.items():
                self._limits[name] = _extract_rpd(pconfig)

        self._path = Path(persistence_path or (Path.home() / ".smartsplit" / "quota.json"))
        self._load()

    # ── Public API ───────────────────────────────────────────

    def record_usage(
        self,
        provider: str,
        task_type: str,
        *,
        is_paid: bool = False,
        prompt: str = "",
    ) -> None:
        self._maybe_reset(provider)
        entry = self._usage.setdefault(provider, {"count": 0, "last_reset": time.time(), "by_type": {}})
        entry["count"] += 1
        entry["by_type"][task_type] = entry["by_type"].get(task_type, 0) + 1

        tokens = estimate_tokens(prompt)
        if is_paid:
            self._savings["paid_calls"] += 1
        else:
            self._savings["free_calls"] += 1
            self._savings["estimated_tokens_saved"] += tokens

        self._dirty = True
        now = time.time()
        if now - self._last_save >= _SAVE_INTERVAL:
            self._save()
            self._last_save = now

    def _maybe_reset(self, provider: str) -> None:
        """Reset daily counter if more than 24h since last reset."""
        entry = self._usage.get(provider)
        if entry is None:
            return
        if time.time() - entry.get("last_reset", time.time()) > _DAY_SECONDS:
            self._usage[provider] = {"count": 0, "last_reset": time.time(), "by_type": {}}
            self._dirty = True

    def get_availability(self, provider: str) -> float:
        """Return a 0.0-1.0 ratio of remaining quota for *provider*."""
        self._maybe_reset(provider)
        limit = self._limits.get(provider, _DEFAULT_RPD)
        entry = self._usage.get(provider, {"count": 0})
        used = entry["count"]
        return max(0.0, (limit - used) / limit)

    def get_usage(self, provider: str) -> int:
        return self._usage.get(provider, {}).get("count", 0)

    def get_savings_report(self) -> SavingsReport:
        total = self._savings["free_calls"] + self._savings["paid_calls"]
        free_pct = (self._savings["free_calls"] / total * 100) if total > 0 else 0.0
        tokens_saved = self._savings["estimated_tokens_saved"]

        return SavingsReport(
            total_requests=total,
            free_requests=self._savings["free_calls"],
            paid_requests=self._savings["paid_calls"],
            free_percentage=round(free_pct, 1),
            estimated_tokens_saved=tokens_saved,
            estimated_cost_saved_usd=round(tokens_saved * _COST_PER_M_TOKENS / 1_000_000, 4),
            providers_usage={name: data["count"] for name, data in self._usage.items()},
        )

    def flush(self) -> None:
        """Force a write to disk if there are pending changes."""
        if self._dirty:
            self._save()

    # ── Persistence ──────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._path.write_text(json.dumps({"usage": self._usage, "savings": self._savings}, indent=2))
            self._path.chmod(0o600)
            self._dirty = False
        except OSError as e:
            logger.warning(f"Could not save quota data: {e}")

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                self._usage = data.get("usage", {})
                self._savings = data.get("savings", self._savings)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load quota data: {e}")
