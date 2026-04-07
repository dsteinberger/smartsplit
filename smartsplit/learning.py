"""MAB (Multi-Armed Bandit) scoring — auto-calibrates provider scores from real results.

Uses UCB1 (Upper Confidence Bound) to balance exploitation (use the best provider)
with exploration (occasionally try less-used providers to detect improvements).

The static competence table serves as the prior — used when no real data exists.
As requests flow, the MAB learns which providers actually perform well for each task type.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

logger = logging.getLogger("smartsplit.learning")

_EXPLORATION_FACTOR = 1.5  # controls exploration vs exploitation (higher = more exploration)
_PRIOR_STRENGTH = 10  # how many "virtual" observations the prior is worth
_SAVE_INTERVAL = 60  # persist every 60 seconds


class BanditScorer:
    """UCB1 bandit that learns provider quality from real routing results."""

    def __init__(self, persistence_path: str | None = None) -> None:
        # stats[task_type][provider] = {"success": int, "total": int}
        self._stats: dict[str, dict[str, dict[str, int]]] = {}
        self._total_pulls = 0
        self._dirty = False
        self._last_save: float = 0.0
        self._path = Path(persistence_path or (Path.home() / ".smartsplit" / "scores.json"))
        self._load()

    def record(self, task_type: str, provider: str, success: bool) -> None:
        """Record a routing result. Called after each provider call."""
        if task_type not in self._stats:
            self._stats[task_type] = {}
        if provider not in self._stats[task_type]:
            self._stats[task_type][provider] = {"success": 0, "total": 0}

        self._stats[task_type][provider]["total"] += 1
        if success:
            self._stats[task_type][provider]["success"] += 1
        self._total_pulls += 1
        self._dirty = True

        now = time.time()
        if now - self._last_save >= _SAVE_INTERVAL:
            self._save()
            self._last_save = now

    def score(self, task_type: str, provider: str, prior: float) -> float:
        """Compute UCB1 score blended with the static prior.

        The prior (from the competence table) is always part of the score,
        weighted inversely to the amount of data. With 0 data → pure prior.
        With lots of data → mostly learned. The exploration bonus encourages
        trying less-used providers.
        """
        stats = self._stats.get(task_type, {}).get(provider)
        if stats is None or stats["total"] == 0:
            return prior

        total = stats["total"]
        success_rate = stats["success"] / total

        # Blend: prior dominates early, learned dominates later
        # At 10 pulls, prior has ~50% weight. At 50 pulls, ~17%. At 100, ~9%.
        prior_weight = _PRIOR_STRENGTH / (_PRIOR_STRENGTH + total)
        blended = prior_weight * prior + (1 - prior_weight) * success_rate

        exploration = _EXPLORATION_FACTOR * math.sqrt(math.log(max(self._total_pulls, 1)) / total)
        return blended + exploration

    def get_stats(self) -> dict[str, dict[str, dict[str, int]]]:
        """Return the raw stats for observability."""
        return dict(self._stats)

    def flush(self) -> None:
        if self._dirty:
            self._save()

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            data = {"stats": self._stats, "total_pulls": self._total_pulls}
            self._path.write_text(json.dumps(data, indent=2))
            self._path.chmod(0o600)
            self._dirty = False
        except OSError as e:
            logger.warning(f"Could not save MAB scores: {e}")

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                self._stats = data.get("stats", {})
                self._total_pulls = data.get("total_pulls", 0)
                logger.info(f"Loaded MAB scores: {self._total_pulls} data points")
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load MAB scores: {e}")
