"""Cloudflare Workers AI provider — free backup via OpenAI-compatible endpoint.

Requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_KEY environment variables.
The api_url is built dynamically from the account ID.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import httpx

from smartsplit.exceptions import ProviderError
from smartsplit.providers.base import OpenAICompatibleProvider

if TYPE_CHECKING:
    from smartsplit.config import ProviderConfig

_ACCOUNT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


class CloudflareProvider(OpenAICompatibleProvider):
    name = "cloudflare"
    api_url = ""  # built dynamically from account_id

    def __init__(self, config: ProviderConfig, http: httpx.AsyncClient) -> None:
        super().__init__(config, http)
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        if not account_id:
            raise ProviderError(self.name, "CLOUDFLARE_ACCOUNT_ID not set")
        if not _ACCOUNT_ID_PATTERN.fullmatch(account_id):
            raise ProviderError(self.name, "CLOUDFLARE_ACCOUNT_ID must be a 32-char hex string")
        self.api_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
