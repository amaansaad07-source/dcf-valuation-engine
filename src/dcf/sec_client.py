"""Polite, cached client for the SEC XBRL REST API.

Handles the three things that break naive scrapers: the mandatory ``User-Agent`` header,
the 10 requests/second fair-access ceiling, and 5-40MB ``companyfacts`` payloads.
"""

import json
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

from .config import CFG, Config, log


class SECError(RuntimeError):
    """Raised when the SEC API cannot satisfy a request."""


class TickerNotFound(SECError):
    """Raised when a ticker has no CIK in the SEC registrant map."""


class SECClient:
    """Thin, polite, cached client for the SEC XBRL REST API."""

    def __init__(self, config: Config = CFG):
        self.cfg = config
        self.session = requests.Session()
        # The SEC's only requirement is a descriptive User-Agent with a real contact.
        # `Host` is deliberately NOT set — requests derives it from the URL, and hardcoding
        # it breaks the moment you call www.sec.gov instead of data.sec.gov.
        self.session.headers.update({
            "User-Agent": config.user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_call = 0.0
        self._ticker_map: Optional[pd.DataFrame] = None

    # ---------------------------------------------------------------- internals ----
    def _throttle(self) -> None:
        """Keep us comfortably inside the SEC's 10 requests/second ceiling."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.cfg.min_request_interval:
            time.sleep(self.cfg.min_request_interval - elapsed)
        self._last_call = time.monotonic()

    def _cache_path(self, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:120]
        return self.cfg.cache_dir / f"{safe}.json"

    def _read_cache(self, key: str) -> Optional[dict]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > self.cfg.cache_ttl_hours:
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                log.info("Cache hit: %s (age %.1fh)", key, age_hours)
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            path.unlink(missing_ok=True)   # corrupt cache entry — drop it
            return None

    def _write_cache(self, key: str, payload: dict) -> None:
        try:
            with self._cache_path(key).open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except OSError as exc:  # a full disk should never kill a valuation
            log.warning("Could not cache %s: %s", key, exc)

    def _get_json(self, url: str, cache_key: Optional[str] = None) -> dict:
        """GET with throttling, retries and optional disk cache."""
        if cache_key:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        last_error: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=self.cfg.request_timeout)
                if resp.status_code == 200:
                    payload = resp.json()
                    if cache_key:
                        self._write_cache(cache_key, payload)
                    return payload
                if resp.status_code == 404:
                    raise SECError(f"404 Not Found — the SEC has no data at {url}")
                if resp.status_code == 403:
                    raise SECError(
                        "403 Forbidden — the SEC rejected the User-Agent. "
                        "Set CFG.user_agent to 'Your Name your@email.com'."
                    )
                last_error = SECError(f"HTTP {resp.status_code}")
            except requests.RequestException as exc:
                last_error = exc

            wait = self.cfg.backoff_base ** attempt
            log.warning("SEC request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, self.cfg.max_retries, last_error, wait)
            time.sleep(wait)

        raise SECError(f"Gave up on {url} after {self.cfg.max_retries} attempts: {last_error}")

    # ------------------------------------------------------------------ public ----
    def ticker_map(self) -> pd.DataFrame:
        """Full SEC registrant map: ticker → CIK → company name."""
        if self._ticker_map is None:
            raw = self._get_json(self.cfg.sec_ticker_map, cache_key="company_tickers")
            frame = pd.DataFrame(raw).T
            frame.columns = [c.lower() for c in frame.columns]
            frame["ticker"] = frame["ticker"].str.upper()
            frame["cik_str"] = frame["cik_str"].astype(int)
            self._ticker_map = frame.rename(columns={"cik_str": "cik", "title": "name"})
            log.info("Loaded SEC registrant map: %d tickers", len(self._ticker_map))
        return self._ticker_map

    def resolve_cik(self, ticker: str) -> Tuple[int, str]:
        """'AAPL' → (320193, 'Apple Inc.'). Raises TickerNotFound if unlisted."""
        ticker = ticker.strip().upper()
        frame = self.ticker_map()
        hit = frame.loc[frame["ticker"] == ticker]
        if hit.empty:
            # Class-share tickers often use a dot or dash: BRK.B / BRK-B
            alt = ticker.replace(".", "-")
            hit = frame.loc[frame["ticker"] == alt]
        if hit.empty:
            raise TickerNotFound(
                f"'{ticker}' is not in the SEC registrant map. Foreign issuers without "
                "US listings and private companies will not appear."
            )
        row = hit.iloc[0]
        return int(row["cik"]), str(row["name"])

    def company_facts(self, cik: int) -> dict:
        """Every XBRL fact the company has ever tagged, in one payload."""
        url = self.cfg.sec_companyfacts.format(cik=cik)
        payload = self._get_json(url, cache_key=f"facts_{cik}")
        if "facts" not in payload:
            raise SECError(f"companyfacts payload for CIK {cik} contains no 'facts' block")
        return payload

    def company_concept(self, cik: int, tag: str) -> dict:
        """One tag's full history — useful for debugging a stubborn line item."""
        url = self.cfg.sec_companyconcept.format(cik=cik, tag=tag)
        return self._get_json(url, cache_key=f"concept_{cik}_{tag}")


SEC = SECClient()
