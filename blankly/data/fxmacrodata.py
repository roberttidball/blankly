"""
    FXMacroData price reader
    Copyright (C) 2026  Blankly Contributors

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
from typing import Optional, Tuple

import pandas as pd
import requests

from blankly.data.data_reader import DataReader, DataTypes, PriceReader

DEFAULT_BASE_URL = "https://api.fxmacrodata.com/v1"
API_KEY_ENV_VARS = ("FXMACRODATA_API_KEY", "FXMD_API_KEY")
PRICE_COLUMNS = ["time", "open", "high", "low", "close", "volume"]
PAGE_SIZE = 100
MAX_PAGES = 1000


class FXMacroDataClient:
    """Small REST client for FXMacroData's public API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30,
    ):
        self.api_key = clean_api_key(api_key or get_env_api_key())
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str, params: Optional[dict] = None):
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        response = requests.get(
            f"{self.base_url}/{path.lstrip('/')}",
            params=clean_params(params or {}),
            headers=headers,
            timeout=self.timeout,
            # Redirects are not followed, so the key never leaves this host.
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise requests.HTTPError(
                f"FXMacroData returned an unexpected redirect ({response.status_code})"
            )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and "detail" in payload and "data" not in payload:
            raise ValueError(f"FXMacroData request failed: {payload['detail']}")
        return payload

    def get_rows(self, path: str, params: Optional[dict] = None, max_rows: Optional[int] = None) -> list:
        """
        Fetch every page of a paginated list endpoint and return the rows.

        List endpoints return at most 100 rows per request (20 by default),
        newest first, with a ``pagination`` object describing the next page.
        """
        params = dict(params or {})
        rows = []
        offset = 0
        for _ in range(MAX_PAGES):
            limit = PAGE_SIZE if max_rows is None else min(PAGE_SIZE, max_rows - len(rows))
            payload = self.get(path, {**params, "limit": limit, "offset": offset})
            page = payload_rows(payload)
            rows.extend(page)
            pagination = payload.get("pagination") if isinstance(payload, dict) else None
            if not page or not isinstance(pagination, dict) or not pagination.get("has_more"):
                break
            if max_rows is not None and len(rows) >= max_rows:
                break
            offset = pagination.get("next_offset") or offset + len(page)
        return rows

    def forex(self, base: str, quote: str, start_date=None, end_date=None, limit=None, offset=None):
        return self.get(
            f"forex/{base.lower()}/{quote.lower()}",
            date_params(start_date, end_date, limit=limit, offset=offset),
        )

    def announcements(
        self, currency: str, indicator: str, start_date=None, end_date=None, limit=None, offset=None
    ):
        return self.get(
            f"announcements/{currency.lower()}/{indicator.lower()}",
            date_params(start_date, end_date, limit=limit, offset=offset),
        )

    def calendar(self, currency: str, indicator: Optional[str] = None, start_date=None, end_date=None):
        return self.get(
            f"calendar/{currency.lower()}",
            {
                "indicator": indicator.lower() if indicator else None,
                "start_date": format_date(start_date) if start_date else None,
                "end_date": format_date(end_date) if end_date else None,
            },
        )

    def predictions(
        self, currency: str, indicator: str, start_date=None, end_date=None, limit=None, offset=None
    ):
        return self.get(
            f"predictions/{currency.lower()}/{indicator.lower()}",
            date_params(start_date, end_date, limit=limit, offset=offset),
        )

    def data_catalogue(self, currency: str, include_coverage: bool = True):
        return self.get(
            f"data_catalogue/{currency.lower()}",
            {"include_coverage": str(include_coverage).lower()},
        )


class FXMacroDataPriceReader(PriceReader):
    """
    PriceReader that loads daily FX spot rates from FXMacroData.

    Args:
        symbol: FX pair such as ``EURUSD``, ``EUR/USD``, or ``EUR-USD``.
        start_date: Start date accepted by ``pandas.Timestamp``.
        stop_date: Stop date accepted by ``pandas.Timestamp``.
        api_key: Optional FXMacroData API key. If omitted, ``FXMACRODATA_API_KEY``
            and ``FXMD_API_KEY`` environment variables are checked.
        base_url: FXMacroData REST API base URL.
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        symbol: str,
        start_date,
        stop_date,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30,
    ):
        self.symbol = self.normalize_symbol(symbol)
        self.client = FXMacroDataClient(api_key=api_key, base_url=base_url, timeout=timeout)

        data = self.fetch_prices(start_date, stop_date)
        super().__init__(data, self.symbol)

    def fetch_prices(self, start_date, stop_date) -> pd.DataFrame:
        base, quote = self.split_symbol(self.symbol)
        rows = self.client.get_rows(f"forex/{base}/{quote}", date_params(start_date, stop_date))
        data = self.rows_to_dataframe(rows)
        if data.empty:
            raise ValueError(f"FXMacroData returned no prices for {self.symbol}")
        return data

    @classmethod
    def rows_to_dataframe(cls, rows: list) -> pd.DataFrame:
        records = []
        for row in rows:
            date = row.get("date") or row.get("timestamp")
            rate = cls.extract_rate(row)
            if date is None or rate is None:
                continue
            records.append(
                {
                    "time": to_epoch(date),
                    "open": rate,
                    "high": rate,
                    "low": rate,
                    "close": rate,
                    "volume": 0.0,
                }
            )
        if not records:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        return pd.DataFrame(records).drop_duplicates("time").sort_values("time")

    @staticmethod
    def extract_rate(row: dict) -> Optional[float]:
        for key in ("val", "value", "close", "rate", "fx_rate"):
            value = row.get(key)
            if value is not None:
                return float(value)
        return None

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        pair = symbol.strip().upper()
        if pair.endswith("=X"):
            pair = pair[:-2]
        pair = pair.replace("/", "").replace("-", "").replace("_", "")
        if len(pair) != 6 or not pair.isalpha():
            raise ValueError("FXMacroData symbols must look like EURUSD or EUR-USD")
        return f"{pair[:3]}-{pair[3:]}"

    @staticmethod
    def split_symbol(symbol: str) -> Tuple[str, str]:
        base, quote = symbol.split("-")
        return base.lower(), quote.lower()


class FXMacroDataEventReader(DataReader):
    """Base event reader for FXMacroData macro rows."""

    def __init__(self, event_type: str, rows: list, time_field: str = "announcement_datetime"):
        super().__init__(DataTypes.event_json)
        times = []
        events = []
        for row in rows:
            event_time = row_time(row, time_field)
            if event_time is None:
                continue
            times.append(event_time)
            events.append(row)
        self._write_dataset({"time": times, "data": events}, event_type, ("time", "data"))
        self._internal_dataset[event_type] = self._internal_dataset[event_type].sort_values("time")


class FXMacroDataAnnouncementReader(FXMacroDataEventReader):
    """Event reader for realized macroeconomic announcements."""

    def __init__(
        self,
        currency: str,
        indicator: str,
        start_date=None,
        stop_date=None,
        limit: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30,
    ):
        client = FXMacroDataClient(api_key=api_key, base_url=base_url, timeout=timeout)
        rows = client.get_rows(
            f"announcements/{currency.lower()}/{indicator.lower()}",
            date_params(start_date, stop_date),
            max_rows=limit,
        )
        event_type = f"fxmacrodata_announcements_{currency.lower()}_{indicator.lower()}"
        super().__init__(event_type, rows)


class FXMacroDataCalendarReader(FXMacroDataEventReader):
    """Event reader for upcoming official release-calendar rows."""

    def __init__(
        self,
        currency: str,
        indicator: Optional[str] = None,
        start_date=None,
        stop_date=None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30,
    ):
        client = FXMacroDataClient(api_key=api_key, base_url=base_url, timeout=timeout)
        rows = payload_rows(client.calendar(currency, indicator, start_date, stop_date))
        indicator_suffix = f"_{indicator.lower()}" if indicator else ""
        event_type = f"fxmacrodata_calendar_{currency.lower()}{indicator_suffix}"
        super().__init__(event_type, rows)


class FXMacroDataPredictionReader(FXMacroDataEventReader):
    """Event reader for FXMacroData forecasts grouped by announcement."""

    def __init__(
        self,
        currency: str,
        indicator: str,
        start_date=None,
        stop_date=None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30,
    ):
        client = FXMacroDataClient(api_key=api_key, base_url=base_url, timeout=timeout)
        rows = client.get_rows(
            f"predictions/{currency.lower()}/{indicator.lower()}",
            date_params(start_date, stop_date),
        )
        event_type = f"fxmacrodata_predictions_{currency.lower()}_{indicator.lower()}"
        super().__init__(event_type, rows)


def clean_params(params: dict) -> dict:
    return {key: value for key, value in params.items() if value is not None}


def date_params(start_date=None, end_date=None, **extra) -> dict:
    params = {
        "start_date": format_date(start_date) if start_date else None,
        "end_date": format_date(end_date) if end_date else None,
    }
    params.update(extra)
    return params


def payload_rows(payload) -> list:
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def row_time(row: dict, time_field: str = "announcement_datetime") -> Optional[int]:
    value = row.get(time_field) or row.get("announcement_datetime")
    if value is not None:
        return int(value)
    date = row.get("date") or row.get("timestamp")
    if date is None:
        return None
    return to_epoch(date)


def clean_api_key(api_key: Optional[str]) -> Optional[str]:
    api_key = (api_key or "").strip()
    if any(char.isspace() or not char.isprintable() for char in api_key):
        raise ValueError("FXMacroData API key contains invalid characters")
    return api_key or None


def get_env_api_key() -> Optional[str]:
    for name in API_KEY_ENV_VARS:
        api_key = os.getenv(name)
        if api_key:
            return api_key
    return None


def format_date(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def to_epoch(value) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp())
