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
from typing import Optional
from typing import Tuple

import pandas as pd
import requests

from blankly.data.data_reader import PriceReader

DEFAULT_BASE_URL = "https://fxmacrodata.com/api/v1"
API_KEY_ENV_VARS = ("FXMACRODATA_API_KEY", "FXMD_API_KEY")
PRICE_COLUMNS = ["time", "open", "high", "low", "close", "volume"]


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
        self.api_key = api_key or self.get_env_api_key()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        data = self.fetch_prices(start_date, stop_date)
        super().__init__(data, self.symbol)

    def fetch_prices(self, start_date, stop_date) -> pd.DataFrame:
        base, quote = self.split_symbol(self.symbol)
        params = {
            "start_date": self.format_date(start_date),
            "end_date": self.format_date(stop_date),
        }
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        response = requests.get(
            f"{self.base_url}/forex/{base}/{quote}",
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = self.rows_to_dataframe(self.payload_rows(response.json()))
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
                    "time": cls.to_epoch(date),
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
    def payload_rows(payload) -> list:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            rows = payload.get("data", [])
            return rows if isinstance(rows, list) else []
        return []

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

    @staticmethod
    def get_env_api_key() -> Optional[str]:
        for name in API_KEY_ENV_VARS:
            api_key = os.getenv(name)
            if api_key:
                return api_key
        return None

    @staticmethod
    def format_date(value) -> str:
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    @staticmethod
    def to_epoch(value) -> int:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return int(timestamp.timestamp())
