"""
    Unit tests for FXMacroData price reader
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

import unittest
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd

API_KEY = "api_key"
ROOT_DIR = Path(__file__).resolve().parents[2]


def load_fxmacrodata_module():
    blankly_module = types.ModuleType("blankly")
    blankly_module.__path__ = [str(ROOT_DIR / "blankly")]
    data_module = types.ModuleType("blankly.data")
    data_module.__path__ = [str(ROOT_DIR / "blankly" / "data")]
    utils_module = types.ModuleType("blankly.utils")
    utils_module.convert_epochs = lambda value: value
    exchanges_module = types.ModuleType("blankly.exchanges")
    interfaces_module = types.ModuleType("blankly.exchanges.interfaces")
    futures_module = types.ModuleType(
        "blankly.exchanges.interfaces.futures_exchange_interface"
    )

    class FuturesExchangeInterface:
        pass

    futures_module.FuturesExchangeInterface = FuturesExchangeInterface
    stubs = {
        "blankly": blankly_module,
        "blankly.data": data_module,
        "blankly.utils": utils_module,
        "blankly.exchanges": exchanges_module,
        "blankly.exchanges.interfaces": interfaces_module,
        "blankly.exchanges.interfaces.futures_exchange_interface": futures_module,
    }
    with patch.dict(sys.modules, stubs):
        data_reader_path = ROOT_DIR / "blankly" / "data" / "data_reader.py"
        data_reader_spec = importlib.util.spec_from_file_location(
            "blankly.data.data_reader", data_reader_path
        )
        data_reader = importlib.util.module_from_spec(data_reader_spec)
        sys.modules["blankly.data.data_reader"] = data_reader
        data_reader_spec.loader.exec_module(data_reader)

        fxmacrodata_path = ROOT_DIR / "blankly" / "data" / "fxmacrodata.py"
        fxmacrodata_spec = importlib.util.spec_from_file_location(
            "blankly.data.fxmacrodata", fxmacrodata_path
        )
        fxmacrodata = importlib.util.module_from_spec(fxmacrodata_spec)
        fxmacrodata_spec.loader.exec_module(fxmacrodata)
    return fxmacrodata


class FXMacroDataResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FXMacroDataPriceReaderTest(unittest.TestCase):
    def setUp(self):
        self.fxmacrodata = load_fxmacrodata_module()
        self.price_reader_class = self.fxmacrodata.FXMacroDataPriceReader

    def test_fetches_daily_fx_prices(self):
        requests = []

        def mock_get(url, params, headers, timeout, allow_redirects=True):
            requests.append((url, params, headers, timeout))
            return FXMacroDataResponse(
                {
                    "data": [
                        {"date": "2024-01-02", "val": "1.101"},
                        {"date": "2024-01-03", "value": 1.102},
                        {"date": "2024-01-04", "close": 1.103},
                    ]
                }
            )

        with patch.object(self.fxmacrodata.requests, "get", side_effect=mock_get):
            reader = self.price_reader_class(
                "EUR/USD",
                "2024-01-02",
                "2024-01-04",
                api_key=API_KEY,
                base_url="https://example.com/api/v1",
            )

        url, params, headers, timeout = requests[0]
        self.assertEqual(url, "https://example.com/api/v1/forex/eur/usd")
        self.assertEqual(
            params,
            {"start_date": "2024-01-02", "end_date": "2024-01-04", "limit": 100, "offset": 0},
        )
        self.assertEqual(headers, {"X-API-Key": API_KEY})
        self.assertEqual(timeout, 30)
        self.assertEqual(list(reader.data), ["EUR-USD"])
        data = reader.data["EUR-USD"]
        self.assertEqual(data["close"].tolist(), [1.101, 1.102, 1.103])
        self.assertEqual(data["volume"].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(reader.prices_info["EUR-USD"]["resolution"], 86400)

    def test_follows_pagination_across_pages(self):
        requests = []
        pages = {
            0: {
                "data": [
                    {"date": "2024-01-04", "val": 1.103},
                    {"date": "2024-01-03", "val": 1.102},
                ],
                "pagination": {"has_more": True, "next_offset": 2},
            },
            2: {
                "data": [{"date": "2024-01-02", "val": 1.101}],
                "pagination": {"has_more": False, "next_offset": None},
            },
        }

        def mock_get(url, params, headers, timeout, allow_redirects=True):
            requests.append(dict(params))
            return FXMacroDataResponse(pages[params["offset"]])

        with patch.object(self.fxmacrodata.requests, "get", side_effect=mock_get):
            reader = self.price_reader_class(
                "EURUSD", "2024-01-02", "2024-01-04", api_key=API_KEY
            )

        self.assertEqual([params["offset"] for params in requests], [0, 2])
        self.assertTrue(all(params["limit"] == 100 for params in requests))
        self.assertEqual(reader.data["EUR-USD"]["close"].tolist(), [1.101, 1.102, 1.103])

    def test_does_not_follow_redirects(self):
        calls = []

        def mock_get(url, params, headers, timeout, allow_redirects=True):
            calls.append(allow_redirects)
            return FXMacroDataResponse({}, status_code=302)

        with patch.object(self.fxmacrodata.requests, "get", side_effect=mock_get):
            with self.assertRaises(self.fxmacrodata.requests.HTTPError):
                self.price_reader_class("EURUSD", "2024-01-02", "2024-01-04", api_key=API_KEY)

        self.assertEqual(calls, [False])

    def test_invalid_api_key_is_not_echoed(self):
        with self.assertRaises(ValueError) as context:
            self.fxmacrodata.FXMacroDataClient(api_key="secret-key\r\nX-Other: 1")

        self.assertNotIn("secret-key", str(context.exception))
        self.assertEqual(self.fxmacrodata.FXMacroDataClient(api_key=" key \n").api_key, "key")

    def test_error_body_and_malformed_rows_raise_clean_errors(self):
        def error_body(url, params, headers, timeout, allow_redirects=True):
            return FXMacroDataResponse({"detail": "Invalid API key"})

        with patch.object(self.fxmacrodata.requests, "get", side_effect=error_body):
            with self.assertRaisesRegex(ValueError, "Invalid API key"):
                self.price_reader_class("EURUSD", "2024-01-02", "2024-01-04", api_key=API_KEY)

        def malformed(url, params, headers, timeout, allow_redirects=True):
            return FXMacroDataResponse({"data": ["oops", 1], "pagination": ["oops"]})

        with patch.object(self.fxmacrodata.requests, "get", side_effect=malformed):
            with self.assertRaisesRegex(ValueError, "no prices"):
                self.price_reader_class("EURUSD", "2024-01-02", "2024-01-04", api_key=API_KEY)

    def test_symbol_formats(self):
        self.assertEqual(self.price_reader_class.normalize_symbol("EURUSD"), "EUR-USD")
        self.assertEqual(self.price_reader_class.normalize_symbol("EUR/USD"), "EUR-USD")
        self.assertEqual(self.price_reader_class.normalize_symbol("EUR_USD"), "EUR-USD")
        self.assertEqual(self.price_reader_class.normalize_symbol("EURUSD=X"), "EUR-USD")

    def test_rows_to_dataframe_sorts_and_shapes_prices(self):
        data = self.price_reader_class.rows_to_dataframe(
            [
                {"date": "2024-01-03", "val": 1.102},
                {"date": "2024-01-02", "value": "1.101"},
            ]
        )

        self.assertEqual(data["close"].tolist(), [1.101, 1.102])
        self.assertEqual(data["open"].tolist(), [1.101, 1.102])
        self.assertEqual(data["high"].tolist(), [1.101, 1.102])
        self.assertEqual(data["low"].tolist(), [1.101, 1.102])
        self.assertEqual(data["volume"].tolist(), [0.0, 0.0])
        self.assertTrue(pd.api.types.is_integer_dtype(data["time"]))


class FXMacroDataMacroReaderTest(unittest.TestCase):
    def setUp(self):
        self.fxmacrodata = load_fxmacrodata_module()

    def test_announcement_reader_fetches_macro_events(self):
        requests = []

        def mock_get(url, params, headers, timeout, allow_redirects=True):
            requests.append((url, params, headers, timeout))
            return FXMacroDataResponse(
                {
                    "data": [
                        {
                            "announcement_id": "usd_inflation_2026-05-31",
                            "date": "2026-05-31",
                            "val": 4.2,
                            "announcement_datetime": 1781094600,
                        }
                    ]
                }
            )

        with patch.object(self.fxmacrodata.requests, "get", side_effect=mock_get):
            reader = self.fxmacrodata.FXMacroDataAnnouncementReader(
                "USD",
                "inflation",
                "2026-01-01",
                "2026-06-30",
                limit=5,
                api_key=API_KEY,
                base_url="https://example.com/api/v1",
            )

        url, params, headers, timeout = requests[0]
        self.assertEqual(url, "https://example.com/api/v1/announcements/usd/inflation")
        self.assertEqual(
            params,
            {"start_date": "2026-01-01", "end_date": "2026-06-30", "limit": 5, "offset": 0},
        )
        self.assertEqual(headers, {"X-API-Key": API_KEY})
        self.assertEqual(timeout, 30)
        event_data = reader.data["fxmacrodata_announcements_usd_inflation"]
        self.assertEqual(event_data["time"].tolist(), [1781094600])
        self.assertEqual(event_data["data"].iloc[0]["val"], 4.2)

    def test_calendar_reader_preserves_events_with_same_timestamp(self):
        def mock_get(url, params, headers, timeout, allow_redirects=True):
            return FXMacroDataResponse(
                {
                    "data": [
                        {
                            "release": "inflation",
                            "date": "2026-06-30",
                            "announcement_datetime": 1784032200,
                            "forecast": 3.9,
                        },
                        {
                            "release": "retail_sales",
                            "date": "2026-06-30",
                            "announcement_datetime": 1784032200,
                            "forecast": 0.2,
                        },
                    ]
                }
            )

        with patch.object(self.fxmacrodata.requests, "get", side_effect=mock_get):
            reader = self.fxmacrodata.FXMacroDataCalendarReader(
                "USD", api_key=API_KEY, base_url="https://example.com/api/v1"
            )

        event_data = reader.data["fxmacrodata_calendar_usd"]
        self.assertEqual(event_data["time"].tolist(), [1784032200, 1784032200])
        self.assertEqual(
            [row["release"] for row in event_data["data"].tolist()],
            ["inflation", "retail_sales"],
        )

    def test_prediction_reader_fetches_forecast_groups(self):
        def mock_get(url, params, headers, timeout, allow_redirects=True):
            return FXMacroDataResponse(
                {
                    "data": [
                        {
                            "announcement_id": "usd_inflation_2026-07-31",
                            "date": "2026-07-31",
                            "announcement_datetime": 1786537800,
                            "predictions": [
                                {
                                    "predicted_value": 3.81,
                                    "prediction_type": "fxmacrodata",
                                }
                            ],
                        }
                    ]
                }
            )

        with patch.object(self.fxmacrodata.requests, "get", side_effect=mock_get):
            reader = self.fxmacrodata.FXMacroDataPredictionReader(
                "USD",
                "inflation",
                api_key=API_KEY,
                base_url="https://example.com/api/v1",
            )

        event_data = reader.data["fxmacrodata_predictions_usd_inflation"]
        self.assertEqual(event_data["time"].tolist(), [1786537800])
        self.assertEqual(
            event_data["data"].iloc[0]["predictions"][0]["predicted_value"], 3.81
        )


if __name__ == "__main__":
    unittest.main()
