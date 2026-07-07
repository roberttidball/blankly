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
    def __init__(self, payload):
        self.payload = payload

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

        def mock_get(url, params, headers, timeout):
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
            params, {"start_date": "2024-01-02", "end_date": "2024-01-04"}
        )
        self.assertEqual(headers, {"X-API-Key": API_KEY})
        self.assertEqual(timeout, 30)
        self.assertEqual(list(reader.data), ["EUR-USD"])
        data = reader.data["EUR-USD"]
        self.assertEqual(data["close"].tolist(), [1.101, 1.102, 1.103])
        self.assertEqual(data["volume"].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(reader.prices_info["EUR-USD"]["resolution"], 86400)

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


if __name__ == "__main__":
    unittest.main()
