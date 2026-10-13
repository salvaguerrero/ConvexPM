"""Fetch and decode Trading Economics chart data into a pandas DataFrame.

Example:
    python examples/tradingeconomics_encoded_response.py

The endpoint currently returns a quoted Base64 string. Its bytes are XOR'd with
``TE_OBFUSCATION_KEY`` and then gzip-compressed JSON.
"""

from __future__ import annotations

import base64
import gzip
import json
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


TE_OBFUSCATION_KEY = "tradingeconomics-charts-core-api-key"
DEFAULT_MARKETS = ("CO1:COM", "XB1:COM")
DEFAULT_URL = "https://d3ii0wo49og5mi.cloudfront.net/markets/{markets}?span={span}&ohlc=0"


def decode_te_response(encoded_response: str, key: str = TE_OBFUSCATION_KEY) -> dict[str, Any]:
    """Decode a Trading Economics obfuscated chart response."""
    payload = encoded_response.strip()
    if payload.startswith('"') and payload.endswith('"'):
        payload = json.loads(payload)

    decoded = bytearray(base64.b64decode(payload))
    key_bytes = key.encode("utf-8")
    for index, value in enumerate(decoded):
        decoded[index] = value ^ key_bytes[index % len(key_bytes)]

    return json.loads(gzip.decompress(decoded).decode("utf-8"))


def te_json_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    """Normalize the decoded Trading Economics JSON into long-form rows."""
    rows: list[dict[str, Any]] = []
    for series in data.get("series", []):
        for point in series.get("data", []):
            rows.append(
                {
                    "date": pd.to_datetime(point[0], unit="s", utc=True).tz_localize(None),
                    "symbol": series.get("symbol"),
                    "name": series.get("name"),
                    "full_name": series.get("full_name"),
                    "unit": series.get("unit"),
                    "value": point[1] if len(point) > 1 else None,
                    "change_percent": point[2] if len(point) > 2 else None,
                    "change": point[3] if len(point) > 3 else None,
                }
            )

    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)


def fetch_te_markets(
    markets: tuple[str, ...] = DEFAULT_MARKETS,
    *,
    span: str = "5y",
    token: str | None = None,
    last_update: str | None = None,
) -> pd.DataFrame:
    """Fetch chart data from the endpoint and return it as a DataFrame."""
    encoded_markets = ",".join(quote(market, safe=":") for market in markets)
    url = DEFAULT_URL.format(markets=encoded_markets, span=quote(span, safe=""))
    headers = {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "origin": "https://tradingeconomics.com",
        "referer": "https://tradingeconomics.com/",
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36"
        ),
    }
    if token:
        headers["x-api-token"] = token
    if last_update:
        headers["x-api-lastupdate"] = last_update

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            encoded_response = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Trading Economics request failed: {exc.code} {detail}") from exc

    return te_json_to_dataframe(decode_te_response(encoded_response))


def main() -> None:
    DEFAULT_MARKETS = ("CO1:COM", "XB1:COM")

    markets = ("spx:ind",)
    span = "50y"
    token = "20260324:loboantunes:1c251cfe5af9b0cea361feb9e4072574c05eb8f5fc12dd953e48ea05af100aab"
    last_update = "202610121600"
    csv_path = None  # Example: "data/te_markets.csv"

    df = fetch_te_markets(
        markets,
        span=span,
        token=token,
        last_update=last_update,
    )
    if csv_path:
        df.to_csv(csv_path, index=False)
    print(df)


if __name__ == "__main__":
    main()
