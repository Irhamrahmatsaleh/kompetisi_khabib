#!/usr/bin/env python3
"""Market risk analyzer for OHLCV data.

This tool calculates volatility, momentum, ATR, trend state, and scenario bands.
It is for research and risk review only. It does not provide trading instructions,
financial advice, or a guarantee of performance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except Exception:
    Console = None
    Panel = None
    Table = None


@dataclass
class RadarReport:
    symbol: str
    interval: str
    last_close: float
    trend_bias: str
    risk_regime: str
    confidence_score: float
    atr_14: float
    atr_percent: float
    momentum_20: float
    realized_volatility_20: float
    rsi_14: float
    ema_12: float
    ema_26: float
    lower_atr_band_1: float
    upper_atr_band_1: float
    lower_atr_band_2: float
    upper_atr_band_2: float
    research_note: str


def load_market_data(symbol: str, period: str, interval: str, csv_path: Optional[str]) -> pd.DataFrame:
    if csv_path:
        data = pd.read_csv(csv_path)
        normalized = {str(c).strip().lower(): c for c in data.columns}
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in normalized]
        if missing:
            raise ValueError(f"CSV is missing required OHLC columns: {missing}")
        data = data.rename(columns={normalized[c]: c.title() for c in required})
        return data.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)

    if yf is None:
        raise RuntimeError("yfinance is unavailable. Run `make install` or pass --csv path/to/data.csv")

    data = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    if data.empty:
        raise RuntimeError(f"No market data returned for {symbol}")
    return data.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr_14"] = true_range.rolling(14).mean()
    df["atr_percent"] = df["atr_14"] / close
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["momentum_20"] = close.pct_change(20)
    df["return"] = close.pct_change()
    df["realized_volatility_20"] = df["return"].rolling(20).std() * math.sqrt(252)
    df["rsi_14"] = rsi(close)
    return df


def build_report(data: pd.DataFrame, symbol: str, interval: str) -> RadarReport:
    if len(data) < 60:
        raise ValueError("At least 60 rows are required for stable calculations.")

    df = add_indicators(data).dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("Indicator output is empty. Provide more OHLCV history.")

    last = df.iloc[-1]
    close = float(last["Close"])
    atr = float(last["atr_14"])
    atr_percent = float(last["atr_percent"])
    momentum = float(last["momentum_20"])
    volatility = float(last["realized_volatility_20"])
    rsi_value = float(last["rsi_14"])
    ema_12 = float(last["ema_12"])
    ema_26 = float(last["ema_26"])

    trend_component = 1.0 if ema_12 > ema_26 else -1.0
    momentum_component = float(np.tanh(momentum * 10))
    rsi_component = float(np.clip((rsi_value - 50) / 25, -1, 1))
    raw_score = 0.45 * trend_component + 0.35 * momentum_component + 0.20 * rsi_component
    confidence = float(round(min(0.99, abs(raw_score)), 4))

    if raw_score > 0.25:
        trend_bias = "upward"
    elif raw_score < -0.25:
        trend_bias = "downward"
    else:
        trend_bias = "mixed"

    if atr_percent > 0.08 or volatility > 0.90:
        risk_regime = "extreme"
    elif atr_percent > 0.04 or volatility > 0.55:
        risk_regime = "high"
    elif atr_percent < 0.015 and volatility < 0.25:
        risk_regime = "compressed"
    else:
        risk_regime = "normal"

    return RadarReport(
        symbol=symbol,
        interval=interval,
        last_close=round(close, 8),
        trend_bias=trend_bias,
        risk_regime=risk_regime,
        confidence_score=confidence,
        atr_14=round(atr, 8),
        atr_percent=round(atr_percent, 6),
        momentum_20=round(momentum, 6),
        realized_volatility_20=round(volatility, 6),
        rsi_14=round(rsi_value, 4),
        ema_12=round(ema_12, 8),
        ema_26=round(ema_26, 8),
        lower_atr_band_1=round(close - 1.0 * atr, 8),
        upper_atr_band_1=round(close + 1.0 * atr, 8),
        lower_atr_band_2=round(close - 2.0 * atr, 8),
        upper_atr_band_2=round(close + 2.0 * atr, 8),
        research_note="Scenario bands are analytical references, not execution instructions.",
    )


def render(report: RadarReport, as_json: bool) -> None:
    payload = asdict(report)
    if as_json or Console is None or Table is None or Panel is None:
        print(json.dumps(payload, indent=2))
        return

    console = Console()
    table = Table(title=f"Market Radar: {report.symbol}", show_lines=True)
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(key, str(value))
    console.print(Panel.fit("Research-only market structure analyzer. No guaranteed outcome."))
    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze volatility, momentum, ATR, and trend structure.")
    parser.add_argument("--symbol", default="BTC-USD", help="Ticker supported by yfinance. Default: BTC-USD")
    parser.add_argument("--period", default="6mo", help="Download period. Default: 6mo")
    parser.add_argument("--interval", default="1d", help="Download interval. Default: 1d")
    parser.add_argument("--csv", default=None, help="Optional local OHLCV CSV path")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_market_data(args.symbol, args.period, args.interval, args.csv)
    report = build_report(data, args.symbol, args.interval)
    render(report, args.json)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
