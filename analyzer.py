#!/usr/bin/env python3
"""Competition-focused market scanner for demo trading.

This tool ranks multi-asset CFD/FX/crypto proxies by momentum, trend, volatility,
and ATR expansion. It is designed for competition planning and risk review only.
It does not provide financial advice, trade instructions, or guaranteed outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional runtime dependency
    yf = None

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except Exception:  # pragma: no cover - rich is optional for JSON/basic output
    Console = None
    Panel = None
    Table = None


DEFAULT_SYMBOLS = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "GC=F",
    "SI=F",
    "CL=F",
    "NQ=F",
    "ES=F",
    "BTC-USD",
    "ETH-USD",
]

SYMBOL_LABELS = {
    "EURUSD=X": "EUR/USD FX proxy",
    "GBPUSD=X": "GBP/USD FX proxy",
    "USDJPY=X": "USD/JPY FX proxy",
    "GC=F": "Gold futures proxy",
    "SI=F": "Silver futures proxy",
    "CL=F": "Crude oil futures proxy",
    "NQ=F": "Nasdaq 100 futures proxy",
    "ES=F": "S&P 500 futures proxy",
    "BTC-USD": "Bitcoin proxy",
    "ETH-USD": "Ethereum proxy",
}


@dataclass
class CompetitionPlan:
    starting_equity: float
    target_equity: float
    days_left: int
    required_total_return_percent: float
    required_daily_return_percent: float
    max_risk_per_attempt: float
    risk_percent_per_attempt: float
    pressure_level: str
    note: str


@dataclass
class InstrumentReport:
    rank: int
    symbol: str
    label: str
    last_close: float
    bias: str
    competition_score: float
    risk_regime: str
    expansion_state: str
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
    risk_note: str


def parse_symbol_list(raw: str) -> list[str]:
    symbols = [item.strip() for item in raw.split(",") if item.strip()]
    if not symbols:
        raise ValueError("Provide at least one symbol.")
    return symbols


def load_market_data(symbol: str, period: str, interval: str, csv_path: Optional[str]) -> pd.DataFrame:
    if csv_path:
        data = pd.read_csv(csv_path)
        normalized = {str(c).strip().lower(): c for c in data.columns}
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in normalized]
        if missing:
            raise ValueError(f"CSV is missing required OHLC columns: {missing}")
        rename_map = {normalized[c]: c.title() for c in required}
        return data.rename(columns=rename_map).dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)

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
    df["atr_percent"] = df["atr_14"] / close.replace(0, np.nan)
    df["atr_percent_median_60"] = df["atr_percent"].rolling(60).median()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["momentum_20"] = close.pct_change(20)
    df["momentum_5"] = close.pct_change(5)
    df["return"] = close.pct_change()
    df["realized_volatility_20"] = df["return"].rolling(20).std() * math.sqrt(252)
    df["rsi_14"] = rsi(close)
    return df


def classify_risk(atr_percent: float, volatility: float) -> str:
    if atr_percent > 0.08 or volatility > 0.90:
        return "extreme"
    if atr_percent > 0.04 or volatility > 0.55:
        return "high"
    if atr_percent < 0.012 and volatility < 0.22:
        return "compressed"
    return "normal"


def classify_expansion(atr_percent: float, median_atr_percent: float) -> str:
    if not math.isfinite(median_atr_percent) or median_atr_percent <= 0:
        return "unknown"
    ratio = atr_percent / median_atr_percent
    if ratio >= 1.35:
        return "expanded"
    if ratio >= 1.10:
        return "expanding"
    if ratio <= 0.75:
        return "compressed"
    return "steady"


def score_instrument(data: pd.DataFrame, symbol: str, rank: int) -> InstrumentReport:
    if len(data) < 80:
        raise ValueError("At least 80 rows are required for stable competition calculations.")

    df = add_indicators(data).dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("Indicator output is empty. Provide more OHLCV history.")

    last = df.iloc[-1]
    close = float(last["Close"])
    atr = float(last["atr_14"])
    atr_percent = float(last["atr_percent"])
    atr_median = float(last["atr_percent_median_60"])
    momentum_20 = float(last["momentum_20"])
    momentum_5 = float(last["momentum_5"])
    volatility = float(last["realized_volatility_20"])
    rsi_value = float(last["rsi_14"])
    ema_12 = float(last["ema_12"])
    ema_26 = float(last["ema_26"])

    trend_component = 1.0 if ema_12 > ema_26 else -1.0
    momentum_component = float(np.tanh(momentum_20 * 8))
    short_momentum_component = float(np.tanh(momentum_5 * 15))
    rsi_component = float(np.clip((rsi_value - 50) / 25, -1, 1))

    directional_raw = (
        0.40 * trend_component
        + 0.30 * momentum_component
        + 0.15 * short_momentum_component
        + 0.15 * rsi_component
    )

    risk_regime = classify_risk(atr_percent, volatility)
    expansion_state = classify_expansion(atr_percent, atr_median)

    volatility_boost = min(1.0, max(0.0, atr_percent / 0.04))
    consistency = min(1.0, abs(directional_raw))
    expansion_bonus = {"expanded": 0.12, "expanding": 0.08, "steady": 0.03, "compressed": -0.05}.get(expansion_state, 0.0)
    risk_penalty = {"extreme": 0.20, "high": 0.08, "normal": 0.0, "compressed": 0.03}.get(risk_regime, 0.0)

    competition_score = 100 * np.clip(
        0.55 * consistency + 0.25 * volatility_boost + expansion_bonus - risk_penalty,
        0,
        1,
    )

    if directional_raw > 0.25:
        bias = "bullish scenario"
    elif directional_raw < -0.25:
        bias = "bearish scenario"
    else:
        bias = "mixed / wait"

    return InstrumentReport(
        rank=rank,
        symbol=symbol,
        label=SYMBOL_LABELS.get(symbol, "custom market proxy"),
        last_close=round(close, 8),
        bias=bias,
        competition_score=round(float(competition_score), 2),
        risk_regime=risk_regime,
        expansion_state=expansion_state,
        atr_14=round(atr, 8),
        atr_percent=round(atr_percent, 6),
        momentum_20=round(momentum_20, 6),
        realized_volatility_20=round(volatility, 6),
        rsi_14=round(rsi_value, 4),
        ema_12=round(ema_12, 8),
        ema_26=round(ema_26, 8),
        lower_atr_band_1=round(close - 1.0 * atr, 8),
        upper_atr_band_1=round(close + 1.0 * atr, 8),
        lower_atr_band_2=round(close - 2.0 * atr, 8),
        upper_atr_band_2=round(close + 2.0 * atr, 8),
        risk_note="ATR bands are scenario references for demo competition review, not entry/TP/SL orders.",
    )


def build_competition_plan(
    account_equity: float,
    target_equity: float,
    days_left: int,
    risk_percent: float,
) -> CompetitionPlan:
    if account_equity <= 0:
        raise ValueError("--account-equity must be greater than zero.")
    if target_equity <= 0:
        raise ValueError("--target-equity must be greater than zero.")
    if days_left <= 0:
        raise ValueError("--days-left must be greater than zero.")
    if risk_percent <= 0:
        raise ValueError("--risk-pct must be greater than zero.")

    required_total = (target_equity / account_equity) - 1
    required_daily = (target_equity / account_equity) ** (1 / days_left) - 1
    max_risk = account_equity * risk_percent / 100

    if required_daily >= 0.10:
        pressure = "extreme"
    elif required_daily >= 0.05:
        pressure = "very aggressive"
    elif required_daily >= 0.02:
        pressure = "aggressive"
    else:
        pressure = "controlled"

    return CompetitionPlan(
        starting_equity=round(account_equity, 2),
        target_equity=round(target_equity, 2),
        days_left=days_left,
        required_total_return_percent=round(required_total * 100, 2),
        required_daily_return_percent=round(required_daily * 100, 2),
        max_risk_per_attempt=round(max_risk, 2),
        risk_percent_per_attempt=round(risk_percent, 2),
        pressure_level=pressure,
        note="Leaderboard math is planning context only; it does not create a predictable edge.",
    )


def scan_symbols(
    symbols: Iterable[str],
    period: str,
    interval: str,
    csv_path: Optional[str],
) -> tuple[list[InstrumentReport], list[str]]:
    symbol_list = list(symbols)
    reports: list[InstrumentReport] = []
    errors: list[str] = []

    for symbol in symbol_list:
        try:
            data = load_market_data(symbol, period, interval, csv_path if len(symbol_list) == 1 else None)
            report = score_instrument(data, symbol, rank=0)
            reports.append(report)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    reports.sort(key=lambda item: item.competition_score, reverse=True)
    reranked = []
    for idx, report in enumerate(reports, start=1):
        payload = asdict(report)
        payload["rank"] = idx
        reranked.append(InstrumentReport(**payload))
    return reranked, errors


def render(plan: CompetitionPlan, reports: list[InstrumentReport], errors: list[str], top: int, as_json: bool) -> None:
    payload = {
        "competition_plan": asdict(plan),
        "watchlist": [asdict(item) for item in reports[:top]],
        "errors": errors,
    }

    if as_json or Console is None or Table is None or Panel is None:
        print(json.dumps(payload, indent=2))
        return

    console = Console()
    console.print(
        Panel.fit(
            "Khabib Demo Competition Assistant\n"
            "Multi-asset scenario scanner for leaderboard planning. No guaranteed outcome."
        )
    )

    plan_table = Table(title="Competition Math", show_lines=True)
    plan_table.add_column("Metric")
    plan_table.add_column("Value")
    for key, value in asdict(plan).items():
        plan_table.add_row(key, str(value))
    console.print(plan_table)

    table = Table(title="Ranked Watchlist", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("Symbol")
    table.add_column("Market")
    table.add_column("Bias")
    table.add_column("Score", justify="right")
    table.add_column("Regime")
    table.add_column("Expansion")
    table.add_column("Last", justify="right")
    table.add_column("ATR%", justify="right")
    table.add_column("Mom20", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("1 ATR Scenario")

    for item in reports[:top]:
        table.add_row(
            str(item.rank),
            item.symbol,
            item.label,
            item.bias,
            f"{item.competition_score:.2f}",
            item.risk_regime,
            item.expansion_state,
            str(item.last_close),
            f"{item.atr_percent * 100:.2f}%",
            f"{item.momentum_20 * 100:.2f}%",
            f"{item.rsi_14:.2f}",
            f"{item.lower_atr_band_1} ↔ {item.upper_atr_band_1}",
        )
    console.print(table)

    if errors:
        error_table = Table(title="Skipped Symbols")
        error_table.add_column("Reason")
        for err in errors:
            error_table.add_row(err)
        console.print(error_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-asset demo competition scanner for trend, volatility, ATR, and leaderboard planning."
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated yfinance symbols or market proxies.",
    )
    parser.add_argument("--period", default="6mo", help="Download period. Default: 6mo")
    parser.add_argument("--interval", default="1d", help="Download interval. Default: 1d")
    parser.add_argument("--csv", default=None, help="Optional local OHLCV CSV path for a single symbol")
    parser.add_argument("--account-equity", type=float, default=50_000, help="Current demo equity. Default: 50000")
    parser.add_argument("--target-equity", type=float, default=1_000_000, help="Leaderboard target equity. Default: 1000000")
    parser.add_argument("--days-left", type=int, default=30, help="Competition days remaining. Default: 30")
    parser.add_argument("--risk-pct", type=float, default=1.0, help="Max planning risk per attempt. Default: 1.0")
    parser.add_argument("--top", type=int, default=8, help="Number of ranked candidates to display. Default: 8")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = parse_symbol_list(args.symbols)
    if args.csv and len(symbols) != 1:
        raise ValueError("--csv can only be used with exactly one symbol.")

    plan = build_competition_plan(
        account_equity=args.account_equity,
        target_equity=args.target_equity,
        days_left=args.days_left,
        risk_percent=args.risk_pct,
    )
    reports, errors = scan_symbols(symbols, args.period, args.interval, args.csv)
    if not reports:
        raise RuntimeError("No symbols could be analyzed. Check internet access, symbol names, or CSV input.")

    render(plan, reports, errors, args.top, args.json)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
