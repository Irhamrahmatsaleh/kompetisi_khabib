#!/usr/bin/env python3
"""Khabib Demo Competition Assistant.

A multi-asset CFD/FX/crypto scenario scanner for demo trading competitions.
It ranks instruments, estimates whether a market is tradable right now, and
prints scenario-based BUY/SELL/WAIT plans with Entry, Stop Loss, TP1, and TP2.

The output is for demo-competition planning and risk review. It is not a
profit guarantee and it cannot read the broker's live order-permission flags.
Always verify the actual BUY/SELL buttons, spread, margin, and contract size
inside the trading platform before opening a position.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, time, timezone
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


@dataclass(frozen=True)
class MarketSpec:
    broker_symbol: str
    data_symbol: str
    label: str
    asset_class: str
    schedule_note: str


MARKET_SPECS: dict[str, MarketSpec] = {
    "XAUUSD": MarketSpec("XAUUSD", "GC=F", "Gold vs US Dollar", "metals", "Usually closed on weekends and around the daily metals break; broker schedule may vary."),
    "XAGUSD": MarketSpec("XAGUSD", "SI=F", "Silver vs US Dollar", "metals", "Usually closed on weekends and around the daily metals break; broker schedule may vary."),
    "USOIL": MarketSpec("USOIL", "CL=F", "Crude Oil CFD proxy", "energy", "Usually closed on weekends and around the daily energy break; broker schedule may vary."),
    "NAS100": MarketSpec("NAS100", "NQ=F", "Nasdaq 100 CFD proxy", "index", "Usually closed on weekends and around the daily index break; broker schedule may vary."),
    "US500": MarketSpec("US500", "ES=F", "S&P 500 CFD proxy", "index", "Usually closed on weekends and around the daily index break; broker schedule may vary."),
    "EURUSD": MarketSpec("EURUSD", "EURUSD=X", "Euro vs US Dollar", "fx", "Usually open from Sunday night to Friday night UTC; broker schedule may vary."),
    "GBPUSD": MarketSpec("GBPUSD", "GBPUSD=X", "British Pound vs US Dollar", "fx", "Usually open from Sunday night to Friday night UTC; broker schedule may vary."),
    "USDJPY": MarketSpec("USDJPY", "JPY=X", "US Dollar vs Japanese Yen", "fx", "Usually open from Sunday night to Friday night UTC; broker schedule may vary."),
    "GBPJPY": MarketSpec("GBPJPY", "GBPJPY=X", "British Pound vs Japanese Yen", "fx", "Usually open from Sunday night to Friday night UTC; broker schedule may vary."),
    "BTCUSD": MarketSpec("BTCUSD", "BTC-USD", "Bitcoin vs US Dollar", "crypto", "Usually available 24/7; broker maintenance may still pause trading."),
    "ETHUSD": MarketSpec("ETHUSD", "ETH-USD", "Ethereum vs US Dollar", "crypto", "Usually available 24/7; broker maintenance may still pause trading."),
}

DEFAULT_SYMBOLS = [
    "XAUUSD",
    "NAS100",
    "USOIL",
    "GBPJPY",
    "GBPUSD",
    "EURUSD",
    "USDJPY",
    "BTCUSD",
    "ETHUSD",
]


@dataclass
class CompetitionPlan:
    account_equity: float
    target_equity: float
    days_left: int
    required_total_return_percent: float
    required_daily_return_percent: float
    max_loss_budget_per_attempt: float
    risk_percent_per_attempt: float
    pressure_level: str
    operating_rule: str


@dataclass
class TradeScenario:
    rank: int
    broker_symbol: str
    data_symbol: str
    market: str
    market_status_estimate: str
    can_buy_now_estimate: bool
    can_sell_now_estimate: bool
    action: str
    setup_direction: str
    entry_reference: float
    stop_loss: Optional[float]
    take_profit_1: Optional[float]
    take_profit_2: Optional[float]
    risk_reward_tp1: Optional[float]
    risk_reward_tp2: Optional[float]
    max_loss_budget: float
    units_at_stop_if_contract_value_1: Optional[float]
    score: float
    edge_score: float
    risk_regime: str
    expansion_state: str
    atr_14: float
    atr_percent: float
    momentum_5: float
    momentum_20: float
    realized_volatility_20: float
    rsi_14: float
    ema_12: float
    ema_26: float
    reason: str
    warning: str


def parse_symbol_list(raw: str) -> list[str]:
    symbols = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not symbols:
        raise ValueError("Provide at least one symbol.")
    return symbols


def resolve_market(symbol: str) -> MarketSpec:
    key = symbol.upper().strip()
    if key in MARKET_SPECS:
        return MARKET_SPECS[key]

    # Allow direct yfinance symbols for custom experiments.
    return MarketSpec(
        broker_symbol=key,
        data_symbol=symbol.strip(),
        label="Custom market proxy",
        asset_class="custom",
        schedule_note="Unknown broker schedule; verify BUY/SELL availability inside the platform.",
    )


def load_market_data(spec: MarketSpec, period: str, interval: str, csv_path: Optional[str]) -> pd.DataFrame:
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

    data = yf.download(spec.data_symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    if data.empty:
        raise RuntimeError(f"No market data returned for {spec.broker_symbol} using data proxy {spec.data_symbol}")
    return data.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(data: pd.DataFrame, interval: str) -> pd.DataFrame:
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

    annualizer = 252
    if interval.endswith("h") or interval.endswith("m"):
        annualizer = 252 * 6.5

    df["atr_14"] = true_range.rolling(14).mean()
    df["atr_percent"] = df["atr_14"] / close.replace(0, np.nan)
    df["atr_percent_median_60"] = df["atr_percent"].rolling(60).median()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["momentum_20"] = close.pct_change(20)
    df["momentum_5"] = close.pct_change(5)
    df["return"] = close.pct_change()
    df["realized_volatility_20"] = df["return"].rolling(20).std() * math.sqrt(annualizer)
    df["rsi_14"] = rsi(close)
    return df


def classify_risk(atr_percent: float, volatility: float) -> str:
    if atr_percent > 0.08 or volatility > 1.10:
        return "extreme"
    if atr_percent > 0.04 or volatility > 0.70:
        return "high"
    if atr_percent < 0.010 and volatility < 0.18:
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


def market_status_estimate(spec: MarketSpec, now_utc: Optional[datetime] = None) -> tuple[str, bool, bool]:
    """Estimate whether BUY/SELL can be opened.

    This is intentionally conservative. The broker can still disable trading due
    to holidays, maintenance, symbol mode, low liquidity, or account restrictions.
    """
    now = now_utc or datetime.now(timezone.utc)
    weekday = now.weekday()  # Monday=0, Sunday=6
    current = now.time()

    if spec.asset_class == "crypto":
        return "OPEN_ESTIMATE_24_7", True, True

    if spec.asset_class in {"fx", "metals", "energy", "index"}:
        weekend_closed = weekday == 5 or (weekday == 4 and current >= time(22, 0)) or (weekday == 6 and current < time(22, 0))
        if weekend_closed:
            return "CLOSED_ESTIMATE_WEEKEND", False, False

        # Many CFD sessions pause around the futures/rollover maintenance window.
        if spec.asset_class in {"metals", "energy", "index"} and time(21, 0) <= current < time(22, 0):
            return "CLOSED_ESTIMATE_DAILY_BREAK", False, False

        return "OPEN_ESTIMATE", True, True

    return "UNKNOWN_VERIFY_PLATFORM", False, False


def round_price(value: Optional[float], reference: float) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    abs_ref = abs(reference)
    if abs_ref >= 1000:
        digits = 2
    elif abs_ref >= 100:
        digits = 3
    elif abs_ref >= 10:
        digits = 4
    elif abs_ref >= 1:
        digits = 5
    else:
        digits = 6
    return round(float(value), digits)


def build_trade_scenario(
    data: pd.DataFrame,
    spec: MarketSpec,
    rank: int,
    interval: str,
    account_equity: float,
    risk_percent: float,
    min_edge: float,
    sl_atr: float,
    tp1_atr: float,
    tp2_atr: float,
) -> TradeScenario:
    if len(data) < 80:
        raise ValueError("At least 80 rows are required for stable scenario calculations.")

    df = add_indicators(data, interval).dropna().reset_index(drop=True)
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
    ema_50 = float(last["ema_50"])

    trend_component = 1.0 if ema_12 > ema_26 else -1.0
    structure_component = 0.50 if close > ema_50 else -0.50
    momentum_component = float(np.tanh(momentum_20 * 8))
    short_momentum_component = float(np.tanh(momentum_5 * 15))
    rsi_component = float(np.clip((rsi_value - 50) / 25, -1, 1))

    directional_raw = (
        0.35 * trend_component
        + 0.20 * structure_component
        + 0.25 * momentum_component
        + 0.12 * short_momentum_component
        + 0.08 * rsi_component
    )
    edge_score = float(np.clip(abs(directional_raw), 0, 1))

    risk_regime = classify_risk(atr_percent, volatility)
    expansion_state = classify_expansion(atr_percent, atr_median)
    status, can_buy, can_sell = market_status_estimate(spec)

    volatility_boost = min(1.0, max(0.0, atr_percent / 0.035))
    expansion_bonus = {"expanded": 0.15, "expanding": 0.10, "steady": 0.04, "compressed": -0.08}.get(expansion_state, 0.0)
    risk_penalty = {"extreme": 0.25, "high": 0.10, "normal": 0.0, "compressed": 0.04}.get(risk_regime, 0.0)
    score = 100 * np.clip(0.60 * edge_score + 0.25 * volatility_boost + expansion_bonus - risk_penalty, 0, 1)

    if directional_raw >= min_edge:
        setup_direction = "BUY/LONG"
    elif directional_raw <= -min_edge:
        setup_direction = "SELL/SHORT"
    else:
        setup_direction = "WAIT"

    entry = close
    stop_loss: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    rr1: Optional[float]
    rr2: Optional[float]

    if setup_direction == "BUY/LONG":
        stop_loss = close - sl_atr * atr
        tp1 = close + tp1_atr * atr
        tp2 = close + tp2_atr * atr
        rr1 = abs(tp1 - close) / abs(close - stop_loss)
        rr2 = abs(tp2 - close) / abs(close - stop_loss)
    elif setup_direction == "SELL/SHORT":
        stop_loss = close + sl_atr * atr
        tp1 = close - tp1_atr * atr
        tp2 = close - tp2_atr * atr
        rr1 = abs(close - tp1) / abs(stop_loss - close)
        rr2 = abs(close - tp2) / abs(stop_loss - close)
    else:
        stop_loss = None
        tp1 = None
        tp2 = None
        rr1 = None
        rr2 = None

    max_loss_budget = account_equity * risk_percent / 100
    stop_distance = abs(entry - stop_loss) if stop_loss is not None else None
    units_at_stop = max_loss_budget / stop_distance if stop_distance and stop_distance > 0 else None

    if setup_direction == "WAIT":
        action = "WAIT - edge too weak"
    elif status.startswith("CLOSED"):
        action = f"WAIT - market closed; prepare {setup_direction} setup"
    elif setup_direction == "BUY/LONG" and not can_buy:
        action = "WAIT - BUY not available; verify platform"
    elif setup_direction == "SELL/SHORT" and not can_sell:
        action = "WAIT - SELL not available; verify platform"
    else:
        action = setup_direction

    reason_bits = []
    reason_bits.append("EMA12>EMA26" if ema_12 > ema_26 else "EMA12<EMA26")
    reason_bits.append("close>EMA50" if close > ema_50 else "close<EMA50")
    reason_bits.append(f"RSI={rsi_value:.1f}")
    reason_bits.append(f"M5={momentum_5 * 100:.2f}%")
    reason_bits.append(f"M20={momentum_20 * 100:.2f}%")
    reason_bits.append(f"ATR={atr_percent * 100:.2f}%")
    reason = "; ".join(reason_bits)

    warning = (
        "Broker can still disable BUY/SELL because of weekend, daily break, holiday, spread widening, "
        "symbol close-only mode, or margin limits. Use the platform buttons as final truth."
    )

    return TradeScenario(
        rank=rank,
        broker_symbol=spec.broker_symbol,
        data_symbol=spec.data_symbol,
        market=spec.label,
        market_status_estimate=status,
        can_buy_now_estimate=can_buy,
        can_sell_now_estimate=can_sell,
        action=action,
        setup_direction=setup_direction,
        entry_reference=round_price(entry, close),
        stop_loss=round_price(stop_loss, close),
        take_profit_1=round_price(tp1, close),
        take_profit_2=round_price(tp2, close),
        risk_reward_tp1=round(float(rr1), 2) if rr1 is not None else None,
        risk_reward_tp2=round(float(rr2), 2) if rr2 is not None else None,
        max_loss_budget=round(max_loss_budget, 2),
        units_at_stop_if_contract_value_1=round(float(units_at_stop), 4) if units_at_stop is not None else None,
        score=round(float(score), 2),
        edge_score=round(edge_score, 4),
        risk_regime=risk_regime,
        expansion_state=expansion_state,
        atr_14=round_price(atr, close) or 0.0,
        atr_percent=round(atr_percent, 6),
        momentum_5=round(momentum_5, 6),
        momentum_20=round(momentum_20, 6),
        realized_volatility_20=round(volatility, 6),
        rsi_14=round(rsi_value, 4),
        ema_12=round_price(ema_12, close) or 0.0,
        ema_26=round_price(ema_26, close) or 0.0,
        reason=reason,
        warning=warning,
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
    max_loss = account_equity * risk_percent / 100

    if required_daily >= 0.10:
        pressure = "extreme"
        rule = "Only take the cleanest high-score setups; random oversized entries usually destroy ranking stability."
    elif required_daily >= 0.05:
        pressure = "very aggressive"
        rule = "Prioritize volatile leaders, use predefined SL, and stop trading after consecutive invalidations."
    elif required_daily >= 0.02:
        pressure = "aggressive"
        rule = "Trade only when direction, momentum, and ATR expansion agree."
    else:
        pressure = "controlled"
        rule = "Preserve equity first; compound only high-quality setups."

    return CompetitionPlan(
        account_equity=round(account_equity, 2),
        target_equity=round(target_equity, 2),
        days_left=days_left,
        required_total_return_percent=round(required_total * 100, 2),
        required_daily_return_percent=round(required_daily * 100, 2),
        max_loss_budget_per_attempt=round(max_loss, 2),
        risk_percent_per_attempt=round(risk_percent, 2),
        pressure_level=pressure,
        operating_rule=rule,
    )


def scan_symbols(
    symbols: Iterable[str],
    period: str,
    interval: str,
    csv_path: Optional[str],
    account_equity: float,
    risk_percent: float,
    min_edge: float,
    sl_atr: float,
    tp1_atr: float,
    tp2_atr: float,
) -> tuple[list[TradeScenario], list[str]]:
    symbol_list = list(symbols)
    scenarios: list[TradeScenario] = []
    errors: list[str] = []

    for symbol in symbol_list:
        try:
            spec = resolve_market(symbol)
            data = load_market_data(spec, period, interval, csv_path if len(symbol_list) == 1 else None)
            scenario = build_trade_scenario(
                data=data,
                spec=spec,
                rank=0,
                interval=interval,
                account_equity=account_equity,
                risk_percent=risk_percent,
                min_edge=min_edge,
                sl_atr=sl_atr,
                tp1_atr=tp1_atr,
                tp2_atr=tp2_atr,
            )
            scenarios.append(scenario)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    scenarios.sort(key=lambda item: item.score, reverse=True)
    reranked: list[TradeScenario] = []
    for idx, item in enumerate(scenarios, start=1):
        payload = asdict(item)
        payload["rank"] = idx
        reranked.append(TradeScenario(**payload))
    return reranked, errors


def render(plan: CompetitionPlan, scenarios: list[TradeScenario], errors: list[str], top: int, as_json: bool) -> None:
    payload = {
        "competition_plan": asdict(plan),
        "ranked_trade_scenarios": [asdict(item) for item in scenarios[:top]],
        "errors": errors,
    }

    if as_json or Console is None or Table is None or Panel is None:
        print(json.dumps(payload, indent=2))
        return

    console = Console()
    console.print(
        Panel.fit(
            "Khabib Demo Competition Assistant\n"
            "BUY = long, SELL = short. Entry/SL/TP are ATR-based demo scenarios, not guaranteed outcomes."
        )
    )

    plan_table = Table(title="Competition Math", show_lines=True)
    plan_table.add_column("Metric")
    plan_table.add_column("Value")
    for key, value in asdict(plan).items():
        plan_table.add_row(key, str(value))
    console.print(plan_table)

    table = Table(title="Ranked Trade Scenarios", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("Symbol")
    table.add_column("Status")
    table.add_column("Action")
    table.add_column("Entry", justify="right")
    table.add_column("SL", justify="right")
    table.add_column("TP1", justify="right")
    table.add_column("TP2", justify="right")
    table.add_column("RR", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Risk")
    table.add_column("Why")

    for item in scenarios[:top]:
        rr = "-" if item.risk_reward_tp1 is None else f"{item.risk_reward_tp1}/{item.risk_reward_tp2}"
        table.add_row(
            str(item.rank),
            item.broker_symbol,
            item.market_status_estimate,
            item.action,
            str(item.entry_reference),
            str(item.stop_loss) if item.stop_loss is not None else "-",
            str(item.take_profit_1) if item.take_profit_1 is not None else "-",
            str(item.take_profit_2) if item.take_profit_2 is not None else "-",
            rr,
            f"{item.score:.2f}",
            item.risk_regime,
            item.reason,
        )
    console.print(table)

    sizing_table = Table(title="Sizing Helper", show_lines=True)
    sizing_table.add_column("Symbol")
    sizing_table.add_column("Max Loss Budget")
    sizing_table.add_column("Units if contract value = 1")
    sizing_table.add_column("Note")
    for item in scenarios[:top]:
        sizing_table.add_row(
            item.broker_symbol,
            str(item.max_loss_budget),
            str(item.units_at_stop_if_contract_value_1) if item.units_at_stop_if_contract_value_1 is not None else "-",
            "Convert to broker lots using the platform contract-size calculator.",
        )
    console.print(sizing_table)

    if errors:
        error_table = Table(title="Skipped Symbols")
        error_table.add_column("Reason")
        for err in errors:
            error_table.add_row(err)
        console.print(error_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo competition scanner: ranked market setups with BUY/SELL/WAIT, Entry, SL, TP1, and TP2."
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated broker-style symbols: XAUUSD,NAS100,USOIL,GBPJPY,GBPUSD,EURUSD,USDJPY,BTCUSD,ETHUSD. Direct yfinance symbols also work.",
    )
    parser.add_argument("--period", default="60d", help="Download period. Default: 60d")
    parser.add_argument("--interval", default="1h", help="Download interval. Default: 1h")
    parser.add_argument("--csv", default=None, help="Optional local OHLCV CSV path for a single symbol")
    parser.add_argument("--account-equity", type=float, default=50_000, help="Current demo equity. Default: 50000")
    parser.add_argument("--target-equity", type=float, default=1_000_000, help="Leaderboard target equity. Default: 1000000")
    parser.add_argument("--days-left", type=int, default=30, help="Competition days remaining. Default: 30")
    parser.add_argument("--risk-pct", type=float, default=1.0, help="Max loss budget per scenario as percent of equity. Default: 1.0")
    parser.add_argument("--min-edge", type=float, default=0.30, help="Minimum directional edge required for BUY/SELL. Default: 0.30")
    parser.add_argument("--sl-atr", type=float, default=1.0, help="Stop Loss distance in ATR. Default: 1.0")
    parser.add_argument("--tp1-atr", type=float, default=1.5, help="Take Profit 1 distance in ATR. Default: 1.5")
    parser.add_argument("--tp2-atr", type=float, default=2.5, help="Take Profit 2 distance in ATR. Default: 2.5")
    parser.add_argument("--top", type=int, default=8, help="Number of ranked scenarios to display. Default: 8")
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
    scenarios, errors = scan_symbols(
        symbols=symbols,
        period=args.period,
        interval=args.interval,
        csv_path=args.csv,
        account_equity=args.account_equity,
        risk_percent=args.risk_pct,
        min_edge=args.min_edge,
        sl_atr=args.sl_atr,
        tp1_atr=args.tp1_atr,
        tp2_atr=args.tp2_atr,
    )
    if not scenarios:
        raise RuntimeError("No symbols could be analyzed. Check internet access, symbol names, or CSV input.")

    render(plan, scenarios, errors, args.top, args.json)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
