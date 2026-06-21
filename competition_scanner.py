#!/usr/bin/env python3
"""Broad market scanner for demo trading competitions.

BUY = long, SELL = short. The scanner compares many market proxies and ranks
where trend, momentum, ATR expansion, and volatility agree. It cannot see the
broker's private live permissions, so the platform buttons, spread, margin, and
contract size are the final truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, time, timezone
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


@dataclass(frozen=True)
class Market:
    symbol: str
    proxy: str
    asset: str
    label: str
    can_buy: bool = True
    can_sell: bool = True


@dataclass
class Scenario:
    rank: int
    symbol: str
    proxy: str
    asset: str
    status: str
    action: str
    entry: float
    sl: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    rr1: Optional[float]
    rr2: Optional[float]
    score: float
    edge: float
    atr: float
    atr_pct: float
    rsi: float
    mom5: float
    mom20: float
    risk: str
    why: str
    max_loss_budget: float
    units_if_contract_value_1: Optional[float]


def m(symbol: str, proxy: str, asset: str, label: str = "") -> Market:
    return Market(symbol, proxy, asset, label or symbol)


FX = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X", "USDCHF": "CHF=X",
    "USDCAD": "CAD=X", "AUDUSD": "AUDUSD=X", "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X", "GBPJPY": "GBPJPY=X", "AUDJPY": "AUDJPY=X", "CADJPY": "CADJPY=X",
    "CHFJPY": "CHFJPY=X", "EURCHF": "EURCHF=X", "EURCAD": "EURCAD=X", "EURAUD": "EURAUD=X",
    "GBPAUD": "GBPAUD=X", "GBPCAD": "GBPCAD=X", "GBPCHF": "GBPCHF=X", "AUDCAD": "AUDCAD=X",
    "AUDCHF": "AUDCHF=X", "AUDNZD": "AUDNZD=X", "NZDJPY": "NZDJPY=X", "NZDCHF": "NZDCHF=X",
    "NZDCAD": "NZDCAD=X", "CADCHF": "CADCHF=X",
}
COMMODITIES = {
    "XAUUSD": ("GC=F", "metals"), "XAGUSD": ("SI=F", "metals"), "XPTUSD": ("PL=F", "metals"),
    "XPDUSD": ("PA=F", "metals"), "USOIL": ("CL=F", "energy"), "UKOIL": ("BZ=F", "energy"),
    "NATGAS": ("NG=F", "energy"), "COPPER": ("HG=F", "commodity"), "CORN": ("ZC=F", "commodity"),
    "WHEAT": ("ZW=F", "commodity"), "SOYBEAN": ("ZS=F", "commodity"),
}
INDICES = {
    "NAS100": "NQ=F", "US500": "ES=F", "US30": "YM=F", "RUSSELL2000": "RTY=F", "VIX": "^VIX",
    "JPN225": "^N225", "HK50": "^HSI", "UK100": "^FTSE", "GER40": "^GDAXI", "FRA40": "^FCHI", "AUS200": "^AXJO",
}
CRYPTO = {
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD", "XRPUSD": "XRP-USD", "BNBUSD": "BNB-USD",
    "DOGEUSD": "DOGE-USD", "ADAUSD": "ADA-USD", "AVAXUSD": "AVAX-USD", "LINKUSD": "LINK-USD", "DOTUSD": "DOT-USD",
}
STOCKS = """
AAPL MSFT NVDA AMZN META GOOGL GOOG TSLA AVGO JPM LLY V MA NFLX COST WMT ORCL AMD BAC KO PEP DIS MCD IBM GE CAT BA GS MS XOM
CVX COP SLB T VZ INTC QCOM CRM ADBE PYPL NKE SBUX BABA NIO PLTR HOOD COIN MSTR SMCI SHOP UBER ABNB RBLX SNOW PANW CRWD NOW MU TXN
AMAT LRCX KLAC ASML TSM ARM MRVL DELL HPQ CSCO ANET JNJ UNH MRK PFE ABBV ABT TMO DHR ISRG REGN HD LOW TGT TJX LULU CMG YUM MAR
BKNG RCL DAL UAL AAL GM F RIVN LCID TM HMC SONY WFC C AXP BLK SCHW BX SPGI ICE CME MCO NEE DUK SO AEP ENPH FSLR FCX NEM GOLD AA
LIN APD SHW MMM HON RTX LMT NOC GD DE
""".split()
ETFS = """SPY QQQ DIA IWM XLK XLF XLE XLV XLY XLI XLP XLU XLB SMH SOXX ARKK GLD SLV USO UNG TLT IEF HYG LQD EEM EFA FXI EWJ EWU EWG""".split()

MARKETS: dict[str, Market] = {}
MARKETS.update({s: m(s, p, "fx", s) for s, p in FX.items()})
MARKETS.update({s: m(s, p, a, s) for s, (p, a) in COMMODITIES.items()})
MARKETS.update({s: m(s, p, "index", s) for s, p in INDICES.items()})
MARKETS.update({s: m(s, p, "crypto", s) for s, p in CRYPTO.items()})
MARKETS.update({s: m(s, s, "stock", f"{s} stock CFD proxy") for s in STOCKS})
MARKETS.update({s: m(s, s, "etf", f"{s} ETF CFD proxy") for s in ETFS})
CORE = "XAUUSD XAGUSD USOIL NAS100 US500 US30 EURUSD GBPUSD USDJPY GBPJPY BTCUSD ETHUSD AAPL MSFT NVDA TSLA AMZN META AMD COIN".split()


def norm(symbol: str) -> str:
    return symbol.strip().upper()


def symbols_from_arg(raw: str) -> list[str]:
    key = norm(raw)
    if key in {"ALL", "BROAD", "HUNDREDS"}:
        return list(MARKETS.keys())
    if key in {"CORE", "DEFAULT"}:
        return CORE
    return [norm(x) for x in raw.split(",") if x.strip()]


def symbols_from_csv(path: str) -> list[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "symbol" not in reader.fieldnames:
            raise ValueError("CSV universe must contain a 'symbol' column")
        out = [norm(row.get("symbol", "")) for row in reader if row.get("symbol")]
    if not out:
        raise ValueError("CSV universe is empty")
    return out


def resolve(symbol: str, blocked_buy: set[str], blocked_sell: set[str]) -> Market:
    key = norm(symbol)
    base = MARKETS.get(key, m(key, symbol.strip(), "custom", "custom yfinance proxy"))
    return Market(base.symbol, base.proxy, base.asset, base.label, base.can_buy and key not in blocked_buy, base.can_sell and key not in blocked_sell)


def status_for(market: Market) -> tuple[str, bool, bool]:
    now = datetime.now(timezone.utc)
    wd = now.weekday()
    t = now.time()
    if market.asset == "crypto":
        return "OPEN_ESTIMATE_24_7", market.can_buy, market.can_sell
    if market.asset in {"fx", "metals", "energy", "commodity", "index"}:
        weekend = wd == 5 or (wd == 4 and t >= time(22, 0)) or (wd == 6 and t < time(22, 0))
        if weekend:
            return "CLOSED_ESTIMATE_WEEKEND", False, False
        if market.asset in {"metals", "energy", "commodity", "index"} and time(21, 0) <= t < time(22, 0):
            return "CLOSED_ESTIMATE_DAILY_BREAK", False, False
        return "OPEN_ESTIMATE", market.can_buy, market.can_sell
    if market.asset in {"stock", "etf"}:
        if wd >= 5:
            return "CLOSED_ESTIMATE_WEEKEND", False, False
        if time(14, 30) <= t < time(21, 0):
            return "OPEN_ESTIMATE_US_SESSION", market.can_buy, market.can_sell
        return "CLOSED_ESTIMATE_OUTSIDE_STOCK_SESSION", False, False
    return "UNKNOWN_VERIFY_PLATFORM", False, False


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        for level in (0, 1):
            names = {str(c).title() for c in out.columns.get_level_values(level)}
            if {"Open", "High", "Low", "Close"}.issubset(names):
                out.columns = out.columns.get_level_values(level)
                break
    cols = {str(c).strip().lower(): c for c in out.columns}
    if not {"open", "high", "low", "close"}.issubset(cols):
        return pd.DataFrame()
    out = out.rename(columns={cols[k]: k.title() for k in ["open", "high", "low", "close"]})
    return out.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)


def fetch_batch(markets: list[Market], period: str, interval: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    if yf is None:
        raise RuntimeError("yfinance unavailable. Run make install first.")
    proxies = sorted({x.proxy for x in markets})
    raw = yf.download(" ".join(proxies), period=period, interval=interval, group_by="ticker", auto_adjust=False, threads=True, progress=False)
    frames, errors = {}, []
    if len(proxies) == 1:
        data = normalize_ohlc(raw)
        return ({proxies[0]: data} if not data.empty else {}, [] if not data.empty else [f"{proxies[0]}: no data"])
    for proxy in proxies:
        try:
            frame = raw[proxy] if isinstance(raw.columns, pd.MultiIndex) and proxy in raw.columns.get_level_values(0) else pd.DataFrame()
            data = normalize_ohlc(frame)
            if data.empty:
                errors.append(f"{proxy}: no data")
            else:
                frames[proxy] = data
        except Exception as exc:
            errors.append(f"{proxy}: {exc}")
    return frames, errors


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def enrich(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    out = df.copy()
    high, low, close = out["High"].astype(float), out["Low"].astype(float), out["Close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    annualizer = 252 * 6.5 if interval.endswith("h") or interval.endswith("m") else 252
    out["atr"] = tr.rolling(14).mean()
    out["atr_pct"] = out["atr"] / close.replace(0, np.nan)
    out["atr_med"] = out["atr_pct"].rolling(60).median()
    out["ema12"] = close.ewm(span=12, adjust=False).mean()
    out["ema26"] = close.ewm(span=26, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["ema100"] = close.ewm(span=100, adjust=False).mean()
    out["mom5"] = close.pct_change(5)
    out["mom20"] = close.pct_change(20)
    out["vol20"] = close.pct_change().rolling(20).std() * math.sqrt(annualizer)
    out["rsi"] = rsi(close)
    return out.dropna().reset_index(drop=True)


def price_round(x: Optional[float], ref: float) -> Optional[float]:
    if x is None or not math.isfinite(x):
        return None
    a = abs(ref)
    digits = 2 if a >= 1000 else 3 if a >= 100 else 4 if a >= 10 else 5 if a >= 1 else 6
    return round(float(x), digits)


def risk_regime(atr_pct: float, vol: float) -> str:
    if atr_pct > 0.10 or vol > 1.40:
        return "extreme"
    if atr_pct > 0.05 or vol > 0.85:
        return "high"
    if atr_pct < 0.008 and vol < 0.16:
        return "compressed"
    return "normal"


def expansion(atr_pct: float, median: float) -> str:
    if not math.isfinite(median) or median <= 0:
        return "unknown"
    ratio = atr_pct / median
    return "expanded" if ratio >= 1.40 else "expanding" if ratio >= 1.15 else "compressed" if ratio <= 0.75 else "steady"


def scenario(market: Market, df: pd.DataFrame, rank: int, interval: str, equity: float, risk_pct: float, min_edge: float, sl_atr: float, tp1_atr: float, tp2_atr: float) -> Scenario:
    if len(df) < 100:
        raise ValueError("not enough bars")
    e = enrich(df, interval)
    if e.empty:
        raise ValueError("empty indicators")
    x = e.iloc[-1]
    close, atr, atr_pct, atr_med = float(x.Close), float(x.atr), float(x.atr_pct), float(x.atr_med)
    mom5, mom20, vol, rv = float(x.mom5), float(x.mom20), float(x.vol20), float(x.rsi)
    trend = 1.0 if x.ema12 > x.ema26 else -1.0
    structure = 0.55 if close > x.ema50 else -0.55
    structure2 = 0.35 if close > x.ema100 else -0.35
    edge_raw = 0.30 * trend + 0.18 * structure + 0.12 * structure2 + 0.22 * np.tanh(mom20 * 8) + 0.12 * np.tanh(mom5 * 15) + 0.06 * np.clip((rv - 50) / 25, -1, 1)
    edge = float(np.clip(abs(edge_raw), 0, 1))
    reg = risk_regime(atr_pct, vol)
    exp = expansion(atr_pct, atr_med)
    vol_boost = min(1.0, max(0.0, atr_pct / 0.04))
    exp_bonus = {"expanded": 0.16, "expanding": 0.11, "steady": 0.04, "compressed": -0.08}.get(exp, 0.0)
    risk_penalty = {"extreme": 0.28, "high": 0.10, "normal": 0.0, "compressed": 0.04}.get(reg, 0.0)
    score = float(100 * np.clip(0.62 * edge + 0.24 * vol_boost + exp_bonus - risk_penalty, 0, 1))
    direction = "BUY/LONG" if edge_raw >= min_edge else "SELL/SHORT" if edge_raw <= -min_edge else "WAIT"
    status, buy_ok, sell_ok = status_for(market)
    if direction == "BUY/LONG":
        sl, tp1, tp2 = close - sl_atr * atr, close + tp1_atr * atr, close + tp2_atr * atr
    elif direction == "SELL/SHORT":
        sl, tp1, tp2 = close + sl_atr * atr, close - tp1_atr * atr, close - tp2_atr * atr
    else:
        sl = tp1 = tp2 = None
    if direction == "WAIT":
        action = "WAIT - edge too weak"
    elif status.startswith("CLOSED"):
        action = f"WAIT - closed; prepare {direction}"
    elif direction == "BUY/LONG" and not buy_ok:
        action = "WAIT - BUY unavailable"
    elif direction == "SELL/SHORT" and not sell_ok:
        action = "WAIT - SELL unavailable"
    else:
        action = direction
    rr1 = abs(tp1 - close) / abs(close - sl) if sl is not None and tp1 is not None else None
    rr2 = abs(tp2 - close) / abs(close - sl) if sl is not None and tp2 is not None else None
    budget = equity * risk_pct / 100
    units = budget / abs(close - sl) if sl is not None and abs(close - sl) > 0 else None
    why = f"EMA12{'>' if x.ema12 > x.ema26 else '<'}EMA26; close{'>' if close > x.ema50 else '<'}EMA50; RSI={rv:.1f}; M5={mom5*100:.2f}%; M20={mom20*100:.2f}%; ATR={atr_pct*100:.2f}%; {exp}"
    return Scenario(rank, market.symbol, market.proxy, market.asset, status, action, price_round(close, close) or 0.0, price_round(sl, close), price_round(tp1, close), price_round(tp2, close), round(rr1, 2) if rr1 else None, round(rr2, 2) if rr2 else None, round(score, 2), round(edge, 4), price_round(atr, close) or 0.0, round(atr_pct, 6), round(rv, 3), round(mom5, 6), round(mom20, 6), reg, why, round(budget, 2), round(units, 4) if units else None)


def competition_math(equity: float, target: float, days: int, risk_pct: float) -> dict:
    total = target / equity - 1
    daily = (target / equity) ** (1 / days) - 1
    pressure = "extreme" if daily >= 0.10 else "very aggressive" if daily >= 0.05 else "aggressive" if daily >= 0.02 else "controlled"
    return {
        "account_equity": round(equity, 2),
        "target_equity": round(target, 2),
        "days_left": days,
        "required_total_return_percent": round(total * 100, 2),
        "required_daily_return_percent": round(daily * 100, 2),
        "risk_percent_per_attempt": round(risk_pct, 2),
        "max_loss_budget_per_attempt": round(equity * risk_pct / 100, 2),
        "pressure_level": pressure,
    }


def scan(args) -> tuple[list[Scenario], list[str], dict]:
    raw_symbols = symbols_from_csv(args.universe_file) if args.universe_file else symbols_from_arg(args.symbols)
    asset_filter = {norm(x).lower() for x in args.asset_class.split(",") if x.strip()}
    blocked_buy = {norm(x) for x in args.blocked_buy.split(",") if x.strip()}
    blocked_sell = {norm(x) for x in args.blocked_sell.split(",") if x.strip()}
    markets = [resolve(s, blocked_buy, blocked_sell) for s in raw_symbols]
    if asset_filter:
        markets = [x for x in markets if x.asset in asset_filter]
    if not markets:
        raise ValueError("no symbols after filter")
    frames, errors = fetch_batch(markets, args.period, args.interval)
    out = []
    for market in markets:
        try:
            if market.proxy not in frames:
                raise ValueError("no usable data")
            out.append(scenario(market, frames[market.proxy], 0, args.interval, args.account_equity, args.risk_pct, args.min_edge, args.sl_atr, args.tp1_atr, args.tp2_atr))
        except Exception as exc:
            errors.append(f"{market.symbol}: {exc}")
    out.sort(key=lambda x: x.score, reverse=True)
    ranked = []
    for i, item in enumerate(out, 1):
        d = asdict(item)
        d["rank"] = i
        ranked.append(Scenario(**d))
    return ranked, errors, competition_math(args.account_equity, args.target_equity, args.days_left, args.risk_pct)


def actionable(x: Scenario) -> bool:
    return x.action in {"BUY/LONG", "SELL/SHORT"}


def print_table(console, title: str, rows: list[Scenario], top: int):
    table = Table(title=title, show_lines=True)
    for col in ["#", "Symbol", "Asset", "Status", "Action", "Entry", "SL", "TP1", "TP2", "RR", "Score", "Risk", "Why"]:
        table.add_column(col, justify="right" if col in {"#", "Entry", "SL", "TP1", "TP2", "RR", "Score"} else "left")
    for x in rows[:top]:
        table.add_row(str(x.rank), x.symbol, x.asset, x.status, x.action, str(x.entry), str(x.sl or "-"), str(x.tp1 or "-"), str(x.tp2 or "-"), "-" if x.rr1 is None else f"{x.rr1}/{x.rr2}", f"{x.score:.2f}", x.risk, x.why)
    console.print(table)


def render(scenarios: list[Scenario], errors: list[str], plan: dict, args):
    buys = [x for x in scenarios if x.action == "BUY/LONG"]
    sells = [x for x in scenarios if x.action == "SELL/SHORT"]
    acts = buys + sells
    waits = [x for x in scenarios if not actionable(x)]
    payload = {"competition_math": plan, "summary": {"scanned": len(scenarios), "actionable": len(acts), "buy": len(buys), "sell": len(sells), "wait_or_closed": len(waits)}, "top_actionable": [asdict(x) for x in sorted(acts, key=lambda y: y.score, reverse=True)[:args.top]], "top_buy": [asdict(x) for x in buys[:args.top]], "top_sell": [asdict(x) for x in sells[:args.top]], "errors": errors[:50]}
    if args.json or Console is None:
        print(json.dumps(payload, indent=2))
        return
    console = Console()
    console.print(Panel.fit("Broad Competition Scanner\nBUY = long, SELL = short. Use platform buttons/spread as final truth."))
    plan_table = Table(title="Competition Math", show_lines=True)
    plan_table.add_column("Metric")
    plan_table.add_column("Value")
    for k, v in plan.items():
        plan_table.add_row(k, str(v))
    console.print(plan_table)
    console.print(f"Scanned: {len(scenarios)} | Actionable: {len(acts)} | BUY: {len(buys)} | SELL: {len(sells)} | WAIT/closed: {len(waits)}")
    if acts:
        print_table(console, "Top Actionable BUY/SELL", sorted(acts, key=lambda y: y.score, reverse=True), args.top)
    if buys:
        print_table(console, "Top BUY / LONG", buys, min(args.top, 10))
    if sells:
        print_table(console, "Top SELL / SHORT", sells, min(args.top, 10))
    if args.show_wait and waits:
        print_table(console, "Top WAIT / Closed / Blocked", waits, min(args.top, 10))
    if errors:
        et = Table(title=f"Skipped / unavailable symbols first {min(20, len(errors))}")
        et.add_column("Reason")
        for e in errors[:20]:
            et.add_row(e)
        console.print(et)


def parse_args():
    p = argparse.ArgumentParser(description="Scan broad markets and rank BUY/SELL demo competition candidates with Entry, SL, TP1, TP2.")
    p.add_argument("--symbols", default="ALL", help="ALL, CORE, or comma-separated symbols. Default: ALL.")
    p.add_argument("--universe-file", default=None, help="CSV with symbol column. Overrides --symbols.")
    p.add_argument("--asset-class", default="", help="Filter: fx,metals,energy,commodity,index,crypto,stock,etf")
    p.add_argument("--period", default="60d")
    p.add_argument("--interval", default="1h")
    p.add_argument("--account-equity", type=float, default=50_000)
    p.add_argument("--target-equity", type=float, default=1_000_000)
    p.add_argument("--days-left", type=int, default=30)
    p.add_argument("--risk-pct", type=float, default=1.0)
    p.add_argument("--min-edge", type=float, default=0.30)
    p.add_argument("--sl-atr", type=float, default=1.0)
    p.add_argument("--tp1-atr", type=float, default=1.5)
    p.add_argument("--tp2-atr", type=float, default=2.5)
    p.add_argument("--blocked-buy", default="", help="Symbols where platform BUY is unavailable, comma-separated.")
    p.add_argument("--blocked-sell", default="", help="Symbols where platform SELL is unavailable, comma-separated.")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--show-wait", action="store_true")
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    scenarios, errors, plan = scan(args)
    if not scenarios:
        raise RuntimeError("No market could be analyzed. Check internet/data source or symbol coverage.")
    render(scenarios, errors, plan, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
