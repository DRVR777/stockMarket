"""Backtester — run SMC + pattern detection against historical data and measure win rates.

Walks through historical candles bar-by-bar, detects setups, simulates entries/exits,
and tracks P&L. Produces a performance report with win rate, profit factor, drawdown.

Usage::

    results = await backtest("BTCUSDT", interval="1h", days=180)
    print(results.summary())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from market_scanner.smc import analyze_smc, Bias

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A single simulated trade."""
    symbol: str
    direction: str       # "long" | "short"
    entry_price: float
    entry_bar: int
    stop_loss: float
    take_profit: float
    setup_type: str
    signals: list[str]
    exit_price: float = 0.0
    exit_bar: int = 0
    exit_reason: str = ""  # "tp" | "sl" | "timeout"
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    """Aggregate backtest results."""
    symbol: str
    interval: str
    total_bars: int
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    avg_hold_bars: float
    trades: list[Trade] = field(default_factory=list)
    setup_breakdown: dict[str, dict] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== Backtest: {self.symbol} ({self.interval}) ===",
            f"Bars: {self.total_bars}  Trades: {self.total_trades}",
            f"Win rate: {self.win_rate:.1%}  ({self.wins}W / {self.losses}L)",
            f"Total PnL: {self.total_pnl_pct:+.2f}%",
            f"Avg win: {self.avg_win_pct:+.2f}%  Avg loss: {self.avg_loss_pct:+.2f}%",
            f"Profit factor: {self.profit_factor:.2f}",
            f"Max drawdown: {self.max_drawdown_pct:.2f}%",
            f"Best: {self.best_trade_pct:+.2f}%  Worst: {self.worst_trade_pct:+.2f}%",
            f"Avg hold: {self.avg_hold_bars:.0f} bars",
        ]
        if self.setup_breakdown:
            lines.append("\nSetup breakdown:")
            for setup, stats in sorted(self.setup_breakdown.items(), key=lambda x: x[1].get("pnl", 0), reverse=True):
                lines.append(
                    f"  {setup:<20} trades={stats['count']:>3}  "
                    f"wr={stats['win_rate']:.0%}  pnl={stats['pnl']:+.2f}%"
                )
        return "\n".join(lines)


def backtest_df(
    df: pd.DataFrame,
    symbol: str = "TEST",
    interval: str = "1h",
    min_confidence: float = 0.4,
    lookback_window: int = 50,
    max_hold_bars: int = 48,
) -> BacktestResult:
    """Run a backtest on a DataFrame of OHLCV data.

    Walks bar-by-bar through the data:
      1. At each bar, run SMC analysis on the last `lookback_window` bars
      2. If a setup is detected with confidence >= min_confidence, enter
      3. Track the position until TP, SL, or timeout

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
        symbol: Asset symbol for labeling
        interval: Candle interval for labeling
        min_confidence: Minimum SMC confidence to trigger entry
        lookback_window: How many bars of history to analyze at each step
        max_hold_bars: Maximum bars to hold a position before force-close
    """
    trades: list[Trade] = []
    in_position = False
    current_trade: Optional[Trade] = None

    for i in range(lookback_window, len(df)):
        # Check if we need to exit current position
        if in_position and current_trade:
            bar_high = float(df["high"].iloc[i])
            bar_low = float(df["low"].iloc[i])
            bar_close = float(df["close"].iloc[i])
            bars_held = i - current_trade.entry_bar

            exited = False

            if current_trade.direction == "long":
                # Check stop loss first (worst case)
                if bar_low <= current_trade.stop_loss:
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.exit_reason = "sl"
                    exited = True
                # Check take profit
                elif bar_high >= current_trade.take_profit:
                    current_trade.exit_price = current_trade.take_profit
                    current_trade.exit_reason = "tp"
                    exited = True
            else:  # short
                if bar_high >= current_trade.stop_loss:
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.exit_reason = "sl"
                    exited = True
                elif bar_low <= current_trade.take_profit:
                    current_trade.exit_price = current_trade.take_profit
                    current_trade.exit_reason = "tp"
                    exited = True

            # Timeout
            if not exited and bars_held >= max_hold_bars:
                current_trade.exit_price = bar_close
                current_trade.exit_reason = "timeout"
                exited = True

            if exited:
                current_trade.exit_bar = i
                if current_trade.direction == "long":
                    current_trade.pnl_pct = (
                        (current_trade.exit_price - current_trade.entry_price)
                        / current_trade.entry_price * 100
                    )
                else:
                    current_trade.pnl_pct = (
                        (current_trade.entry_price - current_trade.exit_price)
                        / current_trade.entry_price * 100
                    )
                trades.append(current_trade)
                in_position = False
                current_trade = None
                continue

        # Look for new entry (only if not in position)
        if not in_position:
            window = df.iloc[i - lookback_window : i + 1].copy()
            window.reset_index(drop=True, inplace=True)

            analysis = analyze_smc(window, symbol, "backtest")
            if analysis is None:
                continue
            if analysis.confidence < min_confidence:
                continue
            if analysis.bias == Bias.NEUTRAL:
                continue
            if not analysis.stop_loss or not analysis.take_profit:
                continue

            direction = "long" if analysis.bias == Bias.BULLISH else "short"
            entry_price = float(df["close"].iloc[i])

            current_trade = Trade(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                entry_bar=i,
                stop_loss=analysis.stop_loss,
                take_profit=analysis.take_profit,
                setup_type=analysis.setup_type or "smc_generic",
                signals=analysis.signals[:5],
            )
            in_position = True

    # Close any open trade at the end
    if in_position and current_trade:
        current_trade.exit_price = float(df["close"].iloc[-1])
        current_trade.exit_bar = len(df) - 1
        current_trade.exit_reason = "end"
        if current_trade.direction == "long":
            current_trade.pnl_pct = (
                (current_trade.exit_price - current_trade.entry_price)
                / current_trade.entry_price * 100
            )
        else:
            current_trade.pnl_pct = (
                (current_trade.entry_price - current_trade.exit_price)
                / current_trade.entry_price * 100
            )
        trades.append(current_trade)

    return _compute_results(trades, symbol, interval, len(df))


def _compute_results(
    trades: list[Trade],
    symbol: str,
    interval: str,
    total_bars: int,
) -> BacktestResult:
    """Compute aggregate statistics from trade list."""
    if not trades:
        return BacktestResult(
            symbol=symbol, interval=interval, total_bars=total_bars,
            total_trades=0, wins=0, losses=0, win_rate=0, total_pnl_pct=0,
            avg_win_pct=0, avg_loss_pct=0, profit_factor=0,
            max_drawdown_pct=0, best_trade_pct=0, worst_trade_pct=0,
            avg_hold_bars=0,
        )

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    total_pnl = sum(t.pnl_pct for t in trades)
    gross_profit = sum(t.pnl_pct for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 0

    # Max drawdown
    equity_curve = []
    running = 0.0
    for t in trades:
        running += t.pnl_pct
        equity_curve.append(running)
    peak = 0.0
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    # Setup breakdown
    setup_stats: dict[str, dict] = {}
    for t in trades:
        st = t.setup_type
        if st not in setup_stats:
            setup_stats[st] = {"count": 0, "wins": 0, "pnl": 0.0}
        setup_stats[st]["count"] += 1
        setup_stats[st]["pnl"] += t.pnl_pct
        if t.pnl_pct > 0:
            setup_stats[st]["wins"] += 1
    for st in setup_stats:
        s = setup_stats[st]
        s["win_rate"] = s["wins"] / s["count"] if s["count"] > 0 else 0

    avg_hold = sum(t.exit_bar - t.entry_bar for t in trades) / len(trades)

    return BacktestResult(
        symbol=symbol,
        interval=interval,
        total_bars=total_bars,
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(trades),
        total_pnl_pct=round(total_pnl, 2),
        avg_win_pct=round(gross_profit / len(wins), 2) if wins else 0,
        avg_loss_pct=round(-gross_loss / len(losses), 2) if losses else 0,
        profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        max_drawdown_pct=round(max_dd, 2),
        best_trade_pct=round(max(t.pnl_pct for t in trades), 2),
        worst_trade_pct=round(min(t.pnl_pct for t in trades), 2),
        avg_hold_bars=round(avg_hold, 1),
        trades=trades,
        setup_breakdown=setup_stats,
    )
