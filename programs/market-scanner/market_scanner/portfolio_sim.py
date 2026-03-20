"""Portfolio simulator — simulate running all signals with position sizing + risk management.

Takes backtest trades and simulates running them as a portfolio:
  - Position sizing: fixed $ per trade or Kelly criterion
  - Max concurrent positions
  - Daily loss limit (circuit breaker)
  - Track equity curve, drawdown, Sharpe ratio

Usage::

    sim = PortfolioSimulator(starting_capital=10000)
    result = sim.run(trades)
    print(result.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimTrade:
    """A trade to simulate."""
    symbol: str
    direction: str       # "long" | "short"
    entry_bar: int
    exit_bar: int
    pnl_pct: float
    setup_type: str = ""
    ml_probability: float = 0.5


@dataclass
class PortfolioResult:
    """Full portfolio simulation results."""
    starting_capital: float
    ending_capital: float
    total_return_pct: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    max_drawdown_usd: float
    avg_trade_pnl_pct: float
    avg_trade_pnl_usd: float
    best_trade_pnl_usd: float
    worst_trade_pnl_usd: float
    avg_position_size: float
    circuit_breaker_hits: int
    equity_curve: list[float] = field(default_factory=list)
    daily_pnl: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{'='*55}",
            f"  PORTFOLIO SIMULATION RESULTS",
            f"{'='*55}",
            f"  Capital: ${self.starting_capital:,.0f} -> ${self.ending_capital:,.0f} ({self.total_return_pct:+.1f}%)",
            f"  Trades:  {self.total_trades}  ({self.wins}W / {self.losses}L = {self.win_rate:.1%})",
            f"  Profit factor: {self.profit_factor:.2f}",
            f"  Sharpe ratio:  {self.sharpe_ratio:.2f}",
            f"",
            f"  Max drawdown:  ${self.max_drawdown_usd:,.0f} ({self.max_drawdown_pct:.1f}%)",
            f"  Avg trade:     ${self.avg_trade_pnl_usd:,.2f} ({self.avg_trade_pnl_pct:+.2f}%)",
            f"  Best trade:    ${self.best_trade_pnl_usd:,.2f}",
            f"  Worst trade:   ${self.worst_trade_pnl_usd:,.2f}",
            f"  Avg position:  ${self.avg_position_size:,.0f}",
            f"  Circuit breaks: {self.circuit_breaker_hits}",
            f"{'='*55}",
        ]
        return "\n".join(lines)


class PortfolioSimulator:
    """Simulate portfolio performance from a list of trades."""

    def __init__(
        self,
        starting_capital: float = 10_000,
        position_size_pct: float = 5.0,      # risk 5% of capital per trade
        max_concurrent: int = 5,
        daily_loss_limit_pct: float = 3.0,    # stop trading if down 3% in a day
        use_kelly: bool = False,
    ) -> None:
        self.starting_capital = starting_capital
        self.position_size_pct = position_size_pct
        self.max_concurrent = max_concurrent
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.use_kelly = use_kelly

    def run(self, trades: list[SimTrade]) -> PortfolioResult:
        """Run portfolio simulation on a list of trades sorted by entry_bar."""
        # Sort by entry bar
        trades = sorted(trades, key=lambda t: t.entry_bar)

        capital = self.starting_capital
        peak_capital = capital
        equity_curve = [capital]
        trade_pnls_usd: list[float] = []
        daily_pnl_list: list[float] = []
        open_positions: list[SimTrade] = []
        circuit_breaker_hits = 0

        # Track daily PnL for circuit breaker
        current_day_bar = 0
        day_pnl = 0.0
        day_paused = False

        for trade in trades:
            # Simple day tracking (every 24 bars on 1h = 1 day)
            trade_day = trade.entry_bar // 24
            if trade_day != current_day_bar:
                daily_pnl_list.append(day_pnl)
                day_pnl = 0.0
                current_day_bar = trade_day
                day_paused = False

            # Circuit breaker check
            if day_paused:
                continue
            if abs(day_pnl) >= capital * self.daily_loss_limit_pct / 100:
                day_paused = True
                circuit_breaker_hits += 1
                continue

            # Max concurrent check
            open_positions = [p for p in open_positions if p.exit_bar > trade.entry_bar]
            if len(open_positions) >= self.max_concurrent:
                continue

            # Position sizing
            if self.use_kelly and trade.ml_probability > 0.5:
                # Kelly criterion: f = (bp - q) / b where b=1, p=win_prob, q=1-p
                kelly_frac = (2 * trade.ml_probability - 1)  # simplified Kelly
                kelly_frac = max(0.01, min(0.25, kelly_frac))  # clamp 1-25%
                pos_size = capital * kelly_frac
            else:
                pos_size = capital * self.position_size_pct / 100

            pos_size = min(pos_size, capital * 0.25)  # never risk more than 25%

            # Execute trade
            pnl_usd = pos_size * trade.pnl_pct / 100
            capital += pnl_usd
            day_pnl += pnl_usd
            trade_pnls_usd.append(pnl_usd)
            equity_curve.append(capital)
            open_positions.append(trade)

            # Track peak for drawdown
            if capital > peak_capital:
                peak_capital = capital

        # Final day
        daily_pnl_list.append(day_pnl)

        # Compute stats
        wins_usd = [p for p in trade_pnls_usd if p > 0]
        losses_usd = [p for p in trade_pnls_usd if p <= 0]
        gross_profit = sum(wins_usd) if wins_usd else 0
        gross_loss = abs(sum(losses_usd)) if losses_usd else 0

        # Max drawdown
        peak = equity_curve[0]
        max_dd_usd = 0
        max_dd_pct = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            dd_pct = dd / peak * 100 if peak > 0 else 0
            if dd > max_dd_usd:
                max_dd_usd = dd
                max_dd_pct = dd_pct

        # Sharpe ratio (annualized, assuming daily returns)
        if daily_pnl_list and len(daily_pnl_list) > 1:
            daily_returns = [p / self.starting_capital * 100 for p in daily_pnl_list]
            import numpy as np
            mean_daily = np.mean(daily_returns)
            std_daily = np.std(daily_returns)
            sharpe = (mean_daily / std_daily * math.sqrt(252)) if std_daily > 0 else 0
        else:
            sharpe = 0

        total_trades = len(trade_pnls_usd)
        return PortfolioResult(
            starting_capital=self.starting_capital,
            ending_capital=round(capital, 2),
            total_return_pct=round((capital - self.starting_capital) / self.starting_capital * 100, 2),
            total_trades=total_trades,
            wins=len(wins_usd),
            losses=len(losses_usd),
            win_rate=len(wins_usd) / total_trades if total_trades > 0 else 0,
            profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            max_drawdown_usd=round(max_dd_usd, 2),
            avg_trade_pnl_pct=round(sum(t.pnl_pct for t in trades[:total_trades]) / total_trades, 3) if total_trades > 0 else 0,
            avg_trade_pnl_usd=round(sum(trade_pnls_usd) / total_trades, 2) if total_trades > 0 else 0,
            best_trade_pnl_usd=round(max(trade_pnls_usd), 2) if trade_pnls_usd else 0,
            worst_trade_pnl_usd=round(min(trade_pnls_usd), 2) if trade_pnls_usd else 0,
            avg_position_size=round(sum(self.starting_capital * self.position_size_pct / 100 for _ in trade_pnls_usd) / total_trades, 0) if total_trades > 0 else 0,
            circuit_breaker_hits=circuit_breaker_hits,
            equity_curve=equity_curve,
            daily_pnl=daily_pnl_list,
        )
