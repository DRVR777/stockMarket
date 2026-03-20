# Paper Trading Rules — Set Before Any Data

Written March 20, 2026. These rules are locked. Don't change them based on results.

## Decision Thresholds

### Minimum sample: 200 trades
Don't evaluate anything before 200 trades. Variance dominates small samples.
At ~8 signals/day, this takes ~25 days.

### GO LIVE criteria (all must be true):
- [ ] 200+ trades completed
- [ ] Win rate >= 52% (above break-even of 44% with margin)
- [ ] No single week with drawdown > 8%
- [ ] Profit factor >= 1.2 (gross wins / gross losses)
- [ ] At least 3 of 4 weeks were net positive

### PAUSE AND DIAGNOSE (any one triggers):
- [ ] Win rate drops below 48% over any 100-trade window
- [ ] Single week drawdown exceeds 10%
- [ ] 3 consecutive losing days
- [ ] Profit factor below 1.0 over any 100-trade window

### KILL SWITCH (stop everything):
- [ ] Win rate below 44% over 200+ trades (below break-even)
- [ ] Total drawdown exceeds 15% from peak
- [ ] Model is clearly broken (e.g. 90%+ of signals are one direction)

## When Going Live

- Start with $100 total capital
- 2% risk per trade = $2 per trade
- Max 5 concurrent positions
- After 100 profitable live trades: increase to $250
- After 200 profitable live trades: increase to $500
- Never increase during a losing week

## What to Track Daily

- Total signals generated
- Signals that passed ML filter
- Win/loss count
- Daily PnL (paper $)
- Rolling 50-trade win rate
- Max drawdown from peak

## What NOT to Do

- Don't override the ML filter ("this one looks good even though ML says no")
- Don't change thresholds mid-evaluation
- Don't go live early because of a hot streak
- Don't increase size after a big win
- Don't revenge trade after a loss
- Don't add new features to the model during evaluation period
