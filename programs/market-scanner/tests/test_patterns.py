"""Tests for pattern detection engine — uses synthetic data, no API calls."""
import sys
import numpy as np
import pandas as pd

from market_scanner.patterns import analyze, Direction


def make_bull_df(n: int = 60) -> pd.DataFrame:
    """Create synthetic bullish OHLCV data (uptrend)."""
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    base = 100.0
    prices = [base + i * 0.5 + np.random.normal(0, 0.5) for i in range(n)]
    return pd.DataFrame({
        "open": [p - 0.2 for p in prices],
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices],
        "close": prices,
        "volume": [1000 + i * 10 for i in range(n)],
    }, index=dates)


def make_bear_df(n: int = 60) -> pd.DataFrame:
    """Create synthetic bearish OHLCV data (downtrend)."""
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    base = 150.0
    prices = [base - i * 0.5 + np.random.normal(0, 0.5) for i in range(n)]
    return pd.DataFrame({
        "open": [p + 0.2 for p in prices],
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices],
        "close": prices,
        "volume": [1000 + i * 10 for i in range(n)],
    }, index=dates)


def test_bull_detection() -> bool:
    df = make_bull_df()
    result = analyze(df, "TEST_BULL", "test", "test_bull")
    assert result is not None
    assert result.direction == Direction.BULL, f"Expected BULL, got {result.direction}"
    assert result.score > 0, f"Score should be > 0, got {result.score}"
    assert result.current_price > 0
    assert result.entry_price is not None
    assert result.stop_loss is not None
    assert result.take_profit is not None
    assert result.take_profit > result.entry_price  # bull TP above entry
    print(f"  bull_detection: PASS  score={result.score}  patterns={result.patterns}")
    return True


def test_bear_detection() -> bool:
    df = make_bear_df()
    result = analyze(df, "TEST_BEAR", "test", "test_bear")
    assert result is not None
    assert result.direction == Direction.BEAR, f"Expected BEAR, got {result.direction}"
    assert result.score > 0
    assert result.entry_price is not None
    assert result.take_profit < result.entry_price  # bear TP below entry
    print(f"  bear_detection: PASS  score={result.score}  patterns={result.patterns}")
    return True


def test_insufficient_data() -> bool:
    df = pd.DataFrame({
        "open": [100], "high": [101], "low": [99],
        "close": [100], "volume": [1000],
    })
    result = analyze(df, "TINY", "test", "tiny")
    assert result is None
    print("  insufficient_data: PASS (returned None)")
    return True


def test_pattern_result_fields() -> bool:
    df = make_bull_df()
    result = analyze(df, "FULL", "crypto", "bitcoin")
    assert result.symbol == "FULL"
    assert result.asset_type == "crypto"
    assert result.asset_id == "bitcoin"
    assert 0 <= result.rsi <= 100
    assert result.volume_ratio > 0
    assert result.support > 0
    assert result.resistance > result.support
    print(f"  result_fields: PASS  rsi={result.rsi}  vol_ratio={result.volume_ratio}")
    return True


def main() -> bool:
    np.random.seed(42)
    tests = [
        test_bull_detection,
        test_bear_detection,
        test_insufficient_data,
        test_pattern_result_fields,
    ]

    print("market-scanner pattern tests\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  {test.__name__}: FAIL -- {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
