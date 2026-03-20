"""ORACLE shared contract models — canonical Pydantic schemas for all inter-program communication."""

from .signal import Signal, SignalCategory, SourceId
from .anomaly_event import AnomalyEvent
from .insight import Insight
from .market_state import MarketState
from .trade_thesis import TradeThesis, ThesisDecision, ThesisOutcome, Hypothesis, EvidenceWeight, ContextAssembly, HistoricalAnalogue
from .wallet_profile import WalletProfile, ReputationTier
from .trade_execution import TradeExecution, MarketType, ExecutionSource, ExecutionStatus, ExitReason
from .post_mortem import PostMortem, SignalSummary
from .operator_alert import OperatorAlert, AlertType, AlertSeverity
from .copy_trade_approval import CopyTradeApproval

__all__ = [
    "Signal", "SignalCategory", "SourceId",
    "AnomalyEvent",
    "Insight",
    "MarketState",
    "TradeThesis", "ThesisDecision", "ThesisOutcome", "Hypothesis", "EvidenceWeight", "ContextAssembly", "HistoricalAnalogue",
    "WalletProfile", "ReputationTier",
    "TradeExecution", "MarketType", "ExecutionSource", "ExecutionStatus", "ExitReason",
    "PostMortem", "SignalSummary",
    "OperatorAlert", "AlertType", "AlertSeverity",
    "CopyTradeApproval",
]
