"""Task 4 — MarketWriter.

Creates/updates vault/markets/{market_id}.md on each TradeThesis
with decision != skip. Maintains sections: Signals, Theses, Resolution, Post-Mortem.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from oracle_shared.contracts.trade_thesis import TradeThesis, ThesisDecision

from knowledge_base.vault import append_section, vault_path, write_md, read_md

logger = logging.getLogger(__name__)


class MarketWriter:
    """Create and maintain market vault files."""

    def write_from_thesis(self, thesis: TradeThesis) -> None:
        """Create or append to vault/markets/{market_id}.md."""
        if thesis.decision == ThesisDecision.SKIP:
            return

        path = vault_path("markets", f"{thesis.market_id}.md")

        if not path.exists():
            # Create new market file
            ms = thesis.context_assembly.market_state
            front_matter = {
                "market_id": thesis.market_id,
                "question": thesis.market_question,
                "resolution_deadline": ms.get("resolution_deadline", ""),
                "status": "open",
            }
            body = (
                f"# {thesis.market_question}\n\n"
                f"## Signals\n\n"
                f"## Theses\n\n"
                f"## Resolution\n\n"
                f"## Post-Mortem\n"
            )
            write_md(path, front_matter, body)
            logger.info("MarketWriter: created %s", path.name)

        # Append thesis reference
        summary = (
            f"- **{thesis.thesis_id[:8]}** [{thesis.direction}] "
            f"confidence={thesis.confidence_score:.2f} "
            f"delta={thesis.probability_delta:+.3f} "
            f"decision={thesis.decision.value}\n"
        )
        append_section(path, "Theses", summary)

    def add_execution(self, market_id: str, execution_id: str, direction: str) -> None:
        """Append execution reference to market file."""
        path = vault_path("markets", f"{market_id}.md")
        content = f"- Execution {execution_id[:8]}… direction={direction}\n"
        append_section(path, "Theses", content)

    def add_postmortem(self, market_id: str, postmortem_text: str) -> None:
        """Append post-mortem text to market file."""
        path = vault_path("markets", f"{market_id}.md")
        append_section(path, "Post-Mortem", postmortem_text)
