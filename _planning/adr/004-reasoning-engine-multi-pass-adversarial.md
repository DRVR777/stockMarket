# ADR-004: Reasoning Engine Uses Multi-Pass Adversarial Reasoning

## Status
accepted — stated explicitly in PRD Section 2.5

## Context
PRD states: "The RE is not a single prompt to an AI model. It operates as a multi-step reasoning pipeline: Step 1 — context assembly, Step 2 — hypothesis generation (adversarial, must argue both sides), Step 3 — evidence weighting, Step 4 — confidence calibration."

## Decision
The Reasoning Engine (RE) uses a four-step multi-pass pipeline for every market analysis:
1. Assemble all available signals, current price, liquidity, time to resolution, whale activity
2. Generate competing hypotheses — model must argue both YES and NO outcomes
3. Score each hypothesis against evidence; compute RE probability vs market implied probability
4. Calibrate confidence based on signal recency, source diversity, and model uncertainty

A TradeThesis is only emitted if the RE probability delta vs market price exceeds the configured threshold AND confidence is above the operator-defined minimum.

## Consequences
- RE makes multiple AI model calls per market analysis (at minimum 2: hypothesis generation + synthesis).
- RE latency is higher than a single-prompt approach — acceptable because RE runs asynchronously.
- Low-confidence theses are flagged for operator review, not auto-executed.
- The four-step structure is fixed; the underlying AI model is swappable (see ADR-002).

## Alternatives Considered
To be completed during planning phase.
