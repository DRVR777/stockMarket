# ADR-002: No Language or Framework Prescription

## Status
accepted — stated explicitly in PRD Section 1 intro

## Context
PRD states: "No specific languages, frameworks, or infrastructure providers are prescribed. The goal is to describe a system that is architecturally sound and extensible enough that implementation can evolve as the platform matures."

## Decision
No language, framework, or infrastructure provider is mandated at the project level. Each program's implementation choices are made when that program reaches the build phase, informed by the accepted ADRs for that program (e.g., ADR-014 for event bus, ADR-015 for state layer).

## Consequences
- Builders choose the right tool per program.
- Contracts define shapes — not serialization formats or transport protocols — until ADR-014 is resolved.
- If a better tool becomes available, swapping it requires changing only the affected program.

## Alternatives Considered
To be completed during planning phase.
