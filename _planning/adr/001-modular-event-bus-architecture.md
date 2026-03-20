# ADR-001: Modular Event-Bus Architecture

## Status
accepted — stated explicitly in PRD Section 2

## Context
PRD states: "The platform is composed of six loosely coupled modules. Each module can be developed, tested, and replaced independently. They communicate through a shared internal event bus and a shared state layer."

## Decision
ORACLE is built as 7 loosely coupled programs (6 core modules + operator dashboard) that communicate exclusively via a shared internal event bus and shared state layer. Programs do not call each other directly. All inter-program data shapes are defined in shared/contracts/.

## Consequences
- No direct imports between programs. All communication is async via event bus.
- Each program can be restarted or replaced without bringing down others.
- Contract shapes in shared/contracts/ are the single source of truth for all inter-program interfaces.
- The event bus and state layer implementations are resolved in ADR-014 and ADR-015.

## Alternatives Considered
To be completed during planning phase.
