"""Task 1 — Vault initializer and file I/O.

Creates the vault directory tree and provides read/write helpers
for markdown files with YAML front-matter.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import knowledge_base.config as _cfg

logger = logging.getLogger(__name__)

VAULT_SUBDIRS = ["markets", "wallets", "signals", "theses", "osint"]


def init_vault() -> Path:
    """Create the vault directory tree. Returns the vault root path."""
    root = Path(_cfg.VAULT_DIR)
    for subdir in VAULT_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    logger.info("Vault initialized at %s", root.resolve())
    return root


def vault_path(*parts: str) -> Path:
    """Build a path under the vault root."""
    return Path(_cfg.VAULT_DIR) / Path(*parts)


def write_md(
    path: Path,
    front_matter: dict[str, Any],
    body: str,
    overwrite: bool = True,
) -> bool:
    """Write a markdown file with YAML front-matter.

    Returns True if written, False if skipped (file exists and overwrite=False).
    """
    if not overwrite and path.exists():
        logger.warning("Vault: skipping existing file %s", path)
        return False

    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["---"]
    for key, value in front_matter.items():
        if isinstance(value, (dict, list)):
            lines.append(f"{key}: {json.dumps(value)}")
        elif value is None:
            lines.append(f"{key}: null")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    lines.append(body)

    path.write_text("\n".join(lines), encoding="utf-8")
    return True


def append_section(path: Path, section_header: str, content: str) -> None:
    """Append content under a section header in an existing markdown file.

    If the section doesn't exist, appends it at the end.
    """
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")

    marker = f"## {section_header}"
    if marker in text:
        # Find the section and append before the next section or EOF
        idx = text.index(marker)
        next_section = text.find("\n## ", idx + len(marker))
        if next_section == -1:
            # Append at end of file
            text = text.rstrip() + "\n" + content + "\n"
        else:
            # Insert before next section
            text = text[:next_section] + content + "\n" + text[next_section:]
    else:
        # Append new section at end
        text = text.rstrip() + f"\n\n{marker}\n{content}\n"

    path.write_text(text, encoding="utf-8")


def read_md(path: Path) -> str:
    """Read a vault markdown file. Returns empty string if not found."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")
