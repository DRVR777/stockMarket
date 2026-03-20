"""Initialize (or reset) the ORACLE database schema.

Usage:
    python scripts/db_init.py          # CREATE TABLE IF NOT EXISTS
    python scripts/db_init.py --reset  # DROP ALL + CREATE ALL
"""
from __future__ import annotations

import asyncio
import os
import sys

# Load .env from the oracle root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from oracle_shared.db import get_engine
from oracle_shared.db.models import Base


async def main(reset: bool = False) -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        if reset:
            print("Dropping all tables...")
            await conn.run_sync(Base.metadata.drop_all)
        print("Creating tables...")
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    print(f"Done. {len(Base.metadata.tables)} tables ready.")


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    if reset:
        print("WARNING: This will destroy all data in Postgres.")
        input("Press Enter to continue or Ctrl+C to cancel...")
    asyncio.run(main(reset=reset))
