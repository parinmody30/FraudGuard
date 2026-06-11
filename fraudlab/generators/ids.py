"""Shared, deterministic transaction-id minting.

A single minter is threaded through the baseline generator and the fraud engine so that
ids remain globally unique and stable for a given seed/config.
"""

from __future__ import annotations


class IdMinter:
    """Sequential transaction-id generator: T0000000000, T0000000001, ..."""

    def __init__(self, start: int = 0) -> None:
        self._counter = start

    def next(self) -> str:
        tid = f"T{self._counter:010d}"
        self._counter += 1
        return tid
