"""Beneficiary model (payees for bank/UPI transfers)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Beneficiary:
    """A payee a user sends money to.

    ``is_trusted`` marks long-standing recurring payees; newly added beneficiaries are a
    weak signal exploited by some fraud scenarios (but also occur legitimately).
    """

    beneficiary_id: str
    owner_user_id: str
    name: str
    account_ref: str
    is_trusted: bool
    first_seen_day: int = 0
