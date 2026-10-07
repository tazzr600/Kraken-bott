from dataclasses import dataclass
from decimal import Decimal

@dataclass
class Book:
    asset_id: str
    ask: Decimal | None
    ask_size: Decimal
    bid: Decimal | None
    bid_size: Decimal

@dataclass
class Market:
    id: str
    question: str
    yes: str
    no: str
    min_size: Decimal

@dataclass
class Opportunity:
    market: Market
    yes: Book
    no: Book
    size: Decimal
    cost: Decimal
    gross: Decimal
    net: Decimal

def val(o, k, d=None):
    if o is None:
        return d
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)
