from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

@dataclass
class Book:
    token_id: str
    bid: Decimal|None=None
    bid_size: Decimal=Decimal("0")
    ask: Decimal|None=None
    ask_size: Decimal=Decimal("0")
    midpoint: Decimal|None=None
    last: Decimal|None=None

@dataclass
class Market:
    id: str
    condition_id: str
    slug: str
    title: str
    series: str
    start_ts: int
    end_ts: int
    yes_token: str
    no_token: str
    min_size: Decimal
    tick_size: Decimal
    liquidity: Decimal

@dataclass
class Underlying:
    symbol: str
    spot: float
    perp: float
    basis: float
    start_price: float
    volatility: float
    momentum_15s: float
    momentum_30s: float
    momentum_60s: float
    speed: float
    samples: int

@dataclass
class Signal:
    fair_up: float
    fair_down: float
    direction: str
    confidence: float
    edge_up: float
    edge_down: float
    selected_edge: float
    expected_move: float
    volatility: float
    reason: str

@dataclass
class Position:
    market_id: str
    yes_shares: Decimal=Decimal("0")
    no_shares: Decimal=Decimal("0")
    yes_cost: Decimal=Decimal("0")
    no_cost: Decimal=Decimal("0")
    paired_shares: Decimal=Decimal("0")
    @property
    def unpaired_yes(self): return max(self.yes_shares-self.no_shares,Decimal("0"))
    @property
    def unpaired_no(self): return max(self.no_shares-self.yes_shares,Decimal("0"))
    @property
    def net_direction(self): return self.yes_shares-self.no_shares
    @property
    def yes_vwap(self): return self.yes_cost/self.yes_shares if self.yes_shares else Decimal("0")
    @property
    def no_vwap(self): return self.no_cost/self.no_shares if self.no_shares else Decimal("0")

@dataclass
class Opportunity:
    market: Market
    underlying: Underlying
    yes: Book
    no: Book
    signal: Signal
    action: str
    outcome: str
    shares: Decimal
    price: Decimal
    spend: Decimal
    metadata: dict[str,Any]=field(default_factory=dict)
