from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class MarketCandidate:
    symbol: str
    base: str
    quote: str

    last: float
    bid: float
    ask: float

    spread_pct: float
    quote_volume: float

    liquidity_score: float

    rank: int = 0

    def to_dict(self):
        return asdict(self)


class KrakenMarketScanner:

    """
    FULL KRAKEN SPOT MARKET SCANNER

    Pipeline:

        Kraken markets
              ↓
        Spot-only filter
              ↓
        USD/allowed quote filter
              ↓
        Active market filter
              ↓
        Price validation
              ↓
        Spread filter
              ↓
        24h volume filter
              ↓
        Liquidity ranking
              ↓
        Top markets → ML engine

    The bot does NOT need to run expensive ML calculations
    against every Kraken market every few seconds.

    Instead, the scanner finds the best liquid markets first,
    then the ML engine analyzes the strongest candidates.
    """

    def __init__(
        self,
        kraken,
        settings
    ):

        self.kraken = kraken
        self.settings = settings

        self.markets: dict[str, dict] = {}

        self.universe: list[
            MarketCandidate
        ] = []

        self.last_refresh = 0.0

        self.last_error = None

        self.last_ticker_count = 0

    # =========================================================
    # MARKET FILTER
    # =========================================================

    def _is_allowed_market(
        self,
        market: dict[str, Any]
    ) -> bool:

        # -----------------------------------------------------
        # ACTIVE
        # -----------------------------------------------------

        if not market.get(
            "active",
            True
        ):
            return False

        # -----------------------------------------------------
        # SPOT ONLY
        # -----------------------------------------------------

        #
        # We want Kraken spot markets here.
        #
        # Futures will be handled separately later if desired.
        #

        if market.get(
            "spot"
        ) is False:

            return False

        # Explicitly reject derivative-style markets.

        if market.get(
            "contract"
        ):

            return False

        if market.get(
            "swap"
        ):

            return False

        if market.get(
            "future"
        ):

            return False

        # -----------------------------------------------------
        # QUOTE CURRENCY
        # -----------------------------------------------------

        quote = str(
            market.get(
                "quote"
            )
            or ""
        ).upper()

        if quote not in (
            self.settings.allowed_quote_list
        ):

            return False

        # -----------------------------------------------------
        # SYMBOL
        # -----------------------------------------------------

        symbol = str(
            market.get(
                "symbol"
            )
            or ""
        )

        upper_symbol = symbol.upper()

        base = str(
            market.get(
                "base"
            )
            or ""
        ).upper()

        # -----------------------------------------------------
        # EXCLUDE LEVERAGED / SYNTHETIC PRODUCTS
        # -----------------------------------------------------

        blocked_fragments = (
            ".D",
            ".I",
            "BULL/",
            "BEAR/",
            "2L/",
            "2S/",
            "3L/",
            "3S/",
            "UP/",
            "DOWN/",
        )

        if any(
            fragment in upper_symbol
            for fragment in blocked_fragments
        ):

            return False

        blocked_base_suffixes = (
            "BULL",
            "BEAR",
            "2L",
            "2S",
            "3L",
            "3S",
        )

        if base.endswith(
            blocked_base_suffixes
        ):

            return False

        # -----------------------------------------------------
        # NO EMPTY BASE
        # -----------------------------------------------------

        if not base:

            return False

        return True

    # =========================================================
    # LIQUIDITY SCORE
    # =========================================================

    def _score(
        self,
        quote_volume: float,
        spread_pct: float
    ) -> float:

        # -----------------------------------------------------
        # VOLUME SCORE
        # -----------------------------------------------------

        #
        # $10M+ 24h quote volume receives the maximum
        # volume component.
        #

        volume_score = min(
            1.0,
            max(
                0.0,
                quote_volume
                /
                10_000_000.0
            )
        )

        # -----------------------------------------------------
        # SPREAD SCORE
        # -----------------------------------------------------

        spread_limit = max(
            self.settings.max_spread_pct,
            0.01
        )

        spread_score = max(
            0.0,
            1.0
            -
            (
                spread_pct
                /
                spread_limit
            )
        )

        # -----------------------------------------------------
        # FINAL SCORE
        # -----------------------------------------------------

        return (
            volume_score
            *
            0.70
            +
            spread_score
            *
            0.30
        )

    # =========================================================
    # SAFE FLOAT
    # =========================================================

    @staticmethod
    def _number(
        value,
        default=0.0
    ):

        try:

            if value is None:
                return default

            return float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return default

    # =========================================================
    # REFRESH FULL MARKET UNIVERSE
    # =========================================================

    def refresh(
        self,
        force: bool = False
    ):

        now = time.time()

        # -----------------------------------------------------
        # USE CACHE
        # -----------------------------------------------------

        if (
            not force
            and self.universe
            and
            (
                now
                -
                self.last_refresh
            )
            <
            self.settings.market_refresh_seconds
        ):

            return self.universe

        try:

            # -------------------------------------------------
            # LOAD KRAKEN MARKETS
            # -------------------------------------------------

            markets = (
                self.kraken.exchange.load_markets(
                    reload=force
                )
            )

            self.markets = markets

            # -------------------------------------------------
            # FETCH KRAKEN TICKERS
            # -------------------------------------------------

            #
            # Kraken/CCXT supports fetching multiple/all
            # tickers through fetch_tickers().
            #

            tickers = (
                self.kraken.exchange.fetch_tickers()
            )

            self.last_ticker_count = len(
                tickers
            )

            candidates = []

            # -------------------------------------------------
            # PROCESS EVERY MARKET
            # -------------------------------------------------

            for symbol, market in markets.items():

                try:

                    if not self._is_allowed_market(
                        market
                    ):

                        continue

                    ticker = (
                        tickers.get(
                            symbol
                        )
                        or {}
                    )

                    # -----------------------------------------
                    # PRICE
                    # -----------------------------------------

                    last = self._number(
                        ticker.get(
                            "last"
                        )
                    )

                    bid = self._number(
                        ticker.get(
                            "bid"
                        )
                    )

                    ask = self._number(
                        ticker.get(
                            "ask"
                        )
                    )

                    if (
                        last <= 0
                        or bid <= 0
                        or ask <= 0
                    ):

                        continue

                    # -----------------------------------------
                    # SPREAD
                    # -----------------------------------------

                    spread_pct = max(
                        0.0,
                        (
                            ask
                            /
                            bid
                            -
                            1.0
                        )
                        *
                        100.0
                    )

                    if (
                        spread_pct
                        >
                        self.settings.max_spread_pct
                    ):

                        continue

                    # -----------------------------------------
                    # 24H QUOTE VOLUME
                    # -----------------------------------------

                    quote_volume = self._number(
                        ticker.get(
                            "quoteVolume"
                        )
                    )

                    if (
                        quote_volume
                        <
                        self.settings.min_quote_volume_usd
                    ):

                        continue

                    # -----------------------------------------
                    # MARKET DATA
                    # -----------------------------------------

                    base = str(
                        market.get(
                            "base"
                        )
                        or ""
                    )

                    quote = str(
                        market.get(
                            "quote"
                        )
                        or ""
                    )

                    # -----------------------------------------
                    # LIQUIDITY SCORE
                    # -----------------------------------------

                    liquidity_score = (
                        self._score(
                            quote_volume,
                            spread_pct
                        )
                    )

                    candidates.append(
                        MarketCandidate(

                            symbol=symbol,

                            base=base,

                            quote=quote,

                            last=last,

                            bid=bid,

                            ask=ask,

                            spread_pct=spread_pct,

                            quote_volume=quote_volume,

                            liquidity_score=
                                liquidity_score,
                        )
                    )

                except Exception as exc:

                    print(
                        "MARKET FILTER ERROR:",
                        symbol,
                        type(exc).__name__,
                        str(exc)
                    )

            # -------------------------------------------------
            # SORT
            # -------------------------------------------------

            candidates.sort(
                key=lambda item: (
                    item.liquidity_score,
                    item.quote_volume,
                    -item.spread_pct,
                ),
                reverse=True,
            )

            # -------------------------------------------------
            # ASSIGN RANK
            # -------------------------------------------------

            for rank, candidate in enumerate(
                candidates,
                start=1
            ):

                candidate.rank = rank

            # -------------------------------------------------
            # SAVE
            # -------------------------------------------------

            self.universe = candidates

            try:
                self.kraken.connected = True
                self.kraken.last_error = None
                self.kraken.last_error_type = None
            except Exception:
                pass

            self.last_refresh = now

            self.last_error = None

            # -------------------------------------------------
            # LOG
            # -------------------------------------------------

            scan_count = min(
                len(candidates),
                self.settings.max_scan_symbols
            )

            print(
                "=" * 60
            )

            print(
                "KRAKEN MARKET SCANNER"
            )

            print(
                f"Markets loaded: "
                f"{len(markets)}"
            )

            print(
                f"Tickers received: "
                f"{self.last_ticker_count}"
            )

            print(
                f"Liquid markets: "
                f"{len(candidates)}"
            )

            print(
                f"ML markets this cycle: "
                f"{scan_count}"
            )

            print(
                f"Allowed quotes: "
                f"{','.join(self.settings.allowed_quote_list)}"
            )

            print(
                "=" * 60
            )

            # -------------------------------------------------
            # SHOW TOP MARKETS
            # -------------------------------------------------

            for candidate in candidates[
                :scan_count
            ]:

                print(
                    f"MARKET #{candidate.rank:02d} "
                    f"{candidate.symbol} "
                    f"vol="
                    f"${candidate.quote_volume:,.0f} "
                    f"spread="
                    f"{candidate.spread_pct:.3f}% "
                    f"liquidity="
                    f"{candidate.liquidity_score:.3f}"
                )

            return self.universe

        except Exception as exc:

            self.last_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            try:
                self.kraken.connected = False
                self.kraken.last_error = str(exc)
                self.kraken.last_error_type = type(exc).__name__
            except Exception:
                pass

            print(
                "MARKET SCANNER ERROR:",
                self.last_error
            )

            # -------------------------------------------------
            # IMPORTANT:
            #
            # If a refresh fails temporarily, keep the previous
            # valid universe rather than destroying it.
            # -------------------------------------------------

            return self.universe

    # =========================================================
    # TOP MARKETS FOR ML
    # =========================================================

    def top_symbols(self):

        universe = self.refresh()

        return universe[
            :self.settings.max_scan_symbols
        ]

    # =========================================================
    # GET MARKET METADATA
    # =========================================================

    def metadata(
        self,
        symbol: str
    ) -> dict:

        for candidate in self.universe:

            if candidate.symbol == symbol:

                return candidate.to_dict()

        return {}

    # =========================================================
    # ALL DISCOVERED MARKETS
    # =========================================================

    def all_symbols(self):

        return [
            candidate.symbol
            for candidate in self.universe
        ]

    # =========================================================
    # SCANNER STATUS
    # =========================================================

    def status(self):

        return {

            "markets_loaded":
                len(
                    self.markets
                ),

            "tickers_received":
                self.last_ticker_count,

            "liquid_markets":
                len(
                    self.universe
                ),

            "markets_sent_to_ml":
                min(
                    len(
                        self.universe
                    ),
                    self.settings.max_scan_symbols
                ),

            "last_refresh":
                self.last_refresh,

            "last_error":
                self.last_error,

            "allowed_quotes":
                self.settings.allowed_quote_list,

            "min_volume":
                self.settings.min_quote_volume_usd,

            "max_spread_pct":
                self.settings.max_spread_pct,
        }
