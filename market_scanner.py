from __future__ import annotations

import json
import time
import urllib.request
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
    market_type: str = "CRYPTO"
    venue: str = "KRAKEN_SPOT"

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
        self.futures_markets: list[dict] = []
        self.last_futures_refresh = 0.0
        self.last_futures_error = None

    # =========================================================
    # MARKET FAMILY CLASSIFICATION
    # =========================================================

    @staticmethod
    def _market_type(
        market: dict[str, Any]
    ) -> str:

        base = str(
            market.get("base") or ""
        ).upper()

        quote = str(
            market.get("quote") or ""
        ).upper()

        if base.endswith("X") and len(base) > 1:
            return "XSTOCKS"

        fiat = {
            "USD", "EUR", "GBP", "CHF", "JPY", "CAD",
            "AUD", "NZD", "SEK", "NOK", "DKK", "SGD",
        }

        if base in fiat and quote in fiat:
            return "FOREX"

        return "CRYPTO"

    # =========================================================
    # FUTURES DISCOVERY
    # =========================================================

    def _refresh_futures(self, force: bool = False):

        now = time.time()

        if (
            not force
            and self.last_futures_refresh
            and now - self.last_futures_refresh
            < self.settings.market_refresh_seconds
        ):
            return self.futures_markets

        url = (
            "https://futures.kraken.com/"
            "derivatives/api/v3/instruments"
        )

        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Kraken-bott/1.0",
                },
            )

            with urllib.request.urlopen(
                request,
                timeout=10,
            ) as response:
                payload = json.loads(
                    response.read().decode("utf-8")
                )

            instruments = payload.get("instruments", [])

            if not isinstance(instruments, list):
                instruments = []

            active = []

            for instrument in instruments:
                if not isinstance(instrument, dict):
                    continue

                if instrument.get("tradeable") is False:
                    continue

                symbol = str(
                    instrument.get("symbol")
                    or instrument.get("instrumentName")
                    or ""
                ).strip()

                if not symbol:
                    continue

                item = dict(instrument)
                item["symbol"] = symbol
                item["market_type"] = "FUTURES"
                item["venue"] = "KRAKEN_FUTURES"
                active.append(item)

            self.futures_markets = active
            self.last_futures_refresh = now
            self.last_futures_error = None

        except Exception as exc:
            self.last_futures_error = (
                f"{type(exc).__name__}: {exc}"
            )

        return self.futures_markets

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

        market_type = self._market_type(market)

        quote = str(
            market.get(
                "quote"
            )
            or ""
        ).upper()

        if (
            market_type != "FOREX"
            and quote not in self.settings.allowed_quote_list
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

                    # Kraken/CCXT may omit quoteVolume. Derive it
                    # from base volume and price when possible.
                    if quote_volume <= 0:
                        base_volume = self._number(
                            ticker.get("baseVolume")
                        )
                        if base_volume > 0:
                            quote_volume = base_volume * last

                    # PAPER mode must not become marketless merely
                    # because a ticker omitted quoteVolume.
                    paper_mode = bool(
                        getattr(self.kraken, "is_paper", False)
                    )

                    if (
                        quote_volume
                        <
                        self.settings.min_quote_volume_usd
                        and not paper_mode
                    ):
                        continue

                    if quote_volume <= 0:
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

                            market_type=
                                self._market_type(market),

                            venue=
                                "KRAKEN_SPOT",
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

            self._refresh_futures(force=force)

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

        discovered_counts = {
            "CRYPTO": 0,
            "FOREX": 0,
            "XSTOCKS": 0,
        }

        for market in self.markets.values():
            if isinstance(market, dict):
                market_type = self._market_type(market)
                discovered_counts[market_type] = (
                    discovered_counts.get(market_type, 0) + 1
                )

        liquid_counts = {
            "CRYPTO": 0,
            "FOREX": 0,
            "XSTOCKS": 0,
        }

        for candidate in self.universe:
            liquid_counts[candidate.market_type] = (
                liquid_counts.get(candidate.market_type, 0) + 1
            )

        return {

            "markets_loaded":
                len(self.markets),

            "tickers_received":
                self.last_ticker_count,

            "liquid_markets":
                len(self.universe),

            "markets_discovered":
                len(self.markets),

            "crypto_markets":
                discovered_counts["CRYPTO"],

            "forex_markets":
                discovered_counts["FOREX"],

            "xstocks_markets":
                discovered_counts["XSTOCKS"],

            "liquid_crypto_markets":
                liquid_counts["CRYPTO"],

            "liquid_forex_markets":
                liquid_counts["FOREX"],

            "liquid_xstocks_markets":
                liquid_counts["XSTOCKS"],

            "futures_markets":
                len(self.futures_markets),

            "markets_sent_to_ml":
                min(
                    len(self.universe),
                    self.settings.max_scan_symbols
                ),

            "last_refresh":
                self.last_refresh,

            "last_futures_refresh":
                self.last_futures_refresh,

            "last_error":
                self.last_error,

            "futures_error":
                self.last_futures_error,

            "allowed_quotes":
                self.settings.allowed_quote_list,

            "min_volume":
                self.settings.min_quote_volume_usd,

            "max_spread_pct":
                self.settings.max_spread_pct,
        }

