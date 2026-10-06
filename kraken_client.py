from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import ccxt


class KrakenTrader:
    """
    Safe Kraken exchange wrapper.

    PAPER:
        Simulates fills.
        NEVER submits an exchange order.

    LIVE:
        Real orders are allowed only when ALL conditions are true:

            LIVE_TRADING=true
            DRY_RUN=false
            runtime mode == LIVE
            authentication succeeded

    Runtime mode always starts as PAPER.
    """

    # Cache market prices for portfolio valuation.
    PRICE_CACHE_SECONDS = 10

    def __init__(self, settings):

        self.settings = settings

        self.exchange = ccxt.kraken({
            "apiKey": str(
                getattr(
                    settings,
                    "kraken_api_key",
                    "",
                ) or ""
            ).strip(),

            "secret": str(
                getattr(
                    settings,
                    "kraken_api_secret",
                    "",
                ) or ""
            ).strip(),

            "enableRateLimit": True,

            "timeout": 30000,
        })

        # --------------------------------------------------------
        # CONNECTION STATE
        # --------------------------------------------------------

        self.connected = False

        self.authenticated = False

        self.last_error = None
        self.last_error_type = None

        self.last_balance = None
        self.last_auth_test = None
        self.last_order = None

        # --------------------------------------------------------
        # RUNTIME MODE
        # --------------------------------------------------------

        # CRITICAL:
        # Always start PAPER.
        self._mode = "PAPER"

        self._mode_lock = threading.Lock()

        # --------------------------------------------------------
        # PRICE CACHE
        # --------------------------------------------------------

        self._price_cache: Dict[str, Any] = {}

        self._price_cache_lock = (
            threading.Lock()
        )

        # --------------------------------------------------------
        # MARKET CACHE
        # --------------------------------------------------------

        self._markets_loaded = False

        self._markets_lock = threading.Lock()

    # ============================================================
    # MODE
    # ============================================================

    @property
    def mode(self) -> str:

        with self._mode_lock:
            return self._mode

    @property
    def is_paper(self) -> bool:

        return self.mode == "PAPER"

    @property
    def is_live(self) -> bool:

        return self.mode == "LIVE"

    def set_mode(
        self,
        mode: str,
    ) -> str:

        mode = str(
            mode or ""
        ).upper().strip()

        if mode not in {
            "PAPER",
            "LIVE",
        }:
            raise ValueError(
                "Mode must be PAPER or LIVE."
            )

        with self._mode_lock:

            # ----------------------------------------------------
            # PAPER
            # ----------------------------------------------------

            if mode == "PAPER":

                self._mode = "PAPER"

                return self._mode

            # ----------------------------------------------------
            # LIVE SAFETY GATES
            # ----------------------------------------------------

            if not bool(
                getattr(
                    self.settings,
                    "live_trading",
                    False,
                )
            ):
                raise RuntimeError(
                    "LIVE trading is disabled by configuration. "
                    "Set LIVE_TRADING=true before enabling LIVE mode."
                )

            if bool(
                getattr(
                    self.settings,
                    "dry_run",
                    True,
                )
            ):
                raise RuntimeError(
                    "LIVE trading is blocked because DRY_RUN=true. "
                    "Set DRY_RUN=false before enabling LIVE mode."
                )

            if not self.authenticated:
                raise RuntimeError(
                    "Kraken authentication has not been verified."
                )

            self._mode = "LIVE"

            return self._mode

    @property
    def live_orders_enabled(self) -> bool:
        """
        Final gate before a real order can ever reach Kraken.
        """

        return (
            self.mode == "LIVE"
            and bool(
                getattr(
                    self.settings,
                    "live_trading",
                    False,
                )
            )
            and not bool(
                getattr(
                    self.settings,
                    "dry_run",
                    True,
                )
            )
            and bool(
                self.authenticated
            )
        )

    # ============================================================
    # CREDENTIALS
    # ============================================================

    def credentials_configured(self) -> bool:

        key = str(
            getattr(
                self.settings,
                "kraken_api_key",
                "",
            ) or ""
        ).strip()

        secret = str(
            getattr(
                self.settings,
                "kraken_api_secret",
                "",
            ) or ""
        ).strip()

        return bool(
            key and secret
        )

    def _require_credentials(self):

        if not self.credentials_configured():

            raise RuntimeError(
                "Kraken API credentials are not configured."
            )

    # ============================================================
    # MARKET LOADING
    # ============================================================

    def _ensure_markets(self):

        if self._markets_loaded:
            return

        with self._markets_lock:

            if self._markets_loaded:
                return

            self.exchange.load_markets()

            self._markets_loaded = True

    # ============================================================
    # CONNECTION
    # ============================================================

    def test_connection(
        self,
    ) -> Dict[str, Any]:

        try:

            self._ensure_markets()

            self.connected = True

            self.last_error = None
            self.last_error_type = None

            return {
                "ok": True,
                "connected": True,
                "message":
                    "Kraken connection successful.",
            }

        except Exception as exc:

            self.connected = False

            self.last_error = str(exc)
            self.last_error_type = (
                type(exc).__name__
            )

            return {
                "ok": False,
                "connected": False,
                "error": str(exc),
                "error_type":
                    type(exc).__name__,
            }

    # ============================================================
    # AUTHENTICATION
    # ============================================================

    def test_authentication(
        self,
    ) -> Dict[str, Any]:

        try:

            self._require_credentials()

            self._ensure_markets()

            balance = (
                self.exchange.fetch_balance()
            )

            self.authenticated = True
            self.connected = True

            self.last_balance = balance

            self.last_auth_test = (
                time.time()
            )

            self.last_error = None
            self.last_error_type = None

            return {
                "ok": True,
                "authenticated": True,
                "message":
                    "Kraken authentication successful.",
            }

        except Exception as exc:

            self.authenticated = False

            self.last_error = str(exc)
            self.last_error_type = (
                type(exc).__name__
            )

            self.last_auth_test = (
                time.time()
            )

            return {
                "ok": False,
                "authenticated": False,
                "error": str(exc),
                "error_type":
                    type(exc).__name__,
            }

    def connection_status(
        self,
    ) -> Dict[str, Any]:

        return {

            "connected":
                bool(self.connected),

            "authenticated":
                bool(self.authenticated),

            "mode":
                self.mode,

            "is_paper":
                self.is_paper,

            "is_live":
                self.is_live,

            "live_orders_enabled":
                bool(
                    self.live_orders_enabled
                ),

            "credentials_configured":
                bool(
                    self.credentials_configured()
                ),

            "last_error":
                self.last_error,

            "last_error_type":
                self.last_error_type,

            "last_auth_test":
                self.last_auth_test,

        }

    # ============================================================
    # MARKET DATA
    # ============================================================

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 720,
    ):

        self._ensure_markets()

        return self.exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=int(limit),
        )

    def fetch_ticker(
        self,
        symbol: str,
    ):

        self._ensure_markets()

        return self.exchange.fetch_ticker(
            symbol
        )

    # ============================================================
    # BALANCE
    # ============================================================

    def fetch_balance(self):

        self._require_credentials()

        balance = (
            self.exchange.fetch_balance()
        )

        self.last_balance = balance

        self.authenticated = True
        self.connected = True

        self.last_error = None
        self.last_error_type = None

        return balance

    def free_quote(
        self,
        currency: str = "USD",
    ) -> float:

        balance = (
            self.fetch_balance()
        )

        free = (
            balance.get(
                "free",
                {},
            )
            or {}
        )

        return float(
            free.get(currency)
            or 0.0
        )

    def total_quote(
        self,
        currency: str = "USD",
    ) -> float:

        balance = (
            self.fetch_balance()
        )

        total = (
            balance.get(
                "total",
                {},
            )
            or {}
        )

        return float(
            total.get(currency)
            or 0.0
        )

    # ============================================================
    # PORTFOLIO PRICE
    # ============================================================

    def _asset_usd_price(
        self,
        asset: str,
    ) -> float:

        asset = str(
            asset or ""
        ).upper().strip()

        if asset in {
            "USD",
            "USDG",
        }:
            return 1.0

        now = time.time()

        # --------------------------------------------------------
        # CACHE
        # --------------------------------------------------------

        with self._price_cache_lock:

            cached = (
                self._price_cache.get(
                    asset
                )
            )

            if cached:

                cached_time, cached_price = (
                    cached
                )

                if (
                    now - cached_time
                    < self.PRICE_CACHE_SECONDS
                ):

                    return float(
                        cached_price
                    )

        # --------------------------------------------------------
        # PRICE SOURCES
        # --------------------------------------------------------

        candidates = [
            f"{asset}/USD",
            f"{asset}/USDT",
        ]

        for symbol in candidates:

            try:

                self._ensure_markets()

                if (
                    symbol
                    not in self.exchange.markets
                ):
                    continue

                ticker = (
                    self.exchange.fetch_ticker(
                        symbol
                    )
                )

                price = (
                    ticker.get("last")
                    or ticker.get("close")
                    or ticker.get("bid")
                    or ticker.get("ask")
                )

                if price is None:
                    continue

                price = float(price)

                if price <= 0:
                    continue

                with self._price_cache_lock:

                    self._price_cache[
                        asset
                    ] = (
                        time.time(),
                        price,
                    )

                return price

            except Exception:
                continue

        return 0.0

    # ============================================================
    # PORTFOLIO VALUATION
    # ============================================================

    def _calculate_portfolio_value(
        self,
        balance,
    ):

        total_balances = (
            balance.get(
                "total",
                {},
            )
            or {}
        )

        portfolio_value = 0.0

        assets = []

        for asset, raw_amount in (
            total_balances.items()
        ):

            try:

                amount = float(
                    raw_amount or 0.0
                )

            except Exception:
                continue

            if amount <= 0:
                continue

            asset = str(
                asset
            ).upper().strip()

            price = (
                self._asset_usd_price(
                    asset
                )
            )

            if price <= 0:
                continue

            value = (
                amount * price
            )

            portfolio_value += value

            assets.append({
                "asset": asset,
                "amount": amount,
                "price_usd": price,
                "value_usd": value,
            })

        assets.sort(
            key=lambda item:
                item["value_usd"],
            reverse=True,
        )

        return (
            float(portfolio_value),
            assets,
        )

    def account_summary(
        self,
        currency: str = "USD",
    ) -> Dict[str, Any]:

        try:

            balance = (
                self.fetch_balance()
            )

            free = (
                balance.get(
                    "free",
                    {},
                )
                or {}
            )

            used = (
                balance.get(
                    "used",
                    {},
                )
                or {}
            )

            total = (
                balance.get(
                    "total",
                    {},
                )
                or {}
            )

            usd_free = float(
                free.get("USD")
                or 0.0
            )

            usd_used = float(
                used.get("USD")
                or 0.0
            )

            usd_total = float(
                total.get("USD")
                or 0.0
            )

            usdg_free = float(
                free.get("USDG")
                or 0.0
            )

            usdg_used = float(
                used.get("USDG")
                or 0.0
            )

            usdg_total = float(
                total.get("USDG")
                or 0.0
            )

            (
                portfolio_value_usd,
                assets,
            ) = (
                self._calculate_portfolio_value(
                    balance
                )
            )

            return {

                "authenticated":
                    True,

                "free":
                    usd_free,

                "usd_free":
                    usd_free,

                "used":
                    usd_used,

                "total":
                    usd_total,

                "usd_total":
                    usd_total,

                "usdg_free":
                    usdg_free,

                "usdg_used":
                    usdg_used,

                "usdg_total":
                    usdg_total,

                "portfolio_value_usd":
                    portfolio_value_usd,

                "portfolio_assets":
                    assets,

                "currency":
                    currency,

                "error":
                    None,

            }

        except Exception as exc:

            self.last_error = str(exc)
            self.last_error_type = (
                type(exc).__name__
            )

            return {

                "authenticated":
                    bool(
                        self.authenticated
                    ),

                "free":
                    0.0,

                "usd_free":
                    0.0,

                "used":
                    0.0,

                "total":
                    0.0,

                "usd_total":
                    0.0,

                "usdg_free":
                    0.0,

                "usdg_used":
                    0.0,

                "usdg_total":
                    0.0,

                "portfolio_value_usd":
                    0.0,

                "portfolio_assets":
                    [],

                "currency":
                    currency,

                "error":
                    str(exc),

                "error_type":
                    type(exc).__name__,

            }

    # ============================================================
    # MARKET LIMITS
    # ============================================================

    def _market_limits(
        self,
        symbol: str,
    ) -> Dict[str, Any]:

        self._ensure_markets()

        market = (
            self.exchange.market(
                symbol
            )
        )

        limits = (
            market.get(
                "limits",
                {},
            )
            or {}
        )

        amount_limits = (
            limits.get(
                "amount",
                {},
            )
            or {}
        )

        cost_limits = (
            limits.get(
                "cost",
                {},
            )
            or {}
        )

        return {

            "min_amount":
                amount_limits.get(
                    "min"
                ),

            "max_amount":
                amount_limits.get(
                    "max"
                ),

            "min_cost":
                cost_limits.get(
                    "min"
                ),

            "max_cost":
                cost_limits.get(
                    "max"
                ),

        }

    def _normalize_amount(
        self,
        symbol: str,
        amount: float,
    ) -> float:

        self._ensure_markets()

        return float(
            self.exchange.amount_to_precision(
                symbol,
                amount,
            )
        )

    # ============================================================
    # TICKER PRICE HELPERS
    # ============================================================

    @staticmethod
    def _ask_price(
        ticker: Dict[str, Any],
    ) -> float:

        value = (
            ticker.get("ask")
            or ticker.get("last")
            or ticker.get("close")
        )

        if value is None:
            raise RuntimeError(
                "Ticker does not contain a usable ask price."
            )

        value = float(value)

        if value <= 0:
            raise RuntimeError(
                "Ticker ask price is invalid."
            )

        return value

    @staticmethod
    def _bid_price(
        ticker: Dict[str, Any],
    ) -> float:

        value = (
            ticker.get("bid")
            or ticker.get("last")
            or ticker.get("close")
        )

        if value is None:
            raise RuntimeError(
                "Ticker does not contain a usable bid price."
            )

        value = float(value)

        if value <= 0:
            raise RuntimeError(
                "Ticker bid price is invalid."
            )

        return value

    # ============================================================
    # MARKET BUY
    # ============================================================

    def market_buy(
        self,
        symbol: str,
        quote_amount: float,
    ) -> Dict[str, Any]:

        symbol = str(
            symbol or ""
        ).strip()

        quote_amount = float(
            quote_amount
        )

        if not symbol:
            raise ValueError(
                "Symbol is required."
            )

        if quote_amount <= 0:
            raise ValueError(
                "Quote amount must be greater than zero."
            )

        # --------------------------------------------------------
        # MARKET
        # --------------------------------------------------------

        self._ensure_markets()

        if symbol not in self.exchange.markets:

            raise RuntimeError(
                f"{symbol} is not available on Kraken."
            )

        # --------------------------------------------------------
        # TICKER
        # --------------------------------------------------------

        ticker = (
            self.fetch_ticker(
                symbol
            )
        )

        ask = self._ask_price(
            ticker
        )

        # --------------------------------------------------------
        # AMOUNT
        # --------------------------------------------------------

        raw_amount = (
            quote_amount / ask
        )

        amount = (
            self._normalize_amount(
                symbol,
                raw_amount,
            )
        )

        if amount <= 0:
            raise RuntimeError(
                f"Calculated order amount is too small for {symbol}."
            )

        limits = (
            self._market_limits(
                symbol
            )
        )

        min_amount = (
            limits["min_amount"]
        )

        max_amount = (
            limits["max_amount"]
        )

        min_cost = (
            limits["min_cost"]
        )

        max_cost = (
            limits["max_cost"]
        )

        estimated_cost = (
            amount * ask
        )

        # --------------------------------------------------------
        # LIMIT VALIDATION
        # --------------------------------------------------------

        if (
            min_amount is not None
            and amount
            < float(min_amount)
        ):

            raise RuntimeError(
                f"Order amount {amount} is below "
                f"Kraken minimum {min_amount} "
                f"for {symbol}."
            )

        if (
            max_amount is not None
            and amount
            > float(max_amount)
        ):

            raise RuntimeError(
                f"Order amount {amount} exceeds "
                f"Kraken maximum {max_amount} "
                f"for {symbol}."
            )

        if (
            min_cost is not None
            and estimated_cost
            < float(min_cost)
        ):

            raise RuntimeError(
                f"Order value ${estimated_cost:.2f} "
                f"is below Kraken minimum "
                f"${float(min_cost):.2f} "
                f"for {symbol}."
            )

        if (
            max_cost is not None
            and estimated_cost
            > float(max_cost)
        ):

            raise RuntimeError(
                f"Order value ${estimated_cost:.2f} "
                f"exceeds Kraken maximum "
                f"${float(max_cost):.2f} "
                f"for {symbol}."
            )

        # ========================================================
        # PAPER
        # ========================================================

        if not self.live_orders_enabled:

            result = {

                "ok":
                    True,

                "paper":
                    True,

                "live":
                    False,

                "symbol":
                    symbol,

                "side":
                    "buy",

                "amount":
                    amount,

                "quote_amount":
                    estimated_cost,

                "requested_quote_amount":
                    quote_amount,

                "price":
                    ask,

                "average":
                    ask,

                "filled":
                    amount,

                "status":
                    "simulated",

                "order":
                    None,

            }

            self.last_order = result

            return result

        # ========================================================
        # LIVE SAFETY
        # ========================================================

        if not self.authenticated:

            raise RuntimeError(
                "Cannot place live order: "
                "Kraken authentication is not verified."
            )

        available_usd = (
            self.free_quote(
                "USD"
            )
        )

        if available_usd < estimated_cost:

            raise RuntimeError(
                f"Insufficient USD balance. "
                f"Available: ${available_usd:.2f}; "
                f"Required: ${estimated_cost:.2f}."
            )

        # ========================================================
        # LIVE ORDER
        # ========================================================

        try:

            order = (
                self.exchange.create_market_buy_order(
                    symbol,
                    amount,
                )
            )

        except Exception as exc:

            self.last_error = str(exc)
            self.last_error_type = (
                type(exc).__name__
            )

            raise

        # --------------------------------------------------------
        # ACTUAL FILL INFORMATION
        # --------------------------------------------------------

        filled = (
            order.get("filled")
            or amount
        )

        average = (
            order.get("average")
            or order.get("price")
            or ask
        )

        filled = float(
            filled
        )

        average = float(
            average
        )

        actual_cost = (
            order.get("cost")
        )

        if actual_cost is None:

            actual_cost = (
                filled * average
            )

        actual_cost = float(
            actual_cost
        )

        result = {

            "ok":
                True,

            "paper":
                False,

            "live":
                True,

            "symbol":
                symbol,

            "side":
                "buy",

            "amount":
                filled,

            "filled":
                filled,

            "quote_amount":
                actual_cost,

            "requested_quote_amount":
                quote_amount,

            "price":
                average,

            "average":
                average,

            "status":
                order.get(
                    "status"
                ),

            "order":
                order,

        }

        self.last_order = result

        return result

    # ============================================================
    # MARKET SELL
    # ============================================================

    def market_sell(
        self,
        symbol: str,
        amount: float,
    ) -> Dict[str, Any]:

        symbol = str(
            symbol or ""
        ).strip()

        amount = float(
            amount
        )

        if not symbol:
            raise ValueError(
                "Symbol is required."
            )

        if amount <= 0:
            raise ValueError(
                "Sell amount must be greater than zero."
            )

        # --------------------------------------------------------
        # MARKET
        # --------------------------------------------------------

        self._ensure_markets()

        if symbol not in self.exchange.markets:

            raise RuntimeError(
                f"{symbol} is not available on Kraken."
            )

        # --------------------------------------------------------
        # TICKER
        # --------------------------------------------------------

        ticker = (
            self.fetch_ticker(
                symbol
            )
        )

        bid = self._bid_price(
            ticker
        )

        # --------------------------------------------------------
        # AMOUNT
        # --------------------------------------------------------

        normalized_amount = (
            self._normalize_amount(
                symbol,
                amount,
            )
        )

        if normalized_amount <= 0:

            raise RuntimeError(
                f"Normalized sell amount is too small for {symbol}."
            )

        limits = (
            self._market_limits(
                symbol
            )
        )

        min_amount = (
            limits["min_amount"]
        )

        max_amount = (
            limits["max_amount"]
        )

        min_cost = (
            limits["min_cost"]
        )

        max_cost = (
            limits["max_cost"]
        )

        estimated_value = (
            normalized_amount * bid
        )

        # --------------------------------------------------------
        # LIMIT VALIDATION
        # --------------------------------------------------------

        if (
            min_amount is not None
            and normalized_amount
            < float(min_amount)
        ):

            raise RuntimeError(
                f"Sell amount {normalized_amount} "
                f"is below Kraken minimum "
                f"{min_amount} for {symbol}."
            )

        if (
            max_amount is not None
            and normalized_amount
            > float(max_amount)
        ):

            raise RuntimeError(
                f"Sell amount {normalized_amount} "
                f"exceeds Kraken maximum "
                f"{max_amount} for {symbol}."
            )

        if (
            min_cost is not None
            and estimated_value
            < float(min_cost)
        ):

            raise RuntimeError(
                f"Estimated order value "
                f"${estimated_value:.2f} "
                f"is below Kraken minimum "
                f"${float(min_cost):.2f} "
                f"for {symbol}."
            )

        if (
            max_cost is not None
            and estimated_value
            > float(max_cost)
        ):

            raise RuntimeError(
                f"Estimated order value "
                f"${estimated_value:.2f} "
                f"exceeds Kraken maximum "
                f"${float(max_cost):.2f} "
                f"for {symbol}."
            )

        # ========================================================
        # PAPER
        # ========================================================

        if not self.live_orders_enabled:

            result = {

                "ok":
                    True,

                "paper":
                    True,

                "live":
                    False,

                "symbol":
                    symbol,

                "side":
                    "sell",

                "amount":
                    normalized_amount,

                "filled":
                    normalized_amount,

                "price":
                    bid,

                "average":
                    bid,

                "quote_amount":
                    estimated_value,

                "status":
                    "simulated",

                "order":
                    None,

            }

            self.last_order = result

            return result

        # ========================================================
        # LIVE SAFETY
        # ========================================================

        if not self.authenticated:

            raise RuntimeError(
                "Cannot place live order: "
                "Kraken authentication is not verified."
            )

        # ========================================================
        # LIVE ORDER
        # ========================================================

        try:

            order = (
                self.exchange.create_market_sell_order(
                    symbol,
                    normalized_amount,
                )
            )

        except Exception as exc:

            self.last_error = str(exc)
            self.last_error_type = (
                type(exc).__name__
            )

            raise

        # --------------------------------------------------------
        # ACTUAL FILL
        # --------------------------------------------------------

        filled = (
            order.get("filled")
            or normalized_amount
        )

        average = (
            order.get("average")
            or order.get("price")
            or bid
        )

        filled = float(
            filled
        )

        average = float(
            average
        )

        actual_proceeds = (
            order.get("cost")
        )

        if actual_proceeds is None:

            actual_proceeds = (
                filled * average
            )

        actual_proceeds = float(
            actual_proceeds
        )

        result = {

            "ok":
                True,

            "paper":
                False,

            "live":
                True,

            "symbol":
                symbol,

            "side":
                "sell",

            "amount":
                filled,

            "filled":
                filled,

            "price":
                average,

            "average":
                average,

            "quote_amount":
                actual_proceeds,

            "status":
                order.get(
                    "status"
                ),

            "order":
                order,

        }

        self.last_order = result

        return result

    # ============================================================
    # SAFE STOP
    # ============================================================

    def force_paper_mode(self) -> str:
        """
        Emergency helper.

        Forces this runtime instance back to PAPER.
        """

        with self._mode_lock:
            self._mode = "PAPER"

        return self._mode

    # ============================================================
    # DEBUG / STATUS
    # ============================================================

    def status(self) -> Dict[str, Any]:

        return {

            "mode":
                self.mode,

            "paper":
                self.is_paper,

            "live":
                self.is_live,

            "live_orders_enabled":
                self.live_orders_enabled,

            "connected":
                self.connected,

            "authenticated":
                self.authenticated,

            "credentials_configured":
                self.credentials_configured(),

            "last_error":
                self.last_error,

            "last_error_type":
                self.last_error_type,

            "last_auth_test":
                self.last_auth_test,

            "last_order":
                self.last_order,

        }
