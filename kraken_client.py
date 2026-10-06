from __future__ import annotations

import ccxt
import threading
from typing import Any, Dict, Optional


class KrakenTrader:

    def __init__(self, settings):

        self.settings = settings

        self.exchange = ccxt.kraken({
            "apiKey": settings.kraken_api_key.strip(),
            "secret": settings.kraken_api_secret.strip(),
            "enableRateLimit": True,
            "timeout": 30000,
        })

        self.connected = False
        self.authenticated = False

        self.last_error = None
        self.last_error_type = None

        self.last_balance = None
        self.last_auth_test = None
        self.last_order = None

        # --------------------------------------------------
        # RUNTIME TRADING MODE
        # --------------------------------------------------
        #
        # Always starts in PAPER.
        #
        # The dashboard can explicitly switch to LIVE.
        # This prevents a Railway restart/redeploy from
        # automatically turning real trading on.
        #
        self._mode = "PAPER"

        self._mode_lock = threading.Lock()

    # ======================================================
    # TRADING MODE
    # ======================================================

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

    def set_mode(self, mode: str):

        mode = str(mode).strip().upper()

        if mode not in {"PAPER", "LIVE"}:

            raise ValueError(
                "Trading mode must be PAPER or LIVE"
            )

        # --------------------------------------------------
        # PAPER
        # --------------------------------------------------

        if mode == "PAPER":

            with self._mode_lock:
                self._mode = "PAPER"

            return {
                "ok": True,
                "mode": "PAPER",
                "live_orders_enabled": False,
            }

        # --------------------------------------------------
        # LIVE
        # --------------------------------------------------

        # LIVE requires both configuration flags AND
        # successful Kraken authentication.

        if not self.settings.live_trading:

            raise RuntimeError(
                "LIVE trading is disabled by configuration. "
                "Set LIVE_TRADING=true before enabling LIVE mode."
            )

        if self.settings.dry_run:

            raise RuntimeError(
                "DRY_RUN is enabled. "
                "Disable DRY_RUN before enabling LIVE mode."
            )

        if not self.authenticated:

            raise RuntimeError(
                "Kraken account is not authenticated. "
                "Test Kraken authentication before enabling LIVE mode."
            )

        with self._mode_lock:
            self._mode = "LIVE"

        return {
            "ok": True,
            "mode": "LIVE",
            "live_orders_enabled": True,
        }

    # ======================================================
    # LIVE ORDER SAFETY
    # ======================================================

    @property
    def live_orders_enabled(self):

        return (
            self.mode == "LIVE"
            and self.settings.live_trading
            and not self.settings.dry_run
            and self.authenticated
        )

    # ======================================================
    # CREDENTIAL CHECK
    # ======================================================

    def credentials_configured(self):

        return bool(
            self.settings.kraken_api_key
            and self.settings.kraken_api_secret
        )

    def _require_credentials(self):

        if not self.settings.kraken_api_key:

            raise RuntimeError(
                "KRAKEN_API_KEY is missing"
            )

        if not self.settings.kraken_api_secret:

            raise RuntimeError(
                "KRAKEN_API_SECRET is missing"
            )

    # ======================================================
    # CONNECTION
    # ======================================================

    def test_connection(self):

        try:

            self.exchange.load_markets()

            self.connected = True
            self.last_error = None
            self.last_error_type = None

            return {
                "connected": True,
                "authenticated": self.authenticated,
                "error": None,
            }

        except Exception as e:

            self.connected = False

            self.last_error_type = type(e).__name__
            self.last_error = (
                f"{type(e).__name__}: {e}"
            )

            return {
                "connected": False,
                "authenticated": False,
                "error": self.last_error,
            }

    # ======================================================
    # AUTHENTICATION
    # ======================================================

    def test_authentication(self):

        try:

            self._require_credentials()

            balance = self.exchange.fetch_balance()

            self.last_balance = balance

            self.authenticated = True
            self.connected = True

            self.last_error = None
            self.last_error_type = None

            self.last_auth_test = {
                "success": True,
                "message":
                    "Kraken private API authenticated",
            }

            return {
                "connected": True,
                "authenticated": True,
                "error": None,
                "message":
                    "Kraken authentication successful",
            }

        except ccxt.AuthenticationError as e:

            self.authenticated = False

            self.last_error_type = (
                "AuthenticationError"
            )

            self.last_error = (
                f"Kraken authentication failed: {e}"
            )

            self.last_auth_test = {
                "success": False,
                "message": self.last_error,
            }

            return {
                "connected": self.connected,
                "authenticated": False,
                "error": self.last_error,
                "error_type":
                    self.last_error_type,
            }

        except ccxt.PermissionDenied as e:

            self.authenticated = False

            self.last_error_type = (
                "PermissionDenied"
            )

            self.last_error = (
                f"Kraken API permission denied: {e}"
            )

            self.last_auth_test = {
                "success": False,
                "message": self.last_error,
            }

            return {
                "connected": self.connected,
                "authenticated": False,
                "error": self.last_error,
                "error_type":
                    self.last_error_type,
            }

        except Exception as e:

            self.authenticated = False

            self.last_error_type = type(e).__name__
            self.last_error = (
                f"{type(e).__name__}: {e}"
            )

            self.last_auth_test = {
                "success": False,
                "message": self.last_error,
            }

            return {
                "connected": self.connected,
                "authenticated": False,
                "error": self.last_error,
                "error_type":
                    self.last_error_type,
            }

    # ======================================================
    # STATUS
    # ======================================================

    def connection_status(self):

        return {
            "connected": self.connected,
            "authenticated": self.authenticated,

            "mode": self.mode,

            "live_orders_enabled":
                self.live_orders_enabled,

            "credentials_configured":
                self.credentials_configured(),

            "error":
                self.last_error,

            "error_type":
                self.last_error_type,
        }

    # ======================================================
    # MARKET DATA
    # ======================================================

    def fetch_ohlcv(
        self,
        symbol,
        timeframe,
        limit,
    ):

        return self.exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit,
        )

    def fetch_ticker(self, symbol):

        return self.exchange.fetch_ticker(
            symbol
        )

    # ======================================================
    # BALANCE
    # ======================================================

    def fetch_balance(self):

        self._require_credentials()

        if not self.authenticated:

            raise RuntimeError(
                "Kraken account is not authenticated"
            )

        balance = self.exchange.fetch_balance()

        self.last_balance = balance

        return balance

    def free_quote(
        self,
        currency="USD",
    ):

        balance = self.fetch_balance()

        free = (
            balance
            .get("free", {})
            .get(currency)
        )

        if free is None:
            return 0.0

        return float(free)

    def total_quote(
        self,
        currency="USD",
    ):

        balance = self.fetch_balance()

        total = (
            balance
            .get("total", {})
            .get(currency)
        )

        if total is None:
            return 0.0

        return float(total)

    def account_summary(
        self,
        currency="USD",
    ):

        if not self.authenticated:

            return {
                "authenticated": False,
                "currency": currency,
                "free": 0.0,
                "used": 0.0,
                "total": 0.0,
                "error":
                    "Kraken account is not authenticated",
            }

        try:

            balance = self.fetch_balance()

            free = float(
                balance
                .get("free", {})
                .get(currency)
                or 0
            )

            used = float(
                balance
                .get("used", {})
                .get(currency)
                or 0
            )

            total = float(
                balance
                .get("total", {})
                .get(currency)
                or 0
            )

            return {
                "authenticated": True,
                "currency": currency,
                "free": free,
                "used": used,
                "total": total,
                "error": None,
            }

        except Exception as e:

            return {
                "authenticated":
                    self.authenticated,

                "currency":
                    currency,

                "free": 0.0,
                "used": 0.0,
                "total": 0.0,

                "error":
                    f"{type(e).__name__}: {e}",
            }

    # ======================================================
    # MARKET PRECISION
    # ======================================================

    def _normalize_amount(
        self,
        symbol,
        amount,
    ):

        try:

            normalized = self.exchange.amount_to_precision(
                symbol,
                amount,
            )

            return float(normalized)

        except Exception:

            return float(amount)

    def _market_limits(
        self,
        symbol,
    ):

        market = self.exchange.market(symbol)

        limits = market.get(
            "limits",
            {},
        )

        amount_limits = limits.get(
            "amount",
            {},
        )

        cost_limits = limits.get(
            "cost",
            {},
        )

        return {
            "min_amount":
                amount_limits.get("min"),

            "max_amount":
                amount_limits.get("max"),

            "min_cost":
                cost_limits.get("min"),

            "max_cost":
                cost_limits.get("max"),
        }

    # ======================================================
    # MARKET BUY
    # ======================================================

    def market_buy(
        self,
        symbol,
        quote_amount,
    ):

        quote_amount = float(quote_amount)

        if quote_amount <= 0:

            raise ValueError(
                "quote_amount must be greater than zero"
            )

        ticker = self.exchange.fetch_ticker(
            symbol
        )

        ask = float(
            ticker.get("ask")
            or ticker.get("last")
            or 0
        )

        if ask <= 0:

            raise RuntimeError(
                f"Unable to determine {symbol} market price"
            )

        amount = quote_amount / ask

        amount = self._normalize_amount(
            symbol,
            amount,
        )

        if amount <= 0:

            raise RuntimeError(
                f"Calculated order amount for {symbol} is zero"
            )

        limits = self._market_limits(symbol)

        min_amount = limits.get("min_amount")
        max_amount = limits.get("max_amount")
        min_cost = limits.get("min_cost")
        max_cost = limits.get("max_cost")

        if (
            min_amount is not None
            and amount < float(min_amount)
        ):

            raise RuntimeError(
                f"{symbol} order amount {amount} "
                f"is below Kraken minimum "
                f"{min_amount}"
            )

        if (
            max_amount is not None
            and amount > float(max_amount)
        ):

            raise RuntimeError(
                f"{symbol} order amount {amount} "
                f"exceeds Kraken maximum "
                f"{max_amount}"
            )

        estimated_cost = amount * ask

        if (
            min_cost is not None
            and estimated_cost < float(min_cost)
        ):

            raise RuntimeError(
                f"{symbol} order value ${estimated_cost:.2f} "
                f"is below Kraken minimum "
                f"${float(min_cost):.2f}"
            )

        if (
            max_cost is not None
            and estimated_cost > float(max_cost)
        ):

            raise RuntimeError(
                f"{symbol} order value ${estimated_cost:.2f} "
                f"exceeds Kraken maximum "
                f"${float(max_cost):.2f}"
            )

        # ==================================================
        # PAPER MODE
        # ==================================================

        if not self.live_orders_enabled:

            result = {
                "order_id":
                    f"PAPER-BUY-{symbol}",

                "symbol":
                    symbol,

                "side":
                    "buy",

                "price":
                    ask,

                "amount":
                    amount,

                "quote_amount":
                    quote_amount,

                "filled":
                    amount,

                "paper":
                    True,

                "raw": {
                    "paper": True,
                    "symbol":
                        symbol,
                    "quote_amount":
                        quote_amount,
                },
            }

            self.last_order = result

            return result

        # ==================================================
        # LIVE MODE
        # ==================================================

        if not self.authenticated:

            raise RuntimeError(
                "Kraken account is not authenticated"
            )

        available = self.free_quote("USD")

        if quote_amount > available:

            raise RuntimeError(
                f"Insufficient Kraken USD balance. "
                f"Requested ${quote_amount:.2f}, "
                f"available ${available:.2f}"
            )

        order = (
            self.exchange
            .create_market_buy_order(
                symbol,
                amount,
            )
        )

        price = float(
            order.get("average")
            or order.get("price")
            or ask
        )

        filled = float(
            order.get("filled")
            or amount
        )

        result = {
            "order_id":
                order.get("id"),

            "symbol":
                symbol,

            "side":
                "buy",

            "price":
                price,

            "amount":
                amount,

            "filled":
                filled,

            "quote_amount":
                quote_amount,

            "paper":
                False,

            "raw":
                order,
        }

        self.last_order = result

        return result

    # ======================================================
    # MARKET SELL
    # ======================================================

    def market_sell(
        self,
        symbol,
        amount,
    ):

        amount = float(amount)

        if amount <= 0:

            raise ValueError(
                "amount must be greater than zero"
            )

        ticker = self.exchange.fetch_ticker(
            symbol
        )

        bid = float(
            ticker.get("bid")
            or ticker.get("last")
            or 0
        )

        if bid <= 0:

            raise RuntimeError(
                f"Unable to determine {symbol} market price"
            )

        amount = self._normalize_amount(
            symbol,
            amount,
        )

        if amount <= 0:

            raise RuntimeError(
                f"Calculated sell amount for {symbol} is zero"
            )

        limits = self._market_limits(symbol)

        min_amount = limits.get("min_amount")
        max_amount = limits.get("max_amount")

        if (
            min_amount is not None
            and amount < float(min_amount)
        ):

            raise RuntimeError(
                f"{symbol} sell amount {amount} "
                f"is below Kraken minimum "
                f"{min_amount}"
            )

        if (
            max_amount is not None
            and amount > float(max_amount)
        ):

            raise RuntimeError(
                f"{symbol} sell amount {amount} "
                f"exceeds Kraken maximum "
                f"{max_amount}"
            )

        # ==================================================
        # PAPER MODE
        # ==================================================

        if not self.live_orders_enabled:

            result = {
                "order_id":
                    f"PAPER-SELL-{symbol}",

                "symbol":
                    symbol,

                "side":
                    "sell",

                "price":
                    bid,

                "amount":
                    amount,

                "filled":
                    amount,

                "paper":
                    True,

                "raw": {
                    "paper": True,
                    "symbol":
                        symbol,
                },
            }

            self.last_order = result

            return result

        # ==================================================
        # LIVE MODE
        # ==================================================

        if not self.authenticated:

            raise RuntimeError(
                "Kraken account is not authenticated"
            )

        order = (
            self.exchange
            .create_market_sell_order(
                symbol,
                amount,
            )
        )

        price = float(
            order.get("average")
            or order.get("price")
            or bid
        )

        filled = float(
            order.get("filled")
            or amount
        )

        result = {
            "order_id":
                order.get("id"),

            "symbol":
                symbol,

            "side":
                "sell",

            "price":
                price,

            "amount":
                amount,

            "filled":
                filled,

            "paper":
                False,

            "raw":
                order,
        }

        self.last_order = result

        return result
