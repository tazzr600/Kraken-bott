from __future__ import annotations

import ccxt


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

    # --------------------------------------------------
    # LIVE ORDER SAFETY
    # --------------------------------------------------

    @property
    def live_orders_enabled(self):

        return (
            self.settings.live_trading
            and not self.settings.dry_run
        )

    # --------------------------------------------------
    # CREDENTIAL CHECK
    # --------------------------------------------------

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

    # --------------------------------------------------
    # PUBLIC CONNECTION
    # --------------------------------------------------

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

            self.last_error_type = (
                type(e).__name__
            )

            self.last_error = (
                f"{type(e).__name__}: {e}"
            )

            return {
                "connected": False,
                "authenticated": False,
                "error": self.last_error,
            }

    # --------------------------------------------------
    # PRIVATE AUTHENTICATION
    # --------------------------------------------------

    def test_authentication(self):

        try:

            self._require_credentials()

            # Force CCXT to use the private Kraken
            # balance endpoint.
            balance = self.exchange.fetch_balance()

            self.last_balance = balance

            self.authenticated = True
            self.connected = True

            self.last_error = None
            self.last_error_type = None

            self.last_auth_test = {
                "success": True,
                "message": "Kraken private API authenticated",
            }

            return {
                "connected": True,
                "authenticated": True,
                "error": None,
                "message": "Kraken authentication successful",
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
                "error_type": self.last_error_type,
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
                "error_type": self.last_error_type,
            }

        except Exception as e:

            self.authenticated = False

            self.last_error_type = (
                type(e).__name__
            )

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
                "error_type": self.last_error_type,
            }

    # --------------------------------------------------
    # STATUS
    # --------------------------------------------------

    def connection_status(self):

        return {
            "connected": self.connected,
            "authenticated": self.authenticated,
            "live_orders_enabled": self.live_orders_enabled,
            "credentials_configured": self.credentials_configured(),
            "error": self.last_error,
            "error_type": self.last_error_type,
        }

    # --------------------------------------------------
    # MARKET DATA
    # --------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol,
        timeframe,
        limit
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

    # --------------------------------------------------
    # ACCOUNT BALANCE
    # --------------------------------------------------

    def free_quote(
        self,
        currency="USD"
    ):

        self._require_credentials()

        balance = self.exchange.fetch_balance()

        free = (
            balance
            .get("free", {})
            .get(currency)
        )

        if free is None:
            return 0.0

        return float(free)

    # --------------------------------------------------
    # MARKET BUY
    # --------------------------------------------------

    def market_buy(
        self,
        symbol,
        quote_amount
    ):

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

        amount = (
            quote_amount / ask
        )

        # PAPER MODE
        #
        # Absolutely no private order is sent.

        if not self.live_orders_enabled:

            return {
                "order_id":
                    f"PAPER-BUY-{symbol}",

                "price":
                    ask,

                "amount":
                    amount,

                "raw": {
                    "paper": True,
                    "symbol": symbol,
                    "quote_amount":
                        quote_amount,
                },
            }

        # LIVE MODE ONLY

        if not self.authenticated:

            raise RuntimeError(
                "Kraken account is not authenticated"
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

        return {
            "order_id":
                order.get("id"),

            "price":
                price,

            "amount":
                filled,

            "raw":
                order,
        }

    # --------------------------------------------------
    # MARKET SELL
    # --------------------------------------------------

    def market_sell(
        self,
        symbol,
        amount
    ):

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

        # PAPER MODE

        if not self.live_orders_enabled:

            return {
                "order_id":
                    f"PAPER-SELL-{symbol}",

                "price":
                    bid,

                "amount":
                    amount,

                "raw": {
                    "paper": True,
                    "symbol": symbol,
                },
            }

        # LIVE MODE ONLY

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

        return {
            "order_id":
                order.get("id"),

            "price":
                price,

            "amount":
                filled,

            "raw":
                order,
        }
