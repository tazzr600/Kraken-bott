from __future__ import annotations

import ccxt


class KrakenTrader:
    def __init__(self, settings):
        self.settings = settings

        self.exchange = ccxt.kraken({
            "apiKey": settings.kraken_api_key,
            "secret": settings.kraken_api_secret,
            "enableRateLimit": True,
            "timeout": 30000,
        })

        self.connected = False
        self.authenticated = False
        self.last_error = None
        self.last_balance = None

    @property
    def live_orders_enabled(self):
        return (
            self.settings.live_trading
            and not self.settings.dry_run
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

    def test_connection(self):
        try:
            self.exchange.load_markets()

            self.connected = True
            self.last_error = None

            return {
                "connected": True,
                "authenticated": self.authenticated,
                "error": None,
            }

        except Exception as e:
            self.connected = False
            self.last_error = (
                f"{type(e).__name__}: {e}"
            )

            return {
                "connected": False,
                "authenticated": False,
                "error": self.last_error,
            }

    def test_authentication(self):
        try:
            self._require_credentials()

            balance = self.exchange.fetch_balance()

            self.authenticated = True
            self.connected = True
            self.last_error = None
            self.last_balance = balance

            return {
                "connected": True,
                "authenticated": True,
                "error": None,
            }

        except Exception as e:
            self.authenticated = False
            self.last_error = (
                f"{type(e).__name__}: {e}"
            )

            return {
                "connected": self.connected,
                "authenticated": False,
                "error": self.last_error,
            }

    def connection_status(self):
        return {
            "connected": self.connected,
            "authenticated": self.authenticated,
            "live_orders_enabled": self.live_orders_enabled,
            "error": self.last_error,
        }

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

    def free_quote(self, currency="USD"):
        balance = self.exchange.fetch_balance()

        free = (
            balance
            .get("free", {})
            .get(currency)
        )

        if free is None:
            return 0.0

        return float(free)

    # =========================================================
    # BUY
    # =========================================================

    def market_buy(
        self,
        symbol,
        quote_amount,
    ):
        """
        DRY_RUN:
            Gets the real Kraken market price but DOES NOT
            submit an order.

        LIVE:
            Actually submits the market order.
        """

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
                f"Unable to determine "
                f"{symbol} market price"
            )

        amount = quote_amount / ask

        # -----------------------------------------------------
        # PAPER MODE
        # -----------------------------------------------------

        if not self.live_orders_enabled:

            return {
                "order_id": (
                    f"PAPER-BUY-{symbol}"
                ),
                "price": ask,
                "amount": amount,
                "raw": {
                    "paper": True,
                    "symbol": symbol,
                    "quote_amount": quote_amount,
                },
            }

        # -----------------------------------------------------
        # LIVE MODE
        # -----------------------------------------------------

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
            "order_id": order.get("id"),
            "price": price,
            "amount": filled,
            "raw": order,
        }

    # =========================================================
    # SELL
    # =========================================================

    def market_sell(
        self,
        symbol,
        amount,
    ):
        """
        DRY_RUN:
            Gets the real Kraken bid price but DOES NOT
            submit an order.

        LIVE:
            Actually submits the market order.
        """

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
                f"Unable to determine "
                f"{symbol} market price"
            )

        # -----------------------------------------------------
        # PAPER MODE
        # -----------------------------------------------------

        if not self.live_orders_enabled:

            return {
                "order_id": (
                    f"PAPER-SELL-{symbol}"
                ),
                "price": bid,
                "amount": amount,
                "raw": {
                    "paper": True,
                    "symbol": symbol,
                },
            }

        # -----------------------------------------------------
        # LIVE MODE
        # -----------------------------------------------------

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
            "order_id": order.get("id"),
            "price": price,
            "amount": filled,
            "raw": order,
        }
