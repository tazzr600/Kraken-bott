from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import ccxt


class KrakenTrader:
    """
    Kraken exchange wrapper.

    PAPER:
        No real orders are submitted.

    LIVE:
        Real Kraken orders are allowed only when:
            LIVE_TRADING=true
            DRY_RUN=false
            Kraken authentication succeeds
            runtime mode is explicitly LIVE
    """

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

        # Runtime mode ALWAYS starts in PAPER.
        self._mode = "PAPER"
        self._mode_lock = threading.Lock()

        # Small price cache so portfolio valuation does not hammer Kraken.
        self._price_cache = {}
        self._price_cache_lock = threading.Lock()

    # ============================================================
    # MODE
    # ============================================================

    @property
    def mode(self):
        with self._mode_lock:
            return self._mode

    @property
    def is_paper(self):
        return self.mode == "PAPER"

    @property
    def is_live(self):
        return self.mode == "LIVE"

    def set_mode(self, mode):
        mode = str(mode or "").upper().strip()

        if mode not in {"PAPER", "LIVE"}:
            raise ValueError("Mode must be PAPER or LIVE.")

        with self._mode_lock:
            if mode == "PAPER":
                self._mode = "PAPER"
                return self._mode

            # LIVE requires every safety condition.
            if not bool(self.settings.live_trading):
                raise RuntimeError(
                    "LIVE trading is disabled by configuration. "
                    "Set LIVE_TRADING=true before enabling LIVE mode."
                )

            if bool(self.settings.dry_run):
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
    def live_orders_enabled(self):
        return (
            self.mode == "LIVE"
            and bool(self.settings.live_trading)
            and not bool(self.settings.dry_run)
            and bool(self.authenticated)
        )

    # ============================================================
    # CREDENTIALS
    # ============================================================

    def credentials_configured(self):
        return bool(
            getattr(self.settings, "kraken_api_key", "").strip()
            and getattr(self.settings, "kraken_api_secret", "").strip()
        )

    def _require_credentials(self):
        if not self.credentials_configured():
            raise RuntimeError(
                "Kraken API credentials are not configured."
            )

    # ============================================================
    # CONNECTION / AUTH
    # ============================================================

    def test_connection(self):
        try:
            self.exchange.load_markets()

            self.connected = True
            self.last_error = None
            self.last_error_type = None

            return {
                "ok": True,
                "connected": True,
                "message": "Kraken connection successful.",
            }

        except Exception as exc:
            self.connected = False
            self.last_error = str(exc)
            self.last_error_type = type(exc).__name__

            return {
                "ok": False,
                "connected": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

    def test_authentication(self):
        try:
            self._require_credentials()

            balance = self.exchange.fetch_balance()

            self.authenticated = True
            self.connected = True
            self.last_balance = balance
            self.last_auth_test = time.time()

            self.last_error = None
            self.last_error_type = None

            return {
                "ok": True,
                "authenticated": True,
                "message": "Kraken authentication successful.",
            }

        except Exception as exc:
            self.authenticated = False
            self.last_error = str(exc)
            self.last_error_type = type(exc).__name__
            self.last_auth_test = time.time()

            return {
                "ok": False,
                "authenticated": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

    def connection_status(self):
        return {
            "connected": bool(self.connected),
            "authenticated": bool(self.authenticated),
            "mode": self.mode,
            "live_orders_enabled": bool(self.live_orders_enabled),
            "credentials_configured": bool(
                self.credentials_configured()
            ),
            "last_error": self.last_error,
            "last_error_type": self.last_error_type,
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
        return self.exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit,
        )

    def fetch_ticker(self, symbol: str):
        return self.exchange.fetch_ticker(symbol)

    # ============================================================
    # BALANCE
    # ============================================================

    def fetch_balance(self):
        self._require_credentials()

        balance = self.exchange.fetch_balance()

        self.last_balance = balance
        self.authenticated = True
        self.connected = True

        return balance

    def free_quote(self, currency="USD"):
        balance = self.fetch_balance()

        free = balance.get("free", {}) or {}

        return float(
            free.get(currency)
            or 0.0
        )

    def total_quote(self, currency="USD"):
        balance = self.fetch_balance()

        total = balance.get("total", {}) or {}

        return float(
            total.get(currency)
            or 0.0
        )

    # ============================================================
    # PORTFOLIO VALUATION
    # ============================================================

    def _asset_usd_price(self, asset: str) -> float:
        """
        Estimate an asset's USD price.

        Direct USD pair is preferred.

        USDT pair is used as a fallback and treated approximately
        1:1 with USD for portfolio display.
        """

        asset = str(asset or "").upper().strip()

        if asset in {"USD", "USDG"}:
            return 1.0

        now = time.time()

        # Cached price for 10 seconds.
        with self._price_cache_lock:
            cached = self._price_cache.get(asset)

            if cached:
                cached_time, cached_price = cached

                if now - cached_time < 10:
                    return float(cached_price)

        candidates = [
            f"{asset}/USD",
            f"{asset}/USDT",
        ]

        for symbol in candidates:
            try:
                if not self.exchange.markets:
                    self.exchange.load_markets()

                if symbol not in self.exchange.markets:
                    continue

                ticker = self.exchange.fetch_ticker(symbol)

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
                    self._price_cache[asset] = (
                        time.time(),
                        price,
                    )

                return price

            except Exception:
                continue

        return 0.0

    def _calculate_portfolio_value(self, balance):
        """
        Calculate the approximate USD value of the entire Kraken
        account.

        USD:
            1:1

        USDG:
            1:1

        Other assets:
            Uses Kraken USD pair when available.
            Falls back to USDT pair.
        """

        total_balances = balance.get("total", {}) or {}

        portfolio_value = 0.0
        assets = []

        for asset, raw_amount in total_balances.items():
            try:
                amount = float(raw_amount or 0.0)
            except Exception:
                continue

            if amount <= 0:
                continue

            asset = str(asset).upper().strip()

            price = self._asset_usd_price(asset)

            if price <= 0:
                continue

            value = amount * price

            portfolio_value += value

            assets.append({
                "asset": asset,
                "amount": amount,
                "price_usd": price,
                "value_usd": value,
            })

        assets.sort(
            key=lambda item: item["value_usd"],
            reverse=True,
        )

        return float(portfolio_value), assets

    def account_summary(self, currency="USD"):
        """
        Return both:

        free:
            actual spendable USD

        total:
            USD wallet total

        portfolio_value_usd:
            total account value including USDG and crypto holdings
        """

        try:
            balance = self.fetch_balance()

            free = balance.get("free", {}) or {}
            used = balance.get("used", {}) or {}
            total = balance.get("total", {}) or {}

            # ----------------------------------------------------
            # USD
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # USDG
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # FULL PORTFOLIO
            # ----------------------------------------------------

            portfolio_value_usd, assets = (
                self._calculate_portfolio_value(balance)
            )

            return {
                "authenticated": True,

                # Actual spendable USD.
                "free": usd_free,
                "usd_free": usd_free,

                # USD wallet.
                "used": usd_used,
                "total": usd_total,
                "usd_total": usd_total,

                # USDG.
                "usdg_free": usdg_free,
                "usdg_used": usdg_used,
                "usdg_total": usdg_total,

                # Full account value.
                "portfolio_value_usd": portfolio_value_usd,

                # Individual assets used to calculate portfolio.
                "portfolio_assets": assets,

                "currency": currency,
                "error": None,
            }

        except Exception as exc:
            self.last_error = str(exc)
            self.last_error_type = type(exc).__name__

            return {
                "authenticated": bool(self.authenticated),
                "free": 0.0,
                "usd_free": 0.0,
                "used": 0.0,
                "total": 0.0,
                "usd_total": 0.0,
                "usdg_free": 0.0,
                "usdg_used": 0.0,
                "usdg_total": 0.0,
                "portfolio_value_usd": 0.0,
                "portfolio_assets": [],
                "currency": currency,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

    # ============================================================
    # ORDER HELPERS
    # ============================================================

    def _normalize_amount(self, symbol, amount):
        return float(
            self.exchange.amount_to_precision(
                symbol,
                amount,
            )
        )

    def _market_limits(self, symbol):
        market = self.exchange.market(symbol)

        limits = market.get("limits", {}) or {}

        amount_limits = limits.get("amount", {}) or {}
        cost_limits = limits.get("cost", {}) or {}

        return {
            "min_amount": amount_limits.get("min"),
            "max_amount": amount_limits.get("max"),
            "min_cost": cost_limits.get("min"),
            "max_cost": cost_limits.get("max"),
        }

    # ============================================================
    # MARKET BUY
    # ============================================================

    def market_buy(self, symbol, quote_amount):
        quote_amount = float(quote_amount)

        if quote_amount <= 0:
            raise ValueError(
                "Quote amount must be greater than zero."
            )

        ticker = self.fetch_ticker(symbol)

        ask = (
            ticker.get("ask")
            or ticker.get("last")
            or ticker.get("close")
        )

        if not ask:
            raise RuntimeError(
                f"Unable to determine ask price for {symbol}."
            )

        ask = float(ask)

        amount = quote_amount / ask
        amount = self._normalize_amount(
            symbol,
            amount,
        )

        if amount <= 0:
            raise RuntimeError(
                f"Calculated order amount is too small for {symbol}."
            )

        limits = self._market_limits(symbol)

        min_amount = limits.get("min_amount")
        min_cost = limits.get("min_cost")

        if min_amount is not None:
            if amount < float(min_amount):
                raise RuntimeError(
                    f"Order amount {amount} is below "
                    f"Kraken minimum {min_amount} for {symbol}."
                )

        if min_cost is not None:
            if quote_amount < float(min_cost):
                raise RuntimeError(
                    f"Order value ${quote_amount:.2f} is below "
                    f"Kraken minimum ${float(min_cost):.2f} "
                    f"for {symbol}."
                )

        # ========================================================
        # PAPER MODE
        # ========================================================

        if not self.live_orders_enabled:
            result = {
                "ok": True,
                "paper": True,
                "live": False,
                "symbol": symbol,
                "side": "buy",
                "amount": amount,
                "quote_amount": quote_amount,
                "price": ask,
                "status": "simulated",
            }

            self.last_order = result

            return result

        # ========================================================
        # LIVE MODE
        # ========================================================

        if not self.authenticated:
            raise RuntimeError(
                "Cannot place live order: Kraken is not authenticated."
            )

        available_usd = self.free_quote("USD")

        if available_usd < quote_amount:
            raise RuntimeError(
                f"Insufficient USD balance. "
                f"Available: ${available_usd:.2f}; "
                f"Required: ${quote_amount:.2f}."
            )

        order = self.exchange.create_market_buy_order(
            symbol,
            amount,
        )

        result = {
            "ok": True,
            "paper": False,
            "live": True,
            "symbol": symbol,
            "side": "buy",
            "amount": amount,
            "quote_amount": quote_amount,
            "price": ask,
            "status": order.get("status"),
            "order": order,
        }

        self.last_order = result

        return result

    # ============================================================
    # MARKET SELL
    # ============================================================

    def market_sell(self, symbol, amount):
        amount = float(amount)

        if amount <= 0:
            raise ValueError(
                "Sell amount must be greater than zero."
            )

        ticker = self.fetch_ticker(symbol)

        bid = (
            ticker.get("bid")
            or ticker.get("last")
            or ticker.get("close")
        )

        if not bid:
            raise RuntimeError(
                f"Unable to determine bid price for {symbol}."
            )

        bid = float(bid)

        amount = self._normalize_amount(
            symbol,
            amount,
        )

        if amount <= 0:
            raise RuntimeError(
                f"Normalized sell amount is too small for {symbol}."
            )

        limits = self._market_limits(symbol)

        min_amount = limits.get("min_amount")
        min_cost = limits.get("min_cost")

        estimated_value = amount * bid

        if min_amount is not None:
            if amount < float(min_amount):
                raise RuntimeError(
                    f"Sell amount {amount} is below "
                    f"Kraken minimum {min_amount} for {symbol}."
                )

        if min_cost is not None:
            if estimated_value < float(min_cost):
                raise RuntimeError(
                    f"Estimated order value ${estimated_value:.2f} "
                    f"is below Kraken minimum "
                    f"${float(min_cost):.2f} for {symbol}."
                )

        # ========================================================
        # PAPER MODE
        # ========================================================

        if not self.live_orders_enabled:
            result = {
                "ok": True,
                "paper": True,
                "live": False,
                "symbol": symbol,
                "side": "sell",
                "amount": amount,
                "price": bid,
                "quote_amount": estimated_value,
                "status": "simulated",
            }

            self.last_order = result

            return result

        # ========================================================
        # LIVE MODE
        # ========================================================

        if not self.authenticated:
            raise RuntimeError(
                "Cannot place live order: Kraken is not authenticated."
            )

        order = self.exchange.create_market_sell_order(
            symbol,
            amount,
        )

        result = {
            "ok": True,
            "paper": False,
            "live": True,
            "symbol": symbol,
            "side": "sell",
            "amount": amount,
            "price": bid,
            "quote_amount": estimated_value,
            "status": order.get("status"),
            "order": order,
        }

        self.last_order = result

        return result
