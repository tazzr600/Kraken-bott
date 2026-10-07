from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from config import settings
from db import (
    add_trade,
    get_positions,
    get_risk,
    register_closed_trade,
    reset_daily_state_if_needed,
    set_risk,
    stats,
    record_equity_snapshot,
)
from kraken_client import KrakenTrader
from market_scanner import KrakenMarketScanner


class KrakenBot:
    """
    Two-sided Kraken inventory engine.

    This is the Kraken adaptation of the paired-inventory idea:
    continuously discover liquid markets, quote both sides, keep inventory
    balanced, and only trade when the executable spread can cover fees and
    the configured safety buffer.

    PAPER is always the runtime default. LIVE requires the existing
    LIVE_TRADING + DRY_RUN safety gates and verified Kraken authentication.
    """

    def __init__(self):
        self.kraken = KrakenTrader(settings)
        self.scanner = KrakenMarketScanner(self.kraken, settings)

        self.running = False
        self.last_scan = None
        self.last_scan_error = None
        self.error = None
        self.signals: list[dict[str, Any]] = []

        self.scan_requested = False
        self.scan_in_progress = False
        self.scan_lock = threading.Lock()
        self.cooldown_until = 0.0
        self.last_quotes: dict[str, dict[str, Any]] = {}
        self.open_order_ids: dict[str, list[str]] = {}
        self.last_quote_refresh = 0.0
        self.filled_pairs = 0

        if get_risk("paper_balance") is None:
            set_risk("paper_balance", settings.paper_start_balance)
        if get_risk("paper_start_balance") is None:
            set_risk("paper_start_balance", settings.paper_start_balance)
        if get_risk("paper_invested") is None:
            set_risk("paper_invested", 0)
        if get_risk("paper_realized_pnl") is None:
            set_risk("paper_realized_pnl", 0)

    # ---------------------------------------------------------
    # lifecycle
    # ---------------------------------------------------------

    def start(self):
        self.running = True
        self.error = None
        return {"started": True, "running": True, "message": "Two-sided bot running."}

    def stop(self):
        self.running = False
        with self.scan_lock:
            self.scan_requested = False
        try:
            self._cancel_all_quotes()
        except Exception:
            pass
        return {"stopped": True, "running": False, "message": "Bot stopped."}

    def request_stop(self):
        return self.stop()

    def request_scan(self):
        with self.scan_lock:
            if self.scan_in_progress:
                return {"accepted": False, "already_scanning": True}
            self.scan_requested = True
            return {"accepted": True, "already_scanning": False}

    def force_paper_mode(self):
        return self.kraken.force_paper_mode()

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------

    @staticmethod
    def _f(v, default=0.0):
        try:
            x = float(v)
            return x if x == x and abs(x) != float("inf") else default
        except Exception:
            return default

    def _runtime_mode(self):
        return str(self.kraken.mode).upper()

    def _risk_ok(self):
        try:
            reset_daily_state_if_needed()
            s = stats()
            if self._f(s.get("daily_pnl")) <= -float(settings.daily_loss_limit_usd):
                return False, "daily loss limit reached"
            if int(s.get("trades_today", 0) or 0) >= int(settings.max_trades_per_day):
                return False, "daily trade limit reached"
            if int(s.get("consecutive_losses", 0) or 0) >= int(settings.max_consecutive_losses):
                return False, "loss circuit breaker active"
            if time.time() < self.cooldown_until:
                return False, "cooldown active"
            return True, ""
        except Exception as exc:
            return False, f"risk check error: {exc}"

    # ---------------------------------------------------------
    # two-sided quote math
    # ---------------------------------------------------------

    def _quote(self, candidate):
        symbol = candidate.symbol
        bid = self._f(candidate.bid)
        ask = self._f(candidate.ask)
        if bid <= 0 or ask <= 0 or ask <= bid:
            return None

        mid = (bid + ask) / 2.0
        market_spread = (ask / bid - 1.0) * 100.0

        # We only quote markets whose displayed spread is large enough
        # to leave room for fees, slippage and a maker margin.
        if market_spread < float(settings.min_market_spread_pct):
            return None
        if market_spread > float(settings.max_spread_pct):
            return None

        target_edge = (
            float(settings.round_trip_cost_pct) / 100.0
            + float(settings.market_maker_buffer_pct) / 100.0
        )
        half = max(
            (ask - bid) / 2.0,
            mid * target_edge / 2.0,
        )

        # Inventory skew: if we hold too much base, lower both quotes so
        # future fills preferentially reduce inventory.
        position = next(
            (p for p in get_positions() if p.get("symbol") == symbol),
            None,
        )
        current_amount = self._f(position.get("amount")) if position else 0.0
        max_base = max(
            0.00000001,
            float(settings.max_position_pct) * max(1.0, float(settings.paper_start_balance)),
        )
        inventory_ratio = max(-1.0, min(1.0, current_amount / max_base))

        skew = mid * float(settings.inventory_skew_pct) / 100.0 * inventory_ratio

        buy_price = mid - half - skew
        sell_price = mid + half - skew

        if buy_price <= 0 or sell_price <= buy_price:
            return None

        spread_capture_pct = (sell_price / buy_price - 1.0) * 100.0
        net_after_cost_pct = spread_capture_pct - float(settings.round_trip_cost_pct)

        if net_after_cost_pct < float(settings.min_net_edge_pct):
            return None

        return {
            "symbol": symbol,
            "base": candidate.base,
            "quote": candidate.quote,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "market_spread_pct": market_spread,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "quote_spread_pct": spread_capture_pct,
            "net_edge_pct": net_after_cost_pct,
            "liquidity_score": self._f(candidate.liquidity_score),
            "quote_volume": self._f(candidate.quote_volume),
            "rank": int(candidate.rank),
            "inventory": current_amount,
        }

    async def analyze_markets(self):
        candidates = await asyncio.to_thread(self.scanner.top_symbols)
        signals = []

        for candidate in candidates:
            q = self._quote(candidate)
            if q:
                signals.append(q)

        signals.sort(key=lambda x: (x["net_edge_pct"], x["liquidity_score"]), reverse=True)
        return signals

    # ---------------------------------------------------------
    # execution
    # ---------------------------------------------------------

    def _cancel_symbol_quotes(self, symbol):
        ids = list(self.open_order_ids.get(symbol, []))
        for order_id in ids:
            try:
                self.kraken.cancel_order(order_id, symbol)
            except Exception:
                pass
        self.open_order_ids[symbol] = []

    def _cancel_all_quotes(self):
        for symbol in list(self.open_order_ids):
            self._cancel_symbol_quotes(symbol)

    def _place_two_sided(self, q):
        ok, reason = self._risk_ok()
        if not ok:
            return False, reason

        symbol = q["symbol"]
        quote_usd = min(
            float(settings.max_trade_usd),
            float(settings.paper_start_balance) * float(settings.max_position_pct),
        )
        if quote_usd <= 0:
            return False, "quote size is zero"

        # Keep one bid and one ask per market. Existing quotes are cancelled
        # before replacement so stale prices cannot accumulate.
        self._cancel_symbol_quotes(symbol)

        amount = quote_usd / q["buy_price"]
        if amount <= 0:
            return False, "invalid amount"

        # Kraken spot cannot sell base that the account does not own.
        # In LIVE mode the ask is therefore capped by actual free base
        # inventory; this bot never assumes margin/shorting.
        if self._runtime_mode() == "LIVE":
            free_base = self.kraken.free_base(symbol)
            amount = min(amount, free_base * float(settings.max_inventory_fraction))
            if amount <= 0:
                return False, "no base inventory available for sell side"

        buy = self.kraken.limit_buy(symbol, amount, q["buy_price"])
        sell = self.kraken.limit_sell(symbol, amount, q["sell_price"])

        ids = []
        for result in (buy, sell):
            order = result.get("order") if isinstance(result, dict) else None
            order_id = ""
            if isinstance(order, dict):
                order_id = str(order.get("id") or order.get("orderId") or "")
            if not order_id and isinstance(result, dict):
                order_id = str(result.get("id") or "")
            if order_id:
                ids.append(order_id)

        self.open_order_ids[symbol] = ids
        self.last_quotes[symbol] = {
            **q,
            "amount": amount,
            "buy_result": buy,
            "sell_result": sell,
            "ts": time.time(),
        }

        return True, "two-sided quotes placed"

    # ---------------------------------------------------------
    # main cycle
    # ---------------------------------------------------------

    async def run(self):
        self.running = True

        while self.running:
            self.scan_in_progress = True
            try:
                self.last_scan = time.time()
                signals = await self.analyze_markets()
                self.signals = signals

                # Keep only the best liquid opportunities. This prevents
                # the bot from scattering inventory across the whole market.
                selected = signals[: int(settings.max_open_markets)]

                selected_symbols = {x["symbol"] for x in selected}
                for symbol in list(self.open_order_ids):
                    if symbol not in selected_symbols:
                        self._cancel_symbol_quotes(symbol)

                if selected:
                    for q in selected:
                        try:
                            self._place_two_sided(q)
                        except Exception as exc:
                            self.last_scan_error = f"{q['symbol']}: {type(exc).__name__}: {exc}"
                else:
                    self._cancel_all_quotes()

                try:
                    record_equity_snapshot()
                except Exception:
                    pass

                self.last_scan_error = None if signals or not self.scanner.last_error else self.scanner.last_error

            except Exception as exc:
                self.last_scan_error = f"{type(exc).__name__}: {exc}"
                self.error = self.last_scan_error
                print("TWO-SIDED ENGINE ERROR:", self.last_scan_error)
            finally:
                self.scan_in_progress = False
                self.scan_requested = False

            await asyncio.sleep(max(2, int(settings.scan_seconds)))

    # ---------------------------------------------------------
    # dashboard compatibility
    # ---------------------------------------------------------

    def get_signals(self):
        return self.signals

    def get_positions(self):
        return get_positions()

    def stats(self):
        s = stats()
        s.update({
            "strategy": "TWO_SIDED_INVENTORY",
            "quotes_active": sum(len(v) for v in self.open_order_ids.values()),
            "markets_quoted": len(self.last_quotes),
            "mode": self._runtime_mode(),
            "last_scan": self.last_scan,
            "last_scan_error": self.last_scan_error,
        })
        return s

    def scanner_status(self):
        result = self.scanner.status()
        result.update({
            "strategy": "TWO_SIDED_INVENTORY",
            "markets_quoted": len(self.last_quotes),
            "quotes_active": sum(len(v) for v in self.open_order_ids.values()),
            "max_scan_symbols": settings.max_scan_symbols,
            "allowed_quotes": settings.allowed_quote_list,
        })
        return result

    def status(self):
        return {
            "running": self.running,
            "mode": self._runtime_mode(),
            "strategy": "TWO_SIDED_INVENTORY",
            "last_scan": self.last_scan,
            "last_scan_error": self.last_scan_error,
            "signals": len(self.signals),
            "quotes_active": sum(len(v) for v in self.open_order_ids.values()),
            "kraken": self.kraken.status(),
            "scanner": self.scanner.status(),
        }


def main():
    bot = KrakenBot()
    bot.start()
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
