from __future__ import annotations

import asyncio
import time
import threading

import pandas as pd

from config import settings

from db import (
    add_trade,
    delete_position,
    get_positions,
    get_risk,
    set_position,
    set_risk,
    stats,
    record_equity_snapshot,
)

from kraken_client import KrakenTrader

from market_scanner import KrakenMarketScanner

from ml_model import (
    predict,
    train_model,
)


class KrakenBot:

    def __init__(self):

        # =====================================================
        # KRAKEN
        # =====================================================

        self.kraken = KrakenTrader(settings)

        # =====================================================
        # MARKET SCANNER
        # =====================================================

        self.scanner = KrakenMarketScanner(
            self.kraken,
            settings,
        )

        # =====================================================
        # AI / ML STATE
        # =====================================================

        self.models = {}
        self.last_train = {}

        # =====================================================
        # BOT STATE
        # =====================================================

        self.running = False

        self.last_scan = None
        self.last_scan_error = None

        self.error = None

        self.signals = []

        # =====================================================
        # POSITION STATE
        # =====================================================

        self.position_signal = None
        self.last_position_check = None
        self.last_position_reason = None

        # =====================================================
        # RISK / COOLDOWN
        # =====================================================

        self.cooldown_until = 0

        # =====================================================
        # SCAN CONTROL
        # =====================================================

        self.scan_requested = False
        self.scan_in_progress = False

        self.scan_lock = threading.Lock()

        # =====================================================
        # PAPER ACCOUNT
        # =====================================================

        try:

            if get_risk("paper_balance") is None:

                set_risk(
                    "paper_balance",
                    settings.paper_start_balance,
                )

        except Exception as exc:

            self.error = (
                f"PAPER ACCOUNT ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            print(self.error)

    # =========================================================
    # MANUAL SCAN REQUEST
    # =========================================================

    def request_scan(self):

        with self.scan_lock:

            if self.scan_in_progress:

                return {
                    "accepted": False,
                    "already_scanning": True,
                }

            self.scan_requested = True

            return {
                "accepted": True,
                "already_scanning": False,
            }

    # =========================================================
    # STOP
    # =========================================================

    def stop(self):

        print("=" * 60)
        print("KRAKEN BOT STOP REQUESTED")
        print("=" * 60)

        self.running = False

        with self.scan_lock:

            self.scan_requested = False

        return {
            "stopped": True,
            "message": "Bot stop requested.",
        }

    # =========================================================
    # REQUEST STOP
    # =========================================================

    def request_stop(self):

        return self.stop()

    # =========================================================
    # DATAFRAME
    # =========================================================

    @staticmethod
    def _df(rows):

        return pd.DataFrame(
            rows,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )

    # =========================================================
    # POSITION
    # =========================================================

    @staticmethod
    def _position(symbol):

        return next(
            (
                position
                for position in get_positions()
                if position["symbol"] == symbol
            ),
            None,
        )

    # =========================================================
    # CURRENT POSITION
    # =========================================================

    @staticmethod
    def _current_position():

        positions = get_positions()

        if not positions:

            return None

        # The strategy intentionally permits only ONE position.
        return positions[0]

    # =========================================================
    # RISK
    # =========================================================

    def _can_trade(self):

        try:

            s = stats()

            if (
                s["last_24h_pnl"]
                <= -settings.daily_loss_limit_usd
            ):

                return (
                    False,
                    "24h loss limit reached",
                )

            if (
                s["trades_24h"]
                >= settings.max_trades_per_day
            ):

                return (
                    False,
                    "24h trade limit reached",
                )

            if (
                s["consecutive_losses"]
                >= settings.max_consecutive_losses
            ):

                return (
                    False,
                    "loss circuit breaker active",
                )

            if time.time() < self.cooldown_until:

                remaining = (
                    self.cooldown_until
                    - time.time()
                )

                return (
                    False,
                    f"risk cooldown active "
                    f"({remaining:.0f}s remaining)",
                )

            return True, ""

        except Exception as exc:

            print(
                "RISK CHECK ERROR:",
                type(exc).__name__,
                exc,
            )

            return (
                False,
                f"risk check error: {exc}",
            )

    # =========================================================
    # AI SYMBOL ANALYSIS
    # =========================================================

    async def scan_symbol(
        self,
        candidate,
    ):

        symbol = (
            candidate.symbol
            if hasattr(candidate, "symbol")
            else str(candidate)
        )

        # -----------------------------------------------------
        # OHLCV
        # -----------------------------------------------------

        try:

            rows = await asyncio.to_thread(
                self.kraken.fetch_ohlcv,
                symbol,
                settings.timeframe,
                settings.candles,
            )

        except Exception as exc:

            print(
                f"OHLCV ERROR {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )

            return None

        if not rows:

            return None

        if len(rows) < 150:

            return None

        try:

            df = self._df(rows)

        except Exception as exc:

            print(
                f"DATAFRAME ERROR {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )

            return None

        now = time.time()

        state = self.models.get(symbol)

        # -----------------------------------------------------
        # TRAIN MODEL
        # -----------------------------------------------------

        if (
            state is None
            or
            now
            -
            self.last_train.get(symbol, 0)
            >= settings.train_every_seconds
        ):

            cost = (
                settings.round_trip_cost_pct / 100
                +
                settings.slippage_buffer_pct / 100
            )

            try:

                state = await asyncio.to_thread(
                    train_model,
                    df,
                    settings.forecast_bars,
                    cost,
                )

            except Exception as exc:

                print(
                    f"MODEL ERROR {symbol}: "
                    f"{type(exc).__name__}: {exc}"
                )

                return None

            self.models[symbol] = state

            self.last_train[symbol] = now

            print(
                f"MODEL TRAINED "
                f"{symbol} "
                f"accuracy="
                f"{state.accuracy:.3f}"
            )

        # -----------------------------------------------------
        # PREDICTION
        # -----------------------------------------------------

        try:

            prediction = predict(
                state,
                df,
                settings.forecast_bars,
            )

        except Exception as exc:

            print(
                f"PREDICTION ERROR {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )

            return None

        if not prediction:

            return None

        # -----------------------------------------------------
        # TICKER
        # -----------------------------------------------------

        try:

            ticker = await asyncio.to_thread(
                self.kraken.fetch_ticker,
                symbol,
            )

        except Exception as exc:

            print(
                f"TICKER ERROR {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )

            return None

        try:

            bid = float(
                ticker.get("bid")
                or ticker.get("last")
                or 0
            )

            ask = float(
                ticker.get("ask")
                or ticker.get("last")
                or 0
            )

            last = float(
                ticker.get("last")
                or 0
            )

        except Exception:

            return None

        if min(
            bid,
            ask,
            last,
        ) <= 0:

            return None

        # -----------------------------------------------------
        # SPREAD
        # -----------------------------------------------------

        spread = max(
            0,
            ask / bid - 1,
        )

        if (
            spread * 100
            > settings.max_spread_pct
        ):

            return None

        # -----------------------------------------------------
        # COST
        # -----------------------------------------------------

        base_cost = (
            settings.round_trip_cost_pct / 100
            +
            settings.slippage_buffer_pct / 100
        )

        total_cost = (
            base_cost
            + spread
        )

        # -----------------------------------------------------
        # MODEL VALUES
        # -----------------------------------------------------

        probability = float(
            prediction.get(
                "probability_up",
                0,
            )
        )

        expected_move = float(
            prediction.get(
                "expected_move",
                0,
            )
        )

        strategy_score = float(
            prediction.get(
                "strategy_score",
                0,
            )
        )

        agreement = float(
            prediction.get(
                "strategy_agreement",
                0,
            )
        )

        confidence = float(
            prediction.get(
                "confidence",
                0,
            )
        )

        direction = str(
            prediction.get(
                "direction",
                "NEUTRAL",
            )
        ).upper()

        # -----------------------------------------------------
        # EDGE
        # -----------------------------------------------------

        ml_edge = (
            probability - 0.5
        ) * 2

        combined_edge = (
            ml_edge * 0.70
            +
            strategy_score * 0.30
        )

        estimated_profit = (
            expected_move
            - total_cost
        )

        score = (
            combined_edge
            *
            max(
                agreement,
                0.25,
            )
            *
            max(
                expected_move,
                0,
            )
            -
            total_cost
        )

        # -----------------------------------------------------
        # REWARD / RISK
        # -----------------------------------------------------

        if settings.stop_loss_pct > 0:

            reward_risk = (
                settings.take_profit_pct
                /
                settings.stop_loss_pct
            )

        else:

            reward_risk = 0

        # -----------------------------------------------------
        # FILTERS
        # -----------------------------------------------------

        reasons = []

        if direction != "LONG":

            reasons.append(
                f"direction={direction}"
            )

        if (
            probability
            < settings.min_probability
        ):

            reasons.append(
                f"probability="
                f"{probability:.3f}"
            )

        if confidence < 0.10:

            reasons.append(
                f"confidence="
                f"{confidence:.3f}"
            )

        if expected_move <= 0:

            reasons.append(
                "no expected movement"
            )

        elif (
            expected_move
            < settings.min_expected_move
        ):

            reasons.append(
                f"expected_move="
                f"{expected_move:.4f} "
                f"< minimum="
                f"{settings.min_expected_move:.4f}"
            )

        if (
            expected_move
            <= total_cost
        ):

            reasons.append(
                "expected move below "
                "trading cost"
            )

        if strategy_score < -0.10:

            reasons.append(
                f"strategy bearish="
                f"{strategy_score:.3f}"
            )

        if agreement < 0.35:

            reasons.append(
                f"low strategy agreement="
                f"{agreement:.3f}"
            )

        if reward_risk < 1.25:

            reasons.append(
                f"reward/risk="
                f"{reward_risk:.2f}"
            )

        # -----------------------------------------------------
        # FINAL DECISION
        # -----------------------------------------------------

        tradeable = (
            not reasons
            and score > 0
            and estimated_profit > 0
            and reward_risk >= 1.25
        )

        return {

            "symbol":
                symbol,

            "price":
                last,

            "last":
                last,

            "bid":
                bid,

            "ask":
                ask,

            "spread":
                spread,

            "cost_estimate":
                total_cost,

            "probability_up":
                probability,

            "probability":
                probability,

            "expected_move":
                expected_move,

            "direction":
                direction,

            "confidence":
                confidence,

            "strategy_score":
                strategy_score,

            "strategy_agreement":
                agreement,

            "combined_edge":
                combined_edge,

            "estimated_profit":
                estimated_profit,

            "reward_risk":
                reward_risk,

            "score":
                score,

            "tradeable":
                tradeable,

            "reasons":
                reasons,

            "accuracy":
                float(
                    state.accuracy
                ),

            "samples":
                int(
                    state.samples
                ),

            "regime":
                prediction.get(
                    "regime",
                    "UNKNOWN",
                ),

            "strategies":
                prediction.get(
                    "strategies",
                    {},
                ),

            "trained_at":
                state.trained_at,
        }

    # =========================================================
    # FULL MARKET SCAN
    # =========================================================

    async def scan(self):

        with self.scan_lock:

            if self.scan_in_progress:

                print(
                    "ML SCAN: already running"
                )

                return self.signals

            self.scan_in_progress = True

            self.scan_requested = False

            self.last_scan_error = None

        try:

            candidates = await asyncio.to_thread(
                self.scanner.top_symbols
            )

            if candidates is None:

                candidates = []

            print(
                f"ML SCAN: "
                f"{len(candidates)} markets"
            )

            results = []

            for candidate in candidates:

                if not self.running:

                    print(
                        "ML SCAN: stop requested"
                    )

                    break

                try:

                    result = await self.scan_symbol(
                        candidate
                    )

                    if result:

                        results.append(
                            result
                        )

                except Exception as exc:

                    print(
                        "SCAN ERROR",
                        getattr(
                            candidate,
                            "symbol",
                            candidate,
                        ),
                        type(exc).__name__,
                        exc,
                    )

            results.sort(
                key=lambda x: x["score"],
                reverse=True,
            )

            for index, signal in enumerate(
                results,
                start=1,
            ):

                signal["rank"] = index

            self.signals = results

            self.last_scan = time.time()

            print(
                f"AI RESULTS: "
                f"{len(results)}"
            )

            for signal in results[:10]:

                print(
                    f"AI "
                    f"{signal['symbol']} "
                    f"{signal['direction']} "
                    f"prob="
                    f"{signal['probability_up']:.3f} "
                    f"move="
                    f"{signal['expected_move']:.4f} "
                    f"score="
                    f"{signal['score']:.5f} "
                    f"RR="
                    f"{signal['reward_risk']:.2f} "
                    f"tradeable="
                    f"{signal['tradeable']}"
                )

            return results

        except Exception as exc:

            self.last_scan_error = (
                f"{type(exc).__name__}: {exc}"
            )

            self.error = (
                "SCAN ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            print(
                self.error
            )

            return self.signals

        finally:

            with self.scan_lock:

                self.scan_in_progress = False

    # =========================================================
    # CURRENT POSITION EDGE ANALYSIS
    # =========================================================

    async def analyze_current_position(
        self,
        position,
    ):

        symbol = position["symbol"]

        print(
            f"POSITION CHECK: {symbol}"
        )

        signal = await self.scan_symbol(
            symbol
        )

        self.last_position_check = time.time()

        if signal is None:

            self.last_position_reason = (
                "Unable to obtain a fresh AI signal."
            )

            print(
                f"POSITION HOLD: {symbol} "
                f"fresh analysis unavailable"
            )

            # We do NOT sell simply because one API/model
            # check failed. The hard stop remains active.
            return True

        self.position_signal = signal

        # -----------------------------------------------------
        # Current price
        # -----------------------------------------------------

        current_price = float(
            signal.get(
                "bid",
                signal.get(
                    "last",
                    0,
                ),
            )
        )

        entry_price = float(
            position.get(
                "entry_price",
                0,
            )
        )

        amount = float(
            position.get(
                "amount",
                0,
            )
        )

        unrealized_pnl = (
            current_price
            -
            entry_price
        ) * amount

        # -----------------------------------------------------
        # AI EDGE
        # -----------------------------------------------------

        probability = float(
            signal.get(
                "probability_up",
                0,
            )
        )

        expected_move = float(
            signal.get(
                "expected_move",
                0,
            )
        )

        strategy_score = float(
            signal.get(
                "strategy_score",
                0,
            )
        )

        agreement = float(
            signal.get(
                "strategy_agreement",
                0,
            )
        )

        confidence = float(
            signal.get(
                "confidence",
                0,
            )
        )

        direction = str(
            signal.get(
                "direction",
                "NEUTRAL",
            )
        ).upper()

        estimated_cost = float(
            signal.get(
                "cost_estimate",
                0,
            )
        )

        score = float(
            signal.get(
                "score",
                0,
            )
        )

        # -----------------------------------------------------
        # EDGE CONDITIONS
        # -----------------------------------------------------
        #
        # We intentionally use a little hysteresis here.
        #
        # Entry requires the stronger original filters.
        # A position is not closed because probability simply
        # moved from 0.70 to 0.59 for one noisy candle.
        #
        # We exit when the position's edge has materially
        # deteriorated.
        # -----------------------------------------------------

        exit_reasons = []

        # Direction reversal is a strong exit signal.

        if direction != "LONG":

            exit_reasons.append(
                f"AI direction changed to {direction}"
            )

        # Expected move must at least cover estimated costs.

        if expected_move <= estimated_cost:

            exit_reasons.append(
                "expected move no longer covers "
                "estimated trading cost"
            )

        # Probability has fallen materially.

        position_exit_probability = max(
            0.50,
            settings.min_probability - 0.05,
        )

        if (
            probability
            < position_exit_probability
        ):

            exit_reasons.append(
                f"probability deteriorated to "
                f"{probability:.3f}"
            )

        # Strategy has become meaningfully bearish.

        if strategy_score < -0.10:

            exit_reasons.append(
                f"strategy turned bearish "
                f"({strategy_score:.3f})"
            )

        # Agreement deterioration.

        if agreement < 0.30:

            exit_reasons.append(
                f"strategy agreement deteriorated "
                f"({agreement:.3f})"
            )

        # Confidence collapse.

        if confidence < 0.08:

            exit_reasons.append(
                f"AI confidence deteriorated "
                f"({confidence:.3f})"
            )

        # Score is no longer positive.

        if score <= 0:

            exit_reasons.append(
                f"AI score became non-positive "
                f"({score:.5f})"
            )

        # -----------------------------------------------------
        # DECISION
        # -----------------------------------------------------

        if not exit_reasons:

            self.last_position_reason = (
                "Current trade still has positive AI edge."
            )

            print(
                f"POSITION HOLD "
                f"{symbol} "
                f"PnL=${unrealized_pnl:.2f} "
                f"prob={probability:.3f} "
                f"move={expected_move:.4f} "
                f"score={score:.5f}"
            )

            return True

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # Require multiple independent deterioration signals
        # before voluntarily exiting.
        #
        # A single weak metric should not cause unnecessary
        # churn.
        # -----------------------------------------------------

        strong_exit = False

        if direction != "LONG":

            strong_exit = True

        elif (
            expected_move <= estimated_cost
            and probability < position_exit_probability
        ):

            strong_exit = True

        elif (
            probability < position_exit_probability
            and strategy_score < -0.10
        ):

            strong_exit = True

        elif (
            score <= 0
            and expected_move <= estimated_cost
        ):

            strong_exit = True

        elif len(exit_reasons) >= 3:

            strong_exit = True

        if not strong_exit:

            self.last_position_reason = (
                "Some AI metrics weakened, but the trade "
                "has not lost enough edge to justify an exit."
            )

            print(
                f"POSITION HOLD "
                f"{symbol} "
                f"edge weakening but not invalidated"
            )

            return True

        self.last_position_reason = (
            "; ".join(exit_reasons)
        )

        print(
            f"POSITION EDGE LOST "
            f"{symbol}: "
            f"{self.last_position_reason}"
        )

        return False

    # =========================================================
    # EXIT CURRENT POSITION
    # =========================================================

    async def exit_position(
        self,
        position,
        reason,
        apply_cooldown=False,
    ):

        symbol = position["symbol"]

        try:

            ticker = await asyncio.to_thread(
                self.kraken.fetch_ticker,
                symbol,
            )

            market_price = float(
                ticker.get("bid")
                or ticker.get("last")
                or 0
            )

            if market_price <= 0:

                raise RuntimeError(
                    f"Invalid market price for {symbol}."
                )

            result = await asyncio.to_thread(
                self.kraken.market_sell,
                symbol,
                position["amount"],
            )

            exit_price = float(
                result.get("price")
                or market_price
            )

            amount = float(
                position["amount"]
            )

            entry_price = float(
                position["entry_price"]
            )

            exit_proceeds = (
                exit_price * amount
            )

            entry_cost = (
                entry_price * amount
            )

            pnl = (
                exit_proceeds
                -
                entry_cost
            )

            # -------------------------------------------------
            # RECORD TRADE
            # -------------------------------------------------

            add_trade({

                "symbol":
                    symbol,

                "side":
                    "SELL",

                "price":
                    exit_price,

                "amount":
                    amount,

                "notional":
                    exit_proceeds,

                "pnl":
                    pnl,

                "status":
                    "CLOSED",

                "mode":
                    (
                        "DRY_RUN"
                        if settings.dry_run
                        else "LIVE"
                    ),

                "reason":
                    reason,
            })

            # -------------------------------------------------
            # PAPER ACCOUNT
            # -------------------------------------------------

            if settings.dry_run:

                balance = float(
                    get_risk(
                        "paper_balance",
                        settings.paper_start_balance,
                    )
                )

                invested = float(
                    get_risk(
                        "paper_invested",
                        0,
                    )
                )

                realized = float(
                    get_risk(
                        "paper_realized_pnl",
                        0,
                    )
                )

                # IMPORTANT:
                #
                # The paper account receives the ACTUAL
                # simulated sale proceeds.
                #
                # Do not add notional + pnl because that
                # double-counts the original capital.

                set_risk(
                    "paper_balance",
                    balance
                    +
                    exit_proceeds,
                )

                set_risk(
                    "paper_invested",
                    max(
                        0,
                        invested
                        -
                        entry_cost,
                    ),
                )

                set_risk(
                    "paper_realized_pnl",
                    realized + pnl,
                )

            delete_position(
                symbol
            )

            self.position_signal = None

            self.last_position_reason = reason

            print(
                f"AI EXIT "
                f"{symbol} "
                f"reason={reason} "
                f"entry=${entry_price:.8f} "
                f"exit=${exit_price:.8f} "
                f"PnL=${pnl:.2f}"
            )

            # -------------------------------------------------
            # COOLDOWN
            # -------------------------------------------------
            #
            # Normal AI edge exits do NOT use the cooldown.
            #
            # Emergency stop-loss exits can use it to prevent
            # immediately jumping back into the same market.
            # -------------------------------------------------

            if apply_cooldown:

                self.cooldown_until = (
                    time.time()
                    +
                    settings.cooldown_minutes
                    * 60
                )

            return {
                "ok": True,
                "symbol": symbol,
                "exit_price": exit_price,
                "amount": amount,
                "pnl": pnl,
                "reason": reason,
            }

        except Exception as exc:

            self.error = (
                "POSITION EXIT ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                self.error
            )

            return {
                "ok": False,
                "error": self.error,
            }

    # =========================================================
    # POSITION MANAGEMENT
    # =========================================================

    async def manage_positions(self):

        positions = get_positions()

        if not positions:

            self.position_signal = None

            return

        # =====================================================
        # SINGLE POSITION RULE
        # =====================================================

        position = positions[0]

        symbol = position["symbol"]

        # =====================================================
        # CURRENT MARKET PRICE
        # =====================================================

        try:

            ticker = await asyncio.to_thread(
                self.kraken.fetch_ticker,
                symbol,
            )

            price = float(
                ticker.get("bid")
                or ticker.get("last")
                or 0
            )

        except Exception as exc:

            print(
                f"POSITION PRICE ERROR {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )

            return

        if price <= 0:

            return

        # =====================================================
        # EMERGENCY STOP LOSS
        # =====================================================
        #
        # Stop loss remains an emergency protection mechanism.
        #
        # It is NOT a normal profit-taking mechanism.
        # =====================================================

        if (
            price
            <= position["stop_price"]
        ):

            reason = (
                "emergency stop loss"
            )

            await self.exit_position(
                position,
                reason,
                apply_cooldown=True,
            )

            return

        # =====================================================
        # AI EDGE RE-EVALUATION
        # =====================================================

        still_has_edge = (
            await self.analyze_current_position(
                position
            )
        )

        if still_has_edge:

            return

        # =====================================================
        # EDGE LOST → EXIT
        # =====================================================

        reason = (
            self.last_position_reason
            or
            "AI edge lost"
        )

        await self.exit_position(
            position,
            reason,
            apply_cooldown=False,
        )

    # =========================================================
    # ENTER BEST TRADE
    # =========================================================

    async def maybe_enter_best(self):

        # =====================================================
        # NEVER ENTER WHILE A POSITION EXISTS
        # =====================================================

        existing_position = (
            self._current_position()
        )

        if existing_position is not None:

            print(
                "ENTRY BLOCKED: "
                "existing position is still open."
            )

            return

        if not self.running:

            return

        if not self.signals:

            print(
                "AI: NO SIGNALS"
            )

            return

        allowed, reason = self._can_trade()

        if not allowed:

            print(
                "TRADE BLOCKED:",
                reason,
            )

            return

        # =====================================================
        # QUALIFIED SIGNALS
        # =====================================================

        candidates = [

            signal

            for signal
            in self.signals

            if signal.get("tradeable")

            and signal.get(
                "accuracy",
                0,
            )
            >= settings.min_training_accuracy
        ]

        if not candidates:

            print(
                "AI: NO QUALIFIED TRADE"
            )

            return

        # =====================================================
        # BEST SIGNAL
        # =====================================================

        candidates.sort(
            key=lambda x: x.get(
                "score",
                0,
            ),
            reverse=True,
        )

        best = candidates[0]

        symbol = best["symbol"]

        if self._position(symbol):

            return

        # =====================================================
        # CAPITAL
        # =====================================================

        if settings.dry_run:

            balance = float(
                get_risk(
                    "paper_balance",
                    settings.paper_start_balance,
                )
            )

            quote = min(
                settings.max_trade_usd,
                balance
                *
                settings.max_position_pct,
            )

        else:

            try:

                free = await asyncio.to_thread(
                    self.kraken.free_quote,
                    "USD",
                )

                quote = min(
                    settings.max_trade_usd,
                    free
                    *
                    settings.max_position_pct,
                )

            except Exception as exc:

                print(
                    "BALANCE ERROR:",
                    type(exc).__name__,
                    exc,
                )

                return

        if quote < 5:

            print(
                "TRADE BLOCKED: "
                "insufficient capital"
            )

            return

        # =====================================================
        # BUY
        # =====================================================

        try:

            result = await asyncio.to_thread(
                self.kraken.market_buy,
                symbol,
                quote,
            )

        except Exception as exc:

            self.error = (
                "ENTRY ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            print(
                self.error
            )

            return

        if not result:

            print(
                "ENTRY BLOCKED: empty order result"
            )

            return

        entry = float(
            result.get("price")
            or best["ask"]
        )

        amount = float(
            result.get("amount")
            or quote / entry
        )

        if entry <= 0 or amount <= 0:

            print(
                "ENTRY BLOCKED: invalid execution values"
            )

            return

        notional = (
            entry * amount
        )

        # =====================================================
        # EMERGENCY STOP ONLY
        # =====================================================

        stop_price = (
            entry
            *
            (
                1
                -
                settings.stop_loss_pct
            )
        )

        # =====================================================
        # POSITION
        # =====================================================
        #
        # We intentionally DO NOT create a fixed take-profit
        # exit. The AI decides when the position has lost edge.
        # =====================================================

        set_position({

            "symbol":
                symbol,

            "entry_price":
                entry,

            "amount":
                amount,

            "notional":
                notional,

            "opened_ts":
                time.time(),

            "stop_price":
                stop_price,

            # Kept for compatibility with existing DB/schema.
            # It is NOT used as an automatic exit trigger.
            "target_price":
                0,
        })

        # =====================================================
        # PAPER BALANCE
        # =====================================================

        if settings.dry_run:

            balance = float(
                get_risk(
                    "paper_balance",
                    settings.paper_start_balance,
                )
            )

            invested = float(
                get_risk(
                    "paper_invested",
                    0,
                )
            )

            set_risk(
                "paper_balance",
                max(
                    0,
                    balance - notional,
                ),
            )

            set_risk(
                "paper_invested",
                invested + notional,
            )

        # =====================================================
        # RECORD ENTRY
        # =====================================================

        add_trade({

            "symbol":
                symbol,

            "side":
                "BUY",

            "price":
                entry,

            "amount":
                amount,

            "notional":
                notional,

            "status":
                "OPEN",

            "mode":
                (
                    "DRY_RUN"
                    if settings.dry_run
                    else "LIVE"
                ),

            "reason":
                (
                    f"AI ENTRY "
                    f"prob="
                    f"{best['probability_up']:.3f} "
                    f"confidence="
                    f"{best['confidence']:.3f} "
                    f"score="
                    f"{best['score']:.5f}"
                ),
        })

        self.position_signal = best

        print("=" * 60)
        print("AI ENTRY")
        print(
            f"SYMBOL: {symbol}"
        )
        print(
            f"PRICE: {entry:.8f}"
        )
        print(
            f"NOTIONAL: ${notional:.2f}"
        )
        print(
            f"STOP: {stop_price:.8f}"
        )
        print(
            "TAKE PROFIT: AI CONTROLLED"
        )
        print(
            "EXIT: WHEN CURRENT TRADE LOSES EDGE"
        )
        print("=" * 60)

    # =========================================================
    # EQUITY
    # =========================================================

    async def mark_equity(self):

        prices = {}

        for position in get_positions():

            if not self.running:

                return

            try:

                ticker = await asyncio.to_thread(
                    self.kraken.fetch_ticker,
                    position["symbol"],
                )

                prices[
                    position["symbol"]
                ] = float(
                    ticker.get("bid")
                    or ticker.get("last")
                    or position["entry_price"]
                )

            except Exception:

                prices[
                    position["symbol"]
                ] = position["entry_price"]

        record_equity_snapshot(
            prices
        )

    # =========================================================
    # AUTONOMOUS LOOP
    # =========================================================

    async def run(self):

        # =====================================================
        # START
        # =====================================================

        self.running = True

        self.error = None

        print("=" * 60)
        print("AUTONOMOUS LOOP ONLINE")
        print("FULL MARKET SCANNER ONLINE")
        print("AI DAY TRADING ENGINE ONLINE")
        print(
            f"PAPER MODE: "
            f"{settings.dry_run}"
        )
        print(
            "POSITION MODE: "
            "ONE TRADE AT A TIME"
        )
        print(
            "EXIT MODE: "
            "AI EDGE LOSS + EMERGENCY STOP"
        )
        print("=" * 60)

        try:

            while self.running:

                try:

                    # =================================================
                    # POSITION FIRST
                    # =================================================
                    #
                    # If we have a position, ONLY manage that position.
                    #
                    # We do NOT search for another trade.
                    # =================================================

                    existing_position = (
                        self._current_position()
                    )

                    if existing_position is not None:

                        await self.manage_positions()

                    else:

                        # =================================================
                        # FLAT
                        # =================================================
                        #
                        # Only when completely flat do we search
                        # the market for the next opportunity.
                        # =================================================

                        if self.scan_requested:

                            print(
                                "MANUAL SCAN REQUEST ACCEPTED"
                            )

                        await self.scan()

                        if not self.running:

                            break

                        # -------------------------------------------------
                        # ENTER ONLY ONE TRADE
                        # -------------------------------------------------

                        await self.maybe_enter_best()

                    # =================================================
                    # EQUITY
                    # =================================================

                    if self.running:

                        await self.mark_equity()

                    # =================================================
                    # CLEAR RECOVERABLE ERROR
                    # =================================================

                    if self.running:

                        self.error = None

                except Exception as exc:

                    self.error = (
                        f"BOT CYCLE ERROR: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                    print(
                        self.error
                    )

                # =================================================
                # NEXT CYCLE
                # =================================================

                if self.running:

                    await asyncio.sleep(
                        max(
                            5,
                            settings.scan_seconds,
                        )
                    )

        except asyncio.CancelledError:

            print(
                "AUTONOMOUS LOOP CANCELLED"
            )

            self.running = False

            raise

        except Exception as exc:

            self.error = (
                f"BOT LOOP FATAL ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                self.error
            )

        finally:

            self.running = False

            print("=" * 60)
            print("AUTONOMOUS LOOP OFFLINE")
            print("=" * 60)

    # =========================================================
    # STATUS
    # =========================================================

    def status(self):

        position = self._current_position()

        return {

            "running":
                bool(self.running),

            "scan_in_progress":
                bool(self.scan_in_progress),

            "scan_requested":
                bool(self.scan_requested),

            "has_position":
                position is not None,

            "position":
                position,

            "last_scan":
                self.last_scan,

            "last_scan_error":
                self.last_scan_error,

            "last_position_check":
                self.last_position_check,

            "last_position_reason":
                self.last_position_reason,

            "error":
                self.error,

            "signals":
                len(self.signals),

            "models":
                len(self.models),

        }

    # =========================================================
    # SIGNALS
    # =========================================================

    def get_signals(self):

        return self.signals

    # =========================================================
    # POSITIONS
    # =========================================================

    def get_positions(self):

        return get_positions()

    # =========================================================
    # STATS
    # =========================================================

    def stats(self):

        return stats()

    # =========================================================
    # SCANNER STATUS
    # =========================================================

    def scanner_status(self):

        position = self._current_position()

        return {

            "running":
                bool(self.running),

            "scanning":
                bool(self.scan_in_progress),

            "scan_requested":
                bool(self.scan_requested),

            "has_position":
                position is not None,

            "position_symbol":
                (
                    position["symbol"]
                    if position
                    else None
                ),

            "last_scan":
                self.last_scan,

            "last_scan_error":
                self.last_scan_error,

            "last_position_check":
                self.last_position_check,

            "last_position_reason":
                self.last_position_reason,

            "error":
                self.error,

            "signals":
                len(self.signals),

            "models":
                len(self.models),

        }
