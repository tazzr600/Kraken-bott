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

        self.kraken = KrakenTrader(settings)

        self.scanner = KrakenMarketScanner(
            self.kraken,
            settings,
        )

        self.models = {}
        self.last_train = {}

        self.running = False

        self.last_scan = None
        self.last_scan_error = None

        self.error = None

        self.signals = []

        self.cooldown_until = 0

        # Manual scan control
        self.scan_requested = False
        self.scan_in_progress = False

        self.scan_lock = threading.Lock()

        # -----------------------------------------------------
        # PAPER ACCOUNT INITIALIZATION
        # -----------------------------------------------------

        if get_risk("paper_balance") is None:

            set_risk(
                "paper_balance",
                settings.paper_start_balance,
            )

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
    # RISK
    # =========================================================

    def _can_trade(self):

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

            return (
                False,
                "cooldown active",
            )

        return True, ""

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

        df = self._df(rows)

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

        # Reject excessive spread.

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

        # LONG ONLY

        if direction != "LONG":

            reasons.append(
                f"direction={direction}"
            )

        # PROBABILITY

        if (
            probability
            < settings.min_probability
        ):

            reasons.append(
                f"probability="
                f"{probability:.3f}"
            )

        # CONFIDENCE

        if confidence < 0.10:

            reasons.append(
                f"confidence="
                f"{confidence:.3f}"
            )

        # EXPECTED MOVE

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

        # COST

        if (
            expected_move
            <= total_cost
        ):

            reasons.append(
                "expected move below "
                "trading cost"
            )

        # STRATEGY

        if strategy_score < -0.10:

            reasons.append(
                f"strategy bearish="
                f"{strategy_score:.3f}"
            )

        # AGREEMENT

        if agreement < 0.35:

            reasons.append(
                f"low strategy agreement="
                f"{agreement:.3f}"
            )

        # REWARD/RISK

        if reward_risk < 1.25:

            reasons.append(
                f"reward/risk="
                f"{reward_risk:.2f}"
            )

        # FINAL DECISION

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

            print(
                f"ML SCAN: "
                f"{len(candidates)} markets"
            )

            results = []

            for candidate in candidates:

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

            # -------------------------------------------------
            # ADD RANK
            # -------------------------------------------------

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

            # -------------------------------------------------
            # PRINT TOP SIGNALS
            # -------------------------------------------------

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
    # POSITION MANAGEMENT
    # =========================================================

    async def manage_positions(self):

        positions = get_positions()

        if not positions:
            return

        for position in positions:

            try:

                ticker = await asyncio.to_thread(
                    self.kraken.fetch_ticker,
                    position["symbol"],
                )

                price = float(
                    ticker.get("bid")
                    or ticker.get("last")
                    or 0
                )

                if price <= 0:
                    continue

                age = (
                    time.time()
                    -
                    position["opened_ts"]
                ) / 60

                reason = None

                if (
                    price
                    <= position["stop_price"]
                ):

                    reason = "stop_loss"

                elif (
                    price
                    >= position["target_price"]
                ):

                    reason = "take_profit"

                elif (
                    age
                    >= settings.max_hold_minutes
                ):

                    reason = "time_exit"

                if not reason:
                    continue

                # -------------------------------------------------
                # SELL
                # -------------------------------------------------

                result = await asyncio.to_thread(
                    self.kraken.market_sell,
                    position["symbol"],
                    position["amount"],
                )

                exit_price = float(
                    result.get("price")
                    or price
                )

                pnl = (
                    exit_price
                    -
                    position["entry_price"]
                ) * position["amount"]

                # -------------------------------------------------
                # RECORD TRADE
                # -------------------------------------------------

                add_trade({

                    "symbol":
                        position["symbol"],

                    "side":
                        "SELL",

                    "price":
                        exit_price,

                    "amount":
                        position["amount"],

                    "notional":
                        exit_price
                        *
                        position["amount"],

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
                # PAPER BALANCE
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

                    set_risk(
                        "paper_balance",
                        balance
                        +
                        position["notional"]
                        +
                        pnl,
                    )

                    set_risk(
                        "paper_invested",
                        max(
                            0,
                            invested
                            -
                            position["notional"],
                        ),
                    )

                    set_risk(
                        "paper_realized_pnl",
                        realized + pnl,
                    )

                delete_position(
                    position["symbol"]
                )

                self.cooldown_until = (
                    time.time()
                    +
                    settings.cooldown_minutes
                    * 60
                )

                print(
                    f"EXIT "
                    f"{position['symbol']} "
                    f"{reason} "
                    f"PnL={pnl:.2f}"
                )

            except Exception as exc:

                self.error = (
                    "POSITION ERROR: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(
                    self.error
                )

    # =========================================================
    # ENTER BEST TRADE
    # =========================================================

    async def maybe_enter_best(self):

        if not self.signals:
            return

        allowed, reason = self._can_trade()

        if not allowed:

            print(
                "TRADE BLOCKED:",
                reason,
            )

            return

        # -----------------------------------------------------
        # QUALIFIED SIGNALS
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # BEST SIGNAL
        # -----------------------------------------------------

        best = candidates[0]

        if self._position(
            best["symbol"]
        ):

            return

        # -----------------------------------------------------
        # CAPITAL
        # -----------------------------------------------------

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
                    exc,
                )

                return

        if quote < 5:

            print(
                "TRADE BLOCKED: "
                "insufficient capital"
            )

            return

        # -----------------------------------------------------
        # BUY
        # -----------------------------------------------------

        try:

            result = await asyncio.to_thread(
                self.kraken.market_buy,
                best["symbol"],
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

        entry = float(
            result.get("price")
            or best["ask"]
        )

        amount = float(
            result.get("amount")
            or quote / entry
        )

        notional = (
            entry * amount
        )

        # -----------------------------------------------------
        # RISK LEVELS
        # -----------------------------------------------------

        stop_price = (
            entry
            *
            (
                1
                -
                settings.stop_loss_pct
            )
        )

        target_price = (
            entry
            *
            (
                1
                +
                settings.take_profit_pct
            )
        )

        # -----------------------------------------------------
        # POSITION
        # -----------------------------------------------------

        set_position({

            "symbol":
                best["symbol"],

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

            "target_price":
                target_price,
        })

        # -----------------------------------------------------
        # PAPER BALANCE
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # RECORD ENTRY
        # -----------------------------------------------------

        add_trade({

            "symbol":
                best["symbol"],

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
                    f"AI "
                    f"prob="
                    f"{best['probability_up']:.3f} "
                    f"confidence="
                    f"{best['confidence']:.3f} "
                    f"score="
                    f"{best['score']:.5f}"
                ),
        })

        print(
            f"AI ENTRY "
            f"{best['symbol']} "
            f"price={entry:.8f} "
            f"notional=${notional:.2f} "
            f"stop={stop_price:.8f} "
            f"target={target_price:.8f}"
        )

    # =========================================================
    # EQUITY
    # =========================================================

    async def mark_equity(self):

        prices = {}

        for position in get_positions():

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

        print("=" * 60)
        print("AUTONOMOUS LOOP ONLINE")
        print("FULL MARKET SCANNER ONLINE")
        print("AI DAY TRADING ENGINE ONLINE")
        print(
            f"PAPER MODE: "
            f"{settings.dry_run}"
        )
        print("=" * 60)

        while self.running:

            try:

                # -------------------------------------------------
                # MANAGE EXISTING POSITION
                # -------------------------------------------------

                await self.manage_positions()

                # -------------------------------------------------
                # SCAN
                # -------------------------------------------------

                if self.scan_requested:

                    print(
                        "MANUAL SCAN REQUEST ACCEPTED"
                    )

                await self.scan()

                # -------------------------------------------------
                # TRADE
                # -------------------------------------------------

                await self.maybe_enter_best()

                # -------------------------------------------------
                # EQUITY
                # -------------------------------------------------

                await self.mark_equity()

                self.error = None

            except Exception as exc:

                self.error = (
                    f"BOT ERROR: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(
                    self.error
                )

            # -----------------------------------------------------
            # NEXT CYCLE
            # -----------------------------------------------------

            await asyncio.sleep(
                max(
                    5,
                    settings.scan_seconds,
                )
            )

        print(
            "AUTONOMOUS LOOP OFFLINE"
        )
