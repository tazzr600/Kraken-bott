from __future__ import annotations

import asyncio
import math
import threading
import time

import pandas as pd

from config import settings

from db import (
    add_trade,
    delete_position,
    get_positions,
    get_risk,
    get_trades,
    register_closed_trade,
    set_position,
    set_risk,
    stats,
    record_equity_snapshot,
)

from kraken_client import KrakenTrader

# Keep the existing scanner.
# Do NOT create another scanner alongside this one.
from market_scanner import KrakenMarketScanner

from ml_model import (
    predict,
    train_model,
)


class KrakenBot:

    # =========================================================
    # DAY-TRADING ENGINE SETTINGS
    # =========================================================

    # A new opportunity must beat the current position by both
    # the score multiplier AND the minimum score difference.
    ROTATION_MIN_SCORE_ADVANTAGE = 0.10
    ROTATION_SCORE_MULTIPLIER = 1.20

    # Current strategy score below this is considered bearish.
    BEARISH_EXIT_SCORE = -0.10

    # Bearish reversal threshold.
    BEARISH_REVERSAL_EXIT = 0.65

    # Minimum positive score required for a trade.
    MIN_REENTRY_EDGE = 0.05

    # =========================================================
    # INIT
    # =========================================================

    def __init__(self):

        self.kraken = KrakenTrader(settings)

        self.scanner = KrakenMarketScanner(
            self.kraken,
            settings,
        )

        # -----------------------------------------------------
        # ML STATE
        # -----------------------------------------------------

        self.models = {}
        self.last_train = {}

        # -----------------------------------------------------
        # ENGINE STATE
        # -----------------------------------------------------

        self.running = False

        self.last_scan = None
        self.last_scan_error = None

        self.error = None

        self.signals = []

        # -----------------------------------------------------
        # POSITION STATE
        # -----------------------------------------------------

        self.position_signal = None

        self.last_position_check = None
        self.last_position_reason = None

        # -----------------------------------------------------
        # ROTATION STATE
        # -----------------------------------------------------

        self.last_rotation_check = None
        self.last_rotation_reason = None

        # -----------------------------------------------------
        # RISK
        # -----------------------------------------------------

        self.cooldown_until = 0

        # -----------------------------------------------------
        # SCAN CONTROL
        # -----------------------------------------------------

        self.scan_requested = False
        self.scan_in_progress = False

        self.scan_lock = threading.Lock()

        # -----------------------------------------------------
        # PAPER ACCOUNT INITIALIZATION
        # -----------------------------------------------------

        try:

            if get_risk("paper_balance") is None:
                set_risk(
                    "paper_balance",
                    settings.paper_start_balance,
                )

            if get_risk("paper_start_balance") is None:
                set_risk(
                    "paper_start_balance",
                    settings.paper_start_balance,
                )

            if get_risk("paper_invested") is None:
                set_risk(
                    "paper_invested",
                    0,
                )

            if get_risk("paper_realized_pnl") is None:
                set_risk(
                    "paper_realized_pnl",
                    0,
                )

        except Exception as exc:

            self.error = (
                "PAPER ACCOUNT ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            print(self.error)

    # =========================================================
    # START
    # =========================================================

    def start(self):

        self.running = True
        self.error = None

        return {
            "started": True,
            "running": True,
            "message": "Bot running.",
        }

    # =========================================================
    # MANUAL SCAN
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
            "running": False,
            "message": "Bot stop requested.",
        }

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
    # SAFE FLOAT
    # =========================================================

    @staticmethod
    def _float(value, default=0.0):

        try:

            result = float(value)

            if math.isfinite(result):
                return result

        except Exception:
            pass

        return float(default)

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

    @staticmethod
    def _current_position():

        positions = get_positions()

        if not positions:
            return None

        # HARD RULE:
        # One position at a time.
        return positions[0]

    # =========================================================
    # RISK
    # =========================================================

    def _today_trade_metrics(self):
        # stats() computes the current local calendar-day window directly
        # from completed SELL trades.
        current = stats()
        return (
            self._float(current.get("daily_pnl"), 0.0),
            int(current.get("trades_today", 0) or 0),
        )

    def _can_trade(self):

        try:

            daily_pnl, trades_today = self._today_trade_metrics()
            s = stats()

            if daily_pnl <= -settings.daily_loss_limit_usd:

                return (
                    False,
                    "daily loss limit reached",
                )

            if trades_today >= settings.max_trades_per_day:

                return (
                    False,
                    "daily trade limit reached",
                )

            if (
                s.get("consecutive_losses", 0)
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

    async def scan_symbol(self, candidate):

        symbol = (
            candidate.symbol
            if hasattr(candidate, "symbol")
            else str(candidate)
        )

        # =====================================================
        # OHLCV
        # =====================================================

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

        # =====================================================
        # TRAIN
        # =====================================================

        if (
            state is None
            or
            now - self.last_train.get(symbol, 0)
            >= settings.train_every_seconds
        ):

            training_cost = (
                settings.round_trip_cost_pct / 100
                +
                settings.slippage_buffer_pct / 100
            )

            try:

                state = await asyncio.to_thread(
                    train_model,
                    df,
                    settings.forecast_bars,
                    training_cost,
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
                f"accuracy={self._float(state.accuracy):.3f}"
            )

        # =====================================================
        # PREDICTION
        # =====================================================

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

        # =====================================================
        # TICKER
        # =====================================================

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

        bid = self._float(
            ticker.get("bid")
            or ticker.get("last")
        )

        ask = self._float(
            ticker.get("ask")
            or ticker.get("last")
        )

        last = self._float(
            ticker.get("last")
        )

        if min(
            bid,
            ask,
            last,
        ) <= 0:

            return None

        # =====================================================
        # SPREAD
        # =====================================================

        spread = max(
            0.0,
            ask / bid - 1.0,
        )

        if (
            spread * 100
            > settings.max_spread_pct
        ):

            return None

        # =====================================================
        # EXECUTION COST
        # =====================================================

        base_cost = (
            settings.round_trip_cost_pct / 100
            +
            settings.slippage_buffer_pct / 100
        )

        total_cost = (
            base_cost
            + spread
        )

        # =====================================================
        # MODEL VALUES
        # =====================================================

        probability = self._float(
            prediction.get(
                "probability_up",
                0,
            )
        )

        expected_move = self._float(
            prediction.get(
                "expected_move",
                0,
            )
        )

        expected_net_move = self._float(
            prediction.get(
                "expected_net_move",
                expected_move - base_cost,
            )
        )

        # Live spread is added on top of model training cost.
        execution_net_move = (
            expected_net_move
            - spread
        )

        strategy_score = self._float(
            prediction.get(
                "strategy_score",
                0,
            )
        )

        agreement = self._float(
            prediction.get(
                "strategy_agreement",
                0,
            )
        )

        confidence = self._float(
            prediction.get(
                "confidence",
                0,
            )
        )

        bearish_reversal = self._float(
            prediction.get(
                "bearish_reversal",
                0,
            )
        )

        direction = str(
            prediction.get(
                "direction",
                "NEUTRAL",
            )
        ).upper()

        # =====================================================
        # COMBINED EDGE
        # =====================================================

        # Probability converted to directional strength.
        #
        # 0.50 = neutral
        # 0.60 = moderately bullish
        # 0.70 = strongly bullish
        ml_edge = (
            probability - 0.50
        ) * 2.0

        # Keep strategy contribution bounded.
        strategy_component = max(
            -1.0,
            min(
                1.0,
                strategy_score,
            ),
        )

        combined_edge = (
            ml_edge * 0.70
            +
            strategy_component * 0.30
        )

        # Bearish reversal reduces LONG quality.
        if direction == "LONG":

            combined_edge *= max(
                0.0,
                1.0 - bearish_reversal * 0.70,
            )

        # =====================================================
        # NORMALIZED EDGE SCORE
        # =====================================================
        #
        # OLD:
        #
        # score =
        #     edge * agreement * expected_return
        #
        # That caused values such as:
        #
        # -0.00000
        #
        # because expected return is a small decimal.
        #
        # NEW:
        #
        # Normalize expected movement against the emergency
        # stop distance.
        # =====================================================

        if settings.stop_loss_pct > 0:

            normalized_move = (
                execution_net_move
                /
                settings.stop_loss_pct
            )

        else:

            normalized_move = (
                execution_net_move * 100.0
            )

        normalized_move = max(
            -3.0,
            min(
                3.0,
                normalized_move,
            ),
        )

        agreement_factor = max(
            0.25,
            min(
                1.0,
                agreement,
            ),
        )

        probability_factor = (
            0.75
            +
            0.50
            *
            max(
                probability - 0.50,
                0.0,
            )
        )

        score = (
            combined_edge
            *
            agreement_factor
            *
            normalized_move
            *
            probability_factor
        )

        # =====================================================
        # DYNAMIC REWARD/RISK
        # =====================================================

        if settings.stop_loss_pct > 0:

            reward_risk = (
                execution_net_move
                /
                settings.stop_loss_pct
            )

        else:

            reward_risk = 0.0

        # =====================================================
        # ESTIMATED PROFIT
        # =====================================================

        estimated_profit = execution_net_move

        # =====================================================
        # ENTRY FILTERS
        # =====================================================

        reasons = []

        if direction != "LONG":

            reasons.append(
                f"direction={direction}"
            )

        if probability < settings.min_probability:

            reasons.append(
                f"probability={probability:.3f}"
            )

        if confidence < 0.10:

            reasons.append(
                f"confidence={confidence:.3f}"
            )

        if expected_move <= 0:

            reasons.append(
                "no expected movement"
            )

        if expected_net_move <= 0:

            reasons.append(
                "expected net move is not positive"
            )

        if execution_net_move <= 0:

            reasons.append(
                "execution-adjusted edge is not positive"
            )

        if (
            expected_move
            < settings.min_expected_move
        ):

            reasons.append(
                f"expected_move={expected_move:.4f} "
                f"< minimum={settings.min_expected_move:.4f}"
            )

        if strategy_score < -0.10:

            reasons.append(
                f"strategy bearish={strategy_score:.3f}"
            )

        if agreement < 0.35:

            reasons.append(
                f"low strategy agreement={agreement:.3f}"
            )

        if (
            bearish_reversal
            >= self.BEARISH_REVERSAL_EXIT
        ):

            reasons.append(
                f"bearish reversal risk="
                f"{bearish_reversal:.3f}"
            )

        if reward_risk < settings.min_reward_risk:

            reasons.append(
                f"dynamic reward/risk="
                f"{reward_risk:.2f}"
                f" < minimum={settings.min_reward_risk:.2f}"
            )

        if score <= self.MIN_REENTRY_EDGE:

            reasons.append(
                f"edge score={score:.3f}"
            )

        # =====================================================
        # TRAINING QUALITY
        # =====================================================

        accuracy = self._float(
            getattr(
                state,
                "accuracy",
                0,
            )
        )

        samples = int(
            self._float(
                getattr(
                    state,
                    "samples",
                    0,
                )
            )
        )

        # =====================================================
        # FINAL TRADE DECISION
        # =====================================================

        tradeable = (
            not reasons
            and
            accuracy
            >= settings.min_training_accuracy
            and
            samples >= 200
            and
            score > self.MIN_REENTRY_EDGE
        )

        # =====================================================
        # VOLUME
        # =====================================================

        volume_24h = 0.0

        if hasattr(candidate, "volume_24h"):

            volume_24h = self._float(
                getattr(
                    candidate,
                    "volume_24h",
                    0,
                )
            )

        elif hasattr(candidate, "quote_volume"):

            volume_24h = self._float(
                getattr(
                    candidate,
                    "quote_volume",
                    0,
                )
            )

        # =====================================================
        # RESULT
        # =====================================================

        return {

            "symbol": symbol,

            "price": last,

            "last": last,

            "bid": bid,

            "ask": ask,

            "spread": spread,

            "cost_estimate": total_cost,

            "probability_up": probability,

            "probability": probability,

            "expected_move": expected_move,

            "expected_net_move": expected_net_move,

            "execution_net_move": execution_net_move,

            "direction": direction,

            "confidence": confidence,

            "strategy_score": strategy_score,

            "strategy_agreement": agreement,

            "bearish_reversal": bearish_reversal,

            "combined_edge": combined_edge,

            "estimated_profit": estimated_profit,

            "reward_risk": reward_risk,

            "score": score,

            "tradeable": tradeable,

            "reasons": reasons,

            "accuracy": accuracy,

            "samples": samples,

            "regime": prediction.get(
                "regime",
                "UNKNOWN",
            ),

            "strategies": prediction.get(
                "strategies",
                {},
            ),

            "trained_at": getattr(
                state,
                "trained_at",
                None,
            ),

            # Dashboard compatibility
            "market_rank": 0,

            "volume_24h": volume_24h,
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
                    break

                try:

                    result = await self.scan_symbol(
                        candidate
                    )

                    if result:
                        results.append(result)

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

            # =================================================
            # SORT BY NEW NORMALIZED SCORE
            # =================================================

            results.sort(
                key=lambda x: x.get(
                    "score",
                    0,
                ),
                reverse=True,
            )

            # =================================================
            # MARKET RANK
            # =================================================

            for index, signal in enumerate(
                results,
                start=1,
            ):

                signal["rank"] = index
                signal["market_rank"] = index

            self.signals = results

            self.last_scan = time.time()

            tradeable_count = sum(
                1
                for signal in results
                if signal.get("tradeable")
            )

            blocked_reasons = {}
            for signal in results:
                for reason in signal.get("reasons", [])[:2]:
                    key = str(reason)
                    blocked_reasons[key] = blocked_reasons.get(key, 0) + 1

            top_blockers = sorted(
                blocked_reasons.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]

            print(
                f"AI RESULTS: "
                f"{len(results)} "
                f"tradeable={tradeable_count}"
            )

            if top_blockers:
                print(
                    "AI BLOCKERS: "
                    + " | ".join(
                        f"{reason} x{count}"
                        for reason, count in top_blockers
                    )
                )

            # =================================================
            # LOG TOP RESULTS
            # =================================================

            for signal in results[:10]:

                print(
                    f"AI "
                    f"{signal['symbol']} "
                    f"{signal['direction']} "
                    f"prob="
                    f"{signal['probability_up']:.3f} "
                    f"move="
                    f"{signal['expected_move']:.4f} "
                    f"net="
                    f"{signal['execution_net_move']:.4f} "
                    f"edge="
                    f"{signal['combined_edge']:.3f} "
                    f"score="
                    f"{signal['score']:.3f} "
                    f"RR="
                    f"{signal['reward_risk']:.2f} "
                    f"bearRev="
                    f"{signal['bearish_reversal']:.3f} "
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

            print(self.error)

            return self.signals

        finally:

            with self.scan_lock:
                self.scan_in_progress = False

    # =========================================================
    # BEST TRADEABLE SIGNAL
    # =========================================================

    def _best_tradeable_signal(
        self,
        exclude_symbol=None,
    ):

        candidates = [

            signal

            for signal in self.signals

            if signal.get("tradeable")

            and signal.get(
                "accuracy",
                0,
            )
            >= settings.min_training_accuracy

            and signal.get(
                "execution_net_move",
                0,
            )
            > 0

            and (
                exclude_symbol is None
                or
                signal.get("symbol")
                != exclude_symbol
            )
        ]

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: x.get(
                "score",
                0,
            ),
            reverse=True,
        )

        return candidates[0]

    # =========================================================
    # CURRENT POSITION SIGNAL
    # =========================================================

    async def _current_position_signal(
        self,
        position,
    ):

        symbol = position["symbol"]

        for signal in self.signals:

            if signal.get("symbol") == symbol:
                return signal

        # Position wasn't in the scanner's top list.
        # Analyze it directly so protection continues.
        return await self.scan_symbol(symbol)

    # =========================================================
    # CURRENT POSITION ANALYSIS
    # =========================================================

    async def analyze_current_position(
        self,
        position,
        best_alternative=None,
    ):

        symbol = position["symbol"]

        signal = await self._current_position_signal(
            position
        )

        self.last_position_check = time.time()

        if signal is None:

            self.last_position_reason = (
                "Fresh AI analysis unavailable; "
                "no confirmed bearish signal."
            )

            return True

        self.position_signal = signal

        # =====================================================
        # PRICE / PNL
        # =====================================================

        current_price = self._float(
            signal.get(
                "bid",
                signal.get(
                    "last",
                    0,
                ),
            )
        )

        entry_price = self._float(
            position.get(
                "entry_price",
                0,
            )
        )

        amount = self._float(
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

        # =====================================================
        # AI VALUES
        # =====================================================

        probability = self._float(
            signal.get(
                "probability_up",
                0,
            )
        )

        expected_move = self._float(
            signal.get(
                "expected_move",
                0,
            )
        )

        expected_net_move = self._float(
            signal.get(
                "expected_net_move",
                expected_move,
            )
        )

        execution_net_move = self._float(
            signal.get(
                "execution_net_move",
                expected_net_move,
            )
        )

        strategy_score = self._float(
            signal.get(
                "strategy_score",
                0,
            )
        )

        agreement = self._float(
            signal.get(
                "strategy_agreement",
                0,
            )
        )

        confidence = self._float(
            signal.get(
                "confidence",
                0,
            )
        )

        bearish_reversal = self._float(
            signal.get(
                "bearish_reversal",
                0,
            )
        )

        direction = str(
            signal.get(
                "direction",
                "NEUTRAL",
            )
        ).upper()

        score = self._float(
            signal.get(
                "score",
                0,
            )
        )

        # =====================================================
        # EXIT CONDITIONS
        # =====================================================

        exit_reasons = []

        # Direction reversal.
        if direction != "LONG":

            exit_reasons.append(
                f"AI direction changed to {direction}"
            )

        # AI says positive net edge is gone.
        if execution_net_move <= 0:

            exit_reasons.append(
                "execution-adjusted net edge is no longer positive"
            )

        # Probability deterioration.
        position_probability_floor = max(
            0.50,
            settings.min_probability - 0.05,
        )

        if probability < position_probability_floor:

            exit_reasons.append(
                f"probability deteriorated "
                f"to {probability:.3f}"
            )

        # Strategy deterioration.
        if strategy_score < self.BEARISH_EXIT_SCORE:

            exit_reasons.append(
                f"strategy turned bearish "
                f"({strategy_score:.3f})"
            )

        # Explicit reversal detector.
        if (
            bearish_reversal
            >= self.BEARISH_REVERSAL_EXIT
        ):

            exit_reasons.append(
                f"bearish reversal detected "
                f"({bearish_reversal:.3f})"
            )

        # Agreement deterioration.
        if agreement < 0.30:

            exit_reasons.append(
                f"strategy agreement deteriorated "
                f"({agreement:.3f})"
            )

        # Confidence deterioration.
        if confidence < 0.08:

            exit_reasons.append(
                f"AI confidence deteriorated "
                f"({confidence:.3f})"
            )

        # Score became non-positive.
        if score <= 0:

            exit_reasons.append(
                f"AI score became non-positive "
                f"({score:.3f})"
            )

        # =====================================================
        # ROTATION
        # =====================================================

        rotation_reason = None

        if best_alternative is not None:

            alternative_symbol = (
                best_alternative.get(
                    "symbol"
                )
            )

            alternative_score = self._float(
                best_alternative.get(
                    "score",
                    0,
                )
            )

            current_score = score

            score_difference = (
                alternative_score
                -
                current_score
            )

            materially_better = (

                alternative_score
                >
                current_score
                *
                self.ROTATION_SCORE_MULTIPLIER

                and

                score_difference
                >=
                self.ROTATION_MIN_SCORE_ADVANTAGE
            )

            if materially_better:

                rotation_reason = (
                    f"better opportunity detected: "
                    f"{alternative_symbol} "
                    f"score="
                    f"{alternative_score:.3f} "
                    f"vs current "
                    f"{current_score:.3f}"
                )

        # =====================================================
        # HOLD
        # =====================================================

        if not exit_reasons:

            if rotation_reason:

                self.last_rotation_reason = (
                    rotation_reason
                )

                self.last_position_reason = (
                    rotation_reason
                )

                print(
                    f"ROTATION SIGNAL "
                    f"{symbol}: "
                    f"{rotation_reason}"
                )

                return False

            self.last_position_reason = (
                "Current trade still has positive AI edge; "
                "no materially superior opportunity."
            )

            print(
                f"POSITION HOLD "
                f"{symbol} "
                f"PnL=${unrealized_pnl:.2f} "
                f"prob={probability:.3f} "
                f"net={execution_net_move:.4f} "
                f"score={score:.3f}"
            )

            return True

        # =====================================================
        # STRONG EXIT DECISION
        # =====================================================

        strong_exit = False

        if direction != "LONG":
            strong_exit = True

        elif (
            bearish_reversal
            >= self.BEARISH_REVERSAL_EXIT
        ):
            strong_exit = True

        elif (
            execution_net_move <= 0
            and
            probability
            < position_probability_floor
        ):
            strong_exit = True

        elif (
            probability
            < position_probability_floor
            and
            strategy_score
            < self.BEARISH_EXIT_SCORE
        ):
            strong_exit = True

        elif (
            score <= 0
            and
            execution_net_move <= 0
        ):
            strong_exit = True

        elif len(exit_reasons) >= 3:
            strong_exit = True

        elif rotation_reason:
            strong_exit = True

        # =====================================================
        # DON'T CHURN ON MINOR WEAKNESS
        # =====================================================

        if not strong_exit:

            self.last_position_reason = (
                "Some AI metrics weakened, but the "
                "position still retains enough edge."
            )

            return True

        if rotation_reason:

            self.last_position_reason = (
                rotation_reason
            )

        else:

            self.last_position_reason = (
                "; ".join(exit_reasons)
            )

        print(
            f"POSITION EXIT SIGNAL "
            f"{symbol}: "
            f"{self.last_position_reason}"
        )

        return False

    # =========================================================
    # PAPER COST
    # =========================================================

    @staticmethod
    def _paper_side_cost_rate():

        return (
            settings.round_trip_cost_pct
            /
            100
            /
            2
        )

    # =========================================================
    # EXIT POSITION
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

            market_price = self._float(
                ticker.get("bid")
                or ticker.get("last")
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

            exit_price = self._float(
                result.get("price")
                or market_price
            )

            amount = self._float(
                position["amount"]
            )

            entry_price = self._float(
                position["entry_price"]
            )

            entry_cost = (
                entry_price
                *
                amount
            )

            gross_exit_proceeds = (
                exit_price
                *
                amount
            )

            # =================================================
            # PAPER COST
            # =================================================

            if self.kraken.is_paper:

                exit_cost = (
                    gross_exit_proceeds
                    *
                    self._paper_side_cost_rate()
                )

                exit_proceeds = (
                    gross_exit_proceeds
                    -
                    exit_cost
                )

                entry_fee = (
                    entry_cost
                    *
                    self._paper_side_cost_rate()
                )

                effective_entry_cost = (
                    entry_cost
                    +
                    entry_fee
                )

            else:

                exit_proceeds = gross_exit_proceeds
                effective_entry_cost = entry_cost

            pnl = (
                exit_proceeds
                -
                effective_entry_cost
            )

            # =================================================
            # RECORD SELL
            # =================================================

            add_trade({

                "symbol": symbol,

                "side": "SELL",

                "price": exit_price,

                "amount": amount,

                "notional": exit_proceeds,

                "pnl": pnl,

                "status": "CLOSED",

                "mode": (
                    "DRY_RUN"
                    if self.kraken.is_paper
                    else "LIVE"
                ),

                "reason": reason,
            })

            # Synchronize persisted risk counters after each completed SELL.
            register_closed_trade(pnl)

            # =================================================
            # PAPER ACCOUNT
            # =================================================

            if self.kraken.is_paper:

                balance = self._float(
                    get_risk(
                        "paper_balance",
                        settings.paper_start_balance,
                    )
                )

                invested = self._float(
                    get_risk(
                        "paper_invested",
                        0,
                    )
                )

                realized = self._float(
                    get_risk(
                        "paper_realized_pnl",
                        0,
                    )
                )

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
                    realized
                    +
                    pnl,
                )

            # =================================================
            # DELETE POSITION
            # =================================================

            delete_position(symbol)

            self.position_signal = None

            self.last_position_reason = reason

            # =================================================
            # LOG
            # =================================================

            print("=" * 60)
            print("AI POSITION CLOSED")
            print(f"SYMBOL: {symbol}")
            print(f"ENTRY: ${entry_price:.8f}")
            print(f"EXIT: ${exit_price:.8f}")
            print(f"PNL: ${pnl:.2f}")
            print(f"REASON: {reason}")
            print("=" * 60)

            # =================================================
            # COOLDOWN
            # =================================================

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

            print(self.error)

            return {
                "ok": False,
                "error": self.error,
            }

    # =========================================================
    # MANAGE POSITION
    # =========================================================

    async def manage_positions(self, price: float | None = None) -> bool:
        """Manage the active short-term day trade.

        Priority:
        1) emergency stop
        2) take profit
        3) maximum hold rotation
        4) otherwise hold and continue scanning

        The maximum-hold timer applies even to losing/flat trades so capital
        cannot become trapped indefinitely.
        """
        position = self._current_position()
        if not position:
            return False

        if price is None or price <= 0:
            try:
                ticker = await asyncio.to_thread(
                    self.kraken.fetch_ticker,
                    position["symbol"],
                )
                price = self._float(
                    ticker.get("bid") or ticker.get("last")
                )
            except Exception as exc:
                self.last_position_reason = (
                    f"price refresh failed: {type(exc).__name__}: {exc}"
                )
                return False

        entry_price = self._float(position.get("entry_price", 0))
        if entry_price <= 0 or price <= 0:
            return False

        profit_pct = price / entry_price - 1.0
        target_pct = max(0.0, float(settings.take_profit_pct))

        # 1. Emergency stop always has priority over everything else.
        stop_price = self._float(position.get("stop_price", 0))
        if stop_price > 0 and price <= stop_price:
            result = await self.exit_position(
                position,
                "emergency stop loss",
                apply_cooldown=True,
            )
            return bool(result.get("ok"))

        # 2. Take profit as soon as the configured target is reached.
        if target_pct > 0 and profit_pct >= target_pct:
            result = await self.exit_position(
                position,
                "take-profit reached",
                apply_cooldown=False,
            )
            return bool(result.get("ok"))

        # 3. Rotate after the maximum hold time regardless of P&L.
        #    This prevents a losing/flat position from bypassing the day-trade timer.
        opened_ts = self._float(position.get("opened_ts", 0))
        max_hold_seconds = max(0, int(settings.max_hold_minutes)) * 60
        if opened_ts > 0 and max_hold_seconds > 0:
            held_seconds = time.time() - opened_ts
            if held_seconds >= max_hold_seconds:
                result = await self.exit_position(
                    position,
                    (
                        "max hold reached; rotating to next day-trade "
                        f"(P&L {profit_pct * 100:.3f}%)"
                    ),
                    apply_cooldown=False,
                )
                return bool(result.get("ok"))

        # 4. While below target, keep the position only until the stop or timer.
        if profit_pct <= 0:
            self.last_position_reason = (
                f"holding short-term position "
                f"(P&L {profit_pct * 100:.3f}%; "
                f"max hold {settings.max_hold_minutes}m)"
            )
        else:
            self.last_position_reason = (
                f"holding profitable position "
                f"({profit_pct * 100:.3f}%) until take profit "
                f"(target {target_pct * 100:.3f}%)"
            )

        return False

    async def maybe_enter_best(self):

        # One position only.
        if self._current_position() is not None:
            return False

        if not self.running:
            return False

        # =====================================================
        # RISK
        # =====================================================

        allowed, reason = self._can_trade()

        if not allowed:

            print(
                "TRADE BLOCKED:",
                reason,
            )

            return False

        # =====================================================
        # BEST SIGNAL
        # =====================================================

        best = self._best_tradeable_signal()
        paper_adaptive_used = False

        # PAPER can use an adaptive entry profile so we can
        # validate the execution/risk loop instead of waiting
        # indefinitely for every secondary heuristic to align.
        # LIVE trading always uses the strict signal gate.
        if best is None and self.kraken.is_paper and settings.paper_adaptive_entry:
            best = self._best_paper_adaptive_signal()
            paper_adaptive_used = best is not None

            if best is not None:
                print(
                    "PAPER ADAPTIVE ENTRY: "
                    f"{best['symbol']} "
                    f"prob={best['probability_up']:.3f} "
                    f"net={best['execution_net_move']:.4f} "
                    f"RR={best['reward_risk']:.2f}"
                )

        if best is None:

            print(
                "AI: NO QUALIFIED TRADE"
            )

            return False

        # =====================================================
        # EXTRA SAFETY
        # =====================================================

        if (
            not paper_adaptive_used
            and
            best.get(
                "bearish_reversal",
                0,
            )
            >= self.BEARISH_REVERSAL_EXIT
        ):

            print(
                "ENTRY BLOCKED: "
                "bearish reversal risk too high"
            )

            return False

        if (
            best.get(
                "execution_net_move",
                0,
            )
            <= 0
            and not paper_adaptive_used
        ):

            print(
                "ENTRY BLOCKED: "
                "execution-adjusted edge is not positive"
            )

            return False

        symbol = best["symbol"]

        # =====================================================
        # CAPITAL
        # =====================================================

        if self.kraken.is_paper:

            balance = self._float(
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

                free = self._float(
                    await asyncio.to_thread(
                        self.kraken.free_quote,
                        "USD",
                    )
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

                return False

        if quote < 5:

            print(
                "TRADE BLOCKED: "
                "insufficient capital"
            )

            return False

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

            print(self.error)

            return False

        if not result:

            print(
                "ENTRY BLOCKED: "
                "empty order result"
            )

            return False

        entry = self._float(
            result.get("price")
            or best["ask"]
        )

        amount = self._float(
            result.get("amount")
            or (
                quote / entry
                if entry > 0
                else 0
            )
        )

        if entry <= 0 or amount <= 0:

            print(
                "ENTRY BLOCKED: "
                "invalid execution values"
            )

            return False

        notional = (
            entry
            *
            amount
        )

        # =====================================================
        # PAPER ENTRY COST
        # =====================================================

        if self.kraken.is_paper:

            entry_fee = (
                notional
                *
                self._paper_side_cost_rate()
            )

            total_entry_cash = (
                notional
                +
                entry_fee
            )

        else:

            entry_fee = 0.0
            total_entry_cash = notional

        # =====================================================
        # EMERGENCY STOP
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
        # PAPER BALANCE CHECK BEFORE POSITION
        # =====================================================

        if self.kraken.is_paper:

            balance = self._float(
                get_risk(
                    "paper_balance",
                    settings.paper_start_balance,
                )
            )

            if total_entry_cash > balance:

                print(
                    "ENTRY BLOCKED: "
                    "paper balance insufficient"
                )

                return False

        # =====================================================
        # CREATE POSITION
        # =====================================================

        set_position({

            "symbol": symbol,

            "entry_price": entry,

            "amount": amount,

            "notional": notional,

            "opened_ts": time.time(),

            "stop_price": stop_price,

            "target_price": (
                entry
                * (1.0 + settings.take_profit_pct)
                if settings.take_profit_pct > 0
                else 0
            ),
        })

        # =====================================================
        # PAPER CASH
        # =====================================================

        if self.kraken.is_paper:

            invested = self._float(
                get_risk(
                    "paper_invested",
                    0,
                )
            )

            set_risk(
                "paper_balance",
                max(
                    0,
                    balance
                    -
                    total_entry_cash,
                ),
            )

            set_risk(
                "paper_invested",
                invested
                +
                notional,
            )

        # =====================================================
        # RECORD BUY
        # =====================================================

        add_trade({

            "symbol": symbol,

            "side": "BUY",

            "price": entry,

            "amount": amount,

            "notional": notional,

            "status": "OPEN",

            "mode": (
                "DRY_RUN"
                if self.kraken.is_paper
                else "LIVE"
            ),

            "reason": (
                f"AI DAY-TRADE ENTRY "
                f"prob="
                f"{best['probability_up']:.3f} "
                f"confidence="
                f"{best['confidence']:.3f} "
                f"expected_net="
                f"{best['execution_net_move']:.4f} "
                f"score="
                f"{best['score']:.3f}"
            ),
        })

        self.position_signal = best

        # =====================================================
        # ENTRY LOG
        # =====================================================

        print("=" * 60)
        print("AI DAY-TRADE ENTRY")
        print("=" * 60)
        print(f"SYMBOL: {symbol}")
        print(f"PRICE: {entry:.8f}")
        print(f"NOTIONAL: ${notional:.2f}")
        print(
            f"PROBABILITY: "
            f"{best['probability_up']:.3f}"
        )
        print(
            f"EXPECTED MOVE: "
            f"{best['expected_move']:.4f}"
        )
        print(
            f"EXPECTED NET: "
            f"{best['execution_net_move']:.4f}"
        )
        print(
            f"COMBINED EDGE: "
            f"{best['combined_edge']:.3f}"
        )
        print(
            f"AI SCORE: "
            f"{best['score']:.3f}"
        )
        print(
            f"REWARD/RISK: "
            f"{best['reward_risk']:.2f}"
        )
        print(
            f"BEARISH REVERSAL: "
            f"{best['bearish_reversal']:.3f}"
        )
        print(
            f"STOP: {stop_price:.8f}"
        )
        print(
            f"TAKE PROFIT BACKSTOP: "
            f"{settings.take_profit_pct * 100:.2f}%"
        )
        print(
            f"MAX HOLD BACKSTOP: "
            f"{settings.max_hold_minutes} minutes"
        )
        print("MARKET SCANNING: CONTINUES")
        print("ROTATION: ENABLED")
        print("=" * 60)

        return True

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
                ] = self._float(
                    ticker.get("bid")
                    or ticker.get("last")
                    or position["entry_price"]
                )

            except Exception:

                prices[
                    position["symbol"]
                ] = self._float(
                    position["entry_price"]
                )

        try:

            record_equity_snapshot(
                prices
            )

        except Exception as exc:

            print(
                "EQUITY SNAPSHOT ERROR:",
                type(exc).__name__,
                exc,
            )

    # =========================================================
    # AUTONOMOUS DAY-TRADING LOOP
    # =========================================================

    async def run(self):

        self.running = True
        self.error = None

        print("=" * 60)
        print(
            "AUTONOMOUS DAY-TRADING ENGINE ONLINE"
        )
        print(
            "FULL MARKET SCANNER ONLINE"
        )
        print(
            "AI EDGE ENGINE ONLINE"
        )
        print(
            f"PAPER MODE: {self.kraken.is_paper}"
        )
        print(
            "ONE POSITION AT A TIME: ENABLED"
        )
        print(
            "CONTINUOUS MARKET SCANNING: ENABLED"
        )
        print(
            "AI POSITION ROTATION: ENABLED"
        )
        print(
            "BEARISH REVERSAL EXIT: ENABLED"
        )
        print(
            "EMERGENCY STOP LOSS: ENABLED"
        )
        print(
            f"TAKE PROFIT: {settings.take_profit_pct * 100:.3f}%"
        )
        print(
            f"MAX HOLD: {settings.max_hold_minutes} MINUTES"
        )
        print("=" * 60)

        try:

            while self.running:

                cycle_started = time.time()

                try:

                    # =================================================
                    # ALWAYS SCAN FIRST
                    # =================================================

                    if self.scan_requested:

                        print(
                            "MANUAL SCAN REQUEST ACCEPTED"
                        )

                    await self.scan()

                    if not self.running:
                        break

                    # =================================================
                    # CURRENT POSITION
                    # =================================================

                    existing_position = (
                        self._current_position()
                    )

                    if existing_position is not None:

                        current_symbol = (
                            existing_position["symbol"]
                        )

                        # =================================================
                        # FIND BETTER ALTERNATIVE
                        # =================================================

                        best_alternative = (
                            self._best_tradeable_signal(
                                exclude_symbol=current_symbol
                            )
                        )

                        self.last_rotation_check = (
                            time.time()
                        )

                        if best_alternative:

                            print(
                                f"BEST ALTERNATIVE: "
                                f"{best_alternative['symbol']} "
                                f"score="
                                f"{best_alternative['score']:.3f}"
                            )

                        # =================================================
                        # MANAGE CURRENT POSITION
                        # =================================================

                        await self.manage_positions()

                        if not self.running:
                            break

                        # =================================================
                        # IF EXITED, LOOK FOR NEXT TRADE
                        # =================================================

                        if (
                            self._current_position()
                            is None
                        ):

                            await self.maybe_enter_best()

                    else:

                        # =================================================
                        # FLAT
                        # =================================================

                        await self.maybe_enter_best()

                    if not self.running:
                        break

                    # =================================================
                    # EQUITY
                    # =================================================

                    await self.mark_equity()

                    self.error = None

                except asyncio.CancelledError:

                    raise

                except Exception as exc:

                    self.error = (
                        "BOT CYCLE ERROR: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                    print(
                        self.error
                    )

                    # A single failed market/API cycle does not
                    # permanently kill the autonomous worker.

                # =================================================
                # CYCLE DELAY
                # =================================================

                if self.running:

                    configured_delay = max(
                        5,
                        int(
                            getattr(
                                settings,
                                "scan_seconds",
                                30,
                            )
                        ),
                    )

                    elapsed = (
                        time.time()
                        -
                        cycle_started
                    )

                    remaining = max(
                        0,
                        configured_delay
                        -
                        elapsed,
                    )

                    if remaining > 0:

                        await asyncio.sleep(
                            remaining
                        )

        except asyncio.CancelledError:

            self.running = False

            print(
                "AUTONOMOUS DAY-TRADING ENGINE CANCELLED"
            )

            raise

        except Exception as exc:

            self.error = (
                "BOT LOOP FATAL ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            print(
                self.error
            )

        finally:

            self.running = False

            print("=" * 60)
            print(
                "AUTONOMOUS DAY-TRADING ENGINE OFFLINE"
            )
            print("=" * 60)

    # =========================================================
    # STATUS
    # =========================================================

    def status(self):

        position = self._current_position()

        scanner_status = self.scanner.status()

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

            "last_rotation_check":
                self.last_rotation_check,

            "last_rotation_reason":
                self.last_rotation_reason,

            "error":
                self.error,

            "signals":
                len(self.signals),

            "models":
                len(self.models),

            "mode":
                getattr(
                    self.kraken,
                    "mode",
                    "PAPER",
                ),

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

        best = self._best_tradeable_signal(
            exclude_symbol=(
                position["symbol"]
                if position
                else None
            )
        )

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

            "last_rotation_check":
                self.last_rotation_check,

            "last_rotation_reason":
                self.last_rotation_reason,

            "best_alternative":
                best,

            "error":
                self.error,

            "signals":
                len(self.signals),

            "models":
                len(self.models),

            "markets_discovered":
                scanner_status.get(
                    "markets_discovered",
                    len(self.scanner.markets),
                ),

            "crypto_markets":
                scanner_status.get(
                    "crypto_markets",
                    0,
                ),

            "forex_markets":
                scanner_status.get(
                    "forex_markets",
                    0,
                ),

            "futures_markets":
                scanner_status.get(
                    "futures_markets",
                    0,
                ),

            "xstocks_markets":
                scanner_status.get(
                    "xstocks_markets",
                    0,
                ),

            "max_scan_symbols":
                settings.max_scan_symbols,

            "allowed_quotes":
                settings.allowed_quote_list,

            "rotation":
                {
                    "enabled": True,

                    "min_score_advantage":
                        self.ROTATION_MIN_SCORE_ADVANTAGE,

                    "score_multiplier":
                        self.ROTATION_SCORE_MULTIPLIER,
                },

            "risk":
                {
                    "emergency_stop_loss":
                        settings.stop_loss_pct,

                    "fixed_take_profit":
                        settings.take_profit_pct > 0,

                    "max_hold":
                        settings.max_hold_minutes > 0,

                    "max_hold_minutes":
                        settings.max_hold_minutes,

                    "take_profit_pct":
                        settings.take_profit_pct,

                    "stop_loss_pct":
                        settings.stop_loss_pct,
                },

        }
