from __future__ import annotations

import asyncio
import time

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
)
from kraken_client import KrakenTrader
from ml_model import predict, train_model


class KrakenBot:

    def __init__(self):
        self.kraken = KrakenTrader(settings)

        self.models = {}
        self.last_train = {}

        self.running = settings.autonomous

        self.last_scan = None
        self.error = None
        self.signals = []

        self.cooldown_until = 0

        if get_risk("paper_balance") is None:
            set_risk(
                "paper_balance",
                settings.paper_start_balance,
            )

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------

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

    @staticmethod
    def _position(symbol):
        return next(
            (
                p
                for p in get_positions()
                if p["symbol"] == symbol
            ),
            None,
        )

    # ---------------------------------------------------------
    # RISK CONTROL
    # ---------------------------------------------------------

    def _can_trade(self):

        s = stats()

        if (
            s["last_24h_pnl"]
            <= -settings.daily_loss_limit_usd
        ):
            return False, "24h loss limit reached"

        if (
            s["trades_24h"]
            >= settings.max_trades_per_day
        ):
            return False, "24h trade limit reached"

        if (
            s["consecutive_losses"]
            >= settings.max_consecutive_losses
        ):
            return False, "loss circuit breaker active"

        if time.time() < self.cooldown_until:
            return False, "cooldown active"

        return True, ""

    # ---------------------------------------------------------
    # SCAN ONE SYMBOL
    # ---------------------------------------------------------

    async def scan_symbol(self, symbol):

        rows = await asyncio.to_thread(
            self.kraken.fetch_ohlcv,
            symbol,
            settings.timeframe,
            settings.candles,
        )

        if not rows:
            return None

        df = self._df(rows)

        if len(df) < 150:
            return None

        now = time.time()

        state = self.models.get(symbol)

        # -----------------------------------------------------
        # TRAIN / RETRAIN
        # -----------------------------------------------------

        if (
            state is None
            or now
            - self.last_train.get(symbol, 0)
            >= settings.train_every_seconds
        ):

            cost = (
                settings.round_trip_cost_pct / 100
                + settings.slippage_buffer_pct / 100
            )

            state = await asyncio.to_thread(
                train_model,
                df,
                settings.forecast_bars,
                cost,
            )

            self.models[symbol] = state
            self.last_train[symbol] = now

            print(
                f"MODEL TRAINED {symbol} "
                f"accuracy={state.accuracy:.3f} "
                f"samples={state.samples} "
                f"regime={state.regime}"
            )

        # -----------------------------------------------------
        # AI PREDICTION
        # -----------------------------------------------------

        prediction = predict(
            state,
            df,
            settings.forecast_bars,
        )

        if not prediction:
            return None

        ticker = await asyncio.to_thread(
            self.kraken.fetch_ticker,
            symbol,
        )

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

        if min(bid, ask, last) <= 0:
            return None

        # -----------------------------------------------------
        # MARKET COST
        # -----------------------------------------------------

        spread = max(
            0,
            ask / bid - 1,
        )

        base_cost = (
            settings.round_trip_cost_pct / 100
            + settings.slippage_buffer_pct / 100
        )

        total_cost = (
            base_cost
            + spread
        )

        probability = prediction[
            "probability_up"
        ]

        expected_move = prediction[
            "expected_move"
        ]

        strategy_score = prediction[
            "strategy_score"
        ]

        agreement = prediction[
            "strategy_agreement"
        ]

        confidence = prediction[
            "confidence"
        ]

        direction = prediction[
            "direction"
        ]

        # -----------------------------------------------------
        # EDGE CALCULATION
        # -----------------------------------------------------

        ml_edge = (
            probability - 0.5
        ) * 2

        strategy_edge = strategy_score

        combined_edge = (
            ml_edge * 0.70
            + strategy_edge * 0.30
        )

        estimated_profit = (
            expected_move
            - total_cost
        )

        # Score rewards:
        #
        # ML confidence
        # strategy agreement
        # expected movement
        # lower costs
        #
        score = (
            combined_edge
            * max(agreement, 0.25)
            * max(expected_move, 0)
            - total_cost
        )

        # -----------------------------------------------------
        # TRADE DECISION
        # -----------------------------------------------------

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

        if expected_move <= total_cost:
            reasons.append(
                "expected move below trading cost"
            )

        if strategy_score < -0.10:
            reasons.append(
                f"strategy bearish={strategy_score:.3f}"
            )

        if agreement < 0.35:
            reasons.append(
                f"low strategy agreement={agreement:.3f}"
            )

        tradeable = (
            len(reasons) == 0
            and score > 0
        )

        # -----------------------------------------------------
        # RETURN SIGNAL
        # -----------------------------------------------------

        return {
            "symbol": symbol,

            "price": last,
            "bid": bid,
            "ask": ask,

            "spread": spread,
            "cost_estimate": total_cost,

            "probability_up": probability,
            "expected_move": expected_move,

            "direction": direction,
            "confidence": confidence,

            "strategy_score": strategy_score,
            "strategy_agreement": agreement,

            "combined_edge": combined_edge,
            "estimated_profit": estimated_profit,

            "score": score,

            "tradeable": tradeable,

            "reasons": reasons,

            "accuracy": state.accuracy,
            "samples": state.samples,

            "regime": prediction[
                "regime"
            ],

            "strategies": prediction[
                "strategies"
            ],

            "trained_at": state.trained_at,
        }

    # ---------------------------------------------------------
    # SCAN ALL MARKETS
    # ---------------------------------------------------------

    async def scan(self):

        results = []

        for symbol in settings.symbols:

            try:

                result = await self.scan_symbol(
                    symbol
                )

                if result:
                    results.append(result)

            except Exception as e:

                print(
                    f"SCAN ERROR {symbol}: "
                    f"{type(e).__name__}: {e}"
                )

        results.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        self.signals = results

        self.last_scan = time.time()

        # Print AI ranking
        for signal in results:

            print(
                f"AI {signal['symbol']} "
                f"direction={signal['direction']} "
                f"prob={signal['probability_up']:.3f} "
                f"confidence={signal['confidence']:.3f} "
                f"strategy={signal['strategy_score']:.3f} "
                f"agreement={signal['strategy_agreement']:.3f} "
                f"move={signal['expected_move']:.4f} "
                f"score={signal['score']:.5f} "
                f"tradeable={signal['tradeable']}"
            )

        return results

    # ---------------------------------------------------------
    # MANAGE OPEN POSITIONS
    # ---------------------------------------------------------

    async def manage_positions(self):

        for position in get_positions():

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
                    - position["opened_ts"]
                ) / 60

                reason = None

                if price <= position[
                    "stop_price"
                ]:
                    reason = "stop_loss"

                elif price >= position[
                    "target_price"
                ]:
                    reason = "take_profit"

                elif age >= settings.max_hold_minutes:
                    reason = "time_exit"

                if not reason:
                    continue

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
                    - position["entry_price"]
                ) * position["amount"]

                mode = (
                    "DRY_RUN"
                    if settings.dry_run
                    else "LIVE"
                )

                add_trade(
                    {
                        "symbol": position["symbol"],
                        "side": "SELL",
                        "price": exit_price,
                        "amount": position["amount"],
                        "notional": (
                            exit_price
                            * position["amount"]
                        ),
                        "pnl": pnl,
                        "status": "CLOSED",
                        "mode": mode,
                        "reason": reason,
                    }
                )

                if settings.dry_run:

                    balance = float(
                        get_risk(
                            "paper_balance",
                            settings.paper_start_balance,
                        )
                    )

                    set_risk(
                        "paper_balance",
                        balance + pnl,
                    )

                delete_position(
                    position["symbol"]
                )

                self.cooldown_until = (
                    time.time()
                    + settings.cooldown_minutes * 60
                )

                print(
                    f"EXIT "
                    f"{position['symbol']} "
                    f"{reason} "
                    f"pnl={pnl:.2f}"
                )

            except Exception as e:

                self.error = (
                    f"POSITION ERROR: "
                    f"{type(e).__name__}: {e}"
                )

                print(self.error)

    # ---------------------------------------------------------
    # SELECT BEST AI TRADE
    # ---------------------------------------------------------

    async def maybe_enter_best(self):

        if not self.signals:
            return

        allowed, reason = self._can_trade()

        if not allowed:
            print(
                f"TRADE BLOCKED: {reason}"
            )
            return

        # Only signals that actually pass
        # every AI/trading filter.
        candidates = [
            signal
            for signal in self.signals
            if signal.get("tradeable")
        ]

        if not candidates:
            print(
                "AI: No trade meets all "
                "risk and strategy requirements."
            )
            return

        best = candidates[0]

        # -----------------------------------------------------
        # EXISTING POSITION
        # -----------------------------------------------------

        if self._position(
            best["symbol"]
        ):
            return

        # -----------------------------------------------------
        # MODEL QUALITY
        # -----------------------------------------------------

        if (
            best["accuracy"]
            < settings.min_training_accuracy
        ):
            print(
                f"AI BLOCKED {best['symbol']}: "
                f"accuracy={best['accuracy']:.3f}"
            )
            return

        # -----------------------------------------------------
        # CAPITAL
        # -----------------------------------------------------

        quote = settings.max_trade_usd

        if settings.dry_run:

            balance = float(
                get_risk(
                    "paper_balance",
                    settings.paper_start_balance,
                )
            )

            quote = min(
                quote,
                balance
                * settings.max_position_pct,
            )

        else:

            try:

                free = await asyncio.to_thread(
                    self.kraken.free_quote,
                    "USD",
                )

                quote = min(
                    quote,
                    free
                    * settings.max_position_pct,
                )

            except Exception as e:

                print(
                    f"BALANCE ERROR: "
                    f"{type(e).__name__}: {e}"
                )

                return

        if quote < 5:
            print(
                f"AI BLOCKED {best['symbol']}: "
                f"trade size below minimum"
            )
            return

        # -----------------------------------------------------
        # EXECUTE BUY
        # -----------------------------------------------------

        result = await asyncio.to_thread(
            self.kraken.market_buy,
            best["symbol"],
            quote,
        )

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
        # POSITION RISK
        # -----------------------------------------------------

        stop_price = (
            entry
            * (
                1
                - settings.stop_loss_pct
            )
        )

        target_price = (
            entry
            * (
                1
                + settings.take_profit_pct
            )
        )

        set_position(
            {
                "symbol": best["symbol"],
                "entry_price": entry,
                "amount": amount,
                "notional": notional,
                "opened_ts": time.time(),
                "stop_price": stop_price,
                "target_price": target_price,
            }
        )

        add_trade(
            {
                "symbol": best["symbol"],
                "side": "BUY",
                "price": entry,
                "amount": amount,
                "notional": notional,
                "status": "OPEN",
                "mode": (
                    "DRY_RUN"
                    if settings.dry_run
                    else "LIVE"
                ),
                "reason": (
                    f"AI "
                    f"p={best['probability_up']:.3f} "
                    f"confidence={best['confidence']:.3f} "
                    f"strategy={best['strategy_score']:.3f} "
                    f"agreement={best['strategy_agreement']:.3f} "
                    f"regime={best['regime']}"
                ),
            }
        )

        print(
            f"AI ENTRY "
            f"{best['symbol']} "
            f"price={entry:.4f} "
            f"amount={amount:.8f} "
            f"score={best['score']:.5f}"
        )

    # ---------------------------------------------------------
    # MAIN ENGINE
    # ---------------------------------------------------------

    async def run(self):

        while True:

            if self.running:

                try:

                    await self.manage_positions()

                    await self.scan()

                    await self.maybe_enter_best()

                    self.error = None

                except Exception as e:

                    self.error = (
                        f"BOT ERROR: "
                        f"{type(e).__name__}: {e}"
                    )

                    print(self.error)

            await asyncio.sleep(
                max(
                    5,
                    settings.scan_seconds,
                )
            )


bot = KrakenBot()
