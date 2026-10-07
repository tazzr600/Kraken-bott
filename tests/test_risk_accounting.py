import os
import tempfile
import time
import unittest


class RiskAccountingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        os.environ["DB_PATH"] = os.path.join(
            cls.tmpdir.name,
            "test.sqlite3",
        )

        import db

        cls.db = db
        cls.db.init_db()

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def setUp(self):
        self.db.clear_positions()
        with self.db.db() as connection:
            connection.execute("DELETE FROM trades")
            connection.execute("DELETE FROM risk_state")
        self.db.init_db()

    def add_trade(self, *, ts, side, status, pnl):
        return self.db.add_trade({
            "ts": ts,
            "symbol": "BTC/USD",
            "side": side,
            "price": 100.0,
            "amount": 1.0,
            "notional": 100.0,
            "pnl": pnl,
            "status": status,
            "mode": "DRY_RUN",
        })

    def test_daily_stats_use_calendar_day_and_completed_sells(self):
        now = time.time()
        local = time.localtime(now)
        midnight = time.mktime((
            local.tm_year,
            local.tm_mon,
            local.tm_mday,
            0, 0, 0,
            local.tm_wday,
            local.tm_yday,
            local.tm_isdst,
        ))

        self.add_trade(
            ts=midnight - 60,
            side="SELL",
            status="CLOSED",
            pnl=-20,
        )
        self.add_trade(
            ts=now,
            side="BUY",
            status="OPEN",
            pnl=0,
        )
        self.add_trade(
            ts=now,
            side="SELL",
            status="CLOSED",
            pnl=-7.5,
        )
        self.add_trade(
            ts=now,
            side="SELL",
            status="CLOSED",
            pnl=3.0,
        )

        stats = self.db.stats()

        self.assertAlmostEqual(stats["daily_pnl"], -4.5)
        self.assertEqual(stats["trades_today"], 2)

    def test_daily_state_resets_persisted_counters_at_new_day(self):
        self.db.set_risk("daily_start_date", "1900-01-01")
        self.db.set_risk("trades_today", 99)
        self.db.set_risk("consecutive_losses", 7)

        self.db.reset_daily_state_if_needed()

        self.assertEqual(
            self.db.get_risk("daily_start_date", ""),
            time.strftime("%Y-%m-%d", time.localtime()),
        )
        self.assertEqual(self.db.get_trades_today(), 0)
        self.assertEqual(self.db.get_consecutive_losses(), 0)

    def test_completed_sell_synchronizes_risk_counters(self):
        self.add_trade(
            ts=time.time(),
            side="SELL",
            status="CLOSED",
            pnl=-4,
        )
        self.db.register_closed_trade(-4)

        self.assertEqual(
            self.db.get_trades_today(),
            1,
        )
        self.assertEqual(
            self.db.get_consecutive_losses(),
            1,
        )

        self.add_trade(
            ts=time.time(),
            side="SELL",
            status="CLOSED",
            pnl=2,
        )
        self.db.register_closed_trade(2)

        self.assertEqual(
            self.db.get_trades_today(),
            2,
        )
        self.assertEqual(
            self.db.get_consecutive_losses(),
            0,
        )

    def test_paper_cost_model_uses_both_sides_of_slippage(self):
        from config import Settings
        from kraken_client import KrakenTrader

        settings = Settings()
        settings.round_trip_cost_pct = 1.60
        settings.slippage_buffer_pct = 0.10

        trader = KrakenTrader(settings)
        self.assertAlmostEqual(
            trader.settings.round_trip_cost_pct / 100
            + (trader.settings.slippage_buffer_pct / 100) * 2,
            0.018,
            places=8,
        )

    def test_paper_market_fills_apply_configured_slippage(self):
        from config import Settings
        from kraken_client import KrakenTrader

        settings = Settings()
        settings.slippage_buffer_pct = 0.10

        trader = KrakenTrader(settings)
        trader.exchange.markets = {"BTC/USD": {}}
        trader._markets_loaded = True
        trader.fetch_ticker = lambda symbol: {
            "ask": 100.0,
            "bid": 99.0,
        }
        trader._normalize_amount = lambda symbol, amount: float(amount)
        trader._market_limits = lambda symbol: {
            "min_amount": None,
            "max_amount": None,
            "min_cost": None,
            "max_cost": None,
        }

        buy = trader.market_buy("BTC/USD", 50.0)
        self.assertAlmostEqual(buy["price"], 100.10, places=8)
        self.assertAlmostEqual(
            buy["quote_amount"],
            buy["amount"] * 100.10,
            places=8,
        )

        sell = trader.market_sell("BTC/USD", buy["amount"])
        self.assertAlmostEqual(sell["price"], 98.901, places=8)
        self.assertAlmostEqual(
            sell["quote_amount"],
            sell["amount"] * 98.901,
            places=8,
        )

    def test_profit_lock_parameters_cover_modeled_round_trip_cost(self):
        from config import Settings

        settings = Settings()
        settings.round_trip_cost_pct = 1.60
        settings.slippage_buffer_pct = 0.10
        settings.profit_lock_trigger_pct = 2.50
        settings.profit_lock_trigger_buffer_pct = 0.25
        settings.profit_lock_min_net_pct = 0.25

        modeled_cost = (
            settings.round_trip_cost_pct / 100
            + (settings.slippage_buffer_pct / 100) * 2
        )
        trigger = max(
            modeled_cost
            + settings.profit_lock_trigger_buffer_pct / 100,
            settings.profit_lock_trigger_pct / 100,
        )
        floor = max(
            modeled_cost
            + settings.profit_lock_min_net_pct / 100,
            0.0,
        )

        self.assertAlmostEqual(modeled_cost, 0.018)
        self.assertAlmostEqual(trigger, 0.025)
        self.assertAlmostEqual(floor, 0.0205)


    def test_live_order_fee_is_normalized_to_quote_currency(self):
        from config import Settings
        from kraken_client import KrakenTrader

        settings = Settings()
        trader = KrakenTrader(settings)
        trader.exchange.markets = {
            "BTC/USD": {"quote": "USD"},
        }
        trader._markets_loaded = True

        fee = trader._order_fee_quote(
            {"fee": {"cost": 0.40, "currency": "USD"}},
            "BTC/USD",
            100.0,
        )
        self.assertAlmostEqual(fee, 0.40, places=8)

        base_fee = trader._order_fee_quote(
            {"fee": {"cost": 0.004, "currency": "BTC"}},
            "BTC/USD",
            100.0,
        )
        self.assertAlmostEqual(base_fee, 0.40, places=8)

    def test_paper_mode_is_the_default_and_live_is_disabled(self):
        from config import Settings
        from kraken_client import KrakenTrader

        settings = Settings()
        self.assertTrue(settings.dry_run)
        self.assertFalse(settings.live_trading)

        trader = KrakenTrader(settings)

        self.assertEqual(trader.mode, "PAPER")
        self.assertFalse(trader.live_orders_enabled)

        with self.assertRaises(RuntimeError):
            trader.set_mode("LIVE")


if __name__ == "__main__":
    unittest.main()
