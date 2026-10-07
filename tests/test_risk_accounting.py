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
