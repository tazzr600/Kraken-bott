import unittest

from market_scanner import KrakenMarketScanner, MarketCandidate


class FakeSettings:
    allowed_quote_list = ["USD"]
    market_refresh_seconds = 1800
    max_scan_symbols = 30
    min_quote_volume_usd = 250000
    max_spread_pct = 0.8


class FakeKraken:
    pass


class MultiMarketScannerTests(unittest.TestCase):

    def setUp(self):
        self.scanner = KrakenMarketScanner(
            FakeKraken(),
            FakeSettings(),
        )

    def test_market_family_classification(self):
        self.assertEqual(
            self.scanner._market_type(
                {"base": "BTC", "quote": "USD"}
            ),
            "CRYPTO",
        )

        self.assertEqual(
            self.scanner._market_type(
                {"base": "EUR", "quote": "USD"}
            ),
            "FOREX",
        )

        self.assertEqual(
            self.scanner._market_type(
                {"base": "AAPLx", "quote": "USD"}
            ),
            "XSTOCKS",
        )

    def test_status_exposes_multi_market_counts(self):
        self.scanner.markets = {
            "BTC/USD": {},
            "EUR/USD": {},
            "AAPLx/USD": {},
        }
        self.scanner.universe = [
            MarketCandidate(
                symbol="BTC/USD",
                base="BTC",
                quote="USD",
                last=1,
                bid=1,
                ask=1,
                spread_pct=0,
                quote_volume=1_000_000,
                liquidity_score=1,
                market_type="CRYPTO",
            ),
            MarketCandidate(
                symbol="EUR/USD",
                base="EUR",
                quote="USD",
                last=1,
                bid=1,
                ask=1,
                spread_pct=0,
                quote_volume=1_000_000,
                liquidity_score=1,
                market_type="FOREX",
            ),
            MarketCandidate(
                symbol="AAPLx/USD",
                base="AAPLx",
                quote="USD",
                last=1,
                bid=1,
                ask=1,
                spread_pct=0,
                quote_volume=1_000_000,
                liquidity_score=1,
                market_type="XSTOCKS",
            ),
        ]

        status = self.scanner.status()

        self.assertEqual(status["crypto_markets"], 1)
        self.assertEqual(status["forex_markets"], 1)
        self.assertEqual(status["xstocks_markets"], 1)
        self.assertEqual(status["futures_markets"], 0)


if __name__ == "__main__":
    unittest.main()
