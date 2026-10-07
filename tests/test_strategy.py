import unittest
from decimal import Decimal
from types import SimpleNamespace
from app.models import Position
from app.strategy import FairValueModel

class StrategyTests(unittest.TestCase):
    def test_paired_vwap(self):
        p=Position("m",Decimal("10"),Decimal("10"),Decimal("4.60"),Decimal("5.10"),Decimal("10"))
        self.assertEqual(p.yes_vwap+p.no_vwap,Decimal("0.97"))

    def test_probability_bounds(self):
        s=SimpleNamespace(price_distance_weight=.35,momentum_weight=.35,speed_weight=.20,basis_weight=.10,fee_buffer=Decimal(".006"),slippage_buffer=Decimal(".004"),volatility_floor=.00035)
        m=SimpleNamespace(end_ts=10**10)
        u=SimpleNamespace(spot=101.0,start_price=100.0,momentum_15s=.001,momentum_30s=.001,momentum_60s=.001,speed=.00001,basis=.0002,volatility=.001)
        yes=SimpleNamespace(ask=Decimal(".55")); no=SimpleNamespace(ask=Decimal(".45"))
        sig=FairValueModel(s).estimate(m,u,yes,no)
        self.assertGreaterEqual(sig.fair_up,.01)
        self.assertLessEqual(sig.fair_up,.99)
        self.assertAlmostEqual(sig.fair_up+sig.fair_down,1.0,places=8)

if __name__=="__main__":
    unittest.main()
