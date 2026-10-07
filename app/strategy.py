import time
from decimal import Decimal
from .config import settings
from .models import Opportunity

class Engine:
    def __init__(self, api, db, log):
        self.api = api
        self.db = db
        self.log = log
        self.failures = 0
        self.last = 0

    def scan(self):
        out = []
        for raw in self.api.markets():
            text = " ".join(str(getattr(raw, x, "") or "") for x in ("question","slug","title")).lower()
            if settings.keywords and not any(k in text for k in settings.keywords):
                continue
            if settings.short_only and not self.api.short(raw):
                continue
            m = self.api.info(raw)
            if not m:
                continue
            try:
                y, n = self.api.books(m.yes, m.no)
            except Exception as e:
                self.log("WARN", "BOOK_ERROR", str(e), m.id)
                continue
            if y.ask is None or n.ask is None:
                continue
            cost = y.ask + n.ask
            gross = Decimal(1) - cost
            net = gross - settings.fee_buffer - settings.slippage_buffer
            if net < settings.min_net_edge:
                continue
            size = min(
                y.ask_size,
                n.ask_size,
                settings.max_pair_size,
                settings.max_pair_spend / cost,
            )
            if m.min_size > 0:
                size = (size / m.min_size).to_integral_value() * m.min_size
            if size < max(settings.min_pair_size, m.min_size):
                continue
            out.append(Opportunity(m, y, n, size, cost, gross, net))
        return sorted(out, key=lambda x: x.net, reverse=True)

    def execute(self, o):
        if self.failures >= settings.max_failures:
            return False
        if time.time() - self.last < settings.cooldown:
            return False
        if self.db.daily_spend() + o.size * o.cost > settings.max_daily_spend:
            return False
        try:
            y = self.api.buy(o.market.yes, o.size, o.yes.ask)
            n = self.api.buy(o.market.no, o.size, o.no.ask)

            def status(x):
                return str(
                    getattr(x, "status", None)
                    or (x.get("status", "SUBMITTED") if isinstance(x, dict) else "SUBMITTED")
                )

            self.db.order(
                ts=int(time.time()), order_id="", market_id=o.market.id,
                asset_id=o.market.yes, side="BUY", shares=float(o.size),
                price=float(o.yes.ask), status=status(y),
                paper=int(not settings.live_trading), raw=str(y),
            )
            self.db.order(
                ts=int(time.time()), order_id="", market_id=o.market.id,
                asset_id=o.market.no, side="BUY", shares=float(o.size),
                price=float(o.no.ask), status=status(n),
                paper=int(not settings.live_trading), raw=str(n),
            )

            if not settings.live_trading:
                for aid, outcome, p in (
                    (o.market.yes, "YES", o.yes.ask),
                    (o.market.no, "NO", o.no.ask),
                ):
                    self.db.fill(
                        ts=int(time.time()), order_id="", market_id=o.market.id,
                        asset_id=aid, outcome=outcome, shares=float(o.size),
                        price=float(p), spend=float(o.size * p), paper=1,
                    )

            self.failures = 0
            self.last = time.time()
            self.log(
                "INFO", "PAIR",
                f"{o.size} YES + {o.size} NO | cost={o.cost:.4f} | net={o.net:.4f}",
                o.market.id,
            )
            return True
        except Exception as e:
            self.failures += 1
            self.log("ERROR", "EXECUTION_ERROR", repr(e), o.market.id)
            return False
