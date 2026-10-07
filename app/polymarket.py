import os
from decimal import Decimal
from polymarket import PublicClient, SecureClient
from .config import settings
from .models import Book, Market, val

class PM:
    def __init__(self):
        self.public = PublicClient()
        self.secure = None
        if settings.live_trading:
            key = os.getenv("POLYMARKET_PRIVATE_KEY")
            wallet = os.getenv("POLYMARKET_DEPOSIT_WALLET")
            if not key or not wallet:
                raise RuntimeError("LIVE_TRADING=true requires Polymarket credentials")
            self.secure = SecureClient.create(private_key=key, wallet=wallet)

    def close(self):
        for c in (self.public, self.secure):
            if c:
                try:
                    c.close()
                except Exception:
                    pass

    def markets(self):
        return self.public.list_markets(
            closed=False,
            liquidity_num_min=float(settings.min_liquidity),
            page_size=settings.max_markets,
            order="liquidityNum",
            ascending=False,
        ).first_page().items

    def short(self, m):
        t = " ".join(str(val(m, x, "") or "") for x in ("question","slug","title")).lower()
        return any(x in t for x in ("5m","5 min","5-minute","15m","15 min","15-minute"))

    def info(self, m):
        o = val(m, "outcomes")
        y = val(val(o, "yes"), "token_id") or val(val(o, "yes"), "position_id")
        n = val(val(o, "no"), "token_id") or val(val(o, "no"), "position_id")
        if not y or not n:
            return None
        tr = val(m, "trading")
        return Market(
            str(val(m, "id", "")),
            str(val(m, "question", "") or val(m, "slug", "")),
            str(y), str(n),
            Decimal(str(val(tr, "minimum_order_size", "0") or "0")),
        )

    def books(self, yid, nid):
        out = []
        for aid in (yid, nid):
            b = self.public.get_order_book(asset_id=aid)
            p = lambda x: Decimal(str(val(x, "price", "0")))
            s = lambda x: Decimal(str(val(x, "size", "0")))
            asks = sorted((p(x), s(x)) for x in b.asks if p(x) > 0 and s(x) > 0)
            bids = sorted((p(x), s(x)) for x in b.bids if p(x) > 0 and s(x) > 0)
            out.append(Book(
                aid,
                asks[0][0] if asks else None,
                asks[0][1] if asks else Decimal(0),
                bids[0][0] if bids else None,
                bids[0][1] if bids else Decimal(0),
            ))
        return out

    def buy(self, aid, shares, price):
        if not settings.live_trading:
            return {"status": "PAPER"}
        return self.secure.place_market_order(
            asset_id=aid, side="BUY", amount=shares * price,
            max_price=price, order_type="FAK"
        )
