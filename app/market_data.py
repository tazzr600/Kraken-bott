import json, math, time, urllib.parse, urllib.request
from decimal import Decimal
from .models import Book, Market, Underlying

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
BINANCE = "https://api.binance.com"
FUTURES = "https://fapi.binance.com"

class Http:
    def get_json(self, url, timeout=5):
        req = urllib.request.Request(url, headers={"User-Agent": "polymarket-paper-engine/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

def parse_json(value, default):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return value if value is not None else default

def parse_ts(value):
    if not value:
        return 0
    try:
        from datetime import datetime
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
    except Exception:
        return 0

def slug_ts(slug):
    try:
        return int(str(slug).rsplit("-", 1)[-1])
    except Exception:
        return 0

class PolymarketData:
    def __init__(self, settings, log):
        self.s = settings
        self.log = log
        self.http = Http()
        self.market_cache = []
        self.market_cache_ts = 0
        self.book_cache = {}
        self.book_cache_ts = {}
        self.underlying_cache = {}
        self.underlying_cache_ts = {}

    def discover(self):
        now = int(time.time())
        if self.market_cache and now - self.market_cache_ts < self.s.market_refresh_seconds:
            return self.market_cache
        found = []
        for series in self.s.series:
            try:
                q = urllib.parse.urlencode({"series_slug": series, "closed": "false", "limit": 500, "order": "endDate", "ascending": "true"})
                events = self.http.get_json(f"{GAMMA}/events?{q}")
                for event in events or []:
                    event_start = parse_ts(event.get("eventStartTime")) or parse_ts(event.get("startTime")) or parse_ts(event.get("startDate")) or slug_ts(event.get("slug", ""))
                    event_end = parse_ts(event.get("endDate")) or parse_ts(event.get("endTime"))
                    for raw in event.get("markets", []) or []:
                        if raw.get("closed") or raw.get("active") is False or raw.get("acceptingOrders") is False:
                            continue
                        if float(raw.get("liquidityNum") or raw.get("liquidity") or 0) < float(self.s.min_liquidity):
                            continue
                        slug = str(raw.get("slug") or event.get("slug") or "")
                        start = parse_ts(raw.get("eventStartTime")) or event_start or parse_ts(raw.get("startDate")) or slug_ts(slug)
                        end = parse_ts(raw.get("endDate")) or event_end
                        if not (start and end and start <= now < end):
                            continue
                        tokens = parse_json(raw.get("clobTokenIds"), [])
                        outcomes = [str(x).lower() for x in parse_json(raw.get("outcomes"), ["Up", "Down"])]
                        if len(tokens) < 2:
                            continue
                        up_index = outcomes.index("up") if "up" in outcomes else 0
                        down_index = outcomes.index("down") if "down" in outcomes else 1
                        found.append(Market(
                            id=str(raw.get("id") or raw.get("conditionId") or slug),
                            condition_id=str(raw.get("conditionId") or ""),
                            slug=slug,
                            title=str(raw.get("question") or event.get("title") or ""),
                            series=series,
                            start_ts=start,
                            end_ts=end,
                            yes_token=str(tokens[up_index]),
                            no_token=str(tokens[down_index]),
                            min_size=Decimal(str(raw.get("orderMinSize") or "5")),
                            tick_size=Decimal(str(raw.get("orderPriceMinTickSize") or "0.01")),
                            liquidity=Decimal(str(raw.get("liquidityNum") or raw.get("liquidity") or "0")),
                        ))
            except Exception as exc:
                self.log("WARN", "DISCOVERY_ERROR", f"{series}: {exc}")
        self.market_cache = found[:self.s.max_markets]
        self.market_cache_ts = now
        return self.market_cache

    def book(self, token):
        now = time.time()
        if token in self.book_cache and now - self.book_cache_ts.get(token, 0) < 0.65:
            return self.book_cache[token]
        raw = self.http.get_json(f"{CLOB}/book?{urllib.parse.urlencode({'token_id': token})}")
        asks = sorted([(Decimal(str(x["price"])), Decimal(str(x["size"]))) for x in raw.get("asks", [])], key=lambda x: x[0])
        bids = sorted([(Decimal(str(x["price"])), Decimal(str(x["size"]))) for x in raw.get("bids", [])], key=lambda x: x[0], reverse=True)
        book = Book(
            token_id=token,
            bid=bids[0][0] if bids else None,
            bid_size=bids[0][1] if bids else Decimal("0"),
            ask=asks[0][0] if asks else None,
            ask_size=asks[0][1] if asks else Decimal("0"),
            midpoint=((asks[0][0] + bids[0][0]) / 2) if asks and bids else None,
            last=Decimal(str(raw.get("last_trade_price"))) if raw.get("last_trade_price") else None,
        )
        self.book_cache[token] = book
        self.book_cache_ts[token] = now
        return book

    def books(self, market):
        return self.book(market.yes_token), self.book(market.no_token)

    def symbol(self, series):
        s = series.lower()
        if s.startswith("btc"): return "BTCUSDT"
        if s.startswith("eth"): return "ETHUSDT"
        if s.startswith("sol"): return "SOLUSDT"
        raise RuntimeError("unsupported underlying series")

    def ticker(self, base, symbol, futures=False):
        path = "/fapi/v1/ticker/bookTicker" if futures else "/api/v3/ticker/bookTicker"
        raw = self.http.get_json(f"{base}{path}?{urllib.parse.urlencode({'symbol': symbol})}")
        bid, ask = float(raw.get("bidPrice", 0)), float(raw.get("askPrice", 0))
        return (bid + ask) / 2 if bid and ask else 0.0

    def klines(self, symbol, start=None, limit=60):
        params = {"symbol": symbol, "interval": "1m", "limit": limit}
        if start:
            params["startTime"] = int(start)
        return self.http.get_json(f"{BINANCE}/api/v3/klines?{urllib.parse.urlencode(params)}")

    def underlying(self, market):
        symbol = self.symbol(market.series)
        now = time.time()
        if symbol in self.underlying_cache and now - self.underlying_cache_ts.get(symbol, 0) < 0.8:
            return self.underlying_cache[symbol]
        spot = self.ticker(BINANCE, symbol)
        perp = self.ticker(FUTURES, symbol, True)
        basis = perp / spot - 1 if spot and perp else 0.0
        start_rows = self.klines(symbol, market.start_ts * 1000, 2)
        start_price = float(start_rows[0][1]) if start_rows else spot
        rows = self.klines(symbol, limit=60)
        closes = [float(x[4]) for x in rows if len(x) > 4]
        returns = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes)) if closes[i-1] > 0]
        if len(returns) > 2:
            mean = sum(returns) / len(returns)
            volatility = math.sqrt(sum((x - mean) ** 2 for x in returns) / max(1, len(returns) - 1))
        else:
            volatility = self.s.volatility_floor
        volatility = min(max(volatility, self.s.volatility_floor), self.s.volatility_cap)
        current = math.log(spot / start_price) if spot and start_price else 0.0
        m60 = math.log(closes[-1] / closes[-2]) if len(closes) > 1 else current
        m30 = (m60 + math.log(closes[-2] / closes[-3])) / 2 if len(closes) > 2 else m60
        speed = abs(current) / max(time.time() - market.start_ts, 1)
        value = Underlying(symbol, spot, perp, basis, start_price, volatility, current, m30, m60, speed, len(closes))
        self.underlying_cache[symbol] = value
        self.underlying_cache_ts[symbol] = now
        return value
