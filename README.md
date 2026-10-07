# Polymarket HFT Pair Engine

The legacy Kraken bot is removed. The repository is now organized around the Polymarket short-duration Up/Down workflow from the supplied research.

## Exact trading flow implemented in PAPER mode

1. Discover live crypto Up/Down markets from Polymarket Gamma and read both CLOB order books.
2. Build fair value from underlying spot vs opening price, time remaining, momentum, movement speed, realized volatility, and spot/perp basis.
3. Compare model fair value against executable Up/Down asks and enter only when modeled edge clears the configured threshold.
4. Continue adding to the primary side while the signal remains valid and inventory limits allow it.
5. When the signal reverses, accumulate the opposite side to reduce unpaired directional exposure.
6. Track equal Up/Down shares as paired inventory. Paired cost is UP_VWAP + DOWN_VWAP. When paired cost is below $1, the matched component has positive structural settlement value before fees and execution effects.
7. Recalculate continuously with daily spend, per-market spend, minimum book size, cooldown, and consecutive-failure controls.

## Dashboard

The dashboard is a dark trading-terminal layout matching the supplied reference direction: P&L/status ribbon, equity graph, fair-value monitor, live order book, central market-flow visualization, inventory builder, opportunity matrix, and execution log.

## Safety

This implementation is intentionally PAPER-only. It does not submit live orders or claim guaranteed profit. Polymarket crypto markets can resolve using Chainlink/TWAP data, while the model uses external spot/perp data as a trading-signal proxy.

Railway starts uvicorn app.main:app --host 0.0.0.0 --port $PORT.

Health: /health

Control:
- POST /api/control/start
- POST /api/control/stop

Data:
- /api/status
- /api/markets
- /api/positions
- /api/orders
- /api/events
- /api/equity
