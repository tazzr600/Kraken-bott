# Kraken Day Trader

A PAPER-first Kraken spot day-trading bot with:

- Kraken market discovery and liquidity filtering
- 1-minute OHLCV analysis with short-horizon forecasting
- ensemble machine-learning signal generation
- technical strategy scoring and confirmation
- one-position-at-a-time execution
- emergency stop-loss and AI-controlled exits
- calendar-day realized P&L risk limits
- calendar-day completed-trade limits
- consecutive-loss circuit breaker
- SQLite trade journal and paper account
- FastAPI dashboard/API
- Railway/Docker deployment
- automated syntax and regression verification

## Safety defaults

The project is intentionally PAPER-first:

- `LIVE_TRADING=false`
- `DRY_RUN=true`
- runtime mode always starts as `PAPER`
- LIVE orders require configuration, authentication, and explicit runtime confirmation

PAPER mode does not require private Kraken API credentials. Public market data is enough to run the scanner and simulated trading.

**This is not a guaranteed-profit system.** Machine learning is a statistical signal generator, not a prediction guarantee.

## Setup

Copy `.env.example` to `.env` for local use. Keep real API credentials out of Git.

For PAPER mode, API credentials are optional.

For Railway, configure environment variables in Railway's secret/environment settings rather than committing them.

If LIVE trading is ever enabled, use a Kraken API key with only the permissions actually required for trading/read access. Never enable withdrawal permissions.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `/`.

Health endpoint: `/health`

## Verification

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

Run both checks locally before deployment. CI is optional and should not be assumed to have run unless a GitHub Actions workflow is present and reports a successful run.

## Risk accounting

Daily loss and daily trade limits use completed SELL trades from the current local calendar day. Rolling 24-hour statistics remain available for dashboard compatibility but are not used for the daily risk gate.

After every completed SELL, the bot synchronizes the persisted trade count, last-trade timestamp, and consecutive-loss state.

## Scanner status

The scanner status exposes:

- markets discovered
- maximum symbols sent to ML
- allowed quote currencies
- liquidity/spread configuration
- last refresh and scan errors

## Deployment

Railway uses the included Dockerfile and `railway.toml`. The HTTP health check is `/health`.

Keep `LIVE_TRADING=false` and `DRY_RUN=true` until paper performance has been independently validated.


## Trading behavior

The bot is designed for fast intraday **Kraken crypto** trading. It uses 1-minute market data by default, refreshes its decision loop frequently, scans a broader liquid-market universe, enters only when the estimated edge is positive after modeled execution costs, and continuously re-evaluates an open position. If the AI direction turns bearish, the expected net edge becomes non-positive, or a bearish reversal is detected, the bot can exit and immediately look for the next qualifying market. Take-profit, stop-loss, and maximum-hold backstops remain enabled.

Kraken provides the real-time crypto markets used by this bot; this repository does not treat U.S. stocks as Kraken-tradable instruments. Stock trading would require a separate brokerage/data integration.
