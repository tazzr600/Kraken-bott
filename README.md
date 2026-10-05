# JEV Crypto Day Trader

JEV is a Kraken spot-crypto day-trading prototype with:

- time-series machine learning (HistGradientBoosting)
- BTC/ETH/SOL scanning
- 5-minute candles
- probability + expected-move scoring
- one best trade selection
- stop loss / take profit / max hold time
- daily loss and trade-count limits
- SQLite trade journal
- FastAPI dashboard
- Railway deployment support

## Important

This is not a guaranteed-profit system. The ML model is a statistical signal generator, not a guarantee of future returns.

The project defaults to:

- LIVE_TRADING=false
- DRY_RUN=true

Do not enable live trading until the paper results have been validated.

## Railway variables

Add:

KRAKEN_API_KEY
KRAKEN_API_SECRET

Then add the risk/config variables from `.env.example`.

Keep:

LIVE_TRADING=false
DRY_RUN=true
AUTONOMOUS_MODE=true

## Kraken API permissions

The API key should be limited to trading/read permissions.

Do NOT enable withdrawal permissions.

## Local run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `/`.

## How the model works

The model uses rolling technical features:

- returns
- EMA distance
- RSI
- ATR/range
- volume z-score
- trend
- volatility

It trains on older candles and tests on a later time segment to reduce look-ahead leakage. The live scanner then ranks symbols by estimated upside probability and expected move after a cost/slippage buffer.

This is intentionally conservative: the highest-ranked market is only eligible if the model clears the accuracy, probability, expected-move and cost filters.
