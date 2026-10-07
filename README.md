# Polymarket Two-Sided Pair Bot — Railway Ready

This rebuild implements the paired-inventory concept for binary Polymarket markets.

For one binary market:
- pair_cost = YES best ask + NO best ask
- gross_edge = 1 - pair_cost
- net_edge = gross_edge - fee_buffer - slippage_buffer

The bot only attempts equal-size YES/NO pairs when the net edge passes the configured threshold.

Example: 0.1658 + 0.7810 = 0.9468, giving 0.0532 gross edge per matched pair before fees/slippage.

It includes a live Polymarket market scanner, liquidity and short-duration filters, YES/NO order-book reads, pair pricing and sizing, daily spend limits, FAK execution in live mode, SQLite logging, FastAPI dashboard, Railway configuration, and PAPER mode by default.

Keep LIVE_TRADING=false while validating. Credentials belong only in Railway environment variables.

Important: the two legs are NOT atomic. A YES fill can happen while NO fails. This is not guaranteed profit. Production hardening must reconcile actual account fills/positions after every live attempt before treating inventory as paired.
