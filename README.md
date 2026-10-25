# Nifty 50 Breakdown Telegram Scanner

A GitHub Actions Python scanner for Nifty 50 spot (`^NSEI`) using 15-minute candles.

## Conditions
- EMA 10 below EMA 20
- Latest close below session VWAP
- RSI between 30 and 40
- Latest candle closes in the bottom 25% of its range
- Volume greater than 2.5x the 10-candle average when spot volume is available

**Important:** Yahoo Finance may report zero volume for the Nifty spot index. As in the supplied strategy, the volume condition is bypassed when volume is unavailable; the Telegram message explicitly flags this. This means such alerts are not volume-confirmed.

## Setup
1. Upload `main.py`, `requirements.txt`, and `.github/workflows/nifty-scan.yml` to a GitHub repository.
2. Open **Settings → Secrets and variables → Actions**.
3. Add repository secrets named `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
4. Open **Actions → Nifty 50 Breakdown Telegram Scanner** and select **Run workflow** to test it.

The workflow also runs on weekdays every 15 minutes during the configured UTC window (approximately 09:30–15:00 IST). GitHub schedule timing can be delayed, and market holidays are not automatically excluded.

## Risk settings
- Stop-loss: 0.5% above the sell trigger
- Target: 1.5% below the sell trigger

These are alert levels, not a guarantee of execution or profit. Verify the signal before trading. The script does not place orders.
