# Intraday Short Scanner — Nifty 500

A GitHub Actions scanner that checks Nifty 500 constituents using completed 15-minute Yahoo Finance candles and sends Telegram alerts when all four configured conditions match.

## Conditions

- 10 EMA < 20 EMA
- Last completed candle close < session VWAP
- RSI(14) between 32 and 45
- Volume > 1.5 × the rolling 10-candle average

## Alert levels

- Sell trigger: latest completed candle low minus ₹0.50
- Stop-loss: 0.75% above the planned sell trigger
- Target: 1.5% below the planned sell trigger
- Planned risk/reward: 1:2 before costs and slippage

These are setup alerts, not trade execution. Confirm the live price and whether the sell trigger has actually broken before entering a trade.

## Repository files

```text
main.py
requirements.txt
README.md
.github/workflows/scanner.yml
```

## Setup

1. Upload these files to your GitHub repository, preserving the `.github/workflows/` path.
2. Open **Settings → Secrets and variables → Actions**.
3. Add repository secrets named `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
4. Open **Actions → Market Scanner Runner → Run workflow** to test manually.
5. Check the workflow logs and Telegram for the scan summary.

## Universe behavior

The script fetches the current Nifty 500 constituent CSV at runtime and checks that the result contains at least 400 tickers. If it cannot retrieve the full universe, it fails visibly rather than silently scanning only 10 stocks. To explicitly use the small built-in test list, set the environment variable `USE_FALLBACK_UNIVERSE=true`; this is not recommended for a full-universe scan.

## Schedule

GitHub Actions uses UTC. The workflow schedules every 15 minutes from 9:30 AM to 2:15 PM IST, plus a 2:30 PM IST run, Monday to Friday. GitHub may delay scheduled workflows, and the schedule is not a real-time execution guarantee.

## Notes and limitations

- Yahoo Finance intraday data can be delayed, missing, throttled, or unavailable for some symbols. Logs report individual symbol errors.
- The newest downloaded candle is dropped conservatively to reduce the chance of using an incomplete candle. This can mean the signal is based on the preceding completed candle.
- This project does not place orders and does not guarantee profitability. Short selling has market, execution, and regulatory risks.
- GitHub-hosted cron jobs are not exchange-grade real-time infrastructure. A run may take several minutes to scan the whole universe, and scheduled runs can overlap or be delayed.
- Telegram alerts are sent on each workflow run for matching signals; persistent cross-run duplicate suppression is not implemented.
