import os
import io
import time
import logging
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

# GitHub repository secrets: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

UNIVERSE_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv?download=1",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.niftyindices.com/",
}

# Used only if explicitly configured; default behavior requires fetching the full Nifty 500 list.
FALLBACK_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "BHARTIARTL.NS", "ITC.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS",
]
USE_FALLBACK_UNIVERSE = os.getenv("USE_FALLBACK_UNIVERSE", "false").lower() == "true"

INTERVAL = "15m"
PERIOD = "5d"
VOLUME_MULTIPLE = 1.5
TARGET_PCT = 0.015
STOP_PCT = 0.0075
SELL_BUFFER_RUPEES = 0.50

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def send_telegram_msg(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Telegram credentials missing. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to GitHub Actions secrets.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            logging.error("Telegram API returned an unsuccessful response: %s", body)
            return False
        return True
    except (requests.RequestException, ValueError) as exc:
        logging.error("Telegram send failed: %s", exc)
        return False


def get_nifty500_tickers() -> list[str]:
    """Fetch the current Nifty 500 constituents. Do not silently scan a small subset."""
    errors = []
    session = requests.Session()
    session.headers.update(HEADERS)
    for url in UNIVERSE_URLS:
        try:
            response = session.get(url, timeout=20)
            response.raise_for_status()
            table = pd.read_csv(io.BytesIO(response.content))
            normalized = {str(col).strip().lower(): col for col in table.columns}
            symbol_col = next((normalized[c] for c in ("symbol", "ticker") if c in normalized), None)
            if symbol_col is None:
                raise ValueError(f"CSV has no Symbol column. Columns: {list(table.columns)}")
            symbols = table[symbol_col].dropna().astype(str).str.strip()
            tickers = sorted({f"{s}.NS" for s in symbols if s and s.lower() != "nan"})
            if len(tickers) < 400:
                raise ValueError(f"Expected roughly 500 constituents, received only {len(tickers)}")
            logging.info("Loaded %d Nifty 500 tickers.", len(tickers))
            return tickers
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    if USE_FALLBACK_UNIVERSE:
        logging.warning("Nifty 500 fetch failed; using explicitly enabled 10-stock fallback. %s", " | ".join(errors))
        return FALLBACK_STOCKS.copy()
    raise RuntimeError(
        "Could not retrieve the full Nifty 500 universe. No partial scan was run. " + " | ".join(errors)
    )


def normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # A single ticker download may still return a two-level column index.
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected market-data columns: {missing}")
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=needed)


def calculate_session_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate typical-price VWAP separately for each trading date."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    volume = df["Volume"].clip(lower=0)
    if isinstance(df.index, pd.DatetimeIndex):
        dates = pd.Index(df.index.date)
    else:
        dates = pd.Index(pd.to_datetime(df.index).date)
    pv = typical_price * volume
    cumulative_pv = pv.groupby(dates).cumsum()
    cumulative_volume = volume.groupby(dates).cumsum().replace(0, float("nan"))
    return cumulative_pv / cumulative_volume


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    # If there are no losses in the lookback, RSI is 100.
    return rsi.where(avg_loss.ne(0), 100.0)


def get_completed_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the newest bar conservatively because Yahoo may return an in-progress candle."""
    if len(df) < 2:
        return df.iloc[0:0]
    return df.iloc[:-1].copy()


def scan_ticker(ticker: str) -> dict | None:
    raw = yf.download(
        ticker,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=20,
    )
    if raw is None or raw.empty:
        return None
    df = normalize_yfinance_columns(raw)
    df = get_completed_candles(df)
    if len(df) < 25:
        return None

    close = df["Close"]
    volume = df["Volume"]
    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    vwap = calculate_session_vwap(df)
    rsi = calculate_rsi(close, 14)
    vol_avg10 = volume.rolling(window=10, min_periods=10).mean()

    values = [close.iloc[-1], ema10.iloc[-1], ema20.iloc[-1], vwap.iloc[-1], rsi.iloc[-1], volume.iloc[-1], vol_avg10.iloc[-1], df["Low"].iloc[-1]]
    if any(pd.isna(v) for v in values):
        return None
    latest_close, latest_ema10, latest_ema20, latest_vwap, latest_rsi, latest_vol, latest_vol_avg, latest_low = map(float, values)
    if latest_vol_avg <= 0:
        return None

    # Original four conditions retained.
    passed = (
        latest_ema10 < latest_ema20
        and latest_close < latest_vwap
        and 32 <= latest_rsi <= 45
        and latest_vol > VOLUME_MULTIPLE * latest_vol_avg
    )
    if not passed:
        return None

    sell_below = round(latest_low - SELL_BUFFER_RUPEES, 2)
    # Short levels are based on the planned entry trigger, not the signal candle close.
    stop_loss = round(sell_below * (1 + STOP_PCT), 2)
    target = round(sell_below * (1 - TARGET_PCT), 2)
    candle_time = df.index[-1]
    if getattr(candle_time, "tzinfo", None) is not None:
        candle_time = candle_time.tz_convert("Asia/Kolkata")
    candle_label = candle_time.strftime("%d-%b-%Y %I:%M %p IST") if hasattr(candle_time, "strftime") else str(candle_time)

    return {
        "ticker": ticker,
        "close": latest_close,
        "sell_below": sell_below,
        "stop_loss": stop_loss,
        "target": target,
        "rsi": latest_rsi,
        "volume_multiple": latest_vol / latest_vol_avg,
        "vwap": latest_vwap,
        "candle_time": candle_label,
    }


def format_alert(signal: dict) -> str:
    symbol = signal["ticker"].removesuffix(".NS")
    return (
        "🔴 *INTRADAY SHORT SETUP* 🔴\n\n"
        f"📌 *Stock:* `{symbol}`\n"
        f"🕒 *Signal candle:* {signal['candle_time']}\n"
        f"💰 *Signal close:* ₹{signal['close']:.2f}\n"
        f"📉 *Sell trigger:* ₹{signal['sell_below']:.2f}\n"
        f"🛑 *Stop-loss (0.75%):* ₹{signal['stop_loss']:.2f}\n"
        f"🎯 *Target (1.5%):* ₹{signal['target']:.2f}\n\n"
        f"📊 *RSI (15m):* {signal['rsi']:.1f}\n"
        f"🔥 *Volume:* {signal['volume_multiple']:.1f}× 10-candle average\n"
        f"📉 *Session VWAP:* ₹{signal['vwap']:.2f}\n\n"
        "⚠️ Setup alert only; entry triggers only if price trades below the sell trigger. Verify live price and liquidity before trading."
    )


def run_scan() -> None:
    logging.info("Starting 15-minute intraday short scan.")
    try:
        tickers = get_nifty500_tickers()
    except Exception as exc:
        logging.exception("Unable to load stock universe: %s", exc)
        raise

    logging.info("Universe size: %d", len(tickers))
    signals = []
    scanned = 0
    failures = 0
    for i, ticker in enumerate(tickers, start=1):
        try:
            signal = scan_ticker(ticker)
            scanned += 1
            if signal:
                signals.append(signal)
                logging.info("SHORT SETUP: %s", ticker)
        except Exception as exc:
            failures += 1
            logging.warning("Error scanning %s: %s", ticker, exc)
        # Be gentle with Yahoo Finance and reduce burst requests.
        if i % 25 == 0:
            time.sleep(1)

    logging.info("Scan complete: %d/%d stocks checked; %d signals; %d errors.", scanned, len(tickers), len(signals), failures)
    summary = f"📊 *Intraday Short Scan Complete*\nScanned: {scanned}/{len(tickers)}\nSignals: {len(signals)}\nErrors: {failures}"
    if not signals:
        send_telegram_msg(summary + "\nNo stocks matched all four conditions on the latest completed 15-minute candle.")
        return

    # Send a summary and each matching setup separately to avoid Telegram message-size limits.
    send_telegram_msg(summary)
    for signal in signals:
        send_telegram_msg(format_alert(signal))
        time.sleep(0.2)


if __name__ == "__main__":
    run_scan()
