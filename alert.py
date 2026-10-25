import os
import io
import json
import time
import logging
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# Add these in GitHub repository Settings > Secrets and variables > Actions.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

STATE_FILE = Path("signal_state.json")
MAX_REMEMBERED_SIGNALS = 1500

UNIVERSE_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv?download=1",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.niftyindices.com/",
}

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
    # Plain text avoids Markdown parse errors from symbols or unusual stock names.
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, json=payload, timeout=20)
        try:
            body = response.json()
        except ValueError:
            body = {"description": response.text[:500]}
        if response.status_code != 200 or not body.get("ok", False):
            logging.error("Telegram send failed: HTTP %s; response: %s", response.status_code, body)
            return False
        return True
    except requests.RequestException as exc:
        logging.error("Telegram request failed: %s", exc)
        return False


def get_nifty500_tickers() -> list[str]:
    errors = []
    session = requests.Session()
    session.headers.update(HEADERS)
    for url in UNIVERSE_URLS:
        try:
            response = session.get(url, timeout=25)
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
    raise RuntimeError("Could not retrieve the full Nifty 500 universe. No scan was run. " + " | ".join(errors))


def normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
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
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    volume = df["Volume"].clip(lower=0)
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
    return rsi.where(avg_loss.ne(0), 100.0)


def get_completed_candles(df: pd.DataFrame) -> pd.DataFrame:
    # Conservative: drop newest Yahoo candle because it may still be forming.
    if len(df) < 2:
        return df.iloc[0:0]
    return df.iloc[:-1].copy()


def candle_timestamp_key(candle_time) -> str:
    stamp = pd.Timestamp(candle_time)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M")


def scan_ticker(ticker: str) -> dict | None:
    raw = yf.download(
        ticker, period=PERIOD, interval=INTERVAL, auto_adjust=False,
        progress=False, threads=False, timeout=20
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

    values = [
        close.iloc[-1], ema10.iloc[-1], ema20.iloc[-1], vwap.iloc[-1],
        rsi.iloc[-1], volume.iloc[-1], vol_avg10.iloc[-1], df["Low"].iloc[-1]
    ]
    if any(pd.isna(v) for v in values):
        return None

    latest_close, latest_ema10, latest_ema20, latest_vwap, latest_rsi, latest_vol, latest_vol_avg, latest_low = map(float, values)
    if latest_vol_avg <= 0:
        return None

    passed = (
        latest_ema10 < latest_ema20
        and latest_close < latest_vwap
        and 32 <= latest_rsi <= 45
        and latest_vol > VOLUME_MULTIPLE * latest_vol_avg
    )
    if not passed:
        return None

    sell_below = round(latest_low - SELL_BUFFER_RUPEES, 2)
    stop_loss = round(sell_below * (1 + STOP_PCT), 2)
    target = round(sell_below * (1 - TARGET_PCT), 2)
    candle_time = df.index[-1]
    candle_key = candle_timestamp_key(candle_time)
    candle_local = pd.Timestamp(candle_time)
    if candle_local.tzinfo is None:
        candle_local = candle_local.tz_localize("UTC")
    candle_label = candle_local.tz_convert("Asia/Kolkata").strftime("%d-%b-%Y %I:%M %p IST")

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
        "candle_key": candle_key,
    }


def format_alert(signal: dict) -> str:
    symbol = signal["ticker"].removesuffix(".NS")
    return (
        "🔴 INTRADAY SHORT SETUP 🔴\n\n"
        f"Stock: {symbol}\n"
        f"Signal candle: {signal['candle_time']}\n"
        f"Signal close: ₹{signal['close']:.2f}\n"
        f"Sell trigger: ₹{signal['sell_below']:.2f}\n"
        f"Stop-loss (0.75%): ₹{signal['stop_loss']:.2f}\n"
        f"Target (1.5%): ₹{signal['target']:.2f}\n\n"
        f"RSI (15m): {signal['rsi']:.1f}\n"
        f"Volume: {signal['volume_multiple']:.1f}x 10-candle average\n"
        f"Session VWAP: ₹{signal['vwap']:.2f}\n\n"
        "Setup alert only; entry triggers only if price trades below the sell trigger. "
        "Verify live price and liquidity before trading."
    )


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {"sent_signal_keys": []}
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        keys = data.get("sent_signal_keys", [])
        if not isinstance(keys, list):
            raise ValueError("sent_signal_keys is not a list")
        return {"sent_signal_keys": [str(k) for k in keys][-MAX_REMEMBERED_SIGNALS:]}
    except Exception as exc:
        logging.warning("Could not read signal state; starting a fresh state file: %s", exc)
        return {"sent_signal_keys": []}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def run_scan() -> None:
    logging.info("Starting 15-minute intraday short scan.")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variable.")

    tickers = get_nifty500_tickers()
    logging.info("Universe size: %d", len(tickers))
    previous = load_state()
    already_sent = set(previous["sent_signal_keys"])

    signals = []
    scanned = 0
    failures = 0
    for i, ticker in enumerate(tickers, start=1):
        try:
            signal = scan_ticker(ticker)
            scanned += 1
            if signal:
                signal_key = f"{signal['ticker']}|{signal['candle_key']}"
                if signal_key not in already_sent:
                    signal["signal_key"] = signal_key
                    signals.append(signal)
                    logging.info("FRESH SHORT SETUP: %s at %s", ticker, signal["candle_time"])
        except Exception as exc:
            failures += 1
            logging.warning("Error scanning %s: %s", ticker, exc)
        if i % 25 == 0:
            time.sleep(1)

    logging.info(
        "Scan complete: %d/%d stocks checked; %d fresh signals; %d errors.",
        scanned, len(tickers), len(signals), failures
    )

    summary = (
        "📊 Intraday Short Scan Complete\n"
        f"Scanned: {scanned}/{len(tickers)}\n"
        f"Fresh signals: {len(signals)}\n"
        f"Errors: {failures}"
    )
    if not signals:
        # Send a heartbeat summary so you can confirm the workflow ran.
        if not send_telegram_msg(summary + "\nNo new stocks matched all four conditions on a new completed 15-minute candle."):
            raise RuntimeError("Telegram summary message failed; see log above.")
        return

    if not send_telegram_msg(summary):
        raise RuntimeError("Telegram summary message failed; see log above.")

    newly_sent_keys = []
    for signal in signals:
        if send_telegram_msg(format_alert(signal)):
            newly_sent_keys.append(signal["signal_key"])
        else:
            logging.error("Could not send signal alert for %s; it will be eligible to retry next run.", signal["ticker"])
        time.sleep(0.3)

    if newly_sent_keys:
        combined = (previous["sent_signal_keys"] + newly_sent_keys)[-MAX_REMEMBERED_SIGNALS:]
        save_state({"sent_signal_keys": combined})
        logging.info("Saved %d newly sent signal keys.", len(newly_sent_keys))


if __name__ == "__main__":
    run_scan()
