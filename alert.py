import io
import logging
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# NSE archive URL; if NSE blocks GitHub Actions, the script reports a clear error.
UNIVERSE_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.niftyindices.com/",
}

INTERVAL = "15m"
PERIOD = "5d"
VOLUME_MULTIPLE = 1.5
TARGET_PCT = 0.015
STOP_PCT = 0.0075
SELL_BUFFER_RUPEES = 0.50
MAX_SIGNALS_PER_RUN = 15
SLEEP_EVERY = 25
SLEEP_SECONDS = 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def send_telegram_msg(message: str) -> bool:
    """Send a Telegram message and print a safe, useful diagnostic on failure."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error(
            "Telegram credentials missing. Add TELEGRAM_BOT_TOKEN and "
            "TELEGRAM_CHAT_ID under repository Settings > Secrets and variables > Actions."
        )
        return False

    endpoint = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(
            endpoint,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=20,
        )
        # Do not print the endpoint or token.
        logging.info("Telegram HTTP status: %s", response.status_code)
        if not response.ok:
            logging.error("Telegram returned HTTP %s: %s", response.status_code, response.text[:500])
            return False
        body = response.json()
        if not body.get("ok"):
            logging.error("Telegram API rejected the message: %s", body)
            return False
        logging.info("Telegram message sent.")
        return True
    except (requests.RequestException, ValueError) as exc:
        logging.error("Telegram send failed: %s: %s", type(exc).__name__, exc)
        return False


def get_nifty500_tickers() -> list[str]:
    errors = []
    with requests.Session() as session:
        session.headers.update(HEADERS)
        for url in UNIVERSE_URLS:
            try:
                response = session.get(url, timeout=25)
                response.raise_for_status()
                table = pd.read_csv(io.BytesIO(response.content))
                normalized = {str(col).strip().lower(): col for col in table.columns}
                symbol_col = next(
                    (normalized[name] for name in ("symbol", "ticker") if name in normalized),
                    None,
                )
                if symbol_col is None:
                    raise ValueError(f"CSV has no Symbol column; received columns: {list(table.columns)}")
                symbols = table[symbol_col].dropna().astype(str).str.strip().str.upper()
                tickers = sorted({
                    symbol if symbol.endswith(".NS") else f"{symbol}.NS"
                    for symbol in symbols
                    if symbol and symbol.lower() != "nan"
                })
                if len(tickers) < 400:
                    raise ValueError(f"Expected about 500 constituents; received {len(tickers)}")
                logging.info("Loaded %d Nifty 500 tickers.", len(tickers))
                return tickers
            except Exception as exc:
                logging.warning("Could not load constituents from %s (%s)", url, exc)
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Could not download the full Nifty 500 list. No partial scan was run. "
        + " | ".join(errors)
    )


def normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # For a single ticker, yfinance can return either level ordering.
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing market-data columns: {missing}")
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=required)


def calculate_session_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    volume = df["Volume"].clip(lower=0)
    dates = pd.Index(pd.to_datetime(df.index).date)
    return (typical * volume).groupby(dates).cumsum() / volume.groupby(dates).cumsum().replace(0, float("nan"))


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
    # Yahoo may include an in-progress 15-minute candle; drop the newest bar.
    return df.iloc[:-1].copy() if len(df) >= 2 else df.iloc[0:0].copy()


def scan_ticker(ticker: str) -> dict | None:
    raw = yf.download(
        ticker, period=PERIOD, interval=INTERVAL,
        auto_adjust=False, progress=False, threads=False, timeout=20,
    )
    if raw is None or raw.empty:
        return None
    df = normalize_yfinance_columns(raw)
    df = get_completed_candles(df)
    if len(df) < 25:
        return None

    close, volume = df["Close"], df["Volume"]
    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    vwap = calculate_session_vwap(df)
    rsi = calculate_rsi(close, 14)
    vol_avg10 = volume.rolling(10, min_periods=10).mean()
    vals = [
        close.iloc[-1], ema10.iloc[-1], ema20.iloc[-1], vwap.iloc[-1],
        rsi.iloc[-1], volume.iloc[-1], vol_avg10.iloc[-1], df["Low"].iloc[-1],
    ]
    if any(pd.isna(value) for value in vals):
        return None
    c, e10, e20, vw, r, vol, avg_vol, low = map(float, vals)
    if avg_vol <= 0:
        return None

    # User's requested intraday SHORT conditions.
    if not (e10 < e20 and c < vw and 32 <= r <= 45 and vol > VOLUME_MULTIPLE * avg_vol):
        return None

    sell_below = round(low - SELL_BUFFER_RUPEES, 2)
    stop_loss = round(sell_below * (1 + STOP_PCT), 2)
    target = round(sell_below * (1 - TARGET_PCT), 2)
    candle_time = df.index[-1]
    try:
        candle_time = candle_time.tz_convert("Asia/Kolkata") if candle_time.tzinfo else candle_time
    except (AttributeError, TypeError):
        pass
    label = candle_time.strftime("%d-%b-%Y %I:%M %p IST") if hasattr(candle_time, "strftime") else str(candle_time)

    return {
        "ticker": ticker, "close": c, "sell_below": sell_below,
        "stop_loss": stop_loss, "target": target, "rsi": r,
        "volume_multiple": vol / avg_vol, "vwap": vw, "candle_time": label,
    }


def format_alert(signal: dict) -> str:
    symbol = signal["ticker"].removesuffix(".NS")
    # Keep Markdown simple to avoid parse errors from unusual symbol text.
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
        "Setup alert only. This is not a recommendation to trade; verify live price, "
        "liquidity, and risk before acting."
    )


def run_scan() -> None:
    started = datetime.now(ZoneInfo("Asia/Kolkata"))
    logging.info("Starting 15-minute intraday short scan at %s", started.isoformat())
    tickers = get_nifty500_tickers()
    signals = []
    scanned = 0
    failures = 0

    for index, ticker in enumerate(tickers, start=1):
        logging.info("Scanning %d/%d: %s", index, len(tickers), ticker)
        try:
            signal = scan_ticker(ticker)
            scanned += 1
            if signal:
                signals.append(signal)
                logging.info("SHORT SETUP: %s", ticker)
        except Exception as exc:
            failures += 1
            logging.warning("Error scanning %s: %s", ticker, exc)
        if index % SLEEP_EVERY == 0:
            time.sleep(SLEEP_SECONDS)

    logging.info(
        "Scan complete: %d/%d stocks checked; %d signals; %d errors.",
        scanned, len(tickers), len(signals), failures,
    )
    summary = (
        "📊 INTRADAY SHORT SCAN COMPLETE\n"
        f"Time: {started:%d-%b-%Y %I:%M %p IST}\n"
        f"Scanned: {scanned}/{len(tickers)}\n"
        f"Signals: {len(signals)}\n"
        f"Data errors: {failures}"
    )
    if not signals:
        if not send_telegram_msg(summary + "\nNo stocks matched all four conditions on the latest completed 15-minute candle."):
            raise RuntimeError("Scan finished, but the Telegram summary could not be sent. Check Telegram HTTP/API log above.")
        return

    summary_sent = send_telegram_msg(summary)
    if not summary_sent:
        raise RuntimeError("Scan finished, but Telegram summary could not be sent.")
    for signal in signals[:MAX_SIGNALS_PER_RUN]:
        if not send_telegram_msg(format_alert(signal)):
            logging.error("Could not send alert for %s; continuing.", signal["ticker"])
        time.sleep(0.2)
    if len(signals) > MAX_SIGNALS_PER_RUN:
        send_telegram_msg(f"Note: {len(signals) - MAX_SIGNALS_PER_RUN} additional matches omitted to keep this run concise.")


if __name__ == "__main__":
    try:
        run_scan()
    except Exception:
        logging.exception("FATAL: stock scan did not complete successfully.")
        raise
