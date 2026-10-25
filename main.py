import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
INDEX_TICKER = "^NSEI"
TARGET_PCT = 0.015
STOP_LOSS_PCT = 0.005


def send_telegram_msg(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing; alert was not sent.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"Telegram API error: HTTP {response.status_code}: {response.text[:300]}")
            return False
        result = response.json()
        if not result.get("ok", False):
            print(f"Telegram API rejected message: {result}")
            return False
        return True
    except requests.RequestException as exc:
        print(f"Telegram request error: {exc}")
        return False


def calculate_session_vwap(df):
    """Calculate VWAP separately for each trading session."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    volume = df["Volume"].fillna(0)
    session_key = pd.Index(df.index.date)
    pv = typical_price * volume
    cumulative_pv = pv.groupby(session_key).cumsum()
    cumulative_volume = volume.groupby(session_key).cumsum()
    return cumulative_pv / cumulative_volume.replace(0, float("nan"))


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def run_scan():
    print("Starting Nifty 50 15-minute breakdown scan...")
    try:
        df = yf.download(
            INDEX_TICKER, period="5d", interval="15m",
            progress=False, auto_adjust=False, threads=False
        )
        if df.empty:
            print("No index data returned by Yahoo Finance.")
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["High", "Low", "Close", "Volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"Required columns missing: {missing}")
            return

        df = df[required].copy().dropna(subset=["High", "Low", "Close"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if len(df) < 25:
            print(f"Insufficient index data: only {len(df)} candles.")
            return

        # Avoid using a candle that is still forming during market hours.
        now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        last_time = df.index[-1]
        if last_time.tzinfo is None:
            last_time = last_time.tz_localize("Asia/Kolkata")
        else:
            last_time = last_time.tz_convert("Asia/Kolkata")
        if (now_ist.date() == last_time.date()
                and now_ist.hour * 60 + now_ist.minute < 15 + last_time.hour * 60 + last_time.minute
                and now_ist.hour < 16):
            df = df.iloc[:-1]
        if len(df) < 25:
            print("Insufficient completed candles after excluding the forming candle.")
            return

        close = df["Close"].astype(float)
        volume = df["Volume"].fillna(0).astype(float)
        ema10 = close.ewm(span=10, adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        vwap = calculate_session_vwap(df)
        rsi = calculate_rsi(close, 14)
        vol_avg10 = volume.rolling(10, min_periods=10).mean()

        latest = df.iloc[-1]
        latest_close = float(close.iloc[-1])
        latest_high = float(latest["High"])
        latest_low = float(latest["Low"])
        latest_ema10 = float(ema10.iloc[-1])
        latest_ema20 = float(ema20.iloc[-1])
        latest_vwap = float(vwap.iloc[-1]) if pd.notna(vwap.iloc[-1]) else None
        latest_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None
        latest_vol = float(volume.iloc[-1])
        latest_vol_avg = float(vol_avg10.iloc[-1]) if pd.notna(vol_avg10.iloc[-1]) else 0.0

        if latest_vwap is None or latest_rsi is None:
            print("Indicator values are not ready; no signal evaluated.")
            return

        candle_range = latest_high - latest_low
        solid_red_candle = candle_range > 0 and latest_close <= latest_low + candle_range * 0.25
        downtrend = latest_ema10 < latest_ema20 and latest_close < latest_vwap
        bearish_rsi = 30 <= latest_rsi <= 40

        # Nifty spot often reports zero volume. In that case volume cannot be
        # verified, so retain the user's fallback but clearly flag it in the alert.
        if latest_vol > 0 and latest_vol_avg > 0:
            has_volume_surge = latest_vol > 2.5 * latest_vol_avg
            vol_ratio = round(latest_vol / latest_vol_avg, 1)
            volume_text = f"{vol_ratio}x 10-candle average"
        else:
            has_volume_surge = True
            vol_ratio = None
            volume_text = "Unavailable for spot index (volume filter bypassed)"

        print(
            f"Latest candle: {df.index[-1]} | close={latest_close:.2f} | "
            f"EMA10={latest_ema10:.2f} | EMA20={latest_ema20:.2f} | "
            f"VWAP={latest_vwap:.2f} | RSI={latest_rsi:.1f} | volume={volume_text}"
        )

        if downtrend and bearish_rsi and has_volume_surge and solid_red_candle:
            sell_below = round(latest_low - 2.0, 2)
            # Prices are calculated from the trigger entry, not the signal candle close.
            sl_price = round(sell_below * (1 + STOP_LOSS_PCT), 2)
            target_price = round(sell_below * (1 - TARGET_PCT), 2)
            candle_time = df.index[-1]
            msg = (
                "🚨 *NIFTY 50 BEARISH BREAKDOWN ALERT* 🚨\\n\\n"
                "📌 *Instrument:* `NIFTY 50 INDEX SPOT`\\n"
                f"🕒 *Signal candle:* `{candle_time}`\\n"
                f"💰 *Signal Close:* ₹{latest_close:.2f}\\n"
                f"📉 *Sell trigger:* ₹{sell_below:.2f}\\n"
                f"🛑 *Stop-loss (0.5%):* ₹{sl_price:.2f}\\n"
                f"🎯 *Target (1.5%):* ₹{target_price:.2f}\\n\\n"
                f"📊 *RSI (15m):* {latest_rsi:.1f}\\n"
                f"📉 *VWAP:* ₹{latest_vwap:.2f}\\n"
                f"📉 *EMA 10 / EMA 20:* {latest_ema10:.2f} / {latest_ema20:.2f}\\n"
                f"🔥 *Volume:* {volume_text}\\n\\n"
                "⚠️ Alert only; verify the trigger and risk before trading. "
                "This script does not place orders."
            )
            sent = send_telegram_msg(msg)
            print("Telegram alert sent." if sent else "Signal found, but Telegram alert was not confirmed.")
        else:
            print(
                "No setup: "
                f"downtrend={downtrend}, RSI-zone={bearish_rsi}, "
                f"volume-surge={has_volume_surge}, strong-red-candle={solid_red_candle}"
            )

    except Exception as exc:
        print(f"Error during Nifty scan: {exc}")
        raise


if __name__ == "__main__":
    run_scan()
