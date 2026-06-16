import os
import sys
import smtplib
from datetime import datetime, date
from email.mime.text import MIMEText

import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal


SHEET_ID = "1_lQwmuBIzjg3kmc43sxML9vFt9qjyvOLmBGYW12LDLM"
GID = "1600436104"
TICKER_COLUMN = "NSE Ticker"

EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")


def is_nse_trading_day(today: date) -> bool:
    """
    Checks whether today is an NSE trading day.
    It automatically excludes Saturdays, Sundays and NSE holidays.
    """
    nse = mcal.get_calendar("NSE")

    schedule = nse.schedule(
        start_date=today.strftime("%Y-%m-%d"),
        end_date=today.strftime("%Y-%m-%d")
    )

    return not schedule.empty


def read_nse_tickers_from_sheet() -> list[str]:
    """
    Reads NSE tickers from Google Sheet column named 'NSE Ticker'.

    Expected format in the sheet:
    RELIANCE.NS
    TCS.NS
    HDFCBANK.NS

    The Google Sheet must be publicly readable or exportable as CSV.
    """
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export"
        f"?format=csv&gid={GID}"
    )

    df = pd.read_csv(csv_url)

    if TICKER_COLUMN not in df.columns:
        raise ValueError(f"Column '{TICKER_COLUMN}' not found in Google Sheet.")

    tickers = (
        df[TICKER_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    tickers = [
        ticker for ticker in tickers
        if ticker and ticker.endswith(".NS")
    ]

    return sorted(set(tickers))


def get_price_volume_history(ticker: str):
    """
    Downloads Close and Volume data from Yahoo Finance.

    We use 9 months of data because we need enough previous trading days
    to calculate 80-day average volume.
    """
    data = yf.download(
        ticker,
        period="9mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if data.empty:
        return None

    # yfinance may sometimes return MultiIndex columns.
    # This flattens them safely for single ticker downloads.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_columns = {"Close", "Volume"}

    if not required_columns.issubset(set(data.columns)):
        return None

    df = data[["Close", "Volume"]].copy()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df = df.dropna(subset=["Close", "Volume"])

    # Need enough data for previous 80-day volume average.
    if len(df) < 85:
        return None

    return df


def add_volume_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds previous 30, 60 and 80 trading-day average volume.

    shift(1) is important.
    It means the current day's volume is excluded from its own average.

    Example:
    For 2026-06-16, avg_vol_30 means average volume of the previous
    30 trading days, not including 2026-06-16.
    """
    df = df.copy()

    df["avg_vol_30"] = df["Volume"].rolling(window=30).mean().shift(1)
    df["avg_vol_60"] = df["Volume"].rolling(window=60).mean().shift(1)
    df["avg_vol_80"] = df["Volume"].rolling(window=80).mean().shift(1)

    return df


def calculate_volume_ratio(volume: float, avg_volume: float):
    """
    Calculates volume ratio.

    Example:
    current volume = 20,00,000
    average volume = 10,00,000
    ratio = 2.00x
    """
    if pd.isna(avg_volume) or avg_volume <= 0:
        return None

    return float(volume / avg_volume)


def detect_latest_streak(df: pd.DataFrame):
    """
    Detects whether the latest movement is UP or DOWN for 3 or more
    consecutive trading sessions.

    Also calculates volume ratios for each streak day.
    """
    closes = df["Close"].dropna()

    if len(closes) < 4:
        return None

    moves = []

    for i in range(1, len(closes)):
        prev_price = float(closes.iloc[i - 1])
        curr_price = float(closes.iloc[i])

        if curr_price > prev_price:
            moves.append("UP")
        elif curr_price < prev_price:
            moves.append("DOWN")
        else:
            moves.append("FLAT")

    if not moves:
        return None

    latest_move = moves[-1]

    if latest_move == "FLAT":
        return None

    streak_days = 1

    for i in range(len(moves) - 2, -1, -1):
        if moves[i] == latest_move:
            streak_days += 1
        else:
            break

    if streak_days < 3:
        return None

    start_price = float(closes.iloc[-streak_days - 1])
    end_price = float(closes.iloc[-1])

    pct_change = ((end_price - start_price) / start_price) * 100

    # These are the actual trading days that form the streak.
    # Example:
    # Close prices: 100, 103, 105, 108
    # Moves: UP, UP, UP
    # Streak days are the last 3 dates.
    streak_dates = closes.index[-streak_days:]

    volume_details = []

    for streak_date in streak_dates:
        row = df.loc[streak_date]

        volume = float(row["Volume"])

        vol_30_ratio = calculate_volume_ratio(volume, row["avg_vol_30"])
        vol_60_ratio = calculate_volume_ratio(volume, row["avg_vol_60"])
        vol_80_ratio = calculate_volume_ratio(volume, row["avg_vol_80"])

        volume_details.append({
            "date": streak_date.strftime("%Y-%m-%d"),
            "close": float(row["Close"]),
            "volume_30_day_ratio": vol_30_ratio,
            "volume_60_day_ratio": vol_60_ratio,
            "volume_80_day_ratio": vol_80_ratio,
        })

    return {
        "direction": latest_move,
        "days": streak_days,
        "start_price": start_price,
        "end_price": end_price,
        "pct_change": pct_change,
        "latest_date": closes.index[-1].strftime("%Y-%m-%d"),
        "volume_details": volume_details,
    }


def build_report():
    """
    Builds report rows for all tickers.

    report_rows:
        Stocks that have 3+ day UP/DOWN streak.

    failed_tickers:
        Tickers where Yahoo Finance data could not be downloaded or processed.
    """
    tickers = read_nse_tickers_from_sheet()

    report_rows = []
    failed_tickers = []

    for ticker in tickers:
        try:
            df = get_price_volume_history(ticker)

            if df is None:
                failed_tickers.append(ticker)
                continue

            df = add_volume_averages(df)

            streak = detect_latest_streak(df)

            if streak:
                report_rows.append({
                    "ticker": ticker,
                    **streak
                })

        except Exception as e:
            failed_tickers.append(f"{ticker}: {str(e)}")

    return report_rows, failed_tickers


def format_ratio(value):
    """
    Formats volume ratio cleanly.
    """
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.2f}x"


def make_email_body(report_rows, failed_tickers):
    """
    Creates plain text email body.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = []
    body.append("NSE 3-Day Stock Movement + Volume Ratio Report")
    body.append(f"Generated at: {now}")
    body.append("")

    if not report_rows:
        body.append("No stock has increased or decreased for 3 or more consecutive trading days.")
    else:
        body.append("Stocks with 3 or more consecutive trading-day movement:")
        body.append("")

        for row in report_rows:
            direction_word = "INCREASED" if row["direction"] == "UP" else "DECREASED"

            body.append(f"Stock: {row['ticker']}")
            body.append(f"Direction: {direction_word}")
            body.append(f"Consecutive trading days: {row['days']}")
            body.append(f"Price change over streak: {row['pct_change']:.2f}%")
            body.append(f"From price: {row['start_price']:.2f}")
            body.append(f"To price: {row['end_price']:.2f}")
            body.append(f"Latest trading date: {row['latest_date']}")
            body.append("")
            body.append("Volume ratio for each streak day:")
            body.append("")

            for vol in row["volume_details"]:
                body.append(f"Date: {vol['date']}")
                body.append(f"Close: {vol['close']:.2f}")
                body.append(f"Volume / previous 30-day avg: {format_ratio(vol['volume_30_day_ratio'])}")
                body.append(f"Volume / previous 60-day avg: {format_ratio(vol['volume_60_day_ratio'])}")
                body.append(f"Volume / previous 80-day avg: {format_ratio(vol['volume_80_day_ratio'])}")
                body.append("")

            body.append("-" * 50)

    body.append("")
    body.append("Note:")
    body.append("- Saturdays, Sundays and NSE holidays are ignored automatically.")
    body.append("- Volume ratios compare that day's volume with previous 30/60/80 trading-day average volume.")
    body.append("- The current day's volume is excluded from its own average.")
    body.append("- 1.00x means normal volume, 2.00x means double average volume, 0.50x means half average volume.")
    body.append("- Price and volume data are taken from Yahoo Finance.")

    if failed_tickers:
        body.append("")
        body.append("Tickers that could not be processed:")
        for item in failed_tickers:
            body.append(f"- {item}")

    return "\n".join(body)


def send_email(subject, body):
    """
    Sends email using Gmail SMTP.

    Required GitHub secrets / environment variables:
    EMAIL_FROM
    EMAIL_APP_PASSWORD
    EMAIL_TO
    """
    if not EMAIL_FROM:
        raise ValueError("EMAIL_FROM secret is missing.")

    if not EMAIL_APP_PASSWORD:
        raise ValueError("EMAIL_APP_PASSWORD secret is missing.")

    if not EMAIL_TO:
        raise ValueError("EMAIL_TO secret is missing.")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        server.send_message(msg)


def main():
    today = date.today()

    if not is_nse_trading_day(today):
        print(f"{today} is not an NSE trading day. Skipping report.")
        return

    report_rows, failed_tickers = build_report()

    body = make_email_body(report_rows, failed_tickers)

    print(body)

    send_email(
        subject="NSE 3-Day Stock Movement + Volume Ratio Report",
        body=body
    )

    print("Email sent successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
