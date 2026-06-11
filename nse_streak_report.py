import os
import sys
import smtplib
from datetime import datetime, date, timedelta
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
    Checks NSE holidays and weekends using NSE market calendar.
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
    The sheet must be publicly readable or published/exportable.
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
        t for t in tickers
        if t and t.endswith(".NS")
    ]

    return sorted(set(tickers))


def get_price_history(ticker: str):
    """
    Downloads recent daily close prices.
    Yahoo Finance already ignores weekends and holidays.
    """
    data = yf.download(
        ticker,
        period="20d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if data.empty or "Close" not in data.columns:
        return None

    closes = data["Close"].dropna()

    if len(closes) < 4:
        return None

    return closes.squeeze()


def detect_latest_streak(closes):
    """
    Detects whether the latest movement is UP or DOWN for 3+ consecutive
    trading sessions.

    Example:
    Day 1: 100
    Day 2: 103 UP
    Day 3: 105 UP
    Day 4: 108 UP

    This is a 3-day UP streak.
    """
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

    return {
        "direction": latest_move,
        "days": streak_days,
        "start_price": start_price,
        "end_price": end_price,
        "pct_change": pct_change,
        "latest_date": closes.index[-1].strftime("%Y-%m-%d"),
    }


def build_report():
    tickers = read_nse_tickers_from_sheet()

    report_rows = []
    failed_tickers = []

    for ticker in tickers:
        try:
            closes = get_price_history(ticker)

            if closes is None:
                failed_tickers.append(ticker)
                continue

            streak = detect_latest_streak(closes)

            if streak:
                report_rows.append({
                    "ticker": ticker,
                    **streak
                })

        except Exception as e:
            failed_tickers.append(f"{ticker}: {str(e)}")

    return report_rows, failed_tickers


def make_email_body(report_rows, failed_tickers):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = []
    body.append("NSE 3-Day Stock Movement Report")
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
            body.append(f"Percentage change: {row['pct_change']:.2f}%")
            body.append(f"From price: {row['start_price']:.2f}")
            body.append(f"To price: {row['end_price']:.2f}")
            body.append(f"Latest trading date: {row['latest_date']}")
            body.append("-" * 40)

    body.append("")
    body.append("Note: Saturdays, Sundays and NSE holidays are ignored automatically.")

    if failed_tickers:
        body.append("")
        body.append("Tickers that could not be processed:")
        for item in failed_tickers:
            body.append(f"- {item}")

    return "\n".join(body)


def send_email(subject, body):
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
        subject="NSE 3-Day Stock Movement Report",
        body=body
    )

    print("Email sent successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)