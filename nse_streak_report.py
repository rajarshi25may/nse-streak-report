import os
import sys
import smtplib
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from typing import Any, Optional

import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal


SHEET_ID = "1_lQwmuBIzjg3kmc43sxML9vFt9qjyvOLmBGYW12LDLM"
GID = "1600436104"
TICKER_COLUMN = "NSE Ticker"

EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")


# ============================================================
# TRADING DAY AND SHEET INPUT
# ============================================================


def is_nse_trading_day(today: date) -> bool:
    """
    Returns True when today is an NSE trading day.
    Saturdays, Sundays and NSE holidays are excluded automatically.
    """
    nse = mcal.get_calendar("NSE")

    schedule = nse.schedule(
        start_date=today.strftime("%Y-%m-%d"),
        end_date=today.strftime("%Y-%m-%d"),
    )

    return not schedule.empty


def read_nse_tickers_from_sheet() -> list[str]:
    """
    Reads unique NSE tickers from the Google Sheet column named 'NSE Ticker'.

    Expected examples:
        RELIANCE.NS
        TCS.NS
        HDFCBANK.NS

    The Google Sheet must be publicly readable/exportable as CSV.
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

    valid_tickers = [
        ticker
        for ticker in tickers
        if ticker and ticker.endswith(".NS")
    ]

    return sorted(set(valid_tickers))


# ============================================================
# PRICE, VOLUME AND STREAK ANALYSIS
# ============================================================


def get_price_volume_history(ticker: str) -> Optional[pd.DataFrame]:
    """
    Downloads adjusted Close and Volume data from Yahoo Finance.

    Nine months normally provides enough observations for the previous
    80-trading-day volume average plus the current streak.
    """
    data = yf.download(
        ticker,
        period="9mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        return None

    # yfinance can return MultiIndex columns even for one ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_columns = {"Close", "Volume"}

    if not required_columns.issubset(set(data.columns)):
        return None

    df = data[["Close", "Volume"]].copy()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna(subset=["Close", "Volume"])

    # At least 85 observations are required to support the 80-day average
    # and leave a few observations for detecting a streak.
    if len(df) < 85:
        return None

    return df


def add_volume_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds previous 30, 60 and 80 trading-day average volume.

    shift(1) excludes the current day's volume from its own average.
    """
    result = df.copy()

    result["avg_vol_30"] = result["Volume"].rolling(window=30).mean().shift(1)
    result["avg_vol_60"] = result["Volume"].rolling(window=60).mean().shift(1)
    result["avg_vol_80"] = result["Volume"].rolling(window=80).mean().shift(1)

    return result


def calculate_volume_ratio(volume: float, avg_volume: float) -> Optional[float]:
    """Returns current volume divided by a previous-volume average."""
    if pd.isna(avg_volume) or avg_volume <= 0:
        return None

    return float(volume / avg_volume)


def detect_latest_streak(df: pd.DataFrame) -> Optional[dict[str, Any]]:
    """
    Detects whether the latest movement is UP or DOWN for at least three
    consecutive trading sessions and calculates volume ratios for each day.
    """
    closes = df["Close"].dropna()

    if len(closes) < 4:
        return None

    moves: list[str] = []

    for index in range(1, len(closes)):
        previous_price = float(closes.iloc[index - 1])
        current_price = float(closes.iloc[index])

        if current_price > previous_price:
            moves.append("UP")
        elif current_price < previous_price:
            moves.append("DOWN")
        else:
            moves.append("FLAT")

    if not moves:
        return None

    latest_move = moves[-1]

    if latest_move == "FLAT":
        return None

    streak_days = 1

    for index in range(len(moves) - 2, -1, -1):
        if moves[index] == latest_move:
            streak_days += 1
        else:
            break

    if streak_days < 3:
        return None

    start_price = float(closes.iloc[-streak_days - 1])
    end_price = float(closes.iloc[-1])
    pct_change = ((end_price - start_price) / start_price) * 100

    streak_dates = closes.index[-streak_days:]
    volume_details: list[dict[str, Any]] = []

    for streak_date in streak_dates:
        row = df.loc[streak_date]
        volume = float(row["Volume"])

        volume_details.append(
            {
                "date": streak_date.strftime("%Y-%m-%d"),
                "close": float(row["Close"]),
                "volume_30_day_ratio": calculate_volume_ratio(
                    volume, row["avg_vol_30"]
                ),
                "volume_60_day_ratio": calculate_volume_ratio(
                    volume, row["avg_vol_60"]
                ),
                "volume_80_day_ratio": calculate_volume_ratio(
                    volume, row["avg_vol_80"]
                ),
            }
        )

    return {
        "direction": latest_move,
        "days": streak_days,
        "start_price": start_price,
        "end_price": end_price,
        "pct_change": pct_change,
        "latest_date": closes.index[-1].strftime("%Y-%m-%d"),
        "volume_details": volume_details,
    }


# ============================================================
# ANALYST CONSENSUS AND TARGETS
# ============================================================


def safe_float(value: Any) -> Optional[float]:
    """Converts a value to float, returning None for missing/invalid values."""
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
        if not pd.notna(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int:
    """Converts recommendation counts to a non-negative integer."""
    number = safe_float(value)
    if number is None or number < 0:
        return 0
    return int(round(number))


def normalise_column_name(value: Any) -> str:
    """Normalises DataFrame column names for tolerant matching."""
    return "".join(character for character in str(value).lower() if character.isalnum())


def select_current_recommendation_row(data: Any) -> Optional[pd.Series]:
    """
    Selects the current analyst-consensus row from yfinance output.

    yfinance normally returns rows for 0m, -1m, -2m and -3m. The 0m row is
    the current snapshot. This function also handles dictionary output and
    minor column-shape changes.
    """
    if data is None:
        return None

    if isinstance(data, dict):
        try:
            data = pd.DataFrame(data)
        except Exception:
            return None

    if not isinstance(data, pd.DataFrame) or data.empty:
        return None

    df = data.copy()

    # Some versions may return period as the index.
    if "period" not in [str(column).lower() for column in df.columns]:
        if normalise_column_name(df.index.name) == "period":
            df = df.reset_index()

    period_column = None
    for column in df.columns:
        if normalise_column_name(column) == "period":
            period_column = column
            break

    if period_column is not None:
        current_rows = df[
            df[period_column].astype(str).str.strip().str.lower() == "0m"
        ]
        if not current_rows.empty:
            return current_rows.iloc[0]

    # Current period is normally the first row when 0m is unavailable.
    return df.iloc[0]


def row_value(row: Optional[pd.Series], *possible_names: str) -> Any:
    """Returns a value from a Series using normalised alternative names."""
    if row is None:
        return None

    normalised_map = {
        normalise_column_name(column): row[column]
        for column in row.index
    }

    for name in possible_names:
        key = normalise_column_name(name)
        if key in normalised_map:
            return normalised_map[key]

    return None


def fetch_analyst_information(
    ticker: str,
    fallback_current_price: Optional[float] = None,
) -> dict[str, Any]:
    """
    Fetches current analyst recommendation counts and target prices.

    Buy = Strong Buy + Buy
    Sell = Sell + Strong Sell

    Missing analyst data does not stop the report. Such fields are returned
    as N/A-compatible None/zero values and the error is recorded.
    """
    result: dict[str, Any] = {
        "ticker": ticker,
        "strong_buy": 0,
        "buy_only": 0,
        "buy": 0,
        "hold": 0,
        "sell_only": 0,
        "strong_sell": 0,
        "sell": 0,
        "total_analysts": 0,
        "buy_pct": None,
        "hold_pct": None,
        "sell_pct": None,
        "consensus": "N/A",
        "recommendation_period": None,
        "current_price": fallback_current_price,
        "target_low": None,
        "target_mean": None,
        "target_median": None,
        "target_high": None,
        "mean_target_return_pct": None,
        "analyst_error": None,
    }

    errors: list[str] = []
    ticker_object = yf.Ticker(ticker)

    # ---------------- Recommendation counts ----------------
    recommendation_data = None

    try:
        recommendation_data = ticker_object.get_recommendations_summary()
    except Exception as exc:
        errors.append(f"recommendations summary: {exc}")

    if recommendation_data is None or (
        isinstance(recommendation_data, pd.DataFrame)
        and recommendation_data.empty
    ):
        try:
            recommendation_data = ticker_object.get_recommendations()
        except Exception as exc:
            errors.append(f"recommendations: {exc}")

    current_row = select_current_recommendation_row(recommendation_data)

    if current_row is not None:
        result["strong_buy"] = safe_int(
            row_value(current_row, "strongBuy", "strong_buy")
        )
        result["buy_only"] = safe_int(row_value(current_row, "buy"))
        result["hold"] = safe_int(row_value(current_row, "hold"))
        result["sell_only"] = safe_int(row_value(current_row, "sell"))
        result["strong_sell"] = safe_int(
            row_value(current_row, "strongSell", "strong_sell")
        )

        period = row_value(current_row, "period")
        if period is not None and not pd.isna(period):
            result["recommendation_period"] = str(period)

    result["buy"] = result["strong_buy"] + result["buy_only"]
    result["sell"] = result["sell_only"] + result["strong_sell"]
    result["total_analysts"] = result["buy"] + result["hold"] + result["sell"]

    total = result["total_analysts"]

    if total > 0:
        result["buy_pct"] = result["buy"] / total * 100
        result["hold_pct"] = result["hold"] / total * 100
        result["sell_pct"] = result["sell"] / total * 100

        category_counts = {
            "BUY": result["buy"],
            "HOLD": result["hold"],
            "SELL": result["sell"],
        }
        maximum = max(category_counts.values())
        leaders = [
            category
            for category, count in category_counts.items()
            if count == maximum
        ]
        result["consensus"] = "/".join(leaders)

    # ---------------- Target-price range ----------------
    targets: Any = None

    try:
        targets = ticker_object.get_analyst_price_targets()
    except Exception as exc:
        errors.append(f"analyst targets: {exc}")

    if not isinstance(targets, dict):
        try:
            targets = ticker_object.analyst_price_targets
        except Exception as exc:
            errors.append(f"analyst target property: {exc}")

    if isinstance(targets, dict):
        result["target_low"] = safe_float(targets.get("low"))
        result["target_mean"] = safe_float(targets.get("mean"))
        result["target_median"] = safe_float(targets.get("median"))
        result["target_high"] = safe_float(targets.get("high"))

        target_current = safe_float(targets.get("current"))
        if result["current_price"] is None:
            result["current_price"] = target_current

    # Older yfinance versions may expose target fields only through info.
    if all(
        result[key] is None
        for key in ("target_low", "target_mean", "target_median", "target_high")
    ):
        try:
            info = ticker_object.get_info()
            result["target_low"] = safe_float(info.get("targetLowPrice"))
            result["target_mean"] = safe_float(info.get("targetMeanPrice"))
            result["target_median"] = safe_float(info.get("targetMedianPrice"))
            result["target_high"] = safe_float(info.get("targetHighPrice"))

            if result["current_price"] is None:
                result["current_price"] = safe_float(
                    info.get("currentPrice") or info.get("regularMarketPrice")
                )
        except Exception as exc:
            errors.append(f"ticker info fallback: {exc}")

    current_price = result["current_price"]
    target_mean = result["target_mean"]

    if current_price is not None and current_price > 0 and target_mean is not None:
        result["mean_target_return_pct"] = (
            (target_mean - current_price) / current_price * 100
        )

    if errors:
        result["analyst_error"] = " | ".join(errors)

    return result


# ============================================================
# REPORT ASSEMBLY
# ============================================================


def build_report() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    """
    Processes every ticker once.

    Returns:
        analyst_rows:
            One analyst-consensus row for every ticker in the Google Sheet.

        streak_rows:
            Only stocks with a latest UP/DOWN streak of at least three days.

        failed_tickers:
            Price/volume failures and unexpected processing errors.
    """
    tickers = read_nse_tickers_from_sheet()

    analyst_rows: list[dict[str, Any]] = []
    streak_rows: list[dict[str, Any]] = []
    failed_tickers: list[str] = []

    for ticker in tickers:
        price_history: Optional[pd.DataFrame] = None
        latest_price: Optional[float] = None
        streak: Optional[dict[str, Any]] = None

        try:
            price_history = get_price_volume_history(ticker)

            if price_history is None:
                failed_tickers.append(f"{ticker}: price/volume history unavailable")
            else:
                price_history = add_volume_averages(price_history)
                latest_price = safe_float(price_history["Close"].iloc[-1])
                streak = detect_latest_streak(price_history)

        except Exception as exc:
            failed_tickers.append(f"{ticker}: price/volume error: {exc}")

        try:
            analyst = fetch_analyst_information(
                ticker=ticker,
                fallback_current_price=latest_price,
            )
        except Exception as exc:
            analyst = {
                "ticker": ticker,
                "strong_buy": 0,
                "buy_only": 0,
                "buy": 0,
                "hold": 0,
                "sell_only": 0,
                "strong_sell": 0,
                "sell": 0,
                "total_analysts": 0,
                "buy_pct": None,
                "hold_pct": None,
                "sell_pct": None,
                "consensus": "N/A",
                "recommendation_period": None,
                "current_price": latest_price,
                "target_low": None,
                "target_mean": None,
                "target_median": None,
                "target_high": None,
                "mean_target_return_pct": None,
                "analyst_error": str(exc),
            }

        analyst_rows.append(analyst)

        if streak is not None:
            streak_rows.append(
                {
                    "ticker": ticker,
                    **streak,
                    "analyst": analyst,
                }
            )

    # Put available analyst coverage first, then sort by ticker.
    analyst_rows.sort(
        key=lambda row: (
            row["total_analysts"] == 0,
            -row["total_analysts"],
            row["ticker"],
        )
    )

    streak_rows.sort(
        key=lambda row: (
            row["direction"] != "UP",
            -abs(row["pct_change"]),
            row["ticker"],
        )
    )

    return analyst_rows, streak_rows, failed_tickers


# ============================================================
# FORMATTING HELPERS
# ============================================================


def format_ratio(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2f}x"


def format_price(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"₹{value:,.2f}"


def format_percent(value: Optional[float], include_sign: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    if include_sign:
        return f"{value:+.1f}%"

    return f"{value:.1f}%"


def format_count_and_pct(count: int, pct: Optional[float]) -> str:
    if pct is None:
        return str(count) if count else "N/A"
    return f"{count} ({pct:.1f}%)"


def format_target_range(row: dict[str, Any]) -> str:
    low = row.get("target_low")
    high = row.get("target_high")

    if low is None and high is None:
        return "N/A"
    if low is None:
        return f"Up to {format_price(high)}"
    if high is None:
        return f"From {format_price(low)}"

    return f"{format_price(low)} – {format_price(high)}"


def html_cell(value: Any, align: str = "left") -> str:
    return (
        f'<td style="border:1px solid #c9c9c9;padding:6px 8px;'
        f'text-align:{align};white-space:nowrap;">{escape(str(value))}</td>'
    )


# ============================================================
# HTML EMAIL
# ============================================================


def build_analyst_html_table(analyst_rows: list[dict[str, Any]]) -> str:
    """Builds the portfolio-wide analyst table shown at the top."""
    if not analyst_rows:
        return "<p>No tickers were found in the Google Sheet.</p>"

    headers = [
        "Ticker",
        "Price",
        "Buy",
        "Hold",
        "Sell",
        "Analysts",
        "Consensus",
        "Target range",
        "Mean target",
        "Median target",
        "Return to mean",
    ]

    header_html = "".join(
        f'<th style="border:1px solid #9e9e9e;padding:7px 8px;'
        f'background:#eeeeee;text-align:center;white-space:nowrap;">'
        f"{escape(header)}</th>"
        for header in headers
    )

    body_rows: list[str] = []

    for row in analyst_rows:
        cells = [
            html_cell(row["ticker"]),
            html_cell(format_price(row.get("current_price")), "right"),
            html_cell(
                format_count_and_pct(row["buy"], row.get("buy_pct")), "right"
            ),
            html_cell(
                format_count_and_pct(row["hold"], row.get("hold_pct")), "right"
            ),
            html_cell(
                format_count_and_pct(row["sell"], row.get("sell_pct")), "right"
            ),
            html_cell(row["total_analysts"], "right"),
            html_cell(row["consensus"], "center"),
            html_cell(format_target_range(row), "right"),
            html_cell(format_price(row.get("target_mean")), "right"),
            html_cell(format_price(row.get("target_median")), "right"),
            html_cell(
                format_percent(row.get("mean_target_return_pct"), include_sign=True),
                "right",
            ),
        ]
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
    <div style="overflow-x:auto;">
      <table style="border-collapse:collapse;font-family:Arial,sans-serif;font-size:12px;">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """


def build_streak_html(streak_rows: list[dict[str, Any]]) -> str:
    if not streak_rows:
        return (
            "<p>No stock has increased or decreased for three or more "
            "consecutive trading days.</p>"
        )

    blocks: list[str] = []

    for row in streak_rows:
        direction_word = "INCREASED" if row["direction"] == "UP" else "DECREASED"
        analyst = row["analyst"]

        volume_rows = []
        for volume in row["volume_details"]:
            volume_rows.append(
                "<tr>"
                + html_cell(volume["date"])
                + html_cell(format_price(volume["close"]), "right")
                + html_cell(format_ratio(volume["volume_30_day_ratio"]), "right")
                + html_cell(format_ratio(volume["volume_60_day_ratio"]), "right")
                + html_cell(format_ratio(volume["volume_80_day_ratio"]), "right")
                + "</tr>"
            )

        block = f"""
        <div style="margin-top:24px;padding-top:8px;border-top:2px solid #777;">
          <h3 style="margin-bottom:8px;">{escape(row['ticker'])}</h3>
          <p>
            <b>Direction:</b> {direction_word}<br>
            <b>Consecutive trading days:</b> {row['days']}<br>
            <b>Price change over streak:</b> {row['pct_change']:+.2f}%<br>
            <b>From:</b> {escape(format_price(row['start_price']))}<br>
            <b>To:</b> {escape(format_price(row['end_price']))}<br>
            <b>Latest trading date:</b> {escape(row['latest_date'])}
          </p>

          <p>
            <b>Analyst consensus:</b> {escape(analyst['consensus'])}<br>
            <b>Buy:</b> {escape(format_count_and_pct(analyst['buy'], analyst.get('buy_pct')))};
            <b>Hold:</b> {escape(format_count_and_pct(analyst['hold'], analyst.get('hold_pct')))};
            <b>Sell:</b> {escape(format_count_and_pct(analyst['sell'], analyst.get('sell_pct')))}<br>
            <b>Analyst target range:</b> {escape(format_target_range(analyst))}<br>
            <b>Mean target:</b> {escape(format_price(analyst.get('target_mean')))};
            <b>Return to mean target:</b>
            {escape(format_percent(analyst.get('mean_target_return_pct'), include_sign=True))}
          </p>

          <p><b>Volume ratio for each streak day</b></p>
          <table style="border-collapse:collapse;font-family:Arial,sans-serif;font-size:12px;">
            <thead>
              <tr>
                <th style="border:1px solid #9e9e9e;padding:6px;background:#eeeeee;">Date</th>
                <th style="border:1px solid #9e9e9e;padding:6px;background:#eeeeee;">Close</th>
                <th style="border:1px solid #9e9e9e;padding:6px;background:#eeeeee;">Vol/30D</th>
                <th style="border:1px solid #9e9e9e;padding:6px;background:#eeeeee;">Vol/60D</th>
                <th style="border:1px solid #9e9e9e;padding:6px;background:#eeeeee;">Vol/80D</th>
              </tr>
            </thead>
            <tbody>{''.join(volume_rows)}</tbody>
          </table>
        </div>
        """
        blocks.append(block)

    return "".join(blocks)


def make_html_email_body(
    analyst_rows: list[dict[str, Any]],
    streak_rows: list[dict[str, Any]],
    failed_tickers: list[str],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    analyst_errors = [
        f"{row['ticker']}: {row['analyst_error']}"
        for row in analyst_rows
        if row.get("analyst_error")
    ]

    failed_html = ""
    if failed_tickers:
        failed_items = "".join(f"<li>{escape(item)}</li>" for item in failed_tickers)
        failed_html = f"<h3>Price/volume processing issues</h3><ul>{failed_items}</ul>"

    analyst_error_html = ""
    if analyst_errors:
        analyst_items = "".join(f"<li>{escape(item)}</li>" for item in analyst_errors)
        analyst_error_html = (
            "<h3>Analyst-data warnings</h3>"
            "<p>These warnings do not stop the rest of the report.</p>"
            f"<ul>{analyst_items}</ul>"
        )

    return f"""
    <html>
      <body style="font-family:Arial,sans-serif;color:#222;">
        <h1>NSE 3-Day Movement, Volume and Analyst Report</h1>
        <p><b>Generated at:</b> {escape(now)}</p>

        <h2>Analyst consensus for all shares</h2>
        <p>
          Buy combines <b>Strong Buy + Buy</b>. Sell combines
          <b>Sell + Strong Sell</b>. The target range is the overall analyst
          low-to-high target range; it is not a separate buy range and sell range.
        </p>
        {build_analyst_html_table(analyst_rows)}

        <h2 style="margin-top:28px;">Stocks with 3+ consecutive trading-day movement</h2>
        {build_streak_html(streak_rows)}

        <h3>Method notes</h3>
        <ul>
          <li>Saturdays, Sundays and NSE holidays are ignored automatically.</li>
          <li>Volume ratios use the previous 30, 60 and 80 trading days.</li>
          <li>The current day's volume is excluded from its own average.</li>
          <li>1.00x means normal volume; 2.00x means double average volume.</li>
          <li>Price, volume and analyst data come from Yahoo Finance through yfinance.</li>
          <li>Analyst coverage can be absent or incomplete for smaller stocks.</li>
        </ul>

        {failed_html}
        {analyst_error_html}
      </body>
    </html>
    """


# ============================================================
# PLAIN-TEXT FALLBACK AND CONSOLE OUTPUT
# ============================================================


def make_text_email_body(
    analyst_rows: list[dict[str, Any]],
    streak_rows: list[dict[str, Any]],
    failed_tickers: list[str],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body: list[str] = [
        "NSE 3-Day Movement, Volume and Analyst Report",
        f"Generated at: {now}",
        "",
        "ANALYST CONSENSUS FOR ALL SHARES",
        "Buy = Strong Buy + Buy; Sell = Sell + Strong Sell",
        "",
    ]

    for row in analyst_rows:
        body.extend(
            [
                f"Stock: {row['ticker']}",
                f"Current price: {format_price(row.get('current_price'))}",
                f"Buy: {format_count_and_pct(row['buy'], row.get('buy_pct'))}",
                f"Hold: {format_count_and_pct(row['hold'], row.get('hold_pct'))}",
                f"Sell: {format_count_and_pct(row['sell'], row.get('sell_pct'))}",
                f"Total analysts: {row['total_analysts']}",
                f"Consensus: {row['consensus']}",
                f"Target range: {format_target_range(row)}",
                f"Mean target: {format_price(row.get('target_mean'))}",
                f"Median target: {format_price(row.get('target_median'))}",
                "Return to mean target: "
                f"{format_percent(row.get('mean_target_return_pct'), include_sign=True)}",
                "-" * 50,
            ]
        )

    body.extend(["", "STOCKS WITH 3+ CONSECUTIVE TRADING-DAY MOVEMENT", ""])

    if not streak_rows:
        body.append(
            "No stock has increased or decreased for three or more consecutive trading days."
        )
    else:
        for row in streak_rows:
            direction_word = "INCREASED" if row["direction"] == "UP" else "DECREASED"
            analyst = row["analyst"]

            body.extend(
                [
                    f"Stock: {row['ticker']}",
                    f"Direction: {direction_word}",
                    f"Consecutive trading days: {row['days']}",
                    f"Price change over streak: {row['pct_change']:+.2f}%",
                    f"From: {format_price(row['start_price'])}",
                    f"To: {format_price(row['end_price'])}",
                    f"Latest trading date: {row['latest_date']}",
                    f"Analyst consensus: {analyst['consensus']}",
                    f"Analyst target range: {format_target_range(analyst)}",
                    "",
                    "Volume ratio for each streak day:",
                ]
            )

            for volume in row["volume_details"]:
                body.extend(
                    [
                        f"Date: {volume['date']}",
                        f"Close: {format_price(volume['close'])}",
                        "Volume / previous 30-day avg: "
                        f"{format_ratio(volume['volume_30_day_ratio'])}",
                        "Volume / previous 60-day avg: "
                        f"{format_ratio(volume['volume_60_day_ratio'])}",
                        "Volume / previous 80-day avg: "
                        f"{format_ratio(volume['volume_80_day_ratio'])}",
                        "",
                    ]
                )

            body.append("=" * 60)

    if failed_tickers:
        body.extend(["", "PRICE/VOLUME PROCESSING ISSUES:"])
        body.extend(f"- {item}" for item in failed_tickers)

    analyst_errors = [
        f"{row['ticker']}: {row['analyst_error']}"
        for row in analyst_rows
        if row.get("analyst_error")
    ]

    if analyst_errors:
        body.extend(["", "ANALYST-DATA WARNINGS:"])
        body.extend(f"- {item}" for item in analyst_errors)

    return "\n".join(body)


# ============================================================
# EMAIL AND ENTRY POINT
# ============================================================


def send_email(subject: str, text_body: str, html_body: str) -> None:
    """
    Sends a multipart Gmail message.

    Required GitHub secrets/environment variables:
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

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = EMAIL_FROM
    message["To"] = EMAIL_TO

    message.attach(MIMEText(text_body, "plain", "utf-8"))
    message.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60) as server:
        server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        server.send_message(message)


def main() -> None:
    today = date.today()

    if not is_nse_trading_day(today):
        print(f"{today} is not an NSE trading day. Skipping report.")
        return

    analyst_rows, streak_rows, failed_tickers = build_report()

    text_body = make_text_email_body(
        analyst_rows=analyst_rows,
        streak_rows=streak_rows,
        failed_tickers=failed_tickers,
    )

    html_body = make_html_email_body(
        analyst_rows=analyst_rows,
        streak_rows=streak_rows,
        failed_tickers=failed_tickers,
    )

    print(text_body)

    send_email(
        subject="NSE Movement, Volume + Analyst Consensus Report",
        text_body=text_body,
        html_body=html_body,
    )

    print("Email sent successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(0)
