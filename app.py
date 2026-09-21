
import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="REAL + OTC Signal Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("REAL + OTC SIGNAL DASHBOARD")
st.caption("Analysis only | Auto Trading Disabled | Paper Test Mode")

# =========================
# SETTINGS
# =========================

OTCHARTS_API_KEY = st.secrets.get("OTCHARTS_API_KEY", "")

REAL_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCAD", "USDCHF", "EURGBP", "EURJPY"
]

OTC_DEFAULT = [
    "EURUSD_otc",
    "GBPUSD_otc",
    "USDJPY_otc",
    "BTCUSD_otc"
]

TIMEFRAMES = {
    "1 Minute": 60,
    "5 Minutes": 300,
    "15 Minutes": 900,
    "30 Minutes": 1800,
    "1 Hour": 3600,
    "4 Hours": 14400
}

# =========================
# INDICATORS
# =========================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def make_signal(df):

    df = df.copy()

    df["EMA9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["RSI"] = calculate_rsi(df["close"])

    df = df.dropna()

    if len(df) < 30:
        return {
            "signal": "WAIT",
            "score": 0,
            "price": float(df.iloc[-1]["close"]),
            "rsi": 0
        }

    last = df.iloc[-1]

    score_call = 0
    score_put = 0

    # EMA 9 / 21
    if last["EMA9"] > last["EMA21"]:
        score_call += 2
    elif last["EMA9"] < last["EMA21"]:
        score_put += 2

    # EMA 21 / 50
    if last["EMA21"] > last["EMA50"]:
        score_call += 2
    elif last["EMA21"] < last["EMA50"]:
        score_put += 2

    # RSI
    if last["RSI"] >= 55:
        score_call += 2
    elif last["RSI"] <= 45:
        score_put += 2

    # Candle direction
    if last["close"] > last["open"]:
        score_call += 1
    elif last["close"] < last["open"]:
        score_put += 1

    if score_call > score_put and score_call >= 5:
        signal = "CALL"
        score = score_call
    elif score_put > score_call and score_put >= 5:
        signal = "PUT"
        score = score_put
    else:
        signal = "WAIT"
        score = max(score_call, score_put)

    return {
        "signal": signal,
        "score": int(score),
        "price": float(last["close"]),
        "rsi": float(last["RSI"]),
        "ema9": float(last["EMA9"]),
        "ema21": float(last["EMA21"]),
        "ema50": float(last["EMA50"])
    }


# =========================
# REAL DATA
# =========================

@st.cache_data(ttl=30)
def get_real_data(pair, timeframe_seconds):

    url = f"https://biquote.io/api/{pair}/ohlc"

    response = requests.get(
        url,
        params={
            "interval": "1m",
            "limit": 150
        },
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["bars"])

    if df.empty:
        return df

    df["openTime"] = pd.to_datetime(df["openTime"], utc=True)

    df = df.sort_values("openTime").reset_index(drop=True)

    if "isOpen" in df.columns:
        df = df[df["isOpen"] == False].copy()

    # Resample for selected timeframe
    if timeframe_seconds > 60:

        minutes = timeframe_seconds // 60

        df = df.set_index("openTime")

        df = df.resample(
            f"{minutes}min"
        ).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        }).dropna().reset_index()

    return df


# =========================
# OTC DATA
# =========================

def get_otc_data(pair, timeframe_seconds):

    if not OTCHARTS_API_KEY:
        return None, "OTCharts API key not configured"

    url = "https://otcharts.com/v1/candles"

    headers = {
        "Authorization": f"Bearer {OTCHARTS_API_KEY}"
    }

    response = requests.get(
        url,
        headers=headers,
        params={
            "venue": "otc",
            "symbol": pair,
            "timeframe": f"{timeframe_seconds}s",
            "limit": 150
        },
        timeout=20
    )

    if response.status_code != 200:
        return None, response.text

    data = response.json()

    candles = data.get("candles", [])

    if not candles:
        return None, "No candles returned"

    df = pd.DataFrame(candles)

    # Handle possible column names
    if "time" in df.columns:
        df["time"] = pd.to_datetime(
            df["time"],
            unit="s",
            utc=True
        )

    return df, "OK"


# =========================
# SIDEBAR
# =========================

st.sidebar.header("Dashboard Settings")

market = st.sidebar.selectbox(
    "Market",
    ["REAL", "OTC"]
)

if market == "REAL":

    pair = st.sidebar.selectbox(
        "Pair",
        REAL_PAIRS
    )

    timeframe_name = st.sidebar.selectbox(
        "Candle Timeframe",
        list(TIMEFRAMES.keys())
    )

else:

    pair = st.sidebar.selectbox(
        "OTC Pair",
        OTC_DEFAULT
    )

    timeframe_name = st.sidebar.selectbox(
        "Candle Timeframe",
        [
            "1 Minute",
            "5 Minutes",
            "15 Minutes",
            "30 Minutes"
        ]
    )

timeframe_seconds = TIMEFRAMES[timeframe_name]

expiry = st.sidebar.selectbox(
    "Expiry",
    [
        "1 Candle",
        "2 Candles",
        "3 Candles"
    ]
)

if st.sidebar.button("REFRESH DATA"):
    st.cache_data.clear()
    st.rerun()


# =========================
# MAIN
# =========================

st.divider()

col1, col2, col3, col4 = st.columns(4)

col1.metric("MARKET", market)
col2.metric("PAIR", pair)
col3.metric("TIMEFRAME", timeframe_name)
col4.metric("EXPIRY", expiry)

st.divider()

if market == "REAL":

    try:

        df = get_real_data(
            pair,
            timeframe_seconds
        )

        if df.empty:
            st.error("No REAL market data received.")
        else:

            result = make_signal(df)

            st.subheader("SIGNAL")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "SIGNAL",
                result["signal"]
            )

            c2.metric(
                "SCORE",
                result["score"]
            )

            c3.metric(
                "ENTRY",
                f"{result['price']:.6f}"
            )

            c4.metric(
                "RSI",
                f"{result['rsi']:.2f}"
            )

            st.subheader("Price Chart")

            chart_df = df.copy()

            if "openTime" in chart_df.columns:
                chart_df = chart_df.set_index("openTime")

            st.line_chart(
                chart_df["close"]
            )

            st.subheader("Latest Candles")

            st.dataframe(
                df.tail(20),
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"REAL DATA ERROR: {e}"
        )

else:

    df, message = get_otc_data(
        pair,
        timeframe_seconds
    )

    if df is None:

        st.warning(
            f"OTC DATA: {message}"
        )

    else:

        # Normalize columns
        rename_map = {}

        for c in df.columns:
            if c.lower() == "open":
                rename_map[c] = "open"
            elif c.lower() == "high":
                rename_map[c] = "high"
            elif c.lower() == "low":
                rename_map[c] = "low"
            elif c.lower() == "close":
                rename_map[c] = "close"

        df = df.rename(
            columns=rename_map
        )

        required = [
            "open",
            "high",
            "low",
            "close"
        ]

        if not all(
            x in df.columns for x in required
        ):

            st.error(
                "OTC candle format is not recognized."
            )

        else:

            result = make_signal(df)

            st.subheader("OTC SIGNAL")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "SIGNAL",
                result["signal"]
            )

            c2.metric(
                "SCORE",
                result["score"]
            )

            c3.metric(
                "ENTRY",
                f"{result['price']:.6f}"
            )

            c4.metric(
                "RSI",
                f"{result['rsi']:.2f}"
            )

            st.subheader("OTC Price Chart")

            if "time" in df.columns:
                chart_df = df.set_index("time")
                st.line_chart(
                    chart_df["close"]
                )
            else:
                st.line_chart(
                    df["close"]
                )

            st.subheader("Latest OTC Candles")

            st.dataframe(
                df.tail(20),
                use_container_width=True
            )


st.divider()

st.success(
    "SYSTEM READY | AUTO TRADING DISABLED | PAPER/ANALYSIS MODE"
)

st.caption(
    "Signal score is a rule-based confirmation score, not a guaranteed probability."
)
