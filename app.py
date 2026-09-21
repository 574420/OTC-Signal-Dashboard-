import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

st.set_page_config(
    page_title="REAL + OTC Signal Bot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 REAL + OTC SIGNAL BOT")
st.caption("Analysis only | Auto Trading Disabled | Demo/Paper Testing")

# =========================================================
# SETTINGS
# =========================================================

OTCHARTS_API_KEY = st.secrets.get("OTCHARTS_API_KEY", "")

REAL_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "EURGBP",
    "EURJPY",
    "NZDUSD",
    "EURCHF",
    "GBPJPY",
    "AUDJPY",
    "EURAUD",
    "GBPAUD",
    "CADJPY",
    "CHFJPY"
]

OTC_PAIRS = [
    "EURUSD_otc",
    "GBPUSD_otc",
    "USDJPY_otc",
    "AUDUSD_otc",
    "USDCAD_otc",
    "USDCHF_otc",
    "EURGBP_otc",
    "EURJPY_otc",
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

# =========================================================
# SESSION MEMORY
# =========================================================

if "last_signal" not in st.session_state:
    st.session_state.last_signal = None

if "signal_history" not in st.session_state:
    st.session_state.signal_history = []

if "last_signal_time" not in st.session_state:
    st.session_state.last_signal_time = None


# =========================================================
# RSI
# =========================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


# =========================================================
# SIGNAL ENGINE
# =========================================================

def make_signal(df):

    df = df.copy()

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.dropna(
        subset=["open", "high", "low", "close"]
    )

    df["EMA9"] = df["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["EMA21"] = df["close"].ewm(
        span=21,
        adjust=False
    ).mean()

    df["EMA50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["RSI"] = calculate_rsi(
        df["close"]
    )

    df = df.dropna().reset_index(drop=True)

    if len(df) < 30:

        return {
            "signal": "WAIT",
            "score": 0,
            "price": 0,
            "rsi": 0,
            "ema9": 0,
            "ema21": 0,
            "ema50": 0
        }

    last = df.iloc[-1]

    call = 0
    put = 0

    # EMA 9 / 21
    if last["EMA9"] > last["EMA21"]:
        call += 2
    elif last["EMA9"] < last["EMA21"]:
        put += 2

    # EMA 21 / 50
    if last["EMA21"] > last["EMA50"]:
        call += 2
    elif last["EMA21"] < last["EMA50"]:
        put += 2

    # RSI
    if last["RSI"] >= 55:
        call += 2
    elif last["RSI"] <= 45:
        put += 2

    # Candle direction
    if last["close"] > last["open"]:
        call += 1
    elif last["close"] < last["open"]:
        put += 1

    if call > put and call >= 5:
        signal = "CALL"
        score = call

    elif put > call and put >= 5:
        signal = "PUT"
        score = put

    else:
        signal = "WAIT"
        score = max(call, put)

    return {
        "signal": signal,
        "score": int(score),
        "price": float(last["close"]),
        "rsi": float(last["RSI"]),
        "ema9": float(last["EMA9"]),
        "ema21": float(last["EMA21"]),
        "ema50": float(last["EMA50"])
    }


# =========================================================
# CANDLE COUNTDOWN
# =========================================================

def candle_countdown(seconds):

    now = int(time.time())

    remaining = seconds - (
        now % seconds
    )

    if remaining == seconds:
        remaining = 0

    return remaining


# =========================================================
# REAL DATA
# =========================================================

@st.cache_data(ttl=10)
def get_real_data(pair, timeframe_seconds):

    url = f"https://biquote.io/api/{pair}/ohlc"

    response = requests.get(
        url,
        params={
            "interval": "1m",
            "limit": 200
        },
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(
        data.get("bars", [])
    )

    if df.empty:
        return df

    df["openTime"] = pd.to_datetime(
        df["openTime"],
        utc=True
    )

    df = df.sort_values(
        "openTime"
    ).reset_index(drop=True)

    if "isOpen" in df.columns:

        df = df[
            df["isOpen"] == False
        ].copy()

    if timeframe_seconds > 60:

        minutes = timeframe_seconds // 60

        df = df.set_index(
            "openTime"
        )

        df = df.resample(
            f"{minutes}min"
        ).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        }).dropna().reset_index()

    return df


# =========================================================
# OTC DATA
# =========================================================

def get_otc_data(
    pair,
    timeframe_seconds
):

    if not OTCHARTS_API_KEY:

        return None, "OTCharts API key not configured"

    url = "https://otcharts.com/v1/candles"

    headers = {
        "Authorization":
        f"Bearer {OTCHARTS_API_KEY}"
    }

    response = requests.get(
        url,
        headers=headers,
        params={
            "venue": "otc",
            "symbol": pair,
            "timeframe":
                f"{timeframe_seconds}s",
            "limit": 200
        },
        timeout=20
    )

    if response.status_code != 200:

        return None, (
            f"OTCharts HTTP "
            f"{response.status_code}: "
            f"{response.text[:300]}"
        )

    data = response.json()

    candles = data.get(
        "candles",
        []
    )

    if not candles:

        return None, "No OTC candles returned"

    df = pd.DataFrame(
        candles
    )

    # Normalize OHLC names

    rename_map = {}

    for c in df.columns:

        name = str(c).lower()

        if name == "open":
            rename_map[c] = "open"

        elif name == "high":
            rename_map[c] = "high"

        elif name == "low":
            rename_map[c] = "low"

        elif name == "close":
            rename_map[c] = "close"

    df = df.rename(
        columns=rename_map
    )

    # Normalize time

    if "time" in df.columns:

        try:

            df["time"] = pd.to_datetime(
                df["time"],
                unit="s",
                utc=True
            )

        except:

            df["time"] = pd.to_datetime(
                df["time"],
                utc=True,
                errors="coerce"
            )

    for col in [
        "open",
        "high",
        "low",
        "close"
    ]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )

    return df, "OK"


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "⚙️ Bot Settings"
)

market = st.sidebar.selectbox(
    "Market",
    ["REAL", "OTC"]
)

if market == "REAL":

    pair = st.sidebar.selectbox(
        "REAL Pair",
        REAL_PAIRS
    )

else:

    pair = st.sidebar.selectbox(
        "OTC Pair",
        OTC_PAIRS
    )

timeframe_name = st.sidebar.selectbox(
    "Candle Timeframe",
    list(TIMEFRAMES.keys())
)

timeframe_seconds = TIMEFRAMES[
    timeframe_name
]

expiry = st.sidebar.selectbox(
    "Expiry",
    [
        "1 Candle",
        "2 Candles",
        "3 Candles"
    ]
)

st.sidebar.divider()

if st.sidebar.button(
    "🔄 REFRESH DATA",
    use_container_width=True
):

    st.cache_data.clear()
    st.rerun()


# =========================================================
# TOP INFO
# =========================================================

st.divider()

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "MARKET",
    market
)

c2.metric(
    "PAIR",
    pair
)

c3.metric(
    "TIMEFRAME",
    timeframe_name
)

c4.metric(
    "EXPIRY",
    expiry
)

st.divider()


# =========================================================
# COUNTDOWN
# =========================================================

remaining = candle_countdown(
    timeframe_seconds
)

mins = remaining // 60
secs = remaining % 60

st.subheader(
    "⏱️ Current Candle"
)

count_col1, count_col2 = st.columns(2)

count_col1.metric(
    "Candle closes in",
    f"{mins:02d}:{secs:02d}"
)

if remaining <= 5:

    count_col2.warning(
        "Candle closing — prepare for next candle"
    )

else:

    count_col2.info(
        "GET SIGNAL will prepare the next-candle setup."
    )


# =========================================================
# GET SIGNAL
# =========================================================

st.divider()

st.subheader(
    "🎯 Signal Generator"
)

if st.button(
    "🚀 GET SIGNAL",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Getting latest candle data..."
    ):

        try:

            if market == "REAL":

                df = get_real_data(
                    pair,
                    timeframe_seconds
                )

                source_message = "REAL"

            else:

                df, source_message = get_otc_data(
                    pair,
                    timeframe_seconds
                )

            if df is None:

                st.error(
                    source_message
                )

            elif df.empty:

                st.error(
                    "No candle data received."
                )

            else:

                result = make_signal(
                    df
                )

                signal_time = datetime.now(
                    timezone.utc
                )

                st.session_state.last_signal = {
                    "market": market,
                    "pair": pair,
                    "timeframe": timeframe_name,
                    "expiry": expiry,
                    "signal": result["signal"],
                    "score": result["score"],
                    "price": result["price"],
                    "rsi": result["rsi"],
                    "time": signal_time
                }

                st.session_state.last_signal_time = signal_time

                # Add to history

                st.session_state.signal_history.insert(
                    0,
                    {
                        "Time": signal_time.strftime(
                            "%H:%M:%S UTC"
                        ),
                        "Market": market,
                        "Pair": pair,
                        "TF": timeframe_name,
                        "Signal": result["signal"],
                        "Score": result["score"],
                        "Entry": round(
                            result["price"],
                            6
                        )
                    }
                )

                # Keep last 20

                st.session_state.signal_history = (
                    st.session_state.signal_history[:20]
                )

        except Exception as e:

            st.error(
                f"Signal error: {e}"
            )


# =========================================================
# SHOW SIGNAL
# =========================================================

if st.session_state.last_signal:

    signal = st.session_state.last_signal

    st.divider()

    st.subheader(
        "📢 NEXT CANDLE SIGNAL"
    )

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "SIGNAL",
        signal["signal"]
    )

    s2.metric(
        "STRENGTH",
        f"{signal['score']}/7"
    )

    s3.metric(
        "ENTRY PRICE",
        f"{signal['price']:.6f}"
    )

    s4.metric(
        "RSI",
        f"{signal['rsi']:.2f}"
    )

    if signal["signal"] == "CALL":

        st.success(
            "🟢 CALL — Next candle setup"
        )

    elif signal["signal"] == "PUT":

        st.error(
            "🔴 PUT — Next candle setup"
        )

    else:

        st.warning(
            "🟡 WAIT — No clear setup"
        )

    st.info(
        f"Expiry selected: {signal['expiry']} | "
        f"Signal generated from the latest completed candle."
    )


# =========================================================
# LOAD DATA FOR CHART
# =========================================================

st.divider()

st.subheader(
    "📊 Market Chart"
)

try:

    if market == "REAL":

        chart_df = get_real_data(
            pair,
            timeframe_seconds
        )

    else:

        chart_df, msg = get_otc_data(
            pair,
            timeframe_seconds
        )

    if chart_df is not None and not chart_df.empty:

        if "time" in chart_df.columns:

            display_df = chart_df.copy()

            display_df["time"] = pd.to_datetime(
                display_df["time"],
                utc=True,
                errors="coerce"
            )

            display_df = display_df.dropna(
                subset=["time"]
            )

            chart_data = display_df.set_index(
                "time"
            )

        elif "openTime" in chart_df.columns:

            chart_data = chart_df.copy()

            chart_data = chart_data.set_index(
                "openTime"
            )

        else:

            chart_data = chart_df.copy()

        if "close" in chart_data.columns:

            st.line_chart(
                chart_data["close"]
            )

        st.subheader(
            "Latest Candles"
        )

        st.dataframe(
            chart_df.tail(20),
            use_container_width=True
        )

    else:

        st.warning(
            "Chart data unavailable."
        )

except Exception as e:

    st.warning(
        f"Chart error: {e}"
    )


# =========================================================
# SIGNAL HISTORY
# =========================================================

st.divider()

st.subheader(
    "🧾 Signal History"
)

if st.session_state.signal_history:

    history_df = pd.DataFrame(
        st.session_state.signal_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No signals generated yet."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.success(
    "SYSTEM READY | AUTO TRADING DISABLED | PAPER / DEMO MODE"
)

st.caption(
    "Signal strength is a rule-based confirmation score, "
    "not a guaranteed probability or guaranteed trade result."
)
