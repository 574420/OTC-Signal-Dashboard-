import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="REAL + OTC Signal Bot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 REAL + OTC SIGNAL BOT")
st.caption(
    "Analysis only | Auto Trading Disabled | Demo/Paper Testing"
)


# =========================================================
# SETTINGS
# =========================================================

OTCHARTS_API_KEY = st.secrets.get(
    "OTCHARTS_API_KEY",
    ""
)


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


CHART_TIMEFRAMES = {
    "Same as Candle": None,
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

defaults = {
    "last_signal": None,
    "signal_history": [],
    "last_signal_time": None,
    "armed": False,
    "armed_candle": None,
    "last_processed_candle": None,
    "signal_for_candle": None
}


for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# RSI
# =========================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        period
    ).mean()

    avg_loss = loss.rolling(
        period
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    return 100 - (
        100 / (1 + rs)
    )


# =========================================================
# SIGNAL ENGINE
# =========================================================

def make_signal(df):

    df = df.copy()

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


    df = df.dropna().reset_index(
        drop=True
    )


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
        score = max(
            call,
            put
        )


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
# TIME HELPERS
# =========================================================

def floor_time(dt, seconds):

    timestamp = int(
        dt.timestamp()
    )

    floored = (
        timestamp // seconds
    ) * seconds

    return datetime.fromtimestamp(
        floored,
        tz=timezone.utc
    )


def format_countdown(seconds):

    seconds = max(
        0,
        int(seconds)
    )

    minutes = seconds // 60

    secs = seconds % 60

    hours = minutes // 60

    minutes = minutes % 60


    if hours > 0:

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:02d}"
        )


    return (
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def candle_times_from_clock(
    timeframe_seconds
):

    now = datetime.now(
        timezone.utc
    )


    current_start = floor_time(
        now,
        timeframe_seconds
    )


    current_end = (
        current_start
        + timedelta(
            seconds=timeframe_seconds
        )
    )


    remaining = int(
        (
            current_end - now
        ).total_seconds()
    )


    remaining = max(
        0,
        remaining
    )


    next_start = current_end


    next_end = (
        next_start
        + timedelta(
            seconds=timeframe_seconds
        )
    )


    return (
        now,
        current_start,
        current_end,
        next_start,
        next_end,
        remaining
    )


def candle_label(dt):

    if dt is None:

        return "N/A"


    return dt.strftime(
        "%H:%M:%S UTC"
    )


# =========================================================
# REAL DATA
# =========================================================

@st.cache_data(ttl=3)
def get_real_data(
    pair,
    timeframe_seconds
):

    url = (
        f"https://biquote.io/api/"
        f"{pair}/ohlc"
    )


    response = requests.get(
        url,
        params={
            "interval": "1m",
            "limit": 500
        },
        timeout=20
    )


    response.raise_for_status()


    data = response.json()


    df = pd.DataFrame(
        data.get(
            "bars",
            []
        )
    )


    if df.empty:

        return df


    if "openTime" not in df.columns:

        return pd.DataFrame()


    df["openTime"] = pd.to_datetime(
        df["openTime"],
        utc=True,
        errors="coerce"
    )


    df = df.dropna(
        subset=["openTime"]
    )


    df = df.sort_values(
        "openTime"
    ).reset_index(
        drop=True
    )


    if "isOpen" in df.columns:

        df = df[
            df["isOpen"] == False
        ].copy()


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


    if timeframe_seconds > 60:

        minutes = (
            timeframe_seconds // 60
        )


        df = (
            df.set_index(
                "openTime"
            )
            .resample(
                f"{minutes}min"
            )
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last"
            })
            .dropna()
            .reset_index()
        )


    return df


# =========================================================
# OTC DATA
# =========================================================

def get_otc_data(
    pair,
    timeframe_seconds
):

    if not OTCHARTS_API_KEY:

        return (
            None,
            "OTCharts API key not configured."
        )


    url = (
        "https://otcharts.com/"
        "v1/candles"
    )


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
            "limit": 500
        },
        timeout=20
    )


    if response.status_code != 200:

        return (
            None,
            f"OTCharts HTTP "
            f"{response.status_code}"
        )


    data = response.json()


    candles = data.get(
        "candles",
        []
    )


    if not candles:

        return (
            None,
            "No OTC candles returned."
        )


    df = pd.DataFrame(
        candles
    )


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


    time_column = None


    for candidate in [
        "time",
        "timestamp",
        "openTime",
        "open_time"
    ]:

        if candidate in df.columns:

            time_column = candidate

            break


    if time_column:

        values = pd.to_numeric(
            df[time_column],
            errors="coerce"
        )


        if (
            values.dropna().size > 0
            and
            values.dropna().median()
            > 100000000000
        ):

            df["time"] = pd.to_datetime(
                values,
                unit="ms",
                utc=True,
                errors="coerce"
            )

        else:

            df["time"] = pd.to_datetime(
                values,
                unit="s",
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


    if "time" in df.columns:

        df = df.dropna(
            subset=["time"]
        )

        df = df.sort_values(
            "time"
        ).reset_index(
            drop=True
        )


    return (
        df,
        "OK"
    )


# =========================================================
# LOAD DATA
# =========================================================

def load_market_data(
    market,
    pair,
    timeframe_seconds
):

    if market == "REAL":

        return (
            get_real_data(
                pair,
                timeframe_seconds
            ),
            "REAL"
        )


    return get_otc_data(
        pair,
        timeframe_seconds
    )


# =========================================================
# LAST COMPLETED CANDLE
# =========================================================

def get_last_completed_candle(
    df,
    timeframe_seconds
):

    if df is None or df.empty:

        return None, None


    work = df.copy()


    if "openTime" in work.columns:

        work["_candle_time"] = (
            pd.to_datetime(
                work["openTime"],
                utc=True,
                errors="coerce"
            )
        )


    elif "time" in work.columns:

        work["_candle_time"] = (
            pd.to_datetime(
                work["time"],
                utc=True,
                errors="coerce"
            )
        )


    else:

        return None, None


    work = work.dropna(
        subset=["_candle_time"]
    )


    if work.empty:

        return None, None


    work = work.sort_values(
        "_candle_time"
    ).reset_index(
        drop=True
    )


    now = datetime.now(
        timezone.utc
    )


    completed = work[
        (
            work["_candle_time"]
            + pd.to_timedelta(
                timeframe_seconds,
                unit="s"
            )
        )
        <= pd.Timestamp(now)
    ].copy()


    if completed.empty:

        return None, None


    last = completed.iloc[-1]


    candle_time = (
        last["_candle_time"]
        .to_pydatetime()
    )


    return (
        completed,
        candle_time
    )


# =========================================================
# GENERATE NEXT CANDLE SIGNAL
# =========================================================

def generate_next_signal(
    market,
    pair,
    timeframe_name,
    timeframe_seconds,
    expiry
):

    df, message = load_market_data(
        market,
        pair,
        timeframe_seconds
    )


    if df is None:

        return None, message


    if df.empty:

        return (
            None,
            "No candle data received."
        )


    completed_df, candle_time = (
        get_last_completed_candle(
            df,
            timeframe_seconds
        )
    )


    if completed_df is None:

        return (
            None,
            "Completed candle is not available yet."
        )


    result = make_signal(
        completed_df
    )


    next_candle_start = (
        candle_time
        + timedelta(
            seconds=timeframe_seconds
        )
    )


    result["source_candle_time"] = (
        candle_time
    )


    result["next_candle_start"] = (
        next_candle_start
    )


    result["market"] = market

    result["pair"] = pair

    result["timeframe"] = (
        timeframe_name
    )

    result["expiry"] = expiry


    return (
        result,
        "OK"
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "⚙️ Bot Settings"
)


market = st.sidebar.selectbox(
    "Market",
    [
        "REAL",
        "OTC"
    ]
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


st.sidebar.subheader(
    "⏱️ Candle Settings"
)


candle_timeframe_name = (
    st.sidebar.selectbox(
        "Candle Timeframe",
        list(
            TIMEFRAMES.keys()
        ),
        index=0
    )
)


candle_seconds = TIMEFRAMES[
    candle_timeframe_name
]


st.sidebar.subheader(
    "📊 Chart Settings"
)


chart_timeframe_name = (
    st.sidebar.selectbox(
        "Chart Timeframe",
        list(
            CHART_TIMEFRAMES.keys()
        ),
        index=0
    )
)


chart_seconds = CHART_TIMEFRAMES[
    chart_timeframe_name
]


if chart_seconds is None:

    chart_seconds = candle_seconds


st.sidebar.subheader(
    "🎯 Expiry"
)


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


c1, c2, c3, c4, c5 = st.columns(5)


c1.metric(
    "MARKET",
    market
)


c2.metric(
    "PAIR",
    pair
)


c3.metric(
    "CANDLE",
    candle_timeframe_name
)


c4.metric(
    "CHART",
    chart_timeframe_name
)


c5.metric(
    "EXPIRY",
    expiry
)


st.divider()


# =========================================================
# LIVE CANDLE CLOCK
# =========================================================

clock_container = st.empty()


def show_live_clock():

    (
        now,
        current_start,
        current_end,
        next_start,
        next_end,
        remaining
    ) = candle_times_from_clock(
        candle_seconds
    )


    with clock_container.container():

        st.subheader(
            "⏱️ LIVE CANDLE CLOCK"
        )


        a, b, c = st.columns(3)


        a.metric(
            "CURRENT CANDLE",
            (
                f"{candle_label(current_start)}"
                " → "
                f"{candle_label(current_end)}"
            )
        )


        b.metric(
            "CANDLE CLOSES IN",
            format_countdown(
                remaining
            )
        )


        c.metric(
            "NEXT CANDLE STARTS",
            candle_label(
                next_start
            )
        )


        if remaining <= 5:

            st.warning(
                "⚠️ Current candle is closing. "
                "Signal will be calculated after 00:00."
            )

        else:

            st.info(
                "Current candle is running. "
                "ARM the signal and wait for 00:00."
            )


# =========================================================
# LIVE CLOCK FRAGMENT
# =========================================================

@st.fragment(
    run_every="1s"
)
def live_clock_fragment():

    show_live_clock()


live_clock_fragment()


# =========================================================
# SIGNAL GENERATOR
# =========================================================

st.divider()


st.subheader(
    "🎯 Signal Generator"
)


if st.button(
    "🚀 ARM NEXT-CANDLE SIGNAL",
    type="primary",
    use_container_width=True
):

    (
        now,
        current_start,
        current_end,
        next_start,
        next_end,
        remaining
    ) = candle_times_from_clock(
        candle_seconds
    )


    st.session_state.armed = True

    st.session_state.armed_candle = (
        current_start
    )

    st.session_state.last_signal = None

    st.session_state.signal_for_candle = None

    st.session_state.last_processed_candle = None


    st.success(
        "🟢 Signal armed for candle "
        f"{candle_label(current_start)} → "
        f"{candle_label(current_end)}. "
        "Wait for 00:00."
    )


if st.session_state.armed:

    st.info(
        "🟢 SIGNAL ARMED — Do not take a trade "
        "from the running candle. The bot will "
        "analyze the completed candle after 00:00."
    )


# =========================================================
# AUTOMATIC NEXT-CANDLE SIGNAL
# =========================================================

@st.fragment(
    run_every="1s"
)
def automatic_signal_fragment():

    if not st.session_state.armed:

        return


    armed_candle = (
        st.session_state.armed_candle
    )


    if armed_candle is None:

        return


    now = datetime.now(
        timezone.utc
    )


    candle_close = (
        armed_candle
        + timedelta(
            seconds=candle_seconds
        )
    )


    # Wait until the exact armed candle closes
    if now < candle_close:

        return


        # -----------------------------------------
    # Small provider-delay protection
    # -----------------------------------------

    result, message = generate_next_signal(
        market,
        pair,
        candle_timeframe_name,
        candle_seconds,
        expiry
    )

    if result is None:
        st.warning(
            "⏳ Waiting for completed candle data..."
        )
        return
