import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone, timedelta


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="REAL + OTC SIGNAL BOT",
    page_icon="🤖",
    layout="wide"
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
    "CHFJPY",
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
    "BTCUSD_otc",
]


TIMEFRAMES = {
    "1 Minute": 60,
    "5 Minutes": 300,
    "15 Minutes": 900,
    "30 Minutes": 1800,
    "1 Hour": 3600,
    "4 Hours": 14400,
}


CHART_TIMEFRAMES = [
    "Same as Candle",
    "1 Minute",
    "5 Minutes",
    "15 Minutes",
    "30 Minutes",
    "1 Hour",
    "4 Hours",
]


EXPIRIES = [
    "1 Candle",
    "2 Candles",
    "3 Candles",
]


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "last_signal": None,
    "signal_history": [],
    "last_signal_time": None,
    "armed": False,
    "armed_candle": None,
    "last_processed_candle": None,
    "signal_for_candle": None,
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def floor_time(dt, seconds):

    timestamp = int(dt.timestamp())

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

    return f"{minutes:02d}:{secs:02d}"


def candle_times_from_clock(
    candle_seconds
):

    now = datetime.now(
        timezone.utc
    )

    current_start = floor_time(
        now,
        candle_seconds
    )

    current_end = (
        current_start
        + timedelta(
            seconds=candle_seconds
        )
    )

    next_start = current_end

    next_end = (
        next_start
        + timedelta(
            seconds=candle_seconds
        )
    )

    remaining = (
        current_end - now
    ).total_seconds()

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
        return "-"

    return dt.strftime(
        "%H:%M:%S UTC"
    )


def timeframe_seconds_from_name(
    name
):

    return TIMEFRAMES.get(
        name,
        60
    )


# =========================================================
# DATA - REAL
# =========================================================

@st.cache_data(
    ttl=5,
    show_spinner=False
)
def get_real_candles(
    pair,
    candle_seconds
):

    try:

        url = (
            f"https://biquote.io/api/"
            f"{pair}/ohlc"
        )

        response = requests.get(
            url,
            params={
                "timeframe": "1m",
                "limit": 500
            },
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict):

            if "data" in data:
                data = data["data"]

            elif "candles" in data:
                data = data["candles"]

            elif "result" in data:
                data = data["result"]


        if not isinstance(
            data,
            list
        ):

            return pd.DataFrame()


        df = pd.DataFrame(data)

        if df.empty:
            return df


        # -----------------------------------------
        # Find timestamp column
        # -----------------------------------------

        time_col = None

        for col in [
            "timestamp",
            "time",
            "datetime",
            "date",
            "openTime",
        ]:

            if col in df.columns:

                time_col = col
                break


        if time_col is None:
            return pd.DataFrame()


        df["time"] = pd.to_datetime(
            df[time_col],
            unit="s",
            errors="coerce",
            utc=True
        )


        # Some providers return milliseconds
        if df["time"].isna().all():

            df["time"] = pd.to_datetime(
                df[time_col],
                unit="ms",
                errors="coerce",
                utc=True
            )


        # -----------------------------------------
        # OHLC columns
        # -----------------------------------------

        rename_map = {}

        for source, target in [
            ("open", "open"),
            ("o", "open"),
            ("high", "high"),
            ("h", "high"),
            ("low", "low"),
            ("l", "low"),
            ("close", "close"),
            ("c", "close"),
        ]:

            if source in df.columns:
                rename_map[source] = target


        df = df.rename(
            columns=rename_map
        )


        required = [
            "time",
            "open",
            "high",
            "low",
            "close",
        ]

        if not all(
            x in df.columns
            for x in required
        ):

            return pd.DataFrame()


        for col in [
            "open",
            "high",
            "low",
            "close",
        ]:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


        df = df.dropna(
            subset=required
        )


        df = df.sort_values(
            "time"
        )

        df = df.drop_duplicates(
            "time"
        )

        df = df.set_index(
            "time"
        )


        # -----------------------------------------
        # Only completed 1-minute candles
        # -----------------------------------------

        now = datetime.now(
            timezone.utc
        )

        df = df[
            df.index
            + timedelta(minutes=1)
            <= now
        ]


        if df.empty:
            return df


        # -----------------------------------------
        # Resample
        # -----------------------------------------

        rule_map = {
            60: "1min",
            300: "5min",
            900: "15min",
            1800: "30min",
            3600: "1h",
            14400: "4h",
        }

        rule = rule_map.get(
            candle_seconds,
            "1min"
        )


        if candle_seconds == 60:

            result = df[
                required[1:]
            ].copy()

        else:

            result = df[
                required[1:]
            ].resample(
                rule,
                label="left",
                closed="left"
            ).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            })


            result = result.dropna()


        return result.tail(500)


    except Exception:

        return pd.DataFrame()


# =========================================================
# DATA - OTC
# =========================================================

@st.cache_data(
    ttl=5,
    show_spinner=False
)
def get_otc_candles(
    pair,
    candle_seconds
):

    if not OTCHARTS_API_KEY:

        return pd.DataFrame()


    try:

        url = (
            "https://otcharts.com/"
            "v1/candles"
        )


        headers = {
            "Authorization":
                f"Bearer {OTCHARTS_API_KEY}",
            "Accept":
                "application/json",
        }


        params = {
            "venue": "otc",
            "symbol": pair,
            "timeframe":
                f"{candle_seconds}s",
            "limit": 500,
        }


        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=15
        )


        response.raise_for_status()


        data = response.json()


        if isinstance(data, dict):

            if "data" in data:
                data = data["data"]

            elif "candles" in data:
                data = data["candles"]

            elif "result" in data:
                data = data["result"]


        if not isinstance(
            data,
            list
        ):

            return pd.DataFrame()


        df = pd.DataFrame(data)


        if df.empty:
            return df


        # -----------------------------------------
        # Timestamp
        # -----------------------------------------

        time_col = None

        for col in [
            "timestamp",
            "time",
            "datetime",
            "date",
            "openTime",
        ]:

            if col in df.columns:

                time_col = col
                break


        if time_col is None:
            return pd.DataFrame()


        numeric_time = pd.to_numeric(
            df[time_col],
            errors="coerce"
        )


        # Detect milliseconds
        if (
            numeric_time.dropna().max()
            > 100000000000
        ):

            df["time"] = pd.to_datetime(
                numeric_time,
                unit="ms",
                errors="coerce",
                utc=True
            )

        else:

            df["time"] = pd.to_datetime(
                numeric_time,
                unit="s",
                errors="coerce",
                utc=True
            )


        # -----------------------------------------
        # OHLC
        # -----------------------------------------

        rename_map = {}

        for source, target in [
            ("open", "open"),
            ("o", "open"),
            ("high", "high"),
            ("h", "high"),
            ("low", "low"),
            ("l", "low"),
            ("close", "close"),
            ("c", "close"),
        ]:

            if source in df.columns:
                rename_map[source] = target


        df = df.rename(
            columns=rename_map
        )


        required = [
            "time",
            "open",
            "high",
            "low",
            "close",
        ]


        if not all(
            x in df.columns
            for x in required
        ):

            return pd.DataFrame()


        for col in [
            "open",
            "high",
            "low",
            "close",
        ]:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


        df = df.dropna(
            subset=required
        )


        df = df.sort_values(
            "time"
        )

        df = df.drop_duplicates(
            "time"
        )

        df = df.set_index(
            "time"
        )


        # -----------------------------------------
        # Only completed candles
        # -----------------------------------------

        now = datetime.now(
            timezone.utc
        )

        df = df[
            df.index
            + timedelta(
                seconds=candle_seconds
            )
            <= now
        ]


        return df.tail(500)


    except Exception:

        return pd.DataFrame()


# =========================================================
# GET CANDLES
# =========================================================

def get_candles(
    market,
    pair,
    candle_seconds
):

    if market == "REAL":

        return get_real_candles(
            pair,
            candle_seconds
        )

    return get_otc_candles(
        pair,
        candle_seconds
    )


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(df):

    df = df.copy()


    if len(df) < 30:
        return df


    df["ema9"] = (
        df["close"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )


    df["ema21"] = (
        df["close"]
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
    )


    df["ema50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )


    delta = df["close"].diff()


    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )


    avg_gain = (
        gain.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
    )


    avg_loss = (
        loss.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
    )


    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan
        )
    )


    df["rsi"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )


    df["rsi"] = df["rsi"].fillna(50)


    return df


# =========================================================
# SIGNAL ENGINE
# =========================================================

def calculate_signal(df):

    if df is None:
        return None


    if len(df) < 30:
        return None


    df = calculate_indicators(
        df
    )


    row = df.iloc[-1]


    close = float(
        row["close"]
    )

    ema9 = float(
        row["ema9"]
    )

    ema21 = float(
        row["ema21"]
    )

    ema50 = float(
        row["ema50"]
    )

    rsi = float(
        row["rsi"]
    )


    call_score = 0
    put_score = 0


    # -----------------------------------------
    # EMA 9 vs EMA 21
    # -----------------------------------------

    if ema9 > ema21:
        call_score += 1

    elif ema9 < ema21:
        put_score += 1


    # -----------------------------------------
    # Price vs EMA 9
    # -----------------------------------------

    if close > ema9:
        call_score += 1

    elif close < ema9:
        put_score += 1


    # -----------------------------------------
    # Price vs EMA 50
    # -----------------------------------------

    if close > ema50:
        call_score += 1

    elif close < ema50:
        put_score += 1


    # -----------------------------------------
    # RSI
    # -----------------------------------------

    if rsi >= 50:

        call_score += 1

    else:

        put_score += 1


    # -----------------------------------------
    # RSI strength
    # -----------------------------------------

    if rsi >= 55:

        call_score += 1

    elif rsi <= 45:

        put_score += 1


    # -----------------------------------------
    # Last candle direction
    # -----------------------------------------

    candle_open = float(
        row["open"]
    )


    if close > candle_open:

        call_score += 1

    elif close < candle_open:

        put_score += 1


    # -----------------------------------------
    # EMA trend confirmation
    # -----------------------------------------

    if (
        ema9 > ema21
        and ema21 > ema50
    ):

        call_score += 1


    elif (
        ema9 < ema21
        and ema21 < ema50
    ):

        put_score += 1


    # -----------------------------------------
    # Final signal
    # -----------------------------------------

    if (
        call_score > put_score
        and call_score >= 5
    ):

        signal = "CALL"

    elif (
        put_score > call_score
        and put_score >= 5
    ):

        signal = "PUT"

    else:

        signal = "WAIT"


    score = max(
        call_score,
        put_score
    )


    return {
        "signal": signal,
        "score": score,
        "price": close,
        "rsi": rsi,
        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,
    }


# =========================================================
# GENERATE NEXT CANDLE SIGNAL
# =========================================================

def generate_next_signal(
    market,
    pair,
    candle_timeframe_name,
    candle_seconds,
    expiry
):

    df = get_candles(
        market,
        pair,
        candle_seconds
    )


    if df is None or df.empty:

        return (
            None,
            "No candle data available."
        )


    df = calculate_indicators(
        df
    )


    if len(df) < 30:

        return (
            None,
            "Not enough candle data."
        )


    # -----------------------------------------
    # Last completed candle
    # -----------------------------------------

    source_candle = df.index[-1]


    if not isinstance(
        source_candle,
        datetime
    ):

        source_candle = pd.Timestamp(
            source_candle
        ).to_pydatetime()


    if source_candle.tzinfo is None:

        source_candle = (
            source_candle.replace(
                tzinfo=timezone.utc
            )
        )

    else:

        source_candle = (
            source_candle.astimezone(
                timezone.utc
            )
        )


    # -----------------------------------------
    # Calculate signal
    # -----------------------------------------

    result = calculate_signal(
        df
    )


    if result is None:

        return (
            None,
            "Unable to calculate signal."
        )


    next_candle_start = (
        source_candle
        + timedelta(
            seconds=candle_seconds
        )
    )


    result["source_candle_time"] = (
        source_candle
    )

    result["next_candle_start"] = (
        next_candle_start
    )


    return (
        result,
        "OK"
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
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


st.sidebar.divider()

st.sidebar.subheader(
    "⏱️ Candle Settings"
)


candle_timeframe_name = (
    st.sidebar.selectbox(
        "Candle Timeframe",
        list(TIMEFRAMES.keys())
    )
)


candle_seconds = (
    timeframe_seconds_from_name(
        candle_timeframe_name
    )
)


st.sidebar.divider()

st.sidebar.subheader(
    "📊 Chart Settings"
)


chart_timeframe_name = (
    st.sidebar.selectbox(
        "Chart Timeframe",
        CHART_TIMEFRAMES
    )
)


st.sidebar.divider()

st.sidebar.subheader(
    "🎯 Expiry"
)


expiry = st.sidebar.selectbox(
    "Expiry",
    EXPIRIES
)


if st.sidebar.button(
    "🔄 Refresh Data",
    use_container_width=True
):

    st.cache_data.clear()

    st.rerun()


# =========================================================
# HEADER
# =========================================================

st.title(
    "🤖 REAL + OTC SIGNAL BOT"
)


st.caption(
    "Analysis only | Auto Trading Disabled | "
    "Demo/Paper Testing"
)


# =========================================================
# BOT INFO
# =========================================================

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
    "EX
