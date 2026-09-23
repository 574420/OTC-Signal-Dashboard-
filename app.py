import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

try:
    from otcharts import Client
except Exception:
    Client = None

st.set_page_config(
    page_title="OTC Signal Dashboard",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# TIMEFRAMES
# =========================================================

TF_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

DEFAULT_REAL = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "EURGBP",
    "EURJPY",
]

DEFAULT_OTC = [
    "EURUSD_otc",
    "GBPUSD_otc",
    "USDJPY_otc",
    "AUDUSD_otc",
    "USDCAD_otc",
    "USDCHF_otc",
    "EURGBP_otc",
    "EURJPY_otc",
]

# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "client": None,
    "real_symbols": DEFAULT_REAL.copy(),
    "otc_symbols": DEFAULT_OTC.copy(),
    "armed": False,
    "armed_candle": None,
    "last_signal": None,
    "last_signal_key": None,
    "history": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================================================
# CLIENT
# =========================================================

def get_client():
    if st.session_state.client is None:

        if Client is None:
            raise RuntimeError(
                "otcharts package is not installed."
            )

        st.session_state.client = Client()

    return st.session_state.client


# =========================================================
# TIME HELPERS
# =========================================================

def candle_start(dt_utc, seconds):

    epoch = int(dt_utc.timestamp())

    start = epoch - (
        epoch % seconds
    )

    return datetime.fromtimestamp(
        start,
        tz=timezone.utc
    )


def candle_times(tf_name):

    now = datetime.now(timezone.utc)

    seconds = TF_SECONDS[tf_name]

    start = candle_start(
        now,
        seconds
    )

    end = (
        start
        + timedelta(
            seconds=seconds
        )
    )

    remaining = max(
        0,
        int(
            (
                end - now
            ).total_seconds()
        )
    )

    return (
        now,
        start,
        end,
        remaining
    )


def fmt_time(dt):

    return dt.astimezone(
        timezone.utc
    ).strftime(
        "%H:%M:%S UTC"
    )


# =========================================================
# NORMALIZE CANDLES
# =========================================================

def normalize_bars(bars):

    rows = []

    for b in bars:

        if isinstance(b, dict):

            t = b.get(
                "time",
                b.get(
                    "openTime",
                    b.get(
                        "timestamp"
                    )
                )
            )

            o = b.get("open")
            h = b.get("high")
            l = b.get("low")
            c = b.get("close")

            v = b.get(
                "volume",
                b.get(
                    "tickVolume",
                    0
                )
            )

        else:

            t = getattr(
                b,
                "time",
                getattr(
                    b,
                    "openTime",
                    None
                )
            )

            o = getattr(
                b,
                "open",
                None
            )

            h = getattr(
                b,
                "high",
                None
            )

            l = getattr(
                b,
                "low",
                None
            )

            c = getattr(
                b,
                "close",
                None
            )

            v = getattr(
                b,
                "volume",
                getattr(
                    b,
                    "tickVolume",
                    0
                )
            )

        if (
            t is None
            or o is None
            or h is None
            or l is None
            or c is None
        ):
            continue

        try:

            if isinstance(
                t,
                (
                    int,
                    float,
                    np.integer,
                    np.floating
                )
            ):

                ts = pd.to_datetime(
                    t,
                    unit="s",
                    utc=True
                )

            else:

                ts = pd.to_datetime(
                    t,
                    utc=True
                )

            rows.append(
                {
                    "time": ts,
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v or 0),
                }
            )

        except Exception:

            continue

    if not rows:

        return pd.DataFrame()

    df = pd.DataFrame(
        rows
    )

    df = df.drop_duplicates(
        subset=["time"]
    )

    df = df.sort_values(
        "time"
    )

    df = df.reset_index(
        drop=True
    )

    return df


# =========================================================
# FETCH DATA
# =========================================================

def fetch_bars(
    market,
    pair,
    tf_name,
    limit=200
):

    client = get_client()

    if market == "REAL":

        venue = "forex"

    else:

        venue = "quotex"

    bars = client.candles(
        venue,
        pair,
        tf=TF_SECONDS[tf_name],
        limit=limit
    )

    df = normalize_bars(
        bars
    )

    return df


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):

    x = df.copy()

    x["ema9"] = (
        x["close"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    x["ema21"] = (
        x["close"]
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
    )

    x["ema50"] = (
        x["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    delta = x["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(14)
        .mean()
    )

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    x["rsi"] = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    x["body"] = (
        x["close"]
        -
        x["open"]
    )

    x["range"] = (
        x["high"]
        -
        x["low"]
    ).replace(
        0,
        np.nan
    )

    x["body_pct"] = (
        x["body"].abs()
        /
        x["range"]
    )

    return x


# =========================================================
# SIGNAL ENGINE
# =========================================================

def signal_from_df(df):

    if len(df) < 60:

        return (
            None,
            "Not enough completed candles."
        )

    x = add_indicators(
        df
    )

    last = x.iloc[-1]

    prev = x.iloc[-2]

    score = 0

    reasons = []

    # EMA 9 / 21

    if last["ema9"] > last["ema21"]:

        score += 1

        reasons.append(
            "EMA9 > EMA21"
        )

    elif last["ema9"] < last["ema21"]:

        score -= 1

        reasons.append(
            "EMA9 < EMA21"
        )

    # EMA 21 / 50

    if last["ema21"] > last["ema50"]:

        score += 1

        reasons.append(
            "EMA21 > EMA50"
        )

    elif last["ema21"] < last["ema50"]:

        score -= 1

        reasons.append(
            "EMA21 < EMA50"
        )

    # RSI

    if last["rsi"] >= 55:

        score += 1

        reasons.append(
            "RSI bullish"
        )

    elif last["rsi"] <= 45:

        score -= 1

        reasons.append(
            "RSI bearish"
        )

    # Candle direction

    if last["close"] > last["open"]:

        score += 1

        reasons.append(
            "Last candle bullish"
        )

    elif last["close"] < last["open"]:

        score -= 1

        reasons.append(
            "Last candle bearish"
        )

    # Previous close comparison

    if last["close"] > prev["close"]:

        score += 1

    elif last["close"] < prev["close"]:

        score -= 1

    # Final signal

    if score >= 3:

        direction = "CALL"

    elif score <= -3:

        direction = "PUT"

    else:

        direction = "WAIT"

    strength = min(
        100,
        50 + abs(score) * 10
    )

    return (
        {
            "direction": direction,
            "score": int(score),
            "strength": int(strength),
            "rsi": (
                float(last["rsi"])
                if pd.notna(last["rsi"])
                else None
            ),
            "close": float(
                last["close"]
            ),
            "candle_time": last["time"],
            "reasons": reasons,
        },
        None
    )


# =========================================================
# GENERATE SIGNAL
# =========================================================

def generate_signal(
    market,
    pair,
    tf_name
):

    df = fetch_bars(
        market,
        pair,
        tf_name,
        200
    )

    if df.empty:

        return (
            None,
            "No candle data returned."
        )

    # Remove current running candle.
    if len(df) > 1:

        df = df.iloc[:-1].copy()

    result, message = signal_from_df(
        df
    )

    if result is None:

        return (
            None,
            message
        )

    result["market"] = market

    result["pair"] = pair

    result["timeframe"] = tf_name

    return (
        result,
        None
    )


# =========================================================
# REFRESH SYMBOL CATALOGUE
# =========================================================

def refresh_symbol_catalog():

    client = get_client()

    real = []

    otc = []

    try:

        for item in client.symbols(
            "forex"
        ):

            name = (
                getattr(
                    item,
                    "symbol",
                    None
                )
                or
                getattr(
                    item,
                    "name",
                    None
                )
            )

            if name:

                real.append(
                    name
                )

    except Exception:

        real = []

    try:

        for item in client.symbols(
            "quotex"
        ):

            name = (
                getattr(
                    item,
                    "symbol",
                    None
                )
                or
                getattr(
                    item,
                    "name",
                    None
                )
            )

            if name:

                otc.append(
                    name
                )

    except Exception:

        otc = []

    if real:

        st.session_state.real_symbols = sorted(
            set(real)
        )

    if otc:

        st.session_state.otc_symbols = sorted(
            set(otc)
        )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "⚙️ Settings"
)

market = st.sidebar.selectbox(
    "Market",
    [
        "REAL",
        "OTC"
    ]
)

if market == "REAL":

    pair_list = (
        st.session_state.real_symbols
    )

else:

    pair_list = (
        st.session_state.otc_symbols
    )

if not pair_list:

    if market == "REAL":

        pair_list = DEFAULT_REAL

    else:

        pair_list = DEFAULT_OTC

pair = st.sidebar.selectbox(
    "Pair",
    pair_list
)

candle_tf = st.sidebar.selectbox(
    "Candle timeframe",
    list(
        TF_SECONDS.keys()
    ),
    index=1
)

chart_tf = st.sidebar.selectbox(
    "Chart timeframe",
    list(
        TF_SECONDS.keys()
    ),
    index=1
)

expiry = st.sidebar.selectbox(
    "Expiry",
    [
        1,
        2,
        3,
        5,
        10,
        15
    ],
    index=0,
    format_func=lambda x:
        f"{x} minute"
)

st.sidebar.divider()

if st.sidebar.button(
    "🔄 Refresh pair catalogue",
    use_container_width=True
):

    try:

        refresh_symbol_catalog()

        st.sidebar.success(
            "Pair catalogue refreshed."
        )

    except Exception as e:

        st.sidebar.error(
            f"Could not refresh catalogue: {e}"
        )

st.sidebar.caption(
    "Analysis/signals only — no automatic trading."
)

# =========================================================
# HEADER
# =========================================================

st.title(
    "📈 REAL + OTC Signal Dashboard"
)

st.caption(
    "Completed-candle analysis only. No auto-trading."
)

# =========================================================
# LIVE CLOCK
# =========================================================

@st.fragment(
    run_every="1s"
)
def clock_fragment():

    (
        now,
        start,
        end,
        remaining
    ) = candle_times(
        candle_tf
    )

    st.metric(
        "Current UTC",
        fmt_time(now),
        f"Next candle in {remaining}s"
    )

    progress = (
        1
        -
        (
            remaining
            /
            TF_SECONDS[candle_tf]
        )
    )

    progress = max(
        0.0,
        min(
            1.0,
            progress
        )
    )

    st.progress(
        progress
    )


clock_fragment()

# =========================================================
# MARKET INFO
# =========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Market",
    market
)

col2.metric(
    "Pair",
    pair
)

col3.metric(
    "Candle TF",
    candle_tf
)

col4.metric(
    "Expiry",
    f"{expiry} min"
)

# =========================================================
# ARM SIGNAL
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
        remaining
    ) = candle_times(
        candle_tf
    )

    # If candle is almost closed,
    # use the next candle.

    if remaining <= 2:

        target = current_end

    else:

        target = current_start

    st.session_state.armed = True

    st.session_state.armed_candle = target

    st.session_state.last_signal = None

    st.session_state.last_signal_key = None

    st.success(
        f"🟢 Armed. Signal will be generated "
        f"after candle {fmt_time(target)} closes."
    )

if st.session_state.armed:

    st.info(
        "🟢 SIGNAL ARMED — wait for the selected "
        "candle to complete. The running candle "
        "is NOT used for the signal."
    )

# =========================================================
# AUTOMATIC SIGNAL
# =========================================================

@st.fragment(
    run_every="1s"
)
def automatic_signal_fragment():

    if not st.session_state.armed:

        return

    target = (
        st.session_state.armed_candle
    )

    if target is None:

        return

    now = datetime.now(
        timezone.utc
    )

    close_time = (
        target
        +
        timedelta(
            seconds=TF_SECONDS[candle_tf]
        )
    )

    if now < close_time:

        remaining = int(
            (
                close_time - now
            ).total_seconds()
        )

        st.info(
            f"⏳ Waiting for candle close: "
            f"{remaining}s"
        )

        return

    result, message = generate_signal(
        market,
        pair,
        candle_tf
    )

    if result is None:

        st.warning(
            "⏳ Waiting for completed candle data..."
        )

        st.caption(
            message
        )

        return

    source_key = (
        f"{market}|"
        f"{pair}|"
        f"{candle_tf}|"
        f"{pd.Timestamp(result['candle_time']).isoformat()}"
    )

    if (
        source_key
        ==
        st.session_state.last_signal_key
    ):

        return

    st.session_state.last_signal_key = source_key

    st.session_state.last_signal = result

    st.session_state.armed = False

    history_item = {
        "time":
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
        "market":
            result["market"],
        "pair":
            result["pair"],
        "tf":
            result["timeframe"],
        "signal":
            result["direction"],
        "strength":
            result["strength"],
        "score":
            result["score"],
    }

    st.session_state.history.insert(
        0,
        history_item
    )

    st.session_state.history = (
        st.session_state.history[:50]
    )

    if result["direction"] == "CALL":

        st.success(
            f"🟢 CALL | {result['pair']} | "
            f"Strength {result['strength']}% | "
            f"Expiry {expiry}m"
        )

    elif result["direction"] == "PUT":

        st.error(
            f"🔴 PUT | {result['pair']} | "
            f"Strength {result['strength']}% | "
            f"Expiry {expiry}m"
        )

    else:

        st.warning(
            f"🟡 WAIT | {result['pair']} | "
            f"Strength {result['strength']}%"
        )

    completed_time = pd.Timestamp(
        result["candle_time"]
    ).to_pydatetime()

    st.write(
        "Completed candle: "
        + fmt_time(
            completed_time
        )
    )

    if result["rsi"] is not None:

        st.write(
            f"RSI: {result['rsi']:.2f} | "
            f"Close: {result['close']}"
        )

    if result["reasons"]:

        st.caption(
            " | ".join(
                result["reasons"]
            )
        )


automatic_signal_fragment()

# =========================================================
# LAST SIGNAL
# =========================================================

st.divider()

st.subheader(
    "📌 Last Signal"
)

if st.session_state.last_signal is None:

    st.info(
        "No signal yet. Press "
        "ARM NEXT-CANDLE SIGNAL."
    )

else:

    r = (
        st.session_state.last_signal
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Signal",
        r["direction"]
    )

    b.metric(
        "Strength",
        f"{r['strength']}%"
    )

    c.metric(
        "Score",
        r["score"]
    )

    d.metric(
        "RSI",
        (
            f"{r['rsi']:.2f}"
            if r["rsi"] is not None
            else "N/A"
        )
    )

# =========================================================
# CHART
# =========================================================

st.divider()

st.subheader(
    "📊 Chart"
)

try:

    chart_df = fetch_bars(
        market,
        pair,
        chart_tf,
        120
    )

    if not chart_df.empty:

        st.line_chart(
            chart_df.set_index(
                "time"
            )["close"],
            height=350
        )

    else:

        st.warning(
            "No chart data available."
        )

except Exception as
