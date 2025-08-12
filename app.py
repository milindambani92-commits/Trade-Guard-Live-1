# app.py
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Trading Platform Lite", page_icon="📈", layout="wide")

# ---------------------------
# Helper functions (safe)
# ---------------------------
def calculate_sma(series: pd.Series, window: int):
    return series.rolling(window=window).mean()

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_bollinger(series: pd.Series, window=20, n_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + (std * n_std)
    lower = sma - (std * n_std)
    return upper, lower

def safe_last(series):
    """Return float of last element or None"""
    try:
        return float(series.iloc[-1])
    except Exception:
        return None

def get_stock_data(symbol, period="1mo", interval="1d"):
    """Fetch data from yfinance; returns DataFrame or None"""
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data is None or data.empty:
            return None
        return data
    except Exception:
        return None

def find_support_resistance(series_close: pd.Series, window=5):
    supports = []
    resistances = []
    s = series_close.dropna()  # ensure no NaN
    for i in range(window, len(s) - window):
        chunk = s.iloc[i - window:i + window + 1]
        center = float(s.iloc[i])
        if center == float(chunk.max()):
            resistances.append(center)
        if center == float(chunk.min()):
            supports.append(center)
    supports = sorted(list(set(supports)))[-3:]
    resistances = sorted(list(set(resistances)))[-3:]
    return supports, resistances

def suggest_levels(current_price, supports, resistances):
    """
    Basic logic:
    - If price is above SMA20 -> bullish bias -> suggest long entry at or near current, take profit at next resistance, SL at nearest support.
    - If price below SMA20 -> bearish bias -> suggest short (we'll present symmetrical suggestions)
    """
    if current_price is None:
        return None
    # nearest support below price
    supports_below = [float(s) for s in supports if float(s) < current_price]
    supports_above = [float(s) for s in supports if float(s) > current_price]
    resistances_above = [float(r) for r in resistances if float(r) > current_price]
    resistances_below = [float(r) for r in resistances if float(r) < current_price]

    nearest_support = min(supports_below) if supports_below else None
    nearest_resistance = min(resistances_above) if resistances_above else (max(resistances) if resistances else None)

    return {
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance
    }

# ---------------------------
# Auto refresh (1 minute)
# ---------------------------
st_autorefresh(interval=60 * 1000, key="refresh")

# ---------------------------
# UI: selectors & inputs
# ---------------------------
st.title("📈 Trading Platform Lite — Guidance Mode (Free Data)")
st.markdown("Use for analysis and learning. Data is provided by Yahoo Finance and may be delayed (~15 min).")

col_left, col_right = st.columns([3, 1])

with col_right:
    st.markdown("**Settings**")
    symbol_input = st.text_input("Stock symbol (e.g., RELIANCE.NS, TCS.NS, AAPL)", value="RELIANCE.NS")
    timeframe = st.selectbox("Historical timeframe", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","max"], index=2)
    intraday_toggle = st.checkbox("Show intraday 1-min chart (today)", value=True)
    intraday_interval = "1m"
    indicator_toggle = st.multiselect("Indicators", ["SMA20","SMA50","RSI","MACD","Bollinger"], default=["SMA20","RSI"])
    # Risk settings
    st.markdown("**Risk Calculator**")
    account_size = st.number_input("Account size (currency)", value=100000.0, step=1000.0)
    risk_percent = st.slider("Risk % per trade", min_value=1, max_value=10, value=2)
    # Optional manual stop loss distance (percent)
    user_stop_loss_pct = st.number_input("Optional SL % from entry (0 to auto)", value=0.0, min_value=0.0, max_value=100.0, step=0.1)

with col_left:
    st.subheader("Market Overview")
    # indices to display
    indices = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'NIFTY 50': '^NSEI',
        'SENSEX': '^BSESN'
    }
    idx_cols = st.columns(len(indices))
    for i, (n, t) in enumerate(indices.items()):
        df_idx = get_stock_data(t, period="2d", interval="1d")
        if df_idx is not None and len(df_idx) >= 2:
            try:
                curr = safe_last(df_idx['Close'])
                prev = float(df_idx['Close'].iloc[-2])
                change_pct = ((curr - prev) / prev) * 100 if prev != 0 else 0
                arrow = "↗" if change_pct >= 0 else "↘"
                idx_cols[i].metric(n, f"{curr:.2f}", f"{arrow} {change_pct:.2f}%")
            except Exception:
                idx_cols[i].metric(n, "N/A", "Error")
        else:
            idx_cols[i].metric(n, "N/A", "No data")

st.markdown("---")

# ---------------------------
# Fetch data
# ---------------------------
symbol = symbol_input.strip().upper()
if not symbol:
    st.warning("Please enter a stock symbol (e.g., RELIANCE.NS or AAPL).")
    st.stop()

# get historical data (user-selected timeframe)
hist_interval = "1d"
hist_period = timeframe
stock_hist = get_stock_data(symbol, period=hist_period, interval=hist_interval)

# get intraday if requested (today 1-min)
stock_intraday = None
if intraday_toggle:
    stock_intraday = get_stock_data(symbol, period="1d", interval=intraday_interval)

if stock_hist is None and (stock_intraday is None):
    st.error("Could not fetch data for this symbol. Check the ticker format and try again.")
    st.stop()

# ---------------------------
# Show top-level metrics
# ---------------------------
latest_price = None
if stock_intraday is not None and len(stock_intraday) >= 1:
    latest_price = safe_last(stock_intraday['Close'])
elif stock_hist is not None and len(stock_hist) >= 1:
    latest_price = safe_last(stock_hist['Close'])

st.header(f"{symbol}  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if latest_price is not None:
    st.metric("Latest Price", f"{latest_price:.2f}")
else:
    st.metric("Latest Price", "N/A")

# ---------------------------
# Charts: Intraday and Historical
# ---------------------------
chart_col1, chart_col2 = st.columns([2,1])

with chart_col1:
    st.subheader("Price Chart")

    # Plot intraday (if available)
    if intraday_toggle and stock_intraday is not None and not stock_intraday.empty:
        df = stock_intraday.copy()
        # prepare fig
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Intraday"
        ))
        # indicators
        if "SMA20" in indicator_toggle and len(df)>=20:
            fig.add_trace(go.Scatter(x=df.index, y=calculate_sma(df['Close'],20), name="SMA20", line=dict(width=1)))
        if "SMA50" in indicator_toggle and len(df)>=50:
            fig.add_trace(go.Scatter(x=df.index, y=calculate_sma(df['Close'],50), name="SMA50", line=dict(width=1)))
        if "Bollinger" in indicator_toggle and len(df)>=20:
            upper, lower = calculate_bollinger(df['Close'], 20)
            fig.add_trace(go.Scatter(x=df.index, y=upper, name="BollingerUpper", line=dict(width=1), opacity=0.5))
            fig.add_trace(go.Scatter(x=df.index, y=lower, name="BollingerLower", line=dict(width=1), opacity=0.5))

        fig.update_layout(title=f"{symbol} Intraday (1-min)", height=500, xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Intraday 1-min chart not available or disabled.")

    # Full historical chart
    if stock_hist is not None and not stock_hist.empty:
        dfh = stock_hist.copy()
        fig2 = go.Figure()
        fig2.add_trace(go.Candlestick(
            x=dfh.index,
            open=dfh['Open'], high=dfh['High'], low=dfh['Low'], close=dfh['Close'], name="Historical"
        ))
        if "SMA20" in indicator_toggle and len(dfh)>=20:
            fig2.add_trace(go.Scatter(x=dfh.index, y=calculate_sma(dfh['Close'],20), name="SMA20", line=dict(width=1)))
        if "SMA50" in indicator_toggle and len(dfh)>=50:
            fig2.add_trace(go.Scatter(x=dfh.index, y=calculate_sma(dfh['Close'],50), name="SMA50", line=dict(width=1)))
        fig2.update_layout(title=f"{symbol} Historical ({hist_period})", height=450, xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig2, use_container_width=True)

with chart_col2:
    st.subheader("Indicators & Levels")
    # Use intraday if available for indicators, else use historical close
    source_df = stock_intraday if (stock_intraday is not None and not stock_intraday.empty) else stock_hist
    if source_df is None or source_df.empty:
        st.warning("Insufficient data for indicators.")
    else:
        close = source_df['Close']
        # RSI
        if "RSI" in indicator_toggle and len(close) >= 14:
            rsi_s = calculate_rsi(close).dropna()
            if not rsi_s.empty:
                current_rsi = float(rsi_s.iloc[-1])
                st.metric("RSI (14)", f"{current_rsi:.1f}", "Oversold" if current_rsi<30 else "Overbought" if current_rsi>70 else "Neutral")

        # MACD
        if "MACD" in indicator_toggle and len(close) >= 26:
            macd_line, sig_line, hist = calculate_macd(close)
            ml = safe_last(macd_line); sl = safe_last(sig_line); hh = safe_last(hist)
            if ml is not None and sl is not None:
                st.write(f"MACD: {ml:.4f}  Signal: {sl:.4f}  Hist: {hh:.4f}")

        # Support/Resistance
        supports, resistances = find_support_resistance(close, window=5)
        st.markdown("**Support (recent)**")
        if supports:
            for s in sorted(supports):
                st.write(f"- {float(s):.2f}")
        else:
            st.write("No clear supports found")

        st.markdown("**Resistance (recent)**")
        if resistances:
            for r in sorted(resistances):
                st.write(f"- {float(r):.2f}")
        else:
            st.write("No clear resistances found")

        # Suggest levels
        latest = safe_last(close)
        suggested = suggest_levels(latest, supports, resistances)
        if suggested:
            st.markdown("**Suggested Levels**")
            st.write(f"Nearest Support: {suggested.get('nearest_support')}")
            st.write(f"Nearest Resistance: {suggested.get('nearest_resistance')}")

# ---------------------------
# Risk & Trade suggestion panel
# ---------------------------
st.markdown("---")
st.subheader("Trade Sizing & Stop-Loss Suggestions")

if latest_price is None:
    st.info("No latest price available to calculate trade suggestions.")
else:
    # default entry as latest_price
    entry_price = st.number_input("Entry price", value=float(latest_price), format="%.4f")
    if user_stop_loss_pct > 0:
        stop_loss_price = entry_price * (1 - user_stop_loss_pct/100)
    else:
        # use nearest support as suggested stop loss if available
        sl = suggested.get("nearest_support") if suggested else None
        stop_loss_price = sl if sl is not None else entry_price * 0.99  # default 1% SL

    stop_loss_price = float(stop_loss_price) if stop_loss_price is not None else entry_price * 0.99

    st.write(f"Suggested Stop Loss: {stop_loss_price:.4f}")

    risk_amount = account_size * (risk_percent / 100.0)
    # per-share risk
    per_unit_risk = abs(entry_price - stop_loss_price)
    if per_unit_risk == 0:
        position_size = 0
    else:
        position_size = int(risk_amount / per_unit_risk)

    st.write(f"Risk Amount: {risk_amount:.2f}")
    st.write(f"Per-unit risk (entry - SL): {per_unit_risk:.4f}")
    st.write(f"Suggested Position Size: {position_size} units (rounded down)")

st.markdown("---")
st.write("⚠️ **Disclaimer:** This app is for educational/guidance purposes only. Do your own research before trading.")
