import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Trading Platform Lite", page_icon="📈", layout="wide")

# -------------------
# Helper functions
# -------------------

def calculate_sma(series: pd.Series, window: int):
    return series.rolling(window=window).mean()

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Simple RSI calculation - returns a Series"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -------------------
# Auto refresh every minute
# -------------------
st_autorefresh(interval=60 * 1000, key="refresh")

# -------------------
# Market Overview
# -------------------
st.subheader("📊 Market Overview (Updated Every Minute)")
indices = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'NIFTY 50': '^NSEI',
    'SENSEX': '^BSESN'
}

cols = st.columns(len(indices))
for i, (name, ticker) in enumerate(indices.items()):
    data = yf.download(ticker, period="2d", interval="1d", progress=False)
    if len(data) >= 2:
    current = float(data['Close'].iloc[-1])
    prev = float(data['Close'].iloc[-2])
    change_pct = ((current - prev) / prev) * 100 if prev != 0 else 0
    arrow = "↗" if change_pct >= 0 else "↘"
    cols[i].metric(name, f"{current:.2f}", f"{arrow} {change_pct:.2f}%")

# -------------------
# Global Stock Search
# -------------------
st.subheader("🌍 Global Stock Search (1-min Data)")
symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS, TCS.NS, AAPL, TSLA):", "RELIANCE.NS")

if symbol:
    stock_data = yf.download(symbol, period="1d", interval="1m")
    if not stock_data.empty:
        current_price = stock_data['Close'].iloc[-1]
        prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
        price_change_pct = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0

        # Display metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"{current_price:.2f}", f"{price_change_pct:.2f}%")
        c2.metric("High", f"{stock_data['High'].iloc[-1]:.2f}")
        c3.metric("Low", f"{stock_data['Low'].iloc[-1]:.2f}")

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'],
                                 mode='lines', name='Price', line=dict(width=2)))
        if len(stock_data) >= 20:
            sma20 = calculate_sma(stock_data['Close'], 20)
            fig.add_trace(go.Scatter(x=stock_data.index, y=sma20, mode='lines', name='SMA 20'))
        if len(stock_data) >= 50:
            sma50 = calculate_sma(stock_data['Close'], 50)
            fig.add_trace(go.Scatter(x=stock_data.index, y=sma50, mode='lines', name='SMA 50'))

        # RSI
        if len(stock_data) >= 14:
            rsi_series = calculate_rsi(stock_data['Close']).dropna()
            if not rsi_series.empty:
                current_rsi = float(rsi_series.iloc[-1])
                rsi_status = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
                st.metric("RSI (14)", f"{current_rsi:.1f}", rsi_status)

        fig.update_layout(title=f"{symbol} - Intraday 1-min Chart",
                          height=400, showlegend=True,
                          xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data found. Check the symbol format.")

# -------------------
# Educational Tips
# -------------------
st.subheader("💡 Quick Trading Tips")
for tip in [
    "Always use stop losses to manage risk",
    "Never risk more than 2-3% of your account on a single trade",
    "RSI below 30 may indicate oversold conditions",
    "RSI above 70 may indicate overbought conditions",
    "High volume often confirms price movements"
]:
    st.markdown(f"• {tip}")

st.markdown("---")
st.warning("⚠️ This is for educational purposes only. Data is delayed ~15 mins for most exchanges.")
