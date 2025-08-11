import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
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

@st.cache_data(ttl=300)
def get_stock_data(symbol, period="1mo"):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_market_overview():
    """Get basic market data (Global + Indian indices)"""
    indices = {
        # Global
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'VIX': '^VIX',
        # Indian
        'NIFTY 50': '^NSEI',
        'NIFTY BANK': '^NSEBANK',
        'SENSEX': '^BSESN'
    }

    results = {}
    for name, ticker in indices.items():
        try:
            data = yf.download(ticker, period='2d', interval='1d', progress=False)
            if data is not None and len(data) >= 2:
                current = data['Close'].iloc[-1]
                prev = data['Close'].iloc[-2]
                change_pct = ((current - prev) / prev) * 100 if prev != 0 else 0
                results[name] = {'price': current, 'change': change_pct}
            else:
                results[name] = {'price': 0, 'change': 0}
        except Exception:
            results[name] = {'price': 0, 'change': 0}
    return results

# -------------------
# Main app
# -------------------

def main():
    st.title("📈 Trading Platform Lite")
    st.markdown("*Fast and simple trading analysis*")

    st.subheader("Market Overview")
    market_data = get_market_overview()
    cols = st.columns(len(market_data))
    for i, (name, data) in enumerate(market_data.items()):
        arrow = "↗" if data['change'] >= 0 else "↘"
        cols[i].metric(name, f"{data['price']:.2f}", f"{arrow} {data['change']:.2f}%")

    st.subheader("Quick Stock Analysis")
    col1, col2 = st.columns([3, 1])

    with col2:
        popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'SPY', 'QQQ']
        symbol = st.selectbox("Select Stock", popular_stocks)
        st.markdown("**Quick Analysis:**")
        period = st.radio("Period", ["1d", "5d", "1mo", "3mo"], index=2, horizontal=True)

        st.markdown("**Position Size Calculator:**")
        account_size = st.number_input("Account Size ($)", value=10000, step=1000)
        risk_percent = st.slider("Risk %", 1, 5, 2)
        stop_loss_percent = st.slider("Stop Loss %", 1, 10, 3)

        if symbol and account_size > 0 and stop_loss_percent > 0:
            risk_amount = account_size * (risk_percent / 100)
            position_size = risk_amount / (stop_loss_percent / 100)
            st.info(f"**Position Size:** ${position_size:,.0f}")
            st.info(f"**Risk Amount:** ${risk_amount:,.0f}")

    with col1:
        if symbol:
            stock_data = get_stock_data(symbol, period)
            if stock_data is not None and not stock_data.empty:
                current_price = stock_data['Close'].iloc[-1]
                prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
                price_change_pct = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0

                st.markdown(f"### {symbol}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"${current_price:.2f}", f"{price_change_pct:.2f}%")
                c2.metric("High", f"${stock_data['High'].iloc[-1]:.2f}")
                c3.metric("Low", f"${stock_data['Low'].iloc[-1]:.2f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Price', line=dict(width=2)))
                if len(stock_data) >= 20:
                    sma20 = calculate_sma(stock_data['Close'], 20)
                    fig.add_trace(go.Scatter(x=stock_data.index, y=sma20, mode='lines', name='SMA 20', line=dict(width=1)))
                if len(stock_data) >= 50:
                    sma50 = calculate_sma(stock_data['Close'], 50)
                    fig.add_trace(go.Scatter(x=stock_data.index, y=sma50, mode='lines', name='SMA 50', line=dict(width=1)))

                fig.update_layout(title=f"{symbol} Price Chart", height=400, showlegend=True, xaxis_title="Date", yaxis_title="Price ($)")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Quick Indicators:**")
                indicator_cols = st.columns(2)

                # RSI (safe extraction)
                if len(stock_data) >= 14:
                    rsi_series = calculate_rsi(stock_data['Close']).dropna()
                    if not rsi_series.empty:
                        current_rsi = float(rsi_series.iloc[-1])
                        if current_rsi < 30:
                            rsi_signal = "Oversold"
                        elif current_rsi > 70:
                            rsi_signal = "Overbought"
                        else:
                            rsi_signal = "Neutral"
                        indicator_cols[0].metric("RSI (14)", f"{current_rsi:.1f}", rsi_signal)
                    else:
                        indicator_cols[0].metric("RSI (14)", "N/A", "Insufficient data")
                else:
                    indicator_cols[0].metric("RSI (14)", "N/A", "Insufficient data")

                avg_volume = stock_data['Volume'].mean()
                current_volume = stock_data['Volume'].iloc[-1] if len(stock_data) >= 1 else 0
                volume_ratio = (current_volume / avg_volume) if avg_volume != 0 else 0
                volume_signal = "High" if volume_ratio > 1.5 else "Low" if volume_ratio < 0.5 else "Normal"
                indicator_cols[1].metric("Volume vs Avg", f"{volume_ratio:.1f}x", volume_signal)
            else:
                st.error(f"Could not fetch data for {symbol}")

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
    st.warning("⚠️ This is for educational purposes only. Always do your own research before making investment decisions.")

if __name__ == "__main__":
    main()
