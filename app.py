import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Trading Platform Lite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple cache decorator for data
@st.cache_data(ttl=300)  # 5 minute cache
def get_stock_data(symbol, period="1mo"):
    """Fetch stock data with caching"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except:
        return None

@st.cache_data(ttl=300)
def get_market_overview():
    """Get basic market data"""
    indices = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC', 
        'VIX': '^VIX'
    }
    
    results = {}
    for name, ticker in indices.items():
        try:
            data = yf.download(ticker, period='2d', interval='1d', progress=False)
            if data is not None and len(data) >= 2:
                current = data['Close'].iloc[-1]
                prev = data['Close'].iloc[-2]
                change_pct = ((current - prev) / prev) * 100
                results[name] = {
                    'price': current,
                    'change': change_pct
                }
            else:
                results[name] = {'price': 0, 'change': 0}
        except:
            results[name] = {'price': 0, 'change': 0}
    
    return results

def calculate_sma(data, window):
    """Simple moving average"""
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Simple RSI calculation"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def main():
    # Header
    st.title("📈 Trading Platform Lite")
    st.markdown("*Fast and simple trading analysis*")
    
    # Market overview
    st.subheader("Market Overview")
    market_data = get_market_overview()
    
    cols = st.columns(len(market_data))
    for i, (name, data) in enumerate(market_data.items()):
        color = "green" if data['change'] >= 0 else "red"
        arrow = "↗" if data['change'] >= 0 else "↘"
        cols[i].metric(
            name,
            f"{data['price']:.2f}",
            f"{arrow} {data['change']:.2f}%"
        )
    
    # Stock analysis
    st.subheader("Quick Stock Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Popular stocks
        popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'SPY', 'QQQ']
        symbol = st.selectbox("Select Stock", popular_stocks)
        
        # Quick actions
        st.markdown("**Quick Analysis:**")
        period = st.radio("Period", ["1d", "5d", "1mo", "3mo"], index=2, horizontal=True)
        
        # Simple risk calculator
        st.markdown("**Position Size Calculator:**")
        account_size = st.number_input("Account Size ($)", value=10000, step=1000)
        risk_percent = st.slider("Risk %", 1, 5, 2)
        stop_loss_percent = st.slider("Stop Loss %", 1, 10, 3)
        
        if symbol and account_size > 0:
            risk_amount = account_size * (risk_percent / 100)
            position_size = risk_amount / (stop_loss_percent / 100)
            st.info(f"**Position Size:** ${position_size:,.0f}")
            st.info(f"**Risk Amount:** ${risk_amount:,.0f}")
    
    with col1:
        if symbol:
            stock_data = get_stock_data(symbol, period)
            
            if stock_data is not None and not stock_data.empty:
                # Current price info
                current_price = stock_data['Close'].iloc[-1]
                prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
                price_change = current_price - prev_price
                price_change_pct = (price_change / prev_price) * 100 if prev_price > 0 else 0
                
                # Price display
                st.markdown(f"### {symbol}")
                price_col1, price_col2, price_col3 = st.columns(3)
                
                with price_col1:
                    st.metric("Current Price", f"${current_price:.2f}", f"{price_change_pct:.2f}%")
                with price_col2:
                    st.metric("High", f"${stock_data['High'].iloc[-1]:.2f}")
                with price_col3:
                    st.metric("Low", f"${stock_data['Low'].iloc[-1]:.2f}")
                
                # Simple chart
                fig = go.Figure()
                
                # Price line
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='blue', width=2)
                ))
                
                # Add simple moving averages if enough data
                if len(stock_data) >= 20:
                    sma20 = calculate_sma(stock_data['Close'], 20)
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=sma20,
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='orange', width=1)
                    ))
                
                if len(stock_data) >= 50:
                    sma50 = calculate_sma(stock_data['Close'], 50)
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=sma50,
                        mode='lines',
                        name='SMA 50',
                        line=dict(color='red', width=1)
                    ))
                
                fig.update_layout(
                    title=f"{symbol} Price Chart",
                    height=400,
                    showlegend=True,
                    xaxis_title="Date",
                    yaxis_title="Price ($)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Simple technical indicators
                st.markdown("**Quick Indicators:**")
                
                indicator_cols = st.columns(2)
                
                # RSI
                if len(stock_data) >= 14:
                    rsi = calculate_rsi(stock_data['Close'])
                    if len(rsi.dropna()) > 0:
                        current_rsi = rsi.iloc[-1]
                        rsi_signal = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
                        indicator_cols[0].metric("RSI (14)", f"{current_rsi:.1f}", rsi_signal)
                
                # Volume
                avg_volume = stock_data['Volume'].mean()
                current_volume = stock_data['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume
                volume_signal = "High" if volume_ratio > 1.5 else "Low" if volume_ratio < 0.5 else "Normal"
                indicator_cols[1].metric("Volume vs Avg", f"{volume_ratio:.1f}x", volume_signal)
                
            else:
                st.error(f"Could not fetch data for {symbol}")
    
    # Quick tips
    st.subheader("💡 Quick Trading Tips")
    tips = [
        "Always use stop losses to manage risk",
        "Never risk more than 2-3% of your account on a single trade",
        "RSI below 30 may indicate oversold conditions",
        "RSI above 70 may indicate overbought conditions",
        "High volume often confirms price movements"
    ]
    
    for tip in tips:
        st.markdown(f"• {tip}")
    
    # Disclaimer
    st.markdown("---")
    st.warning("⚠️ This is for educational purposes only. Always do your own research before making investment decisions.")

if __name__ == "__main__":
    main()
