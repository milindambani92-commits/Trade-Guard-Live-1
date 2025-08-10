import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.data_fetcher import DataFetcher
from utils.technical_indicators import TechnicalIndicators

st.set_page_config(page_title="Market Analysis", page_icon="📊", layout="wide")

def main():
    st.title("📊 Advanced Market Analysis")
    st.markdown("Deep dive into market trends and technical analysis")
    
    # Initialize classes
    data_fetcher = DataFetcher()
    tech_indicators = TechnicalIndicators()
    
    # Sidebar controls
    st.sidebar.header("Analysis Settings")
    
    # Stock selection
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter stock symbol (e.g., AAPL, MSFT)")
    symbol = symbol.upper()
    
    # Time period selection
    period = st.sidebar.selectbox(
        "Time Period",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        index=2,
        help="Select the time period for analysis"
    )
    
    # Indicator selection
    st.sidebar.subheader("Technical Indicators")
    show_ma = st.sidebar.checkbox("Moving Averages", value=True)
    show_bb = st.sidebar.checkbox("Bollinger Bands", value=False)
    show_rsi = st.sidebar.checkbox("RSI", value=True)
    show_macd = st.sidebar.checkbox("MACD", value=True)
    show_stoch = st.sidebar.checkbox("Stochastic", value=False)
    show_volume = st.sidebar.checkbox("Volume", value=True)
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Fetch stock data
    try:
        stock_data = data_fetcher.get_stock_data(symbol, period)
        stock_info = data_fetcher.get_stock_info(symbol)
        
        if stock_data is None or stock_data.empty:
            st.error(f"No data found for symbol {symbol}. Please check the symbol and try again.")
            return
        
        # Display stock info
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = stock_data['Close'].iloc[-1]
        prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        with col1:
            st.metric(
                label=f"{symbol} Price",
                value=f"${current_price:.2f}",
                delta=f"{price_change:.2f} ({price_change_pct:.2f}%)"
            )
        
        with col2:
            if stock_info:
                market_cap = stock_info.get('marketCap', 'N/A')
                if isinstance(market_cap, (int, float)):
                    market_cap = f"${market_cap/1e9:.1f}B"
                st.metric("Market Cap", market_cap)
        
        with col3:
            volume = stock_data['Volume'].iloc[-1]
            avg_volume = stock_data['Volume'].rolling(20).mean().iloc[-1]
            volume_change = ((volume - avg_volume) / avg_volume) * 100 if avg_volume != 0 else 0
            st.metric(
                "Volume",
                f"{volume:,.0f}",
                f"{volume_change:.1f}% vs 20d avg"
            )
        
        with col4:
            high_52w = stock_data['High'].max()
            low_52w = stock_data['Low'].min()
            pct_from_high = ((current_price - high_52w) / high_52w) * 100
            st.metric("52W Range", f"${low_52w:.2f} - ${high_52w:.2f}", f"{pct_from_high:.1f}% from high")
        
        # Calculate technical indicators
        if show_ma:
            stock_data['MA_10'] = tech_indicators.moving_average(stock_data['Close'], 10)
            stock_data['MA_20'] = tech_indicators.moving_average(stock_data['Close'], 20)
            stock_data['MA_50'] = tech_indicators.moving_average(stock_data['Close'], 50)
            stock_data['MA_200'] = tech_indicators.moving_average(stock_data['Close'], 200)
        
        if show_bb:
            bb_upper, bb_middle, bb_lower = tech_indicators.bollinger_bands(stock_data['Close'])
            stock_data['BB_Upper'] = bb_upper
            stock_data['BB_Middle'] = bb_middle
            stock_data['BB_Lower'] = bb_lower
        
        rsi = None
        if show_rsi:
            rsi = tech_indicators.rsi(stock_data['Close'])
        
        macd_line, macd_signal, macd_histogram = None, None, None
        if show_macd:
            macd_line, macd_signal, macd_histogram = tech_indicators.macd(stock_data['Close'])
        
        stoch_k, stoch_d = None, None
        if show_stoch:
            stoch_k, stoch_d = tech_indicators.stochastic_oscillator(
                stock_data['High'], stock_data['Low'], stock_data['Close']
            )
        
        # Create subplots
        subplot_count = 1
        if show_volume:
            subplot_count += 1
        if show_rsi:
            subplot_count += 1
        if show_macd:
            subplot_count += 1
        if show_stoch:
            subplot_count += 1
        
        # Calculate subplot heights
        row_heights = [0.6]  # Main chart gets 60%
        remaining_height = 0.4 / (subplot_count - 1) if subplot_count > 1 else 0
        for _ in range(subplot_count - 1):
            row_heights.append(remaining_height)
        
        subplot_titles = [f'{symbol} Price Chart']
        if show_volume:
            subplot_titles.append('Volume')
        if show_rsi:
            subplot_titles.append('RSI')
        if show_macd:
            subplot_titles.append('MACD')
        if show_stoch:
            subplot_titles.append('Stochastic')
        
        fig = make_subplots(
            rows=subplot_count,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=subplot_titles,
            row_heights=row_heights
        )
        
        # Main price chart
        fig.add_trace(
            go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name=symbol,
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Moving averages
        if show_ma:
            colors = ['orange', 'blue', 'purple', 'brown']
            periods = [10, 20, 50, 200]
            for i, period in enumerate(periods):
                ma_col = f'MA_{period}'
                if ma_col in stock_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data.index,
                            y=stock_data[ma_col],
                            mode='lines',
                            name=f'MA {period}',
                            line=dict(color=colors[i], width=1),
                            opacity=0.8
                        ),
                        row=1, col=1
                    )
        
        # Bollinger Bands
        if show_bb and 'BB_Upper' in stock_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        current_row = 2
        
        # Volume
        if show_volume:
            colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in stock_data.iterrows()]
            fig.add_trace(
                go.Bar(
                    x=stock_data.index,
                    y=stock_data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=current_row, col=1
            )
            current_row += 1
        
        # RSI
        if show_rsi and rsi is not None:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=rsi,
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=current_row, col=1
            )
            
            # RSI overbought/oversold levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, row=current_row, col=1)
            
            current_row += 1
        
        # MACD
        if show_macd and macd_line is not None:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=macd_line,
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=2)
                ),
                row=current_row, col=1
            )
            
            if macd_signal is not None:
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=macd_signal,
                        mode='lines',
                        name='MACD Signal',
                        line=dict(color='red', width=1)
                    ),
                    row=current_row, col=1
                )
            
            if macd_histogram is not None:
                colors = ['green' if x >= 0 else 'red' for x in macd_histogram]
                fig.add_trace(
                    go.Bar(
                        x=stock_data.index,
                        y=macd_histogram,
                        name='MACD Histogram',
                        marker_color=colors,
                        opacity=0.6
                    ),
                    row=current_row, col=1
                )
            
            current_row += 1
        
        # Stochastic
        if show_stoch and stoch_k is not None:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stoch_k,
                    mode='lines',
                    name='%K',
                    line=dict(color='blue', width=2)
                ),
                row=current_row, col=1
            )
            
            if stoch_d is not None:
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=stoch_d,
                        mode='lines',
                        name='%D',
                        line=dict(color='red', width=1)
                    ),
                    row=current_row, col=1
                )
            
            # Stochastic levels
            fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.7, row=current_row, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.7, row=current_row, col=1)
            
            current_row += 1
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            title=f"{symbol} Technical Analysis - {period.upper()}"
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical Analysis Summary
        st.header("📈 Technical Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Current Indicators")
            
            if rsi is not None and not rsi.empty:
                current_rsi = rsi.iloc[-1]
                if current_rsi > 70:
                    st.error(f"🔴 RSI: {current_rsi:.1f} (Overbought)")
                elif current_rsi < 30:
                    st.success(f"🟢 RSI: {current_rsi:.1f} (Oversold)")
                else:
                    st.info(f"🟡 RSI: {current_rsi:.1f} (Neutral)")
            
            if macd_line is not None and macd_signal is not None:
                if len(macd_line) > 1 and len(macd_signal) > 1:
                    if macd_line.iloc[-1] > macd_signal.iloc[-1]:
                        st.success("🟢 MACD: Bullish")
                    else:
                        st.error("🔴 MACD: Bearish")
            
            # Moving average analysis
            if show_ma and len(stock_data) > 50:
                ma20 = stock_data['MA_20'].iloc[-1] if 'MA_20' in stock_data.columns else None
                ma50 = stock_data['MA_50'].iloc[-1] if 'MA_50' in stock_data.columns else None
                
                if ma20 and ma50:
                    if ma20 > ma50:
                        st.success("🟢 MA Trend: Bullish (20>50)")
                    else:
                        st.error("🔴 MA Trend: Bearish (20<50)")
        
        with col2:
            st.subheader("🎯 Support & Resistance")
            
            # Calculate support and resistance levels
            recent_data = stock_data['Close'].tail(50)  # Last 50 days
            support_levels, resistance_levels = tech_indicators.detect_support_resistance(recent_data)
            
            if resistance_levels:
                st.write("**Resistance Levels:**")
                for level in resistance_levels[:3]:  # Top 3
                    distance = ((level - current_price) / current_price) * 100
                    st.write(f"${level:.2f} ({distance:+.1f}%)")
            
            if support_levels:
                st.write("**Support Levels:**")
                for level in support_levels[:3]:  # Top 3
                    distance = ((level - current_price) / current_price) * 100
                    st.write(f"${level:.2f} ({distance:+.1f}%)")
        
        with col3:
            st.subheader("📋 Trading Signals")
            
            # Generate trading signals
            signals = tech_indicators.generate_signals(stock_data)
            
            if not signals.empty:
                latest_signals = signals.iloc[-1]
                
                if latest_signals.get('Buy_Signal', False):
                    st.success("🟢 **BUY SIGNAL DETECTED**")
                    
                    # Show which indicators are bullish
                    bullish_indicators = []
                    if latest_signals.get('RSI_Buy', False):
                        bullish_indicators.append("RSI")
                    if latest_signals.get('MACD_Buy', False):
                        bullish_indicators.append("MACD")
                    if latest_signals.get('MA_Buy', False):
                        bullish_indicators.append("Moving Averages")
                    
                    if bullish_indicators:
                        st.write(f"Bullish: {', '.join(bullish_indicators)}")
                
                elif latest_signals.get('Sell_Signal', False):
                    st.error("🔴 **SELL SIGNAL DETECTED**")
                    
                    # Show which indicators are bearish
                    bearish_indicators = []
                    if latest_signals.get('RSI_Sell', False):
                        bearish_indicators.append("RSI")
                    if latest_signals.get('MACD_Sell', False):
                        bearish_indicators.append("MACD")
                    if latest_signals.get('MA_Sell', False):
                        bearish_indicators.append("Moving Averages")
                    
                    if bearish_indicators:
                        st.write(f"Bearish: {', '.join(bearish_indicators)}")
                
                else:
                    st.info("🟡 **HOLD/NEUTRAL**")
        
        # Sector comparison
        st.header("🏢 Sector Performance")
        
        sector_data = data_fetcher.get_sector_performance()
        if sector_data:
            sector_performance = {}
            
            for sector, data in sector_data.items():
                if not data.empty and len(data) >= 2:
                    current = data['Close'].iloc[-1]
                    previous = data['Close'].iloc[0]
                    performance = ((current - previous) / previous) * 100
                    sector_performance[sector] = performance
            
            if sector_performance:
                # Create sector performance chart
                sectors = list(sector_performance.keys())
                performances = list(sector_performance.values())
                
                fig_sector = px.bar(
                    x=sectors,
                    y=performances,
                    color=performances,
                    color_continuous_scale=['red', 'white', 'green'],
                    title=f"Sector Performance - {period.upper()}",
                    labels={'y': 'Performance (%)', 'x': 'Sector'}
                )
                fig_sector.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_sector, use_container_width=True)
        
        # Raw data table (optional)
        with st.expander("📊 View Raw Data"):
            st.dataframe(stock_data.tail(20))
        
    except Exception as e:
        st.error(f"Error in market analysis: {str(e)}")
        st.info("Please try refreshing the data or selecting a different stock symbol.")

if __name__ == "__main__":
    main()
