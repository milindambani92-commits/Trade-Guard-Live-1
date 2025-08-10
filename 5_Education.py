import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math

from utils.data_fetcher import DataFetcher
from utils.technical_indicators import TechnicalIndicators
from utils.risk_management import RiskManager

st.set_page_config(page_title="Trading Education", page_icon="📚", layout="wide")

def main():
    st.title("📚 Trading Education Center")
    st.markdown("**Learn the fundamentals of trading, technical analysis, and risk management**")
    
    # Educational disclaimer
    st.info("📖 **Educational Purpose**: This content is for educational purposes only and should not be considered as financial advice. Always do your own research and consider consulting with a financial advisor.")
    
    # Initialize classes for examples
    data_fetcher = DataFetcher()
    tech_indicators = TechnicalIndicators()
    risk_manager = RiskManager()
    
    # Main education tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Trading Basics",
        "📊 Technical Analysis", 
        "⚖️ Risk Management",
        "💡 Trading Psychology",
        "📈 Strategy Fundamentals",
        "🎓 Advanced Concepts"
    ])
    
    # Tab 1: Trading Basics
    with tab1:
        st.header("🎯 Trading Fundamentals")
        
        # Market Basics
        st.subheader("📈 Understanding Financial Markets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **What is Trading?**
            
            Trading involves buying and selling financial instruments (stocks, options, futures, etc.) with the goal of making a profit from price movements. Unlike investing, trading typically involves shorter time horizons and more frequent transactions.
            
            **Types of Trading:**
            - **Day Trading**: Positions held for minutes to hours within a single day
            - **Swing Trading**: Positions held for days to weeks
            - **Position Trading**: Positions held for weeks to months
            - **Scalping**: Very short-term trades lasting seconds to minutes
            
            **Key Market Participants:**
            - **Retail Traders**: Individual traders (like you!)
            - **Institutional Investors**: Banks, hedge funds, pension funds
            - **Market Makers**: Provide liquidity to markets
            - **Algorithmic Traders**: Use automated systems
            """)
        
        with col2:
            st.markdown("""
            **How Markets Work:**
            
            **Supply and Demand**: Prices move based on the balance between buyers (demand) and sellers (supply).
            
            **Bid and Ask**: 
            - **Bid**: Highest price buyers are willing to pay
            - **Ask**: Lowest price sellers are willing to accept
            - **Spread**: Difference between bid and ask
            
            **Market Orders vs Limit Orders:**
            - **Market Order**: Buy/sell immediately at current market price
            - **Limit Order**: Buy/sell only at specified price or better
            
            **Volume**: Number of shares traded - indicates interest and liquidity
            
            **Market Hours**: Most US markets open 9:30 AM - 4:00 PM ET
            """)
        
        # Basic Trading Concepts
        st.subheader("💰 Essential Trading Concepts")
        
        # Interactive example
        st.markdown("**💡 Interactive Example: Calculate Profit/Loss**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            shares = st.number_input("Number of Shares", min_value=1, value=100, step=1)
            buy_price = st.number_input("Buy Price ($)", min_value=0.01, value=50.00, step=0.01)
            
        with col2:
            sell_price = st.number_input("Sell Price ($)", min_value=0.01, value=55.00, step=0.01)
            commission = st.number_input("Commission per Trade ($)", min_value=0.0, value=0.0, step=0.01)
        
        with col3:
            # Calculate P&L
            gross_profit = (sell_price - buy_price) * shares
            total_commission = commission * 2  # Buy and sell
            net_profit = gross_profit - total_commission
            profit_percentage = (net_profit / (buy_price * shares)) * 100
            
            st.metric("Gross P&L", f"${gross_profit:.2f}")
            st.metric("Net P&L", f"${net_profit:.2f}")
            st.metric("Return %", f"{profit_percentage:.2f}%")
        
        # Long vs Short explanation
        st.subheader("📊 Long vs Short Positions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **🟢 Long Position (Buy First)**
            
            - You buy shares expecting price to go UP
            - Profit when price increases
            - Loss when price decreases
            - Maximum loss: 100% (if stock goes to $0)
            - Maximum gain: Unlimited (theoretically)
            
            **Example**: Buy 100 shares at $50, sell at $60 = $1,000 profit
            """)
        
        with col2:
            st.error("""
            **🔴 Short Position (Sell First)**
            
            - You borrow and sell shares expecting price to go DOWN
            - Profit when price decreases
            - Loss when price increases
            - Maximum gain: 100% (if stock goes to $0)
            - Maximum loss: Unlimited (if stock keeps rising)
            
            **Example**: Short 100 shares at $50, cover at $40 = $1,000 profit
            """)
        
        # Common mistakes for beginners
        st.subheader("⚠️ Common Beginner Mistakes to Avoid")
        
        mistakes = [
            "**Trading without a plan** - Always have entry, exit, and risk management rules",
            "**Risking too much per trade** - Never risk more than 1-2% of your account on a single trade",
            "**Chasing hot tips** - Do your own research instead of following others blindly",
            "**Emotional trading** - Don't let fear or greed drive your decisions",
            "**Overtrading** - Quality over quantity; wait for good setups",
            "**Ignoring risk management** - Always use stop losses and position sizing",
            "**Not keeping a journal** - Track your trades to learn from mistakes",
            "**Starting with real money** - Practice with paper trading first"
        ]
        
        for mistake in mistakes:
            st.write(f"• {mistake}")
    
    # Tab 2: Technical Analysis
    with tab2:
        st.header("📊 Technical Analysis Fundamentals")
        
        st.markdown("""
        **Technical Analysis** is the study of price charts and trading volume to predict future price movements. 
        It's based on the idea that historical price action tends to repeat itself.
        """)
        
        # Chart types
        st.subheader("📈 Types of Charts")
        
        # Generate sample data for chart examples
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        np.random.seed(42)  # For consistent examples
        prices = 100 + np.cumsum(np.random.randn(30) * 0.5)
        
        sample_data = pd.DataFrame({
            'Date': dates,
            'Open': prices + np.random.randn(30) * 0.2,
            'High': prices + abs(np.random.randn(30)) * 0.5,
            'Low': prices - abs(np.random.randn(30)) * 0.5,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 30)
        })
        
        sample_data['High'] = np.maximum(sample_data[['Open', 'Close']].max(axis=1), sample_data['High'])
        sample_data['Low'] = np.minimum(sample_data[['Open', 'Close']].min(axis=1), sample_data['Low'])
        
        chart_type = st.selectbox("Select Chart Type:", ["Line Chart", "Candlestick Chart", "OHLC Chart"])
        
        fig = go.Figure()
        
        if chart_type == "Line Chart":
            fig.add_trace(go.Scatter(
                x=sample_data['Date'],
                y=sample_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(width=2, color='blue')
            ))
            st.markdown("**Line Chart**: Shows only closing prices connected by lines. Simple but lacks detail about intraday price action.")
            
        elif chart_type == "Candlestick Chart":
            fig.add_trace(go.Candlestick(
                x=sample_data['Date'],
                open=sample_data['Open'],
                high=sample_data['High'],
                low=sample_data['Low'],
                close=sample_data['Close'],
                name='Price'
            ))
            st.markdown("""
            **Candlestick Chart**: Shows open, high, low, and close prices for each period.
            - **Green/White candles**: Close higher than open (bullish)
            - **Red/Black candles**: Close lower than open (bearish)
            - **Body**: Difference between open and close
            - **Wicks/Shadows**: Show high and low of the period
            """)
            
        else:  # OHLC Chart
            for i, row in sample_data.iterrows():
                # High-Low line
                fig.add_trace(go.Scatter(
                    x=[row['Date'], row['Date']],
                    y=[row['Low'], row['High']],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False
                ))
                # Open tick
                fig.add_trace(go.Scatter(
                    x=[row['Date']],
                    y=[row['Open']],
                    mode='markers',
                    marker=dict(symbol='line-ew', size=8, color='blue'),
                    showlegend=False if i > 0 else True,
                    name='Open' if i == 0 else ''
                ))
                # Close tick
                fig.add_trace(go.Scatter(
                    x=[row['Date']],
                    y=[row['Close']],
                    mode='markers',
                    marker=dict(symbol='line-ew', size=8, color='red'),
                    showlegend=False if i > 0 else True,
                    name='Close' if i == 0 else ''
                ))
            st.markdown("**OHLC Chart**: Shows open, high, low, close as bars with ticks. Left tick = open, right tick = close.")
        
        fig.update_layout(
            title=f"Sample {chart_type}",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical Indicators Education
        st.subheader("🔧 Common Technical Indicators")
        
        indicator_type = st.selectbox("Learn About:", [
            "Moving Averages",
            "RSI (Relative Strength Index)",
            "MACD (Moving Average Convergence Divergence)",
            "Bollinger Bands",
            "Support and Resistance",
            "Volume Analysis"
        ])
        
        if indicator_type == "Moving Averages":
            st.markdown("""
            **Moving Averages (MA)** smooth out price data to identify trends.
            
            **Types:**
            - **Simple Moving Average (SMA)**: Average of closing prices over N periods
            - **Exponential Moving Average (EMA)**: Gives more weight to recent prices
            
            **Common Periods:**
            - 20-day: Short-term trend
            - 50-day: Medium-term trend  
            - 200-day: Long-term trend
            
            **Trading Signals:**
            - Price above MA = Uptrend
            - Price below MA = Downtrend
            - MA crossovers = Potential trend changes
            """)
            
            # Moving Average Example
            sample_data['MA_20'] = sample_data['Close'].rolling(window=10).mean()  # Using 10 for visibility
            sample_data['MA_50'] = sample_data['Close'].rolling(window=20).mean()  # Using 20 for visibility
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=sample_data['Close'], 
                                   mode='lines', name='Price', line=dict(color='black')))
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=sample_data['MA_20'], 
                                   mode='lines', name='MA 20', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=sample_data['MA_50'], 
                                   mode='lines', name='MA 50', line=dict(color='red')))
            
            fig.update_layout(title="Moving Averages Example", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        elif indicator_type == "RSI (Relative Strength Index)":
            st.markdown("""
            **RSI** measures the speed and change of price movements on a scale of 0 to 100.
            
            **Interpretation:**
            - **RSI > 70**: Potentially overbought (sell signal)
            - **RSI < 30**: Potentially oversold (buy signal)
            - **RSI = 50**: Neutral momentum
            
            **Formula**: RSI = 100 - (100 / (1 + RS))
            Where RS = Average Gain / Average Loss over 14 periods
            
            **Trading Tips:**
            - Works best in sideways/ranging markets
            - Can stay overbought/oversold longer in strong trends
            - Look for divergences with price for reversal signals
            """)
            
            # RSI Example
            rsi_values = []
            for i in range(len(sample_data)):
                if i < 14:
                    rsi_values.append(50)  # Default for first 14 periods
                else:
                    # Simplified RSI calculation for demonstration
                    gains = []
                    losses = []
                    for j in range(i-13, i+1):
                        change = sample_data['Close'].iloc[j] - sample_data['Close'].iloc[j-1]
                        if change > 0:
                            gains.append(change)
                            losses.append(0)
                        else:
                            gains.append(0)
                            losses.append(abs(change))
                    
                    avg_gain = np.mean(gains)
                    avg_loss = np.mean(losses)
                    
                    if avg_loss == 0:
                        rsi = 100
                    else:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                    
                    rsi_values.append(rsi)
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               subplot_titles=['Price', 'RSI'], row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=sample_data['Close'], 
                                   name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=rsi_values, 
                                   name='RSI', line=dict(color='purple')), row=2, col=1)
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        elif indicator_type == "MACD (Moving Average Convergence Divergence)":
            st.markdown("""
            **MACD** shows the relationship between two moving averages of a security's price.
            
            **Components:**
            - **MACD Line**: 12-period EMA - 26-period EMA
            - **Signal Line**: 9-period EMA of MACD line
            - **Histogram**: MACD line - Signal line
            
            **Trading Signals:**
            - **Bullish**: MACD crosses above signal line
            - **Bearish**: MACD crosses below signal line
            - **Momentum**: Histogram shows increasing/decreasing momentum
            
            **Best Practices:**
            - Confirm with price action
            - Look for divergences
            - Works well in trending markets
            """)
            
            # MACD Example (simplified calculation)
            ema_12 = sample_data['Close'].ewm(span=5).mean()  # Using shorter spans for visibility
            ema_26 = sample_data['Close'].ewm(span=10).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=4).mean()
            histogram = macd_line - signal_line
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=['Price', 'MACD'], row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=sample_data['Close'], 
                                   name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=macd_line, 
                                   name='MACD', line=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=signal_line, 
                                   name='Signal', line=dict(color='red')), row=2, col=1)
            fig.add_trace(go.Bar(x=sample_data['Date'], y=histogram, 
                               name='Histogram', marker_color='green', opacity=0.6), row=2, col=1)
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        elif indicator_type == "Bollinger Bands":
            st.markdown("""
            **Bollinger Bands** consist of a moving average and two standard deviation bands.
            
            **Components:**
            - **Middle Band**: 20-period Simple Moving Average
            - **Upper Band**: Middle Band + (2 × Standard Deviation)
            - **Lower Band**: Middle Band - (2 × Standard Deviation)
            
            **Interpretation:**
            - **Price touches upper band**: Potentially overbought
            - **Price touches lower band**: Potentially oversold
            - **Band squeeze**: Low volatility, potential breakout coming
            - **Band expansion**: High volatility period
            
            **Trading Strategies:**
            - Mean reversion: Buy at lower band, sell at upper band
            - Breakout: Trade in direction of band breakout
            """)
            
            # Bollinger Bands Example
            middle_band = sample_data['Close'].rolling(window=10).mean()
            std = sample_data['Close'].rolling(window=10).std()
            upper_band = middle_band + (std * 2)
            lower_band = middle_band - (std * 2)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=upper_band, 
                                   name='Upper Band', line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=lower_band, 
                                   name='Lower Band', line=dict(color='green', dash='dash'),
                                   fill='tonexty', fillcolor='rgba(0,100,80,0.1)'))
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=middle_band, 
                                   name='Middle Band (MA)', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=sample_data['Close'], 
                                   name='Price', line=dict(color='black', width=2)))
            
            fig.update_layout(title="Bollinger Bands Example", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        elif indicator_type == "Support and Resistance":
            st.markdown("""
            **Support and Resistance** are key price levels where buying or selling pressure is concentrated.
            
            **Support**: A price level where buying interest is strong enough to prevent further declines
            **Resistance**: A price level where selling pressure prevents further advances
            
            **Characteristics:**
            - **More touches = stronger level**
            - **Higher volume at level = more significant**
            - **Psychological levels** (round numbers) often act as support/resistance
            - **Previous highs become resistance, previous lows become support**
            
            **Trading Applications:**
            - Buy near support levels
            - Sell near resistance levels
            - Breakouts above resistance or below support can signal trend changes
            - Use as stop loss or take profit levels
            """)
            
            # Support and Resistance visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=sample_data['Close'], 
                                   mode='lines', name='Price', line=dict(color='black', width=2)))
            
            # Add sample support and resistance lines
            resistance_level = sample_data['Close'].max() * 0.98
            support_level = sample_data['Close'].min() * 1.02
            
            fig.add_hline(y=resistance_level, line_dash="dash", line_color="red", 
                         annotation_text="Resistance")
            fig.add_hline(y=support_level, line_dash="dash", line_color="green", 
                         annotation_text="Support")
            
            fig.update_layout(title="Support and Resistance Example", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        elif indicator_type == "Volume Analysis":
            st.markdown("""
            **Volume** is the number of shares traded and confirms price movements.
            
            **Key Concepts:**
            - **Volume precedes price**: Changes in volume often predict price moves
            - **High volume breakouts** are more reliable than low volume breakouts
            - **Divergence**: Price makes new highs/lows but volume doesn't confirm
            
            **Volume Indicators:**
            - **On Balance Volume (OBV)**: Running total of up/down volume
            - **Volume Moving Average**: Smooths volume to identify trends
            - **Volume Price Trend (VPT)**: Combines price and volume changes
            
            **Trading Rules:**
            - Rising prices + rising volume = healthy uptrend
            - Falling prices + rising volume = healthy downtrend
            - Price breakout + high volume = strong signal
            - Price move + low volume = weak signal
            """)
            
            # Volume analysis example
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=['Price', 'Volume'], row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=sample_data['Close'], 
                                   name='Price'), row=1, col=1)
            
            # Color volume bars based on price change
            colors = ['green' if sample_data['Close'].iloc[i] >= sample_data['Open'].iloc[i] 
                     else 'red' for i in range(len(sample_data))]
            
            fig.add_trace(go.Bar(x=sample_data['Date'], y=sample_data['Volume'], 
                               name='Volume', marker_color=colors), row=2, col=1)
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Chart patterns section
        st.subheader("📊 Common Chart Patterns")
        
        pattern_info = {
            "Head and Shoulders": {
                "description": "Reversal pattern with three peaks, middle one highest",
                "signal": "Bearish reversal",
                "target": "Distance from head to neckline"
            },
            "Double Top/Bottom": {
                "description": "Two similar highs (top) or lows (bottom)",
                "signal": "Reversal pattern",
                "target": "Distance between peaks/troughs and support/resistance"
            },
            "Triangles": {
                "description": "Converging trend lines (ascending, descending, symmetrical)",
                "signal": "Continuation or reversal",
                "target": "Height of triangle added to breakout point"
            },
            "Flags and Pennants": {
                "description": "Brief consolidation after strong move",
                "signal": "Continuation pattern",
                "target": "Length of flagpole"
            }
        }
        
        selected_pattern = st.selectbox("Learn about chart pattern:", list(pattern_info.keys()))
        
        pattern = pattern_info[selected_pattern]
        st.write(f"**{selected_pattern}**")
        st.write(f"• Description: {pattern['description']}")
        st.write(f"• Signal: {pattern['signal']}")
        st.write(f"• Price Target: {pattern['target']}")
    
    # Tab 3: Risk Management
    with tab3:
        st.header("⚖️ Risk Management: The Key to Trading Success")
        
        st.error("🚨 **Most Important Rule**: Risk management is more important than being right. You can be wrong 60% of the time and still be profitable with proper risk management!")
        
        # Position sizing
        st.subheader("📏 Position Sizing")
        
        st.markdown("""
        **Position sizing determines how much of your capital to risk on each trade.**
        
        **The 1% Rule**: Never risk more than 1% of your trading capital on a single trade.
        **The 2% Rule**: More aggressive traders may risk up to 2% per trade.
        """)
        
        # Interactive position sizing calculator
        st.markdown("**💡 Interactive Position Sizing Calculator**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            account_balance = st.number_input("Account Balance ($)", min_value=1000, value=50000, step=1000)
            risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 3.0, 1.0, 0.1)
        
        with col2:
            entry_price = st.number_input("Entry Price ($)", min_value=0.01, value=100.00, step=0.01)
            stop_loss_price = st.number_input("Stop Loss Price ($)", min_value=0.01, value=95.00, step=0.01)
        
        with col3:
            risk_amount = account_balance * (risk_per_trade / 100)
            risk_per_share = abs(entry_price - stop_loss_price)
            shares_to_buy = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            total_investment = shares_to_buy * entry_price
            
            st.metric("Risk Amount", f"${risk_amount:.2f}")
            st.metric("Shares to Buy", f"{shares_to_buy:,}")
            st.metric("Total Investment", f"${total_investment:,.2f}")
        
        # Stop losses
        st.subheader("🛑 Stop Losses")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **What is a Stop Loss?**
            
            A stop loss is a predetermined price level at which you will exit a losing trade to limit your losses.
            
            **Types of Stop Losses:**
            - **Fixed Percentage**: Set at X% below entry (e.g., 5%)
            - **ATR-Based**: Based on average true range (volatility)
            - **Technical**: Based on support/resistance levels
            - **Trailing Stop**: Moves up with profitable trades
            
            **Benefits:**
            - Limits losses to predetermined amount
            - Removes emotion from exit decisions
            - Allows for better risk/reward planning
            - Protects capital for future opportunities
            """)
        
        with col2:
            st.markdown("""
            **Stop Loss Best Practices:**
            
            - **Set before entering trade** - never move stops against you
            - **Honor your stops** - don't hope for reversals
            - **Consider volatility** - wider stops for volatile stocks
            - **Use technical levels** when possible
            - **Trail stops in profitable trades**
            
            **Common Mistakes:**
            - Setting stops too tight (getting stopped out by noise)
            - Moving stops to avoid losses
            - Not using stops at all
            - Setting stops too wide (too much risk)
            """)
        
        # Risk/Reward ratios
        st.subheader("📊 Risk/Reward Ratios")
        
        st.markdown("""
        **Risk/Reward ratio compares potential loss to potential profit.**
        
        **Example**: If you risk $100 to make $300, your risk/reward ratio is 1:3
        """)
        
        # Interactive R/R calculator
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Calculate Risk/Reward Ratio:**")
            entry = st.number_input("Entry Price", value=100.0, key="rr_entry")
            stop = st.number_input("Stop Loss", value=95.0, key="rr_stop")
            target = st.number_input("Target Price", value=120.0, key="rr_target")
            
            risk = abs(entry - stop)
            reward = abs(target - entry)
            ratio = reward / risk if risk > 0 else 0
            
            st.metric("Risk Amount", f"${risk:.2f}")
            st.metric("Reward Amount", f"${reward:.2f}")
            st.metric("Risk/Reward Ratio", f"1:{ratio:.1f}")
        
        with col2:
            st.markdown("""
            **Minimum Recommended Ratios:**
            - **Conservative**: 1:2 (risk $1 to make $2)
            - **Moderate**: 1:3 (risk $1 to make $3)
            - **Aggressive**: 1:4 or higher
            
            **Why This Matters:**
            With a 1:3 ratio, you can be wrong 75% of the time and still break even!
            
            **Win Rate Required for Profitability:**
            - 1:1 ratio needs >50% win rate
            - 1:2 ratio needs >33% win rate  
            - 1:3 ratio needs >25% win rate
            """)
        
        # Portfolio heat
        st.subheader("🔥 Portfolio Heat")
        
        st.markdown("""
        **Portfolio heat measures your total risk exposure across all positions.**
        
        **Best Practices:**
        - Never have more than 6-8% of account at risk simultaneously
        - Correlated positions count as additional risk
        - Reduce position sizes when heat is high
        - Take profits to reduce overall risk
        """)
        
        # Money management rules
        st.subheader("💰 Money Management Rules")
        
        rules = [
            "**Risk only what you can afford to lose** - Trading capital should be separate from living expenses",
            "**Start small** - Begin with small position sizes while learning",
            "**Scale up gradually** - Increase size only after consistent profitability",
            "**Diversify** - Don't put all capital in one trade or sector",
            "**Keep records** - Track all trades for performance analysis",
            "**Review regularly** - Assess and adjust risk parameters monthly",
            "**Plan for drawdowns** - Expect losing periods and plan accordingly",
            "**Protect profits** - Lock in gains as account grows"
        ]
        
        for rule in rules:
            st.write(f"• {rule}")
    
    # Tab 4: Trading Psychology
    with tab4:
        st.header("💡 Trading Psychology: Mastering Your Mind")
        
        st.markdown("""
        **Trading is 20% technical and 80% psychological.**
        
        The biggest enemy of trading success isn't market volatility or economic events - it's your own emotions and cognitive biases.
        """)
        
        # Common psychological traps
        st.subheader("🧠 Common Psychological Traps")
        
        trap_tabs = st.tabs(["Fear", "Greed", "Hope", "FOMO", "Revenge Trading", "Overconfidence"])
        
        with trap_tabs[0]:  # Fear
            st.markdown("""
            **Fear in Trading**
            
            **Fear of Missing Out (FOMO)**: Jumping into trades without proper analysis
            **Fear of Loss**: Avoiding trades even with good setups
            **Fear of Being Wrong**: Refusing to take losses
            
            **How to Overcome Fear:**
            - Start with smaller position sizes
            - Use proper risk management
            - Practice with paper trading
            - Focus on process over results
            - Accept that losses are part of trading
            """)
        
        with trap_tabs[1]:  # Greed
            st.markdown("""
            **Greed in Trading**
            
            **Manifestations:**
            - Risking too much per trade
            - Not taking profits at targets
            - Overleveraging positions
            - Chasing every opportunity
            
            **How to Control Greed:**
            - Set strict position sizing rules
            - Use take profit orders
            - Follow your trading plan
            - Focus on consistent small wins
            - Remember: "Bulls make money, bears make money, pigs get slaughtered"
            """)
        
        with trap_tabs[2]:  # Hope
            st.markdown("""
            **Hope - The Trader's Enemy**
            
            **Dangerous Hopes:**
            - Hoping losing trades will turn around
            - Hoping the market will behave as expected
            - Hoping for quick riches
            
            **Reality Check:**
            - Hope is not a trading strategy
            - Cut losses quickly with stop losses
            - Base decisions on analysis, not emotion
            - Accept losses as part of the business
            """)
        
        with trap_tabs[3]:  # FOMO
            st.markdown("""
            **Fear of Missing Out (FOMO)**
            
            **FOMO Behaviors:**
            - Chasing breakouts after they've moved
            - Entering trades without proper setup
            - Abandoning trading plan for "hot tips"
            - Overtrading to catch every move
            
            **Antidotes to FOMO:**
            - Remember: There's always another opportunity
            - Stick to your trading plan
            - Wait for proper setups
            - Quality over quantity
            """)
        
        with trap_tabs[4]:  # Revenge Trading
            st.markdown("""
            **Revenge Trading**
            
            **What is it?**
            Trading to "get back" at the market after losses, often with larger size and poor setups.
            
            **Warning Signs:**
            - Increasing position sizes after losses
            - Trading outside your plan
            - Feeling angry at the market
            - Making impulsive trades
            
            **Prevention:**
            - Take a break after significant losses
            - Stick to predetermined position sizes
            - Review trades objectively
            - Remember: The market doesn't care about your emotions
            """)
        
        with trap_tabs[5]:  # Overconfidence
            st.markdown("""
            **Overconfidence Bias**
            
            **Dangerous Overconfidence:**
            - Increasing risk after winning streaks
            - Skipping analysis ("I'm hot")
            - Ignoring risk management
            - Trading outside your expertise
            
            **Staying Humble:**
            - Maintain consistent risk management
            - Keep detailed trading records
            - Remember: Past success doesn't guarantee future results
            - Focus on process, not just results
            """)
        
        # Mental preparation
        st.subheader("🧘 Mental Preparation for Trading")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Pre-Market Routine:**
            - Review overnight news and events
            - Check economic calendar
            - Review your trading plan
            - Set daily risk limits
            - Clear your mind of external distractions
            
            **During Trading:**
            - Stick to your plan
            - Take regular breaks
            - Monitor your emotional state
            - Don't overtrade
            - Stay hydrated and eat properly
            """)
        
        with col2:
            st.markdown("""
            **Post-Market Review:**
            - Review all trades objectively
            - Journal your thoughts and emotions
            - Identify what worked and what didn't
            - Plan for tomorrow
            - Celebrate small wins and learn from losses
            
            **Long-term Mental Health:**
            - Maintain life outside of trading
            - Exercise regularly
            - Get adequate sleep
            - Manage stress
            - Seek support when needed
            """)
        
        # Trading journal importance
        st.subheader("📔 The Importance of Trading Journals")
        
        st.markdown("""
        **Why Keep a Trading Journal?**
        
        A trading journal is your path to self-improvement and consistency.
        
        **What to Record:**
        - Entry and exit points with reasons
        - Market conditions
        - Emotional state before/during/after trade
        - What went right or wrong
        - Lessons learned
        
        **Benefits:**
        - Identify patterns in your behavior
        - Learn from mistakes
        - Track improvement over time
        - Maintain discipline
        - Build confidence in your abilities
        """)
    
    # Tab 5: Strategy Fundamentals
    with tab5:
        st.header("📈 Trading Strategy Fundamentals")
        
        st.markdown("""
        **A trading strategy is a systematic approach to buying and selling securities based on predefined rules.**
        """)
        
        # Types of trading strategies
        st.subheader("🎯 Types of Trading Strategies")
        
        strategy_tabs = st.tabs(["Trend Following", "Mean Reversion", "Momentum", "Breakout", "Swing Trading"])
        
        with strategy_tabs[0]:  # Trend Following
            st.markdown("""
            **Trend Following Strategy**
            
            **Philosophy**: "The trend is your friend" - Trade in the direction of the prevailing trend.
            
            **Key Components:**
            - Identify the trend direction
            - Enter trades that align with the trend
            - Stay in trades as long as trend continues
            - Exit when trend shows signs of reversal
            
            **Common Tools:**
            - Moving averages
            - Trend lines
            - ADX (Average Directional Index)
            - MACD
            
            **Pros:**
            - Can capture large moves
            - Works well in trending markets
            - Objective rules
            
            **Cons:**
            - Many false signals in sideways markets
            - Late entries and exits
            - Requires patience
            
            **Example Rules:**
            - Buy when price > 50-day MA and 20-day MA > 50-day MA
            - Sell when price < 20-day MA or MA crossover reverses
            """)
        
        with strategy_tabs[1]:  # Mean Reversion
            st.markdown("""
            **Mean Reversion Strategy**
            
            **Philosophy**: Prices tend to return to their average over time.
            
            **Key Components:**
            - Identify when prices deviate significantly from average
            - Enter trades expecting return to mean
            - Exit when price approaches average or beyond
            
            **Common Tools:**
            - Bollinger Bands
            - RSI
            - Standard deviation
            - Support and resistance levels
            
            **Pros:**
            - High win rate in ranging markets
            - Quick profits possible
            - Works well with oversold/overbought conditions
            
            **Cons:**
            - Can fight strong trends
            - Multiple small losses possible
            - Requires good timing
            
            **Example Rules:**
            - Buy when RSI < 30 and price touches lower Bollinger Band
            - Sell when RSI > 50 or price reaches middle Bollinger Band
            """)
        
        with strategy_tabs[2]:  # Momentum
            st.markdown("""
            **Momentum Strategy**
            
            **Philosophy**: Securities that are moving strongly in one direction tend to continue.
            
            **Key Components:**
            - Identify securities with strong momentum
            - Enter in direction of momentum
            - Hold while momentum continues
            - Exit when momentum weakens
            
            **Common Tools:**
            - Rate of Change (ROC)
            - MACD
            - Volume analysis
            - Relative strength
            
            **Pros:**
            - Can capture explosive moves
            - Works well in volatile markets
            - Clear directional bias
            
            **Cons:**
            - Can reverse quickly
            - Requires quick decision making
            - Higher risk/higher reward
            
            **Example Rules:**
            - Buy when stock breaks 20-day high with volume > 150% of average
            - Sell when momentum indicators turn negative
            """)
        
        with strategy_tabs[3]:  # Breakout
            st.markdown("""
            **Breakout Strategy**
            
            **Philosophy**: Trade securities breaking out of defined price ranges.
            
            **Key Components:**
            - Identify consolidation patterns
            - Wait for breakout with volume
            - Enter in direction of breakout
            - Target based on pattern height
            
            **Common Patterns:**
            - Rectangle breakouts
            - Triangle breakouts
            - Channel breakouts
            - Support/resistance breaks
            
            **Pros:**
            - Clear entry and exit rules
            - Good risk/reward ratios
            - Works in trending markets
            
            **Cons:**
            - Many false breakouts
            - Requires patience for setup
            - Can be whipsawed
            
            **Example Rules:**
            - Buy on break above resistance with volume > 2x average
            - Stop loss below previous resistance (now support)
            - Target = breakout point + (pattern height)
            """)
        
        with strategy_tabs[4]:  # Swing Trading
            st.markdown("""
            **Swing Trading Strategy**
            
            **Philosophy**: Capture swings in stock prices over several days to weeks.
            
            **Key Components:**
            - Hold positions for 2-10 days typically
            - Focus on technical analysis
            - Target intermediate price swings
            - Balance risk and time commitment
            
            **Common Approaches:**
            - Trade pullbacks in trends
            - Play earnings reactions
            - Sector rotation plays
            - Support/resistance bounces
            
            **Pros:**
            - Less time intensive than day trading
            - Can capture significant moves
            - Good for part-time traders
            
            **Cons:**
            - Overnight risk
            - Gap risk
            - Requires patience
            
            **Example Strategy:**
            - Buy pullbacks to 20-day MA in uptrending stocks
            - Target previous highs or 2:1 risk/reward
            - Stop loss below swing low
            """)
        
        # Building a trading plan
        st.subheader("📋 Building Your Trading Plan")
        
        st.markdown("""
        **Essential Elements of a Trading Plan:**
        
        1. **Market Analysis**: How will you analyze markets and find opportunities?
        2. **Entry Criteria**: What conditions must be met to enter a trade?
        3. **Position Sizing**: How much will you risk per trade?
        4. **Risk Management**: Where will you place stops and take profits?
        5. **Exit Strategy**: When and how will you exit trades?
        6. **Performance Review**: How will you track and improve performance?
        """)
        
        # Interactive trading plan builder
        st.subheader("🔧 Trading Plan Builder")
        
        with st.expander("Build Your Trading Plan", expanded=False):
            plan_strategy = st.selectbox("Primary Strategy:", 
                                       ["Trend Following", "Mean Reversion", "Momentum", "Breakout", "Swing Trading"])
            
            plan_timeframe = st.selectbox("Trading Timeframe:", 
                                        ["Intraday", "Daily", "Weekly"])
            
            plan_risk = st.slider("Risk per Trade (%):", 0.5, 3.0, 1.0, 0.1)
            
            plan_ratio = st.selectbox("Minimum Risk/Reward Ratio:", 
                                    ["1:1", "1:2", "1:3", "1:4"])
            
            plan_markets = st.multiselect("Markets to Trade:", 
                                        ["Large Cap Stocks", "Small Cap Stocks", "ETFs", "Options", "Forex"])
            
            if st.button("Generate Trading Plan Summary"):
                st.success(f"""
                **Your Trading Plan Summary:**
                
                - **Strategy**: {plan_strategy}
                - **Timeframe**: {plan_timeframe}
                - **Risk per Trade**: {plan_risk}%
                - **Minimum R/R**: {plan_ratio}
                - **Markets**: {', '.join(plan_markets)}
                
                **Next Steps:**
                1. Backtest this strategy on historical data
                2. Paper trade for at least 30 trades
                3. Start with small real money positions
                4. Track and review performance weekly
                5. Adjust parameters based on results
                """)
        
        # Common strategy mistakes
        st.subheader("⚠️ Common Strategy Mistakes")
        
        mistakes = [
            "**No written plan** - Trading based on gut feelings",
            "**Inconsistent execution** - Not following your own rules",
            "**Over-optimization** - Curve-fitting to past data",
            "**Lack of backtesting** - Not testing strategy before live trading",
            "**Ignoring market conditions** - Using same strategy in all markets",
            "**No performance tracking** - Not measuring what works",
            "**Changing strategies too often** - Not giving strategies time to work",
            "**Focusing only on entries** - Neglecting exits and risk management"
        ]
        
        for mistake in mistakes:
            st.write(f"• {mistake}")
    
    # Tab 6: Advanced Concepts
    with tab6:
        st.header("🎓 Advanced Trading Concepts")
        
        # Market structure
        st.subheader("🏗️ Understanding Market Structure")
        
        st.markdown("""
        **Market structure refers to how prices move and form patterns over time.**
        
        **Key Concepts:**
        
        **Higher Highs and Higher Lows (Uptrend)**: Each peak is higher than the previous, each trough is higher than the previous
        
        **Lower Highs and Lower Lows (Downtrend)**: Each peak is lower than the previous, each trough is lower than the previous
        
        **Sideways/Ranging Market**: Price moves between defined support and resistance levels
        """)
        
        # Market efficiency and edge
        st.subheader("⚖️ Market Efficiency and Finding Your Edge")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Efficient Market Hypothesis (EMH):**
            
            Markets are "informationally efficient" - all available information is already reflected in prices.
            
            **Three Forms:**
            - **Weak**: Past prices don't predict future (challenges technical analysis)
            - **Semi-strong**: Public information already in prices
            - **Strong**: All information (public and private) in prices
            
            **Implications for Traders:**
            - Market movements are largely random
            - Consistent outperformance is difficult
            - Information advantage is key
            - Behavioral biases create opportunities
            """)
        
        with col2:
            st.markdown("""
            **Finding Your Trading Edge:**
            
            An edge is any advantage that gives you a better than random chance of success.
            
            **Types of Edges:**
            - **Informational**: Access to information before others
            - **Analytical**: Better analysis of available information
            - **Behavioral**: Exploiting others' psychological biases
            - **Technological**: Faster execution or better tools
            - **Structural**: Understanding market mechanics
            
            **Maintaining Your Edge:**
            - Continuously learn and adapt
            - Keep strategies secret when possible
            - Stay ahead of technological changes
            - Focus on your strengths
            """)
        
        # Options basics
        st.subheader("📊 Options Trading Basics")
        
        st.markdown("""
        **Options give you the right (but not obligation) to buy or sell a security at a specific price.**
        
        **Key Terms:**
        - **Call Option**: Right to BUY at strike price
        - **Put Option**: Right to SELL at strike price
        - **Strike Price**: The price at which you can exercise
        - **Expiration**: When the option expires
        - **Premium**: Cost of the option
        
        **Basic Strategies:**
        """)
        
        option_tabs = st.tabs(["Buy Calls", "Buy Puts", "Covered Call", "Cash-Secured Put"])
        
        with option_tabs[0]:
            st.markdown("""
            **Buying Call Options**
            
            **When to use**: You're bullish on a stock
            **Max Loss**: Premium paid
            **Max Gain**: Unlimited (theoretically)
            **Breakeven**: Strike + Premium
            
            **Example**: 
            - Stock at $100, buy $105 call for $2
            - Breakeven = $107
            - Profit if stock > $107 at expiration
            """)
        
        with option_tabs[1]:
            st.markdown("""
            **Buying Put Options**
            
            **When to use**: You're bearish on a stock
            **Max Loss**: Premium paid
            **Max Gain**: Strike price - Premium
            **Breakeven**: Strike - Premium
            
            **Example**:
            - Stock at $100, buy $95 put for $2
            - Breakeven = $93
            - Profit if stock < $93 at expiration
            """)
        
        with option_tabs[2]:
            st.markdown("""
            **Covered Call (Income Strategy)**
            
            **Setup**: Own 100 shares + sell 1 call
            **When to use**: Neutral to slightly bullish
            **Max Loss**: Stock loss - premium received
            **Max Gain**: Strike - stock price + premium
            
            **Benefits**: Generate income from stock holdings
            **Risk**: Stock called away if above strike
            """)
        
        with option_tabs[3]:
            st.markdown("""
            **Cash-Secured Put (Income Strategy)**
            
            **Setup**: Hold cash + sell put option
            **When to use**: Want to buy stock at lower price
            **Max Loss**: Strike - premium (if stock goes to $0)
            **Max Gain**: Premium received
            
            **Outcome**: Either keep premium or buy stock at strike
            """)
        
        # Alternative investments
        st.subheader("💎 Alternative Investments")
        
        alt_tabs = st.tabs(["Forex", "Commodities", "Cryptocurrencies", "REITs"])
        
        with alt_tabs[0]:  # Forex
            st.markdown("""
            **Foreign Exchange (Forex) Trading**
            
            **What is Forex?**
            Trading currencies against each other (EUR/USD, GBP/JPY, etc.)
            
            **Key Features:**
            - 24/5 market (Sunday 5 PM - Friday 5 PM EST)
            - High leverage available (up to 50:1 in US)
            - Major pairs: EUR/USD, GBP/USD, USD/JPY, USD/CHF
            - Affected by economic data, central bank policies
            
            **Risks:**
            - High leverage amplifies losses
            - Currency risk
            - Interest rate changes
            - Political/economic instability
            """)
        
        with alt_tabs[1]:  # Commodities
            st.markdown("""
            **Commodities Trading**
            
            **Types of Commodities:**
            - **Energy**: Oil, natural gas, gasoline
            - **Metals**: Gold, silver, copper, platinum
            - **Agriculture**: Wheat, corn, soybeans, sugar
            - **Livestock**: Cattle, pork bellies
            
            **How to Trade:**
            - Commodity ETFs (easier for retail)
            - Futures contracts (requires margin)
            - Mining/agricultural company stocks
            
            **Factors Affecting Prices:**
            - Supply and demand
            - Weather patterns
            - Economic growth
            - Currency fluctuations
            - Geopolitical events
            """)
        
        with alt_tabs[2]:  # Crypto
            st.markdown("""
            **Cryptocurrency Trading**
            
            **Popular Cryptocurrencies:**
            - Bitcoin (BTC) - Digital gold
            - Ethereum (ETH) - Smart contract platform
            - Various altcoins
            
            **Key Characteristics:**
            - Extremely volatile
            - 24/7 trading
            - Regulatory uncertainty
            - Technology-driven
            
            **Risks:**
            - Extreme volatility
            - Regulatory changes
            - Security breaches
            - Technology failures
            - Market manipulation
            
            **Important**: Only trade with money you can afford to lose entirely
            """)
        
        with alt_tabs[3]:  # REITs
            st.markdown("""
            **Real Estate Investment Trusts (REITs)**
            
            **What are REITs?**
            Companies that own/operate income-producing real estate
            
            **Types:**
            - **Equity REITs**: Own properties (apartments, offices, malls)
            - **Mortgage REITs**: Finance real estate purchases
            - **Hybrid REITs**: Combination of equity and mortgage
            
            **Benefits:**
            - High dividend yields
            - Real estate exposure without direct ownership
            - Liquidity (traded like stocks)
            - Diversification
            
            **Risks:**
            - Interest rate sensitivity
            - Real estate market cycles
            - Economic downturns affect occupancy
            - Inflation impact
            """)
        
        # Risk management advanced
        st.subheader("🎯 Advanced Risk Management")
        
        st.markdown("""
        **Portfolio-Level Risk Management**
        
        **Correlation Risk**: Don't hold multiple positions that move together
        **Sector Concentration**: Limit exposure to any single sector
        **Geographic Risk**: Consider international diversification
        **Time Diversification**: Spread entries over time
        
        **Advanced Risk Metrics:**
        - **Value at Risk (VaR)**: Maximum expected loss over time period
        - **Maximum Drawdown**: Largest peak-to-trough decline
        - **Sharpe Ratio**: Risk-adjusted returns
        - **Beta**: Sensitivity to market movements
        """)
        
        # Final thoughts
        st.subheader("🌟 Key Takeaways for Success")
        
        success_factors = [
            "**Education is ongoing** - Markets constantly evolve, keep learning",
            "**Risk management first** - Protect capital above all else",
            "**Develop discipline** - Follow your rules consistently",
            "**Start small** - Build confidence and skills gradually",
            "**Keep records** - Track everything to identify patterns",
            "**Stay humble** - The market will humble you if you don't",
            "**Focus on process** - Control what you can control",
            "**Be patient** - Good opportunities come to those who wait",
            "**Adapt to conditions** - Different markets require different approaches",
            "**Never stop improving** - There's always something new to learn"
        ]
        
        for factor in success_factors:
            st.write(f"• {factor}")
        
        st.success("""
        🎯 **Remember**: Trading is a marathon, not a sprint. Focus on consistent, sustainable growth rather than quick profits. 
        The traders who survive and thrive are those who respect the market, manage risk carefully, and never stop learning.
        
        Good luck on your trading journey! 📈
        """)

if __name__ == "__main__":
    main()
