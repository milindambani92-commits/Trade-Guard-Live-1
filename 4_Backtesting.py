import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.data_fetcher import DataFetcher
from utils.technical_indicators import TechnicalIndicators
from utils.risk_management import RiskManager

st.set_page_config(page_title="Backtesting", page_icon="🔄", layout="wide")

class SimpleBacktester:
    """
    Simple backtesting engine for trading strategies
    """
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.trades = []
        self.portfolio_values = []
        
    def run_strategy(self, data, signals, stop_loss_pct=5, take_profit_pct=10, position_size_pct=10):
        """
        Run backtest on given data and signals
        
        Args:
            data: Price data DataFrame
            signals: DataFrame with Buy_Signal and Sell_Signal columns
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            position_size_pct: Position size as percentage of capital
        """
        capital = self.initial_capital
        position = 0  # Number of shares
        entry_price = 0
        entry_date = None
        
        portfolio_values = []
        trades = []
        
        for i, (date, row) in enumerate(data.iterrows()):
            current_price = row['Close']
            current_signals = signals.iloc[i] if i < len(signals) else None
            
            # Calculate current portfolio value
            portfolio_value = capital + (position * current_price)
            portfolio_values.append({
                'Date': date,
                'Portfolio Value': portfolio_value,
                'Cash': capital,
                'Position Value': position * current_price,
                'Price': current_price
            })
            
            # If we have a position, check for exit conditions
            if position > 0:
                # Check stop loss
                if current_price <= entry_price * (1 - stop_loss_pct / 100):
                    # Stop loss triggered
                    capital += position * current_price
                    trades.append({
                        'Entry Date': entry_date,
                        'Exit Date': date,
                        'Entry Price': entry_price,
                        'Exit Price': current_price,
                        'Shares': position,
                        'P&L': position * (current_price - entry_price),
                        'P&L %': ((current_price - entry_price) / entry_price) * 100,
                        'Exit Reason': 'Stop Loss',
                        'Days Held': (date - entry_date).days
                    })
                    position = 0
                    continue
                
                # Check take profit
                elif current_price >= entry_price * (1 + take_profit_pct / 100):
                    # Take profit triggered
                    capital += position * current_price
                    trades.append({
                        'Entry Date': entry_date,
                        'Exit Date': date,
                        'Entry Price': entry_price,
                        'Exit Price': current_price,
                        'Shares': position,
                        'P&L': position * (current_price - entry_price),
                        'P&L %': ((current_price - entry_price) / entry_price) * 100,
                        'Exit Reason': 'Take Profit',
                        'Days Held': (date - entry_date).days
                    })
                    position = 0
                    continue
                
                # Check sell signal
                elif current_signals is not None and current_signals.get('Sell_Signal', False):
                    # Sell signal triggered
                    capital += position * current_price
                    trades.append({
                        'Entry Date': entry_date,
                        'Exit Date': date,
                        'Entry Price': entry_price,
                        'Exit Price': current_price,
                        'Shares': position,
                        'P&L': position * (current_price - entry_price),
                        'P&L %': ((current_price - entry_price) / entry_price) * 100,
                        'Exit Reason': 'Sell Signal',
                        'Days Held': (date - entry_date).days
                    })
                    position = 0
            
            # Check for buy signal when we don't have a position
            elif position == 0 and current_signals is not None and current_signals.get('Buy_Signal', False):
                # Buy signal triggered
                position_value = capital * (position_size_pct / 100)
                position = int(position_value / current_price)
                if position > 0:
                    capital -= position * current_price
                    entry_price = current_price
                    entry_date = date
        
        # Close any remaining position at the end
        if position > 0:
            final_price = data['Close'].iloc[-1]
            capital += position * final_price
            trades.append({
                'Entry Date': entry_date,
                'Exit Date': data.index[-1],
                'Entry Price': entry_price,
                'Exit Price': final_price,
                'Shares': position,
                'P&L': position * (final_price - entry_price),
                'P&L %': ((final_price - entry_price) / entry_price) * 100,
                'Exit Reason': 'End of Period',
                'Days Held': (data.index[-1] - entry_date).days
            })
        
        self.trades = trades
        self.portfolio_values = portfolio_values
        
        return trades, portfolio_values

def main():
    st.title("🔄 Strategy Backtesting")
    st.markdown("Test your trading strategies on historical data before risking real money")
    
    # Risk warning
    st.error("⚠️ **BACKTESTING DISCLAIMER**: Past performance does not guarantee future results. Backtesting may not account for all market conditions, slippage, and transaction costs.")
    
    # Initialize classes
    data_fetcher = DataFetcher()
    tech_indicators = TechnicalIndicators()
    risk_manager = RiskManager()
    
    # Sidebar for strategy configuration
    st.sidebar.header("🎯 Strategy Configuration")
    
    # Basic parameters
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter stock symbol to backtest")
    
    backtest_period = st.sidebar.selectbox(
        "Backtest Period",
        options=["6mo", "1y", "2y", "5y"],
        index=1,
        help="Historical period for backtesting"
    )
    
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=1000,
        value=10000,
        step=1000,
        help="Starting capital for backtest"
    )
    
    position_size_pct = st.sidebar.slider(
        "Position Size (%)",
        min_value=5,
        max_value=100,
        value=20,
        help="Percentage of capital to use per trade"
    )
    
    # Risk management parameters
    st.sidebar.subheader("⚖️ Risk Management")
    
    stop_loss_pct = st.sidebar.slider(
        "Stop Loss (%)",
        min_value=1,
        max_value=20,
        value=5,
        help="Stop loss percentage"
    )
    
    take_profit_pct = st.sidebar.slider(
        "Take Profit (%)",
        min_value=5,
        max_value=50,
        value=15,
        help="Take profit percentage"
    )
    
    # Strategy selection
    st.sidebar.subheader("📊 Strategy Parameters")
    
    strategy_type = st.sidebar.selectbox(
        "Strategy Type",
        options=["Moving Average Crossover", "RSI Oversold/Overbought", "MACD Signal", "Combined Indicators"],
        help="Select the trading strategy to test"
    )
    
    # Strategy-specific parameters
    if strategy_type == "Moving Average Crossover":
        ma_fast = st.sidebar.slider("Fast MA Period", 5, 50, 20)
        ma_slow = st.sidebar.slider("Slow MA Period", 20, 200, 50)
    
    elif strategy_type == "RSI Oversold/Overbought":
        rsi_period = st.sidebar.slider("RSI Period", 10, 30, 14)
        rsi_oversold = st.sidebar.slider("RSI Oversold Level", 10, 40, 30)
        rsi_overbought = st.sidebar.slider("RSI Overbought Level", 60, 90, 70)
    
    elif strategy_type == "MACD Signal":
        macd_fast = st.sidebar.slider("MACD Fast Period", 8, 20, 12)
        macd_slow = st.sidebar.slider("MACD Slow Period", 20, 35, 26)
        macd_signal = st.sidebar.slider("MACD Signal Period", 5, 15, 9)
    
    # Run backtest button
    run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary")
    
    # Main content area
    if symbol and run_backtest:
        try:
            with st.spinner("Loading data and running backtest..."):
                # Fetch historical data
                stock_data = data_fetcher.get_stock_data(symbol.upper(), backtest_period)
                
                if stock_data is None or stock_data.empty:
                    st.error(f"Could not fetch data for {symbol}. Please check the symbol and try again.")
                    return
                
                # Generate signals based on selected strategy
                signals_df = pd.DataFrame(index=stock_data.index)
                signals_df['Buy_Signal'] = False
                signals_df['Sell_Signal'] = False
                
                if strategy_type == "Moving Average Crossover":
                    # Moving average crossover strategy
                    stock_data['MA_Fast'] = tech_indicators.moving_average(stock_data['Close'], ma_fast)
                    stock_data['MA_Slow'] = tech_indicators.moving_average(stock_data['Close'], ma_slow)
                    
                    # Buy when fast MA crosses above slow MA
                    signals_df['Buy_Signal'] = (stock_data['MA_Fast'] > stock_data['MA_Slow']) & \
                                              (stock_data['MA_Fast'].shift(1) <= stock_data['MA_Slow'].shift(1))
                    
                    # Sell when fast MA crosses below slow MA
                    signals_df['Sell_Signal'] = (stock_data['MA_Fast'] < stock_data['MA_Slow']) & \
                                               (stock_data['MA_Fast'].shift(1) >= stock_data['MA_Slow'].shift(1))
                
                elif strategy_type == "RSI Oversold/Overbought":
                    # RSI strategy
                    rsi = tech_indicators.rsi(stock_data['Close'], rsi_period)
                    
                    # Buy when RSI is oversold
                    signals_df['Buy_Signal'] = (rsi < rsi_oversold) & (rsi.shift(1) >= rsi_oversold)
                    
                    # Sell when RSI is overbought
                    signals_df['Sell_Signal'] = (rsi > rsi_overbought) & (rsi.shift(1) <= rsi_overbought)
                
                elif strategy_type == "MACD Signal":
                    # MACD strategy
                    macd_line, macd_signal_line, macd_histogram = tech_indicators.macd(
                        stock_data['Close'], macd_fast, macd_slow, macd_signal
                    )
                    
                    # Buy when MACD crosses above signal line
                    signals_df['Buy_Signal'] = (macd_line > macd_signal_line) & \
                                              (macd_line.shift(1) <= macd_signal_line.shift(1))
                    
                    # Sell when MACD crosses below signal line
                    signals_df['Sell_Signal'] = (macd_line < macd_signal_line) & \
                                               (macd_line.shift(1) >= macd_signal_line.shift(1))
                
                elif strategy_type == "Combined Indicators":
                    # Combined strategy using multiple indicators
                    # Calculate all indicators
                    stock_data['MA_20'] = tech_indicators.moving_average(stock_data['Close'], 20)
                    stock_data['MA_50'] = tech_indicators.moving_average(stock_data['Close'], 50)
                    rsi = tech_indicators.rsi(stock_data['Close'])
                    macd_line, macd_signal_line, _ = tech_indicators.macd(stock_data['Close'])
                    
                    # Buy when multiple conditions are met
                    buy_conditions = (
                        (stock_data['MA_20'] > stock_data['MA_50']) &  # Uptrend
                        (rsi < 70) &  # Not overbought
                        (macd_line > macd_signal_line) &  # MACD bullish
                        (stock_data['Close'] > stock_data['MA_20'])  # Price above short MA
                    )
                    
                    # Sell when any condition is violated
                    sell_conditions = (
                        (stock_data['MA_20'] < stock_data['MA_50']) |  # Downtrend
                        (rsi > 70) |  # Overbought
                        (macd_line < macd_signal_line)  # MACD bearish
                    )
                    
                    signals_df['Buy_Signal'] = buy_conditions & ~buy_conditions.shift(1).fillna(False)
                    signals_df['Sell_Signal'] = sell_conditions & ~sell_conditions.shift(1).fillna(False)
                
                # Run backtest
                backtester = SimpleBacktester(initial_capital)
                trades, portfolio_values = backtester.run_strategy(
                    stock_data, signals_df, stop_loss_pct, take_profit_pct, position_size_pct
                )
                
                # Display results
                st.header(f"📊 Backtest Results - {symbol.upper()}")
                
                # Performance metrics
                if portfolio_values:
                    portfolio_df = pd.DataFrame(portfolio_values)
                    final_value = portfolio_df['Portfolio Value'].iloc[-1]
                    total_return = ((final_value - initial_capital) / initial_capital) * 100
                    
                    # Calculate more metrics
                    portfolio_df['Daily Return'] = portfolio_df['Portfolio Value'].pct_change()
                    
                    # Annualized return
                    days = len(portfolio_df)
                    annualized_return = ((final_value / initial_capital) ** (252 / days) - 1) * 100
                    
                    # Volatility
                    volatility = portfolio_df['Daily Return'].std() * np.sqrt(252) * 100
                    
                    # Sharpe ratio (assuming 2% risk-free rate)
                    sharpe_ratio = (annualized_return - 2) / volatility if volatility > 0 else 0
                    
                    # Maximum drawdown
                    running_max = portfolio_df['Portfolio Value'].expanding().max()
                    drawdown = (portfolio_df['Portfolio Value'] - running_max) / running_max
                    max_drawdown = drawdown.min() * 100
                    
                    # Display key metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Total Return", f"{total_return:.2f}%", f"${final_value - initial_capital:,.2f}")
                    
                    with col2:
                        st.metric("Annualized Return", f"{annualized_return:.2f}%")
                    
                    with col3:
                        st.metric("Volatility", f"{volatility:.2f}%")
                    
                    with col4:
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    
                    with col5:
                        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                    
                    # Portfolio value chart
                    fig = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('Portfolio Value vs Buy & Hold', 'Stock Price with Signals', 'Drawdown'),
                        row_heights=[0.5, 0.3, 0.2]
                    )
                    
                    # Portfolio value
                    fig.add_trace(
                        go.Scatter(
                            x=portfolio_df['Date'],
                            y=portfolio_df['Portfolio Value'],
                            mode='lines',
                            name='Strategy',
                            line=dict(color='blue', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Buy and hold comparison
                    initial_shares = initial_capital / stock_data['Close'].iloc[0]
                    buy_hold_values = initial_shares * stock_data['Close']
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data.index,
                            y=buy_hold_values,
                            mode='lines',
                            name='Buy & Hold',
                            line=dict(color='gray', width=2, dash='dash')
                        ),
                        row=1, col=1
                    )
                    
                    # Stock price with signals
                    fig.add_trace(
                        go.Candlestick(
                            x=stock_data.index,
                            open=stock_data['Open'],
                            high=stock_data['High'],
                            low=stock_data['Low'],
                            close=stock_data['Close'],
                            name='Price',
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    
                    # Buy signals
                    buy_signals = signals_df[signals_df['Buy_Signal']]
                    if not buy_signals.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=buy_signals.index,
                                y=stock_data.loc[buy_signals.index, 'Close'],
                                mode='markers',
                                name='Buy Signal',
                                marker=dict(color='green', size=10, symbol='triangle-up'),
                                showlegend=False
                            ),
                            row=2, col=1
                        )
                    
                    # Sell signals
                    sell_signals = signals_df[signals_df['Sell_Signal']]
                    if not sell_signals.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=sell_signals.index,
                                y=stock_data.loc[sell_signals.index, 'Close'],
                                mode='markers',
                                name='Sell Signal',
                                marker=dict(color='red', size=10, symbol='triangle-down'),
                                showlegend=False
                            ),
                            row=2, col=1
                        )
                    
                    # Drawdown
                    fig.add_trace(
                        go.Scatter(
                            x=portfolio_df['Date'],
                            y=drawdown * 100,
                            mode='lines',
                            name='Drawdown',
                            fill='tonexty',
                            fillcolor='rgba(255, 0, 0, 0.3)',
                            line=dict(color='red'),
                            showlegend=False
                        ),
                        row=3, col=1
                    )
                    
                    fig.update_layout(
                        height=800,
                        title=f"{symbol.upper()} - {strategy_type} Strategy Backtest",
                        xaxis3_title="Date"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trade analysis
                    if trades:
                        st.header("📋 Trade Analysis")
                        
                        trades_df = pd.DataFrame(trades)
                        
                        # Trade statistics
                        total_trades = len(trades_df)
                        winning_trades = len(trades_df[trades_df['P&L'] > 0])
                        losing_trades = len(trades_df[trades_df['P&L'] < 0])
                        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                        
                        avg_win = trades_df[trades_df['P&L'] > 0]['P&L'].mean() if winning_trades > 0 else 0
                        avg_loss = trades_df[trades_df['P&L'] < 0]['P&L'].mean() if losing_trades > 0 else 0
                        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                        
                        avg_hold_time = trades_df['Days Held'].mean()
                        
                        # Trade metrics
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        
                        with col1:
                            st.metric("Total Trades", total_trades)
                        
                        with col2:
                            st.metric("Win Rate", f"{win_rate:.1f}%", f"{winning_trades}W / {losing_trades}L")
                        
                        with col3:
                            st.metric("Avg Win", f"${avg_win:.2f}")
                        
                        with col4:
                            st.metric("Avg Loss", f"${avg_loss:.2f}")
                        
                        with col5:
                            st.metric("Profit Factor", f"{profit_factor:.2f}")
                        
                        with col6:
                            st.metric("Avg Hold Time", f"{avg_hold_time:.1f} days")
                        
                        # Trade distribution charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # P&L distribution
                            fig_pnl = px.histogram(
                                trades_df,
                                x='P&L',
                                nbins=20,
                                title="Trade P&L Distribution",
                                color_discrete_sequence=['lightblue']
                            )
                            fig_pnl.add_vline(x=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_pnl, use_container_width=True)
                        
                        with col2:
                            # Exit reasons
                            exit_reasons = trades_df['Exit Reason'].value_counts()
                            fig_reasons = px.pie(
                                values=exit_reasons.values,
                                names=exit_reasons.index,
                                title="Exit Reasons"
                            )
                            st.plotly_chart(fig_reasons, use_container_width=True)
                        
                        # Detailed trades table
                        st.subheader("📊 Detailed Trades")
                        
                        # Format the trades dataframe for display
                        display_trades = trades_df.copy()
                        display_trades['Entry Date'] = pd.to_datetime(display_trades['Entry Date']).dt.date
                        display_trades['Exit Date'] = pd.to_datetime(display_trades['Exit Date']).dt.date
                        display_trades['Entry Price'] = display_trades['Entry Price'].apply(lambda x: f"${x:.2f}")
                        display_trades['Exit Price'] = display_trades['Exit Price'].apply(lambda x: f"${x:.2f}")
                        display_trades['P&L'] = display_trades['P&L'].apply(lambda x: f"${x:.2f}")
                        display_trades['P&L %'] = display_trades['P&L %'].apply(lambda x: f"{x:.2f}%")
                        
                        st.dataframe(display_trades, use_container_width=True, hide_index=True)
                    
                    else:
                        st.warning("No trades were generated with the current strategy parameters. Try adjusting the strategy settings.")
                    
                    # Strategy comparison
                    st.header("📈 Strategy vs Buy & Hold Comparison")
                    
                    buy_hold_return = ((buy_hold_values.iloc[-1] - initial_capital) / initial_capital) * 100
                    strategy_return = total_return
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Strategy Return", f"{strategy_return:.2f}%")
                    
                    with col2:
                        st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")
                    
                    with col3:
                        outperformance = strategy_return - buy_hold_return
                        st.metric("Outperformance", f"{outperformance:.2f}%")
                    
                    # Risk-adjusted metrics comparison
                    if volatility > 0:
                        buy_hold_volatility = buy_hold_values.pct_change().std() * np.sqrt(252) * 100
                        strategy_sharpe = (strategy_return - 2) / volatility if volatility > 0 else 0
                        buy_hold_sharpe = (buy_hold_return - 2) / buy_hold_volatility if buy_hold_volatility > 0 else 0
                        
                        st.subheader("📊 Risk-Adjusted Performance")
                        
                        comparison_data = {
                            'Metric': ['Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)'],
                            'Strategy': [strategy_return, volatility, strategy_sharpe, max_drawdown],
                            'Buy & Hold': [buy_hold_return, buy_hold_volatility, buy_hold_sharpe, 
                                         ((buy_hold_values / buy_hold_values.expanding().max() - 1).min() * 100)]
                        }
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.table(comparison_df)
                    
                    # Optimization suggestions
                    st.header("💡 Optimization Suggestions")
                    
                    suggestions = []
                    
                    if win_rate < 40:
                        suggestions.append("• **Low Win Rate**: Consider tightening entry criteria or improving signal quality")
                    
                    if profit_factor < 1.5:
                        suggestions.append("• **Low Profit Factor**: Review stop loss and take profit levels")
                    
                    if max_drawdown < -20:
                        suggestions.append("• **High Drawdown**: Consider smaller position sizes or additional risk management")
                    
                    if avg_hold_time > 30:
                        suggestions.append("• **Long Hold Times**: Consider more responsive exit criteria")
                    
                    if outperformance < 0:
                        suggestions.append("• **Underperformance**: Strategy may need refinement or different market conditions")
                    
                    if sharpe_ratio < 1:
                        suggestions.append("• **Low Sharpe Ratio**: Risk-adjusted returns could be improved")
                    
                    if not suggestions:
                        suggestions.append("• **Good Performance**: Strategy shows promising results, consider live testing with small amounts")
                    
                    for suggestion in suggestions:
                        st.write(suggestion)
                
                else:
                    st.error("No data available for backtesting")
        
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            st.info("Please check your inputs and try again.")
    
    else:
        # Information about backtesting
        st.header("🎯 About Strategy Backtesting")
        
        st.markdown("""
        **What is Backtesting?**
        
        Backtesting is the process of testing a trading strategy using historical data to see how it would have performed in the past. This helps traders:
        
        - Evaluate strategy effectiveness before risking real money
        - Understand potential returns and risks
        - Optimize strategy parameters
        - Build confidence in trading approaches
        
        **Available Strategies:**
        
        1. **Moving Average Crossover**: Buy when fast MA crosses above slow MA, sell when it crosses below
        2. **RSI Oversold/Overbought**: Buy when RSI is oversold, sell when overbought
        3. **MACD Signal**: Buy/sell based on MACD line crossing signal line
        4. **Combined Indicators**: Uses multiple indicators for entry/exit decisions
        
        **Important Considerations:**
        
        ⚠️ **Limitations of Backtesting:**
        - Past performance doesn't guarantee future results
        - Market conditions change over time
        - Doesn't account for slippage, commissions, or execution delays
        - May suffer from overfitting to historical data
        - Survivor bias in data selection
        
        **Best Practices:**
        - Test on multiple time periods and market conditions
        - Use out-of-sample testing
        - Account for transaction costs
        - Start with small amounts when moving to live trading
        - Continuously monitor and adjust strategies
        """)
        
        # Sample backtest results for demonstration
        st.subheader("📊 Sample Backtest Metrics Explained")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Return Metrics:**
            - **Total Return**: Overall percentage gain/loss
            - **Annualized Return**: Return normalized to yearly basis
            - **Win Rate**: Percentage of profitable trades
            - **Profit Factor**: Ratio of gross profit to gross loss
            """)
        
        with col2:
            st.markdown("""
            **Risk Metrics:**
            - **Volatility**: Standard deviation of returns
            - **Sharpe Ratio**: Risk-adjusted return measure
            - **Maximum Drawdown**: Largest peak-to-trough decline
            - **Average Hold Time**: Average days per trade
            """)
        
        st.info("👈 Configure your strategy parameters in the sidebar and click 'Run Backtest' to get started!")

if __name__ == "__main__":
    main()
