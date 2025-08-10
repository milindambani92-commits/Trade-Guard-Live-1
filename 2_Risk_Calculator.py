import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.risk_management import RiskManager
from utils.data_fetcher import DataFetcher
from utils.technical_indicators import TechnicalIndicators

st.set_page_config(page_title="Risk Calculator", page_icon="⚖️", layout="wide")

def main():
    st.title("⚖️ Risk Management Calculator")
    st.markdown("Professional risk management tools to protect your trading capital")
    
    # Risk warning
    st.warning("🚨 **Risk Management is Key**: Never risk more than you can afford to lose. These tools help you calculate appropriate position sizes and risk levels.")
    
    # Initialize classes
    risk_manager = RiskManager()
    data_fetcher = DataFetcher()
    tech_indicators = TechnicalIndicators()
    
    # Tabs for different risk calculators
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📏 Position Sizing", 
        "🎯 Stop Loss Calculator", 
        "📊 Portfolio Risk", 
        "🎲 Monte Carlo Simulation",
        "🧮 Kelly Criterion"
    ])
    
    # Tab 1: Position Sizing Calculator
    with tab1:
        st.header("📏 Position Sizing Calculator")
        st.markdown("Calculate the optimal position size based on your risk tolerance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input parameters
            st.subheader("Account Information")
            account_size = st.number_input(
                "Account Size ($)", 
                min_value=100, 
                value=10000, 
                step=100,
                help="Your total trading account balance"
            )
            
            risk_percentage = st.slider(
                "Risk per Trade (%)", 
                min_value=0.1, 
                max_value=5.0, 
                value=2.0, 
                step=0.1,
                help="Maximum percentage of account to risk per trade (recommended: 1-2%)"
            )
            
            st.subheader("Trade Details")
            symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter the stock symbol")
            
            entry_price = st.number_input(
                "Entry Price ($)", 
                min_value=0.01, 
                value=150.00, 
                step=0.01,
                help="Your planned entry price"
            )
            
            stop_loss_method = st.selectbox(
                "Stop Loss Method",
                ["Percentage", "Dollar Amount", "ATR-based", "Technical Level"],
                help="Choose how to set your stop loss"
            )
            
            if stop_loss_method == "Percentage":
                stop_loss_pct = st.slider(
                    "Stop Loss (%)", 
                    min_value=0.5, 
                    max_value=20.0, 
                    value=3.0, 
                    step=0.1
                )
                stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
                
            elif stop_loss_method == "Dollar Amount":
                stop_loss_amount = st.number_input(
                    "Stop Loss Amount ($)", 
                    min_value=0.01, 
                    value=5.00, 
                    step=0.01
                )
                stop_loss_price = entry_price - stop_loss_amount
                
            elif stop_loss_method == "ATR-based":
                # Get ATR data
                if symbol:
                    try:
                        stock_data = data_fetcher.get_stock_data(symbol.upper(), "1mo")
                        if stock_data is not None and not stock_data.empty:
                            atr = tech_indicators.atr(stock_data['High'], stock_data['Low'], stock_data['Close'])
                            current_atr = atr.iloc[-1] if not atr.empty else 2.0
                            st.info(f"Current ATR: ${current_atr:.2f}")
                        else:
                            current_atr = 2.0
                            st.warning("Could not fetch ATR data, using default value")
                    except:
                        current_atr = 2.0
                        st.warning("Could not fetch ATR data, using default value")
                else:
                    current_atr = 2.0
                
                atr_multiplier = st.slider(
                    "ATR Multiplier", 
                    min_value=0.5, 
                    max_value=5.0, 
                    value=2.0, 
                    step=0.1,
                    help="Multiply ATR by this value to set stop loss distance"
                )
                stop_loss_price = entry_price - (atr_multiplier * current_atr)
                
            else:  # Technical Level
                stop_loss_price = st.number_input(
                    "Stop Loss Price ($)", 
                    min_value=0.01, 
                    value=entry_price * 0.97, 
                    step=0.01,
                    help="Enter your technical stop loss level"
                )
        
        with col2:
            # Calculate position size
            position_size = risk_manager.calculate_position_size(
                account_size, risk_percentage, entry_price, stop_loss_price
            )
            
            # Calculate metrics
            risk_amount = account_size * (risk_percentage / 100)
            total_investment = entry_price * position_size
            investment_percentage = (total_investment / account_size) * 100 if account_size > 0 else 0
            risk_per_share = abs(entry_price - stop_loss_price)
            
            # Display results
            st.subheader("📊 Position Size Results")
            
            # Metrics cards
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.metric("Shares to Buy", f"{position_size:,}", help="Number of shares to purchase")
                st.metric("Risk Amount", f"${risk_amount:,.2f}", help="Total amount at risk")
                st.metric("Risk per Share", f"${risk_per_share:.2f}", help="Risk per individual share")
            
            with col2_2:
                st.metric("Total Investment", f"${total_investment:,.2f}", help="Total capital required")
                st.metric("Investment %", f"{investment_percentage:.1f}%", help="Percentage of account invested")
                st.metric("Stop Loss Price", f"${stop_loss_price:.2f}", help="Your stop loss level")
            
            # Risk/Reward visualization
            st.subheader("📈 Risk/Reward Visualization")
            
            # Calculate potential profit targets
            risk_reward_ratios = [1, 1.5, 2, 3]
            profit_targets = []
            
            for ratio in risk_reward_ratios:
                target_price = risk_manager.calculate_take_profit(entry_price, stop_loss_price, ratio)
                profit_targets.append({
                    'Ratio': f"1:{ratio}",
                    'Target Price': target_price,
                    'Profit': (target_price - entry_price) * position_size,
                    'Return %': ((target_price - entry_price) / entry_price) * 100
                })
            
            # Create profit targets table
            targets_df = pd.DataFrame(profit_targets)
            st.table(targets_df)
            
            # Visual representation
            prices = [stop_loss_price, entry_price] + [target['Target Price'] for target in profit_targets]
            labels = ['Stop Loss', 'Entry'] + [target['Ratio'] for target in profit_targets]
            colors = ['red', 'blue', 'green', 'green', 'green', 'green']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=labels,
                y=prices,
                mode='lines+markers',
                marker=dict(color=colors, size=10),
                line=dict(width=2),
                name='Price Levels'
            ))
            
            fig.update_layout(
                title="Risk/Reward Price Levels",
                xaxis_title="Level",
                yaxis_title="Price ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Stop Loss Calculator
    with tab2:
        st.header("🎯 Stop Loss Calculator")
        st.markdown("Advanced stop loss calculation methods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol_sl = st.text_input("Stock Symbol", value="AAPL", key="sl_symbol")
            entry_price_sl = st.number_input("Entry Price ($)", value=150.00, key="sl_entry")
            
            st.subheader("Stop Loss Methods")
            
            # Percentage-based
            pct_stop = st.slider("Percentage Stop Loss (%)", 1.0, 20.0, 5.0, key="pct_stop")
            pct_stop_price = entry_price_sl * (1 - pct_stop / 100)
            
            # Volatility-based (ATR)
            if symbol_sl:
                try:
                    stock_data = data_fetcher.get_stock_data(symbol_sl.upper(), "2mo")
                    if stock_data is not None and not stock_data.empty:
                        atr = tech_indicators.atr(stock_data['High'], stock_data['Low'], stock_data['Close'])
                        current_atr = atr.iloc[-1] if not atr.empty else 2.0
                        
                        atr_multiplier_sl = st.slider("ATR Multiplier", 0.5, 5.0, 2.0, key="atr_mult_sl")
                        atr_stop_price = entry_price_sl - (atr_multiplier_sl * current_atr)
                        
                        # Support/Resistance levels
                        support_levels, resistance_levels = tech_indicators.detect_support_resistance(stock_data['Close'])
                        
                        if support_levels:
                            nearest_support = max([level for level in support_levels if level < entry_price_sl])
                            support_stop_price = nearest_support * 0.99  # Slightly below support
                        else:
                            support_stop_price = entry_price_sl * 0.95
                            
                    else:
                        current_atr = 2.0
                        atr_stop_price = entry_price_sl - (2.0 * current_atr)
                        support_stop_price = entry_price_sl * 0.95
                except:
                    current_atr = 2.0
                    atr_stop_price = entry_price_sl - (2.0 * current_atr)
                    support_stop_price = entry_price_sl * 0.95
        
        with col2:
            st.subheader("Stop Loss Comparison")
            
            stop_methods = {
                'Percentage': pct_stop_price,
                'ATR-based': atr_stop_price,
                'Support Level': support_stop_price
            }
            
            for method, price in stop_methods.items():
                risk_per_share = entry_price_sl - price
                risk_percentage = (risk_per_share / entry_price_sl) * 100
                
                st.write(f"**{method}**")
                st.write(f"Stop Price: ${price:.2f}")
                st.write(f"Risk per Share: ${risk_per_share:.2f} ({risk_percentage:.1f}%)")
                st.write("---")
            
            # Trailing stop calculator
            st.subheader("Trailing Stop Calculator")
            
            current_price = st.number_input("Current Price ($)", value=entry_price_sl * 1.1, key="current_price")
            trailing_pct = st.slider("Trailing Stop (%)", 1.0, 15.0, 5.0, key="trailing_pct")
            
            trailing_stop_price = current_price * (1 - trailing_pct / 100)
            unrealized_gain = current_price - entry_price_sl
            protected_gain = trailing_stop_price - entry_price_sl
            
            st.metric("Trailing Stop Price", f"${trailing_stop_price:.2f}")
            st.metric("Unrealized Gain", f"${unrealized_gain:.2f}")
            st.metric("Protected Gain", f"${protected_gain:.2f}")
    
    # Tab 3: Portfolio Risk Analysis
    with tab3:
        st.header("📊 Portfolio Risk Analysis")
        st.markdown("Analyze the overall risk of your portfolio")
        
        # Portfolio input
        st.subheader("Portfolio Positions")
        
        if 'portfolio_positions' not in st.session_state:
            st.session_state.portfolio_positions = []
        
        # Add position form
        with st.expander("Add Portfolio Position"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                new_symbol = st.text_input("Symbol", key="portfolio_symbol")
            with col2:
                new_shares = st.number_input("Shares", min_value=1, value=100, key="portfolio_shares")
            with col3:
                new_price = st.number_input("Price ($)", min_value=0.01, value=100.00, step=0.01, key="portfolio_price")
            with col4:
                if st.button("Add Position"):
                    if new_symbol:
                        st.session_state.portfolio_positions.append({
                            'symbol': new_symbol.upper(),
                            'shares': new_shares,
                            'price': new_price,
                            'value': new_shares * new_price
                        })
                        st.success(f"Added {new_symbol} to portfolio")
                        st.rerun()
        
        # Display current positions
        if st.session_state.portfolio_positions:
            portfolio_df = pd.DataFrame(st.session_state.portfolio_positions)
            st.dataframe(portfolio_df)
            
            # Portfolio metrics
            total_value = portfolio_df['value'].sum()
            portfolio_df['weight'] = (portfolio_df['value'] / total_value) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Portfolio Value", f"${total_value:,.2f}")
                st.metric("Number of Positions", len(portfolio_df))
                
            with col2:
                max_position = portfolio_df['weight'].max()
                largest_stock = portfolio_df.loc[portfolio_df['weight'].idxmax(), 'symbol']
                st.metric("Largest Position", f"{largest_stock} ({max_position:.1f}%)")
                
                # Concentration risk
                top_5_weight = portfolio_df.nlargest(5, 'weight')['weight'].sum()
                st.metric("Top 5 Concentration", f"{top_5_weight:.1f}%")
            
            with col3:
                # Risk level assessment
                if max_position > 20:
                    risk_level = "High"
                    risk_color = "red"
                elif max_position > 10:
                    risk_level = "Medium"
                    risk_color = "orange"
                else:
                    risk_level = "Low"
                    risk_color = "green"
                
                st.markdown(f"**Concentration Risk:** <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
            
            # Portfolio allocation chart
            fig = px.pie(
                portfolio_df,
                values='value',
                names='symbol',
                title='Portfolio Allocation'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk analysis for portfolio
            st.subheader("Portfolio Risk Analysis")
            
            try:
                # Fetch price data for correlation analysis
                symbols = portfolio_df['symbol'].tolist()
                price_data = {}
                
                for symbol in symbols:
                    data = data_fetcher.get_stock_data(symbol, "6mo")
                    if data is not None and not data.empty:
                        price_data[symbol] = data['Close']
                
                if price_data:
                    prices_df = pd.DataFrame(price_data)
                    correlation_matrix = risk_manager.correlation_analysis(prices_df)
                    
                    if not correlation_matrix.empty:
                        # Display correlation heatmap
                        fig = px.imshow(
                            correlation_matrix,
                            text_auto=True,
                            aspect="auto",
                            title="Portfolio Correlation Matrix"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate portfolio volatility
                        returns = prices_df.pct_change().dropna()
                        weights = (portfolio_df.set_index('symbol')['value'] / total_value).values
                        
                        portfolio_returns = (returns * weights).sum(axis=1)
                        portfolio_volatility = portfolio_returns.std() * np.sqrt(252) * 100
                        
                        st.metric("Estimated Annual Volatility", f"{portfolio_volatility:.1f}%")
            
            except Exception as e:
                st.warning("Could not perform correlation analysis. Please ensure all symbols are valid.")
            
            # Clear portfolio button
            if st.button("Clear Portfolio", type="secondary"):
                st.session_state.portfolio_positions = []
                st.rerun()
        
        else:
            st.info("Add some positions to analyze portfolio risk")
    
    # Tab 4: Monte Carlo Simulation
    with tab4:
        st.header("🎲 Monte Carlo Simulation")
        st.markdown("Simulate potential portfolio outcomes using historical data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Simulation parameters
            sim_symbol = st.text_input("Stock Symbol for Simulation", value="SPY")
            initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000)
            simulation_days = st.slider("Trading Days to Simulate", 30, 252*2, 252)
            num_simulations = st.slider("Number of Simulations", 100, 5000, 1000, step=100)
            
            if st.button("Run Monte Carlo Simulation"):
                if sim_symbol:
                    try:
                        # Fetch historical data
                        stock_data = data_fetcher.get_stock_data(sim_symbol.upper(), "2y")
                        
                        if stock_data is not None and not stock_data.empty:
                            returns = stock_data['Close'].pct_change().dropna()
                            
                            # Run simulation
                            simulation_results = risk_manager.monte_carlo_simulation(
                                returns, initial_capital, num_simulations, simulation_days
                            )
                            
                            if simulation_results:
                                st.success("Simulation completed!")
                                st.session_state.simulation_results = simulation_results
                                st.session_state.simulation_symbol = sim_symbol.upper()
                        else:
                            st.error("Could not fetch data for simulation")
                    except Exception as e:
                        st.error(f"Simulation error: {str(e)}")
        
        with col2:
            # Display results
            if 'simulation_results' in st.session_state:
                results = st.session_state.simulation_results
                symbol = st.session_state.simulation_symbol
                
                st.subheader(f"Simulation Results - {symbol}")
                
                # Key metrics
                st.metric("Mean Final Value", f"${results['Mean Final Value']:,.2f}")
                st.metric("Median Final Value", f"${results['Median Final Value']:,.2f}")
                
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    st.metric("Best Case (95th %ile)", f"${results['Best Case (95th percentile)']:,.2f}")
                    st.metric("Worst Case (5th %ile)", f"${results['Worst Case (5th percentile)']:,.2f}")
                
                with col2_2:
                    st.metric("Probability of Loss", f"{results['Probability of Loss']*100:.1f}%")
                    st.metric("Standard Deviation", f"${results['Standard Deviation']:,.2f}")
                
                # Interpretation
                expected_return = ((results['Mean Final Value'] - initial_capital) / initial_capital) * 100
                
                if expected_return > 10:
                    st.success(f"🟢 Positive outlook: Expected return of {expected_return:.1f}%")
                elif expected_return > 0:
                    st.info(f"🟡 Modest gains expected: {expected_return:.1f}%")
                else:
                    st.error(f"🔴 Negative outlook: Expected return of {expected_return:.1f}%")
    
    # Tab 5: Kelly Criterion
    with tab5:
        st.header("🧮 Kelly Criterion Calculator")
        st.markdown("Calculate optimal position sizing based on your trading edge")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trading Statistics")
            st.info("💡 **Tip**: Use your historical trading data to get accurate inputs")
            
            win_rate = st.slider("Win Rate (%)", 10, 90, 55) / 100
            avg_win = st.number_input("Average Win ($)", min_value=0.01, value=100.00, step=0.01)
            avg_loss = st.number_input("Average Loss ($)", min_value=0.01, value=50.00, step=0.01)
            
            # Calculate Kelly percentage
            kelly_pct = risk_manager.kelly_criterion(win_rate, avg_win, avg_loss)
            
            st.subheader("Kelly Criterion Results")
            
            # Display results with interpretation
            if kelly_pct > 0:
                st.success(f"📊 **Optimal Position Size**: {kelly_pct*100:.1f}% of capital")
                
                # Practical recommendations
                conservative_kelly = kelly_pct * 0.5  # Half Kelly
                quarter_kelly = kelly_pct * 0.25  # Quarter Kelly
                
                st.write("**Recommended Position Sizes:**")
                st.write(f"• **Full Kelly**: {kelly_pct*100:.1f}% (Aggressive)")
                st.write(f"• **Half Kelly**: {conservative_kelly*100:.1f}% (Conservative)")
                st.write(f"• **Quarter Kelly**: {quarter_kelly*100:.1f}% (Very Conservative)")
                
                # Risk assessment
                if kelly_pct > 0.2:
                    st.warning("⚠️ High Kelly percentage suggests very strong edge but high risk")
                elif kelly_pct > 0.1:
                    st.info("ℹ️ Moderate Kelly percentage indicates good trading edge")
                else:
                    st.success("✅ Conservative Kelly percentage suggests sustainable approach")
            else:
                st.error("🔴 **No Edge Detected**: Your current statistics suggest negative expectancy")
                st.write("**Recommendations:**")
                st.write("• Review and improve your trading strategy")
                st.write("• Focus on risk management")
                st.write("• Consider paper trading to refine your approach")
        
        with col2:
            # Kelly simulation
            st.subheader("Position Size Impact Analysis")
            
            # Calculate expectancy
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            expectancy_pct = (expectancy / avg_loss) * 100 if avg_loss > 0 else 0
            
            st.metric("Trade Expectancy", f"${expectancy:.2f}")
            st.metric("Expectancy %", f"{expectancy_pct:.2f}%")
            
            # Position size comparison
            position_sizes = [0.01, 0.02, 0.05, 0.1, kelly_pct * 0.25, kelly_pct * 0.5, kelly_pct]
            expected_growth_rates = []
            
            for size in position_sizes:
                if size > 0 and size <= 1:
                    # Simplified growth rate calculation
                    growth_rate = (win_rate * np.log(1 + (avg_win * size / 100))) + \
                                 ((1 - win_rate) * np.log(1 - (avg_loss * size / 100)))
                    expected_growth_rates.append(growth_rate * 100)
                else:
                    expected_growth_rates.append(0)
            
            # Create chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[size*100 for size in position_sizes],
                y=expected_growth_rates,
                mode='lines+markers',
                name='Expected Growth Rate',
                line=dict(width=3),
                marker=dict(size=8)
            ))
            
            # Highlight optimal Kelly point
            if kelly_pct > 0:
                fig.add_vline(
                    x=kelly_pct*100, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Optimal Kelly"
                )
            
            fig.update_layout(
                title="Position Size vs Expected Growth Rate",
                xaxis_title="Position Size (% of Capital)",
                yaxis_title="Expected Growth Rate (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trading tips
            st.subheader("💡 Kelly Criterion Tips")
            st.markdown("""
            **Key Points:**
            - Kelly Criterion maximizes long-term growth
            - Full Kelly can be very volatile
            - Half Kelly offers better risk-adjusted returns
            - Overestimating edge can lead to overbetting
            - Consider fractional Kelly for practical trading
            
            **Warning Signs:**
            - Very high Kelly percentages (>20%)
            - Negative Kelly (no trading edge)
            - Inconsistent win rates or profit/loss ratios
            """)

if __name__ == "__main__":
    main()
