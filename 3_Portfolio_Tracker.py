import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

from utils.portfolio_tracker import PortfolioTracker
from utils.data_fetcher import DataFetcher
from utils.risk_management import RiskManager

st.set_page_config(page_title="Portfolio Tracker", page_icon="💼", layout="wide")

def main():
    st.title("💼 Portfolio Tracker")
    st.markdown("Track your trading positions and analyze portfolio performance")
    
    # Risk disclaimer
    st.warning("📊 **Portfolio Tracking**: This tool helps you monitor your positions. Always verify with your broker's official statements.")
    
    # Initialize classes
    portfolio_tracker = PortfolioTracker()
    data_fetcher = DataFetcher()
    risk_manager = RiskManager()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Current Positions", 
        "📈 Performance Analysis", 
        "⚖️ Risk Analysis",
        "📋 Trade Journal"
    ])
    
    # Tab 1: Current Positions
    with tab1:
        st.header("📊 Current Portfolio Positions")
        
        # Add new position section
        with st.expander("➕ Add New Position", expanded=False):
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                new_symbol = st.text_input("Symbol", placeholder="e.g., AAPL", key="new_symbol")
            with col2:
                new_shares = st.number_input("Shares", min_value=1, value=100, key="new_shares")
            with col3:
                new_entry_price = st.number_input("Entry Price ($)", min_value=0.01, value=100.00, step=0.01, key="new_entry_price")
            with col4:
                new_entry_date = st.date_input("Entry Date", value=datetime.now().date(), key="new_entry_date")
            with col5:
                new_position_type = st.selectbox("Position Type", ["long", "short"], key="new_position_type")
            
            col_add, col_clear = st.columns([1, 4])
            with col_add:
                if st.button("Add Position", type="primary"):
                    if new_symbol:
                        portfolio_tracker.add_position(
                            new_symbol.upper(),
                            new_shares,
                            new_entry_price,
                            new_entry_date.strftime('%Y-%m-%d'),
                            new_position_type
                        )
                        st.success(f"Added {new_symbol.upper()} position!")
                        st.rerun()
                    else:
                        st.error("Please enter a stock symbol")
        
        # Display current portfolio
        portfolio_df = portfolio_tracker.get_portfolio_summary()
        
        if not portfolio_df.empty:
            # Portfolio overview metrics
            metrics = portfolio_tracker.calculate_portfolio_metrics()
            
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Value", 
                        f"${metrics['Total Current Value']:,.2f}",
                        f"${metrics['Total P&L']:,.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Total Return", 
                        f"{metrics['Total P&L %']:.2f}%",
                        f"{metrics['Number of Positions']} positions"
                    )
                
                with col3:
                    st.metric(
                        "Win Rate", 
                        f"{metrics['Win Rate']:.1f}%",
                        f"{metrics['Winning Positions']}W / {metrics['Losing Positions']}L"
                    )
                
                with col4:
                    st.metric(
                        "Largest Position", 
                        f"{metrics['Largest Position %']:.1f}%",
                        "of portfolio"
                    )
            
            # Portfolio positions table
            st.subheader("Position Details")
            
            # Format the dataframe for display
            display_df = portfolio_df.copy()
            
            # Format numeric columns
            numeric_columns = ['Entry Price', 'Current Price', 'Entry Value', 'Current Value', 'Total P&L']
            for col in numeric_columns:
                if col in display_df.columns and display_df[col].dtype != 'object':
                    display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
            
            if 'P&L %' in display_df.columns:
                display_df['P&L %'] = display_df['P&L %'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
            
            # Color code P&L
            def highlight_pnl(val):
                if isinstance(val, str) and val != "N/A":
                    try:
                        num_val = float(val.replace('$', '').replace(',', ''))
                        if num_val > 0:
                            return 'background-color: #d4edda; color: #155724'
                        elif num_val < 0:
                            return 'background-color: #f8d7da; color: #721c24'
                    except:
                        pass
                return ''
            
            styled_df = display_df.style.applymap(highlight_pnl, subset=['Total P&L'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Position management
            if len(portfolio_df) > 0:
                st.subheader("Position Management")
                
                # Select position to remove
                col1, col2 = st.columns([2, 1])
                with col1:
                    position_to_remove = st.selectbox(
                        "Select position to remove:",
                        options=range(len(portfolio_df)),
                        format_func=lambda x: f"{portfolio_df.iloc[x]['Symbol']} - {portfolio_df.iloc[x]['Shares']} shares"
                    )
                
                with col2:
                    if st.button("Remove Position", type="secondary"):
                        portfolio_tracker.remove_position(position_to_remove)
                        st.success("Position removed!")
                        st.rerun()
        else:
            st.info("No positions in portfolio. Add your first position above to get started!")
            
            # Sample portfolio suggestions
            st.subheader("💡 Sample Portfolio Ideas")
            st.markdown("""
            **Conservative Portfolio:**
            - SPY (S&P 500 ETF)
            - VTI (Total Stock Market)
            - BND (Bond ETF)
            
            **Growth Portfolio:**
            - QQQ (Tech ETF)
            - Individual growth stocks (AAPL, MSFT, GOOGL)
            
            **Dividend Portfolio:**
            - High dividend yield stocks
            - REITs
            - Utility stocks
            """)
    
    # Tab 2: Performance Analysis
    with tab2:
        st.header("📈 Portfolio Performance Analysis")
        
        if not portfolio_df.empty:
            # Create portfolio visualization charts
            charts = portfolio_tracker.create_portfolio_charts()
            
            if charts:
                col1, col2 = st.columns(2)
                
                # Portfolio allocation chart
                if 'allocation' in charts:
                    with col1:
                        st.plotly_chart(charts['allocation'], use_container_width=True)
                
                # Sector allocation chart
                if 'sector_allocation' in charts:
                    with col2:
                        st.plotly_chart(charts['sector_allocation'], use_container_width=True)
                
                # P&L chart
                if 'pl_chart' in charts:
                    st.plotly_chart(charts['pl_chart'], use_container_width=True)
            
            # Historical performance
            st.subheader("📊 Historical Performance")
            
            days_options = [30, 60, 90, 180, 365]
            selected_days = st.selectbox("Performance Period (days):", days_options, index=2)
            
            if st.button("Calculate Historical Performance"):
                with st.spinner("Calculating performance..."):
                    performance_df = portfolio_tracker.calculate_historical_performance(selected_days)
                    
                    if not performance_df.empty:
                        # Performance chart
                        fig = go.Figure()
                        
                        # Portfolio value line
                        fig.add_trace(go.Scatter(
                            x=performance_df['Date'],
                            y=performance_df['Portfolio Value'],
                            mode='lines',
                            name='Portfolio Value',
                            line=dict(width=3, color='blue')
                        ))
                        
                        fig.update_layout(
                            title=f"Portfolio Performance - Last {selected_days} Days",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Performance metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        if len(performance_df) > 1:
                            total_return = performance_df['Cumulative Return'].iloc[-1]
                            volatility = performance_df['Daily Return'].std() * np.sqrt(252) * 100
                            best_day = performance_df['Daily Return'].max() * 100
                            worst_day = performance_df['Daily Return'].min() * 100
                            
                            with col1:
                                st.metric("Total Return", f"{total_return:.2f}%")
                            with col2:
                                st.metric("Annualized Volatility", f"{volatility:.1f}%")
                            with col3:
                                st.metric("Best Day", f"+{best_day:.2f}%")
                            with col4:
                                st.metric("Worst Day", f"{worst_day:.2f}%")
                        
                        # Daily returns histogram
                        if len(performance_df) > 10:
                            fig_hist = px.histogram(
                                performance_df,
                                x='Daily Return',
                                nbins=20,
                                title="Daily Returns Distribution",
                                labels={'Daily Return': 'Daily Return (%)', 'count': 'Frequency'}
                            )
                            fig_hist.update_traces(marker_color='lightblue', marker_line_color='blue', marker_line_width=1)
                            st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.warning("Could not calculate historical performance. Please check your positions and try again.")
            
            # Portfolio comparison with benchmarks
            st.subheader("📊 Benchmark Comparison")
            
            benchmark_symbols = ['SPY', 'QQQ', 'VTI', 'IWM']
            selected_benchmark = st.selectbox("Compare with:", benchmark_symbols)
            
            if st.button("Compare Performance"):
                try:
                    benchmark_data = data_fetcher.get_stock_data(selected_benchmark, f"{selected_days}d")
                    
                    if benchmark_data is not None and not benchmark_data.empty:
                        # Calculate benchmark returns
                        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
                        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
                        
                        # Create comparison chart
                        fig = go.Figure()
                        
                        # Add benchmark
                        fig.add_trace(go.Scatter(
                            x=benchmark_data.index,
                            y=benchmark_cumulative * 100,
                            mode='lines',
                            name=f'{selected_benchmark} (Benchmark)',
                            line=dict(width=2, color='red', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f"Portfolio vs {selected_benchmark} Performance",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Return (%)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Performance comparison metrics
                        if len(benchmark_cumulative) > 0:
                            benchmark_total_return = benchmark_cumulative.iloc[-1] * 100
                            benchmark_volatility = benchmark_returns.std() * np.sqrt(252) * 100
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(f"{selected_benchmark} Total Return", f"{benchmark_total_return:.2f}%")
                            with col2:
                                st.metric(f"{selected_benchmark} Volatility", f"{benchmark_volatility:.1f}%")
                
                except Exception as e:
                    st.error(f"Error comparing with benchmark: {str(e)}")
        else:
            st.info("Add positions to your portfolio to see performance analysis.")
    
    # Tab 3: Risk Analysis
    with tab3:
        st.header("⚖️ Portfolio Risk Analysis")
        
        if not portfolio_df.empty:
            # Get risk analysis
            risk_analysis = portfolio_tracker.get_risk_analysis()
            
            if risk_analysis:
                # Risk metrics overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    concentration_risk = risk_analysis['Concentration Risk']
                    if concentration_risk > 20:
                        st.error(f"⚠️ High Concentration: {concentration_risk:.1f}%")
                    elif concentration_risk > 10:
                        st.warning(f"⚡ Moderate Concentration: {concentration_risk:.1f}%")
                    else:
                        st.success(f"✅ Low Concentration: {concentration_risk:.1f}%")
                
                with col2:
                    diversification = risk_analysis['Diversification Score']
                    if diversification >= 75:
                        st.success(f"✅ Well Diversified: {diversification}/100")
                    elif diversification >= 50:
                        st.warning(f"⚡ Moderately Diversified: {diversification}/100")
                    else:
                        st.error(f"⚠️ Poor Diversification: {diversification}/100")
                
                with col3:
                    volatility = risk_analysis['Estimated Annual Volatility']
                    st.metric("Est. Annual Volatility", f"{volatility:.1f}%")
                
                with col4:
                    risk_level = risk_analysis['Risk Level']
                    if risk_level == 'Low':
                        st.success(f"✅ Risk Level: {risk_level}")
                    elif risk_level == 'Moderate':
                        st.warning(f"⚡ Risk Level: {risk_level}")
                    else:
                        st.error(f"⚠️ Risk Level: {risk_level}")
            
            # Detailed risk breakdown
            st.subheader("📊 Risk Breakdown by Position")
            
            if not portfolio_df.empty:
                # Calculate position-level risk metrics
                valid_positions = portfolio_df[portfolio_df['Current Value'] != 'N/A'].copy()
                
                if not valid_positions.empty:
                    total_value = valid_positions['Current Value'].sum()
                    
                    risk_breakdown = []
                    for _, position in valid_positions.iterrows():
                        symbol = position['Symbol']
                        weight = (position['Current Value'] / total_value) * 100
                        
                        # Get volatility data
                        try:
                            stock_data = data_fetcher.get_stock_data(symbol, "3mo")
                            if stock_data is not None and not stock_data.empty:
                                returns = stock_data['Close'].pct_change().dropna()
                                volatility = returns.std() * np.sqrt(252) * 100
                            else:
                                volatility = 20  # Default estimate
                        except:
                            volatility = 20
                        
                        risk_contribution = weight * volatility / 100
                        
                        risk_breakdown.append({
                            'Symbol': symbol,
                            'Weight (%)': weight,
                            'Volatility (%)': volatility,
                            'Risk Contribution': risk_contribution,
                            'Risk Level': 'High' if volatility > 30 else 'Medium' if volatility > 20 else 'Low'
                        })
                    
                    risk_df = pd.DataFrame(risk_breakdown)
                    
                    # Risk breakdown chart
                    fig = px.scatter(
                        risk_df,
                        x='Weight (%)',
                        y='Volatility (%)',
                        size='Risk Contribution',
                        color='Risk Level',
                        hover_name='Symbol',
                        title="Position Risk vs Weight Analysis",
                        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk recommendations
                    st.subheader("🎯 Risk Management Recommendations")
                    
                    high_risk_positions = risk_df[risk_df['Risk Level'] == 'High']
                    high_weight_positions = risk_df[risk_df['Weight (%)'] > 15]
                    
                    if not high_risk_positions.empty:
                        st.warning(f"⚠️ **High Volatility Positions**: {', '.join(high_risk_positions['Symbol'].tolist())}")
                        st.write("Consider reducing position sizes or adding hedging instruments.")
                    
                    if not high_weight_positions.empty:
                        st.warning(f"⚠️ **Overweight Positions**: {', '.join(high_weight_positions['Symbol'].tolist())}")
                        st.write("Consider diversifying into other positions to reduce concentration risk.")
                    
                    if len(risk_df) < 5:
                        st.info("💡 **Diversification Opportunity**: Consider adding more positions to improve diversification.")
                    
                    # Position sizing recommendations
                    st.subheader("📏 Optimal Position Sizing")
                    
                    account_size = st.number_input("Account Size for Sizing Analysis ($)", 
                                                 min_value=1000, value=10000, step=1000)
                    max_position_pct = st.slider("Maximum Position Size (%)", 1, 20, 10)
                    
                    st.write("**Recommended Position Sizes:**")
                    for _, position in risk_df.iterrows():
                        symbol = position['Symbol']
                        volatility = position['Volatility (%)']
                        
                        # Risk-adjusted position size (inverse volatility weighting)
                        if volatility > 0:
                            risk_adjusted_size = min(max_position_pct, (20 / volatility) * 5)  # Base calculation
                            recommended_value = account_size * (risk_adjusted_size / 100)
                            
                            st.write(f"• **{symbol}**: {risk_adjusted_size:.1f}% (${recommended_value:,.0f}) - "
                                   f"Vol: {volatility:.1f}%")
        else:
            st.info("Add positions to analyze portfolio risk.")
    
    # Tab 4: Trade Journal
    with tab4:
        st.header("📋 Trade Journal")
        st.markdown("Track your trading decisions and learn from your performance")
        
        # Initialize trade journal in session state
        if 'trade_journal' not in st.session_state:
            st.session_state.trade_journal = []
        
        # Add new trade entry
        with st.expander("➕ Add Trade Entry", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trade_symbol = st.text_input("Symbol", key="journal_symbol")
                trade_type = st.selectbox("Trade Type", ["Buy", "Sell"], key="journal_type")
                trade_shares = st.number_input("Shares", min_value=1, value=100, key="journal_shares")
            
            with col2:
                trade_price = st.number_input("Price ($)", min_value=0.01, value=100.00, step=0.01, key="journal_price")
                trade_date = st.date_input("Trade Date", value=datetime.now().date(), key="journal_date")
                trade_strategy = st.selectbox("Strategy", 
                                            ["Swing Trading", "Day Trading", "Position Trading", "Scalping", "Other"], 
                                            key="journal_strategy")
            
            with col3:
                trade_reason = st.text_area("Entry Reason", placeholder="Why did you enter this trade?", key="journal_reason")
                trade_notes = st.text_area("Additional Notes", placeholder="Any other observations...", key="journal_notes")
            
            if st.button("Add Trade Entry", type="primary"):
                if trade_symbol and trade_reason:
                    trade_entry = {
                        'date': trade_date.strftime('%Y-%m-%d'),
                        'symbol': trade_symbol.upper(),
                        'type': trade_type,
                        'shares': trade_shares,
                        'price': trade_price,
                        'value': trade_shares * trade_price,
                        'strategy': trade_strategy,
                        'reason': trade_reason,
                        'notes': trade_notes,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.session_state.trade_journal.append(trade_entry)
                    st.success("Trade entry added to journal!")
                    st.rerun()
                else:
                    st.error("Please enter symbol and entry reason")
        
        # Display trade journal
        if st.session_state.trade_journal:
            st.subheader("📖 Trade History")
            
            # Convert to DataFrame
            journal_df = pd.DataFrame(st.session_state.trade_journal)
            
            # Sort by date (newest first)
            journal_df = journal_df.sort_values('date', ascending=False)
            
            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_trades = len(journal_df)
            buy_trades = len(journal_df[journal_df['type'] == 'Buy'])
            sell_trades = len(journal_df[journal_df['type'] == 'Sell'])
            unique_symbols = journal_df['symbol'].nunique()
            
            with col1:
                st.metric("Total Trades", total_trades)
            with col2:
                st.metric("Buy Trades", buy_trades)
            with col3:
                st.metric("Sell Trades", sell_trades)
            with col4:
                st.metric("Unique Symbols", unique_symbols)
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                symbol_filter = st.selectbox("Filter by Symbol", 
                                           ["All"] + sorted(journal_df['symbol'].unique().tolist()))
            with col2:
                strategy_filter = st.selectbox("Filter by Strategy", 
                                             ["All"] + sorted(journal_df['strategy'].unique().tolist()))
            with col3:
                type_filter = st.selectbox("Filter by Type", ["All", "Buy", "Sell"])
            
            # Apply filters
            filtered_df = journal_df.copy()
            
            if symbol_filter != "All":
                filtered_df = filtered_df[filtered_df['symbol'] == symbol_filter]
            if strategy_filter != "All":
                filtered_df = filtered_df[filtered_df['strategy'] == strategy_filter]
            if type_filter != "All":
                filtered_df = filtered_df[filtered_df['type'] == type_filter]
            
            # Display filtered results
            if not filtered_df.empty:
                # Format display
                display_columns = ['date', 'symbol', 'type', 'shares', 'price', 'value', 'strategy', 'reason']
                display_df = filtered_df[display_columns].copy()
                
                # Format currency columns
                display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
                display_df['value'] = display_df['value'].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Trading patterns analysis
                st.subheader("📊 Trading Patterns")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Trades by strategy
                    strategy_counts = journal_df['strategy'].value_counts()
                    fig_strategy = px.pie(
                        values=strategy_counts.values,
                        names=strategy_counts.index,
                        title="Trades by Strategy"
                    )
                    st.plotly_chart(fig_strategy, use_container_width=True)
                
                with col2:
                    # Trade value distribution
                    fig_values = px.histogram(
                        journal_df,
                        x='value',
                        nbins=10,
                        title="Trade Value Distribution",
                        labels={'value': 'Trade Value ($)', 'count': 'Number of Trades'}
                    )
                    st.plotly_chart(fig_values, use_container_width=True)
                
                # Most traded symbols
                symbol_counts = journal_df['symbol'].value_counts().head(10)
                if len(symbol_counts) > 0:
                    fig_symbols = px.bar(
                        x=symbol_counts.index,
                        y=symbol_counts.values,
                        title="Most Traded Symbols",
                        labels={'x': 'Symbol', 'y': 'Number of Trades'}
                    )
                    st.plotly_chart(fig_symbols, use_container_width=True)
            
            else:
                st.info("No trades match the selected filters.")
            
            # Export functionality
            st.subheader("📤 Export Journal")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("Export to CSV"):
                    csv_data = journal_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"trade_journal_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col1:
                if st.button("Clear Journal", type="secondary"):
                    if st.checkbox("Confirm clear all entries"):
                        st.session_state.trade_journal = []
                        st.success("Trade journal cleared!")
                        st.rerun()
        
        else:
            st.info("No trades in journal. Add your first trade entry above!")
            
            # Trading journal tips
            st.subheader("💡 Trade Journal Best Practices")
            st.markdown("""
            **Why Keep a Trade Journal?**
            - Track your trading performance over time
            - Identify patterns in your successful and unsuccessful trades
            - Improve your trading discipline and decision-making
            - Learn from past mistakes and replicate successes
            
            **What to Record:**
            - Entry and exit points with reasoning
            - Market conditions at the time of trade
            - Emotional state and confidence level
            - Strategy used and its effectiveness
            - Lessons learned from each trade
            
            **Review Regularly:**
            - Weekly: Review recent trades and performance
            - Monthly: Analyze overall patterns and strategies
            - Quarterly: Assess and adjust your trading plan
            """)

if __name__ == "__main__":
    main()
