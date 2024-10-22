import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objs as go
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import threading
import time

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Global variables for model and data
model = None
data = None
refresh_data_event = threading.Event()

# Initialize demo balance and open positions
demo_balance = 10000.0  # Starting demo balance
open_positions = []  # List to keep track of open positions
transaction_history = []  # List to store transaction history


# Function to fetch market data with caching
@st.cache_data(ttl=3600)
def fetch_market_data(symbol, interval, period='1mo'):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None


# Function to train a simple model on the fetched data
@st.cache_resource
def train_model(data):
    data['Date'] = data.index.map(pd.Timestamp.toordinal)  # Convert dates to ordinal for modeling
    X = data[['Date']]
    y = data['Close']

    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    return model


# Function to predict future prices
def predict_future_prices(model, last_date, days=1):
    next_days = np.array([[pd.Timestamp(last_date + pd.Timedelta(days=i)).toordinal()] for i in range(1, days + 1)])
    predicted_prices = model.predict(next_days)
    return predicted_prices


# Function to calculate EMA
def calculate_ema(data, span):
    data['EMA'] = data['Close'].ewm(span=span, adjust=False).mean()  # Add EMA directly to DataFrame
    return data


# Function to perform sentiment analysis
def analyze_sentiment(news_articles):
    scores = [sia.polarity_scores(article) for article in news_articles]
    return np.mean([score['compound'] for score in scores])


# Symbol mapping for markets and forex pairs
symbol_mapping = {
    'NIFTY 50': '^NSEI',
    'BANKNIFTY': '^NSEBANK',
    'MIDCAP NIFTY': '^NSEMDCP',
    'FINNIFTY': '^NSEFIN',
    'EUR/USD': 'EURUSD=X',
    'USD/JPY': 'JPY=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/CHF': 'CHF=X',
    'AUD/USD': 'AUDUSD=X',
    'USD/CAD': 'CAD=X',
    'NZD/USD': 'NZDUSD=X',
}


# Function to perform the continuous data refresh
def continuous_data_refresh(symbol, interval):
    global data
    while not refresh_data_event.is_set():
        new_data = fetch_market_data(symbol, interval)
        if new_data is not None:
            data = new_data
            # Refresh the Streamlit UI
            st.experimental_rerun()
        time.sleep(0.1)  # Adjust the refresh rate here (0.1 seconds = 100 milliseconds)


# Function to perform trade simulation
def execute_trade(action, trade_amount, last_close):
    global demo_balance  # Declare demo_balance as global at the start of the function
    global open_positions  # Declare open_positions as global
    global transaction_history  # Declare transaction_history as global

    if action == "Buy":
        if trade_amount <= demo_balance:
            demo_balance -= trade_amount
            position = {
                "action": "Buy",
                "amount": trade_amount,
                "price": last_close,
                "symbol": selected_market
            }
            open_positions.append(position)  # Add position to open positions

            # Record the transaction
            transaction_history.append({
                "action": "Buy",
                "amount": trade_amount,
                "price": last_close,
                "symbol": selected_market,
                "balance_after": demo_balance
            })
            return f"Bought {trade_amount} of {selected_market} at {last_close:.2f}. New Balance: ${demo_balance:.2f}"
        else:
            return "Insufficient balance to execute this trade."
    elif action == "Sell":
        # Check if there's a position to sell
        position_to_sell = next((p for p in open_positions if p['symbol'] == selected_market), None)
        if position_to_sell:
            demo_balance += trade_amount  # Assuming selling returns the trade amount
            open_positions.remove(position_to_sell)  # Remove position from open positions

            # Record the transaction
            transaction_history.append({
                "action": "Sell",
                "amount": trade_amount,
                "price": last_close,
                "symbol": selected_market,
                "balance_after": demo_balance
            })
            return f"Sold {trade_amount} of {selected_market} at {last_close:.2f}. New Balance: ${demo_balance:.2f}"
        else:
            return "No open position to sell."


# Function to calculate total profit and loss from open positions
def calculate_total_pnl():
    total_pnl = 0.0
    for position in open_positions:
        current_price = data['Close'].iloc[-1]  # Current market price
        if position['action'] == "Buy":
            pnl = (current_price - position['price']) * position['amount']  # Profit or loss calculation for Buy
        else:
            pnl = (position['price'] - current_price) * position['amount']  # Profit or loss calculation for Sell
        total_pnl += pnl
    return total_pnl


# Streamlit UI
st.set_page_config(page_title="Market Data and Forecasting", layout="wide")
st.title("Market Data and Forecasting Dashboard")

# Live Balance and Open Trades Display
st.sidebar.header("Account Information")
st.sidebar.write(f"**Demo Balance:** ${demo_balance:.2f}")
st.sidebar.write("### Open Trades")
if open_positions:
    for pos in open_positions:
        st.sidebar.write(f"{pos['action']} {pos['amount']} of {pos['symbol']} at {pos['price']:.2f}")
else:
    st.sidebar.write("No open positions.")

# Display total P&L
total_pnl = calculate_total_pnl()
st.sidebar.write(f"**Total P&L:** ${total_pnl:.2f}")

# Sidebar for user input
st.sidebar.header("Market Selection")
market_symbols = list(symbol_mapping.keys())
selected_market = st.sidebar.selectbox("Select Market Symbol", market_symbols)

interval = st.sidebar.selectbox("Select Interval", ['5m', '15m', '1h', '1d'])
forecast_days = st.sidebar.number_input("Number of Days to Forecast", min_value=1, max_value=30, value=1)
ema_span = st.sidebar.number_input("EMA Span", min_value=1, max_value=50, value=10)

# Refresh button
if st.sidebar.button("Refresh Data"):
    refresh_data_event.clear()  # Clear the event to start refreshing
    threading.Thread(target=continuous_data_refresh, args=(symbol_mapping[selected_market], interval),
                     daemon=True).start()

# Fetching the selected market data initially
data = fetch_market_data(symbol_mapping[selected_market], interval)

if data is not None:
    st.subheader(f"Historical Market Data for {selected_market}")
    st.write(data)

    # Train the model on the fetched data
    model = train_model(data)
    st.success("Initial model trained successfully!")

    # Calculate EMA
    data = calculate_ema(data, ema_span)  # Ensure EMA is calculated and added to the DataFrame

    # Prepare data for the chart
    last_date = data.index[-1]
    future_prices = predict_future_prices(model, last_date, forecast_days)

    # Create a DataFrame for future predictions
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
    future_data = pd.DataFrame(data=future_prices, index=future_dates, columns=['Predicted Close'])

    # Candlestick Chart
    st.subheader("Historical Candlestick Chart")
    candlestick_fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    )])
    candlestick_fig.update_layout(title=f'{selected_market} Historical Candlestick Chart', xaxis_title='Date',
                                  yaxis_title='Price', template='plotly_dark')
    st.plotly_chart(candlestick_fig)

    # Separate charts for historical data, EMA, and predicted prices
    st.subheader("Historical Closing Prices Chart")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=data.index, y=data['EMA'], mode='lines', name='EMA', line=dict(color='orange')))
    fig1.update_layout(title=f'{selected_market} Historical Closing Prices with EMA', xaxis_title='Date',
                       yaxis_title='Price', template='plotly_dark')
    st.plotly_chart(fig1)

    # Predicted Prices Chart
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(x=future_data.index, y=future_data['Predicted Close'], mode='lines', name='Predicted Close',
                   line=dict(color='red')))
    fig2.update_layout(title=f'{selected_market} Predicted Prices', xaxis_title='Date', yaxis_title='Price',
                       template='plotly_dark')
    st.plotly_chart(fig2)

    # Combined historical closing and predicted prices
    fig_combined = go.Figure()
    fig_combined.add_trace(
        go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Close', line=dict(color='blue')))
    fig_combined.add_trace(
        go.Scatter(x=future_data.index, y=future_data['Predicted Close'], mode='lines', name='Predicted Close',
                   line=dict(color='red', dash='dash')))
    fig_combined.update_layout(title=f'{selected_market} Combined Prices', xaxis_title='Date', yaxis_title='Price',
                               template='plotly_dark')
    st.plotly_chart(fig_combined)

    # Trade Execution
    st.sidebar.header("Trade Execution")
    trade_action = st.sidebar.selectbox("Select Action", ["Buy", "Sell"])
    trade_amount = st.sidebar.number_input("Trade Amount", min_value=1.0, max_value=demo_balance, value=1.0)

    if st.sidebar.button("Execute Trade"):
        last_close = data['Close'].iloc[-1]
        trade_result = execute_trade(trade_action, trade_amount, last_close)
        st.success(trade_result)

# Display sentiment analysis if applicable
# You may want to fetch and process news articles for sentiment analysis here

# Stop the continuous data refresh when the app is stopped
refresh_data_event.set()
