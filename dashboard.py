#!/usr/bin/env python3
"""Streamlit dashboard for Stock-Grok ETL pipeline."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Stock-Grok Dashboard",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def get_db_connection():
    import os
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'airflow'),
        user=os.getenv('DB_USER', 'airflow'),
        password=os.getenv('DB_PASSWORD', 'airflow')
    )

@st.cache_data(ttl=60)
def load_stock_prices():
    conn = get_db_connection()
    df = pd.read_sql("""
        SELECT ticker, date, open, high, low, close, volume
        FROM raw.stock_prices
        ORDER BY date, ticker
    """, conn)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(ttl=60)
def load_grok_explanations():
    conn = get_db_connection()
    df = pd.read_sql("""
        SELECT ticker, date, explanation, sentiment, topic
        FROM raw.grok_explanations
        ORDER BY date DESC, ticker
    """, conn)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(ttl=60)
def load_price_changes():
    conn = get_db_connection()
    df = pd.read_sql("""
        WITH prices_with_prev AS (
            SELECT ticker, date, close,
                   LAG(close) OVER (PARTITION BY ticker ORDER BY date) as prev_close
            FROM raw.stock_prices
        )
        SELECT p.ticker, p.date, p.close, p.prev_close,
               ((p.close - p.prev_close) / p.prev_close * 100) as pct_change,
               g.sentiment, g.topic, g.explanation
        FROM prices_with_prev p
        LEFT JOIN raw.grok_explanations g ON p.ticker = g.ticker AND p.date = g.date
        WHERE p.prev_close IS NOT NULL
        ORDER BY p.date DESC, p.ticker
    """, conn)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Header
st.title("📈 Stock-Grok Dashboard")
st.markdown("Real-time stock analysis powered by Grok AI")

# Load data
try:
    prices_df = load_stock_prices()
    grok_df = load_grok_explanations()
    changes_df = load_price_changes()
except Exception as e:
    st.error(f"Database connection error: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

tickers = sorted(prices_df['ticker'].unique())
selected_tickers = st.sidebar.multiselect(
    "Select Tickers",
    tickers,
    default=tickers
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(prices_df['date'].min(), prices_df['date'].max()),
    min_value=prices_df['date'].min(),
    max_value=prices_df['date'].max()
)

# Filter data
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (prices_df['date'].dt.date >= start_date) & (prices_df['date'].dt.date <= end_date)
    filtered_prices = prices_df[mask & prices_df['ticker'].isin(selected_tickers)]

    mask_changes = (changes_df['date'].dt.date >= start_date) & (changes_df['date'].dt.date <= end_date)
    filtered_changes = changes_df[mask_changes & changes_df['ticker'].isin(selected_tickers)]
else:
    filtered_prices = prices_df[prices_df['ticker'].isin(selected_tickers)]
    filtered_changes = changes_df[changes_df['ticker'].isin(selected_tickers)]

# Key metrics
st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(prices_df):,}")
with col2:
    st.metric("Grok Explanations", f"{len(grok_df):,}")
with col3:
    st.metric("Date Range", f"{(prices_df['date'].max() - prices_df['date'].min()).days} days")
with col4:
    avg_change = filtered_changes['pct_change'].mean()
    st.metric("Avg Daily Change", f"{avg_change:.2f}%")

# Stock price chart
st.header("Stock Prices")

fig = px.line(
    filtered_prices,
    x='date',
    y='close',
    color='ticker',
    title='Stock Closing Prices Over Time',
    labels={'close': 'Price ($)', 'date': 'Date', 'ticker': 'Ticker'}
)
fig.update_layout(height=400, hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)

# Price changes heatmap
st.header("Daily Price Changes")

col1, col2 = st.columns([2, 1])

with col1:
    # Pivot for heatmap
    if not filtered_changes.empty:
        pivot_df = filtered_changes.pivot_table(
            index='ticker',
            columns='date',
            values='pct_change',
            aggfunc='first'
        )

        fig_heatmap = px.imshow(
            pivot_df,
            labels=dict(x="Date", y="Ticker", color="% Change"),
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            aspect='auto',
            title='Price Change Heatmap'
        )
        fig_heatmap.update_layout(height=300)
        st.plotly_chart(fig_heatmap, use_container_width=True)

with col2:
    # Sentiment distribution
    if not filtered_changes.empty and 'sentiment' in filtered_changes.columns:
        sentiment_counts = filtered_changes['sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='Sentiment Distribution',
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': '#00cc66',
                'negative': '#ff4444',
                'neutral': '#888888'
            }
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

# Topic analysis
st.header("Topic Analysis")

col1, col2 = st.columns(2)

with col1:
    if not filtered_changes.empty and 'topic' in filtered_changes.columns:
        topic_counts = filtered_changes['topic'].value_counts()
        fig_topics = px.bar(
            x=topic_counts.index,
            y=topic_counts.values,
            title='Topics by Frequency',
            labels={'x': 'Topic', 'y': 'Count'},
            color=topic_counts.index
        )
        fig_topics.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_topics, use_container_width=True)

with col2:
    if not filtered_changes.empty and 'topic' in filtered_changes.columns:
        topic_sentiment = filtered_changes.groupby(['topic', 'sentiment']).size().reset_index(name='count')
        fig_topic_sent = px.bar(
            topic_sentiment,
            x='topic',
            y='count',
            color='sentiment',
            title='Sentiment by Topic',
            barmode='group',
            color_discrete_map={
                'positive': '#00cc66',
                'negative': '#ff4444',
                'neutral': '#888888'
            }
        )
        fig_topic_sent.update_layout(height=300)
        st.plotly_chart(fig_topic_sent, use_container_width=True)

# Top movers
st.header("Top Movers")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🚀 Top Gainers")
    top_gainers = filtered_changes.nlargest(10, 'pct_change')[['ticker', 'date', 'pct_change', 'sentiment', 'topic']]
    top_gainers['pct_change'] = top_gainers['pct_change'].apply(lambda x: f"+{x:.2f}%")
    top_gainers['date'] = top_gainers['date'].dt.strftime('%Y-%m-%d')
    st.dataframe(top_gainers, use_container_width=True, hide_index=True)

with col2:
    st.subheader("📉 Top Losers")
    top_losers = filtered_changes.nsmallest(10, 'pct_change')[['ticker', 'date', 'pct_change', 'sentiment', 'topic']]
    top_losers['pct_change'] = top_losers['pct_change'].apply(lambda x: f"{x:.2f}%")
    top_losers['date'] = top_losers['date'].dt.strftime('%Y-%m-%d')
    st.dataframe(top_losers, use_container_width=True, hide_index=True)

# Recent Grok explanations
st.header("Recent Grok Explanations")

# Ticker filter for explanations
selected_ticker_exp = st.selectbox("Filter by Ticker", ["All"] + list(tickers))

if selected_ticker_exp == "All":
    recent_explanations = filtered_changes.head(20)
else:
    recent_explanations = filtered_changes[filtered_changes['ticker'] == selected_ticker_exp].head(20)

for _, row in recent_explanations.iterrows():
    sentiment_color = {
        'positive': '🟢',
        'negative': '🔴',
        'neutral': '⚪'
    }.get(row['sentiment'], '⚪')

    pct = row['pct_change']
    pct_str = f"+{pct:.2f}%" if pct > 0 else f"{pct:.2f}%"
    pct_color = "green" if pct > 0 else "red"

    with st.expander(f"{row['ticker']} | {row['date'].strftime('%Y-%m-%d')} | :{pct_color}[{pct_str}] {sentiment_color}"):
        st.markdown(f"**Topic:** {row['topic']}")
        st.markdown(f"**Explanation:** {row['explanation']}")

# Volume analysis
st.header("Volume Analysis")

fig_vol = px.bar(
    filtered_prices,
    x='date',
    y='volume',
    color='ticker',
    title='Trading Volume Over Time',
    labels={'volume': 'Volume', 'date': 'Date'},
    barmode='group'
)
fig_vol.update_layout(height=400)
st.plotly_chart(fig_vol, use_container_width=True)

# Volatility by ticker
st.header("Volatility by Ticker")

volatility = filtered_changes.groupby('ticker')['pct_change'].agg(['std', 'mean', 'count']).reset_index()
volatility.columns = ['Ticker', 'Volatility (Std)', 'Avg Change (%)', 'Days']
volatility = volatility.sort_values('Volatility (Std)', ascending=False)

fig_vol = px.bar(
    volatility,
    x='Ticker',
    y='Volatility (Std)',
    title='Price Volatility by Ticker (Standard Deviation of Daily Returns)',
    color='Volatility (Std)',
    color_continuous_scale='Reds'
)
fig_vol.update_layout(height=350)
st.plotly_chart(fig_vol, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Data refreshes every 60 seconds. Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*")
