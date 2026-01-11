import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psycopg2
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="Stock Sentiment Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database connection
@st.cache_resource
def get_connection():
    """Create database connection"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'postgres'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'airflow'),
        user=os.getenv('DB_USER', 'airflow'),
        password=os.getenv('DB_PASS', 'airflow')
    )

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(query):
    """Load data from database"""
    conn = get_connection()
    df = pd.read_sql_query(query, conn)
    return df

@st.cache_data(ttl=300)
def get_fact_data():
    """Load main fact table with stock prices and sentiment"""
    query = """
    SELECT
        ticker,
        date,
        open,
        high,
        low,
        close,
        volume,
        price_change,
        pct_change,
        explanation,
        sentiment,
        topic,
        move_category
    FROM analytics.fct_prices_with_grok
    ORDER BY date DESC, ticker
    """
    return load_data(query)

def create_candlestick_chart(df, ticker):
    """Create candlestick chart with volume"""
    df_ticker = df[df['ticker'] == ticker].sort_values('date')

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{ticker} Price Movement', 'Volume'),
        row_heights=[0.7, 0.3]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df_ticker['date'],
            open=df_ticker['open'],
            high=df_ticker['high'],
            low=df_ticker['low'],
            close=df_ticker['close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Volume bar chart
    colors = ['red' if row['close'] < row['open'] else 'green'
              for idx, row in df_ticker.iterrows()]

    fig.add_trace(
        go.Bar(
            x=df_ticker['date'],
            y=df_ticker['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    return fig

def create_sentiment_pie_chart(df, ticker):
    """Create sentiment distribution pie chart"""
    df_ticker = df[df['ticker'] == ticker]
    sentiment_counts = df_ticker['sentiment'].value_counts()

    colors = {
        'positive': '#00CC96',
        'negative': '#EF553B',
        'neutral': '#636EFA'
    }

    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        marker_colors=[colors.get(s.lower(), '#636EFA') for s in sentiment_counts.index],
        hole=0.3
    )])

    fig.update_layout(
        title=f'{ticker} Sentiment Distribution',
        height=400
    )

    return fig

def create_topic_bar_chart(df, ticker):
    """Create topic distribution bar chart"""
    df_ticker = df[df['ticker'] == ticker]
    topic_counts = df_ticker['topic'].value_counts()

    fig = go.Figure(data=[go.Bar(
        x=topic_counts.index,
        y=topic_counts.values,
        marker_color='lightblue'
    )])

    fig.update_layout(
        title=f'{ticker} Topic Distribution',
        xaxis_title='Topic',
        yaxis_title='Count',
        height=400
    )

    return fig

def create_price_change_timeline(df, ticker):
    """Create price change percentage timeline"""
    df_ticker = df[df['ticker'] == ticker].sort_values('date')

    colors = ['red' if x < 0 else 'green' for x in df_ticker['pct_change']]

    fig = go.Figure(data=[go.Bar(
        x=df_ticker['date'],
        y=df_ticker['pct_change'],
        marker_color=colors,
        text=df_ticker['pct_change'].round(2),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Change: %{y:.2f}%<extra></extra>'
    )])

    fig.update_layout(
        title=f'{ticker} Daily Price Change %',
        xaxis_title='Date',
        yaxis_title='Price Change (%)',
        height=400,
        hovermode='x unified'
    )

    return fig

def create_multi_stock_comparison(df, tickers):
    """Create multi-stock price comparison"""
    fig = go.Figure()

    for ticker in tickers:
        df_ticker = df[df['ticker'] == ticker].sort_values('date')
        # Normalize to percentage change from first day
        if len(df_ticker) > 0:
            first_close = df_ticker.iloc[-1]['close']
            df_ticker['normalized'] = ((df_ticker['close'] - first_close) / first_close) * 100

            fig.add_trace(go.Scatter(
                x=df_ticker['date'],
                y=df_ticker['normalized'],
                mode='lines',
                name=ticker,
                hovertemplate=f'<b>{ticker}</b><br>%{{x}}<br>Change: %{{y:.2f}}%<extra></extra>'
            ))

    fig.update_layout(
        title='Stock Performance Comparison (Normalized)',
        xaxis_title='Date',
        yaxis_title='Percentage Change from Start (%)',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def create_sentiment_heatmap(df):
    """Create sentiment heatmap across tickers and dates"""
    # Prepare sentiment score mapping
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}

    df_heatmap = df.copy()
    df_heatmap['sentiment_score'] = df_heatmap['sentiment'].map(sentiment_map)

    # Pivot for heatmap
    pivot_data = df_heatmap.pivot_table(
        values='sentiment_score',
        index='ticker',
        columns='date',
        aggfunc='mean'
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=[[0, 'red'], [0.5, 'yellow'], [1, 'green']],
        zmid=0,
        text=pivot_data.values,
        hovertemplate='Ticker: %{y}<br>Date: %{x}<br>Sentiment: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='Sentiment Heatmap Across Tickers',
        xaxis_title='Date',
        yaxis_title='Ticker',
        height=400
    )

    return fig

# Main app
def main():
    st.title("📈 Stock Sentiment Analysis Dashboard")
    st.markdown("Real-time insights from stock prices and AI-powered sentiment analysis")

    # Load data
    try:
        df = get_fact_data()

        if df.empty:
            st.warning("⚠️ No data available. Please run the ETL pipeline first.")
            return

        # Sidebar filters
        st.sidebar.header("Filters")

        # Date range filter
        min_date = df['date'].min()
        max_date = df['date'].max()

        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]

        # Ticker selection
        all_tickers = sorted(df['ticker'].unique())
        selected_tickers = st.sidebar.multiselect(
            "Select Tickers",
            options=all_tickers,
            default=all_tickers
        )

        if selected_tickers:
            df = df[df['ticker'].isin(selected_tickers)]

        # Sentiment filter
        sentiment_filter = st.sidebar.multiselect(
            "Sentiment",
            options=['positive', 'negative', 'neutral'],
            default=['positive', 'negative', 'neutral']
        )

        if sentiment_filter:
            df = df[df['sentiment'].isin(sentiment_filter)]

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "📈 Stock Analysis",
            "💭 Sentiment Analysis",
            "🔍 AI Explanations",
            "⚠️ Large Moves"
        ])

        with tab1:
            st.header("Overview")

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Records", len(df))

            with col2:
                avg_change = df['pct_change'].mean()
                st.metric("Avg Daily Change", f"{avg_change:.2f}%")

            with col3:
                positive_sentiment = (df['sentiment'] == 'positive').sum()
                sentiment_pct = (positive_sentiment / len(df)) * 100 if len(df) > 0 else 0
                st.metric("Positive Sentiment", f"{sentiment_pct:.1f}%")

            with col4:
                large_moves = (df['move_category'] == 'large_move').sum()
                st.metric("Large Moves (>5%)", large_moves)

            st.markdown("---")

            # Multi-stock comparison
            if len(selected_tickers) > 1:
                st.subheader("Stock Performance Comparison")
                fig_comparison = create_multi_stock_comparison(df, selected_tickers)
                st.plotly_chart(fig_comparison, use_container_width=True)

            # Sentiment heatmap
            st.subheader("Sentiment Heatmap")
            fig_heatmap = create_sentiment_heatmap(df)
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Recent data table
            st.subheader("Recent Data")
            recent_df = df.nlargest(10, 'date')[['date', 'ticker', 'close', 'pct_change', 'sentiment', 'topic']]
            st.dataframe(recent_df, use_container_width=True)

        with tab2:
            st.header("Stock Price Analysis")

            # Ticker selector for detailed view
            ticker_detail = st.selectbox(
                "Select ticker for detailed analysis",
                options=selected_tickers if selected_tickers else all_tickers
            )

            if ticker_detail:
                col1, col2 = st.columns(2)

                with col1:
                    # Candlestick chart
                    fig_candlestick = create_candlestick_chart(df, ticker_detail)
                    st.plotly_chart(fig_candlestick, use_container_width=True)

                with col2:
                    # Price change timeline
                    fig_change = create_price_change_timeline(df, ticker_detail)
                    st.plotly_chart(fig_change, use_container_width=True)

                # Statistics
                df_ticker = df[df['ticker'] == ticker_detail]

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Current Price", f"${df_ticker.iloc[0]['close']:.2f}")

                with col2:
                    avg_volume = df_ticker['volume'].mean()
                    st.metric("Avg Volume", f"{avg_volume:,.0f}")

                with col3:
                    max_gain = df_ticker['pct_change'].max()
                    st.metric("Max Daily Gain", f"{max_gain:.2f}%")

                with col4:
                    max_loss = df_ticker['pct_change'].min()
                    st.metric("Max Daily Loss", f"{max_loss:.2f}%")

        with tab3:
            st.header("Sentiment Analysis")

            ticker_sentiment = st.selectbox(
                "Select ticker for sentiment analysis",
                options=selected_tickers if selected_tickers else all_tickers,
                key='sentiment_ticker'
            )

            if ticker_sentiment:
                col1, col2 = st.columns(2)

                with col1:
                    fig_sentiment = create_sentiment_pie_chart(df, ticker_sentiment)
                    st.plotly_chart(fig_sentiment, use_container_width=True)

                with col2:
                    fig_topic = create_topic_bar_chart(df, ticker_sentiment)
                    st.plotly_chart(fig_topic, use_container_width=True)

                # Sentiment over time
                df_ticker = df[df['ticker'] == ticker_sentiment].sort_values('date')
                sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
                df_ticker['sentiment_score'] = df_ticker['sentiment'].map(sentiment_map)

                fig_sentiment_time = go.Figure()
                fig_sentiment_time.add_trace(go.Scatter(
                    x=df_ticker['date'],
                    y=df_ticker['sentiment_score'],
                    mode='lines+markers',
                    name='Sentiment',
                    marker=dict(
                        color=df_ticker['sentiment_score'],
                        colorscale=[[0, 'red'], [0.5, 'yellow'], [1, 'green']],
                        showscale=True,
                        cmin=-1,
                        cmax=1
                    )
                ))

                fig_sentiment_time.update_layout(
                    title=f'{ticker_sentiment} Sentiment Over Time',
                    xaxis_title='Date',
                    yaxis_title='Sentiment Score',
                    height=400,
                    yaxis=dict(tickvals=[-1, 0, 1], ticktext=['Negative', 'Neutral', 'Positive'])
                )

                st.plotly_chart(fig_sentiment_time, use_container_width=True)

        with tab4:
            st.header("AI-Generated Explanations")

            ticker_explain = st.selectbox(
                "Select ticker",
                options=selected_tickers if selected_tickers else all_tickers,
                key='explain_ticker'
            )

            if ticker_explain:
                df_explain = df[df['ticker'] == ticker_explain].sort_values('date', ascending=False)

                for idx, row in df_explain.iterrows():
                    if pd.notna(row['explanation']):
                        sentiment_color = {
                            'positive': '🟢',
                            'negative': '🔴',
                            'neutral': '🟡'
                        }.get(row['sentiment'], '⚪')

                        with st.expander(
                            f"{sentiment_color} {row['date']} - {row['ticker']} "
                            f"({row['pct_change']:+.2f}%) - {row['topic']}"
                        ):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Price Change", f"{row['pct_change']:+.2f}%")

                            with col2:
                                st.metric("Sentiment", row['sentiment'].title())

                            with col3:
                                st.metric("Topic", row['topic'].title())

                            st.markdown("**AI Explanation:**")
                            st.write(row['explanation'])

        with tab5:
            st.header("Large Price Moves (>5%)")

            df_large = df[df['move_category'] == 'large_move'].sort_values('date', ascending=False)

            if df_large.empty:
                st.info("No large moves (>5%) in the selected date range.")
            else:
                st.write(f"Found {len(df_large)} large price movements")

                for idx, row in df_large.iterrows():
                    move_icon = "📈" if row['pct_change'] > 0 else "📉"
                    sentiment_icon = {
                        'positive': '🟢',
                        'negative': '🔴',
                        'neutral': '🟡'
                    }.get(row['sentiment'], '⚪')

                    with st.expander(
                        f"{move_icon} {sentiment_icon} {row['date']} - {row['ticker']} "
                        f"({row['pct_change']:+.2f}%)"
                    ):
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Price Change", f"{row['pct_change']:+.2f}%")

                        with col2:
                            st.metric("Close Price", f"${row['close']:.2f}")

                        with col3:
                            st.metric("Sentiment", row['sentiment'].title())

                        with col4:
                            st.metric("Topic", row['topic'].title())

                        if pd.notna(row['explanation']):
                            st.markdown("**AI Explanation:**")
                            st.write(row['explanation'])

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Make sure the database is running and the ETL pipeline has been executed.")

if __name__ == "__main__":
    main()
