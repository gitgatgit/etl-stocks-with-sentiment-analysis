# Stock Sentiment Analysis BI Dashboard

A comprehensive, interactive Business Intelligence dashboard for visualizing stock prices and AI-powered sentiment analysis.

## Overview

This Streamlit-based dashboard provides real-time insights into stock market data enriched with sentiment analysis from xAI's Grok API. It connects directly to the PostgreSQL database populated by the ETL pipeline and offers multiple visualization tools for analyzing price movements, sentiment trends, and AI-generated explanations.

## Features

### 📊 Overview Tab
- **Key Metrics**: Total records, average daily change, positive sentiment percentage, and large moves count
- **Multi-Stock Comparison**: Normalized performance chart comparing multiple stocks
- **Sentiment Heatmap**: Visual representation of sentiment across tickers and dates
- **Recent Data Table**: Latest stock data with prices, changes, and sentiment

### 📈 Stock Analysis Tab
- **Candlestick Charts**: Professional OHLC (Open-High-Low-Close) price visualization
- **Volume Analysis**: Trading volume bars with color-coded price movements
- **Price Change Timeline**: Daily percentage changes over time
- **Statistical Metrics**: Current price, average volume, max daily gain/loss

### 💭 Sentiment Analysis Tab
- **Sentiment Distribution**: Pie chart showing positive/negative/neutral breakdown
- **Topic Distribution**: Bar chart of sentiment topics (earnings, macro, company-specific, speculation)
- **Sentiment Timeline**: Historical sentiment trends with color-coded scoring

### 🔍 AI Explanations Tab
- **Detailed Explanations**: AI-generated analysis for each trading day
- **Expandable Cards**: Organized by date with price change, sentiment, and topic
- **Full Context**: Complete Grok API explanations for price movements

### ⚠️ Large Moves Tab
- **Alert System**: Automatically highlights price movements >5%
- **Detailed Analysis**: Each large move includes price data, sentiment, and AI explanation
- **Visual Indicators**: Icons showing movement direction and sentiment

## Tech Stack

- **Frontend**: Streamlit 1.31.0
- **Data Processing**: Pandas 2.2.0
- **Visualizations**: Plotly 5.18.0
- **Database**: PostgreSQL via psycopg2-binary 2.9.9

## Database Connection

The dashboard connects to the PostgreSQL database using the following configuration:

```python
DB_HOST: postgres
DB_PORT: 5432
DB_NAME: airflow
DB_USER: airflow
DB_PASS: airflow
```

These are automatically configured when running via Docker Compose.

## Data Source

The dashboard queries the `analytics.fct_prices_with_grok` fact table, which includes:

- **Stock Data**: ticker, date, OHLC prices, volume
- **Derived Metrics**: price_change, pct_change, move_category
- **Sentiment Data**: explanation, sentiment, topic
- **Categories**: large_move (>5%), medium_move (>2%), small_move

## Running the Dashboard

### Via Docker Compose (Recommended)

The dashboard is automatically started when you run the full stack:

```bash
docker-compose up -d
```

Access the dashboard at: **http://localhost:8501**

### Standalone Development

For local development without Docker:

```bash
cd dashboard
pip install -r requirements.txt

# Set environment variables
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=airflow
export DB_USER=airflow
export DB_PASS=airflow

# Run the app
streamlit run app.py
```

## Usage Guide

### Filters (Sidebar)

1. **Date Range**: Select start and end dates to focus on specific periods
2. **Ticker Selection**: Choose one or multiple stocks to analyze
3. **Sentiment Filter**: Filter by positive, negative, or neutral sentiment

### Interactive Features

- **Hover Tooltips**: Hover over charts to see detailed data points
- **Zoom & Pan**: Use Plotly controls to zoom into specific time periods
- **Expandable Cards**: Click expanders in AI Explanations and Large Moves tabs for details
- **Real-time Updates**: Data refreshes every 5 minutes (cached for performance)

### Best Practices

1. **Start with Overview**: Get a high-level view before diving into details
2. **Compare Stocks**: Select multiple tickers to see relative performance
3. **Investigate Large Moves**: Check the Large Moves tab for significant events
4. **Read AI Explanations**: Understand the "why" behind price movements

## Performance Optimization

- **Caching**: Data is cached for 5 minutes to reduce database load
- **Connection Pooling**: Database connections are reused efficiently
- **Lazy Loading**: Charts are only generated when tabs are viewed

## Troubleshooting

### Dashboard won't load
- Ensure PostgreSQL container is running: `docker ps | grep postgres`
- Check database connectivity: `docker-compose logs dashboard`
- Verify ETL pipeline has run and populated data

### No data showing
- Run the ETL pipeline: `python cli.py trigger`
- Check database contents: `python debug_db.py`
- Verify date range filters aren't too restrictive

### Connection errors
- Ensure all containers are on the same network (`etl-network`)
- Check environment variables in docker-compose.yml
- Restart the dashboard service: `docker-compose restart dashboard`

## Port Configuration

- **Dashboard**: http://localhost:8501
- **Metabase**: http://localhost:3000 (alternative BI tool)
- **Airflow**: http://localhost:8080
- **PostgreSQL**: localhost:5432

## Development

### Adding New Visualizations

1. Create a new function in `app.py` following the pattern:
   ```python
   def create_my_chart(df, params):
       fig = go.Figure()
       # Add your visualization logic
       return fig
   ```

2. Add the chart to the appropriate tab in the `main()` function

3. Test locally before deploying

### Modifying Database Queries

Edit the SQL in the `get_fact_data()` or `load_data()` functions. Remember to:
- Use parameterized queries for user inputs
- Test queries in PostgreSQL first
- Consider performance for large datasets

## License

MIT License - See main project LICENSE file

## Support

For issues or questions:
1. Check the main project README
2. Review Streamlit documentation: https://docs.streamlit.io
3. Check database connectivity using `debug_db.py`

## Credits

Built with:
- [Streamlit](https://streamlit.io/) - The fastest way to build data apps
- [Plotly](https://plotly.com/) - Interactive graphing library
- [Pandas](https://pandas.pydata.org/) - Data analysis library
