import yfinance as yf
from crewai_tools import tool
from datetime import datetime, timedelta


class TickerPrice():
    @tool("Get a stock close price at a specific date")
    def get_stock_close_price(ticker, date):
        """Useful to get the close price of a stock at a specific date.
        The input to this tool should be: ticker, date in the format `YYYY-MM-DD`.
        """
        # Convert date string to datetime object and adjust the end date
        date = datetime.strptime(date, '%Y-%m-%d')
        next_day = date + timedelta(days=2)  # Adding two days to ensure the date is included

        # Fetch data
        stock = yf.Ticker(ticker)
        hist = stock.history(start=date.strftime('%Y-%m-%d'), end=next_day.strftime('%Y-%m-%d'))

        # Retrieve close price if available
        try:
            close_price = hist.loc[date.strftime('%Y-%m-%d'), 'Close']
            return close_price
        except KeyError:
            return "No data available for this date."
