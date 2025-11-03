"""
data_collection.py

Pure data collection from APIs - aligns with architecture diagram.
Separates API calls from data processing as per the flowchart.
Copied and modularized from data.py and news_unstructured_score.py.
"""

import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
import requests
import time
import logging
from datetime import timedelta
import warnings
import os
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"  # Fixed: removed trailing spaces

# === YAHOO FINANCE API DATA COLLECTION ===
def fetch_yahoo_finance_data(ticker, rating_date):
    """
    Fetch financial and market data from Yahoo Finance API.
    This aligns with 'YF - Yahoo Finance API' in the architecture.
    """
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get financial statements
        financials = stock.financials
        balance_sheet = stock.balancesheet  
        cash_flow = stock.cashflow
        
        if financials.empty or balance_sheet.empty:
            logger.warning(f"Insufficient financial statements for {ticker}")
            return None
            
        # Use most recent available fiscal year end
        available_dates = balance_sheet.columns if not balance_sheet.empty else financials.columns
        if available_dates.empty:
            logger.warning(f"No financial data available for {ticker}")
            return None
            
        fye_date = available_dates.max()
        logger.info(f"Using most recent FYE: {fye_date.date()}")
        
        # Get market data around rating date
        hist_start = pd.to_datetime(rating_date) - timedelta(days=30)
        hist_end = pd.to_datetime(rating_date) + timedelta(days=1)
        hist_data = stock.history(start=hist_start, end=hist_end)
        
        # Extract raw financial data
        raw_data = {
            'ticker': ticker,
            'rating_date': rating_date,
            'fye_date': fye_date,
            
            # Income Statement Data
            'total_revenue': _safe_get_yf_data(financials, 'Total Revenue', fye_date),
            'gross_profit': _safe_get_yf_data(financials, 'Gross Profit', fye_date),
            'operating_income': _safe_get_yf_data(financials, 'Operating Income', fye_date),
            'net_income': _safe_get_yf_data(financials, 'Net Income', fye_date),
            
            # Balance Sheet Data  
            'total_assets': _safe_get_yf_data(balance_sheet, 'Total Assets', fye_date),
            'current_assets': _safe_get_yf_data(balance_sheet, 'Current Assets', fye_date),
            'current_liabilities': _safe_get_yf_data(balance_sheet, 'Current Liabilities', fye_date),
            'total_liabilities': _safe_get_yf_data(balance_sheet, 'Total Liabilities Net Minority Interest', fye_date),
            'total_equity': _safe_get_yf_data(balance_sheet, 'Total Equity Gross Minority Interest', fye_date),
            'inventory': _safe_get_yf_data(balance_sheet, 'Inventory', fye_date),
            'cash_and_equivalents': _safe_get_yf_data(balance_sheet, 'Cash And Cash Equivalents', fye_date),
            'accounts_receivable': _safe_get_yf_data(balance_sheet, 'Accounts Receivable', fye_date),
            'accounts_payable': _safe_get_yf_data(balance_sheet, 'Accounts Payable', fye_date),
            
            # Market Data
            'market_data': hist_data,
            'shares_outstanding': _safe_get_yf_data(balance_sheet, 'Share Issued', fye_date)
        }
        
        return raw_data
        
    except Exception as e:
        logger.error(f"Error fetching Yahoo Finance data for {ticker}: {e}")
        return None

def _safe_get_yf_data(df, item, date, default=np.nan):
    """Safely retrieves data from a yfinance DataFrame - copied from data.py"""
    try:
        if item in df.index:
            val = df.loc[item, date]
            return float(val) if pd.notna(val) else default
        
        # Common alternative names for yfinance
        alt_names = {
            'Total Revenue': ['Total Revenue', 'Revenue', 'Total Revenues', 'Net Sales'],
            'Net Income': ['Net Income', 'Net Income Common Stockholders', 'Net Earnings'],
            'Operating Income': ['Operating Income', 'Operating Earnings'],
            'Gross Profit': ['Gross Profit'],
            'Total Assets': ['Total Assets'],
            'Current Assets': ['Current Assets'],
            'Inventory': ['Inventory'],
            'Cash And Cash Equivalents': ['Cash And Cash Equivalents', 'Cash'],
            'Current Liabilities': ['Current Liabilities'],
            'Total Liabilities Net Minority Interest': ['Total Liabilities Net Minority Interest', 'Total Liab'],
            'Total Equity Gross Minority Interest': ['Total Equity Gross Minority Interest', 'Stockholders Equity'],
            'Accounts Receivable': ['Accounts Receivable'],
            'Accounts Payable': ['Accounts Payable'],
            'Share Issued': ['Share Issued', 'Ordinary Shares Number', 'Common Stock Shares Outstanding']
        }
        
        for alt_name in alt_names.get(item, [item]):
            if alt_name in df.index:
                val = df.loc[alt_name, date]
                return float(val) if pd.notna(val) else default
        return default
    except Exception:
        return default

# === FRED API DATA COLLECTION ===
def fetch_fred_macro_data(fred_api_key, start_date, end_date):
    """
    Fetch macroeconomic data from FRED API.
    This aligns with 'FRED - FRED API' in the architecture.
    """
    if not fred_api_key:
        logger.warning("No FRED API key provided. Macroeconomic features will be NaN.")
        return {}
    
    logger.info(f"Fetching FRED macro data from {start_date} to {end_date}")
    
    try:
        fred = Fred(api_key=fred_api_key)
        
        fred_series = {
            'fed_funds_rate': 'FEDFUNDS',
            'treasury_10y': 'GS10', 
            'treasury_3m': 'GS3M',
            'credit_spread_high_yield': 'BAMLH0A0HYM2',
            'credit_spread_investment': 'BAMLC0A0CM',
            'vix': 'VIXCLS',
            'unemployment_rate': 'UNRATE',
        }
        
        macro_dict = {}
        for name, series_id in fred_series.items():
            try:
                data = fred.get_series(series_id, start=start_date, end=end_date)
                if not data.empty:
                    macro_dict[name] = data.resample('D').last().ffill()
                    logger.debug(f"Fetched {name} ({len(data)} points)")
                else:
                    logger.warning(f"No data found for {name} ({series_id})")
            except Exception as e:
                logger.error(f"Error fetching {name} ({series_id}): {e}")
        
        if macro_dict:
            macro_data = pd.DataFrame(macro_dict)
            # Engineer derived features
            if 'treasury_10y' in macro_data.columns and 'treasury_3m' in macro_data.columns:
                macro_data['yield_curve_slope'] = macro_data['treasury_10y'] - macro_data['treasury_3m']
            logger.info(f"Macroeconomic data fetched: {macro_data.shape}")
            return macro_data
        else:
            logger.error("Failed to fetch any macroeconomic data from FRED.")
            return {}
            
    except Exception as e:
        logger.error(f"Error in fetch_fred_macro_data: {e}")
        return {}

# === NEWS API DATA COLLECTION ===
def fetch_news_articles(company_name, days_back=7, max_articles=100):
    """
    Fetch recent news articles from NewsAPI.
    This aligns with 'NEWS - NewsAPI' in the architecture.
    Copied and modularized from news_unstructured_score.py.
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_back)  # Now uses the passed days_back parameter
    
    # Build API request parameters
    params = {
        'q': f'"{company_name}" OR {company_name.replace(" ", " AND ")}',
        'from': start_date.isoformat(),
        'to': end_date.isoformat(),
        'sortBy': 'relevancy',
        'language': 'en',
        'pageSize': min(max_articles, 100),
        'apiKey': NEWS_API_KEY
    }
    
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching news for {company_name} (attempt {attempt + 1})...")
            response = requests.get(NEWS_API_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'error':
                logger.error(f"NewsAPI Error: {data.get('message', 'Unknown error')}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return []
            
            articles = data.get('articles', [])
            
            # Filter out removed articles and duplicates
            valid_articles = []
            seen_titles = set()
            
            for article in articles:
                title = article.get('title', '')
                if (title and 
                    title not in seen_titles and 
                    title != '[Removed]' and
                    len(title) > 10):
                    
                    valid_articles.append({
                        'title': title,
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'publishedAt': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', 'Unknown')
                    })
                    seen_titles.add(title)
            
            logger.info(f"Retrieved {len(valid_articles)} valid articles for {company_name}")
            return valid_articles[:max_articles]
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                logger.error("All retry attempts failed")
                return []
    
    return []

# === HISTORICAL NEWS DATA COLLECTION (NEW FUNCTION) ===
def fetch_historical_news_data(company_name, days_back=28, max_articles=100):
    """
    Fetch historical news data for the specified number of days.
    This is used for historical analysis to avoid multiple API calls.
    """
    logger.info(f"📡 Fetching historical news for {company_name} (last {days_back} days)...")
    return fetch_news_articles(company_name, days_back=days_back, max_articles=max_articles)

# === COMBINED DATA COLLECTION INTERFACE ===
def collect_all_data(company_name, ticker, rating_date, fred_api_key=None, news_days_back=7):
    """
    Orchestrates all data collection according to the architecture.
    Returns separate raw datasets for Yahoo Finance, FRED, and News.
    """
    results = {
        'company_name': company_name,
        'ticker': ticker,
        'rating_date': rating_date,
        'collection_timestamp': datetime.datetime.now(),
    }
    
    # 1. Yahoo Finance Data Collection
    logger.info("=== YAHOO FINANCE DATA COLLECTION ===")
    yf_data = fetch_yahoo_finance_data(ticker, rating_date)
    results['yahoo_finance_data'] = yf_data
    
    # 2. FRED Macro Data Collection  
    logger.info("=== FRED MACRO DATA COLLECTION ===")
    if fred_api_key:
        # Calculate date range for macro data
        rating_date_dt = pd.to_datetime(rating_date)
        start_date = (rating_date_dt - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = (rating_date_dt + timedelta(days=365)).strftime('%Y-%m-%d')
        
        fred_data = fetch_fred_macro_data(fred_api_key, start_date, end_date)
        results['fred_macro_data'] = fred_data
    else:
        results['fred_macro_data'] = {}
    
    # 3. News Data Collection - Now uses the days_back parameter
    logger.info("=== NEWS DATA COLLECTION ===")
    news_data = fetch_news_articles(company_name, days_back=news_days_back, max_articles=20)
    results['news_articles'] = news_data
    
    logger.info(f"Data collection complete for {company_name}")
    return results

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    # Test data collection for Apple
    fred_api_key = "1a39ebc94e4984ff4091baa2f84c0ba7"  # Your FRED API key
    
    result = collect_all_data(
        company_name="Apple Inc.",
        ticker="AAPL", 
        rating_date=datetime.date(2024, 8, 20),
        fred_api_key=fred_api_key,
        news_days_back=7  # Added parameter
    )
    
    print(f"Collected data for {result['company_name']}")
    print(f"Yahoo Finance data: {'✓' if result['yahoo_finance_data'] else '✗'}")
    print(f"FRED macro data: {'✓' if result['fred_macro_data'] else '✗'}")
    print(f"News articles: {len(result['news_articles'])} articles")