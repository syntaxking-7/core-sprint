
"""
data_processing.py

Data storage, cleaning and feature engineering for structured and unstructured data.
Aligns with 'DP - Data Processor' and 'Clean & Feature Engineering' in architecture.
Copied and modularized from data.py and news_unstructured_score.py.
"""

import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

# === STRUCTURED DATA PROCESSING & FEATURE ENGINEERING ===
def process_yahoo_finance_data(raw_yf_data, rating_date):
    """
    Process raw Yahoo Finance data into engineered features.
    This implements the feature engineering shown in the architecture diagram.
    Copied and adapted from data.py calculation logic.
    """
    if not raw_yf_data:
        logger.warning("No Yahoo Finance data to process")
        return {}
    
    logger.info("Processing Yahoo Finance data into engineered features...")
    
    # Calculate market data features
    market_features = _calculate_market_features(raw_yf_data, rating_date)
    
    # Calculate financial ratios
    ratio_features = _calculate_financial_ratios(raw_yf_data)
    
    # Calculate enhanced Z-Score (copied from data.py)
    z_score = _calculate_enhanced_z_score(raw_yf_data, ratio_features, market_features)
    
    # Calculate KMV Distance-to-Default (copied from data.py)  
    kmv_distance = _calculate_rad_kmv(raw_yf_data, rating_date, market_features)
    
    # Combine all features
    processed_features = {
        **ratio_features,
        **market_features,
        'enhanced_z_score': z_score,
        'kmv_distance_to_default': kmv_distance,
        
        # Raw data for further processing
        'total_revenue': raw_yf_data.get('total_revenue', 0),
        'gross_profit': raw_yf_data.get('gross_profit', 0),
        'operating_income': raw_yf_data.get('operating_income', 0),
        'net_income': raw_yf_data.get('net_income', 0),
        'total_assets': raw_yf_data.get('total_assets', 0),
        'current_assets': raw_yf_data.get('current_assets', 0),
        'current_liabilities': raw_yf_data.get('current_liabilities', 0),
        'total_liabilities': raw_yf_data.get('total_liabilities', 0),
        'total_equity': raw_yf_data.get('total_equity', 0),
        'inventory': raw_yf_data.get('inventory', 0),
        'cash_and_equivalents': raw_yf_data.get('cash_and_equivalents', 0),
        'accounts_receivable': raw_yf_data.get('accounts_receivable', 0),
        'accounts_payable': raw_yf_data.get('accounts_payable', 0),
    }
    
    logger.info(f"Generated {len(processed_features)} structured features")
    return processed_features

def _calculate_financial_ratios(raw_data):
    """Calculate financial ratios - copied from data.py"""
    eps = 1e-8
    
    current_assets = raw_data.get('current_assets', 0)
    current_liabilities = raw_data.get('current_liabilities', eps)
    total_assets = raw_data.get('total_assets', eps) 
    total_liabilities = raw_data.get('total_liabilities', 0)
    total_equity = raw_data.get('total_equity', eps)
    total_revenue = raw_data.get('total_revenue', eps)
    gross_profit = raw_data.get('gross_profit', 0)
    operating_income = raw_data.get('operating_income', 0)
    net_income = raw_data.get('net_income', 0)
    
    return {
        'current_ratio': current_assets / max(current_liabilities, eps) if current_assets and current_liabilities else 0,
        'debt_to_equity': total_liabilities / max(total_equity, eps) if total_liabilities and total_equity else 0,
        'debt_ratio': total_liabilities / max(total_assets, eps) if total_liabilities and total_assets else 0,
        'return_on_equity': net_income / max(total_equity, eps) if net_income and total_equity else 0,
        'return_on_assets': net_income / max(total_assets, eps) if net_income and total_assets else 0,
        'gross_margin': gross_profit / max(total_revenue, eps) if gross_profit and total_revenue else 0,
        'operating_margin': operating_income / max(total_revenue, eps) if operating_income and total_revenue else 0,
        'net_margin': net_income / max(total_revenue, eps) if net_income and total_revenue else 0,
        'asset_turnover': total_revenue / max(total_assets, eps) if total_revenue and total_assets else 0,
    }

def _calculate_market_features(raw_data, rating_date):
    """Calculate market-based features including volatility"""
    market_data = raw_data.get('market_data')
    shares_outstanding = raw_data.get('shares_outstanding', 0)
    
    features = {'market_cap': 0, 'volatility': 0.30}  # Default volatility
    
    if market_data is not None and not market_data.empty and shares_outstanding > 0:
        try:
            # Calculate market cap
            stock_price = market_data['Close'].iloc[-1]
            features['market_cap'] = stock_price * shares_outstanding
            
            # Calculate volatility
            if len(market_data) > 5:
                returns = market_data['Close'].pct_change().dropna()
                if len(returns) > 1:
                    features['volatility'] = returns.std() * np.sqrt(252)
        except Exception as e:
            logger.debug(f"Error calculating market features: {e}")
    
    return features

def _calculate_enhanced_z_score(raw_data, ratios, market_features):
    """
    Calculate Enhanced Z-Score - copied and adapted from data.py
    This implements the Z-Score calculation shown in the architecture.
    """
    try:
        eps = 1e-8
        total_assets = max(raw_data.get('total_assets', 1), eps)
        
        # X1: Dynamic Liquidity Stress Index
        current_assets = raw_data.get('current_assets', 0)
        current_liabilities = max(raw_data.get('current_liabilities', eps), eps)
        working_capital = current_assets - current_liabilities
        wc_ta = working_capital / total_assets
        wc_ta = max(-1, min(1, wc_ta))
        
        inventory = raw_data.get('inventory', 0)
        quick_ratio = (current_assets - inventory) / max(current_liabilities, eps)
        quick_ratio_scaled = max(0, min(1, quick_ratio / 2.0))
        
        cash_and_equivalents = raw_data.get('cash_and_equivalents', 0)
        cash_ratio = cash_and_equivalents / max(current_liabilities, eps)
        cash_ratio = max(0, min(1, cash_ratio))
        
        x1 = max(0, min(1, 0.4 * (wc_ta + 1)/2 + 0.3 * quick_ratio_scaled + 0.3 * cash_ratio))
        
        # X2: Multi-Period Earning Quality Score
        total_equity = max(raw_data.get('total_equity', eps), eps)
        retained_earnings_ta = min(2, total_equity / total_assets) / 2.0
        
        operating_margin = ratios.get('operating_margin', 0)
        net_margin = max(ratios.get('net_margin', eps), eps)
        earnings_quality = max(0, min(2, operating_margin / net_margin if net_margin > 0 else 1.0)) / 2.0
        
        roe = ratios.get('return_on_equity', 0)
        roe_stability = max(0, min(1, roe / 0.15 if roe > 0 else 0))
        
        x2 = max(0, min(1, 0.4 * retained_earnings_ta + 0.3 * earnings_quality + 0.3 * roe_stability))
        
        # X3: Risk-Adjusted Operational Performance
        operating_income = raw_data.get('operating_income', 0)
        ebit_ta = max(-0.5, min(1, operating_income / total_assets))
        ebit_ta_scaled = (ebit_ta + 0.5) / 1.5
        
        asset_volatility = market_features.get('volatility', 0.3)
        risk_adjustment = max(0, min(1, 1 - (asset_volatility / 0.4)))
        
        x3 = max(0, min(1, ebit_ta_scaled * risk_adjustment))
        
        # X4: Multi-Dimensional Solvency Score
        market_cap = market_features.get('market_cap', 0)
        total_liabilities = raw_data.get('total_liabilities', 0)
        market_cap_debt = min(3.0, market_cap / max(total_liabilities, eps) if market_cap > 0 else 0) / 3.0
        
        debt_ratio = ratios.get('debt_ratio', 0)
        solvency_score = max(0, min(1, 0.5 * market_cap_debt + 0.5 * (1 - min(1, debt_ratio))))
        
        x4 = solvency_score
        
        # X5: Dynamic Asset Efficiency Index  
        asset_turnover = ratios.get('asset_turnover', 0)
        asset_turnover_scaled = max(0, min(1, asset_turnover / 2.0))
        
        x5 = asset_turnover_scaled
        
        # Enhanced Z-Score calculation
        enhanced_z_score = 1.5*x1 + 1.6*x2 + 3.5*x3 + 0.8*x4 + 1.2*x5
        
        # Normalize to 0-10 range
        normalized_enhanced_z = max(0, min(10, (enhanced_z_score + 5) / 1.5))
        
        return float(normalized_enhanced_z)
        
    except Exception as e:
        logger.debug(f"Error calculating Z-Score: {e}")
        return 2.5  # Default middle value

def _calculate_rad_kmv(raw_data, rating_date, market_features):
    """
    Calculate RAD-KMV Distance-to-Default - copied and adapted from data.py
    This implements the KMV calculation shown in the architecture.
    """
    try:
        eps = 1e-8
        equity_value = market_features.get('market_cap', 0)
        debt_value = raw_data.get('total_liabilities', 0)
        base_volatility = market_features.get('volatility', 0.3)
        
        if not (equity_value > 0 and debt_value > 0):
            return 2.0  # Default middle value
        
        asset_value = equity_value + debt_value
        
        # Default point calculation
        short_term_debt = raw_data.get('current_liabilities', debt_value * 0.3)
        long_term_debt = debt_value - short_term_debt
        default_point = short_term_debt + (long_term_debt * 0.5)
        
        # Risk-free rate (simplified)
        risk_free_rate = 0.03
        time_horizon = 1.0
        
        # Volatility adjustments
        debt_ratio = debt_value / asset_value
        if debt_ratio > 0.6:
            volatility_multiplier = 1.3
        elif debt_ratio < 0.3:
            volatility_multiplier = 0.8
        else:
            volatility_multiplier = 1.0
        
        adjusted_volatility = base_volatility * volatility_multiplier
        
        # KMV calculation
        mu = risk_free_rate - 0.5 * adjusted_volatility**2
        distance_numerator = np.log(asset_value / default_point) + mu * time_horizon
        distance_denominator = adjusted_volatility * np.sqrt(time_horizon)
        distance_to_default = distance_numerator / distance_denominator if distance_denominator != 0 else 2.0
        
        # Bound the result
        rad_kmv_distance = max(-5.0, min(10.0, distance_to_default))
        return float(rad_kmv_distance)
        
    except Exception as e:
        logger.debug(f"Error calculating KMV: {e}")
        return 2.0  # Default middle value

# === MACRO DATA PROCESSING ===
def process_fred_macro_data(raw_fred_data, rating_date):
    """
    Process FRED macro data for the specific rating date.
    Extracts relevant macro features as of the rating date.
    """
    if raw_fred_data is None or (isinstance(raw_fred_data, pd.DataFrame) and raw_fred_data.empty) or (isinstance(raw_fred_data, dict) and not raw_fred_data):
        logger.warning("No FRED data to process")
        return {}
    
    logger.info("Processing FRED macro data...")
    
    try:
        rating_date_dt = pd.to_datetime(rating_date)
        
        # Find the closest available data on or before the rating date
        available_dates = raw_fred_data.index[raw_fred_data.index <= rating_date_dt]
        if available_dates.empty:
            logger.warning("No FRED data available for the rating date")
            return {}
        
        closest_date = available_dates.max()
        macro_row = raw_fred_data.loc[closest_date]
        
        # Extract macro features
        macro_features = {}
        for col in raw_fred_data.columns:
            value = macro_row[col]
            if pd.notna(value):
                macro_features[col] = float(value)
            else:
                # Default values for missing data
                defaults = {
                    'fed_funds_rate': 2.0,
                    'treasury_10y': 3.0,
                    'treasury_3m': 2.0,
                    'credit_spread_high_yield': 4.0,
                    'credit_spread_investment': 1.5,
                    'vix': 20.0,
                    'unemployment_rate': 4.0,
                    'yield_curve_slope': 1.0
                }
                macro_features[col] = defaults.get(col, 0.0)
        
        logger.info(f"Processed {len(macro_features)} macro features")
        return macro_features
        
    except Exception as e:
        logger.error(f"Error processing FRED data: {e}")
        return {}

# === NEWS DATA PROCESSING ===
def process_news_articles(raw_news_articles):
    """
    Clean and preprocess news articles for sentiment analysis.
    This implements the text cleaning shown in the architecture.
    """
    if not raw_news_articles:
        logger.warning("No news articles to process")
        return []
    
    logger.info(f"Processing {len(raw_news_articles)} news articles...")
    
    processed_articles = []
    
    for article in raw_news_articles:
        try:
            # Clean and validate article data
            title = article.get('title', '').strip()
            description = article.get('description', '').strip()
            content = article.get('content', '').strip()
            
            # Skip articles with insufficient content
            if len(title) < 10:
                continue
            
            # Combine text for analysis
            full_text = f"{title} {description} {content}".strip()
            
            # Basic text cleaning
            cleaned_text = _clean_text(full_text)
            
            processed_article = {
                'title': title,
                'description': description,
                'content': content,
                'full_text': cleaned_text,
                'publishedAt': article.get('publishedAt', ''),
                'source': article.get('source', 'Unknown'),
                'url': article.get('url', '')
            }
            
            processed_articles.append(processed_article)
            
        except Exception as e:
            logger.debug(f"Error processing article: {e}")
            continue
    
    logger.info(f"Successfully processed {len(processed_articles)} articles")
    return processed_articles

def _clean_text(text):
    """Basic text cleaning for financial news"""
    if not text:
        return ""
    
    # Remove extra whitespace
    cleaned = ' '.join(text.split())
    
    # Truncate to reasonable length for FinBERT
    if len(cleaned) > 1000:
        cleaned = cleaned[:1000]
    
    return cleaned

# === COMBINED DATA PROCESSING INTERFACE ===
def process_collected_data(collected_data):
    """
    Main processing function that takes raw collected data and returns processed features.
    This orchestrates the entire data processing pipeline per the architecture.
    """
    logger.info("=== STARTING DATA PROCESSING PIPELINE ===")
    
    company_name = collected_data['company_name']
    rating_date = collected_data['rating_date']
    
    # Process each data source
    processed_data = {
        'company_name': company_name,
        'rating_date': rating_date,
        'processing_timestamp': datetime.datetime.now(),
    }
    
    # 1. Process Yahoo Finance Data (Financial + Market features)
    logger.info("Processing Yahoo Finance data...")
    yf_features = process_yahoo_finance_data(
        collected_data.get('yahoo_finance_data'), 
        rating_date
    )
    processed_data['structured_features'] = yf_features
    
    # 2. Process FRED Macro Data  
    logger.info("Processing FRED macro data...")
    macro_features = process_fred_macro_data(
        collected_data.get('fred_macro_data'), 
        rating_date
    )
    processed_data['macro_features'] = macro_features
    
    # 3. Process News Articles
    logger.info("Processing news articles...")
    cleaned_articles = process_news_articles(
        collected_data.get('news_articles', [])
    )
    processed_data['processed_news'] = cleaned_articles
    
    # 4. Combine structured features with macro features
    combined_features = {**yf_features, **macro_features}
    processed_data['combined_structured_features'] = combined_features
    
    logger.info(f"Data processing complete for {company_name}")
    logger.info(f"Generated {len(combined_features)} combined structured features")
    logger.info(f"Processed {len(cleaned_articles)} news articles")
    
    return processed_data

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    # Example: test with dummy collected data
    dummy_collected_data = {
        'company_name': 'Apple Inc.',
        'rating_date': datetime.date(2024, 8, 20),
        'yahoo_finance_data': {
            'total_revenue': 100000000,
            'total_assets': 500000000,
            'current_assets': 200000000,
            'current_liabilities': 100000000,
            'total_liabilities': 300000000,
            'total_equity': 200000000,
            'net_income': 20000000,
            'market_data': None,
            'shares_outstanding': 1000000
        },
        'fred_macro_data': {},
        'news_articles': [
            {'title': 'Apple reports strong quarterly results', 'description': 'Revenue up 10%', 'content': '...'}
        ]
    }
    
    processed = process_collected_data(dummy_collected_data)
    print(f"Processed data for {processed['company_name']}")
    print(f"Structured features: {len(processed['combined_structured_features'])}")
    print(f"Processed news: {len(processed['processed_news'])}")
