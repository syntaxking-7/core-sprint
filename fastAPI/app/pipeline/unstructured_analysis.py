
"""
unstructured_analysis.py

Unstructured data analysis using news sentiment with FinBERT - self-contained implementation.
"""

import logging
import requests
import datetime
import time
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

logger = logging.getLogger(__name__)

# Load FinBERT model (this will happen once when module is imported)
try:
    logger.info("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    finbert_labels = ['positive', 'negative', 'neutral']
    logger.info("FinBERT model loaded successfully")
    FINBERT_LOADED = True
except Exception as e:
    logger.warning(f"Could not load FinBERT: {e}. Using fallback analysis.")
    FINBERT_LOADED = False

# NewsAPI Configuration
NEWS_API_KEY = "9ab5e737f4d345508eb83b0fb4f0a9cc"
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

def analyze_article_sentiment(title: str, description: str = "", content: str = "") -> Dict[str, float]:
    """Analyze sentiment of a single article using FinBERT"""
    if not FINBERT_LOADED:
        return {
            'positive_score': 0.33,
            'negative_score': 0.33,
            'neutral_score': 0.34,
            'net_sentiment': 0.0,
            'confidence': 0.5
        }
    
    try:
        # Combine all available text
        full_text = f"{title} {description} {content}".strip()
        
        # Truncate to reasonable length
        if len(full_text) > 1000:
            full_text = full_text[:1000]
        
        # Tokenize and get model predictions
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            scores = softmax(outputs.logits.detach().numpy(), axis=1)[0]
        
        # Extract sentiment scores
        positive_score = float(scores[finbert_labels.index('positive')])
        negative_score = float(scores[finbert_labels.index('negative')])
        neutral_score = float(scores[finbert_labels.index('neutral')])
        
        # Calculate net sentiment (-1 to 1)
        net_sentiment = positive_score - negative_score
        
        return {
            'positive_score': positive_score,
            'negative_score': negative_score,
            'neutral_score': neutral_score,
            'net_sentiment': net_sentiment,
            'confidence': max(positive_score, negative_score, neutral_score)
        }
        
    except Exception as e:
        logger.warning(f"Sentiment analysis error: {e}")
        return {
            'positive_score': 0.33,
            'negative_score': 0.33,
            'neutral_score': 0.34,
            'net_sentiment': 0.0,
            'confidence': 0.5
        }

def analyze_risk_factors(text: str) -> Dict[str, Any]:
    """Analyze financial risk factors in text"""
    risk_keywords = {
        'liquidity_crisis': ['cash flow', 'liquidity crisis', 'cash shortage', 'working capital', 
                           'credit facility', 'refinancing', 'cash burn', 'funding gap'],
        'debt_distress': ['debt default', 'covenant violation', 'bankruptcy', 'insolvency',
                         'debt restructuring', 'creditor pressure', 'leverage concerns', 'debt burden'],
        'operational_risks': ['supply chain', 'production halt', 'operational disruption', 
                            'regulatory issues', 'compliance violation', 'lawsuit', 'investigation'],
        'market_risks': ['market volatility', 'demand decline', 'competition', 'market share loss',
                       'pricing pressure', 'economic downturn', 'recession concerns']
    }
    
    text_lower = text.lower()
    found_keywords = []
    category_counts = {}
    
    for category, keywords in risk_keywords.items():
        count = 0
        for keyword in keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
                count += 1
        category_counts[category] = count
    
    # Calculate overall risk boost
    total_keywords = len(found_keywords)
    risk_boost = min(total_keywords * 5, 30)  # Max 30 point boost
    
    return {
        'keywords_found': found_keywords,
        'category_counts': category_counts,
        'risk_boost': risk_boost,
        'total_risk_signals': total_keywords
    }

def fetch_company_news(company: str, days_back: int = 7, max_articles: int = 20, start_date: datetime.date = None, end_date: datetime.date = None) -> List[Dict[str, Any]]:
    """Fetch news articles for a company using NewsAPI, optionally for a custom date range."""
    if end_date is None:
        end_date = datetime.date.today()
    if start_date is None:
        start_date = end_date - datetime.timedelta(days=days_back)

    params = {
        'q': f'"{company}" OR {company.replace(" ", " AND ")}',
        'from': start_date.isoformat(),
        'to': end_date.isoformat(),
        'sortBy': 'relevancy',
        'language': 'en',
        'pageSize': min(max_articles, 100),
        'apiKey': NEWS_API_KEY
    }
    
    try:
        logger.info(f"📡 Fetching news for {company}...")
        response = requests.get(NEWS_API_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'error':
            logger.error(f"❌ NewsAPI Error: {data.get('message', 'Unknown error')}")
            return []
        
        articles = data.get('articles', [])
        
        # Filter valid articles
        valid_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '')
            if (title and 
                title not in seen_titles and 
                '[Removed]' not in title and
                title != '[Removed]'):
                seen_titles.add(title)
                valid_articles.append(article)
        
        logger.info(f"Retrieved {len(valid_articles)} valid articles for {company}")
        return valid_articles[:max_articles]
        
    except Exception as e:
        logger.error(f"❌ Error fetching news: {e}")
        return []

def compute_unstructured_score(company_name: str, days_back: int = 7, max_articles: int = 20, start_date: datetime.date = None, end_date: datetime.date = None, articles: List[Dict[str, Any]] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Compute unstructured score using news sentiment analysis with FinBERT.
    Optionally for a custom date range.
    Returns: (unstructured_score, assessment_dict)
    """
    try:
        logger.info(f"📰 Computing unstructured score for {company_name}...")
        # Step 1: Use provided articles or fetch if not provided
        if articles is None:
            articles = fetch_company_news(company_name, days_back, max_articles, start_date, end_date)
        if not articles:
            logger.warning(f"⚠️ No articles found for {company_name}")
            return 50.0, {
                'unstructured_score': 50.0,
                'confidence': 0.3,
                'articles_analyzed': 0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'method': 'No news data available'
            }
        # Step 2: Analyze sentiment for each article
        sentiment_results = []
        all_risk_keywords = []
        for i, article in enumerate(articles):
            # Sentiment analysis
            sentiment = analyze_article_sentiment(
                article['title'], 
                article.get('description', ''), 
                article.get('content', '')
            )
            sentiment_results.append(sentiment)
            # Risk factor analysis
            full_text = f"{article['title']} {article.get('description', '')} {article.get('content', '')}"
            risk_analysis = analyze_risk_factors(full_text)
            all_risk_keywords.extend(risk_analysis['keywords_found'])
        # Step 3: Calculate aggregate sentiment score
        net_sentiments = [result['net_sentiment'] for result in sentiment_results]
        avg_net_sentiment = np.mean(net_sentiments) if net_sentiments else 0.0
        # Convert to 0-100 scale (0 = very positive, 100 = very negative)
        base_sentiment_score = 50 * (1 - avg_net_sentiment)  # Flip so higher = more risky
        # Step 4: Calculate risk factor boost
        risk_keyword_count = len(all_risk_keywords)
        risk_boost = min(risk_keyword_count * 3, 25)  # Max 25 point boost
        # Step 5: Final score calculation
        unstructured_score = (base_sentiment_score)
        unstructured_score = max(0, min(100, unstructured_score))
        # Step 6: Calculate confidence
        if sentiment_results:
            avg_confidence = np.mean([r['confidence'] for r in sentiment_results])
            article_confidence = min(len(articles) / 10, 1.0)  # More articles = more confidence
            confidence = (avg_confidence * 0.7 + article_confidence * 0.3)
        else:
            confidence = 0.3
        # Step 7: Sentiment distribution
        sentiment_dist = {
            'positive': sum(1 for r in sentiment_results if r['net_sentiment'] > 0.1),
            'neutral': sum(1 for r in sentiment_results if -0.1 <= r['net_sentiment'] <= 0.1),
            'negative': sum(1 for r in sentiment_results if r['net_sentiment'] < -0.1)
        }
        logger.info(f"✅ Unstructured Score: {unstructured_score:.1f}, Confidence: {confidence:.2%}")
        assessment = {
            'unstructured_score': unstructured_score,
            'confidence': confidence,
            'articles_analyzed': len(articles),
            'sentiment_distribution': sentiment_dist,
            'risk_keywords': list(Counter(all_risk_keywords).most_common(5)),
            'base_sentiment_score': base_sentiment_score,
            'risk_boost': risk_boost,
            'sample_headlines': [article['title'] for article in articles[:3]],
            'method': 'FinBERT + Risk Keyword Analysis'
        }
        return unstructured_score, assessment
    except Exception as e:
        logger.error(f"❌ Error in unstructured analysis: {e}")
        assessment = {
            'unstructured_score': 50.0,
            'confidence': 0.1,
            'error': str(e),
            'articles_analyzed': 0
        }
        return 50.0, assessment
