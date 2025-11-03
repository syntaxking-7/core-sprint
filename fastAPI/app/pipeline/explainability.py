"""
explainability.py
Comprehensive explainability module combining structured and unstructured explanations.
Self-contained implementation without external dependencies.
Includes integrated comprehensive explainability report generator.
"""
import logging
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, List, Tuple
from collections import Counter
import datetime
# Note: interpret is not imported here anymore as the main functions don't use it directly.
# The EBMExplainer class handles its own import and usage.
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()
# === INTEGRATED COMPREHENSIVE EXPLAINABILITY REPORT ===
MODEL_PATH = os.getenv("MODEL_PATH")
import datetime
import json
from typing import Dict, Any, List

def generate_comprehensive_explainability_report(
    company_name: str,
    final_score: float,
    credit_grade: str,
    structured_result: Dict[str, Any],
    unstructured_result: Dict[str, Any],
    fusion_result: Dict[str, Any],
    market_conditions: Dict[str, Any],
    structured_features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a machine-readable explainability report as JSON instead of plain text.
    """
    try:
        alpha = _calculate_volatility_factor(market_conditions)
        threshold = 0.7  # Standard threshold for market stability
        ebm_output = _extract_ebm_contributions_via_explainer(structured_result, structured_features, company_name)
        headlines, sentiments, sentiment_scores = _extract_news_data(unstructured_result)

        # Component Scores
        structured_score = fusion_result.get('expert_contributions', {}).get('structured_expert', {}).get('score', structured_result.get('structured_score', 50.0))
        unstructured_score = fusion_result.get('expert_contributions', {}).get('news_sentiment_expert', {}).get('score', unstructured_result.get('unstructured_score', 50.0))

        # Dynamic Weights
        dynamic_weights = fusion_result.get('dynamic_weights', {})
        struct_weight = dynamic_weights.get('structured_expert', 0.5)
        news_weight = dynamic_weights.get('news_sentiment_expert', 0.5)

        # Market Conditions
        market_regime = market_conditions.get('regime', 'NORMAL')
        vix = market_conditions.get('vix', 20.0)
        market_status = "STABLE" if alpha > threshold else "VOLATILE"

        # Risk Category
        if final_score >= 80:
            risk_level = "LOW RISK"
            risk_desc = "Strong creditworthiness with minimal default probability"
        elif final_score >= 60:
            risk_level = "MODERATE-LOW RISK"
            risk_desc = "Solid credit profile with acceptable risk levels"
        elif final_score >= 40:
            risk_level = "MODERATE-HIGH RISK"
            risk_desc = "Elevated risk requiring careful monitoring"
        elif final_score >= 20:
            risk_level = "HIGH RISK"
            risk_desc = "Significant credit risk with high default probability"
        else:
            risk_level = "VERY HIGH RISK"
            risk_desc = "Very high credit risk requiring immediate attention"

        # Key Financial Drivers
        top_financial_factors: List[Dict[str, Any]] = []
        if ebm_output:
            total_abs_contribution = sum(abs(feat['raw_contribution']) for feat in ebm_output) or 1e-8
            reverse_sort = bool(final_score >= 50)
            top_contributors = sorted(ebm_output, key=lambda x: float(x['raw_contribution']), reverse=reverse_sort)[:5]

            for feat in top_contributors:
                contrib_percentage = abs(feat['raw_contribution']) / total_abs_contribution
                points_impact_on_structured = structured_score * contrib_percentage
                points_impact_on_final = struct_weight * points_impact_on_structured
                top_financial_factors.append({
                    "feature": feat['feature'],
                    "value": feat['formatted_value'],
                    "interpretation": feat['interpretation'],
                    "raw_contribution": feat['raw_contribution'],
                    "contribution_percentage": round(contrib_percentage * 100, 2),
                    "impact_on_final_score": round(points_impact_on_final, 2)
                })

        # News Impact
        news_impact: List[Dict[str, Any]] = []
        if headlines and sentiments and sentiment_scores:
            top_headlines = sorted(zip(headlines, sentiments, sentiment_scores), key=lambda x: abs(x[2]), reverse=True)[:3]
            for headline, sentiment, score in top_headlines:
                abs_score = abs(score)
                if abs_score > 0.6:
                    impact_level = "High"
                elif abs_score > 0.3:
                    impact_level = "Moderate"
                else:
                    impact_level = "Low"
                news_impact.append({
                    "headline": headline,
                    "sentiment": sentiment,
                    "sentiment_score": score,
                    "impact_level": impact_level
                })

        # Positive & Negative Factors
        positive_factors = _identify_positive_factors(ebm_output, sentiments)
        negative_factors = _identify_negative_factors(ebm_output, sentiments)

        # Final JSON Report
        report = {
            "company": company_name,
            "final_score": round(final_score, 2),
            "credit_grade": credit_grade,
            "assessment_date": datetime.datetime.now().strftime('%Y-%m-%d'),
            "data_sources": ["Structured Analysis (EBM)", "News Sentiment (FinBERT)", "Market Indicators", "Fusion Algorithm"],
            "component_analysis": {
                "structured_score": round(structured_score, 2),
                "unstructured_score": round(unstructured_score, 2),
                "weights": {
                    "structured_expert": struct_weight,
                    "news_sentiment_expert": news_weight
                }
            },
            "market_conditions": {
                "status": market_status,
                "alpha": alpha,
                "vix": vix,
                "regime": market_regime
            },
            "top_financial_drivers": top_financial_factors,
            "news_impact": news_impact,
            "risk_summary": {
                "risk_level": risk_level,
                "description": risk_desc
            },
            "key_insights": {
                "market_influence": "stable" if alpha > threshold else "volatile",
                "strengths": positive_factors[:3],
                "concerns": negative_factors[:3]
            },
            "technical_details": {
                "ml_model": "Explainable Boosting Machine",
                "sentiment_model": "FinBERT",
                "fusion_method": "Dynamic Weighting",
                "articles_analyzed": unstructured_result.get('articles_analyzed', 0),
                "financial_metrics_used": len(structured_features)
            },
            "generated_timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return report

    except Exception as e:
        logger.error(f"Error generating explainability report: {e}")
        return {"error": f"Error generating explainability report for {company_name}: {str(e)}"}

def _calculate_volatility_factor(market_conditions: Dict[str, Any]) -> float:
    """Calculate volatility factor (alpha) from market conditions"""
    vix = market_conditions.get('vix', 20.0)
    regime = market_conditions.get('regime', 'NORMAL')
    # Higher alpha = more stable market (lower volatility)
    # Lower alpha = more volatile market
    if regime == 'CRISIS':
        base_alpha = 0.3
    elif regime == 'STRESS':
        base_alpha = 0.5
    else:  # NORMAL
        base_alpha = 0.7

    # Adjust based on VIX (lower VIX = more stable = higher alpha)
    # Ensure vix is not zero to avoid division issues
    vix = max(vix, 0.1)
    vix_adjustment = max(0.1, min(1.0, (30 - vix) / 20)) # This makes higher VIX lower adjustment
    alpha = base_alpha * vix_adjustment
    return max(0.1, min(1.0, alpha))

# --- NEW FUNCTION: Use EBMExplainer to get contributions ---
def _extract_ebm_contributions_via_explainer(structured_result: Dict[str, Any], structured_features: Dict[str, Any], company_name: str) -> List[Dict[str, Any]]:
    """Use the EBMExplainer to get the correct feature contributions."""
    try:
        logger.debug("Calling EBMExplainer to get feature contributions for report...")
        # Call the explainer function which handles the model loading and explain_local correctly
        explanation_result = explain_structured_score(structured_features, company_name)
        feature_contributions = explanation_result.get('feature_contributions', {})
        logger.debug(f"EBMExplainer returned {len(feature_contributions)} contributions.")

        # Now format these contributions like the old _extract_ebm_contributions did
        ratio_priorities = {
            'debt_to_equity': 10, 'current_ratio': 10, 'return_on_equity': 10, 'return_on_assets': 10,
            'enhanced_z_score': 9, 'kmv_distance_to_default': 9, 'net_margin': 8, 'debt_ratio': 8,
            'volatility': 7, 'quick_ratio': 7, 'asset_turnover': 6, 'interest_coverage': 6,
            'working_capital': 5, 'total_revenue': 3, 'total_assets': 2, 'market_cap': 1
        }

        ebm_output = []
        for feature, raw_contribution in feature_contributions.items():
            # Use the original Yahoo Finance value (not scaled)
            original_value = structured_features.get(feature, 0.0)
            # Format value for display
            formatted_value = _format_feature_value(feature, original_value)
            # Get detailed interpretation using the ebm_exp.py logic
            interpretation = _get_detailed_feature_interpretation(feature, original_value, raw_contribution)

            # Calculate contribution percentage relative to total absolute contributions (if needed elsewhere)
            total_abs_contribution = sum(abs(contrib) for contrib in feature_contributions.values())
            if total_abs_contribution > 0:
                contribution_percentage = abs(raw_contribution) / total_abs_contribution
            else:
                contribution_percentage = 0.0

            # Get priority for sorting (financial ratios first)
            priority = ratio_priorities.get(feature, 0)

            ebm_output.append({
                'feature': feature.replace('_', ' ').title(),
                'value': float(original_value),
                'formatted_value': formatted_value,
                'raw_contribution': float(raw_contribution), # This is the key model output
                'contribution_percentage': contribution_percentage,
                'interpretation': interpretation,
                'priority': priority
            })

        # Sort by priority first, then by absolute contribution for tie-breaking within categories
        ebm_output.sort(key=lambda x: (-x['priority'], -abs(x['raw_contribution'])))
        logger.debug(f"Formatted and sorted EBM output has {len(ebm_output)} items.")
        return ebm_output

    except Exception as e:
        logger.warning(f"Could not extract EBM contributions using EBMExplainer: {e}")
        return []

def _format_feature_value(feature_name: str, value: float) -> str:
    """Format feature values for better readability"""
    # Large dollar amounts in billions/millions
    if feature_name in ['market_cap', 'total_assets', 'total_revenue', 'total_equity', 'current_assets', 'total_liabilities']:
        if abs(value) >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.2f}K"
        else:
            return f"${value:.2f}"
    # Percentages
    elif feature_name in ['return_on_equity', 'return_on_assets', 'net_margin', 'volatility', 'gross_margin', 'operating_margin']:
        return f"{value*100:.2f}%"
    # Ratios (keep as decimal with more precision)
    elif feature_name in ['debt_to_equity', 'current_ratio', 'debt_ratio', 'quick_ratio', 'asset_turnover']:
        return f"{value:.3f}"
    # Scores and distances
    elif feature_name in ['enhanced_z_score', 'kmv_distance_to_default']:
        return f"{value:.2f}"
    # Default formatting
    else:
        return f"{value:.4f}"

def _get_top_financial_features(structured_features: Dict[str, Any]) -> List[tuple]:
    """Get top financial features when EBM contributions are not available"""
    features = []
    # Key financial ratios with thresholds and descriptions
    key_metrics = {
        'debt_to_equity': {
            'threshold_good': 0.5,
            'threshold_concern': 2.0,
            'good_desc': 'strong capital structure',
            'concern_desc': 'high leverage risk'
        },
        'current_ratio': {
            'threshold_good': 1.5,
            'threshold_concern': 1.0,
            'good_desc': 'adequate liquidity',
            'concern_desc': 'liquidity concerns'
        },
        'return_on_equity': {
            'threshold_good': 0.10,
            'threshold_concern': 0.05,
            'good_desc': 'strong profitability',
            'concern_desc': 'weak profitability'
        },
        'return_on_assets': {
            'threshold_good': 0.05,
            'threshold_concern': 0.02,
            'good_desc': 'efficient asset utilization',
            'concern_desc': 'poor asset efficiency'
        },
        'enhanced_z_score': {
            'threshold_good': 3.0,
            'threshold_concern': 1.8,
            'good_desc': 'low bankruptcy risk',
            'concern_desc': 'elevated bankruptcy risk'
        }
    }

    for feature, config in key_metrics.items():
        value = structured_features.get(feature, 0.0)
        if value != 0.0:  # Only include if we have data
            if value >= config['threshold_good']:
                desc = config['good_desc']
            elif value <= config['threshold_concern']:
                desc = config['concern_desc']
            else:
                desc = 'moderate performance levels'
            features.append((feature, value, desc))

    # Sort by relevance (put concerning values first, then good values)
    def sort_key(item):
        feature, value, desc = item
        config = key_metrics[feature]
        if 'concern' in desc:
            return 0  # High priority
        elif 'strong' in desc or 'good' in desc:
            return 1  # Medium priority
        else:
            return 2  # Low priority

    features.sort(key=sort_key)
    return features

def _get_detailed_feature_interpretation(feature_name: str, value: float, contribution: float) -> str:
    """Get detailed business interpretation for a feature using ebm_exp.py logic"""
    # Define comprehensive interpretation rules from ebm_exp.py
    interpretations = {
        'debt_to_equity': {
            'thresholds': [0.5, 1.0, 2.0, 3.0],
            'descriptions': [
                "excellent capital structure with very low leverage",
                "good capital structure with moderate leverage",
                "concerning leverage levels increasing risk",
                "high leverage indicating significant financial risk",
                "extremely high leverage suggesting potential distress"
            ]
        },
        'debt_ratio': {
            'thresholds': [0.3, 0.5, 0.7, 0.8],
            'descriptions': [
                "very low debt burden indicating strong financial position",
                "moderate debt levels within healthy range",
                "elevated debt levels requiring monitoring",
                "high debt burden increasing default risk",
                "excessive debt burden indicating financial distress"
            ]
        },
        'current_ratio': {
            'thresholds': [1.0, 1.5, 2.0, 3.0],
            'descriptions': [
                "insufficient liquidity to meet short-term obligations",
                "adequate liquidity but below optimal levels",
                "good liquidity providing reasonable safety buffer",
                "strong liquidity position with excellent coverage",
                "very strong liquidity position"
            ]
        },
        'return_on_equity': {
            'thresholds': [0.0, 0.05, 0.15, 0.25],
            'descriptions': [
                "negative returns indicating poor management performance",
                "weak profitability suggesting operational challenges",
                "acceptable profitability within industry norms",
                "strong profitability indicating efficient management",
                "exceptional profitability demonstrating superior performance"
            ]
        },
        'enhanced_z_score': {
            'thresholds': [1.8, 3.0, 4.5, 6.0],
            'descriptions': [
                "high bankruptcy risk requiring immediate attention",
                "moderate bankruptcy risk needing close monitoring",
                "low bankruptcy risk indicating stable operations",
                "very low bankruptcy risk with strong fundamentals",
                "minimal bankruptcy risk with excellent financial health"
            ]
        },
        'net_margin': {
            'thresholds': [0.0, 0.05, 0.10, 0.20],
            'descriptions': [
                "negative profitability indicating operational losses",
                "weak profit margins suggesting pricing or cost issues",
                "adequate profit margins within industry standards",
                "strong profit margins indicating competitive advantage",
                "exceptional profit margins demonstrating pricing power"
            ]
        },
        'volatility': {
            'thresholds': [0.15, 0.30, 0.50, 0.75],
            'descriptions': [
                "very low market risk with stable stock performance",
                "moderate market risk typical for established companies",
                "elevated market risk indicating investor uncertainty",
                "high market risk suggesting significant concerns",
                "extreme market risk indicating severe volatility"
            ]
        },
        'kmv_distance_to_default': {
            'thresholds': [0.0, 1.5, 3.0, 5.0],
            'descriptions': [
                "company is at immediate risk of default",
                "elevated default risk requiring urgent attention",
                "moderate default risk needing monitoring",
                "low default risk indicating stable credit profile",
                "minimal default risk with strong credit metrics"
            ]
        },
        'return_on_assets': {
            'thresholds': [0.0, 0.02, 0.05, 0.10],
            'descriptions': [
                "negative asset returns indicating poor operational efficiency",
                "weak asset utilization requiring improvement",
                "adequate asset efficiency within industry standards",
                "strong asset utilization indicating good management",
                "exceptional asset efficiency demonstrating superior operations"
            ]
        },
        'market_cap': {
            'thresholds': [1e9, 10e9, 100e9, 1000e9],
            'descriptions': [
                "small company with limited market presence",
                "mid-cap company with moderate market presence",
                "large-cap company with strong market position",
                "mega-cap company with dominant market position",
                "ultra-large company with exceptional market dominance"
            ]
        },
        'total_assets': {
            'thresholds': [1e9, 10e9, 50e9, 200e9],
            'descriptions': [
                "small asset base indicating limited operational scale",
                "moderate asset base with adequate operational capacity",
                "substantial asset base supporting strong operations",
                "large asset base demonstrating significant operational scale",
                "massive asset base indicating industry-leading scale"
            ]
        },
        'total_revenue': {
            'thresholds': [1e9, 10e9, 50e9, 100e9],
            'descriptions': [
                "low revenue indicating small operational scale",
                "moderate revenue with adequate market presence",
                "strong revenue demonstrating solid market position",
                "high revenue indicating large market presence",
                "exceptional revenue demonstrating market leadership"
            ]
        },
        'total_equity': {
            'thresholds': [1e9, 10e9, 50e9, 100e9],
            'descriptions': [
                "limited equity base indicating potential capital constraints",
                "adequate equity providing reasonable financial cushion",
                "strong equity base supporting stable operations",
                "substantial equity indicating robust financial position",
                "exceptional equity demonstrating superior financial strength"
            ]
        }
    }
    feature_key = feature_name.lower()
    if feature_key in interpretations:
        thresholds = interpretations[feature_key]['thresholds']
        descriptions = interpretations[feature_key]['descriptions']
        # Find appropriate description based on value
        description_idx = 0
        for i, threshold in enumerate(thresholds):
            if value <= threshold:
                description_idx = i
                break
        else:
            description_idx = len(descriptions) - 1
        return descriptions[description_idx]
    else:
        # Generic interpretation based on contribution
        if contribution > 0:
            return "factor contributing positively to credit assessment"
        elif contribution < 0:
            return "factor indicating increased credit risk"
        else:
            return "neutral factor with minimal impact"

def _extract_news_data(unstructured_result: Dict[str, Any]) -> Tuple[List[str], List[str], List[float]]:
    """Extract news headlines, sentiments, and scores"""
    try:
        sample_headlines = unstructured_result.get('sample_headlines', [])
        sentiment_dist = unstructured_result.get('sentiment_distribution', {})
        sentiment_scores = unstructured_result.get('sentiment_scores', []) # Assuming this exists
        headlines = []
        sentiments = []
        scores = []

        # Use actual data if available
        if sample_headlines and sentiment_scores and len(sample_headlines) == len(sentiment_scores):
             # Assume sentiment_scores contains 'negative', 'neutral', 'positive' scores
             # Let's simplify and assume it's a list of primary sentiment labels or scores
             # The original logic was flawed. Let's try to get real data.
             # If sentiment_distribution exists, we can infer sentiments.
             total_articles = sum(sentiment_dist.values()) if sentiment_dist else 1
             if total_articles == 0: total_articles = 1

             neg_ratio = sentiment_dist.get('negative', 0) / total_articles
             pos_ratio = sentiment_dist.get('positive', 0) / total_articles
             neu_ratio = sentiment_dist.get('neutral', 0) / total_articles

             # Assign sentiment based on distribution (simplified)
             for i, headline in enumerate(sample_headlines[:3]):
                 headlines.append(headline)
                 # Assign sentiment based on distribution (simplified)
                 rand_val = np.random.random() if i >= len(sentiment_scores) else sentiment_scores[i] # Fallback to random if not enough scores
                 if isinstance(rand_val, str): # If it's a label
                     sentiments.append(rand_val.title())
                     # Assign a mock score based on label
                     if rand_val.lower() == 'negative':
                         scores.append(-0.7)
                     elif rand_val.lower() == 'positive':
                         scores.append(0.6)
                     else:
                         scores.append(0.0)
                 elif isinstance(rand_val, (int, float)): # If it's a numerical score
                     scores.append(float(rand_val))
                     if rand_val < -0.1:
                         sentiments.append('Negative')
                     elif rand_val > 0.1:
                         sentiments.append('Positive')
                     else:
                         sentiments.append('Neutral')
                 else:
                     # Default fallback
                     if i == 0 and neg_ratio > 0.4:
                         sentiments.append('Negative')
                         scores.append(-0.7)
                     elif i == 1 and pos_ratio > 0.4:
                         sentiments.append('Positive')
                         scores.append(0.6)
                     else:
                         sentiments.append('Neutral')
                         scores.append(0.2)
        else:
            # Generate sample data from available information (fallback)
            for i, headline in enumerate(sample_headlines[:3]):
                headlines.append(headline)
                # Assign sentiment based on distribution (simplified)
                total_articles = sum(sentiment_dist.values())
                if total_articles > 0:
                    neg_ratio = sentiment_dist.get('negative', 0) / total_articles
                    pos_ratio = sentiment_dist.get('positive', 0) / total_articles
                    if i == 0 and neg_ratio > 0.4:
                        sentiments.append('Negative')
                        scores.append(-0.7)
                    elif i == 1 and pos_ratio > 0.4:
                        sentiments.append('Positive')
                        scores.append(0.6)
                    else:
                        sentiments.append('Neutral')
                        scores.append(0.2)
                else:
                    sentiments.append('Neutral')
                    scores.append(0.0)

        return headlines, sentiments, scores

    except Exception as e:
        logger.warning(f"Could not extract news data: {e}")
        return [], [], []

# --- FIXED THRESHOLDS for factor identification ---
POSITIVE_FACTOR_THRESHOLD = 0.005 # Slightly lowered threshold
NEGATIVE_FACTOR_THRESHOLD = -0.005

def _identify_positive_factors(ebm_output: List[Dict[str, Any]], sentiments: List[str]) -> List[str]:
    """Identify positive contributing factors"""
    factors = []
    # From EBM output
    for item in ebm_output:
        # Use a small threshold to identify significant positive contributions
        if item['raw_contribution'] > POSITIVE_FACTOR_THRESHOLD:
            factors.append(f"Strong {item['feature'].lower()}")
    # From sentiments
    positive_count = sentiments.count('Positive')
    if positive_count > 0:
        factors.append(f"Positive news sentiment ({positive_count} articles)")
    return factors

def _identify_negative_factors(ebm_output: List[Dict[str, Any]], sentiments: List[str]) -> List[str]:
    """Identify negative risk factors"""
    factors = []
    # From EBM output
    for item in ebm_output:
        # Use a small threshold to identify significant negative contributions
        if item['raw_contribution'] < NEGATIVE_FACTOR_THRESHOLD:
            factors.append(f"Weak {item['feature'].lower()}")
    # From sentiments
    negative_count = sentiments.count('Negative')
    if negative_count > 0:
        factors.append(f"Negative news sentiment ({negative_count} articles)")
    return factors

# === EBM EXPLAINER ===

class EBMExplainer:
    """Comprehensive explainability class for EBM credit scoring model."""

    def __init__(self, model_path):
        """Initialize the explainer with a trained model."""
        self.model_path = model_path
        self.model_data = None
        self.ebm_model = None
        self.scaler = None
        self.feature_columns = None
        self.load_model()

    def load_model(self):
        """Load the trained EBM model and associated components."""
        logger.info(f"Loading EBM model from {self.model_path}...")
        try:
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            self.ebm_model = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.feature_columns = self.model_data['feature_columns']
            logger.info("EBM model loaded successfully.")
            logger.info(f"Model accuracy: {self.model_data.get('accuracy', 'N/A')}")
            logger.info(f"Features: {len(self.feature_columns)}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_feature_interpretation(self, feature_name, value, contribution):
        """Get business interpretation for a specific feature value."""
        # Reuse the global function for consistency
        return _get_detailed_feature_interpretation(feature_name, value, contribution)

    def explain_single_prediction(self, sample_data, company_name="Unknown Company"):
        """Generate detailed explanation for a single prediction."""
        if isinstance(sample_data, dict):
            sample_df = pd.DataFrame([sample_data])
        elif isinstance(sample_data, pd.Series):
            sample_df = sample_data.to_frame().T
        else:
            sample_df = sample_data.copy()

        available_features = [col for col in self.feature_columns if col in sample_df.columns]
        sample_df = sample_df[available_features]
        sample_df = sample_df.fillna(0)
        sample_df = sample_df.replace([np.inf, -np.inf], 0)

        # Ensure scaler is loaded correctly
        if self.scaler is None:
             logger.error("Scaler is None. Cannot transform data for explanation.")
             raise ValueError("Scaler not loaded. Check model file.")

        sample_scaled = self.scaler.transform(sample_df)

        try:
            prediction = self.ebm_model.predict(sample_scaled)[0]
            prediction_proba = self.ebm_model.predict_proba(sample_scaled)[0]
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            prediction = 0
            prediction_proba = [0.5, 0.5]

        # Get feature contributions using EBM's explain_local
        feature_names = self.feature_columns
        feature_values = sample_df.iloc[0].to_dict()
        feature_contributions = {}

        # --- ONLY USE EBM's built-in explain_local for contributions ---
        try:
            # Use EBM's built-in explain_local for accurate contributions
            logger.debug(f"Attempting explain_local with sample_scaled shape: {sample_scaled.shape}")
            local_explanation = self.ebm_model.explain_local(sample_scaled)
            logger.debug(f"Local explanation object type: {type(local_explanation)}")

            # --- EXTRACT FEATURE CONTRIBUTIONS FROM EBM EXPLANATION ---
            if local_explanation is not None:
                try:
                    # Get explanation data for the first (and only) instance
                    # Assuming InterpretML API where data(index) or data() works
                    if hasattr(local_explanation, 'data'):
                        # Try getting data for instance 0 first, fallback to overall data
                        explanation_data = None
                        try:
                            explanation_data = local_explanation.data(0)
                            logger.debug("Successfully retrieved local explanation data for instance 0.")
                        except:
                            pass
                        if explanation_data is None:
                             try:
                                 explanation_data = local_explanation.data() # Fallback to overall data call
                                 logger.debug("Retrieved local explanation data (overall).")
                             except:
                                 pass

                        if explanation_data is not None and isinstance(explanation_data, dict):
                            feature_scores_all = explanation_data.get('scores', [])
                            feature_names_generic = explanation_data.get('names', [])

                            # Only use main features (not interactions) - they correspond to our feature_columns
                            main_features_count = len(self.feature_columns)
                            if len(feature_scores_all) >= main_features_count:
                                # Take only the first N scores corresponding to main features
                                main_scores = feature_scores_all[:main_features_count]

                                # Map feature names to their scores
                                # Check if names match feature_columns exactly
                                if (len(feature_names_generic) >= main_features_count and
                                    all(name == feature_name for name, feature_name in zip(feature_names_generic[:main_features_count], self.feature_columns))):
                                    # Names match exactly
                                    logger.debug("Feature names from explain_local match feature_columns exactly.")
                                    feature_contributions = dict(zip(feature_names_generic[:main_features_count], main_scores))
                                elif (len(feature_names_generic) >= main_features_count and
                                      all(isinstance(name, str) and name.startswith('feature_') for name in feature_names_generic[:main_features_count])):
                                    # Names are generic 'feature_0000' style. Assume order matches feature_columns.
                                    logger.debug("Feature names are generic. Assuming order matches feature_columns.")
                                    feature_contributions = dict(zip(self.feature_columns, main_scores))
                                elif len(feature_names_generic) >= main_features_count:
                                    # Names are provided and seem specific, but don't match exactly.
                                    # This is the warning case from the debug log.
                                    # Let's try to match names robustly.
                                    logger.warning("Names from explain_local didn't match feature_columns exactly. Attempting robust mapping...")
                                    # Create a mapping from name to index in feature_names_generic
                                    name_to_idx_generic = {name: i for i, name in enumerate(feature_names_generic[:main_features_count])}
                                    mapped_contributions = {}
                                    for feature_name in self.feature_columns:
                                        if feature_name in name_to_idx_generic:
                                            idx = name_to_idx_generic[feature_name]
                                            mapped_contributions[feature_name] = feature_scores_all[idx]
                                        else:
                                            logger.warning(f"Feature '{feature_name}' not found in explain_local names. Using score by index.")
                                            # Fallback to index-based mapping if name not found
                                            generic_idx = self.feature_columns.index(feature_name)
                                            if generic_idx < len(feature_scores_all):
                                                mapped_contributions[feature_name] = feature_scores_all[generic_idx]
                                            else:
                                                mapped_contributions[feature_name] = 0.0
                                    feature_contributions = mapped_contributions
                                else:
                                    # If no names or not enough, assume order matches feature_columns (last resort)
                                    logger.warning("Not enough names returned or names unavailable. Assuming order matches feature_columns.")
                                    feature_contributions = dict(zip(self.feature_columns, main_scores))

                                logger.info(f"Mapped local contributions for {len(feature_contributions)} features.")
                            else:
                                logger.warning(f"Insufficient scores returned by explain_local. Got {len(feature_scores_all)}, expected >= {main_features_count}")
                        else:
                             logger.warning("Local explanation data is None or not a dictionary.")
                    else:
                        logger.warning("Local explanation object does not have a 'data' method.")
                except Exception as extract_e:
                    logger.error(f"Failed to extract local contributions: {extract_e}")
            else:
                logger.warning("Local explanation object is None.")
        except AttributeError as ae:
            logger.error(f"AttributeError while getting local feature contributions: {ae}")
        except Exception as e:
            logger.error(f"Error calling explain_local or processing its output: {e}")

        # --- NO FALLBACK TO GLOBAL ---
        # If feature_contributions is still empty, log an error and initialize with zeros
        if not feature_contributions:
            logger.error("Failed to retrieve local feature contributions from explain_local. No fallback used.")
            feature_contributions = {name: 0.0 for name in feature_names}
        else:
            logger.info("Successfully retrieved local feature contributions.")

        # Ensure all expected features are present in the final dictionary, fill missing with 0
        final_feature_contributions = {name: feature_contributions.get(name, 0.0) for name in feature_names}

        # Convert to list in the correct order for compatibility with _generate_detailed_explanation
        feature_scores = [final_feature_contributions[name] for name in feature_names]

        explanation = self._generate_detailed_explanation(
            company_name, prediction, prediction_proba,
            feature_names, feature_scores, # Pass the correct lists
            feature_values # Pass actual feature values
        )

        return {
            'explanation_text': explanation,
            'prediction': 'Investment Grade' if prediction == 1 else 'Non-Investment Grade',
            'probability_investment_grade': float(prediction_proba[1]), # Ensure float
            'probability_non_investment_grade': float(prediction_proba[0]), # Ensure float
            'feature_contributions': final_feature_contributions # Return the correct contributions dictionary
        }

    def _generate_detailed_explanation(self, company_name, prediction, prediction_proba,
                                     feature_names, feature_scores, feature_values):
        """Generate the detailed textual explanation."""
        feature_data = list(zip(feature_names, feature_scores,
                               [feature_values.get(name, 0) for name in feature_names]))
        feature_data.sort(key=lambda x: abs(x[1]), reverse=True)

        explanation_lines = []
        explanation_lines.append(f"CREDIT RISK ANALYSIS FOR {company_name.upper()}")
        explanation_lines.append("=" * 60)
        explanation_lines.append("")

        grade = "INVESTMENT GRADE" if prediction == 1 else "NON-INVESTMENT GRADE"
        explanation_lines.append(f"OVERALL RATING: {grade}")
        explanation_lines.append(f"Investment Grade Probability: {prediction_proba[1]:.1%}")
        explanation_lines.append(f"Non-Investment Grade Probability: {prediction_proba[0]:.1%}")
        explanation_lines.append("")

        explanation_lines.append("DETAILED FEATURE ANALYSIS:")
        explanation_lines.append("-" * 40)
        explanation_lines.append("")

        total_abs_contribution = sum(abs(score) for _, score, _ in feature_data)
        if total_abs_contribution == 0: total_abs_contribution = 1e-8 # Avoid division by zero

        for i, (feature_name, contribution, value) in enumerate(feature_data[:10]):
            contrib_pct = (abs(contribution) / total_abs_contribution) * 100
            if prediction == 1:
                impact = "INCREASES investment grade probability" if contribution > 0 else "DECREASES investment grade probability"
                impact_symbol = "+" if contribution > 0 else "-"
            else:
                impact = "INCREASES non-investment grade probability" if contribution > 0 else "DECREASES non-investment grade probability"
                impact_symbol = "+" if contribution > 0 else "-"

            interpretation = self.get_feature_interpretation(feature_name, value, contribution)
            explanation_lines.append(f"{i+1}. {feature_name.replace('_', ' ').title()}:")
            explanation_lines.append(f"   Value: {value:.4f}")
            explanation_lines.append(f"   Interpretation: {interpretation}")
            explanation_lines.append(f"   Impact: [{impact_symbol}] {impact} by {contrib_pct:.1f}%")
            explanation_lines.append(f"   Contribution Score: {contribution:+.4f}")
            explanation_lines.append("")

        explanation_lines.append(f"Analysis generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(explanation_lines)

# === NEWS EXPLAINABILITY (copied from news_explainability.py) ===

class NewsExplainabilityEngine:
    """Comprehensive explainability engine for news-based risk analysis."""

    def __init__(self):
        self.risk_categories = {
            'liquidity_crisis': {
                'name': 'Liquidity & Cash Flow Issues',
                'description': 'Problems with immediate cash availability and working capital',
                'impact': 'High immediate risk to operations and debt servicing',
                'keywords': ['cash flow', 'liquidity crisis', 'cash shortage', 'working capital',
                           'credit facility', 'refinancing', 'cash burn', 'funding gap']
            },
            'debt_distress': {
                'name': 'Debt & Financial Distress',
                'description': 'Issues with debt obligations and financial health',
                'impact': 'Very high risk of default or restructuring',
                'keywords': ['debt default', 'covenant violation', 'bankruptcy', 'insolvency',
                           'debt restructuring', 'creditor pressure', 'leverage concerns', 'debt burden']
            }
        }

    def analyze_news_impact(self, news_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Main function to analyze and explain news impact."""
        if not news_assessment or 'detailed_analysis' not in news_assessment:
            return self._create_no_data_explanation()

        detailed = news_assessment['detailed_analysis']
        risk_score = news_assessment.get('risk_score', 50)
        sentiment_analysis = self._analyze_sentiment_impact(detailed)
        risk_factor_analysis = self._analyze_risk_factors(news_assessment)
        temporal_analysis = self._analyze_temporal_trends(detailed)
        confidence_analysis = self._analyze_confidence_factors(news_assessment)

        explanation = self._generate_comprehensive_explanation(
            news_assessment, sentiment_analysis, risk_factor_analysis,
            temporal_analysis, confidence_analysis
        )

        return {
            'company': news_assessment.get('company', 'Unknown'),
            'risk_score': risk_score,
            'risk_level': self._categorize_risk_level(risk_score),
            'confidence': news_assessment.get('confidence', 0.5),
            'explanation': explanation,
            'sentiment_analysis': sentiment_analysis,
            'risk_factor_analysis': risk_factor_analysis,
            'temporal_analysis': temporal_analysis,
            'confidence_analysis': confidence_analysis,
            'actionable_insights': self._generate_actionable_insights(
                risk_score, sentiment_analysis, risk_factor_analysis
            ),
            'data_quality': self._assess_data_quality(news_assessment)
        }

    def _analyze_sentiment_impact(self, detailed_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the sentiment distribution and its impact on risk"""
        sentiment_dist = detailed_analysis.get('sentiment_distribution', {})
        total_articles = sum(sentiment_dist.values())
        if total_articles == 0:
            return {'overall_sentiment': 'unknown', 'impact': 'Cannot assess', 'distribution': {}}

        sentiment_percentages = {
            sentiment: (count / total_articles) * 100
            for sentiment, count in sentiment_dist.items()
        }

        if sentiment_percentages.get('negative', 0) > 60:
            overall_sentiment = 'predominantly_negative'
            impact_desc = 'High negative impact on perceived risk'
        elif sentiment_percentages.get('positive', 0) > 60:
            overall_sentiment = 'predominantly_positive'
            impact_desc = 'Positive impact reducing perceived risk'
        else:
            overall_sentiment = 'mixed'
            impact_desc = 'Neutral impact with mixed signals'

        return {
            'overall_sentiment': overall_sentiment,
            'impact': impact_desc,
            'distribution': sentiment_percentages,
            'total_articles': total_articles,
            'dominant_sentiment': max(sentiment_dist.items(), key=lambda x: x[1])[0] if sentiment_dist else 'unknown'
        }

    def _analyze_risk_factors(self, news_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze specific risk factors identified in the news"""
        risk_keywords = news_assessment.get('detailed_analysis', {}).get('risk_keywords', [])
        if not risk_keywords:
            return {'categories_detected': {}, 'total_risk_signals': 0, 'primary_concerns': []}

        categorized_risks = {}
        for category, info in self.risk_categories.items():
            category_matches = []
            for keyword, count in risk_keywords:
                if keyword.lower() in [k.lower() for k in info['keywords']]:
                    category_matches.append((keyword, count))
            if category_matches:
                total_mentions = sum(count for _, count in category_matches)
                categorized_risks[category] = {
                    'name': info['name'],
                    'description': info['description'],
                    'impact': info['impact'],
                    'matches': category_matches,
                    'total_mentions': total_mentions
                }

        primary_concerns = sorted(
            categorized_risks.items(),
            key=lambda x: x[1]['total_mentions'],
            reverse=True
        )[:3]

        return {
            'categories_detected': categorized_risks,
            'total_risk_signals': len(risk_keywords),
            'primary_concerns': [
                {
                    'category': cat,
                    'name': info['name'],
                    'mentions': info['total_mentions'],
                    'impact': info['impact']
                }
                for cat, info in primary_concerns
            ]
        }

    def _analyze_temporal_trends(self, detailed_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal trends in sentiment"""
        temporal_trend = detailed_analysis.get('temporal_trend', 0)
        if temporal_trend > 5:
            trend_desc = 'Strongly improving'
            impact = 'Positive - risk perception decreasing over time'
        elif temporal_trend > 2:
            trend_desc = 'Improving'
            impact = 'Somewhat positive - slight improvement in sentiment'
        elif temporal_trend > -2:
            trend_desc = 'Stable'
            impact = 'Neutral - consistent sentiment pattern'
        elif temporal_trend > -5:
            trend_desc = 'Deteriorating'
            impact = 'Concerning - sentiment worsening over time'
        else:
            trend_desc = 'Sharply deteriorating'
            impact = 'High concern - rapidly worsening sentiment'

        return {
            'trend_direction': trend_desc,
            'trend_value': temporal_trend,
            'impact_assessment': impact,
            'significance': 'High' if abs(temporal_trend) > 3 else 'Moderate' if abs(temporal_trend) > 1 else 'Low'
        }

    def _analyze_confidence_factors(self, news_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze factors affecting confidence in the assessment"""
        detailed = news_assessment.get('detailed_analysis', {})
        articles_count = detailed.get('articles_analyzed', 0)
        confidence = news_assessment.get('confidence', 0.5)

        confidence_factors = []
        if articles_count >= 15:
            confidence_factors.append('Sufficient news coverage (15+ articles)')
        elif articles_count >= 8:
            confidence_factors.append('Adequate news coverage (8-14 articles)')
        elif articles_count >= 3:
            confidence_factors.append('Limited news coverage (3-7 articles)')
        else:
            confidence_factors.append('Very limited news coverage (<3 articles)')

        confidence_factors.append('Analysis covers recent 7-day period')

        if confidence > 0.8:
            confidence_factors.append('High model confidence in predictions')
        elif confidence > 0.6:
            confidence_factors.append('Moderate model confidence')
        else:
            confidence_factors.append('Lower model confidence - interpret with caution')

        return {
            'overall_confidence': confidence,
            'confidence_level': 'High' if confidence > 0.7 else 'Moderate' if confidence > 0.5 else 'Low',
            'contributing_factors': confidence_factors,
            'data_sufficiency': 'Sufficient' if articles_count >= 10 else 'Limited' if articles_count >= 5 else 'Insufficient'
        }


    def _generate_comprehensive_explanation(
        self,
        news_assessment: Dict[str, Any],
        sentiment_analysis: Dict[str, Any],
        risk_factor_analysis: Dict[str, Any],
        temporal_analysis: Dict[str, Any],
        confidence_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate the main comprehensive explanation as structured JSON"""

        risk_score = news_assessment.get('risk_score', 50)
        company = news_assessment.get('company', 'the company')

        # Determine risk category based on score
        if risk_score < 30:
            risk_category = "LOW RISK"
            summary_text = f"News sentiment indicates LOW RISK for {company}. Recent coverage is predominantly positive with minimal risk indicators."
        elif risk_score < 50:
            risk_category = "MODERATE-LOW RISK"
            summary_text = f"News sentiment indicates MODERATE-LOW RISK for {company}. Mixed sentiment with some areas of concern."
        elif risk_score < 70:
            risk_category = "MODERATE-HIGH RISK"
            summary_text = f"News sentiment indicates MODERATE-HIGH RISK for {company}. Notable negative sentiment and risk factors present."
        else:
            risk_category = "HIGH RISK"
            summary_text = f"News sentiment indicates HIGH RISK for {company}. Predominantly negative coverage with significant risk indicators."

        # Sentiment distribution
        sentiment_distribution = {
            "positive": sentiment_analysis['distribution'].get('positive', 0),
            "neutral": sentiment_analysis['distribution'].get('neutral', 0),
            "negative": sentiment_analysis['distribution'].get('negative', 0),
            "overall_sentiment": sentiment_analysis['overall_sentiment']
        }

        # Primary concerns
        primary_concerns = []
        if risk_factor_analysis.get('primary_concerns'):
            for concern in risk_factor_analysis['primary_concerns']:
                primary_concerns.append({
                    "name": concern['name'],
                    "mentions": concern['mentions'],
                    "impact": concern['impact']
                })

        # Temporal analysis
        temporal_data = {
            "trend_direction": temporal_analysis['trend_direction'],
            "impact_assessment": temporal_analysis['impact_assessment']
        }

        # Sample headlines (if available)
        sample_headlines = []
        if news_assessment.get('sample_headlines'):
            for i, headline in enumerate(news_assessment['sample_headlines'][:3], 1):
                sample_headlines.append({
                    "headline": headline,
                    "approx_published": f"{i + 1} days ago"  # placeholder
                })

        # Confidence assessment
        confidence = {
            "overall_confidence": confidence_analysis['overall_confidence'],
            "confidence_level": confidence_analysis['confidence_level'],
            "data_sufficiency": confidence_analysis['data_sufficiency']
        }

        # Final structured response
        return {
            "company": company,
            "risk_score": round(risk_score, 2),
            "risk_category": risk_category,
            "summary": summary_text,
            "sentiment_breakdown": sentiment_distribution,
            "primary_risk_factors": primary_concerns,
            "temporal_analysis": temporal_data,
            "sample_headlines": sample_headlines,
            "confidence_assessment": confidence
    }
    def _generate_actionable_insights(
    self,
    risk_score: float,
    sentiment_analysis: Dict[str, Any],
    risk_factor_analysis: Dict[str, Any]
) -> Dict[str, Any]:
        """Generate actionable insights in machine-readable format"""
    
        if risk_score > 70:
            priority = "HIGH"
            recommendation = "Monitor for immediate developments that could affect liquidity or operations"
        elif risk_score > 50:
            priority = "MEDIUM"
            recommendation = "Watch for trend continuation and specific risk factor developments"
        else:
            priority = "LOW"
            recommendation = "Maintain standard monitoring protocols"

        top_concern = None
        primary_concerns = risk_factor_analysis.get('primary_concerns', [])
        if primary_concerns:
            top_concern = {
                "focus_area": primary_concerns[0]['name'],
                "mentions": primary_concerns[0]['mentions']
            }

        return {
            "priority_level": priority,
            "recommendation": recommendation,
            "focus_area": top_concern
        }

    def _assess_data_quality(self, news_assessment: Dict[str, Any]) -> Dict[str, str]:
        """Assess the quality and reliability of the underlying data"""
        detailed = news_assessment.get('detailed_analysis', {})
        articles_count = detailed.get('articles_analyzed', 0)

        if articles_count >= 15:
            coverage = "Excellent"
        elif articles_count >= 10:
            coverage = "Good"
        elif articles_count >= 5:
            coverage = "Adequate"
        else:
            coverage = "Limited"

        return {
            'coverage_quality': coverage,
            'data_recency': "Current (7-day window)",
            'source_diversity': "Multiple sources" if articles_count > 5 else "Limited sources",
            'overall_quality': coverage
        }

    def _categorize_risk_level(self, score: float) -> str:
        """Categorize risk level based on score"""
        if score < 25:
            return "LOW RISK"
        elif score < 45:
            return "MODERATE-LOW RISK"
        elif score < 65:
            return "MODERATE-HIGH RISK"
        else:
            return "HIGH RISK"

    def _create_no_data_explanation(self) -> Dict[str, Any]:
        """Create explanation when no data is available"""
        return {
            'risk_score': 50.0,
            'risk_level': 'MODERATE (NO DATA)',
            'explanation': 'No recent news data available for analysis. Using neutral risk assessment.',
            'confidence': 0.1,
            'data_quality': {'overall_quality': 'No Data'},
            'actionable_insights': ['Seek alternative data sources for risk assessment']
        }

def explain_news_assessment(news_assessment: Dict[str, Any]) -> Dict[str, Any]:
    """Main convenience function to generate explanations for news assessments."""
    engine = NewsExplainabilityEngine()
    return engine.analyze_news_impact(news_assessment)

# === MAIN EXPLAINABILITY FUNCTIONS ===

def explain_structured_score(processed_features: Dict[str, Any], company_name: str) -> Dict[str, Any]:
    """
    Generate explanation for structured risk score using embedded EBMExplainer.
    """
    try:
        logger.info(f"📊 Generating structured explanation for {company_name}")
        explainer = EBMExplainer(MODEL_PATH)
        explanation_result = explainer.explain_single_prediction(processed_features, company_name)
        logger.info(f"✅ Structured explanation generated for {company_name}")
        return {
            'explanation_text': explanation_result.get('explanation_text', 'No explanation available'),
            'prediction': explanation_result.get('prediction', 'Unknown'),
            'investment_grade_probability': explanation_result.get('probability_investment_grade', 0.5),
            'non_investment_grade_probability': explanation_result.get('probability_non_investment_grade', 0.5),
            'feature_contributions': explanation_result.get('feature_contributions', {}),
            'explanation_type': 'EBM Structured Analysis',
            'company': company_name,
            'explanation_confidence': 'High',
            'method': 'Explainable Boosting Machine (EBM) with SHAP-style explanations'
        }
    except Exception as e:
        logger.error(f"Error generating structured explanation for {company_name}: {e}")
        return {
            'explanation_text': f"Error generating structured explanation for {company_name}: {str(e)}",
            'prediction': 'Error',
            'investment_grade_probability': 0.5,
            'explanation_type': 'Error',
            'error': str(e),
            'company': company_name
        }

def explain_unstructured_score(unstructured_result: Dict[str, Any], company_name: str) -> Dict[str, Any]:
    """
    Generate explanation for unstructured risk score using embedded NewsExplainabilityEngine.
    """
    try:
        logger.info(f"📰 Generating unstructured explanation for {company_name}")
        news_assessment = {
            'company': company_name,
            'risk_score': unstructured_result.get('risk_score', 50.0),
            'confidence': unstructured_result.get('confidence', 0.5),
            'detailed_analysis': {
                'articles_analyzed': unstructured_result.get('articles_analyzed', 0),
                'sentiment_distribution': unstructured_result.get('sentiment_distribution', {}),
                'temporal_trend': unstructured_result.get('temporal_trend', 0.0),
                'risk_keywords': unstructured_result.get('risk_keywords', []),
                'base_sentiment_score': unstructured_result.get('base_sentiment_score', 50.0),
                'avg_risk_score': unstructured_result.get('avg_risk_score', 0.0)
            },
            'sample_headlines': unstructured_result.get('sample_headlines', [])
        }
        explanation_result = explain_news_assessment(news_assessment)
        logger.info(f"✅ Unstructured explanation generated for {company_name}")
        return {
            'explanation': explanation_result.get('explanation', 'No explanation available'),
            'risk_level': explanation_result.get('risk_level', 'Unknown'),
            'confidence': explanation_result.get('confidence', 0.5),
            'sentiment_analysis': explanation_result.get('sentiment_analysis', {}),
            'risk_factor_analysis': explanation_result.get('risk_factor_analysis', {}),
            'temporal_analysis': explanation_result.get('temporal_analysis', {}),
            'confidence_analysis': explanation_result.get('confidence_analysis', {}),
            'actionable_insights': explanation_result.get('actionable_insights', []),
            'data_quality': explanation_result.get('data_quality', {}),
            'explanation_type': 'News Sentiment Analysis',
            'company': company_name,
            'method': 'FinBERT + Financial Risk Detection + Temporal Analysis'
        }
    except Exception as e:
        logger.error(f"Error generating unstructured explanation for {company_name}: {e}")
        return {
            'explanation': f"Error generating unstructured explanation for {company_name}: {str(e)}",
            'risk_level': 'Error',
            'confidence': 0.1,
            'explanation_type': 'Error',
            'error': str(e),
            'company': company_name
        }
from typing import Dict, Any

def explain_fusion(
    fusion_result: Dict[str, Any],
    structured_result: Dict[str, Any],
    unstructured_result: Dict[str, Any],
    company_name: str
) -> Dict[str, Any]:
    """
    Generate a machine-readable explanation for the fusion process and final score.
    """
    try:
        logger.info(f"🔄 Generating fusion explanation for {company_name}")

        final_score = fusion_result.get('fused_score', 50.0)
        expert_agreement = fusion_result.get('expert_agreement', 0.5)
        market_regime = fusion_result.get('market_regime', 'NORMAL')
        dynamic_weights = fusion_result.get('dynamic_weights', {})
        expert_contributions = fusion_result.get('expert_contributions', {})
        regime_adjustment = fusion_result.get('regime_adjustment', 0.0)

        # Build structured JSON response
        structured_response = {
            "company": company_name,
            "fusion_analysis": {
                "final_score": round(final_score, 2),
                "expert_agreement": expert_agreement,
                "market_regime": market_regime,
                "regime_adjustment": regime_adjustment,
                "fusion_method": "Dynamic Weighted Fusion",
                "methodology": "Dynamic weighted fusion with market condition adjustments"
            },
            "experts": {}
        }

        # Add structured expert details
        if 'structured_expert' in expert_contributions:
            struct_contrib = expert_contributions['structured_expert']
            structured_response["experts"]["structured_analysis"] = {
                "score": struct_contrib.get('score', struct_contrib.get('risk_score', 0)),
                "weight": dynamic_weights.get('structured_expert', 0.5),
                "contribution_points": struct_contrib.get('contribution', 0)
            }

        if 'news_sentiment_expert' in expert_contributions:
            news_contrib = expert_contributions['news_sentiment_expert']
            structured_response["experts"]["news_sentiment_analysis"] = {
                "score": news_contrib.get('score', news_contrib.get('risk_score', 0)),
                "weight": dynamic_weights.get('news_sentiment_expert', 0.5),
                "contribution_points": news_contrib.get('contribution', 0)
            }

        logger.info(f"✅ Fusion explanation generated for {company_name}")
        return structured_response

    except Exception as e:
        logger.error(f"Error generating fusion explanation for {company_name}: {e}")
        return {
            "company": company_name,
            "fusion_analysis": {
                "final_score": fusion_result.get('fused_score', 50.0),
                "fusion_method": "MAESTRO Dynamic Weighted Fusion",
                "error": str(e)
            },
            "status": "error"
        }
