

"""
fusion_engine.py

Fusion engine for risk scoring.
Self-contained implementation without external dependencies.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Union
import statistics
from datetime import datetime

logger = logging.getLogger(__name__)

# === dynamic_fusion GLOBAL FUSION (copied from global_fusion.py) ===

class GlobalFusion:
    """
    Fusion system that:
    1. Combines multiple expert assessments intelligently
    2. Adapts weights based on expert agreement and market conditions
    3. Provides comprehensive explanations for all decisions
    4. Maintains confidence tracking throughout the process
    """
    
    def __init__(self):
        self.expert_weights = {
            'structured_expert': 0.6,  # EBM model
            'news_sentiment_expert': 0.4,  # News analysis
            'market_data_expert': 0.3,  # Market/macro data (if available)
            'technical_expert': 0.2   # Technical indicators (if available)
        }
        
        self.market_regimes = {
            'BULL': {'risk_adjustment': -5, 'volatility_threshold': 0.15},
            'BEAR': {'risk_adjustment': +10, 'volatility_threshold': 0.25},
            'VOLATILE': {'risk_adjustment': +7, 'volatility_threshold': 0.30},
            'NORMAL': {'risk_adjustment': 0, 'volatility_threshold': 0.20}
        }
        
        self.fusion_history = []
        
    def determine_market_regime(self, market_data: Dict[str, Any] = None) -> str:
        """
        Determine current market regime based on available indicators
        """
        if not market_data:
            return 'NORMAL'
        
        try:
            # Check if we have volatility indicators
            vix_level = market_data.get('vix', 20)  # Default VIX
            market_trend = market_data.get('market_trend', 0)  # -1 to 1
            
            if vix_level > 30:
                return 'VOLATILE'
            elif vix_level > 25 and market_trend < -0.3:
                return 'BEAR'
            elif vix_level < 15 and market_trend > 0.3:
                return 'BULL'
            else:
                return 'NORMAL'
                
        except Exception as e:
            logger.warning(f"Error determining market regime: {e}")
            return 'NORMAL'
    
    def calculate_expert_agreement(self, expert_assessments: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate agreement level between expert assessments
        """
        risk_scores = []
        
        for expert_name, assessment in expert_assessments.items():
            if 'risk_score' in assessment:
                risk_scores.append(assessment['risk_score'])
        
        if len(risk_scores) < 2:
            return 0.5  # Neutral agreement when insufficient data
        
        # Calculate coefficient of variation (lower = higher agreement)
        mean_score = statistics.mean(risk_scores)
        if mean_score == 0:
            return 1.0
            
        std_dev = statistics.stdev(risk_scores) if len(risk_scores) > 1 else 0
        cv = std_dev / mean_score
        
        # Convert to agreement score (0 to 1, higher = better agreement)
        agreement = max(0, 1 - (cv / 0.5))  # Normalize CV to 0-1 scale
        
        return min(1.0, agreement)
    
    def adjust_weights_dynamically(self, expert_assessments: Dict[str, Dict[str, Any]], 
                                 expert_agreement: float, market_regime: str) -> Dict[str, float]:
        """
        Dynamically adjust expert weights based on confidence and market conditions
        """
        base_weights = self.expert_weights.copy()
        adjusted_weights = {}
        
        # Adjust based on individual expert confidence
        for expert_name, assessment in expert_assessments.items():
            if expert_name in base_weights:
                base_weight = base_weights[expert_name]
                confidence = assessment.get('confidence', 0.5)
                
                # Boost weight for high-confidence experts
                confidence_adjustment = (confidence - 0.5) * 0.4
                adjusted_weight = base_weight * (1 + confidence_adjustment)
                adjusted_weights[expert_name] = max(0.1, adjusted_weight)
        
        # Market regime adjustments
        if market_regime == 'VOLATILE':
            # In volatile markets, trust structured data more
            if 'structured_expert' in adjusted_weights:
                adjusted_weights['structured_expert'] *= 1.2
            if 'news_sentiment_expert' in adjusted_weights:
                adjusted_weights['news_sentiment_expert'] *= 0.9
                
        elif market_regime == 'BEAR':
            # In bear markets, news sentiment becomes more important
            if 'news_sentiment_expert' in adjusted_weights:
                adjusted_weights['news_sentiment_expert'] *= 1.3
                
        elif market_regime == 'BULL':
            # In bull markets, balance both sources
            pass  # Keep balanced
        
        # Agreement-based adjustment
        if expert_agreement < 0.3:
            # Low agreement - be more conservative, weight towards structured
            if 'structured_expert' in adjusted_weights:
                adjusted_weights['structured_expert'] *= 1.1
        
        # Normalize weights to sum to 1
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for expert in adjusted_weights:
                adjusted_weights[expert] /= total_weight
        
        return adjusted_weights
    
    def calculate_regime_adjustment(self, base_score: float, market_regime: str, 
                                  expert_agreement: float) -> float:
        """
        Calculate market regime-based adjustment to the fused score
        """
        regime_info = self.market_regimes.get(market_regime, self.market_regimes['NORMAL'])
        base_adjustment = regime_info['risk_adjustment']
        
        # Scale adjustment based on expert agreement
        # Lower agreement = larger adjustment (more uncertainty)
        agreement_factor = 1 + (1 - expert_agreement) * 0.5
        
        final_adjustment = base_adjustment * agreement_factor
        
        # Ensure adjustment doesn't push score outside reasonable bounds
        adjusted_score = base_score + final_adjustment
        if adjusted_score < 0:
            final_adjustment = -base_score
        elif adjusted_score > 100:
            final_adjustment = 100 - base_score
            
        return final_adjustment
    
    def fuse_expert_assessments(self, expert_assessments: Dict[str, Dict[str, Any]], 
                              market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main fusion function that combines all expert assessments
        """
        logger.info("🔄 Starting fusion process...")
        
        if not expert_assessments:
            logger.warning("No expert assessments provided for fusion")
            return self._create_default_result()
        
        # Step 1: Determine market regime
        market_regime = self.determine_market_regime(market_data)
        logger.info(f"📊 Market regime detected: {market_regime}")
        
        # Step 2: Calculate expert agreement
        expert_agreement = self.calculate_expert_agreement(expert_assessments)
        logger.info(f"🤝 Expert agreement level: {expert_agreement:.2%}")
        
        # Step 3: Adjust weights dynamically
        dynamic_weights = self.adjust_weights_dynamically(
            expert_assessments, expert_agreement, market_regime
        )
        weights_str = ", ".join([f"{k}: {v:.2f}" for k, v in dynamic_weights.items()])
        logger.info(f"⚖️ Dynamic weights: {weights_str}")
        
        # Step 4: Perform weighted fusion
        weighted_score = 0
        total_weight = 0
        expert_contributions = {}
        
        for expert_name, assessment in expert_assessments.items():
            if expert_name in dynamic_weights:
                weight = dynamic_weights[expert_name]
                risk_score = assessment.get('risk_score', 50)
                
                contribution = weight * risk_score
                weighted_score += contribution
                total_weight += weight
                
                expert_contributions[expert_name] = {
                    'risk_score': risk_score,
                    'weight': weight,
                    'contribution': contribution,
                    'confidence': assessment.get('confidence', 0.5)
                }
        
        # Normalize if needed
        if total_weight > 0:
            base_fused_score = weighted_score / total_weight
        else:
            base_fused_score = 50  # Default neutral score
        
        # Step 5: Apply market regime adjustment
        regime_adjustment = self.calculate_regime_adjustment(
            base_fused_score, market_regime, expert_agreement
        )
        
        final_fused_score = base_fused_score + regime_adjustment
        final_fused_score = max(0, min(100, final_fused_score))  # Clamp to [0,100]
        
        # Step 6: Calculate overall confidence
        confidence_scores = [assessment.get('confidence', 0.5) 
                           for assessment in expert_assessments.values()]
        base_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.5
        
        # Adjust confidence based on agreement
        confidence_adjustment = expert_agreement * 0.3
        overall_confidence = min(1.0, base_confidence + confidence_adjustment)
        
        # Step 7: Compile results
        fusion_result = {
            'fused_score': final_fused_score,
            'base_score': base_fused_score,
            'regime_adjustment': regime_adjustment,
            'market_regime': market_regime,
            'expert_agreement': expert_agreement,
            'overall_confidence': overall_confidence,
            'dynamic_weights': dynamic_weights,
            'expert_contributions': expert_contributions,
            'fusion_method': 'dynamic_fusion',
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_fusion_summary(
                final_fused_score, market_regime, expert_agreement
            )
        }
        
        # Store in history
        self.fusion_history.append(fusion_result)
        
        logger.info(f"✅ Fusion complete. Final score: {final_fused_score:.1f}")
        
        return fusion_result
    
    def _generate_fusion_summary(self, final_score: float, market_regime: str, 
                               expert_agreement: float) -> str:
        """Generate a human-readable summary of the fusion result"""
        
        risk_level = "HIGH" if final_score > 70 else "MODERATE" if final_score > 40 else "LOW"
        agreement_level = "HIGH" if expert_agreement > 0.7 else "MODERATE" if expert_agreement > 0.4 else "LOW"
        
        summary = f"Fusion Result: {risk_level} risk (score: {final_score:.1f}/100) "
        summary += f"with {agreement_level} expert agreement ({expert_agreement:.1%}) "
        summary += f"under {market_regime} market conditions."
        
        return summary
    
    def _create_default_result(self) -> Dict[str, Any]:
        """Create a default result when no expert assessments are available"""
        return {
            'fused_score': 50.0,
            'base_score': 50.0,
            'regime_adjustment': 0.0,
            'market_regime': 'NORMAL',
            'expert_agreement': 0.0,
            'overall_confidence': 0.1,
            'dynamic_weights': {},
            'expert_contributions': {},
            'fusion_method': 'dynamic_fusion',
            'timestamp': datetime.now().isoformat(),
            'summary': 'No expert assessments available for fusion',
            'error': 'No expert assessments provided'
        }
    
    def get_fusion_explanation(self, fusion_result: Dict[str, Any]) -> str:
        """
        Generate detailed explanation of the fusion process
        """
        explanation = []
        
        explanation.append("FUSION EXPLANATION")
        explanation.append("=" * 40)
        explanation.append("")
        
        explanation.append(f"Final Fused Score: {fusion_result['fused_score']:.1f}/100")
        explanation.append(f"Market Regime: {fusion_result['market_regime']}")
        explanation.append(f"Expert Agreement: {fusion_result['expert_agreement']:.1%}")
        explanation.append(f"Overall Confidence: {fusion_result['overall_confidence']:.1%}")
        explanation.append("")
        
        explanation.append("Expert Contributions:")
        for expert, contrib in fusion_result.get('expert_contributions', {}).items():
            explanation.append(f"  {expert}:")
            explanation.append(f"    Score: {contrib['risk_score']:.1f}")
            explanation.append(f"    Weight: {contrib['weight']:.1%}")
            explanation.append(f"    Contribution: {contrib['contribution']:.1f}")
        
        if fusion_result.get('regime_adjustment', 0) != 0:
            explanation.append("")
            explanation.append(f"Market Adjustment: {fusion_result['regime_adjustment']:+.1f}")
        
        return "\n".join(explanation)

def fuse_scores(structured_result: Dict[str, Any], unstructured_result: Dict[str, Any], 
               market_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main function to fuse structured and unstructured scores using dynamic_fusion.
    
    Args:
        structured_result: Result from structured analysis
        unstructured_result: Result from unstructured analysis  
        market_data: Optional market/macro data for regime detection
        
    Returns:
        Dictionary with fused score and detailed fusion explanation
    """
    try:
        logger.info("🔄 Starting score fusion with dynamic_fusion...")
        
        # Initialize fusion engine
        dynamic_fusion = GlobalFusion()
        
        # Prepare expert assessments
        expert_assessments = {}
        
        # Add structured expert
        if structured_result and ('structured_score' in structured_result or 'risk_score' in structured_result):
            risk_score = structured_result.get('structured_score', structured_result.get('risk_score', 50.0))
            expert_assessments['structured_expert'] = {
                'risk_score': risk_score,
                'confidence': structured_result.get('confidence', 0.8),  # EBM typically high confidence
                'assessment_type': 'structured',
                'details': structured_result
            }
            logger.info(f"📊 Structured expert: {risk_score:.1f}")
        
        # Add news sentiment expert
        if unstructured_result and ('unstructured_score' in unstructured_result or 'risk_score' in unstructured_result):
            risk_score = unstructured_result.get('unstructured_score', unstructured_result.get('risk_score', 50.0))
            expert_assessments['news_sentiment_expert'] = {
                'risk_score': risk_score,
                'confidence': unstructured_result.get('confidence', 0.6),
                'assessment_type': 'unstructured',
                'details': unstructured_result
            }
            logger.info(f"📰 News sentiment expert: {risk_score:.1f}")
        
        if not expert_assessments:
            logger.warning("No valid expert assessments for fusion")
            return {
                'fused_score': 50.0,
                'confidence': 0.1,
                'explanation': 'No valid assessments available for fusion',
                'method': 'dynamic_fusion',
                'error': 'No valid assessments'
            }
        
        # Perform fusion
        fusion_result = dynamic_fusion.fuse_expert_assessments(expert_assessments, market_data)
        
        # Add explanation
        fusion_result['detailed_explanation'] = dynamic_fusion.get_fusion_explanation(fusion_result)
        
        logger.info(f"✅ Fusion complete. Final score: {fusion_result['fused_score']:.1f}")
        
        return fusion_result
        
    except Exception as e:
        logger.error(f"Error in score fusion: {e}")
        return {
            'fused_score': 50.0,
            'confidence': 0.1,
            'explanation': f'Fusion error: {str(e)}',
            'method': 'dynamic_fusion',
            'error': str(e)
        }
