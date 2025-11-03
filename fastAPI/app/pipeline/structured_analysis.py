"""
structured_analysis.py

Structured risk analysis using trained EBM model - simplified implementation.
"""

import logging
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import os
from dotenv import load_dotenv
logger = logging.getLogger(__name__)
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
def compute_structured_score(processed_features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Compute structured score using trained EBM model.
    Just takes classifier output and scales to 0-100.
    Returns: (structured_score, assessment_dict)
    """
    try:
        logger.info("Computing structured score...")
        # Load the model
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        ebm_model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        
        # Prepare features
        feature_df = pd.DataFrame([processed_features])
        available_features = [col for col in feature_columns if col in feature_df.columns]
        feature_df = feature_df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Scale and predict
        features_scaled = scaler.transform(feature_df)
        prediction_proba = ebm_model.predict_proba(features_scaled)[0]
        
        # Take classifier output and scale to 0-100
        # prediction_proba[1] = probability of Investment Grade 
        structured_score = prediction_proba[1] * 100
        
        logger.info(f"Structured Score: {structured_score:.1f}")
        assessment = {
            'structured_score': structured_score,
            'features_used': len(available_features),
            # No feature contributions here, let explainability.py handle that
        }
        return structured_score, assessment
    except Exception as e:
        logger.error(f"Error in structured analysis: {e}")
        assessment = {
            'structured_score': 50.0,
            'error': str(e)
        }
        return 50.0, assessment
