import logging
import pickle
from typing import Dict, List, Set, Optional, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
import os

import evaluate

logger = logging.getLogger(__name__)

class EntityMatcher:
    """
    Model for predicting match probabilities for candidate pairs.
    Uses LightGBM optimized for precision (F0.5).
    """
    
    def __init__(self, model_params: Optional[Dict] = None):
        """
        Initialize the LightGBM model.
        
        Args:
            model_params: Dictionary of LightGBM parameters. If None, default 
                parameters favoring precision will be used.
        """
        self.model = None
        self.features = None
        
        if model_params is None:
            self.model_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting': 'gbdt',
                'num_leaves': 63,
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'verbosity': -1
            }
        else:
            self.model_params = model_params

    def train(self, X_train: pd.DataFrame, y_train: np.ndarray, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None):
        """
        Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logger.info("Training LightGBM model...")
        
        # Calculate class distribution to handle imbalance
        num_positives = np.sum(y_train)
        num_negatives = len(y_train) - num_positives
        
        if 'scale_pos_weight' not in self.model_params and num_positives > 0:
            # Cap at 3.0 — full ratio can be 50-100x which destroys precision
            # on the F0.5 metric (precision is 2x more important than recall)
            raw_ratio = num_negatives / num_positives
            scale_pos_weight = min(raw_ratio, 3.0)
            self.model_params['scale_pos_weight'] = scale_pos_weight
            logger.info(f"Auto-computed scale_pos_weight: {scale_pos_weight:.2f} (raw ratio: {raw_ratio:.2f})")

        # Handle categorical features
        categorical_feature = 'auto'
        if isinstance(X_train, pd.DataFrame):
            if 'source_indicator' in X_train.columns:
                categorical_feature = ['source_indicator']
            self.features = list(X_train.columns)
        else:
            self.features = [f'f{i}' for i in range(X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]))]
            
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_feature)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=categorical_feature)
            valid_sets.append(val_data)
            valid_names.append('valid')
            
            # Using Early Stopping Callback
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]
            
            self.model = lgb.train(
                self.model_params,
                train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
        else:
            self.model = lgb.train(
                self.model_params,
                train_data
            )
            
        logger.info("Training completed.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return match probabilities.
        
        Args:
            X: Features dataframe
            
        Returns:
            1D array of probabilities (positive class)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return self.model.predict(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.65) -> np.ndarray:
        """
        Return binary predictions at a given threshold.
        
        Args:
            X: Features dataframe
            threshold: Probability threshold for positive class
            
        Returns:
            1D array of binary predictions
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def find_optimal_threshold(self, X_val, y_val, 
                               gt_dict: Dict[str, Set[str]], s1_ids: List[str], 
                               candidate_ids: List[str]) -> float:
        """
        Sweep thresholds and find the optimal one that maximizes F0.5 score.
        """
        logger.info("Finding optimal threshold for F0.5...")
        probs = self.predict_proba(X_val)
        
        thresholds = np.arange(0.3, 0.9 + 0.025, 0.025)
        best_threshold = 0.65
        best_f05 = -1.0
        
        for thresh in thresholds:
            # Build predictions dict at this threshold
            preds: Dict[str, Set[str]] = {}
            for i, (s1_id, cand_id) in enumerate(zip(s1_ids, candidate_ids)):
                if probs[i] >= thresh:
                    if s1_id not in preds:
                        preds[s1_id] = set()
                    preds[s1_id].add(cand_id)
            
            # Ensure all GT entities present
            for s1_id in gt_dict:
                if s1_id not in preds:
                    preds[s1_id] = set()
            
            f05 = evaluate.macro_f05(preds, gt_dict)
            
            if f05 > best_f05:
                best_f05 = f05
                best_threshold = thresh
                
        logger.info(f"Optimal threshold found: {best_threshold:.3f} (F0.5: {best_f05:.4f})")
        return best_threshold

    def feature_importance(self) -> pd.DataFrame:
        """
        Return feature importances sorted descending.
        
        Returns:
            DataFrame with 'feature' and 'importance' columns
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        importances = self.model.feature_importance(importance_type='gain')
        
        df = pd.DataFrame({
            'feature': self.features,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df

    def save(self, path: str):
        """Save the model to disk using pickle."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'features': self.features,
                'model_params': self.model_params
            }, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load the model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.features = data['features']
            self.model_params = data['model_params']
        logger.info(f"Model loaded from {path}")


def apply_margin_gating(predictions: pd.DataFrame, margin_threshold: float = 0.15) -> pd.DataFrame:
    """
    Apply margin gating to filter ambiguous matches.
    If the top candidate's score minus the score of the next-best candidate 
    is less than margin_threshold, it's considered ambiguous and dropped.
    
    Args:
        predictions: DataFrame with [source1_entity_id, candidate_entity_id, match_prob]
        margin_threshold: Minimum score margin required between top and next-best candidate
        
    Returns:
        Filtered DataFrame
    """
    logger.info(f"Applying margin gating with threshold {margin_threshold}...")
    
    if predictions.empty:
        return predictions
        
    # Sort by S1 ID and match probability descending
    df_sorted = predictions.sort_values(
        ['source1_entity_id', 'match_prob'], 
        ascending=[True, False]
    )
    
    # We will keep track of indices to keep
    indices_to_keep = []
    
    # Group by S1 entity
    for s1_id, group in df_sorted.groupby('source1_entity_id'):
        if len(group) == 1:
            indices_to_keep.append(group.index[0])
            continue
            
        top_idx = group.index[0]
        top_score = group['match_prob'].iloc[0]
        next_best_score = group['match_prob'].iloc[1]
        
        margin = top_score - next_best_score
        
        if margin >= margin_threshold:
            indices_to_keep.append(top_idx)
            
            # Note: The logic specified states "set the top candidate to no-match too" 
            # if the margin is too small. Thus we only keep the top candidate if it clears the margin.
            # Depending on if multiple matches are allowed, we could retain others, 
            # but standard margin gating targets the top match vs 2nd best.
    
    filtered_df = predictions.loc[indices_to_keep].copy()
    logger.info(f"Margin gating dropped {len(predictions) - len(filtered_df)} ambiguous pairs.")
    
    return filtered_df


def generate_matching_results(predictions: pd.DataFrame, all_s1_ids: List[str], 
                              threshold: float, margin_threshold: float = 0.15) -> Dict[str, Set[str]]:
    """
    Generate the final matching dict from scored candidate pairs.
    Applies threshold and margin gating.
    
    Args:
        predictions: DataFrame with candidates and their match probabilities
        all_s1_ids: List of all S1 entity IDs (to ensure singletons are captured)
        threshold: Match probability threshold
        margin_threshold: Minimum margin for top candidate
        
    Returns:
        Dictionary mapping S1 entity ID to set of matched entity IDs
    """
    # First, filter by the match threshold
    thresholded_preds = predictions[predictions['match_prob'] >= threshold].copy()
    
    # To properly compute margin against "next-best candidate that was NOT selected",
    # we need the original predictions, or we can just apply margin gating on the
    # whole predictions dataframe and then filter by threshold.
    # The prompt logic: "If the top candidate's score minus the score of the next-best 
    # candidate that was NOT selected (i.e., below match threshold) is less than margin_threshold"
    # Actually, let's group by and find the highest score above threshold, and highest score below threshold.
    
    results = {s1_id: set() for s1_id in all_s1_ids}
    
    if predictions.empty:
        return results
        
    df_sorted = predictions.sort_values(
        ['source1_entity_id', 'match_prob'], 
        ascending=[True, False]
    )
    
    for s1_id, group in df_sorted.groupby('source1_entity_id'):
        if s1_id not in results:
            continue
            
        matched_candidates = group[group['match_prob'] >= threshold]
        unmatched_candidates = group[group['match_prob'] < threshold]
        
        if matched_candidates.empty:
            continue
            
        # Top matched candidate
        top_matched_idx = matched_candidates.index[0]
        top_matched_score = matched_candidates['match_prob'].iloc[0]
        
        # Best unmatched candidate
        if unmatched_candidates.empty:
            best_unmatched_score = 0.0
        else:
            best_unmatched_score = unmatched_candidates['match_prob'].iloc[0]
            
        margin = top_matched_score - best_unmatched_score
        
        if margin >= margin_threshold:
            # We keep the matches. We add all candidates that passed the threshold.
            for _, row in matched_candidates.iterrows():
                results[s1_id].add(row['candidate_entity_id'])

    return results


def results_to_tsv(results: Dict[str, Set[str]], output_path: str):
    """
    Write matching results to a TSV file in submission format.
    Every S1 entity must have a row. Empty matched_entity_ids for singletons.
    
    Args:
        results: Dictionary mapping S1 entity ID to set of matched entity IDs
        output_path: Path to write the TSV file
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    logger.info(f"Writing results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        
        for s1_id, matched_ids in results.items():
            # Join multiple matched IDs with a comma or space, standard is often space or comma depending on contest,
            # we'll use space as typically standard for list of IDs, or maybe comma. Assuming space/comma.
            # I will assume space separated or comma separated based on common entity resolution formats. Let's use string representation.
            # If nothing specified, comma-separated is safest, or space-separated. Let's join by space.
            matched_str = ",".join(sorted(list(matched_ids)))
            f.write(f"{s1_id}\t{matched_str}\n")
            
    logger.info("Done writing results.")
