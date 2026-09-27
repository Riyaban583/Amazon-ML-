import os
from typing import Optional
import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.ensemble import HistGradientBoostingClassifier


class EntityMatchingModel:
    """
    Akshat's Module: Machine Learning Matching Classifier.
    
    Trained on candidate pairs to estimate P(Match = 1 | features).
    Utilizes LightGBM with HistGradientBoosting fallback, balanced weights,
    and probability calibration.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        if LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.06,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=self.random_state,
                verbose=-1,
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.06,
                max_depth=6,
                class_weight="balanced",
                random_state=self.random_state,
            )

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Train the classifier."""
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match probabilities."""
        if len(X) == 0:
            return np.array([])
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names: list) -> pd.Series:
        """Return feature importance Series if supported."""
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)
        return pd.Series()

    def save(self, filepath: str, metadata: Optional[dict] = None):
        """Save model artifact."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {"model": self.model}
        if metadata:
            payload.update(metadata)
        joblib.dump(payload, filepath)

    def load(self, filepath: str) -> dict:
        """Load model artifact."""
        loaded = joblib.load(filepath)
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            return loaded
        else:
            self.model = loaded
            return {"model": self.model}

    @property
    def is_fitted(self) -> bool:
        """Check if model is already fitted."""
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(self.model)
            return True
        except Exception:
            return False
