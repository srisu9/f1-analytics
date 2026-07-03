\
\
\
\
\
\
\
\
\
\
\
\
\
   

import numpy as np
import joblib
import os


class EnsemblePredictor:
    """
    Weighted soft-voting ensemble of XGBoost, CatBoost, and LightGBM models.
    Falls back gracefully if a model is missing (adjusts weights automatically).
    """

    def __init__(self, models_dir="models", weights=None):
        """
        Parameters
        ----------
        models_dir : str
            Path to the directory containing the saved .joblib model files.
        weights : dict, optional
            Keys: 'xgb', 'cat', 'lgb'. Values: relative weights (will be
            normalised to sum to 1). Defaults to {xgb: 0.4, cat: 0.4, lgb: 0.2}.
        """
        if weights is None:
            weights = {"xgb": 0.4, "cat": 0.4, "lgb": 0.2}

        self.models_dir = models_dir
        self._requested_weights = weights
        self.models = {}
        self.weights = {}
        self._load_models()

                                                                        
    def _load_models(self):
        """Load whichever model files are present and normalise weights."""
        paths = {
            "xgb": os.path.join(self.models_dir, "xgboost_tuned.joblib"),
            "cat": os.path.join(self.models_dir, "catboost_model.joblib"),
            "lgb": os.path.join(self.models_dir, "lightgbm_model.joblib"),
        }

        loaded = {}
        for key, path in paths.items():
            if os.path.exists(path):
                try:
                    loaded[key] = joblib.load(path)
                    print(f"[EnsemblePredictor] Loaded {key} from {path}")
                except Exception as exc:
                    print(f"[EnsemblePredictor] Warning: could not load {key}: {exc}")

        if not loaded:
            raise FileNotFoundError(
                "No ensemble models found in '{}'. "
                "Run 08_walk_forward_eval.py first to train them.".format(self.models_dir)
            )

                                                            
        raw_w = {k: self._requested_weights.get(k, 0.0) for k in loaded}
        total = sum(raw_w.values())
        if total == 0:
            total = len(loaded)
            raw_w = {k: 1.0 for k in loaded}

        self.models  = loaded
        self.weights = {k: v / total for k, v in raw_w.items()}
        print(f"[EnsemblePredictor] Active weights: {self.weights}")

                                                                        
    def predict_proba(self, X):
        """
        Returns blended probability estimates, shape (n_samples, 2).
        Column 0 = P(class=0), Column 1 = P(class=1 / Top10=True).
        """
        blend = np.zeros(len(X))
        for key, model in self.models.items():
            proba  = model.predict_proba(X)[:, 1]               
            blend += self.weights[key] * proba

                                       
        return np.column_stack([1 - blend, blend])

    def predict(self, X, threshold=0.5):
        """Binary predictions at the given probability threshold."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
