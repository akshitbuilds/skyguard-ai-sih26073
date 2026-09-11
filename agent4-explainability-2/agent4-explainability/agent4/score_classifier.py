from __future__ import annotations
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .score_features import SCORE_FEATURE_NAMES, score_feature_vector

DEFAULT_SCORE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "score_gate_model.joblib"
)


class ScoreGateClassifier:
    """
    RandomForest mapping (spatial_deviation_score, ml_anomaly_score,
    physically_coherent + interactions) -> genuine_event | sensor_fault.

    This is ONLY the statistical model. It is deliberately kept separate
    from the safety gate in `screening_gate.py` -- the model's raw output
    is never handed straight to a caller for a high_priority reading
    without passing through that gate first.
    """

    def __init__(self, model: RandomForestClassifier | None = None):
        self.model = model

    def train(self, df, label_col: str = "label", test_size: float = 0.15, verbose: bool = True):
        X = df[SCORE_FEATURE_NAMES].values
        y = df[label_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        if verbose:
            preds = clf.predict(X_test)
            print(classification_report(y_test, preds))
        self.model = clf
        return self

    def predict(self, features: dict):
        if self.model is None:
            raise RuntimeError("Model not trained/loaded yet.")
        x = score_feature_vector(features).reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        classes = self.model.classes_
        label_idx = int(np.argmax(proba))
        label = classes[label_idx]
        confidence = float(proba[label_idx])
        proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}
        return label, confidence, proba_dict

    def save(self, path: str = DEFAULT_SCORE_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str = DEFAULT_SCORE_MODEL_PATH) -> "ScoreGateClassifier":
        model = joblib.load(path)
        return cls(model=model)

    @classmethod
    def load_or_train(cls, path: str = DEFAULT_SCORE_MODEL_PATH) -> "ScoreGateClassifier":
        if os.path.exists(path):
            return cls.load(path)
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from data.generate_score_synthetic import build_score_dataset
        df = build_score_dataset()
        inst = cls().train(df)
        inst.save(path)
        return inst
