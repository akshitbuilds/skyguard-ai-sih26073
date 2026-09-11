from __future__ import annotations
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .features import FEATURE_NAMES, feature_vector

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "root_cause_model.joblib")


class RootCauseClassifier:
    """
    Thin wrapper around a RandomForestClassifier that maps a feature vector
    (see agent4.features) to one of the RootCauseLabel classes.

    RandomForest is chosen deliberately: it gives well-calibrated-enough
    class probabilities for a confidence score, and SHAP's TreeExplainer
    is exact and fast for tree ensembles (no sampling approximation needed),
    which matters for a live demo.
    """

    def __init__(self, model: RandomForestClassifier | None = None):
        self.model = model

    def train(self, df, label_col: str = "label", test_size: float = 0.15, verbose: bool = True):
        X = df[FEATURE_NAMES].values
        y = df[label_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
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
        x = feature_vector(features).reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        classes = self.model.classes_
        label_idx = int(np.argmax(proba))
        label = classes[label_idx]
        confidence = float(proba[label_idx])
        proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}
        return label, confidence, proba_dict

    def save(self, path: str = DEFAULT_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "RootCauseClassifier":
        model = joblib.load(path)
        return cls(model=model)

    @classmethod
    def load_or_train(cls, path: str = DEFAULT_MODEL_PATH) -> "RootCauseClassifier":
        if os.path.exists(path):
            return cls.load(path)
        # train fresh from synthetic data
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from data.generate_synthetic import build_dataset
        df = build_dataset()
        inst = cls().train(df)
        inst.save(path)
        return inst
