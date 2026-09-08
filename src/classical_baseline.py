"""Phase 7: Classical NLP baseline (TF-IDF + Logistic Regression).

Provides a calibrated probabilistic regime classifier that consumes
warning text and returns a RegimeInterpretation compatible with the
existing downstream CausalOptimizer pipeline.

The model is trained on the original 18 Phase-5 templates and evaluated
on the 36 held-out confirmation templates (Phase 6).
"""

from __future__ import annotations

import json
import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from src.events import (
    Regime, REGIME_WARNING_TEMPLATES, REGIME_PARAMS,
    P5_NORMAL_LEAD_TIME, P5_EVENT_DURATION,
)
from src.interpreter import RegimeInterpretation

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase7_classical_baseline"

REGIME_LIST = [Regime.NORMAL, Regime.SUPPLIER_DELAY, Regime.DEMAND_SURGE]
REGIME_LABELS = [r.value for r in REGIME_LIST]

_TEMPLATE_FAMILY_MAP = {
    "delay_clear": "delay", "delay_moderate": "delay", "delay_vague": "delay",
    "normal_clear": "normal", "normal_moderate": "normal", "normal_vague": "normal",
    "surge_clear": "surge", "surge_moderate": "surge", "surge_vague": "surge",
    "cd_clear": "delay", "cd_moderate": "delay", "cd_vague": "delay",
    "cn_clear": "normal", "cn_moderate": "normal", "cn_vague": "normal",
    "cs_clear": "surge", "cs_moderate": "surge", "cs_vague": "surge",
}


def _template_family(template_id: str) -> str:
    prefix = "_".join(template_id.split("_")[:2])
    return _TEMPLATE_FAMILY_MAP.get(prefix, prefix)


def _get_default_params(regime: Regime) -> tuple[int, int, float]:
    if regime == Regime.SUPPLIER_DELAY:
        return (P5_NORMAL_LEAD_TIME + REGIME_PARAMS[Regime.SUPPLIER_DELAY]["lead_time_increase"],
                P5_EVENT_DURATION, 1.0)
    elif regime == Regime.DEMAND_SURGE:
        return (0, P5_EVENT_DURATION, REGIME_PARAMS[Regime.DEMAND_SURGE]["demand_multiplier"])
    return (0, 0, 1.0)


def build_dataset(templates: list[dict]) -> tuple[list[str], list[str], list[str], list[str]]:
    texts, labels, template_ids, families = [], [], [], []
    for t in templates:
        texts.append(t["text"])
        labels.append(t["regime"].value if isinstance(t["regime"], Regime) else t["regime"])
        template_ids.append(t["template_id"])
        families.append(_template_family(t["template_id"]))
    return texts, labels, template_ids, families


def grouped_train_test_split(
    texts, labels, template_ids, families, test_template_ids,
):
    train_idx = [i for i, tid in enumerate(template_ids) if tid not in test_template_ids]
    test_idx = [i for i, tid in enumerate(template_ids) if tid in test_template_ids]
    return (
        [texts[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [template_ids[i] for i in train_idx],
        [families[i] for i in train_idx],
        [texts[i] for i in test_idx],
        [labels[i] for i in test_idx],
        [template_ids[i] for i in test_idx],
        [families[i] for i in test_idx],
    )


@dataclass
class ClassicalBaselineResult:
    regime_probabilities: dict[str, float]
    estimated_lt_increase: int = 0
    estimated_duration: int = 0
    estimated_demand_multiplier: float = 1.0
    model_name: str = "TFIDF_LogReg"

    def to_regime_interpretation(self) -> RegimeInterpretation:
        return RegimeInterpretation(
            regime_probabilities=dict(self.regime_probabilities),
            estimated_lt_increase=self.estimated_lt_increase,
            estimated_duration=self.estimated_duration,
            estimated_demand_multiplier=self.estimated_demand_multiplier,
        )


# Training-pipeline variants.  These are deliberately named as *pipelines*
# rather than "raw vs calibrated", because `CalibratedClassifierCV` changes
# the fitted classifier as well as the probability map (see FOLD_ENSEMBLE).
VARIANT_SINGLE = "single"            # one logistic regression on all training texts
VARIANT_FOLD_ENSEMBLE = "fold_ensemble"  # the CV fold ensemble, WITHOUT calibration
VARIANT_CALIBRATED = "calibrated"    # the CV fold ensemble WITH a calibration map
VARIANTS = (VARIANT_SINGLE, VARIANT_FOLD_ENSEMBLE, VARIANT_CALIBRATED)


class TFIDFLogReg:
    """TF-IDF + logistic-regression regime classifier.

    Three training pipelines are supported so that the effect of *calibration*
    can be separated from the effect of *fold ensembling*:

    ``single``
        One logistic regression fitted on all training texts.
    ``fold_ensemble``
        ``cv`` logistic regressions fitted on the same cross-validation folds
        `CalibratedClassifierCV` uses, averaged, with no calibration map.
        This is the control that isolates ensembling.
    ``calibrated``
        The same fold ensemble with an isotonic or sigmoid calibration map.

    Comparing ``single`` with ``calibrated`` alone therefore compares two
    training pipelines, not an isolated calibration intervention; the
    ``fold_ensemble`` control supplies the missing arm.  With small calibration
    sets, ``cal_method="sigmoid"`` is the recommended alternative to isotonic
    (scikit-learn cautions against isotonic on very few calibration points).

    The vectorizer is fitted *inside* each fold pipeline whenever
    cross-validation is used, so fold vocabularies are not shared.
    """

    def __init__(
        self,
        max_features: int = 500,
        ngram_range: tuple = (1, 2),
        C: float = 1.0,
        calibrate: bool = False,
        cal_method: str = "isotonic",
        seed: int = 42,
        variant: Optional[str] = None,
        cv: int = 3,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.cal_method = cal_method
        self.seed = seed
        self.cv = cv

        if variant is None:
            variant = VARIANT_CALIBRATED if calibrate else VARIANT_SINGLE
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.calibrate = variant == VARIANT_CALIBRATED

        self.vectorizer_ = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.model_ = LogisticRegression(
            C=C, max_iter=1000, solver="lbfgs",
            random_state=seed,
        )
        self.calibrated_model_ = None
        self.fold_models_ = None
        self.is_fitted_ = False
        self.train_size_ = 0
        self.vocabulary_size_ = 0
        self.fold_sizes_ = []
        self.classes_ = None

    def _make_pipeline(self) -> Pipeline:
        """Vectorizer + classifier as one estimator, so CV refits the vocabulary."""
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("lr", LogisticRegression(
                C=self.C, max_iter=1000, solver="lbfgs", random_state=self.seed,
            )),
        ])

    def fit(self, texts: list[str], labels: list[str]) -> "TFIDFLogReg":
        texts = list(texts)
        labels = list(labels)
        self.train_size_ = len(texts)

        # The single-model path always exists: it is the shared reference
        # classifier and supplies the vocabulary used by `predict_*` when no
        # fold pipeline is active.
        X = self.vectorizer_.fit_transform(texts)
        self.vocabulary_size_ = len(self.vectorizer_.vocabulary_)
        self.model_.fit(X, labels)
        self.classes_ = list(self.model_.classes_)

        if self.variant in (VARIANT_FOLD_ENSEMBLE, VARIANT_CALIBRATED):
            splitter = StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.seed,
            )
            self.fold_sizes_ = [
                {"train": len(tr), "calibration": len(te)}
                for tr, te in splitter.split(np.zeros(len(texts)), labels)
            ]

        if self.variant == VARIANT_CALIBRATED:
            splitter = StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.seed,
            )
            self.calibrated_model_ = CalibratedClassifierCV(
                self._make_pipeline(), cv=splitter, method=self.cal_method,
            )
            self.calibrated_model_.fit(np.array(texts, dtype=object), labels)
            self.classes_ = list(self.calibrated_model_.classes_)
        elif self.variant == VARIANT_FOLD_ENSEMBLE:
            # Exactly the classifiers CalibratedClassifierCV would fit, with
            # the calibration map omitted.  Averaging their probabilities
            # isolates the ensembling component of the calibrated pipeline.
            splitter = StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.seed,
            )
            arr = np.array(texts, dtype=object)
            self.fold_models_ = []
            for train_idx, _ in splitter.split(arr, labels):
                pipe = self._make_pipeline()
                pipe.fit(arr[train_idx], [labels[i] for i in train_idx])
                self.fold_models_.append(pipe)
            self.classes_ = list(self.fold_models_[0].classes_)

        self.is_fitted_ = True
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted")
        if self.calibrated_model_ is not None:
            return self.calibrated_model_.predict_proba(np.array(list(texts), dtype=object))
        if self.fold_models_ is not None:
            # Align every fold's columns to the shared class order before
            # averaging; folds can see different label subsets.
            acc = np.zeros((len(texts), len(self.classes_)))
            for pipe in self.fold_models_:
                p = pipe.predict_proba(np.array(list(texts), dtype=object))
                for j, cls in enumerate(pipe.classes_):
                    acc[:, self.classes_.index(cls)] += p[:, j]
            return acc / len(self.fold_models_)
        X = self.vectorizer_.transform(texts)
        return self.model_.predict_proba(X)

    def predict(self, texts: list[str]) -> list[str]:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted")
        probs = self.predict_proba(texts)
        return [self.classes_[i] for i in probs.argmax(axis=1)]

    def predict_regime(self, text: str) -> ClassicalBaselineResult:
        probs = self.predict_proba([text])[0]
        classes = self.classes_
        prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
        for r in REGIME_LABELS:
            if r not in prob_dict:
                prob_dict[r] = 0.0

        most_likely = max(prob_dict, key=prob_dict.get)
        regime_enum = Regime(most_likely) if most_likely in [r.value for r in Regime] else Regime.NORMAL
        lt, dur, mult = _get_default_params(regime_enum)

        return ClassicalBaselineResult(
            regime_probabilities=prob_dict,
            estimated_lt_increase=lt,
            estimated_duration=dur,
            estimated_demand_multiplier=mult,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer_,
                "model": self.model_,
                "calibrated_model": self.calibrated_model_,
                "fold_models": self.fold_models_,
                "classes": self.classes_,
                "fold_sizes": self.fold_sizes_,
                "config": {
                    "max_features": self.max_features,
                    "ngram_range": self.ngram_range,
                    "C": self.C,
                    "cal_method": self.cal_method,
                    "seed": self.seed,
                    "variant": self.variant,
                    "cv": self.cv,
                },
                "is_fitted": self.is_fitted_,
                "train_size": self.train_size_,
                "vocabulary_size": self.vocabulary_size_,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "TFIDFLogReg":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(**data["config"])
        obj.vectorizer_ = data["vectorizer"]
        obj.model_ = data["model"]
        obj.calibrated_model_ = data["calibrated_model"]
        obj.fold_models_ = data.get("fold_models")
        obj.fold_sizes_ = data.get("fold_sizes", [])
        obj.is_fitted_ = data["is_fitted"]
        obj.train_size_ = data["train_size"]
        obj.vocabulary_size_ = data["vocabulary_size"]
        obj.classes_ = data.get("classes") or list(obj.model_.classes_)
        return obj


def evaluate_classification(y_true, y_pred, proba, classes) -> dict:
    acc = accuracy_score(y_true, y_pred)

    present_labels = [c for c in classes if c in y_true]
    cm = confusion_matrix(y_true, y_pred, labels=present_labels) if present_labels else np.zeros((len(classes), len(classes)))

    brier_scores = {}
    for i, cls in enumerate(classes):
        binary_true = [1.0 if y == cls else 0.0 for y in y_true]
        brier_scores[cls] = brier_score_loss(binary_true, proba[:, i])

    present_in_true = [c for c in classes if c in y_true]
    if len(present_in_true) >= 2:
        ll = log_loss(y_true, proba, labels=classes)
    else:
        ll = 0.0

    report = classification_report(y_true, y_pred, labels=present_labels, output_dict=True, zero_division=0)

    return {
        "accuracy": float(acc),
        "brier_macro": float(np.mean(list(brier_scores.values()))),
        "brier_by_class": brier_scores,
        "log_loss": float(ll),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def leakage_audit(
    train_ids, test_ids, train_families, test_families,
) -> dict:
    train_set = set(train_ids)
    test_set = set(test_ids)
    overlap = train_set & test_set

    train_fam = set(train_families)
    test_fam = set(test_families)
    family_overlap = train_fam & test_fam

    train_labels_during_test = set()
    test_families_in_train = set()
    for i, fam in enumerate(test_families):
        for j, tf in enumerate(train_families):
            if tf == fam:
                test_families_in_train.add(fam)
                break

    return {
        "train_templates": len(train_ids),
        "test_templates": len(test_ids),
        "template_id_overlap": list(overlap),
        "no_template_leak": len(overlap) == 0,
        "train_families": sorted(train_fam),
        "test_families": sorted(test_fam),
        "families_shared": sorted(family_overlap),
        "note": "Templates from same family appear in both splits (intentional). "
                "No template ID overlap ensures no exact-text leakage.",
    }


def compute_dataset_manifest(
    original_templates, confirmation_templates,
    train_texts, train_labels, train_ids, train_families,
    test_texts, test_labels, test_ids, test_families,
) -> dict:
    from collections import Counter

    orig_regimes = Counter(t["regime"].value if isinstance(t["regime"], Regime) else t["regime"]
                           for t in original_templates)
    conf_regimes = Counter(t["regime"].value if isinstance(t["regime"], Regime) else t["regime"]
                           for t in confirmation_templates)

    train_amb = Counter()
    test_amb = Counter()
    for t in original_templates:
        if t["template_id"] in train_ids:
            train_amb[t["ambiguity_level"]] += 1
    for t in confirmation_templates:
        if t["template_id"] in test_ids:
            test_amb[t["ambiguity_level"]] += 1

    return {
        "original_templates": len(original_templates),
        "original_regime_distribution": dict(orig_regimes),
        "confirmation_templates": len(confirmation_templates),
        "confirmation_regime_distribution": dict(conf_regimes),
        "train_templates": len(train_ids),
        "train_regime_distribution": dict(Counter(train_labels)),
        "train_ambiguity_distribution": dict(train_amb),
        "train_template_families": sorted(set(train_families)),
        "test_templates": len(test_ids),
        "test_regime_distribution": dict(Counter(test_labels)),
        "test_ambiguity_distribution": dict(test_amb),
        "test_template_families": sorted(set(test_families)),
        "vectorizer_fitted_on_train_only": True,
        "calibration_on_train_only": True,
    }


def cross_validate_on_original(
    templates: list[dict],
    n_splits: int = 3,
    seed: int = 42,
) -> dict:
    texts, labels, template_ids, families = build_dataset(templates)

    cv_results = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        model = TFIDFLogReg(seed=seed)
        model.fit(train_texts, train_labels)

        val_proba = model.predict_proba(val_texts)
        val_pred = model.predict(val_texts)
        classes = model.classes_

        eval_ = evaluate_classification(val_labels, val_pred, val_proba, classes)
        cv_results.append({
            "fold": fold,
            "train_size": len(train_texts),
            "val_size": len(val_texts),
            **eval_,
        })

    avg_acc = np.mean([r["accuracy"] for r in cv_results])
    avg_brier = np.mean([r["brier_macro"] for r in cv_results])
    avg_ll = np.mean([r["log_loss"] for r in cv_results])

    return {
        "n_folds": n_splits,
        "avg_accuracy": float(avg_acc),
        "avg_brier_macro": float(avg_brier),
        "avg_log_loss": float(avg_ll),
        "folds": cv_results,
    }
