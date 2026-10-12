"""Fold-local preprocessing and non-neural classifiers."""

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from scripts.feature_engineering import BuildingFeatures


def make_pipeline(X, kind: str, seed: int = 42) -> Pipeline:
    """Build an unfitted pipeline; the inspected frame determines only column types."""
    schema = BuildingFeatures().fit_transform(X.head(2))
    numeric = schema.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in schema if c not in numeric]
    numeric_steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if kind in ('dummy', 'logistic'):
        numeric_steps.append(('scale', StandardScaler()))
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    else:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    preprocessor = ColumnTransformer([
        ('numeric', Pipeline(numeric_steps), numeric),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('encoder', encoder),
        ]), categorical),
    ], remainder='drop', sparse_threshold=1.0)
    models = {
        'dummy': DummyClassifier(strategy='prior'),
        'logistic': LogisticRegression(max_iter=400, C=1.0, class_weight='balanced', random_state=seed),
        'forest': RandomForestClassifier(n_estimators=100, max_depth=20,
            min_samples_leaf=3, class_weight='balanced_subsample', n_jobs=2, random_state=seed),
        'lightgbm': LGBMClassifier(n_estimators=350, learning_rate=0.06,
            num_leaves=63, max_bin=127, colsample_bytree=0.9,
            reg_lambda=2.0, verbosity=-1, n_jobs=2, random_state=seed,
            deterministic=True, force_col_wise=True),
    }
    pipeline = Pipeline([('features', BuildingFeatures()), ('preprocess', preprocessor), ('model', models[kind])])
    # Ordinal codes are categorical, not continuous distances, for LightGBM.
    pipeline.categorical_indices = list(range(len(numeric), len(numeric) + len(categorical)))
    return pipeline
