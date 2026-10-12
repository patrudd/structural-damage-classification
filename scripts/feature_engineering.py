"""Deterministic, pre-earthquake feature engineering without target information."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BuildingFeatures(TransformerMixin, BaseEstimator):
    """Remove identifiers and derive structural ratios and categorical geography."""

    def fit(self, X: pd.DataFrame, y=None):
        """Record the required input schema without learning from validation rows."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return independent engineered features while preserving row order."""
        X = X.loc[:, self.feature_names_in_].copy()
        X = X.drop(columns=['building_id'], errors='ignore')
        for column in ['district_id', 'vdcmun_id', 'ward_id']:
            X[column] = X[column].astype('string').fillna('missing').astype(object)
        floors = X['count_floors_pre_eq'].replace(0, np.nan)
        X['height_per_floor'] = X['height_ft_pre_eq'] / floors
        X['estimated_floor_area'] = X['plinth_area_sq_ft'] * floors
        X['age_is_999'] = X['age_building'].eq(999).astype(int)
        X['log_plinth_area'] = np.log1p(X['plinth_area_sq_ft'].clip(lower=0))
        return X.replace([np.inf, -np.inf], np.nan)
