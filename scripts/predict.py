"""Prediction alignment and strict competition submission validation."""

from pathlib import Path
import pandas as pd

LABELS = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']


def create_submission(model, test: pd.DataFrame, sample: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Predict original string labels and align them to sample IDs before saving."""
    if list(sample.columns) != ['building_id', 'damage_grade']:
        raise ValueError('Unexpected sample submission columns.')
    if len(sample) != len(test) or sample.building_id.isna().any() or test.building_id.isna().any():
        raise ValueError('Invalid test/sample row counts or missing IDs.')
    if not test.building_id.is_unique or not sample.building_id.is_unique:
        raise ValueError('Submission IDs must be unique.')
    if set(sample.building_id) != set(test.building_id):
        raise ValueError('Sample and test IDs do not match.')
    predictions = pd.Series(model.predict(test), index=test.building_id)
    result = sample.copy()
    result['damage_grade'] = result.building_id.map(predictions)
    if result.damage_grade.isna().any() or not result.damage_grade.isin(LABELS).all():
        raise ValueError('Invalid predicted labels.')
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    reread = pd.read_csv(output)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), reread)
    return result
