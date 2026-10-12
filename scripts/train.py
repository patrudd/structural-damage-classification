"""EDA, baselines, development-only tuning, held-out evaluation and final refit."""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import lightgbm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from scripts.preprocess import make_pipeline
from scripts.predict import LABELS, create_submission


def stratified_subset(X, y, limit: int):
    """Select a reproducible bounded sample without mixing development and holdout."""
    if len(X) <= limit:
        return X, y
    subset, _, labels, _ = train_test_split(X, y, train_size=limit, stratify=y, random_state=42)
    return subset, labels


def inspect_data(data_dir: Path, output_dir: Path):
    """Validate the real competition schema and write an EDA audit and figures."""
    train, test, sample = [pd.read_csv(data_dir / name) for name in
        ['train.csv', 'test.csv', 'sample_submission.csv']]
    if set(train.damage_grade.unique()) != set(LABELS):
        raise ValueError('Unexpected target labels.')
    if set(train.columns) - {'damage_grade'} != set(test.columns):
        raise ValueError('Train and test feature schemas differ.')
    if not train.building_id.is_unique or not test.building_id.is_unique:
        raise ValueError('Repeated building IDs require grouped validation.')
    if set(train.building_id) & set(test.building_id):
        raise ValueError('Train and test building IDs overlap.')
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / 'reports/figures'
    figures.mkdir(parents=True, exist_ok=True)
    audit = {
        'train_shape': list(train.shape), 'test_shape': list(test.shape),
        'class_counts': train.damage_grade.value_counts().sort_index().to_dict(),
        'missing_train': train.isna().sum().to_dict(),
        'missing_test': test.isna().sum().to_dict(),
        'duplicate_rows_train': int(train.duplicated().sum()),
        'identical_feature_rows_train': int(train.drop(columns=['building_id', 'damage_grade']).duplicated().sum()),
        'unique_train_ids': bool(train.building_id.is_unique),
        'age_999_count': int(train.age_building.eq(999).sum()),
        'geography_overlap_test': {c: float(test[c].isin(train[c]).mean()) for c in ['district_id','vdcmun_id','ward_id']},
    }
    (output_dir / 'reports/data_audit.json').write_text(json.dumps(audit, indent=2))
    train.damage_grade.value_counts().sort_index().plot.bar(title='Distribusi kelas train', ylabel='Jumlah bangunan', rot=0)
    plt.tight_layout()
    plt.savefig(figures / 'class_distribution.png', dpi=150)
    plt.close()
    print('DATA_AUDIT', json.dumps(audit), flush=True)
    return train, test, sample


def run(data_dir: Path, output_dir: Path, quick: bool = False) -> dict:
    """Run reproducible experiments without using competition test labels."""
    train, test, sample = inspect_data(data_dir, output_dir)
    X, y = train.drop(columns='damage_grade'), train.damage_grade
    Xdev, Xhold, ydev, yhold = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pd.DataFrame({'building_id': X.building_id,
        'split': np.where(X.index.isin(Xhold.index), 'holdout', 'development')}).to_csv(output_dir/'reports/split.csv', index=False)
    Xsmall, ysmall = stratified_subset(Xdev, ydev, 400 if quick else 40000)
    benchmarks = []
    for kind in ['dummy', 'logistic', 'forest']:
        print('BASELINE_START', kind, flush=True)
        pipe = make_pipeline(Xsmall, kind)
        start = perf_counter()
        pipe.fit(Xsmall, ysmall)
        score = f1_score(yhold, pipe.predict(Xhold), average='macro')
        result = {'model': kind, 'train_rows': len(Xsmall), 'holdout_macro_f1': score,
                  'seconds': perf_counter() - start}
        benchmarks.append(result)
        print('BASELINE_RESULT', json.dumps(result), flush=True)
    base = make_pipeline(Xsmall, 'lightgbm')
    categorical_indices = base.categorical_indices
    params = {'model__num_leaves': [31,63], 'model__min_child_samples': [30,100],
              'model__class_weight': [None, 'balanced']}
    if quick:
        base.set_params(model__n_estimators=10)
    cv = StratifiedKFold(n_splits=2 if quick else 5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(base, params, n_iter=1 if quick else 3, scoring='f1_macro',
        cv=cv, n_jobs=1, random_state=42, refit=True, verbose=1, error_score='raise')
    print('TUNING_START: development subset only; holdout remains outside search', flush=True)
    search.fit(Xsmall, ysmall, model__categorical_feature=categorical_indices)
    pd.DataFrame(search.cv_results_).to_csv(output_dir/'reports/cv_results.csv', index=False)
    print('TUNING_RESULT', search.best_score_, search.best_params_, flush=True)
    selected = clone(search.best_estimator_)
    start = perf_counter()
    selected.fit(Xdev, ydev, model__categorical_feature=categorical_indices)
    predicted = selected.predict(Xhold)
    heldout_score = f1_score(yhold, predicted, average='macro')
    benchmarks.append({'model':'lightgbm_tuned', 'train_rows':len(Xdev),
                       'holdout_macro_f1':heldout_score, 'seconds':perf_counter()-start})
    pd.DataFrame(benchmarks).to_csv(output_dir/'reports/model_comparison.csv', index=False)
    report = classification_report(yhold, predicted, labels=LABELS, output_dict=True, zero_division=0)
    (output_dir/'reports/classification_report.json').write_text(json.dumps(report, indent=2))
    print('HOLDOUT_MACRO_F1', heldout_score, flush=True)
    print(classification_report(yhold, predicted, labels=LABELS, zero_division=0), flush=True)
    confusion = confusion_matrix(yhold, predicted, labels=LABELS)
    ConfusionMatrixDisplay(confusion, display_labels=LABELS).plot(cmap='Blues', xticks_rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir/'reports/figures/confusion_matrix.png', dpi=150)
    plt.close()
    Xi, yi = stratified_subset(Xhold, yhold, 100 if quick else 2000)
    importance = permutation_importance(selected, Xi, yi, scoring='f1_macro', n_repeats=2,
                                       n_jobs=1, random_state=42)
    imp = pd.DataFrame({'feature':Xi.columns, 'mean':importance.importances_mean,
                        'std':importance.importances_std}).sort_values('mean', ascending=False)
    imp.to_csv(output_dir/'reports/permutation_importance.csv', index=False)
    imp.head(12).sort_values('mean').plot.barh(x='feature',y='mean',legend=False,title='Permutation importance (holdout Macro F1)')
    plt.tight_layout()
    plt.savefig(output_dir/'reports/figures/feature_importance.png', dpi=150)
    plt.close()
    print('FINAL_REFIT_START', len(X), flush=True)
    final = clone(search.best_estimator_)
    final.fit(X, y, model__categorical_feature=categorical_indices)
    (output_dir/'models').mkdir(exist_ok=True)
    model_path = output_dir/'models/pipeline_lightgbm.joblib'
    joblib.dump(final, model_path)
    restored = joblib.load(model_path)
    assert np.array_equal(final.predict(test.head(10)), restored.predict(test.head(10)))
    submission = create_submission(restored, test, sample, output_dir/'submissions/submission.csv')
    # Top-level copy is convenient for Kaggle's submission selector.
    submission.to_csv(output_dir/'submission.csv', index=False)
    result = {'holdout_macro_f1':heldout_score, 'cv_macro_f1':search.best_score_,
        'cv_std':float(search.cv_results_['std_test_score'][search.best_index_]),
        'best_params':search.best_params_, 'train_rows':len(train), 'test_rows':len(test),
        'tuning_rows':len(Xsmall), 'holdout_rows':len(Xhold),
        'prediction_counts':submission.damage_grade.value_counts().sort_index().to_dict(),
        'public_score':None, 'private_score':None,
        'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,
            'lightgbm':lightgbm.__version__,'pandas':pd.__version__,'numpy':np.__version__}}
    (output_dir/'reports/results.json').write_text(json.dumps(result, indent=2))
    print('COMPLETE', json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--quick', action='store_true', help='Smoke test only; not a competition result.')
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.quick)
