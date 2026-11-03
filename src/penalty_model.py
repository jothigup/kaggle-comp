"""Model 2: compare L1 and L2 logistic regression, tune C, and test the winner.

Install: pip install pandas numpy scikit-learn matplotlib
Run from the project root: python src/penalty_model.py
Notebook: results = run_penalty_comparison(train_features, test_features)
Default inputs: data/processed/features_train.csv and features_test.csv
Default outputs: model_outputs/penalty_comparison/

The same 80/20 split and five training-only CV folds as baseline_model.py are
used (seed 42). Select penalty and C using CV only. Validation is a development
holdout already inspected in the baseline, not a new independent final test.
Kaggle test labels are unavailable; its rows are used only for predictions.

Default selection metric: negative log loss, appropriate for assessing supplied
probabilities. The actual Kaggle metric has not been confirmed. Override with
--selection-metric roc_auc or average_precision if the competition requires it.
All selection metrics are maximized; higher negative log loss means lower loss.

C is inverse regularization strength: smaller C means stronger regularization.
L1 encourages exact zero coefficients. L2 shrinks coefficients without generally
setting them to zero. liblinear is used for BOTH penalty families and penalizes
the synthetic intercept as well as feature coefficients. Do not interpret this
as a penalty-only controlled comparison with the lbfgs unpenalized baseline.
"""
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, log_loss, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORY_COLUMNS = ['billing.city', 'latest_package', 'latest_location', 'latest_price_level']


def prepare_inputs(train_features, test_features):
    """Normalize types only; do not learn preprocessing from either split."""
    train = train_features.copy()
    test = test_features.copy()
    if 'label' not in train or 'account.id' not in train:
        raise ValueError('train_features must include account.id and label.')
    if 'ID' not in test:
        if 'account.id' in test:
            test = test.rename(columns={'account.id': 'ID'})
        else:
            raise ValueError('test_features must include ID (or account.id).')
    if 'label' in test:
        raise ValueError('test_features must not include training labels.')
    for frame, key in [(train, 'account.id'), (test, 'ID')]:
        frame[key] = frame[key].astype('string').str.strip()
        if frame[key].fillna('').eq('').any() or frame[key].duplicated().any():
            raise ValueError(f'Missing, blank, or duplicate {key} identifiers.')
    if set(train['account.id']) & set(test['ID']):
        raise ValueError('Train and test IDs overlap.')
    y = pd.to_numeric(train['label'], errors='raise')
    if y.isna().any() or not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError('Training labels must contain both classes, 0 and 1, without missing values.')
    X = train.drop(columns=['account.id', 'label'])
    X_test = test.drop(columns='ID')
    if set(X.columns) != set(X_test.columns):
        raise ValueError('Train/test predictor columns differ. Rebuild both with the same feature script.')
    if X.empty:
        raise ValueError('No predictors supplied.')
    X_test = X_test.reindex(columns=X.columns)
    categorical = [c for c in X if c in CATEGORY_COLUMNS or
                   not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X if c not in categorical]
    for frame in [X, X_test]:
        for col in categorical:
            values = frame[col].astype('string')
            # Canonicalize price codes so 1 and 1.0 match across CSV / in-memory inputs.
            if col == 'latest_price_level':
                values = values.str.replace(r'^([0-9]+)\.0$', r'\1', regex=True)
            frame[col] = values.astype(object).where(values.notna(), np.nan)
        for col in numeric:
            frame[col] = pd.to_numeric(frame[col], errors='raise').astype(float)
            if np.isinf(frame[col]).any():
                raise ValueError(f'Infinite values in {col}; review feature engineering.')
    return X, y.astype(int), X_test, train['account.id'], test['ID'], categorical, numeric


def build_pipeline(categorical, numeric):
    numeric_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
        ('scale', StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='__missing__', keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='ignore', min_frequency=10, drop='first')),
    ])
    preprocessing = ColumnTransformer([
        ('numeric', numeric_pipe, numeric),
        ('categorical', categorical_pipe, categorical),
    ])
    version = tuple(int(v) for v in sklearn.__version__.split('.')[:2])
    # sklearn >=1.8 prefers C=inf; older versions specify penalty=None.
    kwargs = {'C': np.inf} if version >= (1, 8) else {'penalty': None}
    model = LogisticRegression(solver='lbfgs', max_iter=5000, tol=1e-5, **kwargs)
    return Pipeline([('preprocess', preprocessing), ('model', model)])


def probability_metrics(y, probability, threshold):
    prediction = (np.asarray(probability) >= threshold).astype(int)
    return {
        'roc_auc': float(roc_auc_score(y, probability)),
        'average_precision': float(average_precision_score(y, probability)),
        'log_loss': float(log_loss(y, probability, labels=[0, 1])),
        'brier_score': float(brier_score_loss(y, probability)),
        'accuracy': float(accuracy_score(y, prediction)),
        'precision': float(precision_score(y, prediction, zero_division=0)),
        'recall': float(recall_score(y, prediction, zero_division=0)),
        'f1': float(f1_score(y, prediction, zero_division=0)),
        'threshold': threshold,
    }


def run_penalty_comparison(train_features, test_features,
                           output_dir='model_outputs/penalty_comparison',
                           selection_metric='neg_log_loss', cv_folds=5,
                           validation_size=0.2, threshold=0.5, random_state=42):
    """Select L1/L2 and C by training CV; evaluate winner; refit for Kaggle."""
    from sklearn.model_selection import GridSearchCV
    if selection_metric not in ['neg_log_loss', 'roc_auc', 'average_precision']:
        raise ValueError('Choose neg_log_loss, roc_auc, or average_precision.')
    if not 0 < threshold < 1:
        raise ValueError('threshold must lie between 0 and 1.')
    X, y, X_test, ids, test_ids, categorical, numeric = prepare_inputs(train_features, test_features)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=validation_size, random_state=random_state)
    if cv_folds < 2 or y_train.value_counts().min() < cv_folds:
        raise ValueError('Insufficient minority-class examples for the requested CV folds.')
    cv = list(StratifiedKFold(n_splits=cv_folds, shuffle=True,
                             random_state=random_state).split(X_train, y_train))
    scoring = {'neg_log_loss': 'neg_log_loss', 'roc_auc': 'roc_auc',
               'average_precision': 'average_precision'}
    baseline = build_pipeline(categorical, numeric)
    baseline_cv = cross_validate(baseline, X_train, y_train, cv=cv, scoring=scoring,
                                 return_train_score=True, error_score='raise', n_jobs=1)
    version = tuple(int(v) for v in sklearn.__version__.split('.')[:2])
    parameter = 'l1_ratio' if version >= (1, 8) else 'penalty'
    values = [1.0, 0.0] if version >= (1, 8) else ['l1', 'l2']
    regularized = clone(baseline)
    # Use one solver for a fairer comparison between L1 and L2 candidates.
    regularized.set_params(model=LogisticRegression(solver='liblinear', max_iter=5000,
                                                    tol=1e-5, random_state=random_state))
    grid = [{f'model__{parameter}': [v], 'model__C': [0.001, 0.01, 0.1, 1, 10, 100]}
            for v in values]
    search = GridSearchCV(regularized, grid, scoring=scoring, refit=selection_metric,
                          cv=cv, return_train_score=True, error_score='raise', n_jobs=1)
    search.fit(X_train, y_train)
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results['penalty'] = cv_results[f'param_model__{parameter}'].map(
        {1.0: 'L1', 0.0: 'L2'} if parameter == 'l1_ratio' else {'l1': 'L1', 'l2': 'L2'})
    cv_results['C'] = cv_results['param_model__C'].astype(float)
    cv_results['mean_validation_log_loss'] = -cv_results['mean_test_neg_log_loss']
    cv_results['mean_training_log_loss'] = -cv_results['mean_train_neg_log_loss']
    cv_results = cv_results.sort_values(f'rank_test_{selection_metric}')
    family_winners = cv_results.groupby('penalty', sort=False).head(1)
    selected = search.best_estimator_
    winner_row = cv_results.loc[search.best_index_]
    selected_penalty = winner_row['penalty']
    comparison = []
    for _, row in family_winners.iterrows():
        comparison.append({'model': row['penalty'], 'C': row['C'],
                           'cv_log_loss': -row['mean_test_neg_log_loss'],
                           'cv_roc_auc': row['mean_test_roc_auc'],
                           'cv_average_precision': row['mean_test_average_precision'],
                           'selection_score': row[f'mean_test_{selection_metric}'],
                           'selection_score_std': row[f'std_test_{selection_metric}']})
    comparison.append({'model': 'Unpenalized baseline', 'C': None,
                       'cv_log_loss': -baseline_cv['test_neg_log_loss'].mean(),
                       'cv_roc_auc': baseline_cv['test_roc_auc'].mean(),
                       'cv_average_precision': baseline_cv['test_average_precision'].mean(),
                       'selection_score': baseline_cv[f'test_{selection_metric}'].mean(),
                       'selection_score_std': baseline_cv[f'test_{selection_metric}'].std(ddof=0)})
    comparison = pd.DataFrame(comparison)
    # Selection is finished before accessing validation outcomes for evaluation.
    baseline.fit(X_train, y_train)
    val_prob = selected.predict_proba(X_val)[:, 1]
    baseline_prob = baseline.predict_proba(X_val)[:, 1]
    val_metrics = pd.DataFrame([
        {'model': f'Selected {selected_penalty}', **probability_metrics(y_val, val_prob, threshold)},
        {'model': 'Unpenalized baseline', **probability_metrics(y_val, baseline_prob, threshold)}])
    cm = confusion_matrix(y_val, (val_prob >= threshold).astype(int), labels=[0, 1])
    confusion = pd.DataFrame(cm, index=['actual_0', 'actual_1'], columns=['predicted_0', 'predicted_1'])
    coefficients = pd.DataFrame({
        'encoded_feature': selected.named_steps['preprocess'].get_feature_names_out(),
        'coefficient': selected.named_steps['model'].coef_[0]})
    coefficients['absolute_coefficient'] = coefficients['coefficient'].abs()
    coefficients['is_zero'] = coefficients['coefficient'].eq(0)
    coefficients = coefficients.sort_values('absolute_coefficient', ascending=False)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(out / 'all_cv_results.csv', index=False)
    comparison.to_csv(out / 'penalty_comparison.csv', index=False)
    val_metrics.to_csv(out / 'validation_metrics.csv', index=False)
    confusion.to_csv(out / 'validation_confusion_matrix.csv')
    coefficients.to_csv(out / 'selected_model_coefficients.csv', index=False)
    pd.DataFrame({'account.id': ids.loc[X_val.index], 'label': y_val,
                  'probability': val_prob, 'prediction': (val_prob >= threshold).astype(int)}).to_csv(
                      out / 'validation_predictions.csv', index=False)
    pd.DataFrame({'account.id': ids, 'split': np.where(X.index.isin(X_val.index),
                  'validation', 'train_cv')}).to_csv(out / 'split_assignments.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=['No subscription', 'Subscription']).plot(
        ax=ax, colorbar=False, values_format='d')
    ax.set_title(f'Model 2: {selected_penalty}, C={search.best_params_["model__C"]:g}\nValidation threshold={threshold:g}')
    fig.tight_layout()
    fig.savefig(out / 'validation_confusion_matrix.png', dpi=180)
    plt.close(fig)
    final_model = clone(selected).fit(X, y)
    submission = pd.DataFrame({'ID': test_ids, 'Predicted': final_model.predict_proba(X_test)[:, 1]})
    submission.to_csv(out / 'submission.csv', index=False)
    baseline_score = baseline_cv[f'test_{selection_metric}'].mean()
    metadata = {'selected_penalty': selected_penalty, 'selected_C': float(search.best_params_['model__C']),
                'selection_metric': selection_metric, 'best_cv_score': float(search.best_score_),
                'unpenalized_cv_score': float(baseline_score),
                'selected_beats_baseline_in_cv': bool(search.best_score_ > baseline_score),
                'training_rows': len(X_train), 'validation_rows': len(X_val), 'test_rows': len(X_test),
                'zero_encoded_coefficients': int(coefficients.is_zero.sum()),
                'total_encoded_coefficients': len(coefficients),
                'random_state': random_state, 'sklearn_version': sklearn.__version__,
                'note': 'Winner chosen by training CV only. Holdout already used during development; score gaps are not proof of superiority.'}
    (out / 'selected_model.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\nCROSS-VALIDATION COMPARISON\n', comparison.to_string(index=False))
    print(f'\nSelected {selected_penalty}, C={metadata["selected_C"]:g}, using {selection_metric}.')
    if not metadata['selected_beats_baseline_in_cv']:
        print('The selected penalty did NOT beat the unpenalized baseline on the selection metric.')
    print('\nVALIDATION METRICS\n', val_metrics.to_string(index=False))
    print('\nCONFUSION MATRIX\n', confusion.to_string())
    print(f'\nSaved outputs to {out.resolve()}')
    return {'search': search, 'validation_model': selected, 'final_model': final_model,
            'cv_comparison': comparison, 'validation_metrics': val_metrics,
            'confusion_matrix': confusion, 'coefficients': coefficients,
            'submission': submission, 'selection': metadata}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', default='data/processed')
    parser.add_argument('--output-dir', default='model_outputs/penalty_comparison')
    parser.add_argument('--selection-metric', default='roc_auc',
                        choices=['neg_log_loss', 'roc_auc', 'average_precision'])
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    folder = Path(args.features_dir)
    types = {'account.id': 'string', 'ID': 'string', **{c: 'string' for c in CATEGORY_COLUMNS}}
    train_features = pd.read_csv(folder / 'features_train.csv', dtype=types)
    test_features = pd.read_csv(folder / 'features_test.csv', dtype=types)
    results = run_penalty_comparison(train_features, test_features,
        output_dir=args.output_dir, selection_metric=args.selection_metric, threshold=args.threshold)
