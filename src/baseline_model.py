"""Unpenalized logistic-regression baseline for subscription prediction.

Install: pip install pandas numpy scikit-learn matplotlib
Run from the project root: python src/baseline_model.py
Default inputs: data/processed/features_train.csv and features_test.csv
Default outputs: model_outputs/baseline/

Notebook:
    from baseline_model import run_baseline
    results = run_baseline(train_features, test_features)

20% of labeled rows are held out for validation. Five-fold stratified CV runs
ONLY on the remaining 80%; imputation, scaling and encoding are fit inside
folds. The Kaggle test set is never used for evaluation or preprocessing fit.
After evaluation, a fresh pipeline is refit on all labeled rows for submission.

The confusion matrix evaluates predictions, not individual features. Coefficients
and validation permutation importance are exploratory, not causal evidence or
proof a feature should be removed. Compare L1/L2 models with identical CV folds
later; avoid repeatedly tuning against the held-out validation set.
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


def run_baseline(train_features, test_features, output_dir='model_outputs/baseline',
                 validation_size=0.2, cv_folds=5, threshold=0.5, random_state=42):
    """Train/evaluate baseline, export reports and probabilities, return results.

    threshold affects the confusion matrix only, never Kaggle probabilities.
    Validation feature importance is exploratory; tune models with training CV.
    """
    if not 0 < threshold < 1:
        raise ValueError('threshold must lie strictly between 0 and 1.')
    X, y, X_test, train_ids, test_ids, categorical, numeric = prepare_inputs(train_features, test_features)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_size, stratify=y, random_state=random_state)
    if y_train.value_counts().min() < cv_folds or cv_folds < 2:
        raise ValueError('Not enough minority-class training samples for the requested CV folds.')
    pipeline = build_pipeline(categorical, numeric)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring={'roc_auc': 'roc_auc', 'average_precision': 'average_precision',
                 'neg_log_loss': 'neg_log_loss'},
        return_train_score=True, error_score='raise', n_jobs=1)
    cv_rows = []
    for split in ['train', 'test']:
        for metric in ['roc_auc', 'average_precision', 'neg_log_loss']:
            values = scores[f'{split}_{metric}']
            if metric == 'neg_log_loss':
                values = -values
            cv_rows.append({'split': 'training_fold' if split == 'train' else 'held_out_fold',
                            'metric': metric.replace('neg_', ''),
                            'mean': float(values.mean()), 'std': float(values.std(ddof=1))})
    cv_summary = pd.DataFrame(cv_rows)
    pipeline.fit(X_train, y_train)
    val_probability = pipeline.predict_proba(X_val)[:, 1]
    val_prediction = (val_probability >= threshold).astype(int)
    baseline_probability = np.full(len(y_val), y_train.mean())
    metrics = {'model': probability_metrics(y_val, val_probability, threshold),
               'constant_probability_baseline': probability_metrics(y_val, baseline_probability, threshold)}
    cm = confusion_matrix(y_val, val_prediction, labels=[0, 1])
    matrix = pd.DataFrame(cm, index=['actual_0', 'actual_1'], columns=['predicted_0', 'predicted_1'])
    feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()
    coefficients = pd.DataFrame({'encoded_feature': feature_names,
                                 'coefficient': pipeline.named_steps['model'].coef_[0]})
    coefficients['absolute_coefficient'] = coefficients.coefficient.abs()
    coefficients = coefficients.sort_values('absolute_coefficient', ascending=False)
    # Permute original columns, so each categorical feature is evaluated together.
    importance = permutation_importance(pipeline, X_val, y_val, scoring='average_precision',
                                        n_repeats=5, random_state=random_state, n_jobs=1)
    importances = pd.DataFrame({'feature': X.columns,
                               'mean_average_precision_drop': importance.importances_mean,
                               'std': importance.importances_std}).sort_values(
                                   'mean_average_precision_drop', ascending=False)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv_summary.to_csv(out / 'cross_validation_summary.csv', index=False)
    matrix.to_csv(out / 'validation_confusion_matrix.csv')
    coefficients.to_csv(out / 'validation_model_coefficients.csv', index=False)
    importances.to_csv(out / 'validation_permutation_importance.csv', index=False)
    pd.DataFrame({'account.id': train_ids.loc[X_val.index], 'label': y_val,
                  'probability': val_probability, 'prediction': val_prediction}).to_csv(
                      out / 'validation_predictions.csv', index=False)
    pd.DataFrame({'account.id': train_ids, 'split': np.where(X.index.isin(X_val.index),
                   'validation', 'train_cv')}).to_csv(out / 'split_assignments.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['No subscription', 'Subscription']).plot(
        ax=ax, colorbar=False, values_format='d')
    ax.set_title(f'Validation: logistic regression (threshold {threshold:g})')
    fig.tight_layout()
    fig.savefig(out / 'validation_confusion_matrix.png', dpi=180)
    plt.close(fig)
    final_model = clone(pipeline).fit(X, y)
    submission = pd.DataFrame({'ID': test_ids, 'Predicted': final_model.predict_proba(X_test)[:, 1]})
    submission.to_csv(out / 'submission.csv', index=False)
    metadata = {'metrics': metrics, 'training_rows': len(X_train), 'validation_rows': len(X_val),
                'kaggle_test_rows': len(X_test), 'cv_folds': cv_folds, 'random_state': random_state,
                'sklearn_version': sklearn.__version__, 'regularization': 'none',
                'numeric_features': numeric, 'categorical_features': categorical,
                'notes': ['Average precision is reported for class imbalance; verify the Kaggle scoring metric.',
                          'Correlated/redundant features can make unpenalized coefficients unstable.',
                          'A confusion matrix cannot determine whether L1 or L2 is needed.',
                          'Unknown categories are encoded as all zeros; rare training categories are grouped.',
                          'Donor-tenure reference year and snapshot timing must be checked before modeling.']}
    (out / 'evaluation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\nCROSS-VALIDATION\n', cv_summary.to_string(index=False))
    print('\nVALIDATION METRICS\n', json.dumps(metrics, indent=2))
    print('\nCONFUSION MATRIX (rows actual, columns predicted)\n', matrix)
    print('\nTOP PERMUTATION IMPORTANCES\n', importances.head(10).to_string(index=False))
    print(f'\nSaved outputs to {out.resolve()}')
    return {'validation_model': pipeline, 'final_model': final_model, 'metrics': metrics,
            'cv_summary': cv_summary, 'confusion_matrix': matrix,
            'coefficients': coefficients, 'permutation_importance': importances,
            'submission': submission}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', default='data/processed')
    parser.add_argument('--output-dir', default='model_outputs/baseline')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    folder = Path(args.features_dir)
    dtypes = {'account.id': 'string', 'ID': 'string', **{c: 'string' for c in CATEGORY_COLUMNS}}
    train_features = pd.read_csv(folder / 'features_train.csv', dtype=dtypes)

    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    # Numerical features and label; exclude account identifiers.
    numeric_df = (
        train_features
        .drop(columns=["account.id", "ID"], errors="ignore")
        .select_dtypes(include="number")
    )

    correlation = numeric_df.corr()

    output_dir = Path("model_outputs/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 12))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax,
    )

    ax.set_title("Training features and label: Pearson correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_matrix.png", dpi=200)
    correlation.to_csv(output_dir / "correlation_matrix.csv")

    plt.show()

    test_features = pd.read_csv(folder / 'features_test.csv', dtype=dtypes)
    results = run_baseline(train_features, test_features, 
                           output_dir=args.output_dir, threshold=args.threshold)
