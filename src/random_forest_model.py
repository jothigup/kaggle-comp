"""Compare random forest with L2 logistic regression, tuned for AUROC.

Install: pip install pandas numpy scikit-learn matplotlib
Run from project root: python src/random_forest_model.py
Notebook: results = run_forest_comparison(train_features, test_features)
Defaults: inputs data/processed; outputs model_outputs/random_forest_comparison.

The same seed-42 80/20 split as earlier scripts is used. Hyperparameters are
selected solely with identical five-fold stratified CV on the training 80%.
Validation has already been inspected in earlier experiments; it is a development
holdout, not an independent final test. Kaggle test rows are prediction-only.
Fold scores from the selected grid candidates are tuning scores, not unbiased
estimates of model generalization. Winner for submission is selected by CV,
never by validation. Imputation and encoding are fit inside each CV fold.
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


def run_forest_comparison(train_features, test_features,
                          output_dir='model_outputs/random_forest_comparison',
                          threshold=0.5, cv_folds=5, random_state=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    if not 0 < threshold < 1:
        raise ValueError('threshold must be between 0 and 1.')
    X, y, X_test, ids, test_ids, categorical, numeric = prepare_inputs(train_features, test_features)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state)
    if cv_folds < 2 or y_train.value_counts().min() < cv_folds:
        raise ValueError('Not enough minority-class samples for CV.')
    cv = list(StratifiedKFold(n_splits=cv_folds, shuffle=True,
                             random_state=random_state).split(X_train, y_train))
    # Retain the previous L2 preprocessing/solver for comparability.
    l2 = build_pipeline(categorical, numeric)
    version = tuple(int(v) for v in sklearn.__version__.split('.')[:2])
    kwargs = {'l1_ratio': 0.0} if version >= (1, 8) else {'penalty': 'l2'}
    l2.set_params(model=LogisticRegression(solver='liblinear', max_iter=5000,
                                          tol=1e-5, random_state=random_state, **kwargs))
    # Trees need no scaling. Keep every category rather than dropping a reference.
    forest_preprocess = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', add_indicator=True,
                                  keep_empty_features=True), numeric),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='__missing__',
                                     keep_empty_features=True)),
            ('encode', OneHotEncoder(handle_unknown='ignore', min_frequency=10)),
        ]), categorical),
    ])
    forest = Pipeline([
        ('preprocess', forest_preprocess),
        ('model', RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                         random_state=random_state, n_jobs=2)),
    ])
    grids = {
        'L2': (l2, {'model__C': [0.001, 0.01, 0.1, 1, 10, 100]}),
        'Random forest': (forest, {'model__max_depth': [5, 10, None],
                                  'model__min_samples_leaf': [2, 10],
                                  'model__class_weight': [None, 'balanced_subsample']}),
    }
    scoring = {'roc_auc': 'roc_auc', 'average_precision': 'average_precision',
               'neg_log_loss': 'neg_log_loss'}
    searches, cv_rows, val_rows, matrices, probabilities = {}, [], [], {}, {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, (pipeline, grid) in grids.items():
        print(f'Tuning {name} with training-only CV...', flush=True)
        search = GridSearchCV(pipeline, grid, scoring=scoring, refit='roc_auc',
                              cv=cv, return_train_score=True, error_score='raise', n_jobs=1)
        search.fit(X_train, y_train)
        searches[name] = search
        row = pd.DataFrame(search.cv_results_).iloc[search.best_index_]
        cv_rows.append({'model': name, 'cv_auroc': row['mean_test_roc_auc'],
                        'cv_auroc_std': row['std_test_roc_auc'],
                        'training_auroc': row['mean_train_roc_auc'],
                        'cv_average_precision': row['mean_test_average_precision'],
                        'cv_log_loss': -row['mean_test_neg_log_loss'],
                        'best_parameters': json.dumps(search.best_params_)})
        safe_name = 'l2' if name == 'L2' else 'forest'
        pd.DataFrame(search.cv_results_).to_csv(out / f'{safe_name}_cv_results.csv', index=False)
    # Select the deployment candidate before calculating any validation metrics.
    winner = max(searches, key=lambda name: searches[name].best_score_)
    for name, search in searches.items():
        prob = search.best_estimator_.predict_proba(X_val)[:, 1]
        probabilities[name] = prob
        val_rows.append({'model': name, **probability_metrics(y_val, prob, threshold)})
        matrices[name] = confusion_matrix(y_val, (prob >= threshold).astype(int), labels=[0, 1])
    cv_comparison = pd.DataFrame(cv_rows)
    validation = pd.DataFrame(val_rows)
    cv_comparison.to_csv(out / 'cv_comparison.csv', index=False)
    validation.to_csv(out / 'validation_metrics.csv', index=False)
    pd.DataFrame({'account.id': ids.loc[X_val.index], 'label': y_val,
                  'l2_probability': probabilities['L2'],
                  'forest_probability': probabilities['Random forest']}).to_csv(
                      out / 'validation_predictions.csv', index=False)
    pd.DataFrame({'account.id': ids, 'split': np.where(X.index.isin(X_val.index),
                  'validation', 'train_cv')}).to_csv(out / 'split_assignments.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, cm) in zip(axes, matrices.items()):
        ConfusionMatrixDisplay(cm, display_labels=['No', 'Yes']).plot(
            ax=ax, colorbar=False, values_format='d')
        ax.set_title(f'{name}: threshold {threshold:g}')
        pd.DataFrame(cm, index=['actual_0', 'actual_1'], columns=['predicted_0', 'predicted_1']).to_csv(
            out / ('l2_confusion_matrix.csv' if name == 'L2' else 'forest_confusion_matrix.csv'))
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrices.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, prob in probabilities.items():
        RocCurveDisplay.from_predictions(y_val, prob, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title('Development validation ROC curves')
    fig.tight_layout()
    fig.savefig(out / 'roc_comparison.png', dpi=180)
    plt.close(fig)
    # Refit BOTH selected models on all labeled accounts. Never fit on Kaggle test.
    final_models, submissions = {}, {}
    for name, search in searches.items():
        final = clone(search.best_estimator_).fit(X, y)
        final_models[name] = final
        submission = pd.DataFrame({'ID': test_ids, 'Predicted': final.predict_proba(X_test)[:, 1]})
        submissions[name] = submission
        filename = 'submission_l2.csv' if name == 'L2' else 'submission_forest.csv'
        submission.to_csv(out / filename, index=False)
    submissions[winner].to_csv(out / 'submission_cv_winner.csv', index=False)
    delta = (validation.set_index('model').loc['Random forest', 'roc_auc'] -
             validation.set_index('model').loc['L2', 'roc_auc'])
    metadata = {'cv_winner': winner, 'forest_minus_l2_validation_auroc': float(delta),
                'training_rows': len(X_train), 'validation_rows': len(X_val), 'test_rows': len(X_test),
                'random_state': random_state, 'cv_folds': cv_folds,
                'sklearn_version': sklearn.__version__,
                'best_parameters': {name: search.best_params_ for name, search in searches.items()},
                'note': 'Validation is a reused development holdout; small differences do not establish superiority.'}
    (out / 'comparison_summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\nCV COMPARISON\n', cv_comparison.to_string(index=False))
    print('\nVALIDATION\n', validation.to_string(index=False))
    print(f'\nCV winner: {winner}. Forest minus L2 validation AUROC: {delta:+.5f}')
    print(f'Outputs: {out.resolve()}')
    return {'searches': searches, 'final_models': final_models, 'cv_comparison': cv_comparison,
            'validation_metrics': validation, 'confusion_matrices': matrices,
            'submissions': submissions, 'summary': metadata}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', default='data/processed')
    parser.add_argument('--output-dir', default='model_outputs/random_forest_comparison')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    folder = Path(args.features_dir)
    types = {'account.id': 'string', 'ID': 'string', **{c: 'string' for c in CATEGORY_COLUMNS}}
    train_features = pd.read_csv(folder / 'features_train.csv', dtype=types)
    test_features = pd.read_csv(folder / 'features_test.csv', dtype=types)
    results = run_forest_comparison(train_features, test_features,
                                    output_dir=args.output_dir, threshold=args.threshold)
