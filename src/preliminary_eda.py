"""Load the nine competition CSVs and run preliminary EDA.

Install: pip install pandas
Terminal: python preliminary_eda.py --data-dir ../data/raw --output-dir eda_output
Notebook: from preliminary_eda import load_data, run_eda
          dfs = load_data('../data/raw')
          run_eda(dfs, 'eda_output')
Paths supplied to these functions are relative to your current working directory.
No source data is modified, imputed, or deduplicated.
"""
from pathlib import Path
import argparse
import pandas as pd

FILES = ['account', 'concerts_2014-15', 'concerts', 'sample_submission',
         'subscriptions', 'test', 'tickets_all', 'train', 'zipcodes']
TEXT_COLUMNS = {'account.id', 'ID', 'shipping.zip.code', 'billing.zip.code', 'Zipcode'}


def load_data(data_dir):
    folder = Path(data_dir).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f'Data folder does not exist: {folder}. Python is running in {Path.cwd()}')
    dfs = {}
    for name in FILES:
        file = folder / f'{name}.csv'
        if not file.is_file():
            raise FileNotFoundError(f'Missing {file}. Put the nine extracted CSVs together in this folder.')
        encoding = 'utf-8'
        try:
            columns = pd.read_csv(file, nrows=0).columns
            df = pd.read_csv(file, dtype={c: 'string' for c in columns if c in TEXT_COLUMNS}, low_memory=False)
        except UnicodeDecodeError:
            encoding = 'cp1252'
            columns = pd.read_csv(file, nrows=0, encoding=encoding).columns
            df = pd.read_csv(file, encoding=encoding,
                             dtype={c: 'string' for c in columns if c in TEXT_COLUMNS}, low_memory=False)
        df.attrs['source_encoding'] = encoding
        dfs[name] = df
        print(f'{name}: {len(df):,} rows x {len(df.columns)} columns ({encoding})')
    return dfs


def run_eda(dfs, output_dir='eda_output'):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries, missing, keys, coverage = [], [], [], []
    for name, df in dfs.items():
        summaries.append({'table': name, 'rows': len(df), 'columns': len(df.columns),
                          'duplicate_rows': int(df.duplicated().sum()),
                          'encoding': df.attrs.get('source_encoding')})
        for col in df.columns:
            missing.append({'table': name, 'column': col, 'dtype': str(df[col].dtype),
                            'missing': int(df[col].isna().sum()),
                            'missing_percent': round(df[col].isna().mean() * 100, 2),
                            'unique_nonmissing': df[col].nunique()})
        numeric = df.select_dtypes('number')
        if not numeric.empty:
            numeric.describe().T.to_csv(out / f'{name}_numeric_summary.csv')
        # Top values for genuinely categorical columns; exclude identifiers and long free text.
        categorical = []
        for col in df.select_dtypes(include=['object', 'string', 'bool']):
            if col not in TEXT_COLUMNS and df[col].nunique() <= 100:
                counts = df[col].value_counts(dropna=False).head(10)
                for value, count in counts.items():
                    categorical.append({'column': col, 'value': str(value), 'count': count})
        if categorical:
            pd.DataFrame(categorical).to_csv(out / f'{name}_top_categories.csv', index=False)
        for col in ['account.id', 'ID', 'Zipcode']:
            if col in df:
                keys.append({'table': name, 'key': col, 'missing': int(df[col].isna().sum()),
                             'unique': df[col].nunique(),
                             'duplicate_nonmissing_keys': int(df[col].dropna().duplicated().sum())})
        if 'season' in df:
            df.groupby('season', dropna=False).size().rename('rows').to_csv(out / f'{name}_by_season.csv')

    summary = pd.DataFrame(summaries)
    missing_df = pd.DataFrame(missing)
    summary.to_csv(out / 'table_overview.csv', index=False)
    missing_df.to_csv(out / 'column_profile.csv', index=False)
    pd.DataFrame(keys).to_csv(out / 'key_checks.csv', index=False)
    train, test = dfs['train'], dfs['test']
    counts = train['label'].value_counts().sort_index()
    pd.DataFrame({'count': counts, 'percent': counts / len(train) * 100}).to_csv(out / 'label_distribution.csv')
    train_ids, test_ids = set(train['account.id']), set(test['ID'])
    for split, ids in [('train', train_ids), ('test', test_ids)]:
        for table in ['account', 'subscriptions', 'tickets_all']:
            other = set(dfs[table]['account.id'].dropna())
            coverage.append({'split': split, 'table': table, 'unique_split_ids': len(ids),
                             'matched_ids': len(ids & other), 'unmatched_ids': len(ids - other)})
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(out / 'account_coverage.csv', index=False)
    # Explore a date without changing the raw table.
    dates = pd.to_datetime(dfs['account']['first.donated'], errors='coerce')
    invalid_dates = int((dfs['account']['first.donated'].notna() & dates.isna()).sum())
    high_missing = missing_df[missing_df.missing_percent >= 30].sort_values('missing_percent', ascending=False)
    report = [
        'PRELIMINARY EDA', '', 'TABLE SIZES', summary.to_string(index=False), '',
        'TRAINING LABELS', counts.to_string(),
        f'Label 1 prevalence: {train["label"].eq(1).mean():.2%}',
        f'Train/test ID overlap: {len(train_ids & test_ids)}',
        f'Sample submission IDs match test IDs in order: {dfs["sample_submission"]["ID"].equals(test["ID"])}',
        '', 'COLUMNS WITH AT LEAST 30% MISSING',
        high_missing[['table', 'column', 'missing', 'missing_percent']].to_string(index=False),
        '', 'ACCOUNT COVERAGE', coverage_df.to_string(index=False),
        '', f'Unparseable nonmissing first.donated dates: {invalid_dates}',
        f'Parsed first.donated range: {dates.min()} to {dates.max()}', '',
        'INTERPRETATION AND NEXT STEPS',
        '* label 1 is rare. A classifier predicting only zero has high accuracy; assess precision, recall, PR-AUC, and the competition metric.',
        '* The meaning of label 1 and the prediction cutoff must be confirmed from competition instructions.',
        '* account.id connects accounts, training labels, subscriptions, and tickets. test uses ID instead.',
        '* Keep IDs and ZIP codes as text. ZIP normalization (including ZIP+4 handling) needs checking before geographic joins.',
        '* Repeated account IDs in subscription/ticket tables can represent legitimate history; aggregate to one row per account before joining to labels.',
        '* Validate account as unique before merging with validate="many_to_one". Use left joins to retain split accounts without history.',
        '* Concert set numbers repeat across seasons and venues. Check composite keys before any concert join to avoid multiplying rows.',
        '* concerts_2014-15 contains a different season and has no location column. Confirm whether this information is allowed for prediction.',
        '* Do not replace missing categories with zero by default. Review missingness and distinguish unknown from no activity.',
        '* Confirm the prediction date before deriving donation or purchase features to avoid future-information leakage.',
        '* Fit preprocessing on training folds only; stratify validation when appropriate and choose a split consistent with the task.',
        '* Text includes encoding artifacts; the cp1252 fallback loads account.csv but does not guarantee all text is clean.',
    ]
    report_text = '\n'.join(report)
    (out / 'eda_summary.txt').write_text(report_text, encoding='utf-8')
    print('\n' + report_text)
    return summary, missing_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/raw', help='Folder containing all nine extracted CSVs')
    parser.add_argument('--output-dir', default='eda_output')
    args = parser.parse_args()
    dataframes = load_data(args.data_dir)
    run_eda(dataframes, args.output_dir)
