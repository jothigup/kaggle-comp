"""Build account and subscription features for the 2014-15 competition.

Requires: pip install pandas
Run: python engineer_features.py --data-dir data/raw --as-of 2014
Use the historical reference year for --as-of, not today's year (default: 2014).

The reference year computes approximate donor tenure only. It does NOT filter donation totals
or purchases within a season: confirm the supplied snapshots were available
at prediction time. Subscription seasons starting in 2014 or later are excluded.
Outputs: features_train.csv (includes label), features_test.csv (includes ID),
feature_quality.csv. No imputation, category encoding, or model fitting occurs.
"""
from pathlib import Path
import argparse
import warnings
import pandas as pd

TARGET_START_YEAR = 2014


def read_csv(path):
    types = {'account.id': 'string', 'ID': 'string',
             'billing.zip.code': 'string', 'shipping.zip.code': 'string'}
    try:
        return pd.read_csv(path, dtype=types, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=types, encoding='cp1252')


def clean_ids(df, column, unique=False):
    df = df.copy()
    df[column] = df[column].astype('string').str.strip().replace('', pd.NA)
    if df[column].isna().any():
        raise ValueError(f'Missing {column} identifiers.')
    if unique and df[column].duplicated().any():
        raise ValueError(f'Duplicate {column} identifiers; review before joining.')
    return df


def account_features(accounts, as_of):
    """Accept raw or cleaned accounts. Keep ID, city, and donation predictors."""
    a = clean_ids(accounts, 'account.id', unique=True)
    amount_cols = ['amount.donated.2013', 'amount.donated.lifetime']
    count_col = 'no.donations.lifetime'
    for col in amount_cols + [count_col]:
        a[col] = pd.to_numeric(a[col], errors='raise')
        if a[col].lt(0).any():
            raise ValueError(f'Negative {col}; review before engineering donation features.')
    if a[count_col].dropna().mod(1).ne(0).any():
        raise ValueError('Fractional donation counts need review.')
    a['billing.city'] = (a['billing.city'].astype('string').str.strip()
                         .str.lower().str.replace(r'\s+', ' ', regex=True).replace('', pd.NA))
    dates = pd.to_datetime(a['first.donated'], errors='raise')
    try:
        reference_year = int(str(as_of))
    except (ValueError, TypeError):
        raise ValueError('Supply a four-digit reference year, such as 2014.') from None
    if len(str(reference_year)) != 4 or not 1000 <= reference_year <= 9999:
        raise ValueError('Supply a four-digit reference year, such as 2014.')
    tenure = reference_year - dates.dt.year
    future_dates = dates.dt.year.gt(reference_year)
    if future_dates.any():
        warnings.warn(f'{int(future_dates.sum())} first-donation years follow the reference year; '
                      'tenure is missing for these accounts. Check snapshot timing.')
    features = a[['account.id', 'billing.city'] + amount_cols + [count_col]].copy()
    features['has_donated'] = a[count_col].gt(0).astype('Int64').mask(a[count_col].isna())
    features['donated_in_2013'] = a[amount_cols[0]].gt(0).astype('Int64').mask(a[amount_cols[0]].isna())
    features['average_donation'] = a[amount_cols[1]] / a[count_col].where(a[count_col].gt(0))
    features['donor_tenure_years'] = tenure.where(a[count_col].gt(0) & ~future_dates)
    return features


def subscription_features(subscriptions):
    """Summarize one row per account-season; raise on ambiguous duplicate keys."""
    s = clean_ids(subscriptions, 'account.id')
    s['season'] = s['season'].astype('string').str.strip()
    parts = s['season'].str.extract(r'^(\d{4})-(\d{4})$')
    start = pd.to_numeric(parts[0], errors='coerce')
    end = pd.to_numeric(parts[1], errors='coerce')
    if (start.isna() | end.ne(start + 1)).any():
        raise ValueError('Invalid subscription season; expected YYYY-YYYY with consecutive years.')
    s['_year'] = start.astype(int)
    excluded = int(s['_year'].ge(TARGET_START_YEAR).sum())
    if excluded:
        warnings.warn(f'Excluded {excluded} subscription records from target/later seasons.')
    s = s.loc[s['_year'].lt(TARGET_START_YEAR)].copy()
    if s.duplicated(['account.id', '_year']).any():
        raise ValueError('Multiple subscription records per account-season; reconcile before summarizing.')
    for col in ['package', 'location']:
        s[col] = (s[col].astype('string').str.strip()
                  .str.replace(r'\s+', ' ', regex=True).replace('', pd.NA))
    s['no.seats'] = pd.to_numeric(s['no.seats'], errors='raise')
    if s['no.seats'].dropna().le(0).any() or s['no.seats'].dropna().mod(1).ne(0).any():
        raise ValueError('Nonpositive or fractional seats need review.')
    # Price level is categorical: its code is not a monetary amount.
    price = pd.to_numeric(s['price.level'], errors='raise')
    s['price.level'] = price.map(lambda x: pd.NA if pd.isna(x) else format(x, 'g')).astype('string')
    rows = []
    for account_id, group in s.groupby('account.id', sort=False):
        group = group.sort_values('_year')
        years = group['_year'].tolist()
        year_set = set(years)
        recent = 0
        y = TARGET_START_YEAR - 1
        while y in year_set:
            recent += 1
            y -= 1
        longest = run = 0
        previous = None
        for y in years:
            run = run + 1 if previous is not None and y == previous + 1 else 1
            longest = max(longest, run)
            previous = y
        # Select the actual latest row, even when an attribute is missing.
        latest = group.iloc[-1]
        row = {'account.id': account_id, 'has_subscription_history': 1,
               'total_subscribed_seasons': len(years),
               'seasons_since_last_subscription': TARGET_START_YEAR - 1 - years[-1],
               'recent_subscription_streak': recent,
               'longest_subscription_streak': longest,
               'latest_subscription_seats': latest['no.seats'],
               'latest_package': latest['package'], 'latest_location': latest['location'],
               'latest_price_level': latest['price.level']}
        for y in [2011, 2012, 2013]:
            row[f'subscribed_{y}_{str(y + 1)[-2:]}'] = int(y in year_set)
        rows.append(row)
    columns = ['account.id', 'has_subscription_history', 'total_subscribed_seasons',
               'seasons_since_last_subscription', 'recent_subscription_streak',
               'longest_subscription_streak', 'latest_subscription_seats',
               'latest_package', 'latest_location', 'latest_price_level',
               'subscribed_2011_12', 'subscribed_2012_13', 'subscribed_2013_14']
    result = pd.DataFrame(rows, columns=columns)
    result['account.id'] = result['account.id'].astype('string')
    return result


def build_features(account_df, subscriptions_df, train_df, test_df, *, as_of=2014):
    """Return train and test feature tables in their original split order.

    Pass a historical reference year as as_of (default 2014). Input DataFrames are unchanged.
    Donor tenure is a calendar-year difference, not exact elapsed time.
    IDs and label must be excluded from model inputs. Encode categorical
    features and fit imputation only after splitting training for validation.
    """
    train = clean_ids(train_df[['account.id', 'label']], 'account.id', unique=True)
    test = clean_ids(test_df[['ID']], 'ID', unique=True)
    if train['label'].isna().any() or not train['label'].isin([0, 1]).all():
        raise ValueError('Training labels must be 0 or 1 and nonmissing.')
    if set(train['account.id']) & set(test['ID']):
        raise ValueError('Train and test IDs overlap.')
    wanted = set(train['account.id']) | set(test['ID'])
    a = account_df.loc[account_df['account.id'].astype('string').str.strip().isin(wanted)].copy()
    s = subscriptions_df.loc[subscriptions_df['account.id'].astype('string').str.strip().isin(wanted)].copy()
    missing = wanted - set(a['account.id'].astype('string').str.strip())
    if missing:
        raise ValueError(f'{len(missing)} competition IDs missing from account data.')
    features = account_features(a, as_of).merge(subscription_features(s), on='account.id',
                                               how='left', validate='one_to_one')
    zero_cols = ['has_subscription_history', 'total_subscribed_seasons',
                 'recent_subscription_streak', 'longest_subscription_streak',
                 'subscribed_2011_12', 'subscribed_2012_13', 'subscribed_2013_14']
    features[zero_cols] = features[zero_cols].fillna(0).astype('Int64')
    for col in ['billing.city', 'latest_package', 'latest_location', 'latest_price_level']:
        features[col] = features[col].astype('string')
    train_features = train.merge(features, on='account.id', how='left', sort=False, validate='one_to_one')
    test_features = test.merge(features.rename(columns={'account.id': 'ID'}), on='ID',
                               how='left', sort=False, validate='one_to_one')
    assert train_features['account.id'].equals(train['account.id'])
    assert test_features['ID'].equals(test['ID'])
    return train_features, test_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/raw')
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--as-of', type=int, default=2014, help='Donor-tenure reference year (default: 2014)')
    args = parser.parse_args()
    folder = Path(args.data_dir)
    train_features, test_features = build_features(
        read_csv(folder / 'account.csv'), read_csv(folder / 'subscriptions.csv'),
        read_csv(folder / 'train.csv'), read_csv(folder / 'test.csv'), as_of=args.as_of)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_features.to_csv(out / 'features_train.csv', index=False)
    test_features.to_csv(out / 'features_test.csv', index=False)
    profiles = []
    for split, frame in [('train', train_features), ('test', test_features)]:
        profile = pd.DataFrame({'split': split, 'column': frame.columns,
                                'dtype': frame.dtypes.astype(str).values,
                                'missing': frame.isna().sum().values,
                                'missing_percent': (100 * frame.isna().mean()).values})
        profiles.append(profile)
    pd.concat(profiles, ignore_index=True).to_csv(out / 'feature_quality.csv', index=False)
    print(f'Train: {train_features.shape}; test: {test_features.shape}')
    print(f'Wrote feature tables and missing-value report to {out.resolve()}')
    print('Confirm snapshot timing. Remove account.id / ID and label from model predictors.')

    print(train_features.head())
    print(test_features.head())