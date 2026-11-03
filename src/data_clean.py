"""Load account.csv into a pandas DataFrame without changing the source file.

Install pandas: pip install pandas
Run from the project root: python data_clean.py
Defaults: data/raw/account.csv, subscriptions.csv, train.csv, test.csv.
Override paths with --file, --subscriptions-file, --train-file, --test-file.
Paths are relative to the folder where you run Python.
"""
from pathlib import Path
import argparse
import pandas as pd


def load_accounts(file_path="data/raw/account.csv"):
    """Preserve identifiers and ZIP codes as text, including leading zeros."""
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path.resolve()}. "
            "Pass the correct path with --file."
        )

    text_columns = {
        "account.id": "string",
        "shipping.zip.code": "string",
        "billing.zip.code": "string",
    }
    try:
        return pd.read_csv(path, dtype=text_columns, encoding="utf-8")
    except UnicodeDecodeError:
        # The supplied account.csv requires a non-UTF-8 encoding.
        return pd.read_csv(path, dtype=text_columns, encoding="cp1252")

def data_clean(df):
    """cleaning the account df"""
    account_df = df.copy()
    cols_drop = ['shipping.zip.code', 'billing.zip.code', 'shipping.city', 'relationship']
    account_df = account_df.drop(columns=cols_drop)

    return account_df



def load_subscriptions(file_path="data/raw/subscriptions.csv"):
    """Load subscription history; preserve account identifiers as text."""
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Could not find {path.resolve()}")
    try:
        return pd.read_csv(path, dtype={"account.id": "string"}, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype={"account.id": "string"}, encoding="cp1252")


def clean_subscriptions(df):
    """Standardize types and text. Preserve rows, missing values, and code values.

    Numeric price/tier codes are retained as supplied; this does not imply
    that their numerical order or spacing has a meaningful interpretation.
    Suspicious values are reported by eda_subscriptions rather than removed.
    """
    subscriptions_df = df.copy()
    text_cols = ["account.id", "season", "package", "location", "section", "multiple.subs"]
    numeric_cols = ["no.seats", "price.level", "subscription_tier"]
    required = text_cols + numeric_cols
    missing = set(required) - set(subscriptions_df.columns)
    if missing:
        raise ValueError(f"Missing subscription columns: {sorted(missing)}")

    for col in text_cols:
        values = subscriptions_df[col].astype("string").str.strip()
        # Preserve ID case; collapse whitespace only in descriptive fields.
        if col != "account.id":
            values = values.str.replace(r"\s+", " ", regex=True)
        subscriptions_df[col] = values.replace("", pd.NA)
    subscriptions_df["multiple.subs"] = subscriptions_df["multiple.subs"].str.lower()

    for col in numeric_cols:
        subscriptions_df[col] = pd.to_numeric(subscriptions_df[col], errors="raise")
    seats = subscriptions_df["no.seats"]
    if seats.dropna().mod(1).ne(0).any():
        raise ValueError("Fractional seat counts found; investigate before cleaning.")
    subscriptions_df["no.seats"] = seats.astype("Int64")
    return subscriptions_df


def eda_subscriptions(df, prediction_start_year=2014):
    """Print and return EDA tables. Run on raw and cleaned data to compare.

    A repeated account across seasons is expected. A repeated account AND
    season needs investigation; neither case is automatically deduplicated.
    A yes in multiple.subs is a source flag, not proof of duplicate rows.
    """
    ids = df["account.id"].astype("string").str.strip().replace("", pd.NA)
    season = df["season"].astype("string").str.strip()
    parts = season.str.extract(r"^(\d{4})-(\d{4})$")
    start = pd.to_numeric(parts[0], errors="coerce")
    end = pd.to_numeric(parts[1], errors="coerce")
    valid_season = start.notna() & end.eq(start + 1)
    repeated = df.duplicated(["account.id", "season"], keep=False)
    repeated_rows = df.loc[repeated].copy()
    if repeated.any():
        duplicate_groups = (repeated_rows.groupby(["account.id", "season"], dropna=False)
                            .nunique(dropna=False))
        conflicting_groups = int(duplicate_groups.gt(1).any(axis=1).sum())
    else:
        conflicting_groups = 0

    flags = pd.DataFrame(index=df.index)
    flags["missing_account_id"] = ids.isna()
    flags["invalid_or_missing_season"] = ~valid_season
    flags["season_at_or_after_target"] = start.ge(prediction_start_year)
    flags["repeated_account_season"] = repeated
    flags["missing_seat_count"] = df["no.seats"].isna()
    flags["nonpositive_seat_count"] = df["no.seats"].le(0)
    for col in ["price.level", "subscription_tier"]:
        flags[f"negative_{col}"] = df[col].lt(0)
    multiple = df["multiple.subs"].astype("string").str.strip().str.lower()
    flags["unexpected_multiple_subs"] = ~multiple.isin(["yes", "no"]) & multiple.notna()
    flags = flags.fillna(False)

    overview = pd.Series({
        "rows": len(df), "columns": len(df.columns),
        "unique_accounts": ids.nunique(),
        "accounts_in_multiple_rows": ids[ids.duplicated(keep=False)].nunique(),
        "exact_duplicate_extra_rows": int(df.duplicated().sum()),
        "rows_in_repeated_account_seasons": int(repeated.sum()),
        "repeated_account_season_groups": len(repeated_rows[["account.id", "season"]].drop_duplicates()),
        "conflicting_account_season_groups": conflicting_groups,
    }, name="value")
    profile = pd.DataFrame({"dtype": df.dtypes.astype(str),
                            "missing": df.isna().sum(),
                            "missing_percent": (df.isna().mean() * 100).round(2),
                            "unique_nonmissing": df.nunique()})
    numeric = df[["no.seats", "price.level", "subscription_tier"]].describe().T
    by_season = (df.groupby("season", dropna=False)
                 .agg(records=("account.id", "size"),
                      accounts=("account.id", "nunique"),
                      missing_seats=("no.seats", lambda s: int(s.isna().sum())),
                      seats=("no.seats", lambda s: s.sum(min_count=1)))
                 .sort_index())
    frequencies = {col: df[col].value_counts(dropna=False)
                   for col in ["package", "location", "section", "price.level",
                               "subscription_tier", "multiple.subs"]}
    print("\nSUBSCRIPTION OVERVIEW\n", overview.to_string())
    print("\nCOLUMN PROFILE\n", profile.to_string())
    print("\nNUMERIC SUMMARY (codes are not necessarily quantities)\n", numeric.to_string())
    print("\nACTIVITY BY SEASON\n", by_season.to_string())
    print("\nQUALITY FLAGS (review; no automatic removal)\n", flags.sum().to_string())
    for col, counts in frequencies.items():
        print(f"\n{col} frequencies\n{counts.to_string()}")
    return {"overview": overview, "column_profile": profile,
            "numeric_summary": numeric, "by_season": by_season,
            "category_counts": frequencies, "quality_flags": flags,
            "repeated_account_season_rows": repeated_rows}


def load_train_test(train_path="data/raw/train.csv", test_path="data/raw/test.csv"):
    """Load and validate competition splits, preserving row order and labels."""
    train_df = pd.read_csv(train_path, dtype={"account.id": "string"})
    test_df = pd.read_csv(test_path, dtype={"ID": "string"})
    validate_splits(train_df, test_df)
    return train_df, test_df


def validate_splits(train_df, test_df):
    """Check split identifiers without changing the supplied DataFrames."""
    if not {"account.id", "label"}.issubset(train_df.columns):
        raise ValueError("train.csv must contain account.id and label.")
    if "ID" not in test_df.columns:
        raise ValueError("test.csv must contain ID.")
    for name, values in [("train", train_df["account.id"]), ("test", test_df["ID"])]:
        if not pd.api.types.is_string_dtype(values.dtype):
            raise ValueError(f"Load {name} identifiers as strings.")
        text = values.astype("string")
        if text.fillna("").str.strip().eq("").any():
            raise ValueError(f"Missing or blank IDs in {name}.")
        if text.ne(text.str.strip()).any():
            raise ValueError(f"Surrounding whitespace in {name} IDs; review before matching.")
        if values.duplicated().any():
            raise ValueError(f"Duplicate IDs in {name}.")
    if set(train_df["account.id"]) & set(test_df["ID"]):
        raise ValueError("Train and test account IDs overlap.")
    if train_df["label"].isna().any() or not train_df["label"].isin([0, 1]).all():
        raise ValueError("Training labels must be nonmissing 0 or 1 values.")


def filter_model_accounts(account_df, subscriptions_df, train_df, test_df):
    """Accept DataFrames; retain all histories for the train/test account union.

    Returns filtered copies plus a unique-ID overlap report. Missing subscription
    history never removes an account. Training labels stay in train_df.
    """
    validate_splits(train_df, test_df)
    train_ids = set(train_df["account.id"])
    test_ids = set(test_df["ID"])
    keep_ids = train_ids | test_ids
    reports = []
    for name, frame in [("account", account_df), ("subscriptions", subscriptions_df)]:
        ids = set(frame["account.id"].dropna())
        reports.append({"table": name, "unique_ids": len(ids),
                        "in_train": len(ids & train_ids), "in_test": len(ids & test_ids),
                        "outside_train_test": len(ids - keep_ids),
                        "train_ids_without_record": len(train_ids - ids),
                        "test_ids_without_record": len(test_ids - ids)})
    overlap_report = pd.DataFrame(reports).set_index("table")
    accounts_filtered = account_df.loc[account_df["account.id"].isin(keep_ids)].copy()
    subscriptions_filtered = subscriptions_df.loc[subscriptions_df["account.id"].isin(keep_ids)].copy()
    print("\nUNIQUE ACCOUNT ID OVERLAP\n", overlap_report.to_string())
    print(f"\nAccount rows retained: {len(accounts_filtered):,}")
    print(f"Subscription history rows retained: {len(subscriptions_filtered):,}")
    return accounts_filtered, subscriptions_filtered, overlap_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, clean, and inspect competition data.")
    parser.add_argument("--file", default="data/raw/account.csv")
    parser.add_argument("--subscriptions-file", default="data/raw/subscriptions.csv")
    parser.add_argument("--train-file", default="data/raw/train.csv")
    parser.add_argument("--test-file", default="data/raw/test.csv")
    args = parser.parse_args()

    df = load_accounts(args.file)
    account_df = data_clean(df)
    subscriptions_raw = load_subscriptions(args.subscriptions_file)
    subscriptions_df = clean_subscriptions(subscriptions_raw)
    train_df, test_df = load_train_test(args.train_file, args.test_file)

    accounts_filtered, subscriptions_filtered, overlap_report = filter_model_accounts(
        account_df, subscriptions_df, train_df, test_df
    )
    # Run EDA on the histories relevant to the competition accounts.
    subscriptions_eda = eda_subscriptions(subscriptions_filtered)

    account_ids = set(account_df["account.id"].dropna())
    subscription_ids = set(subscriptions_df["account.id"].dropna())
    train_ids = set(train_df["account.id"].dropna())
    test_ids = set(test_df["ID"].dropna())

    print(
        f"Accounts shared by account and subscriptions: "
        f"{len(account_ids & subscription_ids):,}\n"
        f"Only in account: {len(account_ids - subscription_ids):,}\n"
        f"Only in subscriptions: {len(subscription_ids - account_ids):,}"
    )

    for name, ids in [
        ("Train", train_ids),
        ("Test", test_ids),
    ]:
        print(
            f"\n{name}: {len(ids):,} accounts\n"
            f"  In account: {len(ids & account_ids):,}\n"
            f"  With subscription history: {len(ids & subscription_ids):,}\n"
            f"  Without subscription history: {len(ids - subscription_ids):,}"
        )   