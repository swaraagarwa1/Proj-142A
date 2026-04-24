"""
LA crime: predict which *crime type* fits a time + location (classification).

We use sklearn LogisticRegression (from your course list). It assigns each row
to one of several crime-type labels and gives probabilities with predict_proba.

LinearRegression is for predicting a number (e.g. price), not a category, so we
do not use it here.

Feature engineering: lat/lon, LAPD area name, reporting district, premise
description, weapon description, four 6-hour time buckets, day of week, month,
year. Regularization: L2 via penalty and strength C on LogisticRegression.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

DEFAULT_DATA = (
    Path.home()
    / ".cache/kagglehub/datasets/cityofLA/crime-in-los-angeles/versions/7"
    / "Crime_Data_2010_2017.csv"
)


def parse_location(loc: pd.Series) -> tuple[pd.Series, pd.Series]:
    def _one(s) -> tuple[float, float]:
        if pd.isna(s):
            return (np.nan, np.nan)
        m = re.search(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", str(s))
        if m:
            return (float(m.group(1)), float(m.group(2)))
        return (np.nan, np.nan)

    lat, lon = zip(*loc.map(_one))
    return (pd.Series(lat, index=loc.index), pd.Series(lon, index=loc.index))


def crime_type_bucket(description: str) -> str:
    """Up to 8 categories; check more specific patterns first."""
    s = str(description).upper()
    if any(k in s for k in ("HOMICIDE", "MANSLAUGHTER", "MURDER")):
        return "Homicide"
    if "RAPE" in s or "SEXUAL" in s or ("SEX" in s and "LEWD" in s) or "SODOMY" in s:
        return "Sexual assault"
    if "ROBBERY" in s:
        return "Robbery"
    if "BURGLARY" in s or "BURGL" in s:
        return "Burglary"
    if any(k in s for k in ("ASSAULT", "BATTERY", "INTIMATE PARTNER")):
        return "Assault"
    if any(
        k in s for k in ("THEFT", "STOLEN", "EMBEZZL", "SHOPLIFT", "PICKPOCKET")
    ) or (s.strip().startswith("VEHICLE") and "STOL" in s):
        return "Theft"
    if "VANDAL" in s:
        return "Vandalism"
    return "Other"


def map_eight_to_coarse4(label8: str) -> str:
    """
    Broader 4-class target (often much easier to learn than 8 fine labels).
    """
    s = str(label8)
    if s in ("Homicide", "Sexual assault", "Robbery", "Assault"):
        return "Violent"
    if s in ("Theft", "Burglary"):
        return "Theft_burglary"
    if s == "Vandalism":
        return "Vandalism"
    return "Other"


def _rare_top_keep(ser: pd.Series, top_n: int) -> set[str]:
    if top_n <= 0 or len(ser) == 0:
        return set(ser.dropna().astype(str).unique())
    top = ser.value_counts().nlargest(top_n).index.astype(str)
    return set(top.tolist())


def apply_rare_category_collapse(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    premise_top_n: int,
    weapon_top_n: int,
    district_top_n: int,
) -> dict:
    """
    Map rare premise / weapon / district values to "Other" using the *train*
    split only (keeps the mapping honest for the test set).
    Returns a dict to save and reuse at prediction time.
    """
    state: dict[str, set[str] | int] = {}
    cols: list[tuple[str, int, str]] = [
        ("premise_description", premise_top_n, "Other"),
        ("weapon_description", weapon_top_n, "Other"),
        ("reporting_district", district_top_n, "Other"),
    ]
    for col, n, other in cols:
        if n <= 0:
            continue
        keep = _rare_top_keep(X_train[col], n)
        state[col] = keep
        for X in (X_train, X_test):
            X.loc[:, col] = X[col].where(X[col].astype(str).isin(keep), other)
    return state


def make_train_test_with_collapse(
    data_path: Path,
    max_rows: int | None,
    top_k: int,
    target: str,
    test_size: float,
    random_state: int,
    premise_top_n: int,
    weapon_top_n: int,
    district_top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict, str]:
    """Load, split, apply rare-bucketing on train (then test)."""
    df, y_name = load_and_prepare(
        data_path, max_rows, top_k, target=target
    )
    y = df[y_name]
    X = build_feature_frame(df)
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    collapse = apply_rare_category_collapse(
        X_train,
        X_test,
        premise_top_n,
        weapon_top_n,
        district_top_n,
    )
    return X_train, X_test, y_train, y_test, collapse, y_name


def apply_saved_collapse(row: dict, collapse: dict) -> None:
    """In-place: map a single input row for prediction."""
    for col, keep in collapse.items():
        if not isinstance(keep, set) or col not in row:
            continue
        v = str(row.get(col, ""))
        if v not in keep:
            row[col] = "Other"


def timeframe_quarter(hour: pd.Series, minute: pd.Series) -> pd.Series:
    """Four equal 6-hour windows in a day (minutes since midnight)."""
    h = hour.astype(int).clip(0, 23)
    m = minute.astype(int).clip(0, 59)
    mins = h * 60 + m
    return pd.Series(
        np.select(
            [mins < 360, mins < 720, mins < 1080],
            ["q1_00_06", "q2_06_12", "q3_12_18"],
            default="q4_18_24",
        ),
        index=hour.index,
    )


def load_and_prepare(
    path: Path, max_rows: int | None, top_k: int, target: str
) -> tuple[pd.DataFrame, str]:
    nrows = max_rows if max_rows else None
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df.columns = df.columns.str.strip()

    loc_col = "Location" if "Location" in df.columns else "Location "
    if loc_col not in df.columns:
        raise ValueError("Expected Location column in the CSV")
    if "Area Name" not in df.columns:
        raise ValueError("Expected Area Name column")

    df["lat"], df["lon"] = parse_location(df[loc_col])
    occ = pd.to_datetime(df["Date Occurred"], format="%m/%d/%Y", errors="coerce")
    df["area_name"] = df["Area Name"].astype(str).str.strip()
    df["premise_description"] = (
        df["Premise Description"].fillna("Unknown").astype(str).str.strip()
    )
    df["weapon_description"] = (
        df["Weapon Description"].fillna("Unknown").astype(str).str.strip()
    )
    df["reporting_district"] = (
        pd.to_numeric(df["Reporting District"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .astype(str)
    )
    good = (
        occ.notna()
        & df["lat"].notna()
        & df["lon"].notna()
        & df["Crime Code Description"].notna()
        & (df["area_name"] != "")
        & (df["area_name"].str.lower() != "nan")
    )
    df = df.loc[good].copy()
    occ = occ.loc[good].reset_index(drop=True)
    df = df.reset_index(drop=True)

    t = df["Time Occurred"].fillna(0).astype(int).clip(0, 2359)
    df["hour"] = t // 100
    df["minute"] = t % 100
    df["timeframe"] = timeframe_quarter(df["hour"], df["minute"])
    df["dayofweek"] = occ.dt.dayofweek
    df["month"] = occ.dt.month
    df["year"] = occ.dt.year

    la = (df["lat"] >= 33.5) & (df["lat"] <= 35.0) & (df["lon"] <= -117.5) & (df["lon"] >= -119.0)
    df = df[la].reset_index(drop=True)

    target_col = "crime_type"
    if target == "fine":
        counts = df["Crime Code Description"].value_counts()
        top = set(counts.head(top_k).index)
        df[target_col] = df["Crime Code Description"].where(
            df["Crime Code Description"].isin(top), "OTHER"
        )
    elif target == "coarse4":
        df["__b8"] = df["Crime Code Description"].map(crime_type_bucket)
        vcb = df["__b8"].value_counts()
        r8 = vcb[vcb < 5].index
        if len(r8):
            df.loc[df["__b8"].isin(r8), "__b8"] = "Other"
        df[target_col] = df["__b8"].map(map_eight_to_coarse4)
        df = df.drop(columns=["__b8"])
    else:
        df[target_col] = df["Crime Code Description"].map(crime_type_bucket)
        vc = df[target_col].value_counts()
        rare = vc[vc < 5].index
        if len(rare):
            df.loc[df[target_col].isin(rare), target_col] = "Other"

    return df, target_col


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "lat",
            "lon",
            "area_name",
            "reporting_district",
            "premise_description",
            "weapon_description",
            "timeframe",
            "dayofweek",
            "month",
            "year",
        ]
    ].copy()


def _one_hot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=True)


def build_pipeline(
    estimator: str,
    random_state: int,
    *,
    logreg_C: float = 0.1,
    rf_n_estimators: int = 300,
    rf_max_depth: int | None = 24,
    rf_min_samples_leaf: int = 1,
    rf_max_features: str | float = 0.35,
    hgb_max_iter: int = 200,
    hgb_max_depth: int = 16,
    hgb_learning_rate: float = 0.08,
    hgb_l2: float = 0.0,
) -> Pipeline:
    num_cols = ["lat", "lon", "dayofweek", "month", "year"]
    cat_cols = [
        "area_name",
        "reporting_district",
        "premise_description",
        "weapon_description",
        "timeframe",
    ]
    if estimator == "logreg":
        pre = ColumnTransformer(
            [
                ("num", StandardScaler(), num_cols),
                ("cat", _one_hot(), cat_cols),
            ],
            remainder="drop",
        )
        clf = LogisticRegression(
            max_iter=3000,
            solver="saga",
            penalty="l2",
            C=float(logreg_C),
            n_jobs=1,
            tol=1e-3,
        )
    elif estimator == "rf":
        pre = ColumnTransformer(
            [
                ("num", "passthrough", num_cols),
                ("cat", _one_hot(), cat_cols),
            ],
            remainder="drop",
        )
        mdepth: int | None
        if rf_max_depth is None:
            mdepth = None
        else:
            mdepth = int(rf_max_depth)

        clf = RandomForestClassifier(
            n_estimators=int(rf_n_estimators),
            max_depth=mdepth,
            min_samples_leaf=int(rf_min_samples_leaf),
            max_features=rf_max_features,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=random_state,
        )
    elif estimator == "hgb":
        return build_hgb_pipeline(
            random_state=random_state,
            hgb_max_iter=hgb_max_iter,
            hgb_max_depth=hgb_max_depth,
            hgb_learning_rate=hgb_learning_rate,
            hgb_l2=hgb_l2,
        )
    else:
        raise ValueError("estimator must be 'logreg', 'rf', or 'hgb'")
    return Pipeline([("pre", pre), ("clf", clf)])


def build_hgb_pipeline(
    random_state: int,
    *,
    hgb_max_iter: int = 200,
    hgb_max_depth: int = 16,
    hgb_learning_rate: float = 0.08,
    hgb_l2: float = 0.0,
) -> Pipeline:
    """
    Dense numeric + integer ordinal categories (native HistGBM support).
    Avoids huge sparse one-hots, so it scales to large row counts.
    """
    num_cols = ["lat", "lon", "dayofweek", "month", "year"]
    cat_cols = [
        "area_name",
        "reporting_district",
        "premise_description",
        "weapon_description",
        "timeframe",
    ]
    n_num = len(num_cols)
    cat_idx: list[int] = list(
        range(n_num, n_num + len(cat_cols))
    )  # indices in transformed matrix

    OE = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        max_categories=100,
    )

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OE, cat_cols),
        ],
    )

    clf = HistGradientBoostingClassifier(
        max_iter=int(hgb_max_iter),
        max_depth=int(hgb_max_depth),
        learning_rate=float(hgb_learning_rate),
        l2_regularization=float(hgb_l2),
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        categorical_features=cat_idx,  # after ColumnTransformer, num then cat
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Train crime-type classifier from time + location: LogisticRegression, "
            "RandomForest, or HistGradientBoosting (ordinal categories + native cat support)."
        )
    )
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--max-rows", type=int, default=200_000)
    p.add_argument(
        "--target",
        type=str,
        choices=("eight", "coarse4", "fine"),
        default="eight",
        help=(
            "eight = 8 crime buckets (default); coarse4 = 4 coarse types (~easier); "
            "fine = top-k LAPD labels + OTHER."
        ),
    )
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--model-out", type=str, default="la_crime_type_model.joblib")
    p.add_argument(
        "--estimator",
        type=str,
        choices=("logreg", "rf", "hgb"),
        default="logreg",
        help="logreg | rf | hgb (HistGradientBoostingClassifier, dense + ordinal cats).",
    )
    p.add_argument(
        "--premise-top-n",
        type=int,
        default=30,
        help="Keep top-N premise labels on train; rest -> Other. 0 = no bucketing.",
    )
    p.add_argument(
        "--weapon-top-n",
        type=int,
        default=25,
        help="Keep top-N weapon labels on train; rest -> Other. 0 = no bucketing.",
    )
    p.add_argument(
        "--district-top-n",
        type=int,
        default=100,
        help="Keep top-N reporting districts on train; rest -> Other. 0 = no bucketing.",
    )
    p.add_argument(
        "--logreg-C",
        type=float,
        default=0.1,
        help="Inverse regularization strength for LogisticRegression (larger = less L2).",
    )
    p.add_argument(
        "--rf-n-estimators",
        type=int,
        default=300,
    )
    p.add_argument(
        "--rf-max-depth",
        type=int,
        default=24,
        help="Max tree depth for RF. Use 0 for None (unlimited).",
    )
    p.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=1,
    )
    p.add_argument(
        "--rf-max-features",
        type=str,
        default="0.35",
        help='RF max_features, e.g. "sqrt" or 0.35',
    )
    p.add_argument(
        "--hgb-max-iter", type=int, default=200, help="HistGradientBoosting max_iter."
    )
    p.add_argument(
        "--hgb-max-depth", type=int, default=16, help="HistGradientBoosting max_depth."
    )
    p.add_argument(
        "--hgb-lr", type=float, default=0.08, help="HistGradientBoosting learning_rate."
    )
    p.add_argument(
        "--hgb-l2", type=float, default=0.0, help="HistGradientBoosting l2 regularization."
    )
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found: {data_path}\n"
            "kagglehub.dataset_download('cityofLA/crime-in-los-angeles')"
        )

    max_rows = None if args.max_rows == 0 else args.max_rows
    X_train, X_test, y_train, y_test, collapse_state, y_name = make_train_test_with_collapse(
        data_path,
        max_rows,
        args.top_k,
        args.target,
        args.test_size,
        args.random_state,
        args.premise_top_n,
        args.weapon_top_n,
        args.district_top_n,
    )
    df_len = len(y_train) + len(y_test)

    try:
        rff = float(args.rf_max_features)
    except ValueError:
        rff = args.rf_max_features
    rf_depth: int | None = None if args.rf_max_depth == 0 else int(args.rf_max_depth)

    pipe = build_pipeline(
        args.estimator,
        args.random_state,
        logreg_C=args.logreg_C,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=rf_depth,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        rf_max_features=rff,
        hgb_max_iter=args.hgb_max_iter,
        hgb_max_depth=args.hgb_max_depth,
        hgb_learning_rate=args.hgb_lr,
        hgb_l2=args.hgb_l2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    acc = (y_pred == y_test).mean()

    est_note = {
        "logreg": "LogisticRegression + predict_proba",
        "rf": "RandomForestClassifier + predict_proba",
        "hgb": "HistGradientBoostingClassifier + predict_proba",
    }[args.estimator]
    print(f"\nModel: {args.estimator} ({est_note})")
    print(f"Rows: {df_len} | Classes: {pd.concat([y_train, y_test]).nunique()}")
    print(f"Test accuracy: {acc:.3f} | macro F1: {f1m:.3f} | weighted F1: {f1w:.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    sample = X_test.iloc[:1]
    proba = pipe.predict_proba(sample)[0]
    classes = pipe.named_steps["clf"].classes_  # type: ignore[index]
    print("Example: estimated probability for each crime type (one test row; values sum to 1):")
    for i in np.argsort(proba)[::-1][: min(8, len(classes))]:
        print(f"  {classes[i]}: {proba[i]:.4f}")

    joblib.dump(
        {
            "pipeline": pipe,
            "estimator": args.estimator,
            "target_mode": args.target,
            "label_column": y_name,
            "feature_columns": list(X_train.columns),
            "class_names": list(classes),
            "rare_collapse": collapse_state,
            "note": "Class probabilities: clf.predict_proba",
        },
        args.model_out,
    )
    print(f"\nSaved: {args.model_out}")


if __name__ == "__main__":
    main()
