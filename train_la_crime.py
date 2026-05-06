# LA crime: predict type from time/place — logreg or RF (--classifier) on one-hot + scaled numbers.
# Compare LogReg / LDA / RandomForest: python compare_logreg_lda.py
# Tune RF hyperparameters: python tune_rf_la_crime.py
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# default CSV path (Kaggle download)
DEFAULT_DATA = (
    Path.home()
    / ".cache/kagglehub/datasets/cityofLA/crime-in-los-angeles/versions/7"
    / "Crime_Data_2010_2017.csv"
)

NUM_COLS = ["lat", "lon", "dayofweek", "month", "year"]
CAT_COLS = [
    "area_name",
    "reporting_district",
    "premise_description",
    "weapon_description",
    "timeframe",
]


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
    # 8 text buckets for Crime Code Description (order matters: specific first)
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
    # merge 8 buckets into 4 labels
    s = str(label8)
    if s in ("Homicide", "Sexual assault", "Robbery", "Assault"):
        return "Violent"
    if s in ("Theft", "Burglary"):
        return "Theft_burglary"
    if s == "Vandalism":
        return "Vandalism"
    return "Other"


def _top_n_categories(ser: pd.Series, top_n: int) -> set[str]:
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
    # learn frequent categories on train, map the rest to "Other" (train+test)
    state: dict[str, set[str] | int] = {}
    cols: list[tuple[str, int, str]] = [
        ("premise_description", premise_top_n, "Other"),
        ("weapon_description", weapon_top_n, "Other"),
        ("reporting_district", district_top_n, "Other"),
    ]
    for col, n, other in cols:
        if n <= 0:
            continue
        keep = _top_n_categories(X_train[col], n)
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
    df, y_name = load_and_prepare(data_path, max_rows, top_k, target=target)
    y = df[y_name]
    X = build_feature_frame(df)
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    collapse = apply_rare_category_collapse(
        X_train, X_test, premise_top_n, weapon_top_n, district_top_n
    )
    return X_train, X_test, y_train, y_test, collapse, y_name


def apply_saved_collapse(row: dict, collapse: dict) -> None:
    for col, keep in collapse.items():
        if not isinstance(keep, set) or col not in row:
            continue
        v = str(row.get(col, ""))
        if v not in keep:
            row[col] = "Other"


def timeframe_quarter(hour: pd.Series, minute: pd.Series) -> pd.Series:
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


def load_and_prepare(path, max_rows, top_k, target: str):
    # read CSV, clean, build y=crime_type; target: coarse4 | eight | fine
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
        df["_b8"] = df["Crime Code Description"].map(crime_type_bucket)
        vcb = df["_b8"].value_counts()
        r8 = vcb[vcb < 5].index
        if len(r8):
            df.loc[df["_b8"].isin(r8), "_b8"] = "Other"
        df[target_col] = df["_b8"].map(map_eight_to_coarse4)
        df = df.drop(columns=["_b8"])
    else:
        df[target_col] = df["Crime Code Description"].map(crime_type_bucket)
        vc = df[target_col].value_counts()
        rare = vc[vc < 5].index
        if len(rare):
            df.loc[df[target_col].isin(rare), target_col] = "Other"

    return df, target_col


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUM_COLS + CAT_COLS].copy()


def build_logreg_pipeline(*, logreg_c: float, random_state: int) -> Pipeline:
    # one Pipeline: scale nums + one-hot cats + logreg (same for train and predict)
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CAT_COLS,
            ),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        max_iter=3000,
        solver="saga",
        penalty="l2",
        C=float(logreg_c),
        n_jobs=1,
        tol=1e-3,
        random_state=random_state,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def build_lda_pipeline() -> Pipeline:
    # Linear Discriminant Analysis: sklearn needs dense X (dense one-hot here).
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
            ),
        ],
        remainder="drop",
    )
    clf = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    return Pipeline([("pre", pre), ("clf", clf)])


def build_rf_pipeline(
    *,
    random_state: int,
    n_estimators: int = 100,
    max_depth: int | None = 20,
) -> Pipeline:
    # Dense one-hot (same as LDA); sklearn RandomForest expects dense X.
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
            ),
        ],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    # --- CLI args
    p = argparse.ArgumentParser(
        description="Train LA crime classifier (logistic regression or random forest)."
    )
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--max-rows", type=int, default=200_000)
    p.add_argument(
        "--target", choices=("coarse4", "eight", "fine"), default="coarse4"
    )
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--model-out", type=str, default="la_crime_type_model.joblib")
    p.add_argument("--premise-top-n", type=int, default=30)
    p.add_argument("--weapon-top-n", type=int, default=25)
    p.add_argument("--district-top-n", type=int, default=100)
    p.add_argument(
        "--classifier",
        choices=("logreg", "rf"),
        default="logreg",
        help="rf uses tuned coarse4 defaults (300 trees, depth 32) unless overridden.",
    )
    p.add_argument(
        "--logreg-c", "--logreg-C", type=float, default=0.1, dest="logreg_c"
    )
    p.add_argument(
        "--rf-n-estimators",
        type=int,
        default=300,
        help="RandomForest only; matches accuracy-CV default for coarse4.",
    )
    p.add_argument(
        "--rf-max-depth",
        type=int,
        default=32,
        help="RandomForest only; use 0 for None.",
    )
    args = p.parse_args()

    if args.classifier == "rf" and args.model_out == "la_crime_type_model.joblib":
        args.model_out = f"la_crime_rf_{args.target}.joblib"

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found: {data_path}\n"
            "kagglehub.dataset_download('cityofLA/crime-in-los-angeles')"
        )

    # --- load, split, bucket rare text fields (function lives above)
    max_rows = None if args.max_rows == 0 else args.max_rows
    X_train, X_test, y_train, y_test, collapse, y_name = make_train_test_with_collapse(
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
    n = len(y_train) + len(y_test)

    # --- model
    if args.classifier == "logreg":
        model = build_logreg_pipeline(
            logreg_c=args.logreg_c, random_state=args.random_state
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_train, y_train)
    else:
        rf_depth = None if args.rf_max_depth == 0 else args.rf_max_depth
        model = build_rf_pipeline(
            random_state=args.random_state,
            n_estimators=args.rf_n_estimators,
            max_depth=rf_depth,
        )
        model.fit(X_train, y_train)

    # --- test metrics
    y_pred = model.predict(X_test)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    acc = float((y_pred == y_test).mean())

    print("\n--- test set ---")
    print(f"target={args.target}  rows={n}  n_classes={pd.concat([y_train, y_test]).nunique()}")
    print(f"accuracy={acc:.3f}  macro_F1={f1m:.3f}  weighted_F1={f1w:.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    proba = model.predict_proba(X_test.iloc[:1])[0]
    classes = model.named_steps["clf"].classes_  # type: ignore[union-attr]
    k_show = min(8, len(classes))
    print("example P(class) for one test row:")
    for j in np.argsort(proba)[::-1][:k_show]:
        print(f"  {classes[j]}: {proba[j]:.4f}")

    # --- save
    joblib.dump(
        {
            "pipeline": model,
            "estimator": args.classifier,
            "target_mode": args.target,
            "label_column": y_name,
            "feature_columns": list(X_train.columns),
            "class_names": list(classes),
            "rare_collapse": collapse,
        },
        args.model_out,
    )
    print(f"\nsaved: {args.model_out}\n")


if __name__ == "__main__":
    main()
