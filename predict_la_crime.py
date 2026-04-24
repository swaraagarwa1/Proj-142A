"""Print crime-type probabilities from a saved train_la_crime.py model (predict_proba)."""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_la_crime import apply_saved_collapse, build_feature_frame, timeframe_quarter


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Show crime-type probabilities for lat/lon, area name, date, time."
    )
    p.add_argument("--model", type=str, default="la_crime_type_model.joblib")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument(
        "--area-name",
        type=str,
        required=True,
        help='LAPD Area Name as in CSV, e.g. "Central" or "77th Street".',
    )
    p.add_argument(
        "--reporting-district",
        type=str,
        default="-1",
        help="Reporting District value from the dataset, if known.",
    )
    p.add_argument(
        "--premise-description",
        type=str,
        default="Unknown",
        help='Premise Description, e.g. "STREET" or "PARKING LOT".',
    )
    p.add_argument(
        "--weapon-description",
        type=str,
        default="Unknown",
        help='Weapon Description, or "Unknown" if not applicable.',
    )
    p.add_argument("--date", type=str, required=True, help="MM/DD/YYYY")
    p.add_argument("--time", type=str, required=True, help="4-digit military, e.g. 1430")
    p.add_argument("--top", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    bundle = joblib.load(Path(args.model))
    pipe = bundle["pipeline"]
    collapse = bundle.get("rare_collapse") or {}

    t = re.sub(r"\D", "", args.time)[:4].zfill(4)
    tt = int(t)
    hour = min(tt // 100, 23)
    minute = min(tt % 100, 59)
    occ = datetime.strptime(args.date, "%m/%d/%Y")

    tf = str(timeframe_quarter(pd.Series([hour]), pd.Series([minute])).iloc[0])

    row = {
        "lat": args.lat,
        "lon": args.lon,
        "area_name": args.area_name.strip(),
        "reporting_district": str(args.reporting_district).strip(),
        "premise_description": args.premise_description.strip() or "Unknown",
        "weapon_description": args.weapon_description.strip() or "Unknown",
        "timeframe": tf,
        "dayofweek": occ.weekday(),
        "month": occ.month,
        "year": occ.year,
    }
    apply_saved_collapse(row, collapse)  # match training rare-category rules
    x = build_feature_frame(pd.DataFrame([row]))
    proba = pipe.predict_proba(x)[0]
    names = pipe.named_steps["clf"].classes_  # type: ignore[union-attr]

    print(f"time quarter: {tf}")
    print("Estimated probability for each crime type:")
    for i in np.argsort(proba)[::-1][: args.top]:
        print(f"  {names[i]}: {proba[i]:.4f}")
    print(f"sum: {proba.sum():.4f}")


if __name__ == "__main__":
    main()
