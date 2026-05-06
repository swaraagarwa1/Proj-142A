# Load a .joblib from train_la_crime.py (logreg or rf bundle) and print P(class) for one place/time.
# Example: python predict_la_crime.py --lat 34.05 --lon -118.25 --area-name Central --date 03/15/2016 --time 2200

import argparse
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_la_crime import apply_saved_collapse, build_feature_frame, timeframe_quarter


def predict_proba_for_inputs(
    bundle,
    *,
    lat,
    lon,
    area_name,
    reporting_district="-1",
    premise_description="Unknown",
    weapon_description="Unknown",
    date_mmddyyyy="01/15/2016",
    time_hhmm="1430",
):
    """One row -> predict_proba; used by CLI and streamlit_app."""
    pipe = bundle["pipeline"]
    collapse = bundle.get("rare_collapse") or {}

    t = re.sub(r"\D", "", time_hhmm)[:4].zfill(4)
    tt = int(t)
    hour = min(tt // 100, 23)
    minute = min(tt % 100, 59)
    occ = datetime.strptime(date_mmddyyyy, "%m/%d/%Y")
    tf = str(timeframe_quarter(pd.Series([hour]), pd.Series([minute])).iloc[0])

    row = {
        "lat": lat,
        "lon": lon,
        "area_name": area_name.strip(),
        "reporting_district": str(reporting_district).strip(),
        "premise_description": premise_description.strip() or "Unknown",
        "weapon_description": weapon_description.strip() or "Unknown",
        "timeframe": tf,
        "dayofweek": occ.weekday(),
        "month": occ.month,
        "year": occ.year,
    }
    apply_saved_collapse(row, collapse)
    x = build_feature_frame(pd.DataFrame([row]))
    proba = pipe.predict_proba(x)[0]
    names = pipe.named_steps["clf"].classes_  # type: ignore[union-attr]
    return proba, names, tf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="la_crime_type_model.joblib")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--area-name", type=str, required=True)
    p.add_argument("--reporting-district", type=str, default="-1")
    p.add_argument("--premise-description", type=str, default="Unknown")
    p.add_argument("--weapon-description", type=str, default="Unknown")
    p.add_argument("--date", type=str, required=True, help="MM/DD/YYYY")
    p.add_argument("--time", type=str, required=True, help="e.g. 1430")
    p.add_argument("--top", type=int, default=8)
    args = p.parse_args()

    # load model, predict, print
    bundle = joblib.load(Path(args.model))
    proba, names, tf = predict_proba_for_inputs(
        bundle,
        lat=args.lat,
        lon=args.lon,
        area_name=args.area_name,
        reporting_district=args.reporting_district,
        premise_description=args.premise_description,
        weapon_description=args.weapon_description,
        date_mmddyyyy=args.date,
        time_hhmm=args.time,
    )
    print(f"time bucket: {tf}")
    for i in np.argsort(proba)[::-1][: args.top]:
        print(f"  {names[i]}: {proba[i]:.4f}")
    print(f"sum: {proba.sum():.4f}")


if __name__ == "__main__":
    main()
