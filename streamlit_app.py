# Run:  streamlit run streamlit_app.py
# Train the model first:  python train_la_crime.py

from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from predict_la_crime import predict_proba_for_inputs


@st.cache_resource
def load_model(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None, f"File not found: {p.resolve()}"
    return joblib.load(p), None


def main():
    st.set_page_config(
        page_title="LA Crime Type Predictor",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("LA crime type predictor")
    st.markdown(
        "Predict **coarse crime category** from where and when an incident occurred "
        "(multinomial **logistic regression** — same as `train_la_crime.py`). "
        "The model does **not** use the crime’s text description as an input."
    )

    # -------- sidebar --------
    with st.sidebar:
        st.header("Model")
        default = Path("la_crime_type_model.joblib")
        model_path = st.text_input(
            "Path to `.joblib`",
            value=str(default.resolve() if default.exists() else default),
            help="Default is `la_crime_type_model.joblib` in this folder after training.",
        )
        bundle, err = load_model(model_path)
        if err:
            st.error(err)
            st.markdown("Train first:\n```\npython train_la_crime.py\n```")
        else:
            st.success("Model loaded")
            st.caption(
                f"**Target:** `{bundle.get('target_mode', '?')}`  "
                f"· **Classes:** {len(bundle.get('class_names', []))}"
            )
        st.divider()
        st.markdown(
            "**Tips**\n"
            "- Use real LAPD **area** names (e.g. *Central*, *77th Street*).\n"
            "- **Time** is 4 digits, 24h (e.g. 1430 = 2:30 PM).\n"
            "- **Lat/lon** should be in the LA area (roughly the training bbox)."
        )

    # -------- inputs --------
    st.subheader("1 · Location")
    c1, c2 = st.columns(2)
    with c1:
        lat = st.number_input(
            "Latitude",
            value=34.0522,
            format="%.5f",
            help="WGS84 decimal degrees",
        )
    with c2:
        lon = st.number_input(
            "Longitude",
            value=-118.2437,
            format="%.5f",
            help="WGS84 decimal degrees (negative for West)",
        )

    st.subheader("2 · LAPD context")
    a1, a2 = st.columns(2)
    with a1:
        area_name = st.text_input(
            "Area name",
            value="Central",
            help='From the dataset, e.g. "Northeast", "77th Street"',
        )
    with a2:
        district = st.text_input(
            "Reporting district",
            value="-1",
            help="As in the data, or -1 if unknown",
        )
    p1, p2 = st.columns(2)
    with p1:
        premise = st.text_input("Premise", value="STREET", help="e.g. STREET, PARKING LOT")
    with p2:
        weapon = st.text_input("Weapon", value="STRONG-ARM", help="Or UNKNOWN")

    st.subheader("3 · Date & time")
    d1, d2 = st.columns(2)
    with d1:
        date_in = st.text_input("Date", value="03/15/2016", help="MM/DD/YYYY")
    with d2:
        time_in = st.text_input("Time (24h)", value="2200", help="HHMM, e.g. 2200 = 10:00 PM")

    run = st.button("Run prediction", type="primary", use_container_width=True)

    if not run:
        st.info("Fill in the fields above, then click **Run prediction**.")
        return
    if bundle is None:
        st.warning("Set a valid model file in the sidebar.")
        return

    try:
        proba, names, tf = predict_proba_for_inputs(
            bundle,
            lat=float(lat),
            lon=float(lon),
            area_name=area_name,
            reporting_district=district,
            premise_description=premise,
            weapon_description=weapon,
            date_mmddyyyy=date_in,
            time_hhmm=time_in,
        )
    except ValueError as e:
        st.error(f"Check **date** (use MM/DD/YYYY) and **time** (4 digits). ({e})")
        return
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    # -------- results --------
    st.divider()
    st.subheader("Results")
    st.caption(f"6-hour time-of-day bucket used in the model: **{tf}**")

    best_i = int(np.argmax(proba))
    c_left, c_right = st.columns([1, 2])
    with c_left:
        st.metric(
            "Most likely class",
            str(names[best_i]),
            f"{100.0 * float(proba[best_i]):.1f}%",
        )
    with c_right:
        order = np.argsort(proba)[::-1]
        table = pd.DataFrame(
            {
                "Class": [str(names[i]) for i in order],
                "Probability": [float(proba[i]) for i in order],
            }
        )
        table["%"] = (100 * table["Probability"]).round(1)
        st.dataframe(
            table[["Class", "%", "Probability"]].rename(
                columns={"%": "Percent", "Probability": "P( class )"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    # horizontal bar chart: classes ordered from most to least likely (top to bottom)
    chart_df = table.rename(columns={"Class": "class"})[["class", "Probability"]]
    class_order = chart_df["class"].tolist()
    ch = (
        alt.Chart(chart_df)
        .mark_bar(color="#2E86AB")
        .encode(
            x=alt.X("Probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("class:N", title=None, sort=class_order),
            tooltip=["class", alt.Tooltip("Probability:Q", format=".1%")],
        )
    )
    st.altair_chart(ch, use_container_width=True)
    st.caption("Probabilities over all classes sum to **1.0**.")


if __name__ == "__main__":
    main()
