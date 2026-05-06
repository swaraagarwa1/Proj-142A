# Run:  streamlit run streamlit_app.py
# Train RF bundle (default UI):  python train_la_crime.py --classifier rf --target coarse4

import os
from pathlib import Path
from typing import Optional, Tuple

import altair as alt
import folium
from folium.plugins import MousePosition
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from predict_la_crime import predict_proba_for_inputs
from train_la_crime import DEFAULT_DATA


@st.cache_data(show_spinner="Resolving crime CSV (first run may download from Kaggle)…")
def resolve_crime_csv_path(fallback: Path) -> str:
    """Use local training CSV if present; otherwise try Kaggle (needs credentials on Streamlit Cloud)."""
    fb = fallback.expanduser()
    if fb.is_file():
        return str(fb.resolve())
    try:
        import kagglehub

        root = Path(kagglehub.dataset_download("cityofLA/crime-in-los-angeles"))
        hit = next(iter(root.rglob("Crime_Data_2010_2017.csv")), None)
        if hit is not None and hit.is_file():
            return str(hit.resolve())
    except Exception:
        pass
    return str(fb)

# Rough LA training bbox (matches train_la_crime.load_and_prepare)
_MAP_SW = (33.5, -119.0)
_MAP_NE = (35.0, -117.5)
_DEFAULT_LAT = 34.0522
_DEFAULT_LON = -118.2437
_MAP_WIDGET_KEY = "crime_loc_map"

# If the crime CSV is missing, these keep selectboxes usable.
_FALLBACK_AREAS = sorted(
    [
        "Central",
        "Devonshire",
        "Foothill",
        "Harbor",
        "Hollenbeck",
        "Hollywood",
        "Mission",
        "Newton",
        "Northeast",
        "North Hollywood",
        "Olympic",
        "Pacific",
        "Rampart",
        "Southeast",
        "Southwest",
        "Topanga",
        "Van Nuys",
        "West LA",
        "West Valley",
        "Wilshire",
        "77th Street",
    ]
)
_FALLBACK_DISTRICTS = ["-1"] + [str(i) for i in range(1, 22)] + ["77", "82", "83", "84", "88"]
_FALLBACK_PREMISES = [
    "STREET",
    "PARKING LOT",
    "SINGLE FAMILY DWELLING",
    "SIDEWALK",
    "APARTMENT/CONDOMINIUM",
    "OTHER PREMISE",
    "UNKNOWN",
]
_FALLBACK_WEAPONS = [
    "STRONG-ARM",
    "UNKNOWN",
    "HAND GUN",
    "KNIFE",
    "FIREARM",
    "VEHICLE",
    "VERBAL THREAT",
    "BLUNT OBJECT",
]


@st.cache_data(show_spinner=False)
def lapd_dropdown_options(csv_path: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Area / district / premise / weapon lists from the training CSV (samples)."""
    path = Path(csv_path)
    if not path.exists():
        return (
            _FALLBACK_AREAS,
            _FALLBACK_DISTRICTS,
            _FALLBACK_PREMISES,
            _FALLBACK_WEAPONS,
        )
    try:
        usecols = [
            "Area Name",
            "Reporting District",
            "Premise Description",
            "Weapon Description",
        ]
        df = pd.read_csv(path, usecols=usecols, nrows=600_000, low_memory=False)
        df.columns = df.columns.str.strip()
        areas = sorted(df["Area Name"].dropna().astype(str).str.strip().unique())

        rd = pd.to_numeric(df["Reporting District"], errors="coerce").fillna(-1).astype(int)
        districts = sorted({str(x) for x in rd.tolist()}, key=lambda x: (x != "-1", int(x)))
        districts = districts[:320]

        prem = df["Premise Description"].fillna("Unknown").astype(str).str.strip()
        premises = prem.value_counts().head(72).index.tolist()

        wea = df["Weapon Description"].fillna("Unknown").astype(str).str.strip()
        weapons = wea.value_counts().head(48).index.tolist()

        return areas, districts, premises, weapons
    except Exception:
        return (
            _FALLBACK_AREAS,
            _FALLBACK_DISTRICTS,
            _FALLBACK_PREMISES,
            _FALLBACK_WEAPONS,
        )


def _hour12_ampm_to_hhmm(hour12: int, ampm: str, minute: int) -> str:
    """Convert 12-hour clock + AM/PM to HHMM string for predict_la_crime."""
    minute = int(min(max(minute, 0), 59))
    ap = ampm.strip().upper()
    if ap == "AM":
        h24 = 0 if hour12 == 12 else hour12
    else:
        h24 = 12 if hour12 == 12 else hour12 + 12
    return f"{h24:02d}{minute:02d}"


def _selectbox_with_default(
    label: str,
    options: list[str],
    default: str,
    key: str,
    help: Optional[str] = None,
) -> str:
    opts = list(dict.fromkeys(options))
    if default not in opts:
        opts.insert(0, default)
    if key not in st.session_state:
        st.session_state[key] = default
    if st.session_state[key] not in opts:
        st.session_state[key] = default if default in opts else opts[0]
    return st.selectbox(label, opts, key=key, help=help)


def _parse_latlng(obj) -> Optional[Tuple[float, float]]:
    """Parse Leaflet LatLng / GeoJSON-like dicts from streamlit-folium."""
    if obj is None:
        return None
    if isinstance(obj, dict) and len(obj) == 0:
        return None
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        try:
            return float(obj[0]), float(obj[1])
        except (TypeError, ValueError):
            return None
    if isinstance(obj, dict):
        nested = obj.get("latlng") or obj.get("LatLng")
        if isinstance(nested, dict):
            return _parse_latlng(nested)
        lat = obj.get("lat")
        lng = obj.get("lng")
        if lat is None:
            lat = obj.get("latitude")
        if lng is None:
            lng = obj.get("longitude")
        coords = obj.get("coordinates")
        if (lat is None or lng is None) and isinstance(coords, (list, tuple)) and len(coords) >= 2:
            lng, lat = float(coords[0]), float(coords[1])  # GeoJSON order
        if lat is None or lng is None:
            return None
        try:
            return float(lat), float(lng)
        except (TypeError, ValueError):
            return None
    return None


def _commit_pick_from_click(raw) -> None:
    parsed = _parse_latlng(raw)
    if parsed is None:
        return
    sig = (round(parsed[0], 8), round(parsed[1], 8))
    if sig == st.session_state.get("_map_click_sig"):
        return
    st.session_state._map_click_sig = sig
    st.session_state.pick_lat = parsed[0]
    st.session_state.pick_lon = parsed[1]


def _sync_pick_from_folium_mirror() -> None:
    """streamlit-folium copies component JSON into session_state[this key] on change."""
    raw = st.session_state.get(_MAP_WIDGET_KEY)
    if not isinstance(raw, dict):
        return
    _commit_pick_from_click(raw.get("last_clicked"))
    _commit_pick_from_click(raw.get("last_object_clicked"))


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

    try:
        sec = st.secrets
        for key in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
            if key in sec:
                os.environ[key] = str(sec[key])
    except Exception:
        pass

    st.title("LA crime type predictor")
    st.markdown(
        "Predict **coarse crime category** (4 classes) from where and when an incident occurred "
        "using a **random forest** trained with `train_la_crime.py --classifier rf` "
        "(**300** trees, **max_depth 32**, tuned for accuracy on coarse labels). "
        "The model does **not** use the crime’s text description as an input."
    )

    # -------- sidebar --------
    with st.sidebar:
        st.header("Model")
        if "_crime_csv_default" not in st.session_state:
            st.session_state._crime_csv_default = resolve_crime_csv_path(DEFAULT_DATA)

        default = Path("la_crime_rf_coarse4.joblib")
        model_path = st.text_input(
            "Path to `.joblib`",
            value=str(default.resolve() if default.exists() else default),
            help="Default `la_crime_rf_coarse4.joblib` from `train_la_crime.py --classifier rf`.",
        )
        dropdown_csv = st.text_input(
            "Crime CSV (LAPD dropdowns)",
            value=st.session_state._crime_csv_default,
            help="Same file training uses for menus; auto-downloads via Kaggle if missing (set Kaggle secrets on Streamlit Cloud).",
        )
        csv_exists = Path(dropdown_csv).expanduser().is_file()
        if not csv_exists:
            st.caption(
                "CSV not found → LAPD menus use built‑in fallbacks. "
                "For full menus on **[Streamlit Community Cloud](https://share.streamlit.io/)**, add secrets "
                "`KAGGLE_USERNAME` and `KAGGLE_KEY` (Kaggle account → API → Create token)."
            )
        bundle, err = load_model(model_path)
        if err:
            st.error(err)
            st.markdown(
                "Train first:\n```\n"
                "python train_la_crime.py --classifier rf --target coarse4\n```"
            )
        else:
            st.success("Model loaded")
            st.caption(
                f"**Target:** `{bundle.get('target_mode', '?')}`  "
                f"· **Classes:** {len(bundle.get('class_names', []))}"
            )
        st.divider()
        st.markdown(
            "**Tips**\n"
            "- **Drag** the map to pan; scroll or buttons to zoom.\n"
            "- **Click** the map to set **latitude / longitude** only.\n"
            "- **LAPD division / district** are **not** inferred from the pin—section 2 has "
            "checkboxes for **unknown division** (blank area) and **unknown district** (-1).\n"
            "- **Time** uses **hour / AM–PM / minute**."
        )

    # -------- location from interactive map --------
    if "pick_lat" not in st.session_state:
        st.session_state.pick_lat = _DEFAULT_LAT
        st.session_state.pick_lon = _DEFAULT_LON
    if "_folium_zoom" not in st.session_state:
        st.session_state._folium_zoom = 11

    # Apply clicks saved on the Streamlit side (mirrors debounced browser updates).
    _sync_pick_from_folium_mirror()

    st.subheader("1 · Location")
    st.markdown(
        "Pan and zoom the map (drag to move), then **click** the map background to move the pin. "
        "The blue outline is non-interactive so clicks reach the map."
    )

    lat0, lon0 = float(st.session_state.pick_lat), float(st.session_state.pick_lon)
    z0 = int(st.session_state._folium_zoom)
    lat_before, lon_before = lat0, lon0

    m = folium.Map(location=[lat0, lon0], zoom_start=z0, tiles="OpenStreetMap")
    # Marker must not capture clicks — otherwise map "last_clicked" never fires.
    folium.Marker(
        [lat0, lon0],
        tooltip="Incident location — click the map to move this pin",
        icon=folium.Icon(color="red"),
        interactive=False,
        draggable=False,
    ).add_to(m)
    # Filled rectangle otherwise eats almost every click inside LA.
    folium.Rectangle(
        bounds=[_MAP_SW, _MAP_NE],
        color="#2563eb",
        fill=True,
        fill_opacity=0.06,
        weight=2,
        tooltip="Approximate training coverage",
        interactive=False,
    ).add_to(m)
    MousePosition(
        position="bottomleft",
        separator="  ",
        prefix="Cursor:",
        num_digits=6,
    ).add_to(m)

    # Do not pass center=/zoom= here: those props update Leaflet client-side and can
    # fight Python-driven marker placement.
    map_out = st_folium(
        m,
        height=440,
        use_container_width=True,
        key=_MAP_WIDGET_KEY,
        returned_objects=["last_clicked", "last_object_clicked", "zoom"],
    )

    if isinstance(map_out, dict):
        z_new = map_out.get("zoom")
        if isinstance(z_new, (int, float)):
            st.session_state._folium_zoom = int(z_new)
        _commit_pick_from_click(map_out.get("last_clicked"))
        _commit_pick_from_click(map_out.get("last_object_clicked"))

    moved = round(float(st.session_state.pick_lat), 8) != round(
        lat_before, 8
    ) or round(float(st.session_state.pick_lon), 8) != round(lon_before, 8)
    if moved:
        st.rerun()

    disp_lat = float(st.session_state.pick_lat)
    disp_lon = float(st.session_state.pick_lon)
    st.caption(
        f"**Latitude:** `{disp_lat:.6f}` **Longitude:** `{disp_lon:.6f}`"
    )
    st.caption(
        "**LAPD division / reporting district** are **not** derived from this pin—they are "
        "separate fields in section 2."
    )

    lat = disp_lat
    lon = disp_lon

    # -------- inputs --------
    prev_csv = st.session_state.get("_lapd_opts_csv_path")
    if prev_csv != dropdown_csv:
        for k in ("lapd_area", "lapd_district", "lapd_premise", "lapd_weapon"):
            st.session_state.pop(k, None)
        st.session_state["_lapd_opts_csv_path"] = dropdown_csv

    areas, districts, premises, weapons = lapd_dropdown_options(dropdown_csv)

    st.subheader("2 · LAPD context")
    st.info(
        "**Coordinates ≠ LAPD labels.** The map only sets **lat/lon**. **Area Name** and "
        "**Reporting District** come from the crime table—this app does **not** look up boundaries.\n\n"
        "- **Unknown division:** check the first box below → **blank Area Name**. Training rows never "
        "used empty names; the fitted **OneHotEncoder** treats it as an unknown category "
        "(`handle_unknown='ignore'`), so there is **no division one-hot signal**—**lat/lon still "
        "carry geography**.\n"
        "- **Unknown district:** check the second box → **-1**, consistent with the dataset’s unknown "
        "district encoding."
    )
    unknown_division = st.checkbox(
        "I don’t know the LAPD division (Area Name) — use a **blank** division label for the model",
        value=False,
        key="lapd_unknown_division",
        help=(
            "Avoids picking a wrong division from the menu; encoder ignores unknown area categories."
        ),
    )
    unknown_district = st.checkbox(
        "I don’t know the reporting district — use **-1** (recommended after picking a location on the map)",
        value=True,
        key="lapd_unknown_district",
        help="Training collapses rare districts; -1 is an explicit unknown code.",
    )

    a1, a2 = st.columns(2)
    with a1:
        if unknown_division:
            area_name = ""
            st.selectbox(
                "Area name (LAPD division)",
                ["(unknown — blank sent to model)"],
                index=0,
                disabled=True,
                key="lapd_area_disabled_display",
                help="Uncheck “unknown division” above to choose a division from the list.",
            )
            st.caption(
                "**Blank Area Name** → no division one-hot; prediction still uses **coordinates** and "
                "other fields."
            )
        else:
            area_name = _selectbox_with_default(
                "Area name (LAPD division)",
                areas,
                "Central",
                "lapd_area",
                help=(
                    "Not inferred from the map pin. If unsure, use the unknown-division checkbox instead "
                    "of guessing."
                ),
            )
            st.caption("_Division label only if known; coordinates above drive geography._")
    with a2:
        if unknown_district:
            district = "-1"
            st.selectbox(
                "Reporting district",
                ["-1"],
                index=0,
                disabled=True,
                key="lapd_district_disabled_display",
                help="Uncheck the box above to choose a district from the training vocabulary.",
            )
            st.caption("Using **-1** — unknown / not applicable.")
        else:
            district = _selectbox_with_default(
                "Reporting district",
                districts,
                "-1",
                "lapd_district",
                help="District code as in the dataset.",
            )
    p1, p2 = st.columns(2)
    with p1:
        premise = _selectbox_with_default(
            "Premise",
            premises,
            "STREET",
            "lapd_premise",
            help="Premise description (top categories from CSV).",
        )
    with p2:
        weapon = _selectbox_with_default(
            "Weapon",
            weapons,
            "STRONG-ARM",
            "lapd_weapon",
            help="Weapon category from CSV.",
        )

    st.subheader("3 · Date & time")
    d1, d2 = st.columns(2)
    with d1:
        date_in = st.text_input("Date", value="03/15/2016", help="MM/DD/YYYY")
    with d2:
        st.markdown("**Time of day**")
        th1, th2, th3 = st.columns(3)
        with th1:
            hour12 = st.selectbox(
                "Hour",
                list(range(1, 13)),
                index=9,
                key="lapd_hour12",
                help="1–12",
            )
        with th2:
            ampm = st.selectbox(
                "AM / PM",
                ["AM", "PM"],
                index=1,
                key="lapd_ampm",
            )
        with th3:
            minute = st.number_input(
                "Minute",
                min_value=0,
                max_value=59,
                value=0,
                step=1,
                key="lapd_minute",
            )
        time_hhmm = _hour12_ampm_to_hhmm(int(hour12), str(ampm), int(minute))
        st.caption(f"Encoded for the model as **{time_hhmm[:2]}:{time_hhmm[2:]}** (24-hour).")

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
            time_hhmm=time_hhmm,
        )
    except ValueError as e:
        st.error(f"Check **date** (use MM/DD/YYYY) and **time** fields. ({e})")
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
