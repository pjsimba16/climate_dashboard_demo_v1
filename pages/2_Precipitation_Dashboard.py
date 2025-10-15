# pages/2_Precipitation_Dashboard.py
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Click-events package (fallback to plain plotly_chart if unavailable)
try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    plotly_events = None
    PLOTLY_EVENTS_AVAILABLE = False

# =========================
# Page defaults
# =========================
st.set_page_config(
    page_title="Precipitation — Global Database of Subnational Climate Indicators",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# Helpers + caching
# =========================
def _find_file(rel_path: str):
    for base in (".", "data"):
        p = os.path.join(base, rel_path)
        if os.path.exists(p):
            return p
    return None

def _read_table(name_or_path: str) -> pd.DataFrame:
    stem = os.path.splitext(name_or_path)[0]
    for ext in (".snappy.parquet", ".zstd.parquet", ".parquet"):
        for base in ("parquet", "."):
            candidate = os.path.join(base, f"{stem}{ext}")
            if os.path.exists(candidate):
                try:
                    return pd.read_parquet(candidate, engine="pyarrow")
                except Exception:
                    pass
    for ext in (".csv",):
        p = _find_file(stem + ext)
        if p:
            try:
                return pd.read_csv(p)
            except Exception:
                try:
                    return pd.read_csv(p, encoding="latin-1")
                except Exception:
                    pass
    return pd.DataFrame()

try:
    import pycountry
except Exception:
    pycountry = None

def _norm(s): return str(s).strip()

def _country_to_iso3(name: str) -> str:
    if not name or pycountry is None:
        return ""
    try:
        c = pycountry.countries.lookup(str(name).strip())
        return getattr(c, "alpha_3", "") or ""
    except Exception:
        return ""

def _iso3_col(df: pd.DataFrame):
    for c in df.columns:
        if c.strip().lower() in {"iso3", "iso_a3"}:
            return c
    if "Country" in df.columns:
        smp = df["Country"].dropna().astype(str).head(20).tolist()
        if smp and all(len(x.strip()) == 3 and x.strip().isalpha() for x in smp):
            return "Country"
    return None

# =========================
# Nav helpers
# =========================
def go_home():
    try: st.query_params.clear()
    except Exception: pass
    try:
        st.switch_page("Home_Page.py")
    except Exception:
        st.rerun()

# =========================
# Header
# =========================
col_back, _ = st.columns([0.2, 0.8])
with col_back:
    if st.button("← Home"):
        go_home()

# --- Hugging Face data loader ---
from pathlib import Path
from typing import Dict
from huggingface_hub import hf_hub_download
try:
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    from huggingface_hub.errors import HfHubHTTPError

HF_REPO_ID = "pjsimba16/adb_climate_dashboard_v1"
HF_REPO_TYPE = "space"

def _get_hf_token():
    try:
        if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return os.getenv("HF_TOKEN", None)

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _download_from_hub(filename: str,
                       repo_id: str = HF_REPO_ID,
                       repo_type: str = HF_REPO_TYPE) -> str:
    token = _get_hf_token()
    try:
        return hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            token=token,
        )
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"Could not download '{filename}' from '{repo_id}' ({repo_type}). HTTP: {e}")

def _read_any_table(local_path: str, **read_kwargs) -> pd.DataFrame:
    ext = Path(local_path).suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(local_path, **read_kwargs)
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else read_kwargs.pop("sep", ",")
        return pd.read_csv(local_path, sep=sep, **read_kwargs)
    if ext == ".feather":
        return pd.read_feather(local_path, **read_kwargs)
    if ext == ".json":
        try:
            return pd.read_json(local_path, lines=True, **read_kwargs)
        except ValueError:
            return pd.read_json(local_path, **read_kwargs)
    raise ValueError(f"Unsupported file extension for '{local_path}'")

@st.cache_data(ttl=24 * 3600)
def read_hf_table(filename: str) -> pd.DataFrame:
    return _read_any_table(_download_from_hub(filename))

@st.cache_data(ttl=24 * 3600)
def load_many_from_hf(files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    return {k: read_hf_table(v) for k, v in files.items()}

st.markdown("### Country Dashboard — Precipitation")

# =========================
# Load precipitation data (cached)
# =========================
FILES = {
    "country_pr":   "country_precipitation.snappy.parquet",
    "city_pr":      "city_precipitation.snappy.parquet",
    "city_mapper":  "city_mapper_with_coords_v2.snappy.parquet",
}
dfs = load_many_from_hf(FILES)
df_country_p = dfs["country_pr"]
df_city_p    = dfs["city_pr"]
df_mapper    = dfs["city_mapper"]

# =========================
# ISO3 selection (URL/session + dropdown)
# =========================
qp = st.query_params or {}
iso3_in = str(qp.get("iso3", "")) if ("iso3" in qp) else str(st.session_state.get("nav_iso3", ""))
iso3_in = iso3_in.upper().strip()

iso_set = set()
for df in (df_country_p, df_city_p, df_mapper):
    if df is None or df.empty:
        continue
    icol = _iso3_col(df)
    if icol:
        iso_set.update(df[icol].dropna().astype(str).str.upper().str.strip().unique())
    elif "Country" in df.columns:
        for nm in df["Country"].dropna().astype(str):
            iso = _country_to_iso3(nm)
            if iso:
                iso_set.add(iso)

iso_list = sorted(iso_set) if iso_set else ([iso3_in] if iso3_in else [])
current = iso3_in if iso3_in in iso_list else (iso_list[0] if iso_list else "")
new_iso = st.selectbox(
    "Select country (ISO3)",
    options=iso_list if iso_list else [current],
    index=(iso_list.index(current) if iso_list and current in iso_list else 0)
)

if new_iso and new_iso != iso3_in:
    try:
        st.query_params.update({"page": "2 Precipitation Dashboard", "iso3": new_iso})
        st.session_state["nav_iso3"] = new_iso
        st.rerun()
    except Exception:
        pass

iso3 = new_iso or iso3_in
if not iso3:
    st.warning("Select a country (ISO3) to begin.")
    st.stop()

# =========================
# Cached helpers for city list & mapper per ISO3
# =========================
@st.cache_data(show_spinner=False)
def _cities_with_pdata(df_city: pd.DataFrame, iso3_val: str):
    need = {"City","Date","Precipitation (Sum)"}
    if df_city.empty or not need <= set(df_city.columns):
        return []
    c_iso = _iso3_col(df_city)
    if c_iso:
        tmp = df_city[df_city[c_iso].astype(str).str.upper().str.strip() == iso3_val]
    elif "Country" in df_city.columns:
        tmp = df_city[df_city["Country"].astype(str).map(_country_to_iso3) == iso3_val]
    else:
        return []
    if tmp.empty:
        return []
    return sorted(tmp["City"].dropna().astype(str).map(_norm).unique().tolist())

@st.cache_data(show_spinner=False)
def _mapper_for_iso(df_map_all: pd.DataFrame, iso3_val: str, city_col: str, lat_c: str, lon_c: str):
    if df_map_all.empty or not all([city_col, lat_c, lon_c]):
        return pd.DataFrame()
    iso_map_c = _iso3_col(df_map_all)
    if iso_map_c:
        dm = df_map_all[df_map_all[iso_map_c].astype(str).str.upper().str.strip() == iso3_val].copy()
    elif "Country" in df_map_all.columns:
        nm = df_map_all["Country"].astype(str).map(_country_to_iso3)
        dm = df_map_all[nm == iso3_val].copy()
    else:
        return pd.DataFrame()
    return dm.dropna(subset=[lat_c, lon_c])

lat_col = next((c for c in df_mapper.columns if c.strip().lower() in {"lat","latitude"}), None) if not df_mapper.empty else None
lon_col = next((c for c in df_mapper.columns if c.strip().lower() in {"lon","lng","longitude"}), None) if not df_mapper.empty else None
city_map_col = next((c for c in df_mapper.columns if c.strip().lower() in {"city","region","province","state","admin1"}), None) if not df_mapper.empty else None

cities_with_data = _cities_with_pdata(df_city_p, iso3)
df_map = _mapper_for_iso(df_mapper, iso3, city_map_col, lat_col, lon_col)
if not df_map.empty and city_map_col and cities_with_data:
    df_map = df_map[df_map[city_map_col].astype(str).map(_norm).isin(cities_with_data)]

# =========================
# Layout
# =========================
left, right = st.columns([0.25, 0.75], gap="large")

# ---------- LEFT ----------
with left:
    st.markdown("#### Regions / Cities")
    base = pd.DataFrame({"iso3":[iso3], "val":[1]})
    fig_map = px.choropleth(
        base, locations="iso3", color="val",
        locationmode="ISO-3", projection="equirectangular",
        color_continuous_scale=[[0, "#E5F2FF"], [1, "#2274A5"]],
        range_color=(0,1)
    )
    fig_map.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=0,t=0,b=0), height=340)
    fig_map.update_geos(fitbounds="locations", visible=False)

    if not df_map.empty and city_map_col and lat_col and lon_col:
        fig_map.add_trace(go.Scattergeo(
            lon=df_map[lon_col],
            lat=df_map[lat_col],
            mode="markers",
            name="Cities",
            marker=dict(size=8, color="red", line=dict(width=0.5, color="#333")),
            text=df_map[city_map_col].astype(str),
            customdata=df_map[city_map_col].astype(str),
            hovertemplate="%{text}<extra></extra>",
        ))

    clicked_city = None
    if PLOTLY_EVENTS_AVAILABLE:
        ev = plotly_events(
            fig_map,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=340,
            override_width="100%"
        )
    else:
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
        ev = []

    if ev:
        e0 = ev[0]
        if "customdata" in e0 and isinstance(e0["customdata"], (str, list)):
            clicked_city = e0["customdata"] if isinstance(e0["customdata"], str) else e0["customdata"][0]
        elif "text" in e0 and isinstance(e0["text"], str):
            clicked_city = e0["text"]
        elif "pointNumber" in e0 and not df_map.empty:
            pn = e0["pointNumber"]
            if isinstance(pn, int) and 0 <= pn < len(df_map):
                clicked_city = str(df_map.iloc[pn][city_map_col])

    st.markdown("#### Select Climate Indicator:")
    INDICATOR_OPTIONS = [
        "Temperature",
        "Precipitation",
        "Temperature Thresholds",
        "Heatwaves",
        "Coldwaves",
        "Dry Conditions",
        "Wet Conditions",
        "Humidity",
        "Windspeeds",
    ]
    current_indicator = "Precipitation"
    sel_indicator = st.radio(
        "Select one indicator",
        options=INDICATOR_OPTIONS,
        index=INDICATOR_OPTIONS.index(current_indicator)
    )
    st.caption("Temperature and Precipitation are implemented; others are placeholders.")

    if sel_indicator != current_indicator and sel_indicator == "Temperature":
        try:
            st.switch_page("pages/1_Temperature_Dashboard.py")
        except Exception:
            if st.query_params.get("page") != "1 Temperature Dashboard":
                st.query_params.update({"page": "1 Temperature Dashboard", "iso3": iso3})
                st.rerun()

# ---------- RIGHT ----------
with right:
    st.markdown("#### Options")

    if "prec_active_ds"   not in st.session_state: st.session_state["prec_active_ds"]   = "Historical Observations"
    if "prec_active_freq" not in st.session_state: st.session_state["prec_active_freq"] = "Monthly"
    if "prec_active_src"  not in st.session_state: st.session_state["prec_active_src"]  = "ERA5"

    with st.form(key="prec_options_form", clear_on_submit=False):
        fc1, fc2, fc3 = st.columns([1, 1, 1], gap="small")
        with fc1:
            pending_ds = st.radio(
                "Dataset type",
                ["Historical Observations", "Projections (SSP)"],
                index=["Historical Observations","Projections (SSP)"].index(st.session_state["prec_active_ds"]),
                horizontal=True,
                key="prec_form_ds"
            )
        with fc2:
            pending_freq = st.radio(
                "Select Frequency",
                ["Monthly", "Seasonal", "Annual"],
                index=["Monthly","Seasonal","Annual"].index(st.session_state["prec_active_freq"]),
                horizontal=True,
                key="prec_form_freq"
            )
        with fc3:
            pending_src = st.radio(
                "Data Source",
                ["CDS/CCKP", "CRU", "ERA5"],
                index=["CDS/CCKP","CRU","ERA5"].index(st.session_state["prec_active_src"]),
                horizontal=True,
                key="prec_form_src"
            )
        apply_clicked = st.form_submit_button("Apply changes", type="primary")

    if apply_clicked:
        changed = (
            pending_ds   != st.session_state["prec_active_ds"] or
            pending_freq != st.session_state["prec_active_freq"] or
            pending_src  != st.session_state["prec_active_src"]
        )
        if changed:
            st.session_state["prec_active_ds"]   = pending_ds
            st.session_state["prec_active_freq"] = pending_freq
            st.session_state["prec_active_src"]  = pending_src
        else:
            st.info("No changes to apply.", icon="ℹ️")

    # City selector (sync with clicks)
    cities_with_data = cities_with_data or []
    city_options = ["Country (all)"] + cities_with_data
    if "opt_city" not in st.session_state or st.session_state["opt_city"] not in city_options:
        st.session_state["opt_city"] = city_options[0]
    if clicked_city and clicked_city in city_options and st.session_state["opt_city"] != clicked_city:
        st.session_state["opt_city"] = clicked_city
    sel_city = st.selectbox("Select Province/City/State", options=city_options, key="opt_city")

    st.markdown("---")

    # Active-choice badges
    st.markdown(
        f"""<div style="margin-bottom:8px">
                <span style="display:inline-block;padding:2px 8px;border-radius:999px;background:#eef2ff;
                             color:#3730a3;font-size:12px;margin-right:6px;">
                    {st.session_state['prec_active_ds']}
                </span>
                <span style="display:inline-block;padding:2px 8px;border-radius:999px;background:#f0f9ff;
                             color:#075985;font-size:12px;margin-right:6px;">
                    {st.session_state['prec_active_freq']}
                </span>
                <span style="display:inline-block;padding:2px 8px;border-radius:999px;background:#ecfdf5;
                             color:#065f46;font-size:12px;">
                    {st.session_state['prec_active_src']}
                </span>
            </div>""",
        unsafe_allow_html=True
    )

    # =========================
    # Series builders (Sum + Var)
    # =========================
    @st.cache_data(show_spinner=False)
    def _country_series_p(df_country: pd.DataFrame, iso3_val: str) -> pd.DataFrame:
        if df_country.empty:
            return pd.DataFrame()
        c_iso = _iso3_col(df_country)
        if c_iso:
            src = df_country[df_country[c_iso].astype(str).str.upper().str.strip() == iso3_val].copy()
        elif "Country" in df_country.columns:
            src = df_country[df_country["Country"].astype(str).map(_country_to_iso3) == iso3_val].copy()
        else:
            src = pd.DataFrame()
        need = {"Date","Precipitation (Sum)"}
        if src.empty or not need.issubset(src.columns):
            return pd.DataFrame()
        out = pd.DataFrame({
            "date": pd.to_datetime(src["Date"], errors="coerce"),
            "sum":  pd.to_numeric(src["Precipitation (Sum)"], errors="coerce"),
        })
        if "Precipitation (Variance)" in src.columns:
            out["var"] = pd.to_numeric(src["Precipitation (Variance)"], errors="coerce")
        return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    @st.cache_data(show_spinner=False)
    def _city_series_p(df_city: pd.DataFrame, iso3_val: str, city_name: str) -> pd.DataFrame:
        need = {"City","Date","Precipitation (Sum)"}
        if df_city.empty or not need <= set(df_city.columns):
            return pd.DataFrame()
        c_iso = _iso3_col(df_city)
        if c_iso:
            tmp = df_city[df_city[c_iso].astype(str).str.upper().str.strip() == iso3_val].copy()
        elif "Country" in df_city.columns:
            tmp = df_city[df_city["Country"].astype(str).map(_country_to_iso3) == iso3_val].copy()
        else:
            tmp = pd.DataFrame()
        if tmp.empty:
            return pd.DataFrame()
        tmp = tmp[tmp["City"].astype(str).map(_norm) == _norm(city_name)].copy()
        if tmp.empty:
            return pd.DataFrame()
        out = pd.DataFrame({
            "date": pd.to_datetime(tmp["Date"], errors="coerce"),
            "sum":  pd.to_numeric(tmp["Precipitation (Sum)"], errors="coerce"),
        })
        if "Precipitation (Variance)" in tmp.columns:
            out["var"] = pd.to_numeric(tmp["Precipitation (Variance)"], errors="coerce")
        return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    series = _city_series_p(df_city_p, iso3, sel_city) if sel_city != "Country (all)" else _country_series_p(df_country_p, iso3)
    if series.empty:
        st.warning("No precipitation data found for this selection.")
        st.stop()

    # Date range slider
    dmin, dmax = series["date"].min(), series["date"].max()
    date_from, date_to = st.slider(
        "Date range",
        min_value=pd.to_datetime(dmin).to_pydatetime(),
        max_value=pd.to_datetime(dmax).to_pydatetime(),
        value=(pd.to_datetime(dmin).to_pydatetime(), pd.to_datetime(dmax).to_pydatetime()),
        key="p_date_range"
    )
    mask = (series["date"] >= pd.to_datetime(date_from)) & (series["date"] <= pd.to_datetime(date_to))
    series = series.loc[mask].reset_index(drop=True)

    # Chart helper (sum + variance on secondary axis)
    def chart_sum_var(title: str, s: pd.DataFrame):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s["date"], y=s["sum"], mode="lines", name="Precipitation (sum)"))
        if "var" in s.columns and s["var"].notna().any():
            fig.add_trace(go.Scatter(x=s["date"], y=s["var"], mode="lines", name="Variance", yaxis="y2"))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Variance"))
        fig.update_layout(height=280, margin=dict(l=20,r=20,t=30,b=20),
                          title=title, xaxis_title="Date", yaxis_title="mm")
        return fig

    # 2×2 grid (placeholders included)
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(chart_sum_var("Total Precipitation", series), use_container_width=True)
    with g2:
        st.plotly_chart(chart_sum_var("Precipitation Intensity (placeholder)", series), use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        st.plotly_chart(chart_sum_var("Wet Conditions (placeholder)", series), use_container_width=True)
    with g4:
        st.plotly_chart(chart_sum_var("Dry Conditions (placeholder)", series), use_container_width=True)

    st.markdown("—")

    # Percentiles
    pct = st.slider("Select percentile for the charts below", min_value=10, max_value=90, value=50, step=1, key="p_pct")

    def percentile_series(s: pd.DataFrame, val_col: str, pct_value: int) -> pd.DataFrame:
        dfp = s.dropna(subset=["date", val_col]).copy()
        if dfp.empty:
            return pd.DataFrame()
        dfp["month"] = dfp["date"].dt.month
        ref = dfp.groupby("month")[val_col].quantile(pct_value/100.0)
        mapped = dfp["month"].map(ref)
        return pd.DataFrame({"date": dfp["date"], "p": mapped})

    def percentile_chart(title: str, s: pd.DataFrame):
        ps = percentile_series(s, "sum", pct)
        if ps.empty:
            st.warning(f"No data for {title.lower()}.")
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ps["date"], y=ps["p"], mode="lines", name=f"P{pct}"))
        fig.update_layout(height=280, margin=dict(l=20,r=20,t=30,b=20),
                          title=title, xaxis_title="Date", yaxis_title="mm")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1: percentile_chart("Percentile — Total Precipitation", series)
    with c2: percentile_chart("Percentile — Wet Conditions (placeholder)", series)
    with c3: percentile_chart("Percentile — Dry Conditions (placeholder)", series)
