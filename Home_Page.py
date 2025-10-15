# Home_Page.py
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px  # kept for convenience
import streamlit.components.v1 as components
from pathlib import Path
from typing import Dict

# Try to use the click-events package; fall back gracefully if unavailable
try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    plotly_events = None
    PLOTLY_EVENTS_AVAILABLE = False

with st.sidebar.expander("🧰 Environment", expanded=False):
    import sys, platform
    import plotly
    st.write("Python:", sys.version)
    st.write("Platform:", platform.platform())
    st.write("plotly:", plotly.__version__)
    try:
        import streamlit_plotly_events as spe
        st.write("streamlit-plotly-events:", getattr(spe, "__version__", "unknown"))
    except Exception as e:
        st.write("streamlit-plotly-events: not importable", str(e))

# =========================
# Page & global style
# =========================
st.set_page_config(
    page_title="Home Page — Global Database of Subnational Climate Indicators",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
  .block-container {
    max-width: 98vw !important;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
  }
  .map-wrap { border-radius: 14px; overflow: hidden; }
  .subtitle { margin-top: -0.5rem; color: #475569; font-size: 0.95rem; text-align: center; }
  .footer-box { border-top: 1px solid #e5e7eb; padding-top: 1rem; margin-top: 1.25rem; color:#334155; }
</style>
""", unsafe_allow_html=True)

# =========================
# Utilities
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

def _to_iso3(name_or_iso):
    if not name_or_iso:
        return ""
    s = str(name_or_iso).strip()
    if len(s) == 3 and s.isalpha():
        return s.upper()
    if pycountry is None:
        return ""
    try:
        c = pycountry.countries.lookup(s)
        return getattr(c, "alpha_3", "").upper()
    except Exception:
        return ""

def _iso3_col(df: pd.DataFrame):
    for c in df.columns:
        if c.strip().lower() in {"iso3", "iso_a3"}:
            return c
    if "Country" in df.columns:
        sample = df["Country"].dropna().astype(str).head(20).tolist()
        if sample and all(len(x.strip()) == 3 and x.strip().isalpha() for x in sample):
            return "Country"
    return None

# --- Hugging Face data loader ---
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
def read_hf_table(filename: str,
                  repo_id: str = HF_REPO_ID,
                  repo_type: str = HF_REPO_TYPE,
                  **read_kwargs) -> pd.DataFrame:
    local_path = _download_from_hub(filename, repo_id=repo_id, repo_type=repo_type)
    return _read_any_table(local_path, **read_kwargs)

@st.cache_data(ttl=24 * 3600)
def load_many_from_hf(files: Dict[str, str],
                      repo_id: str = HF_REPO_ID,
                      repo_type: str = HF_REPO_TYPE,
                      **read_kwargs) -> Dict[str, pd.DataFrame]:
    return {k: read_hf_table(v, repo_id=repo_id, repo_type=repo_type, **read_kwargs) for k, v in files.items()}

# --- availability snapshot loader: local first, then HF fallback ---
def _read_availability_snapshot_local():
    """Look for the snapshot next to this file and in common folders."""
    here = Path(__file__).parent.resolve()
    candidates = [
        here / "availability_snapshot.parquet",
        here / "data" / "availability_snapshot.parquet",
        here / "parquet" / "availability_snapshot.parquet",
        here / "availability_snapshot.csv",
        here / "data" / "availability_snapshot.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                snap = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
                st.caption(f"Loaded availability snapshot from: {p}")
                return snap
            except Exception as e:
                st.caption(f"Found snapshot at {p} but failed to read: {e}")
    return None

@st.cache_data(ttl=24*3600, show_spinner=False)
def _read_availability_snapshot_hf(
    filename_parquet="availability_snapshot.parquet",
    filename_csv="availability_snapshot.csv",
    repo_id=HF_REPO_ID,
    repo_type=HF_REPO_TYPE,
):
    token = _get_hf_token()
    # Prefer parquet in /parquet/
    try:
        p = hf_hub_download(repo_id=repo_id, repo_type=repo_type,
                            filename=f"parquet/{filename_parquet}", token=token)
        return pd.read_parquet(p)
    except Exception:
        pass
    # Fallback to CSV in /data/ or repo root
    for fname in (f"data/{filename_csv}", filename_csv):
        try:
            p = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=fname, token=token)
            return pd.read_csv(p)
        except Exception:
            continue
    return None

def _load_availability_snapshot():
    snap = _read_availability_snapshot_local()
    if snap is not None:
        return snap, "snapshot (local)"
    snap = _read_availability_snapshot_hf()
    if snap is not None:
        return snap, "snapshot (HF)"
    return None, None

# =========================
# Title & subtitle
# =========================
st.markdown("<h1 style='text-align:center'>Global Database of Subnational Climate Indicators</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Built and Maintained by Roshen Fernando and Patrick Jaime Simba</div>", unsafe_allow_html=True)
st.divider()

# =========================
# Load availability (prefer snapshot; else compute from HF datasets)
# =========================
snap, snap_origin = _load_availability_snapshot()

if snap is not None and {"iso3","has_temp","has_prec"} <= set(snap.columns):
    iso_temp = set(snap.loc[snap["has_temp"] == 1, "iso3"].astype(str).str.upper())
    iso_prec = set(snap.loc[snap["has_prec"] == 1, "iso3"].astype(str).str.upper())
    iso_with_data = iso_temp | iso_prec
    country_temp = country_prec = city_temp = city_prec = pd.DataFrame()
else:
    FILES = {
        "country_temp": "country_temperature.snappy.parquet",
        "country_pr":   "country_precipitation.snappy.parquet",
        "city_temp":    "city_temperature.snappy.parquet",
        "city_pr":      "city_precipitation.snappy.parquet",
    }
    dfs = load_many_from_hf(FILES)
    country_temp = dfs["country_temp"]
    country_prec = dfs["country_pr"]
    city_temp    = dfs["city_temp"]
    city_prec    = dfs["city_pr"]

    def _isos_with_indicator(country_df, city_df):
        s = set()
        for df in (country_df, city_df):
            if df is None or df.empty:
                continue
            icol = _iso3_col(df)
            if icol:
                s.update(df[icol].dropna().astype(str).str.upper().str.strip().unique())
            elif "Country" in df.columns:
                s.update(df["Country"].dropna().astype(str).map(_to_iso3))
        s.discard("")
        return s

    iso_temp = _isos_with_indicator(country_temp, city_temp)
    iso_prec = _isos_with_indicator(country_prec, city_prec)
    iso_with_data = iso_temp | iso_prec
    snap_origin = "computed (HF)"

st.caption(f"Availability source: {snap_origin or 'computed (HF)'}")

# Sidebar debug (availability)
with st.sidebar.expander("⚙️ Availability debug", expanded=False):
    st.write("iso_temp:", len(iso_temp))
    st.write("iso_prec:", len(iso_prec))
    st.write("iso_with_data:", len(iso_with_data))
    st.write("origin:", snap_origin)

# =========================
# Build list of all countries & indicator badges
# =========================
if pycountry:
    all_countries = pd.DataFrame(
        [{"iso3": c.alpha_3, "name": c.name} for c in pycountry.countries if hasattr(c, "alpha_3")]
    )
else:
    all_countries = pd.DataFrame({"iso3": sorted(list(iso_with_data))})
    all_countries["name"] = all_countries["iso3"]

def _badges_for_iso(iso3: str):
    tags = []
    if iso3 in iso_temp: tags.append("Temperature")
    if iso3 in iso_prec: tags.append("Precipitation")
    return " • ".join(tags)

all_countries["iso3"]      = all_countries["iso3"].astype(str).str.upper().str.strip()
all_countries["has_data"]  = all_countries["iso3"].isin(iso_with_data)
all_countries["badges"]    = all_countries["iso3"].map(_badges_for_iso)
all_countries["hovertext"] = all_countries.apply(
    lambda r: (f"{r['name']}<br><span>Indicators: {r['badges']}</span>") if r["has_data"]
              else (f"{r['name']}<br><span>No available indicators</span>"),
    axis=1
)
all_countries["val"]  = all_countries["has_data"].astype("float")  # ensure float for z

# Optional color debug
with st.sidebar.expander("🎨 Color debug", expanded=False):
    st.write("val value counts:", all_countries["val"].value_counts(dropna=False).to_dict())

# =========================
# Instruction banner + Quick search
# =========================
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        padding: 14px 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        display: flex; align-items: center; gap: 14px;">
      <div style="font-size:14.5px; color:#0f172a;">
        <strong>Tip:</strong> Click a country on the map to open its dashboard, or use the quick search.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

col_a, col_b = st.columns([0.58, 0.42])
with col_b:
    quick_opts = ["— Type to search —"] + sorted(all_countries["name"].tolist())
    chosen = st.selectbox("Quick search", options=quick_opts, index=0)
    if chosen and chosen != "— Type to search —":
        row = all_countries.loc[all_countries["name"] == chosen].iloc[0]
        iso3_jump = row["iso3"]
        if iso3_jump in iso_with_data:
            st.session_state["nav_iso3"] = iso3_jump
            st.session_state["nav_country"] = chosen
            try:
                st.switch_page("pages/1_Temperature_Dashboard.py")
            except Exception:
                st.query_params.update({"page": "1 Temperature Dashboard", "iso3": iso3_jump})
                st.stop()
        else:
            st.info(f"{chosen}: No available indicators.", icon="ℹ️")

# =========================
# Dynamic sizing
# =========================
vp = components.html(
    """
    <script>
      (function() {
        function send() {
          const payload = {width: window.innerWidth, height: window.innerHeight};
          window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:setComponentValue', value: payload}, '*');
        }
        window.addEventListener('resize', send);
        send();
      })();
    </script>
    """,
    height=0,
)

if isinstance(vp, dict) and "width" in vp and "height" in vp:
    vw, vh = int(vp["width"]), int(vp["height"])
    map_h = max(540, min(int(0.52 * vw), int(0.84 * vh)))
else:
    map_h = 720

# =========================
# World map using go.Choropleth (geo is VISIBLE on Cloud)
# =========================
fig = go.Figure()

fig.add_trace(go.Choropleth(
    locations=all_countries["iso3"],
    z=all_countries["val"],                  # 0.0/1.0
    locationmode="ISO-3",
    colorscale=[[0.0, "#d4d4d8"], [1.0, "#12a39a"]],
    zmin=0.0, zmax=1.0,
    autocolorscale=False,
    showscale=False,
    hoverinfo="text",
    text=all_countries["hovertext"],
    customdata=all_countries[["hovertext", "iso3"]].to_numpy(),
    hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
    marker_line_color="#475569",
    marker_line_width=1.2,
))

# IMPORTANT: keep the geo VISIBLE and scoped to the world
fig.update_geos(
    scope="world",
    showframe=False,
    showcoastlines=False,
    showcountries=True,
    countrycolor="#475569",
    countrywidth=1.2,
    showocean=True,
    oceancolor="#eef2f7",
    showland=True,
    landcolor="#f8fafc",
)

fig.update_layout(
    height=map_h,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="#f8fafc",
    plot_bgcolor="#f8fafc",
)

# Card wrapper
st.markdown(
    """
    <div class='map-wrap' style="
        background:#ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 18px rgba(2, 6, 23, 0.06);
        border-radius: 14px;
        overflow: hidden;">
    """,
    unsafe_allow_html=True
)

if PLOTLY_EVENTS_AVAILABLE:
    events = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=map_h,
        override_width="100%",
    )
else:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.info("Click-to-open is unavailable on this deployment. Use Quick search.", icon="ℹ️")
    events = []

st.markdown("</div>", unsafe_allow_html=True)

# Status chips
st.markdown(
    f"""
    <div style="display:flex; gap:10px; align-items:center; margin: 10px 4px 6px 2px;">
      <span style="display:inline-flex; align-items:center; gap:6px; font-size:12px; color:#334155;">
        <span style="width:12px;height:12px;background:#d4d4d8;border-radius:2px;display:inline-block;border:1px solid #cbd5e1;"></span>
        No data
      </span>
      <span style="display:inline-flex; align-items:center; gap:6px; font-size:12px; color:#334155;">
        <span style="width:12px;height:12px;background:#12a39a;border-radius:2px;display:inline-block;border:1px solid #0e807a;"></span>
        Available indicators
      </span>
      <span style="font-size:12px; color:#64748b; margin-left:8px;">
        • {len(iso_with_data)} countries with data
      </span>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Click handling: robust ISO3 extraction
# =========================
def go_country_page(iso3: str, country_name: str = ""):
    st.session_state["nav_iso3"] = (iso3 or "").upper().strip()
    st.session_state["nav_country"] = country_name or ""
    try:
        st.switch_page("pages/1_Temperature_Dashboard.py")
    except Exception:
        st.query_params.update({"page": "1 Temperature Dashboard", "iso3": st.session_state["nav_iso3"]})
        st.stop()

clicked_iso3 = None
if events:
    e = events[0]
    if "location" in e and isinstance(e["location"], str):
        clicked_iso3 = e["location"].upper()
    elif "customdata" in e and isinstance(e["customdata"], list) and len(e["customdata"]) >= 2:
        clicked_iso3 = str(e["customdata"][1]).upper()

if clicked_iso3:
    if clicked_iso3 in iso_with_data:
        cname = all_countries.loc[all_countries["iso3"] == clicked_iso3, "name"].iloc[0]
        go_country_page(clicked_iso3, cname)
    else:
        st.info("No available indicators for this country.", icon="ℹ️")

# =========================
# Footer
# =========================
st.markdown("""
<div class="footer-box">
  <strong>Acknowledgements & Partners (placeholder)</strong><br/>
  Funding agency • Partner institutions • Data providers (ERA5, CRU, CDS/CCKP) • Other acknowledgements.<br/><br/>
  <em>Disclaimer:</em> This dashboard is provided "as is" without warranty. Data sources may update over time.<br/><br/>
  <a href="#">Project website</a> • <a href="#">Documentation</a> • <a href="#">Contact</a>
</div>
""", unsafe_allow_html=True)
