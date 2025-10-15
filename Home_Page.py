# Home_Page.py
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import streamlit.components.v1 as components

# =========================
# Page & global style
# =========================
st.set_page_config(
    page_title="Home Page — Global Database of Subnational Climate Indicators",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Let the main content take (almost) the full viewport width
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

@st.cache_data(show_spinner=False)
def _read_csv(fname: str) -> pd.DataFrame:
    p = _find_file(fname)
    if not p:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, encoding="latin-1")
        except Exception:
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

def _find_file(rel_path: str):
    for base in (".", "data"):
        p = os.path.join(base, rel_path)
        if os.path.exists(p):
            return p
    return None

def _read_table(name_or_path: str) -> pd.DataFrame:
    """
    Smart reader:
      - If user passes 'country_temperature.csv' → try parquet/country_temperature.parquet (snappy or zstd) first, else CSV file.
      - If user passes 'country_temperature' (no ext) → same behavior.
    Looks in './parquet' first, then in '.' and './data'.
    """
    stem = os.path.splitext(name_or_path)[0]  # strip extension if any, e.g., 'country_temperature'
    # 1) Try Parquet in ./parquet (snappy -> zstd)
    for ext in (".snappy.parquet", ".zstd.parquet", ".parquet"):
        for base in ("parquet", "."):
            candidate = os.path.join(base, f"{stem}{ext}")
            if os.path.exists(candidate):
                try:
                    return pd.read_parquet(candidate, engine="pyarrow")
                except Exception:
                    pass
    # 2) Try CSV in '.' or './data'
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
    # If all fails
    return pd.DataFrame()

# --- Hugging Face data loader for Streamlit ---
# Usage:
#   df = read_hf_table("country_temperature.parquet")
#   dfs = load_many_from_hf({
#       "country_temp": "country_temperature.parquet",
#       "country_pr":   "country_precipitation.parquet",
#       "city_temp":    "city_temperature.parquet",
#       "city_pr":      "city_precipitation.parquet",
#       "city_mapper":  "city_mapper_with_coords_v2.csv",
#   })

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
try:
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    from huggingface_hub.errors import HfHubHTTPError



# Your Space slug (repo ID)
HF_REPO_ID = "pjsimba16/adb_climate_dashboard_v1"
HF_REPO_TYPE = "space"  # <- important when files live in a Space (not a Datasets repo)


def _get_hf_token():
    """
    Return an HF token if one is configured, else None.
    Works locally even when no secrets.toml exists.
    """
    # 1) Prefer Streamlit secrets if available (e.g., on Streamlit Cloud)
    try:
        # Accessing st.secrets can raise if there's no secrets.toml
        if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass

    # 2) Fallback: environment variable (optional)
    return os.getenv("HF_TOKEN", None)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _download_from_hub(filename: str,
                       repo_id: str = HF_REPO_ID,
                       repo_type: str = HF_REPO_TYPE) -> str:
    """
    Download a single file from the Hugging Face Hub and return its local cached path.
    Caches across reruns thanks to HF cache + st.cache_data.
    """
    token = _get_hf_token()
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            token=token,
        )
    except HfHubHTTPError as e:
        raise FileNotFoundError(
            f"Could not download '{filename}' from '{repo_id}' ({repo_type}). "
            f"HTTP error: {e}"
        )
    return local_path


def _read_any_table(local_path: str, **read_kwargs) -> pd.DataFrame:
    """
    Read a tabular file based on extension.
    Supports: .parquet, .csv, .feather, .json (records/lines), .tsv
    """
    ext = Path(local_path).suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(local_path, **read_kwargs)
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else read_kwargs.pop("sep", ",")
        return pd.read_csv(local_path, sep=sep, **read_kwargs)
    if ext == ".feather":
        return pd.read_feather(local_path, **read_kwargs)
    if ext == ".json":
        # Try JSON lines first; fall back to standard JSON
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
    """
    High-level helper: download a file from HF Hub (Space or Dataset) and return a DataFrame.

    Parameters
    ----------
    filename : str
        File path inside the repo (e.g., 'country_temperature.parquet').
    repo_id : str
        HF repo id, default is your Space 'pjsimba16/adb_climate_dashboard_v1'.
    repo_type : str
        'space' (your case) or 'dataset' (if you later move data into a Datasets repo).
    **read_kwargs :
        Extra kwargs for pandas readers (e.g., dtype=..., usecols=..., engine=...).

    Returns
    -------
    pd.DataFrame
    """
    local_path = _download_from_hub(filename, repo_id=repo_id, repo_type=repo_type)
    return _read_any_table(local_path, **read_kwargs)


@st.cache_data(ttl=24 * 3600)
def load_many_from_hf(files: Dict[str, str],
                      repo_id: str = HF_REPO_ID,
                      repo_type: str = HF_REPO_TYPE,
                      **read_kwargs) -> Dict[str, pd.DataFrame]:
    """
    Convenience: load multiple files at once.

    files : mapping of {key: filename}
        Example: {'country_temp': 'country_temperature.parquet', ...}
    Returns mapping {key: DataFrame}
    """
    out = {}
    for key, fname in files.items():
        out[key] = read_hf_table(fname, repo_id=repo_id, repo_type=repo_type, **read_kwargs)
    return out


# =========================
# Title & subtitle (preserved)
# =========================
st.markdown("<h1 style='text-align:center'>Global Database of Subnational Climate Indicators</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Built and Maintained by Roshen Fernando and Patrick Jaime Simba</div>", unsafe_allow_html=True)
st.divider()

# =========================
# Load availability from current indicators
# =========================
#country_temp = _read_csv("country_temperature.csv")
#city_temp    = _read_csv("city_temperature.csv")
#country_prec = _read_csv("country_precipitation.csv")
#city_prec    = _read_csv("city_precipitation.csv")

#country_temp = _read_table("country_temperature")
#city_temp    = _read_table("city_temperature")
#country_prec = _read_table("country_precipitation")
#city_prec    = _read_table("city_precipitation")
#df_mapper    = _read_table("city_mapper_with_coords_v2")

FILES = {
    "country_temp": "country_temperature.snappy.parquet",
    "country_pr":   "country_precipitation.snappy.parquet",
    "city_temp":    "city_temperature.snappy.parquet",
    "city_pr":      "city_precipitation.snappy.parquet",
    "city_mapper":  "city_mapper_with_coords_v2.snappy.parquet",
}

dfs = load_many_from_hf(FILES)
country_temp = dfs["country_temp"]
country_prec   = dfs["country_pr"]
city_temp    = dfs["city_temp"]
city_prec      = dfs["city_pr"]
df_mapper     = dfs["city_mapper"]



def _isos_with_indicator(country_df, city_df, country_col_name=None):
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

# Union for coloring the map
iso_with_data = set().union(iso_temp, iso_prec)

# All countries for the map
if pycountry:
    all_countries = pd.DataFrame(
        [{"iso3": c.alpha_3, "name": c.name} for c in pycountry.countries if hasattr(c, "alpha_3")]
    )
else:
    all_countries = pd.DataFrame({"iso3": sorted(list(iso_with_data))})
    all_countries["name"] = all_countries["iso3"]

# Build indicator badges (for hover)
def _badges_for_iso(iso3: str):
    tags = []
    if iso3 in iso_temp: tags.append("Temperature")
    if iso3 in iso_prec: tags.append("Precipitation")
    return " • ".join(tags)

all_countries["has_data"]  = all_countries["iso3"].isin(iso_with_data)
all_countries["badges"]    = all_countries["iso3"].map(_badges_for_iso)
all_countries["hovertext"] = all_countries.apply(
    lambda r: (f"{r['name']}<br><span>Indicators: {r['badges']}</span>") if r["has_data"]
              else (f"{r['name']}<br><span>No available indicators</span>"),
    axis=1
)

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

# Quick search (optional jump)
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
# Dynamic sizing: read viewport (single value; no loops)
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
    height=0,  # invisible
)

if isinstance(vp, dict) and "width" in vp and "height" in vp:
    vw, vh = int(vp["width"]), int(vp["height"])
    # Height ~ 52% of viewport width, clamped between 540px and 84% of viewport height
    map_h = max(540, min(int(0.52 * vw), int(0.84 * vh)))
else:
    map_h = 720

# =========================
# World map (neutral style + bolder borders + badges in hover)
# =========================
color_vals = all_countries["has_data"].astype(int)  # 0 or 1
fig = px.choropleth(
    all_countries,
    locations="iso3",
    locationmode="ISO-3",
    color=color_vals,
    color_continuous_scale=[[0, "#d4d4d8"], [1, "#12a39a"]],  # neutral grey -> calm teal
    hover_name="name",
    hover_data={"iso3": False, "has_data": False, "name": False, "hovertext": False, "badges": False},
    projection="equirectangular",
)

# Clean, informative hover with badges
fig.update_traces(
    hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
    customdata=all_countries[["hovertext"]]
)

fig.update_layout(
    coloraxis_showscale=False,
    height=map_h,
    autosize=True,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="#f8fafc",
    plot_bgcolor="#f8fafc",
)

# Distinct country borders + neutral oceans/land
fig.update_geos(
    fitbounds="locations",
    visible=False,
    showcountries=True,
    countrycolor="#475569",   # darker slate border
    countrywidth=1.2,         # <--- bolder borders
    showocean=True,
    oceancolor="#eef2f7",
    showland=True,
    landcolor="#f8fafc",
    showsubunits=False
)

# Card wrapper for visual elevation
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
events = plotly_events(
    fig,
    click_event=True,
    hover_event=False,
    select_event=False,
    override_height=map_h,
    override_width="100%",
)
st.markdown("</div>", unsafe_allow_html=True)

# Status chips below map (mini legend + coverage count)
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
# Click handling: direct navigation (no query-param loop)
# =========================
def go_country_page(iso3: str, country_name: str = ""):
    st.session_state["nav_iso3"] = (iso3 or "").upper().strip()
    st.session_state["nav_country"] = country_name or ""
    try:
        st.switch_page("pages/1_Temperature_Dashboard.py")
    except Exception:
        st.query_params.update({"page": "1 Temperature Dashboard", "iso3": st.session_state["nav_iso3"]})
        st.stop()

if events:
    pn = events[0].get("pointNumber")
    if isinstance(pn, int) and 0 <= pn < len(all_countries):
        clicked_iso3 = all_countries.iloc[pn]["iso3"]
        clicked_name = all_countries.iloc[pn]["name"]
        if clicked_iso3 in iso_with_data:
            go_country_page(clicked_iso3, clicked_name)
        else:
            st.info(f"{clicked_name}: No available indicators.", icon="ℹ️")

# =========================
# Footer: logos / acknowledgements / disclaimers / links (preserved)
# =========================
st.markdown("""
<div class="footer-box">
  <strong>Acknowledgements & Partners (placeholder)</strong><br/>
  Funding agency • Partner institutions • Data providers (ERA5, CRU, CDS/CCKP) • Other acknowledgements.<br/><br/>
  <em>Disclaimer:</em> This dashboard is provided "as is" without warranty. Data sources may update over time.<br/><br/>
  <a href="#">Project website</a> • <a href="#">Documentation</a> • <a href="#">Contact</a>
</div>
""", unsafe_allow_html=True)
