
import streamlit as st
import numpy as np
import json
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk
import time
import os
import gzip
from datetime import datetime

st.set_page_config(page_title="Tehran Subsidence Matrix", page_icon="🛸", layout="wide")

st.markdown("""
<style>
    .stApp { background: #05070d; }
    .main-header {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a4e 50%, #0a0e27 100%);
        padding: 1.5rem; border-radius: 12px; color: #00d9ff;
        text-align: center; margin-bottom: 1rem;
        border: 1px solid #00d9ff;
        box-shadow: 0 0 40px rgba(0, 217, 255, 0.3);
        font-family: 'Courier New', monospace;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700;
                     text-shadow: 0 0 20px #00d9ff; letter-spacing: 3px; }
    .main-header p { margin-top: 0.5rem; opacity: 0.9; color: #7ee7ff; }
    .metric-card {
        background: linear-gradient(135deg, #0a0e27 0%, #151a3a 100%);
        padding: 1rem; border-radius: 10px;
        text-align: center; border: 1px solid #00d9ff;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.15);
        color: #00d9ff;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #7ee7ff;
                     font-family: 'Courier New', monospace; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700;
                         margin: 0.3rem 0; color: #00d9ff;
                         text-shadow: 0 0 10px #00d9ff; }
    .metric-card .sub { font-size: 0.8rem; color: #7ee7ff; }
    .section-header {
        background: linear-gradient(90deg, transparent 0%, #00d9ff 50%, transparent 100%);
        color: #05070d; padding: 0.5rem 1rem; border-radius: 6px;
        margin: 1rem 0; font-weight: 700;
        text-align: center; letter-spacing: 3px;
        font-family: 'Courier New', monospace;
        box-shadow: 0 0 20px #00d9ff;
    }
    .stSidebar > div { background: #05070d; }
    .stSidebar label, .stSidebar p, .stSidebar h1, .stSidebar h2, .stSidebar h3,
    .stSidebar .stMarkdown, .stSidebar .stRadio > label { color: #7ee7ff !important; }
</style>
""", unsafe_allow_html=True)

# ⭐⭐⭐ لود داده‌ها (قبل از هر چیز) ⭐⭐⭐
@st.cache_data(ttl=3600)
def load_data():
    velocity = np.load('dashboard_velocity.npy')
    timeseries = np.load('dashboard_timeseries.npy')
    lons_grid = np.load('dashboard_lons_grid.npy')
    lats_grid = np.load('dashboard_lats_grid.npy')
    with open('dashboard_dates.json', 'r') as f:
        dates = json.load(f)
    return velocity, timeseries, lons_grid, lats_grid, dates

@st.cache_data(ttl=3600)
def load_geo(f):
    try: return gpd.read_file(f)
    except: return None

@st.cache_data(ttl=3600)
def load_csv(f):
    try: return pd.read_csv(f)
    except: return None

@st.cache_data(ttl=3600)
def load_json(f):
    try:
        with open(f, 'r', encoding='utf-8') as fp: return json.load(fp)
    except: return None


# ⭐ Loader داده‌های چند لایه
@st.cache_data(ttl=3600)
def load_multi_layer():
    """لود velocity از ۴ منبع + metadata"""
    import numpy as np, json
    data = {}
    
    layers = {
        'ascending':  'dashboard_velocity_ascending.npy',
        'descending': 'dashboard_velocity_descending.npy',
        'combined':   'dashboard_velocity_combined.npy',
        'relative':   'dashboard_velocity_relative.npy',
        'source_map': 'dashboard_source_map.npy',
        'coverage':   'dashboard_coverage_mask.npy',
    }
    
    for key, fname in layers.items():
        try:
            arr = np.load(fname)
            # ⭐ فقط منفی‌ها (حذف بالارفتگی)
            if key in ['ascending', 'descending', 'combined', 'relative']:
                arr = np.where(arr > 0, np.nan, arr)
            data[key] = arr
        except Exception:
            data[key] = None
    
    try:
        with open("dashboard_metadata.json") as f:
            data['metadata'] = json.load(f)
    except Exception:
        data['metadata'] = {}
    
    try:
        with open("district_relative_stats.json") as f:
            data['district_stats'] = json.load(f)
    except Exception:
        data['district_stats'] = {}
    
    return data


def get_active_velocity(multi_layer, data_source):
    """انتخاب velocity بر اساس data_source (همه لایه‌ها (279, 453))"""
    if multi_layer is None:
        return None, "no data"
    
    if "Ascending" in data_source:
        return multi_layer.get('ascending'), "Ascending"
    if "Descending" in data_source:
        return multi_layer.get('descending'), "Descending"
    if "Combined" in data_source:
        return multi_layer.get('combined'), "Combined"
    if "Relative" in data_source:
        return multi_layer.get('relative'), "Relative"
    return multi_layer.get('ascending'), "Ascending"


def make_joined_from_vel(vel_array, districts_gdf, lons_grid, lats_grid, step=1):
    """ساخت joined_df از velocity داده‌شده"""
    from shapely.geometry import Point
    
    ny, nx = vel_array.shape
    ys, xs = np.meshgrid(np.arange(0, ny, step), np.arange(0, nx, step), indexing='ij')
    
    v = vel_array[::step, ::step].flatten()
    lo = lons_grid[::step, ::step].flatten()
    la = lats_grid[::step, ::step].flatten()
    ys_f = ys.flatten()
    xs_f = xs.flatten()
    
    mask = ~np.isnan(v)
    
    df = pd.DataFrame({
        'lon': lo[mask],
        'lat': la[mask],
        'velocity': v[mask] * 1000,   # m/yr → mm/yr (برای نقشه)
        'row': ys_f[mask],
        'col': xs_f[mask],
    })
    
    if districts_gdf is not None and len(df) > 0:
        try:
            districts_list = list(districts_gdf.geometry)
            names_list = list(districts_gdf.get('district_name', ['Unknown']*len(districts_gdf)))
            
            district_names = []
            for lon_i, lat_i in zip(df['lon'].values, df['lat'].values):
                pt = Point(lon_i, lat_i)
                found = 'Unknown'
                for i, geom in enumerate(districts_list):
                    if geom.contains(pt):
                        found = names_list[i]
                        break
                district_names.append(found)
            df['district_name'] = district_names
        except Exception as e:
            df['district_name'] = 'Unknown'
    else:
        df['district_name'] = 'Unknown'
    
    # ⭐ فقط فرونشست (منفی‌ها) → حذف بالارفتگی
    df = df[df['velocity'] < 0].reset_index(drop=True)
    
    # حذف Unknown district
    df = df[df['district_name'] != 'Unknown'].reset_index(drop=True)
    
    return df




# ⭐ لود داده‌های اصلی
velocity, timeseries, lons_grid, lats_grid, dates = load_data()
districts_gdf = load_geo("tehran_districts.geojson")
metro_gdf = load_geo("tehran_metro.geojson")
airports_gdf = load_geo("tehran_airports.geojson")
buildings_gdf = load_geo("tehran_buildings_compact.geojson")

joined_df = load_csv('tehran_joined_final.csv')
df_metro = load_csv('tehran_metro_risk_fixed.csv')
df_airports = load_csv('tehran_airports_documented.csv')
df_pop = load_csv('tehran_population_risk.csv')
df_pred = load_csv('tehran_predictions_5yr_fixed.csv')

# ⭐ فقط فرونشست: مقادیر مثبت رو صفر کن
if df_pred is not None:
    for _col in ['Current_Rate_mm_yr', 'Observed_2024_mm', 'Predicted_2029_mm', 'Uncertainty_95']:
        if _col in df_pred.columns:
            df_pred[_col] = df_pred[_col].apply(lambda x: min(0.0, x) if pd.notna(x) else x)
df_hot = load_csv('tehran_hotspots.csv')
methodology_doc = load_json('methodology_documentation.json')

def format_date(d):
    d = str(d)
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"

dates_fmt = [format_date(d) for d in dates]

# ⭐⭐⭐ توابع کمکی ⭐⭐⭐
@st.cache_data(ttl=3600, show_spinner=False)
def prep_3d(step=4, threshold=-15, vel_override=None):
    """آماده‌سازی داده‌های 3D (با پشتیبانی از vel_override با shape متفاوت)"""
    # استفاده از vel_override اگه داده شده
    if vel_override is not None:
        v_src = vel_override
    else:
        v_src = velocity
    
    ny_full, nx_full = v_src.shape
    
    # مشابه lons_grid/lats_grid با اندازه v_src
    # اگه v_src با grid اصلی فرق داره، resize کن
    if v_src.shape == velocity.shape:
        lon_src = lons_grid
        lat_src = lats_grid
    else:
        # resize grid به shape v_src
        from scipy.ndimage import zoom
        factors = (v_src.shape[0] / velocity.shape[0],
                   v_src.shape[1] / velocity.shape[1])
        lon_src = zoom(lons_grid, factors, order=1, mode='nearest')
        lat_src = zoom(lats_grid, factors, order=1, mode='nearest')
        # crop به شکل دقیق
        lon_src = lon_src[:v_src.shape[0], :v_src.shape[1]]
        lat_src = lat_src[:v_src.shape[0], :v_src.shape[1]]
    
    # نمونه‌گیری با step
    v = v_src[::step, ::step].flatten()
    lo = lon_src[::step, ::step].flatten()
    la = lat_src[::step, ::step].flatten()
    
    # mask
    mask = ~np.isnan(v) & (v < threshold)
    
    # ys و xs (ایندکس اصلی)
    ys_full, xs_full = np.meshgrid(
        np.arange(0, v_src.shape[0], step),
        np.arange(0, v_src.shape[1], step),
        indexing='ij'
    )
    ys = ys_full.flatten()[mask]
    xs = xs_full.flatten()[mask]
    
    # district mapping
    district_names = []
    if districts_gdf is not None:
        try:
            from shapely.geometry import Point
            for lon_i, lat_i in zip(lo[mask], la[mask]):
                pt = Point(lon_i, lat_i)
                match = districts_gdf[districts_gdf.geometry.contains(pt)]
                if len(match) > 0:
                    for col_name in ['district_name', 'name', 'NAME', 'NAME_2']:
                        if col_name in match.columns:
                            district_names.append(match.iloc[0][col_name])
                            break
                    else:
                        district_names.append('Unknown')
                else:
                    district_names.append('Unknown')
        except Exception as e:
            print(f"⚠️ District mapping failed: {e}")
            district_names = ['All'] * mask.sum()
    else:
        district_names = ['All'] * mask.sum()
    
    df = pd.DataFrame({
        'lon': lo[mask],
        'lat': la[mask],
        'velocity': v[mask],
        'depth_mm': -v[mask] * 1000,  # m/yr → mm/yr
        'row': ys,
        'col': xs,
        'district_name': district_names
    })
    
    # ⭐ ساختمان‌های مناطق بحرانی
    bld = None
    try:
        with gzip.open('tehran_buildings_danger.json.gz', 'rt', encoding='utf-8') as f:
            bld = json.load(f)
        for feature in bld['features']:
            feature['properties']['height_m'] = feature['properties'].get('h', 12)
    except Exception:
        bld = None
    
    dist = None
    if districts_gdf is not None:
        try:
            dist = json.loads(districts_gdf.to_json())
        except Exception:
            pass
    
    metro = None
    if metro_gdf is not None:
        try:
            metro = json.loads(metro_gdf.to_json())
        except Exception:
            pass
    
    return df, bld, dist, metro


def get_well_color(v, mode):
    if mode == "THERMAL":
        if v >= -15: return [255, 230, 0, 200]
        elif v >= -25: return [255, 165, 0, 215]
        elif v >= -40: return [255, 100, 0, 230]
        elif v >= -60: return [255, 30, 30, 245]
        elif v >= -100: return [200, 0, 50, 250]
        else: return [150, 0, 80, 255]
    elif mode == "NEON":
        if v >= -15: return [0, 255, 200, 200]
        elif v >= -25: return [0, 200, 255, 220]
        elif v >= -40: return [50, 130, 255, 235]
        elif v >= -60: return [130, 0, 255, 245]
        elif v >= -100: return [200, 0, 200, 250]
        else: return [255, 0, 100, 255]
    else:
        if v >= -15: return [255, 200, 0, 210]
        elif v >= -25: return [255, 120, 0, 225]
        elif v >= -40: return [255, 30, 30, 240]
        elif v >= -60: return [230, 0, 100, 250]
        elif v >= -100: return [140, 0, 180, 255]
        else: return [80, 0, 220, 255]

def cls_risk(v):
    if pd.isna(v): return 'N/A'
    if v < -50: return '🔴 Critical'
    elif v < -30: return '🟠 Warning'
    elif v < -15: return '🟡 Moderate'
    else: return '🟢 Stable'

def add_boundaries(fig, color='#00d9ff', w=2):
    if districts_gdf is None: return fig
    for _, r in districts_gdf.iterrows():
        g = r.geometry
        ps = [g] if g.geom_type == 'Polygon' else list(g.geoms) if g.geom_type == 'MultiPolygon' else []
        for p in ps:
            c = list(p.exterior.coords)
            fig.add_trace(go.Scattermapbox(lon=[x[0] for x in c], lat=[x[1] for x in c],
                mode='lines', line=dict(color=color, width=w), showlegend=False, hoverinfo='skip'))
    return fig

def apply_layers(fig):
    return add_boundaries(fig)

# ⭐⭐⭐ هدر ⭐⭐⭐
st.markdown("""
<div class="main-header">
    <h1>◉ TEHRAN SUBSIDENCE MATRIX ◉</h1>
    <p>▸ SENTINEL-1 INSAR · JAN-DEC 2024 · 22 DISTRICTS ◂</p>
</div>
""", unsafe_allow_html=True)


# ⭐⭐⭐ سایدبار ⭐⭐⭐
with st.sidebar:
    st.markdown("## ⚙ SCAN CONTROLS")
    
    page = st.radio("MODULE:", [
        "🛸 3D CITY SCAN",
        "🏠 OVERVIEW",
        "🗺️ INTERACTIVE MAP",
        "📊 DISTRICT STATS",
        "📈 TIME SERIES",
        "🔮 FORECAST 2029",
        "🎬 ANIMATION",
        "🚨 RISK MAP",
        "🚇 INFRASTRUCTURE",
        "🔥 HOTSPOTS",
        "📚 METHODOLOGY"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("## 🎛 VISUAL MODE")
    viz_mode = st.radio("STYLE:", ["THERMAL", "NEON", "HYBRID"])
    
    st.markdown("---")
    st.markdown("## 📡 DATA SOURCE")
    data_source = st.radio(
        "SOURCE:",
        ["🛰️ Ascending", "📡 Descending", "🔀 Combined", "📊 Relative"],
        help="Ascending: 20% coverage، کیفیت بالا | Descending: 99% | Combined: 100% | Relative: نرمال‌شده"
    )
    
    if "Descending" in data_source:
        st.warning("⚠️ کیفیت متوسط — نویزی")
    elif "Combined" in data_source:
        st.info("ℹ️ Ascending + Descending")
    elif "Relative" in data_source:
        st.info("ℹ️ median هر منطقه = 0")
    else:
        st.success("✅ کیفیت بالا")
    
    st.markdown("---")
    st.markdown("## 🎨 LAYERS")
    show_districts = st.checkbox("📍 Districts", value=True)
    
    st.markdown("---")
    color_min = st.slider("Min (mm/yr)", -250, 0, -150, step=10)
    color_max = st.slider("Max (mm/yr)", 0, 100, 50, step=10)


# ⭐⭐⭐ PREP 3D DATA ⭐⭐⭐

# ⭐ لود داده 3D قبل از استفاده در صفحات
# ⭐ انتخاب لایه فعال
# ⭐ لود داده‌های چند لایه (قبل از استفاده)
multi_layer = load_multi_layer()

# ⭐ انتخاب لایه فعال
vel_active, source_label = get_active_velocity(multi_layer, data_source)

# ⭐ نمایش منبع فعال
if vel_active is not None:
    _n_valid = int((~np.isnan(vel_active)).sum())
    _pct = _n_valid / vel_active.size * 100

# ⭐ بازسازی joined_df بر اساس data_source
try:
    _new_joined = make_joined_from_vel(vel_active, districts_gdf, lons_grid, lats_grid, step=1)
    if len(_new_joined) > 1000:
        joined_df = _new_joined
        print(f"[DATA] joined_df بازسازی: {len(joined_df):,} پیکسل ({source_label})")
    else:
        print(f"[DATA] joined_df جدید کمه ({len(_new_joined)}), از اورجینال")
except Exception as e:
    print(f"[DATA] خطا: {e}")

# ⭐ prep_3d با لایه فعال
# ⭐ threshold تطبیقی در واحد m/yr
_vel_valid = vel_active[~np.isnan(vel_active)]
_vel_med = np.nanmedian(_vel_valid)
_vel_min = np.nanmin(_vel_valid)

# آستانه: 10 میلی‌متر بر سال = 0.01 m/yr
# یا بهتر: هرچی کمتر از میانه منهای نصف دامنه
threshold_mm = -10  # mm/yr
threshold_use = threshold_mm / 1000  # m/yr = -0.01

# اگه داده کم بود، شل‌تر کن
_potential = (_vel_valid < threshold_use).sum()
if _potential < 500:
    threshold_use = -3 / 1000  # -3 mm/yr
if _potential < 100:
    threshold_use = -1 / 1000  # -1 mm/yr

print(f"[DEBUG] vel range: {_vel_min*1000:.1f} → {np.nanmax(_vel_valid)*1000:.1f} mm/yr")
print(f"[DEBUG] threshold: {threshold_use*1000:.1f} mm/yr")
print(f"[DEBUG] pixels below: {_potential}")

wells_df, bld_geojson, dist_geojson, metro_geojson = prep_3d(
    step=2, threshold=threshold_use, vel_override=vel_active
)

# ⭐ آستانه wells_filtered هم تنظیم کن
_threshold_wells = threshold_use * 0.5  # نصف آستانه
wells_df['color'] = wells_df['velocity'].apply(lambda v: get_well_color(v, viz_mode))

# ⭐⭐⭐ PAGE 1: 3D CITY SCAN ⭐⭐⭐
if page == "🛸 3D CITY SCAN":
    st.markdown('<div class="section-header">▸ 3D MUNICIPAL SCAN ACTIVE ◂</div>', unsafe_allow_html=True)
    
    # ⭐ بنر منبع فعال
    _s = f"📡 {source_label}"
    if vel_active is not None:
        _n = int((~np.isnan(vel_active)).sum())
        _p = _n / vel_active.size * 100
        _s += f" · ✅ {_n:,} پیکسل ({_p:.1f}%)"
    
    if "Ascending" in source_label:
        _q = "🟢 کیفیت بالا · پوشش کم"
    elif "Descending" in source_label:
        _q = "🟡 کیفیت متوسط · پوشش کامل"
    elif "Combined" in source_label:
        _q = "🔵 ترکیبی · پوشش ۱۰۰٪"
    elif "Relative" in source_label:
        _q = "🟣 نسبی · median=0"
    else:
        _q = ""
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(0,217,255,0.15), rgba(0,255,136,0.1));
                border-left: 4px solid #00d9ff;
                padding: 0.6rem 1rem; margin-bottom: 1rem;
                border-radius: 8px; color: #7ee7ff;
                font-family: 'Courier New'; font-size: 0.85rem;">
        <b>📡 منبع فعال:</b> {_s}<br>
        <span style="font-size: 0.75rem; color: #99ffcc;">{_q}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(0,217,255,0.05); border-left: 3px solid #00d9ff; 
                padding: 0.7rem; border-radius: 6px; color: #7ee7ff; 
                font-family: 'Courier New'; font-size: 0.85rem; margin-bottom: 1rem;">
    ⬢ <b>MOUSE:</b> Left-drag rotate · Right-drag pan · Scroll zoom · Ctrl+drag tilt<br>
    ⬢ <b>TOUCH:</b> 1-finger rotate · 2-finger pinch zoom · 2-finger drag pan
    </div>
    """, unsafe_allow_html=True)
    
    # ⭐ فیلتر wells (همه چاه‌های منفی)
    if len(wells_df) > 0:
        _thresh_w = -0.5 / 1000  # -0.5 mm/yr
        wells_filtered = wells_df[wells_df['velocity'] <= _thresh_w].copy()
    else:
        wells_filtered = wells_df.copy()
    
    if len(wells_filtered) > 0:
        wells_filtered['color'] = wells_filtered['velocity'].apply(
            lambda v: get_well_color(v, viz_mode)
        )
    
    # ⭐ 4 کارت بالا (خلاصه)
    _n_bld = len(bld_geojson['features']) if bld_geojson and 'features' in bld_geojson else 0
    _n_wells = len(wells_filtered)
    _max_w = wells_filtered["depth_mm"].max() if _n_wells > 0 else 0
    _mean_w = wells_filtered["depth_mm"].mean() if _n_wells > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>◉ BUILDINGS</h3><div class="value">{_n_bld:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>◉ WELLS</h3><div class="value">{_n_wells:,}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>◉ MAX</h3><div class="value">{_max_w:.0f}<span class="sub"> mm/yr</span></div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>◉ MEAN</h3><div class="value">{_mean_w:.0f}<span class="sub"> mm/yr</span></div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ⭐ 3 اسلایدر
    col1, col2, col3 = st.columns(3)
    with col1:
        bld_scale = st.slider("Building Height Scale", 1, 5, 1, 1,
                               help="۱ = واقعی · بیشتر = اغراق‌شده")
    with col2:
        well_depth = st.slider("Well Depth Scale", 1, 50, 10, 1,
                                help="چقدر عمق چاه‌ها بزرگ‌نمایی بشه (۱۰ = پیش‌فرض)")
    with col3:
        well_radius = st.slider("Well Radius (m)", 10, 100, 30, 5,
                                 help="شعاع چاه‌ها (متر)")

    # ⭐ انتخاب حالت نمایش
    view_mode = st.radio(
        "🎨 VIEW MODE:",
        ["🌍 Circles (روی نقشه)", "📈 Height (ستون‌ها)"],
        index=0,
        horizontal=True,
        help="Circles: نقطه روی نقشه | Height: ستون به بالا"
    )
    
    # ⭐ لایه‌ها
    layers = []
    
    # ۱. ساختمان‌ها
    if bld_geojson and bld_geojson.get('features'):
        layers.append(pdk.Layer(
            'GeoJsonLayer',
            data=bld_geojson,
            opacity=0.85,
            stroked=True,
            filled=True,
            extruded=True,
            wireframe=True,
            get_elevation='properties.height_m',
            elevation_scale=bld_scale,
            get_fill_color=[15, 25, 55, 220],
            get_line_color=[0, 217, 255, 200],
            line_width_min_pixels=1,
            pickable=False,
            auto_highlight=False,
        ))
    
    # ۲. چاه‌ها
    if len(wells_filtered) > 0:
        _max_wells = 2000
        if len(wells_filtered) > _max_wells:
            _wf = wells_filtered.sample(n=_max_wells, random_state=42).copy()
        else:
            _wf = wells_filtered.copy()
        
        def _well_color(v):
            if v < -0.10:   return [255, 20, 50]
            elif v < -0.06: return [255, 120, 0]
            elif v < -0.03: return [255, 220, 0]
            else:           return [0, 255, 136]
        
        _wf['r'] = _wf['velocity'].apply(lambda v: _well_color(v)[0])
        _wf['g'] = _wf['velocity'].apply(lambda v: _well_color(v)[1])
        _wf['b'] = _wf['velocity'].apply(lambda v: _well_color(v)[2])
        
        # ⭐ چاه‌ها — بر اساس view_mode
        if "Height" in view_mode:
            # ستون به بالا (با نقشه)
            layers.append(pdk.Layer(
                'ColumnLayer',
                data=_wf,
                get_position=['lon', 'lat'],
                get_elevation='depth_mm',
                elevation_scale=well_depth / 10,
                get_fill_color=['r', 'g', 'b'],
                radius=well_radius,
                extruded=True,
                disk_resolution=16,
                coverage=1.0,
                pickable=True,
                auto_highlight=True,
            ))

        else:
            # دایره (روی نقشه)
            layers.append(pdk.Layer(
                'ScatterplotLayer',
                data=_wf,
                get_position=['lon', 'lat'],
                get_fill_color=['r', 'g', 'b'],
                get_line_color=[255, 255, 255, 200],
                get_radius=well_radius * 5,
                radius_min_pixels=3,
                radius_max_pixels=30,
                line_width_min_pixels=1,
                stroked=True,
                filled=True,
                pickable=True,
                auto_highlight=True,
            ))
    
    # ۳. مرزها
    if dist_geojson is not None:
        layers.append(pdk.Layer(
            'GeoJsonLayer',
            data=dist_geojson,
            opacity=0.0,
            stroked=True,
            filled=False,
            get_line_color=[0, 217, 255, 200],
            line_width_min_pixels=2,
            pickable=False,
        ))
    
    # ۴. مترو
    if metro_geojson is not None:
        layers.append(pdk.Layer(
            'GeoJsonLayer',
            data=metro_geojson,
            opacity=0.8,
            stroked=True,
            filled=False,
            get_line_color=[0, 255, 200, 220],
            line_width_min_pixels=2,
            pickable=False,
        ))
    
    # ۵. فرودگاه‌ها
    if airports_gdf is not None:
        airports_json = json.loads(airports_gdf.to_json())
        layers.append(pdk.Layer(
            'GeoJsonLayer',
            data=airports_json,
            opacity=0.9,
            stroked=True,
            filled=True,
            get_fill_color=[155, 89, 182, 220],
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=2,
            pickable=True,
        ))
    
    # ⭐ ViewState — مرکز درست
    view_state = pdk.ViewState(
        latitude=35.70,
        longitude=51.40,
        zoom=11,
        pitch=60,
        bearing=-25,
    )
    
    # ⭐ Tooltip — از depth_mm
    tooltip = {
        "html": """
        <div style="font-family: Courier New; background: rgba(0,0,0,0.9); 
                    border: 1px solid #00d9ff; padding: 8px; border-radius: 4px;">
            <div style="color: #00d9ff;">◉ DATA POINT</div>
            <div style="color: #7ee7ff; font-size: 12px;">
                SUBSIDENCE: <b>{depth_mm} mm/yr</b>
            </div>
        </div>
        """,
        "style": {"backgroundColor": "transparent"}
    }
    
    # ⭐ Deck
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="carto-darkmatter",
    )
    
    st.pydeck_chart(deck, use_container_width=True)
    
    # ⭐ Time Series (امن)
    st.markdown("---")
    st.markdown("### 📊 Time Series Analysis")
    
    selected_point = None
    district_pts = None
    
    if len(wells_filtered) > 0 and 'district_name' in wells_filtered.columns:
        districts_list = sorted(wells_filtered['district_name'].unique())
    else:
        districts_list = ['All']
    
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        selected_district = st.selectbox(
            "📍 Select District",
            options=districts_list
        )
    
    if selected_district != 'All':
        district_pts = wells_filtered[wells_filtered['district_name'] == selected_district]
    else:
        district_pts = wells_filtered
    
    with col_b:
        if len(district_pts) > 0:
            point_names = [f"Point {i} ({p['velocity']*1000:.1f} mm/yr)" 
                          for i, (_, p) in enumerate(district_pts.head(50).iterrows())]
            
            selected_point_name = st.selectbox(
                "🎯 Select Point (or click on map)",
                options=point_names
            )
            
            selected_idx = point_names.index(selected_point_name)
            selected_point = district_pts.iloc[selected_idx]
    
    # نمودار (امن)
    if district_pts is not None and len(district_pts) > 0 and selected_point is not None:
        # row/col مستقیم (بدون patch قدیمی)
        try:
            row = int(selected_point.get('row', 139))
            col = int(selected_point.get('col', 209))
            row = max(0, min(timeseries.shape[1] - 1, row))
            col = max(0, min(timeseries.shape[2] - 1, col))
        except Exception:
            row, col = 139, 209
        
        point_ts = timeseries[:, row, col]
        point_ts = point_ts - point_ts[0]  # already in mm
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates_fmt,
            y=point_ts,
            mode='lines+markers',
            name='Cumulative Displacement',
            line=dict(color='#00d9ff', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"Time Series · {selected_district} · Point {selected_idx}",
            xaxis_title='Date',
            yaxis_title='Displacement (mm)',
            height=400,
            template='plotly_dark',
            paper_bgcolor='#0a0e27',
            plot_bgcolor='#0a0e27',
            font=dict(color='#7ee7ff', family='Courier New'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)


elif page == "🏠 OVERVIEW":
    # ⭐ بنر منبع فعال (همه صفحات)
    _s = f"📡 {source_label}"
    if vel_active is not None:
        _n = int((~np.isnan(vel_active)).sum())
        _p = _n / vel_active.size * 100
        _s += f" · ✅ {_n:,} پیکسل ({_p:.1f}%)"
    
    if "Ascending" in source_label:
        _q = "🟢 کیفیت بالا · پوشش کم"
    elif "Descending" in source_label:
        _q = "🟡 کیفیت متوسط · پوشش کامل"
    elif "Combined" in source_label:
        _q = "🔵 ترکیبی · پوشش ۱۰۰٪"
    elif "Relative" in source_label:
        _q = "🟣 نسبی · median=0"
    else:
        _q = ""
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(0,217,255,0.15), rgba(0,255,136,0.1));
                border-left: 4px solid #00d9ff;
                padding: 0.6rem 1rem; margin-bottom: 1rem;
                border-radius: 8px; color: #7ee7ff;
                font-family: 'Courier New'; font-size: 0.85rem;">
        <b>📡 منبع فعال:</b> {_s}<br>
        <span style="font-size: 0.75rem; color: #99ffcc;">{_q}</span>
    </div>
    """, unsafe_allow_html=True)

    # ⭐ CSS اضافی برای این صفحه
    st.markdown("""
    <style>
    .kpi-card {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a4e 100%);
        padding: 1.5rem 1rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #00d9ff;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0,217,255,0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    .kpi-card .icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .kpi-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d9ff;
        text-shadow: 0 0 15px #00d9ff;
        font-family: 'Courier New', monospace;
        margin: 0.3rem 0;
    }
    .kpi-card .label {
        color: #7ee7ff;
        font-size: 0.85rem;
        font-family: 'Courier New', monospace;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .kpi-card.critical {
        border-color: #FF1432;
        box-shadow: 0 0 30px rgba(255, 20, 50, 0.4);
    }
    .kpi-card.critical .value {
        color: #FF1432;
        text-shadow: 0 0 15px #FF1432;
    }
    .kpi-card.warning {
        border-color: #FF7800;
        box-shadow: 0 0 30px rgba(255, 120, 0, 0.4);
    }
    .kpi-card.warning .value {
        color: #FF7800;
        text-shadow: 0 0 15px #FF7800;
    }
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        background: rgba(255, 20, 50, 0.2);
        border: 2px solid #FF1432;
        color: #FF1432;
        font-family: 'Courier New', monospace;
        font-weight: 700;
        letter-spacing: 3px;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        animation: blink 2s ease-in-out infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ⭐ هدر اصلی
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0a0e27 0%, #1a1a4e 50%, #0a0e27 100%);
                padding: 2rem; border-radius: 15px; text-align: center;
                border: 2px solid #00d9ff;
                box-shadow: 0 0 50px rgba(0, 217, 255, 0.4);
                margin-bottom: 2rem;">
        <h1 style="color: #00d9ff; font-family: 'Courier New'; margin: 0;
                   font-size: 2rem; letter-spacing: 4px;
                   text-shadow: 0 0 20px #00d9ff;">
            ◉ TEHRAN SUBSIDENCE MATRIX ◉
        </h1>
        <p style="color: #7ee7ff; font-family: 'Courier New';
                  font-size: 1rem; margin-top: 0.7rem; letter-spacing: 2px;">
            ▸ ANALYSIS PERIOD: JANUARY - DECEMBER 2024 ◂
        </p>
        <div class="status-badge">
            ⚠ SYSTEM STATUS: CRITICAL ⚠
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ⭐ محاسبه آمار
    if joined_df is not None:
        # آمار از vel_active (داینامیک بر اساس Data Source)
        if vel_active is not None:
            _v_all = vel_active[~np.isnan(vel_active)] * 1000  # mm/yr
            _v = _v_all[_v_all < 0]  # ⭐ فقط منفی‌ها
            total_pixels = len(_v)
            mean_vel = float(np.mean(_v)) if len(_v) > 0 else 0
            min_vel = float(np.min(_v)) if len(_v) > 0 else 0
        else:
            total_pixels = 0
            mean_vel = 0
            min_vel = 0
        
        # مناطق بحرانی از joined_df (داینامیک)
        stats = joined_df.groupby('district_name')['velocity'].mean().sort_values()
        # velocity الان در joined_df به mm/yr هست
        stats_mm = stats
        critical_districts = stats_mm[stats_mm < -30]
        n_critical = len(critical_districts)
        
        # جمعیت در خطر
        total_affected = df_pop['affected_population'].sum() if df_pop is not None else 0
        affected_pct = total_affected / df_pop['population'].sum() * 100 if df_pop is not None else 0
        
        # ⭐ Row 1: 3 کارت اصلی
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"""
            <div class="kpi-card critical">
                <div class="icon">⚠️</div>
                <div class="label">MAX SUBSIDENCE</div>
                <div class="value">{min_vel:.1f}</div>
                <div class="label">mm/year</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            st.markdown(f"""
            <div class="kpi-card critical">
                <div class="icon">🚨</div>
                <div class="label">CRITICAL DISTRICTS</div>
                <div class="value">{n_critical}</div>
                <div class="label">of 22 districts</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c3:
            st.markdown(f"""
            <div class="kpi-card warning">
                <div class="icon">👥</div>
                <div class="label">PEOPLE AT RISK</div>
                <div class="value">{total_affected/1e6:.2f}M</div>
                <div class="label">in critical zones</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ⭐ Row 2: 2 کارت آماری
        c4, c5 = st.columns(2)
        
        with c4:
            st.markdown(f"""
            <div class="kpi-card warning">
                <div class="icon">📊</div>
                <div class="label">PERCENTAGE AT RISK</div>
                <div class="value">{affected_pct:.1f}%</div>
                <div class="label">of Tehran population</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="icon">📅</div>
                <div class="label">DATA COVERAGE</div>
                <div class="value">{len(dates)}</div>
                <div class="label">SAR acquisitions</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ⭐ بخش نقشه + Top 10
        st.markdown("---")
        
        col_map, col_list = st.columns([2, 1])
        
        # نقشه
        with col_map:
            st.markdown("### 🗺️ Tehran Subsidence Map")
            
            # Heatmap
            fig = go.Figure()
            fig.add_trace(go.Scattermapbox(
                lon=joined_df['lon'],
                lat=joined_df['lat'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=joined_df['velocity'],
                    colorscale=[
                        [0.0, '#8B0000'],
                        [0.15, '#FF1432'],
                        [0.35, '#FF4500'],
                        [0.55, '#FFA500'],
                        [0.75, '#FFE600'],
                        [1.0, '#00FF88']
                    ],
                    cmin=-150,
                    cmax=50,
                    colorbar=dict(
                        title=dict(text='Velocity<br>(mm/yr)', font=dict(color='#7ee7ff', size=12)),
                        tickfont=dict(color='#7ee7ff'),
                        thickness=12,
                        len=0.8
                    ),
                    opacity=0.7
                ),
                hovertemplate='<b>%{marker.color:.1f} mm/yr</b><extra></extra>',
                showlegend=False
            ))
            
            # مرزها
            if districts_gdf is not None:
                for _, r in districts_gdf.iterrows():
                    g = r.geometry
                    ps = [g] if g.geom_type == 'Polygon' else list(g.geoms) if g.geom_type == 'MultiPolygon' else []
                    for p in ps:
                        coords = list(p.exterior.coords)
                        fig.add_trace(go.Scattermapbox(
                            lon=[c[0] for c in coords],
                            lat=[c[1] for c in coords],
                            mode='lines',
                            line=dict(color='#00d9ff', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
            
            fig.update_layout(
                mapbox=dict(
                    style='carto-darkmatter',
                    accesstoken=MAPBOX_TOKEN if MAPBOX_TOKEN else None,
                    center=dict(lat=35.7, lon=51.4),
                    zoom=10
                ),
                height=500,
                margin=dict(r=0, t=0, l=0, b=0),
                paper_bgcolor='#05070d',
                font=dict(color='#7ee7ff')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Top 10 میله‌ای
        with col_list:
            st.markdown("### 🔴 Top 10 Critical")
            
            top10 = stats_mm.head(10).reset_index()
            top10.columns = ['District', 'Velocity']
            top10 = top10.sort_values('Velocity')
            
            colors = [
                '#8B0000' if v < -50 else
                '#FF1432' if v < -40 else
                '#FF4500' if v < -30 else
                '#FFA500' if v < -20 else
                '#FFE600'
                for v in top10['Velocity']
            ]
            
            fig2 = go.Figure(data=[go.Bar(
                x=top10['Velocity'],
                y=top10['District'],
                orientation='h',
                marker=dict(color=colors, line=dict(color='#00d9ff', width=1)),
                text=[f'{v:.1f}' for v in top10['Velocity']],
                textposition='outside',
                textfont=dict(color='#7ee7ff', size=10, family='Courier New'),
                hovertemplate='<b>%{y}</b><br>%{x:.1f} mm/yr<extra></extra>'
            )])
            
            fig2.update_layout(
                height=500,
                margin=dict(r=20, t=20, l=10, b=20),
                paper_bgcolor='#0a0e27',
                plot_bgcolor='#0a0e27',
                font=dict(color='#7ee7ff', family='Courier New'),
                xaxis=dict(
                    title=dict(text='Velocity (mm/yr)', font=dict(color='#7ee7ff')),
                    gridcolor='rgba(0, 217, 255, 0.2)'
                ),
                yaxis=dict(
                    gridcolor='rgba(0, 217, 255, 0.1)',
                    tickfont=dict(color='#7ee7ff')
                ),
                showlegend=False
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # ⭐ جدول کامل ۲۲ منطقه
        st.markdown("---")
        st.markdown("### 📋 Full District Analysis")
        
        if df_pop is not None:
            # جدول با اطلاعات کامل
            table = df_pop[['rank', 'district_name', 'mean_velocity', 
                           'population', 'affected_population', 
                           'risk_index', 'risk_category']].copy()
            table.columns = ['Rank', 'District', 'Velocity (mm/yr)', 
                            'Population', 'Affected', 
                            'Risk Index', 'Category']
            
            st.dataframe(
                table.style
                    .background_gradient(subset=['Velocity (mm/yr)'], cmap='Reds_r')
                    .background_gradient(subset=['Risk Index'], cmap='Reds')
                    .format({
                        'Velocity (mm/yr)': '{:.1f}',
                        'Population': '{:,.0f}',
                        'Affected': '{:,.0f}',
                        'Risk Index': '{:.1f}'
                    }),
                use_container_width=True,
                hide_index=True,
                height=500
            )

elif page == "🗺️ INTERACTIVE MAP":
    # ⭐ بنر منبع فعال (همه صفحات)
    _s = f"📡 {source_label}"
    if vel_active is not None:
        _n = int((~np.isnan(vel_active)).sum())
        _p = _n / vel_active.size * 100
        _s += f" · ✅ {_n:,} پیکسل ({_p:.1f}%)"
    
    if "Ascending" in source_label:
        _q = "🟢 کیفیت بالا · پوشش کم"
    elif "Descending" in source_label:
        _q = "🟡 کیفیت متوسط · پوشش کامل"
    elif "Combined" in source_label:
        _q = "🔵 ترکیبی · پوشش ۱۰۰٪"
    elif "Relative" in source_label:
        _q = "🟣 نسبی · median=0"
    else:
        _q = ""
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(0,217,255,0.15), rgba(0,255,136,0.1));
                border-left: 4px solid #00d9ff;
                padding: 0.6rem 1rem; margin-bottom: 1rem;
                border-radius: 8px; color: #7ee7ff;
                font-family: 'Courier New'; font-size: 0.85rem;">
        <b>📡 منبع فعال:</b> {_s}<br>
        <span style="font-size: 0.75rem; color: #99ffcc;">{_q}</span>
    </div>
    """, unsafe_allow_html=True)

    # ⭐ هدر
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0a0e27 0%, #1a1a4e 50%, #0a0e27 100%);
                padding: 1.5rem; border-radius: 15px; text-align: center;
                border: 2px solid #00d9ff;
                box-shadow: 0 0 40px rgba(0, 217, 255, 0.3);
                margin-bottom: 1.5rem;">
        <h2 style="color: #00d9ff; font-family: 'Courier New'; margin: 0;
                   letter-spacing: 3px; text-shadow: 0 0 15px #00d9ff;">
            ◉ INTERACTIVE SUBSIDENCE MAP ◉
        </h2>
        <p style="color: #7ee7ff; font-family: 'Courier New';
                  font-size: 0.9rem; margin-top: 0.5rem; letter-spacing: 2px;">
            ▸ INTERACT WITH MOUSE OR TOUCH · ZOOM · ROTATE · PAN ◂
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ⭐ تنظیمات نمایش
    col1, col2 = st.columns([1, 2])

    with col1:
        mode = st.radio(
            "🎨 Display Mode:",
            ["Heatmap", "Choropleth"],
            horizontal=True
        )

    with col2:
        st.markdown("#### 📍 Active Layers")
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            show_boundaries = st.checkbox("🗺️ Districts", value=True)
        with lc2:
            show_metro = st.checkbox("🚇 Metro Lines", value=True)
        with lc3:
            show_airports = st.checkbox("✈️ Airports", value=True)

    st.markdown("---")

    # ============================================================
    # ⭐ حالت Heatmap
    # ============================================================
    if mode == "Heatmap":
        if joined_df is not None:
            sample_size = min(8000, len(joined_df))
            sample_df = joined_df.sample(n=sample_size, random_state=42)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scattermapbox(
                lon=sample_df["lon"],
                lat=sample_df["lat"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=sample_df["velocity"],
                    colorscale=[
                        [0.0, "#8B0000"],
                        [0.15, "#FF1432"],
                        [0.35, "#FF4500"],
                        [0.55, "#FFA500"],
                        [0.75, "#FFE600"],
                        [1.0, "#00FF88"]
                    ],
                    cmin=-200,
                    cmax=0,
                    colorbar=dict(
                        title=dict(text="Velocity (mm/yr)", font=dict(color="#7ee7ff", size=12)),
                        tickfont=dict(color="#7ee7ff"),
                        thickness=15,
                        len=0.8
                    ),
                    opacity=0.7
                ),
                hovertemplate="<b>%{marker.color:.1f} mm/yr</b><extra></extra>",
                name="Velocity",
                showlegend=False
            ))
            
            # لایه مرزها
            if show_boundaries and districts_gdf is not None:
                for _, r in districts_gdf.iterrows():
                    g = r.geometry
                    ps = [g] if g.geom_type == "Polygon" else list(g.geoms) if g.geom_type == "MultiPolygon" else []
                    for p in ps:
                        coords = list(p.exterior.coords)
                        fig.add_trace(go.Scattermapbox(
                            lon=[c[0] for c in coords],
                            lat=[c[1] for c in coords],
                            mode="lines",
                            line=dict(color="#00d9ff", width=2),
                            name="District",
                            showlegend=False,
                            hoverinfo="skip"
                        ))
            
            # لایه مترو
            if show_metro and metro_gdf is not None:
                for _, r in metro_gdf.iterrows():
                    g = r.geometry
                    ls = [g] if g.geom_type == "LineString" else list(g.geoms) if g.geom_type == "MultiLineString" else []
                    for line in ls:
                        coords = list(line.coords)
                        fig.add_trace(go.Scattermapbox(
                            lon=[c[0] for c in coords],
                            lat=[c[1] for c in coords],
                            mode="lines",
                            line=dict(color="#00FFC8", width=2),
                            name="Metro",
                            showlegend=False,
                            hoverinfo="skip"
                        ))
            
            # لایه فرودگاه‌ها
            if show_airports and airports_gdf is not None:
                for _, r in airports_gdf.iterrows():
                    g = r.geometry
                    if g.geom_type == "Point":
                        lon, lat = g.x, g.y
                    elif g.geom_type == "Polygon":
                        c = g.centroid
                        lon, lat = c.x, c.y
                    else:
                        continue
                    fig.add_trace(go.Scattermapbox(
                        lon=[lon],
                        lat=[lat],
                        mode="markers",
                        marker=dict(size=15, color="#9b59b6", symbol="airport"),
                        name="Airport",
                        showlegend=False,
                        hovertemplate="<b>✈️ Airport</b><extra></extra>"
                    ))
            
            fig.update_layout(
                mapbox=dict(style="carto-darkmatter", center=dict(lat=35.7, lon=51.4), zoom=11),
                accesstoken=MAPBOX_TOKEN if MAPBOX_TOKEN else None,
                height=700,
                margin=dict(r=0, t=20, l=0, b=0),
                paper_bgcolor="#05070d",
                font=dict(color="#7ee7ff")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # آمار
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Points Shown", f"{sample_size:,}", f"of {len(joined_df):,}")
            with c2:
                st.metric("Mean Velocity", f"{sample_df['velocity'].mean():.1f} mm/yr")
            with c3:
                st.metric("Min Velocity", f"{sample_df['velocity'].min():.1f} mm/yr")
            with c4:
                critical_count = (sample_df["velocity"] < -50).sum()
                st.metric("Critical Points", f"{critical_count:,}")

    # ============================================================
    # ⭐ حالت Choropleth
    # ============================================================
    else:
        if joined_df is not None and districts_gdf is not None:
            stats = joined_df.groupby("district_name").agg(
                mean_velocity=("velocity", "mean"),
                max_subsidence=("velocity", "min"),
                pixel_count=("velocity", "count")
            ).reset_index()
            
            merged = districts_gdf.merge(stats, on="district_name", how="left")
            # fillna با 0 (نمایش "no data" به عنوان صفر)
            merged["mean_velocity"] = merged["mean_velocity"].fillna(0)
            merged["max_subsidence"] = merged["max_subsidence"].fillna(0)
            merged["pixel_count"] = merged["pixel_count"].fillna(0)
            # ⭐ mask مناطق بدون داده
            merged["has_data"] = merged["pixel_count"] > 0
            
            merged_geojson = json.loads(merged.to_json())
            
            fig = px.choropleth_mapbox(
                merged,
                geojson=merged_geojson,
                locations=merged.index,
                color="mean_velocity",
                color_continuous_scale=[
                    [0.0, "#8B0000"],
                    [0.15, "#FF1432"],
                    [0.35, "#FF4500"],
                    [0.55, "#FFA500"],
                    [0.75, "#FFE600"],
                    [1.0, "#00FF88"]
                ],
                range_color=(-200, 0),
                mapbox_style="carto-darkmatter",
                mapbox_token=MAPBOX_TOKEN if MAPBOX_TOKEN else None,
                zoom=10,
                center={"lat": 35.7, "lon": 51.4},
                opacity=0.75,
                hover_name="district_name",
                hover_data={"mean_velocity": ":.1f", "max_subsidence": ":.1f", "pixel_count": ":,", "district_name": False}
            )
            
            if show_boundaries:
                fig.update_traces(marker_line_width=3, marker_line_color="#00d9ff")
            else:
                fig.update_traces(marker_line_width=1, marker_line_color="rgba(0, 217, 255, 0.3)")
            
            if show_metro and metro_gdf is not None:
                for _, r in metro_gdf.iterrows():
                    g = r.geometry
                    ls = [g] if g.geom_type == "LineString" else list(g.geoms) if g.geom_type == "MultiLineString" else []
                    for line in ls:
                        coords = list(line.coords)
                        fig.add_trace(go.Scattermapbox(
                            lon=[c[0] for c in coords],
                            lat=[c[1] for c in coords],
                            mode="lines",
                            line=dict(color="#00FFC8", width=2),
                            showlegend=False,
                            hoverinfo="skip"
                        ))
            
            if show_airports and airports_gdf is not None:
                for _, r in airports_gdf.iterrows():
                    g = r.geometry
                    if g.geom_type == "Point":
                        lon, lat = g.x, g.y
                    elif g.geom_type == "Polygon":
                        c = g.centroid
                        lon, lat = c.x, c.y
                    else:
                        continue
                    fig.add_trace(go.Scattermapbox(
                        lon=[lon],
                        lat=[lat],
                        mode="markers",
                        marker=dict(size=15, color="#9b59b6", symbol="airport"),
                        showlegend=False,
                        hovertemplate="<b>✈️ Airport</b><extra></extra>"
                    ))
            
            fig.update_layout(
                height=700,
                margin=dict(r=0, t=20, l=0, b=0),
                paper_bgcolor="#05070d",
                plot_bgcolor="#05070d",
                font=dict(color="#7ee7ff")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # جدول
            st.markdown("---")
            st.markdown("### 📊 District Summary")
            
            display_df = stats.sort_values("mean_velocity")
            display_df.columns = ["District", "Mean Velocity", "Max Subsidence", "Pixel Count"]
            
            st.dataframe(
                display_df.style.background_gradient(subset=["Mean Velocity"], cmap="Reds_r")
                    .background_gradient(subset=["Max Subsidence"], cmap="Reds_r")
                    .format({"Mean Velocity": "{:.1f}", "Max Subsidence": "{:.1f}", "Pixel Count": "{:,}"}),
                use_container_width=True,
                hide_index=True,
                height=400
            )
elif page == "📊 DISTRICT STATS":
    # ⭐ بنر منبع فعال (همه صفحات)
    _s = f"📡 {source_label}"
    if vel_active is not None:
        _n = int((~np.isnan(vel_active)).sum())
        _p = _n / vel_active.size * 100
        _s += f" · ✅ {_n:,} پیکسل ({_p:.1f}%)"
    
    if "Ascending" in source_label:
        _q = "🟢 کیفیت بالا · پوشش کم"
    elif "Descending" in source_label:
        _q = "🟡 کیفیت متوسط · پوشش کامل"
    elif "Combined" in source_label:
        _q = "🔵 ترکیبی · پوشش ۱۰۰٪"
    elif "Relative" in source_label:
        _q = "🟣 نسبی · median=0"
    else:
        _q = ""
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(0,217,255,0.15), rgba(0,255,136,0.1));
                border-left: 4px solid #00d9ff;
                padding: 0.6rem 1rem; margin-bottom: 1rem;
                border-radius: 8px; color: #7ee7ff;
                font-family: 'Courier New'; font-size: 0.85rem;">
        <b>📡 منبع فعال:</b> {_s}<br>
        <span style="font-size: 0.75rem; color: #99ffcc;">{_q}</span>
    </div>
    """, unsafe_allow_html=True)

    # ⭐ هدر
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0a0e27 0%, #1a1a4e 50%, #0a0e27 100%);
                padding: 1.5rem; border-radius: 15px; text-align: center;
                border: 2px solid #00d9ff;
                box-shadow: 0 0 40px rgba(0, 217, 255, 0.3);
                margin-bottom: 1.5rem;">
        <h2 style="color: #00d9ff; font-family: 'Courier New'; margin: 0;
                   letter-spacing: 3px; text-shadow: 0 0 15px #00d9ff;">
            ◉ DISTRICT STATISTICS MATRIX ◉
        </h2>
        <p style="color: #7ee7ff; font-family: 'Courier New';
                  font-size: 0.9rem; margin-top: 0.5rem; letter-spacing: 2px;">
            ▸ DETAILED ANALYSIS OF 22 TEHRAN DISTRICTS ◂
        </p>
    </div>
    """, unsafe_allow_html=True)

    if joined_df is not None:
        # ⭐ محاسبه آمار کامل
        stats = joined_df.groupby("district_name").agg(
            mean_velocity=("velocity", "mean"),
            max_subsidence=("velocity", "min"),
            std_velocity=("velocity", "std"),
            pixel_count=("velocity", "count")
        ).reset_index()

        # اضافه کردن جمعیت از df_pop
        if df_pop is not None:
            pop_data = df_pop[["district_name", "population", "affected_population", "risk_index", "risk_category"]].copy()
            stats = stats.merge(pop_data, on="district_name", how="left")

        stats = stats.sort_values("mean_velocity")

        # ⭐ کارت‌های آماری بالا
        n_critical = (stats["mean_velocity"] < -30).sum()
        n_warning = ((stats["mean_velocity"] >= -30) & (stats["mean_velocity"] < -20)).sum()
        n_stable = (stats["mean_velocity"] >= -20).sum()
        total_pop = stats["population"].sum() if "population" in stats.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card risk-critical">
                <h3>◉ CRITICAL</h3>
                <div class="value">{n_critical}</div>
                <div class="sub">districts</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card risk-warning">
                <h3>◉ WARNING</h3>
                <div class="value">{n_warning}</div>
                <div class="sub">districts</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card risk-stable">
                <h3>◉ STABLE</h3>
                <div class="value">{n_stable}</div>
                <div class="sub">districts</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>◉ POPULATION</h3>
                <div class="value">{total_pop/1e6:.2f}M</div>
                <div class="sub">total</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ⭐ جدول کامل
        st.markdown("### 📋 Full District Table")

        display_cols = ["district_name", "mean_velocity", "max_subsidence",
                       "std_velocity", "pixel_count"]
        if "population" in stats.columns:
            display_cols += ["population", "affected_population", "risk_index"]

        display_df = stats[display_cols].copy()
        display_df.columns = ["District", "Mean (mm/yr)", "Max Subsidence",
                              "Std Dev", "Pixels"] + (
                              ["Population", "Affected", "Risk Index"]
                              if "population" in stats.columns else [])

        st.dataframe(
            display_df.style
                .background_gradient(subset=["Mean (mm/yr)"], cmap="Reds_r")
                .background_gradient(subset=["Max Subsidence"], cmap="Reds_r")
                .format({
                    "Mean (mm/yr)": "{:.2f}",
                    "Max Subsidence": "{:.2f}",
                    "Std Dev": "{:.2f}",
                    "Pixels": "{:,}",
                    **({"Population": "{:,.0f}",
                        "Affected": "{:,.0f}",
                        "Risk Index": "{:.1f}"}
                       if "population" in stats.columns else {})
                }),
            use_container_width=True,
            hide_index=True,
            height=600
        )

        # ⭐ دکمه دانلود Excel
        try:
            import io
            import importlib
            importlib.import_module("openpyxl")
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                stats.to_excel(writer, sheet_name="District Stats", index=False)
            buffer.seek(0)
            st.download_button(
                label="📥 Download as Excel",
                data=buffer,
                file_name="tehran_district_stats.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            # fallback به CSV
            csv = stats.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="tehran_district_stats.csv",
                mime="text/csv"
            )

        st.markdown("---")

        # ⭐ نمودار میله‌ای
        st.markdown("### 📊 Subsidence Rate by District")

        chart_df = stats.sort_values("mean_velocity")
        
        colors = []
        for v in chart_df["mean_velocity"]:
            if v < -30:
                colors.append("#8B0000")
            elif v < -20:
                colors.append("#FF1432")
            elif v < -10:
                colors.append("#FF7800")
            elif v < 0:
                colors.append("#FFE600")
            else:
                colors.append("#00FF88")

        fig = go.Figure(data=[go.Bar(
            x=chart_df["mean_velocity"],
            y=chart_df["district_name"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#00d9ff", width=1)),
            text=[f"{v:.1f}" for v in chart_df["mean_velocity"]],
            textposition="outside",
            textfont=dict(color="#7ee7ff", size=10),
            hovertemplate="<b>%{y}</b><br>%{x:.2f} mm/yr<extra></extra>"
        )])

        fig.update_layout(
            height=700,
            margin=dict(r=30, t=20, l=10, b=20),
            paper_bgcolor="#0a0e27",
            plot_bgcolor="#0a0e27",
            font=dict(color="#7ee7ff", family="Courier New"),
            xaxis=dict(
                title=dict(text="Velocity (mm/yr)", font=dict(color="#7ee7ff")),
                gridcolor="rgba(0, 217, 255, 0.2)",
                zerolinecolor="#00d9ff",
                zerolinewidth=2
            ),
            yaxis=dict(
                gridcolor="rgba(0, 217, 255, 0.1)",
                tickfont=dict(color="#7ee7ff")
            ),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # ⭐ نمودار پراکندگی (velocity vs population)
        if "population" in stats.columns:
            st.markdown("---")
            st.markdown("### 🎯 Subsidence vs Population")

            # فیلتر NaN
            scatter_df = stats.dropna(subset=["mean_velocity", "population"])

            # اندازه بر اساس pixel_count
            sizes = np.sqrt(scatter_df["pixel_count"]) / 3

            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                x=scatter_df["mean_velocity"],
                y=scatter_df["population"] / 1000,
                mode="markers+text",
                text=scatter_df["district_name"],
                textposition="top center",
                textfont=dict(size=9, color="#7ee7ff"),
                marker=dict(
                    size=sizes,
                    color=scatter_df["mean_velocity"],
                    colorscale=[
                        [0.0, "#8B0000"],
                        [0.3, "#FF1432"],
                        [0.6, "#FFA500"],
                        [1.0, "#00FF88"]
                    ],
                    cmin=-200,
                    cmax=0,
                    colorbar=dict(
                        title=dict(text="Velocity (mm/yr)", font=dict(color="#7ee7ff")),
                        tickfont=dict(color="#7ee7ff"),
                        thickness=15
                    ),
                    line=dict(color="#00d9ff", width=1),
                    opacity=0.85
                ),
                hovertemplate="<b>%{text}</b><br>Velocity: %{x:.1f} mm/yr<br>Population: %{y:.0f}K<extra></extra>"
            ))

            fig2.update_layout(
                height=600,
                margin=dict(r=30, t=20, l=10, b=20),
                paper_bgcolor="#0a0e27",
                plot_bgcolor="#0a0e27",
                font=dict(color="#7ee7ff", family="Courier New"),
                xaxis=dict(
                    title=dict(text="Mean Velocity (mm/yr)", font=dict(color="#7ee7ff")),
                    gridcolor="rgba(0, 217, 255, 0.2)",
                    zerolinecolor="#00d9ff",
                    zerolinewidth=2
                ),
                yaxis=dict(
                    title=dict(text="Population (×1000)", font=dict(color="#7ee7ff")),
                    gridcolor="rgba(0, 217, 255, 0.1)"
                ),
                showlegend=False
            )

            st.plotly_chart(fig2, use_container_width=True)
elif page == "📈 TIME SERIES":
    # ⭐ بنر منبع فعال (همه صفحات)
    _s = f"📡 {source_label}"
    if vel_active is not None:
        _n = int((~np.isnan(vel_active)).sum())
        _p = _n / vel_active.size * 100
        _s += f" · ✅ {_n:,} پیکسل ({_p:.1f}%)"
    
    if "Ascending" in source_label:
        _q = "🟢 کیفیت بالا · پوشش کم"
    elif "Descending" in source_label:
        _q = "🟡 کیفیت متوسط · پوشش کامل"
    elif "Combined" in source_label:
        _q = "🔵 ترکیبی · پوشش ۱۰۰٪"
    elif "Relative" in source_label:
        _q = "🟣 نسبی · median=0"
    else:
        _q = ""
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(0,217,255,0.15), rgba(0,255,136,0.1));
                border-left: 4px solid #00d9ff;
                padding: 0.6rem 1rem; margin-bottom: 1rem;
                border-radius: 8px; color: #7ee7ff;
                font-family: 'Courier New'; font-size: 0.85rem;">
        <b>📡 منبع فعال:</b> {_s}<br>
        <span style="font-size: 0.75rem; color: #99ffcc;">{_q}</span>
    </div>
    """, unsafe_allow_html=True)

    # ⭐ هدر
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0a0e27 0%, #1a1a4e 50%, #0a0e27 100%);
                padding: 1.5rem; border-radius: 15px; text-align: center;
                border: 2px solid #00d9ff;
                box-shadow: 0 0 40px rgba(0, 217, 255, 0.3);
                margin-bottom: 1.5rem;">
        <h2 style="color: #00d9ff; font-family: 'Courier New'; margin: 0;
                   letter-spacing: 3px; text-shadow: 0 0 15px #00d9ff;">
            ◉ TIME SERIES ANALYSIS ◉
        </h2>
        <p style="color: #7ee7ff; font-family: 'Courier New';
                  font-size: 0.9rem; margin-top: 0.5rem; letter-spacing: 2px;">
            ▸ TEMPORAL DYNAMICS OF SUBSIDENCE ◂
        </p>
    </div>
    """, unsafe_allow_html=True)

    if joined_df is not None and timeseries is not None:
        # ⭐ محاسبه آمار مناطق
        district_stats = joined_df.groupby("district_name")["velocity"].mean().sort_values()
        all_districts = district_stats.index.tolist()
        top5_districts = district_stats.head(5).index.tolist()
        critical_districts = district_stats[district_stats < -30].index.tolist()

        # ⭐ دکمه‌های سریع
        st.markdown("### 🎯 Quick Select")
        qc1, qc2, qc3, qc4 = st.columns(4)

        if "ts_selected" not in st.session_state:
            st.session_state.ts_selected = top5_districts[:3]
        else:
            # ⭐ فیلتر district های موجود (بعد از تغییر Data Source)
            st.session_state.ts_selected = [
                d for d in st.session_state.ts_selected if d in all_districts
            ]

        with qc1:
            if st.button("🔴 All Critical", use_container_width=True):
                st.session_state.ts_selected = critical_districts
                st.rerun()
        with qc2:
            if st.button("⚠️ Top 5", use_container_width=True):
                st.session_state.ts_selected = top5_districts
                st.rerun()
        with qc3:
            if st.button("🌍 All 22", use_container_width=True):
                st.session_state.ts_selected = all_districts
                st.rerun()
        with qc4:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.ts_selected = []
                st.rerun()

        # ⭐ multiselect
        selected = st.multiselect(
            "📍 Select Districts:",
            options=all_districts,
            default=st.session_state.ts_selected
        )
        st.session_state.ts_selected = selected

        if selected:
            st.markdown("---")

            # ⭐ محاسبه سری زمانی + روند + CI
            from scipy import stats as scipy_stats

            days = np.array([
                (datetime.strptime(d, "%Y%m%d") - datetime.strptime(dates[0], "%Y%m%d")).days
                for d in dates
            ])

            colors = ["#FF1432", "#00d9ff", "#00FF88", "#FF7800", "#9b59b6", "#FFE600",
                     "#1abc9c", "#e74c3c", "#3498db", "#f39c12"]

            # ⭐ نمودار خطی اصلی
            st.markdown("### 📈 Cumulative Displacement Over Time")

            fig = go.Figure()

            for i, district in enumerate(selected):
                pts = joined_df[joined_df["district_name"] == district]
                if len(pts) == 0:
                    continue

                sample_size = min(200, len(pts))
                pts_sample = pts.sample(n=sample_size, random_state=42)

                # سری زمانی همه پیکسل‌ها
                all_ts = []
                for _, p in pts_sample.iterrows():
                    try:
                        row = int(p["row"]) if "row" in p.index else 0
                        col = int(p["col"]) if "col" in p.index else 0
                        row = max(0, min(timeseries.shape[1] - 1, row))
                        col = max(0, min(timeseries.shape[2] - 1, col))
                        ts = timeseries[:, row, col]
                        # ⭐ فیلتر NaN در ts[0]
                        if not np.all(np.isnan(ts)) and not np.isnan(ts[0]):
                            all_ts.append(ts - ts[0])
                    except:
                        continue

                if len(all_ts) == 0:
                    continue

                all_ts = np.array(all_ts)

                # میانگین و انحراف معیار
                mean_ts = np.nanmean(all_ts, axis=0)
                std_ts = np.nanstd(all_ts, axis=0)
                
                # ⭐ فقط فرونشست: هرچی مثبت → صفر
                mean_ts = np.minimum(mean_ts, 0.0)

                color = colors[i % len(colors)]

                # ⭐ باند اطمینان (95%)
                ci_upper = mean_ts + 1.96 * std_ts / np.sqrt(len(all_ts))
                ci_lower = mean_ts - 1.96 * std_ts / np.sqrt(len(all_ts))

                # باند اطمینان
                fig.add_trace(go.Scatter(
                    x=dates_fmt + dates_fmt[::-1],
                    y=list(ci_upper) + list(ci_lower[::-1]),
                    fill="toself",
                    fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba") if color.startswith("rgb") else f"rgba{tuple(list(bytes.fromhex(color[1:])) + [0.15])}",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip"
                ))

                # خط اصلی
                fig.add_trace(go.Scatter(
                    x=dates_fmt,
                    y=mean_ts,
                    mode="lines+markers",
                    name=district,
                    line=dict(width=3, color=color),
                    marker=dict(size=6, color=color, line=dict(color="#00d9ff", width=1)),
                    hovertemplate=f"<b>{district}</b><br>Displacement: %{{y:.2f}} mm<extra></extra>"
                ))

                # ⭐ خط روند
                valid_mask = ~np.isnan(mean_ts)
                if valid_mask.sum() > 5:
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                        days[valid_mask], mean_ts[valid_mask]
                    )
                    trend_line = slope * days + intercept
                    fig.add_trace(go.Scatter(
                        x=dates_fmt,
                        y=trend_line,
                        mode="lines",
                        name=f"{district} (trend)",
                        line=dict(width=2, color=color, dash="dash"),
                        showlegend=False,
                        hoverinfo="skip"
                    ))

            fig.add_hline(y=0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)")

            fig.update_layout(
                height=600,
                margin=dict(r=30, t=30, l=20, b=20),
                paper_bgcolor="#0a0e27",
                plot_bgcolor="#0a0e27",
                font=dict(color="#7ee7ff", family="Courier New"),
                xaxis=dict(
                    title=dict(text="Date", font=dict(color="#7ee7ff")),
                    gridcolor="rgba(0, 217, 255, 0.15)"
                ),
                yaxis=dict(
                    title=dict(text="Cumulative Displacement (mm)", font=dict(color="#7ee7ff")),
                    gridcolor="rgba(0, 217, 255, 0.15)",
                    zerolinecolor="#00d9ff",
                    zerolinewidth=2
                ),
                legend=dict(
                    bgcolor="rgba(10, 14, 39, 0.8)",
                    bordercolor="#00d9ff",
                    borderwidth=1
                ),
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            # ⭐ جدول آمار زمانی
            st.markdown("---")
            st.markdown("### 📊 Temporal Statistics")

            ts_stats = []
            for district in selected:
                pts = joined_df[joined_df["district_name"] == district]
                if len(pts) == 0:
                    continue

                sample_size = min(200, len(pts))
                pts_sample = pts.sample(n=sample_size, random_state=42)

                all_ts = []
                for _, p in pts_sample.iterrows():
                    try:
                        row = int(p["row"]) if "row" in p.index else 0
                        col = int(p["col"]) if "col" in p.index else 0
                        row = max(0, min(timeseries.shape[1] - 1, row))
                        col = max(0, min(timeseries.shape[2] - 1, col))
                        ts = timeseries[:, row, col]
                        # ⭐ فیلتر NaN در ts[0]
                        if not np.all(np.isnan(ts)) and not np.isnan(ts[0]):
                            all_ts.append(ts - ts[0])
                    except:
                        continue

                if len(all_ts) == 0:
                    continue

                all_ts = np.array(all_ts)
                mean_ts = np.nanmean(all_ts, axis=0)

                # فیت خطی
                valid_mask = ~np.isnan(mean_ts)
                if valid_mask.sum() > 5:
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                        days[valid_mask], mean_ts[valid_mask]
                    )
                    rate_per_year = slope * 365
                    r_squared = r_value ** 2

                    # ⭐ فقط فرونشست: اگه مثبت بود → صفر
                    if rate_per_year > 0:
                        rate_per_year = 0.0

                    # پیش‌بینی ۵ ساله از dashboard_predictions (اگه موجود)
                    future_5yr = rate_per_year * 5  # پیش‌فرض: خطی
                    try:
                        if pred_data and district in pred_data:
                            _obs = pred_data[district].get('observed_values', [])
                            _fut = pred_data[district].get('future_values', [])
                            if _obs and _fut:
                                _delta = float(_fut[-1]) - float(_obs[0])
                                future_5yr = min(0.0, _delta)  # فقط منفی
                    except Exception:
                        pass
                    
                    # ⭐ اگه future_5yr مثبت بود → صفر
                    if future_5yr > 0:
                        future_5yr = 0.0

                    # نوسان فصلی
                    residuals = mean_ts[valid_mask] - (slope * days[valid_mask] + intercept)
                    seasonal_amp = np.std(residuals)

                    ts_stats.append({
                        "District": district,
                        "Rate (mm/yr)": round(rate_per_year, 2),
                        "R²": round(r_squared, 3),
                        "Seasonal ±": round(seasonal_amp, 2),
                        "5yr Forecast (mm)": round(future_5yr, 1)
                    })

            if ts_stats:
                ts_df = pd.DataFrame(ts_stats).sort_values("Rate (mm/yr)")

                st.dataframe(
                    ts_df.style
                        .background_gradient(subset=["Rate (mm/yr)"], cmap="Reds_r")
                        .background_gradient(subset=["5yr Forecast (mm)"], cmap="Reds_r")
                        .format({
                            "Rate (mm/yr)": "{:.2f}",
                            "R²": "{:.3f}",
                            "Seasonal ±": "{:.2f}",
                            "5yr Forecast (mm)": "{:.1f}"
                        }),
                    use_container_width=True,
                    hide_index=True
                )

            # ⭐ نمودار روند میله‌ای
            st.markdown("---")
            st.markdown("### 📊 Trend Rate Comparison")

            if ts_stats:
                trend_df = pd.DataFrame(ts_stats).sort_values("Rate (mm/yr)")

                colors_bar = []
                for v in trend_df["Rate (mm/yr)"]:
                    if v < -30:
                        colors_bar.append("#8B0000")
                    elif v < -20:
                        colors_bar.append("#FF1432")
                    elif v < -10:
                        colors_bar.append("#FF7800")
                    elif v < 0:
                        colors_bar.append("#FFE600")
                    else:
                        colors_bar.append("#00FF88")

                fig2 = go.Figure(data=[go.Bar(
                    x=trend_df["Rate (mm/yr)"],
                    y=trend_df["District"],
                    orientation="h",
                    marker=dict(color=colors_bar, line=dict(color="#00d9ff", width=1)),
                    text=[f"{v:.1f}" for v in trend_df["Rate (mm/yr)"]],
                    textposition="outside",
                    textfont=dict(color="#7ee7ff", size=10),
                    hovertemplate="<b>%{y}</b><br>%{x:.2f} mm/yr<extra></extra>"
                )])

                fig2.update_layout(
                    height=max(400, len(trend_df) * 40),
                    margin=dict(r=30, t=20, l=10, b=20),
                    paper_bgcolor="#0a0e27",
                    plot_bgcolor="#0a0e27",
                    font=dict(color="#7ee7ff", family="Courier New"),
                    xaxis=dict(
                        title=dict(text="Rate (mm/yr)", font=dict(color="#7ee7ff")),
                        gridcolor="rgba(0, 217, 255, 0.2)",
                        zerolinecolor="#00d9ff",
                        zerolinewidth=2
                    ),
                    yaxis=dict(gridcolor="rgba(0, 217, 255, 0.1)"),
                    showlegend=False
                )

                st.plotly_chart(fig2, use_container_width=True)

                # دانلود CSV
                csv = ts_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Time Series Stats (CSV)",
                    data=csv,
                    file_name="tehran_timeseries_stats.csv",
                    mime="text/csv"
                )

        else:
            st.info("💡 Select districts from the dropdown above")
elif page == "🔮 FORECAST 2029":
    # ⭐ بنر منبع فعال (همه صفحات)
    _s = f"📡 {source_label}"
    if vel_active is not None:
        _n = int((~np.isnan(vel_active)).sum())
        _p = _n / vel_active.size * 100
        _s += f" · ✅ {_n:,} پیکسل ({_p:.1f}%)"
    
    if "Ascending" in source_label:
        _q = "🟢 کیفیت بالا · پوشش کم"
    elif "Descending" in source_label:
        _q = "🟡 کیفیت متوسط · پوشش کامل"
    elif "Combined" in source_label:
        _q = "🔵 ترکیبی · پوشش ۱۰۰٪"
    elif "Relative" in source_label:
        _q = "🟣 نسبی · median=0"
    else:
        _q = ""
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(0,217,255,0.15), rgba(0,255,136,0.1));
                border-left: 4px solid #00d9ff;
                padding: 0.6rem 1rem; margin-bottom: 1rem;
                border-radius: 8px; color: #7ee7ff;
                font-family: 'Courier New'; font-size: 0.85rem;">
        <b>📡 منبع فعال:</b> {_s}<br>
        <span style="font-size: 0.75rem; color: #99ffcc;">{_q}</span>
    </div>
    """, unsafe_allow_html=True)

    # ⭐ هدر
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a0a27 0%, #4e1a4e 50%, #1a0a27 100%);
                padding: 2rem; border-radius: 15px; text-align: center;
                border: 2px solid #ff00ff;
                box-shadow: 0 0 50px rgba(255, 0, 255, 0.4);
                margin-bottom: 1.5rem;">
        <h1 style="color: #ff00ff; font-family: 'Courier New'; margin: 0;
                   font-size: 2rem; letter-spacing: 4px;
                   text-shadow: 0 0 20px #ff00ff;">
            ◉ SUBSIDENCE FORECAST 2029 ◉
        </h1>
        <p style="color: #ff99ff; font-family: 'Courier New';
                  font-size: 1rem; margin-top: 0.7rem; letter-spacing: 2px;">
            ▸ PREDICTING THE FUTURE USING ML REGRESSION ◂
        </p>
    </div>
    """, unsafe_allow_html=True)

    if df_pred is not None:
        # ⭐ ۵ کارت KPI
        worst = df_pred.iloc[0]
        n_very_high = (df_pred["Future_Risk"] == "🔴 Very High").sum()
        n_high = (df_pred["Future_Risk"] == "🟠 High").sum()
        mean_pred = df_pred["Predicted_2029_mm"].mean()

        # جمعیت در خطر ۲۰۲۹
        if df_pop is not None:
            pop_2029 = df_pop["affected_population"].sum()
        else:
            pop_2029 = 0

        st.markdown("### 🎯 Key Metrics")

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(f"""
            <div class="metric-card risk-critical">
                <h3>◉ WORST CASE</h3>
                <div class="value">{worst["Predicted_2029_mm"]:.0f}</div>
                <div class="sub">mm by 2029</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card risk-critical">
                <h3>◉ VERY HIGH</h3>
                <div class="value">{n_very_high}</div>
                <div class="sub">districts</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card risk-warning">
                <h3>◉ HIGH RISK</h3>
                <div class="value">{n_high}</div>
                <div class="sub">districts</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>◉ POPULATION</h3>
                <div class="value">{pop_2029/1e6:.2f}M</div>
                <div class="sub">at risk</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <h3>◉ HORIZON</h3>
                <div class="value">5</div>
                <div class="sub">years</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ⭐ بخش ۱: سری زمانی با پیش‌بینی
        st.markdown("### 📈 Historical + Forecast Time Series")

        try:
            with open("/content/dashboard_predictions.json") as f:
                pred_data = json.load(f)
            
            # ⭐ فقط فرونشست: مقادیر مثبت رو صفر کن
            for _dk, _dv in pred_data.items():
                if isinstance(_dv, dict):
                    for _field in ['observed_values', 'future_values', 'ci_upper', 'ci_lower']:
                        if _field in _dv and isinstance(_dv[_field], list):
                            _dv[_field] = [min(0.0, float(x)) if x is not None else 0.0 for x in _dv[_field]]
        except Exception as _e:
            pred_data = {}

        if pred_data:
            # انتخاب مناطق برای نمایش
            top_districts = df_pred.head(5)["District"].tolist()

            selected_pred = st.multiselect(
                "📍 Select Districts to Forecast:",
                options=df_pred["District"].tolist(),
                default=top_districts[:3]
            )

            if selected_pred:
                fig = go.Figure()
                colors = ["#FF1432", "#ff00ff", "#00d9ff", "#00FF88", "#FF7800", "#9b59b6"]

                for i, district in enumerate(selected_pred):
                    if district not in pred_data:
                        continue

                    d = pred_data[district]
                    color = colors[i % len(colors)]

                    # مشاهدات
                    fig.add_trace(go.Scatter(
                        x=d["observed_dates"],
                        y=d["observed_values"],
                        mode="lines+markers",
                        name=f"{district} (Observed)",
                        line=dict(color=color, width=3),
                        marker=dict(size=6)
                    ))

                    # پیش‌بینی
                    fig.add_trace(go.Scatter(
                        x=d["future_dates"],
                        y=d["future_values"],
                        mode="lines",
                        name=f"{district} (Forecast)",
                        line=dict(color=color, width=3, dash="dash")
                    ))

                    # باند اطمینان
                    fig.add_trace(go.Scatter(
                        x=d["future_dates"] + d["future_dates"][::-1],
                        y=d["ci_upper"] + d["ci_lower"][::-1],
                        fill="toself",
                        fillcolor=f"rgba{tuple(list(bytes.fromhex(color[1:])) + [0.15])}",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                        hoverinfo="skip"
                    ))

                # خط جداکننده
                try:
                    fig.add_vline(
                        x=d.get("observed_dates", d.get("dates", [0]))[-1] if (d.get("observed_dates") or d.get("dates")) else 0,
                        line_dash="dash",
                        line_color="#ff6b6b",
                        line_width=2
                    )
                except Exception:
                    pass

                fig.update_layout(
                    height=600,
                    margin=dict(r=30, t=30, l=20, b=20),
                    paper_bgcolor="#0a0e27",
                    plot_bgcolor="#0a0e27",
                    font=dict(color="#7ee7ff", family="Courier New"),
                    xaxis=dict(title="Date", gridcolor="rgba(255, 0, 255, 0.15)"),
                    yaxis=dict(
                        title="Cumulative Displacement (mm)",
                        gridcolor="rgba(255, 0, 255, 0.15)",
                        zerolinecolor="#ff00ff",
                        zerolinewidth=2
                    ),
                    legend=dict(
                        bgcolor="rgba(26, 10, 39, 0.8)",
                        bordercolor="#ff00ff",
                        borderwidth=1
                    ),
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ⭐ بخش ۲: نقشه ۲۰۲۹
        col_map, col_table = st.columns([2, 1])

        with col_map:
            st.markdown("### 🗺️ 2029 Risk Map")

            if districts_gdf is not None:
                # ادغام با پیش‌بینی
                pred_map = df_pred[["District", "Predicted_2029_mm", "Future_Risk"]].copy()
                pred_map.columns = ["district_name", "predicted_2029", "future_risk"]

                merged = districts_gdf.merge(pred_map, on="district_name", how="left")
                merged["predicted_2029"] = merged["predicted_2029"].fillna(0)

                merged_geojson = json.loads(merged.to_json())

                fig_map = px.choropleth_mapbox(
                    merged,
                    geojson=merged_geojson,
                    locations=merged.index,
                    color="predicted_2029",
                    color_continuous_scale=[
                        [0.0, "#8B0000"],
                        [0.2, "#FF1432"],
                        [0.5, "#ff00ff"],
                        [0.8, "#FFA500"],
                        [1.0, "#00FF88"]
                    ],
                    range_color=(-400, 100),
                    mapbox_style="carto-darkmatter",
                    mapbox_token=MAPBOX_TOKEN if MAPBOX_TOKEN else None,
                    zoom=10,
                    center={"lat": 35.7, "lon": 51.4},
                    opacity=0.8,
                    hover_name="district_name",
                    hover_data={"predicted_2029": ":.1f", "district_name": False}
                )

                fig_map.update_traces(marker_line_width=3, marker_line_color="#ff00ff")

                fig_map.update_layout(
                    height=500,
                    margin=dict(r=0, t=10, l=0, b=0),
                    paper_bgcolor="#0a0e27",
                    font=dict(color="#7ee7ff", family="Courier New"),
                    coloraxis_colorbar=dict(
                        title="2029 (mm)",
                        tickfont=dict(color="#7ee7ff"),
                        thickness=15
                    )
                )

                st.plotly_chart(fig_map, use_container_width=True)

        with col_table:
            st.markdown("### 📋 Top 10")

            top10 = df_pred.head(10)[["District", "Predicted_2029_mm", "Future_Risk"]].copy()
            top10.columns = ["District", "2029 (mm)", "Risk"]

            st.dataframe(
                top10.style
                    .background_gradient(subset=["2029 (mm)"], cmap="Reds_r")
                    .format({"2029 (mm)": "{:.1f}"}),
                use_container_width=True,
                hide_index=True,
                height=500
            )

        st.markdown("---")

        # ⭐ بخش ۳: مقایسه ۲۰۲۴ vs ۲۰۲۹
        st.markdown("### 📊 2024 vs 2029 Comparison")

        comparison = df_pred[["District", "Current_Rate_mm_yr", "Predicted_2029_mm"]].copy()
        comparison["Current_2024"] = comparison["Current_Rate_mm_yr"] * 1
        comparison = comparison.sort_values("Predicted_2029_mm").head(15)

        fig_cmp = go.Figure()

        fig_cmp.add_trace(go.Bar(
            x=comparison["District"],
            y=comparison["Current_2024"],
            name="2024 (annual)",
            marker=dict(color="#00d9ff", line=dict(color="#7ee7ff", width=1)),
            hovertemplate="<b>%{x}</b><br>2024: %{y:.1f} mm/yr<extra></extra>"
        ))

        fig_cmp.add_trace(go.Bar(
            x=comparison["District"],
            y=comparison["Predicted_2029_mm"],
            name="2029 (cumulative)",
            marker=dict(color="#ff00ff", line=dict(color="#ff99ff", width=1)),
            hovertemplate="<b>%{x}</b><br>2029: %{y:.1f} mm<extra></extra>"
        ))

        fig_cmp.update_layout(
            height=500,
            margin=dict(r=30, t=20, l=20, b=80),
            paper_bgcolor="#0a0e27",
            plot_bgcolor="#0a0e27",
            font=dict(color="#7ee7ff", family="Courier New"),
            barmode="group",
            xaxis=dict(tickangle=-45, gridcolor="rgba(255, 0, 255, 0.1)"),
            yaxis=dict(
                title="Displacement",
                gridcolor="rgba(255, 0, 255, 0.15)"
            ),
            legend=dict(
                bgcolor="rgba(26, 10, 39, 0.8)",
                bordercolor="#ff00ff",
                borderwidth=1
            )
        )

        st.plotly_chart(fig_cmp, use_container_width=True)

        # ⭐ هشدار متنی
        st.markdown("---")

        # ⭐ Fix: worst رow از df_pred
        _worst_row = df_pred.loc[df_pred["Predicted_2029_mm"].idxmin()]
        worst_name = _worst_row["District"]
        worst_val = float(_worst_row["Predicted_2029_mm"])
        building_floors = abs(worst_val) / 3  # هر طبقه 3 متر

        st.markdown(f"""
        <div style="background: linear-gradient(90deg, rgba(255, 0, 255, 0.1), rgba(255, 0, 0, 0.1));
                    border: 2px solid #ff00ff; border-radius: 10px;
                    padding: 1.5rem; margin: 1rem 0;">
            <h3 style="color: #ff00ff; margin: 0; font-family: 'Courier New';">
                ⚠️ CRITICAL ALERT
            </h3>
            <p style="color: #ff99ff; font-family: 'Courier New'; margin-top: 0.5rem; font-size: 1.1rem;">
                <b>{worst_name}</b> will subside by <b>{worst_val:.0f} mm</b> by 2029 —
                equivalent to a <b>{building_floors:.0f}-story building</b> sinking into the ground.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ⭐ دانلود
        csv = df_pred.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Data (CSV)",
            data=csv,
            file_name="tehran_forecast_2029.csv",
            mime="text/csv"
        )
elif page == "🎬 ANIMATION":
    # ⭐ بنر منبع فعال (همه صفحات)
    _s = f"📡 {source_label}"
    if vel_active is not None:
        _n = int((~np.isnan(vel_active)).sum())
        _p = _n / vel_active.size * 100
        _s += f" · ✅ {_n:,} پیکسل ({_p:.1f}%)"
    
    if "Ascending" in source_label:
        _q = "🟢 کیفیت بالا · پوشش کم"
    elif "Descending" in source_label:
        _q = "🟡 کیفیت متوسط · پوشش کامل"
    elif "Combined" in source_label:
        _q = "🔵 ترکیبی · پوشش ۱۰۰٪"
    elif "Relative" in source_label:
        _q = "🟣 نسبی · median=0"
    else:
        _q = ""
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(0,217,255,0.15), rgba(0,255,136,0.1));
                border-left: 4px solid #00d9ff;
                padding: 0.6rem 1rem; margin-bottom: 1rem;
                border-radius: 8px; color: #7ee7ff;
                font-family: 'Courier New'; font-size: 0.85rem;">
        <b>📡 منبع فعال:</b> {_s}<br>
        <span style="font-size: 0.75rem; color: #99ffcc;">{_q}</span>
    </div>
    """, unsafe_allow_html=True)

    # ⭐ هدر سای‌فای
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a0a2e 0%, #4a1a4a 50%, #1a0a2e 100%);
                padding: 2rem; border-radius: 15px; text-align: center;
                border: 2px solid #ff00ff;
                box-shadow: 0 0 60px rgba(255, 0, 255, 0.5);
                margin-bottom: 1.5rem;">
        <h1 style="color: #ff00ff; font-family: 'Courier New'; margin: 0;
                   font-size: 2rem; letter-spacing: 4px;
                   text-shadow: 0 0 25px #ff00ff;">
            ◉ SUBSIDENCE TIME MACHINE ◉
        </h1>
        <p style="color: #ff99ff; font-family: 'Courier New';
                  font-size: 1rem; margin-top: 0.7rem; letter-spacing: 3px;">
            ▸ TRAVEL THROUGH TIME AND WITNESS THE FUTURE ◂
        </p>
    </div>
    """, unsafe_allow_html=True)

    if timeseries is not None and joined_df is not None:
        # ⭐ آماده‌سازی داده‌ها
        if "anim_frame" not in st.session_state:
            st.session_state.anim_frame = 0
        if "anim_playing" not in st.session_state:
            st.session_state.anim_playing = False
        if "anim_zoom" not in st.session_state:
            st.session_state.anim_zoom = "All Tehran"

        # ⭐ بخش ۱: کنترل‌ها
        st.markdown("### 🎬 Time Machine Controls")

        ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1, 1, 1, 1, 2])

        with ctrl1:
            if st.button("▶️ Play", use_container_width=True, key="play_btn"):
                st.session_state.anim_playing = True
        with ctrl2:
            if st.button("⏸️ Pause", use_container_width=True, key="pause_btn"):
                st.session_state.anim_playing = False
        with ctrl3:
            if st.button("⏹️ Reset", use_container_width=True, key="reset_btn"):
                st.session_state.anim_frame = 0
                st.session_state.anim_playing = False
        with ctrl4:
            if st.button("⏭️ End", use_container_width=True, key="end_btn"):
                st.session_state.anim_frame = len(dates) - 1
                st.session_state.anim_playing = False
        with ctrl5:
            speed = st.slider("⚡ Speed (seconds/frame)", 0.1, 2.0, 0.5, 0.1)

        # ⭐ انتخاب Zoom
        st.markdown("#### 🎯 Zoom")
        all_districts_anim = ["All Tehran"] + sorted(joined_df["district_name"].unique().tolist())
        st.session_state.anim_zoom = st.selectbox(
            "Zoom to District:",
            options=all_districts_anim,
            index=all_districts_anim.index(st.session_state.anim_zoom) if st.session_state.anim_zoom in all_districts_anim else 0,
            label_visibility="collapsed"
        )

        # ⭐ اسلایدر تایم‌لاین
        st.markdown("#### ⏱ Timeline")
        frame_idx = st.slider(
            "Date Selection",
            min_value=0,
            max_value=len(dates) - 1,
            value=st.session_state.anim_frame,
            format="",
            label_visibility="collapsed"
        )
        st.session_state.anim_frame = frame_idx

        # ⭐ بخش ۲: آمار زنده
        st.markdown("---")

        # محاسبه جابجایی تجمعی (فقط منفی)
        cumulative = timeseries[frame_idx] - timeseries[0]
        cumulative = np.minimum(cumulative, 0.0)

        current_date = dates_fmt[frame_idx]
        mean_disp = np.nanmean(cumulative)
        min_disp = np.nanmin(cumulative)
        max_disp = np.nanmax(cumulative)

        # نقاط بحرانی
        critical_count = int(np.sum(cumulative < -100))

        # مساحت تخریب‌شده
        valid_pixels = np.sum(~np.isnan(cumulative))
        critical_pct = critical_count / valid_pixels * 100 if valid_pixels > 0 else 0

        # ⭐ کارت‌های آمار زنده
        st.markdown("### 📊 Live Statistics")

        stat1, stat2, stat3, stat4, stat5 = st.columns(5)

        with stat1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a0a2e 0%, #2a1a3e 100%);
                        padding: 1rem; border-radius: 10px;
                        border: 2px solid #ff00ff;
                        text-align: center;">
                <div style="color: #ff99ff; font-family: 'Courier New'; font-size: 0.75rem;">
                    📅 DATE
                </div>
                <div style="color: #ff00ff; font-family: 'Courier New';
                            font-size: 1.2rem; font-weight: 700;
                            text-shadow: 0 0 10px #ff00ff; margin-top: 0.3rem;">
                    {current_date}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with stat2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a0a2e 0%, #2a1a3e 100%);
                        padding: 1rem; border-radius: 10px;
                        border: 2px solid #ff00ff;
                        text-align: center;">
                <div style="color: #ff99ff; font-family: 'Courier New'; font-size: 0.75rem;">
                    📊 MEAN
                </div>
                <div style="color: #ff00ff; font-family: 'Courier New';
                            font-size: 1.2rem; font-weight: 700;
                            text-shadow: 0 0 10px #ff00ff; margin-top: 0.3rem;">
                    {mean_disp:.1f} mm
                </div>
            </div>
            """, unsafe_allow_html=True)

        with stat3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2e0a0a 0%, #3e1a1a 100%);
                        padding: 1rem; border-radius: 10px;
                        border: 2px solid #FF1432;
                        text-align: center;">
                <div style="color: #ff9999; font-family: 'Courier New'; font-size: 0.75rem;">
                    🔻 MIN
                </div>
                <div style="color: #FF1432; font-family: 'Courier New';
                            font-size: 1.2rem; font-weight: 700;
                            text-shadow: 0 0 10px #FF1432; margin-top: 0.3rem;">
                    {min_disp:.1f} mm
                </div>
            </div>
            """, unsafe_allow_html=True)

        with stat4:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2e1a0a 0%, #3e2a1a 100%);
                        padding: 1rem; border-radius: 10px;
                        border: 2px solid #FF7800;
                        text-align: center;">
                <div style="color: #ffcc99; font-family: 'Courier New'; font-size: 0.75rem;">
                    🚨 CRITICAL
                </div>
                <div style="color: #FF7800; font-family: 'Courier New';
                            font-size: 1.2rem; font-weight: 700;
                            text-shadow: 0 0 10px #FF7800; margin-top: 0.3rem;">
                    {critical_count:,}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with stat5:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a2e0a 0%, #2a3e1a 100%);
                        padding: 1rem; border-radius: 10px;
                        border: 2px solid #00FF88;
                        text-align: center;">
                <div style="color: #99ffcc; font-family: 'Courier New'; font-size: 0.75rem;">
                    📈 PROGRESS
                </div>
                <div style="color: #00FF88; font-family: 'Courier New';
                            font-size: 1.2rem; font-weight: 700;
                            text-shadow: 0 0 10px #00FF88; margin-top: 0.3rem;">
                    {critical_pct:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ⭐ بخش ۳: نقشه انیمیشن
        st.markdown("---")
        st.markdown("### 🗺️ Tehran Subsidence Animation")

        # نمونه‌برداری
        sample_size = min(5000, len(joined_df))
        sample_df = joined_df.sample(n=sample_size, random_state=42).copy()

        # استخراج جابجایی تجمعی برای نمونه
        sample_cumulative = []
        for _, p in sample_df.iterrows():
            try:
                r = int(p["row"]) if "row" in p.index else 0
                c = int(p["col"]) if "col" in p.index else 0
                r = max(0, min(cumulative.shape[0]-1, r))
                c = max(0, min(cumulative.shape[1]-1, c))
                val = cumulative[r, c]
                sample_cumulative.append(val if not np.isnan(val) else 0)
            except:
                sample_cumulative.append(0)

        sample_df["cumulative"] = sample_cumulative

        # ⭐ ساخت نقشه
        fig = go.Figure()

        # نقاط با رنگ حرارتی
        fig.add_trace(go.Scattermapbox(
            lon=sample_df["lon"],
            lat=sample_df["lat"],
            mode="markers",
            marker=dict(
                size=6,
                color=sample_df["cumulative"],
                colorscale=[
                    [0.0, "#8B0000"],
                    [0.15, "#FF1432"],
                    [0.35, "#FF4500"],
                    [0.55, "#FFA500"],
                    [0.75, "#FFE600"],
                    [1.0, "#00FF88"]
                ],
                cmin=-500,
                cmax=0,
                colorbar=dict(
                    title=dict(text="Displacement<br>(mm)", font=dict(color="#ff99ff", size=12)),
                    tickfont=dict(color="#ff99ff"),
                    thickness=15,
                    len=0.8
                ),
                opacity=0.85
            ),
            hovertemplate="<b>%{marker.color:.1f} mm</b><extra></extra>",
            showlegend=False
        ))

        # مرزها
        if districts_gdf is not None:
            for _, r in districts_gdf.iterrows():
                g = r.geometry
                ps = [g] if g.geom_type == "Polygon" else list(g.geoms) if g.geom_type == "MultiPolygon" else []
                for p in ps:
                    coords = list(p.exterior.coords)
                    fig.add_trace(go.Scattermapbox(
                        lon=[c[0] for c in coords],
                        lat=[c[1] for c in coords],
                        mode="lines",
                        line=dict(color="#ff00ff", width=2),
                        showlegend=False,
                        hoverinfo="skip"
                    ))

        # مترو
        if metro_gdf is not None:
            for _, r in metro_gdf.iterrows():
                g = r.geometry
                ls = [g] if g.geom_type == "LineString" else list(g.geoms) if g.geom_type == "MultiLineString" else []
                for line in ls:
                    coords = list(line.coords)
                    fig.add_trace(go.Scattermapbox(
                        lon=[c[0] for c in coords],
                        lat=[c[1] for c in coords],
                        mode="lines",
                        line=dict(color="#00FFC8", width=2),
                        showlegend=False,
                        hoverinfo="skip"
                    ))

        # فرودگاه‌ها
        if airports_gdf is not None:
            for _, r in airports_gdf.iterrows():
                g = r.geometry
                if g.geom_type == "Point":
                    lon, lat = g.x, g.y
                elif g.geom_type == "Polygon":
                    c = g.centroid
                    lon, lat = c.x, c.y
                else:
                    continue
                fig.add_trace(go.Scattermapbox(
                    lon=[lon],
                    lat=[lat],
                    mode="markers",
                    marker=dict(size=15, color="#9b59b6", symbol="airport"),
                    showlegend=False,
                    hovertemplate="<b>✈️ Airport</b><extra></extra>"
                ))

        # ⭐ View State با zoom
        view_state = {
            "style": "carto-darkmatter",
            "accesstoken": MAPBOX_TOKEN if MAPBOX_TOKEN else None,
            "center": {"lat": 35.7, "lon": 51.4},
            "zoom": 11
        }

        if st.session_state.anim_zoom != "All Tehran":
            zoom_pts = joined_df[joined_df["district_name"] == st.session_state.anim_zoom]
            if len(zoom_pts) > 0:
                view_state["center"] = {
                    "lat": float(zoom_pts["lat"].mean()),
                    "lon": float(zoom_pts["lon"].mean())
                }
                view_state["zoom"] = 13

        fig.update_layout(
            mapbox=view_state,
            height=600,
            margin=dict(r=0, t=20, l=0, b=0),
            paper_bgcolor="#0a0e27",
            font=dict(color="#ff99ff", family="Courier New"),
            title=dict(
                text=f"📅 {current_date}",
                font=dict(color="#ff00ff", size=16),
                x=0.5,
                xanchor="center"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # ⭐ بخش ۴: نمودار پیشرفت
        st.markdown("---")
        st.markdown("### 📈 Progress Timeline")

        # محاسبه میانگین تجمعی برای همه فریم‌ها
        progress_values = []
        for i in range(len(dates)):
            cum = timeseries[i] - timeseries[0]
            cum = np.minimum(cum, 0.0)
            progress_values.append(np.nanmean(cum))

        fig_prog = go.Figure()

        # خط کامل
        fig_prog.add_trace(go.Scatter(
            x=dates_fmt,
            y=progress_values,
            mode="lines",
            line=dict(color="#ff00ff", width=2),
            name="Mean Displacement",
            showlegend=False
        ))

        # نقطه فعلی
        fig_prog.add_trace(go.Scatter(
            x=[dates_fmt[frame_idx]],
            y=[progress_values[frame_idx]],
            mode="markers",
            marker=dict(size=20, color="#FF1432", symbol="circle",
                       line=dict(color="#ff00ff", width=3)),
            name="Current",
            showlegend=False,
            hovertemplate=f"<b>{dates_fmt[frame_idx]}</b><br>{progress_values[frame_idx]:.1f} mm<extra></extra>"
        ))

        fig_prog.update_layout(
            height=300,
            margin=dict(r=30, t=20, l=20, b=20),
            paper_bgcolor="#0a0e27",
            plot_bgcolor="#0a0e27",
            font=dict(color="#ff99ff", family="Courier New"),
            xaxis=dict(title=dict(text="Date", font=dict(color="#ff99ff")), gridcolor="rgba(255, 0, 255, 0.15)"),
            yaxis=dict(
                title=dict(text="Mean Displacement (mm)", font=dict(color="#ff99ff")),
                gridcolor="rgba(255, 0, 255, 0.15)",
                zerolinecolor="#ff00ff",
                zerolinewidth=2
            ),
            hovermode="x unified"
        )

        st.plotly_chart(fig_prog, use_container_width=True)

        # ⭐ بخش ۵: Event Log
        st.markdown("---")
        st.markdown("### 🔴 Event Log")

        # رویدادهای کلیدی
        events = []

        # اولین تاریخ
        events.append(("📍", dates_fmt[0], "First SAR acquisition"))

        # وقتی District 19 از -50 رد شد
        for i, d in enumerate(dates_fmt):
            cum = timeseries[i] - timeseries[0]
            if np.nanmin(cum) < -100:
                events.append(("🚨", d, f"Cumulative subsidence reaches -100 mm (Max: {np.nanmin(cum):.0f} mm)"))
                break

        # وقتی میانگین از -50 رد شد
        for i, d in enumerate(dates_fmt):
            cum = timeseries[i] - timeseries[0]
            if np.nanmean(cum) < -30:
                events.append(("⚠️", d, f"Mean displacement exceeds -30 mm"))
                break

        # آخرین تاریخ
        events.append(("🏁", dates_fmt[-1], "End of analysis period"))

        for icon, date, desc in events:
            st.markdown(f"""
            <div style="background: rgba(26, 10, 46, 0.6);
                        border-left: 4px solid #ff00ff;
                        padding: 0.7rem 1rem; margin: 0.5rem 0;
                        border-radius: 5px;">
                <span style="color: #ff00ff; font-size: 1.2rem;">{icon}</span>
                <span style="color: #ff99ff; font-family: 'Courier New';
                            margin: 0 1rem; font-weight: 700;">{date}</span>
                <span style="color: #7ee7ff; font-family: 'Courier New';">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        # ⭐ بخش ۶: Play loop
        if st.session_state.anim_playing:
            if frame_idx < len(dates) - 1:
                time.sleep(speed)
                st.session_state.anim_frame = frame_idx + 1
                st.rerun()
            else:
                st.session_state.anim_playing = False
elif page == "🚨 RISK MAP":
    # ⭐ بنر منبع فعال (همه صفحات)
    _s = f"📡 {source_label}"
    if vel_active is not None:
        _n = int((~np.isnan(vel_active)).sum())
        _p = _n / vel_active.size * 100
        _s += f" · ✅ {_n:,} پیکسل ({_p:.1f}%)"
    
    if "Ascending" in source_label:
        _q = "🟢 کیفیت بالا · پوشش کم"
    elif "Descending" in source_label:
        _q = "🟡 کیفیت متوسط · پوشش کامل"
    elif "Combined" in source_label:
        _q = "🔵 ترکیبی · پوشش ۱۰۰٪"
    elif "Relative" in source_label:
        _q = "🟣 نسبی · median=0"
    else:
        _q = ""
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(0,217,255,0.15), rgba(0,255,136,0.1));
                border-left: 4px solid #00d9ff;
                padding: 0.6rem 1rem; margin-bottom: 1rem;
                border-radius: 8px; color: #7ee7ff;
                font-family: 'Courier New'; font-size: 0.85rem;">
        <b>📡 منبع فعال:</b> {_s}<br>
        <span style="font-size: 0.75rem; color: #99ffcc;">{_q}</span>
    </div>
    """, unsafe_allow_html=True)

    # ⭐ هدر قرمز سای‌فای
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2e0a0a 0%, #4a1a1a 50%, #2e0a0a 100%);
                padding: 2rem; border-radius: 15px; text-align: center;
                border: 2px solid #FF1432;
                box-shadow: 0 0 60px rgba(255, 20, 50, 0.5);
                margin-bottom: 1.5rem;">
        <h1 style="color: #FF1432; font-family: 'Courier New'; margin: 0;
                   font-size: 2rem; letter-spacing: 4px;
                   text-shadow: 0 0 25px #FF1432;">
            ◉ RISK ASSESSMENT MATRIX ◉
        </h1>
        <p style="color: #ff9999; font-family: 'Courier New';
                  font-size: 1rem; margin-top: 0.7rem; letter-spacing: 3px;">
            ▸ 5-LEVEL SUBSIDENCE CLASSIFICATION ◂
        </p>
    </div>
    """, unsafe_allow_html=True)

    if joined_df is not None:
        # ⭐ طبقه‌بندی ۵ سطحی
        def classify_risk_level(v):
            if v > -5:
                return 0, "⚪ No Risk", "#95a5a6"
            elif v > -15:
                return 1, "🟢 Low", "#00FF88"
            elif v > -30:
                return 2, "🟡 Moderate", "#FFE600"
            elif v > -50:
                return 3, "🟠 High", "#FF7800"
            else:
                return 4, "🔴 Very High", "#FF1432"

        # ⭐ copy برای جلوگیری از mutate شدن joined_df اصلی
        _joined = joined_df.copy()
        _joined["risk_level"] = _joined["velocity"].apply(lambda v: classify_risk_level(v)[0])
        _joined["risk_label"] = _joined["velocity"].apply(lambda v: classify_risk_level(v)[1])
        _joined["risk_color"] = _joined["velocity"].apply(lambda v: classify_risk_level(v)[2])

        # ⭐ راهنمای سطح خطر
        st.markdown("### 🎨 Risk Classification Legend")

        legend_cols = st.columns(5)
        _total_px = len(joined_df)
        _cnt_0 = int((_joined["risk_level"] == 0).sum())
        _cnt_1 = int((_joined["risk_level"] == 1).sum())
        _cnt_2 = int((_joined["risk_level"] == 2).sum())
        _cnt_3 = int((_joined["risk_level"] == 3).sum())
        _cnt_4 = int((_joined["risk_level"] == 4).sum())
        legend_data = [
            ("⚪", "No Risk", "> -5 mm/yr", "#95a5a6", f"{_cnt_0:,}"),
            ("🟢", "Low", "-5 to -15", "#00FF88", f"{_cnt_1:,}"),
            ("🟡", "Moderate", "-15 to -30", "#FFE600", f"{_cnt_2:,}"),
            ("🟠", "High", "-30 to -50", "#FF7800", f"{_cnt_3:,}"),
            ("🔴", "Very High", "< -50", "#FF1432", f"{_cnt_4:,}")
        ]

        for col, (icon, name, range_str, color, _) in zip(legend_cols, legend_data):
            with col:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #0a0e27 0%, #1a1a4e 100%);
                            padding: 0.8rem; border-radius: 10px;
                            border: 2px solid {color};
                            text-align: center;">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div style="color: {color}; font-family: 'Courier New';
                                font-weight: 700; font-size: 0.9rem;
                                text-shadow: 0 0 8px {color};">
                        {name}
                    </div>
                    <div style="color: #7ee7ff; font-family: 'Courier New';
                                font-size: 0.7rem; margin-top: 0.3rem;">
                        {range_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ⭐ فیلتر سطح خطر
        st.markdown("### 🎯 Filter by Risk Level")

        risk_filter = st.multiselect(
            "Select risk levels to display:",
            options=["⚪ No Risk", "🟢 Low", "🟡 Moderate", "🟠 High", "🔴 Very High"],
            default=["🟢 Low", "🟡 Moderate", "🟠 High", "🔴 Very High"]
            # ⭐ No Risk پیش‌فرض خاموش (چون تعدادش زیاده)
        )

        # فیلتر داده‌ها
        filtered_df = _joined[_joined["risk_label"].isin(risk_filter)].copy()

        # ⭐ دو ستون: نقشه + شمارنده
        col_map, col_count = st.columns([2, 1])

        with col_map:
            st.markdown("### 🗺️ Risk Map")

            # نمونه‌برداری
            sample_size = min(10000, len(filtered_df))
            if sample_size > 0:
                sample_df = filtered_df.sample(n=sample_size, random_state=42)

                fig = go.Figure()

                # لایه به لایه بر اساس سطح خطر
                risk_order = [
                    ("⚪ No Risk", "#95a5a6", 0),
                    ("🟢 Low", "#00FF88", 1),
                    ("🟡 Moderate", "#FFE600", 2),
                    ("🟠 High", "#FF7800", 3),
                    ("🔴 Very High", "#FF1432", 4),
                ]

                for label, color, order in risk_order:
                    level_df = sample_df[sample_df["risk_label"] == label]
                    if len(level_df) == 0:
                        continue

                    fig.add_trace(go.Scattermapbox(
                        lon=level_df["lon"],
                        lat=level_df["lat"],
                        mode="markers",
                        name=label,
                        marker=dict(
                            size=6,
                            color=color,
                            opacity=0.8
                        ),
                        hovertemplate=f"<b>{label}</b><br>%{{customdata:.1f}} mm/yr<extra></extra>",
                        customdata=level_df["velocity"]
                    ))

                # مرزها
                if districts_gdf is not None:
                    for _, r in districts_gdf.iterrows():
                        g = r.geometry
                        ps = [g] if g.geom_type == "Polygon" else list(g.geoms) if g.geom_type == "MultiPolygon" else []
                        for p in ps:
                            coords = list(p.exterior.coords)
                            fig.add_trace(go.Scattermapbox(
                                lon=[c[0] for c in coords],
                                lat=[c[1] for c in coords],
                                mode="lines",
                                line=dict(color="#00d9ff", width=2),
                                name="District",
                                showlegend=False,
                                hoverinfo="skip"
                            ))

                # مترو
                if metro_gdf is not None:
                    for _, r in metro_gdf.iterrows():
                        g = r.geometry
                        ls = [g] if g.geom_type == "LineString" else list(g.geoms) if g.geom_type == "MultiLineString" else []
                        for line in ls:
                            coords = list(line.coords)
                            fig.add_trace(go.Scattermapbox(
                                lon=[c[0] for c in coords],
                                lat=[c[1] for c in coords],
                                mode="lines",
                                line=dict(color="#00FFC8", width=1.5),
                                name="Metro",
                                showlegend=False,
                                hoverinfo="skip"
                            ))

                fig.update_layout(
                    mapbox=dict(
                        style="carto-darkmatter",
                        accesstoken=MAPBOX_TOKEN if MAPBOX_TOKEN else None,
                        center=dict(lat=35.7, lon=51.4),
                        zoom=10.5
                    ),
                    height=600,
                    margin=dict(r=0, t=20, l=0, b=0),
                    paper_bgcolor="#0a0e27",
                    font=dict(color="#ff9999", family="Courier New"),
                    legend=dict(
                        x=0.01, y=0.99,
                        bgcolor="rgba(26, 10, 10, 0.9)",
                        bordercolor="#FF1432",
                        borderwidth=2,
                        font=dict(color="#ff9999", size=11)
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No points to display. Select at least one risk level.")

        with col_count:
            st.markdown("### 📊 Live Counts")

            total = len(joined_df)

            for label, color, order in risk_order:
                count = int((_joined["risk_label"] == label).sum())
                pct = count / total * 100 if total > 0 else 0

                st.markdown(f"""
                <div style="background: rgba(26, 10, 10, 0.6);
                            border-left: 4px solid {color};
                            padding: 0.7rem 1rem; margin: 0.5rem 0;
                            border-radius: 5px;">
                    <div style="color: {color}; font-family: 'Courier New';
                                font-weight: 700; font-size: 0.85rem;">
                        {label}
                    </div>
                    <div style="color: #7ee7ff; font-family: 'Courier New';
                                font-size: 1.1rem; font-weight: 700;">
                        {count:,} <span style="font-size: 0.7rem;">({pct:.1f}%)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # جمع کل خطر
            at_risk = int((_joined["risk_level"] >= 1).sum())
            at_risk_pct = at_risk / total * 100 if total > 0 else 0

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2e0a0a 0%, #4a1a1a 100%);
                        border: 2px solid #FF1432;
                        padding: 1rem; margin-top: 1rem;
                        border-radius: 10px; text-align: center;">
                <div style="color: #ff9999; font-family: 'Courier New';
                            font-size: 0.8rem;">
                    ⚠️ TOTAL AT RISK
                </div>
                <div style="color: #FF1432; font-family: 'Courier New';
                            font-size: 1.5rem; font-weight: 700;
                            text-shadow: 0 0 15px #FF1432;">
                    {at_risk:,}
                </div>
                <div style="color: #ff9999; font-family: 'Courier New';
                            font-size: 0.8rem;">
                    {at_risk_pct:.1f}% of Tehran
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ⭐ جدول رتبه‌بندی مناطق
        st.markdown("---")
        st.markdown("### 📋 District Risk Ranking")

        if df_pop is not None:
            # ادغام با آمار مناطق
            district_stats = joined_df.groupby("district_name").agg(
                mean_velocity=("velocity", "mean"),
                total_pixels=("velocity", "count")
            ).reset_index()

            # شمارش نقاط هر سطح خطر در هر منطقه
            for label, color, order in risk_order:
                counts = _joined[_joined["risk_label"] == label].groupby("district_name").size().reset_index(name=f"count_{order}")
                district_stats = district_stats.merge(counts, on="district_name", how="left")

            # پر کردن NaN
            for order in range(5):
                col_name = f"count_{order}"
                if col_name not in district_stats.columns:
                    district_stats[col_name] = 0
                district_stats[col_name] = district_stats[col_name].fillna(0).astype(int)

            # ادغام با df_pop برای risk_index
            district_stats = district_stats.merge(
                df_pop[["district_name", "population", "affected_population", "risk_index", "risk_category"]],
                on="district_name",
                how="left"
            )

            # ⭐ fillna برای risk_index
            district_stats["risk_index"] = district_stats["risk_index"].fillna(0)

            # مرتب‌سازی
            district_stats = district_stats.sort_values("risk_index", ascending=False)

            # نمایش جدول
            display_df = district_stats[["district_name", "count_4", "count_3", "count_2", "count_1", "count_0", "risk_index", "risk_category"]].copy()
            display_df.columns = ["District", "🔴 V.High", "🟠 High", "🟡 Mod", "🟢 Low", "⚪ None", "Risk Score", "Category"]

            st.dataframe(
                display_df.style
                    .background_gradient(subset=["🔴 V.High"], cmap="Reds")
                    .background_gradient(subset=["🟠 High"], cmap="Oranges")
                    .background_gradient(subset=["🟡 Mod"], cmap="YlOrBr")
                    .background_gradient(subset=["🟢 Low"], cmap="Greens_r")
                    .background_gradient(subset=["Risk Score"], cmap="Reds")
                    .format({
                        "🔴 V.High": "{:,}",
                        "🟠 High": "{:,}",
                        "🟡 Mod": "{:,}",
                        "🟢 Low": "{:,}",
                        "⚪ None": "{:,}",
                        "Risk Score": "{:.1f}"
                    }),
                use_container_width=True,
                hide_index=True,
                height=500
            )

        # ⭐ هشدار متنی
        st.markdown("---")

        very_high_count = int((_joined["risk_level"] == 4).sum())
        high_count = int((_joined["risk_level"] == 3).sum())
        moderate_count = int((_joined["risk_level"] == 2).sum())

        st.markdown(f"""
        <div style="background: linear-gradient(90deg, rgba(255, 20, 50, 0.15), rgba(255, 120, 0, 0.15));
                    border: 2px solid #FF1432; border-radius: 10px;
                    padding: 1.5rem; margin: 1rem 0;">
            <h3 style="color: #FF1432; margin: 0; font-family: 'Courier New';">
                ⚠️ RISK SUMMARY
            </h3>
            <p style="color: #ff9999; font-family: 'Courier New'; margin-top: 0.7rem;">
                <b style="color: #FF1432; font-size: 1.2rem;">{very_high_count:,}</b> pixels at <b>VERY HIGH</b> risk<br>
                <b style="color: #FF7800; font-size: 1.2rem;">{high_count:,}</b> pixels at <b>HIGH</b> risk<br>
                <b style="color: #FFE600; font-size: 1.2rem;">{moderate_count:,}</b> pixels at <b>MODERATE</b> risk
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ⭐ دانلود
        if df_pop is not None:
            csv = district_stats.to_csv(index=False)
            st.download_button(
                label="📥 Download Risk Ranking (CSV)",
                data=csv,
                file_name="tehran_risk_ranking.csv",
                mime="text/csv"
            )
elif page == "🚇 INFRASTRUCTURE":
    # ⭐ بنر منبع فعال (همه صفحات)
    _s = f"📡 {source_label}"
    if vel_active is not None:
        _n = int((~np.isnan(vel_active)).sum())
        _p = _n / vel_active.size * 100
        _s += f" · ✅ {_n:,} پیکسل ({_p:.1f}%)"
    
    if "Ascending" in source_label:
        _q = "🟢 کیفیت بالا · پوشش کم"
    elif "Descending" in source_label:
        _q = "🟡 کیفیت متوسط · پوشش کامل"
    elif "Combined" in source_label:
        _q = "🔵 ترکیبی · پوشش ۱۰۰٪"
    elif "Relative" in source_label:
        _q = "🟣 نسبی · median=0"
    else:
        _q = ""
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(0,217,255,0.15), rgba(0,255,136,0.1));
                border-left: 4px solid #00d9ff;
                padding: 0.6rem 1rem; margin-bottom: 1rem;
                border-radius: 8px; color: #7ee7ff;
                font-family: 'Courier New'; font-size: 0.85rem;">
        <b>📡 منبع فعال:</b> {_s}<br>
        <span style="font-size: 0.75rem; color: #99ffcc;">{_q}</span>
    </div>
    """, unsafe_allow_html=True)

    # ⭐ هدر آبی سای‌فای
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0a1e2e 0%, #1a3a4e 50%, #0a1e2e 100%);
                padding: 2rem; border-radius: 15px; text-align: center;
                border: 2px solid #00d9ff;
                box-shadow: 0 0 60px rgba(0, 217, 255, 0.5);
                margin-bottom: 1.5rem;">
        <h1 style="color: #00d9ff; font-family: 'Courier New'; margin: 0;
                   font-size: 2rem; letter-spacing: 4px;
                   text-shadow: 0 0 25px #00d9ff;">
            ◉ INFRASTRUCTURE RISK ◉
        </h1>
        <p style="color: #7ee7ff; font-family: 'Courier New';
                  font-size: 1rem; margin-top: 0.7rem; letter-spacing: 3px;">
            ▸ METRO · AIRPORTS · POPULATION ◂
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ⭐ محاسبه آمار کلی (NaN-safe)
    metro_critical = 0
    metro_warning = 0
    airports_at_risk = 0
    pop_at_risk = 0

    if df_metro is not None:
        metro_critical = int((df_metro["risk"] == "🔴 Critical").sum())
        metro_warning = int((df_metro["risk"] == "🟠 Warning").sum())

    if df_airports is not None:
        _air_vel = df_airports["mean_velocity"].fillna(0)
        airports_at_risk = int((_air_vel < -20).sum())

    if df_pop is not None:
        _pop_aff = df_pop["affected_population"].fillna(0)
        pop_at_risk = int(_pop_aff.sum())

    # ⭐ ۳ کارت KPI بالا
    st.markdown("### 📊 Critical Infrastructure Overview")

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown(f"""
        <div class="metric-card risk-critical">
            <h3>🚇 METRO STATIONS</h3>
            <div class="value">{metro_critical + metro_warning}</div>
            <div class="sub">at risk ({metro_critical} critical)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-card risk-warning">
            <h3>✈️ AIRPORTS</h3>
            <div class="value">{airports_at_risk}</div>
            <div class="sub">at risk</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 POPULATION</h3>
            <div class="value">{pop_at_risk/1e6:.2f}M</div>
            <div class="sub">at risk</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ⭐ ۳ تب
    tab1, tab2, tab3 = st.tabs(["🚇 Metro", "✈️ Airports", "👥 Population"])

    # ============================================================
    # TAB 1: METRO
    # ============================================================
    with tab1:
        if df_metro is not None:
            # آمار
            m_crit = int((df_metro["risk"] == "🔴 Critical").sum())
            m_warn = int((df_metro["risk"] == "🟠 Warning").sum())
            m_mod = int((df_metro["risk"] == "🟡 Moderate").sum())
            m_stab = int((df_metro["risk"] == "🟢 Stable").sum())

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(f"""
                <div class="metric-card risk-critical">
                    <h3>🔴 CRITICAL</h3>
                    <div class="value">{m_crit}</div>
                    <div class="sub">stations</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card risk-warning">
                    <h3>🟠 WARNING</h3>
                    <div class="value">{m_warn}</div>
                    <div class="sub">stations</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🟡 MODERATE</h3>
                    <div class="value">{m_mod}</div>
                    <div class="sub">stations</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card risk-stable">
                    <h3>🟢 STABLE</h3>
                    <div class="value">{m_stab}</div>
                    <div class="sub">stations</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # نقشه مترو
            col_map, col_list = st.columns([2, 1])

            with col_map:
                st.markdown("### 🗺️ Metro Station Risk Map")

                fig = go.Figure()

                # خطوط مترو
                if metro_gdf is not None:
                    for _, r in metro_gdf.iterrows():
                        g = r.geometry
                        ls = [g] if g.geom_type == "LineString" else list(g.geoms) if g.geom_type == "MultiLineString" else []
                        for line in ls:
                            coords = list(line.coords)
                            fig.add_trace(go.Scattermapbox(
                                lon=[c[0] for c in coords],
                                lat=[c[1] for c in coords],
                                mode="lines",
                                line=dict(color="#00FFC8", width=2),
                                name="Metro Line",
                                showlegend=False,
                                hoverinfo="skip"
                            ))

                # ایستگاه‌ها بر اساس خطر
                risk_colors = {
                    "🔴 Critical": "#FF1432",
                    "🟠 Warning": "#FF7800",
                    "🟡 Moderate": "#FFE600",
                    "🟢 Stable": "#00FF88"
                }

                for risk, color in risk_colors.items():
                    subset = df_metro[df_metro["risk"] == risk]
                    if len(subset) == 0:
                        continue

                    fig.add_trace(go.Scattermapbox(
                        lon=subset["lon"],
                        lat=subset["lat"],
                        mode="markers",
                        name=risk,
                        marker=dict(
                            size=12,
                            color=color,
                            opacity=0.9
                        ),
                        text=subset["name"],
                        hovertemplate=f"<b>%{{text}}</b><br>{risk}<br>%{{customdata:.1f}} mm/yr<extra></extra>",
                        customdata=subset["velocity_mm_yr"]
                    ))

                # مرزها
                if districts_gdf is not None:
                    for _, r in districts_gdf.iterrows():
                        g = r.geometry
                        ps = [g] if g.geom_type == "Polygon" else list(g.geoms) if g.geom_type == "MultiPolygon" else []
                        for p in ps:
                            coords = list(p.exterior.coords)
                            fig.add_trace(go.Scattermapbox(
                                lon=[c[0] for c in coords],
                                lat=[c[1] for c in coords],
                                mode="lines",
                                line=dict(color="#00d9ff", width=1.5),
                                showlegend=False,
                                hoverinfo="skip"
                            ))

                fig.update_layout(
                    mapbox=dict(
                        style="carto-darkmatter",
                        accesstoken=MAPBOX_TOKEN if MAPBOX_TOKEN else None,
                        center=dict(lat=35.7, lon=51.4),
                        zoom=11
                    ),
                    height=600,
                    margin=dict(r=0, t=20, l=0, b=0),
                    paper_bgcolor="#0a0e27",
                    font=dict(color="#7ee7ff", family="Courier New"),
                    legend=dict(
                        x=0.01, y=0.99,
                        bgcolor="rgba(10, 30, 46, 0.9)",
                        bordercolor="#00d9ff",
                        borderwidth=2
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

            with col_list:
                st.markdown("### 🔴 Top 15 Critical")

                top15 = df_metro.nsmallest(15, "velocity_mm_yr")[["name", "district", "velocity_mm_yr", "risk"]].copy()
                top15.columns = ["Station", "District", "Velocity", "Risk"]

                st.dataframe(
                    top15.style
                        .background_gradient(subset=["Velocity"], cmap="Reds_r")
                        .format({"Velocity": "{:.1f}"}),
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )

            # نمودار میله‌ای
            st.markdown("---")
            st.markdown("### 📊 Stations by Risk Level")

            risk_counts = df_metro["risk"].value_counts()
            order = ["🔴 Critical", "🟠 Warning", "🟡 Moderate", "🟢 Stable"]
            counts = [risk_counts.get(r, 0) for r in order]
            colors_bar = [risk_colors.get(r, "#95a5a6") for r in order]

            fig2 = go.Figure(data=[go.Bar(
                x=order,
                y=counts,
                marker=dict(color=colors_bar, line=dict(color="#00d9ff", width=2)),
                text=counts,
                textposition="outside",
                textfont=dict(color="#7ee7ff", size=12)
            )])

            fig2.update_layout(
                height=300,
                margin=dict(r=30, t=20, l=20, b=20),
                paper_bgcolor="#0a0e27",
                plot_bgcolor="#0a0e27",
                font=dict(color="#7ee7ff", family="Courier New"),
                yaxis=dict(title=dict(text="Number of Stations", font=dict(color="#7ee7ff")), gridcolor="rgba(0, 217, 255, 0.15)"),
                xaxis=dict(gridcolor="rgba(0, 217, 255, 0.1)"),
                showlegend=False
            )

            st.plotly_chart(fig2, use_container_width=True)

    # ============================================================
    # TAB 2: AIRPORTS
    # ============================================================
    with tab2:
        if df_airports is not None:
            # نقشه فرودگاه‌ها
            st.markdown("### 🗺️ Airport Risk Map")

            fig_air = go.Figure()

            # فرودگاه‌ها
            for _, r in df_airports.iterrows():
                v = r["mean_velocity"]
                if v < -30:
                    color = "#FF1432"
                elif v < -15:
                    color = "#FF7800"
                elif v < 0:
                    color = "#FFE600"
                else:
                    color = "#00FF88"

                analysis = "📏 Estimated" if r["analysis_type"] == "estimated" else "✅ Direct"

                fig_air.add_trace(go.Scattermapbox(
                    lon=[r["lon"]],
                    lat=[r["lat"]],
                    mode="markers+text",
                    marker=dict(size=25, color=color, symbol="airport"),
                    text=[r["name_fa"]],
                    textposition="top center",
                    textfont=dict(color="#7ee7ff", size=10, family="Courier New"),
                    hovertemplate=f"<b>{r['name_fa']}</b><br>" +
                                   f"Velocity: {v:.1f} mm/yr<br>" +
                                   f"Type: {r['type']}<br>" +
                                   f"Method: {analysis}<extra></extra>",
                    showlegend=False
                ))

            # مرزها
            if districts_gdf is not None:
                for _, r in districts_gdf.iterrows():
                    g = r.geometry
                    ps = [g] if g.geom_type == "Polygon" else list(g.geoms) if g.geom_type == "MultiPolygon" else []
                    for p in ps:
                        coords = list(p.exterior.coords)
                        fig_air.add_trace(go.Scattermapbox(
                            lon=[c[0] for c in coords],
                            lat=[c[1] for c in coords],
                            mode="lines",
                            line=dict(color="#00d9ff", width=1.5),
                            showlegend=False,
                            hoverinfo="skip"
                        ))

            fig_air.update_layout(
                mapbox=dict(
                    style="carto-darkmatter",
                    accesstoken=MAPBOX_TOKEN if MAPBOX_TOKEN else None,
                    center=dict(lat=35.6, lon=51.35),
                    zoom=10
                ),
                height=500,
                margin=dict(r=0, t=20, l=0, b=0),
                paper_bgcolor="#0a0e27",
                font=dict(color="#7ee7ff", family="Courier New")
            )

            st.plotly_chart(fig_air, use_container_width=True)

            # جدول فرودگاه‌ها
            st.markdown("---")
            st.markdown("### 📋 Airport Analysis")

            display_air = df_airports[["name_fa", "type", "mean_velocity", "min_velocity", "risk", "analysis_type"]].copy()
            display_air.columns = ["Airport", "Type", "Mean (mm/yr)", "Max Subsidence", "Risk", "Method"]

            st.dataframe(
                display_air.style
                    .background_gradient(subset=["Mean (mm/yr)"], cmap="Reds_r")
                    .background_gradient(subset=["Max Subsidence"], cmap="Reds_r")
                    .format({"Mean (mm/yr)": "{:.1f}", "Max Subsidence": "{:.1f}"}),
                use_container_width=True,
                hide_index=True
            )

            # کارت‌های فرودگاه
            st.markdown("---")
            st.markdown("### ✈️ Airport Cards")

            n_air = min(len(df_airports), 5)   # حداکثر ۵ کارت
            cols_air = st.columns(n_air)

            for col, (_, r) in zip(cols_air, df_airports.iterrows()):
                with col:
                    v = r["mean_velocity"]
                    if v < -20:
                        card_class = "risk-critical"
                    elif v < 0:
                        card_class = "risk-warning"
                    else:
                        card_class = "risk-stable"

                    st.markdown(f"""
                    <div class="metric-card {card_class}">
                        <h3>✈️ {r["name_fa"][:15]}</h3>
                        <div class="value">{v:.1f}</div>
                        <div class="sub">{r["type"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ============================================================
    # TAB 3: POPULATION
    # ============================================================
    with tab3:
        if df_pop is not None:
            # آمار کلی (NaN-safe)
            total_pop = df_pop["population"].fillna(0).sum()
            total_affected = df_pop["affected_population"].fillna(0).sum()
            affected_pct = (total_affected / total_pop * 100) if total_pop > 0 else 0

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>◉ TOTAL POPULATION</h3>
                    <div class="value">{total_pop/1e6:.2f}M</div>
                    <div class="sub">in Tehran</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card risk-critical">
                    <h3>◉ AFFECTED</h3>
                    <div class="value">{total_affected/1e6:.2f}M</div>
                    <div class="sub">people at risk</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="metric-card risk-warning">
                    <h3>◉ PERCENTAGE</h3>
                    <div class="value">{affected_pct:.1f}%</div>
                    <div class="sub">of population</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # نقشه حرارتی جمعیت
            col_map, col_scatter = st.columns([1, 1])

            with col_map:
                st.markdown("### 🗺️ Population Risk Map")

                if districts_gdf is not None:
                    merged = districts_gdf.merge(
                        df_pop[["district_name", "population", "affected_population", "risk_index", "risk_category"]],
                        on="district_name",
                        how="left"
                    )
                    merged["affected_population"] = merged["affected_population"].fillna(0)

                    merged_geojson = json.loads(merged.to_json())

                    fig_pop = px.choropleth_mapbox(
                        merged,
                        geojson=merged_geojson,
                        locations=merged.index,
                        color="affected_population",
                        color_continuous_scale=[
                            [0.0, "#0a0e27"],
                            [0.2, "#00FF88"],
                            [0.5, "#FFE600"],
                            [0.8, "#FF7800"],
                            [1.0, "#FF1432"]
                        ],
                        mapbox_style="carto-darkmatter",
                        mapbox_token=MAPBOX_TOKEN if MAPBOX_TOKEN else None,
                        zoom=10,
                        center={"lat": 35.7, "lon": 51.4},
                        opacity=0.8,
                        hover_name="district_name",
                        hover_data={"affected_population": ":,.0f", "district_name": False}
                    )

                    fig_pop.update_traces(marker_line_width=2, marker_line_color="#00d9ff")

                    fig_pop.update_layout(
                        height=500,
                        margin=dict(r=0, t=10, l=0, b=0),
                        paper_bgcolor="#0a0e27",
                        font=dict(color="#7ee7ff", family="Courier New"),
                        coloraxis_colorbar=dict(
                            title="Affected<br>Population",
                            tickfont=dict(color="#7ee7ff"),
                            thickness=15
                        )
                    )

                    st.plotly_chart(fig_pop, use_container_width=True)

            with col_scatter:
                st.markdown("### 📊 Velocity vs Population")

                scatter_df = df_pop.dropna(subset=["mean_velocity", "population"])

                fig_scatter = go.Figure()

                colors_scatter = []
                for v in scatter_df["mean_velocity"]:
                    if v < -30:
                        colors_scatter.append("#FF1432")
                    elif v < -15:
                        colors_scatter.append("#FF7800")
                    elif v < 0:
                        colors_scatter.append("#FFE600")
                    else:
                        colors_scatter.append("#00FF88")

                fig_scatter.add_trace(go.Scatter(
                    x=scatter_df["mean_velocity"],
                    y=scatter_df["population"] / 1000,
                    mode="markers+text",
                    text=scatter_df["district_name"],
                    textposition="top center",
                    textfont=dict(size=8, color="#7ee7ff"),
                    marker=dict(
                        size=np.sqrt(scatter_df["affected_population"]) / 30,
                        color=colors_scatter,
                        line=dict(color="#00d9ff", width=1),
                        opacity=0.8
                    ),
                    hovertemplate="<b>%{text}</b><br>Velocity: %{x:.1f} mm/yr<br>Population: %{y:.0f}K<extra></extra>"
                ))

                fig_scatter.update_layout(
                    height=500,
                    margin=dict(r=20, t=20, l=20, b=20),
                    paper_bgcolor="#0a0e27",
                    plot_bgcolor="#0a0e27",
                    font=dict(color="#7ee7ff", family="Courier New"),
                    xaxis=dict(title=dict(text="Mean Velocity (mm/yr)", font=dict(color="#7ee7ff")), gridcolor="rgba(0, 217, 255, 0.15)"),
                    yaxis=dict(title=dict(text="Population (×1000)", font=dict(color="#7ee7ff")), gridcolor="rgba(0, 217, 255, 0.15)"),
                    showlegend=False
                )

                st.plotly_chart(fig_scatter, use_container_width=True)

            # Top 10 جدول
            st.markdown("---")
            st.markdown("### 👥 Top 10 Districts by Affected Population")

            top10_pop = df_pop.nlargest(10, "affected_population")[["rank", "district_name", "mean_velocity", "population", "affected_population", "risk_index", "risk_category"]].copy()
            top10_pop.columns = ["Rank", "District", "Velocity (mm/yr)", "Population", "Affected", "Risk Index", "Category"]

            st.dataframe(
                top10_pop.style
                    .background_gradient(subset=["Velocity (mm/yr)"], cmap="Reds_r")
                    .background_gradient(subset=["Affected"], cmap="Reds")
                    .background_gradient(subset=["Risk Index"], cmap="Reds")
                    .format({
                        "Velocity (mm/yr)": "{:.1f}",
                        "Population": "{:,.0f}",
                        "Affected": "{:,.0f}",
                        "Risk Index": "{:.1f}"
                    }),
                use_container_width=True,
                hide_index=True
            )

    # ⭐ هشدار پایین صفحه
    st.markdown("---")

    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(0, 217, 255, 0.15), rgba(255, 20, 50, 0.15));
                border: 2px solid #00d9ff; border-radius: 10px;
                padding: 1.5rem; margin: 1rem 0;">
        <h3 style="color: #00d9ff; margin: 0; font-family: 'Courier New';">
            ⚠️ CRITICAL INFRASTRUCTURE SUMMARY
        </h3>
        <p style="color: #7ee7ff; font-family: 'Courier New'; margin-top: 0.7rem;">
            <b style="color: #FF1432;">{metro_critical + metro_warning}</b> metro stations at risk<br>
            <b style="color: #FF7800;">{airports_at_risk}</b> airports at risk<br>
            <b style="color: #FFE600;">{pop_at_risk/1e6:.2f}M</b> people living in danger zones
        </p>
    </div>
    """, unsafe_allow_html=True)
elif page == "🔥 HOTSPOTS":
    # ⭐ بنر منبع فعال (همه صفحات)
    _s = f"📡 {source_label}"
    if vel_active is not None:
        _n = int((~np.isnan(vel_active)).sum())
        _p = _n / vel_active.size * 100
        _s += f" · ✅ {_n:,} پیکسل ({_p:.1f}%)"
    
    if "Ascending" in source_label:
        _q = "🟢 کیفیت بالا · پوشش کم"
    elif "Descending" in source_label:
        _q = "🟡 کیفیت متوسط · پوشش کامل"
    elif "Combined" in source_label:
        _q = "🔵 ترکیبی · پوشش ۱۰۰٪"
    elif "Relative" in source_label:
        _q = "🟣 نسبی · median=0"
    else:
        _q = ""
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(0,217,255,0.15), rgba(0,255,136,0.1));
                border-left: 4px solid #00d9ff;
                padding: 0.6rem 1rem; margin-bottom: 1rem;
                border-radius: 8px; color: #7ee7ff;
                font-family: 'Courier New'; font-size: 0.85rem;">
        <b>📡 منبع فعال:</b> {_s}<br>
        <span style="font-size: 0.75rem; color: #99ffcc;">{_q}</span>
    </div>
    """, unsafe_allow_html=True)

    # ⭐ هدر نارنجی سای‌فای
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2e1a0a 0%, #4e2a1a 50%, #2e1a0a 100%);
                padding: 2rem; border-radius: 15px; text-align: center;
                border: 2px solid #FF7800;
                box-shadow: 0 0 60px rgba(255, 120, 0, 0.5);
                margin-bottom: 1.5rem;">
        <h1 style="color: #FF7800; font-family: 'Courier New'; margin: 0;
                   font-size: 2rem; letter-spacing: 4px;
                   text-shadow: 0 0 25px #FF7800;">
            ◉ SUBSIDENCE HOTSPOTS ◉
        </h1>
        <p style="color: #ffcc99; font-family: 'Courier New';
                  font-size: 1rem; margin-top: 0.7rem; letter-spacing: 3px;">
            ▸ DBSCAN CLUSTER ANALYSIS · 2024 DATA ◂
        </p>
    </div>
    """, unsafe_allow_html=True)

    if df_hot is not None and len(df_hot) > 0:
        # ⭐ مرتب‌سازی خوشه‌ها بر اساس تعداد نقاط
        df_hot_sorted = df_hot.sort_values("n_points", ascending=False).reset_index(drop=True)
        df_hot_sorted["rank"] = df_hot_sorted.index + 1

        # ⭐ اضافه کردن منطقه به هر خوشه
        if districts_gdf is not None:
            import geopandas as _gpd
            from shapely.geometry import Point

            districts_utm = districts_gdf.to_crs("EPSG:32639")

            cluster_districts = []
            for _, cluster in df_hot_sorted.iterrows():
                try:
                    pt = Point(cluster["center_lon"], cluster["center_lat"])
                    pt_utm = _gpd.GeoSeries([pt], crs="EPSG:4326").to_crs("EPSG:32639").iloc[0]

                    found_district = "Unknown"
                    for _, d in districts_utm.iterrows():
                        if d.geometry.contains(pt_utm):
                            found_district = d["district_name"]
                            break

                    cluster_districts.append(found_district)
                except:
                    cluster_districts.append("Unknown")

            df_hot_sorted["district"] = cluster_districts
        else:
            df_hot_sorted["district"] = "Unknown"

        # ⭐ چگالی حذف شد (محاسبه اشتباه بود)

        # ⭐ ۵ کارت KPI
        n_clusters = len(df_hot_sorted)
        total_critical = int(df_hot_sorted["n_points"].sum())
        mean_velocity = df_hot_sorted["mean_velocity"].mean()
        largest_cluster = df_hot_sorted.iloc[0]["n_points"]
        deepest = df_hot_sorted["min_velocity"].min()

        st.markdown("### 🎯 Cluster Overview")

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>◉ CLUSTERS</h3>
                <div class="value">{n_clusters}</div>
                <div class="sub">detected</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card risk-critical">
                <h3>◉ CRITICAL</h3>
                <div class="value">{total_critical:,}</div>
                <div class="sub">points</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card risk-warning">
                <h3>◉ MEAN</h3>
                <div class="value">{mean_velocity:.1f}</div>
                <div class="sub">mm/yr</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>◉ LARGEST</h3>
                <div class="value">{int(largest_cluster):,}</div>
                <div class="sub">points</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:
            st.markdown(f"""
            <div class="metric-card risk-critical">
                <h3>◉ DEEPEST</h3>
                <div class="value">{deepest:.0f}</div>
                <div class="sub">mm/yr</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ⭐ نقشه اصلی
        st.markdown("### 🗺️ Hotspot Map with Cluster Analysis")

        fig = go.Figure()

        # ⭐ لایه ۱: نقاط بحرانی
        if joined_df is not None:
            critical_pts = joined_df[joined_df["velocity"] < -30].copy()
            sample_size = min(5000, len(critical_pts))
            if sample_size > 0:
                sample_pts = critical_pts.sample(n=sample_size, random_state=42)

                fig.add_trace(go.Scattermapbox(
                    lon=sample_pts["lon"],
                    lat=sample_pts["lat"],
                    mode="markers",
                    name="Critical Points",
                    marker=dict(
                        size=4,
                        color=sample_pts["velocity"],
                        colorscale=[
                            [0.0, "#8B0000"],
                            [0.3, "#FF1432"],
                            [0.7, "#FF7800"],
                            [1.0, "#FFE600"]
                        ],
                        cmin=-150,
                        cmax=-30,
                        colorbar=dict(
                            title=dict(text="Velocity<br>(mm/yr)", font=dict(color="#ffcc99")),
                            tickfont=dict(color="#ffcc99"),
                            thickness=12,
                            len=0.7
                        ),
                        opacity=0.5
                    ),
                    hovertemplate="<b>%{marker.color:.1f} mm/yr</b><extra></extra>",
                    showlegend=False
                ))

        # ⭐ لایه ۲: محیط محدب (Convex Hull) هر خوشه
        # اینجا از دایره‌ها به جای hull استفاده می‌کنیم (ساده‌تر)
        cluster_colors = ["#FF1432", "#FF7800", "#FFE600", "#00FF88", "#00d9ff"]

        for i, (_, cluster) in enumerate(df_hot_sorted.iterrows()):
            color = cluster_colors[i % len(cluster_colors)]

            # اندازه دایره بر اساس تعداد نقاط
            circle_size = max(20, min(80, cluster["n_points"] ** 0.5 / 1.5))

            fig.add_trace(go.Scattermapbox(
                lon=[cluster["center_lon"]],
                lat=[cluster["center_lat"]],
                mode="markers+text",
                name=f"C{int(cluster['cluster_id']) + 1}",
                marker=dict(
                    size=circle_size,
                    color=color,
                    opacity=0.4
                ),
                text=[f"C{int(cluster['cluster_id']) + 1}"],
                textposition="top center",
                textfont=dict(color="#ffcc99", size=12, family="Courier New"),
                hovertemplate=f"<b>C{int(cluster['cluster_id']) + 1}</b><br>" +
                               f"Region: {cluster['district']}<br>" +
                               f"Points: {int(cluster['n_points']):,}<br>" +
                               f"Mean: {cluster['mean_velocity']:.1f} mm/yr<br>" +
                               f"Min: {cluster['min_velocity']:.1f} mm/yr<extra></extra>",
                showlegend=True
            ))

        # ⭐ لایه ۳: مرزها
        if districts_gdf is not None:
            for _, r in districts_gdf.iterrows():
                g = r.geometry
                ps = [g] if g.geom_type == "Polygon" else list(g.geoms) if g.geom_type == "MultiPolygon" else []
                for p in ps:
                    coords = list(p.exterior.coords)
                    fig.add_trace(go.Scattermapbox(
                        lon=[c[0] for c in coords],
                        lat=[c[1] for c in coords],
                        mode="lines",
                        line=dict(color="#FF7800", width=1.5),
                        showlegend=False,
                        hoverinfo="skip"
                    ))

        # ⭐ لایه ۴: مترو
        if metro_gdf is not None:
            for _, r in metro_gdf.iterrows():
                g = r.geometry
                ls = [g] if g.geom_type == "LineString" else list(g.geoms) if g.geom_type == "MultiLineString" else []
                for line in ls:
                    coords = list(line.coords)
                    fig.add_trace(go.Scattermapbox(
                        lon=[c[0] for c in coords],
                        lat=[c[1] for c in coords],
                        mode="lines",
                        line=dict(color="#00FFC8", width=1.5),
                        showlegend=False,
                        hoverinfo="skip"
                    ))

        fig.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                accesstoken=MAPBOX_TOKEN if MAPBOX_TOKEN else None,
                center=dict(lat=35.7, lon=51.4),
                zoom=11
            ),
            height=700,
            margin=dict(r=0, t=20, l=0, b=0),
            paper_bgcolor="#0a0e27",
            font=dict(color="#ffcc99", family="Courier New"),
            legend=dict(
                x=0.01, y=0.99,
                bgcolor="rgba(46, 26, 10, 0.9)",
                bordercolor="#FF7800",
                borderwidth=2,
                font=dict(color="#ffcc99", size=11)
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ⭐ بخش پایین: رتبه‌بندی + تحلیل خوشه
        col_rank, col_detail = st.columns([1, 1])

        with col_rank:
            st.markdown("### 📊 Cluster Ranking")

            rank_df = df_hot_sorted[["rank", "cluster_id", "district", "n_points", "mean_velocity"]].copy()
            rank_df["cluster_name"] = rank_df["cluster_id"].apply(lambda x: f"C{int(x) + 1}")
            rank_df = rank_df[["rank", "cluster_name", "district", "n_points", "mean_velocity"]]
            rank_df.columns = ["Rank", "Cluster", "District", "Points", "Mean (mm/yr)"]

            st.dataframe(
                rank_df.style
                    .background_gradient(subset=["Mean (mm/yr)"], cmap="Reds_r")
                    .background_gradient(subset=["Points"], cmap="Reds")
                    .format({
                        "Mean (mm/yr)": "{:.1f}",
                        "Points": "{:,}"
                    }),
                use_container_width=True,
                hide_index=True
            )

        with col_detail:
            st.markdown("### 🎯 Detailed Analysis")

            # انتخاب خوشه
            cluster_options = [f"C{int(c) + 1} · {d} · {int(n):,} points" 
                              for c, d, n in zip(
                                  df_hot_sorted["cluster_id"],
                                  df_hot_sorted["district"],
                                  df_hot_sorted["n_points"]
                              )]

            selected_cluster_name = st.selectbox(
                "📍 Select Cluster to Analyze:",
                options=cluster_options
            )

            selected_idx = cluster_options.index(selected_cluster_name)
            selected_cluster = df_hot_sorted.iloc[selected_idx]

            # اطلاعات خوشه
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2e1a0a 0%, #4e2a1a 100%);
                        border: 2px solid #FF7800; border-radius: 10px;
                        padding: 1rem; margin: 1rem 0;">
                <h4 style="color: #FF7800; margin: 0; font-family: 'Courier New';">
                    Cluster C{int(selected_cluster['cluster_id']) + 1}
                </h4>
                <table style="width: 100%; color: #ffcc99; font-family: 'Courier New';
                              font-size: 0.85rem; margin-top: 0.7rem;">
                    <tr>
                        <td style="padding: 0.3rem;">📍 Region:</td>
                        <td style="text-align: left;"><b>{selected_cluster['district']}</b></td>
                    </tr>
                    <tr>
                        <td style="padding: 0.3rem;">🔢 Points:</td>
                        <td style="text-align: left;"><b>{int(selected_cluster['n_points']):,}</b></td>
                    </tr>
                    <tr>
                        <td style="padding: 0.3rem;">📊 Mean Velocity:</td>
                        <td style="text-align: left;"><b>{selected_cluster['mean_velocity']:.2f} mm/yr</b></td>
                    </tr>
                    <tr>
                        <td style="padding: 0.3rem;">🔻 Min (Deepest):</td>
                        <td style="text-align: left;"><b>{selected_cluster['min_velocity']:.2f} mm/yr</b></td>
                    </tr>
                    <tr>
                        <td style="padding: 0.3rem;">📈 Std Dev:</td>
                        <td style="text-align: left;"><b>{selected_cluster['std_velocity']:.2f} mm/yr</b></td>
                    </tr>
                    <tr>
                        <td style="padding: 0.3rem;">📍 Center:</td>
                        <td style="text-align: left;"><b>({selected_cluster['center_lat']:.4f}, {selected_cluster['center_lon']:.4f})</b></td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

            # آمار اضافی (safe)
            _max_v = abs(df_hot_sorted["mean_velocity"].max())
            if _max_v > 0:
                critical_pct = abs(selected_cluster["mean_velocity"]) / _max_v * 100
            else:
                critical_pct = 0

            st.markdown(f"""
            <div style="background: rgba(46, 26, 10, 0.6); border-left: 4px solid #FF7800;
                        padding: 0.7rem 1rem; border-radius: 5px;">
                <div style="color: #ffcc99; font-family: 'Courier New'; font-size: 0.8rem;">
                    ⚠️ Relative Risk Score
                </div>
                <div style="color: #FF7800; font-family: 'Courier New';
                            font-size: 1.5rem; font-weight: 700;">
                    {critical_pct:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ⭐ نمودار میله‌ای خوشه‌ها
        st.markdown("---")
        st.markdown("### 📊 Cluster Comparison")

        chart_df = df_hot_sorted.sort_values("n_points", ascending=True)
        chart_df["cluster_name"] = chart_df["cluster_id"].apply(lambda x: f"C{int(x) + 1}")

        colors_bar = [cluster_colors[i % len(cluster_colors)] for i in range(len(chart_df))]

        fig2 = go.Figure(data=[go.Bar(
            x=chart_df["n_points"],
            y=chart_df["cluster_name"],
            orientation="h",
            marker=dict(color=colors_bar, line=dict(color="#FF7800", width=2)),
            text=[f"{int(v):,} pts" for v in chart_df["n_points"]],
            textposition="outside",
            textfont=dict(color="#ffcc99", size=11),
            hovertemplate="<b>%{y}</b><br>%{x:,} critical points<extra></extra>"
        )])

        fig2.update_layout(
            height=max(300, len(chart_df) * 60),
            margin=dict(r=80, t=20, l=20, b=20),
            paper_bgcolor="#0a0e27",
            plot_bgcolor="#0a0e27",
            font=dict(color="#ffcc99", family="Courier New"),
            xaxis=dict(
                title=dict(text="Number of Critical Points", font=dict(color="#ffcc99")),
                gridcolor="rgba(255, 120, 0, 0.15)"
            ),
            yaxis=dict(gridcolor="rgba(255, 120, 0, 0.1)"),
            showlegend=False
        )

        st.plotly_chart(fig2, use_container_width=True)

        # ⭐ جدول کامل
        st.markdown("---")
        st.markdown("### 📋 Full Cluster Statistics")

        full_table = df_hot_sorted[[
            "rank", "cluster_id", "district", "n_points",
            "mean_velocity", "min_velocity", "max_velocity", "std_velocity"
        ]].copy()

        full_table["cluster_name"] = full_table["cluster_id"].apply(lambda x: f"C{int(x) + 1}")
        full_table = full_table[["rank", "cluster_name", "district", "n_points",
                                  "mean_velocity", "min_velocity", "max_velocity", "std_velocity"]]
        full_table.columns = ["Rank", "Cluster", "District", "Points",
                              "Mean", "Min", "Max", "Std"]

        st.dataframe(
            full_table.style
                .background_gradient(subset=["Mean"], cmap="Reds_r")
                .background_gradient(subset=["Min"], cmap="Reds_r")
                .background_gradient(subset=["Points"], cmap="Reds")
                .format({
                    "Mean": "{:.2f}",
                    "Min": "{:.2f}",
                    "Max": "{:.2f}",
                    "Std": "{:.2f}",
                    "Points": "{:,}"
                }),
            use_container_width=True,
            hide_index=True
        )

        # ⭐ دانلود CSV
        csv = df_hot_sorted.to_csv(index=False)
        st.download_button(
            label="📥 Download Cluster Data (CSV)",
            data=csv,
            file_name="tehran_hotspots_analysis.csv",
            mime="text/csv"
        )

    else:
        st.warning("⚠️ No hotspot data available")
elif page == "📚 METHODOLOGY":
    # ⭐ بنر منبع فعال (همه صفحات)
    _s = f"📡 {source_label}"
    if vel_active is not None:
        _n = int((~np.isnan(vel_active)).sum())
        _p = _n / vel_active.size * 100
        _s += f" · ✅ {_n:,} پیکسل ({_p:.1f}%)"
    
    if "Ascending" in source_label:
        _q = "🟢 کیفیت بالا · پوشش کم"
    elif "Descending" in source_label:
        _q = "🟡 کیفیت متوسط · پوشش کامل"
    elif "Combined" in source_label:
        _q = "🔵 ترکیبی · پوشش ۱۰۰٪"
    elif "Relative" in source_label:
        _q = "🟣 نسبی · median=0"
    else:
        _q = ""
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(0,217,255,0.15), rgba(0,255,136,0.1));
                border-left: 4px solid #00d9ff;
                padding: 0.6rem 1rem; margin-bottom: 1rem;
                border-radius: 8px; color: #7ee7ff;
                font-family: 'Courier New'; font-size: 0.85rem;">
        <b>📡 منبع فعال:</b> {_s}<br>
        <span style="font-size: 0.75rem; color: #99ffcc;">{_q}</span>
    </div>
    """, unsafe_allow_html=True)

    # ⭐ هدر سبز سای‌فای (تم علمی)
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0a2e1a 0%, #1a4e2a 50%, #0a2e1a 100%);
                padding: 2rem; border-radius: 15px; text-align: center;
                border: 2px solid #00FF88;
                box-shadow: 0 0 60px rgba(0, 255, 136, 0.4);
                margin-bottom: 1.5rem;">
        <h1 style="color: #00FF88; font-family: 'Courier New'; margin: 0;
                   font-size: 2rem; letter-spacing: 4px;
                   text-shadow: 0 0 25px #00FF88;">
            ◉ METHODOLOGY & DOCUMENTATION ◉
        </h1>
        <p style="color: #99ffcc; font-family: 'Courier New';
                  font-size: 1rem; margin-top: 0.7rem; letter-spacing: 3px;">
            ▸ THE SCIENCE BEHIND THE ANALYSIS ◂
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ⭐ بخش ۱: چرا این پروژه مهمه؟
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(0, 255, 136, 0.1), rgba(0, 217, 255, 0.1));
                border-left: 5px solid #00FF88; border-radius: 10px;
                padding: 1.5rem; margin: 1.5rem 0;">
        <h3 style="color: #00FF88; margin: 0; font-family: 'Courier New'; letter-spacing: 2px;">
            💡 WHY THIS PROJECT MATTERS
        </h3>
        <p style="color: #99ffcc; font-family: 'Courier New'; margin-top: 0.8rem; line-height: 1.6;">
            Tehran is sinking — up to <b style="color: #FF1432;">18 cm/year</b> (181 mm/yr) in the
            southwest districts. This project uses <b style="color: #00d9ff;">Sentinel-1 radar
            satellite data</b> to measure ground subsidence with <b style="color: #00FF88;">millimeter
            precision</b>, covering <b>22 districts</b> across <b>12 months</b> of analysis.
            <br><br>
            The results reveal that <b style="color: #FF1432;">16.4% of Tehran's surface</b> is
            subsiding at rates exceeding <b>50 mm/year</b>, affecting over
            <b style="color: #FF7800;">1.28 million people</b> — including critical infrastructure
            such as <b>metro stations</b> and <b>airports</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ⭐ بخش ۲: ۴ کارت تکنولوژی
    st.markdown("### 🛠️ Key Technologies")

    tech_cols = st.columns(4)

    tech_data = [
        ("🛰️", "SENTINEL-1", "29 Images", "#00d9ff"),
        ("🤖", "ASF HyP3", "29 Interferograms", "#FF7800"),
        ("📊", "MintPy", "Time Series Analysis", "#ff00ff"),
        ("🎯", "ML Model", "Linear & Poly-2", "#00FF88")
    ]

    for col, (icon, name, desc, color) in zip(tech_cols, tech_data):
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0a2e1a 0%, #1a4e2a 100%);
                        padding: 1.2rem; border-radius: 12px; text-align: center;
                        border: 2px solid {color};
                        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="color: {color}; font-family: 'Courier New';
                            font-weight: 700; letter-spacing: 2px;
                            margin-top: 0.5rem; text-shadow: 0 0 10px {color};">
                    {name}
                </div>
                <div style="color: #99ffcc; font-family: 'Courier New';
                            font-size: 0.8rem; margin-top: 0.3rem;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ⭐ بخش ۳: فلوچارت پردازش
    st.markdown("### 📊 Processing Pipeline")

    st.markdown("""
    <div style="background: #0a2e1a; padding: 1.5rem; border-radius: 10px;
                border: 2px solid #00FF88; overflow-x: auto;">
        <div style="display: flex; align-items: center; justify-content: space-between;
                    min-width: 800px; gap: 1rem;">
            <div style="text-align: center; flex: 1;">
                <div style="background: #1a4e2a; padding: 0.8rem; border-radius: 8px;
                            border: 2px solid #00d9ff;">
                    <div style="font-size: 1.5rem;">🛰️</div>
                    <div style="color: #00d9ff; font-family: 'Courier New'; font-size: 0.75rem;">
                        Sentinel-1
                    </div>
                </div>
                <div style="color: #99ffcc; font-family: 'Courier New'; font-size: 0.7rem; margin-top: 0.3rem;">
                    Raw SAR
                </div>
            </div>
            <div style="color: #00FF88; font-size: 1.5rem;">→</div>
            <div style="text-align: center; flex: 1;">
                <div style="background: #1a4e2a; padding: 0.8rem; border-radius: 8px;
                            border: 2px solid #FF7800;">
                    <div style="font-size: 1.5rem;">🤖</div>
                    <div style="color: #FF7800; font-family: 'Courier New'; font-size: 0.75rem;">
                        ASF HyP3
                    </div>
                </div>
                <div style="color: #99ffcc; font-family: 'Courier New'; font-size: 0.7rem; margin-top: 0.3rem;">
                    Interferogram
                </div>
            </div>
            <div style="color: #00FF88; font-size: 1.5rem;">→</div>
            <div style="text-align: center; flex: 1;">
                <div style="background: #1a4e2a; padding: 0.8rem; border-radius: 8px;
                            border: 2px solid #ff00ff;">
                    <div style="font-size: 1.5rem;">📊</div>
                    <div style="color: #ff00ff; font-family: 'Courier New'; font-size: 0.75rem;">
                        MintPy
                    </div>
                </div>
                <div style="color: #99ffcc; font-family: 'Courier New'; font-size: 0.7rem; margin-top: 0.3rem;">
                    Time Series
                </div>
            </div>
            <div style="color: #00FF88; font-size: 1.5rem;">→</div>
            <div style="text-align: center; flex: 1;">
                <div style="background: #1a4e2a; padding: 0.8rem; border-radius: 8px;
                            border: 2px solid #00FF88;">
                    <div style="font-size: 1.5rem;">📈</div>
                    <div style="color: #00FF88; font-family: 'Courier New'; font-size: 0.75rem;">
                        Analysis
                    </div>
                </div>
                <div style="color: #99ffcc; font-family: 'Courier New'; font-size: 0.7rem; margin-top: 0.3rem;">
                    Velocity Map
                </div>
            </div>
            <div style="color: #00FF88; font-size: 1.5rem;">→</div>
            <div style="text-align: center; flex: 1;">
                <div style="background: #1a4e2a; padding: 0.8rem; border-radius: 8px;
                            border: 2px solid #FFE600;">
                    <div style="font-size: 1.5rem;">🔮</div>
                    <div style="color: #FFE600; font-family: 'Courier New'; font-size: 0.75rem;">
                        Forecast
                    </div>
                </div>
                <div style="color: #99ffcc; font-family: 'Courier New'; font-size: 0.7rem; margin-top: 0.3rem;">
                    2029 Predict
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ⭐ بخش ۴: ۴ تب موضوعی
    st.markdown("### 📖 Detailed Documentation")

    tab_data, tab_processing, tab_limits, tab_future = st.tabs([
        "🛰️ Data Source",
        "⚙️ Processing",
        "⚠️ Limitations",
        "🚀 Future Work"
    ])

    with tab_data:
        st.markdown("""
        #### 📡 Sentinel-1 Mission

        Sentinel-1 is a **C-band radar satellite** developed by the European Space Agency (ESA).
        Unlike optical satellites, it works **day and night** regardless of weather conditions.

        **Technical Specifications:**
        - 🛰 **Orbit:** Sun-synchronous polar
        - 📡 **Radar Band:** C-band (5.405 GHz)
        - 🔁 **Revisit Time:** 6-12 days
        - 📐 **Spatial Resolution:** 10 meters
        - 📊 **Swath Width:** 250 km (IW mode)

        **Data Details:**
        - 📅 **Period:** January 2024 – December 2024
        - 🔢 **Total Images:** 29 SAR acquisitions
        - 📍 **Orbit Paths:** Descending (Path 35), Ascending (Path 130)
        - 🎯 **Coverage:** Tehran Metropolitan Area
        """)

    with tab_processing:
        st.markdown("""
        #### ⚙️ InSAR Processing Pipeline

        **Step 1: Data Acquisition**
        - Downloaded 29 Sentinel-1 SLC images from ASF
        - Created 29 interferogram pairs (consecutive acquisitions)

        **Step 2: Interferogram Generation (HyP3)**
        - Cloud-based processing by NASA
        - Multilooking: 20x4
        - Phase unwrapping: Minimum Cost Flow

        **Step 3: Time-Series Analysis (MintPy)**
        - Weighted Least Squares inversion
        - Reference point: (35.68156, 51.53089) – Central Tehran
        - Minimum temporal coherence: 0.3

        **Corrections Applied:**
        - ✅ Orbital ramp (linear deramping)
        - ✅ Topographic phase (DEM error)
        - ✅ Tropospheric delay (height correlation)
        """)

    with tab_limits:
        st.markdown("""
        #### ⚠️ Scientific Limitations

        **1. Relative Velocities**
        All velocities are measured **relative to the reference point**.
        Absolute values would require GNSS calibration (not available).

        **2. District Boundaries**
        District boundaries are **approximate** (Voronoi-based around known
        district centers). Real boundaries may differ by 1-2 km.

        **3. Prediction Confidence**
        Three districts have **low prediction confidence** (R² < 0.5):
        - District 3 (R² = 0.10)
        - District 2 (R² = 0.33)
        - District 5 (R² = 0.47)

        **4. Coverage Limitations**
        - Imam Khomeini Airport: outside InSAR coverage
        - Payam Airport: outside InSAR coverage
        - Values for these airports are **estimated**

        **5. Temporal Coverage**
        - Analysis period: **12 months** only
        - Longer periods (>3 years) would improve trend detection
        """)

    with tab_future:
        st.markdown("""
        #### 🚀 Future Work & Recommendations

        **1. GNSS Calibration**
        Install continuous GNSS stations in critical districts
        for absolute velocity measurements.

        **2. Full 2D Decomposition**
        Combine Ascending + Descending orbits to extract:
        - Vertical displacement
        - East-West horizontal motion

        **3. Longer Time Series**
        Extend analysis to **5-10 years** (2020-2029) for
        better trend detection and seasonal analysis.

        **4. Groundwater Integration**
        Overlay groundwater level data from Tehran aquifer
        to establish causal relationship with subsidence.

        **5. Early Warning System**
        Develop a real-time monitoring dashboard with
        automatic alerts when subsidence exceeds thresholds.

        **6. Infrastructure Risk Mapping**
        Identify specific buildings, metro lines, and
        critical infrastructure at immediate risk.
        """)

    st.markdown("---")

    # ⭐ بخش ۵: آمار فنی
    st.markdown("### 📊 Technical Statistics")

    tech_stats = st.columns(5)

    # ⭐ مقادیر dynamic
    _n_dates = len(dates) if dates else 29
    _n_pairs = _n_dates - 1
    _n_valid = int((~np.isnan(vel_active)).sum()) if vel_active is not None else 0
    _n_districts = len(districts_gdf) if districts_gdf is not None else 22

    stats_data = [
        ("📸", f"{_n_dates}", "SAR Images", "#00d9ff"),
        ("🔗", f"{_n_pairs}", "Interferograms", "#FF7800"),
        ("📍", f"{_n_districts}", "Districts", "#ff00ff"),
        ("🎯", f"{_n_valid:,}", "Valid Pixels", "#00FF88"),
        ("📅", "12", "Months", "#FFE600")
    ]

    for col, (icon, value, label, color) in zip(tech_stats, stats_data):
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0a2e1a 0%, #1a4e2a 100%);
                        padding: 1rem; border-radius: 10px; text-align: center;
                        border: 2px solid {color};">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div style="color: {color}; font-family: 'Courier New';
                            font-size: 1.3rem; font-weight: 700;
                            text-shadow: 0 0 10px {color}; margin-top: 0.3rem;">
                    {value}
                </div>
                <div style="color: #99ffcc; font-family: 'Courier New';
                            font-size: 0.75rem; margin-top: 0.2rem;">
                    {label}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ⭐ بخش ۶: مراجع علمی
    st.markdown("### 📚 Scientific References")

    references = [
        ("Amelung et al. (2023)", "InSAR measurements of Tehran subsidence", "Journal of Geophysical Research"),
        ("Fattahi & Amelung (2013)", "DEM error correction in InSAR time series", "IEEE TGRS"),
        ("Doin et al. (2009)", "Tropospheric delay correction by height correlation", "Geophysical Research Letters"),
        ("Yunjun et al. (2019)", "MintPy: Miami InSAR Time-series software in Python", "Computers & Geosciences"),
    ]

    for i, (author, title, journal) in enumerate(references, 1):
        st.markdown(f"""
        <div style="background: rgba(10, 46, 26, 0.6);
                    border-left: 4px solid #00FF88;
                    padding: 0.7rem 1rem; margin: 0.5rem 0;
                    border-radius: 5px;">
            <div style="color: #00FF88; font-family: 'Courier New';
                        font-weight: 700; font-size: 0.9rem;">
                [{i}] {author}
            </div>
            <div style="color: #99ffcc; font-family: 'Courier New';
                        font-size: 0.85rem; margin-top: 0.2rem;">
                {title} · <i>{journal}</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ⭐ بخش ۷: دانلود گزارش
    st.markdown("### 📥 Download Report")

    col_download1, col_download2 = st.columns(2)

    with col_download1:
        if methodology_doc is not None:
            doc_json = json.dumps(methodology_doc, indent=2, ensure_ascii=False)
            st.download_button(
                label="📄 Download Methodology (JSON)",
                data=doc_json,
                file_name="tehran_methodology.json",
                mime="application/json",
                use_container_width=True
            )

    with col_download2:
        # خلاصه‌ی متنی
        summary_text = "TEHRAN SUBSIDENCE ANALYSIS - METHODOLOGY\n" + "=" * 60 + "\n\n"
        summary_text += "Period: Jan-Dec 2024\n"
        summary_text += "Satellite: Sentinel-1 (ESA)\n"
        summary_text += "Processing: ASF HyP3 + MintPy\n"
        _max_sub = np.nanmin(vel_active) * 1000 if vel_active is not None else -181.8
        _critical_px = int((vel_active < -0.05).sum()) if vel_active is not None else 0
        summary_text += "Coverage: 22 districts\n"
        summary_text += f"Max Subsidence: {_max_sub:.1f} mm/yr\n"
        summary_text += f"Critical Pixels: {_critical_px:,}\n"
        st.download_button(
            label="📄 Download Summary (TXT)",
            data=summary_text,
            file_name="tehran_methodology_summary.txt",
            mime="text/plain",
            use_container_width=True
        )