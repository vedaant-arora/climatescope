#milestone 3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
 
# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="ClimateScope",
    page_icon="🌍",
    initial_sidebar_state="expanded",
)
 
TEMPLATE = "plotly_white"
COLOR_SEQ = px.colors.qualitative.Set2
 
# ── helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "final_cleaned_weather.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(".", "_", regex=False)
    )
    if "last_updated" in df.columns:
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    if "year" not in df.columns and "last_updated" in df.columns:
        df["year"] = df["last_updated"].dt.year
    if "month" not in df.columns and "last_updated" in df.columns:
        df["month"] = df["last_updated"].dt.month
    if "date" not in df.columns and "last_updated" in df.columns:
        df["date"] = df["last_updated"].dt.date
    if "wind_turbulence" not in df.columns and {"gust_kph", "wind_kph"}.issubset(df.columns):
        df["wind_turbulence"] = df["gust_kph"] - df["wind_kph"]
    return df
 
 
@st.cache_resource
def train_model(path: str = "final_cleaned_weather.csv"):
    df = load_data(path)
    features = ["humidity", "wind_kph", "pressure_mb", "precip_mm",
                 "uv_index", "visibility_km", "cloud"]
    target = "temperature_celsius"
    X = df[features].dropna()
    y = df.loc[X.index, target]
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf, features, X, y
 
 
def apply_filters(df, selected_countries, date_range):
    out = df.copy()
    if "All" not in selected_countries:
        out = out[out["country"].isin(selected_countries)]
    try:
        start, end = date_range
        if start and end:
            mask = (
                (out["last_updated"].dt.date >= pd.to_datetime(start).date()) &
                (out["last_updated"].dt.date <= pd.to_datetime(end).date())
            )
            out = out[mask]
    except Exception:
        pass
    return out.copy()
 
 
# ── load data ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
data_path = st.sidebar.text_input("CSV path", value="final_cleaned_weather.csv")
df = load_data(data_path)
 
st.sidebar.markdown(
    f"**Rows:** {len(df):,} &nbsp;|&nbsp; **Countries:** {df['country'].nunique()}"
)
 
# ── sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")
 
country_opts = ["All"] + sorted(df["country"].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Countries (empty = All)", options=country_opts, default=["All"]
) or ["All"]
 
min_date = df["last_updated"].min().date()
max_date = df["last_updated"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date))
 
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Chart options")
rolling_window = st.sidebar.selectbox("Rolling window (months)", [1, 3, 6], index=1)
granularity   = st.sidebar.selectbox("Time aggregation", ["monthly", "daily"], index=0)
show_choropleth = st.sidebar.checkbox("Show choropleth", value=True)
lat_global    = st.sidebar.checkbox("Lat–temp: use global data", value=True)
 
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Location compare")
countries_no_all = country_opts[1:] if len(country_opts) > 1 else ["None"]
 
cmp_country_l = st.sidebar.selectbox("Left country",  countries_no_all, key="cl")
cmp_loc_l     = st.sidebar.selectbox("Left location",
    ["None"] + sorted(df[df["country"] == cmp_country_l]["location_name"].unique().tolist()),
    key="ll")
 
cmp_country_r = st.sidebar.selectbox("Right country", countries_no_all, key="cr")
cmp_loc_r     = st.sidebar.selectbox("Right location",
    ["None"] + sorted(df[df["country"] == cmp_country_r]["location_name"].unique().tolist()),
    key="lr")
 
st.sidebar.markdown("---")
st.sidebar.download_button(
    "⬇️ Download dataset",
    data=df.to_csv(index=False).encode(),
    file_name="weather.csv",
    mime="text/csv",
)
 
# ── apply filters ──────────────────────────────────────────────────────────────
fdf = apply_filters(df, selected_countries, date_range)
 
# ── title & KPIs ───────────────────────────────────────────────────────────────
st.title("🌍 ClimateScope — Global Weather Dashboard")
st.caption("Visualizing global weather trends, extremes, and predictions.")
 
if fdf.empty:
    st.warning("No data for current filters. Adjust country or date range.")
    st.stop()
 
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🌡️ Avg Temp (°C)",   f"{fdf['temperature_celsius'].mean():.1f}")
k2.metric("💧 Avg Humidity (%)", f"{fdf['humidity'].mean():.1f}")
k3.metric("🌧️ Avg Precip (mm)", f"{fdf['precip_mm'].mean():.2f}")
k4.metric("💨 Avg Wind (kph)",  f"{fdf['wind_kph'].mean():.1f}")
k5.metric("😮‍💨 Avg PM2.5",       f"{fdf['air_quality_pm25'].mean():.1f}")
 
st.markdown("---")
 
# ── 1. Country comparison & choropleth ────────────────────────────────────────
st.header("1 · Country Comparison")
 
agg_cols = [c for c in
    ["temperature_celsius", "humidity", "precip_mm", "wind_kph", "air_quality_pm25"]
    if c in fdf.columns]
country_summary = fdf.groupby("country").agg({c: "mean" for c in agg_cols}).reset_index()
 
metric_bar = st.selectbox("Metric", agg_cols, index=0, key="metric_bar")
 
col1, col2 = st.columns([1, 1])
with col1:
    fig_bar = px.bar(
        country_summary.sort_values(metric_bar, ascending=False).head(20),
        x="country", y=metric_bar,
        color=metric_bar,
        color_continuous_scale="Reds",
        labels={metric_bar: metric_bar.replace("_", " ").title(), "country": "Country"},
        title=f"Top 20 countries — {metric_bar.replace('_', ' ').title()}",
        template=TEMPLATE,
    )
    fig_bar.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)
 
with col2:
    if show_choropleth:
        try:
            fig_map = px.choropleth(
                country_summary,
                locations="country",
                locationmode="country names",
                color=metric_bar,
                color_continuous_scale="Reds",
                title=f"Choropleth — {metric_bar.replace('_', ' ').title()}",
                template=TEMPLATE,
            )
            fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception:
            st.info("Choropleth unavailable — some country names may not match ISO standard.")
    else:
        st.info("Choropleth disabled. Toggle in sidebar.")
 
st.markdown("---")
 
# ── 2. Time-series trends ─────────────────────────────────────────────────────
st.header("2 · Temperature Trend")
 
if granularity == "monthly":
    ts = fdf.groupby(["year", "month"]).agg(temperature_celsius=("temperature_celsius", "mean")).reset_index()
    ts["period"] = pd.to_datetime(
        ts["year"].astype(int).astype(str) + "-" + ts["month"].astype(int).astype(str) + "-01"
    )
else:
    ts = fdf.groupby("date").agg(temperature_celsius=("temperature_celsius", "mean")).reset_index()
    ts["period"] = pd.to_datetime(ts["date"])
 
ts = ts.sort_values("period")
ts["smoothed"] = ts["temperature_celsius"].rolling(rolling_window, min_periods=1).mean()
 
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=ts["period"], y=ts["temperature_celsius"],
    name="Raw", mode="lines",
    line=dict(color="#b0c4de", width=1),
))
fig_ts.add_trace(go.Scatter(
    x=ts["period"], y=ts["smoothed"],
    name=f"{rolling_window}-period smoothed", mode="lines",
    line=dict(color="#e63946", width=2.5),
))
fig_ts.update_layout(
    title="Temperature over time (raw + smoothed)",
    xaxis_title="Time", yaxis_title="Temperature (°C)",
    template=TEMPLATE, legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig_ts, use_container_width=True)
 
st.markdown("---")
 
# ── 3. Latitude → Temperature gradient ───────────────────────────────────────
st.header("3 · Latitude → Temperature Gradient")
 
lat_source = df if lat_global else fdf
lat_df = lat_source.copy()   # always copy before mutating
 
if "latitude" in lat_df.columns:
    lat_bins = st.select_slider("Latitude bin size (°)", options=[10, 20, 30], value=20)
    bins   = list(np.arange(-90, 90 + lat_bins, lat_bins))
    labels = [f"{int(bins[i])}° to {int(bins[i+1])}°" for i in range(len(bins) - 1)]
    lat_df["lat_band"] = pd.cut(lat_df["latitude"], bins=bins, labels=labels, include_lowest=True)
 
    lat_summary = lat_df.groupby("lat_band", observed=True).agg(
        temperature_celsius=("temperature_celsius", "mean"),
        humidity=("humidity", "mean"),
    ).reset_index()
 
    fig_lat = px.bar(
        lat_summary, x="lat_band", y="temperature_celsius",
        color="temperature_celsius", color_continuous_scale="RdYlBu_r",
        labels={"lat_band": "Latitude band", "temperature_celsius": "Avg Temp (°C)"},
        title="Average temperature by latitude band",
        template=TEMPLATE,
    )
    fig_lat.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_lat, use_container_width=True)
else:
    st.info("Latitude column not available.")
 
st.markdown("---")
 
# ── 4. Scatterplot — correlations ─────────────────────────────────────────────
st.header("4 · Correlation Scatterplot")
 
scatter_opts = [c for c in
    ["humidity", "feels_like_celsius", "precip_mm", "wind_kph",
     "pressure_mb", "uv_index", "air_quality_pm25", "cloud", "visibility_km"]
    if c in fdf.columns]
 
sc1, sc2, sc3 = st.columns(3)
x_axis  = sc1.selectbox("X axis",     scatter_opts, index=0, key="sx")
y_axis  = sc2.selectbox("Y axis",     scatter_opts, index=1, key="sy")
color_by = sc3.selectbox("Color by",  ["season", "country", "none"], index=0, key="sc")
 
sample = fdf.sample(min(3000, len(fdf)), random_state=42)
color_col = None if color_by == "none" else color_by
 
fig_sc = px.scatter(
    sample, x=x_axis, y=y_axis,
    color=color_col,
    opacity=0.55,
    trendline="ols",
    trendline_scope="overall",
    labels={
        x_axis:  x_axis.replace("_", " ").title(),
        y_axis:  y_axis.replace("_", " ").title(),
        color_col: color_col.replace("_", " ").title() if color_col else "",
    },
    title=f"{x_axis.replace('_',' ').title()} vs {y_axis.replace('_',' ').title()}",
    template=TEMPLATE,
    color_discrete_sequence=COLOR_SEQ,
)
fig_sc.update_traces(marker=dict(size=4))
st.plotly_chart(fig_sc, use_container_width=True)
 
st.markdown("---")
 
# ── 5. Correlation heatmap ────────────────────────────────────────────────────
st.header("5 · Correlation Matrix")
 
num_cols = [c for c in fdf.select_dtypes("number").columns if fdf[c].nunique() > 1]
if len(num_cols) >= 2:
    corr = fdf[num_cols].corr()
    fig_corr = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Feature correlation matrix",
        template=TEMPLATE,
    )
    fig_corr.update_layout(height=550)
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Not enough numeric columns.")
 
st.markdown("---")
 
# ── 6. Seasonal heatmap ───────────────────────────────────────────────────────
st.header("6 · Seasonal Temperature Heatmap")
 
season_pivot = fdf.pivot_table(
    values="temperature_celsius", index="season", columns="year", aggfunc="mean"
)
season_pivot = season_pivot.dropna(how="all").dropna(axis=1, how="all")
 
if not season_pivot.empty:
    fig_season = px.imshow(
        season_pivot, text_auto=".1f", aspect="auto",
        color_continuous_scale="RdYlBu_r",
        title="Season × Year — Average Temperature (°C)",
        template=TEMPLATE,
    )
    st.plotly_chart(fig_season, use_container_width=True)
else:
    st.info("Insufficient data for seasonal heatmap with current filters.")
 
st.markdown("---")
 
# ── 7. Extreme events ─────────────────────────────────────────────────────────
st.header("7 · Extreme Events")
 
ex1, ex2 = st.columns(2)
method    = ex1.radio("Detection method", ["Quantile (95th)", "Z-score (> 3)"], horizontal=True)
ex_metric = ex2.selectbox("Metric",
    [c for c in ["temperature_celsius", "precip_mm", "air_quality_pm25", "wind_kph"]
     if c in fdf.columns], index=0)
 
if method.startswith("Quantile"):
    thresh   = fdf[ex_metric].quantile(0.95)
    extremes = fdf[fdf[ex_metric] > thresh].copy()
    st.caption(f"95th percentile threshold = {thresh:.2f} | {len(extremes):,} extreme records")
else:
    from scipy import stats
    valid   = fdf[ex_metric].dropna()
    z       = np.abs(stats.zscore(valid))
    extremes = fdf.loc[valid.index[z > 3]].copy()
    st.caption(f"Z-score > 3 | {len(extremes):,} extreme records")
 
if not extremes.empty:
    top_countries = extremes["country"].value_counts().reset_index()
    top_countries.columns = ["country", "count"]
 
    fig_ex = px.bar(
        top_countries.head(15),
        x="country", y="count",
        color="count", color_continuous_scale="Oranges",
        labels={"country": "Country", "count": "Extreme event count"},
        title=f"Top countries — extreme {ex_metric.replace('_', ' ')}",
        template=TEMPLATE,
    )
    fig_ex.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_ex, use_container_width=True)
 
    show_cols = [c for c in ["country", "location_name", "last_updated", ex_metric] if c in extremes.columns]
    st.dataframe(
        extremes[show_cols].sort_values(ex_metric, ascending=False).head(200),
        use_container_width=True,
    )
else:
    st.info("No extremes found with current settings.")
 
st.markdown("---")
 
# ── 8. Temperature prediction ─────────────────────────────────────────────────
st.header("8 · Temperature Predictor")
st.caption("Random Forest model trained on the full dataset (R² ≈ 0.96). Adjust sliders to predict temperature.")
 
rf_model, feat_names, X_train, y_train = train_model(data_path)
 
p1, p2 = st.columns([1, 1])
with p1:
    st.subheader("Input conditions")
    in_humidity    = st.slider("Humidity (%)",       int(df["humidity"].min()),    int(df["humidity"].max()),    int(df["humidity"].mean()))
    in_wind        = st.slider("Wind speed (kph)",   float(df["wind_kph"].min()),  float(df["wind_kph"].max()),  float(df["wind_kph"].mean()), step=0.5)
    in_pressure    = st.slider("Pressure (mb)",      float(df["pressure_mb"].min()), float(df["pressure_mb"].max()), float(df["pressure_mb"].mean()), step=0.5)
    in_precip      = st.slider("Precipitation (mm)", 0.0, float(df["precip_mm"].max()), 0.0, step=0.1)
    in_uv          = st.slider("UV index",           0.0, float(df["uv_index"].max()),  float(df["uv_index"].mean()), step=0.1)
    in_visibility  = st.slider("Visibility (km)",    0.0, float(df["visibility_km"].max()), float(df["visibility_km"].mean()), step=0.5)
    in_cloud       = st.slider("Cloud cover (%)",    0,   100,                           int(df["cloud"].mean()))
 
    input_vec = np.array([[in_humidity, in_wind, in_pressure, in_precip,
                           in_uv, in_visibility, in_cloud]])
    pred_temp = rf_model.predict(input_vec)[0]
 
    st.markdown("---")
    st.metric("🌡️ Predicted Temperature", f"{pred_temp:.1f} °C")
 
with p2:
    st.subheader("Feature importance")
    imp_df = pd.DataFrame({
        "Feature": [f.replace("_", " ").title() for f in feat_names],
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=True)
 
    fig_imp = px.bar(
        imp_df, x="Importance", y="Feature",
        orientation="h",
        color="Importance", color_continuous_scale="Blues",
        title="What drives temperature predictions?",
        template=TEMPLATE,
    )
    fig_imp.update_layout(coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig_imp, use_container_width=True)
 
st.markdown("---")
 
# ── 9. Compare two locations ──────────────────────────────────────────────────
st.header("9 · Compare Two Locations")
 
def get_loc_df(country, location):
    if not country or not location or location == "None":
        return pd.DataFrame()
    return df[(df["country"] == country) & (df["location_name"] == location)].sort_values("last_updated")
 
left_df  = get_loc_df(cmp_country_l, cmp_loc_l)
right_df = get_loc_df(cmp_country_r, cmp_loc_r)
 
col_l, col_r = st.columns(2)
for side_df, country, location, col in [
    (left_df, cmp_country_l, cmp_loc_l, col_l),
    (right_df, cmp_country_r, cmp_loc_r, col_r),
]:
    with col:
        if side_df.empty:
            st.info("Select a country and location from the sidebar.")
        else:
            smoothed = side_df.set_index("last_updated")[["temperature_celsius"]].rolling(rolling_window).mean()
            fig_loc = px.line(
                smoothed.reset_index(), x="last_updated", y="temperature_celsius",
                labels={"last_updated": "Date", "temperature_celsius": "Temp (°C)"},
                title=f"{location}, {country}",
                template=TEMPLATE,
            )
            fig_loc.update_traces(line_color="#e63946")
            st.plotly_chart(fig_loc, use_container_width=True)
 
st.markdown("---")
st.caption("ClimateScope · Built with Streamlit & Plotly · Data: Kaggle Global Weather Repository")