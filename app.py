import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="ClimateScope — Milestone 2", initial_sidebar_state="expanded")

####################################
# Helpers
####################################
@st.cache_data
def load_data(path="final_cleaned_weather.csv"):
    df = pd.read_csv(path)

    # Only normalize column names (safety)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(".", "_", regex=False)
    )

    # Ensure datetime is parsed (if not already)
    if "last_updated" in df.columns:
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")

    # DO NOT recreate year/month/season if already present
    # Just ensure they exist
    if "year" not in df.columns and "last_updated" in df.columns:
        df["year"] = df["last_updated"].dt.year

    if "month" not in df.columns and "last_updated" in df.columns:
        df["month"] = df["last_updated"].dt.month

    if "date" not in df.columns and "last_updated" in df.columns:
        df["date"] = df["last_updated"].dt.date

    # Turbulence only if not already computed
    if "wind_turbulence" not in df.columns:
        if "gust_kph" in df.columns and "wind_kph" in df.columns:
            df["wind_turbulence"] = df["gust_kph"] - df["wind_kph"]

    return df

def safe_groupby_mean(df, by, cols):
    # helper that returns a DataFrame even if empty
    if len(df) == 0:
        return pd.DataFrame(columns=by + cols)
    return df.groupby(by).agg({c: "mean" for c in cols}).reset_index()

def iqr_filter(df, col, lower=None, upper=None):
    if col not in df.columns or df[col].dropna().empty:
        return df
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    l = q1 - 1.5 * iqr if lower is None else lower
    u = q3 + 1.5 * iqr if upper is None else upper
    return df[(df[col] >= l) & (df[col] <= u)]

####################################
# Load data
####################################
st.sidebar.header("data & settings")
data_path = st.sidebar.text_input("cleaned csv path", value="final_cleaned_weather.csv")
df = load_data(data_path)

# quick schema check
st.sidebar.markdown(f"**rows:** {len(df):,}  \n**columns:** {len(df.columns)}")
if "country" in df.columns:
    st.sidebar.markdown(f"**countries:** {df['country'].nunique():,}")

####################################
# Sidebar: user controls
####################################
st.sidebar.header("filters")
country_all = ["All"] + sorted(df["country"].dropna().unique().tolist()) if "country" in df.columns else ["All"]
default_countries = country_all[1:4] if len(country_all) > 4 else country_all[1:]
selected_countries = st.sidebar.multiselect("countries (empty = ALL)", options=country_all, default=["All"])
if selected_countries == []:
    selected_countries = ["All"]

# location compare
st.sidebar.markdown("---")
st.sidebar.subheader("location compare")
compare_country = st.sidebar.selectbox("country for compare (left)", options=country_all[1:] if len(country_all)>1 else ["None"])
compare_location = st.sidebar.selectbox("location (left)", options=(["None"] + sorted(df[df["country"]==compare_country]["location_name"].unique().tolist())) if compare_country!="All" else ["None"])
compare_country_r = st.sidebar.selectbox("country for compare (right)", options=country_all[1:] if len(country_all)>1 else ["None"])
compare_location_r = st.sidebar.selectbox("location (right)", options=(["None"] + sorted(df[df["country"]==compare_country_r]["location_name"].unique().tolist())) if compare_country_r!="All" else ["None"])

# date range
min_date = df["last_updated"].min()
max_date = df["last_updated"].max()
date_range = st.sidebar.date_input("date range", value=(min_date.date() if min_date is not pd.NaT else None, max_date.date() if max_date is not pd.NaT else None))

# rolling window
rolling_window = st.sidebar.selectbox("rolling window (months) for trend smoothing", options=[1,3,6], index=1)

# aggregation granularity
granularity = st.sidebar.selectbox("Aggregation", options=["monthly", "daily"], index=0)

# mapping toggle
show_choropleth = st.sidebar.checkbox("show choropleth", value=True)

# global vs filtered for latitude band analysis
lat_global_toggle = st.sidebar.checkbox("latitude-temp: use all data (global)", value=True)

# Download processed
csv_bytes = df.to_csv(index=False).encode()

st.sidebar.download_button(
    label="Download dataset",
    data=csv_bytes,
    file_name="weather.csv",
    mime="text/csv"
)
####################################
# Apply filter pipeline (defensive)
####################################
filtered_df = df.copy()
# apply country filter
if "All" not in selected_countries:
    filtered_df = filtered_df[filtered_df["country"].isin(selected_countries)].copy()

# apply date filter
try:
    start_date, end_date = date_range
    if start_date is not None and end_date is not None:
        mask = (filtered_df["last_updated"].dt.date >= pd.to_datetime(start_date).date()) & (filtered_df["last_updated"].dt.date <= pd.to_datetime(end_date).date())
        filtered_df = filtered_df[mask].copy()
except Exception:
    pass

# make sure numeric columns exist for visuals
numeric_cols = [c for c in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[c])]
display_cols = ["temperature_celsius", "humidity", "precip_mm", "wind_kph", "wind_turbulence", "air_quality_pm25"]
display_cols = [c for c in display_cols if c in filtered_df.columns]

####################################
# Top-row KPIs
####################################
st.title("ClimateScope — Interactive Dashboard ")
k1, k2, k3, k4 = st.columns(4)
with k1:
    if "temperature_celsius" in filtered_df.columns:
        st.metric("avg temp (°C)", f"{filtered_df['temperature_celsius'].mean():.2f}")
    else:
        st.metric("avg temp (°C)", "n/a")
with k2:
    if "air_quality_pm25" in filtered_df.columns:
        st.metric("avg pm2.5", f"{filtered_df['air_quality_pm25'].mean():.2f}")
    else:
        st.metric("avg pm2.5", "n/a")
with k3:
    if "precip_mm" in filtered_df.columns:
        st.metric("avg precip (mm)", f"{filtered_df['precip_mm'].mean():.3f}")
    else:
        st.metric("avg precip (mm)", "n/a")
with k4:
    if "wind_kph" in filtered_df.columns:
        st.metric("avg wind (kph)", f"{filtered_df['wind_kph'].mean():.2f}")
    else:
        st.metric("avg wind (kph)", "n/a")

st.markdown("---")

####################################
# Section: Country comparison + choropleth
####################################
st.header("Country comparison & choropleth")
if filtered_df.empty:
    st.warning("No data for selected filters. Clear filters or expand date range.")
else:
    # country summary
    agg_cols = [c for c in ["temperature_celsius", "humidity", "precip_mm", "wind_kph", "air_quality_pm25"] if c in filtered_df.columns]
    country_summary = filtered_df.groupby("country").agg({c: "mean" for c in agg_cols}).reset_index()
    country_summary = country_summary.sort_values(by=agg_cols[0] if agg_cols else country_summary.columns[1], ascending=False)

    col1, col2 = st.columns([2, 1])
    with col1:
        # bar chart for top countries by selected metric
        metric = st.selectbox("metric to compare (bar)", options=agg_cols, index=0 if agg_cols else 0)
        fig = px.bar(country_summary.sort_values(metric, ascending=False).head(20),
                     x="country", y=metric,
                     title=f"Top countries by {metric}")
        st.plotly_chart(fig, use_container_width='stretch')
    with col2:
        if show_choropleth and "country" in country_summary.columns:
            # safe choropleth (some country names may not map exactly)
            try:
                fig_map = px.choropleth(country_summary, locations="country",
                                        locationmode="country names",
                                        color=metric,
                                        title=f"Choropleth: {metric}")
                st.plotly_chart(fig_map, use_container_width='stretch')
            except Exception:
                st.info("choropleth failed for some country names — names may not match standard country names.")

st.markdown("---")

####################################
# Section: Monthly / Daily trend with smoothing
####################################
st.header("Time-series trends")

if filtered_df.empty:
    st.info("No time-series data available for selected filters.")
else:
    if granularity == "monthly":
        ts = filtered_df.groupby(["year", "month"]).agg({"temperature_celsius": "mean"}).reset_index()
        ts["period"] = pd.to_datetime(ts["year"].astype(int).astype(str) + "-" + ts["month"].astype(int).astype(str) + "-01")
    else:
        ts = filtered_df.groupby("date").agg({"temperature_celsius": "mean"}).reset_index()
        ts["period"] = pd.to_datetime(ts["date"])

    # rolling smoothing
    ts = ts.sort_values("period")
    ts["rolling_temp"] = ts["temperature_celsius"].rolling(window=rolling_window, min_periods=1).mean()

    fig_ts = px.line(ts, x="period", y=["temperature_celsius", "rolling_temp"],
                     labels={"value": "Temperature (°C)", "period": "time"},
                     title="Temperature trend (raw + smoothed)")
    st.plotly_chart(fig_ts, use_container_width='stretch')

st.markdown("---")

####################################
# Latitude -> Temperature gradient (global option)
####################################
st.header("Latitude → Temperature gradient")

# compute on global df or filtered_df depending on toggle
lat_df = df if lat_global_toggle else filtered_df

if lat_df.empty or "latitude" not in lat_df.columns or "temperature_celsius" not in lat_df.columns:
    st.info("Latitude or temperature data is missing for this selection.")
else:
    lat_bins = st.select_slider("latitude bin size (degrees)", options=[10, 20, 30], value=30)
    bins = list(np.arange(-90, 90 + lat_bins, lat_bins))
    labels = [f"{int(bins[i])} to {int(bins[i+1])}" for i in range(len(bins)-1)]
    lat_df["lat_band"] = pd.cut(lat_df["latitude"], bins=bins, labels=labels, include_lowest=True)
    lat_summary = lat_df.groupby("lat_band").agg({"temperature_celsius": "mean", "humidity": "mean", "air_quality_pm25": "mean"}).reset_index()
    fig_lat = px.bar(lat_summary, x="lat_band", y="temperature_celsius", title="Latitude band average temperature")
    st.plotly_chart(fig_lat, use_container_width='stretch')

st.markdown("---")

####################################
# Section: correlation heatmap (plotly)
####################################
st.header("Correlation matrix")

if filtered_df.empty:
    st.info("No data to compute correlations.")
else:
    corr_cols = [c for c in filtered_df.select_dtypes("number").columns if filtered_df[c].nunique()>1]
    if len(corr_cols) < 2:
        st.info("Not enough numerical columns for correlation.")
    else:
        corr = filtered_df[corr_cols].corr()
        # use plotly.imshow which handles empty/NaN gracefully
        fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=False, title="Correlation heatmap")
        st.plotly_chart(fig_corr, use_container_width='stretch')

st.markdown("---")

####################################
# Section: Seasonal heatmap (plotly)
####################################
st.header("Seasonal temperature heatmap")

if filtered_df.empty:
    st.info("No data for seasonal heatmap.")
else:
    season_pivot = filtered_df.pivot_table(values="temperature_celsius", index="season", columns="year", aggfunc="mean")
    # drop empty rows/cols
    season_pivot = season_pivot.dropna(how="all").dropna(axis=1, how="all")
    if season_pivot.empty:
        st.info("seasonal heatmap: insufficient data for selected filters.")
    else:
        fig_season = px.imshow(season_pivot, text_auto=".2f", aspect="auto", title="Season vs Year: Avg Temperature")
        st.plotly_chart(fig_season, use_container_width='stretch')

st.markdown("---")

####################################
# Section: extremes & anomaly detection
####################################
st.header("Extreme events & anomalies")

if filtered_df.empty:
    st.info("No data to detect extremes.")
else:
    # choose method: quantile or zscore
    method = st.radio("extreme detection method", options=["quantile (95th)","z-score (>3)"], index=0, horizontal=True)
    metric = st.selectbox("metric for extremes", options=[c for c in ["temperature_celsius","precip_mm","air_quality_pm25","wind_kph"] if c in filtered_df.columns], index=0)
    if method.startswith("quantile"):
        thresh = filtered_df[metric].quantile(0.95)
        extremes = filtered_df[filtered_df[metric] > thresh]
        st.write(f"{metric} 95th percentile = {thresh:.3f}, extreme count = {len(extremes)}")
    else:
        from scipy import stats
        z = np.abs(stats.zscore(filtered_df[metric].dropna()))
        mask = z > 3
        extremes = filtered_df.loc[filtered_df[metric].dropna().index[mask]]
        st.write(f"z-score threshold = 3, extreme count = {len(extremes)}")

    if not extremes.empty:
        top_by_country = extremes["country"].value_counts().reset_index().rename(columns={"index":"country", metric:"count"})
        fig_ex = px.bar(top_by_country.head(15), x="country", y=metric if metric in top_by_country.columns else "count", title=f"Top countries with extreme {metric}")
        st.plotly_chart(fig_ex, use_container_width='stretch')
        st.dataframe(extremes[[c for c in ["country","location_name","last_updated",metric] if c in extremes.columns]].sort_values(metric, ascending=False).head(200))
    else:
        st.info("No extremes found with current settings.")

st.markdown("---")

####################################
# Section: compare two locations (small multiples)
####################################
st.header("Compare two locations")

def get_location_df(country, location):
    if country in (None, "None", "All") or location in (None, "None"):
        return pd.DataFrame()
    d = df[(df["country"] == country) & (df["location_name"] == location)]
    return d.sort_values("last_updated")

left_df = get_location_df(compare_country, compare_location)
right_df = get_location_df(compare_country_r, compare_location_r)

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("Left location")
    if left_df.empty:
        st.info("Select left country & location")
    else:
        st.write(f"{compare_location} — {compare_country}")
        st.line_chart(left_df.set_index("last_updated")[["temperature_celsius"]].rolling(window=rolling_window).mean())

with col_r:
    st.subheader("Right location")
    if right_df.empty:
        st.info("Select right country & location")
    else:
        st.write(f"{compare_location_r} — {compare_country_r}")
        st.line_chart(right_df.set_index("last_updated")[["temperature_celsius"]].rolling(window=rolling_window).mean())

st.markdown("---")