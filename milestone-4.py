import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    layout="wide",
    page_title="ClimateScope",
    page_icon="🌍",
    initial_sidebar_state="expanded",
)

TEMPLATE  = "plotly_white"
COLOR_SEQ = px.colors.qualitative.Set2
MONTH_LABELS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# WHO annual / 24-hour guideline reference values (ug/m3)
WHO = {
    "air_quality_pm25":               5,
    "air_quality_pm10":              15,
    "air_quality_nitrogen_dioxide":  10,
    "air_quality_sulphur_dioxide":   40,
    "air_quality_ozone":            100,
    "air_quality_carbon_monoxide": 4000,
}
AQ_LABELS = {
    "air_quality_pm25":              "PM2.5 (µg/m³)",
    "air_quality_pm10":              "PM10 (µg/m³)",
    "air_quality_nitrogen_dioxide":  "NO₂ (µg/m³)",
    "air_quality_sulphur_dioxide":   "SO₂ (µg/m³)",
    "air_quality_ozone":             "O₃ (µg/m³)",
    "air_quality_carbon_monoxide":   "CO (µg/m³)",
}


# ── data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "final_cleaned_weather.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_").str.replace("-", "_").str.replace(".", "_", regex=False)
    )
    if "last_updated" in df.columns:
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    for col, src in [("year","last_updated"),("month","last_updated"),("date","last_updated")]:
        if col not in df.columns and src in df.columns:
            if col == "date":    df[col] = df[src].dt.date
            elif col == "year":  df[col] = df[src].dt.year
            elif col == "month": df[col] = df[src].dt.month
    if "wind_turbulence" not in df.columns and {"gust_kph","wind_kph"}.issubset(df.columns):
        df["wind_turbulence"] = df["gust_kph"] - df["wind_kph"]
    return df


@st.cache_resource
def train_model(path: str = "final_cleaned_weather.csv"):
    df = load_data(path)
    features = ["humidity","wind_kph","pressure_mb","precip_mm",
                 "uv_index","visibility_km","cloud"]
    X = df[features].dropna()
    y = df.loc[X.index, "temperature_celsius"]
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf, features


@st.cache_data
def build_travel_scores(path: str = "final_cleaned_weather.csv") -> pd.DataFrame:
    df = load_data(path)
    monthly = df.groupby(["country","location_name","month"]).agg(
        temp   = ("temperature_celsius","mean"),
        precip = ("precip_mm",          "mean"),
        wind   = ("wind_kph",           "mean"),
        uv     = ("uv_index",           "mean"),
        vis    = ("visibility_km",      "mean"),
        pm25   = ("air_quality_pm25",   "mean"),
    ).reset_index()
    monthly["temp_score"]   = np.exp(-((monthly["temp"] - 23) / 8) ** 2)
    monthly["precip_score"] = 1 - np.clip(monthly["precip"] / 10, 0, 1)
    monthly["wind_score"]   = 1 - np.clip(monthly["wind"]   / 40, 0, 1)
    monthly["uv_score"]     = 1 - np.clip(monthly["uv"]     / 11, 0, 1)
    monthly["vis_score"]    = np.clip(monthly["vis"] / 20, 0, 1)
    monthly["aq_score"]     = 1 - np.clip(monthly["pm25"]   / 75, 0, 1)
    weights = dict(temp_score=0.35, precip_score=0.20, aq_score=0.20,
                   uv_score=0.10, wind_score=0.10, vis_score=0.05)
    monthly["travel_score"] = sum(monthly[k] * v for k, v in weights.items()) * 100
    monthly["month_label"]  = monthly["month"].map(MONTH_LABELS)
    return monthly


@st.cache_data
def build_comfort_index(path: str = "final_cleaned_weather.csv") -> pd.DataFrame:
    df = load_data(path).copy()

    fl  = df["feels_like_celsius"].values
    uv  = df["uv_index"].values
    w   = df["wind_kph"].values
    pr  = df["precip_mm"].values

    heat_s   = np.where(fl < 0,   0.0,
               np.where(fl < 10,  0.3,
               np.where(fl < 18,  0.7,
               np.where(fl < 28,  1.0,
               np.where(fl < 35,  0.7,
               np.where(fl < 40,  0.4, 0.1))))))
    uv_s     = np.where(uv <= 2,  1.0,
               np.where(uv <= 5,  0.75,
               np.where(uv <= 7,  0.5,
               np.where(uv <= 10, 0.25, 0.0))))
    wind_s   = np.where(w  < 20,  1.0,
               np.where(w  < 40,  0.7,
               np.where(w  < 60,  0.4, 0.1)))
    precip_s = np.where(pr == 0,  1.0,
               np.where(pr <= 1,  0.9,
               np.where(pr <= 5,  0.7,
               np.where(pr <= 10, 0.5, 0.2))))

    df["heat_s"]       = heat_s
    df["uv_s"]         = uv_s
    df["wind_s"]       = wind_s
    df["precip_s"]     = precip_s
    df["comfort_score"] = (heat_s*0.40 + uv_s*0.25 + wind_s*0.20 + precip_s*0.15) * 100
    df["comfort_label"] = np.where(df["comfort_score"] >= 80, "Ideal",
                          np.where(df["comfort_score"] >= 60, "Good",
                          np.where(df["comfort_score"] >= 40, "Fair",
                          np.where(df["comfort_score"] >= 20, "Poor", "Harsh"))))
    return df


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


# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
data_path = st.sidebar.text_input("CSV path", value="final_cleaned_weather.csv")
df = load_data(data_path)
st.sidebar.markdown(f"**Rows:** {len(df):,} &nbsp;|&nbsp; **Countries:** {df['country'].nunique()}")
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

country_opts = ["All"] + sorted(df["country"].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Countries (empty = All)", options=country_opts, default=["All"]
) or ["All"]

min_date   = df["last_updated"].min().date()
max_date   = df["last_updated"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date))

st.sidebar.markdown("---")
st.sidebar.subheader(" Chart options")
rolling_window  = st.sidebar.selectbox("Rolling window (months)", [1,3,6], index=1)
granularity     = st.sidebar.selectbox("Time aggregation", ["monthly","daily"], index=0)
show_choropleth = st.sidebar.checkbox("Show choropleth", value=True)
lat_global      = st.sidebar.checkbox("Lat–temp: use global data", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader(" Location compare")
countries_no_all = country_opts[1:]
cmp_country_l = st.sidebar.selectbox("Left country",  countries_no_all, key="cl")
cmp_loc_l     = st.sidebar.selectbox("Left location",
    ["None"] + sorted(df[df["country"]==cmp_country_l]["location_name"].unique().tolist()), key="ll")
cmp_country_r = st.sidebar.selectbox("Right country", countries_no_all, key="cr")
cmp_loc_r     = st.sidebar.selectbox("Right location",
    ["None"] + sorted(df[df["country"]==cmp_country_r]["location_name"].unique().tolist()), key="lr")

st.sidebar.markdown("---")
st.sidebar.download_button("Download dataset",
    data=df.to_csv(index=False).encode(), file_name="weather.csv", mime="text/csv")

fdf = apply_filters(df, selected_countries, date_range)

st.title("ClimateScope — Global Weather Dashboard")
st.caption("Visualizing global weather trends, extremes, travel insights, air quality, and outdoor comfort.")

if fdf.empty:
    st.warning("No data for current filters.")
    st.stop()


# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    " Dashboard",
    " Travel Planner",
    " Air Quality",
    " Outdoor Comfort",
])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — MAIN DASHBOARD                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab1:
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric(" Avg Temp (°C)",    f"{fdf['temperature_celsius'].mean():.1f}")
    k2.metric(" Avg Humidity (%)", f"{fdf['humidity'].mean():.1f}")
    k3.metric(" Avg Precip (mm)", f"{fdf['precip_mm'].mean():.2f}")
    k4.metric(" Avg Wind (kph)",   f"{fdf['wind_kph'].mean():.1f}")
    k5.metric(" Avg PM2.5",        f"{fdf['air_quality_pm25'].mean():.1f}")
    st.markdown("---")

    # 1. Country comparison
    st.header("1 · Country Comparison")
    agg_cols = [c for c in ["temperature_celsius","humidity","precip_mm","wind_kph","air_quality_pm25"] if c in fdf.columns]
    country_summary = fdf.groupby("country").agg({c:"mean" for c in agg_cols}).reset_index()
    metric_bar = st.selectbox("Metric", agg_cols, index=0, key="mb")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            country_summary.sort_values(metric_bar, ascending=False).head(20),
            x="country", y=metric_bar, color=metric_bar,
            color_continuous_scale="Reds",
            labels={metric_bar: metric_bar.replace("_"," ").title(), "country":"Country"},
            title=f"Top 20 countries — {metric_bar.replace('_',' ').title()}",
            template=TEMPLATE,
        )
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if show_choropleth:
            try:
                fig_map = px.choropleth(
                    country_summary, locations="country", locationmode="country names",
                    color=metric_bar, color_continuous_scale="Reds",
                    title=f"Choropleth — {metric_bar.replace('_',' ').title()}",
                    template=TEMPLATE,
                )
                fig_map.update_layout(margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception:
                st.info("Choropleth unavailable for some country names.")
        else:
            st.info("Choropleth disabled. Toggle in sidebar.")
    st.markdown("---")

    # 2. Time-series
    st.header("2 · Temperature Trend")
    if granularity == "monthly":
        ts = fdf.groupby(["year","month"]).agg(temperature_celsius=("temperature_celsius","mean")).reset_index()
        ts["period"] = pd.to_datetime(ts["year"].astype(int).astype(str)+"-"+ts["month"].astype(int).astype(str)+"-01")
    else:
        ts = fdf.groupby("date").agg(temperature_celsius=("temperature_celsius","mean")).reset_index()
        ts["period"] = pd.to_datetime(ts["date"])
    ts = ts.sort_values("period")
    ts["smoothed"] = ts["temperature_celsius"].rolling(rolling_window, min_periods=1).mean()
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=ts["period"], y=ts["temperature_celsius"],
        name="Raw", mode="lines", line=dict(color="#b0c4de", width=1)))
    fig_ts.add_trace(go.Scatter(x=ts["period"], y=ts["smoothed"],
        name=f"{rolling_window}-period smoothed", mode="lines", line=dict(color="#e63946", width=2.5)))
    fig_ts.update_layout(title="Temperature over time", xaxis_title="Time",
        yaxis_title="Temperature (°C)", template=TEMPLATE, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_ts, use_container_width=True)
    st.markdown("---")

    # 3. Latitude gradient
    st.header("3 · Latitude → Temperature Gradient")
    lat_df = (df if lat_global else fdf).copy()
    if "latitude" in lat_df.columns:
        lat_bins = st.select_slider("Latitude bin size (°)", options=[10,20,30], value=20)
        bins   = list(np.arange(-90, 90+lat_bins, lat_bins))
        labels = [f"{int(bins[i])}° to {int(bins[i+1])}°" for i in range(len(bins)-1)]
        lat_df["lat_band"] = pd.cut(lat_df["latitude"], bins=bins, labels=labels, include_lowest=True)
        lat_sum = lat_df.groupby("lat_band", observed=True).agg(
            temperature_celsius=("temperature_celsius","mean")).reset_index()
        fig_lat = px.bar(lat_sum, x="lat_band", y="temperature_celsius",
            color="temperature_celsius", color_continuous_scale="RdYlBu_r",
            labels={"lat_band":"Latitude band","temperature_celsius":"Avg Temp (°C)"},
            title="Average temperature by latitude band", template=TEMPLATE)
        fig_lat.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_lat, use_container_width=True)
    st.markdown("---")

    # 4. Scatterplot
    st.header("4 · Correlation Scatterplot")
    scatter_opts = [c for c in ["humidity","feels_like_celsius","precip_mm","wind_kph",
        "pressure_mb","uv_index","air_quality_pm25","cloud","visibility_km"] if c in fdf.columns]
    sc1,sc2,sc3 = st.columns(3)
    x_ax   = sc1.selectbox("X axis",  scatter_opts, index=0, key="sx")
    y_ax   = sc2.selectbox("Y axis",  scatter_opts, index=1, key="sy")
    col_by = sc3.selectbox("Color by",["season","country","none"], index=0, key="sc")
    sample = fdf.sample(min(3000, len(fdf)), random_state=42)
    fig_sc = px.scatter(sample, x=x_ax, y=y_ax,
        color=None if col_by=="none" else col_by, opacity=0.55,
        trendline="ols", trendline_scope="overall",
        labels={x_ax: x_ax.replace("_"," ").title(), y_ax: y_ax.replace("_"," ").title()},
        title=f"{x_ax.replace('_',' ').title()} vs {y_ax.replace('_',' ').title()}",
        template=TEMPLATE, color_discrete_sequence=COLOR_SEQ)
    fig_sc.update_traces(marker=dict(size=4))
    st.plotly_chart(fig_sc, use_container_width=True)
    st.markdown("---")

    # 5. Correlation heatmap
    st.header("5 · Correlation Matrix")
    num_cols = [c for c in fdf.select_dtypes("number").columns if fdf[c].nunique()>1]
    if len(num_cols) >= 2:
        fig_corr = px.imshow(fdf[num_cols].corr(), color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, title="Feature correlation matrix", template=TEMPLATE)
        fig_corr.update_layout(height=550)
        st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("---")

    # 6. Seasonal heatmap
    st.header("6 · Seasonal Temperature Heatmap")
    sp = fdf.pivot_table(values="temperature_celsius", index="season", columns="year", aggfunc="mean")
    sp = sp.dropna(how="all").dropna(axis=1, how="all")
    if not sp.empty:
        fig_s = px.imshow(sp, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlBu_r",
            title="Season × Year — Average Temperature (°C)", template=TEMPLATE)
        st.plotly_chart(fig_s, use_container_width=True)
    st.markdown("---")

    # 7. Extreme events
    st.header("7 · Extreme Events")
    ex1,ex2 = st.columns(2)
    method    = ex1.radio("Detection method", ["Quantile (95th)","Z-score (> 3)"], horizontal=True)
    ex_metric = ex2.selectbox("Metric",
        [c for c in ["temperature_celsius","precip_mm","air_quality_pm25","wind_kph"] if c in fdf.columns])
    if method.startswith("Quantile"):
        thresh   = fdf[ex_metric].quantile(0.95)
        extremes = fdf[fdf[ex_metric] > thresh].copy()
        st.caption(f"95th percentile threshold = {thresh:.2f}  |  {len(extremes):,} extreme records")
    else:
        from scipy import stats
        valid    = fdf[ex_metric].dropna()
        z        = np.abs(stats.zscore(valid))
        extremes = fdf.loc[valid.index[z > 3]].copy()
        st.caption(f"Z-score > 3  |  {len(extremes):,} extreme records")
    if not extremes.empty:
        top_c = extremes["country"].value_counts().reset_index()
        top_c.columns = ["country","count"]
        fig_ex = px.bar(top_c.head(15), x="country", y="count",
            color="count", color_continuous_scale="Oranges",
            labels={"country":"Country","count":"Extreme event count"},
            title=f"Top countries — extreme {ex_metric.replace('_',' ')}", template=TEMPLATE)
        fig_ex.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_ex, use_container_width=True)
        show_cols = [c for c in ["country","location_name","last_updated",ex_metric] if c in extremes.columns]
        st.dataframe(extremes[show_cols].sort_values(ex_metric, ascending=False).head(200),
            use_container_width=True)
    st.markdown("---")

    # 8. Temperature predictor
    st.header("8 · Temperature Predictor")
    st.caption("Random Forest trained on full dataset (R² ≈ 0.96).")
    rf_model, feat_names = train_model(data_path)
    p1, p2 = st.columns(2)
    with p1:
        st.subheader("Input conditions")
        in_hum  = st.slider("Humidity (%)",         int(df["humidity"].min()),      int(df["humidity"].max()),      int(df["humidity"].mean()))
        in_wind = st.slider("Wind speed (kph)",      float(df["wind_kph"].min()),    float(df["wind_kph"].max()),    float(df["wind_kph"].mean()), step=0.5)
        in_pres = st.slider("Pressure (mb)",         float(df["pressure_mb"].min()), float(df["pressure_mb"].max()),float(df["pressure_mb"].mean()), step=0.5)
        in_prec = st.slider("Precipitation (mm)",    0.0, float(df["precip_mm"].max()), 0.0, step=0.1)
        in_uv   = st.slider("UV index",              0.0, float(df["uv_index"].max()), float(df["uv_index"].mean()), step=0.1)
        in_vis  = st.slider("Visibility (km)",       0.0, float(df["visibility_km"].max()), float(df["visibility_km"].mean()), step=0.5)
        in_cld  = st.slider("Cloud cover (%)",       0,   100, int(df["cloud"].mean()))
        pred    = rf_model.predict(pd.DataFrame([dict(zip(feat_names,
            [in_hum, in_wind, in_pres, in_prec, in_uv, in_vis, in_cld]))]))[0]
        st.markdown("---")
        st.metric(" Predicted Temperature", f"{pred:.1f} °C")
    with p2:
        st.subheader("Feature importance")
        imp_df = pd.DataFrame({"Feature":[f.replace("_"," ").title() for f in feat_names],
            "Importance": rf_model.feature_importances_}).sort_values("Importance", ascending=True)
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            title="What drives temperature predictions?", template=TEMPLATE)
        fig_imp.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown("---")

    # 9. Location compare
    st.header("9 · Compare Two Locations")
    def get_loc_df(country, location):
        if not country or not location or location == "None":
            return pd.DataFrame()
        return df[(df["country"]==country)&(df["location_name"]==location)].sort_values("last_updated")
    col_l, col_r = st.columns(2)
    for side_df, country, location, col in [
        (get_loc_df(cmp_country_l, cmp_loc_l), cmp_country_l, cmp_loc_l, col_l),
        (get_loc_df(cmp_country_r, cmp_loc_r), cmp_country_r, cmp_loc_r, col_r),
    ]:
        with col:
            if side_df.empty:
                st.info("Select a country and location from the sidebar.")
            else:
                sm = side_df.set_index("last_updated")[["temperature_celsius"]].rolling(rolling_window).mean()
                fig_loc = px.line(sm.reset_index(), x="last_updated", y="temperature_celsius",
                    labels={"last_updated":"Date","temperature_celsius":"Temp (°C)"},
                    title=f"{location}, {country}", template=TEMPLATE)
                fig_loc.update_traces(line_color="#e63946")
                st.plotly_chart(fig_loc, use_container_width=True)
    st.caption("ClimateScope · Streamlit & Plotly · Data: Kaggle Global Weather Repository")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — TRAVEL PLANNER                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.header(" Travel Planner")
    st.markdown(
        "Composite travel comfort score (0–100) per location and month, weighted across "
        "**temperature** (35%, sweet spot 23°C) · **air quality** (20%) · **precipitation** (20%) · "
        "**UV index** (10%) · **wind** (10%) · **visibility** (5%)."
    )
    st.markdown("---")

    travel = build_travel_scores(data_path)
    tp1, tp2 = st.tabs([" Best Month by Country", " Best Countries by Month"])

    with tp1:
        st.subheader("Which month is best to visit a country?")
        sel_country_tp = st.selectbox("Select country", sorted(travel["country"].unique()), key="tp_country")
        ct = travel[travel["country"] == sel_country_tp]

        if ct.empty:
            st.info("No data for this country.")
        else:
            pivot = ct.pivot_table(values="travel_score", index="location_name",
                columns="month_label", aggfunc="mean")
            month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
            fig_tp = px.imshow(pivot, text_auto=".0f", aspect="auto",
                color_continuous_scale="RdYlGn", range_color=[0, 100],
                title=f"Travel comfort score — {sel_country_tp}",
                labels={"x":"Month","y":"Location","color":"Score"}, template=TEMPLATE)
            fig_tp.update_layout(height=max(300, len(pivot)*60 + 100))
            st.plotly_chart(fig_tp, use_container_width=True)

            best = ct.groupby("month")["travel_score"].mean().reset_index()
            best["month_label"] = best["month"].map(MONTH_LABELS)
            best = best.sort_values("travel_score", ascending=False)
            st.markdown(
                f"**Best month to visit {sel_country_tp}:** "
                f" **{best.iloc[0]['month_label']}** "
                f"(avg score: {best.iloc[0]['travel_score']:.1f}/100)"
            )
            fig_bar_tp = px.bar(best.sort_values("month"), x="month_label", y="travel_score",
                color="travel_score", color_continuous_scale="RdYlGn", range_color=[0, 100],
                labels={"month_label":"Month","travel_score":"Comfort score"},
                title=f"Monthly comfort — {sel_country_tp}", template=TEMPLATE)
            fig_bar_tp.add_hline(y=70, line_dash="dot", line_color="green",
                annotation_text="Good threshold (70)", annotation_position="top right")
            fig_bar_tp.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_bar_tp, use_container_width=True)

    with tp2:
        st.subheader("Which countries should I visit this month?")
        sel_month = st.selectbox("Select month", options=list(MONTH_LABELS.keys()),
            format_func=lambda x: MONTH_LABELS[x], key="tp_month")
        top_n = st.slider("Top N countries", 10, 40, 20, key="tp_n")

        month_travel = travel[travel["month"] == sel_month].groupby("country").agg(
            travel_score=("travel_score","mean"),
            temp=("temp","mean"),
            precip=("precip","mean"),
            uv=("uv","mean"),
        ).reset_index().sort_values("travel_score", ascending=False).head(top_n)

        fig_rank = px.bar(month_travel, x="travel_score", y="country", orientation="h",
            color="travel_score", color_continuous_scale="RdYlGn", range_color=[0, 100],
            labels={"travel_score":"Comfort score","country":"Country"},
            title=f"Top {top_n} destinations — {MONTH_LABELS[sel_month]}", template=TEMPLATE)
        fig_rank.update_layout(coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"})
        fig_rank.add_vline(x=70, line_dash="dot", line_color="green")
        st.plotly_chart(fig_rank, use_container_width=True)

        month_all = travel[travel["month"] == sel_month].groupby("country").agg(
            travel_score=("travel_score","mean")).reset_index()
        try:
            fig_world = px.choropleth(month_all, locations="country", locationmode="country names",
                color="travel_score", color_continuous_scale="RdYlGn", range_color=[0, 100],
                title=f"Global travel comfort — {MONTH_LABELS[sel_month]}", template=TEMPLATE)
            fig_world.update_layout(margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_world, use_container_width=True)
        except Exception:
            pass

        st.subheader("Destination details")
        disp = month_travel.copy()
        disp.columns = ["Country","Comfort Score","Avg Temp (°C)","Avg Precip (mm)","Avg UV"]
        disp = disp.set_index("Country")
        st.dataframe(disp.style.background_gradient(
            subset=["Comfort Score"], cmap="RdYlGn", vmin=0, vmax=100),
            use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — AIR QUALITY DEEP DIVE                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.header(" Air Quality Deep Dive")
    st.markdown(
        "Analysis of all six pollutants with **WHO guideline reference lines**. "
        "Composite pollution index normalizes each pollutant against its WHO guideline "
        "(index > 1.0 = exceeds guidelines)."
    )
    st.markdown("---")

    aq_cols_present = [c for c in AQ_LABELS if c in fdf.columns]
    aq1, aq2 = st.tabs([" Country Rankings", " Trends & Correlations"])

    with aq1:
        st.subheader("Country rankings by pollutant")
        sel_poll = st.selectbox("Select pollutant", options=aq_cols_present,
            format_func=lambda x: AQ_LABELS[x], key="aq_poll")
        top_aq_n = st.slider("Top N countries", 10, 40, 20, key="aq_n")

        aq_country = fdf.groupby("country")[aq_cols_present].mean().reset_index()
        aq_ranked  = aq_country.sort_values(sel_poll, ascending=False).head(top_aq_n)

        fig_aq = px.bar(aq_ranked, x="country", y=sel_poll,
            color=sel_poll, color_continuous_scale="YlOrRd",
            labels={"country":"Country", sel_poll: AQ_LABELS[sel_poll]},
            title=f"Top {top_aq_n} countries — {AQ_LABELS[sel_poll]}", template=TEMPLATE)
        who_val = WHO.get(sel_poll)
        if who_val:
            fig_aq.add_hline(y=who_val, line_dash="dash", line_color="red", line_width=1.5,
                annotation_text=f"WHO guideline ({who_val})", annotation_position="top right")
        fig_aq.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_aq, use_container_width=True)

        try:
            fig_aq_map = px.choropleth(aq_country, locations="country", locationmode="country names",
                color=sel_poll, color_continuous_scale="YlOrRd",
                title=f"Global — {AQ_LABELS[sel_poll]}", template=TEMPLATE)
            fig_aq_map.update_layout(margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_aq_map, use_container_width=True)
        except Exception:
            pass

        st.markdown("---")
        st.subheader("Composite Pollution Index")
        st.caption("Mean of (pollutant / WHO guideline) across all 6 pollutants. Index > 1 = above WHO guidelines.")

        for col in aq_cols_present:
            if col in WHO:
                aq_country[f"{col}_norm"] = aq_country[col] / WHO[col]
        norm_cols = [f"{c}_norm" for c in aq_cols_present if c in WHO]
        aq_country["pollution_index"] = aq_country[norm_cols].mean(axis=1)
        aq_indexed = aq_country.sort_values("pollution_index", ascending=False).head(30)

        fig_idx = px.bar(aq_indexed, x="country", y="pollution_index",
            color="pollution_index", color_continuous_scale="RdYlGn_r",
            labels={"country":"Country","pollution_index":"Pollution index (× WHO guideline)"},
            title="Composite pollution index — top 30 most polluted countries", template=TEMPLATE)
        fig_idx.add_hline(y=1, line_dash="dash", line_color="red",
            annotation_text="WHO baseline (1.0)", annotation_position="top right")
        fig_idx.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_idx, use_container_width=True)

    with aq2:
        st.subheader("Monthly pollutant trend by country")
        sel_poll_trend = st.selectbox("Pollutant", options=aq_cols_present,
            format_func=lambda x: AQ_LABELS[x], key="aq_trend_poll")
        sel_countries_aq = st.multiselect("Countries to compare (max 6)",
            options=sorted(fdf["country"].unique()),
            default=sorted(fdf["country"].unique())[:3], key="aq_countries")

        if sel_countries_aq:
            trend_df = fdf[fdf["country"].isin(sel_countries_aq)].copy()
            trend_df["period"] = pd.to_datetime(
                trend_df["year"].astype(int).astype(str) + "-" +
                trend_df["month"].astype(int).astype(str) + "-01"
            )
            trend_agg = trend_df.groupby(["country","period"])[sel_poll_trend].mean().reset_index()
            fig_trend = px.line(trend_agg, x="period", y=sel_poll_trend, color="country",
                labels={"period":"Month","country":"Country", sel_poll_trend: AQ_LABELS[sel_poll_trend]},
                title=f"{AQ_LABELS[sel_poll_trend]} trend by country",
                template=TEMPLATE, color_discrete_sequence=COLOR_SEQ)
            who_val_t = WHO.get(sel_poll_trend)
            if who_val_t:
                fig_trend.add_hline(y=who_val_t, line_dash="dot", line_color="red",
                    annotation_text="WHO guideline", annotation_position="top right")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Select at least one country.")

        st.markdown("---")
        st.subheader("Inter-pollutant correlation")
        if len(aq_cols_present) >= 2:
            corr_matrix = fdf[aq_cols_present].corr()
            clean_labels = [AQ_LABELS.get(c, c) for c in aq_cols_present]
            corr_matrix.index   = clean_labels
            corr_matrix.columns = clean_labels
            fig_aq_corr = px.imshow(corr_matrix, color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, title="Inter-pollutant correlation matrix", template=TEMPLATE)
            st.plotly_chart(fig_aq_corr, use_container_width=True)

    st.caption("WHO guidelines: PM2.5 5µg/m³ · PM10 15µg/m³ · NO₂ 10µg/m³ · SO₂ 40µg/m³ · O₃ 100µg/m³ · CO 4000µg/m³")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — OUTDOOR COMFORT INDEX                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab4:
    st.header(" Outdoor Comfort Index")
    st.markdown(
        "Composite score (0–100): **heat stress** (40%, based on feels-like temp) · "
        "**UV risk** (25%, WHO bands) · **wind comfort** (20%) · **precipitation risk** (15%).  \n"
        "🟢 **Ideal** ≥80 · 🔵 **Good** 60–79 · 🟡 **Fair** 40–59 · 🟠 **Poor** 20–39 · 🔴 **Harsh** <20"
    )
    st.markdown("---")

    comfort_df  = build_comfort_index(data_path)
    comfort_fdf = apply_filters(comfort_df, selected_countries, date_range)

    if comfort_fdf.empty:
        st.info("No data for current filters.")
    else:
        oc1, oc2 = st.tabs([" Global Overview", " Country & Season Breakdown"])

        with oc1:
            country_comfort = comfort_fdf.groupby("country").agg(
                comfort_score=("comfort_score","mean"),
                heat_s=("heat_s","mean"),
                uv_s=("uv_s","mean"),
                wind_s=("wind_s","mean"),
                precip_s=("precip_s","mean"),
            ).reset_index()

            col_oc1, col_oc2 = st.columns(2)
            with col_oc1:
                top_comfort = country_comfort.sort_values("comfort_score", ascending=False).head(25)
                fig_oc = px.bar(top_comfort, x="comfort_score", y="country", orientation="h",
                    color="comfort_score", color_continuous_scale="RdYlGn", range_color=[0,100],
                    labels={"comfort_score":"Comfort score","country":"Country"},
                    title="Top 25 most comfortable countries (avg)", template=TEMPLATE)
                fig_oc.update_layout(coloraxis_showscale=False,
                    yaxis={"categoryorder":"total ascending"})
                fig_oc.add_vline(x=80, line_dash="dot", line_color="green",
                    annotation_text="Ideal threshold")
                st.plotly_chart(fig_oc, use_container_width=True)
            with col_oc2:
                try:
                    fig_oc_map = px.choropleth(country_comfort, locations="country",
                        locationmode="country names", color="comfort_score",
                        color_continuous_scale="RdYlGn", range_color=[0,100],
                        title="Global outdoor comfort score", template=TEMPLATE)
                    fig_oc_map.update_layout(margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig_oc_map, use_container_width=True)
                except Exception:
                    pass

            st.markdown("---")
            st.subheader("Sub-score breakdown — top 20 countries")
            top20 = country_comfort.sort_values("comfort_score", ascending=False).head(20)
            sub_melt = top20.melt(id_vars="country",
                value_vars=["heat_s","uv_s","wind_s","precip_s"],
                var_name="component", value_name="score")
            sub_melt["component"] = sub_melt["component"].map(
                {"heat_s":"Heat stress","uv_s":"UV risk","wind_s":"Wind comfort","precip_s":"Precip risk"})
            fig_sub = px.bar(sub_melt, x="country", y="score", color="component", barmode="group",
                labels={"country":"Country","score":"Sub-score (0–1)","component":"Component"},
                title="Comfort sub-scores breakdown", template=TEMPLATE,
                color_discrete_sequence=COLOR_SEQ)
            fig_sub.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_sub, use_container_width=True)

        with oc2:
            st.subheader("Comfort classification distribution")
            LABEL_ORDER  = ["Ideal","Good","Fair","Poor","Harsh"]
            LABEL_COLORS = {"Ideal":"#2ecc71","Good":"#3498db","Fair":"#f1c40f",
                            "Poor":"#e67e22","Harsh":"#e74c3c"}

            class_pct = (comfort_fdf.groupby(["country","comfort_label"])
                .size().reset_index(name="count"))
            class_pct["pct"] = (class_pct["count"] /
                class_pct.groupby("country")["count"].transform("sum") * 100)

            top_countries_oc = (comfort_fdf.groupby("country")["comfort_score"]
                .mean().sort_values(ascending=False).head(20).index.tolist())
            fig_class = px.bar(
                class_pct[class_pct["country"].isin(top_countries_oc)],
                x="country", y="pct", color="comfort_label",
                category_orders={"comfort_label": LABEL_ORDER},
                color_discrete_map=LABEL_COLORS,
                labels={"country":"Country","pct":"% of records","comfort_label":"Classification"},
                title="Comfort classification distribution — top 20 countries", template=TEMPLATE)
            fig_class.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_class, use_container_width=True)

            st.markdown("---")
            st.subheader("Seasonal outdoor comfort")
            season_comfort = comfort_fdf.groupby("season")["comfort_score"].mean().reset_index()
            season_comfort["season"] = pd.Categorical(
                season_comfort["season"],
                categories=["Spring","Summer","Autumn","Winter"], ordered=True)
            season_comfort = season_comfort.sort_values("season")
            fig_season_oc = px.bar(season_comfort, x="season", y="comfort_score",
                color="comfort_score", color_continuous_scale="RdYlGn", range_color=[0,100],
                labels={"season":"Season","comfort_score":"Avg comfort score"},
                title="Outdoor comfort by season", template=TEMPLATE)
            fig_season_oc.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_season_oc, use_container_width=True)

            st.markdown("---")
            st.subheader("Country deep dive — monthly comfort trend")
            sel_country_oc = st.selectbox("Select country",
                sorted(comfort_fdf["country"].unique()), key="oc_country")
            country_oc_df = comfort_fdf[comfort_fdf["country"] == sel_country_oc].copy()
            country_oc_df["period"] = pd.to_datetime(
                country_oc_df["year"].astype(int).astype(str) + "-" +
                country_oc_df["month"].astype(int).astype(str) + "-01"
            )
            monthly_oc = country_oc_df.groupby("period")["comfort_score"].mean().reset_index()
            fig_oc_ts = px.line(monthly_oc, x="period", y="comfort_score",
                labels={"period":"Month","comfort_score":"Avg comfort score"},
                title=f"Monthly outdoor comfort — {sel_country_oc}", template=TEMPLATE)
            fig_oc_ts.update_traces(line_color="#2ecc71")
            for threshold, label, color in [(80,"Ideal (80)","green"),(60,"Good (60)","blue"),(40,"Fair (40)","orange")]:
                fig_oc_ts.add_hline(y=threshold, line_dash="dot", line_color=color,
                    annotation_text=label)
            st.plotly_chart(fig_oc_ts, use_container_width=True)

    st.caption("ClimateScope · Streamlit & Plotly · Data: Kaggle Global Weather Repository")