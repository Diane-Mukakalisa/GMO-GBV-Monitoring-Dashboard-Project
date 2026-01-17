
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for c in ["date","referred_date","returned_date","referred_date_on_return"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    for c in ["district","sector","cell","village","sex","channel_used","type_of_violence","case_status"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()
    if "date" in df.columns:
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.to_period("M").astype(str)
        df["week"] = df["date"].dt.to_period("W").astype(str)
        df["weekday"] = df["date"].dt.day_name()
    df["reported_flag"] = ~df.get("date", pd.Series([pd.NaT]*len(df))).isna()
    df["referred_flag"] = ~df.get("referred_date", pd.Series([pd.NaT]*len(df))).isna()
    df["returned_flag"] = ~df.get("returned_date", pd.Series([pd.NaT]*len(df))).isna()
    if "sex" in df.columns:
        df["sex"] = df["sex"].str.title().replace({"F":"Female","M":"Male"})
    return df

def kpi_card(label, value, delta=None):
    st.metric(label, value, delta)

st.sidebar.title("Gender Monitoring Office Dashboard")
import base64
from PIL import Image

logo_path = "E:\DSCBI\Project\GMO LOGO.png"

encoded_logo = base64.b64encode(open(logo_path, "rb").read()).decode()
st.sidebar.image("E:\DSCBI\Project\GMO LOGO.png", use_container_width=True)



data_path = "E:\DSCBI\Project\Data\Combined data\GBV_CASE_MANAGEMENT_2022_2025_master_FINAL.csv"
df = load_data(data_path)
if df.empty:
    st.error("No data loaded. Check the CSV path.")
    st.stop()
geojson = None


st.sidebar.markdown("### Filters")
if "date" in df.columns and df["date"].notna().any():
    min_d = df["date"].min()
    max_d = df["date"].max()
    d_from, d_to = st.sidebar.date_input("Case Date Range", value=(min_d.date(), max_d.date()))
else:
    d_from, d_to = None, None

district = st.sidebar.multiselect("District", sorted([x for x in df.get("district", pd.Series()).dropna().unique() if x]))
sector   = st.sidebar.multiselect("Sector", sorted([x for x in df.get("sector", pd.Series()).dropna().unique() if x]))
cell     = st.sidebar.multiselect("Cell", sorted([x for x in df.get("cell", pd.Series()).dropna().unique() if x]))
village  = st.sidebar.multiselect("Village", sorted([x for x in df.get("village", pd.Series()).dropna().unique() if x]))
sex      = st.sidebar.multiselect("Sex", sorted([x for x in df.get("sex", pd.Series()).dropna().unique() if x]))
channel  = st.sidebar.multiselect("Channel Used", sorted([x for x in df.get("channel_used", pd.Series()).dropna().unique() if x]))
violence = st.sidebar.multiselect("Type of Violence", sorted([x for x in df.get("type_of_violence", pd.Series()).dropna().unique() if x]))

if "age" in df.columns and df["age"].notna().any():
    age_min = int(df["age"].min(skipna=True)) if pd.notna(df["age"].min(skipna=True)) else 0
    age_max = int(df["age"].max(skipna=True)) if pd.notna(df["age"].max(skipna=True)) else 100
else:
    age_min, age_max = 0, 100
age_range = st.sidebar.slider("Age Range", min_value=0, max_value=max(100, age_max), value=(age_min, age_max))

mask = pd.Series(True, index=df.index)
if d_from and d_to and "date" in df.columns:
    date_ok = df["date"].isna() | (
    (df["date"] >= pd.to_datetime(d_from)) &
    (df["date"] <= pd.to_datetime(d_to))
)
mask &= date_ok

if district: mask &= df["district"].isin(district)
if sector:   mask &= df["sector"].isin(sector)
if cell:     mask &= df["cell"].isin(cell)
if village:  mask &= df["village"].isin(village)
if sex:      mask &= df["sex"].isin(sex)
if channel:  mask &= df["channel_used"].isin(channel)
if violence: mask &= df["type_of_violence"].isin(violence)
if "age" in df.columns:
    mask &= df["age"].fillna(-1).between(age_range[0], age_range[1], inclusive="both")

fdf = df.loc[mask].copy()

st.title("Interactive Gender Based Violence Monitoring Dashboard")
st.markdown(
    """
    <h3 style="text-align:center; font-size:22px; font-weight:500; margin-top:-10px;">
        Overview of GBV cases by time, location, type, and reporting channel.
    </h3>
    """,
    unsafe_allow_html=True
)

# ----------------- KPI ROW (NEW) -----------------

k1, k2, k3, k4 = st.columns(4)

# Total Cases
with k1:
    total_cases = len(fdf)
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Total Cases</div>
            <div class='kpi-value'>{total_cases:,}</div>
        </div>
    """, unsafe_allow_html=True)

# Top District by Case Count
with k2:
    if "district" in fdf.columns:
        top_dist = (
            fdf["district"]
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .idxmax()
        )
        top_dist_count = fdf["district"].value_counts().max()
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Top District</div>
                <div class='kpi-value'>{top_dist} ({top_dist_count})</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='kpi-card'>
                <div class='kpi-title'>Top District</div>
                <div class='kpi-value'>—</div>
            </div>
        """, unsafe_allow_html=True)

# Top Type of Violence
with k3:
    if "type_of_violence" in fdf.columns:
        top_viol = (
            fdf["type_of_violence"]
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .idxmax()
        )
        top_viol_count = fdf["type_of_violence"].value_counts().max()
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Top Violence Type</div>
                <div class='kpi-value'>{top_viol} ({top_viol_count})</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='kpi-card'>
                <div class='kpi-title'>Top Violence Type</div>
                <div class='kpi-value'>—</div>
            </div>
        """, unsafe_allow_html=True)

# Year Range (2022–2025)
with k4:
    if "date" in fdf.columns and fdf["date"].notna().any():
        start_year = int(fdf["date"].dt.year.min())
        end_year = int(fdf["date"].dt.year.max())
        st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Year Range</div>
                <div class='kpi-value'>{start_year} - {end_year}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='kpi-card'>
                <div class='kpi-title'>Year Range</div>
                <div class='kpi-value'>2022 → 2025</div>
            </div>
        """, unsafe_allow_html=True)


st.markdown("---")

if "date" in fdf.columns:
    ts = fdf.set_index("date").resample("W")["id"].count().rename("Cases").reset_index()
    fig = px.line(ts, x="date", y="Cases", title="GBV Cases Over Time")
    st.plotly_chart(fig, use_container_width=True)


st.subheader("Geographical Distribution")
import pandas as pd
import plotly.express as px
import streamlit as st

fdf["district"] = fdf["district"].astype(str).str.strip().str.title()
fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce")
fdf["month"] = fdf["date"].dt.to_period("M").astype(str)

# ================================
# CASES BY DISTRICT (BAR CHART)
# ================================
st.subheader("📊 GBV Cases by District")

cases_by_district = (
    fdf.groupby("district")["id"]
       .count()
       .reset_index(name="cases")
       .sort_values("cases", ascending=False)
)

st.bar_chart(
    cases_by_district.set_index("district")
)



all_districts = [
    "Bugesera", "Gatsibo", "Kayonza", "Kirehe", "Ngoma", "Rwamagana", "Nyagatare",
    "Gicumbi", "Rulindo", "Burera", "Gakenke", "Musanze",
    "Rubavu", "Nyabihu", "Rutsiro", "Ngororero", "Karongi", "Nyamasheke",
    "Rusizi", "Nyaruguru", "Huye", "Nyanza", "Gisagara", "Nyamagabe",
    "Muhanga", "Kamonyi",
    "Kicukiro", "Gasabo", "Nyarugenge"
]
all_months = sorted(fdf["month"].unique())

heat = (
        fdf.groupby(["district", "month"])["id"]
        .count()
        .reset_index()
        .pivot(index="district", columns="month", values="id")
        .reindex(index=all_districts, columns=all_months)
        .fillna(0)
    )
    # ================================
    # HEATMAP
    # ================================
fig = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale=[
            [0.0, "#fff5f5"],
            [0.2, "#ffcccc"],
            [0.4, "#ff6666"],
            [0.6, "#ff0000"],
            [1.0, "#800000"]
        ],
        labels={"x": "Month", "y": "District", "color": "GBV Cases"},
        title="🔥 GBV Case Intensity Heatmap (Red = High Cases)"
    )

fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Month",
        yaxis_title="District",
        coloraxis_colorbar=dict(
            title="Cases",
            thickness=15
        )
    )

fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Month: %{x}<br>Cases: %{z}<extra></extra>"
    )

st.plotly_chart(fig, use_container_width=True)

st.subheader("Disaggregations")
col_a, col_b = st.columns(2)
with col_a:
    if "sex" in fdf.columns:
        sex_counts = fdf["sex"].replace("", np.nan).dropna().value_counts().reset_index()
        sex_counts.columns = ["Sex","Cases"]
        st.plotly_chart(px.pie(sex_counts, names="Sex", values="Cases", title="Cases by Sex"), use_container_width=True)
with col_b:
    if "age" in fdf.columns and fdf["age"].notna().any():
        st.plotly_chart(px.histogram(fdf, x="age", nbins=20, title="Age Distribution"), use_container_width=True)

st.subheader("Types of Violence")
if "type_of_violence" in fdf.columns:
    tov = (fdf["type_of_violence"].replace("", np.nan).dropna().value_counts().reset_index())
    tov.columns = ["Type of Violence","Cases"]
    st.plotly_chart(px.bar(tov.head(20), x="Cases", y="Type of Violence", orientation="h", title="Top Types of Violence"), use_container_width=True)

st.subheader("Cases by Channel Used")

if "channel_used" in fdf.columns:

        # Clean empty values
        chan = fdf["channel_used"].replace("", np.nan).dropna()

        if chan.empty:
            st.info("No channel data available.")
        else:
            chan_counts = chan.value_counts().reset_index()
            chan_counts.columns = ["Channel", "Cases"]
         # Pie chart – Channel share
            st.plotly_chart(
                px.pie(chan_counts, names="Channel", values="Cases",
                       title="Share of Cases by Reporting Channel"),
                use_container_width=True
            )

else:
        st.info("Column 'channel_used' not found.")

st.subheader("Case Progress Tracking")
k1, k2, k3 = st.columns(3)
progress = pd.DataFrame({
    "Stage": ["Reported", "Referred", "Returned"],
    "Count": [len(fdf),
              fdf["where_case_referred"].replace("", np.nan).notna().sum(),
              fdf["returned_date"].notna().sum()]
})
st.plotly_chart(px.bar(progress, x="Stage", y="Count", title="Case Follow-up"), use_container_width=True)


