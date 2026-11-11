import pandas as pd
import plotly.express as px
import streamlit as st

import kagglehub

path = kagglehub.dataset_download("rankirsh/evolution-of-top-games-on-twitch")

print("Path to dataset files:", path)
st.set_page_config(page_title="Twitch Viewership Analytics", layout="wide")
st.title("Twitch Viewership & Streaming Analytics")
st.caption("Top Games on Twitch, 2016-2023")


@st.cache_data
def load_data():
    df = pd.read_csv("Twitch_game_data.csv", encoding="latin-1")
    df.columns = df.columns.str.strip().str.capitalize()
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str))
    return df


df = load_data()
num_cols = ["Hours_watched", "Hours_streamed", "Peak_viewers", "Peak_channels",
            "Streamers", "Avg_viewers", "Avg_channels"]

st.sidebar.header("Filters")
years = st.sidebar.multiselect("Year", sorted(df["Year"].unique()),
                               default=sorted(df["Year"].unique()))
top_n = st.sidebar.slider("Top N games (by hours watched)", 5, 50, 20)
top_games = df.groupby("Game")["Hours_watched"].sum().nlargest(top_n).index
games = st.sidebar.multiselect("Select games (optional)", list(top_games))
min_avg = st.sidebar.slider("Minimum average viewers", 0, int(df["Avg_viewers"].max()), 0, 1000)

data = df[df["Year"].isin(years) & df["Game"].isin(top_games) & (df["Avg_viewers"] >= min_avg)]
if games:
    data = data[data["Game"].isin(games)]

if data.empty:
    st.warning("No records match the filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total records", f"{len(data):,}")
c2.metric("Total hours watched", f"{data['Hours_watched'].sum() / 1e9:.2f} B")
c3.metric("Average hours watched", f"{data['Hours_watched'].mean() / 1e6:.2f} M")
c4.metric("Average viewers", f"{data['Avg_viewers'].mean():,.0f}")

tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Data Analysis", "Visualizations", "Insights"])

with tab1:
    with st.container():
        st.subheader("Filtered data")
        st.dataframe(data.drop(columns=["Date"]).head(100), width="stretch")
    with st.expander("Summary statistics"):
        st.dataframe(data[num_cols].describe().T, width="stretch")

with tab2:
    st.subheader("Category-wise record counts")
    a, b = st.columns(2)
    a.write("Records per year")
    a.dataframe(data.groupby("Year").size().reset_index(name="Records"), hide_index=True)
    b.write("Records per game (top 10)")
    b.dataframe(data.groupby("Game").size().reset_index(name="Records")
                .sort_values("Records", ascending=False).head(10), hide_index=True)

    st.subheader("Sum and average by category")
    cat = st.selectbox("Group by", ["Game", "Year"])
    num = st.selectbox("Numerical column", num_cols)
    out = data.groupby(cat)[num].agg(["sum", "mean"]).reset_index()
    out.columns = [cat, "Sum", "Average"]
    st.dataframe(out.sort_values("Sum", ascending=False), hide_index=True, width="stretch")

with tab3:
    left, right = st.columns(2)

    with left:
        bar_df = data.groupby("Game")["Hours_watched"].sum().nlargest(10).reset_index()
        fig = px.bar(bar_df, x="Game", y="Hours_watched", title="Top 10 Games by Hours Watched",
                     labels={"Hours_watched": "Total Hours Watched"})
        st.plotly_chart(fig, width="stretch")
        st.caption("Interpretation: a few games take a large share of total viewing.")

    with right:
        line_df = data.groupby("Date")["Hours_watched"].sum().reset_index()
        fig = px.line(line_df, x="Date", y="Hours_watched", title="Monthly Hours Watched",
                      labels={"Date": "Month", "Hours_watched": "Hours Watched"})
        st.plotly_chart(fig, width="stretch")
        st.caption("Interpretation: shows how total viewing changes over time.")

    x_var = st.selectbox("X axis", num_cols, index=1)
    y_var = st.selectbox("Y axis", num_cols, index=0)
    fig = px.scatter(data, x=x_var, y=y_var, color="Year", hover_name="Game",
                     title=f"{y_var} vs {x_var}")
    st.plotly_chart(fig, width="stretch")
    st.caption("Interpretation: an upward trend means both rise together; points far "
               "from the cloud are outliers.")

with tab4:
    st.subheader("Key insights (for current filters)")
    st.write("Most watched game:", data.groupby("Game")["Hours_watched"].sum().idxmax())
    st.write("Year with most hours watched:", data.groupby("Year")["Hours_watched"].sum().idxmax())
    st.write("Highest peak viewers:", f"{data['Peak_viewers'].max():,.0f}")
    st.write("Correlation (hours streamed vs hours watched):",
             round(data["Hours_streamed"].corr(data["Hours_watched"]), 2))