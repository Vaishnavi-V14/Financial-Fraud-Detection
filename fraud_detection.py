import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

# ----------------------------------
# 1. Generate Fake Transactions
# ----------------------------------
@st.cache_data
def generate_data(n=2000):
    np.random.seed(42)
    df = pd.DataFrame({
        "transaction_id": range(1, n+1),
        "customer_id": np.random.randint(100, 200, size=n),
        "amount": np.random.gamma(2, 50, size=n),
        "location": np.random.choice(["US", "IN", "UK", "SG", "DE", "AU"], size=n),
        "device": np.random.choice(["Mobile", "Web", "ATM"], size=n),
        "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(n)]
    })
    df["is_fraud"] = ((df["amount"] > 200) & (df["location"].isin(["SG","UK","DE"]))).astype(int)
    return df

df = generate_data()

# ----------------------------------
# 2. Train Models
# ----------------------------------
X = df[["amount"]]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_train_scaled)

# ----------------------------------
# 3. Streamlit Layout
# ----------------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Financial Fraud Detection Dashboard")

# Tabs for sections
tab1, tab2, tab3 = st.tabs(["📊 Overview", "⚡ Real-Time Alerts", "🔍 Deep Analysis"])

# ----------------------------------
# Tab 1: Overview
# ----------------------------------
with tab1:
    st.sidebar.header("🔎 Filters")
    amount_range = st.sidebar.slider("Amount Range", 0, int(df["amount"].max()), (0, 500))
    fraud_filter = st.sidebar.radio("Show Transactions", ["All", "Fraud Only", "Non-Fraud"])
    
    filtered_df = df[(df["amount"].between(amount_range[0], amount_range[1]))]
    if fraud_filter == "Fraud Only":
        filtered_df = filtered_df[filtered_df["is_fraud"]==1]
    elif fraud_filter == "Non-Fraud":
        filtered_df = filtered_df[filtered_df["is_fraud"]==0]

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(filtered_df))
    col2.metric("Fraudulent Transactions", int(filtered_df["is_fraud"].sum()))
    col3.metric("Fraud %", f"{(filtered_df['is_fraud'].mean()*100):.2f}%")

    # Charts
    colA, colB = st.columns(2)
    with colA:
        fig1 = px.histogram(filtered_df, x="amount", color=filtered_df["is_fraud"].map({0:"Non-Fraud",1:"Fraud"}),
                            nbins=40, title="Fraud vs Non-Fraud by Amount")
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        fig2 = px.pie(filtered_df, names="location", hole=0.4, color="is_fraud",
                      title="Fraud Distribution by Location")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(filtered_df, x="timestamp", y="amount",
                      color=filtered_df["is_fraud"].map({0:"Non-Fraud",1:"Fraud"}),
                      hover_data=["customer_id","device"])
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------
# Tab 2: Real-Time Alerts
# ----------------------------------
with tab2:
    st.subheader("⚡ Real-Time Fraud Simulation")
    sample = df.sample(20)
    sample_scaled = scaler.transform(sample[["amount"]])
    sample["Fraud_Prob"] = clf.predict_proba(sample_scaled)[:,1]
    sample["IsoForest_Flag"] = iso.predict(sample_scaled)
    sample["IsoForest_Flag"] = sample["IsoForest_Flag"].apply(lambda x: 1 if x==-1 else 0)
    sample["Final_Alert"] = ((sample["Fraud_Prob"] > 0.6) | (sample["IsoForest_Flag"]==1)).astype(int)

    def highlight_alert(val):
        return 'background-color: red; color: white;' if val == 1 else ''
    
    st.dataframe(sample[["transaction_id","amount","location","device","Fraud_Prob","Final_Alert"]]
                 .style.applymap(highlight_alert, subset=["Final_Alert"]))

    st.success("✅ Auto-refresh: Refresh the page to simulate new transactions!")

# ----------------------------------
# Tab 3: Deep Analysis
# ----------------------------------
with tab3:
    st.subheader("📌 Correlation Heatmap")
    corr = df[["amount","is_fraud"]].corr()
    fig4 = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="Blues",
        showscale=True
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("🌍 Fraud Map")
    loc_map = {"US": [37.1,-95.7], "IN":[20.5,78.9], "UK":[55.3,-3.4],
               "SG":[1.3,103.8], "DE":[51.1,10.4], "AU":[-25.3,133.8]}
    map_df = df[df["is_fraud"]==1].copy()
    map_df["lat"] = map_df["location"].map(lambda x: loc_map[x][0])
    map_df["lon"] = map_df["location"].map(lambda x: loc_map[x][1])

    fig5 = px.scatter_geo(map_df, lat="lat", lon="lon",
                          hover_name="location", size="amount", color="amount",
                          projection="natural earth", title="Global Fraud Map")
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("📥 Download Fraud Report")
    csv = map_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV Report", csv, "fraud_report.csv", "text/csv")
