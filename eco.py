# Project 1: EcoPredict – Environmental ML Platform (Advanced)
# ---------------------------------------------------------
# Run: streamlit run app.py
# Focus:
# - Advanced data cleaning
# - Probabilistic ML (uncertainty)
# - Explainable AI
# - Geo-spatial + error visualizations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="EcoPredict – Environmental ML", layout="wide")

st.title("🌍 EcoPredict: Environmental ML Platform")
st.caption("End-to-end ML with uncertainty & explainability")

# -----------------------------
# 1. Data Generation (Mock Realistic Environmental Data)
# -----------------------------
np.random.seed(42)

data = pd.DataFrame({
    "latitude": np.random.uniform(8, 35, 800),
    "longitude": np.random.uniform(68, 97, 800),
    "temperature": np.random.normal(30, 6, 800),
    "humidity": np.random.normal(65, 12, 800),
    "pm25": np.random.normal(90, 20, 800),
    "no2": np.random.normal(40, 10, 800)
})

# Introduce missing values
for col in ["temperature", "humidity"]:
    data.loc[data.sample(frac=0.05).index, col] = np.nan

# Target: Environmental Risk
data["risk"] = ((data["pm25"] > 100) | (data["no2"] > 50)).astype(int)

# -----------------------------
# 2. Data Cleaning
# -----------------------------
st.subheader("🧹 Data Cleaning & Validation")

st.write("Missing values before cleaning:")
st.write(data.isna().sum())

# Imputation
data.fillna(data.median(), inplace=True)

st.success("Missing values handled using median imputation")

# -----------------------------
# 3. Train-Test Split
# -----------------------------
features = ["temperature", "humidity", "pm25", "no2"]
X = data[features]
y = data["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# -----------------------------
# 4. Advanced ML Model (Uncertainty-Aware)
# -----------------------------
st.subheader("🤖 Model Training")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42
)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = (probs > 0.5).astype(int)

auc = roc_auc_score(y_test, probs)
st.metric("ROC-AUC", f"{auc:.3f}")

# -----------------------------
# 5. Uncertainty Estimation
# -----------------------------
st.subheader("📉 Prediction Uncertainty")

all_tree_probs = np.stack([
    tree.predict_proba(X_test)[:, 1] for tree in model.estimators_
])

uncertainty = all_tree_probs.std(axis=0)

unc_df = pd.DataFrame({
    "Prediction Probability": probs,
    "Uncertainty": uncertainty
})

fig_unc = px.scatter(
    unc_df,
    x="Prediction Probability",
    y="Uncertainty",
    title="Prediction Confidence vs Uncertainty"
)

st.plotly_chart(fig_unc, use_container_width=True)

# -----------------------------
# 6. Explainable AI – Permutation Importance
# -----------------------------
st.subheader("🔍 Explainable AI (Permutation Importance)")

perm = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42
)

imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": perm.importances_mean
}).sort_values(by="Importance", ascending=False)

st.plotly_chart(
    px.bar(imp_df, x="Feature", y="Importance"),
    use_container_width=True
)

# -----------------------------
# 7. Geo-Spatial Risk & Error Visualization
# -----------------------------
st.subheader("🗺️ Geo-Spatial Risk & Error Map")

data["risk_prob"] = model.predict_proba(X)[:, 1]
data["error"] = abs(data["risk_prob"] - data["risk"])

fig_map = px.scatter_mapbox(
    data,
    lat="latitude",
    lon="longitude",
    color="risk_prob",
    size="error",
    zoom=3,
    mapbox_style="carto-positron",
    title="Environmental Risk Probability & Error"
)

st.plotly_chart(fig_map, use_container_width=True)

# -----------------------------
# 8. Diagnostics
# -----------------------------
st.subheader("📊 Model Diagnostics")

cm = confusion_matrix(y_test, preds)

fig_cm = go.Figure(data=go.Heatmap(
    z=cm,
    x=["Low Risk", "High Risk"],
    y=["Low Risk", "High Risk"],
    colorscale="Blues"
))

fig_cm.update_layout(title="Confusion Matrix")
st.plotly_chart(fig_cm)

st.success("EcoPredict platform ready for production demo 🚀")
