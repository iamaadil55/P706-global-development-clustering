"""
P706 - Global Development Clustering
Streamlit app | Group 5

Loads the K-Means model saved in the model-building notebook and lets the
user (1) look up any country's cluster, (2) predict the cluster of a new
country from its indicator values.
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

st.set_page_config(page_title="Global Development Clustering", layout="wide")

COLORS = {0: "#d95f02", 1: "#7570b3", 2: "#1b9e77"}


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    """Load the saved model, scaler and metadata. Cached so it runs once."""
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return kmeans, scaler, meta


@st.cache_data
def load_data():
    """Load the cleaned country table and the saved cluster labels."""
    country = pd.read_csv("country_clean.csv", index_col=0)
    clusters = pd.read_csv("country_clusters_all.csv", index_col=0)
    return country, clusters


kmeans, scaler, meta = load_model()
country, clusters = load_data()

FEATURES = meta["features"]
LOG_COLS = meta["log_cols"]
# JSON turns dictionary keys into text, so convert them back to numbers
NAMES = {int(k): v for k, v in meta["cluster_names"].items()}


# ------------------------------------------------------------- transform
def transform(df):
    """Apply the exact same steps used during training: log, then scale."""
    X = df[FEATURES].copy()
    X[LOG_COLS] = np.log(X[LOG_COLS].clip(lower=1e-6))
    return scaler.transform(X)


@st.cache_data
def pca_coords():
    """2D coordinates of all 208 countries, for the scatter plot."""
    scaled = transform(country)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    return coords, pca, scaled


coords, pca, scaled_all = pca_coords()
labels = clusters.loc[country.index, "Cluster"].values


def scatter(extra_point=None, extra_label=None):
    """Draw the PCA map. extra_point is an optional new country to mark."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c in sorted(NAMES):
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1], s=28, alpha=0.75,
                   color=COLORS[c], label=f"{c} - {NAMES[c]}")
    if extra_point is not None:
        p = pca.transform(extra_point)
        ax.scatter(p[0, 0], p[0, 1], s=320, marker="*", color="black",
                   zorder=5, label=f"Your country ({NAMES[extra_label]})")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("208 countries in 2 dimensions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- page
st.title("Global Development Clustering")
st.caption("ExcelR project P706 | Group 5 | K-Means, k=3, 21 indicators, "
           "208 countries")

tab1, tab2, tab3 = st.tabs(
    ["Overview", "Look up a country", "Predict a new country"]
)

# ---- Tab 1 -------------------------------------------------------------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Countries", len(country))
    c2.metric("Indicators used", len(FEATURES))
    c3.metric("Clusters found", len(NAMES))

    st.subheader("What the three clusters mean")
    sizes = clusters["Cluster"].value_counts().sort_index()
    summary = pd.DataFrame({
        "Cluster": list(NAMES.keys()),
        "Name": list(NAMES.values()),
        "Countries": [sizes.get(c, 0) for c in NAMES],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.subheader("Average profile of each cluster")
    profile_cols = [c for c in
                    ["GDP per Capita", "Life Expectancy Female",
                     "Internet Usage", "Urban Population",
                     "Health Exp per Capita", "Infant Mortality Rate"]
                    if c in country.columns]
    if profile_cols:
        prof = country[profile_cols].groupby(labels).mean().round(2)
        prof.index = [f"{c} - {NAMES[c]}" for c in prof.index]
        st.dataframe(prof, use_container_width=True)

    st.subheader("Cluster map")
    st.pyplot(scatter())
    st.caption("PCA squeezes the 21 indicators into 2 so the clusters can be "
               "drawn. The clustering itself used all 21.")

# ---- Tab 2 -------------------------------------------------------------
with tab2:
    name = st.selectbox("Choose a country", sorted(country.index))
    row = clusters.loc[name]
    c = int(row["Cluster"])

    st.success(f"**{name}** is in cluster **{c} - {NAMES[c]}**")

    c1, c2 = st.columns(2)
    c1.metric("K-Means", f"{c} - {NAMES[c]}")
    if "Cluster_Hier" in row:
        c2.metric("Hierarchical (check)", int(row["Cluster_Hier"]))

    st.subheader("Its indicator values")
    vals = country.loc[name, FEATURES].to_frame("Value").round(3)
    vals["Cluster average"] = (
        country[labels == c][FEATURES].mean().round(3).values
    )
    st.dataframe(vals, use_container_width=True)

# ---- Tab 3 -------------------------------------------------------------
with tab3:
    st.write("Pick a country to fill the boxes, change any values, then "
             "predict.")

    base = st.selectbox("Start from", sorted(country.index), key="base")
    defaults = country.loc[base, FEATURES]

    with st.form("predict_form"):
        inputs = {}
        cols = st.columns(3)
        for i, f in enumerate(FEATURES):
            inputs[f] = cols[i % 3].number_input(
                f, value=float(defaults[f]), format="%.4f", key=f"in_{f}"
            )
        submitted = st.form_submit_button("Predict cluster")

    if submitted:
        new = pd.DataFrame([inputs])[FEATURES]
        X = transform(new)
        pred = int(kmeans.predict(X)[0])

        st.success(f"Predicted cluster: **{pred} - {NAMES[pred]}**")

        dists = kmeans.transform(X)[0]
        st.write("Distance to each cluster centre (smaller means closer):")
        st.dataframe(
            pd.DataFrame({
                "Cluster": [f"{c} - {NAMES[c]}" for c in range(len(dists))],
                "Distance": dists.round(3),
            }),
            hide_index=True, use_container_width=True,
        )

        st.pyplot(scatter(extra_point=X, extra_label=pred))

st.divider()
st.caption("Model: K-Means (k=3) on 21 log-transformed, standardised "
           "indicators. Validated against Agglomerative clustering, "
           "ARI 0.712.")
