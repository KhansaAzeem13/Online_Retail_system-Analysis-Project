import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
st.title("🛒 Online Retail Business - Customer Segmentation App")

# -----------------------------
# Load and Preprocess Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("OnlineRetail.csv", encoding='ISO-8859-1')
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    return df

df = load_data()

# -----------------------------
# Introduction
# -----------------------------
st.header("📌 Project Introduction")
st.markdown("""
This project analyzes online retail transaction data to segment customers using RFM (Recency, Frequency, Monetary) analysis and K-Means Clustering.

**Goals:**
- Explore customer purchase behavior
- Segment customers into actionable groups
- Provide insights for business strategies
""")

# -----------------------------
# EDA Section
# -----------------------------
st.header("📊 Exploratory Data Analysis")

if st.checkbox("Show Raw Data"):
    st.dataframe(df.head())

# Filter by Country
country_list = df['Country'].unique()
selected_country = st.selectbox("Filter by Country", sorted(country_list))
df_country = df[df['Country'] == selected_country]

# Filter by Date
date_range = st.date_input("Select Invoice Date Range", [df['InvoiceDate'].min(), df['InvoiceDate'].max()])
df_country = df_country[(df_country['InvoiceDate'] >= pd.to_datetime(date_range[0])) &
                        (df_country['InvoiceDate'] <= pd.to_datetime(date_range[1]))]

# Revenue Trend
st.subheader("📈 Revenue Over Time")
daily_revenue = df_country.groupby(df_country['InvoiceDate'].dt.date)['TotalPrice'].sum()
st.line_chart(daily_revenue)

# -----------------------------
# Model Section - PCA + KMeans
# -----------------------------
st.header("🤖 Customer Segmentation")

reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

scaler = MinMaxScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Apply PCA
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

# KMeans on PCA components
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_pca)

score = silhouette_score(rfm_pca, rfm['Cluster'])
st.markdown(f"**Silhouette Score:** `{score:.2f}`")

rfm['PCA1'] = rfm_pca[:, 0]
rfm['PCA2'] = rfm_pca[:, 1]

# -----------------------------
# Runtime Prediction
# -----------------------------
st.subheader("🧮 Predict Cluster for a New Customer")
recency = st.slider("Recency (days)", 0, 365, 30)
frequency = st.slider("Frequency (number of orders)", 1, 100, 10)
monetary = st.slider("Monetary (total spent)", 0, 10000, 500)

input_scaled = scaler.transform([[recency, frequency, monetary]])
input_pca = pca.transform(input_scaled)
pred_cluster = kmeans.predict(input_pca)[0]
st.success(f"Predicted Cluster: **{pred_cluster}**")

# -----------------------------
# Visualize Current Input on Cluster Plot
# -----------------------------
st.subheader("📌 Your Input on Cluster Map")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100, alpha=0.6)
plt.scatter(input_pca[0, 0], input_pca[0, 1], color='red', s=200, label='Your Input', edgecolor='black')
plt.legend()
plt.title("Customer Clusters (PCA) with Your Input")
st.pyplot(fig)

# -----------------------------
# Conclusion
# -----------------------------
st.header("📌 Conclusion")
st.markdown("""
- We segmented customers using RFM analysis and K-Means.
- PCA transformation ensures all three features influence clustering.
- Now your input is shown on the cluster plot.

✅ Project Complete!
""")
