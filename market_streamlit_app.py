import streamlit as st
import os

# Set page config for a wider layout
st.set_page_config(page_title="Market Segmentation", layout="wide")

# Aesthetics
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    h1 { color: #2c3e50; text-align: center; }
    .stDataFrame { background-color: white; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

def main():
    try:
        # Move imports INSIDE try-catch to handle import errors gracefully
        import pandas as pd
        import numpy as np
        import datetime as dt
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        import plotly.express as px
        
        # --- HELPER FUNCTIONS ---
        @st.cache_data
        def load_data_cached():
            import os
            # Use absolute path
            file_path = os.path.join(os.path.dirname(__file__), 'online_retail_II.zip')
            
            if not os.path.exists(file_path):
                return "FILE_NOT_FOUND", file_path

            try:
                # Read CSV
                df = pd.read_csv(file_path, encoding='ISO-8859-1', compression='zip')
                
                # SAMPLE DATA immediately to 5000 rows
                if len(df) > 5000:
                    df = df.sample(n=5000, random_state=42)
                
                return "SUCCESS", df
            except Exception as e:
                return "ERROR", str(e)

        # --- APP LOGIC ---
        st.title("🛒 Market Segmentation & Analysis")
        
        status, result = load_data_cached()
        
        if status == "FILE_NOT_FOUND":
            st.error(f"❌ File not found at: {result}")
            return
        elif status == "ERROR":
            st.error(f"❌ Error loading data: {result}")
            return
        
        df = result

        # --- DATA CLEANING ---
        st.sidebar.header("Data Processing")
        df.dropna(subset=['Customer ID'], inplace=True)
        if 'Invoice' in df.columns: df.rename(columns={'Invoice': 'InvoiceNo'}, inplace=True)
        if 'ï»¿Invoice' in df.columns: df.rename(columns={'ï»¿Invoice': 'InvoiceNo'}, inplace=True)
        df['InvoiceNo'] = df['InvoiceNo'].astype(str)
        df = df[~df['InvoiceNo'].str.contains('C', na=False)]
        df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Total_Price'] = df['Quantity'] * df['Price']

        st.success(f"Data Loaded! Analyzing {len(df)} transactions.")

        # --- RFM ---
        st.subheader("📊 RFM Analysis")
        if df.empty:
            st.warning("No data left after cleaning.")
            return

        ref_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
        rfm = df.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (ref_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'Total_Price': 'sum'
        }).reset_index()
        rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers", rfm['Customer ID'].nunique())
        c2.metric("Avg Recency", f"{rfm['Recency'].mean():.0f} days")
        c3.metric("Avg Monetary", f"${rfm['Monetary'].mean():.0f}")

        # --- CLUSTERING ---
        st.subheader("🤖 Clustering")
        rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)
        
        k = st.sidebar.slider("Clusters (K)", 2, 8, 3)
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        st.write(f"### Segmentation (K={k})")
        
        # 2D Plots only
        row1 = st.columns(2)
        with row1[0]:
            fig = px.scatter(rfm, x='Recency', y='Monetary', color='Cluster', title='Recency vs Monetary', log_x=True, log_y=True)
            st.plotly_chart(fig, use_container_width=True)
        with row1[1]:
            fig = px.scatter(rfm, x='Frequency', y='Monetary', color='Cluster', title='Frequency vs Monetary', log_x=True, log_y=True)
            st.plotly_chart(fig, use_container_width=True)

        # --- MARKET BASKET ---
        st.markdown("---")
        st.title("🛍️ Market Basket")
        
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
        except ImportError:
            st.warning("`mlxtend` library not found. Skipping Market Basket Analysis.")
            return

        country = st.selectbox("Country", df['Country'].unique())
        basket_df = df[df['Country'] == country]
        
        if len(basket_df) > 0:
            # Top 50 Items Only (Super Safe)
            top_items = basket_df['Description'].value_counts().head(50).index
            basket_df = basket_df[basket_df['Description'].isin(top_items)]
            
            basket = (basket_df
                  .groupby(['InvoiceNo', 'Description'])['Quantity']
                  .sum().unstack().fillna(0))
            
            basket_encoded = basket.apply(lambda x: x > 0)
            
            st.write(f"Analyzing top 50 items in {len(basket_encoded)} baskets...")
            
            min_support = st.slider("Min Support", 0.01, 0.5, 0.05)
            frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)
            
            if not frequent_itemsets.empty:
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
                if not rules.empty:
                    # Fix serialization
                    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    st.dataframe(rules[['antecedents', 'consequents', 'lift']].head(10))
                else:
                    st.info("No rules found.")
            else:
                st.info("No frequent items found.")

    except Exception as e:
        st.error(f"❌ CRITICAL ERROR: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
