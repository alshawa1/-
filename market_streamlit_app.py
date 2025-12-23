import streamlit as st
import os

# Set page config for a wider layout
st.set_page_config(page_title="Market Analysis", layout="wide")

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
        # Move imports INSIDE try-catch
        import pandas as pd
        import numpy as np
        import datetime as dt
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        import plotly.express as px
        import matplotlib.pyplot as plt # Needed for Elbow
        
        # --- HELPER FUNCTIONS ---
        @st.cache_data
        def load_data_cached():
            import os
            file_path = os.path.join(os.path.dirname(__file__), 'online_retail_II.zip')
            
            if not os.path.exists(file_path):
                return "FILE_NOT_FOUND", file_path

            try:
                # Read CSV
                df = pd.read_csv(file_path, encoding='ISO-8859-1', compression='zip')
                
                # SAMPLE DATA
                if len(df) > 5000:
                    df = df.sample(n=5000, random_state=42)
                
                return "SUCCESS", df
            except Exception as e:
                return "ERROR", str(e)

        # --- APP HEADER ---
        st.title("🛒 Retail Analytics Dashboard")
        
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
        if df is not None and not df.empty:
            df.dropna(subset=['Customer ID'], inplace=True)
            if 'Invoice' in df.columns: df.rename(columns={'Invoice': 'InvoiceNo'}, inplace=True)
            if 'ï»¿Invoice' in df.columns: df.rename(columns={'ï»¿Invoice': 'InvoiceNo'}, inplace=True)
            df['InvoiceNo'] = df['InvoiceNo'].astype(str)
            df = df[~df['InvoiceNo'].str.contains('C', na=False)]
            df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['Total_Price'] = df['Quantity'] * df['Price']

            st.success(f"Dataloaded: {len(df)} transactions.")
        else:
            st.error("Dataframe is empty.")
            return

        # ==========================================
        # 1. MARKET BASKET ANALYSIS (FIRST)
        # ==========================================
        st.header("🛍️ Market Basket Analysis")
        st.caption("Discover which products are frequently bought together.")
        
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
        except ImportError:
            st.warning("`mlxtend` library not found. Skipping Market Basket.")
            
        country = st.selectbox("Select Country", df['Country'].unique())
        basket_df = df[df['Country'] == country]
        
        if len(basket_df) > 0:
            # Top 50 Items Only for Speed
            top_items = basket_df['Description'].value_counts().head(50).index
            basket_df = basket_df[basket_df['Description'].isin(top_items)]
            
            basket = (basket_df
                  .groupby(['InvoiceNo', 'Description'])['Quantity']
                  .sum().unstack().fillna(0))
            
            basket_encoded = basket.apply(lambda x: x > 0)
            
            st.write(f"Analyzing top 50 items in {len(basket_encoded)} transactions for {country}...")
            
            min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05)
            frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)
            
            if not frequent_itemsets.empty:
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
                if not rules.empty:
                    # Fix serialization
                    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    
                    st.write("### Top Association Rules")
                    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(10))
                else:
                    st.info("No rules found. Try lowering support.")
            else:
                st.info("No frequent itemsets found.")
        else:
            st.warning("No data for this country.")

        st.markdown("---")

        # ==========================================
        # 2. CUSTOMER SEGMENTATION (SECOND)
        # ==========================================
        st.header("👥 Customer Segmentation (RFM & K-Means)")
        
        # --- RFM CALCULATION ---
        if df.empty:
            st.warning("No data.")
            return

        ref_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
        rfm = df.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (ref_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'Total_Price': 'sum'
        }).reset_index()
        rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", rfm['Customer ID'].nunique())
        c2.metric("Avg Recency", f"{rfm['Recency'].mean():.0f} days")
        c3.metric("Avg Monetary", f"${rfm['Monetary'].mean():.0f}")

        # --- CLUSTERING ---
        st.subheader("K-Means Clustering")
        
        rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)
        
        # ELBOW PLOT (Requested)
        if st.checkbox("Show Elbow Plot (Determine Optimal K)", value=True):
            wcss = []
            for i in range(1, 11):
                kmeans_temp = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans_temp.fit(rfm_scaled)
                wcss.append(kmeans_temp.inertia_)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(range(1, 11), wcss, marker='o')
            ax.set_title('Elbow Method')
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel('WCSS')
            st.pyplot(fig)
        
        k = st.slider("Select Number of Clusters (K)", 2, 8, 3)
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        st.write(f"### Segmentation Results (K={k})")
        
        # 2D Plots
        row1 = st.columns(2)
        with row1[0]:
            fig = px.scatter(rfm, x='Recency', y='Monetary', color='Cluster', title='Recency vs Monetary', log_x=True, log_y=True)
            st.plotly_chart(fig, use_container_width=True)
        with row1[1]:
            fig = px.scatter(rfm, x='Frequency', y='Monetary', color='Cluster', title='Frequency vs Monetary', log_x=True, log_y=True)
            st.plotly_chart(fig, use_container_width=True)

        # --- CUSTOMER LOOKUP ---
        st.subheader("🔎 Find Customer Segment")
        
        # Valid IDs for UX
        valid_ids = rfm['Customer ID'].unique()
        import random
        # Handle case where sample is small
        if len(valid_ids) > 0:
            random_examples = sorted(random.sample(list(valid_ids), min(5, len(valid_ids))))
            st.write(f"ℹ️ **Example valid IDs:** {', '.join(map(str, map(int, random_examples)))}")
        
        customer_id_input = st.text_input("Enter Customer ID:")
        
        if customer_id_input:
            rfm['CID_Str'] = rfm['Customer ID'].astype(str).str.split('.').str[0]
            search_id = str(customer_id_input).strip()
            customer_data = rfm[rfm['CID_Str'] == search_id]
            
            if not customer_data.empty:
                c_cluster = customer_data['Cluster'].values[0]
                cluster_avg = rfm[rfm['Cluster'] == c_cluster][['Recency', 'Frequency', 'Monetary']].mean()
                
                st.success(f"Customer **{search_id}** is in **Cluster {c_cluster}**")
                st.write("**Cluster Profile:**")
                st.dataframe(cluster_avg.to_frame().T)
            else:
                st.error("Customer ID not found in this sample.")

    except Exception as e:
        st.error(f"❌ CRITICAL ERROR: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
