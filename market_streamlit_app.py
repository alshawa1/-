import streamlit as st
import os

# Set page config
st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide")

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
        import matplotlib.pyplot as plt 
        
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
                
                # SAMPLE DATA INTELLIGENTLY (By Invoice)
                unique_invoices = df['Invoice'].unique() if 'Invoice' in df.columns else df['InvoiceNo'].unique() if 'InvoiceNo' in df.columns else df.iloc[:,0].unique()
                
                if len(unique_invoices) > 2000:
                    sampled_invoices = np.random.choice(unique_invoices, 2000, replace=False)
                    if 'Invoice' in df.columns:
                        df = df[df['Invoice'].isin(sampled_invoices)]
                    elif 'InvoiceNo' in df.columns:
                        df = df[df['InvoiceNo'].isin(sampled_invoices)]
                
                return "SUCCESS", df
            except Exception as e:
                return "ERROR", str(e)

        # --- SIDEBAR NAV ---
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to:", ["Market Basket Analysis", "Customer Segmentation"])

        # --- DATA LOADING (Common) ---
        status, result = load_data_cached()
        
        if status == "FILE_NOT_FOUND":
            st.error(f"❌ File not found at: {result}")
            return
        elif status == "ERROR":
            st.error(f"❌ Error loading data: {result}")
            return
        
        df = result

        # --- DATA CLEANING (Common) ---
        if df is not None and not df.empty:
            df.dropna(subset=['Customer ID'], inplace=True)
            if 'Invoice' in df.columns: df.rename(columns={'Invoice': 'InvoiceNo'}, inplace=True)
            if 'ï»¿Invoice' in df.columns: df.rename(columns={'ï»¿Invoice': 'InvoiceNo'}, inplace=True)
            df['InvoiceNo'] = df['InvoiceNo'].astype(str)
            df = df[~df['InvoiceNo'].str.contains('C', na=False)]
            df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['Total_Price'] = df['Quantity'] * df['Price']
        else:
            st.error("Dataframe is empty.")
            return

        # ==========================================
        # PAGE 1: MARKET BASKET ANALYSIS
        # ==========================================
        if page == "Market Basket Analysis":
            st.title("🛍️ Market Basket Analysis")
            st.caption("Discover which products are frequently bought together.")
            st.success(f"Analyzing {len(df)} transactions.")
            
            try:
                from mlxtend.frequent_patterns import apriori, association_rules
            except ImportError:
                st.warning("`mlxtend` library not found. Skipping Market Basket.")
                return
                
            country = st.selectbox("Select Country", df['Country'].unique())
            basket_df = df[df['Country'] == country]
            
            if len(basket_df) > 0:
                # Top 50 Items Only
                top_items = basket_df['Description'].value_counts().head(50).index
                basket_df = basket_df[basket_df['Description'].isin(top_items)]
                
                basket = (basket_df
                      .groupby(['InvoiceNo', 'Description'])['Quantity']
                      .sum().unstack().fillna(0))
                
                basket_encoded = basket.apply(lambda x: x > 0)
                
                st.write(f"Analyzing top 50 items in {len(basket_encoded)} transactions for {country}...")
                
                min_support = st.slider("Minimum Support", 0.01, 0.5, 0.01)
                frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)
                
                if not frequent_itemsets.empty:
                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
                    if not rules.empty:
                        # --- PRODUCT RECOMMENDER ---
                        st.subheader("💡 Product Recommender")
                        st.caption(f"Select a product to see best related items.")
                        
                        rules_logic = rules.copy()
                        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                        
                        all_items = set()
                        for itemset in rules_logic['antecedents']:
                            all_items.update(itemset)
                        
                        sorted_items = sorted(list(all_items))
                        
                        if sorted_items:
                            selected_product = st.selectbox("I am buying...", sorted_items)
                            
                            recommendations = rules_logic[rules_logic['antecedents'].apply(lambda x: selected_product in x)]
                            
                            if not recommendations.empty:
                                top_rec = recommendations.sort_values(by='confidence', ascending=False).iloc[0]
                                rec_prod = ', '.join(list(top_rec['consequents']))
                                conf = top_rec['confidence']
                                lift = top_rec['lift']
                                
                                st.success(f"🚀 Best match: **{rec_prod}**")
                                st.metric("Confidence", f"{conf:.1%}", f"Lift: {lift:.2f}")
                                
                                with st.expander("See all recommendations"):
                                    recs_display = recommendations.copy()
                                    recs_display['consequents'] = recs_display['consequents'].apply(lambda x: ', '.join(list(x)))
                                    st.dataframe(recs_display[['consequents', 'confidence', 'lift']].sort_values(by='confidence', ascending=False))
                            else:
                                st.warning(f"No strong rules found starting with {selected_product}.")
                        
                        st.markdown("---")
                        st.write("### All Association Rules")
                        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(10))
                    else:
                        st.info("No rules found. Try lowering support.")
                else:
                    st.info("No frequent itemsets found.")
            else:
                st.warning("No data for this country.")

        # ==========================================
        # PAGE 2: CUSTOMER SEGMENTATION
        # ==========================================
        elif page == "Customer Segmentation":
            st.title("👥 Customer Segmentation (RFM)")
            st.success(f"Analyzing {len(df)} transactions.")
            
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
            
            # ELBOW PLOT
            if st.checkbox("Show Elbow Plot", value=True):
                wcss = []
                for i in range(1, 11):
                    kmeans_temp = KMeans(n_clusters=i, init='k-means++', random_state=42)
                    kmeans_temp.fit(rfm_scaled)
                    wcss.append(kmeans_temp.inertia_)
                
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(range(1, 11), wcss, marker='o')
                ax.set_title('Elbow Method')
                st.pyplot(fig)
                st.caption("ℹ️ The Elbow Method shows the optimal K is where the curve bends (around 3).")
            
            # Fixed K=3
            k = 3
            st.info(f"👉 Using **{k} Clusters** (Gold, Silver, Bronze).")
            
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
            
            # --- MAP CLUSTERS TO LABELS (Gold, Silver, Bronze) ---
            # Sort clusters by average Monetary value to ensure consistent labeling
            cluster_centers = rfm.groupby('Cluster')['Monetary'].mean().sort_values()
            
            # Create a mapping: lowest -> Bronze, middle -> Silver, highest -> Gold
            cluster_mapping = {}
            labels = ["🥉 Bronze", "🥈 Silver", "🥇 Gold"]
            
            for i, cluster_id in enumerate(cluster_centers.index):
                cluster_mapping[cluster_id] = labels[i]
            
            rfm['Cluster Label'] = rfm['Cluster'].map(cluster_mapping)
            
            st.write(f"### Segmentation Results")
            
            # Explain Logic
            with st.expander("ℹ️ Cluster Interpretation Guide"):
                st.markdown("""
                **Based on RFM (Recency, Frequency, Monetary):**
                - 🥇 **Gold:** Recent purchase, High Frequency, High Spender.
                - 🥈 **Silver:** Average behavior, potential to become Gold.
                - 🥉 **Bronze:** Inactive for a while, Low Frequency, Low Spender.
                """)
            
            # Summary Table
            avg_df = rfm.groupby('Cluster Label')[['Recency', 'Frequency', 'Monetary', 'Customer ID']].agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'Customer ID': 'count'
            }).sort_values(by='Monetary', ascending=False) # Gold on top
            
            st.dataframe(avg_df.style.background_gradient(cmap='Greens', subset=['Frequency', 'Monetary']).background_gradient(cmap='Reds_r', subset=['Recency']))
            
            # 2D Plots with Labels and Explanations
            row1 = st.columns(2)
            
            # Define specific colors
            color_map = {
                "🥇 Gold": "gold",
                "🥈 Silver": "silver",
                "🥉 Bronze": "brown"
            }
            
            with row1[0]:
                st.subheader("Recency vs Monetary")
                fig = px.scatter(rfm, x='Recency', y='Monetary', color='Cluster Label', 
                                 title='Recency vs Monetary', log_x=True, log_y=True,
                                 color_discrete_map=color_map,
                                 category_orders={"Cluster Label": ["🥇 Gold", "🥈 Silver", "🥉 Bronze"]})
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **Interpretation:** **Gold** customers are in the *top-left* (High Money, Low Recency days). **Bronze** customers are *bottom-right* (Low Money, High Recency).")

            with row1[1]:
                st.subheader("Frequency vs Monetary")
                fig = px.scatter(rfm, x='Frequency', y='Monetary', color='Cluster Label', 
                                 title='Frequency vs Monetary', log_x=True, log_y=True,
                                 color_discrete_map=color_map,
                                 category_orders={"Cluster Label": ["🥇 Gold", "🥈 Silver", "🥉 Bronze"]})
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **Interpretation:** **Gold** customers are in the *top-right* (High Frequency, High Money). They buy often and spend a lot.")

            # --- CUSTOMER LOOKUP ---
            st.subheader("🔎 Find Customer Segment")
            
            valid_ids = rfm['Customer ID'].unique()
            import random
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
                    st.dataframe(cluster_avg.to_frame().T)
                else:
                    st.error("Customer ID not found in this sample.")

    except Exception as e:
        st.error(f"❌ CRITICAL ERROR: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
