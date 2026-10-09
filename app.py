import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config( 
    page_title="Aura Organics | Brand Strategy Intelligence Platform",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .recommendation-box {
        background-color: #e6f4ea;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #34a853;
        color: #222222;
    }
    .risk-box {
        background-color: #fce8e6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ea4335;
        color: #222222;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA & COMPUTATION CACHING
# ==========================================
@st.cache_data
def load_and_process_data():
    np.random.seed(42)
    N_CUSTOMERS, N_TRANSACTIONS, N_SURVEYS, N_CAMPAIGNS = 10000, 45000, 1200, 100

    # 1. Customers
    customer_ids = [f"CUST_{i+1:05d}" for i in range(N_CUSTOMERS)]
    segments = ["Eco-Conscious Gen Z", "Busy Urban Professionals", "Value-Seeking Families", "Skin-Care Enthusiasts"]
    cities = ["Mumbai", "Delhi NCR", "Bengaluru", "Hyderabad", "Tier-2 Clusters"]
    acq_channels = ["Meta Ads", "Google Search", "Quick-Comm Sponsored", "Organic / Referral"]

    start_date = datetime(2024, 1, 1)
    df_customers = pd.DataFrame({
        "customer_id": customer_ids,
        "segment": np.random.choice(segments, size=N_CUSTOMERS, p=[0.25, 0.35, 0.25, 0.15]),
        "city": np.random.choice(cities, size=N_CUSTOMERS, p=[0.30, 0.25, 0.20, 0.10, 0.15]),
        "acquisition_channel": np.random.choice(acq_channels, size=N_CUSTOMERS, p=[0.35, 0.25, 0.20, 0.20]),
        "signup_date": [start_date + timedelta(days=int(x)) for x in np.random.randint(0, 730, size=N_CUSTOMERS)]
    })

    # 2. Transactions
    categories = ["Haircare", "Bodycare", "Skincare", "Eco-Bundles"]
    price_map = {"Haircare": (450, 80), "Bodycare": (380, 50), "Skincare": (750, 150), "Eco-Bundles": (1200, 200)}
    trans_categories = np.random.choice(categories, size=N_TRANSACTIONS, p=[0.35, 0.30, 0.25, 0.10])
    
    order_values = [max(round(np.random.normal(price_map[c][0], price_map[c][1]), 2), 150.0) for c in trans_categories]
    cust_signup_map = df_customers.set_index("customer_id")["signup_date"].to_dict()
    trans_cust_ids = np.random.choice(customer_ids, size=N_TRANSACTIONS)
    
    trans_dates = [
        cust_signup_map[c] + timedelta(days=int(np.random.randint(0, max((datetime(2026, 1, 1) - cust_signup_map[c]).days, 1))))
        for c in trans_cust_ids
    ]

    df_transactions = pd.DataFrame({
        "transaction_id": [f"TXN_{i+1:06d}" for i in range(N_TRANSACTIONS)],
        "customer_id": trans_cust_ids,
        "transaction_date": trans_dates,
        "channel": np.random.choice(["D2C Website", "Blinkit", "Zepto", "Instamart", "Modern Trade"], size=N_TRANSACTIONS, p=[0.20, 0.30, 0.25, 0.15, 0.10]),
        "category": trans_categories,
        "order_value_inr": order_values
    })

    # 3. Marketing Campaigns
    campaign_channels = ["Meta Ads", "Google Search", "Quick-Comm Sponsored", "TV / Print Regional"]
    campaigns = []
    for i in range(N_CAMPAIGNS):
        ch = np.random.choice(campaign_channels, p=[0.40, 0.25, 0.25, 0.10])
        sp = np.random.uniform(500000, 2000000) if ch == "TV / Print Regional" else (np.random.uniform(50000, 300000) if ch == "Quick-Comm Sponsored" else np.random.uniform(20000, 150000))
        roas = np.random.uniform(1.2, 2.2) if ch == "TV / Print Regional" else (np.random.uniform(3.5, 6.0) if ch == "Quick-Comm Sponsored" else np.random.uniform(2.5, 4.5))
        campaigns.append({
            "campaign_id": f"CMP_{i+1:04d}", "channel": ch, "spend_inr": round(sp, 2),
            "attributed_revenue_inr": round(sp * roas * np.random.uniform(0.85, 1.15), 2),
            "impressions": int(sp * np.random.uniform(15, 30)), "clicks": int(sp * np.random.uniform(0.5, 1.8))
        })
    df_campaigns = pd.DataFrame(campaigns)

    # 4. Brand Surveys & Competitors
    nps_scores = np.random.randint(1, 11, size=N_SURVEYS)
    df_brand_surveys = pd.DataFrame({
        "survey_id": [f"SRV_{i+1:04d}" for i in range(N_SURVEYS)],
        "customer_id": np.random.choice(customer_ids, size=N_SURVEYS, replace=False),
        "unaided_awareness": np.random.choice([0, 1], size=N_SURVEYS, p=[0.35, 0.65]),
        "nps_score": nps_scores,
        "perceived_innovation": np.random.randint(1, 11, size=N_SURVEYS),
        "perceived_affordability": np.random.randint(1, 11, size=N_SURVEYS),
        "sentiment_label": ["Positive" if x >= 8 else ("Neutral" if x >= 5 else "Negative") for x in nps_scores]
    })

    df_competitors = pd.DataFrame({
        "brand_name": ["Aura Organics (Us)", "GlowBotanica", "PureHerbal Co.", "EcoShield", "LegacyBeauty Inc."],
        "market_share_pct": [18.2, 22.5, 15.0, 12.3, 32.0],
        "avg_price_index": [100, 115, 85, 95, 90],
        "perceived_innovation_score": [7.8, 8.5, 5.2, 6.1, 4.8],
        "digital_shelf_share_pct": [21.0, 26.0, 12.5, 14.5, 26.0]
    })

    # Analytics Engine Computations
    REF_DATE = datetime(2026, 1, 1)

    mkt_summary = df_campaigns.groupby("channel").agg(
        spend=("spend_inr", "sum"), revenue=("attributed_revenue_inr", "sum"),
        impressions=("impressions", "sum"), clicks=("clicks", "sum")
    ).reset_index()
    mkt_summary["roas"] = (mkt_summary["revenue"] / mkt_summary["spend"]).round(2)

    cust_summary = df_transactions.groupby("customer_id").agg(
        last_purchase=("transaction_date", "max"),
        orders=("transaction_id", "count"),
        total_spend=("order_value_inr", "sum")
    ).reset_index()
    cust_summary["recency"] = (REF_DATE - cust_summary["last_purchase"]).dt.days

    rfm_log = np.log1p(cust_summary[["recency", "orders", "total_spend"]])
    rfm_scaled = StandardScaler().fit_transform(rfm_log)
    cust_summary["cluster_id"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(rfm_scaled)

    persona_map = {0: "High-Value Champions", 1: "Loyal Regulars", 2: "Promotional Buyers", 3: "At-Risk / Inactive"}
    cust_summary["persona"] = cust_summary["cluster_id"].map(persona_map)

    awareness = df_brand_surveys["unaided_awareness"].mean() * 100
    pos_sentiment = (df_brand_surveys["sentiment_label"] == "Positive").mean() * 100
    innovation = df_brand_surveys["perceived_innovation"].mean() * 10
    nps_val = (((df_brand_surveys["nps_score"] >= 9).sum() - (df_brand_surveys["nps_score"] <= 6).sum()) / len(df_brand_surveys)) * 100
    nps_norm = max(0, min(100, (nps_val + 100) / 2))

    brand_health_score = round((0.30 * awareness) + (0.30 * pos_sentiment) + (0.20 * innovation) + (0.20 * nps_norm), 1)

    return df_customers, df_transactions, df_campaigns, df_brand_surveys, df_competitors, mkt_summary, cust_summary, brand_health_score, awareness, pos_sentiment

# Load data safely through cache
df_customers, df_transactions, df_campaigns, df_brand_surveys, df_competitors, mkt_summary, cust_summary, brand_health_score, awareness, pos_sentiment = load_and_process_data()

# ==========================================
# DASHBOARD UI
# ==========================================
st.sidebar.title("Aura Intelligence")
st.sidebar.caption("FMCG Brand Strategy System v2.5")

st.title("🏆 Brand Strategy Intelligence Platform")
st.caption("Integrated Decision-Support Engine: Personal Care Category with AI Assistant")

# Global KPIs
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Brand Health", f"{brand_health_score}/100", "+2.1 pts")
kpi2.metric("Total Revenue", f"₹{(df_transactions['order_value_inr'].sum()/10000000):.2f} Cr", "+14.2% YoY")
kpi3.metric("Category Share", "18.2%", "+1.5%")
kpi4.metric("Blended ROAS", f"{(df_campaigns['attributed_revenue_inr'].sum()/df_campaigns['spend_inr'].sum()):.2f}x", "Target: 3.5x")
kpi5.metric("Active Customers", f"{len(cust_summary[cust_summary['recency'] <= 90]):,}", "90-Day Window")

st.markdown("---")

# Navigation Tabs
tab_overview, tab_cust, tab_brand, tab_mkt, tab_comp, tab_rec, tab_ai = st.tabs([
    "📊 Overview", "🎯 Customers", "🏷️ Brand Health", "📢 Marketing", "🥊 Competition", "🧠 Strategic Engine", "🤖 AI Analytics Assistant"
])

with tab_overview:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        df_monthly = df_transactions.set_index("transaction_date").resample("ME")["order_value_inr"].sum().reset_index()
        fig_rev = px.line(df_monthly, x="transaction_date", y="order_value_inr", title="<b>Monthly Revenue Trend (INR)</b>", template="plotly_white")
        st.plotly_chart(fig_rev, use_container_width=True)
    with col_b:
        fig_cat = px.pie(df_transactions, names="category", values="order_value_inr", title="<b>Revenue Share by Category</b>", hole=0.4, template="plotly_white")
        st.plotly_chart(fig_cat, use_container_width=True)

with tab_cust:
    st.subheader("Customer Behavior & RFM Machine Learning Clusters")
    fig_rfm = px.scatter(
        cust_summary, x="recency", y="total_spend", size="orders", color="persona",
        title="<b>RFM Clusters: Recency vs. Total Spend</b>",
        labels={"recency": "Days Since Last Order", "total_spend": "Spend (INR)"}, template="plotly_white"
    )
    st.plotly_chart(fig_rfm, use_container_width=True)

with tab_brand:
    st.subheader("Brand Health Sub-Component Breakdown")
    df_health = pd.DataFrame({
        "Metric": ["Unaided Awareness", "Positive Sentiment"],
        "Score": [awareness, pos_sentiment]
    })
    fig_health = px.bar(df_health, x="Metric", y="Score", color="Metric", title="<b>Brand Health Metrics</b>", template="plotly_white")
    st.plotly_chart(fig_health, use_container_width=True)

with tab_mkt:
    st.subheader("Campaign Channel Spend vs. Revenue Generation")
    fig_mkt = px.bar(mkt_summary, x="channel", y=["spend", "revenue"], barmode="group", title="<b>Channel Spend vs. Attributed Revenue</b>", template="plotly_white")
    st.plotly_chart(fig_mkt, use_container_width=True)

with tab_comp:
    st.subheader("Category Strategic Positioning Quadrant")
    fig_comp = px.scatter(
        df_competitors, x="perceived_innovation_score", y="avg_price_index", size="market_share_pct",
        color="brand_name", text="brand_name", title="<b>Price Index vs. Perceived Innovation</b>", template="plotly_white"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with tab_rec:
    st.header("🧠 Automated Strategic Recommendation Engine")
    top_channel = mkt_summary.sort_values("roas", ascending=False).iloc[0]
    st.markdown(f"""
        <div class="recommendation-box">
            <h4>📢 Marketing Capital Re-Allocation Opportunity</h4>
            <p><b>Observation:</b> <b>{top_channel['channel']}</b> delivers the highest ROAS at <b>{top_channel['roas']}x</b>.</p>
        </div>
    """, unsafe_allow_html=True)

with tab_ai:
    st.header("🤖 AI Conversational Analytics Assistant")
    st.caption("Ask questions about brand health, customer segments, marketing ROAS, or sales performance in natural language.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your Aura Organics AI Assistant. Ask me anything about our sales performance, customer segments, or marketing channels."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("e.g., Which marketing channel has the highest ROAS?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        prompt_lower = prompt.lower()
        if "roas" in prompt_lower or "channel" in prompt_lower or "marketing" in prompt_lower:
            best_ch = mkt_summary.sort_values("roas", ascending=False).iloc[0]
            response = f"Based on current campaign telemetry, **{best_ch['channel']}** delivers the highest performance with a blended ROAS of **{best_ch['roas']}x**."
        elif "segment" in prompt_lower or "customer" in prompt_lower:
            persona_counts = cust_summary['persona'].value_counts()
            response = f"Our customer base is segmented via K-Means clustering. The largest behavioral group is **{persona_counts.index[0]}** ({persona_counts.iloc[0]} customers)."
        else:
            total_rev = df_transactions['order_value_inr'].sum()
            response = f"Aura Organics has generated total revenues of **₹{total_rev/10000000:.2f} Cr** across 45,000+ transactions."

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
