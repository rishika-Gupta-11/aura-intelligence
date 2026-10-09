# 🚀 Enterprise Customer Intelligence & Brand Health Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-darkblue.svg)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io/)

An end-to-end, production-grade data science platform designed to ingest multi-source customer telemetry, execute unsupervised machine learning (RFM + K-Means Clustering), dynamically compute a real-time Brand Health Index, and provide conversational insights via an integrated AI analytics assistant. Built with modularity, scalability, and enterprise best practices in mind.

---------------------------------------------------------------------------------------------------

## 🏗️ System Architecture & Multi-Domain Pipeline Flow

The platform is architected around a decoupled, multi-domain ingestion, transformation, and conversational intelligence pipeline.

+---------------------------------------------------------------------------------+
|                         1. DATA INGESTION & VALIDATION                          |
|   [CRM Transactions]       [Web Clickstream Telemetry]       [Customer Surveys] |
+---------------------------------------+-----------------------------------------+
v
+---------------------------------------------------------------------------------+
|                         2. FEATURE ENGINEERING ENGINE                           |
|   - Recency (Days since last interaction)                                       |
|   - Frequency (Total transaction/engagement count)                              |
|   - Monetary Value (Total lifetime spend / CLV proxy)                           |
+---------------------------------------+-----------------------------------------+
v
+---------------------------------------------------------------------------------+
|                       3. UNSUPERVISED MACHINE LEARNING                          |
|   - Log Transformation & Standard Scaler Normalization                          |
|   - Optimal K Selection (Elbow Method & Silhouette Analysis)                   |
|   - K-Means Behavioral Clustering                                               |
+---------------------------------------+-----------------------------------------+
v
+---------------------------------------------------------------------------------+
|                       4. CONVERSATIONAL AI ANALYTICS LAYER                      |
|   - Natural Language Query Processing & Intent Routing                          |
|   - Data-Backed Telemetry Retrieval & Automated Insights Generator              |
+---------------------------------------------------------------------------------+

---------------------------------------------------------------------------------------

## 🤖 AI Conversational Assistant Integration

The platform features an embedded **AI Analytics Assistant** built directly into the business intelligence application:
* **Natural Language Exploration:** Enables executives and stakeholders to query sales performance, customer segments, and marketing metrics using conversational prompts.
* **Deterministic Validation & Data Retrieval:** Dynamically routes queries to underlying Pandas dataframes, pulling live calculations (such as blended ROAS, active customer counts, and sentiment scores) to ensure all generated responses are strictly data-backed.

--------------------------------------------------------------------------------------

## 🧮 RFM + K-Means Clustering Math & Brand Health Formula Breakdown

 1. Feature Transformation Pipeline
Because monetary and frequency distributions in transactional data are typically right-skewed, raw metrics are stabilized using a logarithmic transformation before scaling:

$$X_{transformed} = \ln(X + 1)$$

Features are then standardized using standard scaling to ensure equal variance contribution during distance calculations:

$$Z = \frac{X - \mu}{\sigma}$$

 2. K-Means Objective Function
The segmentation engine groups customers into $K$ distinct clusters by minimizing the Within-Cluster Sum of Squares (WCSS):

$$J = \sum_{i=1}^{K} \sum_{x \in S_i} \vert{}\vert{}x - \mu_i\vert{}\vert{}^2$$

Where:
- $K$ = Number of behavioral segments (determined via Silhouette Optimization).
- $S_i$ = The set of data points assigned to cluster $i$.
- $\mu_i$ = The centroid (mean vector) of cluster $i$.

 3. Dynamic Brand Health Index (BHI) Formula
The overall Brand Health Index is computed as a weighted composite score combining normalized engagement, monetary value, and sentiment metrics:

$$\text{BHI} = w_1(\text{Normalized RFM Score}) + w_2(1 - \text{Churn Probability}) + w_3(\text{CSAT / Sentiment Index})$$

Where $\sum w_i = 1$, allowing stakeholders to adjust business priorities dynamically.

-----------------------------------------------------------------------------------------

## ⚙️ Step-by-Step Installation & Local Execution Guide

### Prerequisites
- Python 3.10 or higher
- Git

1. Clone the Repository
```bash
git clone [https://github.com/rishika-Gupta-11/Banking-Customer-Transaction-Analytics.git](https://github.com/rishika-Gupta-11/Banking-Customer-Transaction-Analytics.git)
cd Banking-Customer-Transaction-Analytics
