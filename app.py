
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="SmartSpend | Expense Intelligence",
    page_icon="💰",
    layout="wide"
)

# ---------- STYLE ----------
st.markdown("""
<style>
.stApp { background: #F8FAFC; color: #0F172A; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 18px;
    border-radius: 14px;
}
h1, h2, h3 { color: #0F172A; }

/* SmartSpend: improve dashboard readability */

.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #334155 !important;
}

.stMarkdown p,
.stMarkdown li,
.stCaption,
label {
    color: #334155;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px;
}

[data-testid="stMetricLabel"] p {
    color: #64748B !important;
}

[data-testid="stMetricValue"] {
    color: #0F172A !important;
}

h1, h2, h3 {
    color: #0F172A !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    data = pd.read_csv("expense_data_1.csv")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Amount"] = pd.to_numeric(data["Amount"], errors="coerce")
    data["Currency"] = data["Currency"].fillna("INR")
    data["Category"] = data["Category"].fillna("Other")
    data["Income/Expense"] = data["Income/Expense"].astype(str).str.strip()

    data = data[
        (data["Currency"] == "INR")
    ].dropna(subset=["Date", "Amount"]).copy()

    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["DayOfWeek"] = data["Date"].dt.dayofweek

    return data

df = load_data()
expenses = df[df["Income/Expense"] == "Expense"].copy()
income = df[df["Income/Expense"] == "Income"].copy()

# ---------- SIDEBAR ----------
st.sidebar.markdown("# 💰 SmartSpend")
st.sidebar.caption("PERSONAL FINANCE INTELLIGENCE")
st.sidebar.divider()

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Overview",
        "Transactions",
        "Spending Analytics",
        "Model Evaluation",
        "K-Means Clustering"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning Expense Tracker")

# ---------- HEADER ----------
st.title("SmartSpend")
st.caption("Understand your spending. Discover patterns. Make informed decisions.")

if page == "Overview":
    st.subheader("Financial Overview")

    total_expenses = expenses["Amount"].sum()
    total_income = income["Amount"].sum()
    net_amount = total_income - total_expenses

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Expenses", f"₹{total_expenses:,.2f}")
    c2.metric("Total Income", f"₹{total_income:,.2f}")
    c3.metric("Net Difference", f"₹{net_amount:,.2f}")
    c4.metric("Expense Transactions", f"{len(expenses):,}")

    st.divider()
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Spending by Category")
        category_sum = expenses.groupby("Category")["Amount"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        category_sum.plot(kind="barh", ax=ax, color="#F59E0B")
        ax.set_xlabel("Amount (INR)")
        ax.set_ylabel("")
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Income vs Expenses")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(
            ["Income", "Expenses"],
            [total_income, total_expenses],
            color=["#16A34A", "#F59E0B"],
            width=0.55
        )
        ax.set_ylabel("Amount (INR)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Recent Transactions")
    st.dataframe(
        df.sort_values("Date", ascending=False)[
            ["Date", "Category", "Amount", "Income/Expense"]
        ].head(10),
        use_container_width=True,
        hide_index=True
    )

elif page == "Transactions":
    st.subheader("Transaction Explorer")

    search = st.text_input("Search category")
    kind = st.selectbox("Transaction type", ["All", "Expense", "Income"])

    view = df.copy()
    if search:
        view = view[view["Category"].str.contains(search, case=False, na=False)]
    if kind != "All":
        view = view[view["Income/Expense"] == kind]

    st.caption(f"{len(view)} transactions found")
    st.dataframe(
        view.sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download filtered transactions",
        view.to_csv(index=False).encode("utf-8"),
        "smartspend_transactions.csv",
        "text/csv"
    )

elif page == "Spending Analytics":
    st.subheader("Spending Analytics")

    monthly = expenses.copy()
    monthly["MonthLabel"] = monthly["Date"].dt.to_period("M").astype(str)
    monthly_sum = monthly.groupby("MonthLabel")["Amount"].sum()

    st.markdown("#### Monthly expense trend")
    st.line_chart(monthly_sum, use_container_width=True)

    st.markdown("#### Category summary")
    summary = expenses.groupby("Category")["Amount"].agg(
        Transactions="count",
        Total="sum",
        Average="mean"
    ).sort_values("Total", ascending=False).round(2)

    st.dataframe(summary, use_container_width=True)

elif page == "Model Evaluation":
    st.subheader("Machine Learning Model Evaluation")
    st.caption(
        "Models are evaluated on held-out test data. Scores may differ "
        "slightly from your earlier notebook if settings differ."
    )

    tab1, tab2, tab3 = st.tabs([
        "Logistic Regression",
        "KNN Classifier",
        "Decision Tree"
    ])

    # Logistic Regression: Expense vs Income
    with tab1:
        log_df = df[df["Income/Expense"].isin(["Expense", "Income"])].copy()
        X = log_df[["Month", "Day", "DayOfWeek", "Amount"]]
        y = log_df["Income/Expense"]

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)

        st.markdown("#### Expense / Income classification")
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{accuracy_score(yte, pred):.1%}")
        b.metric("Precision (macro)", f"{precision_score(yte, pred, average='macro', zero_division=0):.1%}")
        c.metric("Recall (macro)", f"{recall_score(yte, pred, average='macro', zero_division=0):.1%}")
        d.metric("F1 (macro)", f"{f1_score(yte, pred, average='macro', zero_division=0):.1%}")

        labels = ["Expense", "Income"]
        cm = confusion_matrix(yte, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(2), labels=labels)
        ax.set_yticks(range(2), labels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### Classification report")
        st.dataframe(
            pd.DataFrame(classification_report(
                yte, pred, output_dict=True, zero_division=0
            )).T.round(3),
            use_container_width=True
        )

    # Prepare category data shared by KNN and Decision Tree
    category_counts = expenses["Category"].value_counts()
    valid_categories = category_counts[category_counts >= 2].index
    cat_df = expenses[expenses["Category"].isin(valid_categories)].copy()
    Xc = cat_df[["Month", "Day", "DayOfWeek", "Amount"]]
    yc = cat_df["Category"]

    Xtr, Xte, ytr, yte = train_test_split(
        Xc, yc, test_size=0.2, random_state=42, stratify=yc
    )
    scaler_c = StandardScaler()
    Xtr_s = scaler_c.fit_transform(Xtr)
    Xte_s = scaler_c.transform(Xte)

    with tab2:
        st.markdown("#### KNN category classification")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(Xtr_s, ytr)
        pred = knn.predict(Xte_s)

        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{accuracy_score(yte, pred):.1%}")
        b.metric("Precision (macro)", f"{precision_score(yte, pred, average='macro', zero_division=0):.1%}")
        c.metric("Recall (macro)", f"{recall_score(yte, pred, average='macro', zero_division=0):.1%}")
        d.metric("F1 (macro)", f"{f1_score(yte, pred, average='macro', zero_division=0):.1%}")

        labels = sorted(yc.unique())
        cm = confusion_matrix(yte, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("KNN Confusion Matrix")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            pd.DataFrame(classification_report(
                yte, pred, labels=labels, output_dict=True, zero_division=0
            )).T.round(3),
            use_container_width=True
        )

    with tab3:
        st.markdown("#### Decision Tree category classification")
        tree = DecisionTreeClassifier(
            max_depth=5, random_state=42
        )
        tree.fit(Xtr, ytr)
        pred = tree.predict(Xte)

        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{accuracy_score(yte, pred):.1%}")
        b.metric("Precision (macro)", f"{precision_score(yte, pred, average='macro', zero_division=0):.1%}")
        c.metric("Recall (macro)", f"{recall_score(yte, pred, average='macro', zero_division=0):.1%}")
        d.metric("F1 (macro)", f"{f1_score(yte, pred, average='macro', zero_division=0):.1%}")

        labels = sorted(yc.unique())
        cm = confusion_matrix(yte, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(cm, cmap="YlOrBr")
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Decision Tree Confusion Matrix")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            pd.DataFrame(classification_report(
                yte, pred, labels=labels, output_dict=True, zero_division=0
            )).T.round(3),
            use_container_width=True
        )

elif page == "K-Means Clustering":
    st.subheader("Expense Clustering")
    st.caption("Explore groups of transactions with similar amount and date patterns.")

    Xk = expenses[["Amount", "Month", "Day", "DayOfWeek"]].copy()
    scaler_k = StandardScaler()
    Xk_scaled = scaler_k.fit_transform(Xk)

    k = st.slider("Number of clusters (K)", 2, 8, 3)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(Xk_scaled)

    clustered = expenses.copy()
    clustered["Cluster"] = cluster_labels

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Cluster distribution")
        counts = clustered["Cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index.astype(str), counts.values, color="#F59E0B")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Transactions")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown("#### Elbow Method")
        max_k = min(10, len(Xk_scaled))
        inertia = []
        ks = list(range(1, max_k + 1))
        for n in ks:
            temp = KMeans(n_clusters=n, random_state=42, n_init=10)
            temp.fit(Xk_scaled)
            inertia.append(temp.inertia_)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, inertia, marker="o", color="#0F766E")
        ax.set_xlabel("Number of clusters (K)")
        ax.set_ylabel("Inertia")
        ax.set_xticks(ks)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Cluster visualization")
    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        clustered["Month"], clustered["Amount"],
        c=clustered["Cluster"], cmap="viridis",
        alpha=0.75, s=45, edgecolors="white", linewidths=0.3
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Expense Amount (INR)")
    ax.set_title("Expense Amount by Month and Cluster")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Cluster summary")
    summary = clustered.groupby("Cluster")["Amount"].agg(
        Transactions="count",
        Average="mean",
        Minimum="min",
        Maximum="max"
    ).round(2)
    st.dataframe(summary, use_container_width=True)

    st.info(
        "Clusters are based on Amount, Month, Day, and DayOfWeek. "
        "They are exploratory groups, not guaranteed spending categories."
    )
