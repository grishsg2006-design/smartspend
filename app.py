import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="SmartSpend", page_icon="💰", layout="wide")
st.markdown("""
<style>
.stApp {background:#f7f9fc}
.block-container {padding-top:1.5rem}
[data-testid="stMetric"] {background:white;border:1px solid #e2e8f0;padding:14px;border-radius:12px}
</style>
""", unsafe_allow_html=True)
st.title("💰 SmartSpend")
st.caption("Interactive machine-learning expense tracker")
st.info("Models are trained from expense_data_1.csv. Prediction tables and charts on the input pages use only the transactions you enter/upload. A confusion matrix requires actual labels to compare against predictions.")

DATA_FILE = "expense_data_1.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    required = ["Date", "Amount", "Category", "Income/Expense"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV missing columns: " + ", ".join(missing))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    if "Currency" not in df:
        df["Currency"] = "INR"
    df["Currency"] = df["Currency"].fillna("INR").astype(str).str.strip().str.upper()
    df["Category"] = df["Category"].fillna("Other").astype(str).str.strip()
    df["Income/Expense"] = df["Income/Expense"].astype(str).str.strip().str.title()
    df = df.dropna(subset=["Date", "Amount"])
    df = df[(df["Currency"] == "INR") & df["Income/Expense"].isin(["Income", "Expense"])].copy()
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    return df

@st.cache_resource
def train_models(df):
    exp = df[df["Income/Expense"] == "Expense"].copy()
    if len(exp) < 5:
        raise ValueError("At least five INR expense rows are required.")
    # Linear Regression: estimate amount from date features.
    reg = LinearRegression().fit(exp[["Month", "Day", "DayOfWeek"]], exp["Amount"])

    # Logistic Regression: classify Income/Expense.
    Xie, yie = df[["Month", "Day", "DayOfWeek", "Amount"]], df["Income/Expense"]
    if yie.nunique() < 2:
        raise ValueError("Dataset must contain Income and Expense records.")
    scaler_ie = StandardScaler().fit(Xie)
    logistic = LogisticRegression(max_iter=1000, random_state=42)
    logistic.fit(scaler_ie.transform(Xie), yie)

    # KNN and Decision Tree: classify expense category.
    counts = exp["Category"].value_counts()
    cat = exp[exp["Category"].isin(counts[counts >= 2].index)].copy()
    if len(cat) < 5 or cat["Category"].nunique() < 2:
        raise ValueError("Not enough repeated categories to train category models.")
    Xcat, ycat = cat[["Month", "Day", "DayOfWeek", "Amount"]], cat["Category"]
    scaler_cat = StandardScaler().fit(Xcat)
    knn = KNeighborsClassifier(n_neighbors=min(5, len(cat))).fit(scaler_cat.transform(Xcat), ycat)
    tree = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42).fit(Xcat, ycat)

    # K-Means: group transactions by amount/date.
    Xcl = exp[["Amount", "Month", "Day", "DayOfWeek"]]
    scaler_cl = StandardScaler().fit(Xcl)
    kmeans = KMeans(n_clusters=min(3, len(exp)), random_state=42, n_init=10)
    kmeans.fit(scaler_cl.transform(Xcl))
    return {"reg":reg, "log":logistic, "scaler_ie":scaler_ie,
            "knn":knn, "tree":tree, "scaler_cat":scaler_cat,
            "kmeans":kmeans, "scaler_cl":scaler_cl, "exp":exp, "cat":cat}

try:
    df = load_data()
    m = train_models(df)
except FileNotFoundError:
    st.error(f"File `{DATA_FILE}` not found. Put your dataset CSV in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Could not load/train models: {e}")
    st.stop()

page = st.sidebar.radio("Navigate", ["Single prediction", "Batch input & graphs", "Model evaluation"])

def add_date_features(rows):
    out = rows.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Amount"] = pd.to_numeric(out["Amount"], errors="coerce")
    out = out.dropna(subset=["Date", "Amount"]).copy()
    out["Month"] = out["Date"].dt.month
    out["Day"] = out["Date"].dt.day
    out["DayOfWeek"] = out["Date"].dt.dayofweek
    return out

def predict_batch(rows):
    out = add_date_features(rows)
    if out.empty:
        return out
    X = out[["Month", "Day", "DayOfWeek", "Amount"]]
    out["Predicted Income/Expense"] = m["log"].predict(m["scaler_ie"].transform(X))
    out["Predicted Category (KNN)"] = m["knn"].predict(m["scaler_cat"].transform(X))
    out["Predicted Category (Decision Tree)"] = m["tree"].predict(X)
    out["Predicted Cluster"] = m["kmeans"].predict(
        m["scaler_cl"].transform(out[["Amount", "Month", "Day", "DayOfWeek"]]))
    out["Estimated Amount (Linear Regression)"] = np.maximum(
        0, m["reg"].predict(out[["Month", "Day", "DayOfWeek"]]))
    return out

if page == "Single prediction":
    st.subheader("Enter one transaction")
    with st.form("one_transaction"):
        c1, c2 = st.columns(2)
        with c1:
            d = st.date_input("Transaction date")
        with c2:
            amount = st.number_input("Amount (INR)", min_value=0.01, value=150.0, step=10.0)
        go = st.form_submit_button("Predict", type="primary", use_container_width=True)
    if go:
        row = pd.DataFrame([{"Date":pd.Timestamp(d), "Amount":float(amount)}])
        result = predict_batch(row)
        r = result.iloc[0]
        a,b,c,dcol = st.columns(4)
        a.metric("Income / Expense", r["Predicted Income/Expense"])
        b.metric("KNN category", r["Predicted Category (KNN)"])
        c.metric("Decision Tree category", r["Predicted Category (Decision Tree)"])
        dcol.metric("Cluster", f"Cluster {int(r['Predicted Cluster'])}")
        st.metric("Linear Regression estimated amount", f"₹{r['Estimated Amount (Linear Regression)']:,.2f}")
        st.caption("The amount estimate uses date features only; treat it as an experimental estimate, not a reliable personal budget.")
        st.markdown("#### Graphs for your input")
        fig, ax = plt.subplots(figsize=(6,3.5))
        ax.bar(["Entered amount", "Estimated amount"], [amount, r["Estimated Amount (Linear Regression)"]])
        ax.set_ylabel("INR (₹)")
        ax.set_title("Entered vs estimated amount")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(result[["Date","Amount","Predicted Income/Expense","Predicted Category (KNN)","Predicted Category (Decision Tree)","Predicted Cluster","Estimated Amount (Linear Regression)"]], use_container_width=True, hide_index=True)
        st.download_button("Download prediction", result.to_csv(index=False).encode(), "smartspend_prediction.csv", "text/csv")
        st.warning("A confusion matrix cannot be calculated from one input because its correct label is unknown. Use Batch input with actual labels to calculate one.")

elif page == "Batch input & graphs":
    st.subheader("Analyze your own transactions")
    st.write("Upload a CSV with `Date` and `Amount`, or enter several rows below.")
    uploaded = st.file_uploader("Upload your input CSV", type=["csv"])
    if uploaded:
        try:
            user_rows = pd.read_csv(uploaded)
            if not {"Date", "Amount"}.issubset(user_rows.columns):
                st.error("CSV must contain columns named Date and Amount.")
            else:
                result = predict_batch(user_rows)
                if result.empty:
                    st.warning("No valid rows. Check dates and amounts.")
                else:
                    st.success(f"Predictions generated for {len(result)} of your rows.")
                    st.dataframe(result, use_container_width=True, hide_index=True)
                    st.download_button("Download predicted rows", result.to_csv(index=False).encode(), "smartspend_predictions.csv", "text/csv")
                    st.markdown("### Charts from your uploaded rows only")
                    left, right = st.columns(2)
                    with left:
                        counts = result["Predicted Category (Decision Tree)"].value_counts()
                        fig, ax = plt.subplots()
                        ax.bar(counts.index.astype(str), counts.values)
                        ax.set_title("Your predicted category counts")
                        ax.set_ylabel("Transactions")
                        ax.tick_params(axis="x", rotation=35)
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                    with right:
                        counts = result["Predicted Cluster"].value_counts().sort_index()
                        fig, ax = plt.subplots()
                        ax.bar(counts.index.astype(str), counts.values)
                        ax.set_title("Clusters in your input")
                        ax.set_xlabel("Cluster"); ax.set_ylabel("Transactions")
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                    fig, ax = plt.subplots(figsize=(9,4))
                    temp = result.sort_values("Date")
                    ax.plot(temp["Date"], temp["Amount"], marker="o")
                    ax.set_title("Amounts in your submitted transactions")
                    ax.set_ylabel("Amount (₹)"); ax.tick_params(axis="x", rotation=30); ax.grid(alpha=.25)
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                    st.markdown("### Confusion matrix for your own input (optional)")
                    if "Actual Category" in result.columns:
                        actual = result["Actual Category"].astype(str).str.strip()
                        pred = result["Predicted Category (Decision Tree)"].astype(str)
                        labels = sorted(set(actual) | set(pred))
                        cm = confusion_matrix(actual, pred, labels=labels)
                        fig, ax = plt.subplots(figsize=(max(5,len(labels)), max(4,len(labels)*.7)))
                        im = ax.imshow(cm, cmap="Blues")
                        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
                        ax.set_yticks(range(len(labels)), labels=labels)
                        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Your input: category confusion matrix")
                        for i in range(len(labels)):
                            for j in range(len(labels)): ax.text(j,i,int(cm[i,j]),ha="center",va="center")
                        fig.colorbar(im, ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                    elif "Actual Income/Expense" in result.columns:
                        actual = result["Actual Income/Expense"].astype(str).str.strip().str.title()
                        pred = result["Predicted Income/Expense"].astype(str)
                        labels = ["Expense", "Income"]
                        cm = confusion_matrix(actual, pred, labels=labels)
                        fig, ax = plt.subplots(figsize=(5,4))
                        im = ax.imshow(cm, cmap="Blues")
                        ax.set_xticks([0,1], labels=labels); ax.set_yticks([0,1], labels=labels)
                        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Your input: transaction-type confusion matrix")
                        for i in range(2):
                            for j in range(2): ax.text(j,i,int(cm[i,j]),ha="center",va="center")
                        fig.colorbar(im, ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                    else:
                        st.info("To calculate a confusion matrix, add an `Actual Category` or `Actual Income/Expense` column to your CSV. It compares known true labels with predictions; it cannot be meaningfully made from unlabeled inputs.")
        except Exception as e:
            st.error(f"Could not process CSV: {e}")
    else:
        manual = st.data_editor(pd.DataFrame({"Date":[pd.Timestamp.today().strftime("%Y-%m-%d")],"Amount":[150.0]}), num_rows="dynamic", use_container_width=True)
        if st.button("Predict entered rows", type="primary"):
            result = predict_batch(manual)
            if result.empty:
                st.warning("Enter at least one valid date and amount.")
            else:
                st.dataframe(result, use_container_width=True, hide_index=True)
                st.bar_chart(result["Predicted Category (Decision Tree)"].value_counts())
                st.bar_chart(result["Predicted Cluster"].value_counts().sort_index())
                st.caption("These charts summarize only your entered rows.")

else:
    st.subheader("Model evaluation")
    st.warning("This page evaluates on a held-out split of the source dataset, not on your individual input. For a confusion matrix on your own transactions, upload a labeled batch in the previous page.")
    choice = st.selectbox("Classifier", ["Logistic Regression — Income/Expense", "KNN — Expense category", "Decision Tree — Expense category"])
    if choice.startswith("Logistic"):
        ev = df.copy()
        X, y = ev[["Month","Day","DayOfWeek","Amount"]], ev["Income/Expense"]
        Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
        scaler = StandardScaler()
        Xtr, Xte = scaler.fit_transform(Xtr), scaler.transform(Xte)
        model = LogisticRegression(max_iter=1000,random_state=42).fit(Xtr,ytr)
    else:
        ev = m["cat"].copy()
        X, y = ev[["Month","Day","DayOfWeek","Amount"]], ev["Category"]
        strat = y if y.value_counts().min() >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=strat)
        if choice.startswith("KNN"):
            scaler = StandardScaler()
            Xtr, Xte = scaler.fit_transform(Xtr), scaler.transform(Xte)
            model = KNeighborsClassifier(n_neighbors=min(5,len(ytr))).fit(Xtr,ytr)
        else:
            model = DecisionTreeClassifier(max_depth=5,class_weight="balanced",random_state=42).fit(Xtr,ytr)
    pred = model.predict(Xte)
    c1,c2,c3 = st.columns(3)
    c1.metric("Held-out accuracy", f"{accuracy_score(yte,pred):.1%}")
    c2.metric("Test transactions", len(yte))
    c3.metric("Classes", y.nunique())
    labels = list(model.classes_)
    cm = confusion_matrix(yte,pred,labels=labels)
    fig, ax = plt.subplots(figsize=(max(5,len(labels)*.9),max(4,len(labels)*.7)))
    im = ax.imshow(cm,cmap="Blues")
    ax.set_xticks(range(len(labels)),labels=[str(x) for x in labels],rotation=45,ha="right")
    ax.set_yticks(range(len(labels)),labels=[str(x) for x in labels])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix — held-out source data")
    for i in range(len(labels)):
        for j in range(len(labels)): ax.text(j,i,int(cm[i,j]),ha="center",va="center")
    fig.colorbar(im,ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.markdown("#### Classification report")
    st.dataframe(pd.DataFrame(classification_report(yte,pred,labels=labels,output_dict=True,zero_division=0)).T.round(3), use_container_width=True)
