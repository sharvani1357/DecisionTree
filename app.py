import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import confusion_matrix, accuracy_score

# ---------------- Page Config ----------------
st.set_page_config(page_title="Decision Trees Example", layout="wide")

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- Sidebar Controls ----------------
st.sidebar.title("⚙ Model Controls")

max_depth = st.sidebar.slider(
    "Select Tree Depth",
    min_value=1,
    max_value=10,
    value=4
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05
)

# ---------------- Title ----------------
st.markdown("<h1 class='main-title'>Decision Trees Example</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-title'>Bank Marketing Dataset</h3>", unsafe_allow_html=True)

# ---------------- Load Dataset ----------------
df = pd.read_csv("bank_marketing_dataset.csv")

st.header("1️⃣ Dataset Preview")
st.dataframe(df.head())

# ---------------- Outlier Handling ----------------
st.header("2️⃣ Outlier Detection & Handling")

num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
df_handled = df.copy()

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_handled[col] = df_handled[col].clip(lower, upper)

st.success("Outliers handled using IQR capping")

# ---------------- Encoding ----------------
X = df_handled.drop("deposit", axis=1)
deposit = df_handled["deposit"]

le = LabelEncoder()
for col in X.select_dtypes(include='object'):
    X[col] = le.fit_transform(X[col])

deposit = le.fit_transform(deposit)

# ---------------- Train-Test Split ----------------
X_train, X_test, deposit_train, deposit_test = train_test_split(
    X, deposit, test_size=test_size, random_state=42
)

# ---------------- Model Training ----------------
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=max_depth,
    random_state=42
)
model.fit(X_train, deposit_train)

deposit_pred = model.predict(X_test)

# ---------------- Metric Cards ----------------
st.header("3️⃣ Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    label="Accuracy",
    value=f"{accuracy_score(deposit_test, deposit_pred):.2f}"
)

col2.metric(
    label="Tree Depth",
    value=model.get_depth()
)

col3.metric(
    label="Number of Leaves",
    value=model.get_n_leaves()
)

# ---------------- Confusion Matrix ----------------
st.header("4️⃣ Confusion Matrix")

cm = confusion_matrix(deposit_test, deposit_pred)

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Deposit", "Deposit"],
    yticklabels=["No Deposit", "Deposit"]
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ---------------- Decision Tree Visualization ----------------
st.header("5️⃣ Decision Tree Visualization")

fig2, ax2 = plt.subplots(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Deposit", "Deposit"],
    filled=True,
    ax=ax2
)
st.pyplot(fig2)

# ---------------- Task 8: Rules ----------------
st.header("8️⃣ Interpret Learning Rules (IF–ELSE)")

rules = export_text(model, feature_names=list(X.columns))
st.text(rules)

# ---------------- Task 9: Decision Logic ----------------
st.header("9️⃣ Analyze Decision Logic")

top_feature = X.columns[model.tree_.feature[0]]
st.write(f"**Top Feature Used:** {top_feature}")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df)

# ---------------- Task 10: Depth Analysis ----------------
st.header("🔟 Effect of Tree Depth")

depth_results = []

for d in [2, 4, 6, 8, 10]:
    temp = DecisionTreeClassifier(max_depth=d, random_state=42)
    temp.fit(X_train, deposit_train)
    depth_results.append({
        "Depth": d,
        "Train Accuracy": temp.score(X_train, deposit_train),
        "Test Accuracy": temp.score(X_test, deposit_test)
    })

st.dataframe(pd.DataFrame(depth_results))
# ---------------- Footer ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p class='footer'>Decision Tree Explainable Banking Model | Streamlit Deployment</p>",
    unsafe_allow_html=True
)
