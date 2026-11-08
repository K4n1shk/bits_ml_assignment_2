from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(
    page_title="ECG Heartbeat Classification",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"]
FEATURE_COLUMNS = [f"f{i}" for i in range(187)]
LABELS = ["Normal", "Abnormal"]
MODEL_DIR = Path(__file__).resolve().parent / "model"


@st.cache_resource
def load_model(name):
    filename = name.lower().replace(" ", "_") + ".pkl"
    return joblib.load(MODEL_DIR / filename)


def get_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def read_test_file(uploaded):
    try:
        data = pd.read_csv(uploaded)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
        st.error("I couldn't read that CSV. Check the file and try again.")
        st.stop()

    if data.empty:
        st.error("The file has no data rows.")
        st.stop()

    if "target" not in data.columns:
        st.error("The CSV needs a `target` column with the true labels (0 or 1).")
        st.stop()

    missing = [name for name in FEATURE_COLUMNS if name not in data.columns]
    extra = sorted(set(data.columns) - set(FEATURE_COLUMNS) - {"target"})
    if missing or extra:
        details = []
        if missing:
            details.append(f"{len(missing)} missing, starting with {', '.join(missing[:3])}")
        if extra:
            details.append(f"unexpected columns: {', '.join(extra[:3])}")
        st.error("Expected signal columns f0 through f186 (" + "; ".join(details) + ").")
        st.stop()

    try:
        X = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
        y = pd.to_numeric(data["target"], errors="raise")
    except (TypeError, ValueError):
        st.error("The signal values and target labels need to be numbers.")
        st.stop()

    if X.isna().any().any() or y.isna().any():
        st.error("The file has empty values. Fill them in before uploading it.")
        st.stop()
    if set(y.unique()) != {0, 1}:
        st.error("The target column needs both labels: 0 for normal and 1 for abnormal.")
        st.stop()

    return data, X, y.astype(int)


st.title("🫀 ECG Heartbeat Classification")
st.write(
    "Upload the held-out test CSV to see how the five trained models perform. "
    "Each row is one heartbeat with 187 signal values; **0 is normal** and **1 is abnormal**."
)

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox("Select model", MODEL_NAMES)
    st.caption("The models were trained in the notebook. This app only evaluates them.")

uploaded = st.file_uploader(
    "Upload test data (CSV)",
    type="csv",
    help="Use test_data.csv from this project. It contains only the held-out test split.",
)
if uploaded is None:
    st.caption("Use `test_data.csv` from the project folder, not the full training dataset.")
    st.stop()

data, X, y = read_test_file(uploaded)
st.caption(
    f"{len(data):,} beats · {(y == 0).sum():,} normal · "
    f"{(y == 1).sum():,} abnormal · 187 signal values per beat"
)

with st.expander("Preview the uploaded beats"):
    st.dataframe(data.head(5), hide_index=True)
    st.caption("Signal traces for the first three beats")
    traces = X.head(3).T
    traces.columns = [f"Beat {i}" for i in range(1, len(traces.columns) + 1)]
    st.line_chart(traces, x_label="Signal sample", y_label="Value")

try:
    with st.spinner("Evaluating the models..."):
        predictions = {}
        scores_by_model = {}
        for name in MODEL_NAMES:
            model = load_model(name)
            pred = model.predict(X)
            prob = model.predict_proba(X)[:, 1]
            predictions[name] = pred
            scores_by_model[name] = get_metrics(y, pred, prob)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"The models couldn't evaluate this file: {exc}")
    st.stop()

pred = predictions[model_name]
scores = scores_by_model[model_name]

st.subheader(f"Results for {model_name}")
st.caption("Precision, recall and F1 treat abnormal as the positive class.")
for column, (name, value) in zip(st.columns(6), scores.items()):
    column.metric(name, f"{value:.4f}")

left, right = st.columns(2, gap="large")
with left:
    st.markdown("#### Confusion matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion_matrix(y, pred, labels=[0, 1]),
        annot=True,
        fmt="d",
        cmap="Reds",
        cbar=False,
        square=True,
        linewidths=0.5,
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("#### Classification report")
    report = classification_report(y, pred, target_names=LABELS, output_dict=True)
    report_df = pd.DataFrame(report).T.loc[
        LABELS + ["macro avg", "weighted avg"],
        ["precision", "recall", "f1-score", "support"],
    ]
    report_df.index.name = "Class"
    report_df.columns = ["Precision", "Recall", "F1", "Support"]
    st.dataframe(
        report_df,
        width="stretch",
        column_config={
            name: st.column_config.NumberColumn(format="%.4f")
            for name in ("Precision", "Recall", "F1")
        },
    )

st.subheader("Compare all five models")
st.caption("Scores are calculated from the file you uploaded.")
comparison = pd.DataFrame(scores_by_model).T
comparison.index.name = "Model"
st.dataframe(
    comparison,
    width="stretch",
    column_config={name: st.column_config.NumberColumn(format="%.4f") for name in comparison.columns},
)
