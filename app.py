import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report, roc_curve)
from sklearn.preprocessing import label_binarize

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}
TARGET = "Class"

#loading saved artifacts
@st.cache_resource
def load_artifacts():
    models = {n: joblib.load(os.path.join(MODEL_DIR, f)) for n, f in MODEL_FILES.items()}
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    features = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))
    return models, scaler, le, features


def compute_metrics(y_true, y_pred, y_prob, n_classes):
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted",
                            labels=list(range(n_classes)))
    except ValueError:
        auc = np.nan  # happens if the uploaded file lacks some classes
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


models, scaler, le, features = load_artifacts()
classes = list(le.classes_)

#  header 
st.title(" Dry Bean Classification Dashboard")
st.caption("Compare 5 pre-trained ML models on dry-bean shape measurements "
           "(UCI Dry Bean Dataset, 7 bean types)")

#  sidebar 
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
use_sample = st.sidebar.checkbox("Use the bundled test_data.csv", value=uploaded is None,
                                 disabled=uploaded is not None)
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

with st.sidebar.expander("Expected CSV format"):
    st.write(f"Columns needed: {len(features)} features + target `{TARGET}`")
    st.code(", ".join(features + [TARGET]), language=None)

if uploaded is not None:
    data = pd.read_csv(uploaded)
elif use_sample and os.path.exists("test_data.csv"):
    data = pd.read_csv("test_data.csv")
else:
    st.info("Upload a test CSV (or tick the bundled sample) to begin.")
    st.stop()

#  validate 
missing = [c for c in features + [TARGET] if c not in data.columns]
if missing:
    st.error(f"Uploaded file is missing columns: {missing}")
    st.stop()
data = data.dropna(subset=features + [TARGET])
unknown = set(data[TARGET].astype(str)) - set(classes)
if unknown:
    st.error(f"Unknown class labels in target column: {sorted(unknown)}. Expected: {classes}")
    st.stop()

X = scaler.transform(data[features])
y_true = le.transform(data[TARGET].astype(str))

#  predict with every model 
@st.cache_data(show_spinner="Running all models...")
def run_all(X, y_true):
    out = {}
    for n, m in models.items():
        pred = m.predict(X)
        prob = m.predict_proba(X)
        out[n] = (pred, prob, compute_metrics(y_true, pred, prob, len(classes)))
    return out

results = run_all(X, y_true)
pred, prob, metrics = results[model_name]

st.success(f"Loaded **{len(data):,} rows** · evaluating **{model_name}**")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Metrics", "Confusion Matrix", "ROC Curves", "Model Comparison", "Predictions"])

#  tab 1: metrics 
with tab1:
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")
    st.subheader("Classification report")
    rep = pd.DataFrame(classification_report(y_true, pred, labels=list(range(len(classes))),
                                             target_names=classes, output_dict=True,
                                             zero_division=0)).T
    st.dataframe(rep.round(3), width="stretch")
    per_class = rep.loc[classes, ["precision", "recall", "f1-score"]].reset_index() \
        .melt(id_vars="index", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=per_class, x="index", y="Score", hue="Metric", ax=ax)
    ax.set_xlabel("Bean class"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-class scores - {model_name}")
    ax.legend(loc="lower right")
    st.pyplot(fig); plt.close(fig)

#  confusion matrix 
with tab2:
    cm = confusion_matrix(y_true, pred, labels=list(range(len(classes))))
    normalize = st.toggle("Show as % of each actual class", value=False)
    z = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100 if normalize else cm
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(z, annot=True, fmt=".1f" if normalize else "d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {model_name}")
    st.pyplot(fig); plt.close(fig)
    off = cm.copy(); np.fill_diagonal(off, 0)
    if off.max() > 0:
        i, j = np.unravel_index(off.argmax(), off.shape)
        st.info(f"Most common mistake: **{classes[i]}** predicted as **{classes[j]}** ({off[i, j]} times).")

#  tab 3: ROC 
with tab3:
    present = [i for i in range(len(classes)) if (y_true == i).any()]
    if len(present) < 2:
        st.warning("Need at least 2 classes in the uploaded data to draw ROC curves.")
    else:
        y_bin = label_binarize(y_true, classes=list(range(len(classes))))
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in present:
            fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
            ax.plot(fpr, tpr, label=classes[i])
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(f"One-vs-rest ROC curves - {model_name}")
        ax.legend()
        st.pyplot(fig); plt.close(fig)

#  tab 4: comparison 
with tab4:
    comp = pd.DataFrame({n: r[2] for n, r in results.items()}).T.round(4)
    comp.index.name = "ML Model Name"
    st.subheader("All 5 models on your uploaded test data")
    st.dataframe(comp.style.highlight_max(axis=0, color="#c6efce"), width="stretch")
    chosen = st.multiselect("Metrics to plot", list(comp.columns),
                            default=["Accuracy", "F1", "MCC"])
    if chosen:
        long = comp[chosen].reset_index().melt(id_vars="ML Model Name",
                                               var_name="Metric", value_name="Score")
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(data=long, x="ML Model Name", y="Score", hue="Metric", ax=ax)
        ax.set_ylim(max(0, long["Score"].min() - 0.05), 1.0)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig); plt.close(fig)
    best = comp["MCC"].idxmax()
    st.success(f"Best model by MCC on this data: **{best}**")
    st.download_button("Download table as CSV", comp.to_csv().encode(), "model_comparison.csv")

#  tab 5: predictions 
with tab5:
    out = data.copy()
    out["Predicted"] = le.inverse_transform(pred)
    out["Correct"] = out["Predicted"] == out[TARGET].astype(str)
    show = st.radio("Show", ["All rows", "Only wrong predictions"], horizontal=True)
    if show == "Only wrong predictions":
        out = out[~out["Correct"]]
    st.write(f"{len(out):,} rows")
    st.dataframe(out, width="stretch")
    st.download_button("Download predictions", out.to_csv(index=False).encode(), "predictions.csv")

    st.subheader("Explore the data")
    c1, c2 = st.columns(2)
    xf = c1.selectbox("X axis", features, index=0)
    yf = c2.selectbox("Y axis", features, index=min(2, len(features) - 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=data, x=xf, y=yf, hue=TARGET, alpha=0.6, ax=ax)
    st.pyplot(fig); plt.close(fig)
