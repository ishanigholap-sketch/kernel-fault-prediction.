
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

st.set_page_config(
    page_title="Kernel Fault Prediction",
    page_icon="⚙️",
    layout="wide",
)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00601/ai4i2020.csv"
)

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET = "Machine failure"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)


@st.cache_resource
def train_models():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Linear SVM": SVC(
            kernel="linear",
            class_weight="balanced",
        ),
        "RBF SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
        ),
    }

    trained = {}
    for name, classifier in models.items():
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", classifier),
        ])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        trained[name] = {
            "model": model,
            "y_test": y_test,
            "predictions": predictions,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ),
        }

    return df, trained


st.title("⚙️ Kernel-Based Machine Fault Prediction")
st.caption(
    "Machine learning for predictive maintenance | "
    "Linear SVM vs RBF SVM"
)

try:
    df, results = train_models()
except Exception as e:
    st.error(f"Could not load data or train models: {e}")
    st.info("Check your internet connection and try again.")
    st.stop()

st.success("Dataset loaded and both models trained successfully.")

st.subheader("Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total observations", f"{len(df):,}")
c2.metric("Input features", len(FEATURES))
c3.metric("Recorded failures", int(df[TARGET].sum()))

st.dataframe(df[FEATURES + [TARGET]].head(10), use_container_width=True)

st.subheader("Model evaluation")
selected = st.selectbox("Select classifier", list(results.keys()))
result = results[selected]

a, b, c, d = st.columns(4)
a.metric("Accuracy", f"{result['accuracy']:.2%}")
b.metric("Precision", f"{result['precision']:.2%}")
c.metric("Recall", f"{result['recall']:.2%}")
d.metric("F1 score", f"{result['f1']:.2%}")

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    result["matrix"],
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Failure"],
    yticklabels=["Normal", "Failure"],
    ax=ax,
)
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
ax.set_title(f"Confusion Matrix — {selected}")
st.pyplot(fig)
plt.close(fig)

tn, fp, fn, tp = result["matrix"].ravel()

st.subheader("Error analysis")
e1, e2, e3, e4 = st.columns(4)
e1.metric("True negatives", int(tn))
e2.metric("False positives · Type I", int(fp))
e3.metric("False negatives · Type II", int(fn))
e4.metric("True positives", int(tp))

st.subheader("Make a prediction")
st.write(
    "Enter machine operating measurements to estimate "
    "whether the machine resembles the recorded failure class."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.number_input(
            "Air temperature (K)", 290.0, 310.0, 300.0
        )
        process_temp = st.number_input(
            "Process temperature (K)", 300.0, 320.0, 310.0
        )
        rpm = st.number_input(
            "Rotational speed (rpm)", 1000, 3000, 1500
        )

    with col2:
        torque = st.number_input(
            "Torque (Nm)", 3.0, 80.0, 40.0
        )
        wear = st.number_input(
            "Tool wear (min)", 0, 300, 100
        )

    submitted = st.form_submit_button("Predict machine condition")

if submitted:
    sample = pd.DataFrame(
        [[air_temp, process_temp, rpm, torque, wear]],
        columns=FEATURES,
    )
    prediction = result["model"].predict(sample)[0]

    if prediction == 1:
        st.warning(
            "Predicted class: FAILURE. "
            "This is a model prediction, not a confirmed diagnosis."
        )
    else:
        st.success(
            "Predicted class: NORMAL. "
            "This is a model prediction, not a guarantee of safety."
        )

st.subheader("Feature correlations")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.heatmap(
    df[FEATURES + [TARGET]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax2,
)
st.pyplot(fig2)
plt.close(fig2)

st.caption(
    "Educational prototype using recorded data. "
    "Predictions are not validated for real industrial safety."
)
