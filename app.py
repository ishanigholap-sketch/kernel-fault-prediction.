
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

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Kernel AI | Fault Prediction",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# CUSTOM UI STYLING
# ==================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: #171923;
        border-right: 1px solid #292D3A;
    }

    [data-testid="stSidebar"] h2 {
        color: #C4B5FD;
    }

    h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    h2, h3 {
        color: #D8D4F5;
    }

    [data-testid="stMetric"] {
        background-color: #191C27;
        border: 1px solid #303344;
        padding: 14px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #B8B9CB;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #303344;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# DATASET AND FEATURES
# ==================================================

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


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)


# ==================================================
# TRAIN BOTH SVM MODELS
# ==================================================

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

    classifiers = {
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

    for name, classifier in classifiers.items():
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
            "accuracy": accuracy_score(
                y_test, predictions
            ),
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
                y_test,
                predictions,
                labels=[0, 1],
            ),
        }

    return df, trained


# ==================================================
# LOAD APPLICATION DATA
# ==================================================

try:
    df, results = train_models()

except Exception as e:
    st.error(
        "Could not load the dataset or train the models."
    )
    st.caption(f"Technical details: {e}")
    st.info(
        "Check your internet connection and try again."
    )
    st.stop()


# ==================================================
# MODEL COMPARISON DATA
# ==================================================

comparison = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [
        results[name]["accuracy"] * 100
        for name in results
    ],
    "Precision": [
        results[name]["precision"] * 100
        for name in results
    ],
    "Recall": [
        results[name]["recall"] * 100
        for name in results
    ],
    "F1 Score": [
        results[name]["f1"] * 100
        for name in results
    ],
})


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:
    st.markdown("## ⚙️ KERNEL AI")
    st.caption("MACHINE HEALTH MONITOR")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Model Evaluation",
            "Predict",
            "Insights",
        ],
        format_func=lambda x: {
            "Overview": "📊  Overview",
            "Model Evaluation": "📈  Evaluation",
            "Predict": "🔍  Predict",
            "Insights": "💡  Insights",
        }[x],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("**PROJECT DETAILS**")
    st.caption("AI-powered predictive maintenance")
    st.caption("Linear SVM · RBF SVM")
    st.caption("Dataset: AI4I 2020")

    st.divider()

    st.caption("Educational prototype")


# ==================================================
# MAIN HEADER
# ==================================================

st.title("⚙️ Kernel AI")
st.caption(
    "Machine Fault Prediction  ·  Predictive Maintenance"
)

st.success(
    "Dataset loaded and both models trained successfully."
)


# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    st.subheader("Dashboard overview")

    total_records = len(df)
    total_failures = int(df[TARGET].sum())
    failure_rate = total_failures / total_records * 100

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total observations",
        f"{total_records:,}",
    )

    c2.metric(
        "Input features",
        len(FEATURES),
    )

    c3.metric(
        "Recorded failures",
        f"{total_failures:,}",
        f"{failure_rate:.2f}% of dataset",
    )

    st.divider()

    st.subheader("Model performance comparison")

    chart_data = comparison.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ]

    fig, ax = plt.subplots(figsize=(8, 3.6))

    chart_data.plot(
        kind="bar",
        ax=ax,
        color=[
            "#A78BFA",
            "#67C7B7",
            "#F2C879",
            "#8AB4F8",
        ],
        width=0.72,
    )

    ax.set_ylabel("Score (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 105)
    ax.set_title("Linear SVM vs RBF SVM", pad=12)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncols=4,
        frameon=False,
    )

    fig.tight_layout()
    st.pyplot(fig, width=720)
    plt.close(fig)

    st.caption(
        "Both models use the same stratified test split. "
        "Compare recall and precision as well as accuracy."
    )

    st.subheader("Model summary")

    summary = comparison.set_index("Model").copy()

    st.dataframe(
        summary.style.format("{:.2f}"),
        width="stretch",
    )

    with st.expander("Preview dataset"):
        st.dataframe(
            df[FEATURES + [TARGET]].head(10),
            width="stretch",
            hide_index=True,
        )


# ==================================================
# PAGE 2: MODEL EVALUATION
# ==================================================

elif page == "Model Evaluation":

    st.subheader("Model evaluation")

    selected = st.selectbox(
        "Select classifier",
        list(results.keys()),
    )

    result = results[selected]

    a, b, c, d = st.columns(4)

    a.metric(
        "Accuracy",
        f"{result['accuracy']:.2%}",
    )

    b.metric(
        "Precision",
        f"{result['precision']:.2%}",
    )

    c.metric(
        "Recall",
        f"{result['recall']:.2%}",
    )

    d.metric(
        "F1 score",
        f"{result['f1']:.2%}",
    )

    st.divider()

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("#### Confusion matrix")

        fig, ax = plt.subplots(figsize=(5, 3.5))

        sns.heatmap(
            result["matrix"],
            annot=True,
            fmt="d",
            cmap="Purples",
            xticklabels=["Normal", "Failure"],
            yticklabels=["Normal", "Failure"],
            cbar=False,
            linewidths=0.5,
            ax=ax,
        )

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_title(selected, pad=10)

        fig.tight_layout()
        st.pyplot(fig, width=520)
        plt.close(fig)

    with right:
        st.markdown("#### Error analysis")

        tn, fp, fn, tp = result["matrix"].ravel()

        st.metric("Correct normal predictions", int(tn))
        st.metric("Missed failures", int(fn))
        st.metric("False alarms", int(fp))
        st.metric("Correctly detected failures", int(tp))

    st.divider()

    st.markdown("#### Compare both models")

    st.dataframe(
        comparison.set_index("Model").style.format(
            "{:.2f}"
        ),
        width="stretch",
    )

    st.info(
        "A false negative occurs when an actual recorded "
        "failure is predicted as normal. This is particularly "
        "important in predictive maintenance."
    )


# ==================================================
# PAGE 3: PREDICTION
# ==================================================

elif page == "Predict":

    st.subheader("Predict machine condition")

    st.caption(
        "Enter operating measurements and select a trained model."
    )

    selected_model = st.selectbox(
        "Prediction model",
        list(results.keys()),
    )

    with st.form("prediction_form"):

        left, right = st.columns(2)

        with left:
            air_temp = st.number_input(
                "Air temperature (K)",
                min_value=290.0,
                max_value=310.0,
                value=300.0,
                step=0.1,
            )

            process_temp = st.number_input(
                "Process temperature (K)",
                min_value=300.0,
                max_value=320.0,
                value=310.0,
                step=0.1,
            )

            rpm = st.number_input(
                "Rotational speed (rpm)",
                min_value=1000,
                max_value=3000,
                value=1500,
                step=10,
            )

        with right:
            torque = st.number_input(
                "Torque (Nm)",
                min_value=3.0,
                max_value=80.0,
                value=40.0,
                step=0.5,
            )

            wear = st.number_input(
                "Tool wear (min)",
                min_value=0,
                max_value=300,
                value=100,
                step=1,
            )

        submitted = st.form_submit_button(
            "Predict machine condition",
            width="stretch",
        )

    if submitted:

        sample = pd.DataFrame(
            [[
                air_temp,
                process_temp,
                rpm,
                torque,
                wear,
            ]],
            columns=FEATURES,
        )

        model = results[selected_model]["model"]
        predicted_class = model.predict(sample)[0]

        st.divider()
        st.subheader("Prediction result")

        if predicted_class == 1:
            st.error(
                "⚠️ Predicted class: FAILURE\n\n"
                "The model classifies these measurements "
                "as a recorded failure case."
            )

            st.markdown(
                "**Suggested next step:** Inspect the operating "
                "conditions and arrange an appropriate machine "
                "inspection. Do not rely on this prediction alone."
            )

        else:
            st.success(
                "✅ Predicted class: NORMAL\n\n"
                "The model classifies these measurements "
                "as a recorded normal case."
            )

            st.markdown(
                "**Suggested next step:** Continue routine "
                "monitoring and follow the manufacturer's "
                "maintenance schedule."
            )

        st.caption(
            f"Model used: {selected_model}. "
            "This is an educational prediction, not a confirmed "
            "diagnosis or a guarantee of machine safety."
        )


# ==================================================
# PAGE 4: INSIGHTS
# ==================================================

elif page == "Insights":

    st.subheader("Dataset insights")

    st.caption(
        "Explore feature relationships and recorded failure patterns."
    )

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown("#### Feature correlation")

        fig, ax = plt.subplots(figsize=(6, 4.5))

        sns.heatmap(
            df[FEATURES + [TARGET]].corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            cbar=False,
            annot_kws={"size": 8},
            ax=ax,
        )

        ax.set_title("Feature correlation heatmap", pad=10)
        ax.tick_params(axis="x", labelrotation=35)
        ax.tick_params(axis="y", labelrotation=0)

        fig.tight_layout()
        st.pyplot(fig, width=620)
        plt.close(fig)

    with right:
        st.markdown("#### Class distribution")

        normal_count = int((df[TARGET] == 0).sum())
        failure_count = int((df[TARGET] == 1).sum())

        st.metric("Normal records", f"{normal_count:,}")
        st.metric("Failure records", f"{failure_count:,}")

        st.metric(
            "Recorded failure rate",
            f"{failure_count / len(df):.2%}",
        )

        st.warning(
            "The dataset is imbalanced. A model can achieve "
            "high accuracy while still missing important failures."
        )

    st.divider()

    st.markdown("#### Input features")

    feature_info = pd.DataFrame({
        "Feature": [
            "Air temperature",
            "Process temperature",
            "Rotational speed",
            "Torque",
            "Tool wear",
        ],
        "Unit": [
            "Kelvin (K)",
            "Kelvin (K)",
            "Revolutions per minute (rpm)",
            "Newton-metre (Nm)",
            "Minutes",
        ],
    })

    st.dataframe(
        feature_info,
        width="stretch",
        hide_index=True,
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "KERNEL AI  ·  AI4I 2020 dataset  ·  Linear SVM + RBF SVM"
)

st.caption(
    "Educational prototype using recorded data. "
    "Predictions have not been validated for real industrial safety."
)

