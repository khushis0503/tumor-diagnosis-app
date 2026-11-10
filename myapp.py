import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load and Train Model
# -----------------------------
cancer = load_breast_cancer()

# Use 4 key cell-nucleus measurements as input features
features = ["mean radius", "mean texture", "mean smoothness", "mean concave points"]
feature_idx = [list(cancer.feature_names).index(f) for f in features]
X = cancer.data[:, feature_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    cancer.target,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Breast Tumor Diagnosis Classifier",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Breast Tumor Diagnosis Classifier")

st.write(
    "Enter the cell nucleus measurements below and click **Predict** to classify the tumor as Benign or Malignant."
)

st.write(f"### Model Accuracy: **{accuracy:.2%}**")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
mean_radius = st.number_input(
    "Mean Radius",
    min_value=5.0,
    max_value=30.0,
    value=13.54
)

mean_texture = st.number_input(
    "Mean Texture",
    min_value=5.0,
    max_value=40.0,
    value=14.36
)

mean_smoothness = st.number_input(
    "Mean Smoothness",
    min_value=0.0,
    max_value=0.2,
    value=0.0978,
    format="%.4f"
)

mean_concave_points = st.number_input(
    "Mean Concave Points",
    min_value=0.0,
    max_value=0.25,
    value=0.0478,
    format="%.4f"
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Diagnosis"):

    prediction = model.predict([[
        mean_radius,
        mean_texture,
        mean_smoothness,
        mean_concave_points
    ]])

    probability = model.predict_proba([[
        mean_radius,
        mean_texture,
        mean_smoothness,
        mean_concave_points
    ]])

    diagnosis = cancer.target_names[prediction[0]]

    st.success(f"Predicted Diagnosis: **{diagnosis.upper()}**")

    st.subheader("Prediction Confidence")

    st.write(
        {
            cancer.target_names[i]: f"{probability[0][i]*100:.2f}%"
            for i in range(len(cancer.target_names))
        }
    )

    st.progress(float(max(probability[0])))

st.divider()

st.caption("Developed using Streamlit and Scikit-Learn | For educational purposes only")
