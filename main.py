import streamlit as st
import pickle
import pandas as pd

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Student GPA Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student GPA Prediction App")
st.markdown(
    """
    Predict **Student GPA** using a trained **KNN Regression model**.
    The prediction is based on academic effort, support,
    and extracurricular activities.
    """
)

st.divider()

# --------------------------------------------------
# Load Model & Scaler
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("knn_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


knn, scaler = load_artifacts()

# --------------------------------------------------
# Feature Names (MUST MATCH TRAINING)
# --------------------------------------------------
FEATURE_NAMES = [
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "GradeClass",
]

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
st.subheader("📥 Enter Student Details")

study_time = st.number_input(
    "Study Time per Week (hours)",
    min_value=0.0,
    max_value=100.0,
    value=12.5,
    step=0.5
)

absences = st.number_input(
    "Number of Absences",
    min_value=0,
    max_value=100,
    value=3
)

tutoring = st.selectbox(
    "Tutoring Support",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

parental_support = st.slider(
    "Parental Support Level (1 = Low, 5 = High)",
    1, 5, 4
)

extracurricular = st.selectbox(
    "Extracurricular Activities",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

sports = st.selectbox(
    "Sports Participation",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

music = st.selectbox(
    "Music Participation",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

grade_class = st.selectbox(
    "Grade Class",
    [1, 2, 3, 4]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.divider()

if st.button("🎯 Predict GPA"):
    input_df = pd.DataFrame(
        [[
            study_time,
            absences,
            tutoring,
            parental_support,
            extracurricular,
            sports,
            music,
            grade_class
        ]],
        columns=FEATURE_NAMES
    )

    scaled_input = scaler.transform(input_df)
    prediction = knn.predict(scaled_input)

    st.success(f"📊 **Predicted GPA: {prediction[0]:.2f}**")

    st.caption(
        "Prediction generated using a KNN Regression model "
        "trained on historical student data."
    )

# --------------------------------------------------
# Debug Section
# --------------------------------------------------
with st.expander("🔍 View Input Data"):
    st.dataframe(
        input_df if "input_df" in locals()
        else pd.DataFrame(columns=FEATURE_NAMES)
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption("Machine Learning Project | KNN Regression | Streamlit")