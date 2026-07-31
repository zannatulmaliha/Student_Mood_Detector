import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Mood Detector", page_icon="🎓")


@st.cache_resource
def load_model():
    return joblib.load("models/mood_model.joblib")


st.title("🎓 Student Mood Detector")
st.write("Estimate a student's depression risk from a few academic and behavioral factors.")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model not found. Run `python train_model.py` first to train and save it.")
    st.stop()

with st.form("mood_form"):
    age = st.number_input("Age", min_value=15, max_value=60, value=21)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    sleep = st.slider("Sleep Duration (hours/day)", 0.0, 12.0, 7.0, 0.1)
    pressure = st.slider("Academic Pressure (1-10)", 1, 10, 5)
    submitted = st.form_submit_button("Detect Mood")

if submitted:
    row = pd.DataFrame(
        [{"Age": age, "Sleep Duration": sleep, "Academic Pressure": pressure, "Gender": gender}]
    )
    proba = model.predict_proba(row)[0][1]

    if proba >= 0.5:
        st.error(f"⚠️ Higher risk of depression detected ({proba:.0%} probability).")
    else:
        st.success(f"🙂 Low risk of depression detected ({proba:.0%} probability).")

    st.caption("This is an educational demo, not a medical diagnosis.")
