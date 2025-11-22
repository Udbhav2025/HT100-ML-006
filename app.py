import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go # type: ignore
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Cardio Predictor", layout="wide")

# --- Load Model & Imputer ---
@st.cache_resource
def load_model():
    model = joblib.load('heart_model.pkl')
    imputer = joblib.load('imputer.pkl')
    return model, imputer

try:
    model, imputer = load_model()
except:
    st.error("Model not found! Please run 'train_model.py' first.")
    st.stop()

# --- UI Layout ---
st.title("🫀 The Cardio Predictor")
st.markdown("### AI-Powered Early Warning System for Heart Disease")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.header("Patient Data")
    st.write("Enter details below:")
    
    # Inputs corresponding to the 13 features
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], 
                      format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal", 4: "Asymptomatic"}[x])
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 200, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "True" if x == 1 else "False")
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2], format_func=lambda x: "Normal" if x == 0 else ("ST-T Abnormality" if x == 1 else "LV Hypertrophy"))
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST", options=[1, 2, 3])
    ca = st.number_input("Major Vessels (0-3)", 0, 3, 0)
    thal = st.selectbox("Thalassemia", options=[3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}[x])

    # Combine inputs
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Impute (just in case of weird inputs, though Streamlit handles types well)
    features_processed = imputer.transform(features)

with col2:
    st.header("Risk Analysis")
    
    if st.button("Calculate Risk Score", type="primary"):
        # Get Probability
        probability = model.predict_proba(features_processed)[0][1]
        risk_percentage = probability * 100
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_percentage,
            title = {'text': "Heart Disease Risk Probability"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Doctor's Note
        if risk_percentage < 40:
            st.success("✅ Low Risk: Patient appears healthy. Routine checkups recommended.")
        elif risk_percentage < 65:
            st.warning("⚠️ Moderate Risk: Lifestyle changes and further monitoring advised.")
        else:
            st.error("🚨 High Risk: Immediate specialist consultation recommended.")

    # Feature Importance (Explainability)
    st.subheader("Top Risk Factors (Model Logic)")
    importances = model.feature_importances_
    feature_names = ["Age", "Sex", "Chest Pain", "BP", "Cholesterol", "Sugar", "ECG", "Max HR", "Ex. Angina", "ST Depr.", "Slope", "Vessels", "Thal"]
    
    # Sort features
    indices = np.argsort(importances)[-5:] # Top 5
    
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(range(len(indices)), importances[indices], color='#FF4B4B')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Relative Importance")
    st.pyplot(fig)

with col3:
    st.info("ℹ️ **About this Tool**")
    st.markdown("""
    This system analyzes 13 key medical indicators to predict the likelihood of heart disease.
    
    **Key Features:**
    - 🤖 **Random Forest AI:** Robust classification.
    - 🛡️ **Missing Data Handling:** Uses statistical imputation.
    - 📊 **Visual Dashboard:** Instant interpretation for doctors.
    """)