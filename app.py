"""
Cardiovascular Disease (CVD) Risk Predictor - Streamlit Web Application
========================================================================

A clean, modern medical screening tool to evaluate cardiovascular risk
using a trained Machine Learning Pipeline (Support Vector Machine / SGD).

How to run this application:
    1. Ensure dependencies are installed:
       pip install streamlit pandas numpy scikit-learn joblib
       (Or activate your conda/virtual environment)
    2. Run from your terminal:
       streamlit run app.py
    3. The application will open automatically in your default web browser
       (usually at http://localhost:8501).
"""

import os
import sys
import types
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. PIPELINE CUSTOM FUNCTION DEFINITIONS
# ==============================================================================
# The saved scikit-learn pipeline references custom transformer functions
# ('data_preprocessing' and 'identify_obesity') serialized from __main__.
# We define them here and ensure they are registered in sys.modules so that
# pickle / joblib unpickles the pipeline seamlessly.

def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses input data: converts age if needed, drops duplicates and id column."""
    df = df.copy()
    
    # If age is given in days (> 120), convert to years; if already in years, preserve it
    if 'age' in df.columns:
        df['age'] = df['age'].apply(lambda x: x // 365 if x > 120 else x)
        
    # Drop duplicates if any exist
    df = df.drop_duplicates()
    
    # Safely drop 'id' column if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    return df


def identify_obesity(df: pd.DataFrame, bmi_threshold: float = 30.0) -> pd.DataFrame:
    """Calculates BMI from height (cm) and weight (kg) and flags obesity status."""
    df = df.copy()
    if 'height' in df.columns and 'weight' in df.columns:
        height_m = df['height'] / 100.0
        # Prevent division by zero
        height_m = np.where(height_m <= 0, 1.0, height_m)
        df['bmi'] = df['weight'] / (height_m ** 2)
        df['is_obese'] = np.where(df['bmi'] >= bmi_threshold, 1, 0)
    return df


# Register functions into __main__ and calc_obesity modules so unpickler resolves them
main_mod = sys.modules.get('__main__')
if main_mod:
    setattr(main_mod, 'data_preprocessing', data_preprocessing)
    setattr(main_mod, 'identify_obesity', identify_obesity)

if 'calc_obesity' not in sys.modules:
    calc_mod = types.ModuleType('calc_obesity')
    calc_mod.identify_obesity = identify_obesity
    sys.modules['calc_obesity'] = calc_mod


# ==============================================================================
# 2. STREAMLIT PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Cardiovascular Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern medical styling (soft blues, teals, elegant cards)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 50%, #0f766e 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(14, 165, 233, 0.25), 0 8px 10px -6px rgba(14, 165, 233, 0.2);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        opacity: 0.92;
        max-width: 800px;
        line-height: 1.5;
    }

    /* Section Cards */
    .card-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card-container:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .card-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #f1f5f9;
    }

    /* Live Metric Badges */
    .live-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .badge-normal {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .badge-warning {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .badge-danger {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    /* Outcome Result Banners */
    .result-card-high {
        background: linear-gradient(135deg, #fff1f2 0%, #fee2e2 100%);
        border: 2px solid #f87171;
        border-radius: 18px;
        padding: 1.8rem;
        color: #7f1d1d;
        margin-top: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.2);
    }
    .result-card-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #4ade80;
        border-radius: 18px;
        padding: 1.8rem;
        color: #14532d;
        margin-top: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(34, 197, 94, 0.2);
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background: #f8fafc;
        border-left: 4px solid #0284c7;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        color: #475569;
        font-size: 0.88rem;
        line-height: 1.5;
        margin-top: 2rem;
    }

    /* Custom Streamlit Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.75rem 2.5rem;
        border-radius: 12px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 14px 0 rgba(2, 132, 199, 0.39);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.45);
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 3. PIPELINE LOADING WITH CACHING
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_cvd_pipeline():
    """
    Loads the trained SVM machine learning pipeline securely with st.cache_resource.
    Directly targets 'trained_svm_pipeline.pkl'.
    """
    model_filename = "trained_svm_pipeline.pkl"
    candidate_paths = [
        model_filename,
        os.path.join(os.path.dirname(__file__), model_filename),
    ]
    
    pipeline_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            pipeline_path = p
            break
            
    if not pipeline_path:
        raise FileNotFoundError(
            f"Trained pipeline model file '{model_filename}' not found! "
            "Please ensure 'trained_svm_pipeline.pkl' is located in the application directory."
        )

    # Attempt loading using joblib, fallback to pickle
    try:
        pipeline = joblib.load(pipeline_path)
    except Exception:
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)
            
    return pipeline, os.path.basename(pipeline_path)


# ==============================================================================
# 4. CLINICAL HELPER UTILITIES
# ==============================================================================
def calculate_bmi(height_cm: float, weight_kg: float) -> tuple[float, str, str]:
    """Calculates BMI and returns value, clinical category, and style badge."""
    if height_cm <= 0:
        return 0.0, "Invalid Height", "badge-warning"
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 18.5:
        category, badge = "Underweight", "badge-warning"
    elif 18.5 <= bmi < 25.0:
        category, badge = "Normal weight", "badge-normal"
    elif 25.0 <= bmi < 30.0:
        category, badge = "Overweight", "badge-warning"
    else:
        category, badge = "Obese (BMI ≥ 30)", "badge-danger"
    return round(bmi, 1), category, badge


def classify_blood_pressure(systolic: int, diastolic: int) -> tuple[str, str]:
    """Classifies blood pressure based on AHA/ACC clinical guidelines."""
    if systolic < 120 and diastolic < 80:
        return "Normal Blood Pressure (< 120/80)", "badge-normal"
    elif 120 <= systolic <= 129 and diastolic < 80:
        return "Elevated Blood Pressure (120-129 / <80)", "badge-warning"
    elif (130 <= systolic <= 139) or (80 <= diastolic <= 89):
        return "Hypertension Stage 1 (130-139 / 80-89)", "badge-danger"
    elif systolic >= 140 or diastolic >= 90:
        return "Hypertension Stage 2 (≥ 140 / ≥ 90)", "badge-danger"
    else:
        return "Blood Pressure Reading", "badge-warning"


# ==============================================================================
# 5. SIDEBAR: MODEL OVERVIEW & PRESET PATIENT PROFILES
# ==============================================================================
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/cardiology.png",
        width=70
    )
    st.markdown("### **CVD Risk Assessment**")
    st.markdown(
        "This diagnostic screening app uses an end-to-end Machine Learning Pipeline "
        "trained on the **Cardiovascular Disease Dataset** (70,000 patient records)."
    )
    st.markdown("---")
    
    # Model Loading Status
    try:
        model, model_filename = load_cvd_pipeline()
        st.success(f"🟢 **Model Active**: `{model_filename}`")
    except Exception as e:
        st.error(f"🔴 **Model Error**: {e}")
        st.stop()
        
    st.markdown("#### **Quick Load Profiles**")
    st.caption("Populate the form with clinical patient test cases:")
    
    col_demo1, col_demo2 = st.columns(2)
    load_healthy = col_demo1.button("🏃 Healthy", help="Load young active patient with normal vitals")
    load_risk = col_demo2.button("⚠️ High Risk", help="Load patient with hypertension and elevated lipids")
    
    st.markdown("---")
    st.markdown("#### **Clinical Reference Ranges**")
    st.markdown("""
    - **Systolic BP**: Normal < 120 mmHg
    - **Diastolic BP**: Normal < 80 mmHg
    - **Healthy BMI**: 18.5 – 24.9 kg/m²
    - **Cholesterol**: 1: Normal, 2: Above Normal, 3: High
    - **Glucose**: 1: Normal, 2: Above Normal, 3: High
    """)
    st.markdown("---")
    st.caption("Cardiovascular Machine Learning System v2.0")


# Preset default values based on sidebar clicks
if load_healthy:
    st.session_state["age"] = 35
    st.session_state["gender"] = "Female"
    st.session_state["height"] = 168
    st.session_state["weight"] = 62.0
    st.session_state["ap_hi"] = 115
    st.session_state["ap_lo"] = 75
    st.session_state["cholesterol"] = "1: Normal"
    st.session_state["gluc"] = "1: Normal"
    st.session_state["smoke"] = "No"
    st.session_state["alco"] = "No"
    st.session_state["active"] = "Yes"
elif load_risk:
    st.session_state["age"] = 60
    st.session_state["gender"] = "Male"
    st.session_state["height"] = 172
    st.session_state["weight"] = 94.0
    st.session_state["ap_hi"] = 155
    st.session_state["ap_lo"] = 98
    st.session_state["cholesterol"] = "3: Well Above Normal"
    st.session_state["gluc"] = "2: Above Normal"
    st.session_state["smoke"] = "Yes"
    st.session_state["alco"] = "Yes"
    st.session_state["active"] = "No"


# ==============================================================================
# 6. APPLICATION HEADER BANNER
# ==============================================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">
        <span>🫀</span> Cardiovascular Disease Risk Predictor
    </div>
    <div class="hero-subtitle">
        Enter patient biometric, hemodynamic, and lifestyle indicators below to predict 
        the risk of cardiovascular disease using our trained Support Vector Machine pipeline.
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 7. INPUT FORM SECTION
# ==============================================================================
col_demographics, col_vitals, col_lifestyle = st.columns(3)

# ------------------------------------------------------------------------------
# Column 1: Patient Demographics & Body Metrics
# ------------------------------------------------------------------------------
with col_demographics:
    st.markdown("""
    <div class="card-header">
        <span>👤</span> Patient Demographics
    </div>
    """, unsafe_allow_html=True)
    
    age = st.slider(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=st.session_state.get("age", 52),
        help="Patient age in full years."
    )
    
    gender_choice = st.selectbox(
        "Gender",
        options=["Female", "Male"],
        index=0 if st.session_state.get("gender", "Female") == "Female" else 1,
        help="Female (encoded as 1) / Male (encoded as 2) matching dataset standards."
    )
    gender_val = 1 if gender_choice == "Female" else 2
    
    height = st.number_input(
        "Height (cm)",
        min_value=100,
        max_value=220,
        value=st.session_state.get("height", 165),
        step=1,
        help="Patient standing height in centimeters."
    )
    
    weight = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=float(st.session_state.get("weight", 70.0)),
        step=0.5,
        format="%.1f",
        help="Patient weight in kilograms."
    )
    
    # Dynamic BMI Calculation Badge
    current_bmi, bmi_category, bmi_badge_class = calculate_bmi(height, weight)
    st.markdown(
        f'<div class="live-badge {bmi_badge_class}">'
        f'⚖️ BMI: <b>{current_bmi} kg/m²</b> ({bmi_category})'
        f'</div>',
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------------------
# Column 2: Clinical & Vitals Measurements
# ------------------------------------------------------------------------------
with col_vitals:
    st.markdown("""
    <div class="card-header">
        <span>🩺</span> Clinical & Vitals
    </div>
    """, unsafe_allow_html=True)
    
    ap_hi = st.number_input(
        "Systolic Blood Pressure (ap_hi, mmHg)",
        min_value=70,
        max_value=240,
        value=st.session_state.get("ap_hi", 120),
        step=1,
        help="Upper blood pressure measurement during heart contraction."
    )
    
    ap_lo = st.number_input(
        "Diastolic Blood Pressure (ap_lo, mmHg)",
        min_value=40,
        max_value=160,
        value=st.session_state.get("ap_lo", 80),
        step=1,
        help="Lower blood pressure measurement between heartbeats."
    )
    
    # Live Blood Pressure Classification Badge
    bp_category, bp_badge_class = classify_blood_pressure(ap_hi, ap_lo)
    st.markdown(
        f'<div class="live-badge {bp_badge_class}">'
        f'🫀 {bp_category}'
        f'</div>',
        unsafe_allow_html=True
    )
    
    cholesterol_options = [
        "1: Normal",
        "2: Above Normal",
        "3: Well Above Normal"
    ]
    cholesterol_choice = st.selectbox(
        "Cholesterol Level",
        options=cholesterol_options,
        index=cholesterol_options.index(st.session_state.get("cholesterol", "1: Normal")),
        help="Serum total cholesterol clinical category."
    )
    cholesterol_val = int(cholesterol_choice.split(":")[0])
    
    glucose_options = [
        "1: Normal",
        "2: Above Normal",
        "3: Well Above Normal"
    ]
    gluc_choice = st.selectbox(
        "Glucose Level",
        options=glucose_options,
        index=glucose_options.index(st.session_state.get("gluc", "1: Normal")),
        help="Fasting blood glucose clinical category."
    )
    gluc_val = int(gluc_choice.split(":")[0])

# ------------------------------------------------------------------------------
# Column 3: Lifestyle & Behavioral Factors
# ------------------------------------------------------------------------------
with col_lifestyle:
    st.markdown("""
    <div class="card-header">
        <span>🏃</span> Lifestyle & Habits
    </div>
    """, unsafe_allow_html=True)
    
    smoke_choice = st.radio(
        "Tobacco Smoking",
        options=["No", "Yes"],
        index=0 if st.session_state.get("smoke", "No") == "No" else 1,
        horizontal=True,
        help="Does the patient smoke tobacco cigarettes?"
    )
    smoke_val = 1 if smoke_choice == "Yes" else 0
    
    alco_choice = st.radio(
        "Alcohol Intake",
        options=["No", "Yes"],
        index=0 if st.session_state.get("alco", "No") == "No" else 1,
        horizontal=True,
        help="Does the patient consume alcoholic beverages regularly?"
    )
    alco_val = 1 if alco_choice == "Yes" else 0
    
    active_choice = st.radio(
        "Physical Activity",
        options=["Yes", "No"],
        index=0 if st.session_state.get("active", "Yes") == "Yes" else 1,
        horizontal=True,
        help="Engages in regular exercise or moderate physical activity?"
    )
    active_val = 1 if active_choice == "Yes" else 0

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# 8. PREDICTION EXECUTION & RESULTS
# ==============================================================================
predict_btn = st.button("🔍 Predict Cardiovascular Risk", use_container_width=True)

if predict_btn:
    # 1. Assemble DataFrame with exact columns expected by the training pipeline
    input_df = pd.DataFrame([{
        'age': int(age),
        'gender': int(gender_val),
        'height': int(height),
        'weight': float(weight),
        'ap_hi': int(ap_hi),
        'ap_lo': int(ap_lo),
        'cholesterol': int(cholesterol_val),
        'gluc': int(gluc_val),
        'smoke': int(smoke_val),
        'alco': int(alco_val),
        'active': int(active_val)
    }])
    
    with st.spinner("Analyzing biometric markers and computing cardiovascular risk..."):
        try:
            # 2. Model Prediction
            prediction = int(model.predict(input_df)[0])
            
            # 3. Model Probability Score
            # If model supports predict_proba (e.g. LogisticRegression or SVC with probability=True), use it.
            # Otherwise for standard SVC, compute probability using sigmoid of the decision function.
            try:
                probs = model.predict_proba(input_df)
                prob_risk = float(probs[0][1])
            except (AttributeError, NotImplementedError):
                decision_score = float(model.decision_function(input_df)[0])
                prob_risk = float(1.0 / (1.0 + np.exp(-decision_score)))
                
            prob_percent = round(prob_risk * 100, 1)
            prob_healthy = round((1.0 - prob_risk) * 100, 1)
            
        except Exception as err:
            st.error(f"❌ Error during model inference: {err}")
            st.stop()
            
    # 4. Results Presentation
    st.markdown("---")
    st.subheader("📋 Cardiovascular Risk Assessment Report")
    
    if prediction == 1 or prob_risk >= 0.50:
        # High Risk Banner
        st.markdown(f"""
        <div class="result-card-high">
            <h2 style="margin: 0; display: flex; align-items: center; gap: 0.6rem; color: #991b1b;">
                ⚠️ High Risk of Cardiovascular Disease (CVD)
            </h2>
            <p style="font-size: 1.05rem; margin-top: 0.8rem; margin-bottom: 0; color: #7f1d1d;">
                The machine learning model identifies elevated cardiovascular risk based on the submitted clinical profile.
                Estimated probability of cardiovascular disease: <b>{prob_percent}%</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Low Risk / Healthy Banner
        st.markdown(f"""
        <div class="result-card-low">
            <h2 style="margin: 0; display: flex; align-items: center; gap: 0.6rem; color: #166534;">
                ✅ Low Risk / Healthy Cardiovascular Profile
            </h2>
            <p style="font-size: 1.05rem; margin-top: 0.8rem; margin-bottom: 0; color: #14532d;">
                The machine learning model identifies low cardiovascular risk based on the submitted indicators.
                Estimated healthy probability score: <b>{prob_healthy}%</b> (CVD risk: <b>{prob_percent}%</b>).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 5. Key Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Risk Probability",
            value=f"{prob_percent}%",
            delta=f"{'+' if prob_percent >= 50 else '-'}{abs(prob_percent - 50):.1f}% from baseline",
            delta_color="inverse"
        )
    with m2:
        st.metric(
            label="Blood Pressure",
            value=f"{ap_hi} / {ap_lo} mmHg",
            delta="Normal" if ap_hi < 120 and ap_lo < 80 else "Elevated",
            delta_color="normal" if ap_hi < 120 and ap_lo < 80 else "inverse"
        )
    with m3:
        st.metric(
            label="Calculated BMI",
            value=f"{current_bmi} kg/m²",
            delta=bmi_category,
            delta_color="normal" if 18.5 <= current_bmi < 25.0 else "inverse"
        )
    with m4:
        st.metric(
            label="Model Prediction",
            value="Positive (Risk)" if prediction == 1 else "Negative (Normal)"
        )
        
    # Visual Probability Bar
    st.markdown("##### **Probability Distribution**")
    st.progress(min(max(prob_risk, 0.0), 1.0))
    st.caption(f"Low Risk: {prob_healthy}% | High Risk: {prob_percent}%")
    
    # 6. Clinical Risk Factors & Actionable Insights
    with st.expander("🔍 View Detailed Factor Analysis & Recommendations", expanded=True):
        risk_flags = []
        positive_flags = []
        
        if ap_hi >= 140 or ap_lo >= 90:
            risk_flags.append(f"Stage 2 Hypertension measured ({ap_hi}/{ap_lo} mmHg).")
        elif ap_hi >= 130 or ap_lo >= 80:
            risk_flags.append(f"Stage 1 Hypertension measured ({ap_hi}/{ap_lo} mmHg).")
        else:
            positive_flags.append("Blood pressure within optimal reference range.")
            
        if cholesterol_val >= 2:
            risk_flags.append(f"Elevated total cholesterol level ({cholesterol_choice}).")
        else:
            positive_flags.append("Normal cholesterol level.")
            
        if gluc_val >= 2:
            risk_flags.append(f"Elevated blood glucose level ({gluc_choice}).")
        else:
            positive_flags.append("Normal fasting glucose.")
            
        if current_bmi >= 30.0:
            risk_flags.append(f"Obesity indicated with Body Mass Index of {current_bmi} kg/m².")
        elif current_bmi >= 25.0:
            risk_flags.append(f"Overweight status with Body Mass Index of {current_bmi} kg/m².")
        else:
            positive_flags.append("Body Mass Index in healthy range.")
            
        if smoke_val == 1:
            risk_flags.append("Active tobacco smoking is a major independent cardiovascular risk factor.")
        else:
            positive_flags.append("Non-smoker.")
            
        if active_val == 0:
            risk_flags.append("Sedentary lifestyle with lack of regular physical exercise.")
        else:
            positive_flags.append("Engages in regular physical activity.")
            
        c_flags1, c_flags2 = st.columns(2)
        with c_flags1:
            st.markdown("##### ⚠️ Risk Factors Identified")
            if risk_flags:
                for rf in risk_flags:
                    st.markdown(f"- 🔴 {rf}")
            else:
                st.markdown("✅ No significant clinical risk flags identified.")
                
        with c_flags2:
            st.markdown("##### 🛡️ Protective / Normal Markers")
            if positive_flags:
                for pf in positive_flags:
                    st.markdown(f"- 🟢 {pf}")
            else:
                st.markdown("No protective markers flagged.")

# ==============================================================================
# 9. MEDICAL DISCLAIMER
# ==============================================================================
st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ Medical Disclaimer:</b><br>
    This web application is powered by an experimental Machine Learning algorithm trained on population data.
    It is provided for educational, research, and informational screening purposes only.
    This tool does <b>not</b> constitute medical advice, clinical diagnosis, or treatment recommendations.
    Always consult a qualified healthcare provider, cardiologist, or physician regarding personal medical questions or symptoms.
</div>
""", unsafe_allow_html=True)
