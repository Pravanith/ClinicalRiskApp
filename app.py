import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import backend as bk
import datetime

# 1. PAGE CONFIG
st.set_page_config(page_title="Clinical Risk Monitor", page_icon="🛡️", layout="wide")

# --- NEW: INITIALIZE SESSION STATE (Prevents Empty Page) ---
keys = {
    'age': 0, 'gender': 'Male', 'weight_input': 0.0, 
    'sys_bp': 0, 'dia_bp': 0, 'hr': 0, 
    'creat': 0.0, 'anticoagulant': False,
    'entered_app': False
}
for key, default in keys.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Load AI Model (Mocked for reliability)
try:
    bleeding_model = bk.HeuristicFallbackModel() # Fallback for demo
except:
    st.error("Model failed to load.")

# --- MODULE 1: RISK CALCULATOR ---
def render_risk_calculator():
    st.subheader("Acute Risk Calculator")

    # AI SMART FILL SECTION
    with st.container(border=True):
        st.markdown("#### 🪄 AI Smart-Fill (Natural Language)")
        raw_note = st.text_area("Paste a patient description to auto-fill the form:", 
                                placeholder="e.g., A 72 year old female, BP is 150/95, HR 92, on anticoagulants.")
        if st.button("✨ Parse and Fill"):
            if raw_note:
                with st.spinner("AI is reading the note..."):
                    extracted = bk.parse_clinical_note(raw_note)
                    for k, v in extracted.items():
                        st.session_state[k] = v
                    st.rerun()

    # MANUAL ENTRY FORM
    with st.form("risk_form"):
        col_left, col_right = st.columns(2, gap="medium")
        
        with col_left:
            st.markdown("##### 👤 Patient Profile")
            age = st.number_input("Age (Years)", 0, 120, value=st.session_state['age'])
            gender_list = ["Male", "Female"]
            gender = st.selectbox("Gender", gender_list, index=gender_list.index(st.session_state['gender']))
            weight_input = st.number_input("Weight", 0.0, 400.0, value=st.session_state['weight_input'])
            
            st.markdown("##### 🩺 Vitals")
            sys_bp = st.number_input("Systolic BP", 0, 300, value=st.session_state['sys_bp'])
            dia_bp = st.number_input("Diastolic BP", 0, 200, value=st.session_state['dia_bp'])
            hr = st.number_input("Heart Rate", 0, 300, value=st.session_state['hr'])

        with col_right:
            st.markdown("##### 🧪 Critical Labs")
            creat = st.number_input("Creatinine", 0.0, 20.0, value=float(st.session_state['creat']))
            
            st.markdown("##### 📋 Medical History")
            anticoag = st.checkbox("Anticoagulant Use", value=st.session_state['anticoagulant'])
            liver_disease = st.checkbox("Liver Disease")
            
        submitted = st.form_submit_button("🚀 Run Clinical Analysis", type="primary", use_container_width=True)

    if submitted:
        # Perform calculations using the parsed/entered data
        pred_aki = bk.calculate_aki_risk(age, False, False, sys_bp, False, creat, False, False)
        # Store results for display
        st.session_state['last_result'] = {"aki": pred_aki, "status": "Critical" if pred_aki > 50 else "Stable"}
        st.success(f"Analysis Complete: Patient is {st.session_state['last_result']['status']}")

# --- MAIN APP FLOW ---
if not st.session_state['entered_app']:
    st.title("🛡️ Clinical Risk Monitor")
    if st.button("🚀 Launch Dashboard"):
        st.session_state['entered_app'] = True
        st.rerun()
else:
    menu = st.sidebar.radio("Navigation", ["Risk Calculator", "AI Consultant"])
    if menu == "Risk Calculator":
        render_risk_calculator()
    else:
        st.write("Consultant Module Active.")
