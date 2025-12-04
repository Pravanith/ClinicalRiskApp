import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import re
import datetime

# --- IMPORT SAFETY: Dependency Handling ---
# Prevents app crash if drug_data.py is missing
try:
    from drug_data import INTERACTION_DB
except ImportError:
    print("⚠️ Warning: drug_data.py not found. Interaction checker will use empty DB.")
    INTERACTION_DB = {}

# ==========================================
# 1. ADVANCED CLINICAL STANDARDS & CALCULATORS
# ==========================================
class ClinicalStandards:
    """
    Central Logic for Gender-Specific Reference Ranges and Unit Conversions.
    Sources: KDIGO (Renal), WHO (Anemia), CredibleMeds (QTc).
    """
    @staticmethod
    def get_thresholds(gender):
        """Returns a dictionary of safety limits based on gender."""
        is_female = (gender == "Female")
        
        return {
            # --- HEMATOLOGY ---
            # Hgb: Men 13.5-17.5, Women 12.0-15.5 (WHO Criteria)
            'hgb_low': 12.0 if is_female else 13.5, 
            'hgb_high': 15.5 if is_female else 17.5,
            
            # Hematocrit: Typically 3x Hemoglobin
            'hct_low': 36.0 if is_female else 41.0,
            
            # WBC: 4.5 - 11.0 (Generally gender neutral in acute triage)
            'wbc_low': 4.5, 'wbc_high': 11.0,
            
            # Platelets: 150 - 450
            'plt_low': 150, 'plt_high': 450,

            # --- CHEMISTRY (RENAL & LYTES) ---
            # SCr: Women have lower muscle mass -> lower "normal" creatinine threshold
            'creat_high': 1.0 if is_female else 1.3,
            
            # BUN: 7-20 mg/dL
            'bun_high': 20.0,
            
            # Potassium: 3.5 - 5.0 mmol/L (Critical Alert)
            'k_low': 3.5, 'k_high': 5.0,
            
            # Glucose: 70-100 fasting, >180 random concern, >400 Critical
            'glu_low': 70.0, 'glu_high': 180.0,
            
            # --- CARDIOLOGY ---
            # QTc: Women have naturally longer QTc intervals (>460 vs >450)
            'qtc_limit': 460 if is_female else 450,
            
            # --- VITALS ---
            'sys_bp_high': 140, 'dia_bp_high': 90,
            'hr_high': 100, 'hr_low': 60,
            'rr_high': 20, 'rr_low': 12,
            'spo2_low': 95
        }

    @staticmethod
    def convert_units(weight, w_unit, temp, t_unit):
        """Standardizes inputs to Metric (Kg and Celsius) for internal math."""
        # Weight Conversion
        if w_unit == "lbs":
            weight_kg = weight * 0.453592
        else:
            weight_kg = weight
            
        # Temperature Conversion
        if t_unit == "°F":
            temp_c = (temp - 32) * 5.0/9.0
        else:
            temp_c = temp
            
        return weight_kg, temp_c

    @staticmethod
    def calculate_ibw(height_cm, gender, actual_weight_kg):
        """
        Calculates Ideal Body Weight (Devine Formula).
        Essential for dosing in obese patients.
        """
        if height_cm <= 0: return actual_weight_kg 
        
        height_inches = height_cm / 2.54
        base_weight = 45.5 if gender == "Female" else 50.0
        
        if height_inches > 60:
            ibw = base_weight + (2.3 * (height_inches - 60))
        else:
            ibw = base_weight
            
        return ibw

    @staticmethod
    def calculate_crcl(age, actual_weight_kg, height_cm, gender, creat):
        """
        Cockcroft-Gault Equation with Advanced Weight Logic.
        - Uses Actual Weight if Underweight.
        - Uses Adjusted Body Weight if Obese (BMI > 30).
        - Uses Ideal Body Weight otherwise.
        - Applies 0.85 multiplier for Females.
        """
        if creat <= 0 or age == 0: return 0, "N/A"
        
        ibw = ClinicalStandards.calculate_ibw(height_cm, gender, actual_weight_kg)
        
        # BMI Calculation
        bmi = actual_weight_kg / ((height_cm/100)**2) if height_cm > 0 else 0
        
        # Weight Selection Logic
        if actual_weight_kg < ibw:
            dosing_weight = actual_weight_kg
            weight_type = "Actual (Underweight)"
        elif bmi > 30:
            # AdjBW = IBW + 0.4 * (Actual - IBW)
            dosing_weight = ibw + 0.4 * (actual_weight_kg - ibw)
            weight_type = "Adjusted (Obese)"
        else:
            dosing_weight = ibw
            weight_type = "Ideal"
            
        # Standard Formula
        constant = 0.85 if gender == "Female" else 1.0
        crcl = ((140 - age) * dosing_weight * constant) / (72 * creat)
        
        return int(crcl), weight_type

# ==========================================
# 2. DATABASE MANAGEMENT (Thread Safe)
# ==========================================
def get_db_connection():
    # check_same_thread=False is crucial for Streamlit
    return sqlite3.connect('clinical_data.db', check_same_thread=False)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patient_history (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            age INTEGER,
            gender TEXT,
            sbp INTEGER,
            aki_risk_score INTEGER,
            bleeding_risk_score REAL,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_patient_to_db(age, gender, sbp, aki, bleed, status):
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO patient_history (age, gender, sbp, aki_risk_score, bleeding_risk_score, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (age, gender, sbp, aki, bleed, status))
        conn.commit()

def fetch_history():
    if not os.path.exists('clinical_data.db'):
        return pd.DataFrame()
    with get_db_connection() as conn:
        df = pd.read_sql("SELECT * FROM patient_history ORDER BY timestamp DESC", conn)
    return df

def clear_history():
    with get_db_connection() as conn:
        conn.execute("DELETE FROM patient_history")
        conn.commit()

# ==========================================
# 3. ML MODEL LOADING (With Fallback)
# ==========================================
class HeuristicFallbackModel:
    """Fallback logic if XGBoost model file is missing."""
    def predict(self, df):
        risk = 10.0
        row = df.iloc[0]
        if row.get('age', 0) > 65: risk += 15
        if row.get('high_bp', 0) == 1: risk += 20
        if row.get('inr', 1.0) > 1.2: risk += 25
        if row.get('anticoagulant', 0) == 1: risk += 20
        return [min(risk, 95.0)]

def load_bleeding_model():
    model_file = "bleeding_risk_model.json"
    if os.path.exists(model_file):
        try:
            model = xgb.XGBRegressor()
            model.load_model(model_file)
            return model
        except Exception as e:
            print(f"⚠️ XGBoost Error: {e}. Using Fallback.")
            return HeuristicFallbackModel()
    return HeuristicFallbackModel()

# ==========================================
# 4. CLINICAL RISK CALCULATORS
# ==========================================
def calculate_aki_risk(age, diuretic, acei, sys_bp, chemo, creat, nsaid, heart_failure):
    score = 0
    score += 30 if diuretic else 0
    score += 40 if acei else 0
    score += 25 if nsaid else 0
    score += 15 if heart_failure else 0
    score += 20 if chemo else 0
    if age > 0: score += 20 if age > 75 else 0
    if sys_bp > 0:
        score += 10 if sys_bp > 160 else 0
        score += 20 if sys_bp < 90 else 0
    if creat > 0:
        if creat > 1.5: score += 30
        elif creat > 1.2: score += 15
    return min(score, 100)

def calculate_sepsis_risk(sys_bp, resp_rate, altered_mental, temp_c):
    qsofa = 0
    if sys_bp > 0 and sys_bp <= 100: qsofa += 1
    if resp_rate > 0 and resp_rate >= 22: qsofa += 1
    if altered_mental: qsofa += 1
    # Adding Temp to heuristic even though strictly not qSOFA
    if temp_c > 0 and (temp_c > 38.0 or temp_c < 36.0): qsofa += 0.5
    
    if qsofa >= 2: return 90
    if qsofa >= 1: return 45
    return 0

def calculate_hypoglycemic_risk(insulin, renal, hba1c_high, recent_dka):
    score = 0
    score += 30 if insulin else 0
    score += 45 if renal else 0
    score += 20 if hba1c_high else 0
    score += 20 if recent_dka else 0 
    return min(score, 100)

def calculate_sirs_score(temp_c, hr, resp_rate, wbc):
    score = 0
    if temp_c > 0 and (temp_c > 38 or temp_c < 36): score += 1
    if hr > 90: score += 1
    if resp_rate > 20: score += 1
    if wbc > 0 and (wbc > 12 or wbc < 4): score += 1
    return score

# ==========================================
# 5. INTERACTION CHECKER & GLOSSARY
# ==========================================
def normalize_text(text):
    if not isinstance(text, str): return ""
    return text.lower().strip()

def check_interaction(d1, d2):
    d1_clean = normalize_text(d1)
    d2_clean = normalize_text(d2)
    if (d1_clean, d2_clean) in INTERACTION_DB: return INTERACTION_DB[(d1_clean, d2_clean)]
    if (d2_clean, d1_clean) in INTERACTION_DB: return INTERACTION_DB[(d2_clean, d1_clean)]
    return None

def chatbot_response(text):
    """
    Simulated Glossary Function. 
    In full production, this would look up the KNOWLEDGE_BASE dictionary.
    """
    # Simple direct match simulation for key terms
    text = text.lower()
    if "rhabdo" in text: return "**Rhabdomyolysis:** Rapid breakdown of muscle tissue releasing myoglobin into blood. Risk of kidney failure."
    if "hyperkalemia" in text: return "**Hyperkalemia:** High Potassium (>5.0). Risk of cardiac arrhythmia/arrest."
    if "qt" in text: return "**QT Prolongation:** Delayed heart recharge time. Risk of Torsades de Pointes (Fatal arrhythmia)."
    if "sepsis" in text: return "**Sepsis:** Life-threatening organ dysfunction caused by a dysregulated host response to infection."
    
    # Fallback to AI if no local match
    return f"**Consulting AI for definition of:** {text}..."

# ==========================================
# 6. AI INTEGRATION (Google Gemini)
# ==========================================
def consult_ai_doctor(role, user_input, patient_context=None):
    import google.generativeai as genai
    import streamlit as st
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        if role == 'risk_assessment':
            prompt = f"""
            Act as a Senior ICU Consultant. Analyze this patient:
            - Data: {patient_context}
            Task: Identify primary threat and suggest 3 immediate actions.
            """
        elif role == 'provider':
             prompt = f"Expert Medical Consult. Query: {user_input}. Provide differential diagnosis."
        else:
             prompt = user_input
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

def generate_discharge_summary(patient_data):
    import google.generativeai as genai
    import streamlit as st
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"Write a professional hospital discharge summary for this patient data: {patient_data}"
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_drug_interactions(drug_list):
    import google.generativeai as genai
    import streamlit as st
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"Analyze drug interactions for: {', '.join(drug_list)}. detailed mechanism, severity, and management."
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error: {str(e)}"
