import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import backend as bk
import random
import datetime
import json
import google.generativeai as genai
from fpdf import FPDF
import base64

# ==========================================
# 1. CLINICAL GUIDELINE CLASSIFIERS (2025 STANDARDS)
# ==========================================

def get_bp_category(sbp, dbp):
    """Source: 2025 ACC/AHA High Blood Pressure Guidelines"""
    if sbp < 90 or dbp < 60: return "Hypotension (Shock Risk)", "red" 
    elif sbp < 120 and dbp < 80: return "Normal", "green"
    elif 120 <= sbp <= 129 and dbp < 80: return "Elevated", "orange"
    elif (130 <= sbp <= 139) or (80 <= dbp <= 89): return "Stage 1 Hypertension", "orange"
    elif sbp >= 180 or dbp >= 120: return "Severe Hypertension", "red" 
    elif sbp >= 140 or dbp >= 90: return "Stage 2 Hypertension", "red"
    return "Unclassified", "gray"

def get_glucose_category(glucose):
    """Source: 2025 ADA Standards of Care"""
    if glucose < 54: return "Hypoglycemia (Level 2 - Critical)", "red" 
    elif glucose < 70: return "Hypoglycemia (Level 1)", "orange"      
    elif 70 <= glucose <= 99: return "Normal Fasting", "green"
    elif 100 <= glucose <= 125: return "Prediabetes", "orange"
    elif glucose > 180: return "Hyperglycemia (Inpatient Alert)", "red"
    elif glucose >= 126: return "Diabetes Range", "orange"
    return "Normal", "green"

def get_hr_category(hr):
    """Source: AHA / ACLS Bradycardia & Tachycardia Algorithms"""
    if hr < 50: return "Bradycardia (Significant)", "orange"
    elif 50 <= hr <= 100: return "Normal Sinus", "green"
    elif hr > 130: return "Tachycardia (Critical)", "red"
    elif hr > 100: return "Tachycardia", "orange"
    return "Normal", "green"

def get_resp_category(rr):
    """Source: Sepsis-3 & SIRS Criteria"""
    if rr < 8: return "Bradypnea (Critical)", "red"
    elif rr < 12: return "Bradypnea", "orange"
    elif 12 <= rr <= 20: return "Normal", "green"
    elif rr > 30: return "Tachypnea (Critical)", "red"
    elif rr > 20: return "Tachypnea (SIRS Criteria)", "orange"
    return "Normal", "green"

def get_temp_category(temp_f):
    """Source: SIRS Criteria (Fahrenheit)"""
    if temp_f < 96.8: return "Hypothermia (SIRS)", "orange"
    elif 96.8 <= temp_f <= 100.4: return "Normal", "green"
    elif temp_f > 100.4: return "Fever (SIRS)", "orange"
    return "Normal", "green"

# --- CONFIGURATION: CONNECT TO AI ---
try:
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    else:
        st.warning("⚠️ AI Key missing. Please add GEMINI_API_KEY to Streamlit Secrets.")
except Exception as e:
    st.error(f"⚠️ AI Configuration Error: {e}")

# ==========================================
# 2. AI EXTRACTION & REPORTING
# ==========================================

def extract_data_from_soap(note_text):
    candidate_models = ['gemini-flash-latest']
    prompt = f"""
    You are a clinical data extraction assistant. Extract the following values from the note.
    Return ONLY a valid JSON object. Do not add markdown formatting.
    If a value is not mentioned, return null.
    
    Keys to extract:
    - name (string)
    - age (integer)
    - gender (string)
    - weight_lbs (float. If note is in kg, convert to lbs)
    - height_cm (integer)
    - sbp (integer)
    - dbp (integer)
    - hr (integer)
    - rr (integer)
    - temp_f (float. If note is in Celsius, convert to Fahrenheit)
    - o2_sat (integer)
    - creatinine (float)
    - bun (integer)
    - potassium (float)
    - glucose (integer)
    - wbc (float)
    - hgb (float)
    - platelets (integer)
    - inr (float)
    - lactate (float)
    - anticoagulant_use (boolean)
    - liver_disease (boolean)
    - heart_failure (boolean)
    - gi_bleed (boolean)
    - nsaid_use (boolean)
    - active_chemo (boolean)
    - diuretic_use (boolean)
    - acei_arb_use (boolean)
    - insulin_use (boolean)
    - uncontrolled_diabetes (boolean)
    - altered_mental (boolean)

    Clinical Note: "{note_text}"
    """
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_text)
        except Exception as e:
            continue
    st.error("❌ Extraction failed.")
    return None

def create_pdf_report(res, ai_assessment, alerts_list):
    pdf = FPDF()
    pdf.add_page()
    
    def clean_text(text):
        if not isinstance(text, str): return str(text)
        text = text.replace("🚨", "[CRITICAL] ").replace("⚠️", "[WARNING] ")
        text = text.replace("👉", "").replace("✅", "[OK] ")
        return text.encode('latin-1', 'replace').decode('latin-1')

    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="Clinical Risk Monitor - Acute SOAP Note", ln=1, align='C')
    pdf.line(10, 30, 200, 30); pdf.ln(10)

    # Subjective
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, txt="SUBJECTIVE", ln=1, fill=True); pdf.ln(2)
    pdf.set_font("Arial", size=11)
    
    weight_val = res.get('weight_kg', 0) * 2.20462 if res.get('weight_kg') else 0
    patient_info = f"Name: {res.get('name', 'Unknown')} | Age: {res.get('age')} | Gender: {res.get('gender')}"
    pdf.cell(0, 6, txt=clean_text(patient_info), ln=1)
    pdf.cell(0, 6, txt=f"Weight: {weight_val:.1f} lbs", ln=1)
    
    if ai_assessment:
        pdf.ln(2); pdf.set_font("Arial", 'I', 10)
        pdf.multi_cell(0, 5, txt=clean_text(f"AI Assessment: {ai_assessment[:500]}..."))
    pdf.ln(5)

    # Objective
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="OBJECTIVE", ln=1, fill=True); pdf.ln(2)
    
    bp_cat, _ = get_bp_category(res.get('sys_bp', 0), res.get('dia_bp', 0))
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(45, 6, txt="Blood Pressure:", align='R'); pdf.set_font("Arial", size=10)
    pdf.cell(50, 6, txt=clean_text(f"{res.get('sys_bp')}/{res.get('dia_bp')} ({bp_cat})"), ln=0)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(45, 6, txt="Temp:", align='R'); pdf.set_font("Arial", size=10)
    pdf.cell(50, 6, txt=f"{res.get('temp_f')} F", ln=1)
    pdf.ln(5)

    # Alerts
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="ASSESSMENT & ALERTS", ln=1, fill=True); pdf.ln(2)
    pdf.set_font("Arial", size=10)
    problems = [a for a in alerts_list if "Protocol" not in a]
    if problems:
        pdf.set_text_color(200, 0, 0)
        for p in problems: pdf.cell(0, 6, txt=f"  - {clean_text(p)}", ln=1)
    else:
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 6, txt="  - No acute instability detected.", ln=1)
    pdf.set_text_color(0, 0, 0); pdf.ln(5)

    # Plan
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="PLAN", ln=1, fill=True); pdf.ln(2)
    pdf.set_font("Arial", size=10)
    protocols = [a for a in alerts_list if "Protocol" in a or "👉" in a]
    if protocols:
        for prot in protocols: pdf.multi_cell(0, 6, txt=f"[ ] {clean_text(prot)}"); pdf.ln(1)
    else:
        pdf.cell(0, 6, txt="[ ] Continue standard monitoring.", ln=1)

    return pdf.output(dest='S').encode('latin-1', 'replace')

# ==========================================
# 3. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Clinical Risk Monitor", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>[data-testid="stSidebar"] {background-color: #f8f9fa; border-right: 1px solid #e9ecef;} [data-testid="stMetricValue"] {font-size: 1.8rem !important; color: #212529;}</style>""", unsafe_allow_html=True)

st.title("🏥 Clinical Risk Monitor")
st.divider()

def get_timestamp(): return datetime.datetime.now().strftime("%Y%m%d_%H%M")
bk.init_db()
try: bleeding_model = bk.load_bleeding_model()
except Exception as e: st.error(f"Model failed to load: {e}"); st.stop()

if 'patient_data' not in st.session_state: st.session_state['patient_data'] = {}
if 'entered_app' not in st.session_state: st.session_state['entered_app'] = False

# ==========================================
# 4. UI MODULES
# ==========================================

def render_cover_page():
    st.markdown("<h1 style='text-align: center;'>🛡️ Clinical Risk Monitor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-Driven Pharmacovigilance System</p>", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    if c2.button("🚀 Launch Dashboard", use_container_width=True, type="primary"):
        st.session_state['entered_app'] = True; st.rerun()

def render_risk_calculator():
    st.subheader("Acute Risk Calculator")
    defaults = {
        'name_input': "", 'age_input': 0, 'gender_input': "Male", 'weight_input': 0.0, 'height_input': 0,
        'sbp_input': 0, 'dbp_input': 0, 'hr_input': 0, 'rr_input': 0, 'temp_input': 0.0, 'o2_input': 0,
        'creat_input': 0.0, 'bun_input': 0, 'k_input': 0.0, 'glc_input': 0, 'wbc_input': 0.0, 'hgb_input': 0.0,
        'plt_input': 0, 'inr_input': 0.0, 'lac_input': 0.0, 'anticoag_input': False, 'liver_input': False,
        'chf_input': False, 'gib_input': False, 'nsaid_input': False, 'chemo_input': False,
        'diuretic_input': False, 'acei_input': False, 'insulin_input': False, 'dm_input': False, 'ams_input': False
    }
    for key, default_val in defaults.items():
        if key not in st.session_state: st.session_state[key] = default_val

    with st.expander("⚡ AI Auto-Fill (Paste SOAP Note)"):
        st.caption("Paste a clinical note to auto-populate.")
        soap_note = st.text_area("Clinical Note")
        if st.button("✨ Extract Data") and soap_note:
            with st.spinner("Extracting..."):
                data = extract_data_from_soap(soap_note)
                if data:
                    if data.get('name'): st.session_state['name_input'] = data['name']
                    if data.get('age'): st.session_state['age_input'] = int(data['age'])
                    if data.get('gender'): st.session_state['gender_input'] = data['gender']
                    if data.get('weight_lbs'): st.session_state['weight_input'] = float(data['weight_lbs'])
                    if data.get('height_cm'): st.session_state['height_input'] = int(data['height_cm'])
                    if data.get('sbp'): st.session_state['sbp_input'] = int(data['sbp'])
                    if data.get('dbp'): st.session_state['dbp_input'] = int(data['dbp'])
                    if data.get('hr'): st.session_state['hr_input'] = int(data['hr'])
                    if data.get('rr'): st.session_state['rr_input'] = int(data['rr'])
                    if data.get('temp_f'): st.session_state['temp_input'] = float(data['temp_f'])
                    if data.get('o2_sat'): st.session_state['o2_input'] = int(data['o2_sat'])
                    if data.get('glucose'): st.session_state['glc_input'] = int(data['glucose'])
                    if data.get('wbc'): st.session_state['wbc_input'] = float(data['wbc'])
                    if data.get('lactate'): st.session_state['lac_input'] = float(data['lactate'])
                    if data.get('creatinine'): st.session_state['creat_input'] = float(data['creatinine'])
                    st.success("✅ Extracted!"); st.rerun()

    with st.container(border=True):
        st.markdown("#### 📝 Patient Data Entry")
        with st.form("risk_form"):
            # --- NAME INPUT ADDED ---
            name = st.text_input("Patient Name / ID", key='name_input')
            
            col_left, col_right = st.columns([1, 1], gap="medium")
            with col_left:
                st.markdown("##### 👤 Profile & Vitals")
                l1, l2 = st.columns(2)
                age = l1.number_input("Age (Years)", min_value=0, max_value=120, key='age_input')
                gender = l2.selectbox("Gender", ["Male", "Female"], key='gender_input')
                
                w1, w2 = st.columns(2)
                weight_lbs = w1.number_input("Weight (lbs)", 0.0, 500.0, key='weight_input')
                # Restored Height Input
                height_cm = w2.number_input("Height (cm)", 0, 250, key='height_input')
                
                # Logic
                weight_kg = weight_lbs * 0.453592 
                bmi = weight_kg / ((height_cm/100)**2) if height_cm > 0 else 0

                v1, v2 = st.columns(2)
                sys_bp = v1.number_input("Systolic BP", 0, 300, key='sbp_input')
                dia_bp = v2.number_input("Diastolic BP", 0, 200, key='dbp_input')
                v3, v4 = st.columns(2)
                hr = v3.number_input("Heart Rate", 0, 300, key='hr_input')
                resp_rate = v4.number_input("Resp Rate", 0, 60, key='rr_input')
                v5, v6 = st.columns(2)
                temp_f = v5.number_input("Temp °F", 0.0, 115.0, step=0.1, key='temp_input')
                o2_sat = v6.number_input("O2 Sat %", 0, 100, key='o2_input')

            with col_right:
                st.markdown("##### 🧪 Labs & History")
                # Restored Min/Max values for labs
                lab1, lab2 = st.columns(2)
                creat = lab1.number_input("Creatinine", 0.0, 20.0, key='creat_input')
                bun = lab2.number_input("BUN", 0, 100, key='bun_input')
                lab3, lab4 = st.columns(2)
                potassium = lab3.number_input("Potassium", 0.0, 10.0, key='k_input')
                glucose = lab4.number_input("Glucose", 0, 1000, key='glc_input')
                lab5, lab6 = st.columns(2)
                wbc = lab5.number_input("WBC", 0.0, 50.0, key='wbc_input')
                hgb = lab6.number_input("Hgb", 0.0, 20.0, key='hgb_input')
                lab7, lab8 = st.columns(2)
                platelets = lab7.number_input("Platelets", 0, 1000, key='plt_input')
                inr = lab8.number_input("INR", 0.0, 10.0, key='inr_input')
                lactate = st.number_input("Lactate", 0.0, 20.0, key='lac_input')

                h1, h2 = st.columns(2)
                anticoag = h1.checkbox("Anticoagulant", key='anticoag_input')
                liver_disease = h2.checkbox("Liver Disease", key='liver_input')
                h3, h4 = st.columns(2)
                heart_failure = h3.checkbox("Heart Failure", key='chf_input')
                gi_bleed = h4.checkbox("Hx GI Bleed", key='gib_input')
                m1, m2 = st.columns(2)
                diuretic = m1.checkbox("Diuretic", key='diuretic_input')
                acei = m2.checkbox("ACEi/ARB", key='acei_input')
                m3, m4 = st.columns(2)
                insulin = m3.checkbox("Insulin", key='insulin_input')
                altered_mental = m4.checkbox("Altered Mental Status", key='ams_input')
                
                # Hidden inputs for model compatibility
                nsaid = False; active_chemo = False; hba1c_high = False

            st.write(""); submitted = st.form_submit_button("🚀 Run Clinical Analysis", type="primary", use_container_width=True)

    if submitted:
        # 1. Hemodynamics
        map_val = (sys_bp + (2 * dia_bp)) / 3 if sys_bp > 0 else 0
        pulse_pressure = sys_bp - dia_bp
        shock_index = hr / sys_bp if sys_bp > 0 else 0
        
        # 2. Bleeding Risk Model
        input_df = pd.DataFrame({'age': [age], 'inr': [inr], 'systolic_bp': [sys_bp], 'anticoagulant': [1 if anticoag else 0], 'gender': [gender], 'liver_disease': [1 if liver_disease else 0], 'weight': [weight_kg]})
        try: pred_bleeding = bleeding_model.predict_proba(input_df)[0][1] * 100
        except: pred_bleeding = float(bleeding_model.predict(input_df)[0])

        # 3. Clinical Scores (SIRS)
        sirs_score = 0
        if temp_f > 100.4 or (temp_f < 96.8 and temp_f > 0): sirs_score += 1
        if hr > 90: sirs_score += 1
        if resp_rate > 20: sirs_score += 1
        if wbc > 12 or (wbc < 4 and wbc > 0): sirs_score += 1
        
        pred_aki = bk.calculate_aki_risk(age, diuretic, acei, sys_bp, active_chemo, creat, nsaid, heart_failure)
        pred_hypo = bk.calculate_hypoglycemic_risk(insulin, (creat>1.3), hba1c_high, False)
        
        status_calc = 'Critical' if (pred_bleeding > 50 or sirs_score >= 2 or map_val < 65) else 'Stable'
        
        # SAVE TO DB WITH NAME
        bk.save_patient_to_db(name, age, gender, sys_bp, int(pred_aki), float(pred_bleeding), status_calc)
        
        # STORE IN SESSION
        st.session_state['patient_data'] = {
            'name': name, 'id': name if name else "Unknown",
            'status': status_calc, 'sys_bp': sys_bp, 'hr': hr, 'o2_sat': o2_sat,
            'bleeding_risk': pred_bleeding, 'aki_risk': pred_aki, 'sepsis_risk': sirs_score, 'temp_c': (temp_f - 32) * 5/9
        }
        
        st.session_state['analysis_results'] = {
            'name': name,
            'bleeding_risk': float(pred_bleeding), 'aki_risk': int(pred_aki),
            'sirs_score': sirs_score, 'hypo_risk': int(pred_hypo),
            'map_val': map_val, 'shock_index': shock_index, 'pulse_pressure': pulse_pressure,
            'age': age, 'sys_bp': sys_bp, 'dia_bp': dia_bp, 'hr': hr, 'rr': resp_rate,
            'temp_f': temp_f, 'o2_sat': o2_sat, 'glucose': glucose, 'wbc': wbc, 
            'lactate': lactate, 'inr': inr, 'anticoag': anticoag, 'liver_disease': liver_disease,
            'diuretic': diuretic, 'acei': acei, 'ams': altered_mental, 'creat': creat,
            'weight_kg': weight_kg,
            # FIXED KEY ERROR HERE
            'gib_input': gi_bleed 
        }

    if 'analysis_results' in st.session_state:
        res = st.session_state['analysis_results']
        st.divider()
        st.subheader(f"📊 Results for: {res.get('name', 'Unknown')}")
        
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("🩸 Bleeding Risk", f"{res['bleeding_risk']:.1f}%", "High" if res['bleeding_risk'] > 50 else "Normal", delta_color="inverse")
        r2.metric("🔥 SIRS Score", f"{res['sirs_score']}/4", "Sepsis Risk" if res['sirs_score'] >= 2 else "Normal", delta_color="inverse")
        r3.metric("MAP", f"{int(res['map_val'])} mmHg", "Low (<65)" if res['map_val'] < 65 and res['map_val'] > 0 else "Normal", delta_color="inverse")
        r4.metric("Shock Index", f"{res['shock_index']:.2f}", "Critical (>0.9)" if res['shock_index'] > 0.9 else "Normal", delta_color="inverse")

        st.divider()
        st.markdown("### ⚠️ Clinical Alerts & Risk Analysis")
        violations = 0; pdf_alerts = []

        # ALERTS LOGIC (2025)
        if res['sirs_score'] >= 2:
            msg = f"⚠️ SEPSIS ALERT: SIRS Criteria Met ({res['sirs_score']}/4). Screen for infection."
            st.warning(msg); pdf_alerts.append(msg); violations += 1
        
        if res['sirs_score'] >= 2 and res['map_val'] < 65 and res['lactate'] > 2:
            msg = f"🚨 SEPTIC SHOCK ALERT: Hypotension + Hyperlactatemia."
            st.error(msg); pdf_alerts.append(msg); violations += 1

        if res['o2_sat'] > 0:
            if res['o2_sat'] < 88:
                msg = f"🚨 CRITICAL HYPOXIA (SpO2 {res['o2_sat']}%)"
                st.error(msg); pdf_alerts.append(msg); violations += 1
            elif res['o2_sat'] < 94:
                msg = f"⚠️ Hypoxia (SpO2 {res['o2_sat']}%)"
                st.warning(msg); pdf_alerts.append(msg); violations += 1

        bp_cat, _ = get_bp_category(res['sys_bp'], res['dia_bp'])
        if "Severe" in bp_cat:
            msg = f"🚨 {bp_cat.upper()} (BP {res['sys_bp']}/{res['dia_bp']})"
            st.error(msg); pdf_alerts.append(msg); violations += 1
        elif "Hypotension" in bp_cat:
            msg = f"🚨 HYPOTENSION / SHOCK RISK (BP {res['sys_bp']}/{res['dia_bp']})"
            st.error(msg); pdf_alerts.append(msg); violations += 1
        
        if res['age'] >= 65 and res['sys_bp'] >= 130:
            st.warning(f"• **Senior BP Target:** SBP {res['sys_bp']} exceeds 2025 target (<130/80 mmHg).")

        gluc_cat, _ = get_glucose_category(res['glucose'])
        if "Level 2" in gluc_cat:
            msg = f"🚨 {gluc_cat.upper()} ({res['glucose']} mg/dL)"
            st.error(msg); pdf_alerts.append(msg); violations += 1
        elif "Level 1" in gluc_cat:
            msg = f"⚠️ {gluc_cat.upper()} ({res['glucose']} mg/dL)"
            st.warning(msg); pdf_alerts.append(msg); violations += 1
        elif "Action Threshold" in gluc_cat:
            msg = f"🚨 INPATIENT HYPERGLYCEMIA ({res['glucose']} mg/dL) - Action >180."
            st.error(msg); pdf_alerts.append(msg); violations += 1

        if res['inr'] > 3.5: st.error(f"• **Critical INR ({res.get('inr')}):** Major hemorrhage risk.")
        if res['anticoag']: st.warning("• **Medication:** Patient is on anticoagulants.")
        
        # FIXED GIB ALERT LOGIC
        if res.get('gib_input'): st.error("• **History:** Previous GI Bleed (High Recurrence Risk).")
        
        if int(res.get('map_val', 0)) < 65 and int(res.get('map_val', 0)) > 0:
             st.error(f"• **MAP {int(res.get('map_val', 0))}:** Critical hypoperfusion.")

        # STATUS CHECK
        has_demographics = (res.get('age', 0) > 0 or res.get('weight_kg', 0) > 0)
        has_vitals = (res.get('sys_bp', 0) > 0 or res.get('hr', 0) > 0)

        if violations == 0:
            if not has_demographics and not has_vitals:
                st.warning("⚠️ **No Data Entered:** Please input patient data.")
            elif has_demographics and not has_vitals:
                st.warning("⚠️ **Missing Vitals:** Demographics recorded, but Vital Signs required.")
            else:
                st.success("✅ **Patient Stable:** No immediate life-threatening protocol violations detected.")

        st.divider()
        c_ai, c_txt = st.columns([1, 3])
        with c_ai:
            st.markdown("#### 🤖 AI Assessment")
            if st.button("⚡ Consult AI"):
                with st.spinner("Thinking..."):
                    ai_response = bk.consult_ai_doctor("risk_assessment", "", st.session_state['analysis_results'])
                    st.session_state['ai_result'] = ai_response
        with c_txt:
            if 'ai_result' in st.session_state: st.info(st.session_state['ai_result'])

        st.divider()
        if st.button("📄 Generate PDF Report"):
            pdf_bytes = create_pdf_report(res, st.session_state.get('ai_result', ""), pdf_alerts)
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="Clinical_Report.pdf">Download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)

# --- MODULE 2: PATIENT HISTORY---
def render_history_sql():
    st.subheader("🗄️ Patient History Database")
    df = bk.fetch_history()
    if not df.empty:
        # Reorder columns to show Name first
        cols = ['timestamp', 'name', 'age', 'gender', 'sbp', 'aki_risk_score', 'bleeding_risk_score', 'status']
        # Filter existing columns in case of schema drift
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
        st.dataframe(
            df, 
            use_container_width=True, 
            height=400, 
            hide_index=True,
            column_config={
                "name": st.column_config.TextColumn("Patient Name", width="medium"),
                "aki_risk_score": st.column_config.ProgressColumn("AKI Risk", format="%d%%", min_value=0, max_value=100),
                "bleeding_risk_score": st.column_config.ProgressColumn("Bleed Risk", format="%.1f%%", min_value=0, max_value=100),
            }
        )
        if st.button("🗑️ Clear All Records"):
            bk.clear_history()
            st.rerun()
    else:
        st.info("📭 Database is empty.")

# --- MODULE 3: LIVE DASHBOARD ---
def render_dashboard():
    data = st.session_state.get('patient_data', {})
    if not data:
        st.warning("⚠️ No patient data found. Please run the Risk Calculator first.")
        return

    # Name display logic
    patient_name = data.get('name') if data.get('name') else "Unknown Patient"
    
    st.subheader(f"🛏️ Bedside Monitor: {patient_name}")
    st.caption(f"Status: **{data.get('status', 'Unknown')}**")
    
    st.divider()
    c1, c2 = st.columns([3, 1])
    with c1:
        # Mock Live Trace
        base_sbp = data.get('sys_bp', 120)
        base_hr = data.get('hr', 80)
        chart_data = pd.DataFrame({
            'Time': range(20),
            'Systolic BP': np.random.normal(base_sbp, 2, 20),
            'Heart Rate': np.random.normal(base_hr, 2, 20)    
        }).melt('Time', var_name='Metric', value_name='Value')
        c = alt.Chart(chart_data).mark_line(interpolate='basis').encode(
            x=alt.X('Time', axis=None), y=alt.Y('Value', scale=alt.Scale(zero=False)), color='Metric'
        ).properties(height=200)
        st.altair_chart(c, use_container_width=True)
    
    with c2:
        st.metric("SBP", f"{int(base_sbp)}")
        st.metric("HR", f"{int(base_hr)}")
        st.metric("SpO2", f"{int(data.get('o2_sat', 0))}%")
        
        # AI DISCHARGE SUMMARY
        st.write("")
        if st.button("✨ Discharge Note"):
            with st.spinner("Generating..."):
                ai_summary = bk.generate_discharge_summary(data)
                st.session_state['latest_discharge_note'] = ai_summary
        
        if 'latest_discharge_note' in st.session_state:
            st.download_button(
                label="📥 Download",
                data=st.session_state['latest_discharge_note'],
                file_name=f"discharge_{data.get('id')}_{get_timestamp()}.txt",
                mime="text/plain"
            )

# --- MODULE 4: BATCH ANALYSIS (CSV) ---
def render_batch_analysis():
    st.subheader("Bulk Patient Processing & Diagnostic Triage")
    
    # 1. Helper: Download Template
    with st.expander("ℹ️  Download CSV Template"):
        sample_data = {
            'Age': [65, 72], 'Gender': ['Male', 'Female'], 'Weight_kg': [80, 65],
            'Systolic_BP': [130, 90], 'Diastolic_BP': [80, 50], 'Heart_Rate': [72, 110],
            'Resp_Rate': [16, 24], 'Temp_C': [37.0, 38.5], 'O2_Sat': [98, 92],
            'WBC': [6.0, 15.0], 'Glucose': [110, 5.5], 'Creatinine': [1.1, 150],
            'INR': [1.0, 1.2], 'Altered_Mental': [0, 1], 'Anticoagulant': [1, 0],
            'Heart_Failure': [0, 1], 'Liver_Disease': [0, 0], 'Hx_GI_Bleed': [0, 0]
        }
        df_sample = pd.DataFrame(sample_data)
        csv_template = df_sample.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Template", csv_template, "patient_data.csv", "text/csv")

    # 2. CSV PROCESSOR
    st.markdown("#### 📤 Upload Patient Batch")
    uploaded_csv = st.file_uploader("Upload Patient Data (CSV)", type=["csv"])
    
    if uploaded_csv:
        try:
            raw_df = pd.read_csv(uploaded_csv)
            
            # --- A. SMART MAPPING ---
            col_map = {
                'sbp':'Systolic_BP', 'sys':'Systolic_BP', 'hr':'Heart_Rate', 'rr':'Resp_Rate', 'temp':'Temp_C',
                'spo2':'O2_Sat', 'cr':'Creatinine', 'wbc':'WBC', 'glu':'Glucose', 'sugar':'Glucose'
            }
            df = raw_df.rename(columns=lambda x: col_map.get(x.lower(), x))
            
            # Fill missing with safe defaults for calculation
            req_cols = ['Age','Systolic_BP','Heart_Rate','Resp_Rate','Temp_C','O2_Sat','Creatinine','Glucose']
            for c in req_cols:
                if c not in df.columns: df[c] = 0

            if st.button("⚡ Run Clinical Analysis", type="primary"):
                with st.spinner("Validating Data & Calculating Risks..."):
                    
                    # --- B. SMART UNIT CONVERSION & VALIDATION ---
                    def clean_row(row):
                        # 1. Glucose: Convert mmol/L to mg/dL
                        if 0 < row['Glucose'] < 30:
                            row['Glucose'] = row['Glucose'] * 18
                        
                        # 2. Creatinine: Convert umol/L to mg/dL
                        if row['Creatinine'] > 20:
                            row['Creatinine'] = row['Creatinine'] / 88.4
                            
                        return row
                    
                    df = df.apply(clean_row, axis=1)

                    # --- C. ADVANCED SCORING (NEWS-2) ---
                    def calculate_news(row):
                        score = 0
                        # Resp Rate
                        if row['Resp_Rate'] <= 8 or row['Resp_Rate'] >= 25: score += 3
                        elif row['Resp_Rate'] >= 21: score += 2
                        elif row['Resp_Rate'] <= 11: score += 1
                        
                        # SpO2
                        if row['O2_Sat'] <= 91: score += 3
                        elif row['O2_Sat'] <= 93: score += 2
                        elif row['O2_Sat'] <= 95: score += 1
                        
                        # Systolic BP
                        if row['Systolic_BP'] <= 90 or row['Systolic_BP'] >= 220: score += 3
                        elif row['Systolic_BP'] <= 100: score += 2
                        elif row['Systolic_BP'] <= 110: score += 1
                        
                        # Heart Rate
                        if row['Heart_Rate'] <= 40 or row['Heart_Rate'] >= 131: score += 3
                        elif row['Heart_Rate'] >= 111: score += 2
                        elif row['Heart_Rate'] <= 50 or row['Heart_Rate'] >= 91: score += 1
                        
                        # Consciousness
                        if row.get('Altered_Mental', 0) == 1: score += 3
                        
                        # Temp
                        if row['Temp_C'] <= 35.0: score += 3
                        elif row['Temp_C'] >= 39.1: score += 2
                        elif row['Temp_C'] <= 36.0 or row['Temp_C'] >= 38.1: score += 1
                        
                        return score

                    df['NEWS_Score'] = df.apply(calculate_news, axis=1)

                    # --- D. DIAGNOSTIC LABELS ---
                    def get_status(row):
                        alerts = []
                        # 1. NEWS-2 Interpretation
                        if row['NEWS_Score'] >= 7: alerts.append("CRITICAL (NEWS ≥7)")
                        elif row['NEWS_Score'] >= 5: alerts.append("Urgent (NEWS ≥5)")
                        
                        # 2. Specific Organ Failures
                        if row['Systolic_BP'] > 180: alerts.append("HTN Crisis")
                        if row['Creatinine'] > 2.0: alerts.append("AKI Warning")
                        if row.get('WBC', 0) > 12: alerts.append("Leukocytosis")
                        
                        return " + ".join(alerts) if alerts else "Stable"

                    df['Clinical_Status'] = df.apply(get_status, axis=1)
                    
                    # Run AI Prediction (Create simple input df for the XGBoost model)
                    model_inputs = pd.DataFrame()
                    model_inputs['age'] = df.get('Age', 0)
                    model_inputs['inr'] = df.get('INR', 1.0)
                    model_inputs['anticoagulant'] = df.get('Anticoagulant', 0)
                    model_inputs['gi_bleed'] = df.get('Hx_GI_Bleed', 0)
                    model_inputs['high_bp'] = df['Systolic_BP'].apply(lambda x: 1 if x > 140 else 0)
                    model_inputs['antiplatelet'] = 0
                    model_inputs['gender_female'] = 0 
                    model_inputs['weight'] = df.get('Weight_kg', 70)
                    model_inputs['liver_disease'] = df.get('Liver_Disease', 0)
                    
                    df['Bleed_Risk_%'] = bleeding_model.predict(model_inputs)

                    # --- E. DISPLAY ---
                    def color_rows(val):
                        if 'CRITICAL' in str(val): return 'background-color: #ffcdd2; color: black; font-weight: bold;'
                        if 'Urgent' in str(val) or 'Warning' in str(val): return 'background-color: #fff3cd; color: black;'
                        return ''

                    st.success(f"Processed {len(df)} records with enhanced validation.")
                    st.dataframe(
                        df[['Clinical_Status', 'NEWS_Score', 'Age', 'Systolic_BP', 'Bleed_Risk_%']]
                        .style.map(color_rows, subset=['Clinical_Status']), 
                        use_container_width=True
                    )
                    
                    # ---Timestamped Download ---
                    csv_result = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Analyzed Data", 
                        csv_result, 
                        f"analyzed_patients_{get_timestamp()}.csv", 
                        "text/csv"
                    )
                
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# --- MODULE 5: MEDICATION CHECKER ---
def render_medication_checker():
    st.subheader("💊 Drug-Drug Interaction Checker")
    st.caption("Checks for Critical and Major interactions from backend database + AI Analysis.")
    
    col_d1, col_d2 = st.columns(2)
    d1 = col_d1.text_input("Drug A", placeholder="e.g. Warfarin")
    d2 = col_d2.text_input("Drug B", placeholder="e.g. Ibuprofen")

    if d1 and d2:
        st.divider()
        
        # 1. Database Check (Deterministic)
        res = bk.check_interaction(d1, d2)
        
        if res:
            if "CRITICAL" in res: 
                st.error(f"❌ {res}")
            elif "MAJOR" in res: 
                st.warning(f"⚠️ {res}")
            elif "MODERATE" in res: 
                st.info(f"ℹ️ {res}")
        else: 
            st.warning(f"⚠️ **{d1}** + **{d2}** not found in the high-alert database.")
            st.markdown("👉 **Recommendation:** This does not guarantee safety. Please use the **AI Pharmacist** below for a comprehensive check.")

        # 2. AI Analysis Button
        st.divider()
        st.markdown("#### 🧠 AI Pharmacist Analysis")
        st.caption("Get a detailed explanation of mechanism, management, and safety profile.")
        
        if st.button("⚡ Analyze Interaction with AI"):
            with st.spinner("Consulting AI Pharmacist..."):
                ai_report = bk.analyze_drug_interactions([d1, d2])
                st.markdown(ai_report)
                st.caption("⚠️ AI-Generated. Verify with standard drug compendiums.")

# --- MODULE 6: CHATBOT ---
def render_chatbot():
    st.subheader("AI Clinical Assistant")
    st.caption("Database covers 250+ clinical topics (Cardio, Resp, Neuro, Pharm, Labs).")
    
    q = st.text_input("Ask a clinical question:")
    if q:
        with st.chat_message("assistant"):
            st.write(bk.chatbot_response(q))

# --- MODULE 7: AI DIAGNOSTICIAN ---
def render_ai_diagnostician():
    st.subheader("🧠 AI-Powered Clinical Consultant")
    st.caption("Ask complex clinical questions or simulate differential diagnoses.")
    
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I am your AI Clinical Consultant. Describe a patient case or ask a medical question."}]

    # Display History
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Type your clinical query here..."):
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # 2. AI Response
        with st.spinner("Thinking..."):
            # Get context from session if available
            current_data = st.session_state.get('patient_data', {})
            
            # Call Backend
            response = bk.consult_ai_doctor("provider", prompt, current_data)
            
            # 3. Display AI Message
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

# ---------------------------------------------------------
# 3. MAIN APP CONTROLLER
# ---------------------------------------------------------
if not st.session_state['entered_app']:
    render_cover_page()
else:
    with st.sidebar:
        st.title("Navigation")
        menu = st.radio("Module", [
            "Risk Calculator", 
            "Patient History (SQL)",
            "Live Dashboard", 
            "Batch Analysis (CSV)", 
            "Medication Checker", 
            "📚 Medical Glossary",
            "🧠 AI Clinical Consultant"
        ])
        st.info("v3.0 - AI Integrated")

    if menu == "Risk Calculator":
        render_risk_calculator()
    elif menu == "Patient History (SQL)":
        render_history_sql()
    elif menu == "Live Dashboard":
        render_dashboard()
    elif menu == "Batch Analysis (CSV)":
        render_batch_analysis()
    elif menu == "Medication Checker":
        render_medication_checker()
    elif menu == "📚 Medical Glossary":
        render_chatbot()
    elif menu == "🧠 AI Clinical Consultant":
        render_ai_diagnostician()
      # ---------------------------------------------------------
    # DEBUG: ADMIN PANEL (Only visible if URL has ?admin=true)
    # ---------------------------------------------------------
    # Check if the URL is https://.../?admin=true
    if "admin" in st.query_params: 
        st.sidebar.divider()
        st.sidebar.subheader("🔧 Admin Mode")
        if st.sidebar.button("⚡ Force Retrain Model"):
            with st.spinner("Training new model on Cloud Server..."):
                try:
                    import train_model
                    train_model.train()
                    st.success("✅ Model Retrained! Reboot App now.")
                except Exception as e:
                    st.error(f"Failed: {e}")
