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

# --- CLINICAL GUIDELINE CLASSIFIERS (AHA, ADA, SEPSIS-3) ---

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
    """Source: American Diabetes Association (ADA) Standards of Care"""
    if glucose < 70: return "Hypoglycemia (Level 1)", "red"
    elif 70 <= glucose <= 99: return "Normal Fasting", "green"
    elif 100 <= glucose <= 125: return "Prediabetes", "orange"
    elif glucose > 180: return "Hyperglycemia (Inpatient Alert)", "red"
    elif glucose >= 126: return "Diabetes Range", "orange"
    return "Normal", "green"

def get_hr_category(hr):
    """Source: AHA / ACLS Bradycardia & Tachycardia Algorithms"""
    if hr < 60: return "Bradycardia", "orange"
    elif 60 <= hr <= 100: return "Normal Sinus", "green"
    elif hr > 100: return "Tachycardia", "red"
    return "Normal", "green"

def get_resp_category(rr):
    """Source: Sepsis-3 (qSOFA) & Normal Ranges"""
    if rr < 12: return "Bradypnea", "orange"
    elif 12 <= rr <= 20: return "Normal", "green"
    elif rr >= 22: return "Tachypnea (qSOFA Critical)", "red"
    elif rr > 20: return "Tachypnea", "orange"
    return "Normal", "green"

# --- CONFIGURATION: CONNECT TO AI ---
try:
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    else:
        st.warning("⚠️ AI Key missing. Please add GEMINI_API_KEY to Streamlit Secrets.")
except Exception as e:
    st.error(f"⚠️ AI Configuration Error: {e}")

# --- AI Extraction Function (Smart Parser) ---
def extract_data_from_soap(note_text):
    """
    Uses Gemini to extract structured clinical data from unstructured narrative.
    """
    prompt = f"""
    You are a clinical data extraction assistant. Extract the following values from the note.
    Return ONLY a valid JSON object. Do not add markdown formatting.
    If a value is not mentioned, return null.
    
    Keys to extract:
    - age (integer), gender (string: "Male" or "Female")
    - sbp (systolic bp, integer), dbp (diastolic bp, integer)
    - hr (heart rate, integer), rr (respiratory rate, integer)
    - temp_c (temperature in celsius, float), o2_sat (integer)
    - creatinine (float), inr (float), glucose (integer)
    - anticoagulant_use (boolean), liver_disease (boolean), heart_failure (boolean), altered_mental (boolean)

    Clinical Note:
    "{note_text}"
    """
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(prompt)
        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_text)
    except Exception:
        return None

# --- PDF REPORT HELPER ---
def create_pdf_report(res, ai_assessment, alerts_list):
    pdf = FPDF()
    pdf.add_page()
    def clean_text(text):
        if not isinstance(text, str): return str(text)
        return text.encode('latin-1', 'replace').decode('latin-1')
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="Clinical Risk Report", ln=1, align='C')
    pdf.set_font("Arial", size=10)
    pdf.ln(10)
    pdf.cell(0, 10, txt=clean_text(f"Patient: {res.get('age')}yo {res.get('gender')}"), ln=1)
    pdf.cell(0, 10, txt=clean_text(f"Bleeding Risk: {res.get('bleeding_risk'):.1f}%"), ln=1)
    pdf.multi_cell(0, 10, txt=clean_text(f"AI Assessment: {ai_assessment}"))
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Clinical Risk Monitor", 
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e9ecef; }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #212529; }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 Clinical Risk Monitor")
st.divider()

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")

bk.init_db()

try:
    bleeding_model = bk.load_bleeding_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

# Session State Initialization
if 'patient_data' not in st.session_state:
    st.session_state['patient_data'] = {}
if 'entered_app' not in st.session_state:
    st.session_state['entered_app'] = False

# ---------------------------------------------------------
# 2. UI MODULES
# ---------------------------------------------------------

def render_cover_page():
    st.markdown("<h1 style='text-align: center;'>🛡️ Clinical Risk Monitor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-Driven Pharmacovigilance System</p>", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    if c2.button("🚀 Launch Dashboard", use_container_width=True, type="primary"):
        st.session_state['entered_app'] = True
        st.rerun()

def render_risk_calculator():
    st.subheader("Acute Risk Calculator")
    
    # --- NEW: SMART NOTE PARSER (The "Automation" Layer) ---
    with st.expander("✨ Smart Note Parser (Avoid Manual Entry)", expanded=True):
        st.caption("Paste a clinical note below to auto-populate vitals, labs, and history.")
        soap_note = st.text_area("Patient Description / SOAP Note", placeholder="e.g. 75yo male with history of liver disease. BP 160/95, HR 105, on Warfarin...")
        if st.button("🪄 Extract & Populate Form"):
            if soap_note:
                with st.spinner("AI is parsing the clinical data..."):
                    data = extract_data_from_soap(soap_note)
                    if data:
                        st.session_state['age_input'] = data.get('age', 0)
                        st.session_state['gender_input'] = data.get('gender', "Male")
                        st.session_state['sbp_input'] = data.get('sbp', 0)
                        st.session_state['dbp_input'] = data.get('dbp', 0)
                        st.session_state['hr_input'] = data.get('hr', 0)
                        st.session_state['rr_input'] = data.get('rr', 0)
                        st.session_state['temp_input'] = data.get('temp_c', 0.0)
                        st.session_state['o2_input'] = data.get('o2_sat', 0)
                        st.session_state['creat_input'] = data.get('creatinine', 0.0)
                        st.session_state['glc_input'] = data.get('glucose', 0)
                        st.session_state['inr_input'] = data.get('inr', 0.0)
                        st.session_state['anticoag_input'] = data.get('anticoagulant_use', False)
                        st.session_state['liver_input'] = data.get('liver_disease', False)
                        st.session_state['chf_input'] = data.get('heart_failure', False)
                        st.session_state['ams_input'] = data.get('altered_mental', False)
                        st.success("✅ Form updated! Review the data below.")
                        st.rerun()

    # --- INPUTS CONTAINER ---
    with st.container(border=True):
        st.markdown("#### 📝 Patient Data Entry")
        with st.form("risk_form"):
            col_left, col_right = st.columns([1, 1], gap="medium")
            
            with col_left:
                st.markdown("##### 👤 Patient Profile")
                l1, l2 = st.columns(2)
                age = l1.number_input("Age (Years)", 0, 120, key='age_input')
                gender = l2.selectbox("Gender", ["Male", "Female"], key='gender_input')
                
                w_val, w_unit = st.columns([2, 1]) 
                weight_input = w_val.number_input("Weight", 0.0, 400.0, key='weight_input')
                weight_scale = w_unit.selectbox("Unit", ["kg", "lbs"], key="w_unit")
                height = st.number_input("Height (cm)", 0, 250, key='height_input')
                
                weight_kg = weight_input * 0.453592 if weight_scale == "lbs" else weight_input
                bmi = weight_kg / ((height/100)**2) if height > 0 else 0.0

                st.markdown("##### 🩺 Vitals")
                v1, v2 = st.columns(2)
                sys_bp = v1.number_input("Systolic BP", 0, 300, key='sbp_input')
                dia_bp = v2.number_input("Diastolic BP", 0, 200, key='dbp_input')
                v3, v4 = st.columns(2)
                hr = v3.number_input("Heart Rate", 0, 300, key='hr_input')
                resp_rate = v4.number_input("Resp Rate", 0, 60, key='rr_input')
                v5, v6 = st.columns(2)
                temp_c = v5.number_input("Temp °C", 0.0, 45.0, step=0.1, key='temp_input')
                o2_sat = v6.number_input("O2 Sat %", 0, 100, key='o2_input')

            with col_right:
                st.markdown("##### 🧪 Critical Labs")
                lab1, lab2 = st.columns(2)
                creat = lab1.number_input("Creatinine (mg/dL)", key='creat_input')
                bun = lab2.number_input("BUN", 0, 100, key='bun_input')
                lab3, lab4 = st.columns(2)
                potassium = lab3.number_input("Potassium", key='k_input')
                glucose = lab4.number_input("Glucose", key='glc_input')
                lab7, lab8 = st.columns(2)
                platelets = lab7.number_input("Platelets", key='plt_input')
                inr = lab8.number_input("INR", key='inr_input')
                lactate = st.number_input("Lactate", key='lac_input')

                st.markdown("##### 📋 Medical History")
                h1, h2 = st.columns(2)
                anticoag = h1.checkbox("Anticoagulant Use", key='anticoag_input')
                liver_disease = h2.checkbox("Liver Disease", key='liver_input')
                h3, h4 = st.columns(2)
                heart_failure = h3.checkbox("Heart Failure", key='chf_input')
                gi_bleed = h4.checkbox("History of GI Bleed", key='gib_input')
                altered_mental = st.checkbox("Altered Mental Status (Confusion)", key='ams_input')

            st.write("") 
            submitted = st.form_submit_button("🚀 Run Clinical Analysis", type="primary", use_container_width=True)

    if submitted:
        # LOGIC
        input_df = pd.DataFrame({'age': [age], 'inr': [inr], 'systolic_bp': [sys_bp], 'anticoagulant': [1 if anticoag else 0], 'gender': [gender], 'liver_disease': [1 if liver_disease else 0]})
        try: pred_bleeding = bleeding_model.predict_proba(input_df)[0][1] * 100
        except: pred_bleeding = float(bleeding_model.predict(input_df)[0])

        pred_aki = bk.calculate_aki_risk(age, False, False, sys_bp, False, creat, False, heart_failure)
        pred_sepsis = bk.calculate_sepsis_risk(sys_bp, resp_rate, altered_mental, temp_c)
        status_calc = 'Critical' if (pred_bleeding > 50 or pred_aki > 50 or pred_sepsis >= 2) else 'Stable'
        
        st.session_state['patient_data'] = {
            'age': age, 'gender': gender, 'sys_bp': sys_bp, 'dia_bp': dia_bp, 'hr': hr, 'resp_rate': resp_rate,
            'temp_c': temp_c, 'o2_sat': o2_sat, 'creat': creat, 'inr': inr, 'glucose': glucose, 'lactate': lactate,
            'bleeding_risk': float(pred_bleeding), 'aki_risk': int(pred_aki), 'sepsis_risk': int(pred_sepsis), 'status': status_calc,
            'map_val': (sys_bp + (2 * dia_bp)) / 3 if sys_bp > 0 else 0,
            'shock_index': hr / sys_bp if sys_bp > 0 else 0, 'pulse_pressure': sys_bp - dia_bp
        }
        st.session_state['analysis_results'] = st.session_state['patient_data']
        bk.save_patient_to_db(age, gender, sys_bp, int(pred_aki), float(pred_bleeding), status_calc)

    # RESULTS DISPLAY
    if 'analysis_results' in st.session_state:
        res = st.session_state['analysis_results']
        st.divider()
        st.subheader("📊 Risk Stratification Results")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("🩸 Bleeding Risk", f"{res['bleeding_risk']:.1f}%", "High" if res['bleeding_risk'] > 50 else "Normal")
        r2.metric("💧 AKI Risk", f"{res['aki_risk']}%", "High" if res['aki_risk'] > 50 else "Normal")
        r3.metric("🦠 Sepsis Score", f"{res['sepsis_risk']}", "Alert" if res['sepsis_risk'] >= 2 else "Normal")
        r4.metric("🍬 Glycemia", f"{int(res.get('glucose', 0))} mg/dL")
        
        st.divider()
        # Clinical Alerts
        if res.get('sys_bp') > 180: st.error(f"🚨 HYPERTENSIVE CRISIS ({res['sys_bp']} mmHg)")
        if res.get('o2_sat') < 88 and res.get('o2_sat') > 0: st.error(f"🚨 CRITICAL HYPOXIA ({res['o2_sat']}%)")

        # AI Consult
        if st.button("⚡ Consult AI for Clinical Plan"):
            with st.spinner("Thinking..."):
                response = bk.consult_ai_doctor("risk_assessment", "", res)
                st.session_state['ai_result'] = response
                st.info(response)

def render_history_sql():
    st.subheader("🗄️ Patient History Database")
    df = bk.fetch_history()
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History"):
            bk.clear_history()
            st.rerun()
    else: st.info("No records found.")

def render_dashboard():
    data = st.session_state.get('patient_data', {})
    if not data: st.warning("Please run the Risk Calculator first.")
    else:
        st.subheader(f"🛏️ Bedside Monitor Status: {data.get('status')}")
        st.metric("Heart Rate", f"{int(data.get('hr', 0))} BPM")

def render_batch_analysis():
    st.subheader("Bulk Patient Processing")
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("Batch Data Received. Analysis logic can be applied here.")

def render_medication_checker():
    st.subheader("💊 Interaction Checker")
    d1 = st.text_input("Drug A", placeholder="Warfarin")
    d2 = st.text_input("Drug B", placeholder="Aspirin")
    if d1 and d2:
        res = bk.check_interaction(d1, d2)
        if res: st.error(res)
        else: st.success("No critical interaction found in local DB.")

def render_chatbot():
    st.subheader("📚 Medical Glossary")
    q = st.text_input("Search term:")
    if q: st.write(bk.chatbot_response(q))

def render_ai_diagnostician():
    st.subheader("🧠 AI Clinical Consultant")
    prompt = st.chat_input("Ask a clinical question...")
    if prompt:
        st.session_state.messages = st.session_state.get('messages', [])
        st.session_state.messages.append({"role": "user", "content": prompt})
        resp = bk.consult_ai_doctor("provider", prompt, st.session_state.get('patient_data'))
        st.session_state.messages.append({"role": "assistant", "content": resp})
        for m in st.session_state.messages: st.chat_message(m['role']).write(m['content'])

# --- MAIN APP CONTROLLER ---
if not st.session_state['entered_app']:
    render_cover_page()
else:
    with st.sidebar:
        st.title("Navigation")
        menu = st.radio("Select Module", ["Risk Calculator", "Patient History (SQL)", "Live Dashboard", "Batch Analysis (CSV)", "Medication Checker", "📚 Medical Glossary", "🧠 AI Clinical Consultant"])
    if menu == "Risk Calculator": render_risk_calculator()
    elif menu == "Patient History (SQL)": render_history_sql()
    elif menu == "Live Dashboard": render_dashboard()
    elif menu == "Batch Analysis (CSV)": render_batch_analysis()
    elif menu == "Medication Checker": render_medication_checker()
    elif menu == "📚 Medical Glossary": render_chatbot()
    elif menu == "🧠 AI Clinical Consultant": render_ai_diagnostician()
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
