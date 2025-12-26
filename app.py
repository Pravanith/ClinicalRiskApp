import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import backend as bk
import random
import datetime
import json
import google.generativeai as genai

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
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #212529;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 Clinical Risk Monitor")
st.divider()

# Helper for file timestamps
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")

# --- AI EXTRACTION HELPER ---
def extract_data_from_soap(note_text):
    """Uses Gemini to extract structured data for auto-filling the form."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Extract clinical data from this medical note: "{note_text}"
        Return ONLY a JSON object with these exact keys: 
        age, gender, sbp, dbp, hr, rr, temp, o2, cr, bun, pot, glc, wbc, hgb, plt, inr, lac,
        anticoag (bool), liver (bool), hf (bool), gi (bool), nsaid (bool), chemo (bool), diuretic (bool), acei (bool), insulin (bool), hba1c (bool), ams (bool).
        Use 0 for missing numbers and false for missing booleans.
        """
        response = model.generate_content(prompt)
        return json.loads(response.text.replace('```json', '').replace('```', ''))
    except:
        return None

# Initialize Database
bk.init_db()

# Load AI Model
try:
    bleeding_model = bk.load_bleeding_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

# --- SESSION STATE INITIALIZATION ---
if 'entered_app' not in st.session_state:
    st.session_state['entered_app'] = False
if 'patient_data' not in st.session_state:
    st.session_state['patient_data'] = {}

# Initialize form keys to connect the Parser to the Widgets
form_keys = {
    'age_k': 0, 'gen_k': 'Male', 'sbp_k': 0, 'dbp_k': 0, 'hr_k': 0, 'rr_k': 0, 
    'tmp_k': 0.0, 'o2_k': 0, 'cr_k': 0.0, 'bn_k': 0, 'pt_k': 0.0, 'gl_k': 0, 
    'wb_k': 0.0, 'hg_k': 0.0, 'pl_k': 0, 'ir_k': 0.0, 'lc_k': 0.0,
    'an_k': False, 'lv_k': False, 'hf_k': False, 'gi_k': False, 'ns_k': False, 
    'ch_k': False, 'di_k': False, 'ac_k': False, 'in_k': False, 'hb_k': False, 'am_k': False
}
for k, v in form_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------
# 2. UI MODULES
# ---------------------------------------------------------

# --- COVER PAGE ---
def render_cover_page():
    st.markdown("<h1 style='text-align: center;'>🛡️ Clinical Risk Monitor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-Driven Pharmacovigilance System</p>", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    if c2.button("🚀 Launch Dashboard", use_container_width=True, type="primary"):
        st.session_state['entered_app'] = True
        st.rerun()

# --- MODULE 1: RISK CALCULATOR ---
def render_risk_calculator():
    st.subheader("Acute Risk Calculator")

    # --- THE SMART PARSER ---
    with st.expander("✨ Smart Note Parser (Auto-Fill Form)", expanded=True):
        st.caption("Paste a clinical note below to populate the form instantly.")
        soap_note = st.text_area("Patient Narrative", height=150, placeholder="Example: 75yo male, BP 160/95, HR 105, Cr 1.8, on Warfarin...")
        if st.button("🪄 Extract & Populate Form"):
            with st.spinner("AI parsing clinical data..."):
                extracted = extract_data_from_soap(soap_note)
                if extracted:
                    st.session_state['age_k'] = extracted.get('age', 0)
                    st.session_state['gen_k'] = extracted.get('gender', 'Male')
                    st.session_state['sbp_k'] = extracted.get('sbp', 0)
                    st.session_state['dbp_k'] = extracted.get('dbp', 0)
                    st.session_state['hr_k'] = extracted.get('hr', 0)
                    st.session_state['rr_k'] = extracted.get('rr', 0)
                    st.session_state['tmp_k'] = float(extracted.get('temp', 0.0))
                    st.session_state['o2_k'] = extracted.get('o2', 0)
                    st.session_state['cr_k'] = float(extracted.get('cr', 0.0))
                    st.session_state['bn_k'] = extracted.get('bun', 0)
                    st.session_state['pt_k'] = float(extracted.get('pot', 0.0))
                    st.session_state['gl_k'] = extracted.get('glc', 0)
                    st.session_state['wb_k'] = float(extracted.get('wbc', 0.0))
                    st.session_state['hg_k'] = float(extracted.get('hgb', 0.0))
                    st.session_state['pl_k'] = extracted.get('plt', 0)
                    st.session_state['ir_k'] = float(extracted.get('inr', 0.0))
                    st.session_state['lc_k'] = float(extracted.get('lac', 0.0))
                    st.session_state['an_k'] = extracted.get('anticoag', False)
                    st.session_state['lv_k'] = extracted.get('liver', False)
                    st.session_state['hf_k'] = extracted.get('hf', False)
                    st.session_state['gi_k'] = extracted.get('gi', False)
                    st.session_state['ns_k'] = extracted.get('nsaid', False)
                    st.session_state['ch_k'] = extracted.get('chemo', False)
                    st.session_state['di_k'] = extracted.get('diuretic', False)
                    st.session_state['ac_k'] = extracted.get('acei', False)
                    st.session_state['in_k'] = extracted.get('insulin', False)
                    st.session_state['hb_k'] = extracted.get('hba1c', False)
                    st.session_state['am_k'] = extracted.get('ams', False)
                    st.success("Form Populated!")
                    st.rerun()
    
    # --- INPUTS CONTAINER ---
    with st.container(border=True):
        st.markdown("#### 📝 Patient Data Entry")
        with st.form("risk_form"):
            col_left, col_right = st.columns([1, 1], gap="medium")
            with col_left:
                st.markdown("##### 👤 Patient Profile")
                l1, l2 = st.columns(2)
                age = l1.number_input("Age (Years)", 0, 120, key='age_k')
                gender = l2.selectbox("Gender", ["Male", "Female"], key='gen_k')
                
                w_val, w_unit = st.columns([2, 1]) 
                weight_input = w_val.number_input("Weight", 0.0, 400.0, 0.0)
                weight_scale = w_unit.selectbox("Unit", ["kg", "lbs"], key="w_unit")
                height = st.number_input("Height (cm)", 0, 250, 0)
                
                weight_kg = weight_input * 0.453592 if weight_scale == "lbs" else weight_input
                bmi = weight_kg / ((height/100)**2) if height > 0 else 0.0

                st.markdown("##### 🩺 Vitals")
                v1, v2 = st.columns(2)
                sys_bp = v1.number_input("Systolic BP", 0, 300, key='sbp_k')
                dia_bp = v2.number_input("Diastolic BP", 0, 200, key='dbp_k')
                hr = v3, v4 = st.columns(2)
                hr = v3.number_input("Heart Rate", 0, 300, key='hr_k')
                resp_rate = v4.number_input("Resp Rate", 0, 60, key='rr_k')
                v5, v6 = st.columns(2)
                temp_c = v5.number_input("Temp °C", 0.0, 45.0, key='tmp_k', step=0.1)
                o2_sat = v6.number_input("O2 Sat %", 0, 100, key='o2_k')

            with col_right:
                st.markdown("##### 🧪 Critical Labs")
                lab1, lab2 = st.columns(2)
                creat = lab1.number_input("Creatinine", 0.0, 20.0, key='cr_k')
                bun = lab2.number_input("BUN", 0, 100, key='bn_k')
                lab3, lab4 = st.columns(2)
                potassium = lab3.number_input("Potassium", 0.0, 10.0, key='pt_k')
                glucose = lab4.number_input("Glucose", 0, 1000, key='gl_k')
                lab5, lab6 = st.columns(2)
                wbc = lab5.number_input("WBC", 0.0, 50.0, key='wb_k')
                hgb = lab6.number_input("Hemoglobin", 0.0, 20.0, key='hg_k')
                lab7, lab8 = st.columns(2)
                platelets = lab7.number_input("Platelets", 0, 1000, key='pl_k')
                inr = lab8.number_input("INR", 0.0, 10.0, key='ir_k')
                lactate = st.number_input("Lactate", 0.0, 20.0, key='lc_k')

                st.markdown("##### 📋 Medical History")
                h1, h2 = st.columns(2)
                anticoag = h1.checkbox("Anticoagulant Use", key='an_k')
                liver_disease = h2.checkbox("Liver Disease", key='lv_k')
                h3, h4 = st.columns(2)
                heart_failure = h3.checkbox("Heart Failure", key='hf_k')
                gi_bleed = h4.checkbox("History of GI Bleed", key='gi_k')
                m1, m2 = st.columns(2)
                nsaid = m1.checkbox("NSAID Use", key='ns_k')
                active_chemo = m2.checkbox("Active Chemo", key='ch_k')
                m3, m4 = st.columns(2)
                diuretic = m3.checkbox("Diuretic Use", key='di_k')
                acei = m4.checkbox("ACEi/ARB", key='ac_k')
                m5, m6 = st.columns(2)
                insulin = m5.checkbox("Insulin", key='in_k')
                hba1c_high = m6.checkbox("Uncontrolled Diabetes", key='hb_k')
                altered_mental = st.checkbox("Altered Mental Status", key='am_k')

            submitted = st.form_submit_button("🚀 Run Clinical Analysis", type="primary", use_container_width=True)

    if submitted:
        # LOGIC
        input_df = pd.DataFrame({'age': [age], 'inr': [inr], 'systolic_bp': [sys_bp], 'anticoagulant': [1 if anticoag else 0], 'gender': [gender], 'liver_disease': [1 if liver_disease else 0]})
        try: pred_bleeding = bleeding_model.predict_proba(input_df)[0][1] * 100
        except: pred_bleeding = float(bleeding_model.predict(input_df)[0])

        pred_aki = bk.calculate_aki_risk(age, diuretic, acei, sys_bp, active_chemo, creat, nsaid, heart_failure)
        pred_sepsis = bk.calculate_sepsis_risk(sys_bp, resp_rate, altered_mental, temp_c)
        status_calc = 'Critical' if (pred_bleeding > 50 or pred_aki > 50) else 'Stable'
        
        st.session_state['patient_data'] = {'age': age, 'gender': gender, 'sys_bp': sys_bp, 'bleeding_risk': pred_bleeding, 'aki_risk': pred_aki, 'sepsis_risk': pred_sepsis, 'status': status_calc, 'glucose': glucose}
        st.session_state['analysis_results'] = st.session_state['patient_data']
        bk.save_patient_to_db(age, gender, sys_bp, int(pred_aki), float(pred_bleeding), status_calc)

    if 'analysis_results' in st.session_state:
        res = st.session_state['analysis_results']
        st.divider()
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("🩸 Bleeding Risk", f"{res['bleeding_risk']:.1f}%")
        r2.metric("💧 AKI Risk", f"{res['aki_risk']}%")
        r3.metric("🦠 Sepsis Score", f"{res['sepsis_risk']}")
        r4.metric("🍬 Glycemia", f"{int(res.get('glucose', 0))} mg/dL")



def render_history_sql():
    st.subheader("🗄️ Patient History Database")
    df = bk.fetch_history()
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear All Records"):
            bk.clear_history()
            st.rerun()
    else: st.info("Database is empty.")

def render_dashboard():
    data = st.session_state.get('patient_data', {})
    if not data: st.warning("Please run the Risk Calculator first.")
    else:
        st.subheader(f"🛏️ Bedside Monitor: {data.get('status', 'Unknown')}")
        if st.button("✨ Generate Discharge Note", type="primary"):
            st.session_state['latest_discharge_note'] = bk.generate_discharge_summary(data)
        if 'latest_discharge_note' in st.session_state:
            st.text_area("Summary:", value=st.session_state['latest_discharge_note'], height=200)

def render_batch_analysis():
    st.subheader("Bulk Patient Processing")
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("Batch Data Received.")

def render_medication_checker():
    st.subheader("💊 Medication Checker")
    d1 = st.text_input("Drug A")
    d2 = st.text_input("Drug B")
    if d1 and d2:
        res = bk.check_interaction(d1, d2)
        st.write(res)
        if st.button("⚡ AI Analysis"):
            st.markdown(bk.analyze_drug_interactions([d1, d2]))

def render_chatbot():
    st.subheader("AI Clinical Assistant")
    q = st.text_input("Ask a clinical question:")
    if q: st.write(bk.chatbot_response(q))

def render_ai_diagnostician():
    st.subheader("🧠 AI-Powered Clinical Consultant")
    prompt = st.chat_input("Type your query...")
    if prompt:
        st.write(bk.consult_ai_doctor("provider", prompt, st.session_state.get('patient_data', {})))

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
