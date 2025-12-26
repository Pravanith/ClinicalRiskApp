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
        age, gender, sys_bp, dia_bp, hr, resp_rate, temp_c, o2_sat, creat, bun, pot, glc, wbc, hgb, plt, inr, lac,
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
if 'patient_data' not in st.session_state:
    st.session_state['patient_data'] = {}
if 'entered_app' not in st.session_state:
    st.session_state['entered_app'] = False

# Initialize form keys to prevent errors
form_keys = {
    'age_k': 0, 'gen_k': 'Male', 'sbp_k': 0, 'dbp_k': 0, 'hr_k': 0, 'rr_k': 0, 
    'tmp_k': 0.0, 'o2_k': 0, 'cr_k': 0.0, 'bun_k': 0, 'pot_k': 0.0, 'glc_k': 0, 
    'wbc_k': 0.0, 'hgb_k': 0.0, 'plt_k': 0, 'inr_k': 0.0, 'lac_k': 0.0,
    'ant_k': False, 'liv_k': False, 'hf_k': False, 'gi_k': False, 'nsd_k': False, 
    'chm_k': False, 'diu_k': False, 'ace_k': False, 'ins_k': False, 'hba_k': False, 'ams_k': False
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

    # --- NEW: AI AUTO-FILL (SMART PARSER) ---
    with st.expander("✨ Smart Note Parser (Auto-Fill Form)", expanded=True):
        st.caption("Paste a clinical note below to populate the form instantly.")
        soap_note = st.text_area("Patient Narrative", height=150, placeholder="Example: 75yo male, BP 160/95, HR 105, Cr 1.8, on Warfarin, hx of CHF...")
        if st.button("🪄 Extract & Populate Form"):
            with st.spinner("AI parsing clinical data..."):
                extracted = extract_data_from_soap(soap_note)
                if extracted:
                    st.session_state['age_k'] = extracted.get('age', 0)
                    st.session_state['gen_k'] = extracted.get('gender', 'Male')
                    st.session_state['sbp_k'] = extracted.get('sys_bp', 0)
                    st.session_state['dbp_k'] = extracted.get('dia_bp', 0)
                    st.session_state['hr_k'] = extracted.get('hr', 0)
                    st.session_state['rr_k'] = extracted.get('resp_rate', 0)
                    st.session_state['tmp_k'] = float(extracted.get('temp_c', 0.0))
                    st.session_state['o2_k'] = extracted.get('o2_sat', 0)
                    st.session_state['cr_k'] = float(extracted.get('creat', 0.0))
                    st.session_state['bun_k'] = extracted.get('bun', 0)
                    st.session_state['pot_k'] = float(extracted.get('pot', 0.0))
                    st.session_state['glc_k'] = extracted.get('glc', 0)
                    st.session_state['wbc_k'] = float(extracted.get('wbc', 0.0))
                    st.session_state['hgb_k'] = float(extracted.get('hgb', 0.0))
                    st.session_state['plt_k'] = extracted.get('plt', 0)
                    st.session_state['inr_k'] = float(extracted.get('inr', 0.0))
                    st.session_state['lac_k'] = float(extracted.get('lac', 0.0))
                    st.session_state['ant_k'] = extracted.get('anticoag', False)
                    st.session_state['liv_k'] = extracted.get('liver', False)
                    st.session_state['hf_k'] = extracted.get('hf', False)
                    st.session_state['gi_k'] = extracted.get('gi', False)
                    st.session_state['nsd_k'] = extracted.get('nsaid', False)
                    st.session_state['chm_k'] = extracted.get('chemo', False)
                    st.session_state['diu_k'] = extracted.get('diuretic', False)
                    st.session_state['ace_k'] = extracted.get('acei', False)
                    st.session_state['ins_k'] = extracted.get('insulin', False)
                    st.session_state['hba_k'] = extracted.get('hba1c', False)
                    st.session_state['ams_k'] = extracted.get('ams', False)
                    st.success("Form Populated! Please review and click 'Run Analysis'.")
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
                sys_bp = v1.number_input("Systolic BP (Normal: 110-120)", 0, 300, key='sbp_k')
                dia_bp = v2.number_input("Diastolic BP (Normal: 70-80)", 0, 200, key='dbp_k')
                
                v3, v4 = st.columns(2)
                hr = v3.number_input("Heart Rate (Normal: 60-100)", 0, 300, key='hr_k')
                resp_rate = v4.number_input("Resp Rate (Normal: 12-20)", 0, 60, key='rr_k')
                
                v5, v6 = st.columns(2)
                temp_c = v5.number_input("Temp °C (Normal: 36.5-37.5)", 0.0, 45.0, key='tmp_k', step=0.1)
                o2_sat = v6.number_input("O2 Sat % (Normal: >95%)", 0, 100, key='o2_k')

            with col_right:
                st.markdown("##### 🧪 Critical Labs")
                lab1, lab2 = st.columns(2)
                creat = lab1.number_input("Creatinine (0.6-1.2 mg/dL)", 0.0, 20.0, key='cr_k')
                bun = lab2.number_input("Blood Urea Nitrogen (7-20)", 0, 100, key='bun_k')
                
                lab3, lab4 = st.columns(2)
                potassium = lab3.number_input("Potassium (3.5-5.0 mmol/L)", 0.0, 10.0, key='pot_k')
                glucose = lab4.number_input("Glucose (70-100 mg/dL)", 0, 1000, key='glc_k')
                
                lab5, lab6 = st.columns(2)
                wbc = lab5.number_input("WBC (4.5-11.0 10^9/L)", 0.0, 50.0, key='wbc_k')
                hgb = lab6.number_input("Hemoglobin (13.5-17.5 g/dL)", 0.0, 20.0, key='hgb_k')
                
                lab7, lab8 = st.columns(2)
                platelets = lab7.number_input("Platelets (150-450 10^9/L)", 0, 1000, key='plt_k')
                inr = lab8.number_input("INR (Clotting Time) [0.9-1.1]", 0.0, 10.0, key='inr_k')
                
                lactate = st.number_input("Lactate (Normal: < 2.0 mmol/L)", 0.0, 20.0, key='lac_k')

                st.markdown("##### 📋 Medical History")
                h1, h2 = st.columns(2)
                anticoag = h1.checkbox("Anticoagulant Use", key='ant_k')
                liver_disease = h2.checkbox("Liver Disease", key='liv_k')
                
                h3, h4 = st.columns(2)
                heart_failure = h3.checkbox("Heart Failure", key='hf_k')
                gi_bleed = h4.checkbox("History of GI Bleed", key='gi_k')
                
                m1, m2 = st.columns(2)
                nsaid = m1.checkbox("NSAID Use", key='nsd_k')
                active_chemo = m2.checkbox("Active Chemo", key='chm_k')
                
                m3, m4 = st.columns(2)
                diuretic = m3.checkbox("Diuretic Use", key='diu_k')
                acei = m4.checkbox("ACEi/ARB", key='ace_k')
                
                m5, m6 = st.columns(2)
                insulin = m5.checkbox("Insulin", key='ins_k')
                hba1c_high = m6.checkbox("Uncontrolled Diabetes", key='hba_k')
                
                altered_mental = st.checkbox("Altered Mental Status (Confusion)", key='ams_k')
                pain = 0

            st.write("") 
            submitted = st.form_submit_button("🚀 Run Clinical Analysis", type="primary", use_container_width=True)

    # --- LOGIC & RESULTS ---
    if submitted:
        if age > 0 and sys_bp > 0:
            input_df = pd.DataFrame({
                'age': [age], 'inr': [inr], 'systolic_bp': [sys_bp],
                'anticoagulant': [1 if anticoag else 0], 'gender': [gender],
                'liver_disease': [1 if liver_disease else 0]
            })

            try:
                pred_bleeding = bleeding_model.predict_proba(input_df)[0][1] * 100
            except AttributeError:
                pred_bleeding = float(bleeding_model.predict(input_df)[0])

            pred_aki = bk.calculate_aki_risk(age, diuretic, acei, sys_bp, active_chemo, creat, nsaid, heart_failure)
            pred_sepsis = bk.calculate_sepsis_risk(sys_bp, resp_rate, altered_mental, temp_c)
            pred_hypo = bk.calculate_hypoglycemic_risk(insulin, (creat>1.3), hba1c_high, False)
            sirs_score = bk.calculate_sirs_score(temp_c, hr, resp_rate, wbc)
            
            # HAS-BLED Calculator (Heuristic Check)
            has_bled = 0
            if sys_bp > 160: has_bled += 1
            if creat > 2.2 or liver_disease: has_bled += 1
            if gi_bleed: has_bled += 1
            if inr > 1.0: has_bled += 1
            if age > 65: has_bled += 1
            if nsaid or anticoag: has_bled += 1

            status_calc = 'Critical' if (pred_bleeding > 50 or pred_aki > 50 or pred_sepsis >= 2) else 'Stable'
            bk.save_patient_to_db(age, gender, sys_bp, int(pred_aki), float(pred_bleeding), status_calc)
            
            st.session_state['patient_data'] = {
                'id': f"Patient-{age}-{int(sys_bp)}", 
                'age': age, 'gender': gender, 'weight': weight_kg,
                'sys_bp': sys_bp, 'dia_bp': dia_bp, 'hr': hr, 'resp_rate': resp_rate, 
                'temp_c': temp_c, 'o2_sat': o2_sat, 'pain': pain,
                'creat': creat, 'potassium': potassium, 'inr': inr, 'bun': bun,
                'wbc': wbc, 'hgb': hgb, 'platelets': platelets, 'lactate': lactate, 'glucose': glucose,
                'bleeding_risk': float(pred_bleeding), 'aki_risk': int(pred_aki),
                'sepsis_risk': int(pred_sepsis), 'hypo_risk': int(pred_hypo),
                'sirs_score': sirs_score, 'status': status_calc, 'map_val': (sys_bp + (2 * dia_bp)) / 3, 
                'bmi': bmi, 'has_bled': has_bled,
                'shock_index': hr / sys_bp if sys_bp > 0 else 0, 
                'pulse_pressure': sys_bp - dia_bp, 
                'bun_creat_ratio': bun / creat if creat > 0 else 0,
                'anticoag': anticoag, 'liver_disease': liver_disease, 'diuretic': diuretic,
                'acei': acei, 'heart_failure': heart_failure, 'hba1c_high': hba1c_high,
                'altered_mental': altered_mental
            }
            st.session_state['analysis_results'] = st.session_state['patient_data']

    if 'analysis_results' in st.session_state:
        res = st.session_state['analysis_results']
        st.divider()
        st.subheader("📊 Risk Stratification Results")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("🩸 Bleeding Risk", f"{res['bleeding_risk']:.1f}%", "High" if res['bleeding_risk'] > 50 else "Normal")
        r2.metric("💧 AKI Risk", f"{res['aki_risk']}%", "High" if res['aki_risk'] > 50 else "Normal")
        r3.metric("🦠 Sepsis Score", f"{res['sepsis_risk']}", "Alert" if res['sepsis_risk'] >= 2 else "Normal")
        
        current_gluc = res.get('glucose', 0)
        if current_gluc > 180:
             r4.metric("🍬 Glycemia", f"{int(current_gluc)} mg/dL", "Hyper (High)", delta_color="inverse")
        elif current_gluc > 0 and current_gluc < 70:
             r4.metric("🍬 Glycemia", f"{int(current_gluc)} mg/dL", "Hypo (Low)", delta_color="inverse")
        else:
             r4.metric("🍬 Hypo Risk", f"{res.get('hypo_risk', 0)}%", "Normal")

        st.divider()
        c_ai, c_txt = st.columns([1, 3])
        with c_ai:
            st.markdown("#### 🤖 AI Assessment")
            if st.button("⚡ Consult AI"):
                with st.spinner("Thinking..."):
                    ai_context = {'age': res['age'], 'sbp': res['sys_bp'], 'bleeding_risk': res['bleeding_risk'], 'aki_risk': res['aki_risk'], 'shock_index': res['shock_index']}
                    st.session_state['ai_result'] = bk.consult_ai_doctor("risk_assessment", "", ai_context)
        with c_txt:
            if 'ai_result' in st.session_state:
                st.info(st.session_state['ai_result'])

# --- MODULE 2: PATIENT HISTORY ---
def render_history_sql():
    st.subheader("🗄️ Patient History Database")
    df = bk.fetch_history()
    if not df.empty:
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(df, use_container_width=True, height=400, hide_index=True)
        with st.expander("⚙️ Database Management"):
            if st.button("🗑️ Clear All Records", type="secondary"):
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
    st.subheader(f"🛏️ Bedside Monitor: {data.get('id', 'Unknown')}")
    st.caption(f"Status: **{data.get('status', 'Unknown')}**")
    
    if st.button("✨ Generate Discharge Note", type="primary"):
        with st.spinner("Consulting Gemini..."):
            st.session_state['latest_discharge_note'] = bk.generate_discharge_summary(data)
    
    if 'latest_discharge_note' in st.session_state:
        st.text_area("Discharge Summary:", value=st.session_state['latest_discharge_note'], height=200)

# --- MODULE 4: BATCH ANALYSIS ---
def render_batch_analysis():
    st.subheader("Bulk Patient Processing & Diagnostic Triage")
    uploaded_csv = st.file_uploader("Upload Patient Data (CSV)", type=["csv"])
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            st.success(f"Processed {len(df)} records.")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# --- MODULE 5: MEDICATION CHECKER ---
def render_medication_checker():
    st.subheader("💊 Drug-Drug Interaction Checker")
    col_d1, col_d2 = st.columns(2)
    d1 = col_d1.text_input("Drug A", placeholder="e.g. Warfarin")
    d2 = col_d2.text_input("Drug B", placeholder="e.g. Ibuprofen")
    if d1 and d2:
        res = bk.check_interaction(d1, d2)
        if res: st.error(res)
        if st.button("⚡ Analyze Interaction with AI"):
            with st.spinner("Analyzing..."):
                st.markdown(bk.analyze_drug_interactions([d1, d2]))

# --- MODULE 6: CHATBOT ---
def render_chatbot():
    st.subheader("AI Clinical Assistant")
    q = st.text_input("Ask a clinical question:")
    if q: st.write(bk.chatbot_response(q))

# --- MODULE 7: AI DIAGNOSTICIAN ---
def render_ai_diagnostician():
    st.subheader("🧠 AI-Powered Clinical Consultant")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I am your AI Clinical Consultant."}]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Type your clinical query here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.spinner("Thinking..."):
            response = bk.consult_ai_doctor("provider", prompt, st.session_state.get('patient_data', {}))
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
        menu = st.radio("Select Module", ["Risk Calculator", "Patient History (SQL)", "Live Dashboard", "Batch Analysis (CSV)", "Medication Checker", "📚 Medical Glossary", "🧠 AI Clinical Consultant"])
        if "admin" in st.query_params: 
            st.sidebar.divider()
            if st.sidebar.button("⚡ Force Retrain Model"):
                import train_model
                train_model.train()
                st.success("✅ Model Retrained!")

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
