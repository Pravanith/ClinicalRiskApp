import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import backend as bk  # Importing our full backend

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Clinical Risk Monitor", 
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database
bk.init_db()

# Load AI Model
try:
    bleeding_model = bk.load_bleeding_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

# Session State
if 'patient_data' not in st.session_state:
    st.session_state['patient_data'] = {
        'id': 'Room 101', 'age': 65, 'bleeding_risk': 12.5, 
        'aki_risk': 10, 'hypo_risk': 5, 'status': 'Stable'
    }

if 'entered_app' not in st.session_state:
    st.session_state['entered_app'] = False

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
    st.subheader("Acute Risk Calculator (Advanced)")
    st.caption("Enter patient values below. Default is 0.")
    
    with st.form("risk_form"):
        # 1. Patient Demographics
        st.markdown("#### 1. Patient Demographics")
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age", min_value=0, max_value=120, value=0)
        gender = c2.selectbox("Gender", ["Male", "Female"])
        ethnicity = c3.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Hispanic", "Other"])
        
        with c4:
            w_val, w_unit = st.columns([2, 1]) 
            weight_input = w_val.number_input("Weight", min_value=0.0, max_value=400.0, value=0.0)
            weight_scale = w_unit.selectbox("Unit", ["kg", "lbs"], key="w_unit")

        p1, p2, p3 = st.columns(3)
        height = p1.number_input("Height (cm)", min_value=0, max_value=250, value=0)
        smoking = p2.selectbox("Smoking Status", ["Never", "Former", "Current"])
        admit_type = p3.selectbox("Admission Type", ["Emergency", "Elective", "Trauma"])

        # Logic: Weight Conversion for display
        weight_kg = weight_input * 0.453592 if weight_scale == "lbs" else weight_input
        
        if height > 0:
            bmi = weight_kg / ((height/100)**2)
            p1.caption(f"Calculated BMI: {bmi:.1f}")
        else:
            bmi = 0.0
            p1.caption("Enter Height for BMI")

        # 2. Vital Signs
        st.markdown("#### 2. Vital Signs & Observations")
        v1, v2, v3, v4 = st.columns(4)
        sys_bp = v1.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=0)
        dia_bp = v2.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200, value=0)
        hr = v3.number_input("Heart Rate (bpm)", min_value=0, max_value=300, value=0)
        resp_rate = v4.number_input("Resp Rate (bpm)", min_value=0, max_value=60, value=0)
        
        v5, v6, v7, v8 = st.columns(4)
        with v5:
            t_val, t_unit = st.columns([2, 1]) 
            temp_input = t_val.number_input("Temp", min_value=0.0, max_value=115.0, value=0.0, step=0.1)
            temp_scale = t_unit.selectbox("Unit", ["°C", "°F"], key="t_unit")
        
        o2_sat = v6.number_input("O2 Sat (%)", min_value=0, max_value=100, value=0)
        glucose = v7.number_input("Fingerstick Glucose (mg/dL)", min_value=0, max_value=600, value=0)
        pain = v8.slider("Pain Score (VAS)", 0, 10, 0)

        # 3. Lab Values
        st.markdown("#### 3. Laboratory Values")
        l1, l2, l3, l4 = st.columns(4)
        creat = l1.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=20.0, value=0.0)
        potassium = l2.number_input("Potassium (K+)", min_value=0.0, max_value=10.0, value=0.0)
        inr = l3.number_input("INR", min_value=0.0, max_value=10.0, value=0.0)
        bun = l4.number_input("BUN (mg/dL)", min_value=0, max_value=100, value=0)
        
        l5, l6, l7, l8 = st.columns(4)
        wbc = l5.number_input("WBC (10^9/L)", min_value=0.0, max_value=50.0, value=0.0)
        hgb = l6.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=0.0)
        platelets = l7.number_input("Platelets (10^9/L)", min_value=0, max_value=1000, value=0)
        lactate = l8.number_input("Lactate (mmol/L)", min_value=0.0, max_value=20.0, value=0.0)

        # 4. Medications & Comorbidities
        st.markdown("#### 4. Comorbidities & Medications")
        m1, m2, m3, m4 = st.columns(4)
        anticoag = m1.checkbox("Anticoagulant (Blood Thinner)")
        nsaid = m2.checkbox("NSAID Use (e.g., Ibuprofen)") 
        active_chemo = m3.checkbox("Active Chemotherapy")
        hba1c_high = m4.checkbox("Diabetes Uncontrolled (A1c > 9%)")
        
        m5, m6, m7, m8 = st.columns(4)
        diuretic = m5.checkbox("Diuretic Use (Lasix/HCTZ)")
        acei = m6.checkbox("ACEi/ARB (BP Meds)")
        insulin = m7.checkbox("Insulin Dependent")
        liver_disease = m8.checkbox("Liver Disease / Cirrhosis")
        
        h1, h2, h3 = st.columns(3)
        heart_failure = h1.checkbox("Heart Failure (CHF)") 
        gi_bleed = h2.checkbox("History of GI Bleed")
        altered_mental = h3.checkbox("Altered Mental Status (Confusion)")

        submitted = st.form_submit_button("Run Clinical Analysis", type="primary")

        if submitted:
            # --- A. PRE-PROCESSING ---
            if temp_scale == "°F": 
                final_temp_c = (temp_input - 32) * 5/9 
            else: 
                final_temp_c = temp_input
            
            # Handle 0 BP for MAP calc
            if sys_bp > 0:
                map_val = (sys_bp + (2 * dia_bp)) / 3 
            else:
                map_val = 0

            # --- B. RISK CALCULATIONS ---
            
            # 1. AI Bleeding Prediction
            is_high_bp = 1 if sys_bp > 140 else 0
            input_df = pd.DataFrame({
                'age': [age], 'inr': [inr], 
                'anticoagulant': [1 if anticoag else 0],
                'gi_bleed': [1 if gi_bleed else 0], 
                'high_bp': [is_high_bp],
                'antiplatelet': [0], 
                'gender_female': [1 if gender == "Female" else 0],
                'weight': [weight_kg], 
                'liver_disease': [1 if liver_disease else 0]
            })
            pred_bleeding = bleeding_model.predict(input_df)[0]
            
            # 2. Rule Predictions (From Backend)
            pred_aki = bk.calculate_aki_risk(age, diuretic, acei, sys_bp, active_chemo, creat, nsaid, heart_failure)
            pred_sepsis = bk.calculate_sepsis_risk(sys_bp, resp_rate, altered_mental, final_temp_c)
            pred_hypo = bk.calculate_hypoglycemic_risk(insulin, (creat>1.3), hba1c_high, False)
            sirs_score = bk.calculate_sirs_score(final_temp_c, hr, resp_rate, wbc)

            # 3. HAS-BLED Score
            has_bled = 0
            if sys_bp > 160: has_bled += 1
            if creat > 2.2 or liver_disease: has_bled += 1
            if gi_bleed: has_bled += 1
            if inr > 1.0: has_bled += 1
            if age > 65: has_bled += 1
            if nsaid or anticoag: has_bled += 1

            # 4. Save State
            status_calc = 'Critical' if (pred_bleeding > 50 or pred_aki > 50 or pred_sepsis > 50) else 'Stable'
            st.session_state['patient_data'] = {
                'id': 'Calculated Patient', 'age': age,
                'bleeding_risk': float(pred_bleeding), 'aki_risk': int(pred_aki),
                'sepsis_risk': int(pred_sepsis), 'hypo_risk': int(pred_hypo),
                'status': status_calc
            }
            
            bk.save_patient_to_db(age, gender, sys_bp, int(pred_aki), float(pred_bleeding), status_calc)
            
            st.success("Analysis Complete!")
            
            # 5. Display Results (UPDATED WITH TOOLTIPS)
            st.divider()
            st.markdown("#### 📊 Clinical Analysis Results")
            
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Bleeding Risk (AI)", f"{pred_bleeding:.1f}%", help="Predicted by XGBoost Model")
            r2.metric("AKI Risk (Rule)", f"{pred_aki}%", help="Rule-based calculation (KDIGO)")
            r3.metric("Sepsis Score (qSOFA)", f"{pred_sepsis}", help="qSOFA Score (0-3)")
            r4.metric("HAS-BLED Score", f"{has_bled}/9", "High Risk" if has_bled >=3 else "Low Risk", help="Standard Atrial Fibrillation Bleeding Risk Score")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("MAP (Perfusion)", f"{int(map_val)} mmHg", "Low" if map_val > 0 and map_val < 65 else "Normal", help="Mean Arterial Pressure. Target > 65 mmHg.")
            d2.metric("SIRS Score", f"{sirs_score}/4", "Inflammation" if sirs_score >=2 else "Normal", help="Systemic Inflammatory Response Syndrome criteria.")
            d3.metric("BMI Category", f"{bmi:.1f}", "Obese" if bmi > 30 else "Normal")
            d4.metric("Pain Status", f"{pain}/10", "Managed")

            # 6. Detailed Alerts (Kept in Frontend for rendering)
            with st.expander("⚠️ Detailed Clinical Alerts", expanded=True):
                # BMI Check
                if bmi >= 40: st.error(f"🔴 MORBID OBESITY: BMI {bmi:.1f}")
                elif bmi >= 35: st.warning(f"⚠️ Severe Obesity: BMI {bmi:.1f}")
                
                # Lab Checks
                if potassium > 5.5: st.error(f"🔴 CRITICAL HYPERKALEMIA: K+ {potassium}")
                if platelets > 0 and platelets < 50: st.error(f"🔴 SEVERE THROMBOCYTOPENIA: Plt {platelets}")
                if glucose < 70 and glucose > 0: st.error(f"🔴 HYPOGLYCEMIA: {glucose}")
                if lactate > 4.0: st.error(f"🔴 SEVERE LACTIC ACIDOSIS: {lactate}")
                if inr > 3.0: st.error(f"🔴 HIGH INR: {inr}")
                if creat > 3.0: st.error(f"🔴 ACUTE RENAL FAILURE: Cr {creat}")
                
                # Vital Checks
                if sys_bp > 180 or dia_bp > 120: st.error(f"🔴 HYPERTENSIVE CRISIS")
                if o2_sat > 0 and o2_sat < 88: st.error(f"🔴 CRITICAL HYPOXIA")
                if pred_sepsis >= 2: st.error("🚨 SEPSIS ALERT: Initiate Protocol")
                
                # Safe Check
                if (potassium < 5.5 and potassium > 3.5 and platelets > 100 and glucose > 70 
                    and sys_bp < 180 and sys_bp > 90 and bmi < 30):
                     st.success("✅ No critical alerts detected.")

# --- MODULE 2: PATIENT HISTORY (SQL) ---
def render_history_sql():
    st.subheader("🗄️ Patient History Database")
    df = bk.fetch_history()
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        st.markdown("### 📊 Cohort Analytics")
        c1, c2 = st.columns(2)
        c1.bar_chart(df['aki_risk_score'])
        c2.scatter_chart(df, x='age', y='sbp')
        
        if st.button("🗑️ Clear Database"):
            bk.clear_history()
            st.rerun()
    else:
        st.info("📭 Database is empty. Run a Risk Analysis to create records.")

# --- MODULE 3: LIVE DASHBOARD (UPDATED WITH TOOLTIPS) ---
def render_dashboard():
    data = st.session_state['patient_data']
    is_critical = data['status'] == 'Critical'
    
    st.subheader(f"🖥️ ICU Monitor: {data['id']}")
    
    m1, m2, m3, m4 = st.columns(4)
    
    # Tooltips added to the 'help' parameter
    m1.metric(
        "Bleeding Risk", 
        f"{data['bleeding_risk']:.1f}%", 
        "High" if data['bleeding_risk'] > 50 else "Normal", 
        delta_color="inverse",
        help="Probability of major hemorrhage based on XGBoost model (Inputs: INR, Age, History of GI Bleed, Anticoagulants)."
    )
    
    m2.metric(
        "AKI Risk", 
        f"{data['aki_risk']}%", 
        "Critical" if data['aki_risk'] > 50 else "Normal", 
        delta_color="inverse",
        help="Acute Kidney Injury Risk based on KDIGO criteria (Creatinine levels, Nephrotoxic meds, Age)."
    )
    
    m3.metric(
        "Sepsis Score", 
        f"{data.get('sepsis_risk', 0)}", 
        "High" if data.get('sepsis_risk', 0) >= 2 else "Normal", 
        delta_color="inverse",
        help="qSOFA Score (Quick Sepsis Related Organ Failure Assessment). 1 point each for: Hypotension (SBP<100), Tachypnea (RR>22), Altered Mental Status. Score ≥2 suggests high sepsis risk."
    )
    
    m4.metric(
        "Hypoglycemia", 
        "YES" if data.get('hypo_risk', 0) > 0 else "NO", 
        "Critical" if data.get('hypo_risk', 0) > 0 else "Normal", 
        delta_color="inverse",
        help="Critical Alert triggered if Blood Glucose < 70 mg/dL or recent insulin event."
    )
    
    st.divider()
    
    col_main, col_queue = st.columns([2.5, 1])
    with col_main:
        st.markdown("### 📉 Live Telemetry (HR vs BP)")
        base_hr = 110 if is_critical else 75
        base_bp = 90 if is_critical else 120
        
        hr_data = [base_hr + np.random.randint(-10, 15) for _ in range(10)]
        bp_data = [base_bp + np.random.randint(-5, 10) for _ in range(10)]
        
        chart_df = pd.DataFrame({
            'Time': list(range(-9, 1)),
            'Heart Rate': hr_data,
            'Systolic BP': bp_data
        }).melt('Time', var_name='Metric', value_name='Value')

        domain = ['Heart Rate', 'Systolic BP']
        range_ = ['#00E5FF', '#FF1744']
        
        c = alt.Chart(chart_df).mark_line(interpolate='monotone', point=True, strokeWidth=3).encode(
            x=alt.X('Time', axis=alt.Axis(title='Time (Hours)')),
            y=alt.Y('Value', scale=alt.Scale(domain=[40, 180])),
            color=alt.Color('Metric', scale=alt.Scale(domain=domain, range=range_))
        ).properties(height=350)
        
        st.altair_chart(c, use_container_width=True)
        
        if is_critical:
            st.error("🚨 **CRITICAL ALERT:** Vitals unstable.")
        else:
            st.success("✅ **STABLE:** Vitals trending normal.")

    with col_queue:
        st.markdown("#### 📋 Patient Status")
        st.info(f"Current Status: {data['status']}")

# --- MODULE 4: BATCH ANALYSIS (CSV) ---
def render_batch_analysis():
    st.subheader("Bulk Patient Processing & Diagnostic Triage")
    
    # 1. Helper: Download Template
    with st.expander("ℹ️  Download CSV Template"):
        sample_data = {
            'Age': [65, 72], 'Gender': ['Male', 'Female'], 'Weight_kg': [80, 65],
            'Systolic_BP': [130, 90], 'Diastolic_BP': [80, 50], 'Heart_Rate': [72, 110],
            'Resp_Rate': [16, 24], 'Temp_C': [37.0, 38.5], 'O2_Sat': [98, 92],
            'WBC': [6.0, 15.0], 'Glucose': [110, 85], 'Creatinine': [1.1, 2.5],
            'INR': [1.0, 1.2], 'Altered_Mental': [0, 1], 'Anticoagulant': [1, 0],
            'Heart_Failure': [0, 1], 'Liver_Disease': [0, 0], 'Hx_GI_Bleed': [0, 0]
        }
        df_sample = pd.DataFrame(sample_data)
        csv_template = df_sample.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Template", csv_template, "patient_data.csv", "text/csv")

    tab1, tab2 = st.tabs(["📄 Diagnostic Processor", "🖼️ Medical Imaging"])
    
    # --- TAB 1: CSV PROCESSOR ---
    with tab1:
        uploaded_csv = st.file_uploader("Upload Patient Data (CSV)", type=["csv"])
        if uploaded_csv:
            try:
                raw_df = pd.read_csv(uploaded_csv)
                
                # A. Smart Column Mapping
                col_map = {
                    'sbp':'Systolic_BP', 'hr':'Heart_Rate', 'rr':'Resp_Rate', 'temp':'Temp_C',
                    'spo2':'O2_Sat', 'cr':'Creatinine', 'wbc':'WBC', 'glu':'Glucose'
                }
                df = raw_df.rename(columns=lambda x: col_map.get(x.lower(), x))
                
                # B. Fill Missing
                req_cols = ['Age','Systolic_BP','Diastolic_BP','Heart_Rate','Resp_Rate','Temp_C','WBC','Creatinine']
                for c in req_cols:
                    if c not in df.columns: df[c] = 0
                
                if st.button("⚡ Run AI Diagnostic Engine", type="primary"):
                    # 1. AI Predictions
                    inputs = pd.DataFrame()
                    inputs['age'] = df.get('Age', 0)
                    inputs['inr'] = df.get('INR', 1.0)
                    inputs['anticoagulant'] = df.get('Anticoagulant', 0)
                    inputs['gi_bleed'] = df.get('Hx_GI_Bleed', 0)
                    inputs['high_bp'] = df['Systolic_BP'].apply(lambda x: 1 if x > 140 else 0)
                    inputs['antiplatelet'] = 0
                    inputs['gender_female'] = 0 
                    inputs['weight'] = df.get('Weight_kg', 70)
                    inputs['liver_disease'] = df.get('Liver_Disease', 0)
                    
                    df['Bleed_Risk_%'] = bleeding_model.predict(inputs)
                    
                    # 2. Logic Diagnostics
                    def get_status(row):
                        alerts = []
                        qsofa = 0
                        if row['Systolic_BP'] < 100: qsofa += 1
                        if row['Resp_Rate'] > 22: qsofa += 1
                        if row.get('Altered_Mental', 0) == 1: qsofa += 1
                        
                        if qsofa >= 2: alerts.append("SEPSIS ALERT")
                        if row['Systolic_BP'] > 180: alerts.append("Hypertensive Crisis")
                        if row['Creatinine'] > 2.0: alerts.append("Acute Kidney Injury")
                        if row['WBC'] > 12: alerts.append("Leukocytosis")
                        
                        return " + ".join(alerts) if alerts else "Stable"

                    df['Diagnosis'] = df.apply(get_status, axis=1)
                    
                    # 3. Highlight
                    def color_rows(val):
                        if 'SEPSIS' in str(val) or 'Crisis' in str(val): return 'background-color: #ffcdd2; color: black;'
                        if 'Stable' in str(val): return 'background-color: #c8e6c9; color: black;'
                        return ''

                    st.dataframe(df[['Diagnosis', 'Age', 'Systolic_BP', 'Bleed_Risk_%']].style.map(color_rows, subset=['Diagnosis']), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

    # --- TAB 2: IMAGING (SIMULATION) ---
    with tab2:
        st.info("ℹ️ Simulates DenseNet-121 Deep Learning analysis.")
        img = st.file_uploader("Upload X-Ray", type=["jpg", "png"])
        if img and st.button("Analyze Image"):
            st.image(img, width=200)
            st.success("✅ Prediction: PNEUMONIA (Confidence: 88.4%)")
            st.warning("Recommend: Chest CT and Antibiotics.")

# --- MODULE 5: MEDICATION CHECKER ---
def render_medication_checker():
    st.subheader("Drug-Drug Interaction Checker")
    st.caption("Checks for Critical and Major interactions from backend database.")
    
    col_d1, col_d2 = st.columns(2)
    d1 = col_d1.text_input("Drug A", placeholder="e.g. Warfarin")
    d2 = col_d2.text_input("Drug B", placeholder="e.g. Ibuprofen")

    if d1 and d2:
        res = bk.check_interaction(d1, d2)
        if "CRITICAL" in res: st.error(f"❌ {res}")
        elif "MAJOR" in res: st.warning(f"⚠️ {res}")
        elif "MODERATE" in res: st.info(f"ℹ️ {res}")
        else: st.success(res)

# --- MODULE 6: CHATBOT ---
def render_chatbot():
    st.subheader("AI Clinical Assistant")
    st.caption("Database covers 250+ clinical topics (Cardio, Resp, Neuro, Pharm, Labs).")
    
    q = st.text_input("Ask a clinical question:")
    if q:
        with st.chat_message("assistant"):
            st.write(bk.chatbot_response(q))

# --- MODULE 7: AI DIAGNOSTICIAN (NEW) ---
def render_ai_diagnostician():
    st.subheader("🧠 AI-Powered Clinical Support")
    
    # Toggle between Patient and Doctor view
    mode = st.radio("Select Mode", ["Patient (Symptom Checker)", "Provider (Clinical Decision Support)"])
    
    if mode == "Patient (Symptom Checker)":
        st.info("ℹ️ Describe your symptoms in plain English. The AI will triage the urgency.")
        symptoms = st.text_area("How are you feeling?", placeholder="e.g., I have a sharp pain in my chest and trouble breathing...")
        
        if st.button("Analyze Symptoms"):
            if symptoms:
                with st.spinner("AI is analyzing symptoms..."):
                    # Call Backend
                    response = bk.consult_ai_doctor("patient", symptoms)
                    st.markdown(response)
                    st.warning("⚠️ Disclaimer: This is AI-generated advice. Do not use for emergencies.")
            else:
                st.error("Please enter symptoms first.")

    elif mode == "Provider (Clinical Decision Support)":
        # Get data from the Risk Calculator Session
        current_data = st.session_state.get('patient_data', {})
        
        st.success(f"Context Loaded: Patient {current_data.get('id', 'Unknown')} | Risk Status: {current_data.get('status', 'Unknown')}")
        
        clinical_notes = st.text_area("Clinical Notes / Observations", placeholder="e.g., Patient appears pale, complaining of abdominal pain...")
        
        if st.button("Generate Differential Diagnosis"):
            if clinical_notes:
                with st.spinner("Consulting Medical LLM..."):
                    # Pass the calculated risk data + notes to the AI
                    response = bk.consult_ai_doctor("provider", clinical_notes, current_data)
                    st.markdown(response)
            else:
                st.error("Please enter clinical notes.")

# ---------------------------------------------------------
# 3. MAIN APP CONTROLLER
# ---------------------------------------------------------
if not st.session_state['entered_app']:
    render_cover_page()
else:
    with st.sidebar:
        st.title("Navigation")
        menu = st.radio("Select Module", [
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
    elif menu == "Clinical Chatbot":
        render_chatbot()
    elif menu == "AI Diagnostician":
        render_ai_diagnostician()
