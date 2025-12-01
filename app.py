import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import backend as bk

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION (MUST BE FIRST!)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Clinical Risk Monitor", 
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. CSS & SETUP (Everything else comes AFTER)
# ---------------------------------------------------------
def load_css():
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

# --- MODULE 1: RISK CALCULATOR (FINAL CORRECTED VERSION) ---
def render_risk_calculator():
    st.subheader("Acute Risk Calculator")
    
    # --- INPUTS CONTAINER ---
    with st.container(border=True):
        st.markdown("#### 📝 Patient Data Entry")
        
        with st.form("risk_form"):
            
            # Split Screen Layout
            col_left, col_right = st.columns([1, 1], gap="medium")
            
            # --- LEFT COLUMN: Demographics & Vitals ---
            with col_left:
                st.markdown("##### 👤 Patient Profile")
                l1, l2 = st.columns(2)
                age = l1.number_input("Age (Years)", min_value=0, max_value=120, value=0)
                gender = l2.selectbox("Gender", ["Male", "Female"])
                
                w_val, w_unit = st.columns([2, 1]) 
                weight_input = w_val.number_input("Weight", 0.0, 400.0, 0.0)
                weight_scale = w_unit.selectbox("Unit", ["kg", "lbs"], key="w_unit")
                
                # Weight Calc Logic
                weight_kg = weight_input * 0.453592 if weight_scale == "lbs" else weight_input
                height = 170 
                if height > 0:
                    bmi = weight_kg / ((height/100)**2)
                else:
                    bmi = 0.0

                st.markdown("##### 🩺 Vitals")
                v1, v2 = st.columns(2)
                sys_bp = v1.number_input("Systolic BP (Normal: 110-120)", 0, 300, 0)
                dia_bp = v2.number_input("Diastolic BP (Normal: 70-80)", 0, 200, 0)
                
                v3, v4 = st.columns(2)
                hr = v3.number_input("Heart Rate (Normal: 60-100)", 0, 300, 0)
                resp_rate = v4.number_input("Resp Rate (Normal: 12-20)", 0, 60, 0)
                
                v5, v6 = st.columns(2)
                temp_c = v5.number_input("Temp °C (Normal: 36.5-37.5)", 0.0, 45.0, 0.0, step=0.1)
                o2_sat = v6.number_input("O2 Sat % (Normal: >95%)", 0, 100, 0)

            # --- RIGHT COLUMN: Labs & History ---
            with col_right:
                st.markdown("##### 🧪 Critical Labs")
                
                lab1, lab2 = st.columns(2)
                creat = lab1.number_input("Creatinine (0.6-1.2 mg/dL)", 0.0, 20.0, 0.0)
                bun = lab2.number_input("Blood Urea Nitrogen (7-20)", 0, 100, 0)
                
                lab3, lab4 = st.columns(2)
                potassium = lab3.number_input("Potassium (3.5-5.0 mmol/L)", 0.0, 10.0, 0.0)
                glucose = lab4.number_input("Glucose (70-100 mg/dL)", 0, 1000, 0)
                
                lab5, lab6 = st.columns(2)
                wbc = lab5.number_input("WBC (4.5-11.0 10^9/L)", 0.0, 50.0, 0.0)
                hgb = lab6.number_input("Hemoglobin (13.5-17.5 g/dL)", 0.0, 20.0, 0.0)
                
                lab7, lab8 = st.columns(2)
                platelets = lab7.number_input("Platelets (150-450 10^9/L)", 0, 1000, 0)
                inr = lab8.number_input("INR (Clotting Time) [0.9-1.1]", 0.0, 10.0, 0.0)
                
                lactate = st.number_input("Lactate (Normal: < 2.0 mmol/L)", 0.0, 20.0, 0.0)

                st.markdown("##### 📋 Medical History")
                h1, h2 = st.columns(2)
                anticoag = h1.checkbox("Anticoagulant Use")
                liver_disease = h2.checkbox("Liver Disease")
                
                h3, h4 = st.columns(2)
                heart_failure = h3.checkbox("Heart Failure")
                gi_bleed = h4.checkbox("History of GI Bleed")
                
                # Row 2 of checkboxes
                m1, m2 = st.columns(2)
                nsaid = m1.checkbox("NSAID Use (e.g. Ibuprofen)")
                active_chemo = m2.checkbox("Active Chemotherapy")
                
                m3, m4 = st.columns(2)
                diuretic = m3.checkbox("Diuretic Use")
                acei = m4.checkbox("ACEi/ARB Use")
                
                m5, m6 = st.columns(2)
                insulin = m5.checkbox("Insulin Dependent")
                hba1c_high = m6.checkbox("Uncontrolled Diabetes")
                
                altered_mental = st.checkbox("Altered Mental Status (Confusion)")
                
                # Default pain
                pain = 0

            # SUBMIT BUTTON
            st.write("") 
            submitted = st.form_submit_button("🚀 Run Clinical Analysis", type="primary", use_container_width=True)

    # --- LOGIC & RESULTS ---
    if submitted:
        # 1. Pre-Processing
        final_temp_c = temp_c 
        is_high_bp = 1 if sys_bp > 140 else 0
        
        if sys_bp > 0:
            map_val = (sys_bp + (2 * dia_bp)) / 3 
        else:
            map_val = 0
        
        # --- GLOBAL ZERO CHECK ---
        # Only run calculations if valid patient data is entered (Age > 0 AND BP > 0)
        if age > 0 and sys_bp > 0:
            
            # 2. AI Prediction (Bleeding)
            input_df = pd.DataFrame({
                'age': [age], 'inr': [inr], 'anticoagulant': [1 if anticoag else 0],
                'gi_bleed': [1 if gi_bleed else 0], 'high_bp': [is_high_bp],
                'antiplatelet': [0], 'gender_female': [1 if gender == "Female" else 0],
                'weight': [weight_kg], 'liver_disease': [1 if liver_disease else 0]
            })
            pred_bleeding = bleeding_model.predict(input_df)[0]

            # 3. Clinical Rules (Now passing ALL variables correctly)
            pred_aki = bk.calculate_aki_risk(age, diuretic, acei, sys_bp, active_chemo, creat, nsaid, heart_failure)
            pred_sepsis = bk.calculate_sepsis_risk(sys_bp, resp_rate, altered_mental, final_temp_c)
            pred_hypo = bk.calculate_hypoglycemic_risk(insulin, (creat>1.3), hba1c_high, False)
            sirs_score = bk.calculate_sirs_score(final_temp_c, hr, resp_rate, wbc)
            
            # HAS-BLED Score
            has_bled = 0
            if sys_bp > 160: has_bled += 1
            if creat > 2.2 or liver_disease: has_bled += 1
            if gi_bleed: has_bled += 1
            if inr > 1.0: has_bled += 1
            if age > 65: has_bled += 1
            if nsaid or anticoag: has_bled += 1
            
        else:
            # Default to 0 if no data entered
            pred_bleeding = 0.0
            pred_aki = 0
            pred_sepsis = 0
            pred_hypo = 0
            sirs_score = 0
            has_bled = 0

        status_calc = 'Critical' if (pred_bleeding > 50 or pred_aki > 50 or pred_sepsis >= 2) else 'Stable'
        
        # 4. Save Patient Data
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
            'sirs_score': sirs_score, 'status': status_calc, 'map_val': map_val, 'bmi': bmi, 'has_bled': has_bled
        }
        
        # Sync local state
        st.session_state['analysis_results'] = st.session_state['patient_data']

    # --- RESULTS DISPLAY ---
    if 'analysis_results' in st.session_state:
        res = st.session_state['analysis_results']
        
        st.divider()
        st.subheader("📊 Risk Stratification Results")
        
        # ROW 1: Major Risks
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("🩸 Bleeding Risk", f"{res['bleeding_risk']:.1f}%", 
                 "High" if res['bleeding_risk'] > 50 else "Normal", help="XGBoost Prediction")
        r2.metric("💧 AKI Risk", f"{res['aki_risk']}%", 
                 "High" if res['aki_risk'] > 50 else "Normal", help="KDIGO Criteria")
        r3.metric("🦠 Sepsis Score", f"{res['sepsis_risk']}", 
                 "Alert" if res['sepsis_risk'] >= 2 else "Normal", help="qSOFA Score")
        r4.metric("🍬 Hypo Risk", f"{res.get('hypo_risk', 0)}%", 
                  "High" if res.get('hypo_risk', 0) > 50 else "Normal", help="Hypoglycemia Risk Model")

        # ROW 2: Physiology & Vitals
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("MAP", f"{int(res.get('map_val', 0))} mmHg", help="Mean Arterial Pressure (>65 required)")
        d2.metric("⚡ SIRS Score", f"{res.get('sirs_score', 0)}/4", help="Systemic Inflammatory Response Syndrome")
        d3.metric("BMI", f"{res.get('bmi', 0):.1f}", help="Body Mass Index")
        d4.metric("Pain Level", f"{res.get('pain', 0)}/10", "Severe" if res.get('pain', 0) > 7 else "Managed")

        st.divider()
        
        # CLINICAL ALERTS
        st.markdown("### ⚠️ Clinical Alerts & AI Assessment")
        violations = 0 
        
        # 1. Airway/Breathing
        if res.get('o2_sat', 100) > 0 and res.get('o2_sat', 100) < 88: 
            st.error(f"🚨 CRITICAL HYPOXIA (SpO2 {res['o2_sat']}%) - Secure Airway Immediately!")
            violations += 1
        elif res.get('o2_sat', 100) > 0 and res.get('o2_sat', 100) < 92:
            st.warning(f"⚠️ Hypoxia (SpO2 {res['o2_sat']}%) - Oxygen Therapy Indicated")
            violations += 1
        
        if res.get('resp_rate', 0) > 30:
            st.error(f"🚨 SEVERE TACHYPNEA (RR {res['resp_rate']})")
            violations += 1

        # 2. Circulation (NOW CHECKS DIASTOLIC TOO!)
        if res.get('sys_bp', 0) > 180 or res.get('dia_bp', 0) > 120: 
            st.error(f"🚨 HYPERTENSIVE CRISIS (BP {res['sys_bp']}/{res['dia_bp']})")
            violations += 1
        elif res.get('sys_bp', 0) > 0 and res.get('sys_bp', 0) < 90: 
            st.error(f"🚨 SHOCK / HYPOTENSION (BP {res['sys_bp']}/{res['dia_bp']})")
            violations += 1
        elif res.get('dia_bp', 0) > 0 and res.get('dia_bp', 0) < 40: # NEW CHECK
            st.error(f"🚨 CRITICAL DIASTOLIC HYPOTENSION (Dia {res['dia_bp']}) - Check Perfusion")
            violations += 1
            
        if res.get('hr', 0) > 130:
            st.error(f"🚨 SEVERE TACHYCARDIA (HR {res['hr']})")
            violations += 1
        elif res.get('hr', 0) > 0 and res.get('hr', 0) < 40:
            st.error(f"🚨 SEVERE BRADYCARDIA (HR {res['hr']})")
            violations += 1

        # 3. Labs
        if res.get('creat', 0) > 3.0: 
            st.error(f"🚨 ACUTE RENAL FAILURE (Cr {res['creat']})")
            violations += 1
        
        if res.get('potassium', 0) > 6.0:
            st.error(f"🚨 CRITICAL HYPERKALEMIA (K+ {res['potassium']})")
            violations += 1
            
        if res.get('inr', 0) > 4.0:
            st.error(f"🚨 CRITICAL INR ({res['inr']}) - Bleed Risk")
            violations += 1
        
        if res.get('sepsis_risk', 0) >= 2:
             st.error("🚨 SEPSIS ALERT: qSOFA Score ≥ 2")
             violations += 1

        if violations == 0:
            st.success("✅ No immediate Life-Threatening Protocol violations detected.")

        # AI Consultant
        st.divider()
        c_ai, c_txt = st.columns([1, 3])
        with c_ai:
            st.markdown("#### 🤖 AI Assessment")
            if st.button("⚡ Consult AI"):
                with st.spinner("Thinking..."):
                    ai_context = {
                        'age': res['age'], 'sbp': res['sys_bp'], 
                        'bleeding_risk': res['bleeding_risk'], 'aki_risk': res['aki_risk']
                    }
                    response = bk.consult_ai_doctor("risk_assessment", "", ai_context)
                    st.session_state['ai_result'] = response
        
        with c_txt:
            if 'ai_result' in st.session_state:
                st.info(st.session_state['ai_result'])
    else:
        st.info("👈 Fill out the patient data form above and click 'Run Clinical Analysis' to see results.")
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

# --- MODULE 3: LIVE DASHBOARD (LINKED TO CALCULATOR) ---
def render_dashboard():
    # 1. GET DATA FROM SESSION STATE
    # This grabs the exact values you just entered in the Risk Calculator
    data = st.session_state.get('patient_data', {})
    
    # Default values if no analysis has been run yet
    if not data:
        st.warning("⚠️ No patient data found. Please run the Risk Calculator first.")
        return

    is_critical = data.get('status') == 'Critical'
    
    # --- HEADER & AI BUTTON ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader(f"🛏️ Bedside Monitor: {data.get('id', 'Unknown')}")
        st.caption(f"Status: **{data.get('status', 'Unknown')}**")
    
    with c2:
        # AI DISCHARGE SUMMARY (Now uses the real data)
        if st.button("✨ Generate Discharge Note", type="primary"):
            with st.spinner("Consulting Gemini 2.0..."):
                ai_summary = bk.generate_discharge_summary(data)
                st.session_state['latest_discharge_note'] = ai_summary
        
        if 'latest_discharge_note' in st.session_state:
            st.download_button(
                label="📥 Download Note",
                data=st.session_state['latest_discharge_note'],
                file_name=f"discharge_{data.get('id')}.txt",
                mime="text/plain"
            )
            
    # --- PREVIEW AREA (View Generated Summary) ---
    if 'latest_discharge_note' in st.session_state:
        with st.expander("📄 View Generated Summary", expanded=True):
            st.text_area("Edit before downloading:", value=st.session_state['latest_discharge_note'], height=200)

    st.divider()

    # --- REAL-TIME VITALS PANEL (Uses Real Inputs) ---
    with st.container(border=True):
        st.markdown("#### 📉 Real-Time Telemetry")
        col_chart, col_vitals = st.columns([3, 1])
        
        with col_chart:
            # Simulate a live trace based on the INPUT BP and HR
            # We add small random noise to make it look "live" but centered on your inputs
            base_sbp = data.get('sys_bp', 120)
            base_hr = data.get('hr', 80)
            
            chart_data = pd.DataFrame({
                'Time': range(20),
                'Systolic BP': np.random.normal(base_sbp, 2, 20), # Jitters around input BP
                'Heart Rate': np.random.normal(base_hr, 2, 20)    # Jitters around input HR
            }).melt('Time', var_name='Metric', value_name='Value')
            
            c = alt.Chart(chart_data).mark_line(interpolate='basis', strokeWidth=3).encode(
                x=alt.X('Time', axis=None),
                y=alt.Y('Value', scale=alt.Scale(zero=False)), # Auto-scale to show movement
                color=alt.Color('Metric', scale=alt.Scale(range=['#FF4B4B', '#00CC96']))
            ).properties(height=200)
            
            st.altair_chart(c, use_container_width=True)

        with col_vitals:
            # Display the EXACT numbers entered
            st.markdown(f"""
            <div style="background-color:#0E1117; padding:15px; border-radius:10px; text-align:center; border: 1px solid #333;">
                <h3 style="color:#FF4B4B; margin:0;">{int(data.get('sys_bp', 0))}</h3>
                <p style="color:gray; font-size:12px; margin:0;">mmHg (SBP)</p>
                <hr style="margin: 10px 0; border-color:#333;">
                <h3 style="color:#00CC96; margin:0;">{int(data.get('hr', 0))}</h3>
                <p style="color:gray; font-size:12px; margin:0;">BPM (HR)</p>
                <hr style="margin: 10px 0; border-color:#333;">
                <h3 style="color:#00A6ED; margin:0;">{int(data.get('o2_sat', 0))}%</h3>
                <p style="color:gray; font-size:12px; margin:0;">SpO2</p>
            </div>
            """, unsafe_allow_html=True)

    # --- RISK METRICS (From Analysis) ---
    st.markdown("#### ⚠️ Risk Stratification")
    r1, r2, r3, r4 = st.columns(4)
    
    r1.metric("🩸 Bleeding Risk", f"{data.get('bleeding_risk', 0):.1f}%", 
              "High" if data.get('bleeding_risk', 0) > 50 else "Normal", delta_color="inverse",
              help="Probability of major hemorrhage based on XGBoost model.")
    
    r2.metric("💧 AKI Risk", f"{data.get('aki_risk', 0)}%", 
              "Critical" if data.get('aki_risk', 0) > 50 else "Normal", delta_color="inverse",
              help="Acute Kidney Injury Risk based on KDIGO criteria.")
    
    r3.metric("🦠 Sepsis Score", f"{data.get('sepsis_risk', 0)}", 
              "Alert" if data.get('sepsis_risk', 0) >= 2 else "Normal", delta_color="inverse",
              help="qSOFA Score (0-3). ≥2 indicates high sepsis risk.")
    
    r4.metric("🌡️ Temp", f"{data.get('temp_c', 37.0):.1f}°C", "Fever" if data.get('temp_c', 37) > 38 else "Normal", delta_color="inverse")
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

# --- MODULE 7: AI DIAGNOSTICIAN (MODERN CHAT INTERFACE) ---
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
    elif menu == "📚 Medical Glossary":
        render_chatbot()
    elif menu == "🧠 AI Clinical Consultant":
        render_ai_diagnostician()
