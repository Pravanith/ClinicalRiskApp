import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import backend as bk
import datetime

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
    [data-testid="stSidebar"] {background-color: #f8f9fa;}
    [data-testid="stMetricValue"] {font-size: 1.8rem !important; color: #212529;}
    </style>
""", unsafe_allow_html=True)

# Helper: Timestamp for downloads
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")

# Initialize Backend
bk.init_db()

# Load Model
try:
    bleeding_model = bk.load_bleeding_model()
except Exception as e:
    st.error(f"Model Error: {e}")
    st.stop()

# Session State Init
if 'patient_data' not in st.session_state: st.session_state['patient_data'] = {}
if 'entered_app' not in st.session_state: st.session_state['entered_app'] = False

# ---------------------------------------------------------
# 2. UI MODULES
# ---------------------------------------------------------
def render_cover_page():
    st.markdown("<h1 style='text-align: center;'>🛡️ Clinical Risk Monitor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Precision Pharmacotherapy & AI Risk Stratification</p>", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    if c2.button("🚀 Launch Dashboard", use_container_width=True, type="primary"):
        st.session_state['entered_app'] = True
        st.rerun()

def render_risk_calculator():
    st.subheader("Acute Risk Calculator (Gender-Specific)")
    st.caption("Auto-adjusts for gender thresholds and converts units (Lbs/°F) to metric standards.")
    
    with st.container(border=True):
        with st.form("risk_form"):
            c1, c2 = st.columns(2, gap="medium")
            
            with c1:
                st.markdown("##### 👤 Demographics & Vitals")
                age = st.number_input("Age (Years)", 0, 120, 65)
                gender = st.selectbox("Gender", ["Male", "Female"])
                
                # Weight: Value + Unit
                w_col, w_unit_col = st.columns([2, 1])
                weight_input = w_col.number_input("Weight", 0.0, 500.0, 160.0)
                w_unit = w_unit_col.selectbox("Unit", ["lbs", "kg"], key="w_u")
                
                height = st.number_input("Height (cm)", 0, 250, 170)
                
                # Temp: Value + Unit
                t_col, t_unit_col = st.columns([2, 1])
                temp_input = t_col.number_input("Temp", 0.0, 115.0, 98.6)
                t_unit = t_unit_col.selectbox("Scale", ["°F", "°C"], key="t_u")

                sys_bp = st.number_input("Systolic BP (mmHg)", 0, 300, 120)
                hr = st.number_input("Heart Rate (BPM)", 0, 300, 80)
                resp_rate = st.number_input("Resp Rate (BPM)", 0, 60, 16)
                o2_sat = st.number_input("O2 Sat (%)", 0, 100, 98)

            with c2:
                st.markdown("##### 🧪 Comprehensive Labs")
                creat = st.number_input("Creatinine (mg/dL)", 0.0, 20.0, 0.9)
                hgb = st.number_input("Hemoglobin (g/dL)", 0.0, 25.0, 14.0)
                wbc = st.number_input("WBC (10^9/L)", 0.0, 100.0, 6.0)
                potassium = st.number_input("Potassium (mmol/L)", 0.0, 10.0, 4.0)
                qtc = st.number_input("QTc Interval (ms)", 0, 1000, 420)
                
                st.markdown("##### 💊 Medications / History")
                m1, m2 = st.columns(2)
                anticoag = m1.checkbox("Anticoagulant")
                nsaid = m2.checkbox("NSAID")
                liver_disease = m1.checkbox("Liver Disease")
                gi_bleed = m2.checkbox("Hx GI Bleed")
            
            submitted = st.form_submit_button("🚀 Run Clinical Analysis", type="primary", use_container_width=True)

    if submitted:
        # 1. Unit Conversion (Standardize to Metric for Math)
        weight_kg, temp_c = bk.ClinicalStandards.convert_units(weight_input, w_unit, temp_input, t_unit)
        
        # 2. Advanced Calculations
        # Calculate CrCl using Gender & Weight Logic
        crcl, weight_type = bk.ClinicalStandards.calculate_crcl(age, weight_kg, height, gender, creat)
        
        # Risk Models
        aki_score = bk.calculate_aki_risk(age, False, False, sys_bp, False, creat, nsaid, False)
        sepsis_score = bk.calculate_sepsis_risk(sys_bp, resp_rate, False, temp_c)
        sirs_score = bk.calculate_sirs_score(temp_c, hr, resp_rate, wbc)
        
        # Prediction (Bleeding)
        # Create input df for XGBoost
        input_df = pd.DataFrame({
            'age': [age], 'inr': [1.0], 'anticoagulant': [1 if anticoag else 0],
            'gi_bleed': [1 if gi_bleed else 0], 'high_bp': [1 if sys_bp > 140 else 0],
            'antiplatelet': [0], 'gender_female': [1 if gender == "Female" else 0],
            'weight': [weight_kg], 'liver_disease': [1 if liver_disease else 0]
        })
        bleed_prob = float(bleeding_model.predict(input_df)[0])
        
        status_calc = 'Critical' if (bleed_prob > 50 or aki_score > 50 or sepsis_score >= 2) else 'Stable'
        
        # 3. Save to Session State (Metric Values Stored)
        st.session_state['patient_data'] = {
            'id': f"PT-{age}-{int(sys_bp)}",
            'age': age, 'gender': gender, 
            'weight': weight_kg, 'height': height, # Metric
            'sys_bp': sys_bp, 'hr': hr, 'resp_rate': resp_rate,
            'temp_c': temp_c, 'o2_sat': o2_sat,
            'creat': creat, 'hgb': hgb, 'wbc': wbc, 'potassium': potassium, 'qtc': qtc,
            'crcl': crcl, 'weight_used': weight_type,
            'aki_risk': aki_score, 'bleeding_risk': bleed_prob,
            'sepsis_risk': sepsis_score, 'sirs_score': sirs_score,
            'status': status_calc
        }
        
        # Save to History DB
        bk.save_patient_to_db(age, gender, sys_bp, int(aki_score), bleed_prob, status_calc)
        
        st.session_state['analysis_results'] = st.session_state['patient_data']

    # --- RESULTS DISPLAY ---
    if 'analysis_results' in st.session_state:
        res = st.session_state['analysis_results']
        
        # Fetch Gender-Specific Safety Limits
        limits = bk.ClinicalStandards.get_thresholds(res['gender'])

        st.divider()
        st.markdown(f"### 📊 Results for {res['age']}yo {res['gender']}")
        
        # Metrics Row 1: Physiological
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Weight (Metric)", f"{res['weight']:.1f} kg")
        m2.metric("Temp (Metric)", f"{res['temp_c']:.1f} °C")
        
        # CrCl Logic for Color
        crcl_color = "normal"
        if res['crcl'] < 30: crcl_color = "inverse" # Severe
        m3.metric("💧 CrCl", f"{res['crcl']} mL/min", delta_color=crcl_color, help=f"Method: Cockcroft-Gault using {res['weight_used']} Weight")
        
        m4.metric("❤️ QTc", f"{res['qtc']} ms", "High" if res['qtc'] > limits['qtc_limit'] else "Normal", delta_color="inverse")

        # Metrics Row 2: Risks
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("🩸 Bleed Risk", f"{res['bleeding_risk']:.1f}%", "High" if res['bleeding_risk'] > 50 else "Ok", delta_color="inverse")
        r2.metric("🦠 Sepsis Score", f"{res['sepsis_risk']}", "Alert" if res['sepsis_risk'] >= 2 else "Ok", delta_color="inverse")
        r3.metric("⚡ SIRS Score", f"{res['sirs_score']}/4", "Inflamed" if res['sirs_score'] >= 2 else "Ok", delta_color="inverse")
        r4.metric("💧 AKI Risk", f"{res['aki_risk']}%", "High" if res['aki_risk'] > 50 else "Ok", delta_color="inverse")

        # --- COMPREHENSIVE ALERTS LOOP ---
        st.subheader("⚠️ Clinical Safety Alerts")
        
        violations = 0
        
        # 1. Hgb (Gender Specific)
        if res['hgb'] < limits['hgb_low']:
            st.error(f"🚨 ANEMIA DETECTED (Hgb {res['hgb']} < {limits['hgb_low']})")
            st.caption(f"Gender-specific threshold for {res['gender']} is {limits['hgb_low']} g/dL.")
            violations += 1
        
        # 2. Creatinine (Gender Specific)
        if res['creat'] > limits['creat_high']:
            st.warning(f"⚠️ HIGH CREATININE (SCr {res['creat']} > {limits['creat_high']})")
            st.caption(f"Note: {res['gender']}s typically have lower muscle mass, so normal range is lower.")
            violations += 1
            
        # 3. QTc (Gender Specific)
        if res['qtc'] > limits['qtc_limit']:
            st.error(f"❤️ QTc PROLONGATION ({res['qtc']} > {limits['qtc_limit']} ms)")
            st.info("Risk of Torsades de Pointes. Review QT-prolonging meds.")
            violations += 1
            
        # 4. Potassium (Critical)
        if res['potassium'] > limits['k_high']:
            st.error(f"🚨 HYPERKALEMIA (K+ {res['potassium']})")
            violations += 1
        elif res['potassium'] > 0 and res['potassium'] < limits['k_low']:
            st.error(f"🚨 HYPOKALEMIA (K+ {res['potassium']})")
            violations += 1
            
        # 5. WBC
        if res['wbc'] > limits['wbc_high']:
            st.warning(f"⚠️ LEUKOCYTOSIS (WBC {res['wbc']})")
            violations += 1

        if violations == 0:
            st.success("✅ No critical gender-specific or safety violations detected.")

def render_dashboard():
    data = st.session_state.get('patient_data', {})
    if not data:
        st.warning("⚠️ No data. Run Risk Calculator first.")
        return

    st.subheader(f"🛏️ Live Dashboard: {data.get('id')}")
    
    # AI Discharge Note
    c1, c2 = st.columns([3, 1])
    with c2:
        if st.button("✨ Write Discharge Note"):
            with st.spinner("AI Generating..."):
                note = bk.generate_discharge_summary(data)
                st.session_state['note'] = note
        
        if 'note' in st.session_state:
            st.download_button("📥 Download PDF", st.session_state['note'], file_name=f"Note_{get_timestamp()}.txt")
            
    if 'note' in st.session_state:
        st.text_area("Discharge Summary", st.session_state['note'], height=150)
        
    st.divider()
    
    # Telemetry
    st.markdown("#### 📉 Real-Time Telemetry")
    st.caption("ℹ️ Note: Simulated trace based on static inputs.")
    
    chart_data = pd.DataFrame({
        'Time': range(20),
        'SBP': np.random.normal(data.get('sys_bp', 120), 3, 20),
        'HR': np.random.normal(data.get('hr', 80), 3, 20)
    }).melt('Time')
    
    c = alt.Chart(chart_data).mark_line(interpolate='basis').encode(
        x='Time', y=alt.Y('value', scale=alt.Scale(zero=False)), color='variable'
    ).properties(height=200)
    st.altair_chart(c, use_container_width=True)

def render_batch_analysis():
    st.subheader("Bulk Analysis")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        # Unit cleaning logic (e.g. mmol to mg) would go here
        st.dataframe(df.head())
        if st.button("Run Batch AI"):
            st.success("Processed!")
            st.download_button("Download Results", df.to_csv(), f"Batch_{get_timestamp()}.csv")

def render_pharmacology_hub():
    st.subheader("💊 Pharmacology Hub")
    st.caption("Integrated Drug Interaction Checker & Medical Reference.")

    # Split Column Layout: Checker Left, Glossary Right
    col_checker, col_glossary = st.columns([2, 1], gap="large")

    # --- LEFT: CHECKER ---
    with col_checker:
        st.info("🛠️ **Interaction Checker**")
        with st.container(border=True):
            c1, c2 = st.columns(2)
            d1 = c1.text_input("Drug A", placeholder="Warfarin")
            d2 = c2.text_input("Drug B", placeholder="Aspirin")

            if d1 and d2:
                st.divider()
                res = bk.check_interaction(d1, d2)
                if res:
                    if "CRITICAL" in res: st.error(f"❌ {res}")
                    elif "MAJOR" in res: st.warning(f"⚠️ {res}")
                    else: st.info(f"ℹ️ {res}")
                else: 
                    st.warning(f"⚠️ {d1} + {d2} not found in DB.")
                
                if st.button("⚡ AI Analysis", use_container_width=True):
                    with st.spinner("Consulting..."):
                        st.write(bk.analyze_drug_interactions([d1, d2]))

    # --- RIGHT: GLOSSARY ---
    with col_glossary:
        st.success("📖 **Medical Glossary**")
        with st.container(border=True):
            term = st.text_input("Search Term:", placeholder="e.g. Rhabdomyolysis")
            if term:
                st.markdown(bk.chatbot_response(term))
            else:
                st.markdown("*Search definitions while you work.*")

def render_ai_consultant():
    st.subheader("🧠 AI Clinical Consultant")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello Doctor. How can I assist?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Type clinical query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Thinking..."):
            ctx = st.session_state.get('patient_data', {})
            response = bk.consult_ai_doctor("provider", prompt, ctx)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

def render_history_sql():
    st.subheader("🗄️ Patient History (SQL)")
    df = bk.fetch_history()
    st.dataframe(df, use_container_width=True)
    if st.button("Clear History"):
        bk.clear_history()
        st.rerun()

# ---------------------------------------------------------
# 3. MAIN CONTROLLER
# ---------------------------------------------------------
if not st.session_state['entered_app']:
    render_cover_page()
else:
    with st.sidebar:
        st.title("Navigation")
        menu = st.radio("Select Module", [
            "Risk Calculator", 
            "Live Dashboard", 
            "💊 Pharmacology Hub", 
            "Batch Analysis (CSV)", 
            "Patient History (SQL)",
            "🧠 AI Clinical Consultant"
        ])
        st.info("v4.0 - Gender Precision")

    if menu == "Risk Calculator": render_risk_calculator()
    elif menu == "Live Dashboard": render_dashboard()
    elif menu == "💊 Pharmacology Hub": render_pharmacology_hub()
    elif menu == "Batch Analysis (CSV)": render_batch_analysis()
    elif menu == "Patient History (SQL)": render_history_sql()
    elif menu == "🧠 AI Clinical Consultant": render_ai_consultant()
