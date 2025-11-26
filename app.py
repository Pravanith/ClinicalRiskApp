import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sqlite3

# ---------------------------------------------------------
# DATABASE ENGINE (SQL)
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect('clinical_data.db')
    c = conn.cursor()
    # Create table if not exists
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
    conn = sqlite3.connect('clinical_data.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO patient_history (age, gender, sbp, aki_risk_score, bleeding_risk_score, status)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (age, gender, sbp, aki, bleed, status))
    conn.commit()
    conn.close()

# Initialize DB on app start
init_db()

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Clinical Risk Monitor", 
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. LOAD AI MODEL (WITH DUMMY FALLBACK)
# ---------------------------------------------------------
class DummyModel:
    def predict(self, df):
        # Returns a random risk score to simulate AI if file is missing
        return [np.random.randint(10, 30) + (df['age'].values[0] * 0.2)]

@st.cache_resource
def load_bleeding_model():
    """Loads a dummy model if file is missing, for demonstration purposes."""
    model_file = "bleeding_risk_model.json"
    if os.path.exists(model_file):
        model = xgb.XGBRegressor()
        model.load_model(model_file)
        return model
    else:
        # Returns dummy model so app doesn't crash
        return DummyModel()

try:
    bleeding_model = load_bleeding_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. CLINICAL LOGIC ENGINES
# ---------------------------------------------------------

# A. AKI Risk (Kidney)
def calculate_aki_risk(age, diuretic, acei, sys_bp, chemo, creat, nsaid, heart_failure):
    score = 0
    
    # 1. Medications & Conditions (Always count these)
    score += 30 if diuretic else 0
    score += 40 if acei else 0
    score += 25 if nsaid else 0
    score += 15 if heart_failure else 0
    score += 20 if chemo else 0
    
    # 2. Demographics (Only count if valid)
    if age > 0:
        score += 20 if age > 75 else 0
        
    # 3. Vitals (FIX: Only check BP if entered > 0)
    if sys_bp > 0:
        score += 10 if sys_bp > 160 else 0  # Hypertension stress
        score += 20 if sys_bp < 90 else 0   # Hypotension/Shock
    
    # 4. Labs (FIX: Only check Creatinine if entered > 0)
    if creat > 0:
        if creat > 1.5: score += 30
        elif creat > 1.2: score += 15
        
    return min(score, 100)

# B. Sepsis Screen (qSOFA) - UPDATED (Zero-Safe)
def calculate_sepsis_risk(sys_bp, resp_rate, altered_mental, temp_c):
    qsofa = 0
    
    # 1. Hypotension (BP < 100)
    # FIX: We check if sys_bp > 0 so "0" isn't counted as shock
    if sys_bp > 0 and sys_bp <= 100: 
        qsofa += 1
        
    # 2. Tachypnea (High Resp Rate)
    # FIX: We check if resp_rate > 0
    if resp_rate > 0 and resp_rate >= 22: 
        qsofa += 1
        
    # 3. Altered Mental Status
    if altered_mental: 
        qsofa += 1 
    # 4. Fever Check (SIRS Criteria)
    if temp_c > 0 and (temp_c > 38.0 or temp_c < 36.0):
        qsofa += 0.5
    # SCORING
    if qsofa >= 2: return 90   # High Risk
    if qsofa >= 1: return 45   # Moderate Risk
    return 0                   # Normal (0% Risk)

# C. Hypoglycemic Risk
def calculate_hypoglycemic_risk(insulin, renal, hba1c_high, recent_dka):
    score = 0
    score += 30 if insulin else 0
    score += 45 if renal else 0
    score += 20 if hba1c_high else 0
    score += 20 if recent_dka else 0 
    return min(score, 100)
# D. Interaction Database
interaction_db = {
    ("sildenafil", "nitroglycerin"): "CRITICAL: Fatal hypotension. Contraindicated.",
    ("tadalafil", "isosorbide mononitrate"): "CRITICAL: Fatal hypotension. Contraindicated.",
    ("methotrexate", "trimethoprim"): "CRITICAL: Bone marrow toxicity. Avoid.",
    ("sertraline", "linezolid"): "CRITICAL: Fatal Serotonin Syndrome risk.",
    ("fentanyl", "midazolam"): "CRITICAL: Severe respiratory depression.",
    ("spironolactone", "trimethoprim"): "CRITICAL: High risk of sudden death from Hyperkalemia.",
    ("warfarin", "amiodarone"): "MAJOR: Amiodarone increases INR significantly. Reduce Warfarin dose.",
    ("warfarin", "ibuprofen"): "MAJOR: NSAIDs increase bleeding risk and damage gastric mucosa.",
    ("lisinopril", "spironolactone"): "MAJOR: Risk of severe hyperkalemia (high potassium).",
    ("simvastatin", "amiodarone"): "MAJOR: Rhabdomyolysis risk. Simvastatin dose limit 20mg.",
    ("ciprofloxacin", "ondansetron"): "MAJOR: QT Prolongation risk. Monitor ECG.",
    ("levofloxacin", "ondansetron"): "MAJOR: QT Prolongation risk. Monitor ECG.",
    ("citalopram", "fluconazole"): "MAJOR: QT Prolongation risk. Avoid high doses.",
    ("warfarin", "ciprofloxacin"): "MAJOR: Antibiotics kill gut flora, spiking INR.",
    ("warfarin", "fluconazole"): "MAJOR: Fluconazole inhibits CYP2C9, dramatically raising INR.",
    ("clopidogrel", "omeprazole"): "MAJOR: Omeprazole blocks CYP2C19, making Plavix ineffective.",
    ("tramadol", "fluoxetine"): "MAJOR: Risk of Serotonin Syndrome and Seizures.",
    ("lithium", "lisinopril"): "MAJOR: ACE Inhibitors reduce Lithium excretion -> Lithium Toxicity.",
    ("lithium", "ibuprofen"): "MAJOR: NSAIDs reduce Lithium excretion -> Lithium Toxicity.",
    ("digoxin", "clarithromycin"): "MAJOR: Macrolides block P-gp -> Digoxin Toxicity.",
    ("phenytoin", "oral contraceptives"): "MAJOR: Phenytoin induces enzymes, causing contraceptive failure.",
    ("apixaban", "ibuprofen"): "MODERATE: NSAIDs increase bleeding risk with Apixaban.",
    ("clopidogrel", "aspirin"): "MODERATE: Dual antiplatelet therapy, increases bleed risk.",
    ("warfarin", "acetaminophen"): "MODERATE: High/Chronic Tylenol use can elevate INR.",
    ("levothyroxine", "calcium"): "MODERATE: Calcium blocks Thyroid absorption. Separate by 4 hours.",
    ("doxycycline", "iron"): "MODERATE: Iron binds to antibiotic, causing treatment failure.",
}

def check_interaction(d1, d2):
    d1, d2 = d1.lower().strip(), d2.lower().strip()
    if (d1, d2) in interaction_db: return interaction_db[(d1, d2)]
    if (d2, d1) in interaction_db: return interaction_db[(d2, d1)]
    return "✅ No high-alert interaction found."

# E. Chatbot Logic
def chatbot_response(text):
    text = text.lower()
    conditions = {
        "sepsis": "Life-threatening response to infection. Watch for: Fever, Hypotension, Tachycardia (qSOFA score).",
        "pneumonia": "Lung infection. Symptoms: Cough with phlegm, fever, chills, difficulty breathing.",
        "heart failure": "CHF: Heart doesn't pump well. Watch for: Edema, Dyspnea, Fatigue.",
        "copd": "Chronic Obstructive Pulmonary Disease. Watch for O2 desaturation.",
        "afib": "Atrial Fibrillation: Irregular, rapid heart rate. Stroke risk. Needs anticoagulation.",
        "stroke": "Medical emergency. BE-FAST: Balance, Eyes, Face, Arms, Speech, Time.",
        "dvt": "Deep Vein Thrombosis: Leg clot. Symptoms: Swelling, pain, warmth.",
        "pe": "Pulmonary Embolism: Lung clot. Symptoms: Sudden SOB, chest pain.",
        "anemia": "Low RBCs. Symptoms: Fatigue, weakness, pale skin.",
        "gout": "Arthritis from uric acid. Avoid Thiazide diuretics.",
        "uti": "Urinary Tract Infection. Symptoms: Urgency, burning, cloudy urine.",
        "gerd": "Acid reflux. Symptom: Heartburn.",
        "migraine": "Severe throbbing headache, nausea, light sensitivity.",
        "osteoporosis": "Weak bones. Fracture risk.",
        "cirrhosis": "Liver scarring. Causes: Hepatitis, Alcohol.",
        "pancreatitis": "Pancreas inflammation. Symptoms: Upper abdominal pain, nausea.",
        "cellulitis": "Bacterial skin infection. Red, swollen, painful skin.",
        "hypothyroidism": "Underactive thyroid. Fatigue, weight gain.",
        "hyperthyroidism": "Overactive thyroid. Weight loss, rapid heart rate.",
        "asthma": "Airway inflammation. Wheezing, SOB."
    }
    
    if "inr" in text: return "High INR (>3.5) = Bleeding Risk. Target 2.0-3.0."
    if "creatinine" in text: return "Serum Creatinine > 1.2 suggests renal impairment."
    if "metformin" in text: return "Hold Metformin before contrast if eGFR < 30 (Lactic Acidosis)."
    if "hypoglycemia" in text: return "Symptoms: Sweating, confusion. Protocol: 'Rule of 15'."
    if "hyperkalemia" in text: return "High Potassium (>5.5). Arrhythmia risk. Cause: ACEi + Spironolactone."
    if "ibuprofen" in text: return "NSAID: Avoid in CKD, Heart Failure, and with Blood Thinners."
    
    for k, v in conditions.items():
        if k in text: return f"**{k.title()}**: {v}"
        
    return "I didn't recognize that term. Try 'Sepsis', 'Afib', 'INR', or 'Ibuprofen'."

# ---------------------------------------------------------
# 4. SESSION STATE
# ---------------------------------------------------------
if 'patient_data' not in st.session_state:
    st.session_state['patient_data'] = {
        'id': 'Room 101', 'age': 65, 'bleeding_risk': 12.5, 
        'aki_risk': 10, 'hypo_risk': 5, 'status': 'Stable'
    }

if 'entered_app' not in st.session_state:
    st.session_state['entered_app'] = False

# ---------------------------------------------------------
# 5. UI LAYOUT
# ---------------------------------------------------------

# --- COVER PAGE ---
if not st.session_state['entered_app']:
    st.markdown("<h1 style='text-align: center;'>🛡️ Clinical Risk Monitor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-Driven Pharmacovigilance System</p>", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    if c2.button("🚀 Launch Dashboard", use_container_width=True, type="primary"):
        st.session_state['entered_app'] = True
        st.rerun()

# --- MAIN APP ---
else:
    with st.sidebar:
        st.title("Navigation")
        menu = st.radio("Select Module", [
            "Risk Calculator", 
            "Patient History (SQL)",
            "Live Dashboard", 
            "Batch Analysis (CSV)", 
            "Medication Checker", 
            "Clinical Chatbot"
        ])
        st.info("v2.6 - Zero-Base Inputs")
        # --- MODULE 1: RISK CALCULATOR ---
    if menu == "Risk Calculator":
        st.subheader("Acute Risk Calculator (Advanced)")
        st.caption("Enter patient values below. Default is 0.")
        
        with st.form("risk_form"):
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

            # BMI Preview (Visual Only)
            weight_kg_preview = weight_input * 0.453592 if weight_scale == "lbs" else weight_input
            if height > 0:
                bmi_preview = weight_kg_preview / ((height/100)**2)
                p1.caption(f"Calculated BMI: {bmi_preview:.1f}")
            else:
                p1.caption("Enter Height for BMI")

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
                # 1. UNIT CONVERSIONS (Logic Fix)
                if temp_scale == "°F": final_temp_c = (temp_input - 32) * 5/9
                else: final_temp_c = temp_input

                if weight_scale == "lbs": weight_kg = weight_input * 0.453592
                else: weight_kg = weight_input

                # BMI Calculation (Safe Mode)
                if height > 0:
                    bmi = weight_kg / ((height/100)**2)
                else:
                    bmi = 0.0 # Default if height not entered

                map_val = (sys_bp + (2 * dia_bp)) / 3 if sys_bp > 0 else 0

                # 2. AI PREDICTION
                is_high_bp = 1 if sys_bp > 140 else 0
                input_df = pd.DataFrame({
                    'age': [age], 'inr': [inr], 'anticoagulant': [1 if anticoag else 0],
                    'gi_bleed': [1 if gi_bleed else 0], 'high_bp': [is_high_bp],
                    'antiplatelet': [0], 'gender_female': [1 if gender == "Female" else 0],
                    'weight': [weight_kg], 'liver_disease': [1 if liver_disease else 0]
                })
                pred_bleeding = bleeding_model.predict(input_df)[0]
                
                # 3. RULE PREDICTIONS
                pred_aki = calculate_aki_risk(age, diuretic, acei, sys_bp, active_chemo, creat, nsaid, heart_failure)
                pred_sepsis = calculate_sepsis_risk(sys_bp, resp_rate, altered_mental, final_temp_c)
                pred_hypo = calculate_hypoglycemic_risk(insulin, (creat>1.3), hba1c_high, False)

                # 4. SCORES
                has_bled = 0
                if sys_bp > 160: has_bled += 1
                if creat > 2.2 or liver_disease: has_bled += 1
                if gi_bleed: has_bled += 1
                if inr > 1.0: has_bled += 1
                if age > 65: has_bled += 1
                if nsaid or anticoag: has_bled += 1

                sirs_score = 0
                if final_temp_c > 38 or final_temp_c < 36: sirs_score += 1
                if hr > 90: sirs_score += 1
                if resp_rate > 20: sirs_score += 1
                if wbc > 12 or wbc < 4: sirs_score += 1

                # 5. SAVE STATE
                status_calc = 'Critical' if (pred_bleeding > 50 or pred_aki > 50 or pred_sepsis > 50) else 'Stable'
                
                st.session_state['patient_data'] = {
                    'id': 'Calculated Patient', 'age': age,
                    'bleeding_risk': float(pred_bleeding), 'aki_risk': int(pred_aki),
                    'sepsis_risk': int(pred_sepsis), 'hypo_risk': int(pred_hypo),
                    'status': status_calc
                }
                
                save_patient_to_db(age, gender, sys_bp, int(pred_aki), float(pred_bleeding), status_calc)
                
                st.success("Analysis Complete!")
                
                # 6. DISPLAY RESULTS
                st.divider()
                st.markdown("#### 📊 Clinical Analysis Results")
                
                # Row 1: Major Risks
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Bleeding Risk (AI)", f"{pred_bleeding:.1f}%")
                r2.metric("AKI Risk (Rule)", f"{pred_aki}%")
                r3.metric("Sepsis Score (qSOFA)", f"{pred_sepsis}")
                r4.metric("HAS-BLED Score", f"{has_bled}/9", "High Risk" if has_bled >=3 else "Low Risk")

                # Row 2: Clinical Context (Now BMI is safe!)
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("MAP (Perfusion)", f"{int(map_val)} mmHg", "Low" if map_val < 65 else "Normal")
                d2.metric("SIRS Score", f"{sirs_score}/4", "Inflammation" if sirs_score >=2 else "Normal")
                d3.metric("BMI Category", f"{bmi:.1f}", "Obese" if bmi > 30 else "Normal")
                d4.metric("Pain Status", f"{pain}/10", "Managed")

                # 7. ALERTS
                if pred_sepsis > 0:
                    st.error(f"🔴 SEPSIS RISK: qSOFA Score of {pred_sepsis} suggests organ dysfunction.")
                
                with st.expander("⚠️ Detailed Clinical Alerts", expanded=True):
                    if potassium > 5.5: st.error(f"⚠️ HYPERKALEMIA: K+ {potassium}")
                    if platelets > 0 and platelets < 100: st.error(f"⚠️ THROMBOCYTOPENIA: Plt {platelets}")
                    if glucose < 70 and glucose > 0: st.warning(f"⚠️ HYPOGLYCEMIA: Glucose {glucose}")
                    if lactate > 2.0: st.warning(f"⚠️ LACTATE ELEVATED: {lactate} mmol/L")
   # --- MODULE 2: LIVE DASHBOARD (VISUAL OVERHAUL) ---
    elif menu == "Live Dashboard":
        import altair as alt # Import Altair for better charts

        # 1. Get Data
        data = st.session_state['patient_data']
        is_critical = data['status'] == 'Critical'
        
        # Header
        st.subheader(f"🖥️ ICU Monitor: {data['id']}")
        
        # 2. Key Metrics (Styled)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bleeding Risk", f"{data['bleeding_risk']:.1f}%", 
                  "High" if data['bleeding_risk'] > 50 else "Normal", delta_color="inverse")
        m2.metric("AKI Risk", f"{data['aki_risk']}%", 
                  "Critical" if data['aki_risk'] > 50 else "Normal", delta_color="inverse")
        m3.metric("Sepsis Score", f"{data.get('sepsis_risk', 0)}", 
                  "High" if data.get('sepsis_risk', 0) >= 2 else "Normal", delta_color="inverse")
        m4.metric("Hypoglycemia", "YES" if data.get('hypo_risk', 0) > 0 else "NO", 
                  "Critical" if data.get('hypo_risk', 0) > 0 else "Normal", delta_color="inverse")
        
        st.divider()

        # 3. Enhanced Layout
        col_main, col_queue = st.columns([2.5, 1])
        
        with col_main:
            st.markdown("### 📉 Live Telemetry (HR vs BP)")
            
            # GENERATE BETTER LOOKING DATA
            # We create a dataframe with 10 time points for a smoother curve
            base_hr = 110 if is_critical else 75
            base_bp = 90 if is_critical else 120
            
            # Generate random variations
            time_points = list(range(-9, 1)) # -9 hours to Now (0)
            hr_data = [base_hr + np.random.randint(-10, 15) for _ in range(10)]
            bp_data = [base_bp + np.random.randint(-5, 10) for _ in range(10)]
            
            chart_df = pd.DataFrame({
                'Time': time_points,
                'Heart Rate': hr_data,
                'Systolic BP': bp_data
            }).melt('Time', var_name='Metric', value_name='Value')

            # DEFINE CUSTOM COLORS
            # Neon Red for BP, Cyan for Heart Rate
            domain = ['Heart Rate', 'Systolic BP']
            range_ = ['#00E5FF', '#FF1744'] # Cyan and Neon Red

            # ALTAIR CHART (Professional Medical Look)
            c = alt.Chart(chart_df).mark_line(
                interpolate='monotone', # Makes the line smooth/curved
                point=True,
                strokeWidth=3
            ).encode(
                x=alt.X('Time', axis=alt.Axis(title='Time (Hours)', grid=False)),
                y=alt.Y('Value', axis=alt.Axis(title='Value', grid=True), scale=alt.Scale(domain=[40, 180])),
                color=alt.Color('Metric', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(title="Vitals")),
                tooltip=['Time', 'Metric', 'Value']
            ).properties(
                height=350
            ).configure_axis(
                gridColor='#333333' # Dark grid lines for contrast
            ).configure_view(
                strokeWidth=0 # Remove border
            )
            
            st.altair_chart(c, use_container_width=True)
            
            # Dynamic Alert Box
            if is_critical:
                st.error("🚨 **CRITICAL ALERT:** Vitals unstable. Check Sepsis & Bleeding protocols immediately.")
            else:
                st.success("✅ **STABLE:** Vitals are trending within normal limits.")

        with col_queue:
            st.markdown("#### 📋 Patient Status")
            # Only show the current patient (You)
            st.markdown(f"""
            <div style="background-color:#d4edda; color:#155724; padding:10px; border-radius:5px; margin-bottom:10px;">
                <strong>{data['id']} (Current)</strong><br>
                Status: {data['status']}
            </div>
            """, unsafe_allow_html=True)
            
            st.info("ℹ️ Note: Queue is empty. Waiting for new admissions.")
    
    # --- MODULE 3: BATCH ANALYSIS (CSV) ---
    elif menu == "Batch Analysis (CSV)":
        st.subheader("Bulk Patient Processing")
        
        # 1. Download Template Button (Helper for User)
        sample_data = {
            'Age': [65, 80, 45, 72],
            'Creatinine': [1.1, 1.8, 0.9, 2.5],
            'Systolic_BP': [130, 160, 120, 110],
            'Diuretic_Use': [0, 1, 0, 1],
            'ACEI_Use': [1, 1, 0, 0],
            'Chemo_Hx': [0, 0, 1, 0],
            'NSAID_Use': [0, 1, 1, 0],
            'Heart_Failure_Hx': [0, 1, 0, 1]
        }
        df_sample = pd.DataFrame(sample_data)
        
        with st.expander("ℹ️ CSV Format Guide"):
            st.write("Your CSV must contain these columns (0=No, 1=Yes):")
            st.dataframe(df_sample)
            st.download_button("📥 Download Template CSV", 
                               df_sample.to_csv(index=False), 
                               "patient_template.csv", "text/csv")

        # 2. Main Processing Tool
        tab1, tab2 = st.tabs(["📄 CSV Data Processor", "🖼️ Medical Imaging"])
        
        with tab1:
            uploaded_csv = st.file_uploader("Upload Patient Cohort (CSV)", type=["csv"])
            
            if uploaded_csv:
                df = pd.read_csv(uploaded_csv)
                st.write(f"**Loaded {len(df)} patient records.**")
                st.dataframe(df.head(3))

                if st.button("⚡ Run Risk Analysis on All Rows"):
                    try:
                        # PROGRESS BAR
                        progress_bar = st.progress(0)
                        
                        # LOGIC: Iterate via Pandas Apply (Vectorized iteration)
                        # We map the CSV columns to the function arguments
                        def batch_risk(row):
                            return calculate_aki_risk(
                                age=row.get('Age', 60), 
                                diuretic=row.get('Diuretic_Use', 0), 
                                acei=row.get('ACEI_Use', 0), 
                                # Logic: Convert Sys BP to High BP Boolean
                                high_bp=1 if row.get('Systolic_BP', 120) > 140 else 0,
                                chemo=row.get('Chemo_Hx', 0), 
                                creat=row.get('Creatinine', 1.0), 
                                nsaid=row.get('NSAID_Use', 0), 
                                heart_failure=row.get('Heart_Failure_Hx', 0)
                            )

                        # Apply function to create new column
                        df['AKI_Risk_Score'] = df.apply(batch_risk, axis=1)
                        
                        # Add a Risk Label based on score
                        df['Risk_Category'] = df['AKI_Risk_Score'].apply(
                            lambda x: '🔴 CRITICAL' if x > 50 else ('🟠 High' if x > 30 else '🟢 Low')
                        )

                        progress_bar.progress(100)
                        st.success("Batch Processing Complete!")

                        # Display Summary Metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Patients", len(df))
                        m2.metric("High Risk Cases", len(df[df['AKI_Risk_Score'] > 30]))
                        m3.metric("Avg Risk Score", f"{df['AKI_Risk_Score'].mean():.1f}")

                        # Show Result
                        st.dataframe(df.style.map(
                            lambda x: 'color: red; font-weight: bold;' if x == '🔴 CRITICAL' else None, 
                            subset=['Risk_Category']
                        ))

                        # Download Result
                        st.download_button(
                            "📥 Download Analyzed Data",
                            df.to_csv(index=False),
                            "analyzed_patients.csv",
                            "text/csv"
                        )

                    except Exception as e:
                        st.error(f"Error processing CSV: {e}")
                        st.info("Check your column names match the Template above!")

        with tab2:
            uploaded_image = st.file_uploader("Upload Wound/X-Ray (JPG)", type=["jpg"])
            if uploaded_image: st.image(uploaded_image, width=300)
                
    # --- MODULE 4: MEDICATION CHECKER ---
    elif menu == "Medication Checker":
        st.subheader("Drug-Drug Interaction Checker")
        
        # 1. Add a Guide so users know what to test
        with st.expander("ℹ️  Supported Drug Categories (Demo Database)"):
            st.markdown("""
            This demo checks for **High-Alert** interactions in these specific categories:
            * **🫀 Cardiac:** Warfarin, Amiodarone, Digoxin, Sildenafil, Nitroglycerin, Spironolactone, Lisinopril.
            * **💊 Pain/NSAIDs:** Ibuprofen, Tramadol, Fentanyl, Methotrexate, Oxycodone.
            * **🧠 Psych/Neuro:** Sertraline, Fluoxetine, Linezolid, Lithium, Citalopram.
            * **🦠 Antibiotics:** Ciprofloxacin, Erythromycin, Trimethoprim (Bactrim), Claritromycin.
            """)
        
        st.caption("Type two drugs below to check for CRITICAL or MAJOR interactions.")

        # 2. Input Fields
        col_d1, col_d2 = st.columns(2)
        d1 = col_d1.text_input("Drug A", placeholder="e.g., Warfarin")
        d2 = col_d2.text_input("Drug B", placeholder="e.g., Ibuprofen")

        # 3. Logic
        if d1 and d2:
            res = check_interaction(d1, d2)
            
            # Dynamic Styling based on Severity
            if "CRITICAL" in res: 
                st.error(f"❌ {res}")
            elif "MAJOR" in res: 
                st.warning(f"⚠️ {res}")
            elif "MODERATE" in res: 
                st.info(f"ℹ️ {res}")
            else: 
                st.success(res)

    # --- MODULE 5: CHATBOT ---
    elif menu == "Clinical Chatbot":
        st.subheader("AI Clinical Assistant")
        q = st.text_input("Ask a clinical question (e.g. 'Sepsis', 'AFib', 'Metformin'):")
        if q:
            with st.chat_message("assistant"):
                st.write(chatbot_response(q))
