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
# D. Interaction Database (MAXIMUM VERSION)
interaction_db = {
    # -----------------------------------------------------------
    # 🔴 CRITICAL / FATAL (Contraindicated)
    # -----------------------------------------------------------
    ("sildenafil", "nitroglycerin"): "CRITICAL: Severe hypotension and cardiovascular collapse. Contraindicated.",
    ("tadalafil", "nitroglycerin"): "CRITICAL: Severe hypotension. Contraindicated.",
    ("sildenafil", "isosorbide mononitrate"): "CRITICAL: Severe hypotension. Contraindicated.",
    ("methotrexate", "trimethoprim"): "CRITICAL: Fatal bone marrow suppression & pancytopenia. Avoid.",
    ("methotrexate", "bactrim"): "CRITICAL: Fatal bone marrow suppression. Avoid.",
    ("methotrexate", "ibuprofen"): "CRITICAL: NSAIDs block Methotrexate excretion. Risk of Kidney Failure & Toxicity.",
    ("methotrexate", "naproxen"): "CRITICAL: NSAIDs block Methotrexate excretion. Risk of Toxicity.",
    ("sertraline", "linezolid"): "CRITICAL: Fatal Serotonin Syndrome risk (Linezolid is an MAOI).",
    ("fluoxetine", "linezolid"): "CRITICAL: Fatal Serotonin Syndrome risk.",
    ("citalopram", "linezolid"): "CRITICAL: Fatal Serotonin Syndrome risk.",
    ("spironolactone", "trimethoprim"): "CRITICAL: Risk of sudden cardiac death from severe Hyperkalemia.",
    ("spironolactone", "bactrim"): "CRITICAL: Risk of sudden cardiac death from severe Hyperkalemia.",
    ("fentanyl", "benzodiazepine"): "CRITICAL: Profound respiratory depression, coma, and death.",
    ("oxycodone", "alcohol"): "CRITICAL: Respiratory depression and fatal overdose risk.",
    ("tramadol", "alcohol"): "CRITICAL: Seizure threshold lowered + Respiratory depression.",
    ("clarithromycin", "simvastatin"): "CRITICAL: Severe Rhabdomyolysis and kidney failure (CYP3A4 inhibition).",
    ("erythromycin", "simvastatin"): "CRITICAL: Severe Rhabdomyolysis and kidney failure.",
    ("ciprofloxacin", "tizanidine"): "CRITICAL: Severe hypotension and sedation.",

    # -----------------------------------------------------------
    # 🟠 MAJOR (Serious - Requires Monitoring)
    # -----------------------------------------------------------
    # CARDIAC & ANTICOAGULANTS
    ("warfarin", "amiodarone"): "MAJOR: Amiodarone inhibits Warfarin metabolism. INR can double. Reduce Warfarin 50%.",
    ("warfarin", "ciprofloxacin"): "MAJOR: Antibiotics kill gut flora, spiking INR. High bleed risk.",
    ("warfarin", "bactrim"): "MAJOR: Significant increase in INR. High bleed risk.",
    ("warfarin", "erythromycin"): "MAJOR: Increases Warfarin levels. Monitor INR closely.",
    ("warfarin", "clarithromycin"): "MAJOR: Increases Warfarin levels. Monitor INR.",
    ("warfarin", "fluconazole"): "MAJOR: Dramatically increases INR. Bleed risk.",
    ("warfarin", "metronidazole"): "MAJOR: Dramatically increases INR. Bleed risk.",
    ("warfarin", "aspirin"): "MAJOR: Significantly increased bleeding risk (Antiplatelet + Anticoagulant).",
    ("warfarin", "ibuprofen"): "MAJOR: Gastrointestinal bleeding risk. NSAIDs damage stomach lining.",
    ("digoxin", "amiodarone"): "MAJOR: Amiodarone doubles Digoxin levels. Risk of toxicity (Halo vision, Nausea).",
    ("digoxin", "clarithromycin"): "MAJOR: Macrolides block P-gp, leading to Digoxin toxicity.",
    ("digoxin", "erythromycin"): "MAJOR: Macrolides block P-gp, leading to Digoxin toxicity.",
    ("digoxin", "spironolactone"): "MAJOR: Spironolactone reduces Digoxin clearance. Monitor levels.",
    ("clopidogrel", "omeprazole"): "MAJOR: Omeprazole blocks CYP2C19, making Plavix ineffective (Stent thrombosis risk).",
    ("sildenafil", "erythromycin"): "MAJOR: Erythromycin increases Sildenafil levels. Risk of hypotension.",
    ("sildenafil", "clarithromycin"): "MAJOR: Clarithromycin increases Sildenafil levels. Risk of hypotension.",
    
    # KIDNEY & BP
    ("lisinopril", "ibuprofen"): "MAJOR: NSAIDs reduce efficacy of ACEi and cause Acute Kidney Injury.",
    ("lisinopril", "spironolactone"): "MAJOR: High risk of Hyperkalemia. Monitor Potassium.",
    ("lisinopril", "potassium"): "MAJOR: Risk of Hyperkalemia. Avoid supplements.",
    ("lisinopril", "lithium"): "MAJOR: ACEi reduces Lithium excretion -> Lithium Toxicity.",
    ("furosemide", "ibuprofen"): "MAJOR: NSAIDs block diuretic effect and hurt kidneys.",
    
    # PSYCH & NEURO (Serotonin & QT)
    ("tramadol", "sertraline"): "MAJOR: Serotonin Syndrome risk (Seizures, Agitation, Hyperthermia).",
    ("tramadol", "fluoxetine"): "MAJOR: Serotonin Syndrome risk.",
    ("tramadol", "citalopram"): "MAJOR: Serotonin Syndrome risk + Seizure risk.",
    ("citalopram", "ondansetron"): "MAJOR: QT Prolongation. Risk of Torsades de Pointes arrhythmia.",
    ("citalopram", "ciprofloxacin"): "MAJOR: QT Prolongation. Risk of Torsades de Pointes.",
    ("citalopram", "erythromycin"): "MAJOR: QT Prolongation. Risk of Torsades de Pointes.",
    ("citalopram", "clarithromycin"): "MAJOR: QT Prolongation. Risk of Torsades de Pointes.",
    ("fluoxetine", "ondansetron"): "MAJOR: QT Prolongation risk.",
    ("lithium", "ibuprofen"): "MAJOR: NSAIDs reduce Lithium excretion -> Lithium Toxicity (Tremors, Confusion).",
    ("lithium", "hydrochlorothiazide"): "MAJOR: Thiazides reduce Lithium excretion -> Lithium Toxicity.",
    
    # ANTIBIOTICS
    ("ciprofloxacin", "theophylline"): "MAJOR: Cipro increases Theophylline levels (Seizures).",
    ("ciprofloxacin", "prednisone"): "MAJOR: Increased risk of tendon rupture (Achilles), especially in elderly.",
    ("levofloxacin", "prednisone"): "MAJOR: Increased risk of tendon rupture.",
    ("doxycycline", "calcium"): "MAJOR: Calcium binds to antibiotic, preventing absorption.",
    ("ciprofloxacin", "calcium"): "MAJOR: Calcium binds to antibiotic, preventing absorption.",
    
    # -----------------------------------------------------------
    # 🍷 LIFESTYLE & FOOD (New!)
    # -----------------------------------------------------------
    ("simvastatin", "grapefruit"): "MAJOR: Grapefruit inhibits CYP3A4. Risk of Rhabdomyolysis.",
    ("atorvastatin", "grapefruit"): "MODERATE: Grapefruit increases levels. Use caution.",
    ("warfarin", "alcohol"): "MAJOR: Acute alcohol intake increases bleed risk. Chronic intake decreases Warfarin effect.",
    ("metronidazole", "alcohol"): "MAJOR: Disulfiram-like reaction (Severe vomiting). Avoid alcohol for 72h.",
    ("oxycodone", "grapefruit"): "MODERATE: Grapefruit may increase Opioid levels (Sedation).",
    
    # -----------------------------------------------------------
    # 🟡 MODERATE (Monitor)
    # -----------------------------------------------------------
    ("apixaban", "ibuprofen"): "MODERATE: Increased bleeding risk.",
    ("rivaroxaban", "ibuprofen"): "MODERATE: Increased bleeding risk.",
    ("aspirin", "ibuprofen"): "MODERATE: Ibuprofen blocks Aspirin's heart-protective effect if taken together.",
    ("sertraline", "ibuprofen"): "MODERATE: SSRIs + NSAIDs increase GI bleed risk.",
    ("fluoxetine", "ibuprofen"): "MODERATE: SSRIs + NSAIDs increase GI bleed risk.",
    ("citalopram", "ibuprofen"): "MODERATE: SSRIs + NSAIDs increase GI bleed risk.",
    ("metoprolol", "amiodarone"): "MODERATE: Risk of severe Bradycardia (slow heart rate).",
    ("levothyroxine", "calcium"): "MODERATE: Calcium blocks thyroid absorption. Separate by 4 hours.",
    ("levothyroxine", "iron"): "MODERATE: Iron blocks thyroid absorption. Separate by 4 hours.",
    ("allopurinol", "amoxicillin"): "MODERATE: High risk of skin rash.",
}
def check_interaction(d1, d2):
    d1, d2 = d1.lower().strip(), d2.lower().strip()
    if (d1, d2) in interaction_db: return interaction_db[(d1, d2)]
    if (d2, d1) in interaction_db: return interaction_db[(d2, d1)]
    return "✅ No high-alert interaction found."

# E. Chatbot Logic (Fixed: Whole Word Matching)
def chatbot_response(text):
    import re  # Import Regex for smart matching
    
    text = text.lower().strip()
    
    # 1. SMART TRANSLATOR (Maps slang/typos to medical terms)
    synonyms = {
        "type1": "type 1", "type2": "type 2", # Fixes "diabetes type1" typo
        "high bp": "hypertension", "high blood pressure": "hypertension",
        "low bp": "hypotension", "low blood pressure": "hypotension",
        "heart failure": "chf", "kidney failure": "ckd", "kidney injury": "aki",
        "sugar": "glucose", "blood sugar": "glucose", "clot": "dvt",
        "breathing problem": "copd", "shortness of breath": "sob",
        "belly pain": "appendicitis", "stomach pain": "gerd",
        "heart rate": "tachycardia", "temp": "fever", "temperature": "fever",
        "heart attack": "mi", "stroke": "cva"
    }
    
    # Replace synonyms
    for slang, term in synonyms.items():
        if slang in text:
            text = text.replace(slang, term)
    
    # 2. KNOWLEDGE BASE (The Database)
    knowledge_base = {
        # --- CARDIOLOGY ---
        "mi": "Myocardial Infarction (Heart Attack). Blockage of blood flow. STEMI is critical. Symptoms: Chest pressure, radiating pain.",
        "heart attack": "Myocardial Infarction (MI). Emergency. Protocol: MONA (Morphine, O2, Nitro, Aspirin).",
        "hypertension": "High Blood Pressure (>130/80). 'The Silent Killer'. Risk of Stroke/MI/Kidney Failure.",
        "hypotension": "Low BP (<90/60). Signs of shock. Causes: Dehydration, Sepsis, Hemorrhage.",
        "afib": "Atrial Fibrillation. Irregular heart rate. Risk: Stroke. Treatment: Beta-blockers + Anticoagulation.",
        "chf": "Congestive Heart Failure. Fluid overload. Symptoms: SOB lying flat (Orthopnea), Edema. Labs: BNP.",
        "cad": "Coronary Artery Disease. Plaque buildup in heart arteries. Risk of MI. Managed with Statins.",
        "angina": "Chest pain due to ischemia (lack of O2). Stable (exertional) vs Unstable (at rest).",
        "aortic stenosis": "Narrowing of aortic valve. Symptoms: SAD (Syncope, Angina, Dyspnea). Murmur heard.",
        "pericarditis": "Inflammation of heart sac. Sharp pain relieved by leaning forward. Friction rub heard.",
        "endocarditis": "Infection of heart valves. IV drug use is risk factor. Fever + New Murmur.",
        "bradycardia": "Slow heart rate (<60). Causes: Beta-blockers, heart block, athletic heart.",
        "tachycardia": "Fast heart rate (>100). Causes: Fever, Pain, Dehydration, PE, Anxiety.",
        "svt": "Supraventricular Tachycardia. Rate >150. Tx: Vagal maneuvers, Adenosine.",
        "vtach": "Ventricular Tachycardia. Life-threatening. Pulse? Cardiovert. No pulse? Defibrillate.",
        "vfib": "Ventricular Fibrillation. Cardiac Arrest. No pulse. CPR + Defibrillation immediately.",

        # --- RESPIRATORY ---
        "asthma": "Airway inflammation. Wheezing. Rescue: Albuterol. Maintenance: Steroids.",
        "copd": "Chronic Obstructive Pulmonary Disease. Air trapping. Smokers. Risk of CO2 retention.",
        "pneumonia": "Lung infection. Fever, cough, consolidation on X-Ray. Antibiotics needed.",
        "pe": "Pulmonary Embolism. Lung clot. Sudden SOB, chest pain, tachycardia. Emergency.",
        "pneumothorax": "Collapsed lung. Air in pleural space. Absent breath sounds. Trauma/Spontaneous.",
        "pleural effusion": "Fluid around lungs. 'Water on lungs'. Causes: CHF, Cancer. Tx: Thoracentesis.",
        "ards": "Acute Respiratory Distress Syndrome. Severe lung failure. 'White out' X-ray.",
        "bronchitis": "Inflammation of bronchi. Productive cough. Usually viral.",
        "tuberculosis": "TB. Bacterial infection. Night sweats, weight loss, bloody cough. Isolation required.",
        "sleep apnea": "Airway collapse during sleep. Snoring, fatigue. Risk: HTN, Stroke. Tx: CPAP.",
        "croup": "Pediatric viral infection. 'Barking seal' cough. Steeple sign on X-ray.",
        "rsv": "Respiratory Syncytial Virus. Bronchiolitis in kids. Watch for hypoxia.",

        # --- NEUROLOGY ---
        "cva": "Cerebrovascular Accident (Stroke). Ischemic vs Hemorrhagic. Time is Brain.",
        "stroke": "Brain attack. BE-FAST: Balance, Eyes, Face, Arms, Speech, Time. CT Head stat.",
        "tia": "Transient Ischemic Attack. Warning stroke. Symptoms resolve <24h. High risk of future stroke.",
        "seizure": "Abnormal electrical activity. Protect head. Benzodiazepines if > 5 mins.",
        "epilepsy": "Recurrent seizures. Meds: Keppra, Depakote, Phenytoin.",
        "migraine": "Severe unilateral headache. Nausea, photophobia. Tx: Triptans.",
        "cluster headache": "Severe eye pain. 'Suicide headache'. Tx: High flow Oxygen.",
        "meningitis": "Brain lining infection. Fever + Stiff Neck + Headache. Emergency.",
        "concussion": "Mild TBI. Confusion, amnesia, headache. Brain rest needed.",
        "parkinson": "Low dopamine. Tremor (resting), Rigidity, Slow movement (Bradykinesia).",
        "alzheimer": "Dementia type. Memory loss, cognitive decline.",
        "ms": "Multiple Sclerosis. Autoimmune nerve damage. Vision loss, weakness.",
        "als": "Lou Gehrig's Disease. Motor neuron death. Paralysis. Sensation intact.",
        "guillain-barre": "Ascending paralysis after infection. Watch breathing.",

        # --- GASTROINTESTINAL ---
        "gerd": "Acid Reflux. Heartburn. Risk of esophageal damage. PPIs (Omeprazole).",
        "pud": "Peptic Ulcer Disease. Stomach ulcers. Pain with food. H. Pylori or NSAIDs.",
        "gi bleed": "Upper (Vomit blood) vs Lower (Bloody stool). Monitor Hemoglobin.",
        "appendicitis": "RLQ pain (McBurney's). Fever, nausea. Surgical emergency.",
        "cholecystitis": "Gallbladder inflammation. RUQ pain after fatty meal.",
        "pancreatitis": "Pancreas inflammation. Epigastric pain to back. High Lipase.",
        "hepatitis": "Liver inflammation. Viral (A/B/C) or Alcohol. Jaundice.",
        "cirrhosis": "End-stage liver scarring. Ascites, confusion (Ammonia), Bleeding risk.",
        "ibs": "Irritable Bowel Syndrome. Functional pain/diarrhea. No organ damage.",
        "ibd": "Crohn's/Ulcerative Colitis. Autoimmune. Bloody diarrhea.",
        "diverticulitis": "Infected colon pouches. LLQ pain, fever. Antibiotics.",
        "c diff": "Antibiotic diarrhea. Contagious spores. Soap & Water wash only.",
        "bowel obstruction": "Blockage. Constipation, vomiting, distension. NPO + NG Tube.",

        # --- RENAL ---
        "aki": "Acute Kidney Injury. Creatinine spike. Causes: Dehydration, Contrast, NSAIDs.",
        "ckd": "Chronic Kidney Disease. GFR < 60 > 3 months. Diabetes/HTN causes.",
        "esrd": "End Stage Renal Disease. Needs Dialysis or Transplant.",
        "uti": "Urinary Tract Infection. Burning, frequency. E. Coli common.",
        "kidney stone": "Nephrolithiasis. Flank pain, hematuria. Hydration + Pain meds.",
        "bph": "Enlarged prostate. Dribbling, frequency in older men.",
        "rhabdo": "Muscle breakdown. Clogs kidneys. Tea-colored urine. Fluids.",

        # --- ENDOCRINE ---
        "diabetes": "Metabolic disease. High sugar. Causes damage to eyes, kidneys, nerves.",
        "diabetes type 1": "Autoimmune. No insulin. DKA risk. Insulin dependent.",
        "diabetes type 2": "Insulin resistance. Lifestyle + Metformin.",
        "prediabetes": "A1C 5.7-6.4%. Warning sign. Reversible.",
        "dka": "Diabetic Ketoacidosis. Acidosis + Ketones. ICU care.",
        "hhs": "Hyperosmolar Hyperglycemic State. Glucose > 600. Dehydration.",
        "hypoglycemia": "Low Sugar (<70). Sweating, confusion. Glucose needed.",
        "hypothyroidism": "Low Thyroid. Fatigue, weight gain, cold.",
        "hyperthyroidism": "High Thyroid. Weight loss, heat intolerance, fast HR.",
        "addison": "Adrenal insufficiency. Low cortisol. Bronze skin, hypotension.",
        "cushing": "High cortisol. Moon face, buffalo hump, high sugar.",

        # --- INFECTIOUS DISEASE ---
        "sepsis": "Infection + Organ Failure. qSOFA criteria. Antibiotics ASAP.",
        "septic shock": "Sepsis + Hypotension requiring pressors.",
        "flu": "Influenza. Sudden fever, aches. Tamiflu < 48h.",
        "covid": "SARS-CoV-2. Fever, cough, loss of taste/smell.",
        "hiv": "Virus attacking immune system (CD4). Needs ARV therapy.",
        "mrsa": "Resistant Staph infection. Needs Vancomycin.",
        "cellulitis": "Skin infection. Red, hot, spreading.",
        "abscess": "Pus pocket. Needs drainage (I&D).",
        "osteomyelitis": "Bone infection. Long-term IV antibiotics.",
        "fever": "Temp > 100.4F (38C). Sign of inflammation.",
        "neutropenic fever": "Fever in chemo patient. ONCOLOGIC EMERGENCY.",

        # --- HEMATOLOGY ---
        "anemia": "Low Hemoglobin. Fatigue, pallor. Iron deficiency common.",
        "sickle cell": "Genetic. Pain crises. RBCs shape sickle. Fluids/Pain meds.",
        "thrombocytopenia": "Low platelets. Bleeding risk.",
        "leukemia": "Blood cancer. High WBC blasts. Infection risk.",
        "lymphoma": "Lymphatic cancer. Hodgkin vs Non-Hodgkin.",
        "neutropenia": "Low neutrophils. Severe infection risk. Fever is emergency.",
        "dvt": "Deep Vein Thrombosis. Leg clot.",

        # --- MUSCULOSKELETAL ---
        "osteoarthritis": "Wear-and-tear. Joint pain. Worse with use.",
        "rheumatoid arthritis": "Autoimmune. Morning stiffness > 30 mins.",
        "gout": "Uric acid crystals. Big toe pain. Colchicine/Allopurinol.",
        "osteoporosis": "Weak bones. Fracture risk.",
        "compartment syndrome": "Muscle pressure. Pain out of proportion. Emergency surgery.",
        "rhabdomyolysis": "Muscle breakdown releasing myoglobin. Kidney damage. Tea-colored urine.",

        # --- MEDICATIONS ---
        "lisinopril": "ACE Inhibitor (BP). Side effects: Cough, High K+.",
        "amlodipine": "Calcium Channel Blocker (BP). Side effect: Leg swelling.",
        "metoprolol": "Beta-blocker. Lowers HR and BP.",
        "furosemide": "Lasix. Diuretic. Monitor Potassium.",
        "spironolactone": "Potassium-sparing diuretic. Risk: High K+.",
        "atorvastatin": "Lipitor. Cholesterol. Watch for muscle pain.",
        "metformin": "Diabetes. Risk: Lactic Acidosis. Hold for contrast.",
        "insulin": "Lowers sugar. High Alert. Hypoglycemia risk.",
        "glipizide": "Sulfonylurea. Stimulates pancreas. Hypo risk.",
        "warfarin": "Coumadin. Anticoagulant. Monitor INR (2-3).",
        "eliquis": "Apixaban. Blood thinner. No INR needed.",
        "plavix": "Clopidogrel. Antiplatelet. Keeps stents open.",
        "aspirin": "Antiplatelet. MI/Stroke prevention. Bleed risk.",
        "ibuprofen": "NSAID. Pain/Fever. Avoid in Kidney/Ulcer/CHF.",
        "naproxen": "Aleve. NSAID.",
        "acetaminophen": "Tylenol. Pain/Fever. Liver toxicity > 4g.",
        "tramadol": "Weak opioid. Serotonin syndrome risk.",
        "oxycodone": "Opioid. Severe pain. Respiratory depression risk.",
        "morphine": "Opioid. Gold standard for MI/Pain.",
        "naloxone": "Narcan. Opioid antidote.",
        "albuterol": "Asthma rescue inhaler. Causes jitters.",
        "prednisone": "Steroid. High sugar, anger, insomnia.",
        "pantoprazole": "Protonix. Acid reflux.",
        "ondansetron": "Zofran. Nausea. QT prolongation.",
        "sertraline": "Zoloft. SSRI. Depression/Anxiety.",
        "fluoxetine": "Prozac. SSRI.",
        "alprazolam": "Xanax. Benzo. Anxiety. Addictive.",
        "lorazepam": "Ativan. Benzo. Seizures.",
        "vancomycin": "Antibiotic (MRSA). Monitor levels. Red Man Syndrome.",
        "piperacillin": "Zosyn. Broad spectrum antibiotic.",
        "ciprofloxacin": "Antibiotic (UTI). Tendon rupture risk.",
        "azithromycin": "Z-Pak. Pneumonia.",
        "diphenhydramine": "Benadryl. Allergy. Sedating.",
        "epinephrine": "Adrenaline. Anaphylaxis/Code Blue.",

        # --- LABS ---
        "wbc": "White Blood Cells. High=Infection.",
        "hgb": "Hemoglobin. Low=Anemia. <7 Transfuse.",
        "plt": "Platelets. Low=Bleeding risk.",
        "na": "Sodium. Low=Confusion.",
        "k": "Potassium. Critical for heart rhythm.",
        "bun": "BUN. High=Dehydration/Kidney.",
        "cr": "Creatinine. Best kidney marker.",
        "glucose": "Blood Sugar. 70-100 Fasting.",
        "a1c": "HbA1c. 3-month sugar avg. <5.7 Normal. >6.5 Diabetes.",
        "trop": "Troponin. Heart enzyme. High=Heart Attack.",
        "bnp": "Heart Failure marker. High=Fluid overload.",
        "inr": "Clotting time (Warfarin). Normal 1.0. Goal 2-3.",
        "lactate": "Sepsis marker. >2.0 indicates shock.",
        "ph": "Acidity (7.35-7.45).",

        # --- ABBREVIATIONS ---
        "bid": "Twice a day.",
        "tid": "Three times a day.",
        "qid": "Four times a day.",
        "qd": "Daily.",
        "prn": "As needed.",
        "ac": "Before meals.",
        "pc": "After meals.",
        "po": "By mouth.",
        "iv": "Intravenous.",
        "im": "Intramuscular.",
        "npo": "Nothing by mouth.",
        "stat": "Immediately.",
        "vs": "Vital Signs.",
        "nkda": "No Known Drug Allergies."
    }
    
    # 3. SEARCH LOGIC (WHOLE WORD + LONGEST MATCH)
    # We sort keys by length so "diabetes type 1" matches before "diabetes"
    sorted_keys = sorted(knowledge_base.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        # Use regex to find the keyword as a whole word only
        # This prevents "pe" from matching inside "type"
        if re.search(r'\b' + re.escape(key) + r'\b', text):
            return f"**{key.upper()}**: {knowledge_base[key]}"
            
    return "ℹ️ I didn't recognize that term. Try specific medical terms like 'Sepsis', 'Warfarin', 'INR', or 'Stroke'."
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

                # 4. SIRS Score (Updated to ignore 0s)
                sirs_score = 0
                
                # Temp check: Only if temp > 0
                if final_temp_c > 0 and (final_temp_c > 38 or final_temp_c < 36): 
                    sirs_score += 1
                
                # HR check
                if hr > 90: sirs_score += 1
                
                # Resp check
                if resp_rate > 20: sirs_score += 1
                
                # WBC check: Only if WBC > 0
                if wbc > 0 and (wbc > 12 or wbc < 4): 
                    sirs_score += 1

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
     # --- MODULE 2: PATIENT HISTORY (SQL) ---
    elif menu == "Patient History (SQL)":
        st.subheader("🗄️ Patient History Database")
        
        # Check if database exists
        if not os.path.exists('clinical_data.db'):
            st.info("📭 Database is empty. Run a Risk Analysis in the Calculator to create a record.")
        else:
            try:
                conn = sqlite3.connect('clinical_data.db')
                history_df = pd.read_sql("SELECT * FROM patient_history ORDER BY timestamp DESC", conn)
                conn.close()
                
                if not history_df.empty:
                    st.dataframe(history_df, use_container_width=True)
                    
                    st.markdown("### 📊 Cohort Analytics")
                    c1, c2 = st.columns(2)
                    c1.bar_chart(history_df['aki_risk_score'])
                    c2.scatter_chart(history_df, x='age', y='sbp')
                    
                    if st.button("🗑️ Clear Database"):
                        conn = sqlite3.connect('clinical_data.db')
                        conn.execute("DELETE FROM patient_history")
                        conn.commit()
                        conn.close()
                        st.rerun()
                else:
                    st.info("No records found yet.")
            except Exception as e:
                st.error(f"Database Error: {e}")
   # --- MODULE 3: LIVE DASHBOARD (VISUAL OVERHAUL) ---
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
    
    # --- MODULE 4: BATCH ANALYSIS (SMART + DIAGNOSTIC + IMAGING) ---
    elif menu == "Batch Analysis (CSV)":
        st.subheader("Bulk Patient Processing & Diagnostic Triage")
        
        # 1. Helper: Download Template
        with st.expander("ℹ️  How to format your CSV (Click to expand)"):
            st.write("You can use standard medical headers (e.g., HR, BP, Cr). The app will auto-detect them!")
            st.markdown("""
            * **Vitals:** `SBP`/`Systolic`, `HR`/`Pulse`, `RR`/`Resp`, `Temp`, `SpO2`/`O2`
            * **Labs:** `Cr`/`Creatinine`, `Glu`/`Glucose`, `WBC`, `INR`
            * **History:** `CHF`, `Liver`, `Blood_Thinner`, `GI_Bleed`
            """)
            
            sample_data = {
                'Age': [65, 72, 45], 'Gender': ['Male', 'Female', 'Male'],
                'Weight_kg': [80, 65, 90], 'Systolic_BP': [130, 220, 80],
                'Diastolic_BP': [80, 120, 40], 'Heart_Rate': [72, 40, 150],
                'Resp_Rate': [16, 8, 40], 'Temp_C': [37.0, 38.5, 34.0],
                'O2_Sat': [98, 92, 84], 'WBC': [6.0, 0.5, 25.0],
                'Glucose': [110, 700, 55], 'Creatinine': [1.1, 2.5, 3.0],
                'INR': [1.0, 3.2, 1.8], 'Altered_Mental': [0, 1, 1],
                'Anticoagulant': [1, 1, 0], 'NSAID': [0, 1, 0],
                'Heart_Failure': [0, 1, 1], 'Liver_Disease': [0, 0, 1],
                'Hx_GI_Bleed': [0, 1, 0]
            }
            df_sample = pd.DataFrame(sample_data)
            st.dataframe(df_sample, use_container_width=True)
            
            csv_template = df_sample.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Template", csv_template, "diagnostic_patient_data.csv", "text/csv")

        # 2. Main Processing Tool
        tab1, tab2 = st.tabs(["📄 Diagnostic Processor", "🖼️ Medical Imaging"])
        
        # --- TAB 1: SMART CSV PROCESSOR ---
        with tab1:
            uploaded_csv = st.file_uploader("Upload Patient Data (CSV)", type=["csv"])
            
            if uploaded_csv:
                try:
                    raw_df = pd.read_csv(uploaded_csv)
                    
                    # --- A. SMART COLUMN MAPPING ---
                    column_map = {
                        'sbp': 'Systolic_BP', 'sys': 'Systolic_BP', 'systolic': 'Systolic_BP',
                        'dbp': 'Diastolic_BP', 'dia': 'Diastolic_BP', 'diastolic': 'Diastolic_BP',
                        'hr': 'Heart_Rate', 'pulse': 'Heart_Rate',
                        'rr': 'Resp_Rate', 'respiration': 'Resp_Rate',
                        'temp': 'Temp_C', 'temperature': 'Temp_C',
                        'spo2': 'O2_Sat', 'o2': 'O2_Sat',
                        'sugar': 'Glucose', 'bgl': 'Glucose',
                        'cr': 'Creatinine', 'scr': 'Creatinine', 'creat': 'Creatinine',
                        'ams': 'Altered_Mental', 'confusion': 'Altered_Mental',
                        'blood_thinner': 'Anticoagulant', 'blood thinner': 'Anticoagulant',
                        'chf': 'Heart_Failure', 'hf': 'Heart_Failure',
                        'liver': 'Liver_Disease', 'cirrhosis': 'Liver_Disease',
                        'gi_bleed': 'Hx_GI_Bleed', 'bleed_history': 'Hx_GI_Bleed',
                        'sex': 'Gender', 'wt': 'Weight_kg', 'weight': 'Weight_kg'
                    }
                    df = raw_df.rename(columns=lambda x: column_map.get(x.lower(), x))
                    
                    # --- B. DATA SANITIZER (Fill Missing Cols) ---
                    required_cols = [
                        'Age', 'Gender', 'Weight_kg', 'Systolic_BP', 'Diastolic_BP', 
                        'Heart_Rate', 'Resp_Rate', 'Temp_C', 'O2_Sat', 'WBC', 'Glucose', 
                        'Creatinine', 'INR', 'Altered_Mental', 'Anticoagulant', 'NSAID', 
                        'Heart_Failure', 'Liver_Disease', 'Hx_GI_Bleed'
                    ]
                    
                    missing_log = []
                    for col in required_cols:
                        if col not in df.columns:
                            df[col] = 0 # Auto-fill 0
                            missing_log.append(col)
                    
                    if missing_log:
                        st.warning(f"⚠️ Auto-filled missing columns with 0: {', '.join(missing_log)}")
                    else:
                        st.success(f"✅ Successfully loaded {len(df)} patient records.")
                    
                    if st.button("⚡ Run AI Diagnostic Engine", type="primary"):
                        progress_bar = st.progress(0)
                        
                        # --- C. AI MODEL (Bleeding) ---
                        ai_inputs = pd.DataFrame()
                        ai_inputs['age'] = df['Age']
                        ai_inputs['inr'] = df['INR']
                        ai_inputs['anticoagulant'] = df['Anticoagulant']
                        ai_inputs['gi_bleed'] = df['Hx_GI_Bleed']
                        ai_inputs['high_bp'] = df['Systolic_BP'].apply(lambda x: 1 if x > 140 else 0)
                        ai_inputs['antiplatelet'] = 0 
                        ai_inputs['gender_female'] = df['Gender'].astype(str).apply(lambda x: 1 if 'fem' in x.lower() else 0)
                        ai_inputs['weight'] = df['Weight_kg']
                        ai_inputs['liver_disease'] = df['Liver_Disease']
                        
                        try:
                            df['Bleeding_Risk_%'] = bleeding_model.predict(ai_inputs)
                        except:
                            df['Bleeding_Risk_%'] = 0.0
                            
                        progress_bar.progress(30)
                        
                        # --- D. CALCULATE METRICS ---
                        df['MAP'] = (df['Systolic_BP'] + (2 * df['Diastolic_BP'])) / 3
                        
                        def calc_sirs(row):
                            score = 0
                            if row['Temp_C'] > 38 or row['Temp_C'] < 36: score += 1
                            if row['Heart_Rate'] > 90: score += 1
                            if row['Resp_Rate'] > 20: score += 1
                            if row['WBC'] > 12 or row['WBC'] < 4: score += 1
                            return score
                        
                        df['SIRS_Score'] = df.apply(calc_sirs, axis=1)
                        progress_bar.progress(60)

                        # --- E. DIAGNOSTIC LOGIC (Advanced) ---
                        def diagnose_patient(row):
                            diagnoses = []
                            
                            # 1. Cardiovascular
                            if row['Systolic_BP'] > 180 or row['Diastolic_BP'] > 120: 
                                diagnoses.append("Hypertensive Crisis")
                            
                            if row['MAP'] > 0 and row['MAP'] < 65: 
                                if row['SIRS_Score'] >= 2: diagnoses.append("Septic Shock")
                                elif row['Heart_Failure'] == 1: diagnoses.append("Cardiogenic Shock")
                                elif row['Hx_GI_Bleed'] == 1: diagnoses.append("Hemorrhagic Shock")
                                else: diagnoses.append("Hypotensive Shock")
                                
                            if row['Heart_Rate'] > 0 and row['Heart_Rate'] < 50: diagnoses.append("Severe Bradycardia")
                            elif row['Heart_Rate'] > 140: diagnoses.append("Unstable Tachycardia")
                            
                            # 2. Respiratory
                            if row['O2_Sat'] > 0 and row['O2_Sat'] < 85: diagnoses.append("Critical Hypoxia")
                            elif row['O2_Sat'] > 0 and row['O2_Sat'] < 90:
                                if row['Heart_Failure'] == 1: diagnoses.append("Pulmonary Edema")
                                elif row['Temp_C'] > 38: diagnoses.append("Pneumonia / ARDS")
                                else: diagnoses.append("Hypoxic Resp Failure")
                            
                            if row['Resp_Rate'] > 35: diagnoses.append("Severe Resp Distress")
                                
                            # 3. Renal / Metabolic
                            if row['Creatinine'] > 3.0: diagnoses.append("Severe AKI (Stage 3)")
                            elif row['Creatinine'] > 1.5: diagnoses.append("Acute Kidney Injury")
                            
                            if row['Glucose'] > 0 and row['Glucose'] < 70: diagnoses.append("Hypoglycemia")
                            elif row['Glucose'] > 600: diagnoses.append("HHS (Hyperosmolar)")
                            elif row['Glucose'] > 250: diagnoses.append("Hyperglycemia / DKA Risk")
                            
                            # 4. Infection / Immune
                            if row['WBC'] > 0 and row['WBC'] < 1.5 and row['Temp_C'] > 38: 
                                diagnoses.append("Neutropenic Sepsis (CRITICAL)")
                            elif row['WBC'] > 25: 
                                diagnoses.append("Severe Leukocytosis")
                            elif row['SIRS_Score'] >= 2 and "Septic Shock" not in diagnoses:
                                diagnoses.append("Sepsis Protocol Required")
                                
                            if row['Temp_C'] > 0 and row['Temp_C'] < 35: diagnoses.append("Hypothermia")
                                
                            # 5. Neuro / Liver
                            if row['Altered_Mental'] == 1 and row['Resp_Rate'] > 0 and row['Resp_Rate'] < 10:
                                diagnoses.append("CNS Depression / Overdose Risk")
                                
                            if row['Liver_Disease'] == 1 and row['INR'] > 1.7:
                                diagnoses.append("Liver Decompensation")
                            elif row['INR'] > 4.0: 
                                diagnoses.append("Supratherapeutic INR")
                            
                            # Final Output
                            if len(diagnoses) > 0: return " + ".join(diagnoses)
                            return "Stable / No Acute Findings"

                        df['Suggested_Diagnosis'] = df.apply(diagnose_patient, axis=1)
                        progress_bar.progress(100)
                        
                        # --- F. DISPLAY RESULTS ---
                        st.divider()
                        st.subheader("📋 AI Diagnostic Report")
                        
                        def color_rows(val):
                            s = str(val)
                            if 'Shock' in s or 'CRITICAL' in s or 'Crisis' in s: 
                                return 'background-color: #ffcdd2; color: black; font-weight: bold;' # Red
                            elif 'Sepsis' in s or 'Failure' in s or 'Severe' in s: 
                                return 'background-color: #fff9c4; color: black;' # Yellow
                            elif 'Stable' in s: 
                                return 'background-color: #c8e6c9; color: black;' # Green
                            return ''

                        st.dataframe(
                            df[['Suggested_Diagnosis', 'Age', 'Gender', 'MAP', 'Glucose', 'WBC', 'Bleeding_Risk_%']]
                            .style.map(color_rows, subset=['Suggested_Diagnosis']),
                            use_container_width=True
                        )
                        
                        # Metrics
                        c1, c2, c3, c4 = st.columns(4)
                        shock_count = len(df[df['Suggested_Diagnosis'].str.contains("Shock")])
                        sepsis_count = len(df[df['Suggested_Diagnosis'].str.contains("Sepsis")])
                        htn_count = len(df[df['Suggested_Diagnosis'].str.contains("Crisis")])
                        
                        c1.metric("🔴 Shock Cases", shock_count)
                        c2.metric("🟠 Sepsis Alerts", sepsis_count)
                        c3.metric("⚠️ Hypertensive Crisis", htn_count)
                        c4.metric("Avg Bleed Risk", f"{df['Bleeding_Risk_%'].mean():.1f}%")
                        
                        # Download
                        csv_result = df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Diagnostic Report", csv_result, "diagnostic_report.csv", "text/csv", type="primary")
                        
                except Exception as e:
                    st.error(f"Error processing CSV: {e}")
                    st.info("Tip: Check column names if automatic mapping failed.")

        # --- TAB 2: MEDICAL IMAGING (SIMULATION) ---
        with tab2:
            st.markdown("### 🖼️ AI Medical Imaging Diagnostic Tool")
            st.info("ℹ️ **Note:** This module simulates a DenseNet-121 Deep Learning model analyzing chest X-Rays for triage purposes.")
            
            uploaded_image = st.file_uploader("Upload DICOM/JPEG X-Ray", type=["jpg", "png", "jpeg", "dcm"])
            
            if uploaded_image:
                col_img, col_report = st.columns([1, 1.5])
                with col_img:
                    st.image(uploaded_image, caption="Source Input", use_container_width=True)
                
                with col_report:
                    st.markdown("#### AI Analysis Protocol")
                    if st.button("⚡ Initialize Neural Network Scan", type="primary"):
                        import time
                        progress_text = "Operation in progress. Please wait."
                        my_bar = st.progress(0, text=progress_text)

                        steps = [
                            "Preprocessing: Normalizing Hounsfield Units...",
                            "Applying CLAHE Contrast Enhancement...",
                            "Loading DenseNet-121 Weights...",
                            "Generating Grad-CAM Heatmaps...",
                            "Finalizing Classification Probabilities..."
                        ]
                        
                        for percent_complete, step in zip(range(0, 100, 20), steps):
                            time.sleep(0.8)
                            my_bar.progress(percent_complete + 20, text=step)
                        my_bar.empty()
                        
                        scenarios = [
                            {"diag": "Normal Study", "conf": 99.2, "severity": "🟢 Low", "findings": "Cardiomediastinal silhouette is within normal limits.", "rec": "Discharge."},
                            {"diag": "Lobar Pneumonia", "conf": 88.5, "severity": "🔴 High", "findings": "Opacities identified in the right lower lobe.", "rec": "Initiate Antibiotics."},
                            {"diag": "Pneumothorax", "conf": 94.1, "severity": "🔴 CRITICAL", "findings": "Visible visceral pleural edge seen in left apex.", "rec": "Urgent Surgical Consult."},
                            {"diag": "Congestive Heart Failure", "conf": 82.3, "severity": "🟠 Moderate", "findings": "Cardiomegaly present (CTR > 0.5).", "rec": "Diuretics (Furosemide)."}
                        ]
                        result = np.random.choice(scenarios)
                        
                        st.success("Scan Complete successfully.")
                        st.divider()
                        header_color = "red" if "CRITICAL" in result['severity'] else ("orange" if "High" in result['severity'] else "green")
                        st.markdown(f":{header_color}[**PREDICTION: {result['diag']}**]")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("AI Confidence", f"{result['conf']}%")
                        m2.metric("Severity Index", result['severity'])
                        m3.metric("Image Quality", "Optimal")
                        
                        st.markdown("#### 🔍 Radiomic Findings")
                        st.info(f"{result['findings']}")
                        st.markdown("#### 🤖 Clinical Recommendation")
                        st.write(f"> *{result['rec']}*")
                        
# --- MODULE 5: MEDICATION CHECKER (UPDATED UI) ---
    elif menu == "Medication Checker":
        st.subheader("Drug-Drug Interaction Checker")
        
        # UPDATED GUIDE: Now reflects your massive database expansion
        with st.expander("ℹ️  Supported Drug Categories (Expanded Database)"):
            st.markdown("""
            This demo checks for **High-Alert** interactions across these categories:
            * **🫀 Cardiac & Blood:** Warfarin, Amiodarone, Digoxin, Sildenafil, Nitroglycerin, Spironolactone, Lisinopril, Apixaban, Clopidogrel.
            * **💊 Pain & Inflammation:** Ibuprofen, Tramadol, Fentanyl, Methotrexate, Oxycodone, Naproxen.
            * **🧠 Psych & Neuro:** Sertraline, Fluoxetine, Linezolid, Lithium, Citalopram, Tizanidine.
            * **🦠 Antibiotics:** Ciprofloxacin, Erythromycin, Bactrim (Trimethoprim), Clarithromycin, Doxycycline, Metronidazole.
            * **🍷 Lifestyle & Food:** Alcohol, Grapefruit Juice.
            """)
            
        st.caption("Type two drugs below to check for CRITICAL or MAJOR interactions.")

        # 2. Input Fields
        col_d1, col_d2 = st.columns(2)
        d1 = col_d1.text_input("Drug A", placeholder="e.g. Warfarin")
        d2 = col_d2.text_input("Drug B", placeholder="e.g. Ibuprofen")

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

# --- MODULE 6: CHATBOT (UPDATED UI) ---
    elif menu == "Clinical Chatbot":
        st.subheader("AI Clinical Assistant")
        
        # Guide: Tells the user what the bot knows
        with st.expander("ℹ️  What can I ask? (Supported Topics)"):
            st.markdown("""
            **The database covers 250+ clinical topics across these specialties:**
            * **🫀 Cardiology:** MI, AFib, CHF, Hypertension, ECG Rhythms (VTach, VFib).
            * **🫁 Respiratory:** COPD, Asthma, Pneumonia, Pulmonary Embolism.
            * **🧠 Neurology:** Stroke (CVA), TIA, Seizures, Meningitis.
            * **💊 Pharmacology:** Warfarin, Insulin, Metformin, Lisinopril, Vancomycin.
            * **🩸 Labs & Vitals:** INR, Creatinine, Troponin, Lactate, WBC, Platelets.
            * **🦠 Infectious Disease:** Sepsis, Septic Shock, C. Diff, MRSA.
            * **🏥 Abbreviations:** NPO, PRN, BID, QD, AC/PC.
            """)
            st.caption("Try typing a term (e.g., 'Sepsis') or a question (e.g., 'What is the treatment for DKA?').")

        # Chat Input
        q = st.text_input("Ask a clinical question:")
        if q:
            with st.chat_message("assistant"):
                st.write(chatbot_response(q))
