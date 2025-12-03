# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def train_and_save():
    print("⏳ Generating synthetic clinical data (Simulation Mode)...")
    np.random.seed(42)
    n_samples = 5000  

    data = pd.DataFrame({
        'age': np.random.randint(18, 95, n_samples),
        'inr': np.random.uniform(0.8, 6.0, n_samples),
        'anticoagulant': np.random.randint(0, 2, n_samples),
        'gi_bleed': np.random.randint(0, 2, n_samples),
        'high_bp': np.random.randint(0, 2, n_samples),
        'antiplatelet': np.random.randint(0, 2, n_samples),
        'gender_female': np.random.randint(0, 2, n_samples),
        'weight': np.random.normal(75, 15, n_samples),
        'liver_disease': np.random.randint(0, 2, n_samples),
    })

    # Ground Truth Logic (Simulating Clinical Risk Rules like HAS-BLED)
    def rules(row):
        score = 0
        score += 35 if row['anticoagulant'] else 0
        score += 40 if row['inr'] > 3.5 else 0
        score += 30 if row['gi_bleed'] else 0
        score += 15 if row['antiplatelet'] else 0
        score += 10 if row['age'] > 65 else 0  # Updated age threshold
        score += 10 if row['high_bp'] else 0
        score += 15 if row['liver_disease'] else 0
        
        # Add some random noise to simulate real-world biological variance
        noise = np.random.normal(0, 3) 
        return min(max(score + noise, 0), 100)

    data['risk_score'] = data.apply(rules, axis=1)

    # --- THE CRITICAL FIX: Train/Test Split ---
    print("✂️ Splitting data into Training and Validation sets...")
    X = data.drop('risk_score', axis=1)
    y = data['risk_score']
    
    # 80% Train, 20% Test (Standard Data Science Practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    print("🧠 Training XGBoost model...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=150, 
        max_depth=4, 
        learning_rate=0.05
    )
    model.fit(X_train, y_train)

    # Validate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"✅ Model Validation Results:")
    print(f"   - RMSE: {rmse:.2f} (Average prediction error in risk points)")
    print(f"   - R² Score: {r2:.4f} (1.0 is perfect prediction)")

    # Save
    model.save_model("bleeding_risk_model.json")
    print("💾 Model saved as 'bleeding_risk_model.json'")

if __name__ == "__main__":
    train_and_save()
