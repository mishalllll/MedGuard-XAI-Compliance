import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# --- Step 1: Initialize Mock Clinical Feature Space ---
np.random.seed(42)
num_patients = 1000

# Diagnostic indicators: Age, Blood Pressure, Tumor Size (mm), Genetic Score, White Cell Count
clinical_features = pd.DataFrame({
    'Age': np.random.randint(18, 85, size=num_patients),
    'Blood_Pressure_Systolic': np.random.normal(120, 15, size=num_patients),
    'Tumor_Size_MM': np.random.exponential(scale=15, size=num_patients),
    'Genetic_Risk_Score': np.random.uniform(0.0, 1.0, size=num_patients),
    'White_Cell_Count': np.random.normal(7000, 1500, size=num_patients)
})

# Class target: 0 = Benign/Healthy, 1 = Malignant/High-Risk Intervention Required
# Correlating tumor size and genetic scores directly to target probability
risk_logits = (clinical_features['Tumor_Size_MM'] * 0.15) + (clinical_features['Genetic_Risk_Score'] * 3.0) - 4.0
probability = 1 / (1 + np.exp(-risk_logits))
diagnostic_labels = (probability > 0.5).astype(int)

# Stratified Split for Validation
X_train, X_test, y_train, y_test = train_test_split(
    clinical_features, diagnostic_labels, test_size=0.2, random_state=42, stratify=diagnostic_labels
)

# --- Step 2: Train High-Accuracy Classification Model ---
print("Training clinical diagnostic core model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 3: Compute Mathematical SHAP Explainer Values ---
# Initialize TreeExplainer optimized for tree-based ensemble models
explainer = shap.TreeExplainer(model)
# Calculate values specifically for the positive class (Malignant diagnosis)
shap_values = explainer(X_test)


# --- Step 4: Interface Function for Physician Human Oversight Override ---
def generate_patient_diagnostic_dashboard(patient_index_in_test_set):
    """
    Generates a localized explainability breakdown for an individual patient,
    providing the mathematical baseline required for a doctor to execute an override.
    """
    print(f"\n=======================================================")
    print(f" CLINICAL DIAGNOSTIC REPORT - PATIENT LAB REF: #{patient_index_in_test_set}")
    print(f"=======================================================")
    
    # Extract specific patient data metrics
    patient_data = X_test.iloc[[patient_index_in_test_set]]
    raw_prediction = model.predict(patient_data)[0]
    prediction_proba = model.predict_proba(patient_data)[0][1] * 100
    
    print(f"-> Automated AI Classification: {'MALIGNANT ALERT' if raw_prediction == 1 else 'BENIGN / HEALTHY'}")
    print(f"-> Calculated Malignancy Probability: {prediction_proba:.2f}%")
    print("\nPatient Vital Metrics Metrics:")
    print(patient_data.to_string(index=False))
    
    print("\nGenerating SHAP Force Visualization for human oversight verification...")
    # Isolate the explicit shap values for class 1 (Malignant) for this patient
    patient_shap_explanation = shap_values[patient_index_in_test_set, :, 1]
    
    # Render static plot to file for UI inclusion
    plt.figure(figsize=(10, 3))
    shap.plots.plots_force = True  # Enforce visualization rendering flags
    shap.plots.waterfall(patient_shap_explanation, show=False)
    plt.title(f"Diagnostic Explanation Profile (Patient #{patient_index_in_test_set})", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(f"patient_explanation_{patient_index_in_test_set}.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"[Success] Saved explainer graph to disk: 'patient_explanation_{patient_index_in_test_set}.png'")
    print("MANDATORY MANDATE WARNING: Review feature contribution deltas prior to clinical signoff.")

# Test dashboard deployment execution
generate_patient_diagnostic_dashboard(patient_index_in_test_set=5)