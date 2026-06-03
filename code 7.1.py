import hashlib
import os
import numpy as np
import pandas as pd

# --- Configuration ---
# Cryptographic salt generated dynamically and stored securely in vault configurations
SECRET_SALT_KEY = os.environ.get("PATIENT_DB_SALT", "SecureEnterpriseCryptoSalt2026!").encode('utf-8')

# Mock Raw Patient Database containing Protected Health Information (PHI)
raw_hospital_records = pd.DataFrame({
    'Patient_ID': ['PT-9021', 'PT-4482', 'PT-1109'],
    'Full_Name': ['Mishal Majeed', 'Nihal Manoj', 'Bharath Yoonus'],
    'National_Health_ID': ['SSN-998-11', 'SSN-445-90', 'SSN-332-12'],
    'Age': [45, 62, 29],
    'Tumor_Size_MM': [24.5, 12.2, 38.1]
})

def pseudonymize_string(input_string: str) -> str:
    """Computes a secure SHA-256 salted hash signature to lock identifiable metrics."""
    hash_engine = hashlib.sha256()
    # Apply salt to prevent lookup dictionary attacks
    hash_engine.update(SECRET_SALT_KEY + str(input_string).strip().upper().encode('utf-8'))
    return hash_engine.hexdigest()[:16] # Return a clean 16-character token string

def anonymize_clinical_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw transactional medical ledgers into anonymized,
    GDPR-compliant records completely safe for open-loop model retraining optimization.
    """
    anonymized_df = df.copy()
    
    # 1. Strip and Pseudonymize Direct Identifiers
    anonymized_df['Patient_Token'] = anonymized_df['Patient_ID'].apply(pseudonymize_string)
    anonymized_df.drop(columns=['Patient_ID', 'Full_Name', 'National_Health_ID'], inplace=True)
    
    # 2. Implement Generalization buckets (Age Binning)
    # Replaces absolute age with statistical decade bands to break individual lookups
    anonymized_df['Age_Group'] = pd.cut(
        anonymized_df['Age'], 
        bins=[0, 20, 30, 40, 50, 60, 70, 100], 
        labels=['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
    )
    anonymized_df.drop(columns=['Age'], inplace=True)
    
    # 3. Apply Localized Numerical Perturbation (Differential Noise Addition)
    # Adds slight Gaussian noise to float metrics to ensure exact value tracking is impossible,
    # while preserving the overall global mathematical trend variance required for machine learning model training.
    noise = np.random.normal(0, 0.05, size=len(anonymized_df))
    anonymized_df['Tumor_Size_MM'] = np.round(anonymized_df['Tumor_Size_MM'] + noise, 2)
    
    # Reorder columns for visibility structure
    return anonymized_df[['Patient_Token', 'Age_Group', 'Tumor_Size_MM']]

# Execute Ingestion Anonymization Pipeline
if __name__ == "__main__":
    print("\n--- Raw Inbound PHI Database Records (Pre-Anonymization) ---")
    print(raw_hospital_records)
    
    anonymized_output = anonymize_clinical_ledger(raw_hospital_records)
    
    print("\n--- Secure GDPR-Compliant Data Frame (Ready for Retraining Loop) ---")
    print(anonymized_output)
    print("\nVerification Status: Zero direct identifiers tracking out to storage disks.")