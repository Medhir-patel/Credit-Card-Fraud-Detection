"""
Synthetic Dataset Generator for Credit Card Fraud Detection
============================================================
Generates a benchmark-style creditcard.csv dataset if not present.
"""

import os
import numpy as np
import pandas as pd

def generate_synthetic_creditcard_csv(filename="creditcard.csv", num_samples=10000, fraud_ratio=0.02, random_state=42):
    if os.path.exists(filename):
        print(f"[OK] '{filename}' already exists.")
        return

    print(f"Generating synthetic '{filename}' with {num_samples} records...")
    np.random.seed(random_state)
    
    num_fraud = int(num_samples * fraud_ratio)
    num_legit = num_samples - num_fraud
    
    # Generate Time (0 to 172800 seconds = 48 hours)
    time_legit = np.random.uniform(0, 172800, num_legit)
    time_fraud = np.random.uniform(0, 172800, num_fraud)
    time_all = np.concatenate([time_legit, time_fraud])
    
    # Generate V1 to V28 (PCA features)
    v_legit = np.random.randn(num_legit, 28)
    v_fraud = np.random.randn(num_fraud, 28)
    
    # Inject fraud signals into key PCA dimensions (V14, V12, V10, V17, V4, V11)
    v_fraud[:, 13] -= 5.0  # V14
    v_fraud[:, 11] -= 4.0  # V12
    v_fraud[:, 9]  -= 4.0  # V10
    v_fraud[:, 16] -= 3.5  # V17
    v_fraud[:, 3]  += 4.0  # V4
    v_fraud[:, 10] += 3.5  # V11
    
    v_all = np.vstack([v_legit, v_fraud])
    
    # Generate Amount
    amount_legit = np.random.exponential(scale=88.0, size=num_legit)
    amount_fraud = np.random.exponential(scale=120.0, size=num_fraud)
    amount_all = np.concatenate([amount_legit, amount_fraud])
    
    # Target Class
    class_all = np.array([0] * num_legit + [1] * num_fraud)
    
    # Build DataFrame
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    data = np.column_stack([time_all, v_all, amount_all, class_all])
    df = pd.DataFrame(data, columns=columns)
    
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df['Class'] = df['Class'].astype(int)
    
    df.to_csv(filename, index=False)
    print(f"[OK] Successfully saved {len(df)} transactions to '{filename}'.")

if __name__ == "__main__":
    generate_synthetic_creditcard_csv()
