"""
Dataset Loader — Real OASIS Longitudinal Alzheimer's Dataset
Source: Open Access Series of Imaging Studies (OASIS)
https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers

Features:
  Age, M/F, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF
Target:
  Group → Nondemented / Demented / Converted

Auto-downloads from public mirror if not present locally.
"""

import os
import urllib.request
import pandas as pd
import numpy as np

OASIS_URL = (
    'https://raw.githubusercontent.com/MainakRepositor/Datasets/'
    'master/oasis_longitudinal.csv'
)


def load_oasis_dataset(save_path='data/oasis_longitudinal.csv') -> pd.DataFrame:
    """
    Download and return the real OASIS longitudinal dataset.
    Falls back to synthetic data only if download fails.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(save_path):
        print("📥 Downloading real OASIS Alzheimer's dataset...")
        try:
            urllib.request.urlretrieve(OASIS_URL, save_path)
            print(f"✅ OASIS dataset downloaded → {save_path}")
        except Exception as e:
            print(f"⚠️  Download failed ({e}). Falling back to synthetic data.")
            return _generate_synthetic(save_path)

    df = pd.read_csv(save_path)
    print(f"✅ Loaded OASIS dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Class distribution:\n{df['Group'].value_counts().to_string()}")
    return df


def prepare_oasis(df: pd.DataFrame, unit: str = "baseline") -> pd.DataFrame:
    """
    Clean and prepare the OASIS dataset for ML.
    - Drop non-clinical admin columns (retains 'Subject ID' and 'Visit')
    - Rename M/F → Gender
    - Keep only clinically relevant features
    - Select analysis unit: 'baseline' (first visit per subject) or 'all_visits'
    """
    df = df.copy()

    # Drop non-clinical columns (retain Subject ID and Visit)
    drop_cols = ['MRI ID', 'Hand', 'MR Delay']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Rename for clarity
    if 'M/F' in df.columns and 'Gender' not in df.columns:
        df.rename(columns={'M/F': 'Gender'}, inplace=True)

    # Filter analysis unit if Visit and Subject ID exist
    if 'Visit' in df.columns and 'Subject ID' in df.columns:
        if unit == 'baseline':
            df = df.sort_values(['Subject ID', 'Visit']).drop_duplicates(
                subset=['Subject ID'], keep='first'
            ).reset_index(drop=True)
        elif unit == 'all_visits':
            df = df.sort_values(['Subject ID', 'Visit']).reset_index(drop=True)
        else:
            raise ValueError(f"Unknown unit '{unit}'. Expected 'baseline' or 'all_visits'.")

    return df


def _generate_synthetic(save_path: str) -> pd.DataFrame:
    """
    High-quality synthetic fallback based on real OASIS statistics.
    Only used if download fails.
    """
    np.random.seed(42)
    n = 373

    n0 = 190  # Nondemented
    n1 = 146  # Demented
    n2 = 37   # Converted

    def grp(n, label, age_mu, mmse_mu, mmse_sd, cdr_probs, nwbv_mu):
        return pd.DataFrame({
            'Age'   : np.clip(np.random.normal(age_mu, 8, n), 60, 98).astype(int),
            'Gender': np.random.choice(['M', 'F'], n),
            'EDUC'  : np.clip(np.random.normal(14, 3, n), 6, 23).astype(int),
            'SES'   : np.where(np.random.rand(n) < 0.05, np.nan,
                               np.random.choice([1,2,3,4,5], n, p=[0.1,0.2,0.4,0.2,0.1])),
            'MMSE'  : np.clip(np.random.normal(mmse_mu, mmse_sd, n), 0, 30),
            'CDR'   : np.random.choice([0.0,0.5,1.0,2.0], n, p=cdr_probs),
            'eTIV'  : np.clip(np.random.normal(1480, 180, n), 1100, 2000).astype(int),
            'nWBV'  : np.clip(np.random.normal(nwbv_mu, 0.02, n), 0.60, 0.90),
            'ASF'   : np.clip(np.random.normal(1.2, 0.15, n), 0.88, 1.58),
            'Visit' : 1,
            'Group' : label
        })

    df = pd.concat([
        grp(n0, 'Nondemented', 76, 28.5, 1.5, [0.85,0.12,0.02,0.01], 0.760),
        grp(n1, 'Demented',    78, 20.0, 4.5, [0.05,0.30,0.45,0.20], 0.710),
        grp(n2, 'Converted',   75, 26.0, 2.5, [0.30,0.55,0.13,0.02], 0.735),
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    df['Subject ID'] = [f"SYN_{i+1:04d}" for i in range(len(df))]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Synthetic dataset saved → {save_path}")
    return df
