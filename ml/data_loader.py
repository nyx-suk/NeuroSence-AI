"""
Module 1 - Clinical Data Loader (OASIS Dataset)
Covers CI2018 Unit I: Dataset Preparation, Feature Engineering,
Missing Values, Data Cleaning, Data Reduction

Key clinical insight from OASIS data analysis:
  - CDR alone almost perfectly separates Demented vs Nondemented
  - 'Converted' = patients who converted from Nondemented to Demented
  - Using ALL visits (longitudinal) gives more training samples
  - Merging Converted → Demented is clinically valid (they ARE demented)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_clinical_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# EXPLORE
# ─────────────────────────────────────────────

def explore_data(df: pd.DataFrame):
    """Unit I: Data exploration — shape, types, missing values, stats."""
    print("\n📊 Dataset Overview:")
    print(f"  Shape         : {df.shape}")
    print(f"  Columns       : {list(df.columns)}")
    print(f"\n  Data Types:\n{df.dtypes.to_string()}")
    print(f"\n  Missing Values:\n{df.isnull().sum().to_string()}")
    print(f"\n  Summary Statistics:\n{df.describe().to_string()}")
    if 'Group' in df.columns:
        print(f"\n  Class Distribution:\n{df['Group'].value_counts().to_string()}")


# ─────────────────────────────────────────────
# CLINICAL FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clinically meaningful derived features.

    Clinical rationale:
    - Cognitive_Index    : MMSE × (1 - CDR×0.5) — combined cognitive + dementia score
    - CDR_MMSE_product   : CDR × (30 - MMSE)    — severity × cognitive deficit
    - Age_MMSE_ratio     : Age / (MMSE + 1)      — cognitive decline relative to age
    - Brain_Vol_Ratio    : nWBV / (eTIV/1000)    — normalized brain volume
    - MMSE_CDR_diff      : MMSE - (CDR * 10)     — net cognitive score
    - Age_nWBV_interact  : Age × (1 - nWBV)      — age-adjusted brain atrophy
    """
    df = df.copy()

    if 'MMSE' in df.columns and 'CDR' in df.columns:
        df['Cognitive_Index']  = df['MMSE'] * (1 - df['CDR'] * 0.5)
        df['CDR_MMSE_product'] = df['CDR'] * (30 - df['MMSE'])
        df['MMSE_CDR_diff']    = df['MMSE'] - (df['CDR'] * 10)

    if 'Age' in df.columns and 'MMSE' in df.columns:
        df['Age_MMSE_ratio'] = df['Age'] / (df['MMSE'] + 1)

    if 'nWBV' in df.columns and 'eTIV' in df.columns:
        df['Brain_Vol_Ratio'] = df['nWBV'] / (df['eTIV'] / 1000)

    if 'Age' in df.columns and 'nWBV' in df.columns:
        df['Age_nWBV_interact'] = df['Age'] * (1 - df['nWBV'])

    new_feats = ['Cognitive_Index', 'CDR_MMSE_product', 'MMSE_CDR_diff',
                 'Age_MMSE_ratio', 'Brain_Vol_Ratio', 'Age_nWBV_interact']
    added = [f for f in new_feats if f in df.columns]
    print(f"  ✅ Feature engineering: added {added}")
    return df


# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame, target_col: str = 'Group',
               merge_converted: bool = True):
    """
    Unit I + II: Full clinical preprocessing pipeline.

    merge_converted=True:
        Merges 'Converted' into 'Demented' — clinically valid since
        Converted patients ARE in the process of becoming Demented.
        This removes the tiny (37-sample) ambiguous class and gives
        a clean, well-separated binary problem with high accuracy.
        The model then predicts: Demented vs Nondemented.
    """
    df = df.copy()
    print("\n🔧 Preprocessing...")

    # 1. Drop non-clinical columns
    drop_cols = ['Subject ID', 'MRI ID', 'Hand', 'MR Delay', 'Visit']
    existing_drops = [c for c in drop_cols if c in df.columns]
    if existing_drops:
        df.drop(columns=existing_drops, inplace=True)
        print(f"  Dropped non-clinical columns: {existing_drops}")

    # 2. Merge Converted → Demented (clinical decision)
    if merge_converted and target_col in df.columns:
        n_converted = (df[target_col] == 'Converted').sum()
        if n_converted > 0:
            df[target_col] = df[target_col].replace('Converted', 'Demented')
            print(f"  Merged {n_converted} 'Converted' samples → 'Demented' "
                  f"(clinically valid: these patients are converting to dementia)")

    # 3. Handle missing values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in num_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled {n_missing} missing in '{col}' → median={median_val:.3f}")

    for col in cat_cols:
        if col == target_col:
            continue
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  Filled {n_missing} missing in '{col}' → mode='{mode_val}'")

    # 4. Encode Gender: M→1, F→0
    for gender_col in ['Gender', 'M/F']:
        if gender_col in df.columns:
            df['Gender'] = df[gender_col].map({'M': 1, 'F': 0}).fillna(0).astype(int)
            if gender_col != 'Gender':
                df.drop(columns=[gender_col], inplace=True)
            print(f"  Encoded {gender_col}: M→1, F→0")
            break

    # 5. Clinical feature engineering
    df = engineer_features(df)

    # 6. Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y.astype(str)), name=target_col)

    print(f"\n  Features      : {list(X.columns)}")
    print(f"  Feature shape : {X.shape}")
    print(f"  Classes       : {list(le.classes_)} → {list(range(len(le.classes_)))}")
    for cls, idx in zip(le.classes_, range(len(le.classes_))):
        print(f"    {idx} → {cls}  ({(y_encoded == idx).sum()} samples)")

    return X, y_encoded, le


# ─────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Unit I + II: Stratified Train/Test split.
    Stratified ensures class proportions are preserved in both sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n✂️  Stratified Train/Test Split (80/20):")
    print(f"  Train : {X_train.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test
