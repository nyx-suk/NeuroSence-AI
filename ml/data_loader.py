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
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, GroupShuffleSplit
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
        print(f"\n  Class Distribution (Rows):\n{df['Group'].value_counts().to_string()}")
        if 'Subject ID' in df.columns:
            print(f"\n  Distinct Subjects per Class:\n{df.groupby('Group')['Subject ID'].nunique().to_string()}")


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
               merge_converted: bool = True, unit: str = 'baseline'):
    """
    Unit I + II: Full clinical preprocessing pipeline.

    unit='baseline':
        Uses each subject's first visit.
    unit='all_visits':
        Uses every visit and requires group-aware splitting.

    merge_converted=True:
        Merges 'Converted' into 'Demented' — clinically valid since
        Converted patients ARE in the process of becoming Demented.
        This removes the tiny (37-sample) ambiguous class and gives
        a clean, well-separated binary problem with high accuracy.
        The model then predicts: Demented vs Nondemented.

    Returns:
        X (pd.DataFrame): Processed feature matrix.
        y_encoded (pd.Series): Encoded target series.
        groups (pd.Series): Subject ID series corresponding to each sample.
        le (LabelEncoder): Fitted label encoder for target classes.
    """
    df = df.copy()
    print("\n🔧 Preprocessing...")

    # 1. Filter by analysis unit (baseline vs all_visits)
    if 'Visit' in df.columns and 'Subject ID' in df.columns:
        if unit == 'baseline':
            df = df.sort_values(['Subject ID', 'Visit']).drop_duplicates(
                subset=['Subject ID'], keep='first'
            ).reset_index(drop=True)
            print("  Analysis unit: baseline (each subject's first visit)")
        elif unit == 'all_visits':
            df = df.sort_values(['Subject ID', 'Visit']).reset_index(drop=True)
            print("  Analysis unit: all_visits (longitudinal; requires group-aware splitting)")
        else:
            raise ValueError(f"Unknown unit '{unit}'. Expected 'baseline' or 'all_visits'.")

    # 2. Retain Subject ID as separate groups Series
    if 'Subject ID' in df.columns:
        groups = df['Subject ID'].copy().reset_index(drop=True)
    else:
        groups = pd.Series(df.index, name='Subject ID')

    # 3. Drop non-clinical columns from feature dataframe
    drop_cols = ['Subject ID', 'MRI ID', 'Hand', 'MR Delay', 'Visit']
    existing_drops = [c for c in drop_cols if c in df.columns]
    if existing_drops:
        df.drop(columns=existing_drops, inplace=True)
        print(f"  Dropped non-clinical columns: {existing_drops}")

    # 4. Merge Converted → Demented (clinical decision)
    if merge_converted and target_col in df.columns:
        n_converted = (df[target_col] == 'Converted').sum()
        if n_converted > 0:
            df[target_col] = df[target_col].replace('Converted', 'Demented')
            print(f"  Merged {n_converted} 'Converted' samples → 'Demented' "
                  f"(clinically valid: these patients are converting to dementia)")

    # 5. Handle missing values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in num_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled {n_missing} missing in '{col}' → median={median_val:.3f}")

    for col in cat_cols:
        if col == target_col:
            continue
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  Filled {n_missing} missing in '{col}' → mode='{mode_val}'")

    # 6. Encode Gender: M→1, F→0
    for gender_col in ['Gender', 'M/F']:
        if gender_col in df.columns:
            df['Gender'] = df[gender_col].map({'M': 1, 'F': 0}).fillna(0).astype(int)
            if gender_col != 'Gender':
                df.drop(columns=[gender_col], inplace=True)
            print(f"  Encoded {gender_col}: M→1, F→0")
            break

    # 7. Clinical feature engineering
    df = engineer_features(df)

    # 8. Separate features and target
    X = df.drop(columns=[target_col]).reset_index(drop=True)
    y = df[target_col].reset_index(drop=True)

    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y.astype(str)), name=target_col)

    print(f"\n  Features      : {list(X.columns)}")
    print(f"  Feature shape : {X.shape}")
    print(f"  Classes       : {list(le.classes_)} → {list(range(len(le.classes_)))}")
    for cls, idx in zip(le.classes_, range(len(le.classes_))):
        n_rows = (y_encoded == idx).sum()
        n_subjs = groups[y_encoded == idx].nunique()
        print(f"    {idx} → {cls}  ({n_rows} rows, {n_subjs} distinct subjects)")

    return X, y_encoded, groups, le


# ─────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────

def split_data(X, y, groups=None, test_size=0.2, random_state=42):
    """
    Unit I + II: Train/Test split.
    If groups is provided, uses group-aware stratified splitting to prevent
    subject leakage across train and test sets.
    """
    if isinstance(groups, (float, int)):
        test_size = groups
        groups = None

    if groups is not None:
        try:
            n_splits = max(2, int(round(1.0 / test_size)))
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            train_idx, test_idx = next(sgkf.split(X, y, groups))
        except Exception:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
        train_subjs = groups.iloc[train_idx].nunique()
        test_subjs = groups.iloc[test_idx].nunique()
        print(f"\n✂️  Group-Aware Stratified Train/Test Split:")
        print(f"  Train : {X_train.shape[0]} samples ({train_subjs} distinct subjects)")
        print(f"  Test  : {X_test.shape[0]} samples ({test_subjs} distinct subjects)")
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"\n✂️  Stratified Train/Test Split (80/20):")
        print(f"  Train : {X_train.shape[0]} samples")
        print(f"  Test  : {X_test.shape[0]} samples")
        return X_train, X_test, y_train, y_test
