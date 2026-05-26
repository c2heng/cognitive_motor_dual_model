import pandas as pd
import numpy as np
import re
import statsmodels.formula.api as smf
from scipy import stats
import os

# ==========================================
# 1. Configuration and Paths
# ==========================================
PRS_FILE = 'PD_PRS.sscore'
COVA_FILE = 'Cova_with_handedness.csv'
PC_FILE = 'imagen_impute_clean_pca_xic.eigenvec.csv'
AGE_FILE = 'IMAGEN_imaging_age.txt'

# Mediator Variable (Brain)
FMRI_MEDIATOR_FILE = 'Activation_Decomposition.csv'

# Behavioral Data (Outcome)
PATH_BL = 'BL_MID.csv'
PATH_FU2 = 'FU2_MID.csv'
PATH_FU3 = 'FU3_MID.csv'

FD_FILES = {
    'BL': 'MID_ses-baseline_mean_FD.txt',
    'FU2': 'MID_ses-followup2_mean_FD.txt',
    'FU3': 'MID_ses-followup3_mean_FD.txt'
}

# ==========================================
# 2. Basic Loading Functions
# ==========================================
def clean_id(x):
    try:
        x_str = str(x).strip()
        if 'sub-' in x_str: return int(x_str.replace('sub-', '').lstrip('0'))
        return int(float(x_str))
    except: return np.nan

def load_prs(filepath):
    print(f"Loading PRS from: {filepath}")
    df = pd.read_csv(filepath, sep='\s+')
    df = df.iloc[:, [0, -1]] # The second to last column is the score
    df.columns = ['IID', 'PRS']
    df['IID'] = df['IID'].apply(clean_id)
    df.dropna(inplace=True)
    df = df[(np.abs(stats.zscore(df['PRS'])) < 3)]
    return df

def load_static_covariates(cova_path, pc_path):
    print("Loading Static Covariates...")
    cova = pd.read_csv(cova_path, index_col=0)
    cova.index = cova.index.map(clean_id)
    pc = pd.read_csv(pc_path, index_col=0)
    pc.index = pc.index.map(clean_id)
    
    pc_cols = [f'PC{i}' for i in range(1, 11)]
    cova = cova.join(pc[pc_cols], how='left')
    
    if 'Handedness' in cova.columns: cova = cova[cova['Handedness'] == 1]
    if 'ImagingCentreID' in cova.columns:
        cova['ImagingCentreID'] = cova['ImagingCentreID'].fillna(cova['ImagingCentreID'].mode()[0]).astype(int)
    if 'Gender_Male' in cova.columns:
        cova['Gender_Male'] = cova['Gender_Male'].fillna(cova['Gender_Male'].mode()[0])
    return cova

def load_longitudinal_covariates(age_path, fd_files_dict):
    print("Loading Longitudinal Covariates...")
    age_df = pd.read_csv(age_path, sep='\t')
    age_df['IID'] = age_df['SubID'].apply(clean_id)
    age_df.rename(columns={'Time': 'Timepoint', 'age': 'Age'}, inplace=True)
    
    fd_dfs = []
    for tp, fpath in fd_files_dict.items():
        temp_fd = pd.read_csv(fpath, header=None, names=['SubID_Str', 'FD'])
        temp_fd['IID'] = temp_fd['SubID_Str'].apply(clean_id)
        temp_fd['Timepoint'] = tp
        fd_dfs.append(temp_fd[['IID', 'Timepoint', 'FD']])
        
    fd_long = pd.concat(fd_dfs, ignore_index=True)
    return pd.merge(age_df[['IID', 'Timepoint', 'Age']], fd_long, on=['IID', 'Timepoint'], how='outer')

def safe_extract_meta(col_name, pattern):
    match = re.search(pattern, col_name)
    if match:
        return match.group(1)
    return None

def save_model_summary(model_res, filename, output_dir):
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(model_res.summary().as_text())
    print(f"  -> Saved summary to: {filename}")


# ==========================================
# 3. Core Data Processing Workflow
# ==========================================
analysis_modes = ['shared', 'lateralized']

for mode in analysis_modes:
    print(f"\n{'='*60}")
    print(f"  STARTING ANALYSIS FOR: {mode.upper()}")
    print(f"{'='*60}")

    # Output Directory
    OUTPUT_DIR = f'formal_mediation_results/{mode}_final/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- A. Process Mediator (M: fMRI) ---
    print("\n[Step 1] Processing Mediator (fMRI)...")
    df_fmri = pd.read_csv(FMRI_MEDIATOR_FILE, index_col=0)
    df_fmri.index = df_fmri.index.map(clean_id)

    m_cols = [c for c in df_fmri.columns if re.search(rf'^(BW|SW|NW)_(BL|FU2|FU3)_{mode}$', c)]
    print(f"Found {len(m_cols)} Mediator columns.:{m_cols}")

    m_long = df_fmri[m_cols].copy()
    m_long['IID'] = m_long.index
    m_long = m_long.melt(id_vars=['IID'], var_name='M_Col', value_name='M_Value')

    m_long['Condition'] = m_long['M_Col'].apply(lambda x: re.search(r'^(BW|SW|NW)', x).group(1))
    m_long['Timepoint'] = m_long['M_Col'].apply(lambda x: re.search(r'(BL|FU2|FU3)', x).group(1))

    # --- B. Process Outcome (Y: Behavior) ---
    print("\n[Step 2] Processing Outcome (Behavior)...")
    dfs = [pd.read_csv(PATH_BL, index_col=0), pd.read_csv(PATH_FU2, index_col=0), pd.read_csv(PATH_FU3, index_col=0)]
    prefixes = ['BL_', 'FU2_', 'FU3_']
    combined_MID = pd.concat([df.add_prefix(pfx) for df, pfx in zip(dfs, prefixes)], axis=1, join='outer')
    combined_MID.index = combined_MID.index.map(clean_id)

    mapping = {'big_win': 'BW', 'small_win': 'SW', 'no_win': 'NW'}
    for old, new in mapping.items():
        combined_MID.columns = combined_MID.columns.str.replace(old, new, regex=False)

    y_cols = [
        c for c in combined_MID.columns 
        if 'success_rate' in c 
        and any(cond in c for cond in ['BW', 'SW', 'NW'])
        and not ('left' in c or 'right' in c)
    ]
    print(f"Selected {len(y_cols)} Behavioral columns.")
    if len(y_cols) == 0:
        raise ValueError("No valid behavioral columns found! Check column names.")

    y_long = combined_MID[y_cols].copy()
    y_long['IID'] = y_long.index
    y_long = y_long.melt(id_vars=['IID'], var_name='Y_Col', value_name='Y_Value')

    y_long['Timepoint'] = y_long['Y_Col'].apply(lambda x: safe_extract_meta(x, r'(BL|FU2|FU3)'))
    y_long['Condition'] = y_long['Y_Col'].apply(lambda x: safe_extract_meta(x, r'(BW|SW|NW)'))
    y_long.dropna(subset=['Timepoint', 'Condition'], inplace=True)

    # --- C. Merge All Data (Master Merge) ---
    print("\n[Step 3] Merging X, M, and Y...")
    df_med = pd.merge(m_long[['IID', 'Timepoint', 'Condition', 'M_Value']], 
                      y_long[['IID', 'Timepoint', 'Condition', 'Y_Value']], 
                      on=['IID', 'Timepoint', 'Condition'], how='inner')

    prs_df = load_prs(PRS_FILE)
    data = pd.merge(df_med, prs_df, on='IID', how='inner')

    static_cova = load_static_covariates(COVA_FILE, PC_FILE)
    long_cova = load_longitudinal_covariates(AGE_FILE, FD_FILES)

    data = pd.merge(data, static_cova, left_on='IID', right_index=True, how='left')
    data = pd.merge(data, long_cova, on=['IID', 'Timepoint'], how='left')

    data['Timepoint'] = data['Timepoint'].astype('category')
    data['Condition'] = data['Condition'].astype('category')
    data['Gender_Male'] = data['Gender_Male'].astype('category')
    data['ImagingCentreID'] = data['ImagingCentreID'].astype('category')

    print(f"✅ Data aligned. N observations: {len(data)}")

    # ==========================================
    # 4. Standardization - Required Step
    # ==========================================
    print("\n[Step 4] Standardizing variables...")
    std_cols = ['M_Value', 'Y_Value', 'PRS', 'Age', 'FD'] + [f'PC{i}' for i in range(1, 11)]

    for col in std_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val > 0:
                data[col] = (data[col] - mean_val) / std_val

    data.dropna(subset=std_cols, inplace=True)
    print(f"✅ Standardization complete. Final N for modeling: {len(data)}")

    # ==========================================
    # 5. Multilevel Mediation Analysis
    # ==========================================
    print("\n[Step 5] Running Mediation Models (One-Tailed)...")

    # Full covariates formula
    covar_f = "C(Condition) + C(Timepoint) + Age + FD + C(Gender_Male) + C(ImagingCentreID) + " + " + ".join([f"PC{i}" for i in range(1, 11)])

    re_formula = None

    try:
        # Hypothesis: X will decrease Y through M (X decreases M, M increases Y). We apply one-tailed tests based on this directional hypothesis.
        # --- Path a: X -> M (PRS -> Brain) ---
        print("  -> Fitting Path a (PRS -> Brain)...")
        model_a = smf.mixedlm(f"M_Value ~ PRS + {covar_f}", data, groups=data["IID"], re_formula=re_formula)
        res_a = model_a.fit(maxiter=200)
        save_model_summary(res_a, "Path_a_Model_Summary.txt", OUTPUT_DIR)
        
        a_coef = res_a.params['PRS']
        a_p_two_tailed = res_a.pvalues['PRS']
        a_p = a_p_two_tailed / 2
        
        print(f"     Path a Coef: {a_coef:.4f}, P-val (1-tailed): {a_p:.2e}")

        # --- Path b & c': M + X -> Y (Brain + PRS -> Behavior) ---
        print("  -> Fitting Path b (Brain -> Behavior)...")
        model_b = smf.mixedlm(f"Y_Value ~ M_Value + PRS + {covar_f}", data, groups=data["IID"], re_formula=re_formula)
        res_b = model_b.fit(maxiter=200)
        save_model_summary(res_b, "Path_b_Model_Summary.txt", OUTPUT_DIR)
        
        b_coef = res_b.params['M_Value']
        b_p_two_tailed = res_b.pvalues['M_Value']
        b_p = b_p_two_tailed / 2
        
        c_prime_coef = res_b.params['PRS'] 
        c_prime_p_two_tailed = res_b.pvalues['PRS']
        c_prime_p = c_prime_p_two_tailed / 2
        
        print(f"     Path b Coef: {b_coef:.4f}, P-val (1-tailed): {b_p:.2e}")
        
        # --- Calculate Results ---
        indirect = a_coef * b_coef
        total = indirect + c_prime_coef
        
        print("\n" + "="*40)
        print("       MEDIATION RESULTS (One-Tailed)")
        print("="*40)
        print(f"Path a  (X -> M): {a_coef:.4f} (p={a_p:.2e})")
        print(f"Path b  (M -> Y): {b_coef:.4f} (p={b_p:.2e})")
        print(f"Path c' (Direct): {c_prime_coef:.4f} (p={c_prime_p:.2e})")
        print("-" * 40)
        print(f"Indirect Effect (a*b): {indirect:.4f}")
        print(f"Total Effect:          {total:.4f}")
        print("="*40)
        
        if a_p < 0.05 and b_p < 0.05:
            print("✨ SIGNIFICANT MEDIATION DETECTED! (One-Tailed) ✨")
        else:
            print("⚠️ Mediation is not significant.")

        print("\n" + "="*50)
        print("      CALCULATING P-VALUES (ALL ONE-TAILED)")
        print("="*50)

        # -------------------------------------------------
        # 1. Direct Effect (c') P-value
        # -------------------------------------------------
        direct_coef = res_b.params['PRS']
        direct_p_two_tailed = res_b.pvalues['PRS']
        direct_p = direct_p_two_tailed / 2

        # -------------------------------------------------
        # 2. Indirect Effect (a*b)
        # (Sobel Test has been removed per user request)
        # -------------------------------------------------
        indirect_effect = a_coef * b_coef

        # -------------------------------------------------
        # 3. Total Effect (c) P-value (Model C)
        # -------------------------------------------------
        print("Running Model C (Total Effect: Y ~ PRS)...")
        
        formula_c = f"Y_Value ~ PRS + {covar_f}"
        model_c = smf.mixedlm(formula_c, data, groups=data["IID"], re_formula=re_formula)
        res_c = model_c.fit(maxiter=200)
        save_model_summary(res_c, "Path_c_Total_Effect_Summary.txt", OUTPUT_DIR)
        
        total_coef = res_c.params['PRS']
        total_p_two_tailed = res_c.pvalues['PRS']
        total_p = total_p_two_tailed / 2

        # -------------------------------------------------
        # 4. Final Summary Output
        # -------------------------------------------------
        print("\n" + "="*60)
        print(f"{'EFFECT TYPE':<20} | {'COEF':<10} | {'P-VAL(1-T)':<10} | {'SIG'}")
        print("-" * 60)
        
        # Indirect effect (no p-value reported since Sobel test is removed)
        print(f"{'Indirect (a*b)':<20} | {indirect_effect:<10.4f} | {'-':<10} | {'-'}")
        
        # Direct effect
        sig_dir = "*" if direct_p < 0.05 else "ns"
        print(f"{'Direct (c_)':<20} | {direct_coef:<10.4f} | {direct_p:<10.2e} | {sig_dir}")
        
        # Total effect
        sig_tot = "*" if total_p < 0.05 else "ns"
        print(f"{'Total (c)':<20} | {total_coef:<10.4f} | {total_p:<10.2e} | {sig_tot}")
        print("="*60)
        
        print(f"\nDetailed model summaries saved to: {OUTPUT_DIR}")

        # =========================================================
        # Extract all results and generate tables
        # =========================================================
        print("\n[Step 6] Generating Tables...")
        table_rows = []
        
        all_vars = res_b.params.index.tolist()
        
        for var in all_vars:
            row = {'Predictor / Covariate': var}
            
            # --- Path A ---
            if var in res_a.params:
                row['Path a Coef (SE)'] = f"{res_a.params[var]:.3f} ({res_a.bse[var]:.3f})"
                p_val = a_p if var == 'PRS' else res_a.pvalues[var]
                row['Path a P-value'] = f"{p_val:.2e}"
            else:
                row['Path a Coef (SE)'] = "-"
                row['Path a P-value'] = "-"
                
            # --- Path B & C' ---
            if var in res_b.params:
                row["Path b & c' Coef (SE)"] = f"{res_b.params[var]:.3f} ({res_b.bse[var]:.3f})"
                if var == 'M_Value':
                    p_val = b_p
                elif var == 'PRS':
                    p_val = c_prime_p
                else:
                    p_val = res_b.pvalues[var]
                row["Path b & c' P-value"] = f"{p_val:.2e}"
            else:
                row["Path b & c' Coef (SE)"] = "-"
                row["Path b & c' P-value"] = "-"
                
            # --- Total Effect ---
            if var in res_c.params:
                row['Total Effect Coef (SE)'] = f"{res_c.params[var]:.3f} ({res_c.bse[var]:.3f})"
                p_val = total_p if var == 'PRS' else res_c.pvalues[var]
                row['Total Effect P-value'] = f"{p_val:.2e}"
            else:
                row['Total Effect Coef (SE)'] = "-"
                row['Total Effect P-value'] = "-"
                
            table_rows.append(row)
            
        row_re = {'Predictor / Covariate': 'Random Effect (Group Var)'}
        row_re['Path a Coef (SE)'] = f"{res_a.cov_re.iloc[0,0]:.3f}" if hasattr(res_a, 'cov_re') else "-"
        row_re['Path a P-value'] = "-"
        row_re["Path b & c' Coef (SE)"] = f"{res_b.cov_re.iloc[0,0]:.3f}" if hasattr(res_b, 'cov_re') else "-"
        row_re["Path b & c' P-value"] = "-"
        row_re['Total Effect Coef (SE)'] = f"{res_c.cov_re.iloc[0,0]:.3f}" if hasattr(res_c, 'cov_re') else "-"
        row_re['Total Effect P-value'] = "-"
        table_rows.append(row_re)
        
        row_ll = {'Predictor / Covariate': 'Log-Likelihood'}
        row_ll['Path a Coef (SE)'] = f"{res_a.llf:.2f}"
        row_ll['Path a P-value'] = "-"
        row_ll["Path b & c' Coef (SE)"] = f"{res_b.llf:.2f}"
        row_ll["Path b & c' P-value"] = "-"
        row_ll['Total Effect Coef (SE)'] = f"{res_c.llf:.2f}"
        row_ll['Total Effect P-value'] = "-"
        table_rows.append(row_ll)

        row_obs = {'Predictor / Covariate': 'Observations'}
        row_obs['Path a Coef (SE)'] = f"{int(res_a.nobs)}"
        row_obs['Path a P-value'] = "-"
        row_obs["Path b & c' Coef (SE)"] = f"{int(res_b.nobs)}"
        row_obs["Path b & c' P-value"] = "-"
        row_obs['Total Effect Coef (SE)'] = f"{int(res_c.nobs)}"
        row_obs['Total Effect P-value'] = "-"
        table_rows.append(row_obs)

        df_out = pd.DataFrame(table_rows)
        csv_filename = f'Supplementary_Table_{mode.capitalize()}.csv'
        csv_path = os.path.join(OUTPUT_DIR, csv_filename)
        df_out.to_csv(csv_path, index=False)
        print(f"✅ Consolidated Table saved to: {csv_path}")

    except Exception as e:
        print(f"\n❌ Model/Calculation Error for {mode}: {e}")
        print("Hint: If convergence fails, try removing re_formula and use only Random Intercept.")