import pandas as pd
import numpy as np
import re
import statsmodels.formula.api as smf
from scipy import stats
import os

# ==========================================
# 1. Configuration and Paths
# ==========================================
# All paths are set as relative paths for public repository release
PRS_FILE = 'data/prs_score.sscore'
COVA_FILE = 'data/covariates.csv'
PC_FILE = 'data/pca_components.csv'
AGE_FILE = 'data/imaging_age.txt'

# Mediator (Brain)
FMRI_MEDIATOR_FILE = 'data/fmri_mediator.csv'

# Behavioral Data (Outcome)
PATH_BL = 'data/behavioral_bl.csv'
PATH_FU2 = 'data/behavioral_fu2.csv'
PATH_FU3 = 'data/behavioral_fu3.csv'

# Output Directory
OUTPUT_DIR = 'results/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


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
    df = df.iloc[:, [0, -2]] 
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
    
    if 'Handedness' in cova.columns: 
        cova = cova[cova['Handedness'] == 1]
        
    return cova

def load_longitudinal_covariates(age_path):
    print("Loading Longitudinal Covariates...")
    age_df = pd.read_csv(age_path, sep='\t')
    age_df['IID'] = age_df['SubID'].apply(clean_id)
    age_df.rename(columns={'Time': 'Timepoint', 'age': 'Age'}, inplace=True)
    
    return age_df[['IID', 'Timepoint', 'Age']]

# ==========================================
# 3. Core Data Processing Pipeline
# ==========================================

# --- A. Processing Mediator (M: fMRI) ---
print("\n[Step 1] Processing Mediator (fMRI)...")
df_fmri = pd.read_csv(FMRI_MEDIATOR_FILE, index_col=0)
df_fmri.index = df_fmri.index.map(clean_id)

m_cols = [c for c in df_fmri.columns if re.search(r'^(BW|SW|NW)_(BL|FU2|FU3)_shared$', c)]
print(f"Found {len(m_cols)} Mediator columns.:{m_cols}")

m_long = df_fmri[m_cols].copy()
m_long['IID'] = m_long.index
m_long = m_long.melt(id_vars=['IID'], var_name='M_Col', value_name='M_Value')

m_long['Condition'] = m_long['M_Col'].apply(lambda x: re.search(r'^(BW|SW|NW)', x).group(1))
m_long['Timepoint'] = m_long['M_Col'].apply(lambda x: re.search(r'(BL|FU2|FU3)', x).group(1))

# --- B. Processing Outcome (Y: Behavior) ---
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
    if 'W_rt_std' in c 
    and any(cond in c for cond in ['BW', 'SW', 'NW'])
    and not ('left' in c or 'right' in c)
]
print(f"Selected {len(y_cols)} Behavioral columns.", y_cols)
if len(y_cols) == 0:
    raise ValueError("No valid behavioral columns found! Check column names.")

y_long = combined_MID[y_cols].copy()
y_long['IID'] = y_long.index
y_long = y_long.melt(id_vars=['IID'], var_name='Y_Col', value_name='Y_Value')

def safe_extract_meta(col_name, pattern):
    match = re.search(pattern, col_name)
    if match:
        return match.group(1)
    return None

y_long['Timepoint'] = y_long['Y_Col'].apply(lambda x: safe_extract_meta(x, r'(BL|FU2|FU3)'))
y_long['Condition'] = y_long['Y_Col'].apply(lambda x: safe_extract_meta(x, r'(BW|SW|NW)'))
y_long.dropna(subset=['Timepoint', 'Condition'], inplace=True)

# --- C. Master Merge ---
print("\n[Step 3] Merging X, M, and Y...")
df_med = pd.merge(m_long[['IID', 'Timepoint', 'Condition', 'M_Value']], 
                  y_long[['IID', 'Timepoint', 'Condition', 'Y_Value']], 
                  on=['IID', 'Timepoint', 'Condition'], how='inner')

prs_df = load_prs(PRS_FILE)
data = pd.merge(df_med, prs_df, on='IID', how='inner')

static_cova = load_static_covariates(COVA_FILE, PC_FILE)
long_cova = load_longitudinal_covariates(AGE_FILE)

data = pd.merge(data, static_cova, left_on='IID', right_index=True, how='left')
data = pd.merge(data, long_cova, on=['IID', 'Timepoint'], how='left')

data['Timepoint'] = data['Timepoint'].astype('category')
data['Condition'] = data['Condition'].astype('category')
data['Gender_Male'] = data['Gender_Male'].astype('category')
data['ImagingCentreID'] = data['ImagingCentreID'].astype('category')

print(f"✅ Data aligned. N observations: {len(data)}")

# ==========================================
# 4. Standardization
# ==========================================
print("\n[Step 4] Standardizing variables...")
# Hardcoded to include Age and PCs, excluded FD
std_cols = ['M_Value', 'Y_Value', 'PRS', 'Age'] + [f'PC{i}' for i in range(1, 11)]

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

# Hardcoded covariates formulation based on active parameters
covars_list = ["C(Condition)", "C(Timepoint)", "Age", "C(Gender_Male)", "C(ImagingCentreID)"]
covars_list.extend([f"PC{i}" for i in range(1, 11)])

covar_f = " + ".join(covars_list)
print(f"📌 Covariate Formula: {covar_f}")

re_formula = None

# [Helper] Save Summary to file
def save_model_summary(model_res, filename):
    with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
        f.write(model_res.summary().as_text())
    print(f"  -> Saved summary to: {filename}")

try:
    # Hypothesis: X will increase Y through M (X decreases M, M decreases Y). We apply one-tailed tests based on this directional hypothesis.
    # --- Path a: X -> M (PRS -> Brain) ---
    print("  -> Fitting Path a (PRS -> Brain)...")
    model_a = smf.mixedlm(f"M_Value ~ PRS + {covar_f}", data, groups=data["IID"], re_formula=re_formula)
    res_a = model_a.fit(maxiter=200)
    save_model_summary(res_a, "Path_a_Model_Summary.txt")
    
    a_coef = res_a.params['PRS']
    a_p_two_tailed = res_a.pvalues['PRS']
    a_p = a_p_two_tailed / 2
    
    print(f"     Path a Coef: {a_coef:.4f}, P-val (1-tailed): {a_p:.2e}")

    # --- Path b & c': M + X -> Y (Brain + PRS -> Behavior) ---
    print("  -> Fitting Path b (Brain -> Behavior)...")
    model_b = smf.mixedlm(f"Y_Value ~ M_Value + PRS + {covar_f}", data, groups=data["IID"], re_formula=re_formula)
    res_b = model_b.fit(maxiter=200)
    save_model_summary(res_b, "Path_b_Model_Summary.txt")
    
    b_coef = res_b.params['M_Value']
    b_p_two_tailed = res_b.pvalues['M_Value']
    b_p = b_p_two_tailed / 2
    
    c_prime_coef = res_b.params['PRS'] 
    c_prime_p_two_tailed = res_b.pvalues['PRS']
    c_prime_p = c_prime_p_two_tailed / 2
    
    print(f"     Path b Coef: {b_coef:.4f}, P-val (1-tailed): {b_p:.2e}")
    
    # --- Effects Calculation ---
    indirect = a_coef * b_coef
    total = indirect + c_prime_coef
    
    print("\n" + "="*40)
    print("        MEDIATION RESULTS (One-Tailed)")
    print("="*40)
    print(f"Path a  (X -> M): {a_coef:.4f} (p={a_p:.2e})")
    print(f"Path b  (M -> Y): {b_coef:.4f} (p={b_p:.2e})")
    print(f"Path c' (Direct): {c_prime_coef:.4f} (p={c_prime_p:.2e})")
    print("-" * 40)
    print(f"Indirect Effect (a*b): {indirect:.4f}")
    print(f"Total Effect:          {total:.4f}")
    print("="*40)
    

except Exception as e:
    print(f"\n❌ Model Failed: {e}")
    print("Hint: If convergence fails, try removing re_formula and use only Random Intercept.")
    
print("\n" + "="*50)
print("      CALCULATING P-VALUES (ALL ONE-TAILED)")
print("="*50)

try:
    # -------------------------------------------------
    # 1. Direct Effect (c') P-value
    # -------------------------------------------------
    direct_coef = res_b.params['PRS']
    direct_p_two_tailed = res_b.pvalues['PRS']
    direct_p = direct_p_two_tailed / 2

    # -------------------------------------------------
    # 2. Indirect Effect (a*b) P-value (Sobel Test)
    # -------------------------------------------------
    a = res_a.params['PRS']
    b = res_b.params['M_Value']
    se_a = res_a.bse['PRS']
    se_b = res_b.bse['M_Value']

    # Sobel Z
    sobel_z = (a * b) / np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
    indirect_p = 1 - stats.norm.cdf(abs(sobel_z))
    indirect_effect = a * b

    # -------------------------------------------------
    # 3. Total Effect (c) P-value (Model C)
    # -------------------------------------------------
    print("Running Model C (Total Effect: Y ~ PRS)...")
    
    formula_c = f"Y_Value ~ PRS + {covar_f}"
    model_c = smf.mixedlm(formula_c, data, groups=data["IID"], re_formula=re_formula)
    res_c = model_c.fit(maxiter=200)
    save_model_summary(res_c, "Path_c_Total_Effect_Summary.txt")
    
    total_coef = res_c.params['PRS']
    total_p_two_tailed = res_c.pvalues['PRS']
    total_p = total_p_two_tailed / 2

    # -------------------------------------------------
    # 4. Final Output Summary
    # -------------------------------------------------
    print("\n" + "="*60)
    print(f"{'EFFECT TYPE':<20} | {'COEF':<10} | {'P-VAL(1-T)':<10} | {'SIG'}")
    print("-" * 60)
    
    sig_ind = "*" if indirect_p < 0.05 else "ns"
    print(f"{'Indirect (a*b)':<20} | {indirect_effect:<10.4f} | {indirect_p:<10.2e} | {sig_ind}")
    
    sig_dir = "*" if direct_p < 0.05 else "ns"
    print(f"{'Direct (c_)':<20} | {direct_coef:<10.4f} | {direct_p:<10.2e} | {sig_dir}")
    
    sig_tot = "*" if total_p < 0.05 else "ns"
    print(f"{'Total (c)':<20} | {total_coef:<10.4f} | {total_p:<10.2e} | {sig_tot}")
    print("="*60)
    
    print(f"\nDetailed model summaries saved to: {OUTPUT_DIR}")

except NameError:
    print("❌ Error: Missing previous model results. Please ensure you ran Step 5 first.")
except Exception as e:
    print(f"❌ Calculation Error: {e}")
