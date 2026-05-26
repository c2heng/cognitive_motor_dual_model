import random
import numpy as np
from nilearn import image
import nibabel as nib
import copy
import glob
import pandas as pd
import os
from nilearn import image
import statsmodels.api as sm
import numpy as np
import concurrent.futures

cova = pd.read_csv('/Cova_with_handedness.csv', index_col=0)

right_handed_index=cova[cova['Handedness']==1].index
left_handed_index=cova[cova['Handedness']==0].index
decomp=pd.read_csv('activation.csv',index_col=0)

decomp_bl=decomp.loc[:,[i for i in decomp.columns if 'BL' in i]]
decomp_fu2=decomp.loc[:,[i for i in decomp.columns if 'FU2' in i]]
decomp_fu3=decomp.loc[:,[i for i in decomp.columns if 'FU3' in i]]
decomp_bl=decomp_bl.loc[decomp_bl.index.intersection(right_handed_index)]
decomp_fu2=decomp_fu2.loc[decomp_fu2.index.intersection(right_handed_index)]
decomp_fu3=decomp_fu3.loc[decomp_fu3.index.intersection(right_handed_index)]

def clean_dataframe(df):
    try:
        df_cleaned = df.dropna()
        num_cols = df_cleaned.select_dtypes(include=np.number).columns
        z_scores = np.abs((df_cleaned[num_cols] - df_cleaned[num_cols].mean()) / df_cleaned[num_cols].std())
        df_cleaned = df_cleaned[(z_scores <= 3).all(axis=1)]
        return df_cleaned
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
decomp_bl=clean_dataframe(decomp_bl)
decomp_fu2=clean_dataframe(decomp_fu2)
decomp_fu3=clean_dataframe(decomp_fu3)


decomp_bl.index=decomp_bl.index.astype(str) + '_BL'
decomp_fu2.index=decomp_fu2.index.astype(str) + '_FU2'
decomp_fu3.index=decomp_fu3.index.astype(str) + '_FU3'
decomp_bl['BL']=1
decomp_bl['FU2']=0
decomp_bl['FU3']=0
decomp_fu2['BL']=0
decomp_fu2['FU2']=1
decomp_fu2['FU3']=0
decomp_fu3['BL']=0
decomp_fu3['FU2']=0
decomp_fu3['FU3']=1

decomp_bl.dropna(inplace=True)
decomp_fu2.dropna(inplace=True)
decomp_fu3.dropna(inplace=True)

condition_list=['con_Anti_Hit_RIGHT_BIG_WIN','con_Anti_Hit_LEFT_BIG_WIN','con_Anti_Hit_RIGHT_SMALL_WIN','con_Anti_Hit_LEFT_SMALL_WIN','con_Anti_Hit_RIGHT_NO_WIN','con_Anti_Hit_LEFT_NO_WIN']
phase_list=['baseline','followup2','followup3']


condition='con_Anti_Hit_RIGHT_BIG_WIN'
id_from_nii=glob.glob(f'/ses-baseline/sub-*/{condition}.nii.gz')
id_from_nii_BL=[int(i.split('sub-')[1][:12]) for i in id_from_nii]
id_from_nii=glob.glob(f'/ses-followup2/sub-*/{condition}.nii.gz')
id_from_nii_FU2=[int(i.split('sub-')[1][:12]) for i in id_from_nii]
id_from_nii=glob.glob(f'/ses-followup3/sub-*/{condition}.nii.gz')
id_from_nii_FU3=[int(i.split('sub-')[1][:12]) for i in id_from_nii]


decomp_bl.columns=[i.replace('_BL','') for i in decomp_bl.columns]
decomp_fu2.columns=[i.replace('_FU2','') for i in decomp_fu2.columns]
decomp_fu3.columns=[i.replace('_FU3','') for i in decomp_fu3.columns]

df=pd.concat([decomp_bl,decomp_fu2,decomp_fu3])
df=df[[i for i in df.columns if not '_right_' in i and not '_left_' in i]]



output_dir = "/voxel_weight"
os.makedirs(output_dir, exist_ok=True)

# Example definition for col_pairs
col_pairs = [
    ["BW_discriminative", "BW_shared"],
    ["SW_discriminative", "SW_shared"],
    ["NW_discriminative", "NW_shared"],
]

# Map each type of pair to its corresponding condition list
conditions_map = {
    "BW": ["con_Anti_Hit_RIGHT_BIG_WIN", "con_Anti_Hit_LEFT_BIG_WIN"],
    "SW": ["con_Anti_Hit_RIGHT_SMALL_WIN", "con_Anti_Hit_LEFT_SMALL_WIN"],
    "NW": ["con_Anti_Hit_RIGHT_NO_WIN",   "con_Anti_Hit_LEFT_NO_WIN"],
}

# =========================================================
# Helper function to load and concatenate NIfTI files
# =========================================================
def load_and_combine_niis(nii_paths):
    """
    Loads a list of NIfTI files and concatenates them along the 4th dimension (subjects).
    Returns a 4D NumPy array of shape (X, Y, Z, number_of_subjects) and an example NIfTI header/affine.
    """
    loaded_imgs = []
    for path in nii_paths:
        img = nib.load(path)
        loaded_imgs.append(img.get_fdata(dtype=np.float32))
    
    # Shape: (X, Y, Z, n_subjects)
    combined_data = np.stack(loaded_imgs, axis=-1)
    # Keep one sample image for header/affine info
    example_img = nib.load(nii_paths[0])
    return combined_data, example_img


def process_single_voxel(params):
    """
    A helper function to be run in parallel.
    Takes all necessary inputs to perform OLS for one voxel.
    Returns the beta/p-value results for discriminative and shared.
    """
    x, y, z, Y, df_subset, col_pair = params
    
    # ----------------------------------
    #  1) Drop NaN voxel values
    # ----------------------------------
    nan_mask = np.isnan(Y)
    if nan_mask.sum() / len(Y) > 0.20:
        # Too many NaNs, skip
        return x, y, z, np.nan, np.nan, np.nan, np.nan

    valid_mask = ~nan_mask
    Y_valid = Y[valid_mask]
    df_valid = df_subset.iloc[valid_mask].copy()

    pred_cols = col_pair + [
        "BL", "FU2"
    ]

    # Drop rows with NaNs in predictor columns
    mask_non_nan = ~df_valid[pred_cols].isnull().any(axis=1)
    df_valid = df_valid[mask_non_nan]
    Y_valid = Y_valid[mask_non_nan]

    # If fewer than 2 subjects remain, skip
    if len(df_valid) < 2:
        return x, y, z, np.nan, np.nan, np.nan, np.nan

    # ----------------------------------
    #  2) Add subject-level dummies
    # ----------------------------------
    grouping = df_valid.index.str.split('_').str[0].astype(int).astype(str)
    dummy_df = pd.get_dummies(grouping, prefix='', prefix_sep='', drop_first=True)
    dummy_df = dummy_df.astype(int)
    dummy_df.index = df_valid.index
    df_valid = df_valid.join(dummy_df)
    dummy_cols = list(dummy_df.columns)

    all_predictors = pred_cols + dummy_cols
    X = df_valid[all_predictors]
    X = sm.add_constant(X, prepend=True)

    try:
        # ----------------------------------
        #  3) Fit model
        # ----------------------------------
        model = sm.OLS(Y_valid, X)
        fit_res = model.fit()

        params = fit_res.params
        pvals = fit_res.pvalues

        # Extract desired beta & p-values
        beta_lat = params[col_pair[0]]
        pval_lat = pvals[col_pair[0]]
        beta_sha = params[col_pair[1]]
        pval_sha = pvals[col_pair[1]]

        return x, y, z, beta_lat, pval_lat, beta_sha, pval_sha

    except Exception:
        # Catch any model-fitting errors
        return x, y, z, np.nan, np.nan, np.nan, np.nan
# =========================================================
# Main Script (loops, modeling, saving results)
# =========================================================

# Loop over each pair of columns
for col_pair in col_pairs:
    # Determine which set of conditions to use
    if "BW" in col_pair[0]:
        conditions = conditions_map["BW"]
    elif "SW" in col_pair[0]:
        conditions = conditions_map["SW"]
    elif "NW" in col_pair[0]:
        conditions = conditions_map["NW"]
    else:
        continue  # Skip if none of the known prefixes appear

    # Go through each condition
    for condition in conditions:
        # -------------------------------------------------------
        # Construct paths for baseline, FU2, FU3
        # -------------------------------------------------------

        id_from_nii=glob.glob(f'/ses-baseline/sub-*/{condition}.nii.gz')
        id_from_nii_BL=[int(i.split('sub-')[1][:12]) for i in id_from_nii]
        id_from_nii=glob.glob(f'/ses-followup2/sub-*/{condition}.nii.gz')
        id_from_nii_FU2=[int(i.split('sub-')[1][:12]) for i in id_from_nii]
        id_from_nii=glob.glob(f'/ses-followup3/sub-*/{condition}.nii.gz')
        id_from_nii_FU3=[int(i.split('sub-')[1][:12]) for i in id_from_nii]
        
        # Convert DataFrame indices to IDs without suffix
        bl_ids = [idx.split('_')[0] for idx in decomp_bl.index]
        fu2_ids = [idx.split('_')[0] for idx in decomp_fu2.index]
        fu3_ids = [idx.split('_')[0] for idx in decomp_fu3.index]

        # Convert nii IDs to string
        id_from_nii_BL_str = [str(i) for i in id_from_nii_BL]
        id_from_nii_FU2_str = [str(i) for i in id_from_nii_FU2]
        id_from_nii_FU3_str = [str(i) for i in id_from_nii_FU3]

        # Get intersections for BL, FU2, FU3 respectively
        common_ids_bl = set(id_from_nii_BL_str).intersection(bl_ids)
        common_ids_fu2 = set(id_from_nii_FU2_str).intersection(fu2_ids)
        common_ids_fu3 = set(id_from_nii_FU3_str).intersection(fu3_ids)

        # Convert sets to lists
        common_ids_bl_list_int = list(common_ids_bl)
        common_ids_fu2_list_int = list(common_ids_fu2)
        common_ids_fu3_list_int = list(common_ids_fu3)
        common_ids_bl_list=[i + '_BL' for i in common_ids_bl_list_int]
        common_ids_fu2_list=[i + '_FU2' for i in common_ids_fu2_list_int]
        common_ids_fu3_list=[i + '_FU3' for i in common_ids_fu3_list_int]
        
        
        
        nii_files_bl = [
            f"/ses-baseline/sub-"
            + (12 - len(str(i))) * "0" + str(i) + f"/{condition}.nii.gz"
            for i in common_ids_bl_list_int
        ]
        nii_files_fu2 = [
            f"/ses-followup2/sub-"
            + (12 - len(str(i))) * "0" + str(i) + f"/{condition}.nii.gz"
            for i in common_ids_fu2_list_int
        ]
        nii_files_fu3 = [
            f"/ses-followup3/sub-"
            + (12 - len(str(i))) * "0" + str(i) + f"/{condition}.nii.gz"
            for i in common_ids_fu3_list_int
        ]

        # Combine all NIfTI file paths (keeping order: BL -> FU2 -> FU3)
        combined_nii_list = nii_files_bl + nii_files_fu2 + nii_files_fu3

        # Subset the DataFrame accordingly (order matches the NIfTI list)
        # The IDs in df should match the order in the combined lists
        combined_ids = common_ids_bl_list + common_ids_fu2_list + common_ids_fu3_list
        df_subset = df.loc[combined_ids].copy()

        # Load the 4D data array for all subjects
        combined_data, example_img = load_and_combine_niis(combined_nii_list)
        x_dim, y_dim, z_dim, n_subj = combined_data.shape

        for x_slice in range(61):

            flag_path=f'/home1/chenzheng/Lateralization/Motor/FINAL/voxel_weight_mixed_model/{condition}_{x_slice}_flag'
            if os.path.exists(flag_path):
                continue
            with open(flag_path, 'w') as file:
                print(condition,x_slice)
                pass  # Do nothing, just create the file
            # Prepare arrays to store final results
            beta_discriminative = np.full((x_dim, y_dim, z_dim), np.nan, dtype=np.float32)
            pval_discriminative = np.full((x_dim, y_dim, z_dim), np.nan, dtype=np.float32)
            beta_shared      = np.full((x_dim, y_dim, z_dim), np.nan, dtype=np.float32)
            pval_shared      = np.full((x_dim, y_dim, z_dim), np.nan, dtype=np.float32)

            # Build argument list for each voxel
            tasks = []
            for x in [x_slice]:
                for y in range(y_dim):
                    for z in range(z_dim):
                        Y = combined_data[x, y, z, :]
                        tasks.append((x, y, z, Y, df_subset, col_pair))
            # Track total tasks for progress
            total_tasks = len(tasks)
            completed_tasks = 0

            # Now run in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(process_single_voxel, t): t for t in tasks}
                
                for fut in concurrent.futures.as_completed(futures):
                    x, y, z, b_lat, p_lat, b_sha, p_sha = fut.result()
                    
                    beta_discriminative[x, y, z] = b_lat
                    pval_discriminative[x, y, z] = p_lat
                    beta_shared[x, y, z]      = b_sha
                    pval_shared[x, y, z]      = p_sha
            
            # -------------------------------------------------------
            # Save the results as NIfTI
            # -------------------------------------------------------
            
            out_beta_lat_img = nib.Nifti1Image(beta_discriminative, example_img.affine, example_img.header)
            nib.save(out_beta_lat_img, os.path.join(output_dir, f"original_{condition}_beta_{col_pair[0]}_X_{x_slice}.nii.gz"))
            
            out_pval_lat_img = nib.Nifti1Image(pval_discriminative, example_img.affine, example_img.header)
            nib.save(out_pval_lat_img, os.path.join(output_dir, f"original_{condition}_pval_{col_pair[0]}_X_{x_slice}.nii.gz"))
            
            out_beta_sha_img = nib.Nifti1Image(beta_shared, example_img.affine, example_img.header)
            nib.save(out_beta_sha_img, os.path.join(output_dir, f"original_{condition}_beta_{col_pair[1]}_X_{x_slice}.nii.gz"))
            
            out_pval_sha_img = nib.Nifti1Image(pval_shared, example_img.affine, example_img.header)
            nib.save(out_pval_sha_img, os.path.join(output_dir, f"original_{condition}_pval_{col_pair[1]}_X_{x_slice}.nii.gz"))

            
            print(f'--- {condition}_X_{x_slice} done ---')

print("Voxelwise analysis completed.")