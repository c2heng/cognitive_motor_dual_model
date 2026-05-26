import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# --------------------------------------------
# Helpers
# --------------------------------------------
def drop_three_sigma_and_na(df: pd.DataFrame, cols_to_check):
    """
    Remove rows that have any 3-sigma outlier in the selected columns.
    Then drop rows with NaN in those columns.
    """
    df_sel = df.copy()
    keep_mask = pd.Series(True, index=df_sel.index)

    for c in cols_to_check:
        s = df_sel[c]
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True)
        if pd.isna(sd) or sd == 0:
            col_mask = s.notna()
        else:
            lower, upper = mu - 3*sd, mu + 3*sd
            col_mask = s.between(lower, upper) | s.isna()
        keep_mask &= col_mask

    df_filt = df_sel.loc[keep_mask].copy()
    df_filt = df_filt.dropna(subset=cols_to_check)
    return df_filt

def fit_pca_and_append(df_in: pd.DataFrame, feature_cols, n_components: int, prefix: str):
    """
    Standardize feature_cols, fit PCA with n_components,
    append PC scores to a copy of df_in, and return:
    - df_out with PC columns
    - pca model
    - scaler
    """
    X = df_in[feature_cols].values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    # For 4 features, full SVD is used; PC1..PCk are unique up to sign
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(Xz)  # shape: (n_samples, n_components)

    # Append columns PC1..PCk
    df_out = df_in.copy()
    for j in range(n_components):
        df_out[f'{prefix}_PC{j+1}'] = Z[:, j]

    return df_out, pca, scaler

def print_component_weights(pca: PCA, feature_cols, title: str):
    """
    Print component weights (unit-length eigenvectors) and explained variance ratios.
    Rows = PCs, Columns = features.
    """
    loadings = pd.DataFrame(pca.components_,
                            index=[f'PC{i+1}' for i in range(pca.n_components_)],
                            columns=feature_cols)
    evr = pd.Series(pca.explained_variance_ratio_,
                    index=loadings.index,
                    name='ExplainedVarianceRatio')
    # print(f'\n=== {title} ===')
    print('Component weights (loadings):')
    display(loadings)  # Jupyter-friendly; falls back to print if not available
    print('\nExplained variance ratio per PC:')
    display(evr)


df=pd.read_csv('MID_First_Level.csv',index_col=0)
cova = pd.read_csv('Cova_with_handedness.csv', index_col=0)
cova["Handedness"]
df=df.loc[[ i for i in cova.index[cova["Handedness"] == 1] if  i in df.index]]


for phase in ['BL','FU2','FU3']:
    for condition in ['BW','SW','NW']:
    
        cols = [f'{condition}_{phase}_left_lh', f'{condition}_{phase}_left_rh', f'{condition}_{phase}_right_lh', f'{condition}_{phase}_right_rh']


        # --------------------------------------------
        # Main flow (assumes you already have a DataFrame named `df`)
        # --------------------------------------------

        # print(cols)
        # 1) Clean by 3-sigma rule and drop NaNs
        df_clean = drop_three_sigma_and_na(df, cols)

        # 2) PCA with 2 components → new DataFrame with PC1, PC2
        df_pc2, pca2, scaler2 = fit_pca_and_append(df_clean, cols, n_components=2, prefix='BW_BL')
        # print_component_weights(pca2, cols, title='PCA with 2 Components')

        # 3) PCA with 4 components → new DataFrame with PC1..PC4
        df_pc4, pca4, scaler4 = fit_pca_and_append(df_clean, cols, n_components=4, prefix='BW_BL')
        print_component_weights(pca4, cols, title='PCA with 4 Components')
