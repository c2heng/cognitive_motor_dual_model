import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


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
    Fit PCA with n_components (without standardizing),
    append PC scores to a copy of df_in, and return:
    - df_out with PC columns
    - pca model
    """
    X = df_in[feature_cols].values

    # For 4 features, full SVD is used; PC1..PCk are unique up to sign
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)  # shape: (n_samples, n_components)

    # Append columns PC1..PCk
    df_out = df_in.copy()
    for j in range(n_components):
        df_out[f'{prefix}_PC{j+1}'] = Z[:, j]

    return df_out, pca

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


for phase in ['BL','FU2','FU3']:
    for condition in ['BW','SW','NW']:
        cols = [f'{condition}_{phase}_left_lh', f'{condition}_{phase}_left_rh', f'{condition}_{phase}_right_lh', f'{condition}_{phase}_right_rh']
        df_clean = drop_three_sigma_and_na(df, cols)
        df_pc4, pca4 = fit_pca_and_append(df_clean, cols, n_components=4, prefix='BW_BL')
        print_component_weights(pca4, cols, title='PCA with 4 Components')
