import numpy as np
from typing import Tuple

def decompose_fmri_signals(contra_l: float, ipsi_l: float, contra_r: float, ipsi_r: float) -> Tuple[float, float]:
    """
    Decomposes 4 mean fMRI signal values into a shared latent signal and a discriminative latent signal using a multiple linear model.
    
    Parameters:
    contra_l : Contralateral signal when left hand is used.
    ipsi_l   : Ipsilateral signal when left hand is used.
    contra_r : Contralateral signal when right hand is used.
    ipsi_r   : Ipsilateral signal when right hand is used.
    
    Returns:
    Tuple containing (shared_signal, discriminative_signal).
    """
    # 1. Define the dependent variable vector Y
    y = np.array([contra_l, ipsi_l, contra_r, ipsi_r])

    # 2. Define the design matrix X based on the methodology
    # Column 0 corresponds to the latent shared signal (always 1)
    # Column 1 corresponds to the latent discriminative signal (1 for contra, -1 for ipsi)
    X = np.array([
        [1,  1],  # Left hand, Contralateral
        [1, -1],  # Left hand, Ipsilateral
        [1,  1],  # Right hand, Contralateral
        [1, -1]   # Right hand, Ipsilateral
    ])

    # 3. Solve the linear system using Ordinary Least Squares (OLS)
    # beta will contain [shared_signal, discriminative_signal]
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    shared_signal = beta[0]
    discriminative_signal = beta[1]

    return shared_signal, discriminative_signal
