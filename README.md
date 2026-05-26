# A Cognitive-Motor Dual System Model of Bilateral Motor Regulation in Human M1 and Cerebellum

This repository contains the analysis code and brain maps for the manuscript investigating the dual-system architecture of unilateral motor preparation. 

By leveraging longitudinal and cross-sectional fMRI cohorts (IMAGEN and ZIB, N > 6,000), we computationally decomposed bilateral primary motor cortex (M1) and cerebellar signals into two orthogonal mechanisms: 
1. **Cognitive Readiness (Shared Signal):** A hand-invariant, bi-hemispheric signal governing global temporal precision.
2. **Executive Preparation (Discriminative Signal):** A highly lateralized signal controlling effector-specific kinematic execution speed.

## 📁 Repository Structure & Workflow

### 1. fMRI Activation
* `MID_First_Level_SPM.m`: MATLAB script for first-level GLM activation analysis of the Monetary Incentive Delay (MID) task using SPM12 to extract task-related BOLD signals.

### 2. ROI Definition
* `ROIs/`: Contains NIfTI masks for the defined regions of interest, including the left/right primary motor cortex (M1) and the left/right Cerebellum.

### 3. Signal Decomposition
* `Signal_Decomposition_MLR.py`: Core mathematical implementation using Multiple Linear Regression (MLR) to decompose 4-condition M1/Cerebellar signals into shared and discriminative latent signals.
* `Signal_Decomposition_PCA.py`: Model-free validation using Principal Component Analysis (PCA) to extract and verify the identical orthogonal components.

### 4. Voxel Weight (Brain-wide Mapping)
* `Voxel_Weight.py`: Python script performing parallelized voxel-wise OLS regressions to map the whole-brain spatial distributions of the shared and discriminative signals.
* `Voxel_Weight_IMAGEN/` & `Voxel_Weight_ZIB/`: Output directories containing the NIfTI files of voxel-wise statistical maps (betas and p-values) across both cohorts.

### 5. PRS Mediation Analysis
* `PRS_Mediation.py`: Mediation analysis linking Parkinson’s Disease Polygenic Risk Scores (PD-PRS) to behavioral initiation deficits.

### 6. Functional Connectivity
* `MID_FC.m`: MATLAB script for task-based functional connectivity (FC) analysis using the CONN toolbox.
* `FC/`: Output directory containing the condition-specific functional connectivity result maps.

## ⚙️ Dependencies
* **Python 3.x:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `nilearn`, `nibabel`
* **MATLAB:** `SPM12`, `CONN Toolbox`
