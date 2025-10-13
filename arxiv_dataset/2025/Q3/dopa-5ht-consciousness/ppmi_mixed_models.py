import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm

# --- Block 0: Set-up ---
print("--- [Block 0] Loading Analysis-Ready Data ---")
df = pd.read_parquet(r"C:\Users\Diogo Sousa\Downloads\Consicous\Final\ppmi_rem_ready.parquet")
print(f"Successfully loaded data with shape: {df.shape}")

# --- Block 1: Data Quality & Final Filtering ---
print("\n--- [Block 1] Final Data Quality and Filtering ---")

# Create the clean analysis subset (dfc) for modeling
# We now know questionnaires are empty, so we focus on the core variables
model_vars = ['rem_min', 'LEDD', 'age', 'sex', 'UPDRS_III', 'patno']
existing_model_vars = [col for col in model_vars if col in df.columns]
dfc = df.dropna(subset=existing_model_vars).copy()

# Add a binary flag for being on medication or not
dfc['on_meds'] = (dfc['LEDD'] > 0).astype(int)

print(f"Created clean modeling subset ('dfc') with {len(dfc)} nights from {dfc.patno.nunique()} participants.")
print(f"\nBreakdown of nights by medication status:\n{dfc['on_meds'].value_counts()}")

# --- Block 3 (Revised): FITTING ROBUST MODELS ---
print("\n--- [Block 3] Fitting Simplified Mixed-Effects Models ---")

# Standardize continuous predictors for better model convergence
for col in ['LEDD', 'age', 'UPDRS_III']:
    if col in dfc.columns:
        dfc[f'{col}_z'] = (dfc[col] - dfc[col].mean()) / dfc[col].std()

# --- MODEL 1: THE "DISEASE SEVERITY" MODEL ---
# How does REM duration relate to the underlying disease process?
print("\n--- MODEL 1: The Effect of Disease Severity on REM Sleep ---")
model1_formula = "rem_min ~ UPDRS_III_z + age_z + sex"
try:
    model1 = mixedlm(model1_formula, data=dfc, groups=dfc["patno"]).fit(method='powell')
    print(model1.summary())

    # Interpretation for Model 1
    updrs_pval = model1.pvalues['UPDRS_III_z']
    print("\n--- Interpretation of Model 1 ---")
    if updrs_pval < 0.05:
        print(
            f"RESULT: Disease severity (UPDRS_III) has a STATISTICALLY SIGNIFICANT effect on REM duration (p={updrs_pval:.4f}).")
        print("This confirms the trend seen in your box plot and strongly supports your core theory.")
    else:
        print("RESULT: No significant effect of disease severity was found when controlling for other factors.")

except np.linalg.LinAlgError as e:
    print(f"ERROR fitting Model 1: {e}. The variables might still be too correlated.")

# --- MODEL 2: THE "MEDICATION" MODEL ---
# For patients taking medication, does the dose matter?
print("\n--- MODEL 2: The Effect of Medication (LEDD) on REM Sleep ---")
# We filter the dataset to only include nights where the patient was on medication
df_meds_only = dfc.query("on_meds == 1").copy()
print(f"Analyzing {len(df_meds_only)} nights from patients on medication.")

model2_formula = "rem_min ~ LEDD_z + age_z + sex"
try:
    model2 = mixedlm(model2_formula, data=df_meds_only, groups=df_meds_only["patno"]).fit(method='powell')
    print(model2.summary())

    # Interpretation for Model 2
    ledd_pval = model2.pvalues['LEDD_z']
    print("\n--- Interpretation of Model 2 ---")
    if ledd_pval < 0.05:
        print(
            f"RESULT: In patients on medication, LEDD has a STATISTICALLY SIGNIFICANT effect on REM duration (p={ledd_pval:.4f}).")
        print("This directly tests the 'medication' part of your hypothesis.")
    else:
        print(
            "RESULT: For patients already on medication, the specific dose (LEDD) does not seem to significantly affect REM duration.")
        print("This might support your 'receptor binding' theory - just giving more drug isn't the whole story.")

except np.linalg.LinAlgError as e:
    print(f"ERROR fitting Model 2: {e}.")
except ValueError as e:
    print(f"ERROR: Not enough data to fit Model 2. {e}")

# --- MODEL 3 (Advanced): TESTING YOUR RECEPTOR BINDING THEORY ---
print("\n--- MODEL 3 (Advanced): Testing the Interaction of Disease and Medication ---")
# This directly tests: "Does the effect of LEDD depend on the severity of the disease?"
model3_formula = "rem_min ~ LEDD_z * UPDRS_III_z + age_z + sex"
try:
    model3 = mixedlm(model3_formula, data=df_meds_only, groups=df_meds_only["patno"]).fit(method='powell')
    print(model3.summary())

    # Interpretation of the interaction term
    interaction_pval = model3.pvalues['LEDD_z:UPDRS_III_z']
    print("\n--- Interpretation of Model 3 Interaction ---")
    if interaction_pval < 0.05:
        print(f"*** SIGNIFICANT INTERACTION FOUND! (p={interaction_pval:.4f}) ***")
        print(
            "This is strong evidence for your theory! It means the effect of L-DOPA on REM sleep is different for patients with mild vs. severe disease.")
        print(
            "For example, it might help in early stages but have little effect in later stages when terminals/receptors are gone.")
    else:
        print(
            "RESULT: No significant interaction was found. The effect of LEDD does not appear to depend on disease severity.")

except Exception as e:
    print(f"ERROR fitting Model 3: {e}.")
