import pickle
import shap
import numpy as np

SHAP_VALUES_PATH = r"C:\Users\somas\Desktop\shap_values.pkl"

# ✅ Load the SHAP values
with open(SHAP_VALUES_PATH, "rb") as f:
    shap_values = pickle.load(f)

# ✅ Convert `shap.Explanation` object to raw values
if isinstance(shap_values, shap.Explanation):
    print("🔄 Converting `shap.Explanation` to list...")
    shap_values = shap_values.values  # Extract actual SHAP values

# ✅ Ensure it's a list
if isinstance(shap_values, np.ndarray):
    shap_values = shap_values.tolist()

# ✅ Re-save in the correct format
with open(SHAP_VALUES_PATH, "wb") as f:
    pickle.dump(shap_values, f)

print("✅ SHAP values re-saved successfully!")
