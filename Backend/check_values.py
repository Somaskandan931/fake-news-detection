import pickle
import numpy as np
import shap

SHAP_VALUES_PATH = r"C:\Users\somas\Desktop\shap_values.pkl"

# ✅ Load SHAP values
with open(SHAP_VALUES_PATH, "rb") as f:
    shap_values = pickle.load(f)

# 🔍 Print the type and first element
print("📌 SHAP values type:", type(shap_values))
if isinstance(shap_values, list):
    print("✅ SHAP values are a list.")
    print("📌 First element type:", type(shap_values[0]))
elif isinstance(shap_values, np.ndarray):
    print("🔄 SHAP values are a NumPy array. Convert them to a list.")
elif isinstance(shap_values, dict):
    print("📌 SHAP values are stored as a dictionary. Extracting actual values.")
    print("📌 Keys:", shap_values.keys())
elif isinstance(shap_values, shap.Explanation):
    print("📌 SHAP values are stored as a `shap.Explanation` object.")
else:
    print("❌ Unexpected SHAP values format:", type(shap_values))
