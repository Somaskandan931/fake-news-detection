import pickle
import numpy as np
import bz2  # ✅ Use compression

SHAP_VALUES_PATH = r"C:\Users\somas\Desktop\shap_values.pkl"
COMPRESSED_SHAP_VALUES_PATH = r"C:\Users\somas\Desktop\shap_values_compressed.pbz2"

# ✅ Load SHAP values
with open(SHAP_VALUES_PATH, "rb") as f:
    shap_values = pickle.load(f)

# ✅ Convert to float16
if isinstance(shap_values, np.ndarray):
    shap_values = shap_values.astype(np.float16)

# ✅ Save with compression
with bz2.BZ2File(COMPRESSED_SHAP_VALUES_PATH, "wb") as f:
    pickle.dump(shap_values, f)

print("✅ Compressed SHAP values saved!")
