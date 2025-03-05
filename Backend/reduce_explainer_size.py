import dill as pickle
import shap

EXPLAINER_PATH = r"C:\Users\somas\Desktop\shap_explainer.pkl"
REDUCED_EXPLAINER_PATH = r"C:\Users\somas\Desktop\shap_explainer_reduced.pkl"

# ✅ Load SHAP explainer
with open(EXPLAINER_PATH, "rb") as f:
    explainer = pickle.load(f)

# ✅ Remove unnecessary attributes (metadata)
explainer.model = None  # Remove model reference
explainer.expected_value = None  # Remove precomputed expectations

# ✅ Save reduced SHAP explainer
with open(REDUCED_EXPLAINER_PATH, "wb") as f:
    pickle.dump(explainer, f)

print("✅ Reduced SHAP explainer saved!")
