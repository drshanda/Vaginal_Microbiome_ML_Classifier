import pickle
import pandas as pd
import shap

# Load trained model
with open("final_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Extract feature names from the model itself
feature_names = model.feature_names_in_

def predict():
    # Create a dummy test row with all required features
    X = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)

    prediction = model.predict(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return prediction, shap_values

if __name__ == "__main__":
    pred, shap_vals = predict()
    print("Prediction:", pred)
    print("SHAP values:", shap_vals)