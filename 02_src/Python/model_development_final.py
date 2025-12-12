import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import shap
import subprocess, sys, time, webbrowser


import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

%matplotlib inline

TARGET_COLUMN = 'Status_Code'
RANDOM_STATE = 42
N_ESTIMATORS = 500
MAX_DEPTH = 10 
ALPHA_COLUMNS = ['Shannon_Index', 'Observed_Richness'] 

# File Paths
DATA_PATH = "01_data/processed/final_ml_feature_matrix.csv" 
SAVE_DIR_TAB = "03_results/tables"
SAVE_DIR_FIG = "03_results/figures/"
DEPLOYMENT_DIR = "04_app_deployment" 
MODEL_PATH = os.path.join(DEPLOYMENT_DIR, "final_rf_model.pkl") 
EXPERIMENT_NAME = "Metagenome_Classifier_Comparison"
MLFLOW_TRACKING_DIR = "mlruns"

# SHAP plot settings
TOP_FEATURE = 'CLR_1' 
N_SHAP_FEATURES = 10

df = pd.read_csv(DATA_PATH, index_col=0)
clr_columns = [col for col in df.columns if col.startswith('CLR_')]
X = df[clr_columns + ALPHA_COLUMNS]
y_raw = df[TARGET_COLUMN]

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_labels = le.classes_ 

# Split and Reset Index (CRITICAL for SHAP/MLflow alignment)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train_reset = X_train.reset_index(drop=True)
X_test_reset = X_test.reset_index(drop=True)
y_train_reset = pd.Series(y_train).reset_index(drop=True)
y_test_reset = pd.Series(y_test).reset_index(drop=True)

mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
print("MLflow tracking URI:", mlflow.get_tracking_uri())

def start_mlflow_ui(port=5000):
    """Launch MLflow UI from inside the notebook and open it in browser."""
    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", f"file:{MLFLOW_TRACKING_DIR}",
        "--port", str(port)
    ]
    
    print(f"Starting MLflow UI on http://127.0.0.1:{port} ...")
    process = subprocess.Popen(cmd)
    time.sleep(2)  # Wait briefly for UI to start
    webbrowser.open(f"http://127.0.0.1:{port}")

# Auto-start MLflow UI
start_mlflow_ui()

def train_and_log_model(model, X_train, y_train, X_test, y_test, model_name, params):
    """Trains, evaluates, and logs a model using MLflow."""
    
    with mlflow.start_run(run_name=f"{model_name}_Run"):
        print(f"\n--- Starting MLflow Run for: {model_name} ---")

        # --- Train Model ---
        model.set_params(**params)
        model.fit(X_train, y_train)

        # --- Evaluate ---
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Log Parameters and Metrics
        mlflow.log_params(params)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_weighted", f1_weighted)

        # Log Model (for comparison and potential registration)
        mlflow.sklearn.log_model(
             sk_model=model, 
             artifact_path="model", 
             registered_model_name=f"{model_name}_Microbiome_Classifier"
         )
        
        return model, accuracy # Return the model object

mlflow.set_experiment(EXPERIMENT_NAME)

# --- Run 1: Random Forest (The Best Model) ---
rf_params = {'n_estimators': N_ESTIMATORS, 'max_depth': MAX_DEPTH, 'random_state': RANDOM_STATE}
rf_model, rf_accuracy = train_and_log_model(RandomForestClassifier(), X_train_reset, y_train_reset, X_test_reset, y_test_reset, "RandomForest", rf_params)


# --- Run 2: XGBoost Classifier (The Competitor) ---
xgb_params = {
    'n_estimators': N_ESTIMATORS, 'max_depth': 5, 'learning_rate': 0.05,
    'random_state': RANDOM_STATE, 'objective': 'multi:softmax',
    'num_class': len(class_labels), 'use_label_encoder': False, 'eval_metric': 'mlogloss'
}
xgb_model, xgb_accuracy = train_and_log_model(XGBClassifier(), X_train_reset, y_train_reset, X_test_reset, y_test_reset, "XGBoost", xgb_params)

rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

RUN_NAME = "Model_Eval_Confusion_Matrices"
mlflow.set_experiment("Metagenome_Classifier_Comparison")

with mlflow.start_run(run_name=RUN_NAME):

    cm_rf = confusion_matrix(y_test, rf_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf_model.classes_)

    # Plot and save image
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Random Forest Confusion Matrix")
    rf_cm_png = os.path.join(SAVE_DIR_FIG, "rf_confusion_matrix.png")
    plt.savefig(rf_cm_png, dpi=200, bbox_inches="tight")
    plt.close()

    # Save as CSV
    rf_cm_df = pd.DataFrame(cm_rf, index=rf_model.classes_, columns=rf_model.classes_)
    rf_cm_csv = os.path.join(SAVE_DIR_TAB, "rf_confusion_matrix.csv")
    rf_cm_df.to_csv(rf_cm_csv)

    # Log to MLflow
    mlflow.log_artifact(rf_cm_png, artifact_path="confusion_matrix/random_forest")
    mlflow.log_artifact(rf_cm_csv, artifact_path="confusion_matrix/random_forest")

    cm_xgb = confusion_matrix(y_test, xgb_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=xgb_model.classes_)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("XGBoost Confusion Matrix")
    xgb_cm_png = os.path.join(SAVE_DIR_FIG, "xgb_confusion_matrix.png")
    plt.savefig(xgb_cm_png, dpi=200, bbox_inches="tight")
    plt.close()

    # Save as CSV
    xgb_cm_df = pd.DataFrame(cm_xgb, index=xgb_model.classes_, columns=xgb_model.classes_)
    xgb_cm_csv = os.path.join(SAVE_DIR_TAB, "xgb_confusion_matrix.csv")
    xgb_cm_df.to_csv(xgb_cm_csv)
    # Log to MLflow
    mlflow.log_artifact(xgb_cm_png, artifact_path="confusion_matrix/xgboost")
    mlflow.log_artifact(xgb_cm_csv, artifact_path="confusion_matrix/xgboost")

    rf_report = classification_report(y_test, rf_preds, output_dict=True)
    xgb_report = classification_report(y_test, xgb_preds, output_dict=True)

    # Save reports as CSV
    rf_report_df = pd.DataFrame(rf_report).transpose()
    xgb_report_df = pd.DataFrame(xgb_report).transpose()

    rf_report_csv = os.path.join(SAVE_DIR_TAB, "rf_classification_report.csv")
    xgb_report_csv = os.path.join(SAVE_DIR_TAB, "xgb_classification_report.csv")

    rf_report_df.to_csv(rf_report_csv)
    xgb_report_df.to_csv(xgb_report_csv)

    mlflow.log_artifact(rf_report_csv, artifact_path="classification_report/random_forest")
    mlflow.log_artifact(xgb_report_csv, artifact_path="classification_report/xgboost")

    # Also log macro-averaged metrics for dashboarding
    mlflow.log_metric("rf_macro_f1", rf_report['macro avg']['f1-score'])
    mlflow.log_metric("xgb_macro_f1", xgb_report['macro avg']['f1-score'])

    mlflow.log_metric("rf_accuracy", rf_report['accuracy'])
    mlflow.log_metric("xgb_accuracy", xgb_report['accuracy'])

def tune_random_forest(X_train, y_train, n_iter=25, cv=3, random_state=42):
    """
    Hyperparameter tuning for Random Forest with MLflow logging
    of *every model trained* during the search.
    """
    param_dist = {
        "n_estimators": randint(200, 800),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["auto", "sqrt", "log2"],
        "bootstrap": [True, False],
    }

    rf = RandomForestClassifier(random_state=random_state)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        random_state=random_state,
        return_train_score=True
    )

    # -------- Parent MLflow Run --------
    with mlflow.start_run(run_name="RF_Hyperparameter_Tuning") as parent_run:

        search.fit(X_train, y_train)

        # Extract CV results
        results = search.cv_results_
        # -------- Child Runs (one per candidate) --------
        for i in range(n_iter):
            with mlflow.start_run(run_name=f"RF_Candidate_{i}",
                                  nested=True):
                
                params = {k: results["param_%s" % k][i] for k in param_dist.keys()}
                mean_test_score = results["mean_test_score"][i]
                std_test_score = results["std_test_score"][i]

                # Log hyperparameters & CV score for this candidate
                mlflow.log_params(params)
                mlflow.log_metric("mean_test_f1", mean_test_score)
                mlflow.log_metric("std_test_f1", std_test_score)

        # -------- Log Best Model from Search --------
        best_params = search.best_params_
        best_score = search.best_score_
        best_model = search.best_estimator_

        mlflow.log_params(best_params)
        mlflow.log_metric("cv_best_f1_weighted", best_score)

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="best_rf_model",
            registered_model_name="RandomForest_Tuned_Microbiome_Classifier"
        )

        print(f"\nBest F1 (cv): {best_score:.4f}")
        print("Best RF Params:", best_params)

    return best_model, best_params

best_rf_model, best_rf_params = tune_random_forest(
    X_train_reset, y_train_reset,
    n_iter=25,     
    cv=3
)

rf_model, rf_accuracy = train_and_log_model(
    best_rf_model,
    X_train_reset, y_train_reset,
    X_test_reset, y_test_reset,
    "RandomForest_Tuned",
    best_rf_params
)


os.makedirs(DEPLOYMENT_DIR, exist_ok=True) 
with open(MODEL_PATH, 'wb') as file: 
    pickle.dump(best_rf_model, file) 

TOP_FEATURE = X_test.columns[0] if isinstance(X_test, pd.DataFrame) else 0
INTERACTING_FEATURE = None  # or set a feature name/index
X_test_columns = X_test.columns.tolist() if isinstance(X_test, pd.DataFrame) else [f"Feature_{i}" for i in range(X_test.shape[1])]

explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_test)

# Stack the list of arrays (one per class) into a single array
X_columns = clr_columns + ALPHA_COLUMNS

shap_array = np.stack(shap_values, axis=0)  # shape (33, 79, 3)

# mean abs SHAP across samples and classes
global_shap_impact = np.mean(np.abs(shap_array), axis=(0, 2))  # shape = (79,)

# 2. Create the ranking DataFrame using ALL 84 features/scores
shap_ranking_df = pd.DataFrame({
    'Feature': X_columns, 
    'Mean_Abs_SHAP': global_shap_impact
}) 


# 3. Sort and save the top 5
top_5_biomarkers_df = shap_ranking_df.sort_values(by='Mean_Abs_SHAP', ascending=False).head(5)

# --- 4. Save to CSV and Print ---
output_csv_path = "03_results/tables/top_5_shap_biomarkers.csv"
top_5_biomarkers_df.to_csv(output_csv_path, index=False)


if isinstance(shap_values, list):
    shap_values_per_class = shap_values
else:
    shap_values_per_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

n_classes = len(shap_values_per_class)
print(f"Detected {n_classes} classes.")


for c in range(n_classes):
    print(f"Generating summary (dot) plot for class {c}...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_per_class[c],
        X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=X_test_columns),
        feature_names=X_test_columns,
        plot_type="dot",
        show=False
    )
    out_path = os.path.join(SAVE_DIR_FIG, f"shap_summary_dot_class_{c}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

for c in range(n_classes):
    print(f"Generating summary (bar) plot for class {c}...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_per_class[c],
        X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=X_test_columns),
        feature_names=X_test_columns,
        plot_type="bar",
        show=False
    )
    out_path = os.path.join(SAVE_DIR_FIG, f"shap_summary_bar_class_{c}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

i = 0  # sample index
cls = y_test.iloc[i] if isinstance(y_test, (pd.Series, pd.DataFrame)) else y_test[i]

x_vals = X_test.iloc[i, :] if isinstance(X_test, pd.DataFrame) else X_test[i, :]

force_plot_html = shap.force_plot(
    explainer.expected_value[cls],
    shap_values_per_class[cls][i, :],
    x_vals,
    feature_names=X_test_columns,
    matplotlib=False  # HTML backend
)

out_path = os.path.join(SAVE_DIR_FIG, f"shap_force_sample_{i}_class_{cls}.html")
shap.save_html(out_path, force_plot_html)

top_features = ["CLR_1", "CLR_43", "CLR_17", "CLR_14", "CLR_3"]

if isinstance(X_test, pd.DataFrame):
    X_test_df = X_test.copy()
else:
    X_test_df = pd.DataFrame(X_test, columns=X_test_columns)
    
for c in range(n_classes):

    shap_vals_c = shap_values_per_class[c]  # shape = (n_samples, n_features)

    for feature in top_features:
        plt.figure(figsize=(8, 6))
        
        shap.dependence_plot(
            feature,
            shap_vals_c,         # <=== Correct matching dimension
            X_test_df,
            interaction_index="auto",
            show=False
        )

        out = os.path.join(
            SAVE_DIR_FIG, 
            f"shap_dependence_{feature}_class_{c}.png"
        )
        plt.savefig(out, bbox_inches="tight")
        plt.close()

libs = [
    "mlflow",
    "mlflow.sklearn",
    "pandas",
    "numpy",
    "sklearn",
    "xgboost",
    "matplotlib",
    "seaborn",
    "shap"
]

# Create requirements.txt with exact versions
with open("requirements.txt", "w") as f:
    for lib in libs:
        try:
            # Try to get version using __version__
            pkg = __import__(lib.split('.')[0])  # Handles mlflow.sklearn
            version = pkg.__version__
            f.write(f"{lib}=={version}\n")
        except AttributeError:
            # Fallback to pip show
            result = subprocess.run([sys.executable, "-m", "pip", "show", lib.split('.')[0]],
                                    capture_output=True, text=True)
            for line in result.stdout.split("\n"):
                if line.startswith("Version:"):
                    f.write(f"{lib}=={line.split()[1]}\n")





