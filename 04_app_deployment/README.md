# Microbiome ML Diagnostics and Explorer Application

This Shiny application provides an interactive interface for exploring vaginal microbiome diagnostics using statistical summaries, machine learning outputs, and explainable AI visualizations. The app supports clinicians, researchers, and students in examining microbial features, model performance, and biomarker signals through a user-friendly dashboard.

## 1. Purpose

This app enables users to:
- Explore CLR-transformed microbiome features
- Review Random Forest and XGBoost confusion matrices
- Visualize PCA ordination of microbial structure
- Examine biomarker distributions and statistical test results
- Explore SHAP interpretability results via static plots

All machine learning and SHAP computations were performed offline in Python, and the resulting figures are rendered inside the Shiny application.

## 2. Key Modules

2.1 Data Explorer

- Interactive browsing of the processed feature matrix
- Filtering and sorting with DT tables
- Viewing distributions of CLR and diversity features

2.2 Model Validation

- Displays confusion matrices (as static PNG images)
- Shows classification metrics loaded from CSV tables
- PCA visualizations of microbiome community structure
- PERMANOVA summary text

2.3 Biomarker Discovery

- Statistical testing results (Kruskal–Wallis + Dunn tests)
- Boxplots of key CLR biomarkers (static images exported from R/Python)
- Textual interpretation of biomarker significance

2.4 Model Interpretation (SHAP)

- Global feature importance plots (SHAP bar and dot plots)
- Class-specific SHAP dependence plots for top biomarkers
- Local SHAP force plot (HTML widget rendered as an iframe)
All SHAP assets are precomputed offline and displayed as images or embedded HTML.

## 3. Application Architecture

The Shiny app is pure R:
- No Python execution
- No reticulate environment needed
- No live model inference inside Shiny

Instead, the app loads:
- The processed feature matrix (final_ml_feature_matrix.csv)
- Pre-rendered result tables (03_results/tables/)
- Pre-generated figures (03_results/figures/)


## 4. File Structure
   
```
04_app_deployment/
├── ui.R
├── server.R
├── global.R
├── final_ml_feature_matrix.csv
├── final_rf_model.pkl       # for reproducibility only (not used in Shiny)
└── www/                     # location of all static image assets
```

Place all confusion matrices, SHAP plots, biomarker plots, PCA images, and Dunn test images inside www/.

## 5. Running the App Locally

setwd("04_app_deployment")
shiny::runApp()

No Python environment is required.

## 6. Deployment Status

The Shiny application is fully deployed and publicly accessible:
https://drshanda.shinyapps.io/microbiome_diagnostics/
