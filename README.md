# Interpretable ML Classifier for Metagenomic Diagnostics: CLR Transformation & SHAP-Driven Biomarker Discovery

## Project Goal & Overview

This project establishes a reproducible machine learning pipeline to classify complex vaginal clinical states ($\mathbf{Healthy}$, $\mathbf{BV}$, $\mathbf{BVVC}$) using high-dimensional $\text{OTU}$ count data. The core objective is to identify robust microbial biomarkers by focusing on **model interpretability (XAI)**.

-----

## Current Status: Analytical Phase Complete

All core data processing, model training, performance evaluation, and initial interpretability analysis have been successfully completed and are stored within the repository structure.

### 1\. Data Preparation and Feature Engineering (R)

  * **File:** `./02\_src/R/feature\_engineering.Rmd`
  * **Methodology:** Implemented **Compositional Data Analysis (CoDA)** via the **Centered Log-Ratio (CLR) Transformation** to mitigate statistical issues in the $\text{OTU}$ data.
  * **Structural Validation:** Generated the **PCA plot**  and descriptive box plots (e.g., `clr1_boxplot.png`, `shannon_boxplot.png`) confirming that the $\mathbf{CLR}$-transformed feature set separates the clinical status groups.

### 2\. Model Training and Selection (Python)

  * **File:** `./02\_src/Python/model\_development.ipynb`
  * **Comparison:** Trained and evaluated **Random Forest** and **XGBoost** classifiers. The **Random Forest** model was selected as the final classifier.
  * **Evaluation:** Generated the **Confusion Matrix** (implied output) and the `model_performance_grouped_bar_plot.png` for performance comparison.

### 3\. Model Interpretation (SHAP Analysis)

  * **Analysis:** Used $\text{SHAP}$ analysis to explain the Random Forest model's decisions.
  * **Artifacts:** Generated multiple class-specific $\text{SHAP}$ plots (`shap\_class\_0.png`, `shap\_class\_1.png`, `shap\_class\_2.png`) and the final average plot (`shap\_mean\_across\_classes.png`) which verifies the overall feature importance and directional influence.

-----

## Generated Analysis Artifacts

The following key files have been generated and represent the completed analysis:

  * **Dimensionality Reduction:** `pca_plot.png`
  * **Biomarker Analysis:** `clr1_boxplot.png`, `shannon_boxplot.png`
  * **Model Comparison:** `model_performance_grouped_bar_plot.png`
  * **SHAP Interpretation:** `shap_mean_across_classes.png` (Global Summary Plot)

-----

## 📂 Repository Structure (Current State)

```markdown
├── README.md
├── 01_data/
│   ├── processed/
│   │   └── final_ml_feature_matrix.csv # CLR-transformed data used for ML
│   └── raw/
│       ├── zmeh_a_11816030_sm0001.xlsx # Raw OTU Counts
│       └── zmeh_a_11816030_sm0002.xlsx # Raw Metadata
│
├── 02_src/
│   ├── Python/
│   │   ├── model_development.ipynb     # Python code for ML training, evaluation, and SHAP
│   │   └── *.png                       # All Python-generated SHAP and performance plots
│   └── R/
│       ├── feature_engineering.Rmd     # R code for CLR transformation and initial statistics
│       └── *.png, *.html               # R-generated plots (PCA, boxplots)
│
├── 03_results/
│   └── figures/                        # Placeholder for final, curated figures (currently empty)
│
└── 04_app_deployment/                  # Placeholder for R Shiny app and Docker files
```

-----

## ⚙️ Next Steps

The next phase of the project will focus on productionizing the pipeline:

1.  Implement **MLflow** for experiment tracking and **DVC** for data versioning.
2.  Serialize the final $\mathbf{Random\ Forest}$ model object.
3.  Develop the four-module **R Shiny application** in the `./04\_app\_deployment` directory for interactive final delivery.
