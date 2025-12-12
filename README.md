# Interpretable ML Classifier for Metagenomic Diagnostics: CLR Transformation & SHAP-Driven Biomarker Discovery

### AI/ML • Bioinformatics • Compositional Data Analysis • Explainable ML • Clinical Microbiome Modeling

## 1. Overview

This project presents an **end-to-end microbiome machine learning pipeline** integrating:

* **Statistical microbiome analysis (R)**
* **Supervised ML classification (Python)**
* **Explainable AI (SHAP)**
* **MLOps tooling (MLflow, DVC, Docker, AWS S3)**
* **Interactive clinical reporting (R Shiny application)**

The goal is to classify samples into **Healthy**, **Bacterial Vaginosis (BV)**, or **Vulvovaginal Candidiasis (VVC)** using CLR-transformed microbial features and alpha diversity metrics.
This work combines bioinformatics, statistical ecology, machine learning, and software engineering into a **reproducible diagnostic modeling workflow** suitable for research, teaching, and portfolio demonstration.

---

## 2. Project Objectives

1. **Analyze microbial community structure** using PERMANOVA, BETADISPER, PCA, and non-parametric testing.
2. **Develop an interpretable machine learning classifier** (Random Forest, XGBoost).
3. **Identify microbial biomarkers** using Dunn tests, Wilcoxon tests, and SHAP.
4. **Track, version, and deploy the pipeline** using DVC, MLflow, Docker, and AWS S3.
5. **Provide an interactive clinical-facing dashboard** through a Shiny application.

---

## 3. Repository Structure

📁 01_data/
    ├── raw/                     # Raw microbiome input files
    └── processed/               # DVC-tracked CLR feature matrix

📁 02_src/
    ├── Python/                  # ML modeling and SHAP analysis
    │     ├── model_development_final.ipynb
    │     └── model_development_final.py
    └── R/                       # Statistical microbiome analysis
          └── microbiome_feature_analysis.Rmd

📁 03_results/
    ├── figures/                 # PCA, confusion matrices, SHAP plots
    └── tables/                  # Classification reports, SHAP rankings

📁 04_app_deployment/
    ├── ui.R                     # Shiny UI
    ├── server.R                 # Shiny server logic
    ├── global.R                 # App configuration
    ├── final_rf_model.pkl       # Serialized model
    └── www/                     # All displayed figures (PNG/HTML)

📁 docker/
    ├── Dockerfile               # Model inference container
    ├── requirements.txt         # Python dependencies for Docker
    └── app/
          ├── inference.py
          └── final_rf_model.pkl

📁 mlruns/                       # MLflow experiment tracking
📁 mlflow.db                    # MLflow SQLite backend

README.md                        # Project overview & documentation

---

## 4. Methods

### 4.1 Statistical Microbiome Analysis (R)

Tests and visualizations include:

* CLR transformation of compositional microbial data
* PCA ordination
* PERMANOVA and BETADISPER
* Kruskal–Wallis with Dunn post-hoc testing
* Pairwise Mann–Whitney U tests (FDR corrected)

Findings align with Macklaim et al. 2015, showing:

* **BV = high-diversity dysbiosis** with loss of Lactobacillus
* **VVC = microbiologically similar to Healthy** (fungal-driven, not bacterial) 

---

### 4.2 Machine Learning Pipeline (Python)

Models trained:

* **Random Forest (tuned):** Best F1 (CV) = 0.8862
* **XGBoost classifier**

Model evaluation:

* Test accuracy: **78.8%**
* BV predicted with **very high recall**
* BVVC predicted less accurately (weak bacterial signature)

Outputs include:

* Confusion matrices
* Classification reports
* SHAP global and local interpretability
* Top 5 SHAP-ranked biomarkers

---

### 4.3 Explainable AI (SHAP)

The pipeline generates:

* **Global SHAP summary (bar & dot plots)**
* **Class-specific SHAP rankings**
* **Dependence plots for top biomarkers**

Interpretation reveals:

* **CLR_1, CLR_43, CLR_17, CLR_3** behave as **commensal-stability markers**
* **CLR_14** behaves as a **dysbiosis-associated opportunistic feature**
* Patterns align precisely with R statistical testing and published literature

---

## R Shiny Application

This project includes a fully developed and deployed R Shiny application for interactive exploration of the dataset, statistical results, machine learning outputs, and SHAP-based interpretability. The Shiny app now uses static precomputed images (confusion matrices, SHAP plots, PCA, and biomarker figures) instead of running Python or SHAP inside the Shiny environment, ensuring full compatibility with shinyapps.io.

### Application Features

1. Data Explorer
- Interactive table of CLR-transformed features
- Summary statistics and feature visualizations

2. Model Validation

- Random Forest and XGBoost confusion matrices (PNG)
- PCA ordination of CLR-transformed features
- PERMANOVA summary text and interpretation

3. Biomarker Discovery

- Boxplots and significance summaries for CLR biomarkers
- Kruskal–Wallis + Dunn test results

4. Model Interpretation (SHAP)

- Top SHAP Features (Global Importance Ranking)
- Class-specific SHAP dependence plots


All assets displayed in the Shiny app originate from precomputed outputs saved in 03_results/figures/.

### Deployment

The application is publicly accessible at: https://drshanda.shinyapps.io/microbiome_diagnostics/

---

## 6. MLOps Components

### 6.1 DVC (Data Version Control)

Used to track and version:
* Processed feature matrices (`01_data/processed`)
* Intermediate artifacts

Artifacts are now **remote-tracked on AWS S3**, ensuring reproducibility across machines.

Commands used:

```bash
dvc init
dvc remote add -d s3remote s3://microbiome-ml-dvc-store
dvc add 01_data/processed/final_ml_feature_matrix.csv
dvc push
```

---

### 6.2 AWS S3 Integration

AWS S3 is used for storing:

* Versioned feature matrices (DVC artifacts)

This cloud integration ensures:

* Reproducibility
* Centralized artifact tracking
* Long-term storage and auditability

---

### 6.3 Docker Containerization

A Docker image has been created to standardize the Python ML environment.

Typical `Dockerfile` components include:

* Base image
* Installation of Python dependencies
* Inclusion of serialized model (final_rf_model.pkl)
* Exposure of the environment for SHAP computation

Benefits:

* Guaranteed reproducibility
* Portable deployment
* Future compatibility with AWS SageMaker or Kubernetes

---

### 6.4 MLflow Tracking

The notebook logs:

* Model parameters
* Cross-validation metrics
* Final model performance
* SHAP explanation artifacts

MLflow UI allows experiment comparison and model lineage tracking.

---

## 7. Results Summary

### Statistical Results

* BV shows **strong and significant community shifts**
* VVC shows **minimal bacterial change**, consistent with literature
* Top significant CLR biomarkers:
  **CLR_1, CLR_43, CLR_17, CLR_14, CLR_3**

### Machine Learning Results

* Accuracy: **78.8%**
* RF Recall for BV: **0.91**
* XGB Recall for BV: **1.00**
* BVVC remains hardest to classify (biologically consistent)

### SHAP Interpretation

* BV classification driven by **loss of commensals**
* CLR_14 uniquely increases BV probability
* Healthy/BVVC separated by **high commensal abundance**
* Local explanations verify classifier logic on individual samples

---

## 8. Discussion

This project demonstrates the power of combining:

* **Microbiome compositional statistics**
* **Supervised machine learning**
* **Explainable AI**

The alignment between R statistical tests and SHAP interpretability confirms that:

* **BV is characterized by commensal collapse + opportunistic bloom**
* **VVC maintains commensal stability**, explaining weaker ML detection
* The model has learned **true microbial ecology**, not noise

This makes the classifier both **interpretable** and **clinically plausible**—a critical requirement in biomedical AI.

---

## 9. Conclusion

This work provides a reproducible, transparent diagnostic pipeline capable of:

* Identifying key microbial biomarkers
* Explaining predictions biologically
* Supporting research and clinical hypothesis generation
* Being deployed interactively through a Shiny web application

It represents a complete workflow expected of an **AI/ML Bioinformatics Scientist**—from raw data to statistical validation, machine learning, interpretability, and deployment.

---

## 10. Running the Project

### Statistical Analysis (R)

Open:

```
02_src/R/microbiome_feature_analysis.html
```

### Machine Learning Notebook (Python)

```
jupyter notebook ./02_src/Python/model_development_final.ipynb
```

### Run the Shiny App

```r
setwd("04_app_deployment")
shiny::runApp()
```

### Docker Build

```bash
docker build -t microbiome-ml .
```

### DVC Fetch Data

```bash
dvc pull
```

---

