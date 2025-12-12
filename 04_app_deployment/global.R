# ====================================================================
# global.R - Microbiome ML Diagnostics and Explorer (Python-Free)
# ====================================================================

library(tidyverse)
library(DT)
library(plotly)
library(shinythemes)

# -------------------------------
# 1. Load processed feature matrix
# -------------------------------

DATA_PATH <- "final_ml_feature_matrix.csv"

df_processed <- tryCatch({
  readr::read_csv(DATA_PATH) %>%
    mutate(
      Status_Code = factor(
        Status_Code,
        levels = c("bcont", "bbv", "bvvc")
      )
    )
}, error = function(e) {
  message("ERROR: Could not load processed matrix. Using fallback dummy data.")
  tibble(
    SampleID      = paste0("S", seq_len(100)),
    Status_Code   = factor(sample(c("bcont", "bbv", "bvvc"), 100, TRUE),
                           levels = c("bcont", "bbv", "bvvc")),
    Shannon_Index = runif(100, 1, 5),
    CLR_1  = rnorm(100),
    CLR_14 = rnorm(100),
    CLR_17 = rnorm(100),
    CLR_3  = rnorm(100),
    CLR_43 = rnorm(100)
  )
})

# all possible feature columns except IDs / labels
feature_cols <- setdiff(names(df_processed), c("SampleID", "Status_Code"))

# -------------------------------
# 2. Precomputed ML metrics
# -------------------------------

MODEL_PERFORMANCE_METRICS <- list(
  rf = list(
    accuracy = 0.788,  # 78.8%
    f1       = 0.79
  ),
  xgb = list(
    accuracy = 0.788,
    f1       = 0.78
  )
)

# -------------------------------
# 3. Global SHAP ranking (top features)
# -------------------------------

SHAP_RANKING <- tibble::tribble(
  ~Feature,  ~MeanAbsSHAP,
  "CLR_1",    0.40,
  "CLR_43",   0.28,
  "CLR_17",   0.21,
  "CLR_14",   0.18,
  "CLR_3",    0.16
)

# -------------------------------
# 4. SHAP dependence plots (static PNGs in www/)
# -------------------------------
# We have 5 features × 3 classes

shap_dependence_paths <- list(
  CLR_1  = list(
    class0 = "www/shap_dependence_CLR_1_class_0.png",
    class1 = "www/shap_dependence_CLR_1_class_1.png",
    class2 = "www/shap_dependence_CLR_1_class_2.png"
  ),
  CLR_14 = list(
    class0 = "www/shap_dependence_CLR_14_class_0.png",
    class1 = "www/shap_dependence_CLR_14_class_1.png",
    class2 = "www/shap_dependence_CLR_14_class_2.png"
  ),
  CLR_17 = list(
    class0 = "www/shap_dependence_CLR_17_class_0.png",
    class1 = "www/shap_dependence_CLR_17_class_1.png",
    class2 = "www/shap_dependence_CLR_17_class_2.png"
  ),
  CLR_3  = list(
    class0 = "www/shap_dependence_CLR_3_class_0.png",
    class1 = "www/shap_dependence_CLR_3_class_1.png",
    class2 = "www/shap_dependence_CLR_3_class_2.png"
  ),
  CLR_43 = list(
    class0 = "www/shap_dependence_CLR_43_class_0.png",
    class1 = "www/shap_dependence_CLR_43_class_1.png",
    class2 = "www/shap_dependence_CLR_43_class_2.png"
  )
)

# -------------------------------
# 5. Statistical summaries
# -------------------------------

PERMANOVA_GLOBAL <- list(
  R2      = 0.1348,
  p_value = 0.001
)

PERMANOVA_PAIRWISE <- tibble::tribble(
  ~Comparison,        ~R2,    ~p_value,
  "bbv vs bcont",     0.134,  0.001,
  "bbv vs bvvc",      0.125,  0.001,
  "bcont vs bvvc",    0.041,  0.001
)

UNIVARIATE_SUMMARY <- list(
  dunn = list(
    message = "Dunn post-hoc tests show all bbv vs bcont and bbv vs bvvc comparisons are highly significant, while bcont vs bvvc is largely non-significant except for CLR_14."
  ),
  wilcoxon = list(
    message = "Pairwise Wilcoxon tests confirm strong effect sizes for BV vs other groups and weak or absent differences between healthy and VVC."
  )
)

LITERATURE_CONTEXT <- paste(
  "These results are consistent with prior literature (e.g., Macklaim et al. 2015),",
  "where BV is characterized by a strong anaerobic dysbiosis and loss of commensal stability,",
  "while VVC often retains a Lactobacillus-dominant bacterial community similar to healthy controls."
)
