library(shiny)
library(shinythemes)
library(plotly)
library(DT)

ui <- fluidPage(
  navbarPage(
    "Microbiome ML Diagnostics and Explorer",
    theme = shinytheme("yeti"),
    
    # ======================================
    # 1. DATA EXPLORER
    # ======================================
    tabPanel(
      "1. Data Explorer (Data Transparency)",
      h2("Data Explorer Module"),
      
      wellPanel(
        h4("Processed Feature Matrix"),
        DTOutput("processed_data_table")
      ),
      
      fluidRow(
        column(
          width = 6,
          wellPanel(
            h4("Custom Scatter Plot"),
            selectInput("scatter_x", "X-Axis Feature", choices = NULL),
            selectInput("scatter_y", "Y-Axis Feature", choices = NULL),
            plotlyOutput("custom_scatter_plot")
          )
        )
      )
    ),
    
    # ======================================
    # 2. MODEL VALIDATION
    # ======================================
    tabPanel(
      "2. Model Validation (Performance and Structure)",
      h2("Model Validation Module"),
      
      fluidRow(
        column(
          width = 4,
          wellPanel(
            h4("Random Forest Performance"),
            textOutput("rf_accuracy_text"),
            textOutput("rf_f1_text")
          ),
          wellPanel(
            h4("RF Confusion Matrix"),
            plotOutput("rf_confusion_matrix")
          )
        ),
        column(
          width = 4,
          wellPanel(
            h4("XGBoost Performance"),
            textOutput("xgb_accuracy_text"),
            textOutput("xgb_f1_text")
          ),
          wellPanel(
            h4("XGB Confusion Matrix"),
            plotOutput("xgb_confusion_matrix")
          )
        ),
        column(
          width = 4,
          wellPanel(
            h4("PCA Structure"),
            plotlyOutput("pca_plot")
          ),
          wellPanel(
            h4("PERMANOVA Results"),
            verbatimTextOutput("permanova_text")
          )
        )
      )
    ),
    
    # ======================================
    # 3. BIOMARKER DISCOVERY
    # ======================================
    tabPanel(
      "3. Biomarker Discovery (Statistical Evidence)",
      h2("Biomarker Discovery Module"),
      
      fluidRow(
        column(
          width = 4,
          wellPanel(
            h4("Select Feature"),
            selectInput("biomarker_feature", "CLR Feature", choices = NULL),
            h4("Kruskal–Wallis P-value"),
            verbatimTextOutput("kruskal_wallis_p"),
            htmlOutput("biomarker_otu_text")
          )
        ),
        column(
          width = 8,
          wellPanel(
            h4("Feature Distribution Across Clinical Groups"),
            plotlyOutput("biomarker_boxplot", height = "450px")
          )
        )
      )
    ),
    
    # ======================================
    # 4. SHAP INTERPRETATION
    # ======================================
    tabPanel(
      "4. Model Interpretation (SHAP Insights)",
      h2("Model Interpretation Module"),
      
      fluidRow(
        column(
          width = 6,
          wellPanel(
            h4("Top SHAP Features (Global Importance Ranking)"),
            DTOutput("feature_importance_table")
          )
        ),
        column(
          width = 6,
          wellPanel(
            h4("SHAP Dependence Plot"),
            selectInput("shap_dep_feature", "CLR Feature", choices = NULL),
            selectInput("shap_dep_class", "Class", choices = c("class0", "class1", "class2")),
            plotOutput("shap_dependence_plot", height = "350px")
          )
        )
      )
    )
  )
)
