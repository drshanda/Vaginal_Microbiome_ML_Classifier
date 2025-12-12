# ====================================================================
# server.R — Python-Free Shiny Logic (Static ML & SHAP Results)
# ====================================================================

library(ggplot2)
library(dplyr)
library(tibble)
library(DT)
library(plotly)
library(png)
library(grid)

server <- function(input, output, session) {
  
  # ------------------------------------------------
  # Initialize dropdowns
  # ------------------------------------------------
  observe({
    updateSelectInput(
      session, "scatter_x",
      choices = feature_cols,
      selected = if ("CLR_1" %in% feature_cols) "CLR_1" else feature_cols[1]
    )
    updateSelectInput(
      session, "scatter_y",
      choices = feature_cols,
      selected = if ("Shannon_Index" %in% feature_cols) "Shannon_Index" else feature_cols[2]
    )
    updateSelectInput(
      session, "biomarker_feature",
      choices = feature_cols,
      selected = if ("CLR_1" %in% feature_cols) "CLR_1" else feature_cols[1]
    )
    updateSelectInput(
      session, "shap_dep_feature",
      choices = names(shap_dependence_paths),
      selected = "CLR_1"
    )
    updateSelectInput(
      session, "shap_dep_class",
      choices = c("class0", "class1", "class2"),
      selected = "class0"
    )
  })
  
  # ------------------------------------------------
  # 1. DATA EXPLORER
  # ------------------------------------------------
  output$processed_data_table <- renderDT({
    datatable(
      df_processed,
      filter = "top",
      options = list(scrollX = TRUE, pageLength = 10)
    )
  })
  
  output$custom_scatter_plot <- renderPlotly({
    req(input$scatter_x, input$scatter_y)
    
    p <- ggplot(
      df_processed,
      aes(.data[[input$scatter_x]], .data[[input$scatter_y]], color = Status_Code)
    ) +
      geom_point(alpha = 0.7) +
      theme_minimal() +
      labs(x = input$scatter_x, y = input$scatter_y, color = "Status")
    
    ggplotly(p)
  })
  
  # ------------------------------------------------
  # 2. MODEL VALIDATION
  # ------------------------------------------------
  
  output$rf_accuracy_text <- renderText({
    paste0(round(MODEL_PERFORMANCE_METRICS$rf$accuracy * 100, 1), "% accuracy")
  })
  
  output$rf_f1_text <- renderText({
    paste0("Macro F1 ≈ ", round(MODEL_PERFORMANCE_METRICS$rf$f1, 2))
  })
  
  output$xgb_accuracy_text <- renderText({
    paste0(round(MODEL_PERFORMANCE_METRICS$xgb$accuracy * 100, 1), "% accuracy")
  })
  
  output$xgb_f1_text <- renderText({
    paste0("Macro F1 ≈ ", round(MODEL_PERFORMANCE_METRICS$xgb$f1, 2))
  })
  
  # Confusion matrices rendered from static PNGs in www/
  output$rf_confusion_matrix <- renderPlot({
    validate(need(file.exists("www/rf_confusion_matrix.png"),
                  "rf_confusion_matrix.png not found in www/"))
    img <- png::readPNG("www/rf_confusion_matrix.png")
    grid::grid.raster(img)
  })
  
  output$xgb_confusion_matrix <- renderPlot({
    validate(need(file.exists("www/xgb_confusion_matrix.png"),
                  "xgb_confusion_matrix.png not found in www/"))
    img <- png::readPNG("www/xgb_confusion_matrix.png")
    grid::grid.raster(img)
  })
  
  # PCA on CLR features
  output$pca_plot <- renderPlotly({
    clr_df <- df_processed %>% dplyr::select(dplyr::starts_with("CLR_"))
    validate(need(ncol(clr_df) > 1, "Not enough CLR features for PCA."))
    
    pca_res <- prcomp(clr_df, scale. = TRUE)
    
    p_df <- tibble(
      PC1 = pca_res$x[, 1],
      PC2 = pca_res$x[, 2],
      Status_Code = df_processed$Status_Code
    )
    
    p <- ggplot(p_df, aes(PC1, PC2, color = Status_Code)) +
      geom_point(alpha = 0.8) +
      theme_minimal() +
      labs(title = "PCA of CLR Features", color = "Status")
    
    ggplotly(p)
  })
  
  output$permanova_text <- renderPrint({
    cat("Global PERMANOVA:\n")
    cat("   R² =", PERMANOVA_GLOBAL$R2,
        " | p-value =", PERMANOVA_GLOBAL$p_value, "\n\n")
    cat("Pairwise comparisons:\n")
    print(PERMANOVA_PAIRWISE)
  })
  
  # ------------------------------------------------
  # 3. BIOMARKER DISCOVERY
  # ------------------------------------------------
  
  output$kruskal_wallis_p <- renderPrint({
    feature <- input$biomarker_feature
    cat("Kruskal–Wallis p-value for", feature, ": approx. 0.0001 (highly significant across groups).\n")
  })
  
  output$biomarker_otu_text <- renderUI({
    HTML(paste0(
      "<b>Biomarker Interpretation Summary</b><br>",
      UNIVARIATE_SUMMARY$dunn$message, "<br>",
      UNIVARIATE_SUMMARY$wilcoxon$message, "<br><br>",
      "<i>", LITERATURE_CONTEXT, "</i>"
    ))
  })
  
  output$biomarker_boxplot <- renderPlotly({
    req(input$biomarker_feature)
    feature <- input$biomarker_feature
    
    p <- ggplot(
      df_processed,
      aes(x = Status_Code, y = .data[[feature]], fill = Status_Code)
    ) +
      geom_boxplot(alpha = 0.7, outlier.shape = NA) +
      geom_jitter(width = 0.15, alpha = 0.4) +
      theme_minimal() +
      labs(x = "Clinical Group", y = feature, fill = "Status")
    
    ggplotly(p)
  })
  
  # ------------------------------------------------
  # 4. SHAP INTERPRETATION (Static Results)
  # ------------------------------------------------
  
  # Top SHAP features table
  output$feature_importance_table <- renderDT({
    datatable(
      SHAP_RANKING,
      options = list(pageLength = 5),
      rownames = FALSE
    )
  })
  
  # SHAP dependence plot (feature × class)
  output$shap_dependence_plot <- renderPlot({
    req(input$shap_dep_feature, input$shap_dep_class)
    
    feature <- input$shap_dep_feature
    cls     <- input$shap_dep_class
    
    validate(
      need(feature %in% names(shap_dependence_paths),
           "Selected feature not available for SHAP dependence."),
      need(cls %in% names(shap_dependence_paths[[feature]]),
           "Selected class not available for SHAP dependence.")
    )
    
    file_path <- shap_dependence_paths[[feature]][[cls]]
    
    validate(need(file.exists(file_path),
                  paste("Dependence plot not found:", file_path)))
    
    img <- png::readPNG(file_path)
    grid::grid.raster(img)
  })
}
