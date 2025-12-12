## -----------------------------------------------------------------------------
library(readxl)
library(tidyverse)
library(ALDEx2)
library(vegan)
library(rstatix)
library(ggpubr)
library(stats)


## -----------------------------------------------------------------------------
otu_file <- "../../01_data/raw/zmeh_a_11816030_sm0001.xlsx"
meta_file <- "../../01_data/raw/zmeh_a_11816030_sm0002.xlsx"

SAMPLE_ID_COLUMN_META <- "#SampleID"


## -----------------------------------------------------------------------------
df_otu <- read_excel(otu_file, col_names = TRUE, skip = 1)
df_meta <- read_excel(meta_file, col_names = TRUE, skip = 1)

names(df_otu)[1] <- "OTU_ID"
names(df_meta)[1] <- "Sample_ID"



## -----------------------------------------------------------------------------
otu_counts_final <- df_otu %>%
    column_to_rownames("OTU_ID") %>%
    mutate(across(everything(), as.numeric)) %>%
    mutate(across(everything(), ~replace_na(., 0))) %>%
    mutate(across(everything(), as.integer))

STATUS_PATTERN <- "[a-z]+$"

metadata_aligned <- df_meta %>%
    filter(time == 0) %>%
    mutate(Status_Code = str_extract(Sample_ID, pattern = STATUS_PATTERN)) %>%
    mutate(Status_Code = tolower(Status_Code)) %>%
    mutate(Sample_ID = trimws(Sample_ID)) %>%
    column_to_rownames("Sample_ID")


## -----------------------------------------------------------------------------
common_samples <- intersect(colnames(otu_counts_final), rownames(metadata_aligned))
otu_counts_final <- otu_counts_final[, common_samples]
metadata_aligned <- metadata_aligned[common_samples, ]


## -----------------------------------------------------------------------------
min_prevalence <- 5 
min_count <- 10    

otus_to_keep <- rowSums(otu_counts_final > 0) >= min_prevalence &
                rowSums(otu_counts_final) >= min_count

otu_counts_final <- otu_counts_final[otus_to_keep, ]

# --- Remove zero-sum samples ---
zero_samples <- which(colSums(otu_counts_final) == 0)
if (length(zero_samples) > 0) {
    otu_counts_final <- otu_counts_final[, -zero_samples]
    metadata_aligned <- metadata_aligned[colnames(otu_counts_final), ]
}

cat("Final Aligned Samples:", ncol(otu_counts_final), "\n")
cat("Final Features Remaining:", nrow(otu_counts_final), "\n")


## -----------------------------------------------------------------------------
condition_vector <- metadata_aligned[colnames(otu_counts_final), "Status_Code"]

clr_results <- aldex.clr(
    reads = otu_counts_final,
    conds = as.character(condition_vector),
    mc.samples = 128,      # proper MC sampling
    denom = "all",
    verbose = FALSE,
    useMC = TRUE           # <- FORCE Monte Carlo sampling
)

clr_mc <- slot(clr_results, "analysisData")

if (length(clr_mc) < 1) stop("ALDEx2 returned zero CLR Monte Carlo samples.")

num_samples <- length(clr_mc)
num_features <- nrow(clr_mc[[1]])  # number of features
clr_feature_matrix <- matrix(NA, nrow = num_samples, ncol = num_features)

sample_ids <- names(clr_mc)
rownames(clr_feature_matrix) <- sample_ids

for (i in seq_along(clr_mc)) {
  clr_feature_matrix[i, ] <- rowMeans(clr_mc[[i]])
}

colnames(clr_feature_matrix) <- paste0("CLR_", rownames(clr_mc[[1]]))
clr_feature_matrix <- as.data.frame(clr_feature_matrix)






## -----------------------------------------------------------------------------
otu_samples_in_rows <- as.data.frame(t(otu_counts_final))

alpha_diversity <- data.frame(
    SampleID = rownames(otu_samples_in_rows),
    Shannon_Index = diversity(otu_samples_in_rows, index = "shannon"),
    Observed_Richness = specnumber(otu_samples_in_rows)
)

rownames(alpha_diversity) <- alpha_diversity$SampleID
alpha_diversity$SampleID <- NULL


## -----------------------------------------------------------------------------
metadata_to_join <- metadata_aligned %>% 
  rownames_to_column("SampleID")

final_features_temp <- clr_feature_matrix %>%
  rownames_to_column("SampleID") %>%
  inner_join(alpha_diversity %>% rownames_to_column("SampleID"), by = "SampleID")

df_full_features_with_status <- final_features_temp %>%
    inner_join(metadata_to_join, by = "SampleID") %>%
    column_to_rownames("SampleID")

write.csv(df_full_features_with_status, 
          file = "../../01_data/processed/final_ml_feature_matrix.csv", 
          row.names = TRUE)


## -----------------------------------------------------------------------------
df_full <- df_full_features_with_status 
clr_cols <- grep("^CLR_", names(df_full), value = TRUE)
df_pca_input <- df_full[, clr_cols]

# Perform PCA
pca_result <- prcomp(df_pca_input, scale. = TRUE)

# Prepare data for plotting (extract PC scores and merge status code)
pca_scores <- as.data.frame(pca_result$x) %>%
  # --- CRITICAL FIX: Explicitly call dplyr::select to avoid masking error ---
  dplyr::select(PC1, PC2) %>%
  # Merge the status code back for coloring the plot
  mutate(Status_Code = df_full$Status_Code)

# Prepare data for plotting (extract PC scores and merge status code)
pca_scores <- as.data.frame(pca_result$x) %>%
  select(PC1, PC2) %>%
  # Merge the status code back for coloring the plot
  mutate(Status_Code = df_full$Status_Code)

# Calculate variance explained for axis labels
variance_explained <- summary(pca_result)$importance[2, 1:2]
pc1_var <- paste0("PC1 (", round(variance_explained[1] * 100, 1), "%)")
pc2_var <- paste0("PC2 (", round(variance_explained[2] * 100, 1), "%)")

# --- PCA Scatter Plot ---
p3 <- ggplot(pca_scores, aes(x = PC1, y = PC2, color = Status_Code)) +
  geom_point(size = 3, alpha = 0.7) +
  stat_ellipse(geom = "polygon", alpha = 0.1, aes(fill = Status_Code)) + # Add confidence ellipses
  labs(title = "PCA of CLR Microbiome Profiles",
       subtitle = "Color coded by Clinical Status",
       x = pc1_var,
       y = pc2_var) +
  theme_minimal()
print(p3)
ggsave("~/Vaginal_Microbiome_ML_Classifier/03_results/figures/pca_plot.png", plot = p3, width = 7, height = 5)



## -----------------------------------------------------------------------------

df_full <- df_full_features_with_status # Using the full dataframe from the R code execution context
clr_cols <- grep("^CLR_", names(df_full), value = TRUE)
df_clr_input <- df_full[, clr_cols]

dist_matrix <- dist(df_clr_input, method = "euclidean")

permanova_result <- adonis2(dist_matrix ~ Status_Code, data = df_full, permutations = 999)
print(permanova_result)



## -----------------------------------------------------------------------------
dispersion <- betadisper(dist_matrix, df_full$Status_Code)
betadisper_result <- permutest(dispersion, permutations = 999)
print(betadisper_result)


## -----------------------------------------------------------------------------
# 1. --- Setup and Data Loading ---
# Attempt to load the final feature matrix
tryCatch({
    df_full <- read_csv("final_ml_feature_matrix.csv") %>%
        rename(SampleID = '...1') %>%
        column_to_rownames("SampleID")
}, error = function(e) {
    # Fallback path (use if the file is not in the working directory)
    df_full <- read_csv("../../01_data/processed/final_ml_feature_matrix.csv") %>%
        rename(SampleID = '...1') %>%
        column_to_rownames("SampleID")
})

# Define the features for analysis
clr_cols <- grep("^CLR_", names(df_full), value = TRUE)
top_biomarkers <- c("CLR_1", "CLR_43", "CLR_17", "CLR_14", "CLR_3")

# --- 2. Pairwise PERMANOVA (Multivariate Post-Hoc) ---
# Calculate the distance matrix on CLR features
dist_matrix <- dist(df_full[, clr_cols], method = "euclidean")
groups <- df_full$Status_Code

pairwise_permanova_results <- data.frame()
group_levels <- levels(factor(groups))

for (i in 1:(length(group_levels) - 1)) {
    for (j in (i + 1):length(group_levels)) {
        group1 <- group_levels[i]
        group2 <- group_levels[j]

        # Subset data for the pair
        subset_samples <- df_full %>% filter(Status_Code %in% c(group1, group2))
        subset_dist <- dist(subset_samples[, clr_cols], method = "euclidean")

        # Run PERMANOVA on the subset
        res <- adonis2(subset_dist ~ Status_Code, data = subset_samples, permutations = 999)

        # Store results
        pairwise_permanova_results <- rbind(pairwise_permanova_results, data.frame(
            Comparison = paste(group1, "vs", group2),
            R2 = res$R2[1],
            P_Value = res$`Pr(>F)`[1]
        ))
    }
}

# Apply FDR correction (Benjamini-Hochberg)
pairwise_permanova_results <- pairwise_permanova_results %>%
    mutate(FDR_P_Value = p.adjust(P_Value, method = "fdr"))

print(pairwise_permanova_results)



## -----------------------------------------------------------------------------

dunn_test_results <- df_full %>%
    select(Status_Code, all_of(top_biomarkers)) %>%
    pivot_longer(-Status_Code, names_to = "Feature", values_to = "Value") %>%
    group_by(Feature) %>%
    # Performs Dunn's test for all pairs of groups, automatically applying FDR correction
    dunn_test(Value ~ Status_Code, p.adjust.method = "fdr")

print(dunn_test_results)


## -----------------------------------------------------------------------------
plot_clr_features <- function(data, feature_name, dunn_results) {
  
  dunn_feature <- dunn_results %>%
    filter(Feature == feature_name) %>%
    arrange(p.adj) %>%
    mutate(
      y.position = max(data[[feature_name]], na.rm = TRUE) + 0.2 * row_number()
    )
  
  p <- ggplot(data, aes(x = Status_Code, y = !!sym(feature_name), fill = Status_Code)) +
    geom_boxplot(alpha = 0.7) +
    geom_jitter(width = 0.2, size = 1.5, alpha = 0.6) +
    labs(
      title = paste("CLR Abundance of", feature_name, "by Clinical Status"),
      x = "Clinical Status",
      y = feature_name
    ) +
    theme_minimal() +
    stat_pvalue_manual(
      dunn_feature,
      hide.ns = TRUE,
      label = "p.adj.signif"
    )
  
  return(p)
}

clr1_plot <- plot_clr_features(df_full, "CLR_1", dunn_test_results)
print(clr1_plot)
ggsave("../../03_results/figures/clr1_dunn_test_results.png", plot = clr1_plot)

clr43_plot <- plot_clr_features(df_full, "CLR_43", dunn_test_results)
print(clr43_plot)
ggsave("../../03_results/figures/clr43_dunn_test_results.png", plot = clr43_plot)

clr17_plot <- plot_clr_features(df_full, "CLR_17", dunn_test_results)
print(clr17_plot)
ggsave("../../03_results/figures/clr17_dunn_test_results.png", plot = clr17_plot)

clr14_plot <- plot_clr_features(df_full, "CLR_14", dunn_test_results)
print(clr14_plot)
ggsave("../../03_results/figures/clr14_dunn_test_results.png", plot = clr14_plot)

clr3_plot <- plot_clr_features(df_full, "CLR_3", dunn_test_results)
print(clr3_plot)
ggsave("../../03_results/figures/clr3_dunn_test_results.png", plot = clr3_plot)



## -----------------------------------------------------------------------------

wilcox_results <- df_full %>%
    select(Status_Code, all_of(top_biomarkers)) %>%
    pivot_longer(-Status_Code, names_to = "Feature", values_to = "Value") %>%
    group_by(Feature) %>%
    # Performs Wilcoxon test for all pairs of groups, automatically applying FDR correction
    wilcox_test(Value ~ Status_Code, p.adjust.method = "fdr")

print(wilcox_results)

