# Statistical Analysis of Model Performance Across Stages
# This script performs statistical comparisons between models and metrics
# across all three stages of the clinical case analysis

# Load necessary libraries
library(dplyr)
library(tidyr)
library(rstatix)  # For statistical tests
library(ggplot2)  # For potential visualization
library(forcats)  # For factor manipulation

# --- 1. Configuration & Setup ---

# Create results directory if it doesn't exist
output_dir <- "results_simplified_weighted_rank"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
  cat("Created '", output_dir, "' directory for saving output files\n")
}

# Define dataset prefix for file naming
dataset_prefix <- "Results_typical_cases_CPS_concise_selected"

# Construct file paths
file_s1 <- paste0(dataset_prefix, "_stage_1_long_format_ordered_top_k_rated.csv")
file_s2 <- paste0(dataset_prefix, "_stage_2_long_format_ordered_top_k_rated.csv")
file_s3 <- paste0(dataset_prefix, "_stage_3_long_format_ordered_top_k_rated.csv")
cat("Reading data files:\n")
cat("  Stage 1:", file_s1, "\n")
cat("  Stage 2:", file_s2, "\n")
cat("  Stage 3:", file_s3, "\n")

# --- 2. Data Loading and Initial Preparation ---

# Read the CSV files
data_s1 <- read.csv(file_s1)
data_s2 <- read.csv(file_s2)
data_s3 <- read.csv(file_s3)

# Combine Stage 1, Stage 2, and Stage 3 data, adding a stage identifier
all_data <- bind_rows(
  mutate(data_s1, stage_numeric = 1),
  mutate(data_s2, stage_numeric = 2),
  mutate(data_s3, stage_numeric = 3)
)

# --- 3. Data Preprocessing ---

# Model name mapping (essential for readable labels)
model_name_mapping <- data.frame(
  model = c(
    "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-20250219_thinking", "claude-3-opus-20240229", "chatgpt-4o-latest",
    "gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "gpt-4-0613", "gpt-3.5-turbo-0125",
    "o1-2024-12-17_high", "o1-2024-12-17_medium", "o1-mini-2024-09-12", "o1-preview-2024-09-12",
    "o3-mini-2025-01-31_high", "o3-mini-2025-01-31_medium", "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b-001", "gemini-1.5-pro-002", "gemini-2.0-flash-001",
    "gemini-2.0-pro-exp-02-05", "gemini-2.0-flash-thinking-exp-01-21"
  ),
  display_name = c(
    "Claude 3.5 Sonnet", "Claude 3.5 Haiku", "Claude 3.7 Sonnet", "Claude 3.7 Thinking",
    "Claude 3 Opus", "GPT-4o Latest", "GPT-4o", "GPT-4o Mini", "GPT-4", "GPT-3.5 Turbo",
    "O1 (High)", "O1 (Medium)", "O1 Mini", "O1 Preview", "O3-mini (High)", "O3-mini (Medium)",
    "Gemini 1.5 Flash", "Gemini 1.5 Flash 8B", "Gemini 1.5 Pro", "Gemini 2.0 Flash",
    "Gemini 2.0 Pro", "Gemini 2.0 Thinking"
  )
)

all_data <- all_data %>%
  left_join(model_name_mapping, by = "model") %>%
  mutate(model = ifelse(is.na(display_name), model, display_name)) %>%
  select(-display_name) %>%
  mutate(across(starts_with("differential_top_"), ~replace_na(., 0)))

all_data$stage_factor <- factor(
  all_data$stage_numeric,
  levels = c(1, 2, 3),
  labels = c("Stage 1", "Stage 2", "Stage 3")
)

# --- 4. Calculate Mean Scores and Standard Errors ---
model_scores_summary <- all_data %>%
  group_by(model, stage_numeric, stage_factor) %>%
  summarize(
    k1_score = mean(differential_top_1_accuracy, na.rm = TRUE),
    k5_score = mean(differential_top_5_accuracy, na.rm = TRUE),
    k10_score = mean(differential_top_10_accuracy, na.rm = TRUE),
    k1_se = sd(differential_top_1_accuracy, na.rm = TRUE) / sqrt(n()),
    k5_se = sd(differential_top_5_accuracy, na.rm = TRUE) / sqrt(n()),
    k10_se = sd(differential_top_10_accuracy, na.rm = TRUE) / sqrt(n()),
    n_cases = n(),
    .groups = 'drop'
  )

# --- 5. Model Ranking (Weighted Point System) ---
# This is used to determine the top model for comparison
ranked_models_by_points <- model_scores_summary %>%
  # Pivot wider to get stage-specific scores in columns for point calculation
  select(model, stage_numeric, k1_score, k5_score, k10_score) %>%
  pivot_wider(
    names_from = stage_numeric,
    values_from = c(k1_score, k5_score, k10_score),
    names_glue = "stage{stage_numeric}_{.value}" # e.g., stage1_k1_score
  ) %>%
  # Handle cases where a model might not have data for all stages (fill with 0 for point calculation)
  mutate(
    stage1_k1_score = ifelse(is.na(stage1_k1_score), 0, stage1_k1_score),
    stage1_k5_score = ifelse(is.na(stage1_k5_score), 0, stage1_k5_score),
    stage1_k10_score = ifelse(is.na(stage1_k10_score), 0, stage1_k10_score),
    stage2_k1_score = ifelse(is.na(stage2_k1_score), 0, stage2_k1_score),
    stage2_k5_score = ifelse(is.na(stage2_k5_score), 0, stage2_k5_score),
    stage2_k10_score = ifelse(is.na(stage2_k10_score), 0, stage2_k10_score),
    stage3_k1_score = ifelse(is.na(stage3_k1_score), 0, stage3_k1_score),
    stage3_k5_score = ifelse(is.na(stage3_k5_score), 0, stage3_k5_score),
    stage3_k10_score = ifelse(is.na(stage3_k10_score), 0, stage3_k10_score)
  ) %>%
  # Calculate points based on the provided mechanism
  mutate(
    # Original stage 1 and 2 weights
    points_stage1_k1 = stage1_k1_score * 15,
    points_stage1_k5 = stage1_k5_score * 5,
    points_stage1_k10 = stage1_k10_score * 2,
    points_stage2_k1 = stage2_k1_score * 8,
    points_stage2_k5 = stage2_k5_score * 3,
    points_stage2_k10 = stage2_k10_score * 1,
    # New stage 3 weights (assuming 1/2 of stage 2 weights)
    points_stage3_k1 = stage3_k1_score * 4,
    points_stage3_k5 = stage3_k5_score * 1.5,
    points_stage3_k10 = stage3_k10_score * 0.5,
    # Total points calculation including all 3 stages
    total_points = points_stage1_k1 + points_stage1_k5 + points_stage1_k10 +
      points_stage2_k1 + points_stage2_k5 + points_stage2_k10 +
      points_stage3_k1 + points_stage3_k5 + points_stage3_k10
  ) %>%
  arrange(desc(total_points), model) %>% # Rank by total_points, then model name for ties
  pull(model) %>%
  unique()

# Get the top-ranked model for comparisons
top_model <- ranked_models_by_points[1]
cat("Top-ranked model for comparisons:", top_model, "\n")

# --- 6. Statistical Analysis: Within-Model Comparisons ---
# Compare Top-1 vs Top-5 and Top-5 vs Top-10 within each model for each stage

# Function to perform paired Wilcoxon test
perform_paired_test <- function(data, metric1, metric2) {
  # Extract the data for the two metrics
  values1 <- data[[metric1]]
  values2 <- data[[metric2]]
  
  # Perform paired Wilcoxon test
  test_result <- wilcox.test(values1, values2, paired = TRUE)
  
  # Return p-value
  return(test_result$p.value)
}

# Initialize empty dataframe for results
within_model_comparisons <- data.frame()

# Loop through each stage
for (stage in c(1, 2, 3)) {
  stage_data <- all_data %>% filter(stage_numeric == stage)
  
  # Loop through each model
  for (model_name in unique(stage_data$model)) {
    model_data <- stage_data %>% filter(model == model_name)
    
    # Skip if not enough data
    if (nrow(model_data) < 3) {
      next
    }
    
    # Compare Top-1 vs Top-5
    p_value_1_5 <- perform_paired_test(
      model_data, 
      "differential_top_1_accuracy", 
      "differential_top_5_accuracy"
    )
    
    # Compare Top-5 vs Top-10
    p_value_5_10 <- perform_paired_test(
      model_data, 
      "differential_top_5_accuracy", 
      "differential_top_10_accuracy"
    )
    
    # Add results to dataframe
    within_model_comparisons <- rbind(
      within_model_comparisons,
      data.frame(
        Model = model_name,
        Stage = paste("Stage", stage),
        Comparison = "Top-1 vs Top-5",
        P_Value = p_value_1_5,
        Significant = p_value_1_5 < 0.05
      ),
      data.frame(
        Model = model_name,
        Stage = paste("Stage", stage),
        Comparison = "Top-5 vs Top-10",
        P_Value = p_value_5_10,
        Significant = p_value_5_10 < 0.05
      )
    )
  }
}

# Format p-values for readability
within_model_comparisons$P_Value_Formatted <- ifelse(
  within_model_comparisons$P_Value < 0.001, 
  "< 0.001",
  sprintf("%.3f", within_model_comparisons$P_Value)
)

# Add significance stars
within_model_comparisons$Stars <- ifelse(
  within_model_comparisons$P_Value < 0.001, "***",
  ifelse(within_model_comparisons$P_Value < 0.01, "**",
         ifelse(within_model_comparisons$P_Value < 0.05, "*", "ns"))
)

# Sort by model (using the same order as in the ranking), stage, and comparison
within_model_comparisons$Model <- factor(
  within_model_comparisons$Model, 
  levels = ranked_models_by_points
)
within_model_comparisons$Stage <- factor(
  within_model_comparisons$Stage,
  levels = c("Stage 1", "Stage 2", "Stage 3")
)
within_model_comparisons <- within_model_comparisons %>%
  arrange(Model, Stage, Comparison)

# --- 7. Statistical Analysis: Between-Model Comparisons ---
# Compare the top model against all others for Top-1 accuracy in each stage

# Function to perform unpaired Wilcoxon test
perform_unpaired_test <- function(data1, data2, metric) {
  # Extract the data for the metric
  values1 <- data1[[metric]]
  values2 <- data2[[metric]]
  
  # Perform unpaired Wilcoxon test
  test_result <- wilcox.test(values1, values2, paired = FALSE)
  
  # Return p-value
  return(test_result$p.value)
}

# Initialize empty dataframe for results
between_model_comparisons <- data.frame()

# Loop through each stage
for (stage in c(1, 2, 3)) {
  stage_data <- all_data %>% filter(stage_numeric == stage)
  
  # Get data for the top model
  top_model_data <- stage_data %>% filter(model == top_model)
  
  # Skip if not enough data for top model
  if (nrow(top_model_data) < 3) {
    next
  }
  
  # Loop through each model (except the top model)
  for (model_name in setdiff(unique(stage_data$model), top_model)) {
    model_data <- stage_data %>% filter(model == model_name)
    
    # Skip if not enough data
    if (nrow(model_data) < 3) {
      next
    }
    
    # Compare Top-1 accuracy
    p_value_top1 <- perform_unpaired_test(
      top_model_data, 
      model_data, 
      "differential_top_1_accuracy"
    )
    
    # Add results to dataframe
    between_model_comparisons <- rbind(
      between_model_comparisons,
      data.frame(
        Stage = paste("Stage", stage),
        Metric = "Top-1 Accuracy",
        Comparison = paste(top_model, "vs", model_name),
        P_Value = p_value_top1,
        Significant = p_value_top1 < 0.05
      )
    )
  }
}

# Format p-values for readability
between_model_comparisons$P_Value_Formatted <- ifelse(
  between_model_comparisons$P_Value < 0.001, 
  "< 0.001",
  sprintf("%.3f", between_model_comparisons$P_Value)
)

# Add significance stars
between_model_comparisons$Stars <- ifelse(
  between_model_comparisons$P_Value < 0.001, "***",
  ifelse(between_model_comparisons$P_Value < 0.01, "**",
         ifelse(between_model_comparisons$P_Value < 0.05, "*", "ns"))
)

# Sort by stage and p-value
between_model_comparisons$Stage <- factor(
  between_model_comparisons$Stage,
  levels = c("Stage 1", "Stage 2", "Stage 3")
)
between_model_comparisons <- between_model_comparisons %>%
  arrange(Stage, P_Value)

# --- 8. Save Results ---
# Save the within-model comparisons
within_model_file <- file.path(output_dir, "statistical_comparison_within_model_by_stage.csv")
write.csv(within_model_comparisons, within_model_file, row.names = FALSE)
cat("Within-model statistical comparisons saved to:", within_model_file, "\n")

# Save the between-model comparisons
between_model_file <- file.path(output_dir, "statistical_comparison_between_models_top1_by_stage.csv")
write.csv(between_model_comparisons, between_model_file, row.names = FALSE)
cat("Between-model statistical comparisons saved to:", between_model_file, "\n")

# --- 9. Print Summary ---
cat("\n--- Summary of Within-Model Comparisons ---\n")
cat("Number of significant differences (p < 0.05):", sum(within_model_comparisons$Significant), "out of", nrow(within_model_comparisons), "\n")

cat("\n--- Summary of Between-Model Comparisons ---\n")
cat("Number of significant differences (p < 0.05):", sum(between_model_comparisons$Significant), "out of", nrow(between_model_comparisons), "\n")

cat("\nStatistical analysis complete. Results saved to CSV files.\n")
