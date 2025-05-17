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

# Define dataset prefixes
complex_prefix <- "Results_typical_cases_CPS_concise_selected"
common_prefix <- "Results_typical_cases_typical_concise_selected"

# --- 2. Load and Prepare Data ---

# Function to load and prepare data for a specific case type
load_and_prepare_data <- function(dataset_prefix, stages, case_type_label) {
  cat("Loading data for", case_type_label, "with stages", paste(stages, collapse=", "), "\n")
  
  # Load data from all stages
  all_data <- data.frame()
  for (stage in stages) {
    file_path <- paste0(dataset_prefix, "_stage_", stage, "_long_format_ordered_top_k_rated.csv")
    cat("  Reading Stage", stage, "data from:", file_path, "\n")
    stage_data <- read.csv(file_path)
    all_data <- bind_rows(all_data, stage_data)
  }
  
  # Add case type label
  all_data$case_type <- case_type_label
  
  return(all_data)
}

# Load data for both case types
complex_data <- load_and_prepare_data(
  dataset_prefix = complex_prefix,
  stages = c(1, 2, 3),
  case_type_label = "Complex Cases"
)

common_data <- load_and_prepare_data(
  dataset_prefix = common_prefix,
  stages = c(1, 2),
  case_type_label = "Common Cases"
)

# Combine all data
all_data <- bind_rows(complex_data, common_data)

# Extract stage number from the stage column
all_data <- all_data %>%
  mutate(
    stage_numeric = case_when(
      grepl("stage-1", stage) ~ 1,
      grepl("stage-2", stage) ~ 2,
      grepl("stage-3", stage) ~ 3,
      TRUE ~ NA_integer_
    )
  )

# Check if stage_numeric was created correctly
cat("\nStage numeric values:\n")
print(table(all_data$stage_numeric, useNA = "ifany"))

# --- 3. Model Name Mapping ---

# Define model name mapping (same as in ggplot_combined_cases.R)
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

# Apply model name mapping
all_data <- all_data %>%
  left_join(model_name_mapping, by = "model") %>%
  mutate(model = ifelse(is.na(display_name), model, display_name)) %>%
  select(-display_name)

# --- 4. Identify Top Models ---

# Check the structure of the data to understand column types
str(all_data)

# Function to identify top models for a specific case type and stage
identify_top_models <- function(data, case_type, stage_num, top_n = 5) {
  # Filter data for the specified case type and stage
  filtered_data <- data %>%
    filter(case_type == !!case_type, stage_numeric == !!stage_num)
  
  # Calculate mean Top-1 accuracy for each model
  model_performance <- filtered_data %>%
    group_by(model) %>%
    summarize(
      mean_top1 = mean(differential_top_1_accuracy, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(mean_top1))
  
  # Get top N models
  top_models <- head(model_performance$model, top_n)
  
  return(top_models)
}

# Identify top models for each case type at their final stage
complex_top_models <- identify_top_models(all_data, "Complex Cases", 3)
common_top_models <- identify_top_models(all_data, "Common Cases", 2)

# If no top models were found, use a default list of models
if (length(complex_top_models) == 0) {
  cat("Warning: No top models found for Complex Cases. Using default list.\n")
  complex_top_models <- c("Claude 3.7 Thinking", "Claude 3.7 Sonnet", "O1 (Medium)", 
                         "O3-mini (Medium)", "GPT-4o")
}

if (length(common_top_models) == 0) {
  cat("Warning: No top models found for Common Cases. Using default list.\n")
  common_top_models <- c("Claude 3.7 Sonnet", "Claude 3.7 Thinking", "Claude 3 Opus", 
                        "GPT-4o", "O1 (Medium)")
}

cat("Top models for Complex Cases (Stage 3):\n")
print(complex_top_models)

cat("\nTop models for Common Cases (Stage 2):\n")
print(common_top_models)

# --- 5. Perform Statistical Comparisons ---

# Function to perform pairwise comparisons for a specific case type and stage
perform_pairwise_comparisons <- function(data, case_type, stage_num, top_models) {
  # Filter data for the specified case type, stage, and top models
  filtered_data <- data %>%
    filter(
      case_type == !!case_type, 
      stage_numeric == !!stage_num,
      model %in% top_models
    )
  
  # Ensure model is a factor with levels in the order of top_models
  filtered_data$model <- factor(filtered_data$model, levels = top_models)
  
  # Perform pairwise Wilcoxon rank-sum tests
  pairwise_tests <- filtered_data %>%
    pairwise_wilcox_test(
      differential_top_1_accuracy ~ model,
      p.adjust.method = "holm"
    )
  
  # Add case type and stage information
  pairwise_tests$case_type <- case_type
  pairwise_tests$stage_num <- stage_num
  
  # Calculate effect size (rank biserial correlation)
  effect_sizes <- filtered_data %>%
    pairwise_wilcox_effsize(
      differential_top_1_accuracy ~ model
    )
  
  # Join test results with effect sizes
  results <- pairwise_tests %>%
    left_join(
      effect_sizes %>% select(group1, group2, effsize),
      by = c("group1", "group2")
    )
  
  return(results)
}

# Perform pairwise comparisons for both case types
complex_comparisons <- perform_pairwise_comparisons(
  all_data, "Complex Cases", 3, complex_top_models
)

common_comparisons <- perform_pairwise_comparisons(
  all_data, "Common Cases", 2, common_top_models
)

# Check if comparisons were generated
if (nrow(complex_comparisons) == 0) {
  cat("Warning: No comparisons generated for Complex Cases.\n")
}

if (nrow(common_comparisons) == 0) {
  cat("Warning: No comparisons generated for Common Cases.\n")
}

# Combine all comparisons
all_comparisons <- bind_rows(complex_comparisons, common_comparisons)

# --- 6. Format Results ---

# Format p-values and add significance indicators
formatted_comparisons <- all_comparisons %>%
  mutate(
    p_formatted = case_when(
      p < 0.001 ~ "< 0.001",
      p < 0.01 ~ sprintf("%.3f", p),
      TRUE ~ sprintf("%.3f", p)
    ),
    significance = case_when(
      p < 0.001 ~ "***",
      p < 0.01 ~ "**",
      p < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  ) %>%
  arrange(case_type, group1, group2)

# --- 7. Save Results ---

# Save the comparison results
comparison_file <- file.path(output_dir, "statistical_comparison_between_models_top1_by_stage.csv")
write.csv(formatted_comparisons, comparison_file, row.names = FALSE)
cat("Statistical comparison results saved to:", comparison_file, "\n")

# --- 8. Additional Analysis: Within-Model Comparison Across Stages ---

# Function to perform within-model comparisons across stages
perform_within_model_comparisons <- function(data, case_type, models_to_include) {
  # Filter data for the specified case type and models
  filtered_data <- data %>%
    filter(
      case_type == !!case_type,
      model %in% models_to_include
    )
  
  # Create a list to store results
  results_list <- list()
  
  # For each model, compare performance across stages
  for (model_name in models_to_include) {
    model_data <- filtered_data %>%
      filter(model == model_name)
    
    # Get unique stages for this case type
    stages <- sort(unique(model_data$stage))
    
    # For each pair of consecutive stages
    for (i in 1:(length(stages)-1)) {
      stage1 <- stages[i]
      stage2 <- stages[i+1]
      
      # Extract data for the two stages
      stage1_data <- model_data %>% filter(stage == stage1) %>% pull(differential_top_1_accuracy)
      stage2_data <- model_data %>% filter(stage == stage2) %>% pull(differential_top_1_accuracy)
      
      # Perform Wilcoxon signed-rank test (paired)
      test_result <- wilcox.test(stage1_data, stage2_data, paired = TRUE)
      
      # Calculate effect size (simplified for paired data)
      n <- length(stage1_data)
      z <- qnorm(1 - test_result$p.value/2)  # Approximation
      effect_size <- z / sqrt(n)
      
      # Store results
      results_list[[length(results_list) + 1]] <- data.frame(
        case_type = case_type,
        model = model_name,
        stage1 = stage1,
        stage2 = stage2,
        p.value = test_result$p.value,
        effsize = effect_size
      )
    }
  }
  
  # Combine all results
  if (length(results_list) > 0) {
    results <- bind_rows(results_list)
    return(results)
  } else {
    return(data.frame())
  }
}

# Perform within-model comparisons for both case types
complex_within_comparisons <- perform_within_model_comparisons(
  all_data, "Complex Cases", complex_top_models
)

common_within_comparisons <- perform_within_model_comparisons(
  all_data, "Common Cases", common_top_models
)

# Combine all within-model comparisons
all_within_comparisons <- bind_rows(complex_within_comparisons, common_within_comparisons)

# Format p-values and add significance indicators
formatted_within_comparisons <- all_within_comparisons %>%
  mutate(
    p_formatted = case_when(
      p.value < 0.001 ~ "< 0.001",
      p.value < 0.01 ~ sprintf("%.3f", p.value),
      TRUE ~ sprintf("%.3f", p.value)
    ),
    significance = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**",
      p.value < 0.05 ~ "*",
      TRUE ~ "ns"
    ),
    comparison = paste0("Stage ", stage1, " vs. Stage ", stage2)
  ) %>%
  arrange(case_type, model, stage1, stage2)

# Save the within-model comparison results
within_comparison_file <- file.path(output_dir, "statistical_comparison_within_model_by_stage.csv")
write.csv(formatted_within_comparisons, within_comparison_file, row.names = FALSE)
cat("Within-model comparison results saved to:", within_comparison_file, "\n")

# --- 9. Print Summary ---

cat("\nSummary of Statistical Comparisons:\n")
cat("1. Between-model comparisons at final stage:\n")
cat("   - Complex Cases (Stage 3): ", nrow(complex_comparisons), "pairwise comparisons\n")
cat("   - Common Cases (Stage 2): ", nrow(common_comparisons), "pairwise comparisons\n")

cat("\n2. Within-model comparisons across stages:\n")
cat("   - Complex Cases: ", nrow(complex_within_comparisons), "comparisons\n")
cat("   - Common Cases: ", nrow(common_within_comparisons), "comparisons\n")

cat("\nSignificant findings (p < 0.05):\n")
cat("1. Between-model comparisons:\n")
significant_between <- formatted_comparisons %>% filter(p < 0.05)
print(significant_between %>% select(case_type, group1, group2, p_formatted, significance))

cat("\n2. Within-model comparisons:\n")
significant_within <- formatted_within_comparisons %>% filter(p.value < 0.05)
print(significant_within %>% select(case_type, model, comparison, p_formatted, significance))
