# Load necessary libraries
library(dplyr)
library(ggplot2)
library(tidyr) # For pivot_longer, pivot_wider
library(forcats) # For factor manipulation
library(scales)  # For percent_format

# --- 1. Configuration & Setup ---

# Create results directory if it doesn't exist
output_dir <- "results_simplified_weighted_rank" # New output directory
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
    .groups = 'drop'
  )

model_scores_long <- model_scores_summary %>%
  pivot_longer(
    cols = starts_with("k") & ends_with("_score"),
    names_to = "k_metric_type",
    values_to = "score"
  ) %>%
  mutate(
    se = case_when(
      k_metric_type == "k1_score" ~ k1_se,
      k_metric_type == "k5_score" ~ k5_se,
      k_metric_type == "k10_score" ~ k10_se
    ),
    k_metric = factor(
      k_metric_type,
      levels = c("k1_score", "k5_score", "k10_score"),
      labels = c("Top-1 Accuracy", "Top-5 Accuracy", "Top-10 Accuracy")
    )
  ) %>%
  select(model, stage_numeric, stage_factor, k_metric, score, se)

# --- 5. Model Ranking (Weighted Point System) ---
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

# Apply this ranking to the data for plotting
model_scores_long$model <- factor(model_scores_long$model, levels = ranked_models_by_points)

# --- 6. Prepare Data for Layered Bar Segments ---
# We need to calculate the segment heights a bit differently for 3 stages
layered_plot_data <- model_scores_long %>%
  arrange(model, k_metric, stage_numeric) %>%
  group_by(model, k_metric) %>%
  mutate(
    # For each stage, calculate where the segment starts and its height
    segment_start = case_when(
      stage_numeric == 1 ~ 0,
      stage_numeric == 2 ~ lag(score, n = 1, default = 0),
      stage_numeric == 3 ~ lag(score, n = 1, default = 0)
    ),
    segment_height = case_when(
      stage_numeric == 1 ~ score,
      stage_numeric == 2 ~ score - lag(score, n = 1, default = 0),
      stage_numeric == 3 ~ score - lag(score, n = 1, default = 0)
    )
  ) %>%
  ungroup() %>%
  mutate(segment_height = pmax(0, segment_height))

layered_plot_data <- layered_plot_data %>%
  mutate(
    model_numeric = as.numeric(model),
    k_numeric = as.numeric(k_metric)
  )

model_scores_long_for_plot <- model_scores_long %>%
  mutate(
    model_numeric = as.numeric(model),
    k_numeric = as.numeric(k_metric)
  )

# --- 7. Plotting ---
bar_group_width <- 0.8
individual_bar_width <- bar_group_width / 3
bar_spacing <- 0.03

# Update the color palette for 3 stages
stage_colors <- c("Stage 1" = "#4DBBD5", "Stage 2" = "#E64B35", "Stage 3" = "#00A087")

p <- ggplot() +
  geom_rect(
    data = layered_plot_data,
    aes(
      xmin = model_numeric + (k_numeric - 2) * individual_bar_width - (individual_bar_width / 2) + (bar_spacing / 2),
      xmax = model_numeric + (k_numeric - 2) * individual_bar_width + (individual_bar_width / 2) - (bar_spacing / 2),
      ymin = segment_start,
      ymax = segment_start + segment_height,
      fill = stage_factor
    ),
    color = "grey50",
    linewidth = 0.2
  ) +
  geom_errorbar(
    data = model_scores_long_for_plot %>% filter(stage_numeric == 3),
    aes(
      x = model_numeric + (k_numeric - 2) * individual_bar_width,
      y = score,
      ymin = pmax(0, score - se),
      ymax = pmin(1, score + se),
      group = interaction(model, k_metric)
    ),
    width = individual_bar_width * 0.3,
    color = "black",
    alpha = 0.7,
    linewidth = 0.5
  ) +
  geom_text(
    data = model_scores_long_for_plot %>% filter(stage_numeric == 3),
    aes(
      x = model_numeric + (k_numeric - 2) * individual_bar_width,
      y = pmin(1, score + se) + 0.02,
      label = scales::percent(score, accuracy = 1),
      group = interaction(model, k_metric)
    ),
    size = 2.5,
    hjust = 0.5,
    vjust = 0
  ) +
  geom_text(
    data = distinct(layered_plot_data, model, model_numeric, k_metric, k_numeric),
    aes(
      x = model_numeric + (k_numeric - 2) * individual_bar_width,
      y = -0.03,
      label = gsub(" Accuracy", "", k_metric)
    ),
    size = 2.2,
    hjust = 0.5,
    vjust = 1
  ) +
  scale_fill_manual(
    name = "Information Stage",
    values = stage_colors
  ) +
  scale_x_continuous(
    breaks = 1:length(ranked_models_by_points),
    labels = ranked_models_by_points,
    expand = expansion(add = 0.5)
  ) +
  scale_y_continuous(
    limits = c(-0.05, 1.05),
    breaks = seq(0, 1, 0.1),
    labels = scales::percent_format(accuracy = 1)
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, face = "bold", size = 8),
    axis.text.y = element_text(size = 9),
    axis.title.y = element_text(size = 10, face = "bold", margin = margin(r = 10)),
    axis.title.x = element_blank(),
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 10),
    legend.text = element_text(size = 9),
    legend.key.size = unit(0.8, "lines"),
    panel.grid.major.y = element_line(color = "gray90"),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_blank(),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5, margin = margin(b = 15)), # Updated subtitle
    plot.margin = margin(t = 15, r = 20, b = 20, l = 15)
  ) +
  labs(
    title = "Model Performance Comparison (Stages 1, 2 & 3) - Complex Cases",
    subtitle = "Models Ranked by Weighted Performance Score | K1, K5, K10 Accuracy Side-by-Side", 
    y = "Accuracy Score"
  )

print(p)

# --- 8. Save Outputs ---
plot_filename_base <- file.path(output_dir, "model_performance_stage1_stage2_stage3_top_k_weighted_rank") 
ggsave(paste0(plot_filename_base, ".pdf"), plot = p, width = 11, height = 5, dpi = 300)
ggsave(paste0(plot_filename_base, ".png"), plot = p, width = 11, height = 5, dpi = 300)
cat("Plots saved to '", plot_filename_base, ".pdf' and '.png'\n")

summary_table_data <- model_scores_long %>%
  mutate(
    score = round(score, 3),
    se = round(se, 3)
  ) %>%
  select(model, stage_factor, k_metric, score, se) %>%
  # Apply the same factor ordering for the table as for the plot
  mutate(model = factor(model, levels = ranked_models_by_points)) %>%
  arrange(model, k_metric, stage_factor)

table_filename <- file.path(output_dir, "model_performance_stage1_stage2_stage3_top_k_table_weighted_rank.csv")
write.csv(summary_table_data, file = table_filename, row.names = FALSE)
cat("Performance metrics table saved to '", table_filename, "'\n")

# Optional: Print top performers based on the new ranking
cat("\nTop 5 performers based on Weighted Point System:\n")
print(head(ranked_models_by_points, 5))
