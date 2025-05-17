library(dplyr)
library(ggplot2)
library(tidyr)
library(forcats)
library(scales) # Ensure scales is loaded for percent_format

# Create results directory if it doesn't exist
if (!dir.exists("results")) {
  dir.create("results")
  cat("Created 'results' directory for saving output files\n")
}

# Define dataset prefix for file naming
dataset_prefix <- "Results_typical_cases_typical_concise_selected"

# Construct file paths for Stage 1 and Stage 2
file_s1 <- paste0(dataset_prefix, "_stage_1_long_format_ordered_top_k_rated.csv")
file_s2 <- paste0(dataset_prefix, "_stage_2_long_format_ordered_top_k_rated.csv")

cat("Reading data files:\n")
cat("Stage 1:", file_s1, "\n")
cat("Stage 2:", file_s2, "\n")

# Read the CSV files
s1_data <- read.csv(file_s1)
s2_data <- read.csv(file_s2)

# Add stage identifier
s1_data$stage_numeric <- 1
s2_data$stage_numeric <- 2

# Combine Stage 1 and Stage 2 data
all_data <- rbind(s1_data, s2_data)

# Create a mapping of technical model names to simplified display names
model_name_mapping <- data.frame(
  model = c(
    # Anthropic models
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-20250219_thinking",
    "claude-3-opus-20240229",
    
    # OpenAI models
    "chatgpt-4o-latest",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-0613",
    "gpt-3.5-turbo-0125",
    
    # OpenAI with reasoning effort
    "o1-2024-12-17_high",
    "o1-2024-12-17_medium",
    "o1-mini-2024-09-12",
    "o1-preview-2024-09-12",
    "o3-mini-2025-01-31_high",
    "o3-mini-2025-01-31_medium",
    
    # Google models
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b-001",
    "gemini-1.5-pro-002",
    "gemini-2.0-flash-001",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-flash-thinking-exp-01-21"
  ),
  display_name = c(
    # Anthropic models
    "Claude 3.5 Sonnet",
    "Claude 3.5 Haiku",
    "Claude 3.7 Sonnet",
    "Claude 3.7 Thinking",
    "Claude 3 Opus",
    
    # OpenAI models
    "GPT-4o Latest",
    "GPT-4o",
    "GPT-4o Mini",
    "GPT-4",
    "GPT-3.5 Turbo",
    
    # OpenAI with reasoning effort
    "O1 (High)",
    "O1 (Medium)",
    "O1 Mini",
    "O1 Preview",
    "O3-mini (High)",
    "O3-mini (Medium)",
    
    # Google models
    "Gemini 1.5 Flash",
    "Gemini 1.5 Flash 8B",
    "Gemini 1.5 Pro",
    "Gemini 2.0 Flash",
    "Gemini 2.0 Pro",
    "Gemini 2.0 Thinking"
  )
)

# Apply the mapping to the data
all_data <- all_data %>%
  left_join(model_name_mapping, by = "model") %>%
  mutate(model = ifelse(is.na(display_name), model, display_name)) %>%
  select(-display_name)

# Replace NA values with 0
all_data <- replace(all_data, is.na(all_data), 0)

# Create stage factor for better labeling (Stages 1 and 2 only)
all_data$stage_factor <- factor(
  all_data$stage_numeric,
  levels = c(1, 2),
  labels = c("Stage 1", "Stage 2")
)

# Calculate mean scores for each model, stage, and k-metric
model_scores <- all_data %>%
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

# Pivot to long format for plotting
model_scores_long <- model_scores %>%
  pivot_longer(
    cols = c(k1_score, k5_score, k10_score),
    names_to = "k_metric",
    values_to = "score"
  ) %>%
  mutate(
    se = case_when(
      k_metric == "k1_score" ~ k1_se,
      k_metric == "k5_score" ~ k5_se,
      k_metric == "k10_score" ~ k10_se
    ),
    k_metric = case_when(
      k_metric == "k1_score" ~ "Top-1 Accuracy",
      k_metric == "k5_score" ~ "Top-5 Accuracy",
      k_metric == "k10_score" ~ "Top-10 Accuracy"
    ),
    k_metric = factor(k_metric, levels = c("Top-1 Accuracy", "Top-5 Accuracy", "Top-10 Accuracy"))
  )

# Rank models based on Stage 2 Top-1 Accuracy
model_ranking <- model_scores %>%
  filter(stage_numeric == 2) %>%
  arrange(desc(k1_score)) %>%
  pull(model)

# Apply ranking to the plotting data
model_scores_long$model <- factor(model_scores_long$model, levels = model_ranking)

# Prepare data for layered bars (Stages 1 and 2)
layered_data <- model_scores_long %>%
  select(model, stage_numeric, stage_factor, k_metric, score) %>%
  group_by(model, k_metric) %>%
  arrange(stage_numeric) %>%
  mutate(
    segment_height = case_when(
      stage_numeric == 1 ~ score,
      stage_numeric == 2 ~ pmax(0, score - lag(score, 1, default = 0))
    ),
    segment_start = case_when(
      stage_numeric == 1 ~ 0,
      stage_numeric == 2 ~ lag(score, 1, default = 0)
    )
  ) %>%
  ungroup()

# Add a numeric index for each k_metric for positioning
layered_data <- layered_data %>%
  mutate(
    k_numeric = as.numeric(factor(k_metric, levels = c("Top-1 Accuracy", "Top-5 Accuracy", "Top-10 Accuracy"))),
    model_numeric = as.numeric(model)
  )

# Also add k_numeric to model_scores_long for error bars and labels
model_scores_long <- model_scores_long %>%
  mutate(
    k_numeric = as.numeric(factor(k_metric, levels = c("Top-1 Accuracy", "Top-5 Accuracy", "Top-10 Accuracy"))),
    model_numeric = as.numeric(model)
  )

# Define constants for bar positioning
bar_group_width <- 0.8
individual_bar_width <- bar_group_width / 3
bar_spacing <- 0.05


# Create a points-based ranking system
model_ranking_points <- model_scores %>%
  # First, reshape the data to have one row per model with all metrics
  select(model, stage_numeric, k1_score, k5_score, k10_score) %>%
  pivot_wider(
    names_from = stage_numeric,
    values_from = c(k1_score, k5_score, k10_score),
    names_glue = "stage{stage_numeric}_{.value}"
  ) %>%
  # Now calculate points
  mutate(
    # Assign points with Stage 1 K1 having highest weight
    points_stage1_k1 = stage1_k1_score * 12,  # Highest points
    points_stage1_k5 = stage1_k5_score * 5,
    points_stage1_k10 = stage1_k10_score * 2,
    points_stage2_k1 = stage2_k1_score * 8,
    points_stage2_k5 = stage2_k5_score * 3,
    points_stage2_k10 = stage2_k10_score * 1,
    
    # Calculate total points
    total_points = points_stage1_k1 + points_stage1_k5 + points_stage1_k10 + 
      points_stage2_k1 + points_stage2_k5 + points_stage2_k10
  ) %>%
  # Use distinct to ensure no duplicates
  distinct(model, .keep_all = TRUE) %>%
  arrange(desc(total_points)) %>%
  pull(model)

# Apply the ranking to the plotting data
model_scores_long$model <- factor(model_scores_long$model, levels = model_ranking_points)
layered_data$model <- factor(layered_data$model, levels = model_ranking_points)


# Create the plot
p <- ggplot() +
  geom_rect(
    data = layered_data,
    aes(
      xmin = model_numeric + (k_numeric - 2) * individual_bar_width - individual_bar_width/2 + bar_spacing/2,
      xmax = model_numeric + (k_numeric - 2) * individual_bar_width + individual_bar_width/2 - bar_spacing/2,
      ymin = segment_start,
      ymax = segment_start + segment_height,
      fill = stage_factor
    )
  ) +
  geom_errorbar(
    data = model_scores_long %>% filter(stage_numeric == 2), # Error bars for Stage 2
    aes(
      x = model_numeric + (k_numeric - 2) * individual_bar_width,
      y = score,
      ymin = pmax(score - se, 0),
      ymax = pmin(score + se, 1),
      group = interaction(model, k_metric)
    ),
    width = 0.1,
    color = "black",
    alpha = 0.7
  ) +
  geom_text(
    data = model_scores_long %>% filter(stage_numeric == 2), # Text labels for Stage 2
    aes(
      x = model_numeric + (k_numeric - 2) * individual_bar_width,
      y = score + 0.03,
      label = scales::percent(score, accuracy = 1),
      group = interaction(model, k_metric)
    ),
    size = 2.5,
    angle = 0,
    hjust = 0.5,
    vjust = 0
  ) +
  geom_text(
    data = model_scores_long %>% 
      filter(stage_numeric == 2) %>% # K-metric labels based on Stage 2 data
      group_by(model, k_metric, k_numeric, model_numeric) %>%
      summarize(n = n(), .groups = "drop") %>%
      distinct(),
    aes(
      x = model_numeric + (k_numeric - 2) * individual_bar_width,
      y = -0.05,
      label = case_when(
        k_numeric == 1 ~ "K1",
        k_numeric == 2 ~ "K5",
        k_numeric == 3 ~ "K10"
      )
    ),
    size = 2,
    angle = 0,
    hjust = 0.5,
    vjust = 1
  ) +
  scale_fill_manual(
    values = c(
      "Stage 1" = "#4DBBD5", 
      "Stage 2" = "#E64B35"
    )
  ) +
  scale_x_continuous(
    breaks = 1:length(levels(model_scores_long$model)),
    labels = levels(model_scores_long$model),
    expand = c(0.05, 0.05)
  ) +
  scale_y_continuous(
    limits = c(-0.05, 1),
    breaks = seq(0, 1, 0.1),
    labels = scales::percent_format(accuracy = 1)
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(size = 8, face = "bold", angle = 45, hjust = 1, vjust = 1),
    axis.ticks.x = element_blank(),
    axis.text.y = element_text(size = 9, angle = 0),
    axis.title = element_text(size = 11, face = "bold"),
    legend.title = element_text(face = "bold"),
    legend.position = "top",
    legend.key.size = unit(1, "lines"),
    panel.grid.major.y = element_line(color = "gray90"),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    panel.border = element_rect(fill = NA, color = "gray90"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5),
    plot.margin = margin(t = 15, r = 20, b = 15, l = 10),
    plot.background = element_rect(fill = "white", color = NA)
  ) +
  labs(
    title = "Model Performance Comparison (Stages 1 & 2)",
    subtitle = "Models Ranked by Top-1 Accuracy at Stage 2 | K1, K5, K10 Accuracy Side-by-Side",
    y = "Accuracy Score",
    x = "",
    fill = "Information Stage"
  )


# Save plot with updated filenames
ggsave("results/model_performance_stage1_stage2_top_k_comparison.pdf", plot = p, width = 10, height = 12, dpi = 300)
ggsave("results/model_performance_stage1_stage2_top_k_comparison.png", plot = p, width = 10, height = 12, dpi = 300)

cat("Plots saved to 'results/model_performance_stage1_stage2_top_k_comparison.pdf' and 'results/model_performance_stage1_stage2_top_k_comparison.png'\n")

# Prepare data for table
table_data <- model_scores_long %>%
  mutate(
    score = sprintf("%.3f", score),
    se = sprintf("%.3f", se)
  ) %>%
  select(model, stage_factor, k_metric, score, se) %>%
  arrange(k_metric, desc(as.character(model)))

# Write table to CSV with updated filename
write.csv(table_data, file = "results/model_performance_stage1_stage2_top_k_table.csv", row.names = FALSE)

cat("Performance metrics table saved to 'results/model_performance_stage1_stage2_top_k_table.csv'\n")

# Print summary of top performers based on Stage 2
top_performers <- model_scores %>%
  filter(stage_numeric == 2) %>% # Filter for Stage 2
  arrange(desc(k1_score)) %>%
  head(5)

cat("\nTop 5 performers based on Stage 2 Top-1 Accuracy:\n")
print(top_performers %>% select(model, k1_score, k5_score, k10_score))
