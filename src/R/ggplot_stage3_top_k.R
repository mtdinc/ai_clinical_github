library(dplyr)
library(ggplot2)
library(tidyr)
library(forcats)

# Create results directory if it doesn't exist
if (!dir.exists("results")) {
  dir.create("results")
  cat("Created 'results' directory for saving output files\n")
}

# Define dataset prefix for file naming
dataset_prefix <- "Results_typical_cases_CPS_concise_selected"

# Construct file paths
file_s1 <- paste0(dataset_prefix, "_stage_1_long_format_ordered_top_k_rated.csv")
file_s2 <- paste0(dataset_prefix, "_stage_2_long_format_ordered_top_k_rated.csv")
file_s3 <- paste0(dataset_prefix, "_stage_3_long_format_ordered_top_k_rated.csv")

cat("Reading data files:\n")
cat("Stage 1:", file_s1, "\n")
cat("Stage 2:", file_s2, "\n")
cat("Stage 3:", file_s3, "\n")

# Read the CSV files
s1_data <- read.csv(file_s1)
s2_data <- read.csv(file_s2)
s3_data <- read.csv(file_s3)

# Add stage identifier
s1_data$stage_numeric <- 1
s2_data$stage_numeric <- 2
s3_data$stage_numeric <- 3

# Combine all data
all_data <- rbind(s1_data, s2_data, s3_data)

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

# Create stage factor for better labeling
all_data$stage_factor <- factor(
  all_data$stage_numeric,
  levels = c(1, 2, 3),
  labels = c("Stage 1", "Stage 2", "Stage 3")
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
    # Convert k_metric to a factor with specific order
    k_metric = factor(k_metric, levels = c("Top-1 Accuracy", "Top-5 Accuracy", "Top-10 Accuracy"))
  )

# Rank models based on Stage 3 Top-1 Accuracy
model_ranking <- model_scores %>%
  filter(stage_numeric == 3) %>%
  arrange(desc(k1_score)) %>%
  pull(model)

# Apply ranking to the plotting data
model_scores_long$model <- factor(model_scores_long$model, levels = model_ranking)

# Prepare data for layered bars
# We need to create a special dataset where each stage's bar only shows the increment over the previous stage
layered_data <- model_scores_long %>%
  select(model, stage_numeric, stage_factor, k_metric, score) %>%
  group_by(model, k_metric) %>%
  arrange(stage_numeric) %>%
  # Calculate the height of each segment
  mutate(
    # For stage 1, the height is just the score
    # For stage 2, the height is the difference between stage 2 and stage 1 (if positive)
    # For stage 3, the height is the difference between stage 3 and stage 2 (if positive)
    segment_height = case_when(
      stage_numeric == 1 ~ score,
      stage_numeric == 2 ~ pmax(0, score - lag(score, 1, default = 0)),
      stage_numeric == 3 ~ pmax(0, score - lag(score, 1, default = 0))
    ),
    # Calculate the y-position where each segment starts
    segment_start = case_when(
      stage_numeric == 1 ~ 0,
      stage_numeric == 2 ~ lag(score, 1, default = 0),
      stage_numeric == 3 ~ lag(score, 1, default = 0)
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
bar_group_width <- 0.8  # Total width for a group of 3 bars (K1, K5, K10) for one model
individual_bar_width <- bar_group_width / 3  # Width of each individual K-metric bar
bar_spacing <- 0.05  # Space between bars within a group

# Create the plot
p <- ggplot() +
  # Draw the bars for each stage
  geom_rect(
    data = layered_data,
    aes(
      # Position bars side by side for each model based on k_numeric
      xmin = model_numeric + (k_numeric - 2) * individual_bar_width - individual_bar_width/2 + bar_spacing/2,
      xmax = model_numeric + (k_numeric - 2) * individual_bar_width + individual_bar_width/2 - bar_spacing/2,
      ymin = segment_start,
      ymax = segment_start + segment_height,
      fill = stage_factor
    )
  ) +
  # Add error bars for Stage 3 (final) scores
  geom_errorbar(
    data = model_scores_long %>% filter(stage_numeric == 3),
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
  # Add text labels for Stage 3 scores
  geom_text(
    data = model_scores_long %>% filter(stage_numeric == 3),
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
  
  # Add K-metric labels below each group of bars
  geom_text(
    data = model_scores_long %>% 
      filter(stage_numeric == 3) %>%
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
  
  # Set colors for stages
  scale_fill_manual(
    values = c(
      "Stage 1" = "#4DBBD5", # Light blue
      "Stage 2" = "#E64B35", # Light red/orange
      "Stage 3" = "#3C8C3E"  # Green
    )
  ) +
  
  # Set x-axis to use model names
  scale_x_continuous(
    breaks = 1:length(levels(model_scores_long$model)),
    labels = levels(model_scores_long$model),
    expand = c(0.05, 0.05)
  ) +
  # Set y-axis to percentage format
  scale_y_continuous(
    limits = c(-0.05, 1),  # Extended lower limit to make room for K1/K5/K10 labels
    breaks = seq(0, 1, 0.1),
    labels = scales::percent_format(accuracy = 1)
  ) +
  # Theme customization
  theme_minimal(base_size = 12) +
  theme(
    # Fix model names on x-axis with angle
    axis.text.x = element_text(size = 8, face = "bold", angle = 45, hjust = 1, vjust = 1),
    axis.ticks.x = element_blank(),
    # Ensure y-axis labels are readable
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
  # Labels
  labs(
    title = "Model Performance Comparison Across Stages",
    subtitle = "Models Ranked by Top-1 Accuracy at Stage 3 | K1, K5, K10 Accuracy Side-by-Side",
    y = "Accuracy Score",
    x = "",
    fill = "Information Stage"
  )

# Save plot
ggsave("results/model_performance_top_k_comparison.pdf", plot = p, width = 10, height = 12, dpi = 300)
ggsave("results/model_performance_top_k_comparison.png", plot = p, width = 10, height = 12, dpi = 300)

cat("Plots saved to 'results/model_performance_top_k_comparison.pdf' and 'results/model_performance_top_k_comparison.png'\n")

# Prepare data for table
table_data <- model_scores_long %>%
  mutate(
    score = sprintf("%.3f", score),
    se = sprintf("%.3f", se)
  ) %>%
  select(model, stage_factor, k_metric, score, se) %>%
  arrange(k_metric, desc(as.character(model)))

# Write table to CSV
write.csv(table_data, file = "results/model_performance_top_k_table.csv", row.names = FALSE)

cat("Performance metrics table saved to 'results/model_performance_top_k_table.csv'\n")

# Print summary of top performers
top_performers <- model_scores %>%
  filter(stage_numeric == 3) %>%
  arrange(desc(k1_score)) %>%
  head(5)

cat("\nTop 5 performers based on Stage 3 Top-1 Accuracy:\n")
print(top_performers %>% select(model, k1_score, k5_score, k10_score))
