# Load necessary libraries
library(dplyr)
library(ggplot2)
library(tidyr)    # For pivot_longer, pivot_wider
library(forcats)  # For factor manipulation
library(scales)   # For percent_format
library(patchwork) # For combining plots

# --- 1. Configuration & Setup ---

# Create results directory if it doesn't exist
output_dir <- "results_simplified_weighted_rank" 
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
  cat("Created '", output_dir, "' directory for saving output files\n")
}

# --- 2. Model Name Mapping (Common to both scripts) ---

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

# --- 3. Reusable Plot Generation Function ---

generate_model_plot <- function(dataset_prefix, stages, case_type_label) {
  cat("Generating plot for", case_type_label, "with stages", paste(stages, collapse=", "), "\n")
  
  # Construct file paths and read data
  all_data <- data.frame()
  for (stage in stages) {
    file_path <- paste0(dataset_prefix, "_stage_", stage, "_long_format_ordered_top_k_rated.csv")
    cat("  Reading Stage", stage, "data from:", file_path, "\n")
    stage_data <- read.csv(file_path)
    stage_data$stage_numeric <- stage
    all_data <- bind_rows(all_data, stage_data)
  }
  
  # Apply model name mapping
  all_data <- all_data %>%
    left_join(model_name_mapping, by = "model") %>%
    mutate(model = ifelse(is.na(display_name), model, display_name)) %>%
    select(-display_name) %>%
    mutate(across(starts_with("differential_top_"), ~replace_na(., 0)))
  
  # Create stage factor with consistent levels across all plots
  # This ensures the legend will be properly combined
  all_data$stage_factor <- factor(
    all_data$stage_numeric,
    levels = c(1, 2, 3),
    labels = c("Stage 1", "Stage 2", "Stage 3")
  )
  
  # Calculate Mean Scores and Standard Errors
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
  
  # Convert to long format for plotting
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
  
  # Model Ranking (Weighted Point System)
  # First, pivot wider to get stage-specific scores
  ranked_models_data <- model_scores_summary %>%
    select(model, stage_numeric, k1_score, k5_score, k10_score) %>%
    pivot_wider(
      names_from = stage_numeric,
      values_from = c(k1_score, k5_score, k10_score),
      names_glue = "stage{stage_numeric}_{.value}"
    )
  
  # Handle missing values for point calculation
  for (stage in stages) {
    stage_cols <- c(
      paste0("stage", stage, "_k1_score"),
      paste0("stage", stage, "_k5_score"),
      paste0("stage", stage, "_k10_score")
    )
    for (col in stage_cols) {
      if (!col %in% colnames(ranked_models_data)) {
        ranked_models_data[[col]] <- 0
      } else {
        ranked_models_data[[col]] <- ifelse(is.na(ranked_models_data[[col]]), 0, ranked_models_data[[col]])
      }
    }
  }
  
  # Calculate points based on weights
  ranked_models_data <- ranked_models_data %>%
    mutate(
      # Stage 1 weights
      points_stage1_k1 = stage1_k1_score * 15,
      points_stage1_k5 = stage1_k5_score * 5,
      points_stage1_k10 = stage1_k10_score * 2,
      # Stage 2 weights
      points_stage2_k1 = stage2_k1_score * 8,
      points_stage2_k5 = stage2_k5_score * 3,
      points_stage2_k10 = stage2_k10_score * 1
    )
  
  # Add Stage 3 points if applicable
  if (3 %in% stages) {
    ranked_models_data <- ranked_models_data %>%
      mutate(
        # Stage 3 weights (half of stage 2)
        points_stage3_k1 = stage3_k1_score * 4,
        points_stage3_k5 = stage3_k5_score * 1.5,
        points_stage3_k10 = stage3_k10_score * 0.5
      )
  }
  
  # Calculate total points and rank models
  point_cols <- grep("^points_stage", names(ranked_models_data), value = TRUE)
  ranked_models_data$total_points <- rowSums(ranked_models_data[, point_cols], na.rm = TRUE)
  
  ranked_models_by_points <- ranked_models_data %>%
    arrange(desc(total_points), model) %>%
    pull(model) %>%
    unique()
  
  # Apply ranking to the data for plotting
  model_scores_long$model <- factor(model_scores_long$model, levels = ranked_models_by_points)
  
  # Prepare Data for Layered Bar Segments
  layered_plot_data <- model_scores_long %>%
    arrange(model, k_metric, stage_numeric) %>%
    group_by(model, k_metric) %>%
    mutate(
      # For each stage, calculate where the segment starts and its height
      segment_start = case_when(
        stage_numeric == min(stages) ~ 0,
        TRUE ~ lag(score, n = 1, default = 0)
      ),
      segment_height = case_when(
        stage_numeric == min(stages) ~ score,
        TRUE ~ score - lag(score, n = 1, default = 0)
      )
    ) %>%
    ungroup() %>%
    mutate(segment_height = pmax(0, segment_height))
  
  # Add numeric coordinates for plotting
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
  
  # Define bar dimensions
  bar_group_width <- 0.8
  individual_bar_width <- bar_group_width / 3
  bar_spacing <- 0.03
  
  # Define stage colors
  stage_colors <- c(
    "Stage 1" = "#4DBBD5", 
    "Stage 2" = "#E64B35"
  )
  
  # Add Stage 3 color if applicable
  if (3 %in% stages) {
    stage_colors["Stage 3"] <- "#00A087"
  }
  
  # Create the plot
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
      data = model_scores_long_for_plot %>% filter(stage_numeric == max(stages)),
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
      data = model_scores_long_for_plot %>% filter(stage_numeric == max(stages)),
      aes(
        x = model_numeric + (k_numeric - 2) * individual_bar_width,
        y = pmin(1, score + se) + 0.02,
        label = scales::percent(score, accuracy = 1),
        group = interaction(model, k_metric)
      ),
      size = 2.2,   # Decreased from 3.5
      hjust = 0.5,
      vjust = 0
    ) +
    geom_text(
      data = distinct(layered_plot_data, model, model_numeric, k_metric, k_numeric),
      aes(
        x = model_numeric + (k_numeric - 2) * individual_bar_width,
        y = -0.03,
        label = case_when(
          k_metric == "Top-1 Accuracy" ~ "k1",
          k_metric == "Top-5 Accuracy" ~ "k5",
          k_metric == "Top-10 Accuracy" ~ "k10",
          TRUE ~ as.character(k_metric)
        )
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
      legend.title = element_text(face = "bold", size = 10),
      legend.text = element_text(size = 9),
      legend.key.size = unit(0.8, "lines"),
      panel.grid.major.y = element_line(color = "gray90"),
      panel.grid.minor.y = element_blank(),
      panel.grid.major.x = element_blank(),
      plot.margin = margin(t = 15, r = 20, b = 20, l = 15)
    ) +
    labs(y = "Accuracy Score")
  
  # Prepare summary table data
  summary_table_data <- model_scores_long %>%
    mutate(
      score = round(score, 3),
      se = round(se, 3),
      case_type = case_type_label
    ) %>%
    select(model, stage_factor, k_metric, score, se, case_type) %>%
    mutate(model = factor(model, levels = ranked_models_by_points)) %>%
    arrange(model, k_metric, stage_factor)
  
  # Return both the plot and the summary data
  return(list(
    plot = p,
    summary_data = summary_table_data,
    ranked_models = ranked_models_by_points
  ))
}

# --- 4. Generate Individual Plots ---

# Complex Cases (Stages 1, 2, 3)
complex_results <- generate_model_plot(
  dataset_prefix = "Results_typical_cases_CPS_concise_selected",
  stages = c(1, 2, 3),
  case_type_label = "Complex Cases"
)

# Common Cases (Stages 1, 2)
common_results <- generate_model_plot(
  dataset_prefix = "Results_typical_cases_typical_concise_selected",
  stages = c(1, 2),
  case_type_label = "Common Cases"
)

# --- 5. Combine Plots Vertically ---

# Add case type labels to each plot
plot_complex_labeled <- complex_results$plot + 
  labs(title = "Complex Cases") + 
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5))

plot_common_labeled <- common_results$plot + 
  labs(title = "Common Cases") + 
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5))

# Combine plots with a shared legend
combined_plot <- plot_complex_labeled / plot_common_labeled +
  plot_layout(heights = c(1, 1), guides = "collect") +
  plot_annotation(
    title = "Model Performance Comparison Across Case Types",
    theme = theme(
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      plot.margin = margin(t = 20, r = 20, b = 20, l = 20)
    )
  ) &
  theme(
    legend.position = "top",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.margin = margin(t = 0, r = 0, b = 10, l = 0)
  )

# Print the combined plot
print(combined_plot)

# --- 6. Save Combined Outputs ---

# Save the combined plot
combined_plot_filename_base <- file.path(output_dir, "combined_model_performance_comparison")
ggsave(paste0(combined_plot_filename_base, ".pdf"), plot = combined_plot, width = 14, height = 9, dpi = 400)
ggsave(paste0(combined_plot_filename_base, ".png"), plot = combined_plot, width = 14, height = 9, dpi = 400)
cat("Combined plots saved to '", combined_plot_filename_base, ".pdf' and '.png'\n")

# Combine and save summary tables
combined_summary_data <- bind_rows(
  complex_results$summary_data,
  common_results$summary_data
)

combined_table_filename <- file.path(output_dir, "combined_model_performance_table.csv")
write.csv(combined_summary_data, file = combined_table_filename, row.names = FALSE)
cat("Combined performance metrics table saved to '", combined_table_filename, "'\n")

# Print top performers for each case type
cat("\nTop 5 performers for Complex Cases:\n")
print(head(complex_results$ranked_models, 5))

cat("\nTop 5 performers for Common Cases:\n")
print(head(common_results$ranked_models, 5))
