#!/usr/bin/env Rscript
# ClickBench benchmark plotter
# Creates plots comparing baseline vs PAC execution times
# Usage: Rscript plot_clickbench_results.R path/to/clickbench_results.csv [output_dir]

# Configure user-local library path for package installation
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "") {
  user_lib <- file.path(Sys.getenv("HOME"), "R", "libs")
}
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(user_lib, .libPaths()))

required_packages <- c(
  "ggplot2", "dplyr", "readr", "scales", "stringr", "extrafont", "gridExtra", "grid"
)
options(repos = c(CRAN = "https://cloud.r-project.org"))
installed <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!(pkg %in% installed)) {
    message("Installing package: ", pkg)
    install.packages(pkg, dependencies = TRUE, lib = user_lib)
  }
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(scales)
  library(stringr)
  library(extrafont)
  library(gridExtra)
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
# Default to clickbench_micro_results.csv in the script directory
# Get script directory from commandArgs (works with Rscript)
get_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  return(getwd())
}
script_dir <- get_script_dir()
input_csv <- if (length(args) >= 1) args[1] else file.path(script_dir, "clickbench_micro_results.csv")
output_dir <- if (length(args) >= 2) args[2] else dirname(input_csv)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

input_basename <- basename(input_csv)

# Detect if this is a micro benchmark from filename
is_micro <- grepl("micro", input_basename, ignore.case = TRUE)

# Read CSV
raw <- suppressWarnings(readr::read_csv(input_csv, show_col_types = FALSE))

# Validate expected columns
expected_cols <- c("query", "mode", "run", "time_ms")
missing_cols <- setdiff(expected_cols, colnames(raw))
if (length(missing_cols) > 0) {
  stop("Missing expected columns in CSV: ", paste(missing_cols, collapse = ", "))
}

# Normalize query column to character
raw <- raw %>% mutate(query = as.character(query))

# Ensure mode is a character and normalize values
raw <- raw %>% mutate(mode = as.character(mode))

# Check if success column exists, default to TRUE if not
if (!"success" %in% colnames(raw)) {
  raw <- raw %>% mutate(success = TRUE)
} else {
  # Convert success to logical if it's character
  raw <- raw %>% mutate(success = as.logical(success))
}

# Filter out any rows with missing time or missing mode
raw <- raw %>% filter(!is.na(time_ms) & !is.na(mode))
if (nrow(raw) == 0) stop("No valid time data to plot.")

# Compute per-query per-mode summary (mean + sd), only for successful runs
# Also track if any run failed for this query/mode combo
summary_df <- raw %>%
  group_by(query, mode) %>%
  summarize(
    mean_time = mean(time_ms[success == TRUE], na.rm = TRUE),
    sd_time = sd(time_ms[success == TRUE], na.rm = TRUE),
    runs = sum(success == TRUE),
    total_runs = n(),
    any_failed = any(success == FALSE),
    all_failed = all(success == FALSE),
    .groups = "drop"
  )

# Add numeric query number for ordering
summary_df <- summary_df %>% mutate(qnum = as.integer(str_extract(query, "\\d+")))

# Determine number of runs per query
runs_per_query <- raw %>% group_by(query) %>% summarize(n_runs = n_distinct(run), .groups = "drop")
if (nrow(runs_per_query) == 0) n_runs_text <- "unknown" else {
  minr <- min(runs_per_query$n_runs, na.rm = TRUE)
  maxr <- max(runs_per_query$n_runs, na.rm = TRUE)
  if (minr == maxr) n_runs_text <- as.character(minr) else n_runs_text <- paste0(minr, "-", maxr)
}

# Prepare ordering of queries by numeric value
query_order <- summary_df %>%
  arrange(qnum) %>%
  pull(query) %>%
  unique()
if (length(query_order) == 0) query_order <- unique(summary_df$query)

summary_df$query <- factor(summary_df$query, levels = query_order)

# Provide a plotting-safe mean time for log scale (replace non-positive values with a tiny positive number)
summary_df <- summary_df %>% mutate(mean_time_plot = ifelse(mean_time <= 0 | is.na(mean_time), 1e-3, mean_time))

# ============================================================================
# Compute slowdown statistics for PAC compared to baseline
# ============================================================================
compute_slowdown_report <- function(df) {
  # Get baseline times per query
  baseline_df <- df %>% filter(mode == "baseline") %>% select(query, baseline_time = mean_time)

  # Get all non-baseline modes
  other_modes <- df %>% filter(mode != "baseline") %>% select(query, mode, mean_time)

  # Join to compute slowdown ratio for each query/mode
  slowdown_df <- other_modes %>%
    left_join(baseline_df, by = "query") %>%
    filter(!is.na(baseline_time) & baseline_time > 0) %>%
    mutate(slowdown = mean_time / baseline_time)

  # Compute per-mode statistics
  mode_stats <- slowdown_df %>%
    group_by(mode) %>%
    summarize(
      worst_slowdown = max(slowdown, na.rm = TRUE),
      worst_query = query[which.max(slowdown)],
      avg_slowdown = mean(slowdown, na.rm = TRUE),
      median_slowdown = median(slowdown, na.rm = TRUE),
      best_slowdown = min(slowdown, na.rm = TRUE),
      best_query = query[which.min(slowdown)],
      n_queries = n(),
      .groups = "drop"
    )

  return(mode_stats)
}

# Generate text report
generate_report_text <- function(mode_stats) {
  report_lines <- c()

  for (i in seq_len(nrow(mode_stats))) {
    row <- mode_stats[i, ]
    mode_name <- row$mode

    line <- sprintf(
      "%s: worst %.2fx (%s), avg %.2fx, median %.2fx, best %.2fx (%s), n=%d queries",
      mode_name,
      row$worst_slowdown,
      row$worst_query,
      row$avg_slowdown,
      row$median_slowdown,
      row$best_slowdown,
      row$best_query,
      row$n_queries
    )
    report_lines <- c(report_lines, line)
  }

  return(paste(report_lines, collapse = "\n"))
}

# Compute the stats
mode_stats <- compute_slowdown_report(summary_df)
report_text <- generate_report_text(mode_stats)

# Print report to console
message("\n=== ClickBench Slowdown Report (vs baseline) ===")
message(report_text)
message("=================================================\n")

# Color palette
method_colors <- c(
  "baseline" = "#1f77b4",
  "PAC" = "#ff7f0e"
)

# Build plot function for standard output
build_plot <- function(df, out_file, plot_title, width = 4000, height = 2000, res = 200, base_size = 36, base_family = "sans") {
 failed_queries <- df %>% filter(all_failed) %>% pull(query) %>% unique()
  df_success <- df %>% filter(!query %in% failed_queries)
  p <- ggplot(df_success, aes(x = query, y = mean_time_plot, fill = mode)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    scale_fill_manual(values = method_colors, name = "Mode") +
    scale_y_log10(labels = scales::comma) +
    labs(x = "Query", y = "Time (ms, log scale)", fill = "Mode") +
    theme_bw(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.major = element_line(linewidth = 0.8),
      panel.grid.minor = element_blank(),
      legend.position = "top",
      legend.title = element_text(size = base_size - 10),
      legend.text = element_text(size = base_size - 12),
      axis.text.x = element_text(angle = 45, hjust = 1, size = base_size - 14),
      axis.text.y = element_text(size = base_size - 10),
      axis.title = element_text(size = base_size - 8),
      plot.title = element_text(size = base_size + 2, face = "bold", hjust = 0.5)
    ) +
    ggtitle(plot_title)

  png(filename = out_file, width = width, height = height, res = res)
    print(p)
    dev.off()
  message("Plot saved to: ", out_file)
}

# Build plot function for paper (higher quality, Linux Libertine font)
build_plot_paper <- function(df, out_file, plot_title, width = 2400, height = 1000, res = 200, base_size = 36, base_family = "Linux Libertine") {
  # Only plot queries where all modes succeeded
  # Find queries where any mode failed
  failed_queries <- df %>% filter(all_failed) %>% pull(query) %>% unique()
  df_success <- df %>% filter(!query %in% failed_queries)

  p <- ggplot(df_success, aes(x = query, y = mean_time_plot, fill = mode)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    scale_fill_manual(values = method_colors, name = "Mode") +
    scale_y_log10(labels = scales::comma) +
    labs(x = "Query", y = "Time (ms, log scale)", fill = "Mode") +
    theme_bw(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.major = element_line(linewidth = 0.8),
      panel.grid.minor = element_blank(),
      legend.position = "top",
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, -10, 0),
      legend.title = element_text(size = base_size - 10),
      legend.text = element_text(size = base_size - 12),
      axis.text.x = element_text(angle = 45, hjust = 1, size = base_size - 14),
      axis.text.y = element_text(size = base_size - 10),
      axis.title = element_text(size = base_size - 8),
      plot.margin = margin(5, 10, 5, 5)
    )

  png(filename = out_file, width = width, height = height, res = res)
  print(p)
  dev.off()
  message("Plot saved to: ", out_file)
  message("Excluded ", length(failed_queries), " queries with failures: ", paste(failed_queries, collapse = ", "))
}

# Build plot title
dataset_type <- ifelse(is_micro, "micro", "full")
n_queries <- length(unique(summary_df$query))
plot_title <- paste0("ClickBench Benchmark (", dataset_type, ", ", n_queries, " queries, ", n_runs_text, " runs)")

# Output file names
suffix <- ifelse(is_micro, "_micro", "")
out_file <- file.path(output_dir, paste0("clickbench_benchmark_plot", suffix, ".png"))
out_file_paper <- file.path(output_dir, paste0("clickbench_benchmark_plot", suffix, "_paper.png"))

# Generate plots
build_plot(summary_df, out_file, plot_title)
build_plot_paper(summary_df, out_file_paper, plot_title)

message("\nClickBench plotting complete.")
