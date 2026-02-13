#!/usr/bin/env Rscript
# Compact TPC-H benchmark plotter
# Keeps package installation/style and creates a single plot: x=query, y=time, legend=baseline/PAC

# Configure user-local library path for package installation (keep original behavior)
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "") {
  user_lib <- file.path(Sys.getenv("HOME"), "R", "libs")
}
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(user_lib, .libPaths()))

required_packages <- c(
  "ggplot2", "dplyr", "readr", "scales", "stringr", "extrafont", "gridExtra", "grid", "tidyr"
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
  library(stringi)
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(scales)
  library(stringr)
  library(extrafont)
  library(gridExtra)
  library(grid)
  library(tidyr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript plot_tpch_results.R path/to/results.csv [output_dir]")
input_csv <- args[1]
output_dir <- if (length(args) >= 2) args[2] else dirname(input_csv)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

input_basename <- basename(input_csv)

# Try to extract scale factor from filename: look for 'sf' followed by digits and optional '.' or '_' then digits
sf_match <- regmatches(input_basename, regexec("sf([0-9]+(?:[._][0-9]+)?)", input_basename, perl = TRUE))[[1]]
if (length(sf_match) > 1) {
  sf_str <- gsub('_', '.', sf_match[2])
} else {
  sf_str <- NA_character_
}

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

# Filter out any rows with missing time or missing mode
raw <- raw %>% filter(!is.na(time_ms) & !is.na(mode))
if (nrow(raw) == 0) stop("No valid time data to plot.")

# Compute per-query per-mode summary (mean + sd)
summary_df <- raw %>%
  group_by(query, mode) %>%
  summarize(mean_time = mean(time_ms, na.rm = TRUE), sd_time = sd(time_ms, na.rm = TRUE), runs = n(), .groups = "drop")

# Add numeric query number to help filtering (q01, Q1, etc.)
summary_df <- summary_df %>% mutate(qnum = as.integer(str_extract(query, "\\d+")))

# Determine number of runs per query (expect same across queries)
runs_per_query <- raw %>% group_by(query) %>% summarize(n_runs = n_distinct(run), .groups = "drop")
if (nrow(runs_per_query) == 0) n_runs_text <- "unknown" else {
  minr <- min(runs_per_query$n_runs, na.rm = TRUE)
  maxr <- max(runs_per_query$n_runs, na.rm = TRUE)
  if (minr == maxr) n_runs_text <- as.character(minr) else n_runs_text <- paste0(minr, "-", maxr)
}

# Prepare ordering of queries by numeric value (support q01/q1 or just numeric)
query_order <- summary_df %>%
  mutate(qnum = as.integer(str_extract(query, "\\d+"))) %>%
  arrange(qnum) %>%
  pull(query) %>%
  unique()
if (length(query_order) == 0) query_order <- unique(summary_df$query)

summary_df$query <- factor(summary_df$query, levels = query_order)

# Provide a plotting-safe mean time for log scale (replace non-positive values with a tiny positive number)
summary_df <- summary_df %>% mutate(mean_time_plot = ifelse(mean_time <= 0 | is.na(mean_time), 1e-3, mean_time))

# ============================================================================
# Compute slowdown statistics for each mode compared to baseline
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

    # Format the line
    line <- sprintf(
      "%s: worst %.1fx (%s), avg %.1fx, best %.1fx (%s)",
      mode_name,
      row$worst_slowdown,
      row$worst_query,
      row$avg_slowdown,
      row$best_slowdown,
      row$best_query
    )
    report_lines <- c(report_lines, line)
  }

  return(paste(report_lines, collapse = "\n"))
}

# Compute the stats
mode_stats <- compute_slowdown_report(summary_df)
report_text <- generate_report_text(mode_stats)

# Print report to console as well
message("\n=== Slowdown Report (vs baseline) ===")
message(report_text)
message("=====================================\n")

# Choose palette for common modes; if unknown modes are present they'll get default ggplot colors
method_colors <- c(
  "baseline" = "#1f77b4",
  "SIMD PAC" = "#ff7f0e",
  "naive PAC" = "#2ca02c",
  "simple hash PAC" = "#9467bd"
)

# Build a plotting function so we can save two variants (with/without Q01)
build_plot <- function(df, out_file, plot_title, width = 18, height = 8, base_size = 40, base_family = "sans") {
  p <- ggplot(df, aes(x = query, y = mean_time_plot, fill = mode)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    scale_fill_manual(values = method_colors, name = "Mode") +
    scale_y_log10(labels = scales::comma) +
    labs(x = "Query", y = "Time (ms, log scale)", fill = "Mode") +
    theme_bw(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.major = element_line(linewidth = 1.0),
      panel.grid.minor = element_blank(),
      legend.position = "top",
      legend.title = element_text(size = base_size - 14),
      legend.text = element_text(size = base_size - 16),
      axis.text.x = element_text(angle = 45, hjust = 1, size = base_size - 16),
      axis.text.y = element_text(size = base_size - 12),
      axis.title = element_text(size = base_size - 10),
      plot.title = element_text(size = base_size + 4, face = "bold", hjust = 0.5)
    ) +
    ggtitle(plot_title)

  # Save with higher resolution and larger default dimensions
  ggsave(filename = out_file, plot = p, width = width, height = height, dpi = 300)
  message("Plot saved to: ", out_file)
}

build_plot_paper <- function(df, out_file, plot_title, width = 2000, height = 1000, res = 200, base_size = 40, base_family = "Linux Libertine") {
  p <- ggplot(df, aes(x = query, y = mean_time_plot, fill = mode)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    scale_fill_manual(values = method_colors, name = "Mode") +
    scale_y_log10(labels = scales::comma) +
    labs(x = "Query", y = "Time (ms, log scale)", fill = "Mode") +
    theme_bw(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.major = element_line(linewidth = 1.0),
      panel.grid.minor = element_blank(),
      legend.position = "top",
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, -10, 0),
      legend.title = element_text(size = base_size - 14),
      legend.text = element_text(size = base_size - 16),
      axis.text.x = element_text(angle = 45, hjust = 1, size = base_size - 16),
      axis.text.y = element_text(size = base_size - 12),
      axis.title = element_text(size = base_size - 10),
      plot.margin = margin(5, 5, 5, 5)
    )

  # Save as PNG with specified dimensions in pixels
  png(filename = out_file, width = width, height = height, res = res)
  print(p)
  dev.off()
  message("Plot saved to: ", out_file)
}

# Title with sf and runs
title_sf <- ifelse(is.na(sf_str), "sf=unknown", paste0("sf=", sf_str))
plot_title_all <- paste0("TPC-H Benchmark, ", title_sf, ", runs=", n_runs_text)

# Output file names
sf_for_name <- ifelse(is.na(sf_str), "unknown", gsub("\\.", "_", sf_str))
out_file_all <- file.path(output_dir, paste0("tpch_benchmark_plot_sf", sf_for_name, ".png"))
out_file_paper <- file.path(output_dir, paste0("tpch_benchmark_plot_sf", sf_for_name, "_paper.png"))

# 1) Full plot (all queries)
build_plot(summary_df, out_file_all, plot_title_all)
build_plot_paper(summary_df, out_file_paper, plot_title_all)


# End of script
