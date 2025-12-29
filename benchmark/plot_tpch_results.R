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

# Choose palette for common modes; if unknown modes are present they'll get default ggplot colors
method_colors <- c("baseline" = "#1f77b4", "SIMD PAC" = "#ff7f0e", "naive PAC" = "#2ca02c")

# Build a plotting function so we can save two variants (with/without Q01)
build_plot <- function(df, out_file, plot_title, width = 18, height = 8, base_size = 40, base_family = "sans") {
  p <- ggplot(df, aes(x = query, y = mean_time, fill = mode)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    scale_fill_manual(values = method_colors, name = "Mode") +
    scale_y_continuous(labels = scales::comma) +
    labs(x = "Query", y = "Time (ms)", fill = "Mode") +
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

# Title with sf and runs
title_sf <- ifelse(is.na(sf_str), "sf=unknown", paste0("sf=", sf_str))
plot_title_all <- paste0("TPC-H Benchmark, ", title_sf, ", runs=", n_runs_text)

# Output file names
sf_for_name <- ifelse(is.na(sf_str), "unknown", gsub("\\.", "_", sf_str))
out_file_all <- file.path(output_dir, paste0("tpch_benchmark_plot_sf", sf_for_name, "_all.png"))
out_file_no_q01 <- file.path(output_dir, paste0("tpch_benchmark_plot_sf", sf_for_name, "_no_q01.png"))

# 1) Full plot (all queries)
build_plot(summary_df, out_file_all, plot_title_all)

# 2) Exclude query 1 (support q01/Q1 formats)
summary_no_q1 <- summary_df %>% filter(is.na(qnum) | qnum != 1)
if (nrow(summary_no_q1) > 0) {
  plot_title_noq1 <- paste0(plot_title_all, " (excluding Q1)")
  build_plot(summary_no_q1, out_file_no_q01, plot_title_noq1)
} else {
  message("No queries remain after excluding Q1; skipping no-Q1 plot.")
}

# End of script
