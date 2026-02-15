#!/usr/bin/env Rscript
# TPC-H benchmark plotter
# Reads CSV with columns: query, mode, median_ms
# When "simple hash PAC" data is present, splits PAC bars into core + PU-key join overhead

# Configure user-local library path for package installation
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "") {
  user_lib <- file.path(Sys.getenv("HOME"), "R", "libs")
}
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(user_lib, .libPaths()))

required_packages <- c("ggplot2", "dplyr", "readr", "scales", "stringr")
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
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript plot_tpch_results.R path/to/results.csv [output_dir]")
input_csv <- args[1]
output_dir <- if (length(args) >= 2) args[2] else dirname(input_csv)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

input_basename <- basename(input_csv)

# Try to extract scale factor from filename
sf_match <- regmatches(input_basename, regexec("sf([0-9]+(?:[._][0-9]+)?)", input_basename, perl = TRUE))[[1]]
if (length(sf_match) > 1) {
  sf_str <- gsub('_', '.', sf_match[2])
} else {
  sf_str <- NA_character_
}

# Read CSV (format: query, mode, median_ms)
raw <- suppressWarnings(readr::read_csv(input_csv, show_col_types = FALSE))

expected_cols <- c("query", "mode", "median_ms")
missing_cols <- setdiff(expected_cols, colnames(raw))
if (length(missing_cols) > 0) {
  stop("Missing expected columns in CSV: ", paste(missing_cols, collapse = ", "))
}

raw <- raw %>%
  mutate(query = as.character(query), mode = as.character(mode)) %>%
  filter(!is.na(median_ms) & !is.na(mode))

if (nrow(raw) == 0) stop("No valid data to plot.")

# Extract query number
raw <- raw %>% mutate(qnum = as.integer(str_extract(query, "\\d+")))

# Omit queries that don't use PAC or are not allowed
omit_queries <- c(2, 3, 10, 11, 16, 18)
raw <- raw %>% filter(!(qnum %in% omit_queries))

# Keep only base queries (no variants like q08-nolambda)
raw <- raw %>% filter(query == sprintf("q%02d", qnum))

if (nrow(raw) == 0) stop("No data left after filtering.")

# Rename modes
raw <- raw %>% mutate(mode = case_when(
  mode == "baseline" ~ "DuckDB",
  mode == "SIMD PAC" ~ "PAC",
  TRUE ~ mode
))

# Ordered query list
query_order <- raw %>% distinct(query, qnum) %>% arrange(qnum) %>% pull(query)

# Slowdown report
baseline_df <- raw %>% filter(mode == "DuckDB") %>% select(query, baseline_time = median_ms)
other_modes <- raw %>% filter(mode != "DuckDB") %>% select(query, mode, median_ms)
slowdown_df <- other_modes %>%
  left_join(baseline_df, by = "query") %>%
  filter(!is.na(baseline_time) & baseline_time > 0) %>%
  mutate(slowdown = median_ms / baseline_time)

if (nrow(slowdown_df) > 0) {
  mode_stats <- slowdown_df %>%
    group_by(mode) %>%
    summarize(
      worst = max(slowdown), worst_q = query[which.max(slowdown)],
      avg = mean(slowdown), median = median(slowdown),
      best = min(slowdown), best_q = query[which.min(slowdown)],
      .groups = "drop"
    )
  message("\n=== Slowdown Report (vs DuckDB) ===")
  for (i in seq_len(nrow(mode_stats))) {
    r <- mode_stats[i, ]
    message(sprintf("%s: worst %.1fx (%s), avg %.1fx, median %.1fx, best %.1fx (%s)",
                    r$mode, r$worst, r$worst_q, r$avg, r$median, r$best, r$best_q))
  }
  message("===================================\n")
}

# Check if simple hash PAC data is available for the split
has_simple_hash <- "simple hash PAC" %in% raw$mode

# Build plot data
# Each row: query, x_pos (numeric), component (fill), time
duckdb_df <- raw %>% filter(mode == "DuckDB") %>% select(query, median_ms)
pac_df <- raw %>% filter(mode == "PAC") %>% select(query, pac_time = median_ms)

if (has_simple_hash) {
  hash_df <- raw %>% filter(mode == "simple hash PAC") %>% select(query, hash_time = median_ms)
  # Join all three
  combined <- duckdb_df %>%
    inner_join(pac_df, by = "query") %>%
    inner_join(hash_df, by = "query") %>%
    mutate(
      join_overhead = pmax(0, pac_time - hash_time),
      core_time = hash_time
    )
  # Build long-format plot data
  plot_data <- bind_rows(
    combined %>% transmute(query, bar = "DuckDB", component = "DuckDB", time = median_ms),
    combined %>% transmute(query, bar = "PAC", component = "PAC", time = core_time),
    combined %>% transmute(query, bar = "PAC", component = "PU-key join", time = join_overhead)
  )
} else {
  # No split: simple DuckDB vs PAC
  combined <- duckdb_df %>% inner_join(pac_df, by = "query")
  plot_data <- bind_rows(
    combined %>% transmute(query, bar = "DuckDB", component = "DuckDB", time = median_ms),
    combined %>% transmute(query, bar = "PAC", component = "PAC", time = pac_time)
  )
}

# Assign numeric x positions with manual dodge
plot_data <- plot_data %>%
  mutate(
    qidx = match(query, query_order),
    x_pos = qidx + ifelse(bar == "DuckDB", -0.2, 0.2)
  )

# Component ordering: PU-key join on top, core on bottom, DuckDB separate
if (has_simple_hash) {
  comp_levels <- c("PU-key join", "PAC", "DuckDB")
  comp_colors <- c("DuckDB" = "#1f77b4", "PAC" = "#ff7f0e", "PU-key join" = "#d62728")
} else {
  comp_levels <- c("DuckDB", "PAC")
  comp_colors <- c("DuckDB" = "#1f77b4", "PAC" = "#ff7f0e")
}
plot_data$component <- factor(plot_data$component, levels = comp_levels)

# Title
title_sf <- ifelse(is.na(sf_str), "sf=unknown", paste0("SF=", sf_str))
plot_title <- paste0("TPC-H Benchmark, ", title_sf)

# Build plot
# For log scale with split bars, use overlapping bars instead of stacking:
# draw the full PAC bar (PU-key join color) first, then overlay the core bar on top.
# The visible top portion of the PAC bar shows the join overhead.
if (has_simple_hash) {
  pac_total <- plot_data %>% filter(bar == "PAC") %>%
    group_by(query, x_pos) %>%
    summarize(total = sum(time), .groups = "drop")
  pac_core <- plot_data %>% filter(component == "PAC")
  duckdb_bars <- plot_data %>% filter(bar == "DuckDB")

  p <- ggplot() +
    geom_col(data = duckdb_bars, aes(x = x_pos, y = time, fill = component), width = 0.35) +
    geom_col(data = pac_total, aes(x = x_pos, y = total, fill = "PU-key join"), width = 0.35) +
    geom_col(data = pac_core, aes(x = x_pos, y = time, fill = component), width = 0.35) +
    scale_fill_manual(values = comp_colors, name = NULL,
                      breaks = c("DuckDB", "PAC", "PU-key join")) +
    scale_x_continuous(breaks = seq_along(query_order), labels = query_order) +
    scale_y_log10(labels = scales::comma) +
    labs(x = "TPC-H Query", y = "Time (ms, log scale)")
} else {
  p <- ggplot(plot_data, aes(x = x_pos, y = time, fill = component, width = 0.35)) +
    geom_col() +
    scale_fill_manual(values = comp_colors, name = NULL) +
    scale_x_continuous(breaks = seq_along(query_order), labels = query_order) +
    scale_y_log10(labels = scales::comma) +
    labs(x = "TPC-H Query", y = "Time (ms, log scale)")
}

p <- p +
  theme_bw(base_size = 40) +
  theme(
    panel.grid.major = element_line(linewidth = 1.0),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.title = element_blank(),
    legend.text = element_text(size = 28),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 24),
    axis.text.y = element_text(size = 28),
    axis.title.x = element_text(size = 32),
    axis.title.y = element_text(size = 32),
    plot.title = element_text(size = 44, face = "bold", hjust = 0.5)
  ) +
  ggtitle(plot_title)

sf_for_name <- ifelse(is.na(sf_str), "unknown", gsub("\\.", "_", sf_str))
out_file <- file.path(output_dir, paste0("tpch_benchmark_plot_sf", sf_for_name, ".png"))

ggsave(filename = out_file, plot = p, width = 18, height = 8, dpi = 300)
message("Plot saved to: ", out_file)
