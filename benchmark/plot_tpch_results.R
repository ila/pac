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
expected_cols <- c("query", "run", "time_ms", "pac_time_ms")
missing_cols <- setdiff(expected_cols, colnames(raw))
if (length(missing_cols) > 0) {
  stop("Missing expected columns in CSV: ", paste(missing_cols, collapse = ", "))
}

# Normalize query column to character
raw <- raw %>% mutate(query = as.character(query))

# Convert pac_time_ms == -1 (error) to NA so they don't skew means
raw <- raw %>% mutate(pac_time_ms = ifelse(pac_time_ms < 0, NA_real_, pac_time_ms))

# Pivot to long format: baseline vs PAC
long <- raw %>% select(query, run, time_ms, pac_time_ms) %>%
  pivot_longer(cols = c(time_ms, pac_time_ms), names_to = "scenario", values_to = "time") %>%
  mutate(scenario = ifelse(scenario == "time_ms", "baseline", "PAC")) %>%
  filter(!is.na(time))

if (nrow(long) == 0) stop("No valid time data to plot.")

# Compute per-query summary (mean + sd)
summary_df <- long %>%
  group_by(query, scenario) %>%
  summarize(mean_time = mean(time, na.rm = TRUE), sd_time = sd(time, na.rm = TRUE), runs = n(), .groups = "drop")

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

# Colors for baseline and PAC
method_colors <- c("baseline" = "#1f77b4", "PAC" = "#ff7f0e")

# Build plot (grouped bar chart, bigger type)
p <- ggplot(summary_df, aes(x = query, y = mean_time, fill = scenario)) +
  # grouped bars: baseline vs PAC side-by-side per query
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = method_colors, name = "Method") +
  scale_y_continuous(labels = scales::comma) +
  labs(x = "Query", y = "Time (ms)", fill = "Method") +
  theme_bw(base_size = 22, base_family = "Linux Libertine") +
  theme(
    panel.grid.major = element_line(linewidth = 0.6),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 18),
    axis.text.y = element_text(size = 18),
    axis.title = element_text(size = 20),
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5)
  )

# Title with sf and runs
title_sf <- ifelse(is.na(sf_str), "sf=unknown", paste0("sf=", sf_str))
plot_title <- paste0("TPC-H Benchmark, ", title_sf, ", runs=", n_runs_text)
note_lines <- c(
  "Queries without dependence on customer (unchanged): Q2, Q11, Q16",
  "Directly returning Customer data (unchanged, should not return): Q10, Q18",
  "Directly scanning Customer: Q03, Q05, Q07 (?), Q08, Q13 (?), Q22",
  "PAC join with Customer: Q01, Q06, Q09, Q15, Q17, Q19, Q20, Q21 (?)",
  "Using Orders as PU: Q04, Q12, Q14"
)
subtitle_text <- paste(note_lines, collapse = "\n")
p <- p + labs(title = plot_title, subtitle = subtitle_text)

# Output file
sf_for_name <- ifelse(is.na(sf_str), "unknown", gsub("\\.", "_", sf_str))
out_file <- file.path(output_dir, paste0("tpch_benchmark_plot_sf", sf_for_name, ".png"))

# Save plot (overwrite if exists)
ggsave(filename = out_file, plot = p, width = 10, height = 6, dpi = 300)
message("Plot saved to: ", out_file)

# End of script
