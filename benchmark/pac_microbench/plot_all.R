#!/usr/bin/env Rscript
# PAC Microbenchmark Plotting Script
# Converts all Python plotting scripts to R with ggplot2

# Configure user-local library path for package installation
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "") {
  user_lib <- file.path(Sys.getenv("HOME"), "R", "libs")
}
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(user_lib, .libPaths()))

required_packages <- c("ggplot2", "dplyr", "readr", "scales", "stringr", "tidyr", "ggpattern")
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
  library(tidyr)
  library(ggpattern)
})

# ============================================================================
# Platform and Variant Definitions
# ============================================================================

PLATFORM_NAMES <- c(
  'epyc' = 'AMD\n(EPYC 9645)',
  'granite-rapids' = 'Intel Xeon\n(Granite Rapids)',
  'graviton' = 'ARM\n(AWS Graviton4)',
  'macbook' = 'ARM\n(Apple M2 Pro)'
)

PLATFORM_ORDER <- c('graviton', 'macbook', 'granite-rapids', 'epyc')

# Consistent color scheme across all plots:
# - Gray: DuckDB standard
# - Green: Best PAC optimization (with buffering)
# - Blue: Mid-tier optimization (without buffering)
# - Orange: Alternative method (Exact sum)
# - Red: Naive/SIMD-Unfriendly

# Count variant names
COUNT_VARIANT_NAMES <- c(
  'standard' = 'DuckDB Count',
  'default' = 'Buffering+Cascading',
  'nobuffering' = 'Cascading',
  'nocascading' = 'Naive/SIMD-Unfriendly'
)

COUNT_VARIANT_ORDER <- c('DuckDB Count', 'Buffering+Cascading',
                         'Cascading', 'Naive/SIMD-Unfriendly')

COUNT_VARIANT_COLORS <- c(
  'DuckDB Count' = '#95a5a6',
  'Buffering+Cascading' = '#27ae60',
  'Cascading' = '#3498db',
  'Naive/SIMD-Unfriendly' = '#e74c3c'
)

# Sum variant names
SUM_VARIANT_NAMES <- c(
  'standard' = 'DuckDB Sum',
  'default' = 'Approximate+Buffering',
  'nobuffering' = 'Approximate',
  'exactsum' = 'Exact',
  'nocascading' = 'Naive/SIMD-Unfriendly'
)

SUM_VARIANT_ORDER <- c('DuckDB Sum', 'Approximate+Buffering',
                       'Approximate', 'Exact',
                       'Naive/SIMD-Unfriendly')

SUM_VARIANT_COLORS <- c(
  'DuckDB Sum' = '#95a5a6',
  'Approximate+Buffering' = '#27ae60',
  'Approximate' = '#3498db',
  'Exact' = '#f39c12',
  'Naive/SIMD-Unfriendly' = '#e74c3c'
)

# MinMax variant names
MINMAX_VARIANT_NAMES <- c(
  'standard' = 'DuckDB Max',
  'default' = 'Buffering+Pruning',
  'nobuffering' = 'Pruning',
  'noboundopt' = 'Naive/SIMD-Unfriendly'
)

MINMAX_VARIANT_ORDER <- c('DuckDB Max', 'Buffering+Pruning',
                          'Pruning', 'Naive/SIMD-Unfriendly')

MINMAX_VARIANT_COLORS <- c(
  'DuckDB Max' = '#95a5a6',
  'Buffering+Pruning' = '#27ae60',
  'Pruning' = '#3498db',
  'Naive/SIMD-Unfriendly' = '#e74c3c'
)

# SIMD aggregate colors - unique colors for each aggregate
AGG_COLORS <- c(
  'pac_count' = '#9b59b6',    # Purple
  'pac_max' = '#e67e22',       # Orange
  'pac_sum' = '#16a085'        # Teal
)

AGG_ORDER <- c('pac_max', 'pac_count', 'pac_sum')

# Pattern fills for better distinguishability
# These will be used with ggpattern if available
# Valid ggpattern types: 'stripe', 'crosshatch', 'circle', 'none'
VARIANT_PATTERNS <- c(
  'Buffering+Cascading' = 'stripe',
  'Cascading' = 'crosshatch',
  'Approximate+Buffering' = 'wave',
  'Approximate' = 'weave',
  'Exact' = 'stripe',
  'Buffering+Pruning' = 'rose',
  'Pruning' = 'plasma',
  'DuckDB Count' = 'circle',
  'DuckDB Sum' = 'circle',
  'DuckDB Max' = 'circle',
  'Naive/SIMD-Unfriendly' = 'crosshatch'
)

AGG_PATTERNS <- c(
  'pac_count' = 'stripe',
  'pac_max' = 'crosshatch',
  'pac_sum' = 'circle'
)

# ============================================================================
# Helper Functions
# ============================================================================

format_group_label <- function(gv) {
  if (gv == 0) {
    return('0\n(ungrouped)')
  } else if (gv >= 1000000) {
    return(paste0(gv / 1000000, 'M'))
  } else if (gv >= 1000) {
    return(paste0(gv / 1000, 'K'))
  } else {
    return(as.character(gv))
  }
}

# ============================================================================
# 1. Count Optimizations Plot
# ============================================================================

plot_count_optimizations <- function(platform, platform_name, results_dir, output_dir) {
  count_files <- list.files(file.path(results_dir, platform),
                            pattern = "^count_.*\\.csv$", full.names = TRUE)
  if (length(count_files) == 0) {
    message("No count data for ", platform)
    return(NULL)
  }

  df <- read_csv(count_files[1], show_col_types = FALSE)

  # Filter to 1000M rows
  df <- df %>% filter(rows_m == 1000)
  df <- df %>% filter(variant %in% names(COUNT_VARIANT_NAMES))
  df <- df %>% filter(test %in% c('ungrouped', 'grouped_scat'))

  # Rename variants
  df <- df %>% mutate(variant_name = COUNT_VARIANT_NAMES[variant])

  # Map to group values
  df <- df %>% mutate(group_values = ifelse(test == 'ungrouped', 0, groups))

  # Handle timeouts
  df <- df %>% mutate(
    display_time = wall_sec,
    is_timeout = wall_sec < 0
  )

  # Calculate y_max dynamically from non-timeout values
  valid_times <- df %>% filter(!is_timeout, display_time > 0) %>% pull(display_time)
  if (length(valid_times) == 0) return(NULL)
  max_time <- max(valid_times)
  y_max <- max_time / 0.82

  # Prepare display times
  df <- df %>% mutate(
    plot_time = case_when(
      is_timeout ~ y_max * 1.15,  # Overflow above plot area
      display_time > y_max ~ y_max * 0.95,
      display_time > 0 & display_time < y_max * 0.005 ~ y_max * 0.005,
      TRUE ~ display_time
    )
  )

  # Create labels - don't show label for timeouts (will be displayed inside bar)
  df <- df %>% mutate(
    time_label = case_when(
      is_timeout ~ '',  # Empty - will add geom_text separately for inside bar
      display_time > y_max ~ sprintf('%.0fs', display_time),
      display_time > 0 ~ sprintf('%.1fs', display_time),
      TRUE ~ ''
    )
  )

  # Format x labels
  df <- df %>% mutate(group_label = sapply(group_values, format_group_label))
  df$group_label <- factor(df$group_label,
                           levels = unique(df$group_label[order(df$group_values)]))
  df$variant_name <- factor(df$variant_name, levels = COUNT_VARIANT_ORDER)

  # Add pattern mapping
  df <- df %>% mutate(pattern = VARIANT_PATTERNS[as.character(variant_name)])

  # Paper plot settings
  width <- 4000
  height <- 1800
  res <- 200
  base_size <- 40
  base_family <- "Linux Libertine"

  p <- ggplot(df, aes(x = group_label, y = plot_time, fill = variant_name, pattern = variant_name)) +
    geom_col_pattern(position = position_dodge(width = 0.9), width = 0.85,
                     color = 'black', linewidth = 0.5,
                     pattern_density = 0.15, pattern_spacing = 0.05,
                     pattern_fill = 'black', pattern_color = 'black',
                     pattern_angle = 45, pattern_size = 0.3) +
    # Regular time labels on top of bars
    geom_text(aes(label = time_label, color = variant_name),
              position = position_dodge(width = 0.9),
              vjust = -0.3, size = base_size * 0.2, fontface = 'bold') +
    # FAILED labels inside bars - ensure variant_name is factored correctly
    geom_text(data = df %>% filter(is_timeout) %>% mutate(label_y = y_max, variant_name = factor(variant_name, levels = COUNT_VARIANT_ORDER)),
              aes(x = group_label, y = label_y, label = 'FAILED', fill = variant_name),
              position = position_dodge(width = 0.5),
              vjust = 3.6, hjust = 0.5,
              size = base_size * 0.25, fontface = 'bold',
              color = 'white', angle = 90, show.legend = FALSE) +
    scale_fill_manual(values = COUNT_VARIANT_COLORS, name = NULL) +
    scale_color_manual(values = COUNT_VARIANT_COLORS, guide = 'none') +
    scale_pattern_manual(values = VARIANT_PATTERNS, guide = 'none') +
    labs(x = 'distinct GROUP BY values for a COUNT(*)',
         y = 'Time (seconds)') +
    theme_bw(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.major = element_line(linewidth = 1.0),
      panel.grid.minor = element_blank(),
      legend.position = 'top',
      legend.justification = 'right',
      legend.direction = 'horizontal',
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, -10, 0),
      legend.text = element_text(size = base_size + 2),
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = base_size + 2),
      axis.text.y = element_text(size = base_size + 2),
      axis.title.x = element_text(size = base_size + 4, margin = margin(t = 5)),
      axis.title.y = element_text(size = base_size + 4),
      plot.margin = margin(5, 5, 5, 5)
    ) +
    coord_cartesian(ylim = c(0, y_max * 1.1), clip = 'off')

  output_file <- file.path(output_dir, paste0('count_optimizations_', platform, '_paper.png'))
  png(filename = output_file, width = width, height = height, res = res)
  print(p)
  dev.off()
  message("Saved: ", output_file)
}

# ============================================================================
# 2. Sum Optimizations Plot
# ============================================================================

plot_sum_optimizations <- function(platform, platform_name, results_dir, output_dir) {
  sum_files <- list.files(file.path(results_dir, platform),
                         pattern = "^sum_avg_.*\\.csv$", full.names = TRUE)
  if (length(sum_files) == 0) {
    message("No sum_avg data for ", platform)
    return(NULL)
  }

  df <- read_csv(sum_files[1], show_col_types = FALSE)

  # Handle different column names
  test_col <- ifelse('itest' %in% colnames(df), 'itest', 'test')

  # Filter to sum, 1000M rows
  df <- df %>% filter(aggregate == 'sum', rows_m == 1000)

  # Get grouped_scat data
  grouped_df <- df %>% filter(!!sym(test_col) == 'grouped_scat')
  if ('large' %in% grouped_df$dtype) {
    grouped_df <- grouped_df %>% filter(dtype == 'large')
  } else if ('tiny' %in% grouped_df$dtype) {
    grouped_df <- grouped_df %>% filter(dtype == 'tiny')
  }
  grouped_df <- grouped_df %>% mutate(group_values = groups)

  # Get ungrouped data
  ungrouped_df <- df %>% filter(!!sym(test_col) == 'ungrouped_domain')
  if ('large' %in% ungrouped_df$dtype) {
    ungrouped_df <- ungrouped_df %>% filter(dtype == 'large')
  } else if ('tiny' %in% ungrouped_df$dtype) {
    ungrouped_df <- ungrouped_df %>% filter(dtype == 'tiny')
  }
  ungrouped_df <- ungrouped_df %>% mutate(group_values = 0, groups = 1)

  # Combine
  df <- bind_rows(ungrouped_df, grouped_df)
  df <- df %>% filter(variant %in% names(SUM_VARIANT_NAMES))
  df <- df %>% mutate(variant_name = SUM_VARIANT_NAMES[variant])
  df <- df %>% mutate(
    display_time = wall_sec,
    is_timeout = wall_sec < 0
  )

  # Calculate y_max: 10M approximate+buffering at 85%
  approx_buff_df <- df %>%
    filter(variant_name == 'Approximate+Buffering',
           !is_timeout, display_time > 0) %>%
    arrange(desc(group_values)) %>%
    slice(1)

  if (nrow(approx_buff_df) > 0) {
    y_max <- approx_buff_df$display_time[1] / 0.85
  } else {
    valid_times <- df %>% filter(!is_timeout, display_time > 0) %>% pull(display_time)
    y_max <- ifelse(length(valid_times) > 0, max(valid_times) / 0.85, 100)
  }

  # Prepare display times
  df <- df %>% mutate(
    plot_time = case_when(
      is_timeout ~ y_max * 1.15,  # Overflow above plot area (match count plot)
      display_time > y_max ~ y_max * 0.95,
      display_time > 0 & display_time < y_max * 0.005 ~ y_max * 0.005,
      TRUE ~ display_time
    ),
    time_label = case_when(
      is_timeout ~ '',  # Empty - will add geom_text separately for inside bar
      display_time > y_max ~ sprintf('%.0fs', display_time),
      display_time > 0 ~ sprintf('%.1fs', display_time),
      TRUE ~ ''
    )
  )

  df <- df %>% mutate(group_label = sapply(group_values, format_group_label))
  df$group_label <- factor(df$group_label,
                           levels = unique(df$group_label[order(df$group_values)]))
  df$variant_name <- factor(df$variant_name, levels = SUM_VARIANT_ORDER)

  # Add pattern mapping
  df <- df %>% mutate(pattern = VARIANT_PATTERNS[as.character(variant_name)])

  # Paper plot settings (wider for more variants)
  width <- 4800
  height <- 1700
  res <- 200
  base_size <- 40
  base_family <- "Linux Libertine"

  p <- ggplot(df, aes(x = group_label, y = plot_time, fill = variant_name, pattern = variant_name)) +
    geom_col_pattern(position = position_dodge(width = 0.9), width = 0.8,
                     color = 'black', size = 0.5,
                     pattern_density = 0.15, pattern_spacing = 0.05,
                     pattern_fill = 'black', pattern_color = 'black',
                     pattern_angle = 45, pattern_size = 0.3) +
    # Regular time labels on top of bars
    geom_text(aes(label = time_label, color = variant_name),
              position = position_dodge(width = 0.9),
              vjust = -0.3, size = base_size * 0.165, fontface = 'bold') +
    # FAILED labels inside bars - vertical like count plot
    geom_text(data = df %>% filter(is_timeout) %>% mutate(label_y = y_max, variant_name = factor(variant_name, levels = SUM_VARIANT_ORDER)),
              aes(x = group_label, y = label_y, label = 'FAILED', fill = variant_name),
              position = position_dodge(width = 0.8),
              vjust = 0.5, hjust = 0.5,
              size = base_size * 0.22, fontface = 'bold',
              color = 'white', angle = 90, show.legend = FALSE) +
    scale_fill_manual(values = SUM_VARIANT_COLORS, name = NULL) +
    scale_color_manual(values = SUM_VARIANT_COLORS, guide = 'none') +
    scale_pattern_manual(values = VARIANT_PATTERNS, guide = 'none') +
    labs(x = 'distinct GROUP BY values for a SUM',
         y = 'Time (seconds)') +
    theme_bw(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.major = element_line(linewidth = 1.0),
      panel.grid.minor = element_blank(),
      legend.position = 'top',
      legend.justification = 'right',
      legend.direction = 'horizontal',
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, -10, 0),
      legend.text = element_text(size = base_size - 5),
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = base_size + 2),
      axis.text.y = element_text(size = base_size + 2),
      axis.title.x = element_text(size = base_size + 4, margin = margin(t = 5)),
      axis.title.y = element_text(size = base_size + 4),
      plot.margin = margin(5, 5, 5, 5)
    ) +
    coord_cartesian(ylim = c(0, y_max * 1.1), clip = 'off')

  output_file <- file.path(output_dir, paste0('sum_optimizations_', platform, '_paper.png'))
  png(filename = output_file, width = width, height = height, res = res)
  print(p)
  dev.off()
  message("Saved: ", output_file)
}

# ============================================================================
# 3. MinMax Optimizations Plot
# ============================================================================

plot_minmax_optimizations <- function(platform, platform_name, results_dir, output_dir) {
  minmax_files <- list.files(file.path(results_dir, platform),
                             pattern = "^min_max_.*\\.csv$", full.names = TRUE)
  if (length(minmax_files) == 0) {
    message("No min_max data for ", platform)
    return(NULL)
  }

  df <- read_csv(minmax_files[1], show_col_types = FALSE)

  # Filter to dist_test, max, 1000M
  df <- df %>% filter(test == 'dist_test', aggregate == 'max', rows_m == 1000)
  df <- df %>% filter(variant %in% names(MINMAX_VARIANT_NAMES))
  df <- df %>% filter(dtype %in% c('random', 'monotonic_inc'))

  df <- df %>% mutate(variant_name = MINMAX_VARIANT_NAMES[variant])
  df <- df %>% mutate(
    display_time = wall_sec,
    is_timeout = wall_sec < 0
  )

  # Calculate y_max
  valid_times <- df %>% filter(!is_timeout, display_time > 0) %>% pull(display_time)
  if (length(valid_times) == 0) return(NULL)
  max_time <- max(valid_times)
  y_max <- max_time / 0.82

  df <- df %>% mutate(
    plot_time = case_when(
      is_timeout ~ y_max * 1.1,
      display_time > y_max ~ y_max * 0.95,
      display_time > 0 & display_time < y_max * 0.005 ~ y_max * 0.005,
      TRUE ~ display_time
    ),
    time_label = case_when(
      is_timeout ~ 'FAILED',
      display_time > y_max ~ sprintf('%.0fs', display_time),
      display_time > 0 ~ sprintf('%.1fs', display_time),
      TRUE ~ ''
    )
  )

  # Distribution labels
  df <- df %>% mutate(
    dist_label = ifelse(dtype == 'random', 'Random', 'Monotonic Increasing')
  )
  df$dist_label <- factor(df$dist_label, levels = c('Random', 'Monotonic Increasing'))
  df$variant_name <- factor(df$variant_name, levels = MINMAX_VARIANT_ORDER)

  # Add pattern mapping
  df <- df %>% mutate(pattern = VARIANT_PATTERNS[as.character(variant_name)])

  # Paper plot settings
  width <- 4000
  height <- 1500
  res <- 200
  base_size <- 40
  base_family <- "Linux Libertine"

  p <- ggplot(df, aes(x = dist_label, y = plot_time, fill = variant_name, pattern = pattern)) +
    geom_col_pattern(position = position_dodge(width = 0.9), width = 0.85,
                     color = 'black', size = 0.5,
                     pattern_density = 0.15, pattern_spacing = 0.05,
                     pattern_fill = 'black', pattern_color = 'black',
                     pattern_angle = 45, pattern_size = 0.3) +
    geom_text(aes(label = time_label, color = variant_name),
              position = position_dodge(width = 0.9),
              vjust = -0.3, size = base_size * 0.30, fontface = 'bold') +
    scale_fill_manual(values = MINMAX_VARIANT_COLORS, name = NULL) +
    scale_color_manual(values = MINMAX_VARIANT_COLORS, guide = 'none') +
    scale_pattern_manual(values = VARIANT_PATTERNS, guide = 'none') +
    labs(x = 'Data Distribution',
         y = 'Time (seconds)') +
    theme_bw(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.major = element_line(linewidth = 1.0),
      panel.grid.minor = element_blank(),
      legend.position = 'top',
      #legend.justification = 'right',
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, -10, 0),
      legend.text = element_text(size = base_size - 2),
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = base_size + 2),
      axis.text.y = element_text(size = base_size + 2),
      axis.title.x = element_text(size = base_size + 4, margin = margin(t = 5)),
      axis.title.y = element_text(size = base_size + 4),
      plot.margin = margin(5, 5, 5, 5)
    ) +
    ylim(0, y_max * 1.1)

  output_file <- file.path(output_dir, paste0('minmax_optimizations_', platform, '_paper.png'))
  png(filename = output_file, width = width, height = height, res = res)
  print(p)
  dev.off()
  message("Saved: ", output_file)
}

# ============================================================================
# 4. SIMD Improvements Plot
# ============================================================================

load_count_simd_factor <- function(platform_dir) {
  count_files <- list.files(platform_dir, pattern = "^count_.*\\.csv$", full.names = TRUE)
  if (length(count_files) == 0) return(list(factor = NA, n = 0))

  df <- read_csv(count_files[1], show_col_types = FALSE)

  message("DEBUG: load_count_simd_factor for ", basename(platform_dir))
  message("  Total rows in file: ", nrow(df))

  df <- df %>% filter(rows_m == 100, test == 'ungrouped')
  message("  After filtering (rows_m==100, test==ungrouped): ", nrow(df))

  default_df <- df %>% filter(variant == 'default')
  naive_df <- df %>% filter(variant == 'nocascading')

  message("  Rows with variant=='default': ", nrow(default_df))
  message("  Rows with variant=='nocascading': ", nrow(naive_df))

  if (nrow(default_df) == 0 || nrow(naive_df) == 0) {
    message("  SKIPPED: Missing default or nocascading variant")
    return(list(factor = NA, n = 0))
  }

  total_default <- sum(default_df$agg_sec[default_df$agg_sec > 0], na.rm = TRUE)
  total_naive <- sum(naive_df$agg_sec[naive_df$agg_sec > 0], na.rm = TRUE)

  message("  total_default (sum of agg_sec): ", round(total_default, 3))
  message("  total_naive (sum of agg_sec): ", round(total_naive, 3))

  if (total_default == 0) {
    message("  SKIPPED: total_default is 0")
    return(list(factor = NA, n = 0))
  }

  factor <- total_naive / total_default
  message("  FACTOR: ", round(factor, 2), "x")

  list(factor = factor, n = nrow(default_df))
}

load_max_simd_factor <- function(platform_dir) {
  minmax_files <- list.files(platform_dir, pattern = "^min_max_.*\\.csv$", full.names = TRUE)
  if (length(minmax_files) == 0) return(list(factor = NA, n = 0))

  df <- read_csv(minmax_files[1], show_col_types = FALSE)

  message("DEBUG: load_max_simd_factor for ", basename(platform_dir))
  message("  Total rows in file: ", nrow(df))

  # Filter to 100M rows, max aggregate, and dist_test ONLY (not ungrouped_type)
  df <- df %>% filter(rows_m == 100, aggregate == 'max', test == 'dist_test')
  message("  After filtering (rows_m==100, aggregate==max, test==dist_test): ", nrow(df))

  # For dist_test, dtype is the distribution (random, monotonic_inc, monotonic_dec)
  # We want ALL distributions, not filtering by dtype

  default_df <- df %>% filter(variant == 'default')
  naive_df <- df %>% filter(variant == 'nosimd')

  message("  Rows with variant=='default': ", nrow(default_df))
  message("  Rows with variant=='nosimd': ", nrow(naive_df))

  if (nrow(naive_df) > 0) {
    message("  nosimd distributions found:")
    for (i in 1:nrow(naive_df)) {
      message("    dtype=", naive_df$dtype[i], " agg_sec=", naive_df$agg_sec[i])
    }
  }

  if (nrow(default_df) > 0) {
    message("  default distributions found:")
    for (i in 1:nrow(default_df)) {
      message("    dtype=", default_df$dtype[i], " agg_sec=", default_df$agg_sec[i])
    }
  }

  if (nrow(default_df) == 0 || nrow(naive_df) == 0) {
    message("  SKIPPED: Missing default or nosimd variant")
    return(list(factor = NA, n = 0))
  }

  total_default <- sum(default_df$agg_sec[default_df$agg_sec > 0], na.rm = TRUE)
  total_naive <- sum(naive_df$agg_sec[naive_df$agg_sec > 0], na.rm = TRUE)

  message("  total_default (sum of agg_sec): ", round(total_default, 3))
  message("  total_naive (sum of agg_sec): ", round(total_naive, 3))

  if (total_default == 0) {
    message("  SKIPPED: total_default is 0")
    return(list(factor = NA, n = 0))
  }

  factor <- total_naive / total_default
  message("  FACTOR: ", round(factor, 2), "x")

  list(factor = factor, n = nrow(default_df))
}

load_sum_simd_factor <- function(platform_dir) {
  sum_files <- list.files(platform_dir, pattern = "^sum_avg_.*\\.csv$", full.names = TRUE)
  if (length(sum_files) == 0) return(list(factor = NA, n = 0))

  df <- read_csv(sum_files[1], show_col_types = FALSE)
  test_col <- ifelse('itest' %in% colnames(df), 'itest', 'test')

  message("DEBUG: load_sum_simd_factor for ", basename(platform_dir))
  message("  Total rows in file: ", nrow(df))

  df <- df %>% filter(aggregate == 'sum', rows_m == 100,
                     !!sym(test_col) == 'ungrouped_domain')
  message("  After filtering (aggregate==sum, rows_m==100, test==ungrouped_domain): ", nrow(df))

  if ('dtype' %in% colnames(df)) {
    message("  Before dtype filter, unique dtypes: ", paste(unique(df$dtype), collapse=", "))
    # Use all dtypes, not just 'tiny'
    available_dtypes <- unique(df$dtype)
    message("  Using all ", length(available_dtypes), " dtypes for average")
  } else {
    message("  No dtype column found")
  }

  default_df <- df %>% filter(variant == 'default')
  naive_df <- df %>% filter(variant == 'nocascading')

  message("  Rows with variant=='default': ", nrow(default_df))
  message("  Rows with variant=='nocascading': ", nrow(naive_df))

  if (nrow(default_df) == 0 || nrow(naive_df) == 0) {
    message("  SKIPPED: Missing default or nocascading variant")
    return(list(factor = NA, n = 0))
  }

  total_default <- sum(default_df$agg_sec[default_df$agg_sec > 0], na.rm = TRUE)
  total_naive <- sum(naive_df$agg_sec[naive_df$agg_sec > 0], na.rm = TRUE)

  message("  total_default (sum of agg_sec): ", round(total_default, 3))
  message("  total_naive (sum of agg_sec): ", round(total_naive, 3))

  if (total_default == 0) {
    message("  SKIPPED: total_default is 0")
    return(list(factor = NA, n = 0))
  }

  factor <- total_naive / total_default
  message("  FACTOR: ", round(factor, 2), "x")

  list(factor = factor, n = nrow(default_df))
}

plot_simd_improvements <- function(results_dir, output_dir) {
  data <- data.frame()

  for (platform in PLATFORM_ORDER) {
    platform_dir <- file.path(results_dir, platform)
    if (!dir.exists(platform_dir)) next

    count_result <- load_count_simd_factor(platform_dir)
    max_result <- load_max_simd_factor(platform_dir)
    sum_result <- load_sum_simd_factor(platform_dir)

    message(sprintf("%s: count=%.2fx (n=%d), max=%.2fx (n=%d), sum=%.2fx (n=%d)",
                    platform,
                    ifelse(is.na(count_result$factor), 0, count_result$factor), count_result$n,
                    ifelse(is.na(max_result$factor), 0, max_result$factor), max_result$n,
                    ifelse(is.na(sum_result$factor), 0, sum_result$factor), sum_result$n))

    if (!is.na(count_result$factor) || !is.na(max_result$factor) || !is.na(sum_result$factor)) {
      data <- bind_rows(data, data.frame(
        platform = platform,
        platform_name = PLATFORM_NAMES[platform],
        pac_count = ifelse(is.na(count_result$factor), 0, count_result$factor),
        pac_max = ifelse(is.na(max_result$factor), 0, max_result$factor),
        pac_sum = ifelse(is.na(sum_result$factor), 0, sum_result$factor)
      ))
    }
  }

  if (nrow(data) == 0) {
    message("No SIMD data found")
    return(NULL)
  }

  # Reshape for ggplot
  data_long <- data %>%
    pivot_longer(cols = c(pac_count, pac_max, pac_sum),
                 names_to = 'aggregate',
                 values_to = 'factor')

  data_long$platform_name <- factor(data_long$platform_name,
                                     levels = PLATFORM_NAMES[PLATFORM_ORDER])
  data_long$aggregate <- factor(data_long$aggregate, levels = AGG_ORDER)

  # Paper plot settings
  width <- 2000
  height <- 1000
  res <- 200
  base_size <- 40
  base_family <- "Linux Libertine"

  p <- ggplot(data_long, aes(x = platform_name, y = factor, fill = aggregate)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7,
             color = 'black', size = 0.5) +
    geom_text(aes(label = sprintf('%.1fx', factor)),
              position = position_dodge(width = 0.8),
              vjust = -0.3, size = base_size * 0.18, fontface = 'bold') +
    geom_hline(yintercept = 1, linetype = 'dashed', color = 'gray', size = 0.8) +
    scale_fill_manual(values = AGG_COLORS, name = 'Aggregate') +
    labs(x = 'Architecture',
         y = 'Improvement Factor') +
    theme_bw(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.major = element_line(linewidth = 1.0),
      panel.grid.minor = element_blank(),
      legend.position = 'top',
      legend.justification = 'left',
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, -10, 0),
      legend.title = element_text(size = base_size - 14),
      legend.text = element_text(size = base_size - 16),
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = base_size - 20),
      axis.text.y = element_text(size = base_size - 12),
      axis.title = element_text(size = base_size - 10),
      plot.margin = margin(5, 5, 5, 5)
    ) +
    ylim(0, 50)

  output_file <- file.path(output_dir, 'simd_improvements_paper.png')
  png(filename = output_file, width = width, height = height, res = res)
  print(p)
  dev.off()
  message("Saved: ", output_file)
}

# ============================================================================
# 5. Hash Distribution Plot
# ============================================================================

plot_hash_distribution <- function(output_dir) {
  # Data from binomial.out
  num_ones <- 20:44
  observed <- c(4692, 10057, 19764, 35913, 61227, 98015, 147038, 206522, 273576,
                338837, 395711, 434562, 446854, 432363, 395100, 339178, 272061,
                205717, 147131, 97415, 61223, 35483, 19665, 10050, 4879)
  expected <- c(4786.1, 10028.1, 19600.4, 35792.0, 61144.7, 97831.5, 146747.3,
                206533.3, 272919.0, 338796.0, 395262.0, 433513.1, 447060.4,
                433513.1, 395262.0, 338796.0, 272919.0, 206533.3, 146747.3,
                97831.5, 61144.7, 35792.0, 19600.4, 10028.1, 4786.1)

  df <- data.frame(
    num_ones = num_ones,
    observed = observed,
    expected = expected
  )

  # Paper plot settings
  width <- 4000
  height <- 1800
  res <- 200
  base_size <- 40
  base_family <- "Linux Libertine"

  p <- ggplot(df, aes(x = num_ones)) +
    geom_col(aes(y = observed), width = 0.8, alpha = 0.7, fill = '#3498db',
             color = 'black', size = 0.5) +
    geom_line(aes(y = expected), color = '#e74c3c', size = 1.5) +
    geom_point(aes(y = expected), color = '#e74c3c', size = 3) +
    labs(x = "Number of 1's in 64-bit hash",
         y = 'Observed frequency') +
    theme_bw(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.major = element_line(linewidth = 1.0),
      panel.grid.minor = element_blank(),
      legend.position = 'top',
      axis.text.x = element_text(size = base_size - 16),
      axis.text.y = element_text(size = base_size - 12),
      axis.title = element_text(size = base_size - 10),
      plot.margin = margin(5, 5, 5, 5)
    ) +
    annotate('text', x = 40, y = max(observed) * 0.9,
             label = 'DuckDB hash(c_custkey)', color = '#3498db',
             fontface = 'bold', size = base_size * 0.4) +
    annotate('text', x = 40, y = max(observed) * 0.8,
             label = 'Binomial(n=64, p=0.5)', color = '#e74c3c',
             fontface = 'bold', size = base_size * 0.4)

  output_file <- file.path(output_dir, 'hash_distribution_paper.png')
  png(filename = output_file, width = width, height = height, res = res)
  print(p)
  dev.off()
  message("Saved: ", output_file)
}

# ============================================================================
# Main Execution
# ============================================================================

main <- function() {
  # Get script directory - works with both source() and Rscript
  args <- commandArgs(trailingOnly = FALSE)
  script_path <- NULL

  # Try to find --file= argument
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg[1])
    script_dir <- dirname(script_path)
  } else {
    # Fallback to current directory
    script_dir <- getwd()
  }

  results_dir <- file.path(script_dir, 'results')
  output_dir <- file.path(script_dir, 'plots')

  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }

  message("\n=== Plotting PAC Microbenchmarks ===\n")

  # Plot per-platform optimizations
  for (platform in list.dirs(results_dir, full.names = FALSE, recursive = FALSE)) {
    if (platform == "") next

    platform_name <- ifelse(platform %in% names(PLATFORM_NAMES),
                            PLATFORM_NAMES[platform], platform)

    message("\n--- ", platform_name, " ---")

    plot_count_optimizations(platform, platform_name, results_dir, output_dir)
    plot_sum_optimizations(platform, platform_name, results_dir, output_dir)
    plot_minmax_optimizations(platform, platform_name, results_dir, output_dir)
  }

  # Cross-platform plots
  message("\n--- Cross-platform plots ---")
  plot_simd_improvements(results_dir, output_dir)
  plot_hash_distribution(output_dir)

  message("\n=== All plots completed ===\n")
}

# Run main if script is executed directly
if (!interactive()) {
  main()
}



