#!/usr/bin/env Rscript
# plot_microbench_results.R
#
# Scans results/* subfolders and generates simple COUNT plots
# (ungrouped, grouped_seq, grouped_scat)

suppressPackageStartupMessages({
  if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
  if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
  if (!requireNamespace("readr", quietly = TRUE)) install.packages("readr")
  if (!requireNamespace("stringr", quietly = TRUE)) install.packages("stringr")
  if (!requireNamespace("scales", quietly = TRUE)) install.packages("scales")

  library(ggplot2)
  library(dplyr)
  library(readr)
  library(stringr)
  library(scales)
})

results_root <- "results"
if (!dir.exists(results_root)) {
  stop("results directory not found: ", results_root)
}

# ------------------------------------------------------------------------------
# Discover architecture directories
# ------------------------------------------------------------------------------

candidates <- list.files(results_root, full.names = FALSE)
arch_dirs <- file.path(
  results_root,
  candidates[file.info(file.path(results_root, candidates))$isdir]
)

if (length(arch_dirs) == 0) {
  arch_dirs <- results_root
}

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

find_latest_file <- function(dirpath, prefix) {
  files <- list.files(
    dirpath,
    pattern = paste0("^", prefix, "_\\d{8}_\\d{6}.*\\.csv$"),
    full.names = TRUE
  )

  if (length(files) == 0) return(NULL)

  stamps <- str_extract(basename(files), "\\d{8}_\\d{6}")
  valid  <- !is.na(stamps)
  if (!any(valid)) return(NULL)

  files  <- files[valid]
  stamps <- stamps[valid]
  nums   <- as.numeric(gsub("_", "", stamps))

  files[which.max(nums)]
}

normalize_df <- function(df) {
  nm <- names(df)

  if ("wall_sec" %in% nm && !"time_sec" %in% nm) df$time_sec <- df$wall_sec
  if ("time_sec" %in% nm && !"wall_sec" %in% nm) df$wall_sec <- df$time_sec

  if ("rows_m" %in% nm && !"data_size_m" %in% nm) df$data_size_m <- df$rows_m
  if ("data_size_m" %in% nm && !"rows_m" %in% nm) df$rows_m <- df$data_size_m

  if (!"variant" %in% nm) df$variant <- NA_character_
  if (!"test"    %in% nm) df$test    <- NA_character_
  if (!"groups"  %in% nm) df$groups  <- NA
  if (!"rows_m"  %in% nm) df$rows_m  <- NA

  df
}

# ------------------------------------------------------------------------------
# Main processing per architecture
# ------------------------------------------------------------------------------

process_arch <- function(arch_dir) {
  message("Processing: ", arch_dir)

  file <- find_latest_file(arch_dir, "count")
  if (is.null(file)) {
    message("  no count file found")
    return(invisible(NULL))
  }

  message("  using: ", file)

  df <- tryCatch(
    read_csv(file, show_col_types = FALSE),
    error = function(e) {
      message("  failed to read: ", e$message)
      NULL
    }
  )
  if (is.null(df)) return(invisible(NULL))

  df <- normalize_df(df)

  # Remove the 'standard' variant from all further processing
  df <- df %>% filter(variant != "standard")

  raw_time_col <- if ("wall_sec" %in% names(df)) "wall_sec" else "time_sec"
  df$wall_sec_raw   <- suppressWarnings(as.numeric(df[[raw_time_col]]))
  # (NO OVERRIDE) use raw wall times for all rows, including ungrouped

  # treat -1 as missing (NA) so single crash markers don't zero out the mean for a group/variant
  df$wall_sec_clean <- df$wall_sec_raw
  df$wall_sec_clean[!is.na(df$wall_sec_clean) & df$wall_sec_clean == -1] <- NA_real_

  if ("wall_times" %in% names(df)) {
    first_from_times <- function(s) {
      if (is.na(s)) return(NA_real_)
      toks <- str_extract_all(as.character(s), "-?\\d+\\.?\\d*")[[1]]
      if (length(toks) == 0) NA_real_ else as.numeric(toks[1])
    }

    wt_first <- vapply(df$wall_times, first_from_times, NA_real_)
    idx <- which(is.na(df$wall_sec_clean) | df$wall_sec_clean == 0)
    df$wall_sec_clean[idx] <- wt_first[idx]
  }

  df$wall_sec_plot <- as.numeric(df$wall_sec_clean)

  df$groups_label <- vapply(
    df$groups,
    function(x) {
      n <- suppressWarnings(as.numeric(as.character(x)))
      if (is.na(n)) return(as.character(x))
      if (n >= 1e6) return(paste0(n / 1e6, "M"))
      if (n >= 1e3) return(formatC(n, format = "d", big.mark = ","))
      as.character(n)
    },
    FUN.VALUE = ""
  )

  df$rows_m_num <- suppressWarnings(as.numeric(as.character(df$rows_m)))
  df$rows_m_label <- vapply(
    df$rows_m_num,
    function(r) {
      if (is.na(r)) NA_character_
      else if (r %% 1000 == 0 && r >= 1000) paste0(r / 1000, "B rows")
      else paste0(r, "M rows")
    },
    FUN.VALUE = ""
  )

  df$variant <- as.factor(ifelse(is.na(df$variant), "unknown", df$variant))
  df$groups  <- as.factor(df$groups)

  # --------------------------------------------------------------------------
  # Plot helpers
  # --------------------------------------------------------------------------

  make_plot <- function(sub, outname, test_name, arch_name) {
    sub <- sub %>% filter(!is.na(wall_sec_clean))
    if (nrow(sub) == 0) return()

    # Convert seconds to milliseconds for plotting
    sub$wall_ms_plot <- sub$wall_sec_plot * 1000

    # Format the title as: {architecture} - COUNT, {test} (all caps)
    plot_title <- paste0(toupper(arch_name), " - COUNT, ", toupper(test_name))

    p <- ggplot(sub, aes(groups_label, wall_ms_plot, fill = variant)) +
      geom_col(position = position_dodge2(width = 0.9)) +
      facet_wrap(~ rows_m_label, nrow = 1, scales = "free_x") +
      labs(x = "groups", y = "Wall time (ms)", fill = "Variant", title = plot_title) +
      scale_y_log10(labels = scales::label_comma()) +
      theme_bw(base_size = 14) +
      theme(
        legend.position = "top",
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold", hjust = 0.5)
      )

    ggsave(file.path(arch_dir, outname), p, width = 12, height = 6, dpi = 300)
  }

  # --------------------------------------------------------------------------
  # Subsets
  # --------------------------------------------------------------------------

  sub_ungrouped <- df %>%
    filter(test == "ungrouped")

  sub_grouped_seq <- df %>%
    filter(test %in% c("grouped_seq", "grouped"))

  sub_grouped_scat <- df %>%
    filter(grepl("scat", test))

  arch_name <- basename(arch_dir)
  make_plot(sub_ungrouped,     "count_ungrouped.png",     "ungrouped", arch_name)
  make_plot(sub_grouped_seq,   "count_grouped_seq.png",   "grouped_seq", arch_name)
  make_plot(sub_grouped_scat,  "count_grouped_scat.png",  "grouped_scat", arch_name)
}

# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------

for (d in arch_dirs) {
  tryCatch(process_arch(d),
           error = function(e) message("Error in ", d, ": ", e$message))
}

message("Done")
