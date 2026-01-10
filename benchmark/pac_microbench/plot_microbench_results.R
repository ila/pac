#!/usr/bin/env Rscript
# plot_microbench_results.R
# Scans results/* subfolders and generates simple COUNT plots (ungrouped, grouped_seq, grouped_scat)

suppressPackageStartupMessages({
  # core plotting/data packages only
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
if (!dir.exists(results_root)) stop("results directory not found: ", results_root)
# discover architecture directories (subfolders of results)
candidates <- list.files(results_root, full.names = FALSE)
arch_dirs <- file.path(results_root, candidates[file.info(file.path(results_root, candidates))$isdir])
# if no subdirectories, include results_root itself
if (length(arch_dirs) == 0) arch_dirs <- results_root

# Find latest file matching prefix with timestamp YYYYMMDD_HHMMSS
find_latest_file <- function(dirpath, prefix) {
  files <- list.files(dirpath, pattern = paste0("^", prefix, "_\\d{8}_\\d{6}.*\\.csv$"), full.names = TRUE)
  if (length(files) == 0) return(NULL)
  stamps <- stringr::str_extract(basename(files), "\\d{8}_\\d{6}")
  valid <- !is.na(stamps)
  if (!any(valid)) return(NULL)
  files <- files[valid]
  stamps <- stamps[valid]
  nums <- as.numeric(gsub("_", "", stamps))
  files[which.max(nums)]
}

# Normalize common column names so downstream code can rely on wall_sec, rows_m, groups, variant, test
normalize_df <- function(df) {
  nm <- names(df)
  if ("wall_sec" %in% nm && !("time_sec" %in% nm)) df$time_sec <- df$wall_sec
  if ("time_sec" %in% nm && !("wall_sec" %in% nm)) df$wall_sec <- df$time_sec
  if ("rows_m" %in% nm && !("data_size_m" %in% nm)) df$data_size_m <- df$rows_m
  if ("data_size_m" %in% nm && !("rows_m" %in% nm)) df$rows_m <- df$data_size_m
  # ensure these exist to avoid errors
  if (!"variant" %in% nm) df$variant <- NA_character_
  if (!"test" %in% nm) df$test <- NA_character_
  if (!"groups" %in% nm) df$groups <- NA
  if (!"rows_m" %in% nm) df$rows_m <- NA
  if (!"wall_sec" %in% nm && "time_sec" %in% nm) df$wall_sec <- df$time_sec
  df
}

# Plot generator for a single architecture directory
process_arch <- function(arch_dir) {
  message("Processing: ", arch_dir)
  file <- find_latest_file(arch_dir, "count")
  if (is.null(file)) { message("  no count file found in ", arch_dir); return(invisible(NULL)) }
  message("  using: ", file)

  df <- tryCatch(readr::read_csv(file, show_col_types = FALSE), error = function(e) { message("  failed to read ", file, ": ", e$message); return(NULL) })
  if (is.null(df)) return(invisible(NULL))
  df <- normalize_df(df)

  # make sure wall_sec is numeric and build a cleaned timing column
  if (!"wall_sec" %in% names(df) && !"time_sec" %in% names(df)) { message("  missing wall_sec/time_sec in ", file); return(invisible(NULL)) }
  # raw numeric from primary column
  raw_time_col <- if ("wall_sec" %in% names(df)) "wall_sec" else "time_sec"
  df$wall_sec_raw <- suppressWarnings(as.numeric(df[[raw_time_col]]))
  # Map -1 (crash marker) to 0 for display; leave other negatives untouched unless you want to treat them too
  df$wall_sec_clean <- df$wall_sec_raw
  df$wall_sec_clean[which(df$wall_sec_clean == -1)] <- 0

  # If wall_times exists (space-separated), prefer its first sample where cleaned value is NA or 0
  if ("wall_times" %in% names(df)) {
    first_num_from_times <- function(s) {
      if (is.na(s)) return(NA_real_)
      toks <- stringr::str_extract_all(as.character(s), "-?\\d+\\.?\\d*")[[1]]
      if (length(toks) == 0) return(NA_real_) else return(as.numeric(toks[1]))
    }
    wt_first <- vapply(df$wall_times, first_num_from_times, FUN.VALUE = NA_real_)
    idx_fill <- which(is.na(df$wall_sec_clean) | df$wall_sec_clean == 0)
    tofill <- idx_fill[!is.na(wt_first[idx_fill])]
    if (length(tofill) > 0) df$wall_sec_clean[tofill] <- wt_first[tofill]
  }
  # fallback to agg_sec if still missing or zero
  if ("agg_sec" %in% names(df)) {
    idx_fill <- which(is.na(df$wall_sec_clean) | df$wall_sec_clean == 0)
    if (length(idx_fill) > 0) df$wall_sec_clean[idx_fill] <- suppressWarnings(as.numeric(df$agg_sec[idx_fill]))
  }

  # Report and ensure we have a plotting column. We'll display zeros (original -1) as 0, but for log plotting we must replace 0 with a tiny epsilon
  df$wall_sec_clean <- suppressWarnings(as.numeric(df$wall_sec_clean))
  n_before <- nrow(df)
  n_missing <- sum(is.na(df$wall_sec_clean))
  if (n_missing > 0) message("  note: ", n_missing, " rows have no timing after fallback (out of ", n_before, ") and will be skipped in plots")

  # For linear plotting we keep zeros as zeros; prepare plotting column
  df$wall_sec_plot <- suppressWarnings(as.numeric(df$wall_sec_clean))
  # create a readable groups label (format large numbers with commas or suffixes)
  format_group_label <- function(x) {
    n <- suppressWarnings(as.numeric(as.character(x)))
    if (is.na(n)) return(as.character(x))
    if (n >= 1e6) return(paste0(n/1e6, "M"))
    if (n >= 1e3) return(formatC(n, format = "d", big.mark = ","))
    return(as.character(n))
  }
  df$groups_label <- vapply(df$groups, format_group_label, FUN.VALUE = "")
  # facet titles: compute numeric value (millions) and show 'M' or 'B' suffixes; order levels by numeric size
  df$rows_m_num <- suppressWarnings(as.numeric(as.character(df$rows_m)))
  df$rows_m_label <- vapply(df$rows_m_num, function(rnum) {
    if (is.na(rnum)) return(NA_character_)
    if (rnum %% 1000 == 0 && rnum >= 1000) {
      # convert 1000M -> 1B, 2000M -> 2B, etc.
      return(paste0(rnum/1000, "B rows"))
    } else {
      return(paste0(rnum, "M rows"))
    }
  }, FUN.VALUE = "")
  # make the rows_m_label a factor preserving the default appearance order (order of first occurrence in the data)
  # Use unique() on the mapped labels to preserve input order rather than sorting numerically
  labels_in_order <- unique(df$rows_m_label[!is.na(df$rows_m_label)])
  if (length(labels_in_order) > 0) {
    df$rows_m_label <- factor(df$rows_m_label, levels = labels_in_order)
  } else {
    df$rows_m_label <- factor(df$rows_m_label)
  }

  df$variant <- as.factor(ifelse(is.na(df$variant), "unknown", df$variant))
  # Use rows_m as facet variable; convert to factor so each facet is one subplot
  df$rows_m <- as.factor(df$rows_m)
  df$groups <- as.factor(df$groups)

  # Plotting helper: create file and write png
  make_plot <- function(sub, outname, title) {
    if (nrow(sub) == 0) { message("  no data for ", title); return() }
    # remove rows with no timing at all
    sub <- sub %>% filter(!is.na(wall_sec_clean))
    if (nrow(sub) == 0) { message("  no timing rows for ", title); return() }

    p <- ggplot(sub, aes(x = groups_label, y = wall_sec_plot, fill = variant)) +
      geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
      facet_wrap(~ rows_m_label, scales = "free_x", nrow = 1) +
      scale_y_continuous(labels = scales::label_number(accuracy = 0.01, big.mark = ",", decimal.mark = ".")) +
      labs(x = "groups", y = "Wall time (s)", fill = "Variant", title = title) +
      theme_bw(base_size = 14) +
      theme(
        legend.position = "top",
        axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
        axis.text.y = element_text(size = 12),
        axis.title = element_text(size = 14),
        legend.title = element_text(size = 13),
        legend.text = element_text(size = 12),
        strip.text = element_text(size = 13),
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
        strip.background = element_rect(fill = "#f0f0f0")
      )
    out_path <- file.path(arch_dir, outname)
    ggsave(out_path, p, width = 12, height = 4, dpi = 300)
    message("  wrote ", out_path)
  }

  # Un-grouped: test == 'ungrouped'
  sub_ungrouped <- df %>% filter(test == 'ungrouped')
  make_plot(sub_ungrouped, 'count_ungrouped.png', paste(basename(arch_dir), '- ungrouped'))

  # Grouped seq: test == 'grouped_seq' or 'grouped' or contains grouped_seq
  sub_grouped_seq <- df %>% filter(test %in% c('grouped_seq', 'grouped'))
  make_plot(sub_grouped_seq, 'count_grouped_seq.png', paste(basename(arch_dir), '- grouped_seq'))

  # Grouped scat: test == 'grouped_scat' or contains 'scat'
  sub_grouped_scat <- df %>% filter(test %in% c('grouped_scat') | grepl('scat', test))
  make_plot(sub_grouped_scat, 'count_grouped_scat.png', paste(basename(arch_dir), '- grouped_scat'))

  # --- NEW: Variants plot at max rows_m ---
  # Determine the numeric rows_m to use (prefer rows_m_num computed earlier)
  rows_nums <- df$rows_m_num
  max_rows_num <- NA_real_
  if (any(!is.na(rows_nums))) {
    max_rows_num <- max(rows_nums, na.rm = TRUE)
    df_max <- df %>% filter(rows_m_num == max_rows_num)
  } else {
    # fallback: pick the most frequent rows_m value
    if (any(!is.na(df$rows_m))) {
      mode_row <- df %>% count(rows_m) %>% arrange(desc(n)) %>% slice(1) %>% pull(rows_m)
      df_max <- df %>% filter(rows_m == mode_row)
    } else {
      df_max <- df
    }
  }

  if (nrow(df_max) == 0) {
    message("  no rows to plot variants at max rows for ", arch_dir)
  } else {
    # create a clean groups_label if missing
    if (!"groups_label" %in% names(df_max)) {
      df_max$groups_label <- as.character(df_max$groups)
    }
    df_max$variant <- as.factor(ifelse(is.na(df_max$variant), "unknown", df_max$variant))
    # Facet by test type (boxes = test), keep variant as color. Order tests with common order if present.
    desired_tests <- c('ungrouped', 'grouped_seq', 'grouped_scat')
    present_tests <- unique(as.character(df_max$test))
    test_levels <- c(intersect(desired_tests, present_tests), setdiff(present_tests, desired_tests))
    df_max$test <- factor(as.character(df_max$test), levels = test_levels)

    pvar <- ggplot(df_max, aes(x = groups_label, y = wall_sec_plot, fill = variant)) +
      geom_col(position = position_dodge2(width = 0.9, preserve = "single")) +
      facet_wrap(~ test, scales = "free_y", nrow = 1) +
      scale_y_continuous(labels = scales::label_number(accuracy = 0.01, big.mark = ",", decimal.mark = ".")) +
      labs(x = "groups", y = "Wall time (s)", fill = "Variant", title = paste(basename(arch_dir), "- variants at max rows")) +
      theme_bw(base_size = 14) +
      theme(
        legend.position = "top",
        axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
        axis.text.y = element_text(size = 12),
        axis.title = element_text(size = 14),
        strip.text = element_text(size = 13),
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold")
      )
    outp <- file.path(arch_dir, "count_variants_maxrows.png")
    ggsave(outp, pvar, width = 14, height = 4, dpi = 300)
    message("  wrote ", outp)
  }
}

# Run for every arch dir
for (d in arch_dirs) {
  tryCatch(process_arch(d), error = function(e) message("Error processing ", d, ": ", e$message))
}

message("Done")
