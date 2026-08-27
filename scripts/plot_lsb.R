#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(stringr)
  library(forcats)
  library(tidyr)
  library(purrr)
  library(tibble)
  library(scales)
  library(parallel)
})
# lsb_common.R (shared with plot_heatmap.R) owns file parsing, caching,
# region classification, and several of the constants below -- sourced
# rather than reimplemented, so the two tools can never drift apart on how
# an LSB file gets parsed. This script used to carry its own independent
# copy of all of that, which is how it ended up with a compiler-mislabeling
# bug and no device-vs-host pooling that lsb_common.R never had in the
# first place.
.script_dir <- tryCatch({
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    dirname(normalizePath(sub("^--file=", "", file_arg[1])))
  } else {
    getwd()
  }
}, error = function(e) getwd())
source(file.path(.script_dir, "lsb_common.R"))
# lsb_common.R defines its own log_msg() (prefixed "[+Ns]"); redefine it
# here so this script's own log lines stay distinguishable as
# "[plot_lsb +Ns]" when its output is read alongside plot_heatmap.R's.
SCRIPT_START <- Sys.time()
elapsed_s <- function() {
  as.numeric(difftime(Sys.time(), SCRIPT_START, units = "secs"))
}
log_msg <- function(...) {
  cat(sprintf("[plot_lsb +%6.2fs] ", elapsed_s()), sprintf(...), "\n", sep = "")
  flush.console()
}
POINT_SAMPLE_PER_GROUP <- as.integer(Sys.getenv("PLOT_LSB_POINT_SAMPLE", "250"))
if (is.na(POINT_SAMPLE_PER_GROUP) || POINT_SAMPLE_PER_GROUP < 0L) {
  POINT_SAMPLE_PER_GROUP <- 250L
}
SHOW_POINTS <- Sys.getenv("PLOT_LSB_SHOW_POINTS", "1") != "0"
args <- commandArgs(trailingOnly = TRUE)
results_dir <- ifelse(length(args) >= 1, args[[1]], "results")
out_dir <- ifelse(length(args) >= 2, args[[2]], file.path(results_dir, "plots"))
filter_app <- NULL
filter_backend <- NULL
plot_mode <- "full"
if (length(args) > 2) {
  i <- 3
  while (i <= length(args)) {
    if (args[[i]] == "--app" && i + 1 <= length(args)) {
      filter_app <- args[[i + 1]]
      i <- i + 2
    } else if (args[[i]] == "--backend" && i + 1 <= length(args)) {
      filter_backend <- args[[i + 1]]
      i <- i + 2
    } else if (args[[i]] == "--mode" && i + 1 <= length(args)) {
      plot_mode <- args[[i + 1]]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
}
if (!(plot_mode %in% c("light", "full"))) {
  stop("Unknown plot mode: ", plot_mode)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
# backend_colours, implementation_levels, region_class_levels, and
# classify_region() now come from lsb_common.R (sourced above) -- kept
# there so plot_heatmap.R can never disagree with this script about them.
# problem_size_levels stays local: it adds "default" (fixed-size
# benchmarks with no sized variants) on top of lsb_common.R's SIZE_LEVELS,
# which plot_heatmap.R has no need for.
problem_size_levels <- c(SIZE_LEVELS, "default")
theme_pub <- function() {
  theme_bw(base_size = 14) +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1),
      panel.grid.minor = element_blank(),
      legend.position = "right"
    )
}
scale_impl_fill <- function() {
  scale_fill_manual(values = backend_colours, aesthetics = "fill", na.value = "grey70")
}
save_plot <- function(plot, path, width = 10, height = 5) {
  ggsave(path, plot, width = width, height = height)
}
normalise_by_fastest <- function(data, value_col, group_cols) {
  value_col <- enquo(value_col)
  data |>
    group_by(across(all_of(group_cols))) |>
    mutate(.baseline = min(!!value_col, na.rm = TRUE)) |>
    ungroup() |>
    mutate(normalised = (!!value_col) / .baseline) |>
    select(-.baseline)
}
sample_for_points <- function(df, group_cols, max_points_per_group = POINT_SAMPLE_PER_GROUP) {
  if (!SHOW_POINTS || max_points_per_group <= 0L || nrow(df) == 0L) {
    return(df[0, , drop = FALSE])
  }
  df |>
    group_by(across(all_of(group_cols))) |>
    group_modify(function(.x, .y) {
      slice_sample(.x, n = min(nrow(.x), max_points_per_group))
    }) |>
    ungroup()
}
jitter_points <- function(
  df,
  group_cols,
  width = 0.12,
  alpha = 0.25,
  size = 0.7
) {
  if (!SHOW_POINTS || POINT_SAMPLE_PER_GROUP <= 0L || nrow(df) == 0L) {
    return(NULL)
  }
  geom_jitter(
    data = sample_for_points(df, group_cols),
    width = width,
    alpha = alpha,
    size = size
  )
}
# File parsing, region classification, and caching (extract_runtime,
# extract_nodename_fast, parse_lsb_filename, parse_lsb_table_dt,
# read_lsb_file_fast, parse_files_with_progress, read_lsb_cached) all now
# come from lsb_common.R (sourced above) instead of this script's own
# former copies of the same logic. Two behavioural changes fall out of
# this, both intentional:
#   - Rows are now pre-aggregated to one row per (file, region) --
#     total_time_us summed and n_repeats counted across every internal
#     stabilisation-loop pass within that file/region -- rather than one
#     row per raw table entry. Boxplots/jitter below now show spread
#     across independent runs (the repeated ./runner.sh invocations), not
#     a mix of that AND raw within-file repeat-loop rows the way the old
#     parser did -- the old mixing was mild pseudoreplication; this is the
#     statistically correct grouping.
#   - lsb_common.R's `run` column is the digit right after ".r" in the
#     filename (e.g. "r0"), which has been constant (0) in every LSB file
#     seen so far. The actual per-invocation repeat index this script
#     always cared about is the "-N" suffix (e.g. "r0-3"), which
#     lsb_common.R calls `run_variant`. Renamed below so the rest of this
#     file (jitter/boxplot grouping, run-level medians) keeps working
#     exactly as before without needing every "run" reference rewritten.
make_benchmark_plots <- function(bench_df, bench_out) {
  dir.create(bench_out, recursive = TRUE, showWarnings = FALSE)
  bench_name <- unique(bench_df$benchmark)
  if (length(bench_name) != 1) {
    stop("make_benchmark_plots() received multiple benchmarks")
  }
  # problem_size may be genuinely absent for benchmarks with no sized
  # variants (e.g. a fixed problem) -- treat NA as its own facet level
  # rather than silently dropping those rows.
  bench_df <- bench_df |>
    mutate(
      problem_size = factor(
        problem_size,
        levels = c(problem_size_levels, setdiff(sort(unique(as.character(problem_size))), problem_size_levels))
      )
    )
  runtime_df <- bench_df |>
    distinct(device, benchmark, backend, compiler, implementation, problem_size, run, runtime_s) |>
    filter(!is.na(runtime_s))
  if (nrow(runtime_df) > 0) {
    total_runtime_plot <- ggplot(
      runtime_df,
      aes(x = implementation, y = runtime_s * 1e3, fill = implementation)
    ) +
      geom_boxplot(outlier.shape = NA) +
      jitter_points(
        runtime_df,
        c("device", "benchmark", "implementation", "problem_size"),
        alpha = 0.35,
        size = 1.1
      ) +
      facet_grid(problem_size ~ device, scales = "free_x") +
      scale_impl_fill() +
      labs(
        x = NULL,
        y = "Runtime (ms)",
        fill = "Implementation",
        title = paste0(bench_name, ": end-to-end runtime")
      ) +
      theme_pub()
    save_plot(total_runtime_plot, file.path(bench_out, "total_runtime_boxplot.pdf"))
    runtime_norm <- runtime_df |>
      mutate(runtime_ms = runtime_s * 1e3) |>
      group_by(device, benchmark, implementation, problem_size, run) |>
      summarise(runtime_ms = median(runtime_ms), .groups = "drop") |>
      normalise_by_fastest(runtime_ms, c("device", "benchmark", "problem_size"))
    total_runtime_norm_plot <- ggplot(
      runtime_norm,
      aes(x = implementation, y = normalised, fill = implementation)
    ) +
      geom_col() +
      facet_grid(problem_size ~ device, scales = "free_x") +
      scale_impl_fill() +
      labs(
        x = NULL,
        y = "Normalised runtime (fastest = 1.0)",
        fill = "Implementation",
        title = paste0(bench_name, ": normalised end-to-end runtime")
      ) +
      theme_pub()
    save_plot(total_runtime_norm_plot, file.path(bench_out, "total_runtime_normalised.pdf"))
  }
  region_class_breakdown <- bench_df |>
    group_by(device, benchmark, implementation, problem_size, run, region_class) |>
    summarise(
      time_us = sum(time_us),
      # repeat_id doesn't exist any more (lsb_common.R aggregates it away
      # per file+region) -- n_repeats is that same per-row repeat count.
      # max(), not sum(): repeats_to_two_seconds is a property of the
      # stabilisation loop the whole FILE goes through, so every region in
      # the same region_class within one run shares the same repeat count
      # rather than adding up across regions.
      internal_repeats = max(n_repeats),
      .groups = "drop"
    ) |>
    group_by(device, benchmark, implementation, problem_size, region_class) |>
    summarise(
      time_us = median(time_us),
      internal_repeats = median(internal_repeats),
      .groups = "drop"
    ) |>
    mutate(region_class = factor(region_class, levels = region_class_levels))
  region_class_plot <- ggplot(
    region_class_breakdown,
    aes(x = implementation, y = time_us / 1000, fill = region_class)
  ) +
    geom_col() +
    facet_grid(problem_size ~ device, scales = "free_x") +
    labs(
      x = NULL,
      y = "Median measured region time (ms)",
      fill = "Region class",
      title = paste0(bench_name, ": runtime composition")
    ) +
    theme_pub()
  save_plot(region_class_plot, file.path(bench_out, "region_class_breakdown.pdf"))
  region_class_norm <- region_class_breakdown |>
    group_by(device, benchmark, implementation, problem_size) |>
    summarise(time_us = sum(time_us), .groups = "drop") |>
    normalise_by_fastest(time_us, c("device", "benchmark", "problem_size"))
  region_class_norm_plot <- ggplot(
    region_class_norm,
    aes(x = implementation, y = normalised, fill = implementation)
  ) +
    geom_col() +
    facet_grid(problem_size ~ device, scales = "free_x") +
    scale_impl_fill() +
    labs(
      x = NULL,
      y = "Normalised measured region time (fastest = 1.0)",
      fill = "Implementation",
      title = paste0(bench_name, ": normalised measured region total")
    ) +
    theme_pub()
  save_plot(region_class_norm_plot, file.path(bench_out, "region_class_total_normalised.pdf"))
  region_percent <- region_class_breakdown |>
    group_by(device, benchmark, implementation, problem_size) |>
    mutate(percent = 100 * time_us / sum(time_us)) |>
    ungroup()
  region_percent_plot <- ggplot(
    region_percent,
    aes(x = implementation, y = percent, fill = region_class)
  ) +
    geom_col() +
    facet_grid(problem_size ~ device, scales = "free_x") +
    labs(
      x = NULL,
      y = "Share of measured region time (%)",
      fill = "Region class",
      title = paste0(bench_name, ": runtime composition share")
    ) +
    theme_pub()
  save_plot(region_percent_plot, file.path(bench_out, "region_class_percent.pdf"))
  actual_region_breakdown <- bench_df |>
    group_by(device, benchmark, implementation, problem_size, run, region) |>
    summarise(
      time_us = sum(time_us),
      # One row per (run, region) already at this point (lsb_common.R
      # pre-aggregates to file+region granularity), so this sum/max/first
      # are all equivalent -- kept as a sum for symmetry with
      # region_class_breakdown above, and because it's a no-op either way.
      internal_repeats = sum(n_repeats),
      .groups = "drop"
    ) |>
    group_by(device, benchmark, implementation, problem_size, region) |>
    summarise(
      time_us = median(time_us),
      internal_repeats = median(internal_repeats),
      .groups = "drop"
    ) |>
    arrange(device, benchmark, implementation, problem_size, desc(time_us))
  actual_region_labels <- bench_df |>
    filter(region_class == "Kernel") |>
    group_by(device, benchmark, implementation, problem_size, run) |>
    # The old per-row `id` column (a raw table row id) no longer exists --
    # lsb_common.R's n_rows (raw row count per file+region, before
    # aggregation) is the equivalent "how many individual measurements
    # were taken" count. Summed across every Kernel-classified region in
    # this run, matching what n_distinct(id) counted across raw rows.
    summarise(kernel_runs = sum(n_rows), .groups = "drop") |>
    group_by(device, benchmark, implementation, problem_size) |>
    summarise(kernel_runs = median(kernel_runs), .groups = "drop") |>
    left_join(
      actual_region_breakdown |>
        group_by(device, benchmark, implementation, problem_size) |>
        summarise(time_us = sum(time_us), .groups = "drop"),
      by = c("device", "benchmark", "implementation", "problem_size")
    ) |>
    mutate(
      label = case_when(
        kernel_runs <= 1 ~ NA_character_,
        kernel_runs >= 1000 ~ paste0(
          "×",
          format(round(kernel_runs / 100) / 10, nsmall = 1),
          "k"
        ),
        TRUE ~ paste0("×", format(round(kernel_runs)))
      )
    ) |>
    filter(!is.na(label))
  actual_region_plot <- ggplot(
    actual_region_breakdown,
    aes(x = implementation, y = time_us / 1000, fill = region)
  ) +
    geom_col() +
    geom_text(
      data = actual_region_labels,
      aes(
        x = implementation,
        y = time_us / 1000,
        label = label
      ),
      inherit.aes = FALSE,
      vjust = -0.25,
      size = 3
    ) +
    facet_grid(problem_size ~ device, scales = "free_x") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.14))) +
    labs(
      x = NULL,
      y = "Median measured region time over ~2 s window (ms)",
      fill = "Region",
      title = paste0(bench_name, ": actual region breakdown over stabilized ~2 s window")
    ) +
    theme_pub()
  save_plot(actual_region_plot, file.path(bench_out, "actual_region_breakdown.pdf"), width = 11, height = 6)
  actual_region_norm <- actual_region_breakdown |>
    group_by(device, benchmark, implementation, problem_size) |>
    summarise(time_us = sum(time_us), .groups = "drop") |>
    normalise_by_fastest(time_us, c("device", "benchmark", "problem_size"))
  actual_region_norm_plot <- ggplot(
    actual_region_norm,
    aes(x = implementation, y = normalised, fill = implementation)
  ) +
    geom_col() +
    facet_grid(problem_size ~ device, scales = "free_x") +
    scale_impl_fill() +
    labs(
      x = NULL,
      y = "Normalised measured region time (fastest = 1.0)",
      fill = "Implementation",
      title = paste0(bench_name, ": normalised actual-region total")
    ) +
    theme_pub()
  save_plot(actual_region_norm_plot, file.path(bench_out, "actual_region_total_normalised.pdf"))
  all_regions_plot <- ggplot(
    bench_df,
    aes(x = implementation, y = time_us, fill = implementation)
  ) +
    geom_boxplot(outlier.shape = NA) +
    jitter_points(
      bench_df,
      c("device", "benchmark", "implementation", "problem_size", "region_class"),
      alpha = 0.20,
      size = 0.45
    ) +
    facet_grid(region_class ~ device + problem_size, scales = "free_y") +
    scale_impl_fill() +
    labs(
      x = NULL,
      y = "Region time (us)",
      fill = "Implementation",
      title = paste0(bench_name, ": measured regions")
    ) +
    theme_pub()
  save_plot(all_regions_plot, file.path(bench_out, "all_regions_boxplot.pdf"), width = 11, height = 8)
  kernel_df <- bench_df |>
    filter(region_class == "Kernel")
  if (nrow(kernel_df) > 0) {
    kernel_plot <- ggplot(
      kernel_df,
      aes(x = implementation, y = time_us, fill = implementation)
    ) +
      geom_boxplot(outlier.shape = NA) +
      jitter_points(
        kernel_df,
        c("device", "benchmark", "implementation", "problem_size"),
        alpha = 0.30,
        size = 0.9
      ) +
      facet_grid(problem_size ~ device, scales = "free_x") +
      scale_impl_fill() +
      labs(
        x = NULL,
        y = "Kernel region time (us)",
        fill = "Implementation",
        title = paste0(bench_name, ": kernel-region timing")
      ) +
      theme_pub()
    save_plot(kernel_plot, file.path(bench_out, "kernel_regions_boxplot.pdf"))
    kernel_norm <- kernel_df |>
      group_by(device, benchmark, implementation, problem_size, run) |>
      summarise(time_us = sum(time_us), .groups = "drop") |>
      group_by(device, benchmark, implementation, problem_size) |>
      summarise(time_us = median(time_us), .groups = "drop") |>
      normalise_by_fastest(time_us, c("device", "benchmark", "problem_size"))
    kernel_norm_plot <- ggplot(
      kernel_norm,
      aes(x = implementation, y = normalised, fill = implementation)
    ) +
      geom_col() +
      facet_grid(problem_size ~ device, scales = "free_x") +
      scale_impl_fill() +
      labs(
        x = NULL,
        y = "Normalised kernel time (fastest = 1.0)",
        fill = "Implementation",
        title = paste0(bench_name, ": normalised kernel-region time")
      ) +
      theme_pub()
    save_plot(kernel_norm_plot, file.path(bench_out, "kernel_regions_normalised.pdf"))

    # --- Kernel time vs. problem size (line plot) ---
    # Regions that hit the repeat-to-~2s stabilisation loop (SRAD/CFD-style;
    # see EOD repo TODOs item 2 -- not all benchmarks use this) record
    # time_us as the SUM across every internal repeat, not one iteration's
    # cost. Divide by the recorded repeat count first to recover the
    # honest average per-iteration kernel time before comparing across
    # problem sizes -- otherwise benchmarks/sizes that needed more internal
    # repeats to fill 2s would look artificially slower than ones that
    # didn't need to repeat at all. lsb_common.R already gives us exactly
    # this per (file=run, region) -- time_us (renamed from total_time_us)
    # and n_repeats -- so this no longer needs its own group_by(...,
    # region) + sum() stage first the way it did with the old row-level
    # parser; kernel_df is already at that granularity.
    kernel_scaling_df <- kernel_df |>
      mutate(avg_time_us = time_us / pmax(n_repeats, 1)) |>
      group_by(device, benchmark, implementation, problem_size, run) |>
      summarise(avg_time_us = sum(avg_time_us), .groups = "drop") |>
      group_by(device, benchmark, implementation, problem_size) |>
      summarise(avg_time_us = median(avg_time_us), .groups = "drop") |>
      filter(!is.na(problem_size))
    if (nrow(kernel_scaling_df) > 0) {
      kernel_scaling_plot <- ggplot(
        kernel_scaling_df,
        aes(x = problem_size, y = avg_time_us, colour = implementation, group = implementation)
      ) +
        geom_line() +
        geom_point(size = 2) +
        facet_wrap(~device, scales = "free_y") +
        scale_colour_manual(values = backend_colours, aesthetics = "colour", na.value = "grey70") +
        labs(
          x = "Problem size",
          y = "Average kernel time per iteration (µs)",
          colour = "Implementation",
          title = paste0(bench_name, ": kernel time vs. problem size")
        ) +
        theme_pub()
      save_plot(kernel_scaling_plot, file.path(bench_out, "kernel_time_vs_problem_size.pdf"))
    }
  }
  invisible(NULL)
}
log_msg(
  "mode=full-bounded requested_mode=%s app=%s backend=%s jobs=%s point_sample=%d show_points=%s",
  plot_mode,
  ifelse(is.null(filter_app), "all", filter_app),
  ifelse(is.null(filter_backend), "all", filter_backend),
  Sys.getenv("PLOT_LSB_JOBS", "32"),
  POINT_SAMPLE_PER_GROUP,
  ifelse(SHOW_POINTS, "yes", "no")
)
rebuild_cache <- Sys.getenv("PLOT_LSB_REBUILD_CACHE", "0") == "1"
use_cache <- Sys.getenv("PLOT_LSB_USE_CACHE", "1") != "0"
# read_lsb_cached() (lsb_common.R) now owns both parsing and caching. Its
# cache lives under results_dir/.lsb_cache/ (manifest-hash keyed: any
# file added/removed/changed invalidates it, rather than this script's
# old single-mtime-comparison lsb_cache.rds) -- a different location from
# the old lsb_cache.rds/lsb_cache.csv this script used to write directly
# into results_dir/. Those old files are no longer read or written here;
# safe to delete, and harmless to leave in place.
df <- read_lsb_cached(results_dir, force = rebuild_cache || !use_cache)
write_csv(df, file.path(results_dir, "lsb_cache.csv"))
# Column renames to match this script's existing naming below, rather
# than rewriting every group_by/facet_grid/aes() call in this file to
# lsb_common.R's own column names:
#   size          -> problem_size (lsb_common.R's SIZE_LEVELS has no
#                    "default" level -- a size token outside
#                    tiny/small/medium/large already became NA at parse
#                    time there, with a warning logged; NA is still
#                    treated as its own facet level below, same as
#                    before)
#   total_time_us -> time_us (already summed across every internal
#                    stabilisation-loop repeat within one file+region --
#                    see the note above make_benchmark_plots())
#   run_variant   -> run (the actual per-invocation repeat index -- see
#                    the note above make_benchmark_plots() for why this
#                    is run_variant and not lsb_common.R's own `run`)
# device needs no renaming: lsb_common.R already derives it straight from
# the filename's device token, separately from `system` (the hostname).
# The fallback to `system` below only matters if a file's name ever lacks
# a device token at all (lsb_common.R warns but doesn't error on that).
df <- df |>
  rename(problem_size = size, time_us = total_time_us) |>
  mutate(
    problem_size = as.character(problem_size),
    device = ifelse(!is.na(device) & device != "", device, system),
    run = run_variant
  )
if (!is.null(filter_app) && filter_app != "all") {
  wanted_apps <- str_split(filter_app, ",")[[1]]
  df <- df |> filter(benchmark %in% wanted_apps)
}
if (!is.null(filter_backend) && filter_backend != "all") {
  wanted_backends <- str_split(filter_backend, ",")[[1]]
  df <- df |> filter(backend %in% wanted_backends)
}
if (nrow(df) == 0) {
  stop("No readable LSB rows matched selected filters")
}
df <- df |>
  mutate(
    implementation = factor(
      implementation,
      levels = c(
        implementation_levels,
        setdiff(sort(unique(as.character(implementation))),
                implementation_levels)
      )
    ),
    region_class = factor(region_class, levels = region_class_levels)
  )
log_msg(
  "parsed %s rows across %d benchmark(s), %d implementation(s), %d device(s)",
  comma(nrow(df)),
  n_distinct(df$benchmark),
  n_distinct(df$implementation),
  n_distinct(df$device)
)
plot_benchmarks_with_progress <- function(df, out_dir) {
  benches <- sort(unique(df$benchmark))
  n_benches <- length(benches)
  if (n_benches == 0L) {
    return(0L)
  }
  requested <- as.integer(Sys.getenv("PLOT_LSB_PLOT_JOBS", Sys.getenv("PLOT_LSB_JOBS", "32")))
  if (is.na(requested) || requested <= 0L) {
    requested <- 32L
  }
  cores <- parallel::detectCores(logical = FALSE)
  if (is.na(cores) || cores <= 0L) {
    cores <- 1L
  }
  n_workers <- max(1L, min(requested, cores, n_benches))
  log_msg(
    "generating plots for %d benchmark(s) with %d worker(s)",
    n_benches,
    n_workers
  )
  plot_one <- function(bench) {
    bench_df <- df |>
      filter(benchmark == bench)
    make_benchmark_plots(
      bench_df,
      file.path(out_dir, bench)
    )
    bench
  }
  pb <- txtProgressBar(min = 0, max = n_benches, style = 3)
  done <- 0L
  batches <- split(benches, ceiling(seq_along(benches) / n_workers))
  for (batch in batches) {
    batch_result <- if (n_workers <= 1L || length(batch) <= 1L) {
      lapply(batch, plot_one)
    } else {
      parallel::mclapply(
        batch,
        plot_one,
        mc.cores = min(n_workers, length(batch)),
        mc.preschedule = FALSE
      )
    }
    for (result in batch_result) {
      if (inherits(result, "try-error")) {
        close(pb)
        stop(result)
      }
      done <- done + 1L
      setTxtProgressBar(pb, done)
    }
  }
  close(pb)
  n_benches
}
n_plots <- plot_benchmarks_with_progress(df, out_dir)
log_msg(
  "generated full bounded plot set for %d benchmark(s) in %.2fs",
  n_plots,
  elapsed_s()
)
message("Wrote plots and CSV files to: ", out_dir)
