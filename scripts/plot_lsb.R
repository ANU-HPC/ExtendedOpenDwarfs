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

SCRIPT_START <- Sys.time()

elapsed_s <- function() {
  as.numeric(difftime(Sys.time(), SCRIPT_START, units = "secs"))
}

log_msg <- function(...) {
  cat(sprintf("[plot_lsb +%6.2fs] ", elapsed_s()), sprintf(...), "\n", sep = "")
  flush.console()
}

get_worker_count <- function(n_files) {
  requested <- as.integer(Sys.getenv("PLOT_LSB_JOBS", "32"))

  if (is.na(requested) || requested <= 0L) {
    requested <- 32L
  }

  cores <- parallel::detectCores(logical = FALSE)
  if (is.na(cores) || cores <= 0L) {
    cores <- 1L
  }

  max(1L, min(requested, cores, n_files))
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

backend_colours <- c(
  "cuda/nvcc" = "#76B900",
  "cuda/scale-nvidia" = "#3F7F00",
  "cuda/scale-amd" = "#B00020",
  "hip/hipcc" = "#ED1C24",
  "opencl/opencl" = "#1F77B4"
)

region_class_levels <- c(
  "Runtime init",
  "Kernel creation",
  "Buffer/setup",
  "Transfer",
  "Kernel",
  "Input load",
  "Other"
)

classify_region <- function(region) {
  case_when(
    str_detect(region, regex("^runtime_initialization$", ignore_case = TRUE)) ~ "Runtime init",
    str_detect(region, regex("^host_input_load$", ignore_case = TRUE)) ~ "Input load",
    str_detect(region, regex("kernel_creation|program_creation|build", ignore_case = TRUE)) ~ "Kernel creation",
    str_detect(region, regex("^kernel_execution$|kernel|compute|solve|execute", ignore_case = TRUE)) ~ "Kernel",
    str_detect(region, regex("h2d|d2h|copy|transfer|memcpy|read|write", ignore_case = TRUE)) ~ "Transfer",
    str_detect(region, regex("setup|argument|arg|alloc|buffer|init", ignore_case = TRUE)) ~ "Buffer/setup",
    TRUE ~ "Other"
  )
}

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

extract_runtime <- function(lines) {
  runtime_line <- lines |>
    str_subset("^# Runtime:") |>
    first()

  if (is.na(runtime_line)) {
    return(NA_real_)
  }

  as.numeric(str_match(runtime_line, "# Runtime:\\s*([0-9.]+)\\s*s")[1, 2])
}

extract_nodename_fast <- function(lines) {
  idx <- grep("^# Nodename:", lines, fixed = FALSE)[1]

  if (is.na(idx)) {
    return(NA_character_)
  }

  sub("\\..*$", "", sub("^# Nodename:\\s*", "", lines[[idx]]))
}

parse_lsb_filename <- function(path) {
  base <- basename(path)

  meta <- str_match(
    base,
    "^lsb\\.([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9_]+)\\.r[0-9]+(?:-([0-9]+))?$"
  )

  if (is.na(meta[1, 1])) {
    return(NULL)
  }

  list(
    file = base,
    benchmark = meta[1, 2],
    backend = meta[1, 3],
    compiler = str_replace_all(meta[1, 4], "_", "-"),
    run = ifelse(is.na(meta[1, 5]), 0L, as.integer(meta[1, 5]))
  )
}

parse_lsb_table_fast <- function(lines, header_idx, meta, system, runtime_s) {
  header <- str_split(str_squish(lines[[header_idx]]), "\\s+")[[1]]

  region_pos <- match("region", header)
  time_pos <- match("time", header)
  id_pos <- match("id", header)
  overhead_pos <- match("overhead", header)

  if (is.na(region_pos) || is.na(time_pos)) {
    return(NULL)
  }

  data_lines <- lines[(header_idx + 1):length(lines)]
  data_lines <- data_lines[
    !str_detect(data_lines, "^#") &
      !str_detect(data_lines, "^\\s*$")
  ]

  if (length(data_lines) == 0) {
    return(NULL)
  }

  fields <- str_split_fixed(str_squish(data_lines), "\\s+", n = length(header))

  time_us <- suppressWarnings(as.numeric(fields[, time_pos]))
  keep <- !is.na(time_us)

  if (!any(keep)) {
    return(NULL)
  }

  id <- if (!is.na(id_pos)) {
    suppressWarnings(as.integer(fields[keep, id_pos]))
  } else {
    NA_integer_
  }

  overhead <- if (!is.na(overhead_pos)) {
    suppressWarnings(as.numeric(fields[keep, overhead_pos]))
  } else {
    NA_real_
  }

  region <- fields[keep, region_pos]

  tibble(
    file = meta$file,
    system = system,
    benchmark = meta$benchmark,
    backend = meta$backend,
    compiler = meta$compiler,
    implementation = paste(meta$backend, meta$compiler, sep = "/"),
    run = meta$run,
    region = region,
    region_class = classify_region(region),
    id = id,
    time_us = time_us[keep],
    overhead = overhead,
    runtime_s = runtime_s
  )
}

read_lsb_file_fast <- function(path) {
  meta <- parse_lsb_filename(path)
  if (is.null(meta)) {
    return(NULL)
  }

  lines <- readLines(path, warn = FALSE)
  system <- extract_nodename_fast(lines)
  runtime_s <- extract_runtime(lines)

  header_idx <- grep("\\bregion\\b.*\\bid\\b.*\\btime\\b.*\\boverhead\\b", lines)[1]
  if (is.na(header_idx)) {
    header_idx <- which(str_detect(lines, "^\\s*\\S+\\s+.*\\s+region\\s+.*\\stime\\s+"))[1]
  }

  if (is.na(header_idx)) {
    return(NULL)
  }

  parse_lsb_table_fast(lines, header_idx, meta, system, runtime_s)
}

parse_files_with_progress <- function(files) {
  n_workers <- get_worker_count(length(files))

  parse_one <- function(i) {
    rows <- read_lsb_file_fast(files[[i]])

    list(
      rows = rows,
      rows_n = if (is.null(rows)) 0L else nrow(rows)
    )
  }

  pb <- txtProgressBar(min = 0, max = length(files), style = 3)
  parsed <- vector("list", length(files))
  done <- 0L

  batches <- split(seq_along(files), ceiling(seq_along(files) / n_workers))

  for (batch in batches) {
    batch_result <- if (n_workers <= 1L || length(batch) <= 1L) {
      lapply(batch, parse_one)
    } else {
      parallel::mclapply(
        batch,
        parse_one,
        mc.cores = min(n_workers, length(batch)),
        mc.preschedule = FALSE
      )
    }

    for (j in seq_along(batch)) {
      parsed[[batch[[j]]]] <- batch_result[[j]]
      done <- done + 1L
      setTxtProgressBar(pb, done)
    }
  }

  close(pb)

  bind_rows(lapply(parsed, function(x) x$rows))
}

make_benchmark_plots <- function(bench_df, bench_out) {
  dir.create(bench_out, recursive = TRUE, showWarnings = FALSE)

  bench_name <- unique(bench_df$benchmark)
  if (length(bench_name) != 1) {
    stop("make_benchmark_plots() received multiple benchmarks")
  }

  runtime_df <- bench_df |>
    distinct(system, benchmark, backend, compiler, implementation, run, runtime_s) |>
    filter(!is.na(runtime_s))

  if (nrow(runtime_df) > 0) {
    total_runtime_plot <- ggplot(
      runtime_df,
      aes(x = implementation, y = runtime_s * 1e3, fill = implementation)
    ) +
      geom_boxplot(outlier.shape = NA) +
      jitter_points(
        runtime_df,
        c("system", "benchmark", "implementation"),
        alpha = 0.35,
        size = 1.1
      ) +
      facet_wrap(~system, scales = "free_x") +
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
      group_by(system, benchmark, implementation, run) |>
      summarise(runtime_ms = median(runtime_ms), .groups = "drop") |>
      normalise_by_fastest(runtime_ms, c("system", "benchmark"))

    total_runtime_norm_plot <- ggplot(
      runtime_norm,
      aes(x = implementation, y = normalised, fill = implementation)
    ) +
      geom_col() +
      facet_wrap(~system, scales = "free_x") +
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
    group_by(system, benchmark, implementation, run, region_class) |>
    summarise(time_us = sum(time_us), .groups = "drop") |>
    group_by(system, benchmark, implementation, region_class) |>
    summarise(time_us = median(time_us), .groups = "drop") |>
    mutate(region_class = factor(region_class, levels = region_class_levels))

  region_class_plot <- ggplot(
    region_class_breakdown,
    aes(x = implementation, y = time_us / 1000, fill = region_class)
  ) +
    geom_col() +
    facet_wrap(~system, scales = "free_x") +
    labs(
      x = NULL,
      y = "Median measured region time (ms)",
      fill = "Region class",
      title = paste0(bench_name, ": runtime composition")
    ) +
    theme_pub()

  save_plot(region_class_plot, file.path(bench_out, "region_class_breakdown.pdf"))

  region_class_norm <- region_class_breakdown |>
    group_by(system, benchmark, implementation) |>
    summarise(time_us = sum(time_us), .groups = "drop") |>
    normalise_by_fastest(time_us, c("system", "benchmark"))

  region_class_norm_plot <- ggplot(
    region_class_norm,
    aes(x = implementation, y = normalised, fill = implementation)
  ) +
    geom_col() +
    facet_wrap(~system, scales = "free_x") +
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
    group_by(system, benchmark, implementation) |>
    mutate(percent = 100 * time_us / sum(time_us)) |>
    ungroup()

  region_percent_plot <- ggplot(
    region_percent,
    aes(x = implementation, y = percent, fill = region_class)
  ) +
    geom_col() +
    facet_wrap(~system, scales = "free_x") +
    labs(
      x = NULL,
      y = "Share of measured region time (%)",
      fill = "Region class",
      title = paste0(bench_name, ": runtime composition share")
    ) +
    theme_pub()

  save_plot(region_percent_plot, file.path(bench_out, "region_class_percent.pdf"))

  actual_region_breakdown <- bench_df |>
    group_by(system, benchmark, implementation, run, region) |>
    summarise(time_us = sum(time_us), .groups = "drop") |>
    group_by(system, benchmark, implementation, region) |>
    summarise(time_us = median(time_us), .groups = "drop") |>
    arrange(system, benchmark, implementation, desc(time_us))

  actual_region_plot <- ggplot(
    actual_region_breakdown,
    aes(x = implementation, y = time_us / 1000, fill = region)
  ) +
    geom_col() +
    facet_wrap(~system, scales = "free_x") +
    labs(
      x = NULL,
      y = "Median measured region time (ms)",
      fill = "Region",
      title = paste0(bench_name, ": actual region breakdown")
    ) +
    theme_pub()

  save_plot(actual_region_plot, file.path(bench_out, "actual_region_breakdown.pdf"), width = 11, height = 6)

  actual_region_norm <- actual_region_breakdown |>
    group_by(system, benchmark, implementation) |>
    summarise(time_us = sum(time_us), .groups = "drop") |>
    normalise_by_fastest(time_us, c("system", "benchmark"))

  actual_region_norm_plot <- ggplot(
    actual_region_norm,
    aes(x = implementation, y = normalised, fill = implementation)
  ) +
    geom_col() +
    facet_wrap(~system, scales = "free_x") +
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
      c("system", "benchmark", "implementation", "region_class"),
      alpha = 0.20,
      size = 0.45
    ) +
    facet_grid(region_class ~ system, scales = "free_y") +
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
        c("system", "benchmark", "implementation"),
        alpha = 0.30,
        size = 0.9
      ) +
      facet_wrap(~system, scales = "free_x") +
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
      group_by(system, benchmark, implementation, run) |>
      summarise(time_us = sum(time_us), .groups = "drop") |>
      group_by(system, benchmark, implementation) |>
      summarise(time_us = median(time_us), .groups = "drop") |>
      normalise_by_fastest(time_us, c("system", "benchmark"))

    kernel_norm_plot <- ggplot(
      kernel_norm,
      aes(x = implementation, y = normalised, fill = implementation)
    ) +
      geom_col() +
      facet_wrap(~system, scales = "free_x") +
      scale_impl_fill() +
      labs(
        x = NULL,
        y = "Normalised kernel time (fastest = 1.0)",
        fill = "Implementation",
        title = paste0(bench_name, ": normalised kernel-region time")
      ) +
      theme_pub()

    save_plot(kernel_norm_plot, file.path(bench_out, "kernel_regions_normalised.pdf"))
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

files <- list.files(results_dir, pattern = "^lsb\\.", full.names = TRUE)

if (!is.null(filter_app) && filter_app != "all") {
  wanted_apps <- str_split(filter_app, ",")[[1]]
  wanted_apps_regex <- paste(wanted_apps, collapse = "|")
  files <- files[str_detect(basename(files), paste0("^lsb\\.(", wanted_apps_regex, ")_"))]
}

if (!is.null(filter_backend) && filter_backend != "all") {
  wanted_backends <- str_split(filter_backend, ",")[[1]]
  wanted_backends_regex <- paste(wanted_backends, collapse = "|")
  files <- files[str_detect(basename(files), paste0("^lsb\\.[A-Za-z0-9]+_(", wanted_backends_regex, ")_"))]
}

if (length(files) == 0) {
  stop("No LSB files matched selected filters in ", results_dir)
}

sizes <- file.info(files)$size / (1024 * 1024)
log_msg(
  "parsing %d LSB files, %.2f MiB total",
  length(files),
  sum(sizes, na.rm = TRUE)
)

df <- parse_files_with_progress(files)

if (nrow(df) == 0) {
  stop("No readable LSB files found in ", results_dir)
}

if (!is.null(filter_app) && filter_app != "all") {
  wanted_apps <- str_split(filter_app, ",")[[1]]
  df <- df |> filter(benchmark %in% wanted_apps)
}

if (!is.null(filter_backend) && filter_backend != "all") {
  wanted_backends <- str_split(filter_backend, ",")[[1]]
  df <- df |> filter(backend %in% wanted_backends)
}

if (nrow(df) == 0) {
  stop("No readable LSB files matched selected filters")
}

df <- df |>
  mutate(
    implementation = factor(implementation, levels = sort(unique(implementation))),
    region_class = factor(region_class, levels = region_class_levels)
  )

log_msg(
  "parsed %s rows across %d benchmark(s), %d implementation(s), %d system(s)",
  comma(nrow(df)),
  n_distinct(df$benchmark),
  n_distinct(df$implementation),
  n_distinct(df$system)
)

for (bench in sort(unique(df$benchmark))) {
  bench_df <- df |>
    filter(benchmark == bench)

  make_benchmark_plots(
    bench_df,
    file.path(out_dir, bench)
  )
}

n_plots <- length(unique(df$benchmark))
log_msg(
  "generated full bounded plot set for %d benchmark(s) in %.2fs",
  n_plots,
  elapsed_s()
)

message("Wrote plots and CSV files to: ", out_dir)
