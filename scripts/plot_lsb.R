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
})

args <- commandArgs(trailingOnly = TRUE)

results_dir <- ifelse(length(args) >= 1, args[[1]], "results")
out_dir <- ifelse(length(args) >= 2, args[[2]], file.path(results_dir, "plots"))

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
  "Other"
)

classify_region <- function(region) {
  case_when(
    str_detect(region, regex("^runtime_initialization$", ignore_case = TRUE)) ~ "Runtime init",
    str_detect(region, regex("kernel_creation|program_creation|build", ignore_case = TRUE)) ~ "Kernel creation",
    str_detect(region, regex("kernel|compute|solve|execute", ignore_case = TRUE)) ~ "Kernel",
    str_detect(region, regex("h2d|d2h|copy|transfer|memcpy|read|write", ignore_case = TRUE)) ~ "Transfer",
    str_detect(region, regex("setup|argument|arg|alloc|buffer|init", ignore_case = TRUE)) ~ "Buffer/setup",
    TRUE ~ "Other"
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

extract_nodename <- function(lines) {
  nodename <- lines |>
    str_subset("^# Nodename:") |>
    first()

  if (is.na(nodename)) {
    return(NA_character_)
  }

  nodename |>
    str_replace("^# Nodename:\\s*", "") |>
    str_replace("\\..*$", "")
}

parse_lsb_table <- function(lines, header_idx) {
  header <- str_split(str_squish(lines[[header_idx]]), "\\s+")[[1]]

  if (!all(c("region", "id", "time", "overhead") %in% header)) {
    return(NULL)
  }

  region_pos <- match("region", header)
  param_names <- header[seq_len(region_pos - 1)]

  data_lines <- lines[(header_idx + 1):length(lines)]
  data_lines <- data_lines[str_detect(data_lines, "^\\s*[-0-9]+\\s+")]

  if (length(data_lines) == 0) {
    return(NULL)
  }

  tokens <- str_split(str_squish(data_lines), "\\s+")

  rows <- map_dfr(tokens, function(x) {
    n <- length(x)

    if (n < 4) {
      return(tibble())
    }

    param_values <- character(0)

    if (length(param_names) > 0) {
      param_values <- x[seq_len(min(length(param_names), n - 4))]
      names(param_values) <- param_names[seq_along(param_values)]
    }

    tibble(
      !!!as.list(param_values),
      region = x[n - 3],
      id = suppressWarnings(as.integer(x[n - 2])),
      time_us = suppressWarnings(as.numeric(x[n - 1])),
      overhead = suppressWarnings(as.numeric(x[n]))
    )
  })

  rows |>
    mutate(across(
      -c(region),
      ~ suppressWarnings(type.convert(.x, as.is = TRUE))
    ))
}

read_lsb_file <- function(path) {
  lines <- readLines(path, warn = FALSE)
  base <- basename(path)

  meta <- str_match(
    base,
    "^lsb\\.([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9_]+)\\.r[0-9]+(?:-([0-9]+))?$"
  )

  if (is.na(meta[1, 1])) {
    warning("Skipping unrecognised LSB filename: ", base)
    return(NULL)
  }

  header_idx <- which(str_detect(lines, "^\\s*\\S+\\s+.*\\s+region\\s+.*\\stime\\s+"))[1]

  if (is.na(header_idx)) {
    warning("No timing table found in: ", base)
    return(NULL)
  }

  rows <- parse_lsb_table(lines, header_idx)

  if (is.null(rows) || nrow(rows) == 0) {
    warning("No timing rows found in: ", base)
    return(NULL)
  }

  rows |>
    mutate(
      file = base,
      system = extract_nodename(lines),
      benchmark = meta[1, 2],
      backend = meta[1, 3],
      compiler = str_replace_all(meta[1, 4], "_", "-"),
      implementation = paste(backend, compiler, sep = "/"),
      run = ifelse(is.na(meta[1, 5]), 0L, as.integer(meta[1, 5])),
      runtime_s = extract_runtime(lines),
      region_class = classify_region(region)
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
      geom_jitter(width = 0.12, alpha = 0.45, size = 1.4) +
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
    group_by(system, benchmark) |>
    mutate(region_total = sum(time_us)) |>
    ungroup() |>
    arrange(system, benchmark, desc(time_us))

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
    geom_jitter(width = 0.12, alpha = 0.35, size = 1.1) +
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
      geom_jitter(width = 0.12, alpha = 0.45, size = 1.4) +
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

  write_csv(bench_df, file.path(bench_out, "lsb_long.csv"))
  write_csv(region_class_breakdown, file.path(bench_out, "region_class_summary.csv"))
  write_csv(actual_region_breakdown, file.path(bench_out, "actual_region_summary.csv"))
}

files <- list.files(results_dir, pattern = "^lsb\\.", full.names = TRUE)
df <- map_dfr(files, read_lsb_file)

if (nrow(df) == 0) {
  stop("No readable LSB files found in ", results_dir)
}

df <- df |>
  mutate(
    implementation = factor(implementation, levels = sort(unique(implementation))),
    region_class = factor(region_class, levels = region_class_levels)
  )

write_csv(df, file.path(out_dir, "lsb_long.csv"))

for (bench in sort(unique(df$benchmark))) {
  bench_df <- df |>
    filter(benchmark == bench)

  make_benchmark_plots(
    bench_df,
    file.path(out_dir, bench)
  )
}

message("Wrote plots and CSV files to: ", out_dir)
