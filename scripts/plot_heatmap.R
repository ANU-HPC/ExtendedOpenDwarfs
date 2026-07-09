#!/usr/bin/env Rscript
#
# plot_heatmap.R
#
# Heatmap of SCALE vs native-toolchain performance across every
# (benchmark x size x device) combination present in the results directory.
#
# For each device, we compare against exactly one architecture-appropriate
# native toolchain: cuda/scale-nvidia is paired with cuda/nvcc, and
# cuda/scale-amd is paired with hip/hipcc. This is deliberately NOT "whatever
# native toolchain happens to be present on this device" -- hip/hipcc can
# also target NVIDIA hardware (HIP-on-CUDA), so a device can have hip/hipcc
# results without those being the SCALE-nvidia baseline. See the pairing
# logic below for details.
#
# Cell value: median(SCALE metric) / median(native metric), per
# (benchmark, size, device), using each config's median across repeated
# runs.
#   ratio < 1  -> SCALE faster than native
#   ratio = 1  -> parity
#   ratio > 1  -> SCALE slower than native
#
# Metric selection:
#   --metric=total     sum of every measured region's time, normalized by
#                       repeat count (default, run alongside kernel)
#   --metric=kernel     same, but restricted to region_class == "Kernel"
#                        (default, run alongside total)
#   --metric=runtime    raw end-to-end wall time from the "# Runtime:"
#                        header. NOTE: for benchmarks using a
#                        stabilize-until-~2s repeat loop, this measures
#                        "time to hit the stabilization target" rather than
#                        genuine relative performance -- both native and
#                        SCALE converge to nearly the same wall time by
#                        construction, which flattens real differences
#                        toward 1.0x. Use total/kernel instead unless you
#                        have a specific reason to want raw wall time.
#   --metric=auto        legacy behavior: pick runtime if available for
#                        >=50% of configs, else fall back to total. Not the
#                        default -- kept only for explicit opt-in, since
#                        auto-selecting silently makes it easy to lose
#                        track of which metric produced a given figure.
#
#   With NO --metric flag at all, both "kernel" and "total" are run in a
#   single invocation (sharing one parsed dataset), each writing to its own
#   subdirectory. Pass --metric=X explicitly for just one.
#
# Every metric's outputs are nested under out_dir/<metric>/ so multiple
# metrics (or multiple runs with different --metric values) never
# overwrite each other's files.
#
# Usage:
#   Rscript scripts/plot_heatmap.R [results_dir] [out_dir] [--force-reparse] [--metric=total|kernel|runtime|auto]
#
# Env vars (shared with plot_lsb.R where applicable):
#   PLOT_LSB_JOBS       parser worker count (default 32, capped to cores)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(stringr)
  library(tidyr)
  library(scales)
})

SCRIPT_START <- Sys.time()

# Resolve this script's own directory so lsb_common.R can be sourced
# regardless of the working directory Rscript was invoked from.
.script_dir <- tryCatch({
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    dirname(normalizePath(sub("^--file=", "", file_arg[1])))
  } else {
    getwd()
  }
}, error = function(e) getwd())

source(file.path(.script_dir, "lsb_common.R"))

args <- commandArgs(trailingOnly = TRUE)

force_reparse <- any(args == "--force-reparse")

VALID_METRICS <- c("auto", "runtime", "total", "kernel")
DEFAULT_METRICS <- c("kernel", "total")

metric_flag <- str_match(args, "^--metric=(.+)$")[, 2]
metric_flag <- metric_flag[!is.na(metric_flag)]

if (length(metric_flag) > 0) {
  requested_metrics <- str_split(metric_flag[1], ",")[[1]]
} else {
  requested_metrics <- DEFAULT_METRICS
}

bad_metrics <- setdiff(requested_metrics, VALID_METRICS)
if (length(bad_metrics) > 0) {
  stop(
    "Unknown --metric value(s): ", paste(bad_metrics, collapse = ", "),
    " (expected one or more of: ", paste(VALID_METRICS, collapse = ", "), ")"
  )
}

positional <- args[!str_detect(args, "^--(force-reparse|metric=)")]

results_dir <- ifelse(length(positional) >= 1, positional[[1]], "results")
base_out_dir <- ifelse(length(positional) >= 2, positional[[2]], file.path(results_dir, "plots"))

dir.create(base_out_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Load (cached) -- shared across every metric requested, parsed only once.
# ---------------------------------------------------------------------------

df <- read_lsb_cached(results_dir, force = force_reparse)

log_msg(
  "loaded %s rows: %d benchmark(s), %d implementation(s), %d device(s), %d size(s)",
  comma(nrow(df)),
  n_distinct(df$benchmark),
  n_distinct(df$implementation),
  n_distinct(df$device),
  n_distinct(df$size)
)

runtime_coverage_df <- df |>
  distinct(benchmark, size, device, implementation, run, runtime_s)
frac_with_runtime <- mean(!is.na(runtime_coverage_df$runtime_s))

# ---------------------------------------------------------------------------
# Everything from metric resolution through writing files for ONE metric,
# wrapped so it can run once per requested metric against the same df.
# ---------------------------------------------------------------------------

run_for_metric <- function(metric, df, base_out_dir, frac_with_runtime) {
  if (metric == "auto") {
    if (frac_with_runtime >= 0.5) {
      metric <- "runtime"
    } else {
      metric <- "total"
      log_msg(
        "auto metric: only %.0f%% of configs have a '# Runtime:' header -- falling back to metric=total (sum of measured regions). Pass --metric=runtime or --metric=kernel to override.",
        100 * frac_with_runtime
      )
    }
  } else {
    log_msg("metric=%s (%.0f%% of configs have a runtime_s header)", metric, 100 * frac_with_runtime)
  }

  metric_label <- switch(metric,
    runtime = "end-to-end runtime",
    total = "total measured region time",
    kernel = "kernel-region time"
  )

  # Nest outputs under a metric-specific subdirectory (e.g. results/plots/total/,
  # results/plots/kernel/) so running multiple metrics never overwrites a
  # previous metric's output files.
  out_dir <- file.path(base_out_dir, metric)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  log_msg("--- metric=%s: writing outputs to %s ---", metric, out_dir)

  if (metric == "runtime") {
    runtime_df <- df |>
      distinct(system, benchmark, size, device, implementation, run, runtime_s) |>
      filter(!is.na(runtime_s))
  } else {
    # df is pre-aggregated per (file, region) rather than per raw row --
    # total_time_us and n_repeats already reflect the sum/count across every
    # repeat-loop pass for that specific region in that specific file (see
    # parse_lsb_table_dt() in lsb_common.R). Each region is normalized to
    # "time per single pass" INDIVIDUALLY before summing, rather than summing
    # raw totals across regions first and dividing by one shared repeat count
    # -- a one-time setup region and a kernel region measured hundreds of
    # times within the same file do not share a repeat count, so normalizing
    # after combining them would have been wrong.
    region_filter <- df
    if (metric == "kernel") {
      region_filter <- df |> filter(region_class == "Kernel")
    }

    runtime_df <- region_filter |>
      mutate(time_us_per_repeat = total_time_us / n_repeats) |>
      group_by(system, benchmark, size, device, implementation, run) |>
      summarise(runtime_s = sum(time_us_per_repeat) / 1e6, .groups = "drop")
  }

  if (nrow(runtime_df) == 0) {
    log_msg(
      "SKIPPING metric=%s: no data found. If this is 'runtime', check that your LSB files contain a '# Runtime:' header line.",
      metric
    )
    return(invisible(NULL))
  }

  median_runtime <- runtime_df |>
    group_by(benchmark, size, device, implementation) |>
    summarise(
      runtime_s = median(runtime_s),
      n_runs = n(),
      .groups = "drop"
    )

  # -------------------------------------------------------------------------
  # Pair each SCALE implementation with its architecture-appropriate native
  # toolchain, and only that one.
  #
  # We deliberately do NOT use a single global "is this native or scale"
  # classification keyed on implementation name alone. hip/hipcc is not
  # reliably "the AMD native toolchain" -- HIP can also target NVIDIA
  # hardware (hipcc compiling against the CUDA backend), so hip/hipcc
  # legitimately shows up on NVIDIA devices too (e.g.
  # tdm_hip_hipcc_large_rtx3090), as a portability reference rather than the
  # SCALE-nvidia baseline. Pairing by architecture keeps that HIP-on-NVIDIA
  # row out of the AMD comparison (it has no cuda/scale-amd counterpart to
  # pair with, so it's correctly reported as an unmatched/missing pair
  # below, not silently merged in).
  # -------------------------------------------------------------------------

  ARCH_PAIRS <- list(
    nvidia = list(native = "cuda/nvcc", scale = "cuda/scale-nvidia"),
    amd = list(native = "hip/hipcc", scale = "cuda/scale-amd")
  )

  pair_architecture <- function(data, native_impl, scale_impl, architecture) {
    data |>
      filter(implementation %in% c(native_impl, scale_impl)) |>
      mutate(role = ifelse(implementation == native_impl, "native", "scale")) |>
      select(benchmark, size, device, role, implementation, runtime_s, n_runs) |>
      pivot_wider(
        names_from = role,
        values_from = c(implementation, runtime_s, n_runs),
        names_glue = "{role}_{.value}"
      ) |>
      mutate(architecture = architecture)
  }

  wide_df <- bind_rows(
    pair_architecture(median_runtime, ARCH_PAIRS$nvidia$native, ARCH_PAIRS$nvidia$scale, "nvidia"),
    pair_architecture(median_runtime, ARCH_PAIRS$amd$native, ARCH_PAIRS$amd$scale, "amd")
  )

  known_impls <- unlist(ARCH_PAIRS, use.names = FALSE)
  dropped_impls <- setdiff(unique(median_runtime$implementation), known_impls)
  if (length(dropped_impls) > 0) {
    log_msg(
      "excluding implementation(s) not part of a native/SCALE architecture pair: %s",
      paste(dropped_impls, collapse = ", ")
    )
  }

  missing_pairs <- wide_df |>
    filter(is.na(native_runtime_s) | is.na(scale_runtime_s))

  if (nrow(missing_pairs) > 0) {
    log_msg(
      "%d (architecture, benchmark, size, device) combination(s) have only one of {native, scale} -- skipped:",
      nrow(missing_pairs)
    )
    for (i in seq_len(min(nrow(missing_pairs), 20))) {
      row <- missing_pairs[i, ]
      log_msg(
        "  missing %s: %s / %s / %s / %s",
        ifelse(is.na(row$native_runtime_s), "native", "scale"),
        row$architecture, row$benchmark, row$size, row$device
      )
    }
    if (nrow(missing_pairs) > 20) {
      log_msg("  ... and %d more", nrow(missing_pairs) - 20)
    }
  }

  heatmap_df <- wide_df |>
    filter(!is.na(native_runtime_s), !is.na(scale_runtime_s)) |>
    mutate(
      ratio = scale_runtime_s / native_runtime_s,
      log2_ratio = log2(ratio),
      size = factor(size, levels = SIZE_LEVELS)
    )

  if (nrow(heatmap_df) == 0) {
    log_msg(
      "SKIPPING metric=%s: no (benchmark, size, device) combination has both a native and a SCALE result. See missing-pairs log above.",
      metric
    )
    return(invisible(NULL))
  }

  write_csv(heatmap_df, file.path(out_dir, paste0("scale_vs_native_ratio_", metric, ".csv")))
  log_msg(
    "wrote ratio table: %s (%d rows)",
    file.path(out_dir, paste0("scale_vs_native_ratio_", metric, ".csv")),
    nrow(heatmap_df)
  )

  # -------------------------------------------------------------------------
  # Heatmap: rows = benchmark, columns = device, one panel per size.
  # Colour = log2(ratio), diverging around 0 (parity). Blue = SCALE faster,
  # red = SCALE slower. Cell label = ratio as "0.9x" / "1.2x" for readability
  # without requiring the reader to mentally exponentiate log2 values.
  # -------------------------------------------------------------------------

  max_abs_log2 <- max(abs(heatmap_df$log2_ratio), na.rm = TRUE)
  # Symmetric limits so 1.0x always sits at the exact midpoint of the scale,
  # regardless of whether the data skews toward SCALE-faster or SCALE-slower.
  colour_limit <- max(max_abs_log2, 0.1)

  dup_check <- heatmap_df |>
    count(benchmark, size, device) |>
    filter(n > 1)

  if (nrow(dup_check) > 0) {
    log_msg(
      "WARNING: %d (benchmark, size, device) combination(s) have more than one row -- ",
      nrow(dup_check)
    )
    log_msg(
      "this means both an nvidia-architecture pair AND an amd-architecture pair have complete data under the same device tag, which should be physically impossible (an AMD-ISA binary can't run on an NVIDIA GPU or vice versa). This usually indicates a device-tagging bug upstream in the benchmark harness, not in this script. Faceting by architecture below so these don't silently overlap in the plot."
    )
    print(dup_check, n = 20)
  }

  heatmap_plot_fn <- function(data, colour_limit) {
    ggplot(
      data,
      aes(x = device, y = benchmark, fill = log2_ratio)
    ) +
      geom_tile(colour = "white", linewidth = 0.4) +
      geom_text(
        aes(label = paste0(number(ratio, accuracy = 0.01), "x")),
        size = 3,
        colour = "black"
      ) +
      facet_wrap(~size, nrow = 1, labeller = labeller(size = str_to_title)) +
      scale_fill_gradient2(
        low = "#1F77B4",
        mid = "white",
        high = "#D62728",
        midpoint = 0,
        limits = c(-colour_limit, colour_limit),
        breaks = c(-colour_limit, 0, colour_limit),
        labels = c("SCALE faster", "parity", "SCALE slower"),
        name = NULL
      ) +
      theme_bw(base_size = 13) +
      theme(
        axis.text.x = element_text(angle = 40, hjust = 1),
        panel.grid = element_blank(),
        legend.position = "bottom",
        legend.key.width = unit(2.2, "cm")
      )
  }

  # Two independent plots rather than one facet_grid(architecture ~ size) --
  # a shared x-axis across both architectures meant every AMD device showed
  # up as an empty column under the NVIDIA rows and vice versa, which reads
  # as "this device belongs to both architectures" even though it's just an
  # artifact of facet_grid forcing a common axis. Splitting into separate
  # figures means each one's device axis only ever shows devices that
  # actually have data for that architecture.
  architecture_labels <- list(nvidia = "NVIDIA", amd = "AMD")

  for (arch in names(architecture_labels)) {
    arch_df <- heatmap_df |> filter(architecture == arch)

    if (nrow(arch_df) == 0) {
      log_msg("skipping %s heatmap: no complete native/SCALE pairs for this architecture", arch)
      next
    }

    arch_label <- architecture_labels[[arch]]
    n_devices <- n_distinct(arch_df$device)
    n_benchmarks <- n_distinct(arch_df$benchmark)

    p <- heatmap_plot_fn(arch_df, colour_limit) +
      labs(
        x = NULL,
        y = NULL,
        title = paste0("SCALE vs native toolchain (", arch_label, "): ", metric_label, " ratio"),
        subtitle = paste0(
          "Cell = median(SCALE ", metric_label, ") / median(native ", metric_label,
          ") per benchmark, size, device"
        )
      )

    out_path <- file.path(out_dir, paste0("scale_vs_native_heatmap_", arch, "_", metric, ".pdf"))

    ggsave(
      out_path,
      p,
      width = max(6, 2.4 * n_devices + 2),
      height = max(3, 0.55 * n_benchmarks + 2),
      limitsize = FALSE
    )

    log_msg("wrote %s heatmap: %s (%d benchmark(s), %d device(s))", arch_label, out_path, n_benchmarks, n_devices)
  }

  invisible(out_dir)
}

for (m in requested_metrics) {
  run_for_metric(m, df, base_out_dir, frac_with_runtime)
}

message("Done. Wrote outputs for metric(s) [", paste(requested_metrics, collapse = ", "), "] under: ", base_out_dir)
