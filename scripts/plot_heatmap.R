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
# runs. Metric defaults to end-to-end runtime (the "# Runtime:" header) but
# falls back automatically to summed region time if that header is largely
# absent -- see --metric below.
#   ratio < 1  -> SCALE faster than native
#   ratio = 1  -> parity
#   ratio > 1  -> SCALE slower than native
#
# Usage:
#   Rscript scripts/plot_heatmap.R [results_dir] [out_dir] [--force-reparse] [--metric=auto|runtime|total|kernel]
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

metric_flag <- str_match(args, "^--metric=(.+)$")[, 2]
metric_flag <- metric_flag[!is.na(metric_flag)]
metric <- ifelse(length(metric_flag) > 0, metric_flag[1], "auto")

positional <- args[!str_detect(args, "^--(force-reparse|metric=)")]

if (!(metric %in% c("auto", "runtime", "total", "kernel"))) {
  stop("Unknown --metric value '", metric, "' (expected: auto, runtime, total, kernel)")
}

results_dir <- ifelse(length(positional) >= 1, positional[[1]], "results")
out_dir <- ifelse(length(positional) >= 2, positional[[2]], file.path(results_dir, "plots"))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Load (cached)
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

# ---------------------------------------------------------------------------
# Metric selection.
#
# Preferred metric is end-to-end wall time from the "# Runtime:" header
# (runtime_s). Some LSB files -- particularly kernel-focused benchmarks --
# may not carry that header at all, in which case we fall back to the sum
# of measured region time (either every region, or just the Kernel-class
# regions) as a stand-in for "total time this configuration took".
#
# --metric=runtime  force header-based end-to-end time (errors if missing)
# --metric=total    sum of every measured region's time_us
# --metric=kernel   sum of only region_class == "Kernel" time_us
# --metric=auto     (default) use runtime if present for >=50% of configs,
#                    otherwise fall back to total with a warning
# ---------------------------------------------------------------------------

runtime_coverage_df <- df |>
  distinct(benchmark, size, device, implementation, run, runtime_s)
frac_with_runtime <- mean(!is.na(runtime_coverage_df$runtime_s))

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
  log_msg("metric=%s (explicit, %.0f%% of configs have a runtime_s header)", metric, 100 * frac_with_runtime)
}

metric_label <- switch(metric,
  runtime = "end-to-end runtime",
  total = "total measured region time",
  kernel = "kernel-region time"
)

if (metric == "runtime") {
  runtime_df <- df |>
    distinct(system, benchmark, size, device, implementation, run, runtime_s) |>
    filter(!is.na(runtime_s))
} else {
  # IMPORTANT: time_us for a given region is repeated once per pass of the
  # stabilizing loop (the "repeats_to_two_seconds" column, read into
  # repeat_id), which keeps re-running the same measured work until ~2s of
  # wall time has elapsed. A faster implementation therefore completes more
  # passes in that fixed window than a slower one. Summing time_us across
  # all passes without correcting for this just recovers ~2 seconds
  # regardless of which implementation is actually faster -- it measures
  # the stabilization budget, not performance. Dividing by the number of
  # distinct repeat_id values converts the sum back into "time per single
  # pass", which is what's actually comparable across implementations.
  region_filter <- df
  if (metric == "kernel") {
    region_filter <- df |> filter(region_class == "Kernel")
  }

  runtime_df <- region_filter |>
    group_by(system, benchmark, size, device, implementation, run) |>
    summarise(
      summed_time_us = sum(time_us),
      n_repeats = n_distinct(repeat_id),
      .groups = "drop"
    ) |>
    mutate(runtime_s = (summed_time_us / n_repeats) / 1e6)
}

if (nrow(runtime_df) == 0) {
  stop(
    "No data found for metric='", metric, "'. If using --metric=runtime, ",
    "check that your LSB files actually contain a '# Runtime:' header line; ",
    "otherwise try --metric=total or --metric=kernel."
  )
}

median_runtime <- runtime_df |>
  group_by(benchmark, size, device, implementation) |>
  summarise(
    runtime_s = median(runtime_s),
    n_runs = n(),
    .groups = "drop"
  )

# ---------------------------------------------------------------------------
# Pair each SCALE implementation with its architecture-appropriate native
# toolchain, and only that one.
#
# We deliberately do NOT use a single global "is this native or scale"
# classification keyed on implementation name alone. hip/hipcc is not
# reliably "the AMD native toolchain" -- HIP can also target NVIDIA hardware
# (hipcc compiling against the CUDA backend), so hip/hipcc legitimately
# shows up on NVIDIA devices too (e.g. tdm_hip_hipcc_large_rtx3090), as a
# portability reference rather than the SCALE-nvidia baseline. Pairing by
# architecture keeps that HIP-on-NVIDIA row out of the AMD comparison (it
# has no cuda/scale-amd counterpart to pair with, so it's correctly
# reported as an unmatched/missing pair below, not silently merged in).
# ---------------------------------------------------------------------------

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
  stop(
    "No (benchmark, size, device) combination has both a native and a SCALE ",
    "result -- nothing to plot. Check the missing-pairs log above."
  )
}

write_csv(heatmap_df, file.path(out_dir, "scale_vs_native_ratio.csv"))
log_msg("wrote ratio table: %s (%d rows)", file.path(out_dir, "scale_vs_native_ratio.csv"), nrow(heatmap_df))

# ---------------------------------------------------------------------------
# Heatmap: rows = benchmark, columns = device, one panel per size.
# Colour = log2(ratio), diverging around 0 (parity). Blue = SCALE faster,
# red = SCALE slower. Cell label = ratio as "0.9x" / "1.2x" for readability
# without requiring the reader to mentally exponentiate log2 values.
# ---------------------------------------------------------------------------

max_abs_log2 <- max(abs(heatmap_df$log2_ratio), na.rm = TRUE)
# Symmetric limits so 1.0x always sits at the exact midpoint of the scale,
# regardless of whether the data skews toward SCALE-faster or SCALE-slower.
colour_limit <- max(max_abs_log2, 0.1)

heatmap_plot <- ggplot(
  heatmap_df,
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
  labs(
    x = NULL,
    y = NULL,
    title = paste0("SCALE vs native toolchain: ", metric_label, " ratio"),
    subtitle = paste0(
      "Cell = median(SCALE ", metric_label, ") / median(native ", metric_label,
      ") per benchmark, size, device"
    )
  ) +
  theme_bw(base_size = 13) +
  theme(
    axis.text.x = element_text(angle = 40, hjust = 1),
    panel.grid = element_blank(),
    legend.position = "bottom",
    legend.key.width = unit(2.2, "cm")
  )

heatmap_path <- file.path(out_dir, "scale_vs_native_heatmap.pdf")
n_devices <- n_distinct(heatmap_df$device)
n_benchmarks <- n_distinct(heatmap_df$benchmark)

ggsave(
  heatmap_path,
  heatmap_plot,
  width = max(8, 2.4 * n_devices + 2),
  height = max(4, 0.55 * n_benchmarks + 2),
  limitsize = FALSE
)

log_msg("wrote heatmap: %s", heatmap_path)
message("Wrote heatmap and ratio table to: ", out_dir)
