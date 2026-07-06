#!/usr/bin/env Rscript
#
# lsb_common.R
#
# Shared parsing, classification, and caching helpers for LibSciBench (LSB)
# result files. Sourced by both plot_lsb.R and plot_heatmap.R so the two
# tools never disagree about how a result file is parsed or classified.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
  library(purrr)
  library(tibble)
  library(parallel)
  library(digest)
})

SCRIPT_START <- if (exists("SCRIPT_START")) SCRIPT_START else Sys.time()

elapsed_s <- function() {
  as.numeric(difftime(Sys.time(), SCRIPT_START, units = "secs"))
}

log_msg <- function(...) {
  cat(sprintf("[+%6.2fs] ", elapsed_s()), sprintf(...), "\n", sep = "")
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

# ---------------------------------------------------------------------------
# Region classification (unchanged from plot_lsb.R)
# ---------------------------------------------------------------------------

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

backend_colours <- c(
  "cuda/nvcc" = "#76B900",
  "cuda/scale-nvidia" = "#3F7F00",
  "hip/hipcc" = "#ED1C24",
  "cuda/scale-amd" = "#B00020",
  "opencl/opencl" = "#1F77B4"
)

implementation_levels <- c(
  "cuda/nvcc",
  "cuda/scale-nvidia",
  "hip/hipcc",
  "cuda/scale-amd",
  "opencl/opencl"
)

# ---------------------------------------------------------------------------
# Filename / file parsing
#
# Format: lsb.<benchmark>_<backend>_<compiler>[_<vendor>]_<size>_<device>.r<run>[-<variant>]
#
# Examples:
#   lsb.dwt_cuda_nvcc_medium_rtx5090.r0
#   lsb.dwt_cuda_scale_amd_medium_mi300a.r0     (compiler = "scale" + vendor "amd")
#   lsb.dwt_hip_hipcc_medium_mi300a.r0
#
# "compiler" is variable-width: a single token (nvcc, hipcc, opencl) or two
# tokens (scale + nvidia/amd), since SCALE's compiler identity also encodes
# which vendor backend it targeted. We detect this by checking whether the
# token immediately after backend is literally "scale".
#
# "device" is kept distinct from "system" (the node hostname, read from the
# file's own header) because a single node can host more than one GPU --
# device identifies which accelerator actually produced this result.
# ---------------------------------------------------------------------------

SIZE_LEVELS <- c("tiny", "small", "medium", "large")

parse_lsb_filename <- function(path) {
  base <- basename(path)

  m <- str_match(base, "^lsb\\.(.+)\\.r([0-9]+)(?:-([0-9]+))?$")

  if (is.na(m[1, 1])) {
    return(NULL)
  }

  body <- m[1, 2]
  run <- as.integer(m[1, 3])
  run_variant <- ifelse(is.na(m[1, 4]), 0L, as.integer(m[1, 4]))

  tokens <- str_split(body, "_")[[1]]

  if (length(tokens) < 4) {
    warning("Unexpected LSB filename structure (too few tokens): ", base)
    return(NULL)
  }

  idx <- 1L
  benchmark <- tokens[idx]; idx <- idx + 1L
  backend <- tokens[idx]; idx <- idx + 1L

  if (idx <= length(tokens) && tolower(tokens[idx]) == "scale") {
    if (idx + 1L > length(tokens)) {
      warning("LSB filename has 'scale' with no vendor token: ", base)
      return(NULL)
    }
    compiler <- paste(tokens[idx], tokens[idx + 1L], sep = "-")
    idx <- idx + 2L
  } else {
    compiler <- tokens[idx]
    idx <- idx + 1L
  }

  if (idx > length(tokens)) {
    warning("LSB filename missing size/device tokens: ", base)
    return(NULL)
  }

  size <- tolower(tokens[idx]); idx <- idx + 1L

  if (idx > length(tokens)) {
    warning("LSB filename missing device token: ", base)
    return(NULL)
  }

  # Device is whatever tokens remain -- joined back together in case a
  # device name itself contains an underscore (e.g. "rx_7900_xtx").
  device <- paste(tokens[idx:length(tokens)], collapse = "_")

  if (!(size %in% SIZE_LEVELS)) {
    warning(
      "Unrecognised size token '", size, "' in ", base,
      " (expected one of: ", paste(SIZE_LEVELS, collapse = ", "), ")"
    )
  }

  list(
    file = base,
    benchmark = benchmark,
    backend = backend,
    compiler = compiler,
    size = size,
    device = device,
    run = run,
    run_variant = run_variant
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

parse_lsb_table_fast <- function(lines, header_idx, meta, system, runtime_s) {
  header <- str_split(str_squish(lines[[header_idx]]), "\\s+")[[1]]

  region_pos <- match("region", header)
  time_pos <- match("time", header)
  id_pos <- match("id", header)
  overhead_pos <- match("overhead", header)
  repeat_pos <- match("repeats_to_two_seconds", header)

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

  repeat_id <- if (!is.na(repeat_pos)) {
    suppressWarnings(as.integer(fields[keep, repeat_pos]))
  } else {
    rep(0L, sum(keep))
  }
  repeat_id[is.na(repeat_id)] <- 0L

  region <- fields[keep, region_pos]

  tibble(
    file = meta$file,
    system = system,
    benchmark = meta$benchmark,
    backend = meta$backend,
    compiler = meta$compiler,
    implementation = paste(meta$backend, meta$compiler, sep = "/"),
    size = factor(meta$size, levels = SIZE_LEVELS),
    device = meta$device,
    run = meta$run,
    run_variant = meta$run_variant,
    region = region,
    region_class = classify_region(region),
    repeat_id = repeat_id,
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

  # Batches below are processed one wave at a time, and each wave blocks
  # until its slowest file finishes -- so a huge file sharing a wave with
  # small ones stalls otherwise-idle workers for that whole wave. Sorting
  # largest-first (longest-job-first scheduling) means big files get
  # spread across the earliest waves one-per-wave as much as possible,
  # instead of clumping unpredictably with whatever else was nearby in
  # directory listing order.
  file_sizes <- file.info(files)$size
  files <- files[order(-file_sizes)]

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

# ---------------------------------------------------------------------------
# Caching layer
#
# The cache key is a hash of a *manifest* (filename + size + mtime for every
# lsb.* file in results_dir) -- not file contents, so computing the key is
# cheap even over thousands of files. Any addition, removal, or modification
# changes the manifest hash and triggers a full reparse. Otherwise, the
# previously parsed tibble is loaded straight from disk.
# ---------------------------------------------------------------------------

lsb_list_files <- function(results_dir) {
  list.files(results_dir, pattern = "^lsb\\.", full.names = TRUE)
}

lsb_manifest <- function(files) {
  info <- file.info(files)

  tibble(
    file = basename(files),
    size = info$size,
    mtime = as.numeric(info$mtime)
  ) |>
    arrange(file)
}

lsb_manifest_hash <- function(manifest) {
  digest::digest(manifest, algo = "xxhash64")
}

#' Read all LSB files in a directory, using a manifest-hash cache.
#'
#' @param results_dir Directory containing lsb.* files.
#' @param cache_dir Where to store the cache (default: results_dir/.lsb_cache).
#' @param force If TRUE, ignore any existing cache and reparse everything.
#' @return A tibble of parsed LSB rows (same shape as parse_files_with_progress()).
read_lsb_cached <- function(results_dir,
                             cache_dir = file.path(results_dir, ".lsb_cache"),
                             force = FALSE) {
  files <- lsb_list_files(results_dir)

  if (length(files) == 0) {
    stop("No LSB files matched in ", results_dir)
  }

  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  data_file <- file.path(cache_dir, "parsed.rds")
  key_file <- file.path(cache_dir, "manifest.key")

  manifest <- lsb_manifest(files)
  key <- lsb_manifest_hash(manifest)

  if (!force && file.exists(data_file) && file.exists(key_file)) {
    cached_key <- readLines(key_file, warn = FALSE)[1]

    if (identical(cached_key, key)) {
      log_msg(
        "cache hit (%s): loading %d files' worth of parsed data from %s",
        key,
        length(files),
        data_file
      )
      return(readRDS(data_file))
    }

    log_msg("cache stale (manifest changed): reparsing %d files", length(files))
  } else if (!force) {
    log_msg("cache miss: parsing %d files for the first time", length(files))
  } else {
    log_msg("--force-reparse set: parsing %d files", length(files))
  }

  sizes_mib <- sum(manifest$size, na.rm = TRUE) / (1024 * 1024)
  log_msg("parsing %d LSB files, %.2f MiB total", length(files), sizes_mib)

  df <- parse_files_with_progress(files)

  if (nrow(df) == 0) {
    stop("No readable LSB files found in ", results_dir)
  }

  log_msg(
    "parsed %d rows (%.1f MiB in memory) -- writing cache to %s",
    nrow(df),
    as.numeric(object.size(df)) / (1024 * 1024),
    data_file
  )

  # compress = "xz" is much slower than "gzip" for little size benefit on a
  # data frame this large, and gives zero progress feedback while it runs --
  # on a multi-GB object it can look indistinguishable from a hang. "gzip"
  # (the saveRDS default) is dramatically faster for a modest size cost.
  # If the cache directory's disk usage matters more than write time, pass
  # compress = "xz" back in, but expect a long wait with no console output.
  saveRDS(df, data_file, compress = "gzip")
  writeLines(key, key_file)
  log_msg("cache written: %s (%d rows)", data_file, nrow(df))

  df
}

#' Force-invalidate a cache without needing to know the key format.
clear_lsb_cache <- function(results_dir, cache_dir = file.path(results_dir, ".lsb_cache")) {
  if (dir.exists(cache_dir)) {
    unlink(cache_dir, recursive = TRUE)
    log_msg("cleared cache directory: %s", cache_dir)
  }
}
