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
  library(data.table)
})

# File-level parallelism is already handled by mclapply across many files
# at once (see get_worker_count()). fread() also has its own internal
# multithreading via OpenMP -- left at its default, that would mean every
# forked worker ALSO tries to spawn several threads for its own fread()
# call, oversubscribing the machine (e.g. 256 mclapply workers x 4 fread
# threads each = 1024 threads competing for way fewer cores). Pinning fread
# to 1 thread here means all the parallelism comes from one place only.
data.table::setDTthreads(1)

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

  n_workers <- max(1L, min(requested, cores, n_files))

  log_msg(
    "worker count: using %d (requested=%d via PLOT_LSB_JOBS, detected physical cores=%s, n_files=%d)",
    n_workers, requested, ifelse(is.na(parallel::detectCores(logical = FALSE)), "NA", cores), n_files
  )

  n_workers
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
  # NOTE: this used to be `lines |> str_subset(...) |> first()` followed by
  # `if (is.na(runtime_line))`. purrr::first() on zero matches returns NULL,
  # not NA -- and is.na(NULL) is logical(0), not FALSE, so `if (is.na(NULL))`
  # crashes with "argument is of length zero" instead of gracefully
  # returning NA. This silently killed any file lacking a "# Runtime:" line
  # anywhere in the scanned window, which turned out to be the vast
  # majority of files across the suite (most benchmarks apparently don't
  # emit that header at all -- only a minority do). Checking length()
  # directly avoids the NULL/NA confusion entirely.
  matches <- str_subset(lines, "^# Runtime:")

  if (length(matches) == 0) {
    return(NA_real_)
  }

  as.numeric(str_match(matches[1], "# Runtime:\\s*([0-9.]+)\\s*s")[1, 2])
}

extract_nodename_fast <- function(lines) {
  idx <- grep("^# Nodename:", lines, fixed = FALSE)[1]

  if (is.na(idx)) {
    return(NA_character_)
  }

  sub("\\..*$", "", sub("^# Nodename:\\s*", "", lines[[idx]]))
}

parse_lsb_table_dt <- function(dt, meta, system, runtime_s) {
  if (nrow(dt) == 0) {
    return(NULL)
  }

  col_names <- names(dt)
  region_pos <- match("region", col_names)
  time_pos <- match("time", col_names)
  repeat_pos <- match("repeats_to_two_seconds", col_names)

  if (is.na(region_pos) || is.na(time_pos)) {
    return(NULL)
  }

  # IMPORTANT: this aggregates away individual rows *inside this one file*
  # immediately, rather than returning every raw row for bind_rows() to
  # combine across all 16000+ files. The stabilizing loop means a single
  # file can carry hundreds of thousands of near-duplicate rows (the same
  # region measured over and over until ~2s elapses); across the whole
  # results tree that adds up to billions of rows and hundreds of GB if
  # carried through as-is, which is what caused the OOM kill. Nothing
  # downstream (this heatmap, or the repeats-normalization logic) actually
  # needs row-level granularity -- only the per-region total time and how
  # many repeats it was measured over, both of which data.table can compute
  # in C in a fraction of a second per file.
  slim <- data.table(
    region = as.character(dt[[region_pos]]),
    time_us = suppressWarnings(as.numeric(dt[[time_pos]]))
  )

  if (!is.na(repeat_pos)) {
    slim[, repeat_id := suppressWarnings(as.integer(dt[[repeat_pos]]))]
  } else {
    slim[, repeat_id := 0L]
  }
  slim[is.na(repeat_id), repeat_id := 0L]

  slim <- slim[!is.na(time_us)]
  if (nrow(slim) == 0) {
    return(NULL)
  }

  agg <- slim[, .(
    total_time_us = sum(time_us),
    n_rows = .N,
    n_repeats = uniqueN(repeat_id)
  ), by = region]

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
    region = agg$region,
    region_class = classify_region(agg$region),
    total_time_us = agg$total_time_us,
    n_rows = agg$n_rows,
    n_repeats = agg$n_repeats,
    runtime_s = runtime_s
  )
}

# How many leading lines to sniff for the header/metadata block before
# falling back to a full read. Every file we've inspected has its header
# within the first ~10 lines, so 200 leaves generous headroom without ever
# touching the (possibly huge) data section below it.
LSB_HEADER_SCAN_LINES <- 200L

find_lsb_header <- function(path) {
  lines <- readLines(path, n = LSB_HEADER_SCAN_LINES, warn = FALSE)
  header_idx <- grep("\\bregion\\b.*\\bid\\b.*\\btime\\b.*\\boverhead\\b", lines)[1]

  if (is.na(header_idx)) {
    header_idx <- which(str_detect(lines, "^\\s*\\S+\\s+.*\\s+region\\s+.*\\stime\\s+"))[1]
  }

  if (!is.na(header_idx)) {
    return(list(lines = lines, header_idx = header_idx, full_read = FALSE))
  }

  # Header wasn't in the sampled window -- rare based on every file we've
  # seen, but fall back to a full read for this one file rather than
  # silently dropping it. Warn so unusual files are visible in the log
  # instead of just vanishing.
  warning(
    "Header not found in first ", LSB_HEADER_SCAN_LINES, " lines of ",
    basename(path), " -- falling back to a full read for this file"
  )
  lines <- readLines(path, warn = FALSE)
  header_idx <- grep("\\bregion\\b.*\\bid\\b.*\\btime\\b.*\\boverhead\\b", lines)[1]
  if (is.na(header_idx)) {
    header_idx <- which(str_detect(lines, "^\\s*\\S+\\s+.*\\s+region\\s+.*\\stime\\s+"))[1]
  }

  list(lines = lines, header_idx = header_idx, full_read = TRUE)
}

# extract_runtime() only looks within whatever line window it's given. The
# header-sniff window above covers the case where "# Runtime:" sits with
# the other header comments near the top of the file (true in every sample
# we've seen so far). If it's ever instead appended as a trailer at the
# very end of the file, this cheap tail-read catches that case too, without
# falling back to reading the whole file just to find one line.
extract_runtime_from_tail <- function(path, n = 20L) {
  tail_lines <- tryCatch(
    system2("tail", c("-n", as.character(n), shQuote(path)), stdout = TRUE, stderr = FALSE),
    error = function(e) character(0),
    warning = function(w) character(0)
  )
  extract_runtime(tail_lines)
}

# Cheap way to count a file's total lines without reading its content into R
# -- used below to tell fread exactly how many data rows exist, rather than
# relying on its own footer-detection heuristic.
count_lines_fast <- function(path) {
  out <- tryCatch(
    system2("wc", c("-l", shQuote(path)), stdout = TRUE, stderr = FALSE),
    error = function(e) character(0)
  )
  if (length(out) == 0) {
    return(NA_integer_)
  }
  n <- suppressWarnings(as.integer(str_extract(out[1], "^[0-9]+")))
  if (is.na(n)) NA_integer_ else n
}

read_lsb_file_fast <- function(path) {
  meta <- parse_lsb_filename(path)
  if (is.null(meta)) {
    return(NULL)
  }

  header <- find_lsb_header(path)

  if (is.na(header$header_idx)) {
    warning("Could not locate LSB data header in ", basename(path))
    return(NULL)
  }

  system <- extract_nodename_fast(header$lines)
  runtime_s <- extract_runtime(header$lines)

  if (is.na(runtime_s) && !header$full_read) {
    runtime_s <- extract_runtime_from_tail(path)
  }

  col_names <- str_split(str_squish(header$lines[[header$header_idx]]), "\\s+")[[1]]

  # fread() normally auto-detects and discards a trailing "# Runtime:"
  # footer line gracefully (the "Discarded single-line footer" message
  # suppressed below). That auto-detection is a heuristic: it needs enough
  # consistent-looking data rows to confidently recognize the footer as an
  # outlier rather than real data. Benchmarks that don't use the
  # stabilizing repeat-loop (e.g. cwt, which runs its pipeline exactly
  # once) can have as few as 8 data rows total -- not enough for the
  # heuristic to work, so fread instead throws a hard "Can't assign N
  # names to M-column data.table" error on the genuine column-count
  # mismatch between the 7-field header and the 9-field footer line
  # ("# Runtime: 0.227708 s (overhead: 0.000000 %) 8 records" splits into
  # 9 whitespace-separated tokens). Rather than depend on the heuristic at
  # all, explicitly check whether the file's last line is a footer, and if
  # so tell fread exactly how many data rows to read via nrows= -- so the
  # footer is never handed to the parser in the first place, regardless of
  # file size.
  nrows_arg <- NA_integer_
  tail_line <- tryCatch(
    system2("tail", c("-n", "1", shQuote(path)), stdout = TRUE, stderr = FALSE),
    error = function(e) character(0)
  )
  if (length(tail_line) > 0 && str_detect(tail_line[1], "^# Runtime:")) {
    total_lines <- count_lines_fast(path)
    if (!is.na(total_lines)) {
      nrows_arg <- total_lines - header$header_idx - 1L
    }
  }

  # suppressWarnings() here is deliberately narrow in intent (even though
  # syntactically it suppresses all warnings from this call): on files
  # where nrows_arg above wasn't determined (e.g. tail read failed) fread
  # may still fall back to its own footer-detection and warn about it --
  # expected, not a problem, and capturing it as a diagnostic warning (see
  # parse_one() in parse_files_with_progress()) buried genuinely
  # informative warnings under thousands of copies of this one.
  fread_args <- list(
    input = path,
    skip = header$header_idx,
    header = FALSE,
    col.names = col_names,
    fill = TRUE,
    showProgress = FALSE
  )
  if (!is.na(nrows_arg)) {
    fread_args$nrows <- max(nrows_arg, 0L)
  }

  dt <- tryCatch(
    suppressWarnings(do.call(data.table::fread, fread_args)),
    error = function(e) {
      warning("fread failed on ", basename(path), ": ", conditionMessage(e))
      NULL
    }
  )

  if (is.null(dt) || nrow(dt) == 0) {
    return(NULL)
  }

  parse_lsb_table_dt(dt, meta, system, runtime_s)
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
    path <- files[[i]]
    err_msg <- NULL
    warn_msgs <- character(0)

    # Wrapping the whole per-file call is deliberate: with 16000+ files
    # pulled from multiple hosts and harness invocations, some fraction
    # being truncated, zero-byte, or otherwise malformed is basically
    # guaranteed. Without this, mclapply silently converts an error inside
    # any single forked worker into a try-error object instead of the
    # list(rows=, rows_n=) structure parse_one() is supposed to return --
    # which then crashes bind_rows() far downstream with a confusing
    # "$ operator is invalid for atomic vectors" error that doesn't say
    # which file caused it. This keeps one bad file from taking down the
    # entire parse, and reports exactly which file and why.
    #
    # withCallingHandlers() around the tryCatch additionally captures
    # warning() calls (e.g. "Could not locate LSB data header in ...",
    # raised when a file's header doesn't match the expected pattern and
    # read_lsb_file_fast() returns NULL cleanly rather than erroring).
    # Without this, such warnings vanish entirely: R does NOT automatically
    # propagate warnings raised inside forked mclapply workers back to the
    # parent process's console the way it does for a plain sequential
    # lapply(). A file could silently contribute zero rows with no error
    # AND no visible explanation of why -- exactly the failure mode that
    # made a missing benchmark look like a mystery instead of a logged
    # warning.
    rows <- withCallingHandlers(
      tryCatch(
        read_lsb_file_fast(path),
        error = function(e) {
          err_msg <<- conditionMessage(e)
          NULL
        }
      ),
      warning = function(w) {
        warn_msgs <<- c(warn_msgs, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )

    list(
      rows = rows,
      rows_n = if (is.null(rows)) 0L else nrow(rows),
      file = basename(path),
      error = err_msg,
      warnings = if (length(warn_msgs) > 0) paste(warn_msgs, collapse = " | ") else NULL
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

  failed <- Filter(function(x) !is.null(x$error), parsed)
  warned <- Filter(function(x) is.null(x$error) && !is.null(x$warnings), parsed)

  if (length(failed) > 0) {
    log_msg(
      "%d of %d file(s) failed to parse and were skipped (not included in results):",
      length(failed), length(files)
    )
    show_n <- min(length(failed), 30)
    for (f in failed[seq_len(show_n)]) {
      log_msg("  %s: %s", f$file, f$error)
    }
    if (length(failed) > show_n) {
      log_msg("  ... and %d more (see full list by re-running with a smaller file set to isolate)", length(failed) - show_n)
    }
  }

  if (length(warned) > 0) {
    log_msg(
      "%d of %d file(s) produced a warning during parsing (may still have contributed 0 rows -- check rows_n if a whole benchmark looks missing):",
      length(warned), length(files)
    )
    show_n <- min(length(warned), 30)
    for (f in warned[seq_len(show_n)]) {
      log_msg("  %s (rows=%d): %s", f$file, f$rows_n, f$warnings)
    }
    if (length(warned) > show_n) {
      log_msg("  ... and %d more", length(warned) - show_n)
    }
  }

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
  # recursive = TRUE is required: rsync-based collection layouts commonly
  # drop each host's results into a subdirectory (results/alpha/,
  # results/excl/, ...), and list.files() does not descend into
  # subdirectories by default. Without this, files outside the top level
  # of results_dir are silently invisible -- no error, no warning, just an
  # incomplete file list.
  list.files(results_dir, pattern = "^lsb\\.", full.names = TRUE, recursive = TRUE)
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
