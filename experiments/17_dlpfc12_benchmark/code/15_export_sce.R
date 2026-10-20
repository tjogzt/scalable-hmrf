# Export SingleCellExperiment objects for the R baselines.
#
# Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
# Created: 2026-10-02
# Updated: 2026-10-20
#
# Inputs
#   data/dlpfc12/<section>/ matrices; config/slices_dlpfc12.txt;
#   repository root from HMRF_ROOT or config/defaults.yaml discovery.
#
# Outputs
#   results/sce/ RDS objects.
#
# Usage
#   Rscript 15_export_sce.R [all | <section id>]

# --- portable project root (no hardcoded user paths) ---
.hmrf_script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f) > 0) normalizePath(dirname(f)) else getwd()
}
.hmrf_find_root <- function(start) {
  d <- normalizePath(start)
  while (!file.exists(file.path(d, "config", "defaults.yaml")) && dirname(d) != d) d <- dirname(d)
  if (!file.exists(file.path(d, "config", "defaults.yaml"))) stop("project root not found")
  d
}
ROOT <- Sys.getenv("HMRF_ROOT", unset = "")
if (ROOT == "") ROOT <- .hmrf_find_root(.hmrf_script_dir())
# ------------------------------------------------------
# H9: export the 12 slices from the cached H³MRF DLPFC SCE to a generic format (for the Scalable-HMRF benchmark)
# Output: data/dlpfc12/<sample>/{counts.mtx.gz, spots.csv, genes.txt}
suppressPackageStartupMessages({library(SingleCellExperiment); library(Matrix)})

CACHE <- Sys.getenv("HMRF_DLPFC_CACHE", unset = "")
if (CACHE == "" || !file.exists(CACHE)) stop("please set HMRF_DLPFC_CACHE to point at dlpfc_sce_cache.rds (see README)")
OUT <- file.path(ROOT, "data/dlpfc12")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

x <- readRDS(CACHE)
cat("loaded:", dim(x), "\n")
samples <- sort(unique(as.character(x$sample_name)))
cat("samples:", paste(samples, collapse = ", "), "\n\n")

for (s in samples) {
  xs <- x[, x$sample_name == s]
  keep <- !is.na(xs$layer_guess_reordered) & as.character(xs$layer_guess_reordered) != ""
  xs <- xs[, keep]
  d <- file.path(OUT, s); dir.create(d, recursive = TRUE, showWarnings = FALSE)
  m <- counts(xs)
  writeMM(m, file.path(d, "counts.mtx"))
  system(paste("gzip -f", shQuote(file.path(d, "counts.mtx"))))
  df <- data.frame(barcode = colnames(xs),
                   row = as.integer(xs$row), col = as.integer(xs$col),
                   label = as.character(xs$layer_guess_reordered))
  write.csv(df, file.path(d, "spots.csv"), row.names = FALSE, quote = FALSE)
  writeLines(rownames(xs), file.path(d, "genes.txt"))
  cat(sprintf("%s: spots=%d genes=%d labels=%s\n", s, ncol(xs), nrow(xs),
              paste(names(table(df$label)), table(df$label), sep = "=", collapse = " ")))
  flush.console()
}
cat("EXPORT DONE\n")
