# Run the BayesSpace baseline on the 12 DLPFC sections.
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
#   results/bayesspace/ labels and per-slice metrics.
#
# Usage
#   Rscript 03_run_bayesspace.R [all | <section id>]

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
# H9: BayesSpace baseline (12 DLPFC slices, standard pipeline spatialPreprocess + spatialCluster)
# Usage: Rscript run_bayesspace.R [slice_id | all]
suppressPackageStartupMessages({
  library(SingleCellExperiment); library(SpatialExperiment); library(Matrix); library(BayesSpace)
})
args <- commandArgs(trailingOnly = TRUE)
target <- if (length(args) >= 1) args[1] else "all"
SEED <- as.integer(Sys.getenv("HMRF_SEED", unset = "49"))
set.seed(SEED)
DATA <- file.path(ROOT, "data/dlpfc12")
OUT  <- file.path(ROOT, "experiments/h9-dlpfc12-benchmark/results/bayesspace")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
ALL <- readLines(file.path(ROOT, "config", "slices_dlpfc12.txt"))

nmi_fun <- function(a, b) {
  a <- as.character(a); b <- as.character(b)
  ua <- sort(unique(a)); ub <- sort(unique(b))
  ct <- table(factor(a, ua), factor(b, ub))
  p <- ct / sum(ct); pi <- rowSums(p); pj <- colSums(p)
  nz <- p > 0
  I <- sum(p[nz] * log(p[nz] / (outer(pi, pj)[nz])))
  Ha <- -sum(pi[pi > 0] * log(pi[pi > 0])); Hb <- -sum(pj[pj > 0] * log(pj[pj > 0]))
  if (Ha + Hb <= 0) return(1.0)
  I / ((Ha + Hb) / 2)
}

run_slice <- function(sid) {
  t0 <- Sys.time()
  m <- readMM(gzfile(file.path(DATA, sid, "counts.mtx.gz")))
  sp <- read.csv(file.path(DATA, sid, "spots.csv"), stringsAsFactors = FALSE)
  genes <- readLines(file.path(DATA, sid, "genes.txt"))
  rownames(m) <- genes; colnames(m) <- sp$barcode
  k <- length(unique(sp$label))
  cd <- DataFrame(array_col = sp$col, array_row = sp$row, truth = sp$label)
  sc <- as.matrix(sp[, c("col", "row")]); colnames(sc) <- c("col", "row")
  sce <- SpatialExperiment(assays = list(counts = m), colData = cd, spatialCoords = sc)
  sce <- spatialPreprocess(sce, platform = "Visium", n.PCs = 15, n.HVGs = 2000, log.normalize = TRUE)
  sce <- spatialCluster(sce, q = k, platform = "Visium", d = 15, nrep = 1000, burn.in = 100)
  pred <- colData(sce)$spatial.cluster
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ari <- mclust::adjustedRandIndex(sp$label, pred)
  nmi <- nmi_fun(sp$label, pred)
  write.csv(data.frame(barcode = sp$barcode, pred = pred), file.path(OUT, sprintf("%s_labels.csv", sid)), row.names = FALSE)
  res <- data.frame(slice = sid, method = "BayesSpace", k = k, ari = round(ari, 4), nmi = round(nmi, 4), t_s = round(dt, 1))
  write.table(res, file.path(OUT, sprintf("%s_result.csv", sid)), sep = ",", row.names = FALSE, quote = FALSE)
  cat(sprintf("%s: ARI=%.4f NMI=%.4f k=%d t=%.1fs\n", sid, ari, nmi, k, dt)); flush.console()
  res
}

if (target == "smoke") {
  run_slice("151673")
} else {
  slices <- if (target == "all") ALL else target
  all <- NULL
  for (s in slices) {
    r <- try(run_slice(s), silent = FALSE)
    if (!inherits(r, "try-error")) all <- rbind(all, r)
  }
  if (!is.null(all)) {
    f <- file.path(OUT, "bayesspace_all.csv")
    prev <- if (file.exists(f)) read.csv(f) else NULL
    prev <- prev[!(prev$slice %in% all$slice), , drop = FALSE]
    write.csv(rbind(prev, all), f, row.names = FALSE)
  }
  cat("BAYESSPACE DONE\n")
}
