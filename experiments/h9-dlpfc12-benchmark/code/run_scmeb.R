# --- portable project root (no hardcoded user paths) ---
.hmrf_script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f) > 0) normalizePath(dirname(f)) else getwd()
}
.hmrf_find_root <- function(start) {
  d <- normalizePath(start)
  while (!file.exists(file.path(d, "src", "hmrf_solvers.py")) && dirname(d) != d) d <- dirname(d)
  if (!file.exists(file.path(d, "src", "hmrf_solvers.py"))) stop("project root not found")
  d
}
ROOT <- Sys.getenv("HMRF_ROOT", unset = "")
if (ROOT == "") ROOT <- .hmrf_find_root(.hmrf_script_dir())
# ------------------------------------------------------
# H9: SC.MEB baseline (same pipeline: PC50 exported by this method + the same hexagonal adjacency)
# Usage: Rscript run_scmeb.R [slice_id | all | smoke]
suppressPackageStartupMessages({
  library(Matrix); library(SC.MEB); library(SingleCellExperiment)
  library(scuttle); library(scater); library(scran); library(BiocSingular)
})
args <- commandArgs(trailingOnly = TRUE)
target <- if (length(args) >= 1) args[1] else "all"
set.seed(49)
FEAT <- file.path(ROOT, "experiments/h9-dlpfc12-benchmark/results/features")
OUT  <- file.path(ROOT, "experiments/h9-dlpfc12-benchmark/results/scmeb")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
ALL <- c("151507","151508","151509","151510","151669","151670","151671","151672","151673","151674","151675","151676")

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
  # Official config C: SC.MEB::spatialPreprocess (counts -> lognorm -> 2000 HVG -> 15 PC)
  # Diagnostic evidence on 151673: A (this method, PC50)=0.186 / B (PC15)=0.282 / C (official preprocessing)=0.357
  m <- readMM(gzfile(file.path(ROOT, "data/dlpfc12", sid, "counts.mtx.gz")))
  sp <- read.csv(file.path(ROOT, "data/dlpfc12", sid, "spots.csv"), stringsAsFactors = FALSE)
  genes <- readLines(file.path(ROOT, "data/dlpfc12", sid, "genes.txt"))
  rownames(m) <- genes; colnames(m) <- sp$barcode
  sce <- SingleCellExperiment(assays = list(counts = m))
  colData(sce)$col <- sp$col; colData(sce)$row <- sp$row
  sce <- SC.MEB::spatialPreprocess(sce, platform = "Visium", n.PCs = 15,
                                   n.HVGs = 2000, log.normalize = TRUE)
  pc <- reducedDim(sce, "PCA")
  edges <- as.matrix(read.csv(file.path(FEAT, sprintf("%s_edges.csv", sid)), header = FALSE))
  truth <- read.csv(file.path(FEAT, sprintf("%s_truth.csv", sid)))$truth
  n <- nrow(pc); k <- length(unique(truth))
  D <- sparseMatrix(i = edges[, 1], j = edges[, 2], x = 1, dims = c(n, n))
  res <- SC.MEB(y = pc, Adj_sp = D, K_set = c(k), beta_grid = seq(0, 4, 0.2),
                parallel = TRUE, num_core = 5, PX = TRUE, maxIter_ICM = 10, maxIter = 50)
  if (target == "smoke") { cat("struct of res:\n"); str(res, max.level = 1) }
  labels <- as.integer(as.vector(res[[1]]))
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ari <- mclust::adjustedRandIndex(truth, labels)
  nmi <- nmi_fun(truth, labels)
  write.csv(data.frame(barcode = seq_len(n), pred = labels),
            file.path(OUT, sprintf("%s_labels.csv", sid)), row.names = FALSE)
  cat(sprintf("%s: ARI=%.4f NMI=%.4f k=%d t=%.1fs\n", sid, ari, nmi, k, dt)); flush.console()
  data.frame(slice = sid, method = "SC.MEB", k = k, ari = round(ari, 4), nmi = round(nmi, 4), t_s = round(dt, 1))
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
    f <- file.path(OUT, "scmeb_all.csv")
    prev <- if (file.exists(f)) read.csv(f) else NULL
    prev <- prev[!(prev$slice %in% all$slice), , drop = FALSE]
    write.csv(rbind(prev, all), f, row.names = FALSE)
  }
  cat("SCMEB DONE\n")
}
