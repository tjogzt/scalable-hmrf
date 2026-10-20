# BayesSpace multi-restart control runs.
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
#   results/bayesspace_multi/ outputs.
#
# Usage
#   Rscript 05_run_bayesspace_multi.R [all | <section id>]

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
# H9 matched-budget extension: BayesSpace under 5 independent seeds per slice
# (identical official pipeline as run_bayesspace.R: spatialPreprocess + spatialCluster nrep=1000).
# Usage: Rscript run_bayesspace_multi.R <slice_id> <seed>
suppressPackageStartupMessages({
  library(SingleCellExperiment); library(SpatialExperiment); library(Matrix); library(BayesSpace)
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript run_bayesspace_multi.R <slice_id> <seed>")
sid <- args[1]; seed <- as.integer(args[2])
set.seed(seed)
DATA <- file.path(ROOT, "data/dlpfc12")
OUT  <- file.path(ROOT, "experiments/h9-dlpfc12-benchmark/results/matched_budget/bayesspace")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

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
write.csv(data.frame(barcode = sp$barcode, pred = pred),
          file.path(OUT, sprintf("%s_seed%d_labels.csv", sid, seed)), row.names = FALSE)
res <- data.frame(slice = sid, method = "BayesSpace", seed = seed, k = k,
                  ari = round(ari, 4), nmi = round(nmi, 4), t_s = round(dt, 1))
write.csv(res, file.path(OUT, sprintf("%s_seed%d_result.csv", sid, seed)), row.names = FALSE)
cat(sprintf("BS %s seed=%d: ARI=%.4f NMI=%.4f t=%.1fs\n", sid, seed, ari, nmi, dt)); flush.console()
