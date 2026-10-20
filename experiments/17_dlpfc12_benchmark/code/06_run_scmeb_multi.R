# SC-MEB multi-restart control runs.
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
#   results/scmeb_multi/ outputs.
#
# Usage
#   Rscript 06_run_scmeb_multi.R [all | <section id>]

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
# H9 matched-budget extension: SC.MEB under 3 independent seeds per slice
# (identical official pipeline as run_scmeb.R: official spatialPreprocess 15 PC + beta_grid).
# Usage: Rscript run_scmeb_multi.R <slice_id> <seed>
suppressPackageStartupMessages({
  library(Matrix); library(SC.MEB); library(SingleCellExperiment)
  library(scuttle); library(scater); library(scran); library(BiocSingular)
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript run_scmeb_multi.R <slice_id> <seed>")
sid <- args[1]; seed <- as.integer(args[2])
set.seed(seed)
FEAT <- file.path(ROOT, "experiments/h9-dlpfc12-benchmark/results/features")
OUT  <- file.path(ROOT, "experiments/h9-dlpfc12-benchmark/results/matched_budget/scmeb")
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
labels <- as.integer(as.vector(res[[1]]))
dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
ari <- mclust::adjustedRandIndex(truth, labels)
nmi <- nmi_fun(truth, labels)
write.csv(data.frame(barcode = seq_len(n), pred = labels),
          file.path(OUT, sprintf("%s_seed%d_labels.csv", sid, seed)), row.names = FALSE)
out <- data.frame(slice = sid, method = "SC.MEB", seed = seed, k = k,
                  ari = round(ari, 4), nmi = round(nmi, 4), t_s = round(dt, 1))
write.csv(out, file.path(OUT, sprintf("%s_seed%d_result.csv", sid, seed)), row.names = FALSE)
cat(sprintf("SCMEB %s seed=%d: ARI=%.4f NMI=%.4f t=%.1fs\n", sid, seed, ari, nmi, dt)); flush.console()
