# Diagnose the SC-MEB adjacency construction.
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
#   Console diagnostics.
#
# Usage
#   Rscript 17_diagnose_scmeb_adjacency.R [all | <section id>]

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
# Fixed version of config C: SC.MEB official spatialPreprocess (explicitly load the namespaces its internals depend on)
suppressPackageStartupMessages({
  library(Matrix); library(SingleCellExperiment); library(SC.MEB)
  library(scuttle); library(scater); library(scran); library(BiocSingular)
})
SEED <- as.integer(Sys.getenv("HMRF_SEED", unset = "49"))
set.seed(SEED)
FEAT <- file.path(ROOT, "experiments/h9-dlpfc12-benchmark/results/features")
sid <- "151673"
edges <- as.matrix(read.csv(file.path(FEAT, sprintf("%s_edges.csv", sid)), header = FALSE))
truth <- read.csv(file.path(FEAT, sprintf("%s_truth.csv", sid)))$truth
n <- length(truth); k <- length(unique(truth))
D <- sparseMatrix(i = edges[, 1], j = edges[, 2], x = 1, dims = c(n, n))

m <- readMM(gzfile(file.path(ROOT, "data/dlpfc12", sid, "counts.mtx.gz")))
sp <- read.csv(file.path(ROOT, "data/dlpfc12", sid, "spots.csv"), stringsAsFactors = FALSE)
genes <- readLines(file.path(ROOT, "data/dlpfc12", sid, "genes.txt"))
rownames(m) <- genes; colnames(m) <- sp$barcode
sce <- SingleCellExperiment(assays = list(counts = m))
colData(sce)$col <- sp$col; colData(sce)$row <- sp$row

sce <- SC.MEB::spatialPreprocess(sce, platform = "Visium", n.PCs = 15, n.HVGs = 2000, log.normalize = TRUE)
pcC <- reducedDim(sce, "PCA")
cat("PCA dims:", dim(pcC), "\n")
resC <- SC.MEB(y = pcC, Adj_sp = D, K_set = c(k), beta_grid = seq(0, 4, 0.2),
               parallel = TRUE, num_core = 5, PX = TRUE, maxIter_ICM = 10, maxIter = 50)
ariC <- mclust::adjustedRandIndex(truth, as.vector(resC[[1]]))
cat(sprintf("C(official preproc): ARI=%.4f\n", ariC))
cat("DIAG C DONE\n")
