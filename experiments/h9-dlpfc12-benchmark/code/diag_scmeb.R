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
# SC.MEB config diagnostics (151673): A=PC50 (already run), B=first 15 PCs, C=SC.MEB official spatialPreprocess (computes its own 15 PCs from counts)
suppressPackageStartupMessages({
  library(Matrix); library(SingleCellExperiment); library(SpatialExperiment); library(SC.MEB)
})
set.seed(49)
FEAT <- file.path(ROOT, "experiments/h9-dlpfc12-benchmark/results/features")
sid <- "151673"
pc <- as.matrix(read.csv(file.path(FEAT, sprintf("%s_pc50.csv", sid)), header = FALSE))
edges <- as.matrix(read.csv(file.path(FEAT, sprintf("%s_edges.csv", sid)), header = FALSE))
truth <- read.csv(file.path(FEAT, sprintf("%s_truth.csv", sid)))$truth
n <- nrow(pc); k <- length(unique(truth))
D <- sparseMatrix(i = edges[, 1], j = edges[, 2], x = 1, dims = c(n, n))

# B: first 15 PCs
resB <- SC.MEB(y = pc[, 1:15], Adj_sp = D, K_set = c(k), beta_grid = seq(0, 4, 0.2),
               parallel = TRUE, num_core = 5, PX = TRUE, maxIter_ICM = 10, maxIter = 50)
ariB <- mclust::adjustedRandIndex(truth, as.vector(resB[[1]]))
cat(sprintf("B(PC1-15): ARI=%.4f\n", ariB))

# C: official spatialPreprocess (counts -> lognorm -> 2000 HVG -> 15 PC)
m <- readMM(gzfile(file.path(ROOT, "data/dlpfc12", sid, "counts.mtx.gz")))
sp <- read.csv(file.path(ROOT, "data/dlpfc12", sid, "spots.csv"), stringsAsFactors = FALSE)
genes <- readLines(file.path(ROOT, "data/dlpfc12", sid, "genes.txt"))
rownames(m) <- genes; colnames(m) <- sp$barcode
sce <- SingleCellExperiment(assays = list(counts = m))
colData(sce)$col <- sp$col; colData(sce)$row <- sp$row
sce <- SC.MEB::spatialPreprocess(sce, platform = "Visium", n.PCs = 15, n.HVGs = 2000, log.normalize = TRUE)
pcC <- reducedDim(sce, "PCA")
resC <- SC.MEB(y = pcC, Adj_sp = D, K_set = c(k), beta_grid = seq(0, 4, 0.2),
               parallel = TRUE, num_core = 5, PX = TRUE, maxIter_ICM = 10, maxIter = 50)
ariC <- mclust::adjustedRandIndex(truth, as.vector(resC[[1]]))
cat(sprintf("C(official preproc): ARI=%.4f\n", ariC))
cat("DIAG DONE\n")
