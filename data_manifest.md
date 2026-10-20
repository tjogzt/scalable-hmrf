# 数据清单（不纳入 git）

`data/` 共约 263 MB 公开原始数据，已由 `.gitignore` 排除；本文件记录其来源与用途，便于重建。

## data/dlpfc12/（249 MB，12 切片）

- 切片编号：151507、151508、151509、151510、151669、151670、151671、151672、151673、151674、151675、151676
- 每切片内容：`counts.mtx.gz`（基因×spot 计数矩阵）、`genes.txt`、`spots.csv`（坐标）
- 来源：10x Visium 人背外侧前额叶（DLPFC）公开数据，spatialLIBD 项目（Lieber Institute；Maynard et al., 2021）；原始矩阵可由 spatialLIBD（Bioconductor）公开数据重建
- 用途：`experiments/h4*`、`h9-dlpfc12-benchmark`（K=7 层注释基准、匹配预算对比）

## data/dlpfc/（14 MB，单切片扁平常量）

- `151673_raw_feature_bc_matrix.h5`、`cluster_labels.csv`、`tissue_positions_list.txt`
- 用途：早期单切片验证（h4 之前的原型）

## 外部数据（不在本机仓库内）

- Xenium 乳腺 IDC（574,852 细胞 × 280 基因）：`/Volumes/tjogzt4T/data/Xenium_V1_FFPE_Human_Breast_IDC_outs`（外置盘；`experiments/h10-xenium574k` 使用，驱动脚本用 `HMRF_XENIUM_DIR` 指定路径）
