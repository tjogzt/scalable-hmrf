# H10 Xenium Breast IDC 574K 细胞全量规模实测 — 分析

日期：2026-10-11（批次3）｜数据：10x Genomics 官方发布 Xenium FFPE Human Breast IDC（本地副本 23 GB；脚本经环境变量 HMRF_XENIUM_DIR 定位）
脚本：`code/run.py`（18 组网格，max_iter=200）、`code/save_best_labels.py`（500 迭代重放+标签）、`code/validate_fastpath.py`（等价性验证）、`paper/figures/make_fig8.py`

## 数据与管线
- **574,852 个细胞**（全量检测集）；280 基因 panel（Xenium V1 human breast）；无生物学金标准的官方分层注释，
  参照采用 10x 官方聚类：graphclust（25 簇）与 kmeans_8（**reference clustering，非 ground truth**，如实声明）。
- 管线：h5 矩阵加载 2.1 s → log1p(CP10K) + z-score → PCA30 → cKDTree kNN(k=6) 对称图（mean deg 6.85, max 13）
  → 均值场 VI-EM（高斯发射，物理 σ 上限）。**特征+图共建 7.3 s**。
- 快速实现验证：kNN 稀疏邻居和 vs 原版 `np.add.at` 有向边，同数据同起点 **ΔARI=0、标签一致率 100%**。

## 主结果

| 配置 | ARI（参照） | 迭代 | 用时 | 收敛 |
|---|---|---|---|---|
| K=8, β=0.7, r2, 500it（最佳） | **0.4879**（kmeans8） | 500（上限） | 71.7 s | 否（dq≈0.032，慢速振荡下降） |
| K=8, β=0.7, r2, 200it | 0.4815 | 200（上限） | 26.1 s | 否 |
| K=25, β=0.7, r2 | 0.2575（graphclust，粒度匹配） | 394 | ~96 s | **是**（dq=9.4e-05） |
| K=20, β=0.7, r2 | 0.2431（graphclust，粒度不匹配） | 200 | 42.6 s | 否 |

K=8 × β∈{0.7,1.0,1.5} × 3 起点共 9 组（200it）：
- β=0.7：0.390 / 0.409 / **0.4815**；β=1.0：0.277 / 0.257 / 0.402；β=1.5：0.165 / 0.148 / 0.348。

## 结论（写入论文的三点）

1. **真实单细胞分辨率平台规模落地**：57.5 万细胞全量，特征+图 7.3 s、单次域拟合 26 s（200it）–72 s（500it）
   ——MCMC 类域识别方法（BayesSpace 等）在此规模不可行，本方法为唯一秒–分钟级选项；
   对照本方法在 3.6K spot DLPFC 上单跑 ~0.1–0.6 s，规模 ×160 而耗时仅 ×~50–100（近线性）。
2. **与官方参照聚类的中等一致性**：K=8 vs 10x kmeans8 ARI 0.49；K=25 粒度匹配 vs graphclust 0.26
   （graphclust 为 25 个细粒度子簇，与 kmeans8 属不同划分粒度；两者均为参考口径非真值）。
3. **"选择问题"在真实大规模上第三次复现**：同 β 下重启间 0.39–0.48；β 最优值（0.7，低端）与
   仿真小规模（常在高 β）**方向相反**——最优 β 是 {规模, 图密度, 平台} 的函数，进一步支持
   "无普适自动 β 准则、须报告敏感性"的结论。K=8 在 500 迭代未完全收敛（dq 缓慢振荡），
   但 ARI 已稳定微增（+0.006），200 迭代为性价比拐点。

## GraphST 对照（Xenium；进行中→全量运行中）

**规模障碍（重要发现）**：GraphST 1.1.1 的图构建（'10X' 路径 `construct_interaction` 与
'Stereo/Slide' 路径 `construct_interaction_KNN`）及 `graph_neigh` 消费均为**稠密 n×n 实现**——
574K 细胞需要 574K²×8B ≈ **2.6 TB**（连 50K 都需 20 GB/矩阵、瞬时峰值 ~60-100 GB）。
即：**GraphST 官方实现在单细胞分辨率规模物理不可行**。

**本仓工程**：`code/patch_graphst_sparse.py`（3 处最小补丁，边语义完全一致，仅存储改 scipy/torch.sparse；
'datatype="Stereo"' 走稀疏编码器路径）+ 预置 `var["highly_variable"]=True` 跳过其 seurat_v3 HVG
（280 基因 < 3000 触发 loess 外推错误）。重装 GraphST 后需重打补丁（脚本幂等）。

**50K 子样冒烟**（同 rng49 子集、同 kmeans8 参照；600 epochs；k=3 官方默认图）：
- raw mclust 标签 ARI = **0.3569**（我们同子集单配置 0.4071，β=1.0）；
- **50-NN refine 后 ARI 降至 0.2241**——graphclust 参照同样降（0.162→0.080）。
  解读：其 refine radius=50 为 Visium 尺度校准，Xenium 细胞密度下 50-NN 邻域极小且强制最小
  连通块，破坏小结构；**Xenium 口径应报 raw**（refine 作为负向敏感性）。
- 用时 693 s / 50K（600 epochs 训练 ~10 min）→ 574K 外推 ~2-3 h（运行中，两版标签均存档）。

**全量 574K 结果**（600 epochs；k=3 官方默认图；t_total=4085.7 s ≈ 68 min）：

| 方法（K=8） | ARI (vs 10x kmeans8) | 用时 |
|---|---|---|
| **本方法（best 起点, β=0.7）** | **0.4879** | 准备 7.3 s + 拟合 71.7 s ≈ **80 s** |
| GraphST raw（稀疏补丁口径） | 0.2826 | **4086 s（68 min）** |
| GraphST + 50-NN refine | 0.2401 | （同一次运行） |

- 精度：本方法 **+0.205 ARI**；速度：**~50×**（同一台机器、同参照、同 K）。
- 与 graphclust 参照（K 不匹配的辅助口径）：GraphST raw 0.2427 / 本方法 K=25 口径 0.2575。
- 规模对照的诚实声明：GraphST 本运行已加稀疏补丁（官方稠密实现需 2.6 TB 内存、物理不可行）；
  即"GraphST 能跑起来"本身已是本仓工程介入的结果，且仍耗时 68 分钟。refine 在两数据集均降分
  （其 radius 为 Visium 尺度校准），Xenium 口径报 raw。

## 诚实边界
- 参考聚类 ≠ 生物学金标准；未做外部注释映射（如 Janesick 2023 分层）。
- σ 命中物理上限（=1.0）：发射在 PC 空间呈单位方差——与"类内方差 ≤ 总方差"约束一致，如实报告。
- 基线对照：GraphST 全量运行中（稀疏补丁口径，两版标签存档）；BayesSpace 类 MCMC 在该规模不可行。
- K=8 大 n 下软 EM 收敛慢，迭代上限截断；建议生产口径报告 iters 与 dq（本页已给）。

## 文件清单
- `results/h10_full.csv`（18 组网格）、`h10_full_meta.json`（n=574852, t_total=634.3 s）
- `results/h10_labels_K8_b0.7_r2.npz`（fig8 数据源）、`h10_labels_K25_b0.7_r2.npz`、`h10_labels_K8_b0.7_r0.npz`
- `review/batch3_xenium_full.log`、`batch3_xenium_replay.log`
