# 实验 H1：均值场变分 ≈ Gibbs 后验（正确性闸门）

## 目的
方向二用 VI/图割替代 Gibbs 之前，先证明：均值场变分对 Potts HMRF 的后验模态近似是忠实的。
若 VI 代理本身失真，后续所有提速主张失去基础。

## 设计
- 仿真：30x30 网格，K=5 域（Checkerboard-Gibbs 从低温度 Potts 采样得到真实标签，含自然边界）；
  高斯发射 d=20 基因，mu_k 分离度 1.5，噪声 σ=0.8。
- 对照：① Gibbs 全贝叶斯（oracle 发射参数，后验模态标签）② 均值场 VI（相同一元项，不动点迭代）。
- 扫描：模型 beta ∈ {0.2, 0.5, 1.0}（= 真值 beta），另测 beta 失配 {0.5 真 vs 1.5 模型}。
- 指标：ARI(方法, 真值)、ARI(VI模态, Gibbs模态)、墙钟时间。
- 种子：3 个（噪声/真值采样变化）。

## 预测（机制判断）
- 两个方法 ARI(·, 真值) 相近；ARI(VI, Gibbs) >= 0.95（beta=1.0 时）；VI 快 100x+。

## 判定（成功标准）
- 3 个 beta 档位中 >=2 档：|ARI(VI,真值) - ARI(Gibbs,真值)| <= 0.02，且 ARI(VI,Gibbs) >= 0.90。
- 不满足 → VI 不可直接替代 Gibbs（需查 beta 高值时的 MF 退化），记录负结果仍算进展。

## 输出
- results/h1_results.csv（每行：seed, beta, method, ari_truth, ari_vi_vs_gibbs, wall_s, iters）
