# Scalable-HMRF：面向百万级空间转录组的可扩展变分/图割 HMRF（综述方向二）

自主研究项目（autoresearch 全流程，2026-10-02，6 轮）。
来源综述：《隐马尔可夫随机场（HMRF）在空间转录组肿瘤微环境研究中的进展、价值与未来方向》→ 方向二。

## 一句话结论
HMRF 的统计严谨性可以带到百万级 spot：均值场变分是主引擎（≥250K 多种子 ARI 0.9934–0.9937；
vs **实测** Gibbs：单次 154–201×、3 起点稳健政策 16–65×，h6 修正后口径），
精确推断（图割）不买标签精度（负结果），低分平台的分差首先归因于数据信息量（oracle 归因）。
真实数据：DLPFC 12 切片（best-of-25 平均 ARI 0.463，~10× 快于基线；配对 Wilcoxon vs GraphST p=0.0015 显著、vs BayesSpace/SC-MEB 不显著→"竞争力"口径，见批次4）＋ **Xenium 乳腺 574K 细胞**
（批次3：全流程 10 s 级准备 + 26–72 s 拟合，vs 参照聚类 ARI 0.49；GraphST 同规模对照 0.283/68 min，+0.205 ARI 且 ~50× 更快）。

## 目录
- paper/main.pdf — 论文稿（7 页，LaTeX 编译）；draft.md / main.tex / refs.bib（17 条 Crossref/EuropePMC 核验 + 2 arXiv，全部定稿）
- src/ — simulate.py（高保真仿真）、hmrf_solvers.py（MF-VI / VI-EM / Gibbs / 图结构版）、graphcut_hmrf.py（α-expansion + pymaxflow）
- experiments/ — 13 个实验（protocol + code + results + analysis）：
  h1-vi-vs-gibbs, h1b-vi-em, h2-scale（已被 h2d 取代，见其 results/SUPERSEDED_note.md）,
  h2b-nb-redesign, h2c-nb-convergence, h3-graphcut, h3b-stress, h4-dlpfc, h5-beta-adaptive,
  h5b-boundary-criterion；**审查后批次1 新增**：h2d-multiseed-metrics（多种子+NMI/边界F1）、
  h6-gibbs-measured（Gibbs 实测替代外推）、h4d-dlpfc-metrics（指标补全+标签留档）；
  **批次2 新增**：h9-dlpfc12-benchmark（12 切片四方法基准，含 fig6）、h7-predictive-beta（留出似然 β/起点选择，负结果）；
  **批次3 新增**：h10-xenium574k（574K 细胞真实大规模，含 fig8）
- paper/figures/ — 图件（fig1–fig8，PDF+PNG；生成脚本 make_fig1.py、make_figures_batch1*.py、make_fig6.py、make_fig8.py）
- review/ — 全量审查报告（Scalable-HMRF-Review-2026-10-10.pdf/.tex）+ 批次1/2 运行日志（batch1_chain.log、batch2_*.log）
- data/dlpfc/ — SpatialLIBD 151673（h5 计数 + 坐标 + 层标签）；data/dlpfc12/ — DLPFC 全 12 切片（自 H³MRF 缓存导出，计数逐位核验一致）
- to_human/ — 6 份阶段汇报（progress-1..6.html）
- research-state.yaml / research-log.md / findings.md — 全程状态与决策

## 复现
- 运行时 Python（纯 numpy）：仿真 + MF/VI 实验
- 系统 Python 3.9 + pymaxflow + h5py：图割（h3/h3b）与 h5 读取（h4/h5b）
- 论文编译：TinyTeX（pdflatex + bibtex）

## 已知边界（论文 Discussion 如实声明）
批次2 已完成 12 切片 + 四方法同管线对照（口径不对称声明）；批次3 已补真实单细胞分辨率大规模
（Xenium 574K）——Slide-seqV2 全量仍未跑，如实声明；留出似然/ELBO/边界三准则均不可靠（选择问题=瓶颈，
真实数据三度复现）；先验各向同性；disp 固定。批次4 收尾：统一指标（NMI/边界F1）+ 配对检验、引用全量定稿（scGCO=Nat Commun 2022）、fig1 方法总览、NB 口径修正（1M 级成立）、Zenodo DOI 办理中。

## 版本管理（git）

- 本目录自 2026-10-17 起纳入 git（分支 `main`），作为工作树；发布镜像在 `~/Documents/scalable-hmrf`。
- 纳入：`paper/`（tex、figures、PDF）、`experiments/`（代码与结果表）、`src/`、`review/`、`to_human/`、研究日志与状态文件。
- 排除：`data/`（公开原始数据，约 263 MB，见 `data_manifest.md`）、`literature/`（第三方论文 PDF）、LaTeX 中间产物与 `*.bak_*` 修订备份。

