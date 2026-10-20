# R1–R10 修订落地日志（2026-10-16）

> 来源：`review/ars-panel-2026-10-16/06_editorial_decision.md`（Major Revision：10 必修 R1–R10 + 8 建议）
> 实验编号映射：E1–E11 实验 + 主稿两段式重写脚本（A/B）+ 图件修补
> 状态：**全部 R1–R10 已落地**；主稿 14 页（0 错误/0 overfull/0 未解析引用）、补充材料 4 页（同绿）

## 必修项落地映射

| R | 问题（06 综合裁决） | 落地动作 | 关键证据/数字 |
|---|---|---|---|
| R1 | 摘要 emission 口径混用 + boundary-F1 缺失 | 摘要重写：Gaussian 与 NB 分列；boundary-F1 0.94→0.66 明示为 metric-dependent；方法节新增 boundary-F1 无容差定义 | main.tex 摘要 (1)；§3.5 |
| R2 | 摘要对比主张降级 | "最高 ARI"改为匹配预算双协议陈述：ours best-of-5 0.427 ≈ SC-MEB 0.441（p=0.68）< BayesSpace 0.478（p=0.077）；全 20 跑包络 0.463 | E1；Table 3/4 |
| R3 | 对称预算比较缺失 | 新增匹配重启预算对照：BayesSpace 5 seeds + SC-MEB 5 seeds（新增 103/104）×12 切片；Table 4 + 补充 S3 全表 + donor-blocked LMM | E1/E6；matched_budget_summary.csv |
| R4 | novelty 重定位 + 经典谱系 | Related Work 重写：Zhang 2001（HMRF+EM）/Krähenbühl 2011/EM 与 VI 经典脉络；新增方法定位表（Table 1，11 行×6 列） | refs.bib +6 条（Crossref 核验） |
| R5 | exactness 范围修复 [C1] | "never raise ARI"删改；保证范围（二值子模=精确 / 多标签=强局部最优）；kNN 拓扑补充实验（wellspec 平手；stress −0.05/+0.04 符号不一；K=5 处能量更低而 ARI 更差） | E2/E5；h3c_results.csv |
| R6 | 计时可审计 | 计时段落重写：逐迭代成本线性拟合（2.8e-4 ms/spot，R²=0.95）、迭代数 12–300、562.5K 触顶右尾、硬件（M4 Max/16 核/128GB）；表值改中位口径 | E10；h2d_timing_table.csv；补充 S2 |
| R7 | oracle 更名 + β 网格取上确 [C2] | 更名"管线条件参照"（类均值一元项+纯 MF，β 网格 {0.3–3.0} 取上确）；0.47/0.466 明确定性为单 β 口径假象；网格上限：重设 0.995/0.997/0.998、原参数 0.891/0.895；参照构造稳健性 0.21–0.63 | E3/E4；h2e_results.csv；ref_robustness.csv |
| R8 | 图割分散度 + 等价检验 + donor 分析 | h3b 5 种子×6 设定（均值差 +0.003，90% CI [+0.0003,+0.005]，±0.01 等价界内）；donor 随机截距 LMM（vs GraphST −0.155 p<0.001；vs SC-MEB −0.067 p=0.071）；Gibbs 单线程 NumPy 实现披露 | E5/E6；h3b_seeds_*.csv；donor_lmm.csv |
| R9 | Xenium 真值锚定 + 披露 [C3] | 8 域 marker 合理性核验（上皮/增殖/肌上皮/血管/T 细胞/髓系/脂肪；补充 Fig S1）；σ 命物理上限 + 500it 截断披露；稠密/稀疏补丁等价性 10K 验证（图逐位相同、ΔARI=0、一致率 100%）；raw/refine 双口径声明 | E7/E8；equivalence_report.json；figS1 |
| R10 | 指南边界 + K 开放问题 | Discussion 重写（8 条局限含 K 假设、精确性范围 ≤10K、等预算=次数非算力）；K 敏感性 K=5–10（best 0.30–0.44/median 0.18–0.30，缺口恒在）；"报参照即可诊断"加构造敏感性限定 | E9；k_sensitivity.csv |

## 数值口径变更汇总（全稿已同步）

- best-of-25 / median-of-25 → **best-of-20 / median-of-20**（4β×5 起点；含图注、表、正文、fig6 标签）
- 稳健提速 16–65× → **40–66×**（per-instance 3-restart 中位口径；单跑 154–201× 不变；fig2 面板 b/c 同步重绘）
- NB"上限 0.47" → **β 网格取上确 ≥0.89（原参数）/≥0.98（重设）**
- 参照"0.647" → **0.65（构造敏感 0.21–0.63，报告协议注明）**
- 图件：fig2（median 口径+gate 标签去代号）、fig3（5 种子均值±SD 重绘）、fig4/5/7 重生成、figS1 新增
- 内部实验编号（h1/h2d/h6/h9/h10/kt1 等）自正文/标题/图内全部剥离

## 新建/修改文件

- 实验代码：`experiments/h9-dlpfc12-benchmark/code/{run_bayesspace_multi.R, run_scmeb_multi.R, run_matched_budget.py, analyze_matched_budget.py, ref_robustness.py, donor_lmm.py, k_sensitivity.py}`；`experiments/h2e-nb-ceiling-grid/code/run.py`；`experiments/h3c-knn-topology/code/run.py`；`experiments/h3b-stress/code/run_seeds.py`；`experiments/h2d-multiseed-metrics/code/timing_analysis.py`；`experiments/h10-xenium574k/code/{marker_check.py, patch_equivalence_10k.py, e8_driver.sh}`；`src/graphcut_hmrf.py`（通用图 α-expansion）
- 稿件：`paper/main.tex`（两段式脚本 A 15 处 + B 全节 + 修补）、`paper/supplementary.tex`（新，4 页）、`paper/draft.md` v0.5、`paper/refs.bib` 25 条
- 发布仓：`~/Documents/scalable-hmrf` 提交 **72b3287** 已推送（41 文件 + 240 per-run CSV）

## 遗留事项

- Zenodo DOI：待用户在 `zenodo.org/account/settings/github/` 将 `tjogzt/scalable-hmrf` 开关拨 ON（发布说明已备）
- Slide-seqV2 全量：本机仍缺（Limitations 已如实声明为投影）
- 建议项 S1–S8：已随 R1–R10 顺带完成（n/CI 标注、经典引用、指标定义、K 敏感性等）；未单列
