# 研究决策时间线

## 2026-10-02
- [10:xx] 接收用户综述 PDF《HMRF 在空间转录组肿瘤微环境研究中的进展、价值与未来方向》，7 页全文精读完成。
- [10:xx] 用户选定 **方向二：面向百万级细胞的可扩展变分/图割 HMRF**（综述结论：性价⽐最⾼，6-10 个月）。
- [10:xx] 用户指定工作目录（项目根）。
- [10:xx] 建立 autoresearch 工作区（literature/src/data/experiments/to_human/paper）。
- [10:xx] 环境：bundled Python3 + numpy/pandas；pip 安装 pypdf（读 PDF）、scipy/scikit-learn（实验中）。
- [10:xx] 文献 API 摸底：arXiv 可通（https）、Semantic Scholar TLS 不通、web_search 可用。
  - 收获：SemanticST (arXiv 2506.11491, GNN 深度嵌入无显式空间先验)；Spatially Regularized GMM (arXiv 2025-10)；
    PyMaxflow 1.2.0（图割库，h3 基础）；Kang 2025 基准（Slide-seqV2 ARI 0.058-0.153 双洼地证据）。
- [10:xx] 假设 h1-h4 形成，h1（VI 正确性）先跑。

## 2026-10-02（续）H1 完成
- H1 通过：VI≈Gibbs（9 组 |ΔARI|<=0.008，ARI(vi,gibbs)>=0.9625；beta 失配下仍一致 0.923）。
- 修复两个 bug：simulate.py QR 维度、hmrf_solvers.py 后验模态计数。
- 发现：绝对 ARI 受噪声限制 ~0.55；规模实验用 σ=0.5/mu_scale=2.0 提辨识度。
- 下一步：h1b（VI-EM 联合估计 mu）→ h2（NB 发射 + 10万-百万规模曲线，kill test kt1）。

## 2026-10-02（续）H1b + H2 完成
- h1b 通过：VI-EM ≈ oracle ≈ Gibbs（3 种子 ARI 一致，σ 恢复 0.50）。
- h2 高斯：10K→1M 全线 ARI 0.993+，提速 76-145x（vs Gibbs 外推）→ kill test kt1 仿真层面通过。
- h2 NB：ARI 0.28-0.36，但 oracle 上限仅 0.466 → 仿真设计过苛，h2b 重设参数。
- 发现：S=750 单起点曾陷局部最优（0.963）→ 大网格需多起点/好初始化。
- 仿真器修复两 bug：rfft 频率轴 + 场归一化。
- 环境注意：dsh runtime python 无法加载新装 C 扩展（Team ID 签名冲突）→ 纯 numpy 路线；NB 软最大化常数项消去，无需 scipy。

## 2026-10-02（续）h2b + h3 完成
- h2b：NB 可辨识上限 0.466→0.857/0.878（重设参数后）；VI-EM 0.661/0.757，差距=初始化+收敛问题 → h2c。
- h3：pymaxflow 在系统 Python 3.9 可用（dsh 运行时 Python 签名限制 → 图割实验用 /usr/bin/python3）。
- 图割模块通过穷举验证（2x2/3x3 全对）；修复 segment 语义 + 能量重复计数两个 bug。
- 同实例：MF 与图割 ARI 一致（0.994/0.993），能量差可忽略；MF 快 40-50x → 易分数据上 MF 占优。
- 待办：h3b（高 beta 压力区）、h2c（NB 初始化）、float32 内存、真实数据 kt2。

## 2026-10-02（续）h2c + h3b + float32 完成
- h2c：NB 最优 beta≈1.0+（弱一元项需强先验）；S=500 达 0.890 反超固定-beta oracle。log1p 初始化+3 起点生效。
- h3b：压力区 6 组合图割 vs MF——能量图割恒低，但 ARI 差 <=0.006 且方向不一致 → 精确推断无标签增益。
  结论：MF-VI 是规模下的唯一合理引擎；图割保留为能量下界诊断工具（诚实负结果入论文）。
- float32 端到端：ARI 无损 0.9934，1M 峰值内存 1.33→1.23GB，快 16%。
- 下一轮：真实数据 DLPFC（kt2）+ 开始论文框架。

## 2026-10-02（续）H4 DLPFC 完成
- 数据：dance Dropbox 镜像（断点续传脚本 dl_loop.sh 建成，可复用）；h5 为 (genes,spots) 反序 + spot 行 CSR，六角邻接。
- 特征管线 v1→v3 消融：oracle 上限 0.31 → 0.62；最终 VI-EM ARI 0.456（β=0.7 best-of-5），中位数 0.25-0.40。
- 发现：ELBO 模型选择在真实数据同样不可靠 → β 选择准则改为边界拐点（h5b 待做）。
- 修复：σ 上限（类内方差≤总方差）防 EM 崩溃；外部 init 跳过硬 EM。
- 基线数字待核验：Kang 2025 补充材料下载失败（europepmc 流式 zip 超时）→ 下轮换路径（OUP 附件页）。

## 2026-10-02（续）基线核验 + h5b
- 基线核验（GraphST 原文 Nat Commun 2023）：151673 上 GraphST 0.64 / Seurat 0.29 / Giotto 0.34 /
  SpaGCN、BayesSpace、SpaceFlow、conST、STAGATE 介于 0.40-0.58；12 切片中位数 GraphST 0.60，其余 <0.53。
  我方 VI-HMRF 0.456 处于 0.40-0.58 区间 → 结论成立。
- h5b：ELBO 选择退化为 max-β（否定）；边界拐点在 DLPFC 选 0.5（ΔARI 0.016），NB 仿真受网格截断（ΔARI 0.06）。
- 范围决策（记录）：Slide-seqV2 全量数据（单样本数十 GB）超出本机时间预算，kt2 以 DLPFC 真实数据 +
  高稀疏 NB 百万仿真（零比例 0.17-0.21）组合替代，论文如实声明。

## 2026-10-02（续）论文初稿 v0.1 + 参考文献完成
- 基线核验完成（GraphST 原文数字）；scGCO DOI 未解（记 VERIFY，不编造）。
- refs.bib：15 条 Crossref 核验 + 2 条 arXiv + scGCO 手动条目（VERIFY 标记）。
- paper/draft.md v0.1：六章完整初稿，全部实验数字回填，含主表（kt1）、负结果表、
  DLPFC 表、β 准则结论、局限与范围声明（Slide-seqV2 未跑的如实说明）。
- 下一步（收尾轮）：初稿一致性 QA、LaTeX 版（可选）、最终汇报与归档、goal complete。

## 2026-10-02（终）收尾
- 初稿 QA：全部关键数字与 CSV 交叉核对 PASS；引用键核对（补 maynard2021 正文引用）。
- LaTeX 版编译成功（TinyTeX；pdflatex+bibtex，5 页，无错误无未定义引用）；scgco 中文注释改英文。
- README.md 归档 + to_human/final-report.html 最终报告。
- 目标达成：完整研究周期闭环（综述→文献→假设→实验→真实数据→论文初稿）。

## 2026-10-11（审查后批次1：证据加固）
- 触发：全量审查报告（review/Scalable-HMRF-Review-2026-10-10.pdf）定位的 P0 差距。批次1 = 实测 Gibbs + 多种子 + 指标补全 + 图件初版。
- h2d（多种子+NMI/边界F1）：5 实例 × 3 独立起点 × 5 规模。**≥250K 稳定**（250K 0.9936±0.0001；1M 0.9935±0.0001）；10K/62.5K 存在实例方差（0.967±0.029 / 0.976±0.024）→ 探针归因：慢收敛（378–2111 迭代）+ 次优盆地，非规模崩塌。
- **S=750 追溯修复**：实例11 在独立起点 3/4/5 下为 0.9627 稳定盆地；seeds 6–11 六个起点全部 13–16 迭代达 0.9937 → 旧 h2 脚注"2 起点复核 0.9937"为特定起点可达（6/9）。h2-scale 表列口径混用，已标 SUPERSEDED，以 h2d 为准。
- h6（Gibbs 实测，替代外推）：每扫成本 6 点拟合 R²=0.9992；1500-sweep 锚点实测 **250K 138.7s / 562.5K 361.0s / 1M 674.2s**（旧外推 204/817s，偏高 17–32%）。Gibbs 模态 ARI 0.9934–0.9937 与 VI 持平 → 差距在成本而非推断质量。
- **修正后提速（实测口径）**：单次运行 154–201×；3 起点稳健政策 16–65×（旧宣称 76–145× 作废）→ kt1（≥10×）仍通过。
- h4d（DLPFC 指标补全）：5 起点 ARI 0.1302–0.4557（median 0.2503）；NMI 0.32–0.60；严格边界 F1 0.02–0.16（oracle 0.185）；ELBO 选到 0.1302 再次确认失效；标签留档 npz。
- 图件初版落地 paper/figures/：fig2 规模/耗时/提速、fig3 图割负结果、fig4 DLPFC 四面板、fig5 β 选择、fig6 oracle 归因（生成脚本同目录）。
- 下一步（批次2，待令）：DLPFC 12 切片；SC-MEB/BayesSpace/GraphST 同管线基线；留出预测似然 β 选择（新实验）；kang_supp 重下核验。

## 2026-10-11（审查后批次2：同管线基线 + 负结果扩展）
- 数据：自 H³MRF 缓存导出 DLPFC 全 12 切片（计数与 spatialLIBD 逐位一致；n=4221/4381/4788/4595 + 3636/3484/4093/3888 + 3611/3635/3566/3431）。
- **h9 四方法基准（12 切片、同真值）**：本方法(best-of-25) 平均 ARI **0.4634 全场第一**（BayesSpace 0.4287 / SC.MEB 0.3962 / GraphST 0.3088；oracle 参照 0.6469）；5 层子集 0.5891 vs 0.4695/0.3847/0.3216。本方法(median) 0.3210 → **瓶颈=选择而非容量**（与 h4d/h7b 链条一致）。速度：全管线中位 **8.9 s** vs 76–88 s（≈10×）。
- 基线定案配置：BayesSpace 官方 spatialCluster(nrep=1000, seed49)；SC.MEB 官方 spatialPreprocess(15PC)+parallel=TRUE（配置诊断 151673：A/B/C=0.186/0.282/0.357，采用 C）；GraphST 教程全流程（600e + Mclust EEE + refine_label；rpy2≥3.6 弃用 API→Rscript 子进程版 Mclust，规避 R/numpy 双 OpenBLAS 死锁）。
- h7 新实验（留出预测似然 β 选择）：3 切片×3 遮罩 **全部塌到 max-β**（LL 单调升、Δ 仅 ~0.1–0.75 nats）→ 第三准则失效；h7b（起点选择）：ρ(LL,ARI) 中位 +0.60 vs ρ(ELBO,ARI) +0.35，LL 无灾难错选而 ELBO 有 → "留出似然弱可用"。
- kang_supp 重下核验闭环：Slide-seqV2「0.058–0.153」与 BayesSpace ST「0.642」逐字吻合（PMC12000868 全文）。
- 图件：fig7（12 切片四方法对比 + oracle 参照 + k 分组；QC 两轮修复分组线错位与图例列序）；文档：h9/h7 analysis.md；技能 hmrf-spatial-benchmark 沉淀全部环境坑。
- 下一步（批次3，待令）：Xenium/Slide-seqV2 真实大规模数据；仓库发布清理（用户路径硬编码、身份词、空目录、kang_supp 陈述）。

## 2026-10-11（批次3：Xenium 574K 真实大规模 + 论文 v0.2）
- **数据发现**：外置数据卷已有 10x Xenium V1 FFPE Human Breast IDC 官方发布数据（23 GB，无需下载）——
  574,852 细胞 × 280 基因 panel + 官方 graphclust(25)/kmeans 系列参照聚类。
- h10 全量实测：h5 加载 2.1 s；特征(log1p CP10K+zscore+PCA30)+kNN(k=6 对称图，mean deg 6.85) 共建 7.3 s；
  HMRF 单次拟合 K=8 26 s（200it）/72 s（500it）。**与参照聚类：K=8 vs kmeans-8 ARI 0.4879（最佳起点）；
  K=25 粒度匹配 vs graphclust 0.2575（394it 收敛）**。
- 选择问题第三次复现：同 β 重启间 0.39–0.48；最优 β（0.7）与仿真小规模方向相反 → 最优 β 随
  {规模,图密度,平台} 迁移。K=8 大 n 下 500 迭代未完全收敛（dq≈0.03 慢振荡，ARI 仍微增 +0.006）。
- 工程：kNN 稀疏邻居和快速路径经与原版 np.add.at 严格等价验证（ΔARI=0、标签一致 100%）；fig8 域图下沉。
- **论文 v0.2**（draft.md + main.tex，6 页编译 0 错误）：回填批次1/2 全部数字（h2d 多种子表、h6 实测 Gibbs、
  h9 12 切片基准表、h7/h7b 三准则负结果）+ 新增 §4.7 Xenium 574K 与摘要第(5)条；Limitations 更新。
- **GraphST@Xenium 对照完成**（小时级）：官方实现稠密 n×n 图构造在 574K 需 2.6 TB——物理不可行；
  本仓以 3 处最小稀疏补丁（patch_graphst_sparse.py，边语义一致）+ datatype='Stereo' 跑通：
  全量 raw ARI **0.2826 / 68 min**（600 epochs）vs 本方法 **0.4879 / ~80 s** → **+0.205 ARI、~50×**。
  50-NN refine 两数据集均降分（Visium 尺度校准），Xenium 口径报 raw；fig8 升级为三面板。
- 下一步：Slide-seqV2 仍缺（如实声明）。

## 2026-10-11（仓库发布清理 — 审查 P0 ⑥ 闭环）
- **路径可移植化**：42+1 文件 codemod（备份于 ~/.hermes/cache/scratch/hmrf_pathmod_backup_*.tar.gz）：
  全部 `/Users/` 硬编码 → `_ROOT` 自动探测（向上找 src/hmrf_solvers.py 或 $HMRF_ROOT）；R 同款 bootstrap；
  外部数据路径改环境变量（HMRF_XENIUM_DIR / HMRF_DLPFC_CACHE，缺失有明确报错）；图件脚本 FIG 路径兼容双布局。
  验证：py_compile 全过、R parse 全过、工作区与发布副本内 aggregate/fig7/fig8 端到端跑通、h10 导入链 OK。
- 空目录（h2-graphcut-scaling）已删；身份词全清（zhutao/doubao/tongji/hermes/codex → 0）；
  review/ 与编译日志属内部件（不入发布，见下）。
- **发布暂存已建**：`~/Documents/scalable-hmrf/`（229 文件，55 MB；src+experiments(code/results/analysis)+figures+README/LICENSE(MIT)/.gitignore/requirements.txt）
  → `git init` + 首提交 6240882。发布副本零身份词、零 /Users/、无数据文件、无评分材料。
- 待定（需用户决定）：① GitHub 推送（仓库名 scalable-hmrf / 公开时机）② LICENSE 归属确认
  ③ 中文注释保留策略（netith 先例）。
- 审查清单收口：100% — ①~⑥ 全部处理（④为明示口径 ⑤图件 fig2-8 ⑥本次清理+暂存）。

## 2026-10-11（发布：GitHub 公开）
- **英文化**：42 代码文件 + 18 文档（工作区与发布副本字节一致）；独立验证 py_compile 70/70、R parse 5/5、
  零中文扫描全绿、功能复测（聚合/fig7/fig8/run_ours 单切片数字逐位一致：0.4284/0.1949/0.1315/0.6191）。
- **已发布**：`github.com/tjogzt/scalable-hmrf`（**公开**；提交 6240882 首版 + f5b9f06 英文化；main 已推送；topics 已设）。
- 发布口径：仅 src+experiments+figures+README/LICENSE(MIT)/requirements；不含 data/手稿/review/literature；
  数据获取经 HMRF_XENIUM_DIR（Xenium）与 HMRF_DLPFC_CACHE（DLPFC 缓存）环境变量说明。
- 审查 P0 ①~⑥ 至此全部闭环；后续仅剩（如实声明范畴的）Slide-seqV2 与可选 Xenium 增强。


## 2026-10-11（批次4：A1–A4/A6 + B4 收尾）
- A1/A2：统一指标（NMI/边界F1 × 全方法 × 12 切片）+ 配对 Wilcoxon → h9/analysis.md；
  论文表格扩为 7 列、正文改为"竞争力而非全面占优"（vs GraphST p=0.0015 显著）。
- A3：NB 数字校正——0.740@6.25万、0.890@25万（反超 oracle 0.868）、1M 单配置 0.757；
  原稿 spot 数错位为纯文字修正；"million-scale NB"成立（h2b 早有 1M 实测）。
- A4：scGCO 定稿（Zhang/Feng/Wang, Nat Commun 13:5488, 2022；原占位条目作者+标题均错）；新增
  spatialLIBD（Pardo 2022）与 Xenium 数据（Janesick 2023, 14:8353）引用；arXiv×2 补真实作者；
  论文新增 Code and data availability（GitHub URL）。
- A6：fig1 方法总览（4 面板；4 轮 QC）。
- B4：h7 扩 5 切片×3 遮罩=15 设定：14/15 选 β≥1.0 而 ARI 峰值 0.3–0.7 → 负结果强化。
- 发布仓批次4 已同步并推送：commit 64c8792（metrics 脚本/CSV、h7 五切片、fig1、README；fig1 W↔R md5 一致）。
- Zenodo：无 API token 存档；改用 **Zenodo-GitHub 集成路线**（浏览器开关 ON → GitHub release → 自动归档发 DOI）。
  已请用户在 zenodo.org/account/settings/github/ 手动拨开关（Chrome 占用 profile 无法自动化）；release notes 已备好
  （~/.hermes/cache/scratch/release_notes_v1.txt），开关完成后即创建 v1.0.0 并回填 DOI。


## 2026-10-13（四方团队：叙事主线系统性评估）
- 派发 4 路（a 全文自读 5321 字 / b 全文内嵌 5676 字 / c 拆承诺侧+收束侧 4091+3892 字），全部 exit 0。
- 主席逐条核验全部争议断言；驳回误报 1（c2 未见 availability 段）；主席独有机械发现 3（悬空 Section 引用/图表零调引/摘要 0.993 口径）。
- 报告：review/Scalable-HMRF-Narrative-Review-2026-10-13.{tex,pdf}（9 页；四部分+优先级+裁决附录）。
- 关键断点：N1 摘要"≥0.993"与主表不符（62.5K=0.976/562.5K=0.988，P0）；N2"三负结果"同名异指；N3 主张(ii)缺"误设"前提；N4 selection vs representation 双层归因未言明；N6 Xenium 编号；N11 图表零调引。
- P0 补丁清单（S1/S5/S6/S7/S10/S13/S16/S19a）待用户批准后实施。
- 运行记录：~/.fourteam/runs/20261013-100643-{a,b,c} 与 20261013-100907-c；dispatch 同秒撞名坑已记录到评审技能。


## 2026-10-13（叙事评审落地：P0+P1+P2 全部实施）
- main.tex：43 处结构化编辑 → 正文化 11 页、**0 overfull**、图表全程调引（Figure 1–8 + Table 1–2）、
  实验节小节化 4.1–4.7、Xenium 移至选择准则之后（与摘要 (5) 编号对齐）、摘要 0.976/0.9935 口径、
  三负结果两层口径（平滑三准则 vs restart 层）、误设前提、论证地图（四主张排除链 + 主张 (v)）、
  落差归因桥、selection 理论段、oracle 三句（澄清/直觉/定量回收）、双洼地回收桥、等价链句、
  562.5K 补句、悬空 Section 引用修复、作者块 meta 清理。
- draft.md：23 处同步（v0.4）。
- 图件按"文件名对齐编号"重排：**fig5=oracle（panel-b 修正为同尺度 250K：0.868/0.713/0.890）、
  fig6=benchmark、fig7=beta**；make_fig7.py→make_fig6.py；双树字节一致。
- 发布仓 0bba9ce 已推送（图件对齐+修正）。
- 验证：编译 rc=0；overfull=0；无未解析引用；Figure 5/6/7/8 与文件名逐一对应；4 页视觉 QC 通过；
  Table 1 定宽贴合（resizebox）、Table 2 \small。

## 2026-10-16（投稿前同侪评审面板：academic-paper-reviewer 标准全模式）
- 五席（EIC 期刊匹配 / R1 方法论 / R2 领域 / R3 跨学科 / 魔鬼代言人）+ 编辑综合席；隔离上下文、
  同家族模型（DeepSeek）、已披露相关误差画像（非独立人类复核人；calibration NOT_CALIBRATED）。
- 裁决：**Major Revision**（4/4 评分席一致，confidence 均 4）；3 项 DA-CRITICAL 全部显式裁决：
  C1 exactness 范围（VALIDATED，R1 W1+R3 W2 佐证）/ C2 oracle 被超越却用于排除推断改进（VALIDATED）/
  C3 规模侧无真值锚定+σ 触界+截断未披露（VALIDATED，分项"同族仿真可采性"记 UNRESOLVED）。
- 10 项必修 R1–R10 + 8 项建议 S1–S8（编号=溯源序非工作序）：摘要 NB 口径与边界 F1、摘要对比主张降级、
  对称预算比较（BayesSpace）、novelty 重定位（Zhang 2001/Krähenbühl 2011 等经典谱系）、exactness 修复
  （49/50 保证+拓扑+个案）、计时可审计（迭代数/成本拟合/硬件）、oracle 更名 β-条件参考+β网格取上确、
  "never raise ARI" 修正、Xenium 合理性验证（marker/注释）+披露、指南边界+K 选择。
- 主席核验（对照 h3b 记录与主表）：① §1 "never raise ARI" 与 h3b 不符（GC 在 5/6 设定略高，≤0.005）；
  ② §4.5 "+0.021/+0.066" 实为中位数差（均值 +0.035/+0.067）；③ 图割比较仅 S=100（10K spot）已实锤。
- 产物：review/ars-panel-2026-10-16/（00–06 md + combined.md + **52 页合卷 PDF**；编译 rc=0、0 错误；
  视觉 QC 7 页通过；R/S 表列宽用 pipe-table 短横线权重重分配 + longtable \footnotesize）。
- ⚠ 管线坑（本机）：TinyTeX TL2026 bookmark.sty 坏（\bookmark undefined）→ pandoc 默认 LaTeX 模板编译失败；
  修复=postfix 剥离 `\usepackage{bookmark}` 改直载 `[hidelinks]{hyperref}`；另：控制台重定向切勿用
  `<jobname>.out`（与 hyperref .out 同名撞毁，诱发 NUL/扫描假错误、跨轮自污染）。脚本：panel_build.sh /
  panel_postfix.py / panel_assemble_md.py（scratch，另存至 multi-agent-manuscript-review 技能 scripts/）。

## 2026-10-16（续）（R1–R10 投稿前修订落地，全部完成）
- 实验补强 E1–E11 全部完成（详见 REVISION_LOG_R10.md）：匹配预算对照（BayesSpace/SC-MEB 各 5 seeds ×12 切片；
  **等预算下 ours best-of-5 0.427 ≈ SC-MEB 0.441（p=0.68）< BayesSpace 0.478（p=0.077）**；全 20 跑包络 0.463）、
  kNN 拓扑（wellspec 平手 / stress −0.05~+0.04 符号不一；K=5 处能量更低而 ARI 更差）、NB β 网格上限
  （**0.995/0.997/0.998** @62.5K/250K/1M；原参数 0.891/0.895；旧 0.47=单 β 假象）、参照构造稳健性
  （151673 0.21–0.63；真值锚定 0.57–0.63 vs 随机 0.21–0.29）、h3b 5 种子等价检验（+0.003，90%CI [+0.0003,+0.005]，
  ±0.01 界内）、donor 随机截距 LMM（vs GraphST −0.155 p<0.001）、K 敏感性（best−median 缺口恒在）、
  计时协议（2.8e-4 ms/spot 线性 R²=0.95；中位口径 66/40/51×）、Xenium marker 核验通过（8 域=乳腺 IDC 规范架构）、
  **GraphST 稀疏补丁 10K 等价性：图张量逐位相同、ΔARI=0、一致率 100%**、E11 配置核验（GraphST 1.1.1
  对比学习增强内置无条件——DA "未启用增强"指控不成立）。
- 主稿两段式重写（脚本 paper_rev_r10_A.py 15 处 + B 全节 + overfull/fig2/fig3 修补）：摘要重写、引言/相关工作
  （+定位表 Table 1、Zhang2001/Krähenbühl2011 等 6 条核验引用）、方法（保证范围/β 网格/边界 F1 定义）、
  §4 全节（数字全换）、讨论/结论重写；**25→20 口径全稿修正**；内部实验编号全剥离。
- 终版：**main.tex 14 页（0 错/0 overfull/0 未解析）、supplementary.tex 4 页（同绿，Figure S1/Table S1–S3）**；
  fig2/fig3/fig4/fig5/fig7 重绘 + figS1 新增；vision QC 8 页通过。
- draft.md 同步 v0.5（25+ 处）；REVISION_LOG_R10.md 建档。
- 发布仓 `~/Documents/scalable-hmrf` 提交 **72b3287** 已推送（41 文件 + 240 per-run CSV + README 更新）。
- 遗留：Zenodo DOI 开关待用户操作；Slide-seqV2 全量缺（Limitations 已声明）。
