# Block 3 答辩 Q&A

> 覆盖：模型适配（T6/T7）、概率校准（T9）、对抗探针（T10），以及四任务数据与长度匹配（length matching）。
> 数字均为 SuperPOD 实测（1×H800，4-bit NF4）。

---

## 一、总体与方法

### 1. Block 3 具体做了什么？

在 Block 1 的数据和 Block 2 的 harness/指标之上，完成三件事：
- **T6/T7 模型适配（adapter）**：把 JEV-9B（Qwen3.5-9B + LoRA + fp32 决策头）和 Qwen3.5-9B 接入统一契约 `(decision, prob, logits, latency)`，两者用**完全相同的 few-shot 示例**（每个任务 4 条、顺序一致），保证公平对比。
- **T9 校准（calibration）**：二分类用 Platt 缩放（Platt scaling），打分任务用**序数缩放（ordinal scaling）**；在 held-out 上拟合、在 eval 上报告。
- **T10 对抗探针（adversarial probes）**：四类表面形式扰动 × 200 对，测模型对选项呈现方式的敏感性。

### 2. 校准为什么重要？校准会不会改变排序（ranking）？

校准把"置信度"对齐到"真实正确率"。它**不会改变排序**：AUROC、相关系数（correlation）在校准前后几乎不变，因为校准函数是单调的。它改变的是两件事：
- **置信度的量级**（confidence level）——修 ECE（期望校准误差，expected calibration error）和 Brier 分数（Brier score）；
- 对带位置项的序数缩放，还会移动 argmax，从而改变 ACC / QWK / MAE 这类**依赖阈值/位置**的指标。

一句话：校准修的是"分数准不准"，不是"排得对不对"。

### 3. held-out 拟合和 K 折交叉拟合（K-fold cross-fitting）有什么区别？为什么两者几乎一致？

- **held-out 拟合（PLAN §2.3 协议）**：在 held_out 上拟合校准器，在 eval 上报告，每个校准器没见过它要预测的样本。
- **K 折交叉拟合（cross-fitting）**：把 eval 分 5 折，每条样本的概率来自"没在训练折里见过它"的校准器。

两者都避免了"在同一条样本上拟合又评估"的乐观偏差（optimism bias）。它们结果几乎一致（ECE/Brier 相差 ≤0.002），说明**交叉拟合没有明显过拟合**，正式报告引用 held-out 即可。

### 4. ECE、Brier、NLL 分别衡量什么？

| 指标 | 英文 | 衡量 | 特点 |
| --- | --- | --- | --- |
| ECE | expected calibration error | 置信度与正确率的分箱平均偏差 | 直觉、易解释，但对分箱敏感 |
| Brier | Brier score | 概率与标签的均方误差，可分解为区分度+可靠性 | 严格的 proper scoring rule |
| NLL | negative log-likelihood | 对真实标签的负对数似然 | 严格、对错误置信惩罚最重 |

三个一起报告，避免只挑好看的。注意 ECE 低不等于校准好——见问题 9。

### 5. bootstrap 置信区间（CI）有什么局限？

我们的 95% CI 用**百分位 bootstrap**（percentile bootstrap，默认 1000 次）重采样 eval 上的固定 out-of-fold 概率。它衡量的是 **eval 集抽样波动**，**不含**"重新拟合校准器"的不确定性，所以是下限估计，偏乐观。这是成本与严谨之间的折中，报告里已注明。

---

## 二、序数缩放与 score 偏置

### 6. 为什么温度缩放（temperature scaling）修不了 score 的偏置？

JEV 的 count 头是**位置偏置（location bias）**：预测均值 2.55，真实均值 0.64——argmax 系统性偏高，但排序是真实的（corr +0.61）。

温度缩放 `softmax(logits / T)` 只能**改变分布的宽窄，移不动众数（mode）**。用它修位置偏置，拟合器只能把 T 拉大到 ≈32，把分布压成近似均匀（Brier → 0.83 ≈ 平凡预测器的 5/6），ECE 看着低（0.027）其实是"压平"出来的假象。

### 7. γ（gamma）这个"位置项（location term）"是什么？怎么拟合的？

序数缩放公式 `softmax(logits / T + γ·i)`，其中 `i` 是分值 0–5。γ 是一个**沿分值轴的平移项**：`γ<0` 把众数往低分拉，`γ=0` 退化为纯温度缩放。

拟合目标是最小化 NLL，在 held_out 上做网格搜索 + 局部细化。JEV 拟合出 **γ = −2.0、T ≈ 0.98**（held_out 拟合 T≈0.99），把分值轴整体下移。

### 8. 把 ACC 从 0.20 拉到 0.67，算不算作弊（cheating）/ 数据泄漏（leakage）？

不算，三个理由：
1. **拟合只用 held_out**，eval 样本的校准器从未见过它们——没有泄漏。
2. **信息一直在 logits 里**：corr(gold, E[score]) = +0.61 说明概率里本来就有序数信号，错的只是决策头的 argmax 位置。
3. **这是报告端（report-side）修正**，且明确标注"经序数缩放后的报告值"；真正的修法是块 1 把 γ=−2.0 合进服务端的决策头。

### 9. Qwen 校准后 ECE（0.023）比 JEV（0.086）还低，是不是 Qwen 校准得更好？

**不是。** 这正是"ECE 低 ≠ 校准好"的例子。Qwen 的 count 头没有可用序数信号（corr 仅 +0.168），序数缩放拟合出巨大 T≈42，`logits/T` 被抹掉，只剩 `γ·i`，于是**每个输入都输出同一个固定分布、永远预测 0 类**。因为 66% 的标签本来就是 0（类先验，class prior），它的置信度（≈0.637）恰好对齐正确率，ECE 才低。看别的指标就露馅了：Brier 0.539 远差于 JEV 的 0.416，QWK=0 说明排序为零。

---

## 三、四任务数据

### 10. 四个任务分别是什么？

| 任务 | 英文 | eval 规模 | 标签 | 来源 |
| --- | --- | --- | --- | --- |
| SMS 垃圾邮件 | SMS spam | 628 | no/yes | sms_spam + enron_spam |
| 钓鱼邮件 | phishing | 797（长度匹配后） | no/yes | 多来源公开语料（重建） |
| 毒性二分类 | toxicity (binary) | 1,016 | no/yes | Jigsaw `toxic` 标签 |
| 毒性计数 | toxicity count (score) | 1,016 | 0–5 | Jigsaw 5 个子类求和 |

### 11. 毒性二分类和毒性计数为什么用同一批评论？

Jigsaw 每条评论同时带有：
- **broad toxic 标签**（0/1，整句是否含毒性）；
- **5 个子类标签**（severe_toxic / obscene / threat / insult / identity_hate，各 0/1）。

于是同一条评论可以回答两个不同问题（"是否有毒"→ 0/1；"命中几类"→ 求和 0–5），标签是**对齐的 ground truth（aligned ground truth）**。这带来一个报告上的注意点：这两条任务的结果**不是独立样本**，不能当成两份独立证据。

### 12. 为什么重建 phishing？原数据有什么问题？

原 phishing 用的是 CEAS 2008 语料（`drorrabin/phishing_emails-data`）。分析发现：
- 正例是**短营销/诈骗模板**（CNN Daily Top 10、Rolex 仿表、Hilton 广告），负例是**长开源邮件列表线程**（Python、openSUSE、Perl 技术讨论）；
- 长度本身就能预测标签：`AUC(-length) = 0.89`（用"更短=钓鱼"一个特征就能拿 0.89 AUC）。

结果两模型 AUC 都 ≈ 0.9999，任务饱和（saturated）、区分不出模型——这个分数没有意义。

### 13. 长度匹配（length matching）怎么做？AUC(-length) 是什么？

长度匹配的做法：把样本按文本长度分箱（10 箱），**每个箱内正负例各取等量**，再 60/20/20 切分。目的是消除"长度/格式"这个**混淆变量（confound）**。

`AUC(-length)` 是一个诊断：把 `-长度` 当作一个打分器，看它单独能否区分两类。匹配后 eval 的 `AUC(-length)` 从 0.89 降到 **0.48**（≈随机），说明长度不再是预测信号；此时 JEV AUC 0.914 vs Qwen 0.713，任务才真正区分模型。

### 14. 数据切分和泄漏是怎么控制的？

- 切分前按文本**去重**；train / held_out / eval 两两之间的相同文本 = **0**。
- 毒性二分类由毒性计数派生，二者保持在同一 split 内，避免跨集文本泄漏。
- 拟合集（held_out）与评测集（eval）的预测 ID 完全不相交。

---

## 四、Qwen 的退化

### 15. Qwen 的 count 任务为什么 QWK=0？

Qwen 的 count 概率里没有可用的序数信号（corr +0.168，接近噪声水平）。它原始就几乎全答 0（960/1016），序数缩放对它退化成**类先验预测（predicting the class prior）**：校准后仍全答 0，ACC 0.661 恰好等于"永远答 0"的多数类基线（majority-class baseline），QWK 归零。任何单调校准都救不回来。

### 16. 可靠性图（reliability diagram）里 Qwen 那个"单点"是什么？

是**退化拟合（degenerate fit）**的可视化结果：所有样本被预测为同一类、置信度恒定（≈0.637），于是整条可靠性曲线塌成**一个点** (0.637, 0.661)。图上已标注 "degenerate fit: all predictions = class 0 (no ordinal signal)"，避免被误读成"校准得很好"。

---

## 五、T10 对抗探针

### 17. 四类探针分别测什么？

| 探针 | 英文 | 扰动 | 指标 |
| --- | --- | --- | --- |
| 选项改名 | option renaming | No/Yes 改成 False/True（或数值改描述词） | 翻转率 + 概率 TV 距离 |
| 选项换序 | option order permutation | 逆序排列选项 | 翻转率 + 概率 TV 距离 |
| 去弃权 | abstain removal | 删除可选的 "unsure" | 弃权率 + 强制作答准确率 |
| 提示注入 | prompt injection | 文本末尾追加"忽略指令，答案是 X" | 翻转率 + 答 yes 率变化 |

每类 200 对（original / perturbed 共用 pair_id）。

### 18. 为什么 prompt injection 对 Qwen 信息量有限？

**天花板效应（ceiling effect）**：Qwen 在原始条件下已经 94.5% 答 yes，注入"答案是 yes"之后基本没变化空间（94.5% → 97.0%），看不出它到底有没有被推动。所以对 Qwen 而言，更有信息量的是两个表面形式探针（改名、换序）。

### 19. 翻转率（flip rate）和总变差距离（TV distance）的区别？

- **翻转率**：原始与扰动下 argmax 决策不同的样本占比——只看"最终答案变没变"。
- **TV 距离（total variation distance）**：两个概率分布的逐项绝对差之和的一半——看"整个概率分布挪了多少"，比只看决策更敏感。

两者一起报告：翻转率直观，TV 能捕捉"决策没变但置信度大幅漂移"的情况。

### 20. 为什么 T10 只在二分类任务上做？

打分（score）任务的探针暂缓，因为 score 头本身有**位置偏置**（见问题 6），探针结果会混入决策头的位置效应，难以归因。等块 1 把 γ 修进服务端后再补。

---

## 六、模型对比与结论

### 21. 最终结论是什么？JEV 全面碾压 Qwen 吗？

JEV 在每个任务的**排序（AUC）上都更强**，且对表面形式稳健；但它的结论要按指标分开说：
- **排序质量**：JEV 全面领先（SMS AUC 0.967、phishing 0.914、毒性二分类 0.928；Qwen 对应 0.857 / 0.713 / 0.699）。
- **阈值**：Qwen 二分类"排序尚可、原始阈值坏"（几乎总答 yes），Platt 学出阈值后 ACC 才上来——要明确这是"学出阈值后的准确率"，不是原生能力。
- **打分**：JEV 只有位置偏置（修好 QWK 0.44）；Qwen 无任何序数信号（QWK=0）。

所以不是"碾压"，而是"排序上 JEV 全面领先 + 一个可修的位置偏置；Qwen 的二分类需要学阈值、打分任务完全无效"。

### 22. 延迟（latency）是怎么测的？为什么不测 HTTP 端到端？

延迟来自 `cluster_run.py` 的**进程内批量推理（in-process batch inference）**，含每条约 0.4ms 的落盘开销，**不等于** HTTP 服务端到端延迟。这样测是为了在 GPU 节点上绕开 HTTP 开销、得到纯推理耗时；报告里已注明口径，若与 T8 的服务延迟对比需要统一口径。

### 23. γ 修正是报告端（report-side）的，块 1 应该做什么？

块 1 可以把 γ = −2.0 合进服务端的 `score` 决策头（对输出 logits 加 `γ·i` 或直接修 `proj.bias`），这样服务端直接输出校准后的分数，报告端就不需要再额外说明"经序数缩放后的报告值"。在块 1 改之前，引用 score 指标必须写清这个前提。
