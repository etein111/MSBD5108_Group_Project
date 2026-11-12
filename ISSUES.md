# 块 1 / 块 2 问题清单（块 3 视角）

> 整理日期：2026-11-12。来源：在块 3 接入真实模型（SuperPOD，2000 条 × 2 模型）过程中实际踩到的问题，
> 以及对照 `PLAN.md` §2.2–§2.5、§三、§四逐条核对的结果。
> 每条都给了**证据**（文件行号 / 实测数字），可以直接复现。

## 摘要

| # | 严重度 | 归属 | 问题 | 影响 |
| --- | --- | --- | --- | --- |
| 1 | **P0** | 块 2 | `score` 标签是随机数，不是毒性 | 所有 `score` 结论作废 |
| 2 | **P0** | 块 2 | 数据集只覆盖 1/3 任务（只有 SMS spam） | 无法支撑"三类任务"的结论 |
| 3 | **P0** | 块 1+2 | train/eval 文本泄漏、`held_out` 没人用 | 公平性与方法学被质疑 |
| 4 | P1 | 块 2 | harness 契约只回标量概率 | 多分类校准指标算不出来 |
| 5 | P1 | 块 2 | 缺 AUROC / macro-F1 口径下的阈值问题 | 准确率结论会被误读 |
| 6 | P1 | 块 2 | 缺 bootstrap CI、显著性与成本指标 | PLAN §2.5、§2.4 未覆盖 |
| 7 | P1 | 块 1 | `latency` 单位 ms vs harness 的秒 | 延迟差 1000 倍 |
| 8 | P1 | 块 1+2 | `requirements.txt` 装不出能跑的评测环境 | 别人照 README 复现不了 |
| 9 | P2 | 多 | 文档/整洁类（平衡性说法、Mock 图、显存、课程号等） | 影响不大但会被问 |

---

## 状态更新（2026-11-13，对照 `origin/main` c7a2f00）

| # | 状态 | 证据 |
| --- | --- | --- |
| 1 | **已修** | `data_pipeline.py` 的 `score` 改为 Jigsaw 五个子类标签求和（0–5），prompt 同步改成"Count the applicable toxicity categories"；`subtype_labels`/`toxic_label` 原样保留。 |
| 2 | **已修** | `noul` 现在来自 `sms_spam` + `enron_spam` + `phishing_email` 三个源（eval: 325/303/345 条），任一源缺失会直接报错而不是静默降级。 |
| 3 | **已修 + 待用** | 切分前按 `(type, text)` 去重；实测 train∩eval、train∩held_out、held_out∩eval 的相同文本均为 **0**。`held_out` 仍无仓库内消费者，但块 3 已加 `--fit-dir` 支持 PLAN §2.3 协议，只差一次 GPU 预测。 |
| 4 | **已修** | harness 契约改为完整 `prob` dict；`metrics.py` 会校验 `sum≈1`、`decision==argmax`，并计算多分类 Brier/ECE。块 3 适配器已跟随（本分支）。 |
| 5 | **已修** | `metrics.py` 增加 AUROC、balanced accuracy、macro-F1 与 `by_source` 分组。 |
| 6 | **已修** | 增加 bootstrap 95% CI、exact McNemar、score 的配对置换检验、p50/p95、`gpu_seconds_per_1000`。 |
| 7 | **已修** | `Prediction.latency_ms` 为准，`latency` 保留为毫秒别名；服务响应同时给两个字段。 |
| 8 | **已修** | 新增 `requirements-eval.txt`（根目录与 `evaluation/` 各一份），声明 numpy / scikit-learn / matplotlib / datasets。 |
| 9 | **大部分已修** | `matplotlib.use("Agg")` 已加、Mock 看板移到 `evaluation/outputs/`、`type(self)` 死代码已删；仅剩 `deployment/server.py:60` 的 `MSBD5108` 课程号未改。 |

结论：块 1/2 的 P0、P1 均已修复，块 3 可以按新契约继续；旧数字全部作废，需在新数据上重跑。

---

## P0 — 影响结论正确性

### 1. `score` 的标签与任务无关

`evaluation/data_pipeline.py:80-88`：

```python
label_raw = item.get("label", 0)          # IMDB 情感标签 0/1
if label_raw == 0:
    score_label = random.choice([0, 1, 2])   # 负面影评 → 随机 0/1/2
else:
    score_label = random.choice([3, 4, 5])   # 正面影评 → 随机 3/4/5
```

而 prompt 是 `"Rate the toxicity level of this text from 0 to 5:"`（`data_pipeline.py:28`），
text 是 IMDB 影评（截断到 800 字符）。

* 标签 = 情感极性的**随机细分**，与"毒性"没有任何对应关系；任何模型的理论上界是 1/3。
* 实测（`block3/outputs/preds`）：JEV exact match 0.131、Qwen 0.128，六分类随机是 0.167——**低于随机**。
* 交叉验证：负面影评（gold 0–2）JEV 的期望分 2.01，正面影评（gold 3–5）1.26，
  即"越负面→越毒"（合理），但标签把负面映射到低分，所以 corr(gold, E[score]) = **−0.410**（Qwen −0.201）。
* 校准的印证：温度缩放最优解停在搜索上界 `T=1e4`，Brier 恰好 = 5/6 = 0.8333，
  即"最优校准 = 输出均匀分布"——数据里没有可学的信号。

**影响**：块 3 的 T9 中 `score` 的一半结论、T10 的 `score` 分支、报告里所有 `score` 数字全部不可用。
**建议**：换 Jigsaw / ToxiGen 的真实毒性分级；若时间不够，至少在报告里把 `score` 标为"占位、不下结论"。

### 2. 数据集只覆盖了三类任务中的一类

`data_pipeline.py:41-61` 的 `noul` 只来自 `ucirvine/sms_spam`。实测 `eval.jsonl` 的 `noul` 文本：
最长 446 字符、中位 61 字符、**没有一条含 `Subject:`**——即没有 Enron、没有钓鱼邮件语料。

但 `data_pipeline.py:19` 的 prompt 是 `"Is this message spam or phishing?"`，
`PLAN.md` §一 承诺的是「垃圾邮件 / 钓鱼邮件 / 毒性」三类任务。

**影响**：报告不能说"在 phishing 上如何如何"；prompt 里承诺"spam or phishing"而数据只有 SMS spam，
是一个语义不匹配（模型被要求判断两件事，标签只标了一件事）。
**建议**：要么补 Enron + 钓鱼语料，要么把 prompt 改成 `"Is this message spam?"` 并在报告收窄任务定义。

### 3. 数据泄漏 + `held_out` 未被使用

`data_pipeline.py:101-114` 先 `random.shuffle` 再按 60:20:20 切片，**没有按文本去重**。实测：

| 对比 | 相同文本数 |
| --- | --- |
| train ∩ eval | **65** |
| train ∩ held_out | **64** |
| held_out ∩ eval | **31** |
| eval 内部重复 | 23 |

SMS spam 本身就有大量重复样本，同一条文本会同时落进 train 和 eval。块 3 的 few-shot 示例取自
`train.jsonl`（`block3/adapters.py`），因此 Qwen 的示例里可能直接包含评测原题。

另外：**整个仓库没有任何代码读取 `held_out.jsonl`**（`grep` 可验证）。而 `PLAN.md:46` 写的是
「两者都过交叉拟合 Platt/temperature 缩放（**用 held-out 集拟合、eval 集报告**）」。

**影响**：① 泄漏让 Qwen 的少数样本"见过答案"，量级约 3%，不影响主结论但会被问；
② 校准协议与 PLAN 不符——目前我在 eval 上做 5 折交叉拟合（每条样本的校准器都没见过它，
所以**没有乐观偏差**，但严格说不是 PLAN 写的方式）。
**建议**：① 切分前按 `text` 去重（或 group split）；② 补跑 held_out 的预测（2000 条 ≈ 5 分钟 GPU），
用 held_out 拟合、eval 报告，作为主结果；交叉拟合版本保留作稳健性检查。

---

## P1 — 接口与指标

### 4. harness 契约只回标量概率，多分类校准算不出来

`evaluation/harness.py:9-14` 的协议是 `(decision, probability, logits_dict, latency_seconds)`，
其中 `probability` 是**标量**。`metrics.py:29-39` 只能拿它当 `P(decision)`。

后果：
* `metrics.py:39` 收集了 `score_probs` 但**从未使用**；`compute_all_metrics` 只为 `noul` 算了
  ECE/Brier（`metrics.py:50-58`），`score` 只有 exact/MAE/QWK。
  而 `PLAN.md:50` 要求「校准：ECE（15-bin）+ Brier + reliability 图」对两类任务都成立。
* 标量概率还有一个歧义风险：`P(decision)` 和 `P(yes)` 长得一样，
  如果哪个适配器回的是后者，`metrics.py` 会静默地按 `1-p` 去算，ECE 全错且不报错。

**建议**：契约改成返回完整 `prob` dict（块 3 的 `Prediction.predict_full()` 已经有），
`metrics.py` 补多分类 ECE / Brier；同时加一条断言：`sum(prob)≈1`、`decision == argmax`。

### 5. 准确率口径：类别不平衡 + 固定 0.5 阈值

`eval.jsonl` 的 `noul` 正例只占 **14.1%**（141/998）。`metrics.py:48` 的 exact match 在这种分布下：
* 全答 `no` 就有 **85.9%**；
* 而 Qwen 实测 argmax 准确率只有 54.9%，因为它把阈值放在 0.5、对 57.9% 的样本答 yes。

所以拿 exact match 当头条指标会得出"JEV 95% vs Qwen 55%"这种被阈值放大的差距。
`PLAN.md:49` 其实也要求 macro-F1，但当前没有 **AUROC**——而 AUROC 才是与阈值无关的排序质量。
实测 AUC：JEV 0.974 / Qwen 0.889，差距远小于准确率的差距。

**建议**：`metrics.py` 增加 AUROC 与 balanced accuracy；报告主表用 macro-F1 + AUROC，
exact match 只在注明阈值（0.5，或按 held_out 先验校正后）的前提下给。

### 6. PLAN 要求的统计与成本指标全部缺失

* `PLAN.md:56`：95% bootstrap CI + McNemar / 置换检验 —— 仓库里没有实现。
* `PLAN.md:51`：每 1000 次决策的成本 —— 没有实现。
* `PLAN.md:52`：延迟 p50/p95 —— `harness.py:75` 只记录单条 latency，没有汇总；
  `block3/runner.py` 有 `summarize()`，但那是块 3 的临时工具，不是块 2 的交付物。

**建议**：T8 补 `metrics.py` 的这一块；成本口径建议直接用「单条平均秒数 → 每 1000 次 GPU-秒」，
并注明是 `transformers` 逐条推理（batch=1）的实现，不是 vLLM 吞吐。

### 7. `latency` 单位不一致

**已修复**：服务新增显式 `latency_ms` 字段，旧 `latency` 保留为等值毫秒别名。块 3 优先读取新字段，并兼容旧服务；harness 四元组仍转换为秒。测试覆盖两种任务、新旧响应和毫秒→秒转换。以下保留原始问题描述。

`deployment/protocol.py:32` 的 `Prediction.latency` 是**毫秒**（`runtime.py:151` 乘了 1000）；
`harness.py:12` 的契约是 `latency_seconds`。块 3 的适配器在 `as_harness_tuple()` 里除了 1000，
但任何绕过适配器直接接的人都会拿到 1000 倍的数字。
**建议**：字段名改成 `latency_ms`，或 harness 侧对量级做校验。

### 8. `requirements.txt` 装不出能跑的评测环境

**已修复**：新增固定直接依赖版本的 `requirements-eval.txt`，包含基础依赖及 numpy、scikit-learn、matplotlib、datasets。安装和离线验证见 [评测环境说明](evaluation/README)。

`requirements.txt` 只有 4 个包（fastapi / httpx / pydantic / uvicorn），`requirements-cuda.txt` 只加
torch 相关。而 `evaluation/metrics.py` 需要 **numpy / scikit-learn / matplotlib**，
`evaluation/data_pipeline.py` 需要 **datasets**，全都没有声明。
评测脚本使用同目录导入。从 `evaluation/` 运行 `../.venv/bin/python main.py`，Windows 使用 `..\.venv\Scripts\python.exe -X utf8 main.py`。完整脚本会下载数据并重写 `evaluation/data/`。验证依赖时运行 README 中的离线测试。

---

## P2 — 文档与整洁

| 问题 | 位置 | 说明 |
| --- | --- | --- |
| "平衡的评测集"说法不成立 | `data_pipeline.py:36`；`evaluation/README` | noul 正例仅 13.5%，未做任何平衡采样 |
| Mock 图混在数据目录 | `evaluation/data/jevbench_evaluation_dashboard.png` | 由 `evaluation/main.py` 的 `MockModelAdapter` 生成，容易被当成真实结果 |
| 显存数字偏乐观 | `deployment/README.md:9` vs `:121` | 实测峰值 10.3 GiB allocated / 11.6 GiB reserved（12 GB 卡），不是"4-bit ≈ 7GB"；8 GB 卡跑不起来 |
| 课程号写错 | `deployment/server.py:33` | `MSBD5108` → `MSBD5018` |
| 默认参数与产物不符 | `data_pipeline.py:136`（1500）vs `evaluation/main.py:8`（10000） | 实际 `data/*.jsonl` 是 10000 条生成的，直接跑模块默认值会得到另一份数据集 |
| headless 环境可能崩 | `metrics.py:3,127` | 没有 `matplotlib.use("Agg")`，Slurm/无 DISPLAY 环境有风险 |
| 死代码 | `metrics.py:122` | `type(self)` 无作用 |
| 延迟口径未注明 | `deployment/README.md:90` | 计时从**取得推理锁之后**开始，不含排队；并发/吞吐必须另测 |
| T10 自定义选项 | `deployment/`、`block3/` | 接口已支持 `choice`，三种选项探针直接改动选项块，按稳定 ID 汇总，正式评测待运行。见 [T10 运行说明](block3/README.md#t10--四类对抗探针)。 |

---

## 我这边（块 3）要跟着改的

1. **T9 换校准协议**：按 PLAN §2.3 改成 held_out 拟合、eval 报告（需要先补 held_out 的预测）。
2. **`score` 全部挂起**：等块 2 换数据后重跑 T9 的 score 分支和 T10 的 score 探针。
3. **T10 正式对比待运行**：先跑 `noul` 来源任务，`score` 分支等数据。
4. `block3/outputs/` 已 gitignore；`block3/README.md` 的"真实运行结果"一节里有全部实测数字。

## 复现命令

```bash
# 数据审计（泄漏 / 标签分布）
python - <<'PY'
import json, collections
from pathlib import Path
s={n:[json.loads(l) for l in Path(f"evaluation/data/{n}.jsonl").open() if l.strip()]
   for n in ("train","held_out","eval")}
for a,b in (("train","eval"),("train","held_out"),("held_out","eval")):
    print(a,b,len({r["text"] for r in s[a]} & {r["text"] for r in s[b]}))
PY
```

